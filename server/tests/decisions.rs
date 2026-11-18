//! A decision model through HTTP: LiquidAI's d1, which answers questions and
//! generates no text. The model lists show it with its checkpoint's context,
//! Codex's catalog leaves it out, and the chat endpoints refuse it; the server
//! loads no weights for these. POST /v1/decisions answers questions about a
//! text or JSON input with the probabilities of LiquidAI's own code.
//!
//! Set EREMITE_D1_CHECKPOINT to the d1 checkpoint directory. EREMITE_SERVER
//! picks the server binary (default build/eremite-server). The tests run one
//! model process at a time (they share a lock):
//!
//!     EREMITE_D1_CHECKPOINT=DIR cargo test --release \
//!       --manifest-path server/Cargo.toml --test decisions -- --nocapture

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

mod common;
use common::{Server, build_config, http, one_model, root};

/// The tokens that one pass of d1-2B holds.
const CONTEXT: u64 = 128000;
/// The largest difference in probability of LiquidAI's own BF16 run from its
/// FP32 run, over four short requests.
const PROBABILITY_BOUND: f64 = 0.0067;

/// Starts a server of the d1 checkpoint `checkpoint`. It writes no saved
/// states.
fn start_server(checkpoint: &str) -> Server {
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("decisions-test-{}-{port}-key", std::process::id());
    let config = build_config(
        port,
        json!({"memory": 16, "state_disk": 0}),
        checkpoint,
        None,
        json!({}),
    );
    Server::start(&program, &config, &key, port)
}

/// Sends `method` to `path` with `body`. Returns the status and the JSON
/// answer, or null for another answer.
fn ask(server: &Server, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let (code, answer) = http(server.port, &server.key, method, path, body);
    (code, serde_json::from_slice(&answer).unwrap_or(Value::Null))
}

#[test]
fn a_decision_model_is_listed_with_its_context_and_refuses_chat() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
        return;
    };
    let folder = Path::new(&checkpoint)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the checkpoint's folder");
    let id = folder.to_lowercase();
    let _one = one_model();
    let server = start_server(&checkpoint);
    let (code, models) = ask(&server, "GET", "/v1/models", None);
    assert_eq!(code, 200, "{models}");
    let model = &models["data"][0];
    println!("listed: {model}");
    assert_eq!(model["id"], id.as_str());
    assert_eq!(model["name"], format!("LiquidAI {folder} (eremite)"));
    assert_eq!(model["context_length"].as_u64(), Some(CONTEXT));
    assert_eq!(model.get("max_output_tokens"), None, "it generates no text");
    let (code, catalog) = ask(&server, "GET", "/v1/codex/models", None);
    assert_eq!((code, &catalog["models"]), (200, &json!([])), "{catalog}");
    let refusal = format!("{id} is a decision model: it answers questions and generates no text");
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let body =
            json!({"model": id, "messages": [{"role": "user", "content": "Hi"}], "input": "Hi"});
        let (code, answer) = ask(&server, "POST", path, Some(&body));
        assert_eq!(code, 400, "{path}: {answer}");
        assert_eq!(answer["error"]["message"], refusal.as_str(), "{path}");
    }
    server.stop();
}

/// Checks that `answer`'s probabilities, option by option, are within
/// PROBABILITY_BOUND of `expected`.
fn check_probabilities(answer: &Value, expected: &[f64]) {
    let probabilities: Vec<f64> = answer["probabilities"]
        .as_array()
        .expect("the answer's probabilities")
        .iter()
        .map(|option| option["probability"].as_f64().expect("a probability"))
        .collect();
    assert_eq!(probabilities.len(), expected.len(), "{answer}");
    for (probability, expected) in probabilities.iter().zip(expected) {
        assert!(
            (probability - expected).abs() <= PROBABILITY_BOUND,
            "{answer}: {expected} in LiquidAI's FP32 run"
        );
    }
}

#[test]
fn a_decision_model_answers_as_liquidais_code_answers() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
        return;
    };
    let _one = one_model();
    let server = start_server(&checkpoint);
    // A text input with a predicate, a choice and a score. The expected
    // probabilities are those of LiquidAI's own code in FP32.
    let ticket = json!({
        "input": "I was charged twice and my parcel never arrived. Please refund one charge.",
        "questions": [
            {"type": "predicate", "name": "refund",
             "instructions": "Is the customer asking for a refund?"},
            {"type": "choice", "name": "team", "instructions": "Which team should handle this?",
             "choices": [
                 {"value": "billing", "description": "Charges, refunds, invoices"},
                 {"value": "shipping", "description": "Delivery and parcels"},
                 {"value": "technical", "description": "App or site faults"}]},
            {"type": "score", "name": "urgency", "instructions": "How urgent is this ticket?",
             "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]}
        ]
    });
    let (code, decision) = ask(&server, "POST", "/v1/decisions", Some(&ticket));
    assert_eq!(code, 200, "{decision}");
    println!("{decision}");
    let answers = &decision["answers"];
    assert_eq!(answers[0]["name"], "refund");
    let refund = answers[0]["probability"].as_f64().expect("a probability");
    assert!(
        (refund - 0.9974407257466943).abs() <= PROBABILITY_BOUND,
        "{refund}"
    );
    assert_eq!(answers[1]["choice"], "billing");
    check_probabilities(
        &answers[1],
        &[0.6963505675216248, 0.25612062167091604, 0.04752881080745921],
    );
    check_probabilities(
        &answers[2],
        &[0.166013676544358, 0.6243465916565784, 0.20963973179906356],
    );
    assert_eq!(decision["usage"]["input_tokens"], 123);
    assert_eq!(decision["usage"]["output_tokens"], 0);

    // A JSON input in one user message.
    let order = json!({
        "input": [{"role": "user", "content": [{"type": "input_json", "data": {
            "ticket": {"body": "Please refund the duplicate charge."},
            "order": {"charges": [49, 49.5], "paid": true, "note": null}}}]}],
        "questions": [{"type": "predicate",
            "instructions": "Does the ticket describe a duplicate charge supported by the order?"}]
    });
    let (code, decision) = ask(&server, "POST", "/decisions", Some(&order));
    assert_eq!(code, 200, "{decision}");
    let duplicate = decision["answers"][0]["probability"]
        .as_f64()
        .expect("a probability");
    assert!(
        (duplicate - 0.921040621318548).abs() <= PROBABILITY_BOUND,
        "{duplicate}"
    );
    assert_eq!(decision["usage"]["input_tokens"], 89);

    // A request without questions gets 400, and another method 405.
    let (code, refused) = ask(
        &server,
        "POST",
        "/v1/decisions",
        Some(&json!({"input": "Hi", "questions": []})),
    );
    assert_eq!(code, 400, "{refused}");
    assert_eq!(server.get("/v1/decisions").0, 405);
    // The status counts the two decisions.
    let status = server.settled(2);
    assert_eq!(status["totals"]["prompt_tokens"], 123 + 89);
    server.stop();
}
