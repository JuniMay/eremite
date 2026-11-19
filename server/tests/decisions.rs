//! A decision model through HTTP: LiquidAI's d1, which answers questions and
//! generates no text. The model lists show it with its checkpoint's context,
//! Codex's catalog leaves it out, and the chat endpoints refuse it; the server
//! loads no weights for these. POST /v1/decisions answers questions about a
//! text or JSON input, and about images, with the probabilities of
//! LiquidAI's own code.
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
use common::{Server, build_config, encode_base64, http, make_test_bmp, one_model, root};

/// The tokens that one pass of d1-2B holds.
const CONTEXT: u64 = 128000;
/// The largest difference in probability of LiquidAI's own BF16 run from its
/// FP32 run, over four short requests of text, and over four requests about
/// images.
const PROBABILITY_BOUND: f64 = 0.0067;
const IMAGE_PROBABILITY_BOUND: f64 = 0.0112;

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

/// Returns an input_image part of the test pattern at width x height pixels,
/// as a BMP file in a base64 data URL.
fn image_part(width: usize, height: usize) -> Value {
    let data = encode_base64(&make_test_bmp(width, height));
    json!({"type": "input_image", "image_url": format!("data:image/bmp;base64,{data}")})
}

/// Checks that `answer`'s probabilities, option by option, are within `bound`
/// of `expected`.
fn check_probabilities_within(answer: &Value, expected: &[f64], bound: f64) {
    let probabilities: Vec<f64> = answer["probabilities"]
        .as_array()
        .expect("the answer's probabilities")
        .iter()
        .map(|option| option["probability"].as_f64().expect("a probability"))
        .collect();
    assert_eq!(probabilities.len(), expected.len(), "{answer}");
    for (probability, expected) in probabilities.iter().zip(expected) {
        assert!(
            (probability - expected).abs() <= bound,
            "{answer}: {expected} in LiquidAI's FP32 run"
        );
    }
}

/// Checks that `answer`'s probabilities, option by option, are within
/// PROBABILITY_BOUND of `expected`.
fn check_probabilities(answer: &Value, expected: &[f64]) {
    check_probabilities_within(answer, expected, PROBABILITY_BOUND);
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

#[test]
fn a_decision_model_answers_about_images_as_liquidais_code_answers() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
        return;
    };
    let _one = one_model();
    let server = start_server(&checkpoint);
    // A text state after a 400 x 208 image, with a predicate, a choice and a
    // score. The expected probabilities are those of LiquidAI's own code in
    // FP32.
    let card = json!({
        "input": [{"role": "user", "content": [
            image_part(400, 208),
            {"type": "input_text", "text": "A test card: color ramps and a striped blue texture."}]}],
        "questions": [
            {"type": "predicate", "instructions": "Is the picture mostly blue?"},
            {"type": "choice", "instructions": "Which color is strongest at the right edge?",
             "choices": [{"value": "red"}, {"value": "green"}, {"value": "blue"}]},
            {"type": "score", "instructions": "How colorful is the picture?",
             "levels": [{"label": "dull"}, {"label": "plain"}, {"label": "bright"}]}
        ]
    });
    let (code, decision) = ask(&server, "POST", "/v1/decisions", Some(&card));
    assert_eq!(code, 200, "{decision}");
    println!("{decision}");
    let answers = &decision["answers"];
    let blue = answers[0]["probability"].as_f64().expect("a probability");
    assert!(
        (blue - 0.08439659339814178).abs() <= IMAGE_PROBABILITY_BOUND,
        "{blue}"
    );
    assert_eq!(answers[1]["choice"], "red");
    check_probabilities_within(
        &answers[1],
        &[
            0.9709278128599893,
            0.010602843523026886,
            0.018469343616983654,
        ],
        IMAGE_PROBABILITY_BOUND,
    );
    check_probabilities_within(
        &answers[2],
        &[
            0.009501194952973599,
            0.027155231986241753,
            0.9633435730607847,
        ],
        IMAGE_PROBABILITY_BOUND,
    );
    assert_eq!(
        decision["usage"]["input_tokens"], 185,
        "the image's span counts whole"
    );

    // A 200 x 150 image alone: the question follows it.
    let alone = json!({
        "input": [{"role": "user", "content": [image_part(200, 150)]}],
        "questions": [{"type": "predicate", "instructions": "Does the picture show a person?"}]
    });
    let (code, decision) = ask(&server, "POST", "/v1/decisions", Some(&alone));
    assert_eq!(code, 200, "{decision}");
    let person = decision["answers"][0]["probability"]
        .as_f64()
        .expect("a probability");
    assert!(
        (person - 0.01885821318423426).abs() <= IMAGE_PROBABILITY_BOUND,
        "{person}"
    );
    assert_eq!(decision["usage"]["input_tokens"], 97);

    // An image that does not decode gets 400.
    let broken = json!({
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,aGk="}]}],
        "questions": [{"type": "predicate", "instructions": "?"}]
    });
    let (code, refused) = ask(&server, "POST", "/v1/decisions", Some(&broken));
    assert_eq!(code, 400, "{refused}");
    server.stop();
}
