//! What the server gives Codex beyond the Responses API, through HTTP. GET
//! /v1/codex/models returns Codex's model catalog: an entry for each served
//! model with its served context, Codex's instructions and the apply_patch
//! tool. POST /v1/alpha/search runs the commands of Codex's web tool. The
//! server loads no weights, since no chat request runs.
//!
//! Set EREMITE_CHECKPOINT to the DeepSeek checkpoint directory and
//! EREMITE_EXPERTS optionally to GGUF experts. EREMITE_SERVER picks the server
//! binary (default build/eremite-server). The tests run one model process at
//! a time (they share a lock):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test codex -- --nocapture

use std::net::TcpListener;
use std::path::PathBuf;

use serde_json::{Value, json};

mod common;
use common::{Server, build_config, http, one_model, root};

/// The context that the tests' server serves, in tokens.
const CTX: u64 = 32768;

/// Starts a server of the checkpoint `checkpoint`. It writes no saved states.
fn start_server(checkpoint: &str) -> Server {
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("codex-test-{}-{port}-key", std::process::id());
    let experts = std::env::var("EREMITE_EXPERTS").ok();
    let config = build_config(
        port,
        json!({"memory": 56, "state_disk": 0}),
        checkpoint,
        experts.as_deref(),
        json!({"ctx": CTX}),
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
fn the_catalog_lists_the_served_model_with_its_context() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let server = start_server(&checkpoint);
    for path in ["/v1/codex/models", "/codex/models"] {
        let (code, catalog) = ask(&server, "GET", path, None);
        assert_eq!(code, 200, "{path}: {catalog}");
        let entry = &catalog["models"][0];
        println!(
            "{path}: {} at {} tokens, reasoning {}",
            entry["slug"], entry["context_window"], entry["default_reasoning_level"]
        );
        assert_eq!(entry["slug"], "deepseek-v4.1-flash", "{path}");
        assert_eq!(entry["context_window"].as_u64(), Some(CTX), "{path}");
        assert_eq!(entry["apply_patch_tool_type"], "freeform", "{path}");
        assert!(
            entry["model_messages"]["instructions_template"]
                .as_str()
                .is_some_and(|instructions| !instructions.is_empty()),
            "{path}: the entry has no instructions"
        );
    }
    let (code, _) = ask(&server, "POST", "/v1/codex/models", Some(&json!({})));
    assert_eq!(code, 405);
}

/// A command that the web tool lacks gets a note, and so does a call without
/// commands. A body that is no search request gets 400. No command here
/// reaches SearXNG or the web.
#[test]
fn the_web_tool_answers_each_call_with_its_output() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let server = start_server(&checkpoint);
    let call = json!({"id": "codex-test", "commands": {"image_query": [{}]}});
    let (code, answer) = ask(&server, "POST", "/v1/alpha/search", Some(&call));
    assert_eq!(code, 200, "{answer}");
    let output = answer["output"].as_str().unwrap_or("");
    println!("the web tool's output: {output:?}");
    assert!(output.contains("image_query: not available"), "{output}");
    assert_eq!(answer["encrypted_output"], Value::Null);
    let (code, answer) = ask(
        &server,
        "POST",
        "/alpha/search",
        Some(&json!({"id": "codex-test"})),
    );
    assert_eq!(code, 200, "{answer}");
    assert!(
        answer["output"]
            .as_str()
            .is_some_and(|output| output.contains("No command was given.")),
        "{answer}"
    );
    let (code, _) = ask(
        &server,
        "POST",
        "/v1/alpha/search",
        Some(&json!({"commands": 5})),
    );
    assert_eq!(code, 400);
}
