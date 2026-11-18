//! A decision model through HTTP: LiquidAI's d1, which answers questions and
//! generates no text. The model lists show it with its checkpoint's context,
//! Codex's catalog leaves it out, and the chat endpoints refuse it. The server
//! loads no weights for these.
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
