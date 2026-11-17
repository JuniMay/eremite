//! What the server gives Codex beyond the Responses API, through HTTP. GET
//! /v1/codex/models returns Codex's model catalog: an entry for each served
//! model with its served context, Codex's instructions and the apply_patch
//! tool. The server loads no weights, since no chat request runs.
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
use common::{Server, http, one_model, root};

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
    let ctx = CTX.to_string();
    let mut args: Vec<String> = [
        "-m",
        checkpoint,
        "--memory",
        "56",
        "--ctx",
        &ctx,
        "--state-disk",
        "0",
    ]
    .map(String::from)
    .to_vec();
    if let Ok(experts) = std::env::var("EREMITE_EXPERTS") {
        args.extend(["--experts".to_string(), experts]);
    }
    args.extend(["--serve".to_string(), port.to_string()]);
    Server::start(&program, &args, &key, port)
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
