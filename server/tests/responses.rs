//! The Responses API through HTTP (POST /v1/responses). A request gets its
//! reply as one Responses object: its output items hold the reply's text, and
//! its usage counts the input tokens.
//!
//! Set EREMITE_CHECKPOINT to the DeepSeek checkpoint directory and
//! EREMITE_EXPERTS optionally to GGUF experts. EREMITE_SERVER picks the server
//! binary (default build/eremite-server):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test responses -- --nocapture

use std::net::TcpListener;
use std::path::PathBuf;

use serde_json::{Value, json};

mod common;
use common::{Server, http, root};

/// Starts a server of the checkpoint `checkpoint`. It writes no saved states.
fn start_server(checkpoint: &str) -> Server {
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("responses-test-{}-{port}-key", std::process::id());
    let mut args: Vec<String> = [
        "-m",
        checkpoint,
        "--memory",
        "56",
        "--ctx",
        "32768",
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

#[test]
fn a_responses_request_gets_its_reply_as_a_response() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let server = start_server(&checkpoint);
    let body = json!({"model": "deepseek-v4.1-flash", "input": "Name a color. Answer in one word.",
                      "max_output_tokens": 64, "temperature": 0, "reasoning": {"effort": "none"},
                      "stream": false});
    let (code, answer) = http(
        server.port,
        &server.key,
        "POST",
        "/v1/responses",
        Some(&body),
    );
    let answer: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
    assert_eq!(code, 200, "{answer}");
    assert!(
        answer["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("resp_")),
        "{answer}"
    );
    let text: String = answer["output"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter_map(|part| part["text"].as_str())
        .collect();
    println!(
        "the response's text: {text:?}; its usage: {}",
        answer["usage"]
    );
    assert!(!text.trim().is_empty(), "no text: {answer}");
    assert!(
        answer["usage"]["input_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0),
        "{answer}"
    );
    // The path without /v1 is the same route. It takes POST only.
    let (code, _) = http(server.port, &server.key, "GET", "/responses", None);
    assert_eq!(code, 405);
}
