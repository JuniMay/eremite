//! Drafts and context-copy proposals through HTTP (--draft, --lookup). A greedy
//! reply with them has the tokens of the reply without them. The log counts
//! the passes that ran them and the drafts that the passes kept.
//!
//! Set EREMITE_CHECKPOINT to the DeepSeek checkpoint directory and
//! EREMITE_EXPERTS optionally to GGUF experts. EREMITE_SERVER picks the server
//! binary (default build/eremite-server):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test drafts -- --nocapture

use std::net::TcpListener;
use std::path::PathBuf;

use serde_json::{Value, json};

mod common;
use common::{Server, add_keys, build_config, http, one_model, root};

/// The tokens that each reply has at most.
const MAX_TOKENS: u32 = 96;

/// Starts a server of the checkpoint `checkpoint` with the model settings
/// `model`, besides a context of 32768 tokens. It writes no saved states.
fn start_server(checkpoint: &str, model: Value) -> Server {
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("drafts-test-{}-{port}-key", std::process::id());
    let experts = std::env::var("EREMITE_EXPERTS").ok();
    let mut settings = json!({"ctx": 32768});
    add_keys(&mut settings, model);
    let config = build_config(
        port,
        json!({"memory": 56, "state_disk": 0}),
        checkpoint,
        experts.as_deref(),
        settings,
    );
    Server::start(&program, &config, &key, port)
}

/// Sends a greedy chat request of `messages` with thinking off, and returns the
/// reply's content. The request must succeed.
fn reply(server: &Server, messages: &Value) -> String {
    let body = json!({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": MAX_TOKENS,
                      "temperature": 0, "thinking": false, "stream": false});
    let (status, answer) = http(
        server.port,
        &server.key,
        "POST",
        "/v1/chat/completions",
        Some(&body),
    );
    assert_eq!(
        status,
        200,
        "chat completion: {}",
        String::from_utf8_lossy(&answer)
    );
    let answer: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
    answer["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// Waits for the log line of request 1 that contains `label`, and returns it.
fn read_line(server: &Server, label: &str) -> String {
    server.wait_for_line(0, |line| line.contains("req 1: ") && line.contains(label))
}

/// Returns the number in `line` just before `word`, or None.
fn read_count_before(line: &str, word: &str) -> Option<u64> {
    let before = &line[..line.find(word)?];
    before.split_whitespace().last()?.parse().ok()
}

/// The reply copies text from the prompt, so the lookup finds proposals there.
/// The draft head drafts in every pass that the draft choice gives drafts; its
/// first passes do.
#[test]
fn drafts_and_proposals_keep_the_greedy_reply() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let messages = json!([{"role": "user", "content":
        "Copy this text twice, exactly, and write nothing else:\n\nThe engine thread runs every \
         model's calls. A stream holds the buffers of one call, and the worker reuses it."}]);
    let plain = reply(&start_server(&checkpoint, json!({})), &messages);
    let lookup = start_server(&checkpoint, json!({"lookup": 15}));
    let with_lookup = reply(&lookup, &messages);
    let proposals = read_line(&lookup, "context lookup");
    drop(lookup);
    let drafts = start_server(&checkpoint, json!({"draft": 2}));
    let with_drafts = reply(&drafts, &messages);
    let drafted = read_line(&drafts, "draft passes");
    drop(drafts);
    println!("{proposals}\n{drafted}");
    assert_eq!(with_lookup, plain, "the lookup changed the reply");
    assert_eq!(with_drafts, plain, "the drafts changed the reply");
    assert!(
        read_count_before(&proposals, " passes").is_some_and(|passes| passes > 0)
            && read_count_before(&proposals, " accepted").is_some_and(|accepted| accepted > 0),
        "the lookup's proposals were not kept: {proposals}"
    );
    assert!(
        read_count_before(&drafted, " draft passes").is_some_and(|passes| passes > 0),
        "no pass ran drafts: {drafted}"
    );
}
