//! States saved on the SSD through HTTP (--state-disk, --state-dir,
//! --state-disk-tokens). A state evicted from memory is written to the SSD. A
//! later prompt that starts like it continues from it as it would from memory,
//! byte for byte in its greedy replies. A second server on the same folder runs
//! without the saved states.
//!
//! Set EREMITE_CHECKPOINT to the DeepSeek checkpoint directory and
//! EREMITE_EXPERTS optionally to GGUF experts. The tests run one model process
//! at a time, in temporary folders they remove:
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test state_disk -- --nocapture
//!
//! EREMITE_SERVER picks the server binary (default build/eremite-server).

use std::net::TcpListener;
use std::path::PathBuf;

use serde_json::{Value, json};

mod common;
use common::{Server, http, one_model, root};

/// What the tests run: the checkpoint, its experts and the server's program.
struct Setup {
    /// The checkpoint's directory.
    checkpoint: String,
    /// The GGUF experts, if any.
    experts: Option<String>,
    /// The server's program.
    server: PathBuf,
}

/// Reads the tests' setup from the environment. Returns None without
/// EREMITE_CHECKPOINT.
fn setup() -> Option<Setup> {
    let checkpoint = std::env::var("EREMITE_CHECKPOINT").ok()?;
    Some(Setup {
        checkpoint,
        experts: std::env::var("EREMITE_EXPERTS").ok(),
        server: std::env::var("EREMITE_SERVER")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root().join("build/eremite-server")),
    })
}

/// A temporary folder for a test's saved states, removed when dropped.
struct Folder(PathBuf);

impl Folder {
    /// Names the empty folder of the test `name`, and removes what an earlier
    /// run left there.
    fn new(name: &str) -> Folder {
        let dir =
            std::env::temp_dir().join(format!("eremite-state-disk-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Folder(dir)
    }

    /// Returns the folder's path as text.
    fn path(&self) -> &str {
        self.0.to_str().expect("a UTF-8 path")
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Starts a server of the checkpoint with saved states in `dir`, or none
/// without it. The states are written for 256 tokens of their own. `extra`
/// comes last.
fn start_server(setup: &Setup, dir: Option<&Folder>, extra: &[&str]) -> Server {
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("state-disk-test-{}-{port}-key", std::process::id());
    let mut args: Vec<String> = ["-m", &setup.checkpoint, "--memory", "56", "--ctx", "32768"]
        .map(String::from)
        .to_vec();
    if let Some(experts) = &setup.experts {
        args.extend(["--experts".to_string(), experts.clone()]);
    }
    match dir {
        Some(dir) => {
            args.extend(["--state-dir", dir.path(), "--state-disk-tokens", "256"].map(String::from))
        }
        None => args.extend(["--state-disk", "0"].map(String::from)),
    }
    args.extend(["--serve".to_string(), port.to_string()]);
    args.extend(extra.iter().map(|arg| arg.to_string()));
    Server::start(&setup.server, &args, &key, port)
}

/// Sends a greedy chat completion of `messages` and returns its answer, which
/// must be a success.
fn chat(server: &Server, messages: &Value, max_tokens: u32) -> Value {
    let body = json!({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
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
    serde_json::from_slice(&answer).unwrap_or(Value::Null)
}

/// Returns the prompt tokens that `answer` took from the cache, or -1.
fn hit(answer: &Value) -> i64 {
    answer["usage"]["prompt_cache_hit_tokens"]
        .as_i64()
        .unwrap_or(-1)
}

/// Returns the prompt tokens of `answer`, or -1.
fn prompt_tokens(answer: &Value) -> i64 {
    answer["usage"]["prompt_tokens"].as_i64().unwrap_or(-1)
}

/// Returns `answer`'s content.
fn content(answer: &Value) -> String {
    answer["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// Returns a conversation whose system prompt is about `chars` characters of
/// the repository's file `doc`, with the user's `question`.
fn conversation(doc: &str, chars: usize, question: &str) -> Value {
    let text = std::fs::read_to_string(root().join(doc)).expect("a document");
    let cut: String = text.chars().take(chars).collect();
    json!([{"role": "system", "content": format!("Answer from this reference.\n\n{cut}")},
           {"role": "user", "content": question}])
}

/// Returns the conversation `messages` continued: the reply in `answer`, and
/// one more question.
fn continued(messages: &Value, answer: &Value, question: &str) -> Value {
    let mut messages = messages.as_array().expect("messages").clone();
    messages.push(json!({"role": "assistant", "content": content(answer)}));
    messages.push(json!({"role": "user", "content": question}));
    Value::Array(messages)
}

/// Reads the saved states' count, saves and loads from the model's `status`, or
/// -1 for each.
fn disk(status: &Value) -> (i64, i64, i64) {
    let disk = &status["state_disk"];
    let read = |key: &str| disk[key].as_i64().unwrap_or(-1);
    (read("states"), read("saves"), read("loads"))
}

/// A state evicted for another one is saved. Its conversation's next turn reads
/// it back and continues as the state kept in memory does.
#[test]
fn an_evicted_state_continues_from_the_ssd_as_from_memory() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let first_conversation = conversation(
        "src/eremite.h",
        6000,
        "Which calls run on the engine thread?",
    );
    let second_conversation = conversation(
        "src/model/model.h",
        6000,
        "What does a model family fill in?",
    );
    // The reference: the state stays in memory.
    let (first, next, then, room) = {
        let server = start_server(&setup, None, &[]);
        let first = chat(&server, &first_conversation, 24);
        let room = server.settled(1)["state_cache_gib"]
            .as_f64()
            .expect("the cache's size");
        let next = continued(&first_conversation, &first, "And which run on the CPU?");
        let then = chat(&server, &next, 24);
        (first, next, then, room)
    };
    // Room for one such state: the second conversation's state evicts the
    // first's, and the first conversation's next turn evicts the second's.
    let folder = Folder::new("evict");
    let limit = format!("{:.4}", room * 1.5);
    let server = start_server(&setup, Some(&folder), &["--state-cache", &limit]);
    let first_again = chat(&server, &first_conversation, 24);
    assert_eq!(
        content(&first_again),
        content(&first),
        "the first turn as on the reference server"
    );
    chat(&server, &second_conversation, 8);
    let after_second = server.settled(2);
    let then_again = chat(&server, &next, 24);
    let end = server.settled(3);
    println!(
        "state-cache {limit} GiB: after the second conversation {:?} (states, saves, loads); the next turn reused {} of {} tokens (in memory: {}); {:?}",
        disk(&after_second),
        hit(&then_again),
        prompt_tokens(&then_again),
        hit(&then),
        disk(&end)
    );
    assert!(
        disk(&after_second).1 >= 1,
        "the first state was not saved when the second evicted it: {:?}",
        disk(&after_second)
    );
    assert!(
        disk(&end).2 >= 1,
        "the next turn did not read the saved state: {:?}",
        disk(&end)
    );
    assert_eq!(
        hit(&then_again),
        hit(&then),
        "the saved state resumes where the state in memory does"
    );
    assert_eq!(
        content(&then_again),
        content(&then),
        "the reply continues as from memory"
    );
}

/// One server uses a folder at a time: a second one there runs without saved
/// states. Neither loads its weights, since no request runs.
#[test]
fn a_second_server_on_the_same_folder_runs_without_saved_states() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("lock");
    let first = start_server(&setup, Some(&folder), &[]);
    let second = start_server(&setup, Some(&folder), &[]);
    assert!(
        first.status()["state_disk"].is_object(),
        "the first server has its saved states"
    );
    assert!(
        second.status()["state_disk"].is_null(),
        "the second runs without them"
    );
}
