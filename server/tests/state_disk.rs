//! States saved on the SSD through HTTP (state_disk, state_dir,
//! state_disk_tokens). A state evicted from memory, or idle at quit, is
//! written to the SSD. A later prompt that starts like it continues from
//! it as it would from memory, byte for byte in its greedy replies. A saved
//! state damaged after the start is refused when it is read: the prompt is
//! prefilled cold and the saved state goes. Saved states stay within their
//! budget, the most recently used kept. A second server on the same folder
//! runs without them.
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
use common::{Server, add_keys, build_config, http, one_model, root};

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

    /// Lists the files in it whose names end in `suffix`.
    fn files(&self, suffix: &str) -> Vec<PathBuf> {
        std::fs::read_dir(&self.0)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.to_string_lossy().ends_with(suffix))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Starts a server of the checkpoint with saved states in `dir`, or none
/// without it. The states are written for 256 tokens of their own.
fn start_server(setup: &Setup, dir: Option<&Folder>) -> Server {
    start_server_with(setup, dir, json!({}), json!({}))
}

/// Starts a server of the checkpoint with saved states in `dir`, or none
/// without it, and with the server-wide settings `server` and the model's
/// settings `model` in place of those of the same keys. The states are written
/// for 256 tokens of their own.
fn start_server_with(setup: &Setup, dir: Option<&Folder>, server: Value, model: Value) -> Server {
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("state-disk-test-{}-{port}-key", std::process::id());
    let mut settings = match dir {
        Some(dir) => json!({"memory": 56, "state_dir": dir.path(), "state_disk_tokens": 256}),
        None => json!({"memory": 56, "state_disk": 0}),
    };
    add_keys(&mut settings, server);
    let mut model_settings = json!({"ctx": 32768});
    add_keys(&mut model_settings, model);
    let config = build_config(
        port,
        settings,
        &setup.checkpoint,
        setup.experts.as_deref(),
        model_settings,
    );
    Server::start(&setup.server, &config, &key, port)
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
    let (first, next, then, room, required) = {
        let server = start_server(&setup, None);
        let first = chat(&server, &first_conversation, 24);
        let room = server.settled(1)["state_cache_gib"]
            .as_f64()
            .expect("the cache's size");
        // What a full prompt pass of the loaded model needs, in GiB.
        let required = server.status()["own_memory"]["required"]
            .as_f64()
            .expect("the memory that a full prompt pass needs")
            / (1u64 << 30) as f64;
        let next = continued(&first_conversation, &first, "And which run on the CPU?");
        let then = chat(&server, &next, 24);
        (first, next, then, room, required)
    };
    // Room for one such state: the second conversation's state evicts the
    // first's, and the first conversation's next turn evicts the second's. The
    // model's own memory that much above what a full prompt pass needs gives
    // its states that room.
    let folder = Folder::new("evict");
    let limit = room * 1.5;
    let server = start_server_with(
        &setup,
        Some(&folder),
        json!({}),
        json!({"memory": required + limit}),
    );
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
        "a state room of {limit} GiB: after the second conversation {:?} (states, saves, loads); the next turn reused {} of {} tokens (in memory: {}); {:?}",
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

/// The states idle at quit are saved. After a restart the same prompt is
/// answered from its saved prompt end, without a prefill.
#[test]
fn a_state_saved_at_quit_answers_its_prompt_again_after_a_restart() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("quit");
    let messages = conversation(
        "src/engine/expert_store.h",
        6000,
        "When does the store read ahead?",
    );
    let first = {
        let server = start_server(&setup, Some(&folder));
        let first = chat(&server, &messages, 24);
        server.settled(1);
        server.stop();
        first
    };
    assert_eq!(
        folder.files(".state").len(),
        1,
        "one saved state after the quit"
    );
    let server = start_server(&setup, Some(&folder));
    let start = server.status();
    let again = chat(&server, &messages, 24);
    let end = server.settled(1);
    println!(
        "after the restart: {:?} (states, saves, loads); the prompt again reused {} of {} tokens; last prefill {}",
        disk(&start),
        hit(&again),
        prompt_tokens(&again),
        end["last_prefill"]["tokens"]
    );
    assert_eq!(
        disk(&start).0,
        1,
        "the restarted server reads the saved state's index"
    );
    assert_eq!(
        hit(&again),
        prompt_tokens(&again),
        "the whole prompt from the saved prompt end"
    );
    assert_eq!(
        content(&again),
        content(&first),
        "the same reply as before the quit"
    );
    assert_eq!(disk(&end).2, 1);
}

/// A part damaged after the start is refused when it is read. The prompt is
/// prefilled cold and gives the cold reply, and the saved state is deleted.
#[test]
fn a_saved_state_damaged_after_the_start_is_refused_when_read() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("damage");
    let messages = conversation("src/eremite.h", 5000, "What does a stream do?");
    let first = {
        let server = start_server(&setup, Some(&folder));
        let first = chat(&server, &messages, 24);
        server.settled(1);
        server.stop();
        first
    };
    let server = start_server(&setup, Some(&folder));
    assert_eq!(disk(&server.status()).0, 1);
    for part in folder.files(".rows") {
        let mut bytes = std::fs::read(&part).expect("a part");
        // A byte in the part's first region of rows.
        bytes[16384 + 100] ^= 0x40;
        std::fs::write(&part, bytes).expect("damaged");
    }
    let again = chat(&server, &messages, 24);
    let end = server.settled(1);
    println!(
        "damaged: reused {} of {}; {:?}",
        hit(&again),
        prompt_tokens(&again),
        disk(&end)
    );
    assert_eq!(hit(&again), 0, "a cold prefill");
    assert_eq!(content(&again), content(&first), "the cold reply");
    assert_eq!(disk(&end).0, 0, "the damaged saved state is deleted");
    assert!(folder.files(".state").is_empty());
}

/// Saved states stay within state_disk. At quit the most recently used are
/// written first, and an older one does not take their room.
#[test]
fn saved_states_stay_within_their_budget_most_recent_first() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("budget");
    let prompts = [
        conversation("src/eremite.h", 4000, "Summarize the first section."),
        conversation("src/model/model.h", 4000, "Summarize the first section."),
        conversation(
            "src/engine/expert_store.h",
            4000,
            "Summarize the first section.",
        ),
    ];
    // What each saved state takes, all three written. They differ: the second
    // conversation also keeps a snapshot where it parts from the first. At quit
    // the most recently used is written first, so its number is the lowest.
    let sizes: Vec<f64> = {
        let server = start_server(&setup, Some(&folder));
        for prompt in &prompts {
            chat(&server, prompt, 1);
        }
        server.settled(3);
        server.stop();
        let mut bytes = std::collections::BTreeMap::<u64, u64>::new();
        for entry in std::fs::read_dir(&folder.0).expect("the folder").flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name
                .split(['-', '.'])
                .next()
                .and_then(|number| number.parse().ok())
            {
                *bytes.entry(id).or_insert(0) +=
                    entry.metadata().map_or(0, |metadata| metadata.len());
            }
        }
        let _ = std::fs::remove_dir_all(&folder.0);
        bytes
            .values()
            .map(|&size| size as f64 / (1u64 << 30) as f64)
            .collect()
    };
    assert_eq!(sizes.len(), 3, "all three saved: {sizes:?}");
    // Room for the two most recent.
    let limit = sizes[0] + sizes[1] + sizes[2] / 2.0;
    {
        let server = start_server_with(
            &setup,
            Some(&folder),
            json!({"state_disk": limit}),
            json!({}),
        );
        for prompt in &prompts {
            chat(&server, prompt, 1);
        }
        server.settled(3);
        server.stop();
    }
    let server = start_server_with(
        &setup,
        Some(&folder),
        json!({"state_disk": limit}),
        json!({}),
    );
    let start = server.status();
    let hits: Vec<i64> = prompts
        .iter()
        .map(|prompt| hit(&chat(&server, prompt, 1)))
        .collect();
    println!(
        "saved states of {sizes:.4?} GiB, state_disk {limit}: {:?} at the start; the prompts again reused {hits:?}",
        disk(&start)
    );
    assert_eq!(disk(&start).0, 2, "two saved states fit");
    let gib = start["state_disk"]["gib"].as_f64().unwrap_or(f64::NAN);
    assert!(
        gib <= start["state_disk"]["limit_gib"].as_f64().unwrap_or(0.0),
        "{gib} GiB"
    );
    // Each state's own tokens are nearly all of its prompt. The prompts share
    // only their first few tokens with each other.
    assert!(
        hits[1] >= 256 && hits[2] >= 256,
        "the two most recent are kept: {hits:?}"
    );
    assert!(hits[0] < 256, "the oldest is not: {hits:?}");
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
    let first = start_server(&setup, Some(&folder));
    let second = start_server(&setup, Some(&folder));
    assert!(
        first.status()["state_disk"].is_object(),
        "the first server has its saved states"
    );
    assert!(
        second.status()["state_disk"].is_null(),
        "the second runs without them"
    );
}
