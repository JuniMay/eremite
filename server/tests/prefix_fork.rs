//! Requests through HTTP to a DeepSeek checkpoint.
//!
//! A prompt too long for the context is refused with its token count. The
//! model's status follows a running prefill, and quitting stops a long prefill.
//! A request is answered while four others decode.
//!
//! Forks of a conversation reuse its history up to where they part from it,
//! also while the conversation is still decoding. The cache resumes each fork
//! the cheaper way: a replay at the shared prefix, or a snapshot before it that
//! leaves fewer rows to compute. So a fork reuses at least its shared prefix
//! less what a replay there computes again, and never more than the prefix.
//! Without ced, each fork must give the tokens that the same prompt gives a
//! server that starts without a cache. With ced, forks are approximate by
//! design: the replay is bounded to a ring, and their tokens are printed. The
//! other cache tests check a model's state room.
//!
//! A growing generation frees room in the cache before it grows. A session
//! stays while its idle model unloads (idle_unload).
//!
//! Set EREMITE_CHECKPOINT to the checkpoint directory. EREMITE_EXPERTS
//! optionally selects GGUF experts; otherwise the tests use native MXFP4. The
//! tests run one model process at a time (they share a lock):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test prefix_fork -- --nocapture
//!
//! EREMITE_SERVER picks the server binary (default build/eremite-server) and
//! EREMITE_CLI the tokenizer's (default build/eremite). The fork tests print
//! every fork's shared and reported reuse and fail if any lies outside those
//! bounds.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::time::Instant;

use serde_json::{Value, json};

mod common;
use common::{Server, add_keys, build_config, http, one_model, root};
use eremite_server::chat::deepseek;

/// The requests that decode while another request arrives.
const DECODING_REQUESTS: usize = 4;
/// The tokens that each decoding request asks for. 500 tokens take about 45 s
/// with IQ2 experts and 100 s with MXFP4 experts. The arriving request's answer
/// takes less than 20 s.
const DECODING_TOKENS: u32 = 500;

/// The positions of a layer's ring: the window's 128 and the rewind margin of
/// 15, with the default lookup's 15 proposals.
const RING_POSITIONS: usize = 143;

/// Returns at least the rows that an exact replay at `k` computes again, on
/// average per layer (the cache's cost of a replay, eremite_replay_rows). Layer
/// 39 starts a ring before `k`, or one position earlier for a ratio-2
/// compressor's pair. Each layer below starts at most 128 positions earlier:
/// 127, plus a pair.
fn replay_rows(k: usize) -> usize {
    (0..40)
        .map(|layer| k.min(RING_POSITIONS + 1 + 128 * layer))
        .sum::<usize>()
        .div_ceil(40)
}

/// Returns what a bounded replay at `k` computes again (ced): a ring's
/// positions.
fn bounded_rows(k: usize) -> usize {
    k.min(RING_POSITIONS)
}

/// What the tests run: the checkpoint, its experts, and the server's and the
/// CLI's programs.
struct Setup {
    /// The checkpoint's directory.
    checkpoint: String,
    /// The GGUF experts, if any.
    experts: Option<String>,
    /// The server's program.
    server: PathBuf,
    /// The CLI's program, for its tokenizer.
    cli: PathBuf,
}

/// Reads the tests' setup from the environment. Returns None without
/// EREMITE_CHECKPOINT.
fn setup() -> Option<Setup> {
    let checkpoint = std::env::var("EREMITE_CHECKPOINT").ok()?;
    let experts = std::env::var("EREMITE_EXPERTS").ok();
    let pick = |var: &str, default: &str| {
        std::env::var(var)
            .map(PathBuf::from)
            .unwrap_or_else(|_| root().join(default))
    };
    Some(Setup {
        checkpoint,
        experts,
        server: pick("EREMITE_SERVER", "build/eremite-server"),
        cli: pick("EREMITE_CLI", "build/eremite"),
    })
}

// ---- HTTP/1.1: a streamed chat request ----

/// Sends a streamed chat request and reads its answer to the end. When the
/// answer's first token arrives, it sends on `first_token`.
fn stream_chat(port: u16, key: &str, body: &Value, first_token: mpsc::Sender<()>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let sent_body = body.to_string();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        sent_body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(sent_body.as_bytes()))
        .expect("send");
    // A token's chunk holds "content":" and then a character other than the
    // closing quote.
    let token = b"\"content\":\"";
    let (mut received, mut buffer, mut sent) = (Vec::new(), [0u8; 4096], false);
    loop {
        let count = stream.read(&mut buffer).expect("read");
        if count == 0 {
            return;
        }
        received.extend_from_slice(&buffer[..count]);
        if !sent
            && received
                .windows(token.len() + 1)
                .any(|window| window.starts_with(token) && window[token.len()] != b'"')
        {
            first_token
                .send(())
                .expect("the test waits for the first token");
            sent = true;
        }
    }
}

/// Starts a server of the checkpoint with the tests' settings.
fn start_server(setup: &Setup) -> Server {
    start_server_with(setup, json!({}), json!({}))
}

/// Starts a server of the checkpoint with the tests' settings, and with the
/// server-wide settings `server` and the model's settings `model` in place of
/// those of the same keys.
fn start_server_with(setup: &Setup, server: Value, model: Value) -> Server {
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let key = format!(
        "fork-test-{}-{:x}",
        std::process::id(),
        nanos ^ (port as u128) << 64
    );
    // No saved states: a cold server stays cold, and nothing is written to the
    // SSD.
    let mut settings = json!({"memory": 56, "state_disk": 0});
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

/// Sends a chat completion and returns its answer, which must be a success.
fn chat(server: &Server, body: &Value) -> Value {
    let (status, answer) = http(
        server.port,
        &server.key,
        "POST",
        "/v1/chat/completions",
        Some(body),
    );
    let parsed: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
    assert_eq!(
        status,
        200,
        "chat completion: {}",
        String::from_utf8_lossy(&answer)
    );
    parsed
}

// ---- prompt tokens from the server's renderer and the CLI's tokenizer ----

/// Returns the tokens of `messages`' prompt, with thinking off.
fn tokens(setup: &Setup, messages: &Value) -> Vec<i64> {
    rendered_tokens(setup, messages, false)
}

/// Returns the tokens of `messages`' prompt, with thinking on or off.
fn rendered_tokens(setup: &Setup, messages: &Value, thinking: bool) -> Vec<i64> {
    let mode = if thinking { "thinking" } else { "chat" };
    let prompt = deepseek::render_test_case(&json!({"messages": messages, "thinking_mode": mode}))
        .unwrap_or_else(|error| panic!("render: {}", error.message));
    let ids = Command::new(&setup.cli)
        .args([
            "-m",
            &setup.checkpoint,
            "--rendered",
            "--ctx",
            "32768",
            "--tokenize",
            "-p",
        ])
        .arg(prompt.trim_end_matches('\n'))
        .output()
        .expect("tokenize");
    assert!(
        ids.status.success(),
        "tokenize: {}",
        String::from_utf8_lossy(&ids.stderr)
    );
    String::from_utf8_lossy(&ids.stdout)
        .trim()
        .split(',')
        .filter_map(|token| token.parse().ok())
        .collect()
}

/// Counts the tokens at the start of `first` and `second` that are the same.
fn count_shared(first: &[i64], second: &[i64]) -> usize {
    first
        .iter()
        .zip(second)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Builds a greedy chat request of `messages` with thinking off, for at most
/// `max_tokens`.
fn request(messages: &Value, max_tokens: u32) -> Value {
    body(messages, max_tokens, false)
}

/// Builds a greedy chat request of `messages`, for at most `max_tokens`, with
/// thinking on or off.
fn body(messages: &Value, max_tokens: u32, thinking: bool) -> Value {
    json!({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
           "temperature": 0, "thinking": thinking, "stream": false})
}

/// Returns the prompt tokens that `answer` took from the cache, or -1.
fn hit(answer: &Value) -> i64 {
    answer["usage"]["prompt_cache_hit_tokens"]
        .as_i64()
        .unwrap_or(-1)
}

/// Returns `answer`'s content.
fn content(answer: &Value) -> String {
    answer["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// Returns the reply's reasoning and its content.
fn reply(answer: &Value) -> String {
    let message = &answer["choices"][0]["message"];
    format!(
        "{}|{}",
        message["reasoning_content"].as_str().unwrap_or(""),
        message["content"].as_str().unwrap_or("")
    )
}

/// Returns the rows that the server's last prefill computed (the model's
/// status): the prompt's new tokens, and with a replay also the earlier rows it
/// computed again.
fn last_prefill(server: &Server) -> i64 {
    server.status()["last_prefill"]["tokens"]
        .as_i64()
        .unwrap_or(-1)
}

/// Returns the conversation's text, from the repository's own documents: a long
/// system prompt and a long user turn.
fn texts() -> (String, String) {
    let design = std::fs::read_to_string(root().join("src/eremite.h")).expect("src/eremite.h");
    let usage =
        std::fs::read_to_string(root().join("src/model/model.h")).expect("src/model/model.h");
    let cut = |text: &str, chars: usize| text.chars().take(chars).collect::<String>();
    (
        format!(
            "You review this project's design. Reference:\n\n{}",
            cut(&design, 15000)
        ),
        cut(&usage, 4000),
    )
}

/// Returns character offsets in a text of `len` characters: fixed ones near the
/// ends, and `random` ones from `seed`.
fn offsets(len: usize, random: usize, seed: u64) -> Vec<usize> {
    let mut offsets = vec![1, len / 3, len / 2, len.saturating_sub(2)];
    let mut state = seed;
    for _ in 0..random {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        offsets.push(1 + (state as usize) % len.saturating_sub(2).max(1));
    }
    offsets.sort();
    offsets.dedup();
    offsets
}

/// Returns the conversation, a long system prompt and a long user turn, and its
/// forks. Each fork cuts the user turn or the system prompt at an offset and
/// continues differently.
fn conversation_and_forks() -> (Value, Vec<(String, Value)>) {
    let (system, user) = texts();
    let tail =
        " Leave the rest aside and answer in one short sentence: what does this part describe?";
    let conversation =
        json!([{"role": "system", "content": system}, {"role": "user", "content": user}]);
    let mut forks: Vec<(String, Value)> = Vec::new();
    for offset in offsets(user.chars().count(), 8, 0x9E37_79B9) {
        let cut: String = user.chars().take(offset).collect();
        forks.push((
            format!("user turn at character {offset}"),
            json!([{"role": "system", "content": system}, {"role": "user", "content": cut + tail}]),
        ));
    }
    for offset in offsets(system.chars().count(), 4, 0x85EB_CA6B) {
        let cut: String = system.chars().take(offset).collect();
        forks.push((
            format!("system prompt at character {offset}"),
            json!([{"role": "system", "content": cut + tail}, {"role": "user", "content": user}]),
        ));
    }
    (conversation, forks)
}

/// A fork's outcome on the cached server.
struct Forked {
    /// Where the fork parts from the conversation.
    what: String,
    /// The fork's messages.
    messages: Value,
    /// The tokens that the fork shares with the conversation.
    shared: usize,
    /// The tokens that the server reused.
    reused: i64,
    /// The fork's reply.
    text: String,
}

/// Runs the forks, one after another on one server with the model settings
/// `model`, while the conversation decodes there. A fork whose reuse lies
/// outside [shared - rows(shared), shared] is added to `wrong`.
fn run_forks(
    setup: &Setup,
    model: Value,
    rows: fn(usize) -> usize,
    wrong: &mut Vec<String>,
) -> Vec<Forked> {
    let (conversation, forks) = conversation_and_forks();
    let source = tokens(setup, &conversation);
    let mut results = Vec::new();
    let server = start_server_with(setup, json!({}), model);
    let port = server.port;
    let key = server.key.clone();
    let mut long = request(&conversation, 6000);
    long["stream"] = json!(true);
    let (first_token, first_tokens) = mpsc::channel();
    let busy = std::thread::spawn(move || stream_chat(port, &key, &long, first_token));
    // Past the conversation's prefill: it decodes.
    first_tokens.recv().expect("the conversation's first token");
    for (what, messages) in forks {
        let shared = count_shared(&tokens(setup, &messages), &source);
        let answer = chat(&server, &request(&messages, 16));
        let (reused, least) = (hit(&answer), shared - rows(shared));
        println!("{what}: shares {shared} tokens, reused {reused} (at least {least})");
        if reused < least as i64 || reused > shared as i64 {
            wrong.push(format!(
                "{what}: shares {shared} tokens, reused {reused}; at least {least} expected"
            ));
        }
        results.push(Forked {
            what,
            messages,
            shared,
            reused,
            text: content(&answer),
        });
    }
    // Dropping the server ends the decoding request too.
    drop(server);
    let _ = busy.join();
    results
}

#[test]
fn forks_reuse_the_shared_prefix_and_give_the_cold_tokens() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let mut wrong = Vec::new();
    let results = run_forks(&setup, json!({}), replay_rows, &mut wrong);
    // The same prompts on servers that start without a cache.
    for fork in &results {
        let cold = start_server(&setup);
        let reference = content(&chat(&cold, &request(&fork.messages, 16)));
        drop(cold);
        let same = reference == fork.text;
        println!(
            "{} (shares {}, reused {}): tokens {}",
            fork.what,
            fork.shared,
            fork.reused,
            if same {
                "equal to the cold server's"
            } else {
                "DIFFERENT from the cold server's"
            }
        );
        if !same {
            wrong.push(format!(
                "{}: {:?} against the cold server's {reference:?}",
                fork.what, fork.text
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "forks that reused too little or changed the output:\n{}",
        wrong.join("\n")
    );
}

/// With ced 128 a fork resumes by a bounded replay or by a snapshot at most a
/// ring before its shared prefix. A bounded replay computes a ring's positions
/// again, with their windows cut off there.
#[test]
fn ced_forks_reuse_the_shared_prefix_less_a_ring() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let mut wrong = Vec::new();
    for fork in run_forks(&setup, json!({"ced": 128}), bounded_rows, &mut wrong) {
        println!("{}: {:?}", fork.what, fork.text);
    }
    assert!(
        wrong.is_empty(),
        "forks that reused too little:\n{}",
        wrong.join("\n")
    );
}

/// A thinking prompt ends with <think>. The next turn renders that reply with
/// </think> there. So it shares all of the prompt but its last token. It
/// resumes from the turn's prompt-end snapshot one position back, without
/// computing any earlier rows again, with ced too. Without ced it gives the
/// tokens that a server starting without a cache gives.
#[test]
fn a_thinking_turn_s_next_turn_resumes_one_position_before_its_prompt_end() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let (system, user) = texts();
    let first = json!([{"role": "system", "content": system}, {"role": "user", "content": user}]);
    let second = json!([{"role": "system", "content": system}, {"role": "user", "content": user},
                        {"role": "assistant", "content": "It describes how the server is started and checked."},
                        {"role": "user", "content": "Which of those options change the output?"}]);
    let (first_tokens, second_tokens) = (
        rendered_tokens(&setup, &first, true),
        rendered_tokens(&setup, &second, true),
    );
    let shared = count_shared(&first_tokens, &second_tokens);
    assert_eq!(
        shared,
        first_tokens.len() - 1,
        "the next turn parts at the first turn's last token"
    );
    let mut wrong = Vec::new();
    for model in [json!({}), json!({"ced": 128})] {
        let without_ced = model.get("ced").is_none();
        let what = if without_ced {
            "without ced"
        } else {
            "with ced 128"
        };
        let server = start_server_with(&setup, json!({}), model);
        chat(&server, &body(&first, 16, true));
        let answer = chat(&server, &body(&second, 16, true));
        let (reused, computed) = (hit(&answer), last_prefill(&server));
        println!(
            "{what}: the next turn shares {shared} of its {} tokens, reused {reused}, computed {computed} rows",
            second_tokens.len()
        );
        if answer["usage"]["prompt_tokens"].as_i64() != Some(second_tokens.len() as i64) {
            wrong.push(format!(
                "{what}: the server's prompt has {} tokens, the test's {}",
                answer["usage"]["prompt_tokens"],
                second_tokens.len()
            ));
        }
        if reused != shared as i64 || computed != (second_tokens.len() - shared) as i64 {
            wrong.push(format!(
                "{what}: reused {reused} and computed {computed} rows; {shared} and {} expected",
                second_tokens.len() - shared
            ));
        }
        if without_ced {
            let text = reply(&answer);
            drop(server);
            let cold = start_server(&setup);
            let reference = reply(&chat(&cold, &body(&second, 16, true)));
            println!(
                "{what}: tokens {}",
                if text == reference {
                    "equal to the cold server's"
                } else {
                    "DIFFERENT from the cold server's"
                }
            );
            if text != reference {
                wrong.push(format!(
                    "{what}: {text:?} against the cold server's {reference:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the next thinking turn:\n{}",
        wrong.join("\n")
    );
}

// ---- an image in the prompt ----

/// Encodes `data` as base64.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let bits = (chunk[0] as u32) << 16
            | (*chunk.get(1).unwrap_or(&0) as u32) << 8
            | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= chunk.len() {
                ALPHABET[(bits >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

/// Returns the CRC-32 of `data`, as PNG chunks carry it.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Returns a small original image as a data URL: a 32 x 32 PNG in four colored
/// quarters (red, green, blue, yellow), its pixels in one stored deflate block.
fn image_url() -> String {
    let (width, height) = (32u32, 32u32);
    let mut pixels = Vec::new();
    for y in 0..height {
        // No filter.
        pixels.push(0);
        for x in 0..width {
            pixels.extend(match (x < width / 2, y < height / 2) {
                (true, true) => [220, 40, 40],
                (false, true) => [40, 200, 40],
                (true, false) => [40, 60, 220],
                (false, false) => [230, 220, 30],
            });
        }
    }
    // Adler-32.
    let (mut low, mut high) = (1u32, 0u32);
    for &pixel in &pixels {
        low = (low + pixel as u32) % 65521;
        high = (high + low) % 65521;
    }
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend((pixels.len() as u16).to_le_bytes());
    zlib.extend((!(pixels.len() as u16)).to_le_bytes());
    zlib.extend(&pixels);
    zlib.extend((high << 16 | low).to_be_bytes());
    let chunk = |png: &mut Vec<u8>, kind: &[u8], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        let body: Vec<u8> = kind.iter().chain(data).copied().collect();
        png.extend(&body);
        png.extend(crc32(&body).to_be_bytes());
    };
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    // 8-bit RGB.
    header.extend([8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    format!("data:image/png;base64,{}", base64(&png))
}

/// A fork after an image reuses the rows through the image. It gives the tokens
/// that a server starting without a cache gives: the image's rows are the same,
/// also those that an exact replay at the fork computes again.
#[test]
fn image_forks_give_the_cold_tokens() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let (system, _) = texts();
    let image = image_url();
    let turn = |question: &str| {
        json!([{"role": "system", "content": system},
               {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                            {"type": "text", "text": question}]}])
    };
    let first = turn("Describe the colors of this picture in one sentence.");
    let fork = turn("Which color is in the top left quarter? Answer in one word.");
    let (reused, text) = {
        let server = start_server(&setup);
        chat(&server, &request(&first, 8));
        let answer = chat(&server, &request(&fork, 16));
        (hit(&answer), content(&answer))
    };
    let reference = {
        let cold = start_server(&setup);
        content(&chat(&cold, &request(&fork, 16)))
    };
    println!(
        "image fork: reused {reused}; tokens {}",
        if text == reference {
            "equal to the cold server's"
        } else {
            "DIFFERENT"
        }
    );
    assert!(reused > 0, "the fork after the image reused nothing");
    assert_eq!(
        text, reference,
        "the fork after the image gives other tokens than a cold server"
    );
}

/// A prompt too long for the context is refused with its token count. With an
/// image, the count includes the image's span. So it is more than that of the
/// same text alone.
#[test]
fn a_prompt_with_an_image_that_does_not_fit_is_refused_with_its_token_count() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let (system, _) = texts();
    let server = start_server_with(&setup, json!({}), json!({"ctx": 1024}));
    let refused_tokens = |content: Value| {
        let messages = json!([{"role": "user", "content": content}]);
        let (status, answer) = http(
            server.port,
            &server.key,
            "POST",
            "/v1/chat/completions",
            Some(&request(&messages, 4)),
        );
        let message = serde_json::from_slice::<Value>(&answer)
            .ok()
            .and_then(|error| error["error"]["message"].as_str().map(String::from))
            .unwrap_or_default();
        assert_eq!(status, 400, "a prompt longer than the context: {message}");
        println!("refused: {message}");
        message
            .strip_prefix("the prompt has ")
            .and_then(|rest| rest.split(' ').next())
            .and_then(|count| count.parse::<usize>().ok())
            .unwrap_or_else(|| panic!("no token count in the refusal: {message}"))
    };
    let text_alone = refused_tokens(json!(system));
    let with_image = refused_tokens(json!([
        {"type": "text", "text": system},
        {"type": "image_url", "image_url": {"url": image_url()}}
    ]));
    assert!(
        with_image > text_alone,
        "the prompt with an image is refused with {with_image} tokens, no more than the \
         {text_alone} of its text alone"
    );
}

// ---- a model's state room ----

/// What the model's status tells of the cache.
struct Status {
    /// The GiB that the cache counts.
    states: f64,
    /// The GiB that the model's states may take: its state room.
    limit: f64,
    /// What the model does.
    phase: String,
    /// The positions of each state.
    sessions: Vec<i64>,
}

/// Reads what the server's status tells of the cache.
fn status(server: &Server) -> Status {
    read_status(&server.status())
}

/// Reads what the model's status `status` tells of the cache.
fn read_status(status: &Value) -> Status {
    Status {
        states: status["state_cache_gib"].as_f64().unwrap_or(f64::NAN),
        limit: status["state_cache_limit_gib"].as_f64().unwrap_or(f64::NAN),
        phase: status["phase"].as_str().unwrap_or("").to_string(),
        sessions: status["sessions"]
            .as_array()
            .map(|sessions| {
                sessions
                    .iter()
                    .filter_map(|session| session["tokens"].as_i64())
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Reads the cache's status once `requests` requests have ended and the server
/// is idle.
fn settled(server: &Server, requests: i64) -> Status {
    read_status(&server.settled(requests))
}

/// The bytes in one GiB.
const GIB: f64 = (1u64 << 30) as f64;

/// Returns the GiB that a full prompt pass of the server's loaded model needs,
/// as its status tells it. A model whose own memory is that much above a room
/// has that room for its states.
fn required_gib(server: &Server) -> f64 {
    let required = server.status()["own_memory"]["required"].as_f64();
    required.expect("the memory that a full prompt pass needs") / GIB
}

/// Returns a short conversation with a one-word answer.
fn short_prompt() -> Value {
    json!([{"role": "system", "content": "You answer in one word."},
           {"role": "user", "content": "Name a color."}])
}

/// A retry of a saved prompt resumes at its prompt end from the snapshot there.
/// It takes no snapshot of its own. So it fits a cache that the first try
/// filled to within less than a snapshot (5.3 MiB) of its state room.
#[test]
fn a_saved_prompt_again_fits_a_full_state_cache() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let prompt = short_prompt();
    let count = tokens(&setup, &prompt).len() as i64;
    // What the first try takes: its state and its prompt-end snapshot.
    let (first, required) = {
        let server = start_server(&setup);
        chat(&server, &request(&prompt, 1));
        (settled(&server, 1).states, required_gib(&server))
    };
    let limit = first + 2.0 / 1024.0;
    let server = start_server_with(&setup, json!({}), json!({"memory": required + limit}));
    chat(&server, &request(&prompt, 1));
    let again = chat(&server, &request(&prompt, 1));
    let cache = settled(&server, 2);
    println!(
        "saved prompt again: the first try took {first:.3} GiB, a state room of {limit} GiB; the retry reused {} of {count}; the cache takes {:.3} GiB",
        hit(&again),
        cache.states
    );
    assert_eq!(
        hit(&again),
        count,
        "the retry did not resume at its prompt end"
    );
    assert!(
        cache.states <= cache.limit,
        "the cache takes {:.3} GiB of {:.3}",
        cache.states,
        cache.limit
    );
}

/// A prompt that shares only its first tokens with the one idle state, in a
/// cache with room for one state, runs in that state's place. Reusing those
/// tokens would keep the state, and the two do not fit together.
#[test]
fn a_prompt_runs_in_place_of_the_state_whose_reuse_does_not_fit() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let cut = |doc: &str, question: &str| {
        let text = std::fs::read_to_string(root().join(doc)).expect("a document");
        let reference: String = text.chars().take(6000).collect();
        json!([{"role": "system", "content": format!("Answer from this reference.\n\n{reference}")},
               {"role": "user", "content": question}])
    };
    let first = cut("src/eremite.h", "Which calls run on the engine thread?");
    let second = cut("src/model/model.h", "What does a model family fill in?");
    // What the first conversation takes alone: its state and its prompt-end
    // snapshot.
    let (one, required) = {
        let server = start_server(&setup);
        chat(&server, &request(&first, 8));
        (settled(&server, 1).states, required_gib(&server))
    };
    // Room for one such state, not two.
    let limit = one * 1.5;
    let server = start_server_with(&setup, json!({}), json!({"memory": required + limit}));
    chat(&server, &request(&first, 8));
    let answer = chat(&server, &request(&second, 8));
    let cache = settled(&server, 2);
    println!(
        "a prompt beside a state it cannot keep: one state takes {one:.3} GiB, a state room of {limit} GiB; \
         the second reused {} tokens; the cache holds states of {:?} positions in {:.3} GiB",
        hit(&answer),
        cache.sessions,
        cache.states
    );
    assert_eq!(
        cache.sessions.len(),
        1,
        "the first state stayed beside the second: {:?}",
        cache.sessions
    );
    assert!(
        cache.states <= cache.limit,
        "the cache takes {:.3} GiB of {:.3}",
        cache.states,
        cache.limit
    );
}

/// Returns a conversation of `low` to `high` prompt tokens, and its token
/// count. Its system prompt is a cut of src/eremite.h, repeated as often as
/// that size needs. Its question has a long answer.
fn sized_prompt(setup: &Setup, low: usize, high: usize) -> (Value, usize) {
    let text = std::fs::read_to_string(root().join("src/eremite.h")).expect("src/eremite.h");
    let conversation = |reference: &str| {
        json!([{"role": "system", "content": format!("You review this project's design. Reference:\n\n{reference}")},
               {"role": "user", "content": "List every heading of the reference, one per line, each with a one-sentence summary."}])
    };
    let once = tokens(setup, &conversation(&text)).len();
    let design: Vec<char> = text.repeat(high / once + 1).chars().collect();
    let make = |chars: usize| conversation(&design[..chars].iter().collect::<String>());
    let (mut shortest, mut longest) = (0, design.len());
    while shortest < longest {
        let middle = (shortest + longest) / 2;
        let messages = make(middle);
        let count = tokens(setup, &messages).len();
        if count < low {
            shortest = middle + 1;
        } else if count >= high {
            longest = middle;
        } else {
            return (messages, count);
        }
    }
    panic!("no cut of src/eremite.h gives {low} to {high} prompt tokens");
}

/// Four requests decode while another request arrives. The server admits it at
/// once: its answer comes while the four still decode.
#[test]
fn a_request_is_answered_while_four_others_decode() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let server = start_server(&setup);
    let (first_token, first_tokens) = mpsc::channel();
    let decoding: Vec<_> = (0..DECODING_REQUESTS)
        .map(|i| {
            // A task that does not end within the tokens the request asks for.
            let messages = json!([{"role": "user", "content": format!(
                "[{i}] Count from 1 to 100000, one number per line, and write nothing else.")}]);
            let mut body = request(&messages, DECODING_TOKENS);
            body["stream"] = json!(true);
            let (port, key, first_token) = (server.port, server.key.clone(), first_token.clone());
            std::thread::spawn(move || stream_chat(port, &key, &body, first_token))
        })
        .collect();
    for _ in 0..DECODING_REQUESTS {
        first_tokens
            .recv()
            .expect("a decoding request's first token");
    }
    chat(&server, &request(&short_prompt(), 1));
    let still_decoding = decoding.iter().filter(|job| !job.is_finished()).count();
    for job in decoding {
        job.join().expect("a decoding request");
    }
    println!(
        "{still_decoding} of {DECODING_REQUESTS} requests still decoded when the arriving request's answer came"
    );
    assert_eq!(
        still_decoding, DECODING_REQUESTS,
        "the arriving request waited for a decoding request to end"
    );
}

/// The model's status follows a running prefill: before the answer's first
/// token comes, the engine thread's seconds in calls grow in it. A status that
/// is published only when a call ends shows the same seconds until the
/// prefill's end.
#[test]
fn the_status_follows_a_running_prefill() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let server = start_server(&setup);
    // The first request loads the model.
    chat(&server, &request(&short_prompt(), 1));
    let (long, _) = sized_prompt(&setup, 4000, 4096);
    let mut body = request(&long, 1);
    body["stream"] = json!(true);
    let (first_token, first_tokens) = mpsc::channel();
    let (port, key) = (server.port, server.key.clone());
    let job = std::thread::spawn(move || stream_chat(port, &key, &body, first_token));
    let seconds_in_calls = |status: &Value| -> f64 {
        ["pass", "wait_gpu", "wait_read", "wait_other"]
            .iter()
            .map(|state| {
                status["engine_times"]["thread_seconds"][state]
                    .as_f64()
                    .unwrap_or(0.0)
            })
            .sum()
    };
    let (mut first, mut grew) = (None, false);
    // The loop also ends when the request's thread ends without a first token.
    while !grew && first_tokens.try_recv() == Err(mpsc::TryRecvError::Empty) {
        let status = server.status();
        if status["phase"] != "prefill" {
            continue;
        }
        let seconds = seconds_in_calls(&status);
        grew = first.is_some_and(|first| seconds > first);
        first.get_or_insert(seconds);
    }
    job.join().expect("the long request");
    assert!(
        grew,
        "the status showed the same seconds in calls until the prefill ended"
    );
}

/// A generation that takes its state past the room it started with (16,384
/// positions) gets that growth counted before the state grows. So an idle state
/// goes then, not after the generation. The model's states stay within their
/// room all along. The idle conversation ends more than the rewind margin (15
/// positions, with the default lookup) after the tokens that it shares with
/// the long prompt. So the long prompt does not continue it in place.
#[test]
fn a_growing_generation_frees_room_before_it_grows() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let short = json!([
        {"role": "system", "content": "You answer in one word: the name of a color, in lower \
                                       case, with no period and no other word."},
        {"role": "user", "content": "Name a color."}
    ]);
    let (long, count) = sized_prompt(&setup, 16_330, 16_376);
    let body = request(&long, (16_384 - count + 150) as u32);
    // Leaves the short conversation's state idle, then sends the long request
    // and reads the status until its answer comes. Each status read waits for
    // the worker to publish: the samples follow its turns. Returns what the
    // idle state took, the tokens generated and the samples.
    let run = |server: &Server| {
        chat(server, &request(&short, 1));
        let idle = settled(server, 1).states;
        let (port, key, body) = (server.port, server.key.clone(), body.clone());
        let job = std::thread::spawn(move || {
            http(port, &key, "POST", "/v1/chat/completions", Some(&body))
        });
        let mut samples = Vec::new();
        while !job.is_finished() {
            samples.push(status(server));
        }
        let (code, answer) = job.join().expect("the long request");
        assert_eq!(
            code,
            200,
            "the long request: {}",
            String::from_utf8_lossy(&answer)
        );
        let answer: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
        let generated = answer["usage"]["completion_tokens"].as_i64().unwrap_or(0) as usize;
        (idle, generated, samples)
    };
    // With no limit, the idle state stays beside the long one: the cache's
    // largest size while the long one decodes, with its growth counted.
    let (idle, both, required) = {
        let server = start_server(&setup);
        let (idle, _, samples) = run(&server);
        let both = samples
            .iter()
            .filter(|sample| sample.phase == "decode")
            .map(|sample| sample.states)
            .fold(0.0, f64::max);
        (idle, both, required_gib(&server))
    };
    // Room for the idle state, the long state as it starts (its first room, as
    // the idle one's) and half of the long state's growth.
    let growth = both - 2.0 * idle;
    let limit = 2.0 * idle + growth / 2.0;
    let server = start_server_with(&setup, json!({}), json!({"memory": required + limit}));
    let (_, generated, samples) = run(&server);
    // Samples published after the long state grew: it holds more than 16,384
    // positions.
    let grown: Vec<&Status> = samples
        .iter()
        .filter(|sample| {
            sample.phase == "decode" && sample.sessions.iter().any(|&tokens| tokens > 16_384)
        })
        .collect();
    let over = samples
        .iter()
        .filter(|sample| sample.states > sample.limit)
        .count();
    println!(
        "growing generation: {count} prompt tokens, {generated} generated; with no limit the idle state took {idle:.4} GiB \
         and both {both:.4} GiB, a growth of {growth:.4} GiB; a state room of {limit} GiB; {} samples, {} while decoding past \
         16,384 positions, of which {} with the idle state still kept; {over} over the limit",
        samples.len(),
        grown.len(),
        grown
            .iter()
            .filter(|sample| sample.sessions.len() > 1)
            .count()
    );
    assert!(
        growth > 0.0,
        "the long state took no more than its first room"
    );
    assert!(
        count + generated > 16_390,
        "the reply ended at {} positions, before the state grew",
        count + generated
    );
    assert!(
        !grown.is_empty(),
        "no status sample while it decoded past 16,384 positions"
    );
    assert!(
        grown.iter().all(|sample| sample.sessions.len() == 1),
        "the idle state stayed while the generation grew past its room"
    );
    assert_eq!(over, 0, "the states took more than their room");
}

/// Quitting stops a long prompt's prefill. SIGTERM 5 s into the prefill of
/// about 15,000 tokens ends the server within 5 s, not when the prefill ends.
/// The request's answer is the server's stop (503). A short request loads the
/// model first, so the long prompt's prefill starts as soon as it arrives.
#[test]
fn quitting_stops_a_long_prefill() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let (long, count) = sized_prompt(&setup, 15_000, 16_000);
    let mut server = start_server(&setup);
    chat(&server, &request(&short_prompt(), 1));
    let body = request(&long, 8);
    let (port, key) = (server.port, server.key.clone());
    let job =
        std::thread::spawn(move || http(port, &key, "POST", "/v1/chat/completions", Some(&body)));
    // Each status read waits for the worker to publish: the loop ends once the
    // prefill runs.
    while server.status()["phase"] != "prefill" {}
    let stop = Instant::now();
    let _ = Command::new("kill")
        .args(["-TERM", &server.child.id().to_string()])
        .status();
    let exit = server.child.wait().ok();
    let took = stop.elapsed().as_secs_f64();
    let (code, answer) = job.join().expect("the long request");
    println!(
        "quitting a {count}-token prefill: the server exited {took:.1} s after SIGTERM ({exit:?}); the request: {code} {}",
        String::from_utf8_lossy(&answer)
    );
    assert!(
        exit.is_some_and(|status| status.success()),
        "the server did not exit cleanly"
    );
    assert!(
        took < 5.0,
        "the server exited {took:.1} s after SIGTERM, during a {count}-token prefill"
    );
    assert_eq!(code, 503, "the stopped request's answer");
}

/// A model that goes its idle_unload time without requests gives back its
/// weights and experts. Its status reads unloaded. The memory it held is back
/// in the engine's count and in the process's footprint, and its session stays.
/// It stays unloaded while the server keeps calling its idle hook (35 s: past
/// DeepSeek's 32 calm calls, after which its cache raises its cap). The same
/// request then loads it again and continues that session with the same greedy
/// reply.
#[test]
fn an_idle_model_unloads_and_its_next_request_loads_it_again() {
    let Some(setup) = setup() else {
        eprintln!("skipped: set EREMITE_CHECKPOINT and EREMITE_EXPERTS");
        return;
    };
    let _one = one_model();
    let server = start_server_with(&setup, json!({"idle_unload": 3}), json!({}));
    let footprint = |status: &Value| {
        status["memory"]["footprint_gib"]
            .as_f64()
            .unwrap_or(f64::MAX)
    };
    let ask = request(&short_prompt(), 16);
    let first = content(&chat(&server, &ask));
    let replied = Instant::now();
    let before = server.status();
    assert_eq!(before["loaded"], json!(true), "{before}");
    let bytes = |status: &Value| {
        status["model_memory_budget"]["allocated"]
            .as_u64()
            .unwrap_or(0)
    };
    let held = bytes(&before);
    // Each status read waits for the worker to publish; the model unloads a
    // little after the 3 s.
    let mut after = before.clone();
    while after["loaded"] != json!(false) {
        after = server.status();
    }
    let took = replied.elapsed().as_secs_f64();
    let back = held.saturating_sub(bytes(&after));
    // Then it stays unloaded, and freed pages reach the counters.
    let unloaded = Instant::now();
    let mut stayed = after["loaded"] == json!(false);
    while stayed && unloaded.elapsed().as_secs() < 35 {
        after = server.status();
        stayed = after["loaded"] == json!(false);
    }
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "unloaded {took:.1} s after the reply: {:.2} of {:.2} GiB of model memory back; \
         the process's footprint {:.1} GiB loaded, {:.1} after 35 s unloaded; sessions {}",
        gib(back),
        gib(held),
        footprint(&before),
        footprint(&after),
        after["sessions"]
    );
    assert!(
        took >= 2.5,
        "unloaded {took:.1} s after the reply, before 3 s"
    );
    assert!(stayed, "it did not stay unloaded: {after}");
    assert!(back * 10 >= held * 9, "{back} of {held} bytes back");
    // The pages go back to the system too, not only the engine's count. The
    // session, the workspace and the compiled kernels stay.
    assert!(
        footprint(&after) <= footprint(&before) / 10.0,
        "the footprint fell from {:.1} to only {:.1} GiB",
        footprint(&before),
        footprint(&after)
    );
    assert!(
        after["sessions"]
            .as_array()
            .is_some_and(|sessions| !sessions.is_empty()),
        "the session stays: {after}"
    );
    let again = chat(&server, &ask);
    assert_eq!(content(&again), first, "the reply after the reload");
    assert_eq!(server.status()["loaded"], json!(true));
}
