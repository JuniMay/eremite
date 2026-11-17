//! Requests through HTTP to a DeepSeek checkpoint.
//!
//! A prompt too long for the context is refused with its token count. The
//! model's status follows a running prefill, and quitting stops a long prefill.
//! A request is answered while four others decode.
//!
//! Set EREMITE_CHECKPOINT to the checkpoint directory. EREMITE_EXPERTS
//! optionally selects GGUF experts; otherwise the tests use native MXFP4. The
//! tests run one model process at a time (they share a lock):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test prefix_fork -- --nocapture
//!
//! EREMITE_SERVER picks the server binary (default build/eremite-server) and
//! EREMITE_CLI the tokenizer's (default build/eremite).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::time::Instant;

use serde_json::{Value, json};

mod common;
use common::{Server, http, one_model, root};

/// The requests that decode while another request arrives.
const DECODING_REQUESTS: usize = 4;
/// The tokens that each decoding request asks for. 500 tokens take about 45 s
/// with IQ2 experts and 100 s with MXFP4 experts. The arriving request's answer
/// takes less than 20 s.
const DECODING_TOKENS: u32 = 500;

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

/// Starts a server with the tests' settings, and `extra`.
fn start_server(setup: &Setup, extra: &[&str]) -> Server {
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
    let mut args: Vec<String> = ["-m", &setup.checkpoint, "--memory", "56"]
        .iter()
        .map(|arg| arg.to_string())
        .collect();
    if let Some(experts) = &setup.experts {
        args.extend(["--experts".to_string(), experts.clone()]);
    }
    args.extend(["--ctx", "32768", "--serve", &port.to_string()].map(String::from));
    args.extend(extra.iter().map(|arg| arg.to_string()));
    Server::start(&setup.server, &args, &key, port)
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
    let case = std::env::temp_dir().join(format!("eremite-fork-{}.json", std::process::id()));
    let mode = if thinking { "thinking" } else { "chat" };
    std::fs::write(
        &case,
        json!({"messages": messages, "thinking_mode": mode}).to_string(),
    )
    .expect("a render case");
    let rendered = Command::new(&setup.server)
        .arg("--render")
        .arg(&case)
        .output()
        .expect("render");
    let _ = std::fs::remove_file(&case);
    assert!(
        rendered.status.success(),
        "render: {}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    let prompt = String::from_utf8(rendered.stdout).expect("UTF-8");
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
    let server = start_server(&setup, &["--ctx", "1024"]);
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

/// Returns a short conversation with a one-word answer.
fn short_prompt() -> Value {
    json!([{"role": "system", "content": "You answer in one word."},
           {"role": "user", "content": "Name a color."}])
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
    let server = start_server(&setup, &[]);
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
    let server = start_server(&setup, &[]);
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
    let mut server = start_server(&setup, &[]);
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
