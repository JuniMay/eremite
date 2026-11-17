//! Requests through HTTP to a DeepSeek checkpoint.
//!
//! A prompt too long for the context is refused with its token count.
//!
//! Set EREMITE_CHECKPOINT to the checkpoint directory. EREMITE_EXPERTS
//! optionally selects GGUF experts; otherwise the tests use native MXFP4. The
//! tests run one model process at a time (they share a lock):
//!
//!     EREMITE_CHECKPOINT=DIR EREMITE_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test prefix_fork -- --nocapture
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
    })
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
