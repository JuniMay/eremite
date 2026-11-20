//! MiniMax-H3-Image through HTTP: POST /v1/images/generations (JSON) and
//! POST /v1/images/edits (multipart/form-data). Verifies model listing, Codex
//! catalog exclusion, chat and decision refusals, HTTP 400 validation errors,
//! exact requested output dimensions (including non-32 resizing), PNG and JPEG
//! encoding, and multi-part reference image editing.
//!
//! Set EREMITE_MINIMAX_H3_MODEL_DIR, EREMITE_MINIMAX_H3_TEXT_ENCODER,
//! EREMITE_MINIMAX_H3_DIT, EREMITE_MINIMAX_H3_VAE, and optionally
//! EREMITE_MINIMAX_H3_REF_IMAGE. Run one model process at a time:
//!
//!     EREMITE_MINIMAX_H3_MODEL_DIR=... \
//!     EREMITE_MINIMAX_H3_TEXT_ENCODER=... \
//!     EREMITE_MINIMAX_H3_DIT=... \
//!     EREMITE_MINIMAX_H3_VAE=... \
//!       cargo test --release --manifest-path server/Cargo.toml --test images -- --nocapture

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use serde_json::{Value, json};

mod common;
use common::{Server, build_config, http, one_model, root};

/// Starts a server configured with the MiniMax-H3-Image checkpoints from the
/// environment, or returns None when the env vars are unset.
fn start_image_server() -> Option<Server> {
    let model_dir = std::env::var("EREMITE_MINIMAX_H3_MODEL_DIR").ok()?;
    let text_encoder = std::env::var("EREMITE_MINIMAX_H3_TEXT_ENCODER").ok()?;
    let dit = std::env::var("EREMITE_MINIMAX_H3_DIT").ok()?;
    let vae = std::env::var("EREMITE_MINIMAX_H3_VAE").ok()?;
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("images-test-{}-{port}-key", std::process::id());
    let config = build_config(
        port,
        json!({"reserve": 4, "state_disk": 0, "http_timeout": 300}),
        &model_dir,
        None,
        json!({
            "text_encoder": text_encoder,
            "dit": dit,
            "vae": vae
        }),
    );
    Some(Server::start(&program, &config, &key, port))
}

/// Sends a JSON HTTP request and parses the response body as JSON.
fn ask(server: &Server, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let (code, answer) = http(server.port, &server.key, method, path, body);
    (code, serde_json::from_slice(&answer).unwrap_or(Value::Null))
}

/// Sends a raw POST request with custom Content-Type and body bytes.
fn post_raw(server: &Server, path: &str, content_type: &str, body: &[u8]) -> (u16, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).expect("connect");
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        server.key,
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body))
        .expect("send");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("HTTP response");
    let headers = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut rest = raw[split + 4..].to_vec();
    if headers.contains("transfer-encoding: chunked") {
        let (mut out, mut at) = (Vec::new(), 0);
        while let Some(end) = rest[at..].windows(2).position(|window| window == b"\r\n") {
            let size =
                usize::from_str_radix(String::from_utf8_lossy(&rest[at..at + end]).trim(), 16)
                    .unwrap_or(0);
            if size == 0 {
                break;
            }
            let start = at + end + 2;
            out.extend_from_slice(&rest[start..start + size]);
            at = start + size + 2;
        }
        rest = out;
    }
    (status, serde_json::from_slice(&rest).unwrap_or(Value::Null))
}

/// Decodes standard RFC 4648 base64 into raw bytes.
fn decode_base64(input: &str) -> Vec<u8> {
    let url = format!("data:application/octet-stream;base64,{input}");
    data_url::DataUrl::process(&url)
        .expect("valid data URL")
        .decode_to_vec()
        .expect("valid base64")
        .0
}

/// Reads the (width, height) from a PNG file's IHDR chunk.
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((w, h))
}

#[test]
fn minimax_h3_image_serves_generations_and_edits_over_http() {
    let _one = one_model();
    let Some(server) = start_image_server() else {
        eprintln!("skipped: set EREMITE_MINIMAX_H3_MODEL_DIR, _TEXT_ENCODER, _DIT and _VAE");
        return;
    };

    // 1. Model catalog and endpoint isolation.
    let (code, models) = ask(&server, "GET", "/v1/models", None);
    assert_eq!(code, 200, "{models}");
    let model = &models["data"][0];
    assert_eq!(model["id"], "minimax-h3-image");
    assert_eq!(model["context_length"].as_u64(), Some(4096));
    assert_eq!(model.get("max_output_tokens"), None);

    let (code, catalog) = ask(&server, "GET", "/v1/codex/models", None);
    assert_eq!((code, &catalog["models"]), (200, &json!([])), "{catalog}");

    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model": "minimax-h3-image", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "/v1/responses",
            json!({"model": "minimax-h3-image", "input": "hi"}),
        ),
        (
            "/v1/decisions",
            json!({"model": "minimax-h3-image", "input": "hi", "questions": [{"type": "predicate", "instructions": "yes?"}]}),
        ),
    ] {
        let (code, answer) = ask(&server, "POST", path, Some(&body));
        assert_eq!(code, 400, "{path}: {answer}");
    }

    // 2. Request validation errors return HTTP 400 before loading weights.
    for bad in [
        json!({"prompt": "a cat", "size": "512x512", "num_inference_steps": 12}),
        json!({"prompt": "a cat", "size": "512x512", "guidance_scale": 5.0}),
        json!({"prompt": "a cat", "size": "512x512", "negative_prompt": "ugly"}),
        json!({"prompt": "a cat", "size": "512x512", "response_format": "url"}),
    ] {
        let (code, answer) = ask(&server, "POST", "/v1/images/generations", Some(&bad));
        assert_eq!(code, 400, "bad={bad} -> {answer}");
    }

    // 3. Text-to-image PNG with non-32 exact requested size (500x500).
    let t2i_req = json!({
        "model": "minimax-h3-image",
        "prompt": "A bright red apple resting on a white ceramic plate.",
        "size": "500x500",
        "seed": 42,
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
        "min_generation_area": 1024
    });
    let (code, answer) = ask(&server, "POST", "/v1/images/generations", Some(&t2i_req));
    assert_eq!(code, 200, "t2i PNG failed: {answer}");
    let png_b64 = answer["data"][0]["b64_json"]
        .as_str()
        .expect("b64_json string");
    let png_bytes = decode_base64(png_b64);
    assert_eq!(png_dimensions(&png_bytes), Some((500, 500)));

    // 4. Text-to-image JPEG with model omitted (default single image model).
    let jpg_req = json!({
        "prompt": "A blue sailboat on calm water under a clear sky.",
        "size": "512x512",
        "seed": 7,
        "output_format": "jpeg",
        "output_compression": 90,
        "min_generation_area": 1024
    });
    let (code, answer) = ask(&server, "POST", "/v1/images/generations", Some(&jpg_req));
    assert_eq!(code, 200, "t2i JPEG failed: {answer}");
    let jpg_b64 = answer["data"][0]["b64_json"]
        .as_str()
        .expect("b64_json string");
    let jpg_bytes = decode_base64(jpg_b64);
    assert!(
        jpg_bytes.len() > 100 && jpg_bytes[0] == 0xFF && jpg_bytes[1] == 0xD8,
        "valid JPEG SOI header"
    );

    // 5. Reference image editing via multipart/form-data on /v1/images/edits.
    let ref_bytes = if let Ok(ref_path) = std::env::var("EREMITE_MINIMAX_H3_REF_IMAGE") {
        std::fs::read(&ref_path).expect("read EREMITE_MINIMAX_H3_REF_IMAGE")
    } else {
        png_bytes.clone()
    };
    let boundary = "eremite-images-test-boundary";
    let mut form = Vec::new();
    for (name, val) in [
        (
            "prompt",
            "Keep <Picture 1> with warm golden sunset lighting.",
        ),
        ("size", "512x512"),
        ("seed", "1234"),
        ("generator_device", "cpu"),
        ("response_format", "b64_json"),
        ("output_format", "png"),
        ("min_generation_area", "1024"),
    ] {
        form.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n"
            )
            .as_bytes(),
        );
    }
    form.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"ref.png\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .as_bytes(),
    );
    form.extend_from_slice(&ref_bytes);
    form.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let (code, answer) = post_raw(
        &server,
        "/v1/images/edits",
        &format!("multipart/form-data; boundary={boundary}"),
        &form,
    );
    assert_eq!(code, 200, "edit failed: {answer}");
    let edit_b64 = answer["data"][0]["b64_json"]
        .as_str()
        .expect("edit b64_json");
    let edit_bytes = decode_base64(edit_b64);
    assert_eq!(png_dimensions(&edit_bytes), Some((512, 512)));

    // 6. Status endpoint reports loaded image model.
    let (code, status) = ask(&server, "GET", "/v1/models/minimax-h3-image", None);
    assert_eq!(code, 200, "{status}");
    assert_eq!(status["status"]["loaded"], json!(true));

    server.stop();
}
