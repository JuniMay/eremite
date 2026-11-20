//! MiniMax-H3-Image through HTTP: POST /v1/images/generations (JSON) and
//! POST /v1/images/edits (JSON or multipart/form-data). Verifies model
//! listing, Codex catalog exclusion, chat and decision refusals, HTTP 400
//! validation errors, exact and auto output sizes (including non-32
//! resizing), PNG and JPEG encoding, and reference image editing.
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

/// The most bytes in one stored, uncompressed deflate block.
const STORED_BLOCK_BYTES: usize = 65_535;
/// The modulus of zlib's Adler-32 checksum.
const ADLER_MODULUS: u32 = 65_521;
/// The reversed polynomial of PNG's CRC-32.
const CRC_POLYNOMIAL: u32 = 0xEDB8_8320;

/// Returns the CRC-32 of `bytes`, as PNG chunks carry it.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC_POLYNOMIAL
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Returns a PNG file of `width` by `height` RGBA pixels that `pixel` gives
/// for each x and y, with its data stored without compression.
fn rgba_png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut rows = Vec::new();
    for y in 0..height {
        // Each row starts with its filter type, 0: none.
        rows.push(0);
        for x in 0..width {
            rows.extend_from_slice(&pixel(x, y));
        }
    }
    // A zlib stream with no compression, then its Adler-32 checksum.
    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = rows.chunks(STORED_BLOCK_BYTES).collect();
    for (index, block) in blocks.iter().enumerate() {
        zlib.push(u8::from(index + 1 == blocks.len()));
        let len = block.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut low, mut high) = (1u32, 0u32);
    for &byte in &rows {
        low = (low + u32::from(byte)) % ADLER_MODULUS;
        high = (high + low) % ADLER_MODULUS;
    }
    zlib.extend_from_slice(&((high << 16) | low).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    // 8 bits a channel, color type 6 (RGBA), and the default methods.
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    for (kind, data) in [
        (&b"IHDR"[..], &header[..]),
        (b"IDAT", &zlib),
        (b"IEND", &[]),
    ] {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        let crc = crc32(&png[start..]);
        png.extend_from_slice(&crc.to_be_bytes());
    }
    png
}

/// Returns `bytes` as a base64 data URL of a PNG image.
fn png_data_url(bytes: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        eremite_server::image::encode_base64(bytes)
    )
}

#[test]
fn rgba_png_writes_a_png_that_imageio_reads() {
    // Rows of 3 RGBA pixels and a filter byte; one row more than one stored
    // block holds, so that the data spans two blocks.
    let row_bytes = 1 + 3 * 4;
    let height = (STORED_BLOCK_BYTES / row_bytes + 1) as u32;
    let png = rgba_png(3, height, |x, y| [x as u8, y as u8, 0, 128]);
    assert_eq!(
        eremite_server::engine::read_image_size(&png),
        Ok((3, height as i32))
    );
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
    assert_eq!(answer["size"], "500x500", "{answer}");

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

    // 4a. Text-to-image WebP at the lowest compression value.
    let webp_req = json!({
        "prompt": "A yellow rubber duck on a bathroom tile floor.",
        "size": "512x512",
        "seed": 9,
        "output_format": "webp",
        "output_compression": 0,
        "min_generation_area": 1024
    });
    let (code, answer) = ask(&server, "POST", "/v1/images/generations", Some(&webp_req));
    assert_eq!(code, 200, "t2i WebP failed: {answer}");
    assert_eq!(answer["output_format"], "webp", "{answer}");
    let webp_bytes = decode_base64(answer["data"][0]["b64_json"].as_str().expect("b64_json"));
    assert!(
        webp_bytes.len() > 12 && &webp_bytes[..4] == b"RIFF" && &webp_bytes[8..12] == b"WEBP",
        "valid WebP header"
    );

    // 4b. Text-to-image with size omitted gets the model's default size.
    let auto_req = json!({"prompt": "A small green cactus in a terracotta pot.", "seed": 3});
    let (code, answer) = ask(&server, "POST", "/v1/images/generations", Some(&auto_req));
    assert_eq!(code, 200, "auto t2i failed: {answer}");
    assert_eq!(answer["size"], "2048x2048", "{answer}");
    let auto_png = decode_base64(answer["data"][0]["b64_json"].as_str().expect("b64_json"));
    assert_eq!(png_dimensions(&auto_png), Some((2048, 2048)));

    // 5. Reference image editing via multipart/form-data on /v1/images/edits.
    let ref_bytes = if let Ok(ref_path) = std::env::var("EREMITE_MINIMAX_H3_REF_IMAGE") {
        std::fs::read(&ref_path).expect("read EREMITE_MINIMAX_H3_REF_IMAGE")
    } else {
        png_bytes.clone()
    };
    let boundary = "eremite-images-test-boundary";
    let edit_form = |size: Option<&str>| {
        let mut form = Vec::new();
        for (name, val) in [
            (
                "prompt",
                "Keep <Picture 1> with warm golden sunset lighting.",
            ),
            ("seed", "1234"),
            ("generator_device", "cpu"),
            ("response_format", "b64_json"),
            ("output_format", "png"),
            ("min_generation_area", "1024"),
        ]
        .into_iter()
        .chain(size.map(|size| ("size", size)))
        {
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
        form
    };
    let content_type = format!("multipart/form-data; boundary={boundary}");

    let (code, answer) = post_raw(
        &server,
        "/v1/images/edits",
        &content_type,
        &edit_form(Some("512x512")),
    );
    assert_eq!(code, 200, "edit failed: {answer}");
    let edit_b64 = answer["data"][0]["b64_json"]
        .as_str()
        .expect("edit b64_json");
    let edit_bytes = decode_base64(edit_b64);
    assert_eq!(png_dimensions(&edit_bytes), Some((512, 512)));

    // 5b. An edit with size omitted keeps the size of its first image.
    let ref_size = png_dimensions(&ref_bytes).expect("a PNG reference image");
    let (code, answer) = post_raw(&server, "/v1/images/edits", &content_type, &edit_form(None));
    assert_eq!(code, 200, "auto edit failed: {answer}");
    assert_eq!(answer["size"], format!("{}x{}", ref_size.0, ref_size.1));
    let auto_edit = decode_base64(answer["data"][0]["b64_json"].as_str().expect("b64_json"));
    assert_eq!(png_dimensions(&auto_edit), Some(ref_size));

    // 5c. A JSON edit gives its image as a base64 data URL, as Codex sends it.
    let json_edit = json!({
        "model": "gpt-image-2",
        "prompt": "Keep <Picture 1> with warm golden sunset lighting.",
        "images": [{"image_url": png_data_url(&ref_bytes)}],
        "size": "auto",
        "quality": "auto",
        "background": "opaque",
        "seed": 1234,
        "min_generation_area": 1024
    });
    let (code, answer) = ask(&server, "POST", "/v1/images/edits", Some(&json_edit));
    assert_eq!(code, 200, "JSON edit failed: {answer}");
    assert_eq!(answer["size"], format!("{}x{}", ref_size.0, ref_size.1));

    // 5d. A masked edit: the mask keeps the first image's bottom half. A size
    // other than the first image's is refused before the weights run.
    let mask = rgba_png(ref_size.0, ref_size.1, |_, y| {
        [0, 0, 0, if y < ref_size.1 / 2 { 0 } else { 255 }]
    });
    let masked = |size: &str| {
        json!({
            "prompt": "Turn <Picture 1> into a pencil sketch.",
            "images": [{"image_url": png_data_url(&ref_bytes)}],
            "mask": {"image_url": png_data_url(&mask)},
            "size": size,
            "seed": 5,
            "min_generation_area": 1024
        })
    };
    let (code, answer) = ask(&server, "POST", "/v1/images/edits", Some(&masked("64x64")));
    assert_eq!(code, 400, "a mask with another size: {answer}");
    let (code, answer) = ask(&server, "POST", "/v1/images/edits", Some(&masked("auto")));
    assert_eq!(code, 200, "masked edit failed: {answer}");
    assert_eq!(answer["size"], format!("{}x{}", ref_size.0, ref_size.1));

    // 6. Status endpoint reports loaded image model with separated counters.
    let (code, status) = ask(&server, "GET", "/v1/models/minimax-h3-image", None);
    assert_eq!(code, 200, "{status}");
    let st = &status["status"];
    assert_eq!(st["loaded"], json!(true));
    assert_eq!(st["phase"], json!("idle"));
    assert_eq!(st["totals"]["generated_images"], json!(8));
    assert_eq!(st["totals"]["generated_tokens"], json!(0));
    assert!(
        st["totals"]["prefilled_tokens"].as_u64().unwrap_or(0) > 0,
        "{st}"
    );
    assert!(
        st["totals"]["prefill_seconds"].as_f64().unwrap_or(0.0) > 0.0,
        "{st}"
    );
    assert!(
        st["totals"]["image_seconds"].as_f64().unwrap_or(0.0) > 0.0,
        "{st}"
    );
    assert_eq!(st["engine_counters"]["generated_images"], json!(8));
    assert_eq!(st["engine_counters"]["image_steps"], json!(48));
    assert_eq!(st["engine_counters"]["image_stage"], json!(0));

    server.stop();
}
