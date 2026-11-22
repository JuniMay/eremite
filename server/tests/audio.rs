//! Nemotron-3.5-Transcribe through HTTP: POST /v1/audio/transcriptions and
//! POST /v1/audio/translations (multipart/form-data). Verifies model listing,
//! Codex catalog exclusion, cross-endpoint refusals, HTTP 400 validation
//! errors, JSON, plain text, verbose_json and SSE streaming responses, and
//! MoE SSD expert streaming under a 5 GiB model memory cap.
//!
//! Set EREMITE_NEMOTRON_TRANSCRIBE_MODEL_DIR (and optionally
//! EREMITE_NEMOTRON_TRANSCRIBE_WAV). Build `build/eremite-server` first and
//! run one model process at a time:
//!
//!     make server && \
//!     EREMITE_NEMOTRON_TRANSCRIBE_MODEL_DIR=... \
//!       cargo test --release --manifest-path server/Cargo.toml \
//!       --target-dir build/cargo --test audio -- --nocapture

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;

use serde_json::{Value, json};

mod common;
use common::{Server, build_config, http, one_model, root};

/// Sample rate in Hz of synthesized test WAV files.
const TEST_WAV_SAMPLE_RATE: u32 = 16_000;
/// Bit depth of synthesized 16-bit PCM WAV files.
const TEST_WAV_BITS_PER_SAMPLE: u16 = 16;
/// RIFF header payload bytes preceding the PCM sample data.
const WAV_RIFF_PREAMBLE_BYTES: u32 = 36;
/// PCM fmt subchunk payload length in bytes.
const WAV_FMT_CHUNK_BYTES: u32 = 16;
/// Spoken test sentence used when synthesizing speech with macOS `say`.
const TEST_SPEECH_SENTENCE: &str = "The quick brown fox jumps over the lazy dog. Local speech recognition on Apple silicon runs reliably.";

/// Encodes mono 16-bit PCM samples at 16 kHz into a RIFF/WAVE byte vector.
fn encode_pcm16_wav(samples: &[i16]) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let byte_rate = TEST_WAV_SAMPLE_RATE * 2;
    let mut wav = Vec::with_capacity(44 + data_bytes as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(WAV_RIFF_PREAMBLE_BYTES + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&WAV_FMT_CHUNK_BYTES.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&TEST_WAV_SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&TEST_WAV_BITS_PER_SAMPLE.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

/// Builds a multipart/form-data body and matching Content-Type header value.
fn build_multipart(parts: &[(&str, Option<&str>, &[u8])]) -> (String, Vec<u8>) {
    let boundary = "eremite-audio-test-boundary";
    let mut out = Vec::new();
    for &(name, filename, data) in parts {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match filename {
            Some(fname) => out.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => out.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), out)
}

/// Sends a raw HTTP POST request and returns `(status, content_type, body)`.
fn post_raw(
    server: &Server,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> (u16, String, Vec<u8>) {
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
    let headers_text = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let status = headers_text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let resp_ct = headers_text
        .lines()
        .find_map(|line| line.strip_prefix("content-type:"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut rest = raw[split + 4..].to_vec();
    if headers_text.contains("transfer-encoding: chunked") {
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
    (status, resp_ct, rest)
}

/// Sends a JSON HTTP request and parses the response body as JSON.
fn ask(server: &Server, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let (code, answer) = http(server.port, &server.key, method, path, body);
    (code, serde_json::from_slice(&answer).unwrap_or(Value::Null))
}

/// Starts a server with the Nemotron-3.5-Transcribe checkpoint and `model_opts`.
fn start_audio_server(model_dir: &str, model_opts: Value) -> Server {
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("audio-test-{}-{port}-key", std::process::id());
    let config = build_config(
        port,
        json!({"reserve": 4, "state_disk": 0, "http_timeout": 300}),
        model_dir,
        None,
        model_opts,
    );
    Server::start(&program, &config, &key, port)
}

/// Loads `EREMITE_NEMOTRON_TRANSCRIBE_WAV` or synthesizes spoken English audio
/// using macOS `say`.
fn load_or_synthesize_speech_wav() -> Vec<u8> {
    if let Ok(path) = std::env::var("EREMITE_NEMOTRON_TRANSCRIBE_WAV") {
        return std::fs::read(&path).expect("read EREMITE_NEMOTRON_TRANSCRIBE_WAV");
    }
    let tmp = std::env::temp_dir().join(format!("eremite-audio-test-{}.wav", std::process::id()));
    let status = Command::new("say")
        .args([
            "-o",
            tmp.to_str().unwrap(),
            "--data-format=LEI16@16000",
            TEST_SPEECH_SENTENCE,
        ])
        .status()
        .expect("run macOS say");
    assert!(status.success(), "macOS say failed");
    let bytes = std::fs::read(&tmp).expect("read synthesized wav");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

#[test]
fn wav_helper_writes_valid_riff_wave_header() {
    let samples = vec![0i16; 1600];
    let wav = encode_pcm16_wav(&samples);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(wav.len(), 44 + samples.len() * 2);
}

#[test]
fn nemotron_transcribe_serves_audio_endpoints_full_and_ssd_streaming() {
    let Ok(model_dir) = std::env::var("EREMITE_NEMOTRON_TRANSCRIBE_MODEL_DIR") else {
        return;
    };
    let _guard = one_model();
    let speech_wav = load_or_synthesize_speech_wav();

    // 1. Full-resident mode: verify catalog, refusals, validation, json, text,
    // verbose_json, SSE streaming and English translation.
    let expected_text = {
        let server = start_audio_server(&model_dir, json!({}));

        let (code, models) = ask(&server, "GET", "/v1/models", None);
        assert_eq!(code, 200);
        let entry = &models["data"][0];
        assert_eq!(entry["id"], "nemotron-3.5-transcribe");
        assert_eq!(entry["context_length"], 65536);

        let (code, codex_models) = ask(&server, "GET", "/v1/codex/models", None);
        assert_eq!(code, 200);
        assert_eq!(codex_models["models"].as_array().unwrap().len(), 0);

        // Non-audio endpoints refuse the audio transcription model with 400.
        let (code, _) = ask(
            &server,
            "POST",
            "/v1/chat/completions",
            Some(&json!({
                "model": "nemotron-3.5-transcribe",
                "messages": [{"role": "user", "content": "hi"}]
            })),
        );
        assert_eq!(code, 400);
        let (code, _) = ask(
            &server,
            "POST",
            "/v1/decisions",
            Some(&json!({
                "model": "nemotron-3.5-transcribe",
                "input": "hello",
                "questions": [{"type": "predicate", "instructions": "yes?"}]
            })),
        );
        assert_eq!(code, 400);
        let (code, _) = ask(
            &server,
            "POST",
            "/v1/images/generations",
            Some(&json!({"model": "nemotron-3.5-transcribe", "prompt": "cat"})),
        );
        assert_eq!(code, 400);
        let (code, _) = ask(
            &server,
            "POST",
            "/v1/videos",
            Some(&json!({"model": "nemotron-3.5-transcribe", "prompt": "cat"})),
        );
        assert_eq!(code, 400);

        // Validation errors return HTTP 400.
        let (code, _, _) = post_raw(
            &server,
            "/v1/audio/transcriptions",
            "application/json",
            b"{}",
        );
        assert_eq!(code, 400);

        let (ct, body) = build_multipart(&[("model", None, b"whisper-1")]);
        let (code, _, _) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 400);

        let (ct, body) = build_multipart(&[("file", Some("bad.wav"), b"not audio data at all")]);
        let (code, _, _) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 400);

        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("response_format", None, b"srt"),
        ]);
        let (code, _, _) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 400);

        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("response_format", None, b"text"),
            ("stream", None, b"true"),
        ]);
        let (code, _, _) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 400);

        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("language", None, b"en"),
        ]);
        let (code, _, _) = post_raw(&server, "/v1/audio/translations", &ct, &body);
        assert_eq!(code, 400);

        // Transcription (default json, using OpenAI model alias whisper-1).
        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("model", None, b"whisper-1"),
        ]);
        let (code, resp_ct, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200, "{}", String::from_utf8_lossy(&resp_body));
        assert!(resp_ct.starts_with("application/json"), "{resp_ct}");
        let json_resp: Value = serde_json::from_slice(&resp_body).expect("JSON");
        let text = json_resp["text"].as_str().expect("text field").to_string();
        let lower = text.to_ascii_lowercase();
        assert!(
            lower.contains("quick brown fox") && lower.contains("apple silicon"),
            "unexpected transcript: {text:?}"
        );
        assert!(json_resp["usage"]["input_tokens"].as_u64().unwrap() > 0);
        assert!(json_resp["usage"]["output_tokens"].as_u64().unwrap() > 0);

        // Plain text format matches JSON text.
        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("response_format", None, b"text"),
        ]);
        let (code, resp_ct, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200);
        assert!(resp_ct.starts_with("text/plain"), "{resp_ct}");
        assert_eq!(String::from_utf8(resp_body).unwrap(), text);

        // Verbose JSON with language and context prompt.
        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("language", None, b"en"),
            ("prompt", None, b"Apple silicon"),
            ("response_format", None, b"verbose_json"),
        ]);
        let (code, _, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200);
        let verbose: Value = serde_json::from_slice(&resp_body).expect("verbose JSON");
        assert_eq!(verbose["task"], "transcribe");
        assert_eq!(verbose["language"], "english");
        assert!(verbose["duration"].as_f64().unwrap() > 1.0);
        assert!(
            verbose["text"]
                .as_str()
                .unwrap()
                .to_ascii_lowercase()
                .contains("quick brown fox")
        );

        // SSE streaming transcription.
        let (ct, body) = build_multipart(&[
            ("file", Some("speech.wav"), &speech_wav),
            ("stream", None, b"true"),
        ]);
        let (code, resp_ct, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200);
        assert!(resp_ct.starts_with("text/event-stream"), "{resp_ct}");
        let sse_text = String::from_utf8(resp_body).expect("UTF-8 SSE");
        let mut streamed = String::new();
        let mut done_text = None;
        for block in sse_text.split("\n\n") {
            let Some(data) = block.lines().find_map(|line| line.strip_prefix("data: ")) else {
                continue;
            };
            let ev: Value = serde_json::from_str(data).expect("SSE JSON");
            match ev["type"].as_str() {
                Some("transcript.text.delta") => {
                    streamed.push_str(ev["delta"].as_str().unwrap_or(""));
                }
                Some("transcript.text.done") => {
                    done_text = ev["text"].as_str().map(String::from);
                    assert!(ev["usage"]["total_tokens"].as_u64().unwrap() > 0);
                }
                other => panic!("unexpected SSE event type: {other:?}"),
            }
        }
        assert_eq!(done_text.as_deref(), Some(streamed.as_str()));
        assert_eq!(streamed.trim(), text);

        // Translation endpoint (/v1/audio/translations).
        let (ct, body) = build_multipart(&[("file", Some("speech.wav"), &speech_wav)]);
        let (code, _, resp_body) = post_raw(&server, "/v1/audio/translations", &ct, &body);
        assert_eq!(code, 200, "{}", String::from_utf8_lossy(&resp_body));
        let trans: Value = serde_json::from_slice(&resp_body).expect("JSON");
        assert!(
            trans["text"]
                .as_str()
                .unwrap()
                .to_ascii_lowercase()
                .contains("quick brown fox")
        );

        text
    };

    // 2. SSD-streaming mode (5 GiB memory cap): verify bit-identical
    // transcript and live expert cache misses reported in model status.
    {
        let server = start_audio_server(&model_dir, json!({"memory": 5}));
        let (ct, body) = build_multipart(&[("file", Some("speech.wav"), &speech_wav)]);
        let (code, _, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200, "{}", String::from_utf8_lossy(&resp_body));
        let json_resp: Value = serde_json::from_slice(&resp_body).expect("JSON");
        assert_eq!(json_resp["text"].as_str().unwrap(), expected_text);

        let (code, status) = ask(&server, "GET", "/v1/models/nemotron-3.5-transcribe", None);
        assert_eq!(code, 200);
        let cache = &status["status"]["expert_cache"];
        let counters = &status["status"]["engine_counters"];
        let slots = cache["slots"].as_u64().unwrap();
        let total_experts = counters["routed_layers"].as_u64().unwrap()
            * counters["experts_per_layer"].as_u64().unwrap();
        let misses = cache["misses"].as_u64().unwrap();
        assert!(
            slots < total_experts && misses > 0,
            "expected SSD streaming with partial expert cache, got cache={cache} counters={counters}"
        );
    }

    // 3. MTP speculative decoding + SSD streaming (6 GiB cap, draft: 7):
    // verify model-mtp-fp8.safetensors loads, status reports drafts=7 with
    // partial expert cache, and transcript matches baseline verbatim.
    {
        let server = start_audio_server(&model_dir, json!({"memory": 6, "draft": 7}));
        let (ct, body) = build_multipart(&[("file", Some("speech.wav"), &speech_wav)]);
        let (code, _, resp_body) = post_raw(&server, "/v1/audio/transcriptions", &ct, &body);
        assert_eq!(code, 200, "{}", String::from_utf8_lossy(&resp_body));
        let json_resp: Value = serde_json::from_slice(&resp_body).expect("JSON");
        assert_eq!(json_resp["text"].as_str().unwrap(), expected_text);

        let (code, status) = ask(&server, "GET", "/v1/models/nemotron-3.5-transcribe", None);
        assert_eq!(code, 200);
        assert_eq!(status["status"]["drafts"], 7);
        let cache = &status["status"]["expert_cache"];
        let counters = &status["status"]["engine_counters"];
        let slots = cache["slots"].as_u64().unwrap();
        let total_experts = counters["routed_layers"].as_u64().unwrap()
            * counters["experts_per_layer"].as_u64().unwrap();
        let misses = cache["misses"].as_u64().unwrap();
        assert!(
            slots < total_experts && misses > 0,
            "expected SSD streaming with MTP active, got cache={cache} counters={counters}"
        );
    }
}
