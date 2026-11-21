//! MiniMax-H3 Video through HTTP: POST /v1/videos, GET /v1/videos,
//! GET /v1/videos/{id}, GET /v1/videos/{id}/content and DELETE /v1/videos/{id}.
//! Verifies model listing, Codex catalog exclusion, chat/decision/image
//! refusals, HTTP 400 validation errors, asynchronous video job polling, valid
//! MP4 content download, job deletion and strict prefill-vs-video status
//! counters.
//!
//! Set EREMITE_MINIMAX_H3_VIDEO_MODEL_DIR, EREMITE_MINIMAX_H3_TEXT_ENCODER,
//! EREMITE_MINIMAX_H3_VIDEO_DIT, EREMITE_MINIMAX_H3_VIDEO_VAE,
//! EREMITE_MINIMAX_H3_AUDIO_VAE, and optionally EREMITE_MINIMAX_H3_VIDEO_LORA
//! and EREMITE_MINIMAX_H3_VIDEO_STEPS. The test starts build/eremite-server,
//! so build it first. Run one model process at a time:
//!
//!     make server && \
//!     EREMITE_MINIMAX_H3_VIDEO_MODEL_DIR=... \
//!     EREMITE_MINIMAX_H3_TEXT_ENCODER=... \
//!     EREMITE_MINIMAX_H3_VIDEO_DIT=... \
//!     EREMITE_MINIMAX_H3_VIDEO_VAE=... \
//!     EREMITE_MINIMAX_H3_AUDIO_VAE=... \
//!     EREMITE_MINIMAX_H3_VIDEO_LORA=... \
//!     EREMITE_MINIMAX_H3_VIDEO_STEPS=4 \
//!       cargo test --release --manifest-path server/Cargo.toml \
//!       --target-dir build/cargo --test videos -- --nocapture

use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

mod common;
use common::{Server, build_config, http, root};

/// Starts a server configured with the MiniMax-H3 Video checkpoints from the
/// environment, or returns None when the env vars are unset.
fn start_video_server() -> Option<(Server, PathBuf)> {
    let model_dir = std::env::var("EREMITE_MINIMAX_H3_VIDEO_MODEL_DIR").ok()?;
    let text_encoder = std::env::var("EREMITE_MINIMAX_H3_TEXT_ENCODER").ok()?;
    let dit = std::env::var("EREMITE_MINIMAX_H3_VIDEO_DIT").ok()?;
    let vae = std::env::var("EREMITE_MINIMAX_H3_VIDEO_VAE").ok()?;
    let audio_vae = std::env::var("EREMITE_MINIMAX_H3_AUDIO_VAE").ok()?;
    let lora = std::env::var("EREMITE_MINIMAX_H3_VIDEO_LORA").ok();
    let steps: i32 = std::env::var("EREMITE_MINIMAX_H3_VIDEO_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("videos-test-{}-{port}-key", std::process::id());
    let video_dir =
        std::env::temp_dir().join(format!("eremite-videos-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&video_dir);
    let mut model_opts = json!({
        "text_encoder": text_encoder,
        "dit": dit,
        "vae": vae,
        "audio_vae": audio_vae,
        "steps": steps,
        "variant": "fl2va"
    });
    if let Some(lora_path) = lora {
        model_opts["lora"] = json!(lora_path);
    }
    let config = build_config(
        port,
        json!({
            "reserve": 2,
            "state_disk": 0,
            "http_timeout": 300,
            "video_dir": video_dir
        }),
        &model_dir,
        None,
        model_opts,
    );
    Some((Server::start(&program, &config, &key, port), video_dir))
}

/// Sends a JSON HTTP request and parses the response body as JSON.
fn ask(server: &Server, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let (code, answer) = http(server.port, &server.key, method, path, body);
    (code, serde_json::from_slice(&answer).unwrap_or(Value::Null))
}

#[test]
fn minimax_h3_video_serves_openai_videos_api_over_http() {
    let Some((server, video_dir)) = start_video_server() else {
        return;
    };

    // 1. Model listing and non-video endpoint refusals.
    let (code, models) = ask(&server, "GET", "/v1/models", None);
    assert_eq!(code, 200);
    assert_eq!(models["data"][0]["id"], json!("minimax-h3-video"));

    let (code, codex) = ask(&server, "GET", "/v1/codex/models", None);
    assert_eq!(code, 200);
    assert_eq!(codex["models"], json!([]));

    let (code, _) = ask(
        &server,
        "POST",
        "/v1/chat/completions",
        Some(
            &json!({"model": "minimax-h3-video", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    );
    assert_eq!(code, 400);

    let (code, _) = ask(
        &server,
        "POST",
        "/v1/images/generations",
        Some(&json!({"model": "minimax-h3-video", "prompt": "hi"})),
    );
    assert_eq!(code, 400);

    // 2. Empty list, 404 lookups and HTTP 400 validation checks.
    let (code, list) = ask(&server, "GET", "/v1/videos", None);
    assert_eq!(code, 200);
    assert_eq!(list["object"], json!("list"));
    assert_eq!(list["data"], json!([]));

    let (code, _) = ask(&server, "GET", "/v1/videos/video_missing", None);
    assert_eq!(code, 404);
    let (code, _) = ask(&server, "DELETE", "/v1/videos/video_missing", None);
    assert_eq!(code, 404);

    for bad in [
        json!({"model": "minimax-h3-video"}),
        json!({"model": "minimax-h3-video", "prompt": "hi", "size": "bad"}),
        json!({"model": "minimax-h3-video", "prompt": "hi", "seconds": 4, "frames": 22}),
        json!({"model": "minimax-h3-video", "prompt": "hi", "guidance_scale": 5.0}),
        json!({"model": "minimax-h3-video", "prompt": "hi", "ref_images": ["AQID"]}),
    ] {
        let (code, _) = ask(&server, "POST", "/v1/videos", Some(&bad));
        assert_eq!(code, 400, "expected 400 for {bad}");
    }

    // 3. Submit a short T2VA video job and poll until completion.
    let (code, created) = ask(
        &server,
        "POST",
        "/v1/videos",
        Some(&json!({
            "model": "minimax-h3-video",
            "prompt": "A calm lake reflecting pine trees at golden hour, gentle water ripples",
            "size": "512x288",
            "frames": 22,
            "steps": 4,
            "seed": 1234
        })),
    );
    assert_eq!(code, 200, "create failed: {created}");
    let id = created["id"].as_str().expect("video id").to_string();
    assert!(id.starts_with("video_"), "{created}");
    assert_eq!(created["object"], json!("video"));

    let deadline = Instant::now() + Duration::from_secs(180);
    let completed = loop {
        let (code, job) = ask(&server, "GET", &format!("/v1/videos/{id}"), None);
        assert_eq!(code, 200, "get failed: {job}");
        match job["status"].as_str() {
            Some("completed") => break job,
            Some("failed") => panic!("video job failed: {job}"),
            _ => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for video: {job}"
                );
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    };
    assert_eq!(completed["progress"], json!(100));
    assert_eq!(completed["size"], json!("512x288"));
    assert_eq!(completed["frames"], json!(22));
    assert_eq!(completed["fps"], json!(24));

    // 4. Download MP4 content and check ISO BMFF ftyp signature.
    let (code, mp4) = http(
        server.port,
        &server.key,
        "GET",
        &format!("/v1/videos/{id}/content"),
        None,
    );
    assert_eq!(code, 200);
    assert!(mp4.len() > 1024, "MP4 too small: {} bytes", mp4.len());
    assert_eq!(&mp4[4..8], b"ftyp", "expected MP4 ftyp header");

    let (code, _) = ask(
        &server,
        "GET",
        &format!("/v1/videos/{id}/content?variant=thumbnail"),
        None,
    );
    assert_eq!(code, 400);

    // 5. Verify list and status counters (strict prefill vs video separation).
    let (code, list) = ask(&server, "GET", "/v1/videos", None);
    assert_eq!(code, 200);
    assert_eq!(list["data"][0]["id"], json!(id));

    let (code, model_status) = ask(&server, "GET", "/v1/models/minimax-h3-video", None);
    assert_eq!(code, 200);
    let st = &model_status["status"];
    assert_eq!(st["model_decode"]["computed_rows"], json!(0));
    assert_eq!(st["totals"]["generated_videos"], json!(1));
    assert!(
        st["totals"]["video_seconds"].as_f64().unwrap_or(0.0) > 0.0,
        "expected positive video_seconds: {st}"
    );
    assert!(
        st["totals"]["prefilled_tokens"].as_u64().unwrap_or(0) > 0,
        "expected positive prefilled_tokens: {st}"
    );
    assert!(
        st["totals"]["prefill_seconds"].as_f64().unwrap_or(0.0) > 0.0,
        "expected positive prefill_seconds: {st}"
    );
    assert_eq!(st["engine_counters"]["generated_videos"], json!(1));
    assert_eq!(st["engine_counters"]["video_steps"], json!(4));
    assert_eq!(st["engine_counters"]["video_stage"], json!(0));

    // 6. Delete video and verify MP4 removal and 404.
    let mp4_path = video_dir.join(format!("{id}.mp4"));
    assert!(mp4_path.exists());
    let (code, deleted) = ask(&server, "DELETE", &format!("/v1/videos/{id}"), None);
    assert_eq!(code, 200);
    assert_eq!(deleted["deleted"], json!(true));
    assert!(!mp4_path.exists(), "deleted video file should be removed");
    let (code, _) = ask(&server, "GET", &format!("/v1/videos/{id}"), None);
    assert_eq!(code, 404);

    let _ = std::fs::remove_dir_all(&video_dir);
}
