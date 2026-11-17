//! The yield and resume control. A yield to the least memory takes effect
//! between engine calls, and a resume restores the configured limits. A retry
//! of either changes nothing, and another operation cannot yield while one owns
//! the yield. No yield starts while the server stops. The first test runs the
//! worker without a model. The second asks a server of the checkpoint over
//! HTTP, and is skipped without EREMITE_CHECKPOINT. EREMITE_EXPERTS optionally
//! selects GGUF experts, and EREMITE_SERVER picks the server binary (default
//! build/eremite-server):
//!
//!     EREMITE_CHECKPOINT=DIR cargo test --release \
//!       --manifest-path server/Cargo.toml --test worker_control -- --nocapture

use std::net::TcpListener;
use std::path::PathBuf;

use eremite_server::{config, control, worker};
use serde_json::{Value, json};

mod common;
use common::{Server, http, root};

/// Returns the yield request of `operation_id` for the least memory.
fn ask_for_least(operation_id: &str) -> control::YieldRequest {
    control::YieldRequest {
        operation_id: operation_id.into(),
        memory_gib: Some(0.0),
        state_cache_gib: Some(0.0),
    }
}

/// Returns the resume request of `operation_id`.
fn ask_to_resume(operation_id: &str) -> control::ResumeRequest {
    control::ResumeRequest {
        operation_id: operation_id.into(),
    }
}

/// Reads the limit `kind` of the limits `name` in the control's `status`.
fn read_limit(status: &Value, name: &str, kind: &str) -> u64 {
    status[name][kind].as_u64().expect("a limit")
}

/// Asks the worker to publish until the control leaves `state`, and returns
/// the control's status then.
fn wait_while(runtime: &tokio::runtime::Runtime, state: &str) -> Value {
    loop {
        let status = worker::build_control_status_json();
        if status["state"] != state {
            return status;
        }
        runtime.block_on(worker::refresh_status());
    }
}

#[test]
fn a_yield_takes_effect_and_a_resume_restores_the_limits() {
    assert!(
        worker::start(Vec::new(), config::Budgets::default(), None, None, None),
        "the worker starts"
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let before = worker::build_control_status_json();
    assert_eq!(before["state"], "running", "{before}");
    assert_eq!(before["effective_limits"], before["configured_limits"]);
    let (code, status) = worker::start_yield(ask_for_least("first")).expect("the yield starts");
    assert_eq!((code, status["state"].as_str()), (202, Some("yielding")));
    let yielded = wait_while(&runtime, "yielding");
    assert_eq!(yielded["state"], "yielded", "{yielded}");
    for kind in ["model_bytes", "state_cache_bytes"] {
        assert!(
            read_limit(&yielded, "effective_limits", kind)
                <= read_limit(&before, "configured_limits", kind),
            "{yielded}"
        );
    }
    // A retry of the yield that is done gets 200. Another operation cannot
    // yield now.
    let retry = worker::start_yield(ask_for_least("first")).map(|(code, _)| code);
    assert_eq!(retry.ok(), Some(200));
    let refused = worker::start_yield(ask_for_least("second")).expect_err("a conflict");
    assert_eq!(
        (refused.http_status, refused.code),
        (409, "operation_conflict")
    );
    let (code, _) = worker::start_resume(ask_to_resume("first")).expect("the resume starts");
    assert_eq!(code, 202);
    let resumed = wait_while(&runtime, "resuming");
    assert_eq!(resumed["state"], "running", "{resumed}");
    assert_eq!(resumed["effective_limits"], before["configured_limits"]);
    // A retry of the resume changes nothing and names the operation that
    // ended.
    let (code, status) = worker::start_resume(ask_to_resume("first")).expect("a retry");
    assert_eq!(
        (code, status["resumed_operation_id"].as_str()),
        (200, Some("first"))
    );
    worker::request_stop();
    let refused =
        worker::start_yield(ask_for_least("third")).expect_err("no yield during the stop");
    assert_eq!(
        (refused.http_status, refused.code),
        (503, "server_stopping")
    );
    runtime.block_on(worker::wait_for_stop());
    worker::stop();
}

/// GET /v1/control shows the control's status. POST /v1/control/yield and
/// /v1/control/resume start a yield and a resume, and the model's status shows
/// the yield. A body that is no control request gets 400, and another method
/// gets 405. The server loads no weights, since no chat request runs.
#[test]
fn the_control_routes_answer_over_http() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT (and EREMITE_EXPERTS)");
        return;
    };
    let program = std::env::var("EREMITE_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/eremite-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port();
    let key = format!("control-test-{}-{port}-key", std::process::id());
    // No saved states: the test writes nothing to the SSD.
    let mut args: Vec<String> = ["-m", &checkpoint, "--memory", "56", "--state-disk", "0"]
        .map(String::from)
        .to_vec();
    if let Ok(experts) = std::env::var("EREMITE_EXPERTS") {
        args.extend(["--experts".to_string(), experts]);
    }
    args.extend(["--serve".to_string(), port.to_string()]);
    let server = Server::start(&program, &args, &key, port);
    let ask = |method: &str, path: &str, body: Option<&Value>| {
        let (code, answer) = http(port, &key, method, path, body);
        (
            code,
            serde_json::from_slice::<Value>(&answer).unwrap_or(Value::Null),
        )
    };
    // Each model status read waits for the worker to publish, so the control's
    // status reads follow the worker's turns.
    let wait_while = |state: &str, mut status: Value| {
        while status["state"] == state {
            server.status();
            status = ask("GET", "/v1/control", None).1;
        }
        status
    };
    let (code, status) = ask("GET", "/v1/control", None);
    assert_eq!(
        (code, status["state"].as_str()),
        (200, Some("running")),
        "{status}"
    );
    let least = json!({"operation_id": "http-test", "memory_gib": 0, "state_cache_gib": 0});
    let (code, status) = ask("POST", "/v1/control/yield", Some(&least));
    assert_eq!(code, 202, "{status}");
    let status = wait_while("yielding", status);
    assert_eq!(status["state"], "yielded", "{status}");
    assert_eq!(server.status()["phase"], "yielded");
    let (code, error) = ask(
        "POST",
        "/v1/control/yield",
        Some(&json!({"operation_id": ""})),
    );
    assert_eq!(
        (code, error["error"]["code"].as_str()),
        (400, Some("invalid_control_request")),
        "{error}"
    );
    let (code, error) = ask(
        "POST",
        "/v1/control/yield",
        Some(&json!({"operation_id": "other", "memory": 1})),
    );
    assert_eq!(code, 400, "an unknown field: {error}");
    let (code, _) = ask("GET", "/v1/control/resume", None);
    assert_eq!(code, 405);
    let (code, status) = ask(
        "POST",
        "/v1/control/resume",
        Some(&json!({"operation_id": "http-test"})),
    );
    assert_eq!(code, 202, "{status}");
    let status = wait_while("resuming", status);
    assert_eq!(status["state"], "running", "{status}");
    println!("the yield and the resume ran: {status}");
}
