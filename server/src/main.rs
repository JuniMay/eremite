//! eremite-server: the OpenAI Chat Completions and Responses APIs and Codex's
//! model catalog, over eremite's engine, for one model or several in one
//! process.
//!
//! - The engine (src/eremite.h, and engine.rs here) runs the models. It owns
//!   the stop on SIGINT or SIGTERM.
//! - The worker (worker.rs) runs every model's requests on the engine thread.
//! - The state cache (state_cache.rs) keeps every model's states between
//!   requests.
//! - Each model family's chat format (chat/) renders its requests and reads its
//!   replies.

use eremite_server::{chat, codex, config, control, engine, http, worker};

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use eremite_server::chat::{Api, Family, deepseek};
use eremite_server::config::ModelConfig;

/// The fewest characters that the API key has.
/// The reason for this value is unknown.
const MIN_KEY_CHARS: usize = 16;
/// The most characters that the API key has.
/// The reason for this value is unknown.
const MAX_KEY_CHARS: usize = 256;

/// The command line's help.
const USAGE: &str =
    "usage: eremite-server [--config FILE | -m MODEL [model options]] [server options]
  Without --config or -m, read ~/.config/eremite/server.json.
  -m DIR          a supported model checkpoint directory, with its options below
  --config FILE   server settings and models in JSON; a request's model field picks its model
  --serve PORT    override the JSON port; serve OpenAI Chat Completions at http://HOST:PORT/v1
  --host ADDR     the IPv4 address to serve on (default 127.0.0.1)
  --api-key-file FILE  the API key clients must send (16 to 256 characters)
  --http-timeout S   seconds a request's head, each part of its body and a stalled write may take
                  (default 60)
  --keepalive S   seconds between the comment lines of a streamed response that waits for its
                  first token (default 15)
  --search-timeout S  seconds each fetch, search and health check of Codex's web tool may take,
                  and SearXNG's start (default 30)
  --searxng-idle S    seconds without a search before SearXNG stops (default 600)
  --search-session-idle S  seconds without a call before the web tool forgets a session's
                  references (default 7200)
  --memory GB     an upper limit of model memory for every model, state memory included
                  (default: none; the memory that other programs leave, less --reserve, decides)
  --reserve GB    memory kept available to other programs, 2-1024 GiB (default 10)
  --gpu-timeout S seconds a wait for the GPU may last before the server ends (default 60)
  --progress-log S   seconds between a decoding request's progress lines in the log, 0 none
                  (default 10)
  --state-cache GB   total states and snapshots across all models, GiB (default 8)
  --state-disk GB    room on the SSD for evicted states, written when evicted and at quit and
                  read back for a prompt that starts like one, GiB; 0 off (default 32)
  --state-dir DIR    the saved states' folder (default ~/.cache/eremite/states)
  --state-disk-tokens N  save a state only when it holds at least N tokens no other kept state
                  holds (default 8192)
  --idle-unload S seconds a model may go without requests before its weights and experts
                  unload (its states stay; the next request loads it again), 0 never
                  (default 600; a model's own setting wins)
  --max-waiting-requests N  requests that may wait for one model; another gets HTTP 429
                  (default 16)
  --max-cached-states N  states that the state cache keeps, for all models together
                  (default 256)
  --snapshot-interval N  positions between the snapshots of a long prompt, for a model that
                  resumes only from snapshots (default 2048)
  --render FILE   print the prompt of a JSON test case of DeepSeek's encoding/tests, and exit
 a model's options:
  --id NAME       the name requests give in model (default: deepseek-v4.1-flash)
  --rows-per-call N  rows of one decoding call at most, at most the model's own; fewer rows per
                  call run more calls beside each other (default: the model's own, 6 for
                  DeepSeek)
  -t T, --temperature T  the temperature of requests that set none (default 1)
  --top-p P       the top_p of requests that set none, 0 to 1 (default 1)
  --top-k K       the top_k of requests that set none, 0 for all (default 0)
  --seed N        the seed of requests that set none (default: random for each)
  --think N       thinking effort 1-100 of requests that set none; 0 turns thinking off
  --ctx N         context capacity in tokens, at most 1048576 and the model's own (default: the
                  most)
  --draft N       at most N DSpark proposals per pass, 0-5; each pass runs the count that has
                  given the most tokens per second
  --lookup off|N  verify up to N (1-15) context-copy drafts; --draft supplies fallback
  DeepSeek only:
  --ced off|N     approximate prefill: replay N decoder tokens, 128-1048576 (default off), plus
                  the rewind margin (8)
  --prefill-prune  skip decoder batches outside required history; CED and substitution must be
                  off (default off)
  --experts GGUF  read the routed experts from a GGUF file or set of shards
  --packed-state-rows  keep a state's compressed rows and index keys packed: the same values in
                  about 28% of BF16's bytes (default off)
";

/// Prints the command line's help and ends the process with status 2.
fn exit_with_usage() -> ! {
    eprint!("{USAGE}");
    exit(2)
}

/// Prints `message` and ends the process with status 1.
fn die(message: &str) -> ! {
    eprintln!("eremite-server: {message}");
    exit(1)
}

/// The settings that the command line gives.
struct Arguments {
    /// -m's model and its options, or --config's models.
    models: Vec<ModelConfig>,
    /// The server-wide budgets and limits.
    budgets: config::Budgets,
    /// The server-wide settings besides the budgets.
    server: config::Server,
    /// The JSON test case whose prompt --render prints, if given.
    render: Option<String>,
}

/// Parses the command line's arguments `args`, without the program's name.
/// Without -m and --config, it reads the default configuration file. The
/// command line's server settings override the file's. An invalid argument ends
/// the process with the help or a message.
fn parse_arguments(args: &[String]) -> Arguments {
    let mut parsed = Arguments {
        models: Vec::new(),
        budgets: config::Budgets::default(),
        server: config::Server::default(),
        render: None,
    };
    let mut model: Option<ModelConfig> = None;
    // -m's options, in their order.
    let mut options: Vec<(String, Value)> = Vec::new();
    let mut budgets: Vec<(String, Value)> = Vec::new();
    let mut server: Vec<(String, Value)> = Vec::new();
    let mut file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let switch = |name: &str| (name.to_string(), Value::Bool(true));
        match flag {
            "--prefill-prune" | "--packed-state-rows" => {
                options.push(switch(&flag[2..].replace('-', "_")))
            }
            _ => {
                let Some(text) = args.get(i + 1) else {
                    exit_with_usage()
                };
                i += 1;
                match flag {
                    "-m" => {
                        if model.is_some() {
                            die("-m serves one model; several go in a --config file");
                        }
                        model = Some(ModelConfig::new(text));
                    }
                    "--config" => file = Some(text.into()),
                    "--memory"
                    | "--reserve"
                    | "--gpu-timeout"
                    | "--progress-log"
                    | "--state-cache"
                    | "--state-disk"
                    | "--state-disk-tokens"
                    | "--max-waiting-requests"
                    | "--max-cached-states"
                    | "--snapshot-interval"
                    | "--idle-unload" => {
                        let name = flag[2..].replace('-', "_");
                        let value = serde_json::from_str::<Value>(text).unwrap_or(Value::Null);
                        budgets.push((name, value));
                    }
                    "--serve" => {
                        let port: u16 = text
                            .parse()
                            .unwrap_or_else(|_| die(&format!("--serve: not a port: {text}")));
                        server.push(("port".into(), Value::from(port)));
                    }
                    "--host" | "--api-key-file" | "--state-dir" => {
                        server.push((flag[2..].replace('-', "_"), Value::String(text.clone())))
                    }
                    "--http-timeout"
                    | "--keepalive"
                    | "--search-timeout"
                    | "--searxng-idle"
                    | "--search-session-idle" => {
                        let value = serde_json::from_str::<Value>(text).unwrap_or(Value::Null);
                        server.push((flag[2..].replace('-', "_"), value));
                    }
                    "--render" => parsed.render = Some(text.clone()),
                    _ => {
                        let name = match flag {
                            "-t" => "temperature".to_string(),
                            long if long.starts_with("--") => long[2..].replace('-', "_"),
                            _ => exit_with_usage(),
                        };
                        // Numbers stay numbers. The rest, such as off, paths
                        // and names, is text.
                        let value = serde_json::from_str::<Value>(text)
                            .ok()
                            .filter(Value::is_number)
                            .unwrap_or(Value::String(text.clone()));
                        options.push((name, value));
                    }
                }
            }
        }
        i += 1;
    }
    if model.is_none() && file.is_none() && parsed.render.is_none() {
        file = Some(config::default_path().unwrap_or_else(|error| die(&error)));
    }
    match (model, file) {
        (Some(_), Some(_)) => {
            die("-m and --config: give one model with -m, or several in the file")
        }
        (Some(mut model), None) => {
            for (name, value) in &options {
                model
                    .set(name, value)
                    .unwrap_or_else(|error| die(&format!("--{}: {error}", name.replace('_', "-"))));
            }
            parsed.models.push(model);
        }
        (None, Some(file)) => {
            if let Some((name, _)) = options.first() {
                die(&format!(
                    "--{}: with --config, each model's options are in the file",
                    name.replace('_', "-")
                ));
            }
            let config = config::read(&file).unwrap_or_else(|error| die(&error));
            parsed.models = config.models;
            parsed.budgets = config.budgets;
            parsed.server = config.server;
        }
        (None, None) => {}
    }
    for (name, value) in budgets {
        parsed
            .budgets
            .set(&name, &value)
            .unwrap_or_else(|error| die(&error));
    }
    for (name, value) in server {
        parsed
            .server
            .set(&name, &value)
            .unwrap_or_else(|error| die(&error));
    }
    parsed
}

#[test]
fn one_model_takes_its_options_from_the_command_line() {
    let parsed = parse_arguments(
        &[
            "-m",
            "/m/d",
            "--prefill-prune",
            "--packed-state-rows",
            "--lookup",
            "15",
            "--draft",
            "2",
            "--ced",
            "off",
            "--serve",
            "8008",
        ]
        .map(String::from),
    );
    let model = &parsed.models[0];
    assert!(model.engine.deepseek().unwrap().prefill_prune);
    assert!(model.engine.deepseek().unwrap().packed_state_rows);
    assert_eq!(
        (
            model.engine.lookup,
            model.engine.draft,
            model.engine.deepseek().unwrap().ced
        ),
        (15, 2, 0)
    );
    assert_eq!(
        (
            model.temperature,
            model.top_p,
            parsed.budgets.state_cache_gb
        ),
        (None, None, 8.0)
    );
    assert_eq!(parsed.server.port, Some(8008));
    let parsed = parse_arguments(
        &[
            "-m",
            "/m/small",
            "-t",
            "0.6",
            "--top-k",
            "20",
            "--id",
            "small",
            "--state-cache",
            "4",
        ]
        .map(String::from),
    );
    let model = &parsed.models[0];
    assert_eq!(
        (
            model.temperature,
            model.top_k,
            model.id.as_deref(),
            parsed.budgets.state_cache_gb
        ),
        (Some(0.6), Some(20), Some("small"), 4.0)
    );
}

#[test]
fn the_queue_and_state_limits_come_from_the_command_line() {
    let parsed = parse_arguments(
        &[
            "-m",
            "/m/d",
            "--max-waiting-requests",
            "4",
            "--max-cached-states",
            "32",
            "--snapshot-interval",
            "1024",
        ]
        .map(String::from),
    );
    let budgets = &parsed.budgets;
    assert_eq!(
        (
            budgets.max_waiting_requests,
            budgets.max_cached_states,
            budgets.snapshot_interval
        ),
        (4, 32, 1024)
    );
}

/// A file that a test writes. It is removed when the test ends.
#[cfg(test)]
struct TestFile(PathBuf);

#[cfg(test)]
impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn command_line_server_settings_override_the_file_in_either_order() {
    let path = std::env::temp_dir().join(format!("eremite-server-{}.json", std::process::id()));
    let config = json!({"port": 8008, "host": "127.0.0.1", "api_key_file": "/dev/null",
                        "reserve": 10, "state_cache": 8, "models": [{"model": "unused-model"}]});
    std::fs::write(&path, config.to_string()).unwrap();
    let written = TestFile(path);
    let file = written.0.to_string_lossy().into_owned();
    let parsed = parse_arguments(&["--config".into(), file.clone()]);
    assert_eq!(parsed.server.port, Some(8008));
    assert_eq!(parsed.server.api_key_file.as_deref(), Some("/dev/null"));
    for before in [false, true] {
        let flags: Vec<String> = [
            "--serve",
            "9008",
            "--host",
            "127.0.0.2",
            "--api-key-file",
            "override-key",
            "--reserve",
            "12",
            "--state-cache",
            "16",
            "--idle-unload",
            "30",
        ]
        .map(String::from)
        .into();
        let config = vec!["--config".into(), file.clone()];
        let parsed = parse_arguments(&if before {
            [flags, config].concat()
        } else {
            [config, flags].concat()
        });
        assert_eq!(
            (
                parsed.server.port,
                parsed.server.host.as_str(),
                parsed.server.api_key_file.as_deref()
            ),
            (Some(9008), "127.0.0.2", Some("override-key"))
        );
        assert_eq!(
            (
                parsed.budgets.reserve_gb,
                parsed.budgets.state_cache_gb,
                parsed.budgets.idle_unload_s
            ),
            (12.0, 16.0, 30.0)
        );
    }
}

/// A served model: its names, its context, its tokenizer and its family's chat
/// format.
struct Served {
    /// The name that requests give in `model`.
    id: String,
    /// The name that the model lists show.
    name: String,
    /// The number of tokens in the model's context.
    ctx: usize,
    /// The model's tokenizer.
    tokenizer: engine::Tokenizer,
    /// The model's chat format, with the model's defaults.
    format: Format,
}

/// A model family's chat format, with a served model's defaults.
enum Format {
    /// DeepSeek V4.1's format (chat/deepseek.rs).
    DeepSeek(deepseek::Defaults),
}

/// What the HTTP handlers share.
struct Shared {
    /// The API key's SHA-256 digest.
    key: [u8; 32],
    /// When the server started, in seconds since the Unix epoch.
    started: u64,
    /// The number of chat requests that arrived. The log names each request by
    /// its number.
    arrivals: AtomicU64,
    /// The served models, in their configured order.
    models: Vec<Served>,
    /// Codex's web tool (POST /v1/alpha/search).
    search: codex::search::Service,
    /// The limits on connections and requests.
    limits: http::Limits,
    /// The time between the comment lines of a streamed response that waits for
    /// its first event.
    keepalive: Duration,
}

/// Finds the index of the model that the request's `body` names, in any case.
/// With one model served, it returns that model for any name. Among several
/// models, it returns an error message when the body names an unknown model, or
/// none.
fn find_model_index(shared: &Shared, body: &[u8]) -> Result<usize, String> {
    /// The part of a request's body that names its model.
    #[derive(serde::Deserialize)]
    struct Named {
        /// The model's name.
        model: Option<String>,
    }
    let name = serde_json::from_slice::<Named>(body)
        .ok()
        .and_then(|named| named.model);
    if let Some(name) = &name
        && let Some(index) = shared
            .models
            .iter()
            .position(|model| name.eq_ignore_ascii_case(&model.id))
    {
        return Ok(index);
    }
    if shared.models.len() == 1 {
        return Ok(0);
    }
    let known: Vec<&str> = shared
        .models
        .iter()
        .map(|model| model.id.as_str())
        .collect();
    Err(match name {
        Some(name) => format!(
            "no model {name:?} here; the models are {}",
            known.join(", ")
        ),
        None => format!(
            "the request names no model; the models are {}",
            known.join(", ")
        ),
    })
}

/// The handlers' state: what they share.
type App = State<Arc<Shared>>;

/// Lets a request through when it carries the API key, as Authorization: Bearer
/// KEY or x-api-key: KEY. GET /health needs no key. It compares the digests in
/// constant time, so the time taken tells nothing about the key. It answers any
/// other request with 401.
async fn authorize(
    State(shared): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    /// The scheme at the start of an Authorization header that carries the key.
    const BEARER: &str = "bearer ";
    if matches!(request.uri().path(), "/health" | "/v1/health") {
        return next.run(request).await;
    }
    let headers: &HeaderMap = request.headers();
    let key = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .get(..BEARER.len())
                .filter(|scheme| scheme.eq_ignore_ascii_case(BEARER))
                .map(|_| value[BEARER.len()..].trim_start())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        });
    let given: [u8; 32] = Sha256::digest(key.unwrap_or_default().as_bytes()).into();
    let same = given
        .iter()
        .zip(shared.key.iter())
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0;
    if key.is_some() && same {
        return next.run(request).await;
    }
    engine::log(&format!(
        "{}: 401: {} {} with {}",
        peer.ip(),
        request.method(),
        request.uri().path(),
        if key.is_some() {
            "a wrong API key"
        } else {
            "no API key"
        }
    ));
    chat::build_error_response(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "a valid API key is required (Authorization: Bearer KEY)",
    )
}

/// Builds a 200 response with the JSON text `body`.
fn build_json_response(body: String) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// Answers GET /health: the server is up.
async fn report_health() -> Response {
    build_json_response(r#"{"status": "ok"}"#.into())
}

/// Builds the response of a control request from its `result`: the HTTP status
/// and the control's status, or the control's error.
fn build_control_response(result: Result<(u16, Value), control::Error>) -> Response {
    let (status, body) = match result {
        Ok((status, body)) => (status, body),
        Err(error) => (
            error.http_status,
            json!({"error": {
                "type":"control_error", "code":error.code, "message":error.message,
            }}),
        ),
    };
    (
        StatusCode::from_u16(status).unwrap(),
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

/// Answers GET /v1/control with the control's status.
async fn report_control_status() -> Response {
    build_json_response(worker::build_control_status_json().to_string())
}

/// Answers POST /v1/control/yield: starts a yield of the memory that `body`
/// asks for.
async fn request_yield(body: axum::body::Bytes) -> Response {
    let request = match serde_json::from_slice::<control::YieldRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            return chat::build_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &error.to_string(),
            );
        }
    };
    build_control_response(worker::start_yield(request))
}

/// Answers POST /v1/control/resume: starts the resume that `body` names.
async fn request_resume(body: axum::body::Bytes) -> Response {
    let request = match serde_json::from_slice::<control::ResumeRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            return chat::build_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &error.to_string(),
            );
        }
    };
    build_control_response(worker::start_resume(request))
}

/// Builds `model`'s object as GET /v1/models lists it: OpenAI's fields, and the
/// context's length as servers commonly add it.
fn build_model_object(shared: &Shared, model: &Served) -> Value {
    json!({
        "id": model.id, "object": "model", "created": shared.started, "owned_by": "eremite",
        "name": model.name, "context_length": model.ctx, "max_model_len": model.ctx,
        "max_output_tokens": (model.ctx as i64).min(chat::DEFAULT_MAX_TOKENS),
    })
}

/// Answers GET /v1/models with the served models' objects.
async fn list_served_models(State(shared): App) -> Response {
    let data: Vec<Value> = shared
        .models
        .iter()
        .map(|model| build_model_object(&shared, model))
        .collect();
    build_json_response(json!({"object": "list", "data": data}).to_string())
}

/// Answers GET /v1/codex/models with Codex's model catalog (codex/mod.rs) of
/// the DeepSeek models. They answer the Responses API that Codex uses.
async fn list_codex_models(State(shared): App) -> Response {
    let models = shared.models.iter().map(|model| match &model.format {
        Format::DeepSeek(defaults) => codex::CatalogModel {
            model_id: &model.id,
            display_name: &model.name,
            context_window: model.ctx,
            default_effort: defaults.is_thinking_on.then_some(defaults.effort),
        },
    });
    build_json_response(codex::build_catalog(models).to_string())
}

/// Answers POST /v1/alpha/search: runs the commands of Codex's web tool
/// (codex/search.rs). Each command's result, or why it failed, is text in
/// `output`.
async fn answer_search(
    State(shared): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    body: Body,
) -> Response {
    let peer = peer.ip().to_string();
    let body = match http::read_body(body, shared.limits).await {
        Ok(body) => body,
        Err(error) => return refuse_request(&peer, error),
    };
    let request: codex::search::SearchRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return refuse_request(
                &peer,
                chat::Error::new_bad_request(format!("not a search request: {error}")),
            );
        }
    };
    let output = shared.search.answer_call(request).await;
    build_json_response(
        json!({"output": output, "encrypted_output": null, "results": null}).to_string(),
    )
}

/// Answers GET /v1/models/{id} with the model's object and its live status
/// under "status". The status shows what the model is doing, its requests, its
/// state cache, its counters, its expert cache and the process's memory.
async fn report_model_status(State(shared): App, Path(id): Path<String>) -> Response {
    let Some(index) = shared
        .models
        .iter()
        .position(|model| id.eq_ignore_ascii_case(&model.id))
    else {
        return chat::build_error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            &format!("no model {id:?} here"),
        );
    };
    let mut object = build_model_object(&shared, &shared.models[index]);
    worker::refresh_status().await;
    object["status"] = worker::build_model_status_json(index);
    build_json_response(object.to_string())
}

/// Logs `error` from `peer`, which ends a request before its response starts,
/// and builds the error's response.
fn refuse_request(peer: &str, error: chat::Error) -> Response {
    engine::log(&format!(
        "{peer}: {}: {}",
        error.status.as_u16(),
        error.message
    ));
    let error_type = match error.status {
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::BAD_REQUEST | StatusCode::REQUEST_TIMEOUT | StatusCode::PAYLOAD_TOO_LARGE => {
            "invalid_request_error"
        }
        _ => "server_error",
    };
    chat::build_error_response(error.status, error_type, &error.message)
}

/// Submits `request` to the worker. Returns its cancel guard and the receiver
/// of its events, or a 429 error when the worker is busy or stopping.
fn submit_request(
    request: worker::Request,
) -> Result<
    (
        worker::CancelGuard,
        tokio::sync::mpsc::UnboundedReceiver<worker::Event>,
    ),
    chat::Error,
> {
    worker::submit(request).map_err(|error| match error {
        worker::SubmitError::Busy(message) => chat::Error {
            status: StatusCode::TOO_MANY_REQUESTS,
            message,
        },
    })
}

/// Answers POST /v1/chat/completions.
async fn answer_chat_completions(
    State(shared): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    body: Body,
) -> Response {
    answer_generation(shared, peer, body, Api::ChatCompletions).await
}

/// Answers POST /v1/responses.
async fn answer_responses(
    State(shared): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    body: Body,
) -> Response {
    answer_generation(shared, peer, body, Api::Responses).await
}

/// Answers a generation request of either API from `peer`. It reads the
/// request's `body` and finds its model. It renders and queues the request. It
/// answers as a stream or as one response.
async fn answer_generation(
    shared: Arc<Shared>,
    peer: SocketAddr,
    body: Body,
    api: Api,
) -> Response {
    let peer = peer.ip().to_string();
    let body = match http::read_body(body, shared.limits).await {
        Ok(body) => body,
        Err(error) => return refuse_request(&peer, error),
    };
    let arrival = shared.arrivals.fetch_add(1, Ordering::Relaxed) + 1;
    engine::log(&format!(
        "http {arrival}: arrived from {peer}, {} body bytes",
        body.len()
    ));
    let index = match find_model_index(&shared, &body) {
        Ok(index) => index,
        Err(message) => {
            return chat::build_error_response(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                &message,
            );
        }
    };
    // Rendering and tokenization take a while for long prompts.
    let (from, handlers) = (peer.clone(), shared.clone());
    let queued = tokio::task::spawn_blocking(move || match &handlers.models[index].format {
        Format::DeepSeek(defaults) => {
            let prepared =
                deepseek::prepare_request(&body, defaults, handlers.models[index].tokenizer, api)?;
            let note = match api {
                Api::ChatCompletions => format!(
                    "{} messages, {} tools",
                    prepared.message_count, prepared.tool_count
                ),
                Api::Responses => format!(
                    "Responses API: {} input items, {} tools ({} not supported and left out)",
                    prepared.message_count, prepared.tool_count, prepared.dropped_tool_count
                ),
            };
            let note = format!(
                "http {arrival} from {from}: {note}, {}; {}",
                if prepared.is_streamed {
                    "streamed"
                } else {
                    "one response"
                },
                prepared.describe_thinking()
            );
            let (cancel, events) = submit_request(prepared.build_worker_request(index, note)?)?;
            Ok::<_, chat::Error>((Prepared::DeepSeek(prepared), cancel, events))
        }
    })
    .await;
    match queued {
        Ok(Ok((Prepared::DeepSeek(prepared), cancel, events))) => {
            if prepared.is_streamed {
                deepseek::stream_response(prepared, cancel, events, shared.keepalive)
            } else {
                deepseek::collect_response(prepared, cancel, events).await
            }
        }
        Ok(Err(error)) => refuse_request(&peer, error),
        Err(error) => chat::build_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &error.to_string(),
        ),
    }
}

/// A request that its family's chat format prepared for the worker.
enum Prepared {
    /// A request in DeepSeek V4.1's format.
    DeepSeek(deepseek::PreparedRequest),
}

/// Answers a request for an unknown path with 404.
async fn refuse_unknown_endpoint() -> Response {
    chat::build_error_response(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "no such endpoint",
    )
}

/// Answers a request with a method other than GET with 405.
async fn refuse_other_than_get() -> Response {
    chat::build_error_response(
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "use GET",
    )
}

/// Answers a request with a method other than POST with 405.
async fn refuse_other_than_post() -> Response {
    chat::build_error_response(
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "use POST",
    )
}

/// Starts the server: it reads the settings, listens, starts the worker with
/// the models, and serves the routes until the stop.
fn main() {
    let arguments = parse_arguments(&std::env::args().skip(1).collect::<Vec<_>>());
    if let Some(file) = &arguments.render {
        exit(deepseek::render_test_case_file(file));
    }
    let (Some(port), false) = (arguments.server.port, arguments.models.is_empty()) else {
        exit_with_usage()
    };
    let key = match &arguments.server.api_key_file {
        Some(file) => {
            std::fs::read_to_string(file).unwrap_or_else(|_| die(&format!("cannot open {file}")))
        }
        None => std::env::var("EREMITE_API_KEY").unwrap_or_default(),
    };
    let key = key.trim_end_matches(['\n', '\r', ' ']);
    if !(MIN_KEY_CHARS..=MAX_KEY_CHARS).contains(&key.len()) {
        die(&format!(
            "the API key (EREMITE_API_KEY or --api-key-file) must have {MIN_KEY_CHARS} to {MAX_KEY_CHARS} characters"
        ));
    }
    let host: Ipv4Addr = if arguments.server.host == "localhost" {
        Ipv4Addr::LOCALHOST
    } else {
        arguments.server.host.parse().unwrap_or_else(|_| {
            die(&format!(
                "host takes an IPv4 address: {}",
                arguments.server.host
            ))
        })
    };
    // The port is taken before the models load, so that a busy port fails at
    // once.
    let listener = std::net::TcpListener::bind((host, port))
        .unwrap_or_else(|error| die(&format!("cannot listen on {host}:{port}: {error}")));
    listener
        .set_nonblocking(true)
        .unwrap_or_else(|error| die(&format!("cannot configure the listening socket: {error}")));

    // Each model's family, name and context.
    let mut served = Vec::new();
    let mut families = Vec::new();
    for model in &arguments.models {
        let family = Family::of(&model.engine.model).unwrap_or_else(|error| die(&error));
        let mut options = model.engine.clone();
        options
            .configure(family)
            .unwrap_or_else(|error| die(&error));
        let id = model.id.clone().unwrap_or_else(|| match family {
            Family::DeepSeek => deepseek::DEFAULT_MODEL_ID.into(),
        });
        if served
            .iter()
            .any(|settings: &worker::ModelSettings| settings.id.eq_ignore_ascii_case(&id))
        {
            die(&format!("two models named {id}: give one an id"));
        }
        served.push(worker::ModelSettings {
            id,
            options,
            rows_per_call: model.rows_per_call,
            idle_unload: model.idle_unload.unwrap_or(arguments.budgets.idle_unload_s),
        });
        families.push(family);
    }
    // The folder of the saved states. There is none without a folder, or with
    // --state-disk 0.
    let state_dir = (arguments.budgets.state_disk_gb > 0.0)
        .then(|| {
            arguments
                .server
                .state_dir
                .as_ref()
                .map(PathBuf::from)
                .or_else(config::default_state_dir)
        })
        .flatten();
    if !worker::start(
        served.clone(),
        arguments.budgets.clone(),
        state_dir,
        None,
        None,
        worker::ViewOptions::default(),
    ) {
        exit(2);
    }
    let loaded = worker::list_models();
    let mut served_models = Vec::new();
    for ((model, settings), family) in arguments.models.iter().zip(&served).zip(&families) {
        let (ctx, id) = (settings.options.ctx as usize, settings.id.clone());
        let format = match family {
            Family::DeepSeek => Format::DeepSeek(deepseek::Defaults {
                model_id: id.clone(),
                sampling: worker::Sampling {
                    temperature: model.temperature.unwrap_or(1.0),
                    top_p: model.top_p.unwrap_or(1.0),
                    top_k: model.top_k.unwrap_or(0),
                    seed: model.seed,
                },
                is_thinking_on: model.think != Some(0),
                effort: model
                    .think
                    .filter(|&effort| effort > 0)
                    .unwrap_or(deepseek::EFFORT_HIGH),
            }),
        };
        let name = match family {
            Family::DeepSeek => "DeepSeek V4.1 Flash (eremite)".to_string(),
        };
        worker::log_model_line(
            served_models.len(),
            &format!(
                "serving {id} at http://{host}:{port}/v1 (context {ctx} tokens{}{})",
                if settings
                    .options
                    .deepseek()
                    .is_some_and(|deepseek| deepseek.experts.is_some())
                {
                    ", GGUF experts"
                } else {
                    ""
                },
                if settings.options.draft > 0 {
                    ", drafts"
                } else {
                    ""
                }
            ),
        );
        served_models.push(Served {
            id,
            name,
            ctx,
            tokenizer: loaded[served_models.len()].1,
            format,
        });
    }
    if !host.is_loopback() {
        engine::log(&format!(
            "warning: plain HTTP on {host}: keys and conversations cross the network unencrypted"
        ));
    }
    let shared = Arc::new(Shared {
        key: Sha256::digest(key.as_bytes()).into(),
        started: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs()),
        arrivals: AtomicU64::new(0),
        models: served_models,
        search: {
            let dir = arguments
                .server
                .searxng_dir
                .as_deref()
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .filter(|home| !home.is_empty())
                        .map(|home| PathBuf::from(home).join(".local/share/searxng"))
                });
            let dir = dir.unwrap_or_else(|| PathBuf::from(".local/share/searxng"));
            let socket = arguments
                .server
                .searxng_socket
                .as_deref()
                .map_or_else(|| dir.join("eremite.sock"), PathBuf::from);
            codex::search::Service::new(
                dir,
                socket,
                codex::search::AddressPolicy::Public,
                codex::search::Times {
                    timeout: Duration::from_secs_f64(arguments.server.search_timeout_s),
                    searxng_idle: Duration::from_secs_f64(arguments.server.searxng_idle_s),
                    session_idle: Duration::from_secs_f64(arguments.server.search_session_idle_s),
                },
            )
        },
        limits: http::Limits {
            timeout: Duration::from_secs_f64(arguments.server.http_timeout_s),
            ..http::DEFAULT_LIMITS
        },
        keepalive: Duration::from_secs_f64(arguments.server.keepalive_s),
    });
    let (stopper, limits) = (shared.clone(), shared.limits);
    let app = Router::new()
        .route("/health", get(report_health))
        .route("/v1/health", get(report_health))
        .route(
            "/v1/control",
            get(report_control_status).fallback(refuse_other_than_get),
        )
        .route(
            "/v1/control/yield",
            post(request_yield).fallback(refuse_other_than_post),
        )
        .route(
            "/v1/control/resume",
            post(request_resume).fallback(refuse_other_than_post),
        )
        .route(
            "/v1/models",
            get(list_served_models).fallback(refuse_other_than_get),
        )
        .route(
            "/v1/codex/models",
            get(list_codex_models).fallback(refuse_other_than_get),
        )
        .route(
            "/v1/alpha/search",
            post(answer_search).fallback(refuse_other_than_post),
        )
        .route(
            "/alpha/search",
            post(answer_search).fallback(refuse_other_than_post),
        )
        .route(
            "/codex/models",
            get(list_codex_models).fallback(refuse_other_than_get),
        )
        .route(
            "/models",
            get(list_served_models).fallback(refuse_other_than_get),
        )
        .route(
            "/v1/models/{id}",
            get(report_model_status).fallback(refuse_other_than_get),
        )
        .route(
            "/models/{id}",
            get(report_model_status).fallback(refuse_other_than_get),
        )
        .route(
            "/v1/chat/completions",
            post(answer_chat_completions).fallback(refuse_other_than_post),
        )
        .route(
            "/chat/completions",
            post(answer_chat_completions).fallback(refuse_other_than_post),
        )
        .route(
            "/v1/responses",
            post(answer_responses).fallback(refuse_other_than_post),
        )
        .route(
            "/responses",
            post(answer_responses).fallback(refuse_other_than_post),
        )
        .fallback(refuse_unknown_endpoint)
        .layer(middleware::from_fn_with_state(shared.clone(), authorize))
        .with_state(shared);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|error| die(&format!("cannot start the runtime: {error}")));
    let idle = stopper.clone();
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::from_std(listener)
            .unwrap_or_else(|error| die(&error.to_string()));
        // The engine owns the stop on SIGINT or SIGTERM. The server then stops
        // accepting, and the engine ends every open response.
        let stopped = worker::wait_for_stop();
        // SearXNG, if the web tool started it, stops when idle.
        tokio::spawn(async move { idle.search.stop_searxng_when_idle().await });
        http::serve(listener, app, stopped, limits).await;
    });
    stopper.search.stop_searxng();
    worker::stop();
}
