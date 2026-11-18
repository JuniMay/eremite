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
//! - With the setting tui, the terminal UI (ui/) shows the server's status, its
//!   requests, its memory and its log on stderr's terminal instead of the log.
//!
//! The configuration file (config.rs) holds every setting. The command line
//! names only that file.

use eremite_server::{chat, codex, config, control, engine, http, ui, worker};

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use eremite_server::chat::{Api, Family, deepseek};
use eremite_server::decision::d1;

/// The fewest characters that the API key has.
/// The reason for this value is unknown.
const MIN_KEY_CHARS: usize = 16;
/// The most characters that the API key has.
/// The reason for this value is unknown.
const MAX_KEY_CHARS: usize = 256;

/// eremite-server serves the models that a configuration file lists, over the
/// OpenAI and Codex APIs. The file holds every setting.
#[derive(Parser)]
struct Arguments {
    /// The configuration file in JSON: the server's settings and the models it
    /// serves [default: ~/.config/eremite/server.json]
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
}

/// Prints `message` and ends the process with status 1.
fn die(message: &str) -> ! {
    eprintln!("eremite-server: {message}");
    exit(1)
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
    /// d1's prompt (decision/d1.rs). A d1 model is a decision model: it
    /// answers questions and generates no text.
    D1,
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

/// Builds `model`'s object as GET /v1/models lists it: OpenAI's fields, the
/// context's length as servers commonly add it, and for a model that generates
/// text, the most tokens of its output.
fn build_model_object(shared: &Shared, model: &Served) -> Value {
    let mut object = json!({
        "id": model.id, "object": "model", "created": shared.started, "owned_by": "eremite",
        "name": model.name, "context_length": model.ctx, "max_model_len": model.ctx,
    });
    if let Format::DeepSeek(_) = model.format {
        object["max_output_tokens"] = (model.ctx as i64).min(chat::DEFAULT_MAX_TOKENS).into();
    }
    object
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
    let models = shared
        .models
        .iter()
        .filter_map(|model| match &model.format {
            Format::DeepSeek(defaults) => Some(codex::CatalogModel {
                model_id: &model.id,
                display_name: &model.name,
                context_window: model.ctx,
                default_effort: defaults.is_thinking_on.then_some(defaults.effort),
            }),
            Format::D1 => None,
        });
    build_json_response(codex::build_catalog(models).to_string())
}

/// Answers POST /v1/alpha/search: runs the commands of Codex's web tool
/// (codex/search.rs). Each command's result, or why it failed, is text in
/// `output`. A successful query also returns its sources in `results`.
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
    let answer = shared.search.answer_call(request).await;
    build_json_response(serde_json::to_string(&answer).expect("serialize the search answer"))
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
        Format::D1 => Err(chat::Error::new_bad_request(format!(
            "{} is a decision model: it answers questions and generates no text",
            handlers.models[index].id
        ))),
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
    let started = Instant::now();
    let path = Arguments::parse()
        .config
        .unwrap_or_else(|| config::default_path().unwrap_or_else(|error| die(&error)));
    let configuration = config::read(&path).unwrap_or_else(|error| die(&error));
    let Some(port) = configuration.server.port else {
        die(&format!(
            "{}: no port; port sets the port to serve on",
            path.display()
        ))
    };
    let key = match &configuration.server.api_key_file {
        Some(file) => {
            std::fs::read_to_string(file).unwrap_or_else(|_| die(&format!("cannot open {file}")))
        }
        None => std::env::var("EREMITE_API_KEY").unwrap_or_default(),
    };
    let key = key.trim_end_matches(['\n', '\r', ' ']);
    if !(MIN_KEY_CHARS..=MAX_KEY_CHARS).contains(&key.len()) {
        die(&format!(
            "the API key (EREMITE_API_KEY or api_key_file) must have {MIN_KEY_CHARS} to {MAX_KEY_CHARS} characters"
        ));
    }
    let host: Ipv4Addr = if configuration.server.host == "localhost" {
        Ipv4Addr::LOCALHOST
    } else {
        configuration.server.host.parse().unwrap_or_else(|_| {
            die(&format!(
                "host takes an IPv4 address: {}",
                configuration.server.host
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
    for model in &configuration.models {
        let family = Family::of(&model.engine.model).unwrap_or_else(|error| die(&error));
        let mut model = model.clone();
        if family == Family::D1 {
            let context = d1::read_context(&model.engine.model).unwrap_or_else(|error| die(&error));
            model
                .fit_decision_model(context)
                .unwrap_or_else(|error| die(&error));
        }
        let mut options = model.engine.clone();
        options
            .configure(family)
            .unwrap_or_else(|error| die(&error));
        let id = model.id.clone().unwrap_or_else(|| match family {
            Family::DeepSeek => deepseek::DEFAULT_MODEL_ID.into(),
            Family::D1 => d1::derive_model_names(&model.engine.model).0,
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
            idle_unload: model
                .idle_unload
                .unwrap_or(configuration.budgets.idle_unload_s),
            max_cached_states: model.max_cached_states,
        });
        families.push(family);
    }
    // The folder of the saved states. There is none without a folder, or with
    // state_disk 0.
    let state_dir = (configuration.budgets.state_disk_gb > 0.0)
        .then(|| {
            configuration
                .server
                .state_dir
                .as_ref()
                .map(PathBuf::from)
                .or_else(config::default_state_dir)
        })
        .flatten();
    // The terminal UI takes the terminal before the engine starts to log.
    let settings = &configuration.server;
    let opened = if settings.tui {
        let wait = Duration::from_secs_f64(settings.tui_background_answer_wait_s);
        let opened = ui::terminal::enter(settings.tui_theme, wait);
        if opened.is_none() {
            engine::log("tui: stderr is not a terminal that can show the terminal UI");
        }
        opened
    } else {
        None
    };
    let (on_log, on_die, view_options) = match opened {
        Some(_) => (
            Some(ui::log::ENGINE_LOG_HOOK),
            Some(ui::terminal::leave_before_fatal_error as extern "C" fn()),
            worker::ViewOptions {
                kept_route_tokens: settings.tui_expert_map_tokens,
                on_model_log: Some(ui::log::keep_model_line),
            },
        ),
        None => (None, None, worker::ViewOptions::default()),
    };
    if !worker::start(
        served.clone(),
        configuration.budgets.clone(),
        state_dir,
        on_log,
        on_die,
        view_options,
    ) {
        // The exit gives the terminal back first, as enter() arranged.
        exit(2);
    }
    let loaded = worker::list_models();
    let mut served_models = Vec::new();
    for (settings, family) in served.iter().zip(&families) {
        let (ctx, id) = (settings.options.ctx as usize, settings.id.clone());
        let format = match family {
            Family::DeepSeek => Format::DeepSeek(deepseek::Defaults::new(id.clone())),
            Family::D1 => Format::D1,
        };
        let name = match family {
            Family::DeepSeek => "DeepSeek V4.1 Flash (eremite)".to_string(),
            Family::D1 => d1::derive_model_names(&settings.options.model).1,
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
    let view = opened.map(|opened| {
        let models = served_models
            .iter()
            .zip(&families)
            .map(|(model, family)| ui::view::ViewModel {
                info: ui::frame::ModelInfo {
                    name: model.id.clone(),
                    kind: match family {
                        Family::DeepSeek => ui::frame::Kind::Llm,
                        Family::D1 => ui::frame::Kind::Decision,
                    },
                },
                tokenizer: model.tokenizer,
                find_role: match family {
                    Family::DeepSeek => deepseek::find_role_of_token,
                    // A decision model keeps no sessions whose text the view
                    // shows.
                    Family::D1 => |_| None,
                },
            })
            .collect();
        let windows = ui::samples::Windows {
            now: settings.tui_now_window_s,
            recent: settings.tui_recent_window_s,
            history: settings.tui_history_window_s,
        };
        ui::view::start_view(opened, models, format!("{host}:{port}"), started, windows)
    });
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
            let dir = configuration
                .server
                .searxng_dir
                .as_deref()
                .map(PathBuf::from)
                .or_else(config::default_searxng_dir);
            // Without HOME, the folder is relative to the current one.
            let dir = dir.unwrap_or_else(|| PathBuf::from(".local/share/eremite/searxng"));
            let socket = configuration
                .server
                .searxng_socket
                .as_deref()
                .map_or_else(|| dir.join("eremite.sock"), PathBuf::from);
            codex::search::Service::new(
                dir,
                socket,
                codex::search::AddressPolicy::Public,
                codex::search::Times {
                    timeout: Duration::from_secs_f64(configuration.server.search_timeout_s),
                    searxng_idle: Duration::from_secs_f64(configuration.server.searxng_idle_s),
                    session_idle: Duration::from_secs_f64(
                        configuration.server.search_session_idle_s,
                    ),
                },
            )
        },
        limits: http::Limits {
            timeout: Duration::from_secs_f64(configuration.server.http_timeout_s),
            ..http::DEFAULT_LIMITS
        },
        keepalive: Duration::from_secs_f64(configuration.server.keepalive_s),
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
    if let Some(view) = view {
        view.finish();
    }
}
