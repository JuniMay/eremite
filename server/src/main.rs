//! eremite-server: eremite's engine behind an HTTP server, for one model or
//! several in one process.
//!
//! - The engine (src/eremite.h, and engine.rs here) runs the models. It owns
//!   the stop on SIGINT or SIGTERM.
//! - The worker (worker.rs) runs every model's requests on the engine thread.

use eremite_server::{chat, config, engine, http, worker};

use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::process::exit;
use std::time::Duration;

use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::Value;

use eremite_server::chat::{Family, deepseek};
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
  --serve PORT    override the JSON port; serve at http://HOST:PORT/v1
  --host ADDR     the IPv4 address to serve on (default 127.0.0.1)
  --api-key-file FILE  the API key clients must send (16 to 256 characters)
  --http-timeout S   seconds a request's head, each part of its body and a stalled write may take
                  (default 60)
  --keepalive S   seconds between the comment lines of a streamed response that waits for its
                  first token (default 15)
  --memory GB     an upper limit of model memory for every model, state memory included
                  (default: none; the memory that other programs leave, less --reserve, decides)
  --reserve GB    memory kept available to other programs, 2-1024 GiB (default 10)
  --gpu-timeout S seconds a wait for the GPU may last before the server ends (default 60)
  --max-waiting-requests N  requests that may wait for one model; another gets HTTP 429
                  (default 16)
  --render FILE   print the prompt of a JSON test case of DeepSeek's encoding/tests, and exit
 a model's options:
  --id NAME       the name requests give in model (default: deepseek-v4.1-flash)
  -t T, --temperature T  the temperature of requests that set none (default 1)
  --top-p P       the top_p of requests that set none, 0 to 1 (default 1)
  --top-k K       the top_k of requests that set none, 0 for all (default 0)
  --seed N        the seed of requests that set none (default: random for each)
  --think N       thinking effort 1-100 of requests that set none; 0 turns thinking off
  --ctx N         context capacity in tokens, at most 1048576 and the model's own (default: the
                  most)
  DeepSeek only:
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
                    "--memory" | "--reserve" | "--gpu-timeout" | "--max-waiting-requests" => {
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
                    "--host" | "--api-key-file" => {
                        server.push((flag[2..].replace('-', "_"), Value::String(text.clone())))
                    }
                    "--http-timeout" | "--keepalive" => {
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
            "--serve",
            "8008",
        ]
        .map(String::from),
    );
    let model = &parsed.models[0];
    assert!(model.engine.deepseek().unwrap().prefill_prune);
    assert!(model.engine.deepseek().unwrap().packed_state_rows);
    assert_eq!((model.temperature, model.top_p), (None, None));
    assert_eq!(parsed.server.port, Some(8008));
    let parsed = parse_arguments(
        &[
            "-m", "/m/small", "-t", "0.6", "--top-k", "20", "--id", "small",
        ]
        .map(String::from),
    );
    let model = &parsed.models[0];
    assert_eq!(
        (model.temperature, model.top_k, model.id.as_deref()),
        (Some(0.6), Some(20), Some("small"))
    );
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

/// Answers a request for an unknown path with 404.
async fn refuse_unknown_endpoint() -> Response {
    chat::build_error_response(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "no such endpoint",
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
        served.push(worker::ModelSettings { id, options });
    }
    if !worker::start(served.clone(), arguments.budgets.clone(), None, None) {
        exit(2);
    }
    for settings in &served {
        engine::log(&format!(
            "serving {} at http://{host}:{port}/v1 (context {} tokens{})",
            settings.id,
            settings.options.ctx,
            if settings
                .options
                .deepseek()
                .is_some_and(|deepseek| deepseek.experts.is_some())
            {
                ", GGUF experts"
            } else {
                ""
            }
        ));
    }
    if !host.is_loopback() {
        engine::log(&format!(
            "warning: plain HTTP on {host}: keys and conversations cross the network unencrypted"
        ));
    }
    let limits = http::Limits {
        timeout: Duration::from_secs_f64(arguments.server.http_timeout_s),
        ..http::DEFAULT_LIMITS
    };
    let app = Router::new()
        .route("/health", get(report_health))
        .route("/v1/health", get(report_health))
        .fallback(refuse_unknown_endpoint);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|error| die(&format!("cannot start the runtime: {error}")));
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::from_std(listener)
            .unwrap_or_else(|error| die(&error.to_string()));
        // The engine owns the stop on SIGINT or SIGTERM. The server then stops
        // accepting, and the engine ends every open response.
        let stopped = worker::wait_for_stop();
        http::serve(listener, app, stopped, limits).await;
    });
    worker::stop();
}
