//! The server's settings.
//!
//! `-m DIR` serves one model, with its options on the command line. `--config
//! FILE` serves the models that a JSON file lists. Each model there has its own
//! options. They take the command line's names without the dashes, with `_` for
//! `-`. The file's root holds the server-wide settings.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use crate::engine;

/// The most tokens that a context holds. A model also takes at most its own
/// context.
const MAX_CONTEXT_TOKENS: i32 = 1 << 20;
/// The fewest decoder tokens that a bounded replay (ced) computes again.
/// The reason for this value is unknown.
const MIN_CED_TOKENS: i32 = 128;
/// The most GiB that a memory budget takes. The reason for this value is
/// unknown.
const MAX_MEMORY_GIB: f64 = 1024.0;
/// The fewest GiB that the reserve takes. The reason for this value is unknown.
const MIN_RESERVE_GIB: f64 = 2.0;
/// The most GiB that saved states may take on the SSD. The reason for this
/// value is unknown.
const MAX_STATE_DISK_GIB: f64 = 65536.0;
/// The most requests or states that a count setting takes. The reason for this
/// value is unknown.
const MAX_COUNT: f64 = 65536.0;
/// The most seconds that a time setting takes. The reason for this value is
/// unknown.
const MAX_SECONDS: f64 = 1e6;

/// One model's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// The name that requests give in `model`; None: the family's default name.
    pub id: Option<String>,
    /// The engine's options for the model.
    pub engine: engine::Options,
    /// The most rows that a decoding call takes, at most the model's own; None:
    /// the model's own. DeepSeek's own is 6.
    pub rows_per_call: Option<usize>,
    /// The temperature of requests that set none; None: 1.
    pub temperature: Option<f64>,
    /// The top_p of requests that set none; None: 1.
    pub top_p: Option<f64>,
    /// The top_k of requests that set none; None: 0, every token.
    pub top_k: Option<u32>,
    /// The seed of requests that set none; None: a random seed for each
    /// request.
    pub seed: Option<u64>,
    /// The thinking effort of requests that set none, from 1 to 100, or 0 for
    /// thinking off; None: the format's default.
    pub think: Option<u32>,
    /// The seconds without requests before the model's weights and experts
    /// unload, 0 for never; None: the server-wide setting.
    pub idle_unload: Option<f64>,
}

impl ModelConfig {
    /// Creates the settings of the model in the directory `model`. The context
    /// holds the most tokens. No other option is set.
    pub fn new(model: &str) -> ModelConfig {
        ModelConfig {
            id: None,
            engine: engine::Options {
                model: model.into(),
                ctx: MAX_CONTEXT_TOKENS,
                ..Default::default()
            },
            rows_per_call: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            think: None,
            idle_unload: None,
        }
    }

    /// Sets the option `name` to `value`. The name is the command line's,
    /// without the dashes and with `_` for `-`. Returns the option's message
    /// when the name or the value is invalid.
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let options = &mut self.engine;
        let number = || value.as_f64().ok_or(format!("{name} takes a number"));
        let integer = || {
            value
                .as_i64()
                .filter(|integer| i32::try_from(*integer).is_ok())
                .map(|integer| integer as i32)
                .ok_or(format!("{name} takes an integer"))
        };
        let switch = || value.as_bool().ok_or(format!("{name} takes true or false"));
        let text = || {
            value
                .as_str()
                .filter(|text| !text.is_empty())
                .map(String::from)
                .ok_or(format!("{name} takes a nonempty string"))
        };
        match name {
            "model" => options.model = text()?,
            "id" => self.id = Some(text()?),
            "experts" => options.deepseek_mut()?.experts = Some(text()?),
            "ctx" => options.ctx = integer()?,
            "ced" => options.deepseek_mut()?.ced = match value {
                Value::String(text) if text == "off" => 0,
                _ => integer()
                    .ok()
                    .filter(|tokens| (MIN_CED_TOKENS..=MAX_CONTEXT_TOKENS).contains(tokens))
                    .ok_or(format!(
                        "ced takes off or {MIN_CED_TOKENS} to {MAX_CONTEXT_TOKENS} replay tokens"
                    ))?,
            },
            "prefill_prune" => options.deepseek_mut()?.prefill_prune = switch()?,
            "packed_state_rows" => options.deepseek_mut()?.packed_state_rows = switch()?,
            "memory"
            | "reserve"
            | "gpu_timeout"
            | "progress_log"
            | "state_cache"
            | "state_disk"
            | "state_disk_tokens"
            | "max_waiting_requests"
            | "max_cached_states"
            | "snapshot_interval"
            | "state_dir"
            | "port"
            | "host"
            | "api_key_file"
            | "http_timeout"
            | "keepalive" => {
                return Err(format!("{name} is server-wide; move it outside models"));
            }
            "rows_per_call" => {
                self.rows_per_call = Some(
                    integer()
                        .ok()
                        .filter(|rows| *rows >= 1)
                        .ok_or("rows_per_call takes a positive integer")?
                        as usize,
                )
            }
            "temperature" => {
                self.temperature = Some(
                    number()
                        .ok()
                        .filter(|temperature| temperature.is_finite() && *temperature >= 0.0)
                        .ok_or("temperature must be finite and nonnegative")?,
                )
            }
            "top_p" => {
                self.top_p = Some(
                    number()
                        .ok()
                        .filter(|top_p| (0.0..=1.0).contains(top_p))
                        .ok_or("top_p must be in [0, 1]")?,
                )
            }
            "top_k" => {
                self.top_k = Some(
                    value
                        .as_u64()
                        .and_then(|top_k| u32::try_from(top_k).ok())
                        .ok_or("top_k takes a nonnegative integer")?,
                )
            }
            "seed" => self.seed = Some(value.as_u64().ok_or("seed takes a nonnegative integer")?),
            "think" => {
                self.think = Some(
                    value
                        .as_u64()
                        .filter(|effort| *effort <= 100)
                        .ok_or("think takes 0 (off) or 1 to 100")? as u32,
                )
            }
            "idle_unload" => self.idle_unload = Some(read_idle_seconds(value)?),
            _ => return Err(format!("no option {name}")),
        }
        Ok(())
    }
}

/// The server-wide settings besides the budgets.
#[derive(Clone, Debug, PartialEq)]
pub struct Server {
    /// The port that the server listens on. Without one, the server shows its
    /// usage and exits.
    pub port: Option<u16>,
    /// The IPv4 address that the server listens on, or localhost.
    pub host: String,
    /// The file that holds the API key; None: the key is in EREMITE_API_KEY.
    pub api_key_file: Option<String>,
    /// The folder of the states saved on the SSD (disk.rs); None:
    /// [`default_state_dir`].
    pub state_dir: Option<String>,
    /// The seconds that a request's head, each part of its body and a stalled
    /// write may take (http.rs).
    pub http_timeout_s: f64,
    /// The seconds between the comment lines that a streamed response sends
    /// while it waits for its first event.
    pub keepalive_s: f64,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            port: None,
            host: "127.0.0.1".into(),
            api_key_file: None,
            state_dir: None,
            http_timeout_s: 60.0,
            keepalive_s: 15.0,
        }
    }
}

/// Returns $HOME/.cache/eremite/states: the folder of the saved states unless
/// state_dir sets another one. Returns None when HOME is unset.
pub fn default_state_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".cache/eremite/states"))
}

impl Server {
    /// Sets the server setting `name` to `value`. Returns the setting's message
    /// when either is invalid.
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let text = || {
            value
                .as_str()
                .filter(|text| !text.is_empty())
                .ok_or(format!("{name} takes nonempty text"))
        };
        match name {
            "port" => {
                self.port = Some(
                    value
                        .as_u64()
                        .and_then(|port| u16::try_from(port).ok())
                        .ok_or("port takes an integer from 0 to 65535")?,
                )
            }
            "host" => {
                let host = text()?;
                if host != "localhost" && host.parse::<std::net::Ipv4Addr>().is_err() {
                    return Err("host takes an IPv4 address or localhost".into());
                }
                self.host = host.into();
            }
            "api_key_file" => self.api_key_file = Some(text()?.into()),
            "state_dir" => self.state_dir = Some(text()?.into()),
            "http_timeout" | "keepalive" => {
                let seconds = value
                    .as_f64()
                    .filter(|seconds| *seconds > 0.0 && *seconds <= MAX_SECONDS)
                    .ok_or(format!(
                        "{name} takes more than 0 and up to {MAX_SECONDS} seconds"
                    ))?;
                *match name {
                    "http_timeout" => &mut self.http_timeout_s,
                    _ => &mut self.keepalive_s,
                } = seconds;
            }
            _ => return Err(format!("no server setting {name}")),
        }
        Ok(())
    }
}

/// Reads the seconds without requests before a model unloads: 0 for never, up
/// to MAX_SECONDS.
fn read_idle_seconds(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|seconds| (0.0..=MAX_SECONDS).contains(seconds))
        .ok_or(format!(
            "idle_unload takes 0 (never) to {MAX_SECONDS} seconds"
        ))
}

/// The server-wide budgets and limits. The engine's memory policy keeps its
/// defaults. Settings that can slow the server stay out of the configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Budgets {
    /// The GiB of model memory at most, for all models together, state memory
    /// included; 0: no limit.
    pub memory_gb: f64,
    /// The GiB of memory kept available to other programs.
    pub reserve_gb: f64,
    /// A wait for the GPU that lasts this many seconds ends the server. Only a
    /// failed command buffer or a stopped GPU takes that long.
    pub gpu_timeout_s: f64,
    /// The seconds between a decoding request's progress lines in the log; 0:
    /// none.
    pub progress_log_s: f64,
    /// The GiB of states and snapshots that the state cache keeps, for all
    /// models together.
    pub state_cache_gb: f64,
    /// The seconds without requests before a model unloads; 0: never. A model's
    /// own idle_unload setting overrides it.
    pub idle_unload_s: f64,
    /// The GiB that saved states may take on the SSD; 0: none are saved or
    /// read.
    pub state_disk_gb: f64,
    /// A state is saved when it holds at least this many tokens that no other
    /// kept state holds.
    pub state_disk_tokens: usize,
    /// The most requests that wait for one model. Another request then gets
    /// HTTP 429, too many requests.
    pub max_waiting_requests: usize,
    /// The most states that the state cache keeps, for all models together.
    pub max_cached_states: usize,
    /// For a model that resumes only from snapshots, a prompt prefilled in
    /// several calls takes a snapshot at every multiple of this many positions.
    pub snapshot_interval: usize,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            memory_gb: 0.0,
            reserve_gb: 10.0,
            gpu_timeout_s: 60.0,
            progress_log_s: 10.0,
            state_cache_gb: 8.0,
            idle_unload_s: 600.0,
            state_disk_gb: 32.0,
            state_disk_tokens: 8192,
            max_waiting_requests: 16,
            max_cached_states: 256,
            snapshot_interval: 2048,
        }
    }
}

impl Budgets {
    /// Sets the budget `name` to `value`. Returns the budget's message when
    /// either is invalid.
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        if name == "idle_unload" {
            self.idle_unload_s = read_idle_seconds(value)?;
            return Ok(());
        }
        let number = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or(format!("{name} takes a finite number"))?;
        let is_count = |most: f64| (1.0..=most).contains(&number) && number.fract() == 0.0;
        match name {
            "memory" if (0.0..=MAX_MEMORY_GIB).contains(&number) => self.memory_gb = number,
            "memory" => {
                return Err(format!(
                    "memory takes 0 (automatic) or a positive GiB budget up to {MAX_MEMORY_GIB}"
                ));
            }
            "reserve" if (MIN_RESERVE_GIB..=MAX_MEMORY_GIB).contains(&number) => {
                self.reserve_gb = number
            }
            "reserve" => {
                return Err(format!(
                    "reserve takes {MIN_RESERVE_GIB} to {MAX_MEMORY_GIB} GiB"
                ));
            }
            "gpu_timeout" if number > 0.0 => self.gpu_timeout_s = number,
            "gpu_timeout" => return Err("gpu_timeout takes a positive number of seconds".into()),
            "progress_log" if (0.0..=MAX_SECONDS).contains(&number) => self.progress_log_s = number,
            "progress_log" => {
                return Err(format!(
                    "progress_log takes 0 (none) to {MAX_SECONDS} seconds"
                ));
            }
            "state_cache" if number > 0.0 && number <= MAX_MEMORY_GIB => {
                self.state_cache_gb = number
            }
            "state_cache" => {
                return Err(format!(
                    "state_cache takes a positive budget up to {MAX_MEMORY_GIB} GiB"
                ));
            }
            "state_disk" if (0.0..=MAX_STATE_DISK_GIB).contains(&number) => {
                self.state_disk_gb = number
            }
            "state_disk" => {
                return Err(format!(
                    "state_disk takes 0 (off) or a GiB budget up to {MAX_STATE_DISK_GIB}"
                ));
            }
            "state_disk_tokens" if is_count(MAX_CONTEXT_TOKENS as f64) => {
                self.state_disk_tokens = number as usize
            }
            "state_disk_tokens" => {
                return Err(format!(
                    "state_disk_tokens takes 1 to {MAX_CONTEXT_TOKENS} tokens"
                ));
            }
            "max_waiting_requests" if is_count(MAX_COUNT) => {
                self.max_waiting_requests = number as usize
            }
            "max_waiting_requests" => {
                return Err(format!(
                    "max_waiting_requests takes 1 to {MAX_COUNT} requests"
                ));
            }
            "max_cached_states" if is_count(MAX_COUNT) => self.max_cached_states = number as usize,
            "max_cached_states" => {
                return Err(format!("max_cached_states takes 1 to {MAX_COUNT} states"));
            }
            "snapshot_interval" if is_count(MAX_CONTEXT_TOKENS as f64) => {
                self.snapshot_interval = number as usize
            }
            "snapshot_interval" => {
                return Err(format!(
                    "snapshot_interval takes 1 to {MAX_CONTEXT_TOKENS} positions"
                ));
            }
            _ => return Err(format!("no server budget {name}")),
        }
        Ok(())
    }

    /// Returns the engine's process settings that these budgets give.
    pub fn process_settings(&self) -> engine::ProcessSettings {
        engine::ProcessSettings {
            memory_gb: self.memory_gb,
            reserve_gb: self.reserve_gb,
            gpu_timeout_seconds: self.gpu_timeout_s,
        }
    }
}

/// The server's settings and the models it serves.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The server-wide settings besides the budgets.
    pub server: Server,
    /// The server-wide budgets and limits.
    pub budgets: Budgets,
    /// The models, in the file's order.
    pub models: Vec<ModelConfig>,
}

/// Returns $HOME/.config/eremite/server.json: the configuration file that the
/// server reads when its arguments give neither --config nor -m. Returns an
/// error when HOME is unset.
pub fn default_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".config/eremite/server.json"))
        .ok_or("HOME is unset; specify --config FILE or -m DIR".into())
}

/// Reads the server settings and the models they serve from the JSON file at
/// `path`.
pub fn read(path: &Path) -> Result<Config, String> {
    let name = path.display();
    let text = std::fs::read(path).map_err(|error| format!("cannot read {name}: {error}"))?;
    let root: Value = serde_json::from_slice(&text).map_err(|error| format!("{name}: {error}"))?;
    parse(&root).map_err(|error| format!("{name}: {error}"))
}

/// Parses the server settings and the models they serve from the JSON object
/// `root`. Returns an error for an unknown setting, an invalid value or a
/// missing model list.
pub fn parse(root: &Value) -> Result<Config, String> {
    let Some(object) = root.as_object() else {
        return Err("not a JSON object".into());
    };
    if let Some(key) = object.keys().find(|key| {
        ![
            "models",
            "memory",
            "reserve",
            "gpu_timeout",
            "progress_log",
            "state_cache",
            "state_disk",
            "state_disk_tokens",
            "max_waiting_requests",
            "max_cached_states",
            "snapshot_interval",
            "state_dir",
            "idle_unload",
            "port",
            "host",
            "api_key_file",
            "http_timeout",
            "keepalive",
        ]
        .contains(&key.as_str())
    }) {
        return Err(format!("no server setting {key}"));
    }
    let mut budgets = Budgets::default();
    for name in [
        "memory",
        "reserve",
        "gpu_timeout",
        "progress_log",
        "state_cache",
        "state_disk",
        "state_disk_tokens",
        "max_waiting_requests",
        "max_cached_states",
        "snapshot_interval",
        "idle_unload",
    ] {
        if let Some(value) = object.get(name) {
            budgets.set(name, value)?;
        }
    }
    let mut server = Server::default();
    for name in [
        "port",
        "host",
        "api_key_file",
        "state_dir",
        "http_timeout",
        "keepalive",
    ] {
        if let Some(value) = object.get(name) {
            server.set(name, value)?;
        }
    }
    let list = object
        .get("models")
        .and_then(Value::as_array)
        .filter(|list| !list.is_empty())
        .ok_or("models must be a nonempty list")?;
    let mut models = Vec::new();
    for (i, entry) in list.iter().enumerate() {
        let entry: &Map<String, Value> = entry
            .as_object()
            .ok_or(format!("models[{i}] is not an object"))?;
        let dir = entry
            .get("model")
            .and_then(Value::as_str)
            .ok_or(format!("models[{i}] has no model directory"))?;
        let mut model = ModelConfig::new(dir);
        for (name, value) in entry {
            model
                .set(name, value)
                .map_err(|error| format!("models[{i}]: {error}"))?;
        }
        models.push(model);
    }
    Ok(Config {
        server,
        budgets,
        models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_budget_takes_values_in_its_range_only() {
        let mut budgets = Budgets::default();
        assert_eq!(
            (budgets.memory_gb, budgets.reserve_gb, budgets.gpu_timeout_s),
            (0.0, 10.0, 60.0)
        );
        budgets.set("memory", &json!(64)).unwrap();
        budgets.set("reserve", &json!(12)).unwrap();
        budgets.set("gpu_timeout", &json!(120)).unwrap();
        budgets.set("max_waiting_requests", &json!(4)).unwrap();
        assert_eq!(budgets.max_waiting_requests, 4);
        let settings = budgets.process_settings();
        assert_eq!(
            (
                settings.memory_gb,
                settings.reserve_gb,
                settings.gpu_timeout_seconds
            ),
            (64.0, 12.0, 120.0)
        );
        for (name, value) in [
            ("memory", json!(-1)),
            ("memory", json!(2048)),
            ("reserve", json!(1)),
            ("gpu_timeout", json!(0)),
            ("gpu_timeout", json!("long")),
            ("max_waiting_requests", json!(0)),
            ("max_waiting_requests", json!(1.5)),
            ("cache", json!(8)),
        ] {
            assert!(budgets.set(name, &value).is_err(), "{name} {value}");
        }
    }

    #[test]
    fn idle_unload_is_server_wide_and_a_model_can_set_its_own() {
        let config = parse(&json!({"idle_unload": 120, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/second", "idle_unload": 0}
        ]}))
        .unwrap();
        assert_eq!(config.budgets.idle_unload_s, 120.0);
        assert_eq!(
            (config.models[0].idle_unload, config.models[1].idle_unload),
            (None, Some(0.0))
        );
        // Ten minutes when nothing sets it.
        let config = parse(&json!({"models": [{"model": "/models/deepseek"}]})).unwrap();
        assert_eq!(config.budgets.idle_unload_s, 600.0);
        for bad in [json!(-1), json!(2e6), json!("soon")] {
            assert!(parse(&json!({"idle_unload": bad, "models": [{"model": "/m"}]})).is_err());
            assert!(parse(&json!({"models": [{"model": "/m", "idle_unload": bad}]})).is_err());
        }
    }

    #[test]
    fn model_memory_is_global_and_separate_from_states() {
        let root = json!({"memory": 64, "reserve": 10, "state_cache": 12, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/second"}
        ]});
        assert!(
            parse(&root).is_ok(),
            "one model-memory budget must serve every architecture"
        );
        for key in ["cache", "memory"] {
            let nested = json!({"models": [{"model": "/models/any-model", key: 32}]});
            assert!(
                parse(&nested).is_err(),
                "{key} must not create a per-model memory budget"
            );
        }
    }

    #[test]
    fn reserve_and_state_cache_are_server_wide() {
        let root = json!({"memory": 56, "reserve": 10, "state_cache": 12, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/second"}
        ]});
        assert!(
            parse(&root).is_ok(),
            "one reserve and one state budget must cover every model"
        );
    }

    #[test]
    fn model_entries_cannot_create_separate_memory_budgets() {
        for setting in [
            "memory",
            "reserve",
            "gpu_timeout",
            "progress_log",
            "state_cache",
            "http_timeout",
            "keepalive",
        ] {
            let mut model = json!({"model": "/models/second"});
            model[setting] = json!(8);
            assert!(
                parse(&json!({"models": [model]})).is_err(),
                "{setting} belongs at the server level"
            );
        }
    }

    #[test]
    fn server_settings_are_global_and_cli_values_can_override_them() {
        let mut config = parse(
            &json!({"port": 8008, "host": "localhost", "api_key_file": "key-file",
                    "models": [{"model": "/models/deepseek"}]}),
        )
        .unwrap();
        assert_eq!(
            (
                config.server.port,
                config.server.host.as_str(),
                config.server.api_key_file.as_deref()
            ),
            (Some(8008), "localhost", Some("key-file"))
        );
        config.server.set("port", &json!(9008)).unwrap();
        assert_eq!(config.server.port, Some(9008));
        assert_eq!(
            (config.server.http_timeout_s, config.server.keepalive_s),
            (60.0, 15.0)
        );
        config.server.set("http_timeout", &json!(120)).unwrap();
        config.server.set("keepalive", &json!(2.5)).unwrap();
        assert_eq!(
            (config.server.http_timeout_s, config.server.keepalive_s),
            (120.0, 2.5)
        );
        for (name, value) in [
            ("port", json!(-1)),
            ("port", json!(65536)),
            ("port", json!(1.5)),
            ("port", json!("8008")),
            ("host", json!("not-an-address")),
            ("api_key_file", json!("")),
            ("http_timeout", json!(0)),
            ("keepalive", json!(-1)),
            ("keepalive", json!("15")),
        ] {
            assert!(config.server.set(name, &value).is_err(), "{name} {value}");
        }
    }

    #[test]
    fn a_file_gives_each_model_its_own_options() {
        let config = parse(&json!({"memory": 72, "state_cache": 16, "models": [
            {"model": "/m/deepseek", "ced": 128},
            {"model": "/m/small", "id": "small", "ctx": 32768, "top_k": 20, "think": 0}
        ]}))
        .unwrap();
        let models = config.models;
        assert_eq!(models.len(), 2);
        let (deepseek, small) = (&models[0], &models[1]);
        assert_eq!(
            (
                deepseek.engine.model.as_str(),
                config.budgets.memory_gb,
                deepseek.engine.deepseek().unwrap().ced,
                config.budgets.state_cache_gb
            ),
            ("/m/deepseek", 72.0, 128, 16.0)
        );
        assert_eq!(deepseek.id.as_deref(), None);
        assert_eq!(
            (
                small.id.as_deref(),
                small.engine.ctx,
                small.top_k,
                small.think
            ),
            (Some("small"), 32768, Some(20), Some(0))
        );
        assert!(small.engine.deepseek().is_none());
    }

    #[test]
    fn rows_per_call_is_per_model_and_positive() {
        let models = parse(&json!({"models": [
            {"model": "/m/deepseek", "rows_per_call": 2}, {"model": "/m/small"}
        ]}))
        .unwrap()
        .models;
        assert_eq!(
            (models[0].rows_per_call, models[1].rows_per_call),
            (Some(2), None)
        );
        assert!(parse(&json!({"models": [{"model": "/m", "rows_per_call": 0}]})).is_err());
        assert!(parse(&json!({"models": [{"model": "/m", "rows_per_call": "all"}]})).is_err());
    }

    #[test]
    fn the_queue_and_state_limits_are_server_settings() {
        let defaults = Budgets::default();
        assert_eq!(
            (
                defaults.max_waiting_requests,
                defaults.max_cached_states,
                defaults.snapshot_interval
            ),
            (16, 256, 2048)
        );
        let budgets = parse(
            &json!({"models": [{"model": "/m"}], "max_waiting_requests": 8,
                              "max_cached_states": 64, "snapshot_interval": 512}),
        )
        .unwrap()
        .budgets;
        assert_eq!(
            (
                budgets.max_waiting_requests,
                budgets.max_cached_states,
                budgets.snapshot_interval
            ),
            (8, 64, 512)
        );
        for (name, value) in [
            ("max_waiting_requests", json!(0)),
            ("max_cached_states", json!(1.5)),
            ("snapshot_interval", json!(-1)),
        ] {
            let mut root = json!({"models": [{"model": "/m"}]});
            root[name] = value;
            assert!(parse(&root).is_err(), "{name}");
        }
    }

    #[test]
    fn packed_state_rows_is_a_deepseek_switch() {
        let models = parse(&json!({"models": [
            {"model": "/m/deepseek", "packed_state_rows": true}
        ]}))
        .unwrap()
        .models;
        assert!(models[0].engine.deepseek().unwrap().packed_state_rows);
        let models = parse(&json!({"models": [{"model": "/m/deepseek"}]}))
            .unwrap()
            .models;
        assert!(
            !models[0]
                .engine
                .deepseek()
                .is_some_and(|deepseek| deepseek.packed_state_rows)
        );
        assert!(parse(&json!({"models": [{"model": "/m", "packed_state_rows": 1}]})).is_err());
    }

    #[test]
    fn deepseek_options_stay_through_the_family_check() {
        let mut models = parse(&json!({"models": [
            {"model": "/models/deepseek", "ced": 128}, {"model": "/models/plain"}
        ]}))
        .unwrap()
        .models;
        models[0]
            .engine
            .configure(engine::Architecture::DeepSeek)
            .unwrap();
        assert_eq!(models[0].engine.deepseek().unwrap().ced, 128);
        assert!(models[1].engine.deepseek().is_none());
        models[1]
            .engine
            .configure(engine::Architecture::DeepSeek)
            .unwrap();
        assert_eq!(
            models[1].engine.deepseek(),
            Some(&engine::DeepSeekOptions::default())
        );
    }

    #[test]
    fn unknown_or_invalid_options_are_refused() {
        assert!(parse(&json!({"models": []})).is_err());
        assert!(parse(&json!({"models": [{"cache": 1}]})).is_err());
        assert!(
            parse(&json!({"models": [{"model": "/m", "cahce": 1}]}))
                .unwrap_err()
                .contains("cahce")
        );
        assert!(parse(&json!({"models": [{"model": "/m", "ced": 64}]})).is_err());
        assert!(parse(&json!({"models": [{"model": "/m", "top_p": 2}]})).is_err());
        assert!(parse(&json!({"prt": 8008, "models": [{"model": "/m"}]})).is_err());
        for timeout in [json!(0), json!(-1), json!("long")] {
            assert!(
                parse(&json!({"gpu_timeout": timeout, "models": [{"model": "/m"}]})).is_err(),
                "gpu_timeout {timeout} is refused"
            );
        }
        for period in [json!(-1), json!(2e6)] {
            assert!(
                parse(&json!({"progress_log": period, "models": [{"model": "/m"}]})).is_err(),
                "progress_log {period} is refused"
            );
        }
        assert_eq!(
            parse(&json!({"progress_log": 0, "models": [{"model": "/m"}]}))
                .unwrap()
                .budgets
                .progress_log_s,
            0.0,
            "progress_log 0 logs no progress lines"
        );
        let timeout = |root: Value| parse(&root).unwrap().budgets.gpu_timeout_s;
        assert_eq!(timeout(json!({"models": [{"model": "/m"}]})), 60.0);
        assert_eq!(
            timeout(json!({"gpu_timeout": 120, "models": [{"model": "/m"}]})),
            120.0
        );
        assert_eq!(
            parse(&json!({"models": [{"model": "/m", "ced": "off"}]}))
                .unwrap()
                .models[0]
                .engine
                .deepseek()
                .unwrap()
                .ced,
            0
        );
    }
}
