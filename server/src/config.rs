//! The server's settings, from a JSON configuration file. The file's root holds
//! the server-wide settings. Its models list holds the models that the server
//! serves, each with its own options.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use crate::engine;
use crate::ui::terminal::ThemeChoice;

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
/// The most requests, states or tokens that a count setting takes. The reason
/// for this value is unknown.
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
            idle_unload: None,
        }
    }

    /// Sets the option `name`, a key of the model's entry in the configuration
    /// file, to `value`. Returns the option's message when the name or the
    /// value is invalid.
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let options = &mut self.engine;
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
            "draft" => options.draft = integer()?,
            "lookup" => {
                options.lookup = match value {
                    Value::String(text) if text == "off" => 0,
                    _ => integer()?,
                }
            }
            "prefill_prune" => options.deepseek_mut()?.prefill_prune = Some(switch()?),
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
            | "keepalive"
            | "search_timeout"
            | "searxng_idle"
            | "search_session_idle" => {
                return Err(format!("{name} is server-wide; move it outside models"));
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
    /// The port that the server listens on. The server needs one.
    pub port: Option<u16>,
    /// The IPv4 address that the server listens on, or localhost.
    pub host: String,
    /// The file that holds the API key; None: the key is in EREMITE_API_KEY.
    pub api_key_file: Option<String>,
    /// The folder of the states saved on the SSD (disk.rs); None:
    /// [`default_state_dir`].
    pub state_dir: Option<String>,
    /// The SearXNG for Codex's web tool (codex/search.rs), as
    /// scripts/setup-searxng.sh sets it up; None: [`default_searxng_dir`].
    pub searxng_dir: Option<String>,
    /// The Unix socket that the server starts SearXNG on; None: eremite.sock in
    /// the SearXNG folder.
    pub searxng_socket: Option<String>,
    /// The seconds that a request's head, each part of its body and a stalled
    /// write may take (http.rs).
    pub http_timeout_s: f64,
    /// The seconds between the comment lines that a streamed response sends
    /// while it waits for its first event.
    pub keepalive_s: f64,
    /// The seconds that each fetch, query and health check of the web tool may
    /// take, and SearXNG's start (codex/search.rs).
    pub search_timeout_s: f64,
    /// The seconds without a search after which SearXNG stops.
    pub searxng_idle_s: f64,
    /// The seconds without a call after which the web tool forgets a session's
    /// references.
    pub search_session_idle_s: f64,
    /// Whether the terminal UI (ui/) shows on stderr's terminal instead of the
    /// log.
    pub tui: bool,
    /// How the terminal UI chooses its colors.
    pub tui_theme: ThemeChoice,
    /// The seconds that the terminal UI waits for the terminal's answer about
    /// its background color.
    pub tui_background_answer_wait_s: f64,
    /// The seconds that the terminal UI's "now" figures average.
    pub tui_now_window_s: f64,
    /// The seconds that the terminal UI's "recent" figures average.
    pub tui_recent_window_s: f64,
    /// The seconds that the terminal UI's history charts span, one sample a
    /// column.
    pub tui_history_window_s: f64,
    /// The latest tokens of a decoding request whose experts the terminal
    /// UI's expert map shows.
    pub tui_expert_map_tokens: usize,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            port: None,
            host: "127.0.0.1".into(),
            api_key_file: None,
            state_dir: None,
            searxng_dir: None,
            searxng_socket: None,
            http_timeout_s: 60.0,
            keepalive_s: 15.0,
            search_timeout_s: 30.0,
            searxng_idle_s: 600.0,
            search_session_idle_s: 7200.0,
            tui: false,
            tui_theme: ThemeChoice::Auto,
            tui_background_answer_wait_s: 1.0,
            tui_now_window_s: 5.0,
            tui_recent_window_s: 60.0,
            tui_history_window_s: 60.0,
            tui_expert_map_tokens: 12,
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

/// Returns $HOME/.local/share/eremite/searxng: the folder of eremite's own
/// SearXNG unless searxng_dir sets another one. scripts/setup-searxng.sh sets
/// it up there by default. Returns None when HOME is unset.
pub fn default_searxng_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".local/share/eremite/searxng"))
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
            "searxng_dir" => self.searxng_dir = Some(text()?.into()),
            "searxng_socket" => self.searxng_socket = Some(text()?.into()),
            "tui" => self.tui = value.as_bool().ok_or("tui takes true or false")?,
            "tui_theme" => {
                self.tui_theme = match value.as_str() {
                    Some("auto") => ThemeChoice::Auto,
                    Some("light") => ThemeChoice::Light,
                    Some("dark") => ThemeChoice::Dark,
                    _ => return Err("tui_theme takes auto, light or dark".into()),
                }
            }
            "tui_expert_map_tokens" => {
                self.tui_expert_map_tokens = value
                    .as_u64()
                    .filter(|tokens| (1..=MAX_COUNT as u64).contains(tokens))
                    .ok_or(format!(
                        "tui_expert_map_tokens takes 1 to {MAX_COUNT} tokens"
                    ))? as usize
            }
            "http_timeout"
            | "keepalive"
            | "search_timeout"
            | "searxng_idle"
            | "search_session_idle"
            | "tui_background_answer_wait"
            | "tui_now_window"
            | "tui_recent_window"
            | "tui_history_window" => {
                let seconds = value
                    .as_f64()
                    .filter(|seconds| *seconds > 0.0 && *seconds <= MAX_SECONDS)
                    .ok_or(format!(
                        "{name} takes more than 0 and up to {MAX_SECONDS} seconds"
                    ))?;
                *match name {
                    "http_timeout" => &mut self.http_timeout_s,
                    "keepalive" => &mut self.keepalive_s,
                    "search_timeout" => &mut self.search_timeout_s,
                    "searxng_idle" => &mut self.searxng_idle_s,
                    "search_session_idle" => &mut self.search_session_idle_s,
                    "tui_background_answer_wait" => &mut self.tui_background_answer_wait_s,
                    "tui_now_window" => &mut self.tui_now_window_s,
                    "tui_recent_window" => &mut self.tui_recent_window_s,
                    _ => &mut self.tui_history_window_s,
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
/// server reads without --config. Returns an error when HOME is unset.
pub fn default_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".config/eremite/server.json"))
        .ok_or("HOME is unset; give the configuration file with --config FILE".into())
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
            "searxng_dir",
            "searxng_socket",
            "http_timeout",
            "keepalive",
            "search_timeout",
            "searxng_idle",
            "search_session_idle",
            "tui",
            "tui_theme",
            "tui_background_answer_wait",
            "tui_now_window",
            "tui_recent_window",
            "tui_history_window",
            "tui_expert_map_tokens",
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
        "searxng_dir",
        "searxng_socket",
        "http_timeout",
        "keepalive",
        "search_timeout",
        "searxng_idle",
        "search_session_idle",
        "tui",
        "tui_theme",
        "tui_background_answer_wait",
        "tui_now_window",
        "tui_recent_window",
        "tui_history_window",
        "tui_expert_map_tokens",
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
        if let Some(deepseek) = model.engine.deepseek()
            && deepseek.prefill_prune == Some(true)
            && deepseek.ced != 0
        {
            return Err(format!("models[{i}]: prefill_prune needs ced off"));
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
    fn the_default_folders_are_eremite_s_own_under_home() {
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
        assert_eq!(
            default_state_dir(),
            Some(home.join(".cache/eremite/states"))
        );
        assert_eq!(
            default_searxng_dir(),
            Some(home.join(".local/share/eremite/searxng"))
        );
    }

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
            "search_timeout",
            "searxng_idle",
            "search_session_idle",
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
    fn server_settings_take_values_in_their_range_only() {
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
        assert_eq!(
            (
                config.server.search_timeout_s,
                config.server.searxng_idle_s,
                config.server.search_session_idle_s
            ),
            (30.0, 600.0, 7200.0)
        );
        config.server.set("searxng_idle", &json!(60)).unwrap();
        assert_eq!(config.server.searxng_idle_s, 60.0);
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
            ("search_timeout", json!(0)),
            ("search_session_idle", json!(2e6)),
            ("tui", json!("yes")),
            ("tui_theme", json!("blue")),
            ("tui_now_window", json!(0)),
            ("tui_expert_map_tokens", json!(0)),
        ] {
            assert!(config.server.set(name, &value).is_err(), "{name} {value}");
        }
        // the terminal UI's settings
        let mut config = parse(&json!({"tui": true, "tui_theme": "dark",
            "tui_history_window": 120, "models": [{"model": "/models/deepseek"}]}))
        .unwrap();
        assert_eq!(
            (
                config.server.tui,
                config.server.tui_theme,
                config.server.tui_history_window_s
            ),
            (true, ThemeChoice::Dark, 120.0)
        );
        assert_eq!(
            (
                config.server.tui_background_answer_wait_s,
                config.server.tui_now_window_s,
                config.server.tui_recent_window_s
            ),
            (1.0, 5.0, 60.0)
        );
        config.server.set("tui", &json!(false)).unwrap();
        config
            .server
            .set("tui_expert_map_tokens", &json!(20))
            .unwrap();
        assert_eq!(
            (config.server.tui, config.server.tui_expert_map_tokens),
            (false, 20)
        );
    }

    #[test]
    fn a_file_gives_each_model_its_own_options() {
        let config = parse(&json!({"memory": 72, "state_cache": 16, "models": [
            {"model": "/m/deepseek", "ced": 128},
            {"model": "/m/small", "id": "small", "draft": 1, "ctx": 32768}
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
        assert_eq!((deepseek.engine.draft, deepseek.id.as_deref()), (0, None));
        assert_eq!(
            (small.id.as_deref(), small.engine.draft, small.engine.ctx),
            (Some("small"), 1, 32768)
        );
        assert!(small.engine.deepseek().is_none());
    }

    #[test]
    fn prefill_prune_is_on_unless_ced_is_on() {
        let models = parse(&json!({"models": [
            {"model": "/m/a", "experts": "/m/a.gguf"},
            {"model": "/m/b", "ced": 128},
            {"model": "/m/c", "prefill_prune": false},
            {"model": "/m/d", "ced": "off", "prefill_prune": true}
        ]}))
        .unwrap()
        .models;
        let prunes: Vec<bool> = models
            .iter()
            .map(|model| model.engine.deepseek().unwrap().prunes_prefill())
            .collect();
        assert_eq!(prunes, [true, false, false, true]);
        let both = parse(&json!({"models": [{"model": "/m", "ced": 128, "prefill_prune": true}]}));
        assert!(both.unwrap_err().contains("prefill_prune needs ced off"));
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
            parse(&json!({"models": [{"model": "/m", "ced": "off", "lookup": "off"}]}))
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
