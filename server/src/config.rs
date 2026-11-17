//! The server's settings.
//!
//! `-m DIR` serves one model, with its options on the command line. `--config
//! FILE` serves the models that a JSON file lists. Each model there has its own
//! options. They take the command line's names without the dashes, with `_` for
//! `-`. The file's root holds the server-wide settings.

use serde_json::{Map, Value};

use crate::engine;

/// The most tokens that a context holds. A model also takes at most its own
/// context.
const MAX_CONTEXT_TOKENS: i32 = 1 << 20;
/// The most GiB that a memory budget takes. The reason for this value is
/// unknown.
const MAX_MEMORY_GIB: f64 = 1024.0;
/// The fewest GiB that the reserve takes. The reason for this value is unknown.
const MIN_RESERVE_GIB: f64 = 2.0;
/// The most requests or states that a count setting takes. The reason for this
/// value is unknown.
const MAX_COUNT: f64 = 65536.0;

/// One model's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// The name that requests give in `model`; None: the family's default name.
    pub id: Option<String>,
    /// The engine's options for the model.
    pub engine: engine::Options,
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
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            think: None,
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
            "prefill_prune" => options.deepseek_mut()?.prefill_prune = switch()?,
            "packed_state_rows" => options.deepseek_mut()?.packed_state_rows = switch()?,
            "memory" | "reserve" | "gpu_timeout" | "max_waiting_requests" => {
                return Err(format!("{name} is server-wide; move it outside models"));
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
            _ => return Err(format!("no option {name}")),
        }
        Ok(())
    }
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
    /// The most requests that wait for one model. Another request then gets
    /// HTTP 429, too many requests.
    pub max_waiting_requests: usize,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            memory_gb: 0.0,
            reserve_gb: 10.0,
            gpu_timeout_s: 60.0,
            max_waiting_requests: 16,
        }
    }
}

impl Budgets {
    /// Sets the budget `name` to `value`. Returns the budget's message when
    /// either is invalid.
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
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
            "max_waiting_requests" if is_count(MAX_COUNT) => {
                self.max_waiting_requests = number as usize
            }
            "max_waiting_requests" => {
                return Err(format!(
                    "max_waiting_requests takes 1 to {MAX_COUNT} requests"
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
    /// The server-wide budgets and limits.
    pub budgets: Budgets,
    /// The models, in the file's order.
    pub models: Vec<ModelConfig>,
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
            "max_waiting_requests",
        ]
        .contains(&key.as_str())
    }) {
        return Err(format!("no server setting {key}"));
    }
    let mut budgets = Budgets::default();
    for name in ["memory", "reserve", "gpu_timeout", "max_waiting_requests"] {
        if let Some(value) = object.get(name) {
            budgets.set(name, value)?;
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
    Ok(Config { budgets, models })
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
    fn model_entries_cannot_create_separate_memory_budgets() {
        for setting in ["memory", "reserve", "gpu_timeout"] {
            let mut model = json!({"model": "/models/second"});
            model[setting] = json!(8);
            assert!(
                parse(&json!({"models": [model]})).is_err(),
                "{setting} belongs at the server level"
            );
        }
    }

    #[test]
    fn a_file_gives_each_model_its_own_options() {
        let config = parse(&json!({"memory": 72, "models": [
            {"model": "/m/deepseek"},
            {"model": "/m/small", "id": "small", "ctx": 32768, "top_k": 20, "think": 0}
        ]}))
        .unwrap();
        let models = config.models;
        assert_eq!(models.len(), 2);
        let (deepseek, small) = (&models[0], &models[1]);
        assert_eq!(
            (deepseek.engine.model.as_str(), config.budgets.memory_gb),
            ("/m/deepseek", 72.0)
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
    fn unknown_or_invalid_options_are_refused() {
        assert!(parse(&json!({"models": []})).is_err());
        assert!(parse(&json!({"models": [{"cache": 1}]})).is_err());
        assert!(
            parse(&json!({"models": [{"model": "/m", "cahce": 1}]}))
                .unwrap_err()
                .contains("cahce")
        );
        assert!(parse(&json!({"models": [{"model": "/m", "top_p": 2}]})).is_err());
        assert!(parse(&json!({"prt": 8008, "models": [{"model": "/m"}]})).is_err());
        for timeout in [json!(0), json!(-1), json!("long")] {
            assert!(
                parse(&json!({"gpu_timeout": timeout, "models": [{"model": "/m"}]})).is_err(),
                "gpu_timeout {timeout} is refused"
            );
        }
        let timeout = |root: Value| parse(&root).unwrap().budgets.gpu_timeout_s;
        assert_eq!(timeout(json!({"models": [{"model": "/m"}]})), 60.0);
        assert_eq!(
            timeout(json!({"gpu_timeout": 120, "models": [{"model": "/m"}]})),
            120.0
        );
    }
}
