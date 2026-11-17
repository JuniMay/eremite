//! The server's settings.

use serde_json::Value;

use crate::engine;

/// The most GiB that a memory budget takes. The reason for this value is
/// unknown.
const MAX_MEMORY_GIB: f64 = 1024.0;
/// The fewest GiB that the reserve takes. The reason for this value is unknown.
const MIN_RESERVE_GIB: f64 = 2.0;
/// The most requests or states that a count setting takes. The reason for this
/// value is unknown.
const MAX_COUNT: f64 = 65536.0;

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
}
