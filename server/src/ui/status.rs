//! What the terminal UI reads of a model's status: the parts of the JSON of
//! worker::build_model_status_json that its views show. The memory of the
//! system and of all models is the same in every model's status.

use std::collections::BTreeMap;

use serde::Deserialize;

/// What a model does, as its status says.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// The model loads its weights.
    Loading,
    /// A prompt prefills.
    Prefill,
    /// Requests decode, and no prompt prefills.
    Decode,
    /// A yield holds the model: the server yields, has yielded, or resumes.
    #[serde(alias = "yielding", alias = "yielded", alias = "resuming")]
    Held,
    /// No request runs. A phase that the view does not know reads as idle.
    #[default]
    #[serde(other)]
    Idle,
}

/// What a request does.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Activity {
    /// It waits for its prefill.
    #[default]
    Waiting,
    /// Its prompt prefills.
    Prefill,
    /// It decodes its reply.
    Decoding,
}

/// A model's status.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct ModelStatus {
    /// Whether the model's weights are in memory.
    pub loaded: bool,
    /// Why the model's last load failed. None after a load that worked.
    pub load_error: Option<String>,
    /// What the model does.
    pub phase: Phase,
    /// The seconds that the model has been in its phase.
    pub phase_seconds: f64,
    /// The model's requests that run, then those that wait.
    pub requests: Vec<Request>,
    /// The model's sessions, in the order of their tokens.
    pub sessions: Vec<Session>,
    /// The GiB that the state cache counts for this model's states.
    pub state_cache_gib: f64,
    /// The GiB that the model's states may take.
    pub state_cache_limit_gib: f64,
    /// The states saved on the SSD, or None without them.
    pub state_disk: Option<StateDisk>,
    /// The GiB kept available to other programs.
    pub reserve_gib: f64,
    /// The setting of the models' memory limit in GiB, or 0 without one.
    pub memory_gib: f64,
    /// The memory of every model.
    pub model_memory: Vec<ModelMemory>,
    /// The limit of all models' memory now.
    pub model_memory_budget: MemoryBudget,
    /// The counts of the model's ended requests and prefills.
    pub totals: Totals,
    /// The number of requests that started with each kind of plan, by the
    /// plan's word.
    pub plans: BTreeMap<String, u64>,
    /// The model's decoding counters since the start.
    pub model_decode: Decode,
    /// The engine's other counters and the system's memory.
    pub engine_counters: Counters,
}

/// A request of a model.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Request {
    /// The request's number in the log.
    #[serde(rename = "request")]
    pub id: u64,
    /// What the request does.
    pub status: Activity,
    /// The number of its session. None while it waits.
    pub session: Option<u64>,
    /// Its prompt's tokens.
    pub prompt_tokens: u64,
    /// The prompt's tokens that came from the state cache.
    pub reused_tokens: u64,
    /// The tokens that its reply generated.
    pub generated_tokens: u64,
    /// The experts of its latest tokens, the oldest first.
    pub routes: Vec<Route>,
}

/// The experts that one token chose in each layer with routed experts, and
/// where each one was.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Route {
    /// The experts, layer by layer.
    pub experts: Vec<u16>,
    /// Where each expert was: 0 in memory, 1 read ahead, 2 late, 3
    /// substituted, 4 from the SSD.
    pub sources: Vec<u8>,
}

/// A session of a model: a state that the state cache keeps.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Session {
    /// The session's number in the log.
    #[serde(rename = "session")]
    pub number: u64,
    /// The tokens whose rows its state holds.
    pub tokens: u64,
    /// The seconds since a request last used it.
    pub idle_seconds: f64,
    /// Whether a request runs in it.
    pub busy: bool,
    /// The bytes that the state cache counts for it.
    pub bytes: u64,
    /// The position of its latest prompt-end snapshot, if it has one.
    pub prompt_end: Option<u64>,
    /// The tokens that it shares from position 0 with the next session.
    pub shared_with_next: u64,
}

/// The states saved on the SSD.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct StateDisk {
    /// Their folder.
    pub dir: String,
    /// The GiB that they take.
    pub gib: f64,
    /// The GiB that they may take.
    pub limit_gib: f64,
}

/// A model's memory, in bytes.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct ModelMemory {
    /// The model's name.
    pub model: String,
    /// Its dense weights.
    pub weights: u64,
    /// Its expert cache.
    pub experts: u64,
    /// Its working buffers.
    pub workspace: u64,
    /// Its states and their snapshots.
    pub states: u64,
}

/// The limit of all models' memory.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct MemoryBudget {
    /// The bytes that the models may take now.
    pub limit: u64,
}

/// The counts of a model's ended requests and prefills.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Totals {
    /// The requests that ended.
    pub requests: u64,
    /// Their prompts' tokens.
    pub prompt_tokens: u64,
    /// The tokens that they generated.
    pub generated_tokens: u64,
    /// Their prompts' tokens that came from the state cache.
    pub reused_tokens: u64,
    /// The prompt tokens that the ended prefills computed.
    pub prefilled_tokens: u64,
    /// The seconds that the ended prefills took.
    pub prefill_seconds: f64,
}

/// A model's decoding counters since the start.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Decode {
    /// The rows that decoding computed.
    pub computed_rows: u64,
    /// The seconds of decoding calls.
    pub seconds: f64,
    /// The seconds in which the GPU's main queue was busy with them.
    pub main_queue_gpu_seconds: f64,
    /// The GB of experts that the decoded rows read.
    pub requested_expert_gb: f64,
    /// The seconds that the decoded rows waited for expert reads.
    pub read_wait_seconds: f64,
}

/// The engine's counters of a model, and the system's memory.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Counters {
    /// The model's layers. It is 0 while the model is not loaded.
    pub layers: u64,
    /// The experts that a decoded row chooses in all layers together. It is
    /// 0 for a model without routed experts.
    pub experts_per_row: u64,
    /// The layers with routed experts.
    pub routed_layers: u64,
    /// The experts of each such layer.
    pub experts_per_layer: u64,
    /// The decoded rows' experts by where each one was: in memory, read
    /// ahead, late, substituted, from the SSD.
    pub decode_expert_sources: [u64; 5],
    /// The prompt rows computed.
    pub prefill_rows: u64,
    /// The bytes of experts that the prompt rows read.
    pub prefill_read_bytes: u64,
    /// The running prefill's layer, or 0 without one.
    pub prefill_layer: u64,
    /// The running prefill's rows.
    pub prefill_rows_now: u64,
    /// The bytes of experts read since the start, counted as each read ends.
    pub expert_read_bytes: u64,
    /// The bytes of the experts mapped for the GPU.
    pub mapped_expert_bytes: u64,
    /// The system's memory.
    pub memory: SystemMemory,
}

/// The system's memory, in bytes.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct SystemMemory {
    /// The server's memory footprint.
    pub footprint: u64,
    /// The memory available to the server.
    pub available: u64,
    /// The GPU's memory in use by all programs.
    pub gpu_in_use: u64,
    /// The GPU's memory limit. None before the GPU starts.
    pub gpu_limit: Option<u64>,
    /// The memory installed.
    pub total: u64,
    /// The free memory.
    pub free: u64,
    /// The file cache.
    pub file: u64,
    /// The part of the file cache that the system keeps rather than give up
    /// for other memory. The rest of the file cache counts as free.
    pub file_kept: u64,
    /// The wired memory.
    pub wired: u64,
    /// The swap in use.
    pub swap_used: u64,
    /// The memory pressure: 1 normal, 2 warn, 4 critical.
    pub pressure: u64,
}

impl ModelStatus {
    /// Returns the tokens that the model's requests generated: the ended
    /// requests' and the running ones'.
    pub fn count_generated_tokens(&self) -> u64 {
        self.totals.generated_tokens
            + self
                .requests
                .iter()
                .map(|request| request.generated_tokens)
                .sum::<u64>()
    }

    /// Returns the prompt rows that the model computed. The running prefill's
    /// rows count by the share of the layers that they went through.
    pub fn count_prompt_rows(&self) -> f64 {
        let counters = &self.engine_counters;
        let through = counters.prefill_layer as f64 / counters.layers.max(1) as f64;
        counters.prefill_rows as f64 + counters.prefill_rows_now as f64 * through
    }

    /// Returns how many of the model's requests run, and how many wait.
    pub fn count_requests(&self) -> (usize, usize) {
        let waiting = self
            .requests
            .iter()
            .filter(|request| request.status == Activity::Waiting)
            .count();
        (self.requests.len() - waiting, waiting)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A status in the form that the worker builds: a decoding request and a
    /// waiting one, two sessions, and a prefill at layer 31 of 43.
    const STATUS: &str = r#"{
        "loaded": true, "load_error": null, "phase": "decode", "phase_seconds": 2.5,
        "decoding": 1, "waiting": 1,
        "requests": [
            {"request": 5, "status": "decoding", "session": 5, "prompt_tokens": 23,
             "reused_tokens": 2, "generated_tokens": 516,
             "routes": [{"experts": [3, 200], "sources": [0, 4]}]},
            {"request": 6, "status": "waiting", "session": null, "prompt_tokens": 12000,
             "reused_tokens": 0, "generated_tokens": 0, "routes": []}],
        "sessions": [
            {"session": 2, "tokens": 550, "idle_seconds": 23, "busy": false,
             "bytes": 53687091, "prompt_end": 518, "shared_with_next": 7},
            {"session": 5, "tokens": 604, "idle_seconds": 0, "busy": true,
             "bytes": 53687091, "prompt_end": null, "shared_with_next": 0}],
        "state_cache_gib": 0.33, "state_cache_limit_gib": 8.0,
        "state_disk": {"dir": "/home/me/.cache/eremite/states", "states": 0, "gib": 0.0,
                       "limit_gib": 32.0, "loads": 0, "saves": 0, "written_gib": 0.0},
        "reserve_gib": 10.0, "memory_gib": 64.0,
        "model_memory": [{"model": "deepseek-v4.1-flash", "weights": 7730941132,
                          "experts": 47352328192, "workspace": 64424509, "states": 354334801}],
        "model_memory_budget": {"limit": 56264570880, "allocated": 1, "committed": 1, "peak": 1},
        "totals": {"requests": 4, "prompt_tokens": 1938, "generated_tokens": 192,
                   "reused_tokens": 12, "prefilled_tokens": 1926, "prefill_seconds": 24.7},
        "plans": {"cold": 1, "forked": 4},
        "model_decode": {"computed_rows": 708, "seconds": 30.5, "main_queue_gpu_seconds": 20.4,
                         "requested_expert_gb": 201.5, "read_wait_seconds": 8.6},
        "engine_counters": {"layers": 43, "experts_per_row": 240, "routed_layers": 40,
            "experts_per_layer": 384, "decode_expert_sources": [150000, 2000, 3000, 0, 9000],
            "prefill_rows": 1926, "prefill_read_bytes": 291000000000, "prefill_layer": 31,
            "prefill_rows_now": 430, "expert_read_bytes": 492500000000,
            "mapped_expert_bytes": 45742809088,
            "memory": {"footprint": 55942467584, "available": 11489037926,
                       "gpu_in_use": 56693760819, "gpu_limit": 115427033088,
                       "total": 137438953472, "free": 1181116006, "file": 7838315315,
                       "file_kept": 0, "wired": 60880148070, "swap_used": 0, "pressure": 1}},
        "engine_times": {"gpu_busy_seconds": 20.4}
    }"#;

    #[test]
    fn a_status_gives_the_requests_sessions_counters_and_memory() {
        let status: ModelStatus = serde_json::from_str(STATUS).expect("the worker's form");
        assert_eq!(status.phase, Phase::Decode);
        assert_eq!(status.requests[0].status, Activity::Decoding);
        assert_eq!(status.requests[0].routes[0].sources, [0, 4]);
        assert_eq!(status.requests[1].session, None);
        assert_eq!(status.count_requests(), (1, 1));
        assert_eq!(status.count_generated_tokens(), 192 + 516);
        // 1,926 rows and 31 of 43 layers of the running 430
        let rows = 1926.0 + 430.0 * 31.0 / 43.0;
        assert!((status.count_prompt_rows() - rows).abs() < 1e-9);
        assert_eq!(status.sessions[0].prompt_end, Some(518));
        assert_eq!(status.plans["forked"], 4);
        assert_eq!(status.engine_counters.memory.gpu_limit, Some(115427033088));
        assert_eq!(status.state_disk.map(|disk| disk.limit_gib), Some(32.0));
    }

    #[test]
    fn missing_parts_and_unknown_phases_take_their_defaults() {
        let status: ModelStatus =
            serde_json::from_str(r#"{"phase": "something new", "requests": []}"#).unwrap();
        assert_eq!(status.phase, Phase::Idle);
        assert_eq!(status.count_prompt_rows(), 0.0);
        for (phase, read) in [
            ("loading", Phase::Loading),
            ("yielded", Phase::Held),
            ("resuming", Phase::Held),
        ] {
            let status: ModelStatus =
                serde_json::from_str(&format!(r#"{{"phase": "{phase}"}}"#)).unwrap();
            assert_eq!(status.phase, read, "{phase}");
        }
    }
}
