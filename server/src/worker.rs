//! The worker: one engine thread that runs the requests of every model.
//!
//! The engine thread registers the configured models. It makes every engine
//! call. So the GPU work of all models goes through one command queue. It loads
//! a model at the model's first request. HTTP handlers submit each request to
//! its model's queue. At most max_waiting_requests requests wait for one model.
//! Each handler gets its request's events back on a channel.
//!
//! Each turn of the worker's loop gives every model one step. A step admits
//! every waiting request. It starts a prefill call for each prefilling request
//! with no call running. Prefills run beside each other and beside decoding. A
//! model that prefills a whole prompt in one call gets the rest of the prompt.
//! Otherwise the call takes at most the model's prefill rows. The step then
//! starts decoding calls for the decoding requests with no call running. A
//! request starts decoding as soon as its prefill ends. Several decoding
//! requests run together, at most the model's rows per call.
//!
//! Calls run on streams. The calls of all models run at the same time. After
//! the steps, the loop advances the streams until a call ends. It then finishes
//! that call. A prompt resumes from a state in the state cache. The worker also
//! builds each model's status.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::calls::CallChooser;
use crate::control::GIB;
use crate::engine::{
    self, CacheCounts, ExpandError, Logits, Media, Model, PrefillInfo, Sampler, SnapshotHandle,
    StateHandle, Stream,
};
use crate::state_cache::{self, Cache, Evicted, Plan, PlanLimits, Snapshot, SnapshotKind};

/// The engine needs a stack of at least 16 MiB on its thread.
const ENGINE_STACK_BYTES: usize = 16 << 20;
/// The bytes in one GB, for expert reads.
const GB: f64 = 1e9;
/// The engine thread's states, in the order of eremite_stats's thread_s and
/// gpu_idle_s.
const THREAD_STATES: [&str; 5] = ["outside", "pass", "wait_gpu", "wait_read", "wait_other"];

/// Limits the rows_per_call `setting` to 1 through `model_max_rows`. Without a
/// setting, it returns `model_max_rows`, at least 1. The result is the most
/// rows a decoding call takes.
fn limit_rows_per_call(model_max_rows: usize, setting: Option<usize>) -> usize {
    let max_rows = model_max_rows.max(1);
    setting.map_or(max_rows, |rows| rows.clamp(1, max_rows))
}

/// What the threads share. The engine thread sets it when it starts.
static SHARED: Mutex<Option<Arc<Shared>>> = Mutex::new(None);
/// The engine thread's handle. Stopping the worker joins it.
static ENGINE_THREAD: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// What wait_for_stop waits on: the engine's stop hook wakes it.
static STOPPED: tokio::sync::Notify = tokio::sync::Notify::const_new();

// ---- requests: the handlers' side ----

/// Why a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The reply reached a stop token.
    StopToken,
    /// The reply reached max_tokens or the end of the context.
    Length,
    /// The client cancelled the request.
    Cancelled,
    /// The server stopped.
    ServerStop,
}

/// What the worker reports about a request.
#[derive(Debug)]
pub enum Event {
    /// The request starts decoding. Its prefill ended, or it needed none.
    /// `prompt_tokens` counts the prompt with its media expanded.
    /// `reused_tokens` counts the prompt tokens taken from the state cache.
    DecodingStarted {
        prompt_tokens: usize,
        reused_tokens: usize,
    },
    /// The tokens kept since the last Tokens event. They never hold a stop
    /// token.
    Tokens(Vec<u32>),
    /// The request ended. This is the last event.
    Done { reason: FinishReason },
    /// The worker refused the request before its prefill. This is the last
    /// event. The string says why.
    Refused(String),
    /// The request failed: its model did not load, or memory was short. This is
    /// the last event. The string says why. The other models still serve
    /// requests.
    Failed(String),
}

/// How a request samples its tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// 0 takes the most likely token.
    pub temperature: f64,
    /// Sampling keeps the most likely tokens until their probabilities add up
    /// to top_p.
    pub top_p: f64,
    /// Sampling keeps at most this many of the most likely tokens. 0 keeps
    /// every token.
    pub top_k: u32,
    /// The random seed. None takes a random seed.
    pub seed: Option<u64>,
}

/// A request for one model.
pub struct Request {
    /// The index of the request's model, in the configured order.
    pub model_index: usize,
    /// The prompt's token ids. Each image's place holds the model's image
    /// token.
    pub prompt: Vec<i32>,
    /// The bytes of each media file, in the prompt's order.
    pub media: Vec<Vec<u8>>,
    /// How the request samples its tokens.
    pub sampling: Sampling,
    /// The most tokens the reply may have.
    pub max_tokens: i64,
    /// The log shows "req ID NOTE" with this note when the request is queued.
    pub note: String,
}

/// Why the worker did not queue a request.
pub enum SubmitError {
    /// Too many requests wait for the model, or the server is stopping. The
    /// string says why.
    Busy(String),
}

/// The flags that a request's cancel guard and the worker share.
#[derive(Default)]
struct RequestFlags {
    /// The cancel guard was dropped before the request ended.
    cancelled: AtomicBool,
    /// The worker sent the request's last event.
    last_event_sent: AtomicBool,
}

/// Cancels its request when it is dropped. The response drops it at its end.
/// The response ends when the request finished or its client went away. A stop
/// sequence can also end the response early. A request that ended already is
/// left alone.
pub struct CancelGuard {
    /// The request's id.
    id: u64,
    /// The flags it shares with the worker.
    flags: Arc<RequestFlags>,
}

impl CancelGuard {
    /// Returns the request's id: its number in the log.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns a guard for a request that was never submitted, for the tests.
    #[cfg(test)]
    pub fn new_unsubmitted(id: u64) -> CancelGuard {
        CancelGuard {
            id,
            flags: Arc::default(),
        }
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.flags.last_event_sent.load(Ordering::SeqCst)
            && !self.flags.cancelled.swap(true, Ordering::SeqCst)
        {
            engine::log(&format!("req {}: cancellation requested", self.id));
            wake_worker();
        }
    }
}

/// A submitted request, as the worker holds it.
struct Job {
    /// The request's id: its number in the log.
    id: u64,
    /// The request itself.
    request: Request,
    /// Sends the request's events to its handler.
    events: mpsc::UnboundedSender<Event>,
    /// The flags it shares with its cancel guard.
    flags: Arc<RequestFlags>,
    /// When the request was queued.
    queued_at: Instant,
}

impl Job {
    /// Sends `event` as the request's last event.
    fn send_last_event(&self, event: Event) {
        let _ = self.events.send(event);
        self.flags.last_event_sent.store(true, Ordering::SeqCst);
    }
}

// ---- what the threads share ----

/// A served model's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSettings {
    /// The model's name, as requests give it.
    pub id: String,
    /// The engine's options for the model.
    pub options: engine::Options,
    /// The most rows a decoding call takes. None takes the model's own limit.
    pub rows_per_call: Option<usize>,
}

/// What the HTTP handlers and the worker share under one lock.
#[derive(Default)]
struct Queue {
    /// Each model's waiting requests, the oldest first.
    waiting: Vec<VecDeque<Job>>,
    /// Set at the stop. No request is queued after it.
    closed: bool,
    /// The id of the last queued request. The first request gets 1.
    last_request_id: u64,
}

/// What a model is doing and has done, for its status.
#[derive(Default)]
struct ModelStatus {
    /// Why the model's last load failed. None after a load that worked.
    load_error: Option<String>,
    /// The model's engine counters.
    stats: engine::Stats,
    /// Whether the model's weights are in memory.
    loaded: bool,
    /// What the model does now: idle, loading, prefill or decode.
    phase: &'static str,
    /// When the model started its phase.
    phase_since: Option<Instant>,
    /// The number of the model's requests that decode.
    decoding_count: usize,
    /// The id of the request shown in the status. It is the first request that
    /// prefills, or else the first request.
    shown_request_id: u64,
    /// The number of prompt tokens of the shown request.
    shown_prompt_tokens: usize,
    /// The number of the shown request's prompt tokens that came from the state
    /// cache.
    shown_reused_tokens: usize,
    /// The number of tokens that the shown request generated.
    shown_generated_tokens: usize,
    /// The number of requests that ended.
    total_requests: u64,
    /// The number of prompt tokens of the ended requests.
    total_prompt_tokens: u64,
    /// The number of tokens that the ended requests generated.
    total_generated_tokens: u64,
    /// The plan of the latest prefill.
    last_prefill: PrefillInfo,
    /// Whether the decoding requests split evenly over at least two calls.
    split_calls: bool,
    /// The model's states: each one's number, token count and last use.
    sessions: Vec<(u64, usize, Instant)>,
    /// The state bytes committed for the model: what the state cache counts for
    /// its states and their snapshots, the reserved ones included.
    state_committed_bytes: u64,
    /// The bytes that the engine allocated for the model's states and their
    /// snapshots.
    state_allocated_bytes: u64,
}

/// A model as the HTTP handlers and the worker share it. The tokenizer works on
/// any thread.
pub struct SharedModel {
    /// The model's settings.
    pub settings: ModelSettings,
    /// The model's tokenizer.
    pub tokenizer: engine::Tokenizer,
    /// The model's status. The worker updates it.
    status: Mutex<ModelStatus>,
}

/// What the HTTP handlers and the worker share.
struct Shared {
    /// The waiting requests.
    queue: Mutex<Queue>,
    /// Wakes the worker when it has work: a new request, a cancellation, a
    /// status reader's request, or the stop.
    worker_wakeup: Condvar,
    /// The served models, in their configured order.
    models: Vec<SharedModel>,
    /// The server-wide budgets and limits.
    budgets: crate::config::Budgets,
    /// The state bytes committed for all models together.
    state_committed_bytes: AtomicU64,
    /// Whether a status reader asked for fresh counters and states. The worker
    /// publishes them at its next turn, also between the steps of a running
    /// call.
    status_wanted: AtomicBool,
    /// Wakes the status readers once the worker has published, when a long
    /// operation starts, and at the stop.
    status_published: tokio::sync::Notify,
    /// The worker's long operations running now (LongOperation).
    long_operations: AtomicUsize,
}

/// Marks a worker operation that may take seconds and publishes nothing, as
/// long as it lives. Status readers then take the last published status at
/// once.
struct LongOperation<'a>(&'a Shared);

impl<'a> LongOperation<'a> {
    /// Starts a long operation in `shared`. It wakes the status readers that
    /// wait for a publish.
    fn start(shared: &'a Shared) -> LongOperation<'a> {
        shared.long_operations.fetch_add(1, Ordering::AcqRel);
        shared.status_published.notify_waiters();
        LongOperation(shared)
    }
}

impl Drop for LongOperation<'_> {
    fn drop(&mut self) {
        self.0.long_operations.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Clones the handle to what the threads share. Panics before the worker
/// starts.
fn clone_shared() -> Arc<Shared> {
    SHARED.lock().unwrap().clone().expect("the worker started")
}

/// Locks `mutex`. It also takes the data after a thread panicked with the lock
/// held.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Asks the worker for fresh counters and states, and returns once it has
/// published them. While the server stops, and while the worker runs a long
/// operation, it returns at once: the status is then the last one published.
pub async fn refresh_status() {
    let shared = clone_shared();
    // A notified future gets the wake from the time it is made, before it is
    // polled.
    let published = shared.status_published.notified();
    if engine::is_stopping() || shared.long_operations.load(Ordering::Acquire) > 0 {
        return;
    }
    shared.status_wanted.store(true, Ordering::Release);
    wake_worker();
    published.await;
}

/// Lists the models the worker serves, in their configured order: each one's
/// id, tokenizer and settings. A request's model_index is an index into this
/// list.
pub fn list_models() -> Vec<(String, engine::Tokenizer, ModelSettings)> {
    clone_shared()
        .models
        .iter()
        .map(|shared_model| {
            (
                shared_model.settings.id.clone(),
                shared_model.tokenizer,
                shared_model.settings.clone(),
            )
        })
        .collect()
}

/// Starts the engine thread. The thread registers each model without loading
/// its weights. This call returns true once the worker takes requests. It
/// returns false when the engine cannot start or a model's settings are
/// invalid. The log says why a model's settings are invalid.
pub fn start(
    model_settings: Vec<ModelSettings>,
    budgets: crate::config::Budgets,
    on_log: Option<extern "C" fn(*const std::ffi::c_char)>,
    on_die: Option<extern "C" fn()>,
) -> bool {
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let handle = thread::Builder::new()
        .name("eremite-engine".into())
        .stack_size(ENGINE_STACK_BYTES)
        .spawn(move || {
            if !engine::init(
                &budgets.process_settings(),
                on_log,
                on_die,
                Some(wake_after_stop),
            ) {
                let _ = started_sender.send(false);
                return;
            }
            engine::refresh_stats();
            let mut models = Vec::new();
            let mut model_handles = Vec::new();
            for settings in &model_settings {
                let Some(model) = Model::open(&settings.options) else {
                    let _ = started_sender.send(false);
                    return;
                };
                models.push(SharedModel {
                    settings: settings.clone(),
                    tokenizer: model.tokenizer(),
                    status: Mutex::new(ModelStatus {
                        stats: model.read_stats(),
                        ..ModelStatus::default()
                    }),
                });
                model_handles.push(Rc::new(model));
            }
            let model_count = models.len();
            let shared = Arc::new(Shared {
                queue: Mutex::new(Queue {
                    waiting: (0..model_count).map(|_| VecDeque::new()).collect(),
                    ..Queue::default()
                }),
                worker_wakeup: Condvar::new(),
                models,
                budgets,
                state_committed_bytes: AtomicU64::new(0),
                status_wanted: AtomicBool::new(false),
                status_published: tokio::sync::Notify::new(),
                long_operations: AtomicUsize::new(0),
            });
            *SHARED.lock().unwrap() = Some(shared.clone());
            // A panic here ends the process, as the engine's fatal errors do.
            // The waiting requests could not end otherwise. Rust has already
            // printed the panic's message.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut worker = Worker::new(shared.clone(), model_handles);
                let _ = started_sender.send(true);
                worker.run_until_stop();
            }));
            if run.is_err() {
                std::process::exit(1);
            }
        })
        .expect("the engine's thread");
    let ok = started_receiver.recv().unwrap_or(false);
    if ok {
        *ENGINE_THREAD.lock().unwrap() = Some(handle);
    } else {
        let _ = handle.join();
    }
    ok
}

/// Requests the stop as SIGINT does.
pub fn request_stop() {
    engine::request_stop();
}

/// Wakes what waits for the stop: the worker and wait_for_stop. The engine runs
/// it after each stop request, on a thread of its own.
extern "C" fn wake_after_stop() {
    wake_worker();
    if let Some(shared) = SHARED.lock().unwrap().clone() {
        // The status readers then take the last published status.
        shared.status_published.notify_waiters();
    }
    STOPPED.notify_waiters();
}

/// Returns once a stop is requested.
pub async fn wait_for_stop() {
    loop {
        // A notified future gets the wake from the time it is made, before it
        // is polled.
        let stopped = STOPPED.notified();
        if engine::is_stopping() {
            return;
        }
        stopped.await;
    }
}

/// Wakes the worker, once it has started.
fn wake_worker() {
    if let Some(shared) = SHARED.lock().unwrap().clone() {
        let _queue = lock(&shared.queue);
        notify_worker(&shared);
    }
}

/// Wakes the worker: while it waits for its queue, and while the engine runs
/// its calls.
fn notify_worker(shared: &Shared) {
    shared.worker_wakeup.notify_all();
    engine::wake();
}

/// Stops the worker and waits for its thread to end. It requests the stop
/// unless a stop was requested already. Before its thread ends, the worker ends
/// every request.
pub fn stop() {
    if !engine::is_stopping() {
        request_stop();
    }
    if let Some(handle) = ENGINE_THREAD.lock().unwrap().take() {
        let _ = handle.join();
    }
}

/// Submits `request` to its model's queue. Returns the guard and the receiver
/// of its events. Dropping the guard cancels the request. The guard also holds
/// the request's id. Returns Busy when too many requests wait or the server is
/// stopping.
pub fn submit(
    request: Request,
) -> Result<(CancelGuard, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let shared = clone_shared();
    let (events, receiver) = mpsc::unbounded_channel();
    let flags = Arc::new(RequestFlags::default());
    let mut queue = lock(&shared.queue);
    if queue.closed
        || queue.waiting[request.model_index].len() >= shared.budgets.max_waiting_requests
    {
        let message = if queue.closed {
            "the server is stopping"
        } else {
            "the server is busy: too many requests are waiting"
        };
        return Err(SubmitError::Busy(message.into()));
    }
    queue.last_request_id += 1;
    let request_id = queue.last_request_id;
    if !request.note.is_empty() {
        // The log shows the note before the worker can take the request.
        engine::log(&format!("req {request_id} {}", request.note));
    }
    let (prompt_tokens, max_tokens, model_index) = (
        request.prompt.len(),
        request.max_tokens,
        request.model_index,
    );
    queue.waiting[model_index].push_back(Job {
        id: request_id,
        request,
        events,
        flags: flags.clone(),
        queued_at: Instant::now(),
    });
    engine::log(&format!(
        "req {request_id}: queued for {}, waiting={}, prompt_tokens={prompt_tokens}, max_new={max_tokens}",
        shared.models[model_index].settings.id,
        queue.waiting[model_index].len()
    ));
    notify_worker(&shared);
    Ok((
        CancelGuard {
            id: request_id,
            flags,
        },
        receiver,
    ))
}

// ---- a model's status ----

/// Loads `model`'s weights. Status readers take the last published status
/// meanwhile.
fn load_model(shared: &Shared, model: &Model) -> Result<(), String> {
    let _long = LongOperation::start(shared);
    model.load()
}

/// Rounds `value` to `digits` decimal places.
fn round(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

/// Builds each model's memory as JSON: its weights, experts, workspace and
/// state bytes.
fn build_model_memory_json(shared: &Shared) -> Vec<Value> {
    shared
        .models
        .iter()
        .map(|shared_model| {
            let status = lock(&shared_model.status);
            let stats = status.stats;
            let tracked = stats.dense_bytes
                + stats.cache_bytes
                + stats.model_workspace
                + status.state_allocated_bytes;
            json!({
                "model": shared_model.settings.id,
                "weights": stats.dense_bytes,
                "experts": stats.cache_bytes,
                "workspace": stats.model_workspace,
                "states": status.state_allocated_bytes,
                "states_committed": status.state_committed_bytes,
                "sessions": status.sessions.len(),
                "tracked": tracked,
            })
        })
        .collect()
}

/// Builds the status of the model at `model_index` as JSON. It shows what the
/// model is doing, its requests, its state cache, its counters, its expert
/// cache and the process's memory. Returns null for an unknown index.
pub fn build_model_status_json(model_index: usize) -> Value {
    let shared = clone_shared();
    let Some(shared_model) = shared.models.get(model_index) else {
        return Value::Null;
    };
    let waiting = lock(&shared.queue).waiting[model_index].len();
    let state_limit = (shared.budgets.state_cache_gb * GIB) as u64;
    let stats = lock(&shared_model.status).stats;
    let allocations = build_model_memory_json(&shared);
    let status = lock(&shared_model.status);
    let now = Instant::now();
    // The status rounds idle times to whole seconds. It rounds other seconds
    // and memory GiB to tenths, and state GiB to thousandths. Why these digits
    // were chosen is unknown.
    let sessions: Vec<Value> = status
        .sessions
        .iter()
        .map(|&(number, tokens, last_used_at)| {
            json!({
                "session": number,
                "tokens": tokens,
                "idle_seconds": round((now - last_used_at).as_secs_f64(), 0),
            })
        })
        .collect();
    let phase = if status.phase.is_empty() {
        "idle"
    } else {
        status.phase
    };
    let phase_seconds = status
        .phase_since
        .map_or(0.0, |since| (now - since).as_secs_f64());
    let total_state_bytes = shared.state_committed_bytes.load(Ordering::Relaxed);
    let last_prefill = &status.last_prefill;
    let options = &shared_model.settings.options;
    json!({
        "loaded": status.loaded,
        "load_error": status.load_error,
        "phase": phase,
        "phase_seconds": round(phase_seconds, 1),
        "decoding": status.decoding_count,
        "request": status.shown_request_id,
        "prompt_tokens": status.shown_prompt_tokens,
        "reused_tokens": status.shown_reused_tokens,
        "generated_tokens": status.shown_generated_tokens,
        "waiting": waiting,
        "sessions": sessions,
        "state_cache_gib": round(status.state_committed_bytes as f64 / GIB, 3),
        "state_cache_limit_gib": round(state_limit as f64 / GIB, 3),
        "state_cache_scope": "server",
        "state_cache_total_gib": round(total_state_bytes as f64 / GIB, 3),
        "reserve_gib": shared.budgets.reserve_gb,
        "memory_gib": shared.budgets.memory_gb,
        "model_memory": allocations,
        "model_memory_budget": {
            "limit": stats.model_memory_limit,
            "allocated": stats.model_memory_used,
            "committed": stats.model_memory_committed,
            "peak": stats.model_memory_peak,
        },
        "totals": {
            "requests": status.total_requests,
            "prompt_tokens": status.total_prompt_tokens,
            "generated_tokens": status.total_generated_tokens,
        },
        "ced_depth": options.deepseek().map_or(0, |deepseek| deepseek.ced),
        "prefill_prune": options.deepseek().is_some_and(|deepseek| deepseek.prefill_prune),
        "rows_per_call": shared_model.settings.rows_per_call,
        "split_decoding_calls": status.split_calls,
        "model_decode": {
            "computed_rows": stats.decode_rows,
            "seconds": stats.decode_seconds,
            "main_queue_gpu_seconds": stats.decode_gpu_seconds,
            "requested_expert_gb": stats.decode.bytes as f64 / GB,
            "read_wait_seconds": stats.decode.read_s,
        },
        "last_prefill": {
            "chunks": last_prefill.chunks,
            "tokens": last_prefill.tokens,
            "decoder_tokens": last_prefill.decoder_tokens,
            "ced_applied": last_prefill.ced_applied != 0,
            "prune_applied": last_prefill.prune_applied != 0,
            "prefix_tokens": last_prefill.prefix_tokens,
            "prefix_encoder_only": false,
        },
        "expert_cache": {
            "slots": stats.slots,
            "gib": round(stats.cache_bytes as f64 / GIB, 1),
            "hits": stats.cache.hits + stats.cache.prefetch_hits,
            "misses": stats.cache.misses,
            "read_gb": round(stats.cache.bytes as f64 / GB, 1),
            "slots_released": stats.cache.given_back,
        },
        "memory": {
            "available_gib": round(stats.available as f64 / GIB, 1),
            "footprint_gib": round(stats.footprint as f64 / GIB, 1),
        },
        "engine_times": build_engine_times_json(&stats),
    })
}

/// Builds where the process's time went since the start, as JSON. It is the
/// same for every model. It holds the engine thread's seconds in each of its
/// states. It holds the GPU main queue's busy seconds. It holds the queue's
/// idle seconds between command buffers, by the engine thread's state at the
/// time. It holds the SSD's seconds while an expert read runs. It also holds
/// the summed seconds of demand reads and of background reads.
fn build_engine_times_json(stats: &engine::Stats) -> Value {
    let by_state = |seconds: &[f64; THREAD_STATES.len()]| {
        Value::Object(
            THREAD_STATES
                .iter()
                .zip(seconds)
                .map(|(state, &value)| (state.to_string(), json!(value)))
                .collect(),
        )
    };
    json!({
        "thread_seconds": by_state(&stats.thread_s),
        "gpu_busy_seconds": stats.gpu_busy_s,
        "gpu_idle_seconds": by_state(&stats.gpu_idle_s),
        "ssd_busy_seconds": stats.ssd_busy_s,
        "demand_read_seconds": stats.demand_read_s,
        "background_read_seconds": stats.background_read_s,
    })
}

// ---- the worker ----

/// A prefill in progress: its prompt and how far it has run.
struct Prefill {
    /// The prompt's tokens, with the media expanded.
    prompt: Vec<i32>,
    /// The position where the next prefill call starts.
    next_position: usize,
    /// Where the prefill takes a branch snapshot, if anywhere.
    capture_position: Option<usize>,
    /// The position of the rows that the prefill continues, and their stale
    /// rows.
    origin: state_cache::Origin,
    /// Whether the prefill's first call replays.
    replays: bool,
    /// The prompt's media. They are kept until the prefill ends.
    media: Option<Media>,
    /// When the prefill started.
    started_at: Instant,
    /// The expert cache's counters when the prefill started.
    expert_counts_at_start: CacheCounts,
    /// Whether its calls run only while no other prefill call runs. It is set
    /// when its working memory did not fit beside another prefill's.
    must_run_alone: bool,
}

/// A request from its prefill to its end.
struct Generation {
    /// The submitted request.
    job: Job,
    /// The request's sampler.
    sampler: Sampler,
    /// The key of the request's state in the state cache.
    state_id: state_cache::StateId,
    /// The number of prompt tokens, with the media expanded.
    prompt_tokens: usize,
    /// The number of prompt tokens taken from the state cache. The prefill
    /// starts after them.
    reused_tokens: usize,
    /// Some while the prompt prefills.
    prefill: Option<Prefill>,
    /// The position of the next token.
    pos: usize,
    /// The number of tokens that the reply generated, a stop token included.
    generated: usize,
    /// The number of the reply's tokens inside its think block.
    reasoning_tokens: usize,
    /// Whether the reply is inside its think block.
    thinking: bool,
    /// Whether the reply reached a stop token.
    reached_stop_token: bool,
    /// Whether the client cancelled the request.
    cancelled: bool,
    /// The token sampled next. No pass has run it yet.
    next_token: i32,
    /// The kept tokens not sent to the client yet.
    unsent_tokens: Vec<u32>,
    /// The most tokens that the reply may generate.
    max_new_tokens: usize,
    /// When the request started decoding.
    decode_started_at: Instant,
    /// When the worker last logged the request's progress.
    progress_logged_at: Instant,
    /// The expert cache's counters when the request started decoding.
    expert_counts_at_decode_start: CacheCounts,
    /// Whether a call of this request runs on a stream.
    in_call: bool,
    /// The request's turn number since it last became ready to decode (Turns).
    turn: u64,
    /// When the request got its last token in a steady round, if the rounds
    /// since then stayed steady and kept their arrangement. The call chooser
    /// measures the gap from it to the next token.
    measured_token_at: Option<Instant>,
}

impl Generation {
    /// Returns whether the request decodes: its prefill has ended.
    fn is_decoding(&self) -> bool {
        self.prefill.is_none()
    }
}

/// An engine call that runs on a stream, and what to do when it ends.
enum Call {
    /// One prefill call of the request. It prefills the prompt up to `end`.
    /// `call_capture` is the position where the engine takes a snapshot during
    /// the call, if any.
    Prefill {
        request_id: u64,
        end: usize,
        call_capture: Option<usize>,
    },
    /// One token for each of these requests, one row each.
    DecodeRows {
        request_ids: Vec<u64>,
        tokens: Vec<i32>,
    },
}

/// A running call and the stream it runs on.
struct RunningCall {
    /// The stream that runs the call.
    stream: Stream,
    /// The call, and what to do when it ends.
    call: Call,
}

/// One model as the worker runs it: its engine handle, settings, requests and
/// streams. Its states are in the state cache that all models share. A prompt
/// resumes only from this model's states.
struct WorkerModel {
    /// The model's index in the configured order.
    model_index: usize,
    /// The model's engine handle.
    model: Rc<Model>,
    /// The model's settings.
    settings: ModelSettings,
    /// The model's context length, in tokens.
    ctx: usize,
    /// The requests from their prefill to their end.
    generations: Vec<Generation>,
    /// Admitted requests that wait for their turn to plan their prefill.
    admitted: VecDeque<Job>,
    /// The token that starts a think block, if the model has one.
    think_start_token: Option<i32>,
    /// The token that ends a think block, if the model has one.
    think_end_token: Option<i32>,
    /// The number of decoding requests in the last log line about them.
    logged_decoding_count: usize,
    /// The experts that a decoded row chooses in all layers together. It is 0
    /// for a model without routed experts.
    experts_per_row: usize,
    /// The calls that run now.
    running_calls: Vec<RunningCall>,
    /// The streams that ended a prefill call of a prompt that is still
    /// prefilling, and wait for another. They hold large working buffers. The
    /// worker drops them when no request is left.
    idle_prefill_streams: Vec<Stream>,
    /// The streams that ended a decoding call and wait for the next one.
    idle_decode_streams: Vec<Stream>,
    /// Chooses how the decoding requests split into calls.
    call_chooser: CallChooser,
    /// The round of decoding that the call chooser measures, if one runs.
    round: Option<Round>,
    /// The order of the decoding requests' turns.
    turns: Turns,
}

/// The order in which decoding requests take turns in calls. Each request that
/// becomes ready takes the next turn number, and the lowest number goes first.
/// Requests that become ready together, the rows of one call, take numbers in
/// the order they had in that call. So a request that a call left out goes
/// before every row of that call, and no request waits behind the same others
/// again and again.
#[derive(Default)]
struct Turns {
    /// The number of the next request that becomes ready.
    next: u64,
}

impl Turns {
    /// Returns the turn numbers of `count` requests that become ready together,
    /// in their order.
    fn take(&mut self, count: usize) -> std::ops::Range<u64> {
        let start = self.next;
        self.next += count as u64;
        start..self.next
    }
}

/// A round of decoding that the call chooser measures. It lasts until every
/// decoding request has a new token.
struct Round {
    /// The decoding requests that have no new token yet.
    waiting: Vec<u64>,
    /// The number of decoding requests.
    decoding_count: usize,
    /// Whether the requests' calls split.
    is_split: bool,
    /// Whether only these requests' decoding calls ran: no prompt, no other
    /// model's calls, and no calls of another arrangement.
    is_steady: bool,
    /// The gaps between two tokens of a request that ended in the round.
    gaps: usize,
    /// The seconds of those gaps in all.
    gap_seconds: f64,
}

/// A state's engine handles: the engine state and its snapshots. Dropping them
/// frees them through the model that owns them.
struct StateHandles {
    /// The model that owns the state and its snapshots.
    owner: Rc<Model>,
    /// The engine's state.
    engine_state: StateHandle,
    /// The state's snapshots, by position.
    snapshots: HashMap<usize, SnapshotHandle>,
    /// When a request last used the state.
    last_used_at: Instant,
}

impl Drop for StateHandles {
    fn drop(&mut self) {
        for (_, snapshot) in self.snapshots.drain() {
            self.owner.snapshot_free(snapshot);
        }
        self.owner.state_free(self.engine_state);
    }
}

/// The states of all models: the state cache's records and each state's engine
/// handles.
struct States {
    /// The state cache's records of the states.
    cache: Cache,
    /// Each state's engine handles, by the state's key.
    handles: HashMap<state_cache::StateId, StateHandles>,
    /// For a model that resumes only from snapshots, a prefill takes a snapshot
    /// at every multiple of this many positions.
    snapshot_interval: usize,
}

impl States {
    /// Creates the states of all models: an empty state cache of at most
    /// `limit` bytes and `max_states` states. `snapshot_interval` is the
    /// prefills' snapshot interval.
    fn new(limit: u64, max_states: usize, snapshot_interval: usize) -> Self {
        Self {
            cache: Cache::new(limit, max_states),
            handles: HashMap::new(),
            snapshot_interval,
        }
    }

    /// Frees what the state cache evicted: the snapshots, then the states.
    fn free_evicted(&mut self, Evicted { snapshots, states }: Evicted) {
        for (state_id, snapshot) in snapshots {
            self.free_snapshot(state_id, snapshot.pos);
        }
        for (state_id, _) in states {
            self.handles.remove(&state_id);
        }
    }

    /// Adds a new, empty state of the model at `model_index`. Returns its key,
    /// or None when the state's memory does not fit.
    fn add_state(&mut self, model_index: usize, model: Rc<Model>) -> Option<state_cache::StateId> {
        let engine_state = model.state_new()?;
        let state_id = self.cache.add_state(model_index, 0);
        self.handles.insert(
            state_id,
            StateHandles {
                owner: model,
                engine_state,
                snapshots: HashMap::new(),
                last_used_at: Instant::now(),
            },
        );
        Some(state_id)
    }

    /// Gets state `state_id`'s snapshot at `pos`. Panics when there is none.
    fn get_snapshot(&self, state_id: state_cache::StateId, pos: usize) -> SnapshotHandle {
        self.handles[&state_id].snapshots[&pos]
    }

    /// Inserts `snapshot` as state `state_id`'s snapshot at `pos`. It frees the
    /// snapshot it replaces.
    fn insert_snapshot(
        &mut self,
        state_id: state_cache::StateId,
        pos: usize,
        snapshot: SnapshotHandle,
    ) {
        let state_handles = self
            .handles
            .get_mut(&state_id)
            .expect("a state owns its snapshots");
        if let Some(previous) = state_handles.snapshots.insert(pos, snapshot) {
            state_handles.owner.snapshot_free(previous);
        }
    }

    /// Frees state `state_id`'s snapshot at `pos`, if there is one.
    fn free_snapshot(&mut self, state_id: state_cache::StateId, pos: usize) {
        let state_handles = self
            .handles
            .get_mut(&state_id)
            .expect("a state owns its snapshots");
        if let Some(snapshot) = state_handles.snapshots.remove(&pos) {
            state_handles.owner.snapshot_free(snapshot);
        }
    }

    /// Frees the snapshots in `snapshots` and the states in `states`.
    fn free_snapshots_and_states(
        &mut self,
        snapshots: Vec<(state_cache::StateId, Snapshot)>,
        states: Vec<(state_cache::StateId, Vec<Snapshot>)>,
    ) {
        for (state_id, snapshot) in snapshots {
            self.free_snapshot(state_id, snapshot.pos);
        }
        for (state_id, _) in states {
            self.handles.remove(&state_id);
        }
    }

    /// Publishes the states to the status: each model's states and state bytes.
    fn publish_states(&mut self, shared: &Shared) {
        shared
            .state_committed_bytes
            .store(self.cache.count_bytes(), Ordering::Relaxed);
        for (model_index, shared_model) in shared.models.iter().enumerate() {
            let entries: Vec<_> = self
                .cache
                .entries
                .iter()
                .filter(|(_, entry)| entry.model == model_index)
                .collect();
            let mut status = lock(&shared_model.status);
            status.sessions = entries
                .iter()
                .map(|(state_id, entry)| {
                    (
                        entry.number,
                        entry.tokens.len(),
                        self.handles[state_id].last_used_at,
                    )
                })
                .collect();
            status.state_committed_bytes = self.cache.count_model_bytes(model_index);
            status.state_allocated_bytes = entries
                .iter()
                .map(|(state_id, entry)| {
                    self.handles[state_id]
                        .owner
                        .state_bytes(Some(self.handles[state_id].engine_state), 0)
                        + entry
                            .snapshots
                            .iter()
                            .map(|snapshot| snapshot.bytes)
                            .sum::<u64>()
                })
                .sum();
        }
    }
}

/// The worker on the engine thread.
struct Worker {
    /// What the threads share.
    shared: Arc<Shared>,
    /// The models that the worker serves, in their configured order.
    models: Vec<WorkerModel>,
    /// The states of all models.
    states: States,
}

impl Worker {
    /// Creates the worker for the models in `shared` and their engine handles.
    fn new(shared: Arc<Shared>, model_handles: Vec<Rc<Model>>) -> Worker {
        let models: Vec<WorkerModel> = shared
            .models
            .iter()
            .zip(model_handles)
            .enumerate()
            .map(|(model_index, (shared_model, model))| {
                let settings = shared_model.settings.clone();
                WorkerModel {
                    model_index,
                    ctx: settings.options.ctx as usize,
                    generations: Vec::new(),
                    admitted: VecDeque::new(),
                    think_start_token: model.tokenizer().find_added_token("<think>"),
                    think_end_token: model.tokenizer().find_added_token("</think>"),
                    logged_decoding_count: 0,
                    experts_per_row: model.read_stats().experts_per_row.max(0) as usize,
                    running_calls: Vec::new(),
                    idle_prefill_streams: Vec::new(),
                    idle_decode_streams: Vec::new(),
                    call_chooser: CallChooser::new(),
                    round: None,
                    turns: Turns::default(),
                    model,
                    settings,
                }
            })
            .collect();
        let states = States::new(
            (shared.budgets.state_cache_gb * GIB) as u64,
            shared.budgets.max_cached_states,
            shared.budgets.snapshot_interval,
        );
        let mut worker = Worker {
            shared,
            models,
            states,
        };
        worker.states.publish_states(&worker.shared);
        worker
    }

    /// Publishes each model's engine counters to its status.
    fn publish_counters(&self) {
        for worker_model in &self.models {
            let mut status = lock(&self.shared.models[worker_model.model_index].status);
            status.stats = worker_model.model.read_stats();
            status.loaded = worker_model.model.is_loaded();
        }
    }

    /// Publishes the counters and the states when a status reader asked for
    /// them, and wakes the readers.
    fn publish_status_if_wanted(&mut self) {
        if self.shared.status_wanted.swap(false, Ordering::AcqRel) {
            engine::refresh_stats();
            self.states.publish_states(&self.shared);
            self.publish_counters();
            self.shared.status_published.notify_waiters();
        }
    }

    /// Watches each model's round of decoding. A round stops being steady while
    /// a prompt prefills or another model's calls run. It ends without a record
    /// when its requests change: when their number changes, or when a request
    /// that it waits for no longer decodes. Either way, the call chooser
    /// measures no gap that spans the change. The status shows whether the
    /// calls split.
    fn watch_rounds(&mut self) {
        let busy: Vec<bool> = self
            .models
            .iter()
            .map(|worker_model| {
                !worker_model.running_calls.is_empty() || !worker_model.admitted.is_empty()
            })
            .collect();
        for (i, worker_model) in self.models.iter_mut().enumerate() {
            let decoding_count = worker_model
                .generations
                .iter()
                .filter(|generation| generation.is_decoding())
                .count();
            let is_prefilling = worker_model
                .generations
                .iter()
                .any(|generation| !generation.is_decoding())
                || !worker_model.admitted.is_empty();
            let others = busy
                .iter()
                .enumerate()
                .any(|(j, &is_busy)| j != i && is_busy);
            if let Some(round) = worker_model.round.as_mut() {
                let left = round.waiting.iter().any(|&id| {
                    !worker_model
                        .generations
                        .iter()
                        .any(|generation| generation.job.id == id && generation.is_decoding())
                });
                if round.decoding_count != decoding_count || left {
                    worker_model.round = None;
                    worker_model.forget_token_gaps();
                } else if is_prefilling || others {
                    round.is_steady = false;
                    worker_model.forget_token_gaps();
                }
            }
            lock(&self.shared.models[worker_model.model_index].status).split_calls =
                worker_model.call_chooser.is_split();
        }
    }

    /// Advances the running calls of every model until one ends, and finishes
    /// the call that ended. A new request, a cancellation or the stop wakes the
    /// engine, so that the worker serves it without waiting for a call to end.
    fn advance_running_calls(&mut self) {
        let mut streams = Vec::new();
        let mut owners = Vec::new();
        for (model_index, worker_model) in self.models.iter().enumerate() {
            for (call_index, running_call) in worker_model.running_calls.iter().enumerate() {
                streams.push(&running_call.stream);
                owners.push((model_index, call_index));
            }
        }
        if streams.is_empty() {
            return;
        }
        if let Some(i) = engine::run_streams(&streams) {
            let (model_index, call_index) = owners[i];
            let shared = self.shared.clone();
            let worker_model = &mut self.models[model_index];
            let ended_call = worker_model.running_calls.remove(call_index);
            worker_model.finish_call(ended_call, &shared, &mut self.states);
        }
    }

    /// Finishes the running calls of every model. It advances them until none
    /// runs.
    fn finish_running_calls(&mut self) {
        while self
            .models
            .iter()
            .any(|worker_model| !worker_model.running_calls.is_empty())
        {
            self.advance_running_calls();
            self.publish_status_if_wanted();
        }
    }

    /// Runs the worker's loop until the stop. Each turn gives every model one
    /// step. A step admits requests, starts a prefill call and starts decoding
    /// calls. At the stop, every request ends with FinishReason::ServerStop.
    fn run_until_stop(&mut self) {
        let shared = self.shared.clone();
        while !engine::is_stopping() {
            self.publish_status_if_wanted();
            let mut queue = lock(&shared.queue);
            let idle = |queue: &Queue, models: &[WorkerModel]| {
                queue.waiting.iter().all(|jobs| jobs.is_empty())
                    && models.iter().all(|worker_model| {
                        worker_model.generations.is_empty() && worker_model.admitted.is_empty()
                    })
            };
            while idle(&queue, &self.models) && !engine::is_stopping() {
                if shared.status_wanted.load(Ordering::Acquire) {
                    drop(queue);
                    self.publish_status_if_wanted();
                    queue = lock(&shared.queue);
                    continue;
                }
                queue = shared
                    .worker_wakeup
                    .wait(queue)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            let mut admitted_by_model: Vec<Vec<Job>> = Vec::new();
            for worker_model in &self.models {
                let admitted: Vec<Job> = if engine::is_stopping() {
                    Vec::new()
                } else {
                    queue.waiting[worker_model.model_index].drain(..).collect()
                };
                if !admitted.is_empty() {
                    engine::log(&format!(
                        "{}: admitted={}, running={}",
                        worker_model.settings.id,
                        admitted.len(),
                        worker_model.generations.len()
                    ));
                }
                admitted_by_model.push(admitted);
            }
            drop(queue);
            for (worker_model, admitted) in self.models.iter_mut().zip(admitted_by_model) {
                worker_model.run_step(&shared, admitted, &mut self.states);
                worker_model
                    .model
                    .set_request_running(!worker_model.generations.is_empty());
            }
            self.advance_running_calls();
            self.watch_rounds();
            self.publish_status_if_wanted();
        }
        self.finish_running_calls();
        for worker_model in &mut self.models {
            for generation in std::mem::take(&mut worker_model.generations) {
                worker_model.end_generation(&shared, generation, true, &mut self.states);
            }
        }
        let mut waiting: Vec<Job> = {
            let mut queue = lock(&shared.queue);
            queue.closed = true;
            queue
                .waiting
                .iter_mut()
                .flat_map(|jobs| jobs.drain(..))
                .collect()
        };
        for worker_model in &mut self.models {
            waiting.extend(worker_model.admitted.drain(..));
        }
        for job in waiting {
            end_before_prefill(job, FinishReason::ServerStop);
        }
        for worker_model in &mut self.models {
            worker_model.update_status(&shared, "idle");
        }
    }
}

/// Refuses `job` with `message`. It logs the refusal and sends it as the last
/// event.
fn refuse(job: Job, message: &str) {
    engine::log(&format!("req {}: refused: {message}", job.id));
    job.send_last_event(Event::Refused(message.into()));
}

/// The refusal of a request whose new state does not fit in the memory.
const STATE_MEMORY_SHORT: &str = "the conversation's attention state does not fit: memory is short";

/// The refusal of a prompt of `tokens` tokens, too many for a context of `ctx`.
fn format_prompt_too_long(tokens: usize, ctx: usize) -> String {
    format!("the prompt has {tokens} tokens; the context holds {ctx}")
}

/// Ends `job` before its prefill with `reason`. It logs the end and sends Done
/// as the last event.
fn end_before_prefill(job: Job, reason: FinishReason) {
    engine::log(&format!(
        "req {}: ended before prefill ({}), waited {:.3} s",
        job.id,
        if reason == FinishReason::Cancelled {
            "cancelled"
        } else {
            "shutdown"
        },
        job.queued_at.elapsed().as_secs_f64()
    ));
    job.send_last_event(Event::Done { reason });
}

impl WorkerModel {
    /// Lists the ways `prompt` can resume from this model's states, the
    /// cheapest first. The last way is a cold start.
    fn list_resume_plans(&self, prompt: &[i32], states: &States) -> Vec<Plan> {
        let model = self.model.clone();
        let cost = move |position: usize| model.replay_rows(position).unwrap_or(0);
        let limits = PlanLimits {
            rewind_margin: self.model.rewind_margin(),
            // Any position tells whether the model rebuilds its rolling rows
            // from the rows it keeps.
            can_replay: self.model.replay_rows(1).is_some(),
            replay_cost: &cost,
        };
        states.cache.list_plans(self.model_index, prompt, limits)
    }

    /// Counts the bytes that a prompt of `prompt_tokens` tokens adds to the
    /// state cache on `plan`. It counts the state's growth to the prompt and
    /// its first token. It also counts the snapshots that the plan takes.
    fn count_plan_bytes(&self, plan: &Plan, prompt_tokens: usize, states: &States) -> u64 {
        let growth = match *plan {
            Plan::InPlace { state, .. } => self
                .model
                .state_bytes(Some(states.handles[&state].engine_state), prompt_tokens + 1)
                .saturating_sub(states.cache.entries[&state].bytes),
            _ => self.model.state_bytes(None, prompt_tokens + 1),
        };
        growth
            + plan.count_snapshot_bytes(
                prompt_tokens,
                self.model.snapshot_bytes(false),
                self.model.snapshot_bytes(true),
            )
    }

    /// Runs one step of the model. Admitted requests start, one at a time after
    /// the prefills before them. One prefill call starts. The generations that
    /// are done end. The others start decoding calls.
    fn run_step(&mut self, shared: &Shared, admitted: Vec<Job>, states: &mut States) {
        self.admitted.extend(admitted);
        // A request plans its prefill only after the prefills before it have
        // ended. It can then resume from their states. So one request at a time
        // starts its prefill.
        while !self
            .generations
            .iter()
            .any(|generation| !generation.is_decoding())
        {
            let Some(job) = self.admitted.pop_front() else {
                break;
            };
            if engine::is_stopping() {
                end_before_prefill(job, FinishReason::ServerStop);
            } else if job.flags.cancelled.load(Ordering::SeqCst) {
                engine::log(&format!("req {}: client gone while waiting", job.id));
                end_before_prefill(job, FinishReason::Cancelled);
            } else {
                let was_loaded = self.model.is_loaded();
                if !was_loaded {
                    self.update_status(shared, "loading");
                }
                // A running call uses the model's buffers. A load may resize
                // them. So the load runs only while no call runs.
                if self.running_calls.is_empty()
                    && let Err(error) = load_model(shared, &self.model)
                {
                    if engine::is_stopping() {
                        // The stop cut the load short.
                        end_before_prefill(job, FinishReason::ServerStop);
                        continue;
                    }
                    lock(&shared.models[self.model_index].status).load_error = Some(error.clone());
                    engine::log(&format!("{}: {error}", self.settings.id));
                    job.send_last_event(Event::Failed(error));
                    continue;
                }
                lock(&shared.models[self.model_index].status).load_error = None;
                self.experts_per_row = self.model.read_stats().experts_per_row.max(0) as usize;
                if !was_loaded && self.model.is_loaded() {
                    engine::log(&format!(
                        "{}: loaded; shares the {:.1} GiB state cache limit",
                        self.settings.id, shared.budgets.state_cache_gb
                    ));
                }
                self.start_request(shared, job, states);
            }
        }
        if engine::is_stopping() {
            return;
        }
        self.start_prefill_calls(states);
        let ctx = self.ctx;
        let mut i = 0;
        while i < self.generations.len() {
            if !self.generations[i].in_call
                && self.generations[i].is_decoding()
                && Self::check_ended(&self.model, &mut self.generations[i], ctx)
            {
                let generation = self.generations.swap_remove(i);
                self.end_generation(shared, generation, false, states);
            } else if !self.generations[i].in_call
                && !self.generations[i].is_decoding()
                && self.generations[i]
                    .job
                    .flags
                    .cancelled
                    .load(Ordering::SeqCst)
            {
                let mut generation = self.generations.swap_remove(i);
                generation.cancelled = true;
                self.end_prefill_early(&mut generation, states);
                self.end_generation(shared, generation, false, states);
            } else {
                i += 1;
            }
        }
        let decoding_count = self
            .generations
            .iter()
            .filter(|generation| generation.is_decoding())
            .count();
        if decoding_count > 1 && decoding_count != self.logged_decoding_count {
            engine::log(&format!(
                "{}: {decoding_count} requests decoding together",
                self.settings.id
            ));
        }
        self.logged_decoding_count = decoding_count;
        if self.generations.is_empty() {
            self.idle_prefill_streams.clear();
            self.idle_decode_streams.clear();
            self.update_status(shared, "idle");
            return;
        }
        if decoding_count > 0 {
            self.start_decode_calls(states);
        }
        let progress_log_s = shared.budgets.progress_log_s;
        for generation in &mut self.generations {
            if !generation.unsent_tokens.is_empty() {
                let _ = generation
                    .job
                    .events
                    .send(Event::Tokens(std::mem::take(&mut generation.unsent_tokens)));
            }
            if generation.is_decoding()
                && progress_log_s > 0.0
                && generation.progress_logged_at.elapsed().as_secs_f64() >= progress_log_s
            {
                generation.progress_logged_at = Instant::now();
                engine::log(&format!(
                    "req {}: {} tokens, {:.1} tokens/s",
                    generation.job.id,
                    generation.generated,
                    generation.generated as f64
                        / (generation.progress_logged_at - generation.decode_started_at)
                            .as_secs_f64()
                ));
            }
        }
        let phase = if self
            .generations
            .iter()
            .any(|generation| !generation.is_decoding())
        {
            "prefill"
        } else {
            "decode"
        };
        self.update_status(shared, phase);
    }

    /// Updates the model's status: its phase, its counts and the request it
    /// shows.
    fn update_status(&mut self, shared: &Shared, phase: &'static str) {
        let mut status = lock(&shared.models[self.model_index].status);
        if status.phase != phase || status.phase_since.is_none() {
            status.phase_since = Some(Instant::now());
        }
        status.phase = phase;
        status.decoding_count = self
            .generations
            .iter()
            .filter(|generation| generation.is_decoding())
            .count();
        let first = self
            .generations
            .iter()
            .find(|generation| !generation.is_decoding())
            .or(self.generations.first());
        (
            status.shown_request_id,
            status.shown_prompt_tokens,
            status.shown_reused_tokens,
            status.shown_generated_tokens,
        ) = first.map_or((0, 0, 0, 0), |generation| {
            (
                generation.job.id,
                generation.prompt_tokens,
                generation.reused_tokens,
                generation.generated,
            )
        });
    }

    /// Starts an admitted request. It checks the request's media and prompt. It
    /// carries out the cheapest plan that fits in the state cache, and the
    /// prefill follows. For the same prompt again, the first token comes from
    /// the prompt-end snapshot instead. It runs after the prefills before the
    /// request have ended. It refuses a request that does not fit.
    fn start_request(&mut self, shared: &Shared, job: Job, states: &mut States) {
        // A reply has at least one token, so a prompt has at most ctx - 1.
        let most_prompt_tokens = self.ctx - 1;
        let mut prompt = job.request.prompt.clone();
        let mut media = None;
        if !job.request.media.is_empty() {
            match self.model.open_media(&job.request.media) {
                Err(message) => return refuse(job, &message),
                Ok(mut opened) => match opened.expand(&prompt, most_prompt_tokens) {
                    Ok(expanded) => {
                        prompt = expanded;
                        media = Some(opened);
                    }
                    Err(ExpandError::TooLong(tokens)) => {
                        return refuse(job, &format_prompt_too_long(tokens, self.ctx));
                    }
                    Err(ExpandError::PlacesAndMediaDiffer) => {
                        let message = format!(
                            "the prompt's image places and its {} media differ",
                            job.request.media.len()
                        );
                        return refuse(job, &message);
                    }
                },
            }
        }
        let prompt_tokens = prompt.len();
        if prompt_tokens > most_prompt_tokens {
            return refuse(job, &format_prompt_too_long(prompt_tokens, self.ctx));
        }
        let sampling = job.request.sampling;
        let Some(sampler) = Sampler::new(
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
            sampling.seed,
        ) else {
            return refuse(job, "invalid sampling settings");
        };
        // A reply has at least one token.
        let max_new_tokens = (job.request.max_tokens.max(1) as usize).min(self.ctx - prompt_tokens);
        // Take the cheapest plan that fits. Each plan keeps the state it
        // reuses. The last plan is a cold start, and it keeps nothing. With it,
        // the cache can evict a state that does not fit beside the new one.
        let mut need = 0;
        let mut chosen = None;
        for plan in self.list_resume_plans(&prompt, states) {
            need = self.count_plan_bytes(&plan, prompt_tokens, states);
            let keep: Vec<state_cache::StateId> = plan.find_source_state().into_iter().collect();
            if let Some(eviction) =
                states
                    .cache
                    .find_eviction(need, !matches!(plan, Plan::InPlace { .. }), &keep)
            {
                chosen = Some((plan, eviction));
                break;
            }
        }
        let Some((plan, eviction)) = chosen else {
            let busy: u64 = states
                .cache
                .entries
                .values()
                .filter(|entry| entry.busy)
                .map(|entry| entry.bytes)
                .sum();
            // The state's bytes count the prompt and its first token.
            let message = format!(
                "the conversation's attention state needs {:.1} GiB ({:.1} GiB with its snapshots); the state cache allows {:.1} GiB, and the requests running take {:.1} GiB of it",
                self.model.state_bytes(None, prompt_tokens + 1) as f64 / GIB,
                need as f64 / GIB,
                states.cache.byte_limit as f64 / GIB,
                busy as f64 / GIB
            );
            return refuse(job, &message);
        };
        let origin = states.cache.find_origin(&plan, prompt_tokens);
        let capture_position = plan.find_branch_snapshot_pos(prompt_tokens);
        let wait = job.queued_at.elapsed().as_secs_f64();
        let evicted = states.cache.evict(&eviction);
        states.free_evicted(evicted);
        let reused_tokens = plan.count_reused_tokens();
        // The plan's state: the one that it continues in place, or a new one. A
        // request whose new state does not fit is refused.
        let planned_state = match plan {
            Plan::InPlace { state, .. } => state,
            _ => match states.add_state(self.model_index, self.model.clone()) {
                Some(state_id) => state_id,
                None => return refuse(job, STATE_MEMORY_SHORT),
            },
        };
        let state_id = match plan {
            Plan::Cold => planned_state,
            Plan::InPlace { state, .. } => {
                states.cache.record_reuse(state);
                state
            }
            Plan::Snapshot {
                source, from: at, ..
            }
            | Plan::Replay { source, from: at } => {
                let state_id = planned_state;
                let engine_state = states.handles[&state_id].engine_state;
                self.model
                    .state_copy(engine_state, states.handles[&source].engine_state, at);
                states.cache.record_reuse(source);
                states.handles.get_mut(&source).expect("state").last_used_at = Instant::now();
                if let Plan::Snapshot { snapshot_pos, .. } = plan {
                    self.model
                        .snapshot_restore(engine_state, states.get_snapshot(source, snapshot_pos));
                }
                state_id
            }
            Plan::Load { .. } => {
                unreachable!("a saved state's plan needs the store of saved states")
            }
        };
        // The state's bytes count the prompt and its first token.
        let bytes = self.model.state_bytes(
            Some(states.handles[&state_id].engine_state),
            prompt_tokens + 1,
        );
        let dropped = states.cache.start_request(
            state_id,
            &prompt,
            reused_tokens,
            bytes,
            state_cache::count_stale_rows_at(origin, reused_tokens),
        );
        let mut reserved = Vec::new();
        if reused_tokens < prompt_tokens {
            reserved.push(Snapshot {
                pos: prompt_tokens,
                kind: SnapshotKind::PromptEnd,
                bytes: self.model.snapshot_bytes(true),
                stale_rows: state_cache::count_stale_rows_at(origin, prompt_tokens),
            });
        }
        if let Some(pos) = capture_position {
            reserved.push(Snapshot {
                pos,
                kind: SnapshotKind::Branch,
                bytes: self.model.snapshot_bytes(false),
                stale_rows: state_cache::count_stale_rows_at(origin, pos),
            });
        }
        states.cache.set_reserved_snapshots(state_id, reserved);
        for snapshot in dropped {
            states.free_snapshot(state_id, snapshot.pos);
        }
        states
            .handles
            .get_mut(&state_id)
            .expect("state")
            .last_used_at = Instant::now();
        states.publish_states(shared);
        let number = states.cache.entries[&state_id].number;
        let plan_kind = plan.describe_for_log(prompt_tokens);
        let thinking = self.think_start_token == Some(prompt[prompt_tokens - 1]);
        engine::log(&format!(
            "req {}: {} {prompt_tokens} prompt tokens, session {number} ({plan_kind}), {reused_tokens} reused; thinking {}; at most {max_new_tokens} new tokens, {}; waited {wait:.1} s",
            job.id,
            self.settings.id,
            if thinking { "on" } else { "off" },
            if sampling.temperature > 0.0 {
                "sampled"
            } else {
                "greedy"
            }
        ));
        let replays = plan.starts_with_replay();
        if let Some(opened) = media.as_mut() {
            // Encode the media that the prefill reaches. A replay computes
            // again from its first position. That position lies before the
            // branch point.
            let first = if replays {
                self.model.replay_start(reused_tokens)
            } else {
                reused_tokens
            };
            let encode_started_at = Instant::now();
            let encoded = match opened.encode(first) {
                Ok(count) => count,
                Err(error) => {
                    let snapshots = states.cache.remove_state(state_id);
                    states.free_snapshots_and_states(Vec::new(), vec![(state_id, snapshots)]);
                    states.publish_states(shared);
                    job.send_last_event(Event::Failed(error));
                    return;
                }
            };
            if encoded > 0 {
                engine::log(&format!(
                    "req {}: {encoded} of {} media encoded in {:.2} s",
                    job.id,
                    job.request.media.len(),
                    encode_started_at.elapsed().as_secs_f64()
                ));
            }
        }
        let now = Instant::now();
        let mut generation = Generation {
            job,
            sampler,
            state_id,
            prompt_tokens,
            reused_tokens,
            prefill: Some(Prefill {
                prompt,
                next_position: reused_tokens,
                capture_position,
                origin,
                replays,
                media,
                started_at: now,
                expert_counts_at_start: self.model.read_stats().cache,
                must_run_alone: false,
            }),
            pos: prompt_tokens,
            generated: 0,
            reasoning_tokens: 0,
            thinking,
            reached_stop_token: false,
            cancelled: false,
            next_token: 0,
            unsent_tokens: Vec::new(),
            max_new_tokens,
            decode_started_at: now,
            progress_logged_at: now,
            expert_counts_at_decode_start: CacheCounts::default(),
            in_call: false,
            turn: 0,
            measured_token_at: None,
        };
        if reused_tokens == prompt_tokens {
            // The same prompt again: its prompt-end snapshot holds its logits.
            let source = plan.find_source_state().unwrap_or(state_id);
            let snapshot = states.get_snapshot(source, prompt_tokens);
            if source == state_id {
                self.model
                    .snapshot_restore(states.handles[&state_id].engine_state, snapshot);
            }
            let prompt = generation
                .prefill
                .as_ref()
                .map(|prefill| prefill.prompt.clone())
                .unwrap_or_default();
            self.model.set_history(
                states.handles[&state_id].engine_state,
                &prompt,
                prompt_tokens,
            );
            let logits = self
                .model
                .snapshot_logits(snapshot)
                .expect("a prompt-end snapshot holds its logits");
            self.start_decoding(
                shared,
                &mut generation,
                logits,
                PrefillInfo::default(),
                states,
            );
        }
        self.generations.push(generation);
    }

    /// Finds the index in `generations` of the request with `request_id`.
    /// Panics when there is none.
    fn find_generation_index(&self, request_id: u64) -> usize {
        self.generations
            .iter()
            .position(|generation| generation.job.id == request_id)
            .expect("a running call's request")
    }

    /// Starts a prefill call for each request still prefilling with no call
    /// running. The calls run beside each other and beside the decoding calls.
    fn start_prefill_calls(&mut self, states: &mut States) {
        for i in 0..self.generations.len() {
            self.start_prefill_call(i, states);
        }
    }

    /// Starts a prefill call for request `i` if it is still prefilling with no
    /// call running. A request that must run alone waits while another prefill
    /// call runs. A model that prefills a whole prompt in one call gets the
    /// rest of the prompt. Otherwise the call takes at most the model's prefill
    /// rows. Such a call also ends at the capture position. For a model that
    /// resumes only from snapshots, it also ends at the next multiple of the
    /// snapshot interval. Such a model takes a snapshot there, while the state
    /// cache has room. A later prompt may share a long prefix with this prompt
    /// but part from it before its end. That prompt then resumes from the last
    /// snapshot before the point where they part. It computes the positions
    /// after that snapshot again.
    fn start_prefill_call(&mut self, i: usize, states: &mut States) {
        let generation = &self.generations[i];
        let Some(prefill) = generation.prefill.as_ref() else {
            return;
        };
        if generation.in_call
            || (prefill.must_run_alone
                && self
                    .running_calls
                    .iter()
                    .any(|running_call| matches!(running_call.call, Call::Prefill { .. })))
        {
            return;
        }
        let prefill_rows = self.model.prefill_rows();
        // Any position tells whether the model rebuilds its rolling rows from
        // the rows it keeps.
        let can_replay = self.model.replay_rows(1).is_some();
        let (state_id, prompt_tokens) = (
            self.generations[i].state_id,
            self.generations[i].prompt_tokens,
        );
        let engine_state = states.handles[&state_id].engine_state;
        let prefill = self.generations[i].prefill.as_ref().expect("a prefill");
        let (next_position, capture_position) = (prefill.next_position, prefill.capture_position);
        let end = match prefill_rows {
            None => prompt_tokens,
            Some(max_rows) => {
                let mut end = (next_position + max_rows).min(prompt_tokens);
                if !can_replay {
                    // The next multiple of the snapshot interval.
                    let interval = states.snapshot_interval;
                    end = end.min((next_position / interval + 1) * interval);
                }
                if let Some(at) = capture_position
                    && next_position < at
                    && at < end
                {
                    end = at;
                }
                end
            }
        };
        if self.generations[i]
            .job
            .flags
            .cancelled
            .load(Ordering::SeqCst)
        {
            // The next step ends the cancelled request.
            return;
        }
        let replays = prefill.replays && next_position == self.generations[i].reused_tokens;
        // In one call, the engine takes the capture snapshot during the
        // prefill. In several calls, the worker takes it where a call ends.
        let call_capture = if prefill_rows.is_none() {
            capture_position
        } else {
            None
        };
        let stream = self
            .idle_prefill_streams
            .pop()
            .unwrap_or_else(|| Stream::new(&self.model));
        stream.prefill(
            engine_state,
            &prefill.prompt[..end],
            next_position,
            replays,
            call_capture,
            prefill.media.as_ref(),
        );
        let request_id = self.generations[i].job.id;
        self.generations[i].in_call = true;
        self.running_calls.push(RunningCall {
            stream,
            call: Call::Prefill {
                request_id,
                end,
                call_capture,
            },
        });
    }

    /// Finishes a prefill call that ended. It keeps the request's new
    /// snapshots. At the prompt's end, the request starts decoding.
    fn finish_prefill_call(
        &mut self,
        stream: Stream,
        request_id: u64,
        end: usize,
        call_capture: Option<usize>,
        shared: &Shared,
        states: &mut States,
    ) {
        let i = self.find_generation_index(request_id);
        self.generations[i].in_call = false;
        let prefill_rows = self.model.prefill_rows();
        // Any position tells whether the model rebuilds its rolling rows from
        // the rows it keeps.
        let can_replay = self.model.replay_rows(1).is_some();
        let (state_id, prompt_tokens) = (
            self.generations[i].state_id,
            self.generations[i].prompt_tokens,
        );
        let engine_state = states.handles[&state_id].engine_state;
        let Some(logits) = stream.logits() else {
            if engine::is_stopping() {
                // The stop cut the prefill short. The stop also ends the
                // request.
                return;
            }
            // Its working memory did not fit beside another prefill call. It
            // runs again alone.
            if self
                .running_calls
                .iter()
                .any(|running_call| matches!(running_call.call, Call::Prefill { .. }))
            {
                self.generations[i]
                    .prefill
                    .as_mut()
                    .expect("a prefill")
                    .must_run_alone = true;
                self.idle_prefill_streams.push(stream);
                return;
            }
            let generation = self.generations.remove(i);
            let snapshots = states.cache.remove_state(generation.state_id);
            states.free_snapshots_and_states(Vec::new(), vec![(generation.state_id, snapshots)]);
            states.publish_states(shared);
            generation.job.send_last_event(Event::Failed(
                "the prefill's working memory does not fit: memory is short".into(),
            ));
            return;
        };
        let (captured, info) = (stream.captured_snapshot(), stream.prefill_info());
        let prefill = self.generations[i].prefill.as_ref().expect("a prefill");
        let (origin, capture_position) = (prefill.origin, prefill.capture_position);
        states.cache.add_prefilled_tokens(
            state_id,
            &prefill.prompt[..end],
            state_cache::count_stale_rows_at(origin, end),
        );
        if let Some(at) = call_capture
            && let Some(captured) = captured
        {
            states.insert_snapshot(state_id, at, captured);
            let stale = state_cache::count_stale_rows_at(origin, at);
            states.cache.add_snapshot(
                state_id,
                Snapshot {
                    pos: at,
                    kind: SnapshotKind::Branch,
                    bytes: self.model.snapshot_bytes(false),
                    stale_rows: stale,
                },
            );
        }
        let last = end == prompt_tokens;
        if prefill_rows.is_some() && !last {
            let at_interval = !can_replay
                && end.is_multiple_of(states.snapshot_interval)
                && states.cache.find_snapshot(state_id, end).is_none();
            if capture_position == Some(end) || at_interval {
                let bytes = self.model.snapshot_bytes(false);
                let fits = capture_position == Some(end)
                    || states
                        .cache
                        .find_eviction(bytes, false, &[state_id])
                        .is_some_and(|eviction| {
                            let evicted = states.cache.evict(&eviction);
                            states.free_evicted(evicted);
                            true
                        });
                if fits {
                    let snapshot = self.model.snapshot_take(engine_state, end, None);
                    states.insert_snapshot(state_id, end, snapshot);
                    let stale = state_cache::count_stale_rows_at(origin, end);
                    states.cache.add_snapshot(
                        state_id,
                        Snapshot {
                            pos: end,
                            kind: SnapshotKind::Branch,
                            bytes,
                            stale_rows: stale,
                        },
                    );
                }
            }
        }
        if !last {
            self.generations[i]
                .prefill
                .as_mut()
                .expect("a prefill")
                .next_position = end;
            self.idle_prefill_streams.push(stream);
            return;
        }
        let snapshot = self
            .model
            .snapshot_take(engine_state, prompt_tokens, Some(logits));
        states.insert_snapshot(state_id, prompt_tokens, snapshot);
        let stale = state_cache::count_stale_rows_at(origin, prompt_tokens);
        states.cache.add_snapshot(
            state_id,
            Snapshot {
                pos: prompt_tokens,
                kind: SnapshotKind::PromptEnd,
                bytes: self.model.snapshot_bytes(true),
                stale_rows: stale,
            },
        );
        let mut generation = self.generations.remove(i);
        self.start_decoding(shared, &mut generation, logits, info, states);
        self.generations.insert(i, generation);
        // The stream owns the logits until here. Dropping it frees the prompt's
        // large buffers.
        drop(stream);
    }

    /// Finishes a call that ended. Its requests then go on.
    fn finish_call(&mut self, running_call: RunningCall, shared: &Shared, states: &mut States) {
        let RunningCall { stream, call } = running_call;
        match call {
            Call::Prefill {
                request_id,
                end,
                call_capture,
            } => self.finish_prefill_call(stream, request_id, end, call_capture, shared, states),
            Call::DecodeRows {
                request_ids,
                tokens,
            } => self.finish_decode_rows_call(stream, request_ids, tokens, states),
        }
    }

    /// Ends the request's prefill before the prompt's end. It cuts the state's
    /// tokens to the positions that the prefill ran.
    fn end_prefill_early(&mut self, generation: &mut Generation, states: &mut States) {
        if let Some(prefill) = generation.prefill.take()
            && let Some(entry) = states.cache.entries.get_mut(&generation.state_id)
        {
            entry.tokens.truncate(prefill.next_position);
            generation.pos = prefill.next_position;
        }
    }

    /// Starts decoding after the prefill ended with `logits`. It logs the
    /// prefill and sends DecodingStarted. It samples the first token from
    /// `logits`.
    fn start_decoding(
        &mut self,
        shared: &Shared,
        generation: &mut Generation,
        logits: Logits,
        info: PrefillInfo,
        states: &mut States,
    ) {
        states.cache.clear_reserved_snapshots(generation.state_id);
        let prefill = generation.prefill.take().expect("a prefill");
        let prompt_tokens = generation.prompt_tokens;
        let reused_tokens = generation.reused_tokens;
        let now = Instant::now();
        let prefill_seconds = (now - prefill.started_at).as_secs_f64();
        let stats = self.model.read_stats();
        let read = (stats.cache.bytes - prefill.expert_counts_at_start.bytes) as f64 / GB;
        if reused_tokens < prompt_tokens {
            let replayed = if prefill.replays {
                format!(
                    "; {} replay at {reused_tokens}: up to {} earlier rows computed again",
                    if self
                        .settings
                        .options
                        .deepseek()
                        .is_some_and(|deepseek| deepseek.ced > 0)
                    {
                        "a bounded"
                    } else {
                        "an exact"
                    },
                    (info.tokens as usize).saturating_sub(prompt_tokens - reused_tokens)
                )
            } else {
                String::new()
            };
            // 1e-9 keeps the rate finite when no time passed. Why 1e-9 exactly
            // is unknown.
            engine::log(&format!(
                "req {}: prefill of {} tokens in {prefill_seconds:.2} s ({:.0} tokens/s), {read:.2} GB of experts read{replayed}",
                generation.job.id,
                prompt_tokens - reused_tokens,
                (prompt_tokens - reused_tokens) as f64 / prefill_seconds.max(1e-9)
            ));
            engine::log(&format!(
                "req {}: prefill plan: {} chunks, {} encoder tokens, {} decoder tokens, ced_applied={}, prune_applied={}, prefix={} (full)",
                generation.job.id,
                info.chunks,
                info.tokens,
                info.decoder_tokens,
                info.ced_applied,
                info.prune_applied,
                info.prefix_tokens
            ));
        }
        let _ = generation.job.events.send(Event::DecodingStarted {
            prompt_tokens,
            reused_tokens,
        });
        {
            let mut status = lock(&shared.models[self.model_index].status);
            status.last_prefill = info;
        }
        drop(prefill.media);
        generation.next_token = self.model.sample(logits, &mut generation.sampler);
        generation.decode_started_at = now;
        generation.progress_logged_at = now;
        generation.turn = self.turns.take(1).start;
        generation.expert_counts_at_decode_start = stats.cache;
        states.publish_states(shared);
    }

    /// Keeps `token`. The token ran at the generation's position. It adds the
    /// token to the state's tokens and to the tokens to send. A stop token ends
    /// the reply and is not sent.
    fn keep_token(
        model: &Model,
        cache: &mut Cache,
        generation: &mut Generation,
        token: i32,
        think_end: Option<i32>,
    ) {
        if generation.thinking {
            generation.reasoning_tokens += 1;
            generation.thinking = Some(token) != think_end;
        }
        cache.add_decoded_token(generation.state_id, token);
        generation.pos += 1;
        generation.generated += 1;
        // A kept stop token ends the reply.
        generation.reached_stop_token = model.is_stop_token(token);
        if !generation.reached_stop_token {
            generation.unsent_tokens.push(token as u32);
        }
    }

    /// Checks whether the generation ends before its next pass. It ends at a
    /// stop token, at its token limit, at the context's end, or when its client
    /// cancelled it. Returns true when it ends.
    fn check_ended(model: &Model, generation: &mut Generation, ctx: usize) -> bool {
        if generation.job.flags.cancelled.load(Ordering::SeqCst) {
            generation.cancelled = true;
        }
        if generation.reached_stop_token || generation.cancelled {
            return true;
        }
        if model.is_stop_token(generation.next_token) {
            generation.generated += 1;
            generation.reached_stop_token = true;
            return true;
        }
        if generation.generated + 1 >= generation.max_new_tokens || generation.pos + 1 >= ctx {
            // The waiting token is the reply's last. It goes out without a
            // pass. The state then ends at the last token that ran. A model
            // that cannot rewind needs that for the next turn to continue the
            // state.
            if generation.thinking {
                generation.reasoning_tokens += 1;
            }
            generation.unsent_tokens.push(generation.next_token as u32);
            generation.generated += 1;
            return true;
        }
        false
    }

    /// Ends the generation. It sends its last tokens and its Done event. It
    /// keeps a snapshot at the generation's end when the state cache has room.
    /// It logs the request's counts and adds them to the model's totals.
    fn end_generation(
        &mut self,
        shared: &Shared,
        mut generation: Generation,
        at_stop: bool,
        states: &mut States,
    ) {
        if !generation.unsent_tokens.is_empty() {
            let _ = generation
                .job
                .events
                .send(Event::Tokens(std::mem::take(&mut generation.unsent_tokens)));
        }
        let reason = if generation.cancelled {
            FinishReason::Cancelled
        } else if at_stop {
            FinishReason::ServerStop
        } else if generation.reached_stop_token {
            FinishReason::StopToken
        } else {
            FinishReason::Length
        };
        let reason_text = match reason {
            FinishReason::StopToken => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "client gone",
            FinishReason::ServerStop => "stopped by the server's stop",
        };
        generation.job.send_last_event(Event::Done { reason });
        self.end_prefill_early(&mut generation, states);
        // Take a snapshot at the generation's end when the cache keeps it. A
        // later fork there resumes from it.
        let stale = states.cache.entries[&generation.state_id].stale_rows;
        let end_snapshot = states
            .cache
            .find_snapshot(generation.state_id, generation.pos)
            .is_none()
            .then(|| Snapshot {
                pos: generation.pos,
                kind: SnapshotKind::GenerationEnd,
                bytes: self.model.snapshot_bytes(false),
                stale_rows: stale,
            });
        let (take, evicted) = states.cache.finish_request(
            generation.state_id,
            self.model.state_bytes(
                Some(states.handles[&generation.state_id].engine_state),
                generation.pos,
            ),
            end_snapshot,
        );
        if take {
            let snapshot = self.model.snapshot_take(
                states.handles[&generation.state_id].engine_state,
                generation.pos,
                None,
            );
            states.insert_snapshot(generation.state_id, generation.pos, snapshot);
        }
        states.free_evicted(evicted);
        states
            .handles
            .get_mut(&generation.state_id)
            .expect("state")
            .last_used_at = Instant::now();
        let stats = self.model.read_stats();
        {
            let mut status = lock(&shared.models[self.model_index].status);
            status.stats = stats;
            status.loaded = self.model.is_loaded();
        }
        let (expert_counts, decode_seconds) = (
            stats.cache,
            generation.decode_started_at.elapsed().as_secs_f64(),
        );
        let hits = expert_counts.hits - generation.expert_counts_at_decode_start.hits
            + expert_counts.prefetch_hits
            - generation.expert_counts_at_decode_start.prefetch_hits;
        let misses = expert_counts.misses - generation.expert_counts_at_decode_start.misses;
        let experts = if self.experts_per_row > 0 {
            // 100 turns the share of hits into a percent.
            format!(
                "; experts {:.1}% hit, {:.2} GB read",
                100.0 * hits as f64 / ((hits + misses).max(1)) as f64,
                (expert_counts.bytes - generation.expert_counts_at_decode_start.bytes) as f64 / GB
            )
        } else {
            String::new()
        };
        // 1e-9 keeps the rate finite when no time passed. Why 1e-9 exactly is
        // unknown.
        engine::log(&format!(
            "req {}: {reason_text}: {} tokens ({} reasoning) in {decode_seconds:.1} s, {:.1} tokens/s{experts}; states {} sessions, {:.2} GiB",
            generation.job.id,
            generation.generated,
            generation.reasoning_tokens,
            generation.generated as f64 / decode_seconds.max(1e-9),
            states.cache.entries.len(),
            states.cache.count_bytes() as f64 / GIB,
        ));
        let sampling = generation.job.request.sampling;
        engine::log(&format!(
            "req {}: sampling temperature {}, top_p {}, top_k {}",
            generation.job.id, sampling.temperature, sampling.top_p, sampling.top_k,
        ));
        let mut status = lock(&shared.models[self.model_index].status);
        status.total_requests += 1;
        status.total_prompt_tokens += generation.prompt_tokens as u64;
        status.total_generated_tokens += generation.generated as u64;
        drop(status);
        states.publish_states(shared);
    }

    /// Ends the requests of a decoding pass that could not load its experts.
    /// The stop may have ended their reads. Or memory stayed short after idle
    /// memory was freed. During the stop, the requests end as every request
    /// does at the stop. Otherwise the requests end with an error. The pass may
    /// have changed their states in part. So the states are freed.
    fn end_failed_pass(&mut self, generation_indices: &[usize], states: &mut States) {
        if engine::is_stopping() {
            return;
        }
        let mut indices = generation_indices.to_vec();
        indices.sort_unstable_by(|left, right| right.cmp(left));
        for i in indices {
            let generation = self.generations.remove(i);
            let snapshots = states.cache.remove_state(generation.state_id);
            states.free_snapshots_and_states(Vec::new(), vec![(generation.state_id, snapshots)]);
            engine::log(&format!(
                "req {}: a pass could not load its experts: memory is short",
                generation.job.id
            ));
            generation.job.send_last_event(Event::Failed(
                "a pass could not load the model's experts: memory is short".into(),
            ));
        }
    }

    /// Takes an idle decoding stream, or makes a new one.
    fn take_decode_stream(&mut self) -> Stream {
        self.idle_decode_streams
            .pop()
            .unwrap_or_else(|| Stream::new(&self.model))
    }

    /// Starts decoding calls for the decoding requests with no call running.
    /// Several requests run together, at most the model's rows per call. Each
    /// row then gives what it would give alone.
    fn start_decode_calls(&mut self, states: &mut States) {
        // The call chooser sets how many calls run at once and the rows of
        // each. A call starts as soon as its requests are free and the chooser
        // allows one more call. Groups of requests run independently. The GPU
        // work of one call fills the waits of the others.
        let max_rows_per_call =
            limit_rows_per_call(self.model.max_rows(), self.settings.rows_per_call);
        let decoding_count = self
            .generations
            .iter()
            .filter(|generation| generation.is_decoding())
            .count();
        if decoding_count >= 2 && self.round.is_none() {
            let was_split = self.call_chooser.is_split();
            let is_split = self
                .call_chooser
                .begin_round(decoding_count, max_rows_per_call);
            // A gap counts only when both of its tokens came in this
            // arrangement. Calls of the other arrangement that still run would
            // give tokens in the round.
            if is_split != was_split {
                self.forget_token_gaps();
            }
            let calls_run = self
                .running_calls
                .iter()
                .any(|running_call| !matches!(running_call.call, Call::Prefill { .. }));
            self.round = Some(Round {
                waiting: self
                    .generations
                    .iter()
                    .filter(|generation| generation.is_decoding())
                    .map(|generation| generation.job.id)
                    .collect(),
                decoding_count,
                is_split,
                is_steady: !(calls_run && is_split != was_split),
                gaps: 0,
                gap_seconds: 0.0,
            });
        }
        let (allowed, rows_per_call) = self
            .call_chooser
            .choose_calls(decoding_count, max_rows_per_call);
        let running = self
            .running_calls
            .iter()
            .filter(|running_call| !matches!(running_call.call, Call::Prefill { .. }))
            .count();
        if running >= allowed {
            return;
        }
        let mut decoding_indices: Vec<usize> = (0..self.generations.len())
            .filter(|&i| self.generations[i].is_decoding() && !self.generations[i].in_call)
            .collect();
        if decoding_indices.is_empty() {
            return;
        }
        decoding_indices.sort_by_key(|&i| self.generations[i].turn);
        if decoding_count == 1 {
            let index = decoding_indices[0];
            let state_id = self.generations[index].state_id;
            let engine_state = states.handles[&state_id].engine_state;
            let stream = self.take_decode_stream();
            let generation = &mut self.generations[index];
            generation.in_call = true;
            let request_id = generation.job.id;
            stream.decode_rows(
                &[engine_state],
                &[generation.next_token],
                &[generation.pos as i32],
            );
            let call = Call::DecodeRows {
                request_ids: vec![request_id],
                tokens: vec![generation.next_token],
            };
            self.running_calls.push(RunningCall { stream, call });
            return;
        }
        // Each group gets one call, while the chooser allows one more call.
        for group in decoding_indices
            .chunks(rows_per_call)
            .take(allowed - running)
        {
            let engine_states: Vec<StateHandle> = group
                .iter()
                .map(|&i| states.handles[&self.generations[i].state_id].engine_state)
                .collect();
            let tokens: Vec<i32> = group
                .iter()
                .map(|&i| self.generations[i].next_token)
                .collect();
            let pos: Vec<i32> = group
                .iter()
                .map(|&i| self.generations[i].pos as i32)
                .collect();
            let request_ids: Vec<u64> = group.iter().map(|&i| self.generations[i].job.id).collect();
            let stream = self.take_decode_stream();
            stream.decode_rows(&engine_states, &tokens, &pos);
            for &i in group {
                self.generations[i].in_call = true;
            }
            self.running_calls.push(RunningCall {
                stream,
                call: Call::DecodeRows {
                    request_ids,
                    tokens,
                },
            });
        }
    }

    /// Finishes a decoding call of one token per request. Each request keeps
    /// its token and samples the next one.
    fn finish_decode_rows_call(
        &mut self,
        stream: Stream,
        request_ids: Vec<u64>,
        tokens: Vec<i32>,
        states: &mut States,
    ) {
        let indices: Vec<usize> = request_ids
            .iter()
            .map(|&request_id| self.find_generation_index(request_id))
            .collect();
        let ended_at = Instant::now();
        let turns = self.turns.take(indices.len());
        for (&i, turn) in indices.iter().zip(turns) {
            self.generations[i].in_call = false;
            self.generations[i].turn = turn;
        }
        let Some(logits) = stream.logits() else {
            self.idle_decode_streams.push(stream);
            self.end_failed_pass(&indices, states);
            return;
        };
        let vocab = self.model.vocab_size();
        let model = self.model.clone();
        let think_end = self.think_end_token;
        for (row, &i) in indices.iter().enumerate() {
            let generation = &mut self.generations[i];
            Self::keep_token(
                &model,
                &mut states.cache,
                generation,
                tokens[row],
                think_end,
            );
            // Safety: the call returns one row of logits for each state. Each
            // row holds vocab_size floats.
            generation.next_token =
                model.sample(unsafe { logits.row(row, vocab) }, &mut generation.sampler);
        }
        self.idle_decode_streams.push(stream);
        self.count_round_tokens(&indices, ended_at);
    }

    /// Counts the new tokens that the requests at `indices` got at `ended_at`,
    /// and the gaps since their previous tokens. Once every request of the
    /// round has a token, a steady round goes to the call chooser, and the next
    /// calls begin the next round.
    fn count_round_tokens(&mut self, indices: &[usize], ended_at: Instant) {
        let Some(round) = self.round.as_mut() else {
            for &i in indices {
                self.generations[i].measured_token_at = None;
            }
            return;
        };
        for &i in indices {
            let generation = &mut self.generations[i];
            round.waiting.retain(|&id| id != generation.job.id);
            if !round.is_steady {
                generation.measured_token_at = None;
            } else if let Some(previous) = generation.measured_token_at.replace(ended_at) {
                round.gaps += 1;
                round.gap_seconds += ended_at.duration_since(previous).as_secs_f64();
            }
        }
        if round.waiting.is_empty() {
            if round.is_steady {
                self.call_chooser.record_round(
                    round.decoding_count,
                    round.is_split,
                    round.gaps,
                    round.gap_seconds,
                );
            }
            self.round = None;
        }
    }

    /// Forgets each request's last measured token, so that the call chooser
    /// measures no gap that spans a change of arrangement, a prompt, another
    /// model's calls or a change of requests.
    fn forget_token_gaps(&mut self) {
        for generation in &mut self.generations {
            generation.measured_token_at = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six requests take turns in one call of four rows at a time. Each call's
    /// rows become ready together. With turn numbers, every request runs in two
    /// calls out of three. Ordered by when they became ready, with ties by
    /// index, requests 0 and 1 would run in every call and the others in every
    /// second one.
    #[test]
    fn requests_that_a_call_leaves_out_go_first() {
        const REQUESTS: usize = 6;
        const ROWS: usize = 4;
        const CALLS: usize = 30;
        let mut turns = Turns::default();
        let mut turn: Vec<u64> = turns.take(REQUESTS).collect();
        let mut calls_run = [0; REQUESTS];
        for _ in 0..CALLS {
            let mut order: Vec<usize> = (0..REQUESTS).collect();
            order.sort_by_key(|&i| turn[i]);
            let call = &order[..ROWS];
            for (&i, next) in call.iter().zip(turns.take(ROWS)) {
                calls_run[i] += 1;
                turn[i] = next;
            }
        }
        assert_eq!(calls_run, [CALLS * ROWS / REQUESTS; REQUESTS]);
    }

    /// DeepSeek V4.1 Flash's most rows per call.
    const MAX_ROWS_PER_CALL: usize = 6;

    #[test]
    fn the_rows_per_call_setting_stays_between_one_and_the_models_most_rows() {
        assert_eq!(
            limit_rows_per_call(MAX_ROWS_PER_CALL, None),
            MAX_ROWS_PER_CALL
        );
        assert_eq!(limit_rows_per_call(MAX_ROWS_PER_CALL, Some(1)), 1);
        assert_eq!(limit_rows_per_call(MAX_ROWS_PER_CALL, Some(2)), 2);
        // Never more than the model takes.
        assert_eq!(
            limit_rows_per_call(MAX_ROWS_PER_CALL, Some(9)),
            MAX_ROWS_PER_CALL
        );
        assert_eq!(limit_rows_per_call(MAX_ROWS_PER_CALL, Some(0)), 1);
        let chooser = CallChooser::new();
        assert_eq!(
            chooser.choose_calls(4, limit_rows_per_call(MAX_ROWS_PER_CALL, Some(1))),
            (4, 1)
        );
        assert_eq!(
            chooser.choose_calls(4, limit_rows_per_call(MAX_ROWS_PER_CALL, Some(2))),
            (2, 2)
        );
        assert_eq!(
            chooser.choose_calls(12, limit_rows_per_call(MAX_ROWS_PER_CALL, Some(4))),
            (3, 4)
        );
    }
}
