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
//! requests run together, at most the model's rows per call. A request that
//! decodes alone may also run its drafts or its lookup proposals.
//!
//! A decision model's step runs its decisions one at a time, one pass per
//! call, and returns each pass's logits at the decision's end.
//!
//! Calls run on streams. The calls of all models run at the same time. After
//! the steps, the loop advances the streams until a call ends. It then finishes
//! that call. A prompt resumes from a state in the state cache. The worker also
//! builds each model's status.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::calls::CallChooser;
use crate::control::{self, Action, Control, GIB, Phase, YieldTarget};
use crate::decision::d1;
use crate::disk::{EngineState, LoadError, Store};
use crate::engine::{
    self, CacheCounts, DraftChoice, ExpandError, ImageCallParams, ImageError, ImageFormat,
    ImageInputs, Logits, Lookup, Media, Model, PrefillInfo, Sampler, SnapshotHandle, StateHandle,
    Stream,
};
use crate::state_cache::{
    self, Cache, Entry, Evicted, ModelLimits, Plan, PlanLimits, Snapshot, SnapshotKind,
};

/// The engine needs a stack of at least 16 MiB on its thread.
const ENGINE_STACK_BYTES: usize = 16 << 20;
/// The bytes in one GB, for expert reads.
const GB: f64 = 1e9;
/// The engine thread's states, in the order of eremite_stats's thread_s and
/// gpu_idle_s.
const THREAD_STATES: [&str; 5] = ["outside", "pass", "wait_gpu", "wait_read", "wait_other"];

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
    /// A decision ended: the logits of each of its passes, in the passes'
    /// order, and the tokens that the passes read, each image's span counted
    /// whole. This is the last event.
    Decided {
        logits: Vec<Vec<f32>>,
        tokens: usize,
    },
    /// An image generation or editing request finished: the encoded PNG or
    /// JPEG bytes of each produced image. This is the last event.
    Images { images: Vec<Vec<u8>> },
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

impl Default for Sampling {
    /// OpenAI's defaults: temperature 1, top_p 1 and every token, with a random
    /// seed.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
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

/// A decision model's request: passes that it runs one after another.
pub struct DecisionRequest {
    /// The index of the request's model, in the configured order.
    pub model_index: usize,
    /// The passes, each a state and questions after it (decision/d1.rs).
    pub passes: Vec<d1::Pass>,
    /// The bytes of each image file, in the order of the image places in each
    /// pass's state.
    pub media: Vec<Vec<u8>>,
    /// The log shows "req ID NOTE" with this note when the request is queued.
    pub note: String,
}

impl DecisionRequest {
    /// Counts the tokens of all its passes.
    fn count_tokens(&self) -> usize {
        self.passes.iter().map(|pass| pass.tokens.len()).sum()
    }
}

/// An image model's generation or editing request: one or more images generated
/// from a prompt and optional reference images.
pub struct ImageRequest {
    /// The index of the request's model, in the configured order.
    pub model_index: usize,
    /// Raw prompt text passed verbatim to the model.
    pub prompt: String,
    /// Exact requested output width in pixels.
    pub width: i32,
    /// Exact requested output height in pixels.
    pub height: i32,
    /// Number of images to generate (`n`).
    pub count: usize,
    /// Base seed for the first image (`seed.wrapping_add(i)` for later ones).
    pub seed: u64,
    /// Minimum generation canvas area in pixels.
    pub min_generation_area: i32,
    /// Encoded output image format.
    pub format: ImageFormat,
    /// JPEG quality in 1..=100.
    pub quality: i32,
    /// Optional reference image bytes in upload order (`<Picture 1>`, ...).
    pub references: Vec<Vec<u8>>,
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

/// A submitted request, as the worker holds it: a Request, or a
/// DecisionRequest.
struct Job<R = Request> {
    /// The request's id: its number in the log.
    id: u64,
    /// The request itself.
    request: R,
    /// Sends the request's events to its handler.
    events: mpsc::UnboundedSender<Event>,
    /// The flags it shares with its cancel guard.
    flags: Arc<RequestFlags>,
    /// When the request was queued.
    queued_at: Instant,
}

impl<R> Job<R> {
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
    /// The seconds without requests before the model's weights and experts
    /// unload. 0 keeps them loaded.
    pub idle_unload: f64,
    /// The most states of the model that the state cache keeps.
    pub max_cached_states: usize,
}

/// What the terminal UI asks of the worker. The default asks for nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewOptions {
    /// The number of tokens whose experts the status keeps for each decoding
    /// request. 0 keeps none.
    pub kept_route_tokens: usize,
    /// Receives each log line about a model, with the model's index, instead of
    /// the log.
    pub on_model_log: Option<fn(usize, &str)>,
}

impl ViewOptions {
    /// Writes `line`, a line about the model at `model_index`, to the terminal
    /// UI's hook. Without the hook, it writes the line to the log.
    fn write_model_line(&self, model_index: usize, line: &str) {
        match self.on_model_log {
            Some(hook) => hook(model_index, line),
            None => engine::log(line),
        }
    }
}

/// What the HTTP handlers and the worker share under one lock.
#[derive(Default)]
struct Queue {
    /// Each model's waiting requests, the oldest first.
    waiting: Vec<VecDeque<Job>>,
    /// Each model's waiting decisions, the oldest first.
    decisions: Vec<VecDeque<Job<DecisionRequest>>>,
    /// Each model's waiting image requests, the oldest first.
    images: Vec<VecDeque<Job<ImageRequest>>>,
    /// Set at the stop. No request is queued after it.
    closed: bool,
    /// The id of the last queued request. The first request gets 1.
    last_request_id: u64,
    /// The yield and resume control.
    control: Control,
    /// The sessions whose states the terminal UI asked to delete from memory,
    /// each as its model's index and its number, the oldest first.
    states_to_delete: Vec<(usize, u64)>,
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
    /// The drafts per pass that the model makes itself.
    drafts_per_pass: usize,
    /// What the model does now: idle, loading, prefill, decode or generating.
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
    /// The number of prompt tokens of the ended requests that came from the
    /// state cache.
    total_reused_tokens: u64,
    /// The prompt tokens that the ended prefills computed.
    total_prefilled_tokens: u64,
    /// The seconds that the ended prefills took.
    total_prefill_seconds: f64,
    /// The number of images that the ended requests generated.
    total_generated_images: u64,
    /// The seconds that the ended requests spent denoising, decoding and
    /// encoding images.
    total_image_seconds: f64,
    /// The number of requests that started with each kind of plan, by the
    /// plan's word in the log.
    plan_counts: HashMap<&'static str, u64>,
    /// The model's requests that run, and those that wait for their prefill's
    /// turn.
    requests: Vec<RequestStatus>,
    /// The plan of the latest prefill.
    last_prefill: PrefillInfo,
    /// Whether the decoding requests split evenly over at least two calls.
    split_calls: bool,
    /// The model's sessions, in the order of their tokens.
    sessions: Vec<SessionStatus>,
    /// The state bytes committed for the model: what the state cache counts for
    /// its states and their snapshots, the reserved ones included.
    state_committed_bytes: u64,
    /// The bytes that the model's states and their snapshots may take.
    state_limit_bytes: u64,
    /// The bytes that the engine allocated for the model's states and their
    /// snapshots.
    state_allocated_bytes: u64,
}

/// A request of a model, for the model's status.
#[derive(Clone, Debug, PartialEq)]
struct RequestStatus {
    /// The request's id: its number in the log.
    id: u64,
    /// What the request does: "waiting", "prefill", "decoding" or "generating".
    status: &'static str,
    /// The number of the session whose state the request uses. None while the
    /// request waits.
    session: Option<u64>,
    /// The number of prompt tokens. Once the prefill starts, it counts the
    /// media expanded.
    prompt_tokens: usize,
    /// The number of prompt tokens that came from the state cache.
    reused_tokens: usize,
    /// The number of tokens that the reply generated.
    generated_tokens: usize,
    /// The images requested by an image request; 0 for text and decision
    /// requests.
    images_requested: usize,
    /// The images finished so far for an image request.
    generated_images: usize,
    /// The experts of the request's latest tokens, the oldest first.
    routes: Vec<Route>,
}

/// The experts that one token of a request chose, and where each one was.
#[derive(Clone, Debug, PartialEq)]
struct Route {
    /// The experts of each layer with routed experts, layer by layer.
    experts: Vec<u16>,
    /// Where each expert was: 0 in memory, 1 read ahead, 2 late, 3
    /// substituted, 4 from the SSD.
    sources: Vec<u8>,
}

impl RequestStatus {
    /// Returns the status of the decision `job`, which does `status`: "waiting"
    /// or "prefill".
    fn for_decision(job: &Job<DecisionRequest>, status: &'static str) -> RequestStatus {
        RequestStatus {
            id: job.id,
            status,
            session: None,
            prompt_tokens: job.request.count_tokens(),
            reused_tokens: 0,
            generated_tokens: 0,
            images_requested: 0,
            generated_images: 0,
            routes: Vec::new(),
        }
    }

    /// Returns the status of an image request `job` with `prompt_tokens`
    /// encoded prompt tokens and `generated_images` finished outputs.
    fn for_image(
        job: &Job<ImageRequest>,
        status: &'static str,
        prompt_tokens: usize,
        generated_images: usize,
    ) -> RequestStatus {
        RequestStatus {
            id: job.id,
            status,
            session: None,
            prompt_tokens,
            reused_tokens: 0,
            generated_tokens: 0,
            images_requested: job.request.count,
            generated_images,
            routes: Vec::new(),
        }
    }

    /// Returns the status of `job`, a request that waits for its prefill.
    fn for_waiting_job(job: &Job) -> RequestStatus {
        RequestStatus {
            id: job.id,
            status: "waiting",
            session: None,
            prompt_tokens: job.request.prompt.len(),
            reused_tokens: 0,
            generated_tokens: 0,
            images_requested: 0,
            generated_images: 0,
            routes: Vec::new(),
        }
    }
}

/// A session of a model, for the model's status. A session is a state that
/// the state cache keeps.
#[derive(Clone, Debug, PartialEq)]
struct SessionStatus {
    /// The session's number in the log.
    number: u64,
    /// The number of tokens whose rows the state holds.
    tokens: usize,
    /// When a request last used the state.
    last_used_at: Instant,
    /// Whether a request runs in the state.
    busy: bool,
    /// The bytes that the state cache counts for the state and its snapshots.
    bytes: u64,
    /// The position of the state's latest prompt-end snapshot, if it has one.
    prompt_end: Option<usize>,
    /// The number of tokens that the session shares from position 0 with the
    /// next session in the order of their tokens. It is 0 for the last one.
    shared_with_next: usize,
}

/// A model as the HTTP handlers and the worker share it. The tokenizer works on
/// any thread.
pub struct SharedModel {
    /// The model's settings.
    pub settings: ModelSettings,
    /// The model's tokenizer.
    pub tokenizer: engine::Tokenizer,
    /// Image generation limits reported by the opened model, or None for text
    /// and decision models.
    pub image_limits: Option<engine::ImageLimits>,
    /// The model's status. The worker updates it.
    status: Mutex<ModelStatus>,
}

/// What the HTTP handlers and the worker share.
struct Shared {
    /// The waiting requests and the control.
    queue: Mutex<Queue>,
    /// Wakes the worker when it has work: a new request, a cancellation, a
    /// status reader's request, a control request, or the stop.
    worker_wakeup: Condvar,
    /// The served models, in their configured order.
    models: Vec<SharedModel>,
    /// The server-wide budgets and limits.
    budgets: crate::config::Budgets,
    /// The state bytes committed for all models together.
    state_committed_bytes: AtomicU64,
    /// The status of the saved states on the SSD.
    disk_status: Mutex<DiskStatus>,
    /// Whether a status reader asked for fresh counters and states. The worker
    /// publishes them at its next turn, also between the steps of a running
    /// call.
    status_wanted: AtomicBool,
    /// Wakes the status readers once the worker has published, when a long
    /// operation starts, and at the stop.
    status_published: tokio::sync::Notify,
    /// The worker's long operations running now (LongOperation).
    long_operations: AtomicUsize,
    /// What the terminal UI asks of the worker.
    view: ViewOptions,
    /// The session whose tokens a reader asked for: its model's index and its
    /// number. The worker takes it at its next publish of the states.
    session_tokens_wanted: Mutex<Option<(usize, u64)>>,
    /// The tokens of the session asked for last, with its model's index and
    /// its number, once the worker has published them.
    session_tokens: Mutex<Option<(usize, u64, Vec<i32>)>>,
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

/// The status of the saved states on the SSD.
#[derive(Clone, Debug, Default)]
pub struct DiskStatus {
    /// The store's folder. None without a store.
    pub dir: Option<String>,
    /// The number of saved states.
    pub states: usize,
    /// The bytes that the saved states take.
    pub bytes: u64,
    /// The most bytes that the saved states may take.
    pub limit: u64,
    /// The saved states read in this run.
    pub loads: u64,
    /// The states saved in this run.
    pub saves: u64,
    /// The bytes written in this run.
    pub written: u64,
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

/// Builds the control's status as JSON.
pub fn build_control_status_json() -> Value {
    let shared = clone_shared();
    lock(&shared.queue).control.build_status_json()
}

/// Starts a yield for `request`, and wakes the worker. Returns the HTTP status
/// and the control's status as JSON. Returns an error while the server is
/// stopping.
pub fn start_yield(request: control::YieldRequest) -> Result<(u16, Value), control::Error> {
    let shared = clone_shared();
    let mut queue = lock(&shared.queue);
    if queue.closed || engine::is_stopping() {
        return Err(control::Error {
            http_status: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
            code: "server_stopping",
            message: "the server is stopping".into(),
        });
    }
    let http_status = queue.control.start_yield(request)?;
    let status_json = queue.control.build_status_json();
    notify_worker(&shared);
    Ok((http_status, status_json))
}

/// Starts the resume for `request`, and wakes the worker. Returns the HTTP
/// status and the control's status as JSON. For an operation that ended
/// already, the JSON also holds resumed_operation_id. Returns an error while
/// the server is stopping.
pub fn start_resume(request: control::ResumeRequest) -> Result<(u16, Value), control::Error> {
    let shared = clone_shared();
    let mut queue = lock(&shared.queue);
    if queue.closed || engine::is_stopping() {
        return Err(control::Error {
            http_status: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
            code: "server_stopping",
            message: "the server is stopping".into(),
        });
    }
    let ended = queue.control.start_resume(&request.operation_id)?;
    let mut status_json = queue.control.build_status_json();
    if ended {
        status_json["resumed_operation_id"] = json!(request.operation_id);
    }
    notify_worker(&shared);
    let http_status = if ended {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    Ok((http_status.as_u16(), status_json))
}

/// Asks for the tokens of session `session` of the model at `model_index`. The
/// worker publishes them at its next publish of the states, such as the one
/// that refresh_status asks for.
pub fn ask_session_tokens(model_index: usize, session: u64) {
    *lock(&clone_shared().session_tokens_wanted) = Some((model_index, session));
}

/// Returns the tokens of session `session` of the model at `model_index`, once
/// the worker has published them after ask_session_tokens. Returns None
/// before that, and for a session that the state cache no longer keeps.
pub fn read_session_tokens(model_index: usize, session: u64) -> Option<Vec<i32>> {
    match &*lock(&clone_shared().session_tokens) {
        Some((model, number, tokens)) if (*model, *number) == (model_index, session) => {
            Some(tokens.clone())
        }
        _ => None,
    }
}

/// Asks the worker to delete the state of session `session` of the model at
/// `model_index` from memory, without saving it to the SSD. The worker deletes
/// it between engine calls and logs what it did. It deletes no state that a
/// request runs in. Once the server stops, it deletes no more states.
pub fn ask_to_delete_session_state(model_index: usize, session: u64) {
    let shared = clone_shared();
    let mut queue = lock(&shared.queue);
    if queue.closed || engine::is_stopping() {
        drop(queue);
        let line = format!("states: session {session} not deleted: the server stops");
        shared.view.write_model_line(model_index, &line);
        return;
    }
    queue.states_to_delete.push((model_index, session));
    notify_worker(&shared);
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

/// Returns the image generation limits of the served model at `model_index`, or
/// None when it generates no images.
pub fn image_limits(model_index: usize) -> Option<engine::ImageLimits> {
    clone_shared().models.get(model_index)?.image_limits
}

/// Starts the engine thread. The thread registers each model without loading
/// its weights. `view` holds what the terminal UI asks of the worker. This call
/// returns true once the worker takes requests. It returns false when the
/// engine cannot start or a model's settings are invalid. The log says why a
/// model's settings are invalid.
pub fn start(
    model_settings: Vec<ModelSettings>,
    budgets: crate::config::Budgets,
    state_dir: Option<PathBuf>,
    on_log: Option<extern "C" fn(*const std::ffi::c_char)>,
    on_die: Option<extern "C" fn()>,
    view: ViewOptions,
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
                let image_limits = model.image_limits();
                let mut resolved = settings.clone();
                if resolved.options.ctx == 0
                    && let Some(lim) = image_limits
                {
                    resolved.options.ctx = lim.max_prompt_tokens;
                }
                models.push(SharedModel {
                    settings: resolved,
                    tokenizer: model.tokenizer(),
                    image_limits,
                    status: Mutex::new(ModelStatus {
                        stats: model.read_stats(),
                        drafts_per_pass: model.drafts_per_pass(),
                        ..ModelStatus::default()
                    }),
                });
                model_handles.push(Rc::new(model));
            }
            let model_count = models.len();
            let shared = Arc::new(Shared {
                queue: Mutex::new(Queue {
                    waiting: (0..model_count).map(|_| VecDeque::new()).collect(),
                    decisions: (0..model_count).map(|_| VecDeque::new()).collect(),
                    images: (0..model_count).map(|_| VecDeque::new()).collect(),
                    control: Control::new(engine::read_memory().limit),
                    ..Queue::default()
                }),
                worker_wakeup: Condvar::new(),
                models,
                budgets,
                state_committed_bytes: AtomicU64::new(0),
                disk_status: Mutex::new(DiskStatus::default()),
                status_wanted: AtomicBool::new(false),
                status_published: tokio::sync::Notify::new(),
                long_operations: AtomicUsize::new(0),
                view,
                session_tokens_wanted: Mutex::new(None),
                session_tokens: Mutex::new(None),
            });
            *SHARED.lock().unwrap() = Some(shared.clone());
            // A panic here ends the process, as the engine's fatal errors do.
            // The waiting requests could not end otherwise. Rust has already
            // printed the panic's message.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut worker = Worker::new(shared.clone(), model_handles, state_dir);
                let _ = started_sender.send(true);
                worker.run_until_stop();
                worker.save_states_at_stop();
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

/// Writes `line`, a line about the model at `model_index`, to the log, as the
/// worker writes its own lines about that model.
pub fn log_model_line(model_index: usize, line: &str) {
    clone_shared().view.write_model_line(model_index, line);
}

/// Requests the stop as SIGINT does. A second request ends the saving of states
/// at the stop.
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
/// unless a stop was requested already. A second request would end the saving
/// of states. Before its thread ends, the worker ends every request and saves
/// the states.
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
    let counts = format!(
        "prompt_tokens={}, max_new={}",
        request.prompt.len(),
        request.max_tokens
    );
    let (model_index, note) = (request.model_index, request.note.clone());
    queue_job(request, model_index, &note, &counts, |queue| {
        &mut queue.waiting
    })
}

/// Submits the decision `request` to its model's queue, as submit does.
pub fn submit_decision(
    request: DecisionRequest,
) -> Result<(CancelGuard, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let counts = format!(
        "tokens={}, passes={}",
        request.count_tokens(),
        request.passes.len()
    );
    let (model_index, note) = (request.model_index, request.note.clone());
    queue_job(request, model_index, &note, &counts, |queue| {
        &mut queue.decisions
    })
}

/// Submits the image `request` to its model's queue, as submit does.
pub fn submit_image(
    request: ImageRequest,
) -> Result<(CancelGuard, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let counts = format!(
        "size={}x{}, n={}, refs={}",
        request.width,
        request.height,
        request.count,
        request.references.len()
    );
    let (model_index, note) = (request.model_index, request.note.clone());
    queue_job(request, model_index, &note, &counts, |queue| {
        &mut queue.images
    })
}

/// Queues `request` for the model at `model_index` in the queues that
/// `waiting` picks. The log shows its `note` and its `counts`. Returns the
/// guard and the receiver of its events, or Busy when too many requests wait
/// or the server is stopping.
fn queue_job<R>(
    request: R,
    model_index: usize,
    note: &str,
    counts: &str,
    waiting: fn(&mut Queue) -> &mut Vec<VecDeque<Job<R>>>,
) -> Result<(CancelGuard, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let shared = clone_shared();
    let (events, receiver) = mpsc::unbounded_channel();
    let flags = Arc::new(RequestFlags::default());
    let mut queue = lock(&shared.queue);
    if queue.closed || waiting(&mut queue)[model_index].len() >= shared.budgets.max_waiting_requests
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
    if !note.is_empty() {
        // The log shows the note before the worker can take the request.
        shared
            .view
            .write_model_line(model_index, &format!("req {request_id} {note}"));
    }
    let jobs = &mut waiting(&mut queue)[model_index];
    jobs.push_back(Job {
        id: request_id,
        request,
        events,
        flags: flags.clone(),
        queued_at: Instant::now(),
    });
    shared.view.write_model_line(
        model_index,
        &format!(
            "req {request_id}: queued for {}, waiting={}, {counts}",
            shared.models[model_index].settings.id,
            jobs.len()
        ),
    );
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
    let (waiting, control_phase, queued) = {
        let queue = lock(&shared.queue);
        let queued: Vec<RequestStatus> = queue.waiting[model_index]
            .iter()
            .map(RequestStatus::for_waiting_job)
            .chain(
                queue.decisions[model_index]
                    .iter()
                    .map(|job| RequestStatus::for_decision(job, "waiting")),
            )
            .chain(
                queue.images[model_index]
                    .iter()
                    .map(|job| RequestStatus::for_image(job, "waiting", 0, 0)),
            )
            .collect();
        (queued.len(), queue.control.phase, queued)
    };
    let stats = lock(&shared_model.status).stats;
    let allocations = build_model_memory_json(&shared);
    let status = lock(&shared_model.status);
    let now = Instant::now();
    // The status rounds idle times to whole seconds. It rounds other seconds
    // and memory GiB to tenths, and state and disk GiB to thousandths. Why
    // these digits were chosen is unknown.
    let sessions: Vec<Value> = status
        .sessions
        .iter()
        .map(|session| {
            json!({
                "session": session.number,
                "tokens": session.tokens,
                "idle_seconds": round((now - session.last_used_at).as_secs_f64(), 0),
                "busy": session.busy,
                "bytes": session.bytes,
                "prompt_end": session.prompt_end,
                "shared_with_next": session.shared_with_next,
            })
        })
        .collect();
    let phase = if control_phase != Phase::Running {
        control_phase.as_str()
    } else if status.phase.is_empty() {
        "idle"
    } else {
        status.phase
    };
    let phase_seconds = status
        .phase_since
        .map_or(0.0, |since| (now - since).as_secs_f64());
    let total_state_bytes = shared.state_committed_bytes.load(Ordering::Relaxed);
    let requests: Vec<RequestStatus> = status.requests.iter().cloned().chain(queued).collect();
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
        "requests": build_requests_json(&requests),
        "sessions": sessions,
        "state_cache_gib": round(status.state_committed_bytes as f64 / GIB, 3),
        "state_cache_limit_gib": round(status.state_limit_bytes as f64 / GIB, 3),
        "state_cache_total_gib": round(total_state_bytes as f64 / GIB, 3),
        "state_disk": build_disk_status_json(&shared),
        "reserve_gib": shared.budgets.reserve_gb,
        "memory_gib": shared.budgets.memory_gb,
        "model_memory": allocations,
        "model_memory_budget": {
            "limit": stats.model_memory_limit,
            "allocated": stats.model_memory_used,
            "committed": stats.model_memory_committed,
            "peak": stats.model_memory_peak,
        },
        "own_memory": {
            "limit": stats.own_memory_limit,
            "counted": stats.own_memory_counted,
            "states": stats.own_state_bytes,
            "required": stats.required_bytes,
        },
        "totals": {
            "requests": status.total_requests,
            "prompt_tokens": status.total_prompt_tokens,
            "generated_tokens": status.total_generated_tokens,
            "reused_tokens": status.total_reused_tokens,
            "prefilled_tokens": status.total_prefilled_tokens,
            "prefill_seconds": status.total_prefill_seconds,
            "generated_images": status.total_generated_images,
            "image_seconds": status.total_image_seconds,
        },
        "plans": status.plan_counts,
        "ced_depth": options.deepseek().map_or(0, |deepseek| deepseek.ced),
        "lookup_depth": options.lookup,
        "prefill_prune": options.deepseek().is_some_and(|deepseek| deepseek.prunes_prefill()),
        "split_decoding_calls": status.split_calls,
        "drafts": status.drafts_per_pass,
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
        "engine_counters": build_engine_counters_json(&stats),
    })
}

/// Builds `requests` as JSON: each request's id, what it does, its session,
/// its prompt tokens, its reused tokens, its generated tokens, and the experts
/// of its latest tokens.
fn build_requests_json(requests: &[RequestStatus]) -> Value {
    requests
        .iter()
        .map(|request| {
            json!({
                "request": request.id,
                "status": request.status,
                "session": request.session,
                "prompt_tokens": request.prompt_tokens,
                "reused_tokens": request.reused_tokens,
                "generated_tokens": request.generated_tokens,
                "images_requested": request.images_requested,
                "generated_images": request.generated_images,
                "routes": request
                    .routes
                    .iter()
                    .map(|route| json!({"experts": route.experts, "sources": route.sources}))
                    .collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// Builds the engine's counters that the rest of the status leaves out or
/// rounds, as JSON. They hold the model's layers and routed experts, and where
/// the decoded rows' experts were. They hold the prompt rows and the running
/// prefill's layer and rows. They also hold the bytes of experts read, the
/// bytes of experts mapped for the GPU, and the system's memory in bytes.
fn build_engine_counters_json(stats: &engine::Stats) -> Value {
    let decode = &stats.decode;
    // The engine reports i64::MAX as the GPU's limit before the GPU starts.
    let gpu_limit = (stats.gpu_limit != i64::MAX).then_some(stats.gpu_limit);
    json!({
        "layers": stats.layers,
        "experts_per_row": stats.experts_per_row,
        "routed_layers": stats.routed_layers,
        "experts_per_layer": stats.experts_per_layer,
        "decode_expert_sources": [
            decode.row_own,
            decode.row_ahead,
            decode.row_late,
            decode.row_sub,
            decode.row_ssd,
        ],
        "prefill_rows": stats.prefill_rows,
        "prefill_read_bytes": stats.prefill.bytes,
        "prefill_layer": stats.prefill_layer,
        "prefill_rows_now": stats.prefill_rows_now,
        "generated_images": stats.generated_images,
        "image_steps": stats.image_steps,
        "image_seconds": stats.image_seconds,
        "image_stage": stats.image_stage,
        "image_step": stats.image_step,
        "image_steps_total": stats.image_steps_total,
        "expert_read_bytes": stats.cache.bytes,
        "mapped_expert_bytes": stats.mapped.max(0) as u64 * stats.slot_bytes,
        "memory": {
            "footprint": stats.footprint,
            "available": stats.available,
            "gpu_in_use": stats.gpu_in_use,
            "gpu_limit": gpu_limit,
            "total": stats.total,
            "free": stats.free,
            "file": stats.file,
            "file_kept": stats.file_kept,
            "wired": stats.wired,
            "swap_used": stats.swap_used,
            "pressure": stats.pressure,
        },
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

/// Builds the status of the saved states as JSON, or null without a store. It
/// holds their folder, their count, their GiB and the GiB limit. It also holds
/// this run's loads, saves and GiB written.
fn build_disk_status_json(shared: &Shared) -> Value {
    let disk_status = lock(&shared.disk_status).clone();
    match disk_status.dir {
        None => Value::Null,
        Some(dir) => json!({
            "dir": dir,
            "states": disk_status.states,
            "gib": round(disk_status.bytes as f64 / GIB, 3),
            "limit_gib": round(disk_status.limit as f64 / GIB, 3),
            "loads": disk_status.loads,
            "saves": disk_status.saves,
            "written_gib": round(disk_status.written as f64 / GIB, 3),
        }),
    }
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
    /// The number of the session whose state the request uses.
    session: u64,
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
    /// The number of verify passes that ran drafts or proposals.
    draft_passes: usize,
    /// The number of drafts and proposals that the verify passes accepted.
    drafts_accepted: usize,
    /// The lookup that proposes tokens from the context, when the model uses
    /// one.
    lookup: Option<Lookup>,
    /// The number of tokens that the lookup holds.
    lookup_tokens: usize,
    /// The number of passes that ran the lookup's proposals.
    lookup_passes: usize,
    /// The number of proposals that the lookup offered.
    lookup_offered: usize,
    /// The number of proposals that the passes accepted.
    lookup_accepted: usize,
    /// Chooses how many drafts each pass runs, when the model drafts.
    draft_choice: Option<DraftChoice>,
    /// The start time and the draft count of the running pass. It is set only
    /// while no other call of the model runs beside the pass. The draft choice
    /// learns from such passes only.
    timed_pass: Option<(Instant, usize)>,
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
    /// The experts of the request's latest tokens, the oldest first, as many
    /// as the status keeps.
    routes: VecDeque<Route>,
}

impl Generation {
    /// Returns whether the request decodes: its prefill has ended.
    fn is_decoding(&self) -> bool {
        self.prefill.is_none()
    }
}

/// A decision from its first pass to its last.
struct Decision {
    /// The submitted request. Its passes hold their images' spans.
    job: Job<DecisionRequest>,
    /// Its images, encoded, if it has any.
    media: Option<Media>,
    /// The logits that its finished passes read, in the passes' order.
    logits: Vec<Vec<f32>>,
    /// Why its last pass failed, if it did.
    failure: Option<String>,
    /// Whether a pass of the decision runs on a stream.
    in_call: bool,
    /// When the decision's first pass started.
    started_at: Instant,
}

/// An image generation or editing request from its first image to its last.
struct ImageTask {
    /// The submitted image request.
    job: Job<ImageRequest>,
    /// Prompt C string and reference bytes kept alive across stream calls.
    inputs: ImageInputs,
    /// Encoded PNG/JPEG images generated so far.
    outputs: Vec<Vec<u8>>,
    /// Encoded prompt tokens of this request, including reference image tokens.
    prompt_tokens: usize,
    /// Engine prefill_rows counter when the active image pass started.
    pass_start_prefill_rows: u64,
    /// Sum of prompt tokens encoded across completed passes of this request.
    prefilled_tokens: u64,
    /// Seconds spent in the prompt encoder across completed passes.
    prefill_seconds: f64,
    /// Seconds spent denoising, decoding and encoding images across completed
    /// passes.
    image_seconds: f64,
    /// Error from the last failed call, if any.
    failure: Option<ImageError>,
    /// Whether an image generation call is currently running on a stream.
    in_call: bool,
    /// When the first image started generating.
    started_at: Instant,
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
    /// One verify pass of the request: its token and its drafts in `rows`.
    /// `proposed_count` counts the lookup's proposals among them. With 0, the
    /// model made the drafts.
    Verify {
        request_id: u64,
        rows: Vec<i32>,
        proposed_count: usize,
    },
    /// The next pass of a decision model's request. The request keeps the
    /// pass's tokens until the call ends.
    Decide { request_id: u64 },
    /// One image generation or editing call of an image model's request.
    GenerateImage { request_id: u64 },
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
    /// Whether the model is a decision model: it runs decisions, and generates
    /// no text.
    is_decision_model: bool,
    /// Admitted decisions that wait for their turn.
    admitted_decisions: VecDeque<Job<DecisionRequest>>,
    /// The decision that runs its passes.
    decision: Option<Decision>,
    /// Whether the model is an image generation model: it generates images,
    /// not text or decisions.
    is_image_model: bool,
    /// Admitted image requests that wait for their turn.
    admitted_images: VecDeque<Job<ImageRequest>>,
    /// The image request currently running.
    image: Option<ImageTask>,
    /// The token that starts a think block, if the model has one.
    think_start_token: Option<i32>,
    /// The token that ends a think block, if the model has one.
    think_end_token: Option<i32>,
    /// The number of decoding requests in the last log line about them.
    logged_decoding_count: usize,
    /// The experts that a decoded row chooses in all layers together. It is 0
    /// for a model without routed experts.
    experts_per_row: usize,
    /// What the terminal UI asks of the worker.
    view: ViewOptions,
    /// When the model last had a request. The idle unload counts from then.
    last_used_at: Instant,
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

/// The states of all models: the state cache's records, each state's engine
/// handles and the store of saved states.
struct States {
    /// The state cache's records of the states.
    cache: Cache,
    /// Each state's engine handles, by the state's key.
    handles: HashMap<state_cache::StateId, StateHandles>,
    /// The store of saved states on the SSD. None without one.
    store: Option<Store>,
    /// States that the stop cut short in a prefill or a decoding pass. Their
    /// rows are not saved.
    cut_short: HashSet<state_cache::StateId>,
    /// For a model that resumes only from snapshots, a prefill takes a snapshot
    /// at every multiple of this many positions.
    snapshot_interval: usize,
}

impl States {
    /// Creates the states of all models: an empty state cache with each
    /// model's `limits`, by the model's index, without a store.
    /// `snapshot_interval` is the prefills' snapshot interval.
    fn new(limits: Vec<ModelLimits>, snapshot_interval: usize) -> Self {
        Self {
            cache: Cache::new(limits),
            handles: HashMap::new(),
            store: None,
            cut_short: HashSet::new(),
            snapshot_interval,
        }
    }

    /// Frees what the state cache evicted. It frees the snapshots. It saves the
    /// states to the SSD, as the cache decides, then frees them. The most
    /// recently used state is saved first. A newer saved state may then hold an
    /// older state's tokens. That older state is then not written.
    fn free_evicted(
        &mut self,
        shared: &Shared,
        Evicted {
            snapshots,
            mut states,
        }: Evicted,
    ) {
        for (state_id, snapshot) in snapshots {
            self.free_snapshot(state_id, snapshot.pos);
        }
        states.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.last_used));
        for (state_id, entry) in states {
            self.save_state(shared, state_id, &entry, true);
            self.handles.remove(&state_id);
        }
    }

    /// Saves state `state_id` to the SSD, as the state cache decides. `entry`
    /// is its record, already out of the cache. `others_stay_in_memory` says
    /// whether the states still in the cache stay in memory. A state that the
    /// stop cut short is not saved.
    fn save_state(
        &mut self,
        shared: &Shared,
        state_id: state_cache::StateId,
        entry: &Entry,
        others_stay_in_memory: bool,
    ) {
        let States {
            store: Some(store),
            cache,
            handles,
            cut_short,
            ..
        } = self
        else {
            return;
        };
        let Some(state_handles) = handles.get(&state_id) else {
            return;
        };
        if cut_short.contains(&state_id) {
            return;
        }
        let snapshot = |pos: usize| state_handles.snapshots.get(&pos).copied();
        let last_used_at = std::time::SystemTime::now() - state_handles.last_used_at.elapsed();
        let _long = LongOperation::start(shared);
        if let Some(line) = store.save_state(
            cache,
            &EngineState {
                model: &state_handles.owner,
                state: state_handles.engine_state,
            },
            &snapshot,
            entry,
            last_used_at,
            others_stay_in_memory,
        ) {
            engine::log(&format!("states: {line}"));
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

    /// Deletes the state of session `number` of the model at `model_index` from
    /// memory, with its snapshots, without saving it to the SSD. It deletes
    /// nothing when the state cache keeps no such session, or when a request
    /// runs in it. Returns the line for the log: what it deleted, or why it
    /// deleted nothing.
    fn delete_session_state(&mut self, model_index: usize, number: u64) -> String {
        let found = self
            .cache
            .entries
            .iter()
            .find(|(_, entry)| entry.model == model_index && entry.number == number);
        let Some((&state_id, entry)) = found else {
            return format!("session {number} not deleted: it is no longer in memory");
        };
        if entry.busy {
            return format!("session {number} not deleted: a request runs in it");
        }
        let tokens = entry.tokens.len();
        let bytes = entry.bytes
            + entry
                .snapshots
                .iter()
                .map(|snapshot| snapshot.bytes)
                .sum::<u64>();
        let snapshots = self.cache.remove_state(state_id);
        self.free_snapshots_and_states(Vec::new(), vec![(state_id, snapshots)]);
        format!(
            "session {number} deleted from memory, not saved to the SSD: {tokens} tokens, {:.2} GiB",
            bytes as f64 / GIB
        )
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

    /// Publishes the states to the status: each model's states and state bytes,
    /// and the saved states' figures.
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
            let token_lists: Vec<&[i32]> =
                entries.iter().map(|(_, entry)| &entry.tokens[..]).collect();
            let mut status = lock(&shared_model.status);
            status.sessions = order_by_tokens(&token_lists)
                .into_iter()
                .map(|(i, shared_with_next)| {
                    let (state_id, entry) = entries[i];
                    SessionStatus {
                        number: entry.number,
                        tokens: entry.tokens.len(),
                        last_used_at: self.handles[state_id].last_used_at,
                        busy: entry.busy,
                        bytes: entry.bytes
                            + entry
                                .snapshots
                                .iter()
                                .map(|snapshot| snapshot.bytes)
                                .sum::<u64>(),
                        prompt_end: entry
                            .snapshots
                            .iter()
                            .filter(|snapshot| snapshot.kind == SnapshotKind::PromptEnd)
                            .map(|snapshot| snapshot.pos)
                            .max(),
                        shared_with_next,
                    }
                })
                .collect();
            status.state_committed_bytes = self.cache.count_model_bytes(model_index);
            status.state_limit_bytes = self.cache.limits[model_index].bytes;
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
        if let Some((model_index, number)) = lock(&shared.session_tokens_wanted).take() {
            *lock(&shared.session_tokens) = find_session_tokens(&self.cache, model_index, number)
                .map(|tokens| (model_index, number, tokens));
        }
        *lock(&shared.disk_status) = match &self.store {
            Some(store) => DiskStatus {
                dir: Some(store.get_folder().display().to_string()),
                states: self.cache.saved_states.len(),
                bytes: self.cache.count_saved_bytes(),
                limit: self.cache.saved_byte_limit,
                loads: store.load_count,
                saves: store.save_count,
                written: store.bytes_written,
            },
            None => DiskStatus::default(),
        };
    }
}

/// Orders `token_lists` by their tokens, as words are ordered by their letters.
/// Returns each list's index in that order, with the number of tokens that the
/// list shares from position 0 with the next one. The last one shares 0.
fn order_by_tokens(token_lists: &[&[i32]]) -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = (0..token_lists.len()).collect();
    order.sort_by(|&a, &b| token_lists[a].cmp(token_lists[b]));
    order
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let shared = order.get(k + 1).map_or(0, |&next| {
                state_cache::count_shared_prefix(token_lists[i], token_lists[next])
            });
            (i, shared)
        })
        .collect()
}

/// Returns a copy of the tokens of session `number` of the model at
/// `model_index`, or None when `cache` keeps no such session.
fn find_session_tokens(cache: &Cache, model_index: usize, number: u64) -> Option<Vec<i32>> {
    cache
        .entries
        .values()
        .find(|entry| entry.model == model_index && entry.number == number)
        .map(|entry| entry.tokens.clone())
}

/// The worker on the engine thread.
struct Worker {
    /// What the threads share.
    shared: Arc<Shared>,
    /// The models that the worker serves, in their configured order.
    models: Vec<WorkerModel>,
    /// The states of all models.
    states: States,
    /// When the worker last kept the memory reserve free.
    reserve_kept_free_at: Instant,
}

impl Worker {
    /// Creates the worker for the models in `shared` and their engine handles.
    /// It opens the store of saved states in `state_dir`, if given.
    fn new(
        shared: Arc<Shared>,
        model_handles: Vec<Rc<Model>>,
        state_dir: Option<PathBuf>,
    ) -> Worker {
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
                    is_decision_model: model.is_decision_model(),
                    admitted_decisions: VecDeque::new(),
                    decision: None,
                    is_image_model: model.is_image_model(),
                    admitted_images: VecDeque::new(),
                    image: None,
                    think_start_token: model.tokenizer().find_added_token("<think>"),
                    think_end_token: model.tokenizer().find_added_token("</think>"),
                    logged_decoding_count: 0,
                    experts_per_row: model.read_stats().experts_per_row.max(0) as usize,
                    view: shared.view,
                    last_used_at: Instant::now(),
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
        let limits = models
            .iter()
            .map(|worker_model| ModelLimits {
                bytes: 0,
                states: worker_model.settings.max_cached_states,
            })
            .collect();
        let mut states = States::new(limits, shared.budgets.snapshot_interval);
        for worker_model in &models {
            worker_model.update_state_room(&mut states);
        }
        if let Some(dir) = state_dir {
            Self::open_store(&mut states, &models, &shared.budgets, &dir);
        }
        let mut worker = Worker {
            shared,
            models,
            states,
            reserve_kept_free_at: Instant::now(),
        };
        worker.states.publish_states(&worker.shared);
        worker
    }

    /// Opens the store of saved states in `dir`. A saved state with a model's
    /// identity can serve that model's prompts. The other saved states stay
    /// until their room is needed. When the saved states take more than the
    /// disk limit, some are deleted now.
    fn open_store(
        states: &mut States,
        models: &[WorkerModel],
        budgets: &crate::config::Budgets,
        dir: &std::path::Path,
    ) {
        let (store, found_states) = match Store::open(dir) {
            Ok(opened) => opened,
            Err(message) => {
                engine::log(&format!("states: no saved states: {message}"));
                return;
            }
        };
        let identities: Vec<_> = models
            .iter()
            .map(|worker_model| worker_model.model.state_identity())
            .collect();
        states.cache.saved_byte_limit = (budgets.state_disk_gb * GIB) as u64;
        states.cache.min_unshared_tokens = budgets.state_disk_tokens;
        let (mut usable, mut tokens) = (0, 0);
        for mut found in found_states {
            found.saved_state.model = identities
                .iter()
                .position(|identity| *identity == found.identity);
            if found.saved_state.model.is_some() {
                usable += 1;
                tokens += found.saved_state.tokens.len();
            }
            states.cache.add_saved_state(found.id, found.saved_state);
        }
        engine::log(&format!(
            "states: saved states in {}: {} ({:.1} GiB of {:.1} GiB); {usable} of them, {tokens} tokens, for the models here",
            dir.display(),
            states.cache.saved_states.len(),
            states.cache.count_saved_bytes() as f64 / GIB,
            budgets.state_disk_gb
        ));
        states.store = Some(store);
        // The saved states may take more than the disk limit. Delete them until
        // they fit: first those of no model here, then the oldest.
        if let (Some(to_delete), Some(store)) = (
            states.cache.find_saved_states_to_delete(0, None, None),
            states.store.as_mut(),
        ) {
            for saved_id in to_delete {
                store.delete_saved_state(&mut states.cache, saved_id);
            }
        }
    }

    /// Deletes the states of the sessions in `asked` from memory, without
    /// saving them to the SSD, and logs what it did. Each session comes as its
    /// model's index and its number.
    fn delete_session_states(&mut self, asked: Vec<(usize, u64)>) {
        for (model_index, number) in asked {
            let line = self.states.delete_session_state(model_index, number);
            self.shared
                .view
                .write_model_line(model_index, &format!("states: {line}"));
        }
    }

    /// Saves every state to the SSD at the stop, after the requests ended. The
    /// state cache decides what each save writes. The most recently used state
    /// goes first. A state that the stop cut short is not saved. A second stop
    /// request ends the saving.
    fn save_states_at_stop(&mut self) {
        if self.states.store.is_none() {
            return;
        }
        let mut order: Vec<(u64, state_cache::StateId)> = self
            .states
            .cache
            .entries
            .iter()
            .map(|(&state_id, entry)| (entry.last_used, state_id))
            .collect();
        order.sort_unstable_by(|left, right| right.cmp(left));
        let (count, started_at) = (order.len(), Instant::now());
        for (handled, (_, state_id)) in order.into_iter().enumerate() {
            // The second stop request ends the saving.
            if engine::count_stop_requests() >= 2 {
                engine::log(&format!(
                    "states: a second stop: {} states not saved",
                    count - handled
                ));
                break;
            }
            // Take the state out of the cache. The states not saved yet do not
            // hold its tokens for it.
            let Some(entry) = self.states.cache.entries.remove(&state_id) else {
                continue;
            };
            self.states
                .save_state(&self.shared, state_id, &entry, false);
        }
        if count > 0 {
            engine::log(&format!(
                "states: {count} states considered at quit in {:.1} s",
                started_at.elapsed().as_secs_f64()
            ));
        }
    }

    /// Unloads each model that had no request for its idle_unload seconds and
    /// has none waiting. It frees the model's expert cache and weights. The
    /// model's states stay. Its next request loads it again. Nothing unloads
    /// during a yield.
    fn unload_idle_models(&mut self) {
        let waiting: Vec<bool> = {
            let queue = lock(&self.shared.queue);
            if queue.control.phase != Phase::Running {
                return;
            }
            (0..queue.waiting.len())
                .map(|i| {
                    !queue.waiting[i].is_empty()
                        || !queue.decisions[i].is_empty()
                        || !queue.images[i].is_empty()
                })
                .collect()
        };
        let mut drained = false;
        for worker_model in &self.models {
            let after = worker_model.settings.idle_unload;
            if after <= 0.0
                || worker_model.runs_requests()
                || worker_model.has_admitted_requests()
                || waiting
                    .get(worker_model.model_index)
                    .copied()
                    .unwrap_or(false)
                || worker_model.last_used_at.elapsed().as_secs_f64() < after
                || !worker_model.model.is_loaded()
            {
                continue;
            }
            let _long = LongOperation::start(&self.shared);
            if !drained {
                // Wait for the GPU work and the reads. No running work then
                // reads the memory that the unload frees.
                engine::drain();
                drained = true;
            }
            let freed = worker_model.model.unload();
            if freed > 0 {
                worker_model.log_line(&format!(
                    "{}: unloaded after {after:.0} s without requests; {:.1} GiB back",
                    worker_model.settings.id,
                    freed as f64 / GIB
                ));
            }
        }
    }

    /// Keeps the memory reserve free, between engine calls. It first unloads
    /// idle models. A model with routed experts then runs its idle work.
    /// Another model runs its idle work only while it has no request. The idle
    /// work frees memory that other programs need. When memory is still short,
    /// the engine frees idle memory. After that, the state cache evicts idle
    /// states. Last, it publishes the counters.
    fn keep_reserve_free(&mut self) {
        self.unload_idle_models();
        engine::refresh_stats();
        let mut freed = 0;
        for worker_model in &self.models {
            if worker_model.experts_per_row > 0 {
                let before = worker_model.model.read_stats().cache_bytes;
                worker_model.model.run_idle_work();
                freed += before.saturating_sub(worker_model.model.read_stats().cache_bytes);
            } else if !worker_model.runs_requests() {
                worker_model.model.run_idle_work();
            }
        }
        if let Some(worker_model) = self.models.first() {
            let available = worker_model.model.read_stats().available;
            let reserve = (self.shared.budgets.reserve_gb * GIB) as u64;
            // Freed expert pages can take a moment to show in the system's
            // memory counters. Count them here. This turn then does not free
            // the same shortfall twice.
            let shortfall = reserve.saturating_sub(available).saturating_sub(freed);
            let shortfall = shortfall.saturating_sub(engine::reclaim(shortfall));
            if shortfall > 0 {
                let evicted = self.states.cache.evict_bytes(shortfall);
                self.states.free_evicted(&self.shared, evicted);
                self.states.publish_states(&self.shared);
            }
        }
        self.publish_counters();
        self.reserve_kept_free_at = Instant::now();
    }

    /// Returns how long the worker may wait before it keeps the memory reserve
    /// free again: the engine's idle work expects a call once per
    /// available-memory reading interval.
    fn wait_before_keeping_reserve(&self) -> Duration {
        Duration::from_secs_f64(engine::idle_work_interval_seconds())
            .saturating_sub(self.reserve_kept_free_at.elapsed())
    }

    /// Publishes each model's engine counters to its status, and the memory in
    /// use to the control.
    fn publish_counters(&self) {
        for worker_model in &self.models {
            let mut status = lock(&self.shared.models[worker_model.model_index].status);
            let stats = worker_model.model.read_stats();
            status.stats = stats;
            status.loaded = worker_model.model.is_loaded();
            status.drafts_per_pass = worker_model.model.drafts_per_pass();
            if let Some(image) = &worker_model.image {
                let live_phase = WorkerModel::active_image_phase(&stats);
                let live_prompt = WorkerModel::live_image_prompt_tokens(image, &stats);
                if status.phase != "loading" {
                    if status.phase != live_phase || status.phase_since.is_none() {
                        status.phase_since = Some(Instant::now());
                    }
                    status.phase = live_phase;
                }
                status.shown_prompt_tokens = live_prompt;
                status.shown_generated_tokens = 0;
                if let Some(request) = status.requests.iter_mut().find(|r| r.id == image.job.id) {
                    request.status = live_phase;
                    request.prompt_tokens = live_prompt;
                    request.generated_images = image.outputs.len();
                }
            }
        }
        let memory_in_use = engine::read_memory();
        lock(&self.shared.queue).control.retained = control::MemoryBytes {
            model_bytes: memory_in_use.model_bytes,
            state_bytes: memory_in_use.state_bytes,
        };
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
                !worker_model.running_calls.is_empty() || worker_model.has_admitted_requests()
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

    /// Restarts each model that has running requests. It loads the model when
    /// needed, and sets it as running requests. A model that fails to load ends
    /// its requests with the error.
    fn restart_models_with_running_requests(&mut self) {
        for worker_model in &mut self.models {
            if !worker_model.runs_requests() {
                continue;
            }
            if !worker_model.model.is_loaded()
                && let Err(error) = load_model(&self.shared, &worker_model.model)
            {
                if engine::is_stopping() {
                    // The stop cut the load short. The stop also ends the
                    // model's requests.
                    continue;
                }
                lock(&self.shared.models[worker_model.model_index].status).load_error =
                    Some(error.clone());
                for generation in std::mem::take(&mut worker_model.generations) {
                    let snapshots = self.states.cache.remove_state(generation.state_id);
                    self.states.free_snapshots_and_states(
                        Vec::new(),
                        vec![(generation.state_id, snapshots)],
                    );
                    generation.job.send_last_event(Event::Failed(error.clone()));
                }
                if let Some(mut decision) = worker_model.decision.take() {
                    decision.failure = Some(error.clone());
                    worker_model.end_decision(&self.shared, decision);
                }
                if let Some(mut image) = worker_model.image.take() {
                    image.failure = Some(ImageError {
                        message: error.clone(),
                        invalid_input: false,
                    });
                    worker_model.end_image(&self.shared, image);
                }
                continue;
            }
            worker_model.model.set_request_running(true);
        }
    }

    /// Advances the running calls of every model until one ends, and finishes
    /// the call that ended. A new request, a cancellation, a control request or
    /// the stop wakes the engine, so that the worker serves it without waiting
    /// for a call to end.
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

    /// Runs a yield or a resume. It first finishes the running calls, and waits
    /// for the GPU work and the reads. A yield stops every model from running
    /// requests. It frees the models' idle memory, then their idle states, down
    /// to the target. If more memory than the target is still in use, the yield
    /// fails and the models restart. Otherwise the lower limit takes effect. A
    /// resume restores the configured limit and restarts the models. Each
    /// model's state room then follows the limit.
    fn run_control_action(&mut self, action: Action) {
        let shared = self.shared.clone();
        let _long = LongOperation::start(&shared);
        self.finish_running_calls();
        let configured = lock(&self.shared.queue).control.configured_limit;
        engine::drain();
        match action {
            Action::Yield(target) => {
                for worker_model in &self.models {
                    worker_model.model.set_request_running(false);
                }
                // The models' idle memory goes first, then their idle states,
                // the least recently used first.
                let before = engine::read_memory();
                let excess = match target {
                    // u64::MAX frees all that can be freed.
                    YieldTarget::Minimum => u64::MAX,
                    YieldTarget::Bytes(cap) => before.model_bytes.saturating_sub(cap),
                };
                let freed = engine::reclaim(excess);
                let evicted = self.states.cache.evict_bytes(excess.saturating_sub(freed));
                self.states.free_evicted(&self.shared, evicted);
                engine::drain();
                let memory_in_use = engine::read_memory();
                let retained = control::MemoryBytes {
                    model_bytes: memory_in_use.model_bytes,
                    state_bytes: memory_in_use.state_bytes,
                };
                let limit = match target {
                    YieldTarget::Minimum => memory_in_use.model_bytes,
                    YieldTarget::Bytes(cap) => cap,
                };
                if memory_in_use.model_bytes > limit {
                    self.restart_models_with_running_requests();
                    let memory_in_use = engine::read_memory();
                    let message = format!(
                        "the limit cannot retain protected buffers: {} model bytes are required, {} of them for states",
                        retained.model_bytes, retained.state_bytes
                    );
                    lock(&self.shared.queue).control.fail_yield(
                        message,
                        control::MemoryBytes {
                            model_bytes: memory_in_use.model_bytes,
                            state_bytes: memory_in_use.state_bytes,
                        },
                    );
                } else {
                    engine::set_memory_limit(limit);
                    lock(&self.shared.queue)
                        .control
                        .finish_yield(limit, retained);
                    engine::log(&format!(
                        "yielded: {} model bytes retained, {} of them for states",
                        retained.model_bytes, retained.state_bytes
                    ));
                }
            }
            Action::Resume => {
                engine::set_memory_limit(configured);
                lock(&self.shared.queue).control.effective_limit = configured;
                self.restart_models_with_running_requests();
                let memory_in_use = engine::read_memory();
                lock(&self.shared.queue)
                    .control
                    .finish_resume(control::MemoryBytes {
                        model_bytes: memory_in_use.model_bytes,
                        state_bytes: memory_in_use.state_bytes,
                    });
                engine::log("resumed: the configured memory limit holds again");
            }
        }
        // Each model's state room follows the limit.
        for worker_model in &self.models {
            worker_model.update_state_room(&mut self.states);
        }
        engine::refresh_stats();
        self.states.publish_states(&self.shared);
        self.publish_counters();
    }

    /// Runs the worker's loop until the stop. Each turn gives every model one
    /// step. A step admits requests, starts a prefill call and starts decoding
    /// calls. Once per available-memory reading interval, the loop keeps the
    /// memory reserve free between calls. At the stop, every request ends with
    /// FinishReason::ServerStop.
    fn run_until_stop(&mut self) {
        let shared = self.shared.clone();
        while !engine::is_stopping() {
            if self.wait_before_keeping_reserve().is_zero() {
                self.keep_reserve_free();
            }
            self.publish_status_if_wanted();
            let mut queue = lock(&shared.queue);
            if let Some(action) = queue.control.take_next_action() {
                drop(queue);
                self.run_control_action(action);
                continue;
            }
            if !queue.states_to_delete.is_empty() {
                let asked = std::mem::take(&mut queue.states_to_delete);
                drop(queue);
                self.delete_session_states(asked);
                continue;
            }
            if queue.control.phase != Phase::Running {
                // A status reader's request is published at the loop's start,
                // and so is the reserve's keeping once its interval has passed.
                if !shared.status_wanted.load(Ordering::Acquire) {
                    let (guard, _) = shared
                        .worker_wakeup
                        .wait_timeout(queue, self.wait_before_keeping_reserve())
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    drop(guard);
                }
                continue;
            }
            let idle = |queue: &Queue, models: &[WorkerModel]| {
                queue.waiting.iter().all(|jobs| jobs.is_empty())
                    && queue.decisions.iter().all(|jobs| jobs.is_empty())
                    && queue.images.iter().all(|jobs| jobs.is_empty())
                    && queue.states_to_delete.is_empty()
                    && queue.control.phase == Phase::Running
                    && models.iter().all(|worker_model| {
                        !worker_model.runs_requests() && !worker_model.has_admitted_requests()
                    })
            };
            while idle(&queue, &self.models) && !engine::is_stopping() {
                if shared.status_wanted.load(Ordering::Acquire) {
                    drop(queue);
                    self.publish_status_if_wanted();
                    queue = lock(&shared.queue);
                    continue;
                }
                // A wake before the interval's end, such as a status read, does
                // not delay the reserve's keeping.
                let wait = self.wait_before_keeping_reserve();
                if wait.is_zero() {
                    drop(queue);
                    self.keep_reserve_free();
                    queue = lock(&shared.queue);
                    continue;
                }
                let (guard, _) = shared
                    .worker_wakeup
                    .wait_timeout(queue, wait)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                queue = guard;
            }
            // A deletion runs at the loop's start.
            if queue.control.phase != Phase::Running || !queue.states_to_delete.is_empty() {
                continue;
            }
            /// Requests drained from the queue for one model in one worker turn.
            struct AdmittedModelJobs {
                /// Queued generation requests for the model.
                generations: Vec<Job>,
                /// Queued decision requests for the model.
                decisions: Vec<Job<DecisionRequest>>,
                /// Queued image requests for the model.
                images: Vec<Job<ImageRequest>>,
            }
            let mut admitted_by_model = Vec::with_capacity(self.models.len());
            for worker_model in &self.models {
                let index = worker_model.model_index;
                let batch = if engine::is_stopping() {
                    AdmittedModelJobs {
                        generations: Vec::new(),
                        decisions: Vec::new(),
                        images: Vec::new(),
                    }
                } else {
                    AdmittedModelJobs {
                        generations: queue.waiting[index].drain(..).collect(),
                        decisions: queue.decisions[index].drain(..).collect(),
                        images: queue.images[index].drain(..).collect(),
                    }
                };
                if !batch.generations.is_empty()
                    || !batch.decisions.is_empty()
                    || !batch.images.is_empty()
                {
                    worker_model.log_line(&format!(
                        "{}: admitted={}, running={}",
                        worker_model.settings.id,
                        batch.generations.len() + batch.decisions.len() + batch.images.len(),
                        worker_model.generations.len()
                            + usize::from(worker_model.decision.is_some())
                            + usize::from(worker_model.image.is_some())
                    ));
                }
                admitted_by_model.push(batch);
            }
            drop(queue);
            for (worker_model, batch) in self.models.iter_mut().zip(admitted_by_model) {
                let AdmittedModelJobs {
                    generations: admitted,
                    decisions,
                    images,
                } = batch;
                if worker_model.is_decision_model {
                    for job in images {
                        worker_model.refuse_image(
                            job,
                            "the model is a decision model: it generates no images",
                        );
                    }
                    worker_model.run_decision_step(&shared, admitted, decisions);
                } else if worker_model.is_image_model {
                    worker_model.run_image_step(&shared, admitted, decisions, images);
                } else {
                    for job in decisions {
                        worker_model.refuse_decision(
                            job,
                            "the model generates text: it makes no decisions",
                        );
                    }
                    for job in images {
                        worker_model
                            .refuse_image(job, "the model generates text: it generates no images");
                    }
                    worker_model.run_step(&shared, admitted, &mut self.states);
                }
                if worker_model.runs_requests() {
                    worker_model.last_used_at = Instant::now();
                }
                worker_model
                    .model
                    .set_request_running(worker_model.runs_requests());
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
            if let Some(decision) = worker_model.decision.take() {
                worker_model.end_decision(&shared, decision);
            }
            if let Some(image) = worker_model.image.take() {
                worker_model.end_image(&shared, image);
            }
        }
        let mut waiting: Vec<Job>;
        let decisions: Vec<Vec<Job<DecisionRequest>>>;
        let images: Vec<Vec<Job<ImageRequest>>>;
        {
            let mut queue = lock(&shared.queue);
            queue.closed = true;
            waiting = queue
                .waiting
                .iter_mut()
                .flat_map(|jobs| jobs.drain(..))
                .collect();
            decisions = queue
                .decisions
                .iter_mut()
                .map(|jobs| jobs.drain(..).collect())
                .collect();
            images = queue
                .images
                .iter_mut()
                .map(|jobs| jobs.drain(..).collect())
                .collect();
        }
        // The deletions asked before the stop run before the stop saves the
        // states. No request runs now, and the closed queue takes no more.
        let states_to_delete = std::mem::take(&mut lock(&shared.queue).states_to_delete);
        self.delete_session_states(states_to_delete);
        for ((worker_model, decisions), images) in self.models.iter_mut().zip(decisions).zip(images)
        {
            waiting.extend(worker_model.admitted.drain(..));
            let ended_decisions: Vec<Job<DecisionRequest>> = worker_model
                .admitted_decisions
                .drain(..)
                .chain(decisions)
                .collect();
            for job in ended_decisions {
                worker_model.end_waiting_decision(job, FinishReason::ServerStop);
            }
            let ended_images: Vec<Job<ImageRequest>> = worker_model
                .admitted_images
                .drain(..)
                .chain(images)
                .collect();
            for job in ended_images {
                worker_model.end_waiting_image(job, FinishReason::ServerStop);
            }
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
    log_model_line(
        job.request.model_index,
        &format!("req {}: refused: {message}", job.id),
    );
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
    log_model_line(
        job.request.model_index,
        &format!(
            "req {}: ended before prefill ({}), waited {:.3} s",
            job.id,
            if reason == FinishReason::Cancelled {
                "cancelled"
            } else {
                "shutdown"
            },
            job.queued_at.elapsed().as_secs_f64()
        ),
    );
    job.send_last_event(Event::Done { reason });
}

impl WorkerModel {
    /// Writes `line`, a line about the model, to the log.
    fn log_line(&self, line: &str) {
        self.view.write_model_line(self.model_index, line);
    }

    /// Returns whether the model runs requests: generations, a decision or an
    /// image task.
    fn runs_requests(&self) -> bool {
        !self.generations.is_empty() || self.decision.is_some() || self.image.is_some()
    }

    /// Returns whether admitted requests, decisions or image tasks wait for
    /// their turn.
    fn has_admitted_requests(&self) -> bool {
        !self.admitted.is_empty()
            || !self.admitted_decisions.is_empty()
            || !self.admitted_images.is_empty()
    }

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
    /// its first token. It counts the snapshot that a load reads. It counts
    /// that snapshot with logits. For the same prompt again, the snapshot holds
    /// them. It also counts the snapshots that the plan takes.
    fn count_plan_bytes(&self, plan: &Plan, prompt_tokens: usize, states: &States) -> u64 {
        let growth = match *plan {
            Plan::InPlace { state, .. } => self
                .model
                .state_bytes(Some(states.handles[&state].engine_state), prompt_tokens + 1)
                .saturating_sub(states.cache.entries[&state].bytes),
            _ => self.model.state_bytes(None, prompt_tokens + 1),
        };
        let loaded_snapshot = match plan {
            Plan::Load {
                snapshot_pos: Some(_),
                ..
            } => self.model.snapshot_bytes(true),
            _ => 0,
        };
        growth
            + loaded_snapshot
            + plan.count_snapshot_bytes(
                prompt_tokens,
                self.model.snapshot_bytes(false),
                self.model.snapshot_bytes(true),
            )
    }

    /// Sets the model's byte limit in the state cache to its state room: the
    /// part of its own memory limit, or else of the memory limit now, that a
    /// full prompt pass leaves to its states. Before the model's first load, a
    /// full prompt pass's memory is unknown, and the room is the whole limit.
    fn update_state_room(&self, states: &mut States) {
        let stats = self.model.read_stats();
        let limit = match stats.own_memory_limit {
            0 => stats.model_memory_limit,
            own => own.min(stats.model_memory_limit),
        };
        states.cache.limits[self.model_index].bytes = limit.saturating_sub(stats.required_bytes);
    }

    /// Runs one step of the model. Admitted requests start, one at a time after
    /// the prefills before them. One prefill call starts. The generations that
    /// are done end. The others start decoding calls.
    fn run_step(&mut self, shared: &Shared, admitted: Vec<Job>, states: &mut States) {
        self.update_state_room(states);
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
                self.log_line(&format!("req {}: client gone while waiting", job.id));
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
                    self.log_line(&format!("{}: {error}", self.settings.id));
                    job.send_last_event(Event::Failed(error));
                    continue;
                }
                lock(&shared.models[self.model_index].status).load_error = None;
                self.last_used_at = Instant::now();
                self.experts_per_row = self.model.read_stats().experts_per_row.max(0) as usize;
                self.update_state_room(states);
                if !was_loaded && self.model.is_loaded() {
                    self.log_line(&format!(
                        "{}: loaded; its states may take {:.1} GiB",
                        self.settings.id,
                        states.cache.limits[self.model_index].bytes as f64 / GIB
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
            self.log_line(&format!(
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
            self.start_decode_calls(shared, states);
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
                self.view.write_model_line(
                    self.model_index,
                    &format!(
                        "req {}: {} tokens, {:.1} tokens/s",
                        generation.job.id,
                        generation.generated,
                        generation.generated as f64
                            / (generation.progress_logged_at - generation.decode_started_at)
                                .as_secs_f64()
                    ),
                );
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

    /// Stops timing the running passes. A call that starts beside them changes
    /// their times. The draft choice then learns nothing from them.
    fn stop_timing_passes(&mut self) {
        for generation in &mut self.generations {
            generation.timed_pass = None;
        }
    }

    /// Makes room in the state cache for state `state_id` to hold `positions`
    /// positions, before a pass grows it. The cache counts the state's new size
    /// now. The worker frees what the cache evicts for it. With only decoding
    /// states left, the cache can go over its limit. The worker then logs the
    /// sizes.
    fn make_room_to_grow(
        &mut self,
        shared: &Shared,
        state_id: state_cache::StateId,
        positions: usize,
        states: &mut States,
    ) {
        let before = states.cache.entries[&state_id].bytes;
        let evicted = states.cache.grow_state(
            state_id,
            self.model
                .state_bytes(Some(states.handles[&state_id].engine_state), positions),
        );
        states.free_evicted(shared, evicted);
        let after = states.cache.entries[&state_id].bytes;
        if after > before && states.cache.count_bytes_over_limit(self.model_index) > 0 {
            self.log_line(&format!(
                "{}: a decoding state grows to {:.2} GiB at {positions} positions; with only decoding states left, the model's states take {:.2} GiB of the {:.1} GiB they may",
                self.settings.id,
                after as f64 / GIB,
                states.cache.count_model_bytes(self.model_index) as f64 / GIB,
                states.cache.limits[self.model_index].bytes as f64 / GIB
            ));
        }
        if after > before {
            states.publish_states(shared);
        }
    }

    /// Updates the model's status: its phase, its counts, the request it shows
    /// and its requests.
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
        let image_stats = self.image.as_ref().map(|_| self.model.read_stats());
        (
            status.shown_request_id,
            status.shown_prompt_tokens,
            status.shown_reused_tokens,
            status.shown_generated_tokens,
        ) = match (first, &self.decision, &self.image) {
            (Some(generation), _, _) => (
                generation.job.id,
                generation.prompt_tokens,
                generation.reused_tokens,
                generation.generated,
            ),
            (None, Some(decision), _) => {
                (decision.job.id, decision.job.request.count_tokens(), 0, 0)
            }
            (None, None, Some(image)) => (
                image.job.id,
                Self::live_image_prompt_tokens(image, image_stats.as_ref().unwrap()),
                0,
                0,
            ),
            (None, None, None) => (0, 0, 0, 0),
        };
        status.requests = self
            .generations
            .iter()
            .map(|generation| RequestStatus {
                id: generation.job.id,
                status: if generation.is_decoding() {
                    "decoding"
                } else {
                    "prefill"
                },
                session: Some(generation.session),
                prompt_tokens: generation.prompt_tokens,
                reused_tokens: generation.reused_tokens,
                generated_tokens: generation.generated,
                images_requested: 0,
                generated_images: 0,
                routes: generation.routes.iter().cloned().collect(),
            })
            .chain(
                self.decision
                    .iter()
                    .map(|decision| RequestStatus::for_decision(&decision.job, "prefill")),
            )
            .chain(self.image.iter().map(|image| {
                let stats = image_stats.as_ref().unwrap();
                RequestStatus::for_image(
                    &image.job,
                    Self::active_image_phase(stats),
                    Self::live_image_prompt_tokens(image, stats),
                    image.outputs.len(),
                )
            }))
            .chain(self.admitted.iter().map(RequestStatus::for_waiting_job))
            .chain(
                self.admitted_decisions
                    .iter()
                    .map(|job| RequestStatus::for_decision(job, "waiting")),
            )
            .chain(
                self.admitted_images
                    .iter()
                    .map(|job| RequestStatus::for_image(job, "waiting", 0, 0)),
            )
            .collect();
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
            if let Some(eviction) = states.cache.find_eviction(
                self.model_index,
                need,
                !matches!(plan, Plan::InPlace { .. }),
                &keep,
            ) {
                chosen = Some((plan, eviction));
                break;
            }
        }
        let Some((plan, eviction)) = chosen else {
            let busy: u64 = states
                .cache
                .entries
                .values()
                .filter(|entry| entry.busy && entry.model == self.model_index)
                .map(|entry| entry.bytes)
                .sum();
            // The state's bytes count the prompt and its first token.
            let message = format!(
                "the conversation's attention state needs {:.1} GiB ({:.1} GiB with its snapshots); the model's states may take {:.1} GiB, and its requests running take {:.1} GiB of it",
                self.model.state_bytes(None, prompt_tokens + 1) as f64 / GIB,
                need as f64 / GIB,
                states.cache.limits[self.model_index].bytes as f64 / GIB,
                busy as f64 / GIB
            );
            return refuse(job, &message);
        };
        let origin = states.cache.find_origin(&plan, prompt_tokens);
        let capture_position = plan.find_branch_snapshot_pos(prompt_tokens);
        // Pin the saved state that the plan reads, and note its version. The
        // saves that the eviction below makes then leave it unchanged.
        let planned = match plan {
            Plan::Load {
                saved_id: saved, ..
            } => {
                states.cache.pinned = Some(saved);
                states
                    .cache
                    .saved_states
                    .get(&saved)
                    .map(|saved_state| saved_state.version)
            }
            _ => None,
        };
        let wait = job.queued_at.elapsed().as_secs_f64();
        let evicted = states.cache.evict(&eviction);
        states.free_evicted(shared, evicted);
        if let Plan::Load {
            saved_id: saved, ..
        } = plan
            && states
                .cache
                .saved_states
                .get(&saved)
                .map(|saved_state| saved_state.version)
                != planned
        {
            states.cache.pinned = None;
            // It changed during the eviction: plan again.
            return self.start_request(shared, job, states);
        }
        let reused_tokens = plan.count_reused_tokens();
        // The plan's state: the one that it continues in place, or a new one. A
        // request whose new state does not fit is refused.
        let planned_state = match plan {
            Plan::InPlace { state, .. } => state,
            _ => match states.add_state(self.model_index, self.model.clone()) {
                Some(state_id) => state_id,
                None => {
                    states.cache.pinned = None;
                    return refuse(job, STATE_MEMORY_SHORT);
                }
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
                states.cache.copy_saved_link(state_id, source, at);
                states.handles.get_mut(&source).expect("state").last_used_at = Instant::now();
                if let Plan::Snapshot { snapshot_pos, .. } = plan {
                    self.model
                        .snapshot_restore(engine_state, states.get_snapshot(source, snapshot_pos));
                }
                state_id
            }
            Plan::Load {
                saved_id: saved,
                snapshot_pos,
                from: at,
                ..
            } => {
                let state_id = planned_state;
                let engine_state = states.handles[&state_id].engine_state;
                let load_started_at = Instant::now();
                let store = states
                    .store
                    .as_mut()
                    .expect("saved states have their store");
                let long = LongOperation::start(shared);
                let read = store.load_saved_state(
                    &mut states.cache,
                    &EngineState {
                        model: &self.model,
                        state: engine_state,
                    },
                    saved,
                    snapshot_pos,
                    at,
                );
                drop(long);
                states.cache.pinned = None;
                match read {
                    Ok(snapshot) => {
                        states.cache.link_to_saved_state(state_id, saved, at);
                        if let Some(snapshot) = snapshot {
                            if at == prompt_tokens {
                                // The same prompt again. The snapshot is
                                // restored below.
                                states.insert_snapshot(state_id, prompt_tokens, snapshot);
                            } else {
                                self.model.snapshot_restore(engine_state, snapshot);
                                self.model.snapshot_free(snapshot);
                            }
                        }
                        self.log_line(&format!(
                            "req {}: {at} positions of saved state {saved} read in {:.2} s",
                            job.id,
                            load_started_at.elapsed().as_secs_f64()
                        ));
                        state_id
                    }
                    Err(error) => {
                        // Plan again without this saved state. A damaged one is
                        // deleted. One that cannot be read now is skipped for
                        // this request.
                        match error {
                            LoadError::Damaged(message) => {
                                self.log_line(&format!(
                                    "req {}: saved state {saved} cannot be read ({message}); it is deleted",
                                    job.id
                                ));
                                store.delete_saved_state(&mut states.cache, saved);
                            }
                            LoadError::Unavailable(message) => {
                                self.log_line(&format!(
                                    "req {}: saved state {saved} cannot be read now ({message}); planned without it",
                                    job.id
                                ));
                                states.cache.skipped = Some(saved);
                            }
                        }
                        let snapshots = states.cache.remove_state(state_id);
                        states.free_snapshots_and_states(Vec::new(), vec![(state_id, snapshots)]);
                        self.start_request(shared, job, states);
                        states.cache.skipped = None;
                        return;
                    }
                }
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
        if let Plan::Load {
            from: at,
            snapshot_pos: Some(_),
            ..
        } = plan
            && at == prompt_tokens
        {
            states.cache.add_snapshot(
                state_id,
                Snapshot {
                    pos: prompt_tokens,
                    kind: SnapshotKind::PromptEnd,
                    bytes: self.model.snapshot_bytes(true),
                    stale_rows: state_cache::count_stale_rows_at(origin, prompt_tokens),
                },
            );
        }
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
        *lock(&shared.models[self.model_index].status)
            .plan_counts
            .entry(plan_kind)
            .or_default() += 1;
        let thinking = self.think_start_token == Some(prompt[prompt_tokens - 1]);
        self.log_line(&format!(
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
                self.log_line(&format!(
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
            session: number,
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
            draft_passes: 0,
            drafts_accepted: 0,
            lookup: None,
            lookup_tokens: prompt_tokens,
            lookup_passes: 0,
            lookup_offered: 0,
            lookup_accepted: 0,
            draft_choice: None,
            timed_pass: None,
            max_new_tokens,
            decode_started_at: now,
            progress_logged_at: now,
            expert_counts_at_decode_start: CacheCounts::default(),
            in_call: false,
            turn: 0,
            measured_token_at: None,
            routes: VecDeque::new(),
        };
        if reused_tokens == prompt_tokens {
            // The same prompt again: its prompt-end snapshot holds its logits.
            // A loaded saved state's prompt-end snapshot belongs to the new
            // state.
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
        self.stop_timing_passes();
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
                        .find_eviction(self.model_index, bytes, false, &[state_id])
                        .is_some_and(|eviction| {
                            let evicted = states.cache.evict(&eviction);
                            states.free_evicted(shared, evicted);
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
            Call::Verify {
                request_id,
                rows,
                proposed_count,
            } => self.finish_verify_call(stream, request_id, rows, proposed_count, states),
            Call::Decide { request_id } => self.finish_decide_call(stream, request_id),
            Call::GenerateImage { request_id } => self.finish_image_call(stream, request_id),
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
    /// `logits`. It also sets up the lookup and the draft choice.
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
            self.log_line(&format!(
                "req {}: prefill of {} tokens in {prefill_seconds:.2} s ({:.0} tokens/s), {read:.2} GB of experts read{replayed}",
                generation.job.id,
                prompt_tokens - reused_tokens,
                (prompt_tokens - reused_tokens) as f64 / prefill_seconds.max(1e-9)
            ));
            self.log_line(&format!(
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
            if reused_tokens < prompt_tokens {
                status.total_prefilled_tokens += (prompt_tokens - reused_tokens) as u64;
                status.total_prefill_seconds += prefill_seconds;
            }
        }
        drop(prefill.media);
        generation.next_token = self.model.sample(logits, &mut generation.sampler);
        if self.settings.options.lookup > 0 {
            let mut lookup =
                Lookup::new(self.ctx).unwrap_or_else(|| panic!("cannot initialize context lookup"));
            if !lookup.append(&prefill.prompt) {
                panic!("cannot initialize context lookup");
            }
            generation.lookup = Some(lookup);
        }
        if self.model.drafts_per_pass() > 0 {
            generation.draft_choice = Some(DraftChoice::new());
        }
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
        if let Some(lookup) = generation.lookup.as_mut()
            && generation.lookup_tokens == generation.pos
        {
            if !lookup.append(&[token]) {
                panic!("context lookup exceeds the configured context");
            }
            generation.lookup_tokens += 1;
        }
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
        if let Some(prefill) = &generation.prefill {
            // Do not save the state when the stop cut its prefill short. Do not
            // save it either when its replay never ran. Its rolling rows were
            // never rebuilt.
            if at_stop || (prefill.replays && prefill.next_position == generation.reused_tokens) {
                states.cut_short.insert(generation.state_id);
            }
            self.end_prefill_early(&mut generation, states);
        }
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
        states.free_evicted(shared, evicted);
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
            status.drafts_per_pass = self.model.drafts_per_pass();
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
        self.log_line(&format!(
            "req {}: {reason_text}: {} tokens ({} reasoning) in {decode_seconds:.1} s, {:.1} tokens/s{experts}; states {} sessions, {:.2} GiB; available {:.1} GiB, footprint {:.1} GiB",
            generation.job.id,
            generation.generated,
            generation.reasoning_tokens,
            generation.generated as f64 / decode_seconds.max(1e-9),
            states.cache.entries.len(),
            states.cache.count_bytes() as f64 / GIB,
            stats.available as f64 / GIB,
            stats.footprint as f64 / GIB
        ));
        let sampling = generation.job.request.sampling;
        self.log_line(&format!(
            "req {}: sampling temperature {}, top_p {}, top_k {}; {} draft passes, {} drafts accepted",
            generation.job.id,
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
            generation.draft_passes,
            generation.drafts_accepted
        ));
        if generation.lookup.is_some() {
            self.log_line(&format!(
                "req {}: context lookup {} passes, {} offered, {} accepted",
                generation.job.id,
                generation.lookup_passes,
                generation.lookup_offered,
                generation.lookup_accepted
            ));
        }
        let mut status = lock(&shared.models[self.model_index].status);
        status.total_requests += 1;
        status.total_prompt_tokens += generation.prompt_tokens as u64;
        status.total_generated_tokens += generation.generated as u64;
        status.total_reused_tokens += generation.reused_tokens as u64;
        drop(status);
        states.publish_states(shared);
    }

    /// Ends the requests of a decoding pass that could not load its experts.
    /// The stop may have ended their reads. Or memory stayed short after idle
    /// memory was freed. During the stop, the requests end as every request
    /// does at the stop. Their states are not saved. Otherwise the requests end
    /// with an error. The pass may have changed their states in part. So the
    /// states are freed.
    fn end_failed_pass(&mut self, generation_indices: &[usize], states: &mut States) {
        if engine::is_stopping() {
            for &i in generation_indices {
                // The pass may have changed the state in part. Do not save it.
                states.cut_short.insert(self.generations[i].state_id);
            }
            return;
        }
        let mut indices = generation_indices.to_vec();
        indices.sort_unstable_by(|left, right| right.cmp(left));
        for i in indices {
            let generation = self.generations.remove(i);
            let snapshots = states.cache.remove_state(generation.state_id);
            states.free_snapshots_and_states(Vec::new(), vec![(generation.state_id, snapshots)]);
            self.log_line(&format!(
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

    /// Starts decoding calls for the decoding requests with no call running. A
    /// request that decodes alone may run its drafts or proposals in its call.
    /// Several requests run together, at most the model's rows per call. Each
    /// row then gives what it would give alone.
    fn start_decode_calls(&mut self, shared: &Shared, states: &mut States) {
        // The call chooser sets how many calls run at once and the rows of
        // each. A call starts as soon as its requests are free and the chooser
        // allows one more call. Groups of requests run independently. The GPU
        // work of one call fills the waits of the others.
        let max_rows_per_call = self.model.max_rows().max(1);
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
        let ctx = self.ctx;
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
            let lookup_depth = self.settings.options.lookup as usize;
            // A verify pass's rows hold the token too.
            let verify_room = self.model.max_verify().saturating_sub(1);
            let (mut rows, proposed_count, draft_count, pos) = {
                let generation = &mut self.generations[index];
                let mut rows = vec![generation.next_token];
                // The drafts or proposals that the pass may run. Suppose the
                // pass keeps the token and all of them. The token sampled after
                // them must still fit the token limit. It must also fit a
                // position before the context's end. That token may go out
                // without a pass. The 2 counts the token and the token sampled
                // after them.
                let room = (generation.max_new_tokens as isize - generation.generated as isize - 2)
                    .min(ctx as isize - generation.pos as isize - 2)
                    .min(verify_room as isize);
                let mut proposed_count = 0;
                if let Some(lookup) = generation.lookup.as_mut() {
                    if generation.lookup_tokens != generation.pos
                        || !lookup.append(&[generation.next_token])
                    {
                        panic!("context lookup history is not aligned with the generation");
                    }
                    generation.lookup_tokens += 1;
                    if room > 0 {
                        let mut proposals = vec![0i32; self.model.max_verify()];
                        proposed_count =
                            lookup.propose(lookup_depth.min(room as usize), &mut proposals);
                        rows.extend_from_slice(&proposals[..proposed_count]);
                    }
                }
                let draft_count = if proposed_count > 0 {
                    generation.lookup_passes += 1;
                    generation.lookup_offered += proposed_count;
                    proposed_count
                } else {
                    let max_drafts = self.model.drafts_per_pass().min(room.max(0) as usize);
                    match generation.draft_choice.as_mut() {
                        Some(draft_choice) if max_drafts > 0 => draft_choice.next_count(max_drafts),
                        _ => max_drafts,
                    }
                };
                (rows, proposed_count, draft_count, generation.pos)
            };
            // The rows the pass writes: the token and its drafts or proposals.
            self.make_room_to_grow(shared, state_id, pos + 1 + draft_count, states);
            let stream = self.take_decode_stream();
            self.stop_timing_passes();
            let alone = self.running_calls.is_empty();
            let generation = &mut self.generations[index];
            generation.in_call = true;
            if alone && proposed_count == 0 && generation.draft_choice.is_some() {
                generation.timed_pass = Some((Instant::now(), draft_count));
            }
            let request_id = generation.job.id;
            let call = if draft_count > 0 {
                if proposed_count == 0 {
                    rows.resize(draft_count + 1, 0);
                }
                stream.verify(
                    engine_state,
                    &mut rows,
                    proposed_count > 0,
                    generation.pos,
                    &mut generation.sampler,
                );
                Call::Verify {
                    request_id,
                    rows,
                    proposed_count,
                }
            } else {
                stream.decode_rows(
                    &[engine_state],
                    &[generation.next_token],
                    &[generation.pos as i32],
                );
                Call::DecodeRows {
                    request_ids: vec![request_id],
                    tokens: vec![generation.next_token],
                }
            };
            self.running_calls.push(RunningCall { stream, call });
            return;
        }
        // Each group gets one call, while the chooser allows one more call.
        for group in decoding_indices
            .chunks(rows_per_call)
            .take(allowed - running)
        {
            for &i in group {
                let (state_id, positions) =
                    (self.generations[i].state_id, self.generations[i].pos + 1);
                self.make_room_to_grow(shared, state_id, positions, states);
            }
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
            self.stop_timing_passes();
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
        // The timed passes in this call. Their draft choice chose 0 drafts.
        let mut timed_passes = Vec::new();
        let turns = self.turns.take(indices.len());
        for (&i, turn) in indices.iter().zip(turns) {
            self.generations[i].in_call = false;
            self.generations[i].turn = turn;
            if let Some(timed_pass) = self.generations[i].timed_pass.take() {
                timed_passes.push((i, timed_pass));
            }
        }
        let Some(logits) = stream.logits() else {
            self.idle_decode_streams.push(stream);
            self.end_failed_pass(&indices, states);
            return;
        };
        for (i, (started_at, draft_count)) in timed_passes {
            if let Some(draft_choice) = self.generations[i].draft_choice.as_mut() {
                draft_choice.record_pass(
                    draft_count,
                    0,
                    ended_at.duration_since(started_at).as_secs_f64(),
                );
            }
        }
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
        for (row, &i) in indices.iter().enumerate() {
            self.keep_routes(i, row);
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

    /// Finishes a verify call of one request. The request keeps its token and
    /// the accepted drafts. They go out with the next Tokens event.
    fn finish_verify_call(
        &mut self,
        stream: Stream,
        request_id: u64,
        rows: Vec<i32>,
        proposed_count: usize,
        states: &mut States,
    ) {
        let index = self.find_generation_index(request_id);
        self.generations[index].in_call = false;
        self.generations[index].turn = self.turns.take(1).start;
        let timed_pass = self.generations[index].timed_pass.take();
        if stream.logits().is_none() {
            self.idle_decode_streams.push(stream);
            self.end_failed_pass(&[index], states);
            return;
        }
        let (next, accepted) = stream.next_token();
        self.idle_decode_streams.push(stream);
        let model = self.model.clone();
        let think_end = self.think_end_token;
        let generation = &mut self.generations[index];
        generation.draft_passes += 1;
        generation.drafts_accepted += accepted;
        if let (Some((started_at, draft_count)), Some(draft_choice)) =
            (timed_pass, generation.draft_choice.as_mut())
        {
            draft_choice.record_pass(draft_count, accepted, started_at.elapsed().as_secs_f64());
        }
        if proposed_count > 0
            && let Some(lookup) = generation.lookup.as_mut()
        {
            // The drafts offered: every row but the token's.
            lookup.record_verification(rows.len() - 1, accepted);
            generation.lookup_accepted += accepted;
        }
        let mut kept = 0;
        for &token in &rows[..=accepted] {
            if generation.reached_stop_token {
                break;
            }
            Self::keep_token(&model, &mut states.cache, generation, token, think_end);
            kept += 1;
        }
        // The pass's rows after the kept tokens took places in the state's
        // rings. The cache counts them as stale.
        states
            .cache
            .record_rejected_rows(generation.state_id, rows.len() - kept);
        generation.next_token = next;
        // Row 0 is the request's own token.
        self.keep_routes(index, 0);
    }

    /// Keeps the experts of row `row` of the latest decoding pass as those of
    /// the latest token of the request at `index`, when the status keeps
    /// experts. It keeps nothing while another decoding call of the model
    /// runs, because the latest pass may then be that call's.
    fn keep_routes(&mut self, index: usize, row: usize) {
        let other_decoding_call = self
            .running_calls
            .iter()
            .any(|running_call| !matches!(running_call.call, Call::Prefill { .. }));
        if self.view.kept_route_tokens == 0 || self.experts_per_row == 0 || other_decoding_call {
            return;
        }
        let Some((experts, sources)) = self.model.read_routes(row) else {
            return;
        };
        let routes = &mut self.generations[index].routes;
        if routes.len() == self.view.kept_route_tokens {
            routes.pop_front();
        }
        routes.push_back(Route { experts, sources });
    }

    // ---- a decision model's requests ----

    /// Runs one step of a decision model. It refuses `replies`, since the
    /// model generates no text. The admitted `decisions` wait in order. The
    /// running decision goes on between its passes. Without one, the next
    /// admitted decision starts.
    fn run_decision_step(
        &mut self,
        shared: &Shared,
        replies: Vec<Job>,
        decisions: Vec<Job<DecisionRequest>>,
    ) {
        for job in replies {
            refuse(job, "the model is a decision model: it generates no text");
        }
        self.admitted_decisions.extend(decisions);
        if let Some(decision) = self.decision.take() {
            self.continue_decision(shared, decision);
        }
        while self.decision.is_none() {
            let Some(job) = self.admitted_decisions.pop_front() else {
                break;
            };
            self.start_decision(shared, job);
        }
        let phase = if self.decision.is_some() {
            "prefill"
        } else {
            "idle"
        };
        self.update_status(shared, phase);
    }

    /// Goes on with the running `decision`. While a pass runs, it waits. After
    /// a pass, the decision ends when the pass failed, when it has all its
    /// logits, when its client cancelled it, and at the stop. Otherwise its
    /// next pass starts.
    fn continue_decision(&mut self, shared: &Shared, decision: Decision) {
        if decision.in_call {
            self.decision = Some(decision);
        } else if decision.failure.is_some()
            || decision.logits.len() == decision.job.request.passes.len()
            || engine::is_stopping()
            || decision.job.flags.cancelled.load(Ordering::SeqCst)
        {
            self.end_decision(shared, decision);
        } else {
            self.start_decision_pass(decision);
        }
    }

    /// Starts the admitted decision `job`. It ends a cancelled decision, and
    /// every decision at the stop. It loads the model. It opens the decision's
    /// images and puts their spans in place of their places in each pass. It
    /// refuses a decision whose images or passes do not fit the model. It
    /// encodes the images, and the first pass starts.
    fn start_decision(&mut self, shared: &Shared, mut job: Job<DecisionRequest>) {
        if engine::is_stopping() {
            return self.end_waiting_decision(job, FinishReason::ServerStop);
        }
        if job.flags.cancelled.load(Ordering::SeqCst) {
            return self.end_waiting_decision(job, FinishReason::Cancelled);
        }
        let was_loaded = self.model.is_loaded();
        if !was_loaded {
            self.update_status(shared, "loading");
        }
        // No call runs while no decision runs, so the load may resize the
        // model's buffers.
        if let Err(error) = load_model(shared, &self.model) {
            if engine::is_stopping() {
                // The stop cut the load short.
                return self.end_waiting_decision(job, FinishReason::ServerStop);
            }
            lock(&shared.models[self.model_index].status).load_error = Some(error.clone());
            self.log_line(&format!("{}: {error}", self.settings.id));
            job.send_last_event(Event::Failed(error));
            return;
        }
        lock(&shared.models[self.model_index].status).load_error = None;
        if !was_loaded {
            self.log_line(&format!("{}: loaded", self.settings.id));
        }
        self.last_used_at = Instant::now();
        let mut media = None;
        if !job.request.media.is_empty() {
            let mut opened = match self.model.open_media(&job.request.media) {
                Ok(opened) => opened,
                Err(message) => return self.refuse_decision(job, &message),
            };
            for pass in &mut job.request.passes {
                if let Err(message) = place_spans(&mut opened, pass, self.ctx) {
                    return self.refuse_decision(job, &message);
                }
            }
            media = Some(opened);
        }
        let vocab = self.model.vocab_size();
        if let Some(message) = job
            .request
            .passes
            .iter()
            .find_map(|pass| check_decision_pass(pass, self.ctx, vocab, media.is_some()))
        {
            return self.refuse_decision(job, &message);
        }
        let started_at = Instant::now();
        let mut failure = None;
        if let Some(media) = media.as_mut() {
            match media.encode(0) {
                Ok(count) => self.log_line(&format!(
                    "req {}: {count} images encoded in {:.2} s",
                    job.id,
                    started_at.elapsed().as_secs_f64()
                )),
                Err(message) => failure = Some(message),
            }
        }
        let decision = Decision {
            job,
            media,
            logits: Vec::new(),
            failure,
            in_call: false,
            started_at,
        };
        if decision.failure.is_some() {
            return self.end_decision(shared, decision);
        }
        if decision.job.request.passes.is_empty() {
            // A decision without passes has all its logits.
            return self.end_decision(shared, decision);
        }
        self.start_decision_pass(decision);
    }

    /// Posts the next pass of `decision` on a new stream, and keeps the
    /// decision as the running one.
    fn start_decision_pass(&mut self, mut decision: Decision) {
        let pass = &decision.job.request.passes[decision.logits.len()];
        let stream = Stream::new(&self.model);
        stream.decide(
            &pass.tokens,
            pass.state_tokens,
            &pass.question_ends,
            &pass.rows,
            &pass.ids,
            decision.media.as_ref(),
        );
        decision.in_call = true;
        self.running_calls.push(RunningCall {
            stream,
            call: Call::Decide {
                request_id: decision.job.id,
            },
        });
        self.decision = Some(decision);
    }

    /// Finishes a pass of the decision `request_id` that ended on `stream`. It
    /// keeps the pass's logits, or why the pass failed when it did not fit in
    /// memory. The model's next step goes on with the decision: a request ends
    /// only in a step, which then marks whether the model still runs
    /// requests.
    fn finish_decide_call(&mut self, stream: Stream, request_id: u64) {
        let Some(decision) = self
            .decision
            .as_mut()
            .filter(|decision| decision.job.id == request_id)
        else {
            return;
        };
        decision.in_call = false;
        let count = decision.job.request.passes[decision.logits.len()].ids.len();
        match stream.copy_decision_logits(count) {
            Some(logits) => decision.logits.push(logits),
            None => {
                decision.failure = Some("a decision's pass does not fit: memory is short".into())
            }
        }
    }

    /// Ends `decision`: with why its last pass failed, when it did; with its
    /// logits after its last pass; else at its client's cancellation or the
    /// stop. It logs the end and counts the passes that ran in the model's
    /// totals.
    fn end_decision(&mut self, shared: &Shared, decision: Decision) {
        let passes = &decision.job.request.passes;
        let ran = decision.logits.len();
        let ran_tokens: usize = passes[..ran].iter().map(|pass| pass.tokens.len()).sum();
        let (count, tokens) = (passes.len(), decision.job.request.count_tokens());
        let seconds = decision.started_at.elapsed().as_secs_f64();
        let (how, event) = if let Some(message) = decision.failure {
            (format!("failed: {message}"), Event::Failed(message))
        } else if ran == count {
            (
                "decided".to_string(),
                Event::Decided {
                    logits: decision.logits,
                    tokens,
                },
            )
        } else if decision.job.flags.cancelled.load(Ordering::SeqCst) {
            (
                "client gone".to_string(),
                Event::Done {
                    reason: FinishReason::Cancelled,
                },
            )
        } else {
            (
                "stopped by the server's stop".to_string(),
                Event::Done {
                    reason: FinishReason::ServerStop,
                },
            )
        };
        self.log_line(&format!(
            "req {}: {how}: {ran} of {count} passes, {ran_tokens} of {tokens} tokens in {seconds:.2} s",
            decision.job.id
        ));
        let mut status = lock(&shared.models[self.model_index].status);
        status.total_requests += 1;
        status.total_prompt_tokens += tokens as u64;
        status.total_prefilled_tokens += ran_tokens as u64;
        status.total_prefill_seconds += seconds;
        drop(status);
        decision.job.send_last_event(event);
    }

    /// Ends the decision `job` before its first pass with `reason`. It logs the
    /// end and sends Done as the last event.
    fn end_waiting_decision(&self, job: Job<DecisionRequest>, reason: FinishReason) {
        self.log_line(&format!(
            "req {}: ended before its first pass ({}), waited {:.3} s",
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

    /// Refuses the decision `job` with `message`. It logs the refusal and sends
    /// it as the last event.
    fn refuse_decision(&self, job: Job<DecisionRequest>, message: &str) {
        self.log_line(&format!("req {}: refused: {message}", job.id));
        job.send_last_event(Event::Refused(message.into()));
    }

    // ---- an image model's requests ----

    /// Returns the status phase for a running image request from engine stats:
    /// "prefill" only while the prompt encoder runs, and "generating" while
    /// denoising, decoding or preparing the image.
    fn active_image_phase(stats: &engine::Stats) -> &'static str {
        if engine::ImageStage::from_raw(stats.image_stage) == engine::ImageStage::Prefill {
            "prefill"
        } else {
            "generating"
        }
    }

    /// Returns the encoded prompt token count for an active image task, using
    /// live engine counters while the first pass is still in flight.
    fn live_image_prompt_tokens(task: &ImageTask, stats: &engine::Stats) -> usize {
        if stats.prefill_rows_now > 0 {
            stats.prefill_rows_now as usize
        } else if stats.prefill_rows > task.pass_start_prefill_rows {
            (stats.prefill_rows - task.pass_start_prefill_rows) as usize
        } else {
            task.prompt_tokens
        }
    }

    /// Runs one step of an image generation model. Refuses chat and decision
    /// requests, queues admitted image jobs, advances any running image task,
    /// and starts the next waiting image job when idle.
    fn run_image_step(
        &mut self,
        shared: &Shared,
        replies: Vec<Job>,
        decisions: Vec<Job<DecisionRequest>>,
        images: Vec<Job<ImageRequest>>,
    ) {
        for job in replies {
            refuse(job, "the model is an image model: it generates no text");
        }
        for job in decisions {
            self.refuse_decision(job, "the model is an image model: it makes no decisions");
        }
        self.admitted_images.extend(images);
        if let Some(image) = self.image.take() {
            self.continue_image(shared, image);
        }
        while self.image.is_none() {
            let Some(job) = self.admitted_images.pop_front() else {
                break;
            };
            self.start_image(shared, job);
        }
        let phase = if self.image.is_some() {
            Self::active_image_phase(&self.model.read_stats())
        } else {
            "idle"
        };
        self.update_status(shared, phase);
    }

    /// Advances the running `image` task: signals cooperative cancellation if
    /// the client dropped its connection while a GPU call is in flight, ends
    /// the task once all images or a failure/cancellation occur, or starts the
    /// next image pass for `n > 1`.
    fn continue_image(&mut self, shared: &Shared, image: ImageTask) {
        if image.in_call {
            if image.job.flags.cancelled.load(Ordering::SeqCst) {
                for running in &self.running_calls {
                    if matches!(running.call, Call::GenerateImage { request_id } if request_id == image.job.id)
                    {
                        running.stream.cancel();
                    }
                }
            }
            self.image = Some(image);
        } else if image.failure.is_some()
            || image.outputs.len() == image.job.request.count
            || engine::is_stopping()
            || image.job.flags.cancelled.load(Ordering::SeqCst)
        {
            self.end_image(shared, image);
        } else {
            self.start_image_pass(image);
        }
    }

    /// Starts the admitted image request `job`: loads weights on first use,
    /// packages stable C FFI input buffers, and posts the first image pass.
    fn start_image(&mut self, shared: &Shared, mut job: Job<ImageRequest>) {
        if engine::is_stopping() {
            return self.end_waiting_image(job, FinishReason::ServerStop);
        }
        if job.flags.cancelled.load(Ordering::SeqCst) {
            return self.end_waiting_image(job, FinishReason::Cancelled);
        }
        let was_loaded = self.model.is_loaded();
        if !was_loaded {
            self.update_status(shared, "loading");
        }
        if let Err(error) = load_model(shared, &self.model) {
            if engine::is_stopping() {
                return self.end_waiting_image(job, FinishReason::ServerStop);
            }
            lock(&shared.models[self.model_index].status).load_error = Some(error.clone());
            self.log_line(&format!("{}: {error}", self.settings.id));
            job.send_last_event(Event::Failed(error));
            return;
        }
        lock(&shared.models[self.model_index].status).load_error = None;
        if !was_loaded {
            self.log_line(&format!("{}: loaded", self.settings.id));
        }
        self.last_used_at = Instant::now();
        let refs = std::mem::take(&mut job.request.references);
        let inputs = ImageInputs::new(&job.request.prompt, refs);
        let task = ImageTask {
            job,
            inputs,
            outputs: Vec::new(),
            prompt_tokens: 0,
            pass_start_prefill_rows: 0,
            prefilled_tokens: 0,
            prefill_seconds: 0.0,
            image_seconds: 0.0,
            failure: None,
            in_call: false,
            started_at: Instant::now(),
        };
        if task.job.request.count == 0 {
            return self.end_image(shared, task);
        }
        self.start_image_pass(task);
    }

    /// Posts the next image generation pass of `task` on a new engine stream.
    fn start_image_pass(&mut self, mut task: ImageTask) {
        let index = task.outputs.len() as u64;
        task.pass_start_prefill_rows = self.model.read_stats().prefill_rows;
        let req = &task.job.request;
        let stream = Stream::new(&self.model);
        stream.generate_image(
            &task.inputs,
            ImageCallParams {
                width: req.width,
                height: req.height,
                min_generation_area: req.min_generation_area,
                format: req.format,
                quality: req.quality,
                seed: req.seed.wrapping_add(index),
            },
        );
        task.in_call = true;
        self.running_calls.push(RunningCall {
            stream,
            call: Call::GenerateImage {
                request_id: task.job.id,
            },
        });
        self.image = Some(task);
    }

    /// Collects the finished image bytes or error from `stream` into the
    /// active image task.
    fn finish_image_call(&mut self, stream: Stream, request_id: u64) {
        let Some(task) = self.image.as_mut().filter(|task| task.job.id == request_id) else {
            return;
        };
        task.in_call = false;
        match stream.take_image_result() {
            Ok(output) => {
                task.prompt_tokens = output.prompt_tokens;
                task.prefilled_tokens += output.prompt_tokens as u64;
                task.prefill_seconds += output.prefill_seconds;
                task.image_seconds += output.image_seconds;
                task.outputs.push(output.bytes);
            }
            Err(error) => {
                if !task.job.flags.cancelled.load(Ordering::SeqCst) && !engine::is_stopping() {
                    task.failure = Some(error);
                }
            }
        }
    }

    /// Ends `task`, logging elapsed time and sending the final worker event.
    fn end_image(&mut self, shared: &Shared, task: ImageTask) {
        let done = task.outputs.len();
        let count = task.job.request.count;
        let seconds = task.started_at.elapsed().as_secs_f64();
        let (how, event) = if let Some(error) = task.failure {
            if error.invalid_input {
                (
                    format!("refused: {}", error.message),
                    Event::Refused(error.message),
                )
            } else {
                (
                    format!("failed: {}", error.message),
                    Event::Failed(error.message),
                )
            }
        } else if done == count {
            (
                "generated".to_string(),
                Event::Images {
                    images: task.outputs,
                },
            )
        } else if task.job.flags.cancelled.load(Ordering::SeqCst) {
            (
                "client gone".to_string(),
                Event::Done {
                    reason: FinishReason::Cancelled,
                },
            )
        } else {
            (
                "stopped by the server's stop".to_string(),
                Event::Done {
                    reason: FinishReason::ServerStop,
                },
            )
        };
        self.log_line(&format!(
            "req {}: {how}: {done} of {count} images ({}x{}) in {seconds:.2} s",
            task.job.id, task.job.request.width, task.job.request.height
        ));
        let mut status = lock(&shared.models[self.model_index].status);
        status.total_requests += 1;
        status.total_prompt_tokens += task.prompt_tokens as u64;
        status.total_prefilled_tokens += task.prefilled_tokens;
        status.total_prefill_seconds += task.prefill_seconds;
        status.total_generated_images += done as u64;
        status.total_image_seconds += task.image_seconds;
        drop(status);
        task.job.send_last_event(event);
    }

    /// Ends an image request `job` before it started running.
    fn end_waiting_image(&self, job: Job<ImageRequest>, reason: FinishReason) {
        self.log_line(&format!(
            "req {}: ended before image generation ({}), waited {:.3} s",
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

    /// Refuses the image request `job` with `message`.
    fn refuse_image(&self, job: Job<ImageRequest>, message: &str) {
        self.log_line(&format!("req {}: refused: {message}", job.id));
        job.send_last_event(Event::Refused(message.into()));
    }
}

/// Puts the spans of the images of `media` in place of their places in the
/// state of `pass`, at most `ctx` tokens with them, and moves the pass's
/// state, question ends and rows after them. Returns why it cannot: the pass
/// does not fit, or its places in the state and the images differ.
fn place_spans(media: &mut Media, pass: &mut d1::Pass, ctx: usize) -> Result<(), String> {
    let tokens = media
        .expand(&pass.tokens, ctx)
        .map_err(|error| match error {
            ExpandError::TooLong(rows) => {
                format!("a pass has {rows} tokens; the context holds {ctx}")
            }
            ExpandError::PlacesAndMediaDiffer => {
                "a pass's image places and its images differ".to_string()
            }
        })?;
    let added = tokens.len() - pass.tokens.len();
    let state = pass.state_tokens + added;
    // A span's tokens are negative: every one must be in the state.
    if tokens[state..].iter().any(|&token| token < 0) {
        return Err("a pass's image places must be in its state".into());
    }
    pass.tokens = tokens;
    pass.state_tokens = state;
    for at in pass.question_ends.iter_mut().chain(&mut pass.rows) {
        *at += added as i32;
    }
    Ok(())
}

/// Returns why the engine cannot run the decision pass `pass` with a context
/// of `ctx` tokens and a vocabulary of `vocab` tokens, or None when it can:
/// the pass fits the context, its questions follow its state up to its last
/// token, and it reads the logits of tokens of the vocabulary at its own
/// rows. Its tokens are of the vocabulary, or negative in its images' spans
/// when `reads_images`.
fn check_decision_pass(
    pass: &d1::Pass,
    ctx: usize,
    vocab: usize,
    reads_images: bool,
) -> Option<String> {
    let rows = pass.tokens.len();
    if rows > ctx {
        return Some(format!("a pass has {rows} tokens; the context holds {ctx}"));
    }
    let is_token = |id: &i32| usize::try_from(*id).is_ok_and(|id| id < vocab);
    let is_row = |id: &i32| is_token(id) || (reads_images && *id < 0);
    let questions_follow = pass.question_ends.last() == Some(&(rows as i32))
        && pass
            .question_ends
            .iter()
            .try_fold(pass.state_tokens as i32, |start, &end| {
                (end > start).then_some(end)
            })
            .is_some();
    let reads_fit = pass.rows.len() == pass.ids.len()
        && pass
            .rows
            .iter()
            .all(|row| usize::try_from(*row).is_ok_and(|row| row < rows))
        && pass.ids.iter().all(is_token);
    (!(questions_follow && reads_fit && pass.tokens.iter().all(is_row)))
        .then(|| "a pass's questions, rows or tokens do not fit the model".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lines that the test's hook for model lines received.
    static MODEL_LINES: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());

    /// Keeps a line about a model, as the terminal UI's hook does.
    fn keep_model_line(model_index: usize, line: &str) {
        MODEL_LINES
            .lock()
            .unwrap()
            .push((model_index, line.to_string()));
    }

    #[test]
    fn a_line_about_a_model_reaches_the_hook_with_the_models_index() {
        let view = ViewOptions {
            on_model_log: Some(keep_model_line),
            ..ViewOptions::default()
        };
        view.write_model_line(1, "req 6: queued for qwen3.8-27b");
        assert_eq!(
            *MODEL_LINES.lock().unwrap(),
            [(1, "req 6: queued for qwen3.8-27b".to_string())]
        );
    }

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

    /// A pass runs when it fits the context, its questions follow its state up
    /// to its last token, and it reads tokens of the vocabulary at its rows.
    #[test]
    fn a_decision_pass_is_checked_before_the_engine_runs_it() {
        let pass = || d1::Pass {
            tokens: vec![5, 6, 7, 8, 9, 10],
            state_tokens: 2,
            question_ends: vec![4, 6],
            rows: vec![3, 5, 5],
            ids: vec![11, 12, 13],
        };
        assert_eq!(check_decision_pass(&pass(), 6, 16, false), None);
        assert_eq!(
            check_decision_pass(&pass(), 5, 16, false).as_deref(),
            Some("a pass has 6 tokens; the context holds 5")
        );
        let mut with_image = pass();
        with_image.tokens[1] = -7;
        assert_eq!(
            check_decision_pass(&with_image, 6, 16, true),
            None,
            "an image's span holds negative tokens"
        );
        assert!(check_decision_pass(&with_image, 6, 16, false).is_some());
        let changes: [fn(&mut d1::Pass); 7] = [
            |pass| pass.question_ends = vec![4, 5],
            |pass| pass.question_ends = vec![2, 6],
            |pass| pass.question_ends = vec![5, 4, 6],
            |pass| pass.question_ends.clear(),
            |pass| pass.rows[0] = 6,
            |pass| pass.ids[2] = 16,
            |pass| pass.tokens[0] = -1,
        ];
        for change in changes {
            let mut wrong = pass();
            change(&mut wrong);
            assert_eq!(
                check_decision_pass(&wrong, 6, 16, false).as_deref(),
                Some("a pass's questions, rows or tokens do not fit the model"),
                "{:?} {:?} {:?} {:?}",
                wrong.tokens,
                wrong.question_ends,
                wrong.rows,
                wrong.ids
            );
        }
    }

    /// A request that waits shows no session yet. The running ones show what
    /// they do, their tokens, and the experts of their latest tokens.
    #[test]
    fn the_status_lists_each_request_with_what_it_does() {
        let (events, _receiver) = mpsc::unbounded_channel();
        let job = Job {
            id: 7,
            request: Request {
                model_index: 0,
                prompt: vec![1, 2, 3],
                media: Vec::new(),
                sampling: Sampling {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: None,
                },
                max_tokens: 16,
                note: String::new(),
            },
            events,
            flags: Arc::default(),
            queued_at: Instant::now(),
        };
        let decoding = RequestStatus {
            id: 5,
            status: "decoding",
            session: Some(5),
            prompt_tokens: 23,
            reused_tokens: 2,
            generated_tokens: 516,
            images_requested: 0,
            generated_images: 0,
            routes: vec![Route {
                experts: vec![17, 300],
                sources: vec![0, 4],
            }],
        };
        let (image_events, _image_rx) = mpsc::unbounded_channel();
        let image_job = Job {
            id: 8,
            request: ImageRequest {
                model_index: 0,
                prompt: "red mug".into(),
                width: 512,
                height: 512,
                count: 4,
                seed: 1,
                min_generation_area: 1024,
                format: engine::ImageFormat::Png,
                quality: 0,
                references: Vec::new(),
                note: String::new(),
            },
            events: image_events,
            flags: Arc::default(),
            queued_at: Instant::now(),
        };
        let requests = [
            decoding,
            RequestStatus::for_waiting_job(&job),
            RequestStatus::for_image(&image_job, "generating", 19, 2),
        ];
        assert_eq!(
            build_requests_json(&requests),
            json!([
                {"request": 5, "status": "decoding", "session": 5,
                 "prompt_tokens": 23, "reused_tokens": 2, "generated_tokens": 516,
                 "images_requested": 0, "generated_images": 0,
                 "routes": [{"experts": [17, 300], "sources": [0, 4]}]},
                {"request": 7, "status": "waiting", "session": null,
                 "prompt_tokens": 3, "reused_tokens": 0, "generated_tokens": 0,
                 "images_requested": 0, "generated_images": 0,
                 "routes": []},
                {"request": 8, "status": "generating", "session": null,
                 "prompt_tokens": 19, "reused_tokens": 0, "generated_tokens": 0,
                 "images_requested": 4, "generated_images": 2,
                 "routes": []},
            ])
        );
    }

    /// The engine's counters keep their whole values. The GPU's limit is null
    /// until the GPU starts.
    #[test]
    fn the_engine_counters_keep_whole_bytes_and_counts() {
        let mut stats = engine::Stats {
            layers: 43,
            experts_per_row: 240,
            routed_layers: 40,
            experts_per_layer: 384,
            prefill_rows: 1_938,
            prefill_layer: 31,
            prefill_rows_now: 399,
            mapped: 3,
            slot_bytes: 1 << 20,
            gpu_limit: i64::MAX,
            total: 128 << 30,
            pressure: 1,
            ..engine::Stats::default()
        };
        stats.decode.row_own = 222;
        stats.decode.row_ssd = 11;
        stats.cache.bytes = 291_000_000_123;
        let counters = build_engine_counters_json(&stats);
        assert_eq!(counters["layers"], 43);
        assert_eq!(counters["decode_expert_sources"], json!([222, 0, 0, 0, 11]));
        assert_eq!(counters["prefill_layer"], 31);
        assert_eq!(counters["expert_read_bytes"], 291_000_000_123u64);
        assert_eq!(counters["mapped_expert_bytes"], 3u64 << 20);
        assert_eq!(counters["memory"]["gpu_limit"], Value::Null);
        assert_eq!(counters["memory"]["total"], 128u64 << 30);
        stats.gpu_limit = 107 << 30;
        let counters = build_engine_counters_json(&stats);
        assert_eq!(counters["memory"]["gpu_limit"], 107i64 << 30);
    }

    /// Sessions in the order of their tokens tell the tree of their shared
    /// prefixes: each one shares with the next its prefix in that tree.
    #[test]
    fn sessions_come_in_the_order_of_their_tokens_with_their_shared_prefixes() {
        let lists: [&[i32]; 4] = [&[1, 2, 3, 9], &[1, 2, 3], &[1, 2, 7], &[5]];
        assert_eq!(
            order_by_tokens(&lists),
            [(1, 3), (0, 2), (2, 0), (3, 0)],
            "[1, 2, 3] comes before [1, 2, 3, 9], then [1, 2, 7], then [5]"
        );
        assert!(order_by_tokens(&[]).is_empty());
        let limits = ModelLimits {
            bytes: 1 << 30,
            states: 8,
        };
        let mut cache = Cache::new(vec![limits; 2]);
        let first = cache.add_state(0, 0);
        let other_model = cache.add_state(1, 0);
        cache.entries.get_mut(&first).unwrap().tokens = vec![4, 5, 6];
        let number = cache.entries[&first].number;
        assert_eq!(find_session_tokens(&cache, 0, number), Some(vec![4, 5, 6]));
        let number = cache.entries[&other_model].number;
        assert_eq!(
            find_session_tokens(&cache, 0, number),
            None,
            "a session of another model"
        );
    }

    /// The terminal UI's d key deletes an idle session's state from memory.
    /// A session that a request runs in keeps its state. A session of another
    /// model is not found.
    #[test]
    fn a_deletion_frees_an_idle_sessions_state_and_keeps_a_busy_one() {
        let limits = ModelLimits {
            bytes: 1 << 30,
            states: 8,
        };
        // The test takes no snapshots, so any snapshot interval works.
        let mut states = States::new(vec![limits; 2], 1);
        // One idle state of 3 tokens and 512 MiB, without engine handles.
        let idle = states.cache.add_state(0, 0);
        states.cache.entries.get_mut(&idle).unwrap().tokens = vec![4, 5, 6];
        states.cache.set_idle(idle, 1 << 29);
        let busy = states.cache.add_state(0, 0);
        let other_model = states.cache.add_state(1, 0);
        assert_eq!(
            states.delete_session_state(0, busy),
            format!("session {busy} not deleted: a request runs in it")
        );
        assert_eq!(
            states.delete_session_state(0, other_model),
            format!("session {other_model} not deleted: it is no longer in memory")
        );
        assert_eq!(
            states.delete_session_state(0, idle),
            format!("session {idle} deleted from memory, not saved to the SSD: 3 tokens, 0.50 GiB")
        );
        assert!(!states.cache.entries.contains_key(&idle));
        assert!(states.cache.entries.contains_key(&busy));
        assert!(states.cache.entries.contains_key(&other_model));
    }
}
