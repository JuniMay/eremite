//! The worker: one engine thread that runs the requests of every model.
//!
//! The engine thread registers the configured models. It makes every engine
//! call. So the GPU work of all models goes through one command queue.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::engine::{self, Model};

/// The engine needs a stack of at least 16 MiB on its thread.
const ENGINE_STACK_BYTES: usize = 16 << 20;

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
    /// The request starts decoding. Its prefill ended. `prompt_tokens` counts
    /// the prompt. `reused_tokens` counts the prompt tokens taken from the
    /// state cache.
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
    /// The prompt's token ids.
    pub prompt: Vec<i32>,
    /// How the request samples its tokens.
    pub sampling: Sampling,
    /// The most tokens the reply may have.
    pub max_tokens: i64,
}

/// Why the worker did not queue a request.
pub enum SubmitError {
    /// Too many requests wait for the model, or the server is stopping. The
    /// string says why.
    Busy(String),
}

// ---- what the threads share ----

/// A served model's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSettings {
    /// The model's name, as requests give it.
    pub id: String,
    /// The engine's options for the model.
    pub options: engine::Options,
}

/// A model as the HTTP handlers and the worker share it. The tokenizer works on
/// any thread.
pub struct SharedModel {
    /// The model's settings.
    pub settings: ModelSettings,
    /// The model's tokenizer.
    pub tokenizer: engine::Tokenizer,
}

/// What the HTTP handlers and the worker share.
struct Shared {
    /// The served models, in their configured order.
    models: Vec<SharedModel>,
}

/// Clones the handle to what the threads share. Panics before the worker
/// starts.
fn clone_shared() -> Arc<Shared> {
    SHARED.lock().unwrap().clone().expect("the worker started")
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
/// its weights. This call returns true once the models are registered. It
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
            let mut models = Vec::new();
            for settings in &model_settings {
                let Some(model) = Model::open(&settings.options) else {
                    let _ = started_sender.send(false);
                    return;
                };
                models.push(SharedModel {
                    settings: settings.clone(),
                    tokenizer: model.tokenizer(),
                });
            }
            let shared = Arc::new(Shared { models });
            *SHARED.lock().unwrap() = Some(shared);
            let _ = started_sender.send(true);
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

/// Wakes what waits for the stop: wait_for_stop. The engine runs it after each
/// stop request, on a thread of its own.
extern "C" fn wake_after_stop() {
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

/// Stops the worker and waits for its thread to end. It requests the stop
/// unless a stop was requested already.
pub fn stop() {
    if !engine::is_stopping() {
        request_stop();
    }
    if let Some(handle) = ENGINE_THREAD.lock().unwrap().take() {
        let _ = handle.join();
    }
}
