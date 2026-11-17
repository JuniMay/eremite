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
//! with no call running. The call prefills the whole prompt. The step then
//! starts decoding calls for the decoding requests with no call running. A
//! request starts decoding as soon as its prefill ends. A model runs one
//! request at a time.
//!
//! Calls run on streams. The calls of all models run at the same time. After
//! the steps, the loop advances the streams until a call ends. It then finishes
//! that call.

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

use tokio::sync::mpsc;

use crate::engine::{self, Logits, Model, Sampler, StateHandle, Stream};

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
    /// The waiting requests.
    queue: Mutex<Queue>,
    /// Wakes the worker when it has work: a new request, a cancellation, or the
    /// stop.
    worker_wakeup: Condvar,
    /// The served models, in their configured order.
    models: Vec<SharedModel>,
    /// The server-wide budgets and limits.
    budgets: crate::config::Budgets,
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
    let model_index = request.model_index;
    queue.waiting[model_index].push_back(Job {
        id: request_id,
        request,
        events,
        flags: flags.clone(),
    });
    notify_worker(&shared);
    Ok((
        CancelGuard {
            id: request_id,
            flags,
        },
        receiver,
    ))
}

// ---- the worker ----

/// A prefill in progress: its prompt.
struct Prefill {
    /// The prompt's tokens.
    prompt: Vec<i32>,
}

/// A request from its prefill to its end.
struct Generation {
    /// The submitted request.
    job: Job,
    /// The request's sampler.
    sampler: Sampler,
    /// The request's engine state.
    state: StateHandle,
    /// The number of prompt tokens.
    prompt_tokens: usize,
    /// Some while the prompt prefills.
    prefill: Option<Prefill>,
    /// The position of the next token.
    pos: usize,
    /// The number of tokens that the reply generated, a stop token included.
    generated: usize,
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
    /// Whether a call of this request runs on a stream.
    in_call: bool,
}

impl Generation {
    /// Returns whether the request decodes: its prefill has ended.
    fn is_decoding(&self) -> bool {
        self.prefill.is_none()
    }
}

/// An engine call that runs on a stream, and what to do when it ends.
enum Call {
    /// One prefill call of the request. It prefills the whole prompt.
    Prefill { request_id: u64 },
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
/// streams.
struct WorkerModel {
    /// The model's index in the configured order.
    model_index: usize,
    /// The model's engine handle.
    model: Rc<Model>,
    /// The model's context length, in tokens.
    ctx: usize,
    /// The requests from their prefill to their end.
    generations: Vec<Generation>,
    /// Admitted requests that wait for their turn to plan their prefill.
    admitted: VecDeque<Job>,
    /// The calls that run now.
    running_calls: Vec<RunningCall>,
    /// The streams that ended a decoding call and wait for the next one.
    idle_decode_streams: Vec<Stream>,
}

/// The worker on the engine thread.
struct Worker {
    /// What the threads share.
    shared: Arc<Shared>,
    /// The models that the worker serves, in their configured order.
    models: Vec<WorkerModel>,
}

impl Worker {
    /// Creates the worker for the models in `shared` and their engine handles.
    fn new(shared: Arc<Shared>, model_handles: Vec<Rc<Model>>) -> Worker {
        let models: Vec<WorkerModel> = shared
            .models
            .iter()
            .zip(model_handles)
            .enumerate()
            .map(|(model_index, (shared_model, model))| WorkerModel {
                model_index,
                ctx: shared_model.settings.options.ctx as usize,
                generations: Vec::new(),
                admitted: VecDeque::new(),
                running_calls: Vec::new(),
                idle_decode_streams: Vec::new(),
                model,
            })
            .collect();
        Worker { shared, models }
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
            let worker_model = &mut self.models[model_index];
            let ended_call = worker_model.running_calls.remove(call_index);
            worker_model.finish_call(ended_call);
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
        }
    }

    /// Runs the worker's loop until the stop. Each turn gives every model one
    /// step. A step admits requests, starts a prefill call and starts decoding
    /// calls. At the stop, every request ends with FinishReason::ServerStop.
    fn run_until_stop(&mut self) {
        let shared = self.shared.clone();
        while !engine::is_stopping() {
            let mut queue = lock(&shared.queue);
            let idle = |queue: &Queue, models: &[WorkerModel]| {
                queue.waiting.iter().all(|jobs| jobs.is_empty())
                    && models.iter().all(|worker_model| {
                        worker_model.generations.is_empty() && worker_model.admitted.is_empty()
                    })
            };
            while idle(&queue, &self.models) && !engine::is_stopping() {
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
                admitted_by_model.push(admitted);
            }
            drop(queue);
            for (worker_model, admitted) in self.models.iter_mut().zip(admitted_by_model) {
                worker_model.run_step(admitted);
                worker_model
                    .model
                    .set_request_running(!worker_model.generations.is_empty());
            }
            self.advance_running_calls();
        }
        self.finish_running_calls();
        for worker_model in &mut self.models {
            for generation in std::mem::take(&mut worker_model.generations) {
                worker_model.end_generation(generation, true);
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
    }
}

/// Refuses `job` with `message`. It sends the refusal as the last event.
fn refuse(job: Job, message: &str) {
    job.send_last_event(Event::Refused(message.into()));
}

/// The refusal of a request whose new state does not fit in the memory.
const STATE_MEMORY_SHORT: &str = "the conversation's attention state does not fit: memory is short";

/// The refusal of a prompt of `tokens` tokens, too many for a context of `ctx`.
fn format_prompt_too_long(tokens: usize, ctx: usize) -> String {
    format!("the prompt has {tokens} tokens; the context holds {ctx}")
}

/// Ends `job` before its prefill with `reason`. It sends Done as the last
/// event.
fn end_before_prefill(job: Job, reason: FinishReason) {
    job.send_last_event(Event::Done { reason });
}

impl WorkerModel {
    /// Runs one step of the model. Admitted requests start one at a time, when
    /// the model runs no other request. A prefill call starts. The generations
    /// that are done end. The others start decoding calls.
    fn run_step(&mut self, admitted: Vec<Job>) {
        self.admitted.extend(admitted);
        // The model runs one request at a time.
        while self.generations.is_empty() {
            let Some(job) = self.admitted.pop_front() else {
                break;
            };
            if engine::is_stopping() {
                end_before_prefill(job, FinishReason::ServerStop);
            } else if job.flags.cancelled.load(Ordering::SeqCst) {
                end_before_prefill(job, FinishReason::Cancelled);
            } else {
                if let Err(error) = self.model.load() {
                    if engine::is_stopping() {
                        // The stop cut the load short.
                        end_before_prefill(job, FinishReason::ServerStop);
                        continue;
                    }
                    job.send_last_event(Event::Failed(error));
                    continue;
                }
                self.start_request(job);
            }
        }
        if engine::is_stopping() {
            return;
        }
        self.start_prefill_calls();
        let ctx = self.ctx;
        let mut i = 0;
        while i < self.generations.len() {
            if !self.generations[i].in_call
                && self.generations[i].is_decoding()
                && Self::check_ended(&self.model, &mut self.generations[i], ctx)
            {
                let generation = self.generations.swap_remove(i);
                self.end_generation(generation, false);
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
                self.end_generation(generation, false);
            } else {
                i += 1;
            }
        }
        let decoding_count = self
            .generations
            .iter()
            .filter(|generation| generation.is_decoding())
            .count();
        if self.generations.is_empty() {
            self.idle_decode_streams.clear();
            return;
        }
        if decoding_count > 0 {
            self.start_decode_calls();
        }
        for generation in &mut self.generations {
            if !generation.unsent_tokens.is_empty() {
                let _ = generation
                    .job
                    .events
                    .send(Event::Tokens(std::mem::take(&mut generation.unsent_tokens)));
            }
        }
    }

    /// Starts an admitted request. It checks the request's prompt. It makes the
    /// request's state, and the prefill follows. It refuses a request that does
    /// not fit.
    fn start_request(&mut self, job: Job) {
        // A reply has at least one token, so a prompt has at most ctx - 1.
        let most_prompt_tokens = self.ctx - 1;
        let prompt = job.request.prompt.clone();
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
        // A request whose new state does not fit is refused.
        let Some(state) = self.model.state_new() else {
            return refuse(job, STATE_MEMORY_SHORT);
        };
        self.generations.push(Generation {
            job,
            sampler,
            state,
            prompt_tokens,
            prefill: Some(Prefill { prompt }),
            pos: prompt_tokens,
            generated: 0,
            reached_stop_token: false,
            cancelled: false,
            next_token: 0,
            unsent_tokens: Vec::new(),
            max_new_tokens,
            in_call: false,
        });
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
    /// running.
    fn start_prefill_calls(&mut self) {
        for i in 0..self.generations.len() {
            self.start_prefill_call(i);
        }
    }

    /// Starts a prefill call for request `i` if it is still prefilling with no
    /// call running. The call prefills the whole prompt.
    fn start_prefill_call(&mut self, i: usize) {
        let generation = &self.generations[i];
        let Some(prefill) = generation.prefill.as_ref() else {
            return;
        };
        if generation.in_call {
            return;
        }
        if generation.job.flags.cancelled.load(Ordering::SeqCst) {
            // The next step ends the cancelled request.
            return;
        }
        let stream = Stream::new(&self.model);
        stream.prefill(generation.state, &prefill.prompt, 0, false, None, None);
        let request_id = generation.job.id;
        self.generations[i].in_call = true;
        self.running_calls.push(RunningCall {
            stream,
            call: Call::Prefill { request_id },
        });
    }

    /// Finishes a prefill call that ended. The request starts decoding.
    fn finish_prefill_call(&mut self, stream: Stream, request_id: u64) {
        let i = self.find_generation_index(request_id);
        self.generations[i].in_call = false;
        let Some(logits) = stream.logits() else {
            if engine::is_stopping() {
                // The stop cut the prefill short. The stop also ends the
                // request.
                return;
            }
            let generation = self.generations.remove(i);
            self.model.state_free(generation.state);
            generation.job.send_last_event(Event::Failed(
                "the prefill's working memory does not fit: memory is short".into(),
            ));
            return;
        };
        let mut generation = self.generations.remove(i);
        self.start_decoding(&mut generation, logits);
        self.generations.insert(i, generation);
        // The stream owns the logits until here. Dropping it frees the prompt's
        // large buffers.
        drop(stream);
    }

    /// Finishes a call that ended. Its requests then go on.
    fn finish_call(&mut self, running_call: RunningCall) {
        let RunningCall { stream, call } = running_call;
        match call {
            Call::Prefill { request_id } => self.finish_prefill_call(stream, request_id),
            Call::DecodeRows {
                request_ids,
                tokens,
            } => self.finish_decode_rows_call(stream, request_ids, tokens),
        }
    }

    /// Starts decoding after the prefill ended with `logits`. It sends
    /// DecodingStarted. It samples the first token from `logits`.
    fn start_decoding(&mut self, generation: &mut Generation, logits: Logits) {
        generation.prefill.take().expect("a prefill");
        let prompt_tokens = generation.prompt_tokens;
        let _ = generation.job.events.send(Event::DecodingStarted {
            prompt_tokens,
            reused_tokens: 0,
        });
        generation.next_token = self.model.sample(logits, &mut generation.sampler);
    }

    /// Keeps `token`. The token ran at the generation's position. It adds the
    /// token to the tokens to send. A stop token ends the reply and is not
    /// sent.
    fn keep_token(model: &Model, generation: &mut Generation, token: i32) {
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
            generation.unsent_tokens.push(generation.next_token as u32);
            generation.generated += 1;
            return true;
        }
        false
    }

    /// Ends the generation. It sends its last tokens and its Done event. It
    /// frees the request's state.
    fn end_generation(&mut self, mut generation: Generation, at_stop: bool) {
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
        generation.job.send_last_event(Event::Done { reason });
        self.model.state_free(generation.state);
    }

    /// Ends the requests of a decoding pass that could not load its experts.
    /// The stop may have ended their reads. Or memory stayed short after idle
    /// memory was freed. During the stop, the requests end as every request
    /// does at the stop. Otherwise the requests end with an error. The pass may
    /// have changed their states in part. So the states are freed.
    fn end_failed_pass(&mut self, generation_indices: &[usize]) {
        if engine::is_stopping() {
            return;
        }
        let mut indices = generation_indices.to_vec();
        indices.sort_unstable_by(|left, right| right.cmp(left));
        for i in indices {
            let generation = self.generations.remove(i);
            self.model.state_free(generation.state);
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

    /// Starts a decoding call for the decoding request with no call running.
    fn start_decode_calls(&mut self) {
        let decoding_indices: Vec<usize> = (0..self.generations.len())
            .filter(|&i| self.generations[i].is_decoding() && !self.generations[i].in_call)
            .collect();
        if decoding_indices.is_empty() {
            return;
        }
        let index = decoding_indices[0];
        let stream = self.take_decode_stream();
        let generation = &mut self.generations[index];
        generation.in_call = true;
        let request_id = generation.job.id;
        stream.decode_rows(
            &[generation.state],
            &[generation.next_token],
            &[generation.pos as i32],
        );
        let call = Call::DecodeRows {
            request_ids: vec![request_id],
            tokens: vec![generation.next_token],
        };
        self.running_calls.push(RunningCall { stream, call });
    }

    /// Finishes a decoding call of one token per request. Each request keeps
    /// its token and samples the next one.
    fn finish_decode_rows_call(&mut self, stream: Stream, request_ids: Vec<u64>, tokens: Vec<i32>) {
        let indices: Vec<usize> = request_ids
            .iter()
            .map(|&request_id| self.find_generation_index(request_id))
            .collect();
        for &i in &indices {
            self.generations[i].in_call = false;
        }
        let Some(logits) = stream.logits() else {
            self.idle_decode_streams.push(stream);
            self.end_failed_pass(&indices);
            return;
        };
        let vocab = self.model.vocab_size();
        let model = self.model.clone();
        for (row, &i) in indices.iter().enumerate() {
            let generation = &mut self.generations[i];
            Self::keep_token(&model, generation, tokens[row]);
            // Safety: the call returns one row of logits for each state. Each
            // row holds vocab_size floats.
            generation.next_token =
                model.sample(unsafe { logits.row(row, vocab) }, &mut generation.sampler);
        }
        self.idle_decode_streams.push(stream);
    }
}
