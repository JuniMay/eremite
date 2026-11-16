//! The engine's C interface (src/eremite.h) and thin safe wrappers over it:
//! the process's settings, its log and its stop; a model's settings; a model
//! with its tokenizer; and the model's states, calls and sampling. A
//! [`Model`] is one registered checkpoint, whose weights load on demand. Its
//! functions and its streams run on the engine's thread only; the tokenizer's
//! run on any thread.

use std::ffi::{CString, c_char, c_int};
use std::ptr::{self, NonNull};

/// The version of src/eremite.h's interface that these declarations follow
/// (EREMITE_ABI).
const ABI: c_int = 2;

/// The process's settings and hooks (eremite_process).
#[repr(C)]
struct RawProcess {
    abi: c_int,
    memory_gb: f64,
    reserve_gb: f64,
    gpu_timeout_seconds: f64,
    freed_memory_delay_seconds: f64,
    available_memory_reading_interval_seconds: f64,
    available_memory_reading_max_age_seconds: f64,
    keep_recently_used_weights_seconds: f64,
    free_pages_only_after_swapout_seconds: f64,
    expert_cache_growth_margin_gb: f64,
    gpu_memory_limit_margin_gb: f64,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
    on_stop: Option<extern "C" fn()>,
}

/// DeepSeek's settings (eremite_deepseek_config).
#[repr(C)]
#[derive(Clone, Copy)]
struct RawDeepSeekConfig {
    experts: *const c_char,
    substitute: f32,
    draft: c_int,
    ced: c_int,
    prefill_prune: c_int,
    packed_state_rows: c_int,
}

/// A model's settings (eremite_config).
#[repr(C)]
struct RawConfig {
    abi: c_int,
    model: *const c_char,
    ctx: c_int,
    lookup: c_int,
    deepseek: RawDeepSeekConfig,
}

/// A registered model (eremite_model).
#[repr(C)]
struct RawModel {
    _opaque: [u8; 0],
}

/// A checkpoint's tokenizer (eremite_tokenizer).
#[repr(C)]
struct RawTokenizer {
    _opaque: [u8; 0],
}

/// A state of one model (eremite_state).
#[repr(C)]
struct RawState {
    _opaque: [u8; 0],
}

/// A stream of one model's calls (eremite_stream).
#[repr(C)]
struct RawStream {
    _opaque: [u8; 0],
}

/// A sampler (eremite_sampler).
#[repr(C)]
struct RawSampler {
    _opaque: [u8; 0],
}

/// A prompt's decoded media (eremite_media).
#[repr(C)]
struct RawMedia {
    _opaque: [u8; 0],
}

/// eremite_tokenize's mode that also takes the added tokens' spellings as
/// those tokens.
const TOKENIZE_ADDED_TOKENS: c_int = 1;
/// eremite_tokenize's mode for a rendered prompt: as TOKENIZE_ADDED_TOKENS,
/// with the byte 0xFF for an image's place.
const TOKENIZE_PROMPT: c_int = 2;

unsafe extern "C" {
    fn eremite_init(p: *const RawProcess) -> c_int;
    fn eremite_idle_work_interval_seconds() -> f64;
    fn eremite_is_stopping() -> c_int;
    fn eremite_request_stop();
    fn eremite_count_stop_requests() -> c_int;
    fn eremite_log(line: *const c_char);
    fn eremite_open(c: *const RawConfig) -> *mut RawModel;
    fn eremite_load(m: *mut RawModel) -> c_int;
    fn eremite_is_loaded(m: *const RawModel) -> c_int;
    fn eremite_unload(m: *mut RawModel) -> u64;
    fn eremite_drain();
    fn eremite_set_request_running(m: *mut RawModel, active: c_int);
    fn eremite_max_rows(m: *const RawModel) -> c_int;
    fn eremite_vocab_size(m: *const RawModel) -> c_int;
    fn eremite_is_stop_token(m: *const RawModel, token: i32) -> c_int;
    fn eremite_state_new(m: *mut RawModel) -> *mut RawState;
    fn eremite_state_free(m: *mut RawModel, s: *mut RawState);
    fn eremite_stream_new(m: *mut RawModel) -> *mut RawStream;
    fn eremite_stream_free(x: *mut RawStream);
    fn eremite_stream_prefill(
        x: *mut RawStream,
        s: *mut RawState,
        tokens: *const i32,
        n: c_int,
        from: c_int,
        replay: c_int,
        capture: c_int,
        images: *const RawMedia,
    );
    fn eremite_stream_decode_rows(
        x: *mut RawStream,
        states: *mut *mut RawState,
        tokens: *const i32,
        pos: *const i32,
        count: c_int,
    );
    fn eremite_run_streams(streams: *const *mut RawStream, n: c_int) -> c_int;
    fn eremite_stream_logits(x: *const RawStream) -> *const f32;
    fn eremite_sampler_new(
        temperature: f64,
        top_p: f64,
        top_k: c_int,
        seed: u64,
        seed_set: c_int,
    ) -> *mut RawSampler;
    fn eremite_sampler_free(s: *mut RawSampler);
    fn eremite_sample(m: *const RawModel, logits: *const f32, s: *mut RawSampler) -> i32;
    fn eremite_model_tokenizer(m: *const RawModel) -> *mut RawTokenizer;
    fn eremite_tokenize(
        t: *mut RawTokenizer,
        text: *const c_char,
        len: usize,
        mode: c_int,
        out: *mut i32,
        max: c_int,
    ) -> c_int;
    fn eremite_token_bytes(t: *const RawTokenizer, id: i32, len: *mut usize) -> *const c_char;
    fn eremite_token_kind(t: *const RawTokenizer, id: i32) -> c_int;
    fn eremite_find_added_token(t: *const RawTokenizer, text: *const c_char, len: usize) -> i32;
}

/// Returns `s` as a C string. A NUL byte would end the string early, so the
/// NUL bytes are left out.
fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no NUL bytes left")
}

/// The process's settings for the engine (eremite_process). A setting of 0
/// takes the engine's default. The engine's memory policy always keeps its
/// defaults here.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessSettings {
    /// GiB of model memory at most, state memory included; 0: no limit.
    pub memory_gb: f64,
    /// GiB kept available to other programs.
    pub reserve_gb: f64,
    /// Seconds that a wait for the GPU may last before the process ends.
    pub gpu_timeout_seconds: f64,
}

/// Installs the log (on_log receives its lines, with their time, under the
/// log's lock, and must not call [`log`]), the stop's signal handlers and the
/// failure hook, on this thread, which from then on is the engine's (it
/// needs a 16 MiB stack), with `settings`. on_stop runs after each stop
/// request, on a thread of its own. False when called twice, and for settings
/// the engine refuses.
pub fn init(
    settings: &ProcessSettings,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
    on_stop: Option<extern "C" fn()>,
) -> bool {
    let p = RawProcess {
        abi: ABI,
        memory_gb: settings.memory_gb,
        reserve_gb: settings.reserve_gb,
        gpu_timeout_seconds: settings.gpu_timeout_seconds,
        freed_memory_delay_seconds: 0.0,
        available_memory_reading_interval_seconds: 0.0,
        available_memory_reading_max_age_seconds: 0.0,
        keep_recently_used_weights_seconds: 0.0,
        free_pages_only_after_swapout_seconds: 0.0,
        expert_cache_growth_margin_gb: 0.0,
        gpu_memory_limit_margin_gb: 0.0,
        on_log,
        on_die,
        on_stop,
    };
    unsafe { eremite_init(&p) == 0 }
}

/// Returns the seconds between two runs of a model's idle work: the engine's
/// available-memory reading interval after [`init`], or 0 before it.
pub fn idle_work_interval_seconds() -> f64 {
    unsafe { eremite_idle_work_interval_seconds() }
}

/// Whether a stop was requested: SIGINT, SIGTERM or [`request_stop`].
pub fn is_stopping() -> bool {
    unsafe { eremite_is_stopping() != 0 }
}

/// Requests the stop as SIGINT does: init's on_stop runs after it.
pub fn request_stop() {
    unsafe { eremite_request_stop() }
}

/// Returns how many stops were requested.
pub fn count_stop_requests() -> u32 {
    unsafe { eremite_count_stop_requests().max(0) as u32 }
}

/// Waits for the GPU's work and the expert reads to finish.
pub fn drain() {
    unsafe { eremite_drain() };
}

/// Writes a line to the log, with the time: on stderr, or to init's on_log.
pub fn log(line: &str) {
    let line = c_string(line);
    unsafe { eremite_log(line.as_ptr()) }
}

/// Which format a checkpoint's requests take: its family, from config.json's
/// model_type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// DeepSeek V4.1, model_type deepseek_v41.
    DeepSeek,
}

impl Architecture {
    /// Returns the family of the checkpoint in `dir`, or why there is none.
    pub fn of(dir: &str) -> Result<Architecture, String> {
        let path = std::path::Path::new(dir).join("config.json");
        let text =
            std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let config: serde_json::Value =
            serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        match config.get("model_type").and_then(serde_json::Value::as_str) {
            Some("deepseek_v41") => Ok(Architecture::DeepSeek),
            Some(t) => Err(format!(
                "{}: model_type {t} is not one this server runs",
                path.display()
            )),
            None => Err(format!("{}: no model_type", path.display())),
        }
    }
}

/// The settings of a model: the ones every family shares, and its family's
/// own.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Options {
    /// The checkpoint's directory.
    pub model: String,
    /// The context's capacity in tokens.
    pub ctx: i32,
    /// The family's own settings.
    pub backend: BackendOptions,
}

/// A family's own settings.
#[derive(Default, Clone, Debug, PartialEq)]
pub enum BackendOptions {
    /// None set yet: the checkpoint's family decides which settings apply.
    #[default]
    Unspecified,
    /// DeepSeek's settings.
    DeepSeek(DeepSeekOptions),
}

/// DeepSeek's own settings.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct DeepSeekOptions {
    /// A GGUF file or shard whose routed experts replace the checkpoint's.
    pub experts: Option<String>,
    /// The decoder replay tokens of an approximate prefill, or 0 for none.
    pub ced: i32,
    /// Whether a prefill skips the decoder batches that no later row needs.
    pub prefill_prune: bool,
    /// Whether a state keeps its compressed rows and index keys packed: the
    /// same values in about 28% of BF16's bytes.
    pub packed_state_rows: bool,
}

impl Options {
    /// Sets the family's own settings to the checkpoint's family. Returns an
    /// error if they were set for another family.
    pub fn configure(&mut self, architecture: Architecture) -> Result<(), String> {
        self.backend = match (&self.backend, architecture) {
            (BackendOptions::Unspecified, Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(DeepSeekOptions::default())
            }
            (BackendOptions::DeepSeek(d), Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(d.clone())
            }
        };
        Ok(())
    }

    /// Returns DeepSeek's settings, or None for another family.
    pub fn deepseek(&self) -> Option<&DeepSeekOptions> {
        match &self.backend {
            BackendOptions::DeepSeek(d) => Some(d),
            BackendOptions::Unspecified => None,
        }
    }

    /// Returns DeepSeek's settings to change. Settings not set yet become
    /// DeepSeek's.
    pub fn deepseek_mut(&mut self) -> Result<&mut DeepSeekOptions, String> {
        if matches!(self.backend, BackendOptions::Unspecified) {
            self.backend = BackendOptions::DeepSeek(DeepSeekOptions::default());
        }
        match &mut self.backend {
            BackendOptions::DeepSeek(d) => Ok(d),
            BackendOptions::Unspecified => unreachable!("set above"),
        }
    }
}

/// The handle of a state that Model::state_new made. It stays valid until
/// Model::state_free frees the state.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHandle(NonNull<RawState>);

/// Logits of one row or more, from a finished call. They stay valid until the
/// stream's next call.
#[derive(Clone, Copy, Debug)]
pub struct Logits(NonNull<f32>);

impl Logits {
    /// Returns the logits of row `row`, where each row holds `vocab` floats.
    ///
    /// # Safety
    ///
    /// The logits hold more than `row` rows of `vocab` floats.
    pub unsafe fn row(self, row: usize, vocab: usize) -> Logits {
        Logits(unsafe { self.0.add(row * vocab) })
    }
}

/// A registered model whose weights may still be unloaded. It stays on the
/// engine's thread.
pub struct Model {
    raw: *mut RawModel,
    /// The checkpoint's directory.
    directory: String,
    /// Whether the weight files passed validate_weight_files.
    validated: std::cell::Cell<bool>,
}

/// The largest safetensors header that validate_weight_files reads. A larger
/// one is taken as damaged. The reason for this value is unknown.
const MAX_WEIGHT_HEADER_BYTES: u64 = 64 << 20;

/// Checks that every weight of the checkpoint in `directory` lies inside its
/// safetensors file: the files of model.safetensors.index.json's weight_map,
/// or model.safetensors alone. Returns what is missing or damaged.
fn validate_weight_files(directory: &str) -> Result<(), String> {
    use std::io::Read;
    let root = std::path::Path::new(directory);
    let index = root.join("model.safetensors.index.json");
    let bytes = if index.exists() {
        std::fs::read(&index).map_err(|e| format!("model weight index is unavailable: {e}"))?
    } else {
        b"null".to_vec()
    };
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid model weight index: {e}"))?;
    let mut files = std::collections::BTreeMap::<&str, Vec<&str>>::new();
    if json.is_null() {
        files.insert("model.safetensors", Vec::new());
    } else {
        let weights = json
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .filter(|m| !m.is_empty())
            .ok_or("model weight index has no weight_map")?;
        for (name, file) in weights {
            files
                .entry(file.as_str().ok_or("weight shard name is not text")?)
                .or_default()
                .push(name);
        }
    }
    for (name, tensors) in files {
        let mut file = std::fs::File::open(root.join(name))
            .map_err(|e| format!("model weight shard {name} is unavailable: {e}"))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        // A safetensors file starts with its header's length, 8 bytes.
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("cannot read weight shard {name}: {e}"))?;
        let header = u64::from_le_bytes(prefix);
        if header > MAX_WEIGHT_HEADER_BYTES || header > size.saturating_sub(8) {
            return Err(format!("invalid weight header in {name}"));
        }
        let mut data = vec![0; header as usize];
        file.read_exact(&mut data)
            .map_err(|e| format!("cannot read weight header {name}: {e}"))?;
        let values: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|e| format!("invalid weight header {name}: {e}"))?;
        let tensors = if tensors.is_empty() {
            values
                .as_object()
                .ok_or("weight header is not an object")?
                .keys()
                .filter(|key| key.as_str() != "__metadata__")
                .map(String::as_str)
                .collect()
        } else {
            tensors
        };
        for tensor in tensors {
            let offsets = values
                .get(tensor)
                .and_then(|v| v.get("data_offsets"))
                .and_then(serde_json::Value::as_array)
                .filter(|v| v.len() == 2)
                .ok_or_else(|| format!("missing weight {tensor} in {name}"))?;
            let start = offsets[0].as_u64().ok_or("invalid weight offset")?;
            let end = offsets[1].as_u64().ok_or("invalid weight offset")?;
            if start > end || end > size - header - 8 {
                return Err(format!("weight {tensor} extends past {name}"));
            }
        }
    }
    Ok(())
}

impl Model {
    /// Registers the checkpoint's model with settings `o`: its metadata and
    /// its tokenizer, without its weights. None, with the reason in the log,
    /// for a checkpoint or settings the engine refuses.
    pub fn open(o: &Options) -> Option<Model> {
        let architecture = match Architecture::of(&o.model) {
            Ok(architecture) => architecture,
            Err(error) => {
                log(&error);
                return None;
            }
        };
        let mut o = o.clone();
        if let Err(error) = o.configure(architecture) {
            log(&error);
            return None;
        }
        let deepseek = o.deepseek().cloned().unwrap_or_default();
        let model = c_string(&o.model);
        let experts = deepseek.experts.as_deref().map(c_string);
        let raw = RawConfig {
            abi: ABI,
            model: model.as_ptr(),
            ctx: o.ctx,
            // No context-copy proposals.
            lookup: 0,
            deepseek: RawDeepSeekConfig {
                experts: experts.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                // The engine's approximate substitution stays off: it is for
                // the engine's tests only.
                substitute: 0.0,
                // No proposals of the draft head.
                draft: 0,
                ced: deepseek.ced,
                prefill_prune: deepseek.prefill_prune.into(),
                packed_state_rows: deepseek.packed_state_rows.into(),
            },
        };
        let m = unsafe { eremite_open(&raw) };
        (!m.is_null()).then_some(Model {
            raw: m,
            directory: o.model.clone(),
            validated: std::cell::Cell::new(false),
        })
    }

    /// Loads the model's weights, or loads them again after they were given
    /// back. The first load checks the weight files first. After a load, the
    /// model runs admitted requests. Returns why the model did not load.
    pub fn load(&self) -> Result<(), String> {
        if !self.validated.get() {
            validate_weight_files(&self.directory)?;
            self.validated.set(true);
        }
        if unsafe { eremite_load(self.raw) != 0 } {
            Ok(())
        } else {
            Err("the model could not be loaded".into())
        }
    }

    /// Returns whether the model is loaded and its weights are in memory.
    pub fn is_loaded(&self) -> bool {
        unsafe { eremite_is_loaded(self.raw) != 0 }
    }

    /// Sets whether the model runs admitted requests. While it does, its
    /// weights stay in memory.
    pub fn set_request_running(&self, active: bool) {
        unsafe { eremite_set_request_running(self.raw, active.into()) };
    }

    /// Gives the model's expert cache and weights back to the system while
    /// the model runs no request. Its states stay, and the next load reads
    /// the weights again. Returns the bytes given back.
    pub fn unload(&self) -> u64 {
        unsafe { eremite_unload(self.raw) }
    }

    /// Returns how many rows of different states one decoding call takes at
    /// most.
    pub fn max_rows(&self) -> usize {
        unsafe { eremite_max_rows(self.raw) as usize }
    }

    /// Returns the number of floats in a row of logits: the vocabulary's size.
    pub fn vocab_size(&self) -> usize {
        unsafe { eremite_vocab_size(self.raw) as usize }
    }

    /// Returns whether a token ends a reply. The loaded model knows its end
    /// token.
    pub fn is_stop_token(&self, token: i32) -> bool {
        unsafe { eremite_is_stop_token(self.raw, token) != 0 }
    }

    /// Makes a new state, or returns None when its memory does not fit.
    pub fn state_new(&self) -> Option<StateHandle> {
        let s = unsafe { eremite_state_new(self.raw) };
        NonNull::new(s).map(StateHandle)
    }

    /// Frees a state that no call uses.
    pub fn state_free(&self, s: StateHandle) {
        unsafe { eremite_state_free(self.raw, s.0.as_ptr()) }
    }

    /// Returns a token sampled from one row of logits.
    pub fn sample(&self, logits: Logits, sampler: &mut Sampler) -> i32 {
        unsafe { eremite_sample(self.raw, logits.0.as_ptr(), sampler.0) }
    }

    /// Returns the model's tokenizer.
    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer(unsafe { eremite_model_tokenizer(self.raw) })
    }
}

/// A checkpoint's tokenizer. It lives as long as the process, and any thread
/// may use it.
#[derive(Clone, Copy)]
pub struct Tokenizer(*mut RawTokenizer);

// Safety: each engine tokenizer has its own lock, and it is never freed.
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Tokenizer {
    /// Returns the token ids of `text` in eremite_tokenize's `mode`.
    fn tokenize_in_mode(&self, text: &[u8], mode: c_int) -> Vec<i32> {
        // A token holds one byte at least, and the tokenizer may add one
        // token before the text and one after it.
        let mut out = vec![0i32; text.len() + 2];
        let n = unsafe {
            eremite_tokenize(
                self.0,
                text.as_ptr().cast(),
                text.len(),
                mode,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        };
        out.truncate(n.max(0) as usize);
        out
    }

    /// Returns the token ids of `text`, in which the added tokens' spellings
    /// are those tokens.
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        self.tokenize_in_mode(text.as_bytes(), TOKENIZE_ADDED_TOKENS)
    }

    /// Returns the token ids of a rendered prompt: the added tokens'
    /// spellings are those tokens, and the byte 0xFF is an image's place.
    pub fn tokenize_prompt(&self, prompt: &[u8]) -> Vec<i32> {
        self.tokenize_in_mode(prompt, TOKENIZE_PROMPT)
    }

    /// Returns the id of the added token spelled `text`, if there is one.
    pub fn find_added_token(&self, text: &str) -> Option<i32> {
        let id = unsafe { eremite_find_added_token(self.0, text.as_ptr().cast(), text.len()) };
        (id >= 0).then_some(id)
    }

    /// Returns whether a token is one of the tokenizer's added tokens.
    pub fn is_added(&self, id: i32) -> bool {
        unsafe { eremite_token_kind(self.0, id) != 0 }
    }

    /// Returns a token's bytes, or none for an unknown id.
    pub fn token_bytes(&self, id: i32) -> &'static [u8] {
        let mut n = 0usize;
        // Safety: the tokenizer and its bytes stay loaded while the process
        // runs.
        let p = unsafe { eremite_token_bytes(self.0, id, &mut n) };
        if p.is_null() || n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) }
        }
    }
}

/// How a model picks a token from a row of logits.
pub struct Sampler(*mut RawSampler);

impl Sampler {
    /// Returns a sampler, or None for settings the engine refuses. A
    /// temperature of 0 is greedy, top_k 0 takes every token, and no seed
    /// gives a random one.
    pub fn new(temperature: f64, top_p: f64, top_k: u32, seed: Option<u64>) -> Option<Sampler> {
        let s = unsafe {
            eremite_sampler_new(
                temperature,
                top_p,
                top_k as c_int,
                seed.unwrap_or(0),
                seed.is_some().into(),
            )
        };
        (!s.is_null()).then_some(Sampler(s))
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { eremite_sampler_free(self.0) }
    }
}

/// One model's calls, one after another, on their own stack and working
/// buffers (eremite_stream). The calls of several streams run at the same
/// time. A posted call runs when run_streams advances its stream. What the
/// call was given must stay valid until it finishes.
pub struct Stream(*mut RawStream);

impl Stream {
    /// Makes a stream for `model`'s calls.
    pub fn new(model: &Model) -> Stream {
        Stream(unsafe { eremite_stream_new(model.raw) })
    }

    /// Posts the prefill of a prompt's `tokens` from position `from` on, in
    /// `s`, which holds every position before `from`.
    pub fn prefill(&self, s: StateHandle, tokens: &[i32], from: usize) {
        unsafe {
            eremite_stream_prefill(
                self.0,
                s.0.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as c_int,
                from as c_int,
                // No replay, no snapshot and no images.
                0,
                -1,
                ptr::null(),
            )
        }
    }

    /// Posts one decoding call: for each state, its token at its position.
    pub fn decode_rows(&self, states: &[StateHandle], tokens: &[i32], pos: &[i32]) {
        assert!(
            tokens.len() == states.len() && pos.len() == states.len(),
            "a decoding call has a token and a position for each state"
        );
        // The engine copies the handles and writes none of them.
        let handles = states.as_ptr().cast::<*mut RawState>().cast_mut();
        unsafe {
            eremite_stream_decode_rows(
                self.0,
                handles,
                tokens.as_ptr(),
                pos.as_ptr(),
                states.len() as c_int,
            )
        }
    }

    /// Returns the finished call's logits, or None when the call did not fit
    /// in memory.
    pub fn logits(&self) -> Option<Logits> {
        let lg = unsafe { eremite_stream_logits(self.0) };
        NonNull::new(lg.cast_mut()).map(Logits)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { eremite_stream_free(self.0) }
    }
}

/// Advances the streams until a call finishes, and returns that stream's
/// index. Returns None when no stream has a call posted.
pub fn run_streams(streams: &[&Stream]) -> Option<usize> {
    let raw: Vec<*mut RawStream> = streams.iter().map(|x| x.0).collect();
    let i = unsafe { eremite_run_streams(raw.as_ptr(), raw.len() as c_int) };
    (i >= 0).then_some(i as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder for one test. It is removed when the test ends.
    struct TestDir(std::path::PathBuf);

    impl TestDir {
        /// Makes an empty folder named after `name` and the process.
        fn new(name: &str) -> TestDir {
            let dir =
                std::env::temp_dir().join(format!("eremite-engine-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }

        /// Returns the folder's path as text.
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_checkpoint_s_family_comes_from_its_model_type() {
        let dir = TestDir::new("family");
        let config = dir.0.join("config.json");
        let error = Architecture::of(dir.path()).unwrap_err();
        assert!(error.starts_with("cannot read"), "{error}");
        std::fs::write(&config, r#"{"model_type": "deepseek_v41"}"#).unwrap();
        assert_eq!(Architecture::of(dir.path()), Ok(Architecture::DeepSeek));
        std::fs::write(&config, r#"{"model_type": "llama"}"#).unwrap();
        let error = Architecture::of(dir.path()).unwrap_err();
        assert!(
            error.ends_with("model_type llama is not one this server runs"),
            "{error}"
        );
        std::fs::write(&config, "{}").unwrap();
        let error = Architecture::of(dir.path()).unwrap_err();
        assert!(error.ends_with("no model_type"), "{error}");
        std::fs::write(&config, "{").unwrap();
        assert!(Architecture::of(dir.path()).is_err(), "not JSON");
    }

    #[test]
    fn deepseek_settings_are_kept_when_the_family_is_deepseek() {
        let mut options = Options::default();
        assert_eq!(options.deepseek(), None, "no family's settings yet");
        options.deepseek_mut().unwrap().ced = 512;
        options.configure(Architecture::DeepSeek).unwrap();
        assert_eq!(options.deepseek().map(|d| d.ced), Some(512));
        let mut unset = Options::default();
        unset.configure(Architecture::DeepSeek).unwrap();
        assert_eq!(unset.deepseek(), Some(&DeepSeekOptions::default()));
    }

    /// Writes a safetensors file `name` in `dir`: its header's length, the
    /// header `header`, then `data_bytes` bytes of data.
    fn write_safetensors(dir: &TestDir, name: &str, header: &str, data_bytes: usize) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + data_bytes, 0);
        std::fs::write(dir.0.join(name), bytes).unwrap();
    }

    #[test]
    fn every_weight_must_lie_inside_its_file() {
        let dir = TestDir::new("weights");
        let error = validate_weight_files(dir.path()).unwrap_err();
        assert!(
            error.contains("model.safetensors is unavailable"),
            "{error}"
        );
        let header = r#"{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        write_safetensors(&dir, "model.safetensors", header, 8);
        assert_eq!(validate_weight_files(dir.path()), Ok(()));
        write_safetensors(&dir, "model.safetensors", header, 4);
        let error = validate_weight_files(dir.path()).unwrap_err();
        assert_eq!(error, "weight w extends past model.safetensors");
        std::fs::write(dir.0.join("model.safetensors"), 1u64.to_le_bytes()).unwrap();
        let error = validate_weight_files(dir.path()).unwrap_err();
        assert_eq!(error, "invalid weight header in model.safetensors");
        // With an index, each shard holds the weights the index names.
        write_safetensors(&dir, "part.safetensors", header, 8);
        let index = r#"{"weight_map": {"w": "part.safetensors", "v": "part.safetensors"}}"#;
        std::fs::write(dir.0.join("model.safetensors.index.json"), index).unwrap();
        let error = validate_weight_files(dir.path()).unwrap_err();
        assert_eq!(error, "missing weight v in part.safetensors");
        let index = r#"{"weight_map": {"w": "part.safetensors"}}"#;
        std::fs::write(dir.0.join("model.safetensors.index.json"), index).unwrap();
        assert_eq!(validate_weight_files(dir.path()), Ok(()));
    }

    #[test]
    fn a_model_stays_on_its_thread() {
        // Resolves to the first impl unless T is Send; then both impls apply
        // and the call is ambiguous, which fails to compile.
        trait NotSend<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> NotSend<()> for T {}
        struct SendMarker;
        impl<T: ?Sized + Send> NotSend<SendMarker> for T {}
        let _ = <Model as NotSend<_>>::check;

        // The same for Sync.
        trait NotSync<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> NotSync<()> for T {}
        struct SyncMarker;
        impl<T: ?Sized + Sync> NotSync<SyncMarker> for T {}
        let _ = <Model as NotSync<_>>::check;
    }
}
