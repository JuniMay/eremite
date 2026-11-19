//! The engine's C interface (src/eremite.h) and thin safe wrappers over it:
//! the process's settings, its log and its stop; a model's settings; a model
//! with its tokenizer; the model's states, snapshots, saved states, calls,
//! drafts, media and sampling; and the statistics and memory. A
//! [`Model`] is one registered checkpoint, whose weights load on demand. Its
//! functions and its streams run on the engine's thread only; the tokenizer's
//! run on any thread.

use std::ffi::{CString, c_char, c_int};
use std::ptr::{self, NonNull};

use deepseek_recipe::stream::StreamTokenizer;

/// The version of src/eremite.h's interface that these declarations follow
/// (EREMITE_ABI).
const ABI: c_int = 5;

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
    memory_gb: f64,
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

/// A snapshot of a state's rolling rows (eremite_snapshot).
#[repr(C)]
struct RawSnapshot {
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

/// Context-copy proposals (eremite_lookup).
#[repr(C)]
struct RawLookup {
    _opaque: [u8; 0],
}

/// The choice of how many drafts a pass runs (eremite_draft_choice).
#[repr(C)]
struct RawDraftChoice {
    _opaque: [u8; 0],
}

/// A prompt's decoded media (eremite_media).
#[repr(C)]
struct RawMedia {
    _opaque: [u8; 0],
}

/// What a prefill ran (eremite_prefill_info).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillInfo {
    /// The chunks that the prefill ran its rows in.
    pub chunks: c_int,
    /// The rows that the prefill ran from its first row, the replayed ones
    /// included.
    pub tokens: c_int,
    /// The rows that the later layers ran: fewer than tokens when the
    /// approximate prefill or the pruning skipped some.
    pub decoder_tokens: c_int,
    /// Whether the approximate prefill applied.
    pub ced_applied: c_int,
    /// Whether the pruning applied.
    pub prune_applied: c_int,
    /// The position at which the prefill captured a snapshot; 0 when it
    /// captured none.
    pub prefix_tokens: c_int,
}

/// An expert cache's counts (eremite_cache_counts).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheCounts {
    /// Experts that a row found in memory.
    pub hits: u64,
    /// Experts that a row had to read.
    pub misses: u64,
    /// Experts that a read ahead brought in time.
    pub prefetch_hits: u64,
    /// Experts read ahead.
    pub prefetches: u64,
    /// Bytes read.
    pub bytes: u64,
    /// Experts evicted from the cache.
    pub evictions: u64,
    /// Reads ahead that came too late.
    pub late: u64,
    /// Reads ahead that were cancelled.
    pub cancelled: u64,
    /// Experts replaced by a substitute.
    pub substituted: u64,
    /// Slots mapped to another expert.
    pub remaps: u64,
    /// Slots given back to the system.
    pub given_back: u64,
    /// Decoded rows' experts that were in memory.
    pub row_own: u64,
    /// Decoded rows' experts that a read ahead brought.
    pub row_ahead: u64,
    /// Decoded rows' experts that came late.
    pub row_late: u64,
    /// Decoded rows' experts that were substituted.
    pub row_sub: u64,
    /// Decoded rows' experts read from the SSD.
    pub row_ssd: u64,
    /// Seconds spent waiting for reads.
    pub read_s: f64,
}

/// A model's counters and the process's memory (eremite_stats).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// All of the expert cache's counts.
    pub cache: CacheCounts,
    /// The counts of decoded rows.
    pub decode: CacheCounts,
    /// The counts of prompt rows.
    pub prefill: CacheCounts,
    /// Rows decoded.
    pub decode_rows: u64,
    /// Prompt rows computed.
    pub prefill_rows: u64,
    /// Seconds of decoding calls.
    pub decode_seconds: f64,
    /// GPU seconds of decoding calls.
    pub decode_gpu_seconds: f64,
    /// The running prefill's layer; 0: none.
    pub prefill_layer: c_int,
    /// The running prefill's rows.
    pub prefill_rows_now: c_int,
    /// The expert cache's slots.
    pub slots: c_int,
    /// The slots that hold an expert.
    pub mapped: c_int,
    /// The slots that the cache may have now.
    pub cap: c_int,
    /// The slots that the cache may have at most.
    pub max_slots: c_int,
    /// The experts in all.
    pub experts: c_int,
    /// The experts on the SSD only.
    pub uncached: c_int,
    /// Bytes of one slot.
    pub slot_bytes: u64,
    /// Bytes of the experts on the SSD only.
    pub uncached_bytes: u64,
    /// Bytes of the expert cache.
    pub cache_bytes: u64,
    /// Bytes of the dense weights.
    pub dense_bytes: u64,
    /// The process's memory footprint.
    pub footprint: u64,
    /// Bytes that the GPU has in use.
    pub gpu_in_use: u64,
    /// Bytes available to the process: the free memory, and the file cache
    /// above the part that the system keeps.
    pub available: u64,
    /// Metal's working-set recommendation; i64::MAX before the GPU starts.
    pub gpu_limit: i64,
    /// The system's memory in all.
    pub total: u64,
    /// Free memory.
    pub free: u64,
    /// File-backed memory.
    pub file: u64,
    /// The part of the file cache that the system keeps rather than give up
    /// for other memory: the smallest file cache read, or more when the
    /// system compressed other memory instead.
    pub file_kept: u64,
    /// Wired memory.
    pub wired: u64,
    /// The compressor's memory.
    pub compressor: u64,
    /// The memory that the compressor holds compressed.
    pub compressed: u64,
    /// Swap in use.
    pub swap_used: u64,
    /// Pages swapped out since the system started.
    pub swapouts: u64,
    /// The system's memory pressure: 1 normal, 2 warn, 4 critical.
    pub pressure: c_int,
    /// The model's layers.
    pub layers: c_int,
    /// The experts that a row chooses; 0 without routed experts.
    pub experts_per_row: c_int,
    /// The layers with routed experts.
    pub routed_layers: c_int,
    /// Each such layer's experts.
    pub experts_per_layer: c_int,
    /// The most model memory now.
    pub model_memory_limit: u64,
    /// What the models hold, state memory included.
    pub model_memory_used: u64,
    /// What is counted for the models.
    pub model_memory_committed: u64,
    /// The most ever counted.
    pub model_memory_peak: u64,
    /// Bytes allocated for states.
    pub state_allocated: u64,
    /// Bytes of the model's working buffers.
    pub model_workspace: u64,
    /// The model's own memory limit; 0: none.
    pub own_memory_limit: u64,
    /// What is counted for the model, its states included.
    pub own_memory_counted: u64,
    /// Bytes of the model's states.
    pub own_state_bytes: u64,
    /// The least memory that the model needs to run a full prompt pass: its
    /// weights, the pass's workspace and its smallest expert cache; 0 before
    /// it loads.
    pub required_bytes: u64,
    /// Seconds since the start: the engine thread outside run_streams,
    /// running a pass, and with no pass ready while one waits for the GPU,
    /// for reads, or for something else.
    pub thread_s: [f64; 5],
    /// Seconds that the GPU's main queue was busy.
    pub gpu_busy_s: f64,
    /// Seconds that the GPU's main queue was idle between command buffers, in
    /// each of thread_s's five states of the engine thread.
    pub gpu_idle_s: [f64; 5],
    /// Seconds with an expert read in flight.
    pub ssd_busy_s: f64,
    /// The summed seconds of demand reads.
    pub demand_read_s: f64,
    /// The summed seconds of background reads.
    pub background_read_s: f64,
}

/// The process's memory (eremite_memory_info).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemoryInfo {
    /// The memory limit in bytes.
    pub limit: u64,
    /// The bytes counted for the models, their state memory included.
    pub model_bytes: u64,
    /// The bytes of the models' state memory.
    pub state_bytes: u64,
}

/// eremite_tokenize's mode that also takes the added tokens' spellings as
/// those tokens.
const TOKENIZE_ADDED_TOKENS: c_int = 1;
/// eremite_tokenize's mode for a rendered prompt: as TOKENIZE_ADDED_TOKENS,
/// with the byte 0xFF for an image's place.
const TOKENIZE_PROMPT: c_int = 2;
/// eremite_token_kind's answer for an added token that tokenizer.json marks
/// special.
const TOKEN_SPECIAL: c_int = 2;

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
    fn eremite_run_idle_work(m: *mut RawModel) -> c_int;
    fn eremite_reclaim(bytes: u64) -> u64;
    fn eremite_read_memory() -> MemoryInfo;
    fn eremite_set_memory_limit(bytes: u64);
    fn eremite_refresh_stats();
    fn eremite_read_stats(m: *const RawModel, s: *mut Stats);
    fn eremite_read_routes(
        m: *const RawModel,
        row: c_int,
        experts: *mut u16,
        sources: *mut u8,
    ) -> c_int;
    fn eremite_wake();
    fn eremite_max_rows(m: *const RawModel) -> c_int;
    fn eremite_drafts_per_pass(m: *const RawModel) -> c_int;
    fn eremite_max_verify(m: *const RawModel) -> c_int;
    fn eremite_vocab_size(m: *const RawModel) -> c_int;
    fn eremite_is_stop_token(m: *const RawModel, token: i32) -> c_int;
    fn eremite_is_decision_model(m: *const RawModel) -> c_int;
    fn eremite_state_new(m: *mut RawModel) -> *mut RawState;
    fn eremite_state_free(m: *mut RawModel, s: *mut RawState);
    fn eremite_state_bytes(m: *const RawModel, s: *const RawState, positions: c_int) -> usize;
    fn eremite_state_copy(
        m: *mut RawModel,
        dst: *mut RawState,
        src: *const RawState,
        positions: c_int,
    );
    fn eremite_rewind_margin(m: *const RawModel) -> c_int;
    fn eremite_snapshot_take(
        m: *mut RawModel,
        s: *mut RawState,
        pos: c_int,
        logits: *const f32,
    ) -> *mut RawSnapshot;
    fn eremite_snapshot_restore(m: *mut RawModel, s: *mut RawState, x: *const RawSnapshot);
    fn eremite_snapshot_free(m: *mut RawModel, x: *mut RawSnapshot);
    fn eremite_snapshot_bytes(m: *const RawModel, with_logits: c_int) -> usize;
    fn eremite_snapshot_logits(m: *const RawModel, x: *const RawSnapshot) -> *const f32;
    fn eremite_replay_rows(m: *const RawModel, k: c_int) -> c_int;
    fn eremite_replay_start(m: *const RawModel, k: c_int) -> c_int;
    fn eremite_set_history(m: *mut RawModel, s: *mut RawState, tokens: *const i32, pos: c_int);
    fn eremite_state_write(
        m: *mut RawModel,
        s: *mut RawState,
        from: c_int,
        to: c_int,
        fd: c_int,
    ) -> i64;
    fn eremite_state_read(
        m: *mut RawModel,
        s: *mut RawState,
        positions: c_int,
        fds: *const c_int,
        count: c_int,
    ) -> c_int;
    fn eremite_snapshot_write(
        m: *mut RawModel,
        x: *const RawSnapshot,
        s: *mut RawState,
        pos: c_int,
        fd: c_int,
    ) -> i64;
    fn eremite_snapshot_read(m: *mut RawModel, fd: c_int) -> *mut RawSnapshot;
    fn eremite_state_identity(m: *mut RawModel, out: *mut u8);
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
    fn eremite_stream_verify(
        x: *mut RawStream,
        s: *mut RawState,
        rows: *mut i32,
        drafts: c_int,
        proposed: c_int,
        pos: c_int,
        sampler: *mut RawSampler,
    );
    fn eremite_stream_decide(
        x: *mut RawStream,
        tokens: *const i32,
        n: c_int,
        state_tokens: c_int,
        question_ends: *const i32,
        questions: c_int,
        rows: *const i32,
        ids: *const i32,
        count: c_int,
        images: *const RawMedia,
    );
    fn eremite_stream_next_token(x: *const RawStream, accepted: *mut c_int) -> i32;
    fn eremite_run_streams(streams: *const *mut RawStream, n: c_int) -> c_int;
    fn eremite_stream_logits(x: *const RawStream) -> *const f32;
    fn eremite_stream_captured_snapshot(x: *const RawStream) -> *mut RawSnapshot;
    fn eremite_stream_prefill_info(x: *const RawStream, info: *mut PrefillInfo);
    fn eremite_sampler_new(
        temperature: f64,
        top_p: f64,
        top_k: c_int,
        seed: u64,
        seed_set: c_int,
    ) -> *mut RawSampler;
    fn eremite_sampler_free(s: *mut RawSampler);
    fn eremite_sample(m: *const RawModel, logits: *const f32, s: *mut RawSampler) -> i32;
    fn eremite_lookup_new(capacity: c_int) -> *mut RawLookup;
    fn eremite_lookup_free(x: *mut RawLookup);
    fn eremite_lookup_append(x: *mut RawLookup, tokens: *const i32, n: c_int) -> c_int;
    fn eremite_lookup_propose(x: *const RawLookup, max: c_int, drafts: *mut i32) -> c_int;
    fn eremite_lookup_record_verification(x: *mut RawLookup, offered: c_int, accepted: c_int);
    fn eremite_draft_choice_new() -> *mut RawDraftChoice;
    fn eremite_draft_choice_free(x: *mut RawDraftChoice);
    fn eremite_draft_choice_next_count(x: *mut RawDraftChoice, most: c_int) -> c_int;
    fn eremite_draft_choice_record_pass(
        x: *mut RawDraftChoice,
        count: c_int,
        kept: c_int,
        seconds: f64,
    );
    fn eremite_media_open(
        m: *mut RawModel,
        data: *const *const u8,
        len: *const usize,
        n: c_int,
        message: *mut c_char,
        cap: usize,
    ) -> *mut RawMedia;
    fn eremite_media_expand(
        x: *mut RawMedia,
        input: *const i32,
        n: c_int,
        out: *mut i32,
        max: c_int,
    ) -> c_int;
    fn eremite_media_encode(x: *mut RawMedia, from: c_int) -> c_int;
    fn eremite_media_close(x: *mut RawMedia);
    fn eremite_model_tokenizer(m: *const RawModel) -> *mut RawTokenizer;
    #[cfg(test)]
    fn eremite_tokenizer_open(dir: *const c_char, image: *const c_char) -> *mut RawTokenizer;
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

/// Reads the process's memory once, for every model's statistics.
pub fn refresh_stats() {
    unsafe { eremite_refresh_stats() };
}

/// Evicts idle memory until `bytes` are freed or none is left. Returns the
/// bytes freed.
pub fn reclaim(bytes: u64) -> u64 {
    unsafe { eremite_reclaim(bytes) }
}

/// Returns the memory limit, the bytes counted for the models and their state
/// memory's bytes.
pub fn read_memory() -> MemoryInfo {
    unsafe { eremite_read_memory() }
}

/// Sets the memory limit in bytes.
pub fn set_memory_limit(bytes: u64) {
    unsafe { eremite_set_memory_limit(bytes) };
}

/// Makes run_streams return None soon. Any thread may call it.
pub fn wake() {
    unsafe { eremite_wake() }
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
    /// LiquidAI's d1 decision models: model_type lfm3_vl, with d1's question
    /// layout (d1_attention "set").
    D1,
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
            Some("lfm3_vl") if config["d1_attention"] == "set" => Ok(Architecture::D1),
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
    /// The model's own memory limit in GiB, its states included; 0: none.
    pub memory_gb: f64,
    /// The most proposals of the model's draft head that a pass verifies; 0:
    /// none.
    pub draft: i32,
    /// The most context-copy proposals that a pass verifies; 0: none.
    pub lookup: i32,
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
    /// It changes no result. None: it does whenever ced is 0.
    pub prefill_prune: Option<bool>,
    /// Whether a state keeps its compressed rows and index keys packed: the
    /// same values in about 28% of BF16's bytes.
    pub packed_state_rows: bool,
}

impl DeepSeekOptions {
    /// Returns whether a prefill skips the decoder batches that no later row
    /// needs: as prefill_prune says, or whenever ced is 0.
    pub fn prunes_prefill(&self) -> bool {
        self.prefill_prune.unwrap_or(self.ced == 0)
    }
}

impl Options {
    /// Sets the family's own settings to the checkpoint's family. Returns an
    /// error if they were set for another family, and for a decision model
    /// with drafts or lookup.
    pub fn configure(&mut self, architecture: Architecture) -> Result<(), String> {
        self.backend = match (&self.backend, architecture) {
            (BackendOptions::Unspecified, Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(DeepSeekOptions::default())
            }
            (BackendOptions::DeepSeek(d), Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(d.clone())
            }
            (BackendOptions::Unspecified, Architecture::D1) => {
                if self.draft != 0 || self.lookup != 0 {
                    return Err(format!(
                        "{}: a decision model takes no drafts and no lookup",
                        self.model
                    ));
                }
                BackendOptions::Unspecified
            }
            (BackendOptions::DeepSeek(_), Architecture::D1) => {
                return Err(format!(
                    "{}: DeepSeek's settings for a decision model",
                    self.model
                ));
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

/// The handle of a snapshot that the engine took or read. It stays valid
/// until Model::snapshot_free frees the snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotHandle(NonNull<RawSnapshot>);

/// Logits of one row or more, from a finished call or a snapshot. A call's
/// logits stay valid until the stream's next call, and a snapshot's until the
/// snapshot is freed.
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

    /// Returns a copy of the first row's `vocab` floats.
    ///
    /// # Safety
    ///
    /// The logits hold a row of `vocab` floats.
    pub unsafe fn copy_row(self, vocab: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), vocab) }.to_vec()
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

/// The room for the message that says why a media input cannot be decoded.
const MEDIA_MESSAGE_BYTES: usize = 256;

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
            lookup: o.lookup,
            memory_gb: o.memory_gb,
            deepseek: RawDeepSeekConfig {
                experts: experts.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                // The engine's approximate substitution stays off: it is for
                // the engine's tests only.
                substitute: 0.0,
                draft: o.draft,
                ced: deepseek.ced,
                prefill_prune: deepseek.prunes_prefill().into(),
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

    /// Gives memory back to other programs if they need it, while the model
    /// runs no request. Returns the model's expert cache slots after that, or
    /// 0 without a cache.
    pub fn run_idle_work(&self) -> i32 {
        unsafe { eremite_run_idle_work(self.raw) }
    }

    /// Returns the model's counters and the process's memory, as
    /// refresh_stats last read it.
    pub fn read_stats(&self) -> Stats {
        let mut stats = Stats::default();
        unsafe { eremite_read_stats(self.raw, &mut stats) };
        stats
    }

    /// Returns the experts that row `row` of the model's latest decoding pass
    /// chose in each layer with routed experts, layer by layer. With them, it
    /// returns where each one was: 0 in memory, 1 read ahead, 2 late, 3
    /// substituted, 4 from the SSD. Returns None for a model without routed
    /// experts, and for a row that did not run every layer.
    pub fn read_routes(&self, row: usize) -> Option<(Vec<u16>, Vec<u8>)> {
        let count = self.read_stats().experts_per_row.max(0) as usize;
        let mut experts = vec![0u16; count];
        let mut sources = vec![0u8; count];
        // Safety: the engine writes experts_per_row values to each buffer, and
        // each buffer holds that many.
        let read = unsafe {
            eremite_read_routes(
                self.raw,
                row as c_int,
                experts.as_mut_ptr(),
                sources.as_mut_ptr(),
            )
        };
        (read != 0 && count > 0).then_some((experts, sources))
    }

    /// Returns how many drafts the model makes itself for one pass: the
    /// draft setting, at most its draft head's.
    pub fn drafts_per_pass(&self) -> usize {
        unsafe { eremite_drafts_per_pass(self.raw) as usize }
    }

    /// Returns how many rows one verify call takes at most: the token and its
    /// drafts.
    pub fn max_verify(&self) -> usize {
        unsafe { eremite_max_verify(self.raw) as usize }
    }

    /// Returns the most rows that one prefill call takes, or None when one
    /// call takes the whole prompt, as it does for every family the engine
    /// runs.
    pub fn prefill_rows(&self) -> Option<usize> {
        None
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

    /// Returns whether the model is a decision model: it answers questions in
    /// one pass each, and keeps no states.
    pub fn is_decision_model(&self) -> bool {
        unsafe { eremite_is_decision_model(self.raw) != 0 }
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

    /// Returns the bytes that `s`, or a new state for None, takes when it
    /// holds `positions` positions.
    pub fn state_bytes(&self, s: Option<StateHandle>, positions: usize) -> u64 {
        let s = s.map_or(ptr::null_mut(), |s| s.0.as_ptr());
        unsafe { eremite_state_bytes(self.raw, s, positions as c_int) as u64 }
    }

    /// Copies the rows of `src` before `positions` that later positions do
    /// not change into `dst`. `src` may be a running request's state.
    pub fn state_copy(&self, dst: StateHandle, src: StateHandle, positions: usize) {
        unsafe { eremite_state_copy(self.raw, dst.0.as_ptr(), src.0.as_ptr(), positions as c_int) }
    }

    /// Returns how many positions before a snapshot's position a state may
    /// continue at.
    pub fn rewind_margin(&self) -> usize {
        unsafe { eremite_rewind_margin(self.raw) as usize }
    }

    /// Returns a snapshot of the rolling rows of `s` as they are at `pos`,
    /// with the logits that `pos`'s token is sampled from, if given.
    pub fn snapshot_take(
        &self,
        s: StateHandle,
        pos: usize,
        logits: Option<Logits>,
    ) -> SnapshotHandle {
        let logits = logits.map_or(ptr::null_mut(), |lg| lg.0.as_ptr());
        let x = unsafe { eremite_snapshot_take(self.raw, s.0.as_ptr(), pos as c_int, logits) };
        SnapshotHandle(NonNull::new(x).expect("the engine takes every snapshot asked for"))
    }

    /// Restores the rolling rows of snapshot `x` into `s`, which then
    /// continues at the snapshot's position.
    pub fn snapshot_restore(&self, s: StateHandle, x: SnapshotHandle) {
        unsafe { eremite_snapshot_restore(self.raw, s.0.as_ptr(), x.0.as_ptr()) }
    }

    /// Frees a snapshot.
    pub fn snapshot_free(&self, x: SnapshotHandle) {
        unsafe { eremite_snapshot_free(self.raw, x.0.as_ptr()) }
    }

    /// Returns the bytes that a snapshot takes, with its logits or without.
    pub fn snapshot_bytes(&self, with_logits: bool) -> u64 {
        unsafe { eremite_snapshot_bytes(self.raw, with_logits.into()) as u64 }
    }

    /// Returns the logits that a snapshot keeps, if it keeps any.
    pub fn snapshot_logits(&self, x: SnapshotHandle) -> Option<Logits> {
        let lg = unsafe { eremite_snapshot_logits(self.raw, x.0.as_ptr()) };
        NonNull::new(lg.cast_mut()).map(Logits)
    }

    /// Returns how many rows before position `k` a replay at `k` computes
    /// again, per layer on average. Returns None for a model that resumes
    /// only from snapshots.
    pub fn replay_rows(&self, k: usize) -> Option<usize> {
        let n = unsafe { eremite_replay_rows(self.raw, k as c_int) };
        (n >= 0).then_some(n as usize)
    }

    /// Returns the first position that a replay at `k` computes again.
    pub fn replay_start(&self, k: usize) -> usize {
        unsafe { eremite_replay_start(self.raw, k as c_int).max(0) as usize }
    }

    /// Sets the token history of `s` at `pos` from `tokens`, the tokens
    /// before `pos`, for a state that continues without a prefill.
    pub fn set_history(&self, s: StateHandle, tokens: &[i32], pos: usize) {
        unsafe { eremite_set_history(self.raw, s.0.as_ptr(), tokens.as_ptr(), pos as c_int) }
    }

    /// Writes one part of a saved state into `fd`: the rows of `s` that
    /// positions from `from` on wrote, up to `to`. Returns the bytes written,
    /// or None when a write failed or a second stop request ended it. With
    /// `fd` -1, it writes nothing and returns the bytes it would write.
    pub fn state_write(&self, s: StateHandle, from: usize, to: usize, fd: c_int) -> Option<u64> {
        let n =
            unsafe { eremite_state_write(self.raw, s.0.as_ptr(), from as c_int, to as c_int, fd) };
        (n >= 0).then_some(n as u64)
    }

    /// Fills `s`, a new state, with the rows before `positions` from a saved
    /// state's parts in `fds`, the first written first. Returns false when a
    /// part is short, damaged or of another shape; `s` is then of no use.
    pub fn state_read(&self, s: StateHandle, positions: usize, fds: &[c_int]) -> bool {
        unsafe {
            eremite_state_read(
                self.raw,
                s.0.as_ptr(),
                positions as c_int,
                fds.as_ptr(),
                fds.len() as c_int,
            ) == 0
        }
    }

    /// Writes snapshot `x` without its logits into `fd`, or for None, the
    /// rolling rows of `s` as they are at its end, `pos`. Returns the bytes
    /// written, or None when a write failed.
    pub fn snapshot_write(
        &self,
        x: Option<SnapshotHandle>,
        s: StateHandle,
        pos: usize,
        fd: c_int,
    ) -> Option<u64> {
        let x = x.map_or(ptr::null_mut(), |x| x.0.as_ptr());
        let n = unsafe { eremite_snapshot_write(self.raw, x, s.0.as_ptr(), pos as c_int, fd) };
        (n >= 0).then_some(n as u64)
    }

    /// Returns the snapshot that snapshot_write wrote into `fd`, or None when
    /// the file is short, damaged or of another shape.
    pub fn snapshot_read(&self, fd: c_int) -> Option<SnapshotHandle> {
        NonNull::new(unsafe { eremite_snapshot_read(self.raw, fd) }).map(SnapshotHandle)
    }

    /// Returns the identity of this model's saved states: a hash of what
    /// they depend on, the weights and the settings that change a state's
    /// rows. A state saved with another identity is not read.
    pub fn state_identity(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        unsafe { eremite_state_identity(self.raw, out.as_mut_ptr()) };
        out
    }

    /// Returns a token sampled from one row of logits.
    pub fn sample(&self, logits: Logits, sampler: &mut Sampler) -> i32 {
        unsafe { eremite_sample(self.raw, logits.0.as_ptr(), sampler.0) }
    }

    /// Decodes a prompt's media files, in the prompt's order, for this
    /// model's encoder. Returns why a file cannot be decoded.
    pub fn open_media(&self, media: &[Vec<u8>]) -> Result<Media, String> {
        let data: Vec<*const u8> = media.iter().map(|i| i.as_ptr()).collect();
        let len: Vec<usize> = media.iter().map(|i| i.len()).collect();
        let mut message = [0 as c_char; MEDIA_MESSAGE_BYTES];
        let x = unsafe {
            eremite_media_open(
                self.raw,
                data.as_ptr(),
                len.as_ptr(),
                media.len() as c_int,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        if x.is_null() {
            return Err(unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Media(x))
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
    /// Opens the tokenizer of the checkpoint in `dir` without its model, for
    /// the tests. `image` is the spelling of its model's image token, if it
    /// has one. A tokenizer that the engine cannot run ends the process with
    /// a message.
    #[cfg(test)]
    pub fn open(dir: &str, image: Option<&str>) -> Tokenizer {
        let dir = c_string(dir);
        let image = image.map(c_string);
        Tokenizer(unsafe {
            eremite_tokenizer_open(
                dir.as_ptr(),
                image.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        })
    }

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

    /// Returns whether a token is one of the tokenizer's added tokens that
    /// tokenizer.json marks special.
    pub fn is_special(&self, id: i32) -> bool {
        unsafe { eremite_token_kind(self.0, id) == TOKEN_SPECIAL }
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

/// The tokenizer as deepseek-recipe's stream processor reads it: a token's text
/// is its bytes, and the special tokens are the tokenizer's added tokens.
impl StreamTokenizer for Tokenizer {
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special_tokens && self.is_special_token(id) {
                continue;
            }
            bytes.extend_from_slice(self.token_bytes(id as i32));
        }
        // An unfinished character at the end becomes U+FFFD, so the processor
        // waits for the token that finishes it.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        Some(self.tokenize(text).len())
    }

    fn is_special_token(&self, id: u32) -> bool {
        self.is_added(id as i32)
    }
}

/// Context-copy proposals: drafts copied from where the context's last tokens
/// appeared before.
pub struct Lookup(*mut RawLookup);

impl Lookup {
    /// Returns a lookup for `capacity` tokens, or None if it cannot hold that
    /// many.
    pub fn new(capacity: usize) -> Option<Lookup> {
        let x = unsafe { eremite_lookup_new(capacity as c_int) };
        (!x.is_null()).then_some(Lookup(x))
    }

    /// Appends `tokens` to the context. Returns false when the lookup is full.
    pub fn append(&mut self, tokens: &[i32]) -> bool {
        unsafe { eremite_lookup_append(self.0, tokens.as_ptr(), tokens.len() as c_int) != 0 }
    }

    /// Writes up to `max` proposals into `drafts`, and returns how many.
    pub fn propose(&self, max: usize, drafts: &mut [i32]) -> usize {
        unsafe {
            eremite_lookup_propose(self.0, max.min(drafts.len()) as c_int, drafts.as_mut_ptr())
                .max(0) as usize
        }
    }

    /// Records how many of the `offered` proposals a verify call kept.
    pub fn record_verification(&mut self, offered: usize, accepted: usize) {
        unsafe { eremite_lookup_record_verification(self.0, offered as c_int, accepted as c_int) }
    }
}

impl Drop for Lookup {
    fn drop(&mut self) {
        unsafe { eremite_lookup_free(self.0) }
    }
}

/// How many of the draft head's drafts each pass of one request runs: the
/// count with the most expected tokens per second, from the drafts kept and
/// the times of earlier passes. The tokens do not depend on it.
pub struct DraftChoice(*mut RawDraftChoice);

impl DraftChoice {
    /// Returns a choice with no passes recorded.
    pub fn new() -> DraftChoice {
        DraftChoice(unsafe { eremite_draft_choice_new() })
    }

    /// Returns how many drafts the next pass runs, 0 to `most`.
    pub fn next_count(&mut self, most: usize) -> usize {
        unsafe {
            eremite_draft_choice_next_count(self.0, most.min(c_int::MAX as usize) as c_int).max(0)
                as usize
        }
    }

    /// Records a pass with `count` drafts that kept the first `kept` of them
    /// and took `seconds`, with no other call beside it.
    pub fn record_pass(&mut self, count: usize, kept: usize, seconds: f64) {
        unsafe { eremite_draft_choice_record_pass(self.0, count as c_int, kept as c_int, seconds) }
    }
}

impl Default for DraftChoice {
    fn default() -> DraftChoice {
        DraftChoice::new()
    }
}

impl Drop for DraftChoice {
    fn drop(&mut self) {
        unsafe { eremite_draft_choice_free(self.0) }
    }
}

/// A prompt's decoded media. Dropping it closes them and gives their memory
/// back.
pub struct Media(*mut RawMedia);

/// Why the media's spans could not be put in a prompt.
#[derive(Debug, PartialEq, Eq)]
pub enum ExpandError {
    /// The prompt has another number of image places than there are media.
    PlacesAndMediaDiffer,
    /// With the spans, the prompt has this many tokens, more than allowed.
    TooLong(usize),
}

impl Media {
    /// Returns the prompt with each image place replaced by its image's span,
    /// at most `max` tokens.
    pub fn expand(&mut self, prompt: &[i32], max: usize) -> Result<Vec<i32>, ExpandError> {
        let mut out = vec![0i32; max];
        let n = unsafe {
            eremite_media_expand(
                self.0,
                prompt.as_ptr(),
                prompt.len() as c_int,
                out.as_mut_ptr(),
                max as c_int,
            )
        };
        let Ok(n) = usize::try_from(n) else {
            return Err(ExpandError::PlacesAndMediaDiffer);
        };
        if n > max {
            return Err(ExpandError::TooLong(n));
        }
        out.truncate(n);
        Ok(out)
    }

    /// Encodes the media at positions from `from` on, for the next prefill.
    /// Returns how many it encoded, or why the encoder does not fit.
    pub fn encode(&mut self, from: usize) -> Result<usize, String> {
        let count = unsafe { eremite_media_encode(self.0, from as c_int) };
        if count < 0 {
            Err(
                "the global model-memory budget cannot fit the image encoder and its workspace"
                    .into(),
            )
        } else {
            Ok(count as usize)
        }
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        unsafe { eremite_media_close(self.0) }
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
    /// `s`. Without `replay`, `s` holds every position before `from`. With
    /// it, `s` holds only the rows that Model::state_copy copies, and the
    /// prefill computes the rest again. With `capture`, the prefill also takes
    /// a snapshot at that position, from `from` on and before the prompt's
    /// end. `media` gives the encoded media of the prompt's images.
    pub fn prefill(
        &self,
        s: StateHandle,
        tokens: &[i32],
        from: usize,
        replay: bool,
        capture: Option<usize>,
        media: Option<&Media>,
    ) {
        unsafe {
            eremite_stream_prefill(
                self.0,
                s.0.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as c_int,
                from as c_int,
                replay.into(),
                capture.map_or(-1, |p| p as c_int),
                media.map_or(ptr::null(), |media| media.0),
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

    /// Posts one verify call in `s`: the token `rows[0]` at `pos` and its
    /// drafts in the other rows, decoded together and sampled with
    /// `sampler`. Without `proposed`, the model's draft head fills the other
    /// rows; with it, they hold the caller's proposals.
    pub fn verify(
        &self,
        s: StateHandle,
        rows: &mut [i32],
        proposed: bool,
        pos: usize,
        sampler: &mut Sampler,
    ) {
        unsafe {
            eremite_stream_verify(
                self.0,
                s.0.as_ptr(),
                rows.as_mut_ptr(),
                rows.len() as c_int - 1,
                proposed.into(),
                pos as c_int,
                sampler.0,
            )
        }
    }

    /// Posts one pass of a decision model: `tokens`, a state's first
    /// `state_tokens` tokens and questions after it that end at
    /// `question_ends`. The pass reads the logit of token `ids[i]` at row
    /// `rows[i]` for each i. `media` gives its encoded images, whose spans
    /// the tokens hold.
    pub fn decide(
        &self,
        tokens: &[i32],
        state_tokens: usize,
        question_ends: &[i32],
        rows: &[i32],
        ids: &[i32],
        media: Option<&Media>,
    ) {
        assert_eq!(rows.len(), ids.len(), "a pass reads one token at each row");
        unsafe {
            eremite_stream_decide(
                self.0,
                tokens.as_ptr(),
                tokens.len() as c_int,
                state_tokens as c_int,
                question_ends.as_ptr(),
                question_ends.len() as c_int,
                rows.as_ptr(),
                ids.as_ptr(),
                ids.len() as c_int,
                media.map_or(ptr::null(), |media| media.0),
            )
        }
    }

    /// Returns a copy of the finished decision pass's `count` logits, or None
    /// when the pass did not fit in memory.
    pub fn copy_decision_logits(&self, count: usize) -> Option<Vec<f32>> {
        let logits = self.logits()?;
        Some(unsafe { std::slice::from_raw_parts(logits.0.as_ptr(), count) }.to_vec())
    }

    /// Returns the finished verify call's next token, and how many of its
    /// drafts it kept.
    pub fn next_token(&self) -> (i32, usize) {
        let mut accepted = 0;
        let token = unsafe { eremite_stream_next_token(self.0, &mut accepted) };
        (token, accepted.max(0) as usize)
    }

    /// Returns the finished call's logits, or None when the call did not fit
    /// in memory.
    pub fn logits(&self) -> Option<Logits> {
        let lg = unsafe { eremite_stream_logits(self.0) };
        NonNull::new(lg.cast_mut()).map(Logits)
    }

    /// Returns the snapshot that the finished prefill captured, if it took
    /// one.
    pub fn captured_snapshot(&self) -> Option<SnapshotHandle> {
        NonNull::new(unsafe { eremite_stream_captured_snapshot(self.0) }).map(SnapshotHandle)
    }

    /// Returns what the finished prefill ran.
    pub fn prefill_info(&self) -> PrefillInfo {
        let mut info = PrefillInfo::default();
        unsafe { eremite_stream_prefill_info(self.0, &mut info) };
        info
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { eremite_stream_free(self.0) }
    }
}

/// Advances the streams until a call finishes, and returns that stream's
/// index. Returns None when no stream has a call posted, and when wake asks
/// it to return.
pub fn run_streams(streams: &[&Stream]) -> Option<usize> {
    let raw: Vec<*mut RawStream> = streams.iter().map(|x| x.0).collect();
    let i = unsafe { eremite_run_streams(raw.as_ptr(), raw.len() as c_int) };
    (i >= 0).then_some(i as usize)
}

#[cfg(test)]
pub(crate) use tests::TestDir;

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder for one test. It is removed when the test ends.
    pub(crate) struct TestDir(
        /// The path owned by this test.
        pub(crate) std::path::PathBuf,
    );

    impl TestDir {
        /// Makes an empty folder named after `name` and the process.
        pub(crate) fn new(name: &str) -> TestDir {
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
        std::fs::write(
            &config,
            r#"{"model_type": "lfm3_vl", "d1_attention": "set"}"#,
        )
        .unwrap();
        assert_eq!(Architecture::of(dir.path()), Ok(Architecture::D1));
        std::fs::write(&config, r#"{"model_type": "lfm3_vl"}"#).unwrap();
        assert!(
            Architecture::of(dir.path()).is_err(),
            "an LFM3-VL checkpoint without d1's layout"
        );
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
    fn a_decision_model_takes_no_drafts_lookup_or_deepseek_settings() {
        let mut options = Options::default();
        options.configure(Architecture::D1).unwrap();
        assert_eq!(options.backend, BackendOptions::Unspecified);
        for change in [
            |o: &mut Options| o.draft = 1,
            |o: &mut Options| o.lookup = 3,
            |o: &mut Options| o.deepseek_mut().unwrap().ced = 512,
        ] {
            let mut options = Options::default();
            change(&mut options);
            assert!(options.configure(Architecture::D1).is_err(), "{options:?}");
        }
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

    /// The test reads the tokenizer of the checkpoint that EREMITE_CHECKPOINT
    /// names. Without it, the test returns at once without checking.
    #[test]
    fn the_stream_processor_reads_tokens_as_their_bytes() {
        let Ok(dir) = std::env::var("EREMITE_CHECKPOINT") else {
            return;
        };
        let tokenizer = Tokenizer::open(&dir, Some("<｜deepseek_image｜>"));
        let image = tokenizer.find_added_token("<｜deepseek_image｜>").unwrap();
        let mut ids: Vec<u32> = tokenizer
            .tokenize("Hello")
            .iter()
            .map(|&t| t as u32)
            .collect();
        ids.push(image as u32);
        assert!(tokenizer.is_special_token(image as u32));
        assert_eq!(tokenizer.decode_ids(&ids, true), Ok("Hello".into()));
        assert_eq!(
            tokenizer.decode_ids(&ids, false),
            Ok("Hello<｜deepseek_image｜>".into())
        );
        assert_eq!(
            tokenizer.count_tokens("Hello, world"),
            Some(tokenizer.tokenize("Hello, world").len())
        );
    }

    #[test]
    fn a_lookup_proposes_what_followed_the_last_tokens_before() {
        // A match needs the last 12 tokens.
        let repeated: Vec<i32> = (1..=12).collect();
        let mut lookup = Lookup::new(32).expect("a lookup for 32 tokens");
        assert!(lookup.append(&repeated));
        assert!(lookup.append(&[100, 101, 102]));
        let mut drafts = [0; 4];
        assert_eq!(lookup.propose(3, &mut drafts), 0, "no earlier match yet");
        assert!(lookup.append(&repeated));
        let count = lookup.propose(3, &mut drafts);
        assert_eq!(&drafts[..count], &[100, 101, 102], "what followed 1 to 12");
        lookup.record_verification(count, 2);
        assert!(!lookup.append(&[1; 6]), "33 tokens do not fit");
    }

    #[test]
    fn the_draft_choice_stays_within_its_most() {
        let mut choice = DraftChoice::new();
        for most in 0..4 {
            assert!(choice.next_count(most) <= most);
        }
        for _ in 0..8 {
            choice.record_pass(2, 2, 0.05);
        }
        assert!(choice.next_count(2) <= 2);
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
