//! The engine's C interface (src/eremite.h) and thin safe wrappers over it:
//! the process's settings, its log and its stop; a model's settings; and a
//! model with its tokenizer. A [`Model`] is one registered checkpoint. Its
//! functions run on the engine's thread only; the tokenizer's run on any
//! thread.

use std::ffi::{CString, c_char, c_int};
use std::ptr;

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
    fn eremite_max_rows(m: *const RawModel) -> c_int;
    fn eremite_vocab_size(m: *const RawModel) -> c_int;
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

/// A registered model whose weights may still be unloaded. It stays on the
/// engine's thread.
pub struct Model {
    raw: *mut RawModel,
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
        (!m.is_null()).then_some(Model { raw: m })
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
