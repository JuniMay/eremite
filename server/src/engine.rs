//! The engine's C interface (src/eremite.h) and thin safe wrappers over it:
//! the process's settings, its log and its stop, and a model's settings.

use std::ffi::{CString, c_char, c_int};

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

unsafe extern "C" {
    fn eremite_init(p: *const RawProcess) -> c_int;
    fn eremite_idle_work_interval_seconds() -> f64;
    fn eremite_is_stopping() -> c_int;
    fn eremite_request_stop();
    fn eremite_count_stop_requests() -> c_int;
    fn eremite_log(line: *const c_char);
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
}
