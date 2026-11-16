//! The engine's C interface (src/eremite.h) and thin safe wrappers over it:
//! the process's settings, its log and its stop.

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
