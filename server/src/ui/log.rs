//! The terminal UI's log: the lines of the engine, of the server and of each
//! model, each with the model it is about. The engine's log hook and the
//! worker's hook for model lines keep them here.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine;

/// The lines that the log keeps. The hooks reach them here, so they are
/// global.
static KEPT: Mutex<Kept> = Mutex::new(Kept {
    lines: VecDeque::new(),
    logged: 0,
    printed: 0,
});
/// Whether the view is on the screen. While it is not, the hooks also write
/// each line to stderr.
static ON_SCREEN: AtomicBool = AtomicBool::new(false);

/// The log lines that the view keeps: the newest ones. The view copies them
/// for each frame. The reason for this count is unknown.
const KEPT_LINES: usize = 2000;

/// A line of the log.
#[derive(Clone, Debug, PartialEq)]
pub struct LogLine {
    /// The time of day when it was logged, such as 19:19:22.784.
    pub time: String,
    /// The index of the model that it is about. None for a line about no
    /// model.
    pub model: Option<usize>,
    /// The line's text.
    pub text: String,
}

/// The lines that the log keeps, and how many of them stderr showed.
struct Kept {
    /// The newest lines.
    lines: VecDeque<LogLine>,
    /// The lines logged since the start.
    logged: u64,
    /// Of those, the first ones that stderr showed.
    printed: u64,
}

/// Locks the kept lines. It also takes them after a thread panicked with the
/// lock held.
fn lock_kept() -> MutexGuard<'static, Kept> {
    KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Keeps `line`. The oldest line goes when the log is full.
fn keep(line: LogLine) {
    let mut kept = lock_kept();
    if !ON_SCREEN.load(Ordering::Acquire) {
        eprintln!("{} {}", line.time, line.text);
        kept.printed = kept.logged + 1;
    }
    kept.logged += 1;
    if kept.lines.len() == KEPT_LINES {
        kept.lines.pop_front();
    }
    kept.lines.push_back(line);
}

/// The engine's log hook: it keeps each line of the engine's log.
pub const ENGINE_LOG_HOOK: extern "C" fn(*const c_char) = keep_engine_line;

/// Keeps a line of the engine's log. The line starts with its time of day.
/// The engine passes a NUL-terminated line that stays valid during the call.
extern "C" fn keep_engine_line(line: *const c_char) {
    // Safety: the engine passes a NUL-terminated line, valid during the call.
    let line = unsafe { CStr::from_ptr(line) }.to_string_lossy();
    let (time, text) = line.split_once(' ').unwrap_or(("", &line));
    keep(LogLine {
        time: time.to_string(),
        model: None,
        text: text.to_string(),
    });
}

/// Keeps `line`, a line about the model at `model_index`: the worker's hook
/// for model lines. While the view is not on the screen, the line goes to the
/// engine's log instead.
pub fn keep_model_line(model_index: usize, line: &str) {
    if !ON_SCREEN.load(Ordering::Acquire) {
        engine::log(line);
        return;
    }
    keep(LogLine {
        time: format_time_of_day(SystemTime::now()),
        model: Some(model_index),
        text: line.to_string(),
    });
}

/// Sets whether the view is on the screen.
pub fn set_on_screen(on: bool) {
    let _kept = lock_kept();
    ON_SCREEN.store(on, Ordering::Release);
}

/// Returns a copy of the kept lines, the oldest first.
pub fn copy_lines() -> Vec<LogLine> {
    lock_kept().lines.iter().cloned().collect()
}

/// Returns the kept lines that stderr has not shown, as the engine's log
/// writes them, and counts them as shown.
pub fn take_unprinted_lines() -> Vec<String> {
    let mut kept = lock_kept();
    let first = kept.logged - kept.lines.len() as u64;
    let from = kept.printed.saturating_sub(first) as usize;
    let lines = kept
        .lines
        .iter()
        .skip(from)
        .map(|line| format!("{} {}", line.time, line.text))
        .collect();
    kept.printed = kept.logged;
    lines
}

/// Formats the local time of day at `now` as the engine's log does, such as
/// 19:19:22.784.
fn format_time_of_day(now: SystemTime) -> String {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = since.as_secs() as libc::time_t;
    // Safety: localtime_r fills the zeroed struct, and keeps no pointer to it.
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&seconds, &mut local) };
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        local.tm_hour,
        local.tm_min,
        local.tm_sec,
        since.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn the_hooks_keep_the_lines_with_their_time_and_model() {
        let line = CString::new("19:19:22.784 http 4: arrived from 127.0.0.1").unwrap();
        set_on_screen(true);
        ENGINE_LOG_HOOK(line.as_ptr());
        keep_model_line(1, "req 6: queued for qwen3.8-27b");
        let kept = copy_lines();
        assert!(kept.contains(&LogLine {
            time: "19:19:22.784".into(),
            model: None,
            text: "http 4: arrived from 127.0.0.1".into(),
        }));
        let model_line = kept
            .iter()
            .find(|line| line.text == "req 6: queued for qwen3.8-27b")
            .expect("the model's line");
        assert_eq!(model_line.model, Some(1));
        assert_eq!(model_line.time.len(), "19:19:22.784".len());
        let unprinted = take_unprinted_lines();
        assert!(unprinted.contains(&"19:19:22.784 http 4: arrived from 127.0.0.1".to_string()));
        assert!(take_unprinted_lines().is_empty(), "each line once");
        // the seconds and the milliseconds do not depend on the time zone
        let at = UNIX_EPOCH + std::time::Duration::from_millis(3_723_456);
        assert!(format_time_of_day(at).ends_with(":03.456"));
    }
}
