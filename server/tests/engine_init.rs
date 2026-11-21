//! The process settings that the server gives the engine. The engine refuses a
//! negative GPU timeout. A line logged after the engine starts reaches the
//! server's log hook. The idle work's interval is the engine's default
//! available-memory reading interval, 1 s.
use std::ffi::{CStr, c_char};
use std::sync::Mutex;

use eremite_server::engine;

/// The lines that the log hook received.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

extern "C" fn keep_line(line: *const c_char) {
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    LINES.lock().unwrap().push(line);
}

#[test]
fn a_logged_line_reaches_the_log_hook() {
    // The thread that starts the engine becomes the engine's. It needs a 16 MiB
    // stack.
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            assert!(
                !engine::init(
                    &engine::ProcessSettings {
                        gpu_timeout_seconds: -1.0,
                        ..Default::default()
                    },
                    Some(keep_line),
                    None,
                    None
                ),
                "the engine refuses a negative GPU timeout"
            );
            assert!(
                !engine::init(
                    &engine::ProcessSettings {
                        system_gpu_share: -0.1,
                        ..Default::default()
                    },
                    Some(keep_line),
                    None,
                    None
                ),
                "the engine refuses a negative system GPU share"
            );
            assert!(
                !engine::init(
                    &engine::ProcessSettings {
                        system_gpu_share: 0.6,
                        ..Default::default()
                    },
                    Some(keep_line),
                    None,
                    None
                ),
                "the engine refuses a system GPU share above 0.5"
            );
            assert!(
                !engine::init(
                    &engine::ProcessSettings {
                        system_gpu_window_seconds: -1.0,
                        ..Default::default()
                    },
                    Some(keep_line),
                    None,
                    None
                ),
                "the engine refuses a negative system GPU window"
            );
            assert!(
                engine::init(
                    &engine::ProcessSettings::default(),
                    Some(keep_line),
                    None,
                    None
                ),
                "the engine starts"
            );
            assert_eq!(
                engine::idle_work_interval_seconds(),
                1.0,
                "the idle work runs once per default available-memory reading interval"
            );
            engine::log("a line for the hook");
        })
        .unwrap()
        .join()
        .unwrap();
    let lines = LINES.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|line| line.ends_with("a line for the hook")),
        "the hook received {lines:?}"
    );
}
