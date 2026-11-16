//! A stop request: the engine reports that it is stopping and counts the
//! request, and the on_stop hook runs on a thread of its own.
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

use eremite_server::engine;

/// The threads that ran the on_stop hook.
static STOP_THREADS: Mutex<Vec<ThreadId>> = Mutex::new(Vec::new());
/// Notified when the on_stop hook has run.
static STOPPED: Condvar = Condvar::new();

extern "C" fn keep_stop_thread() {
    STOP_THREADS
        .lock()
        .unwrap()
        .push(std::thread::current().id());
    STOPPED.notify_all();
}

#[test]
fn a_stop_request_runs_the_stop_hook_on_a_thread_of_its_own() {
    // The thread that starts the engine becomes the engine's. It needs a 16 MiB
    // stack.
    let engine_thread = std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            assert!(engine::init(
                &engine::ProcessSettings::default(),
                None,
                None,
                Some(keep_stop_thread)
            ));
            assert!(!engine::is_stopping());
            assert_eq!(engine::count_stop_requests(), 0);
            engine::request_stop();
            assert!(engine::is_stopping());
            assert_eq!(engine::count_stop_requests(), 1);
            std::thread::current().id()
        })
        .unwrap()
        .join()
        .unwrap();
    let threads = STOPPED
        .wait_while(STOP_THREADS.lock().unwrap(), |threads| threads.is_empty())
        .unwrap();
    assert_eq!(threads.len(), 1, "the hook ran once");
    assert_ne!(
        threads[0], engine_thread,
        "the hook ran on a thread of its own"
    );
}
