//! The worker runs each model's requests one at a time and reports their
//! events. A request decodes until its token limit or a stop token. The worker
//! refuses a prompt that does not fit the context, and invalid sampling
//! settings. A dropped guard cancels its request. The stop ends a decoding
//! request, and the worker queues no request after it. The log shows each
//! request's note when it is queued, and its counts at its end. Set
//! EREMITE_CHECKPOINT to a DeepSeek V4.1 checkpoint's directory, and
//! EREMITE_EXPERTS to GGUF experts if wanted. Without EREMITE_CHECKPOINT, the
//! test is skipped.
use std::ffi::{CStr, c_char};
use std::sync::Mutex;

use eremite_server::{config, engine, worker};
use tokio::sync::mpsc::UnboundedReceiver;

/// The tokens in the test model's context.
const CTX: usize = 4096;
/// The most tokens that the short replies have.
const MAX_TOKENS: i64 = 8;
/// The most tokens that the long replies have. They outlast the test's checks.
const LONG_REPLY_TOKENS: i64 = 1000;
/// Greedy sampling: the most likely token each time.
const GREEDY: worker::Sampling = worker::Sampling {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 0,
    seed: None,
};
/// The note of every request that the test submits.
const NOTE: &str = "from the worker test";

/// The lines that the log hook received.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

extern "C" fn keep_line(line: *const c_char) {
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    LINES.lock().unwrap().push(line);
}

/// What the worker reported about one request.
struct Reply {
    /// The prompt tokens that DecodingStarted counted, if it came.
    prompt_tokens: Option<usize>,
    /// The tokens of the Tokens events.
    tokens: Vec<u32>,
    /// The last event. It is None when the reading stopped at the first token.
    last: Option<worker::Event>,
}

/// Reads `events` until the last one. With `until_first_token`, it stops once
/// the reply's first token came.
fn read_events(events: &mut UnboundedReceiver<worker::Event>, until_first_token: bool) -> Reply {
    let mut reply = Reply {
        prompt_tokens: None,
        tokens: Vec::new(),
        last: None,
    };
    loop {
        match events
            .blocking_recv()
            .expect("the worker sends a last event")
        {
            worker::Event::DecodingStarted {
                prompt_tokens,
                reused_tokens,
            } => {
                assert_eq!(reused_tokens, 0, "the worker keeps no state");
                reply.prompt_tokens = Some(prompt_tokens);
            }
            worker::Event::Tokens(tokens) => {
                reply.tokens.extend(tokens);
                if until_first_token && !reply.tokens.is_empty() {
                    return reply;
                }
            }
            last => {
                reply.last = Some(last);
                return reply;
            }
        }
    }
}

/// Submits a request for `prompt` with `sampling` and `max_tokens` to the first
/// model.
fn submit(
    prompt: &[i32],
    sampling: worker::Sampling,
    max_tokens: i64,
) -> (worker::CancelGuard, UnboundedReceiver<worker::Event>) {
    let request = worker::Request {
        model_index: 0,
        prompt: prompt.to_vec(),
        sampling,
        max_tokens,
        note: NOTE.into(),
    };
    match worker::submit(request) {
        Ok(submitted) => submitted,
        Err(worker::SubmitError::Busy(message)) => panic!("the worker is busy: {message}"),
    }
}

/// Submits a request for `prompt` with `sampling` and `max_tokens`, and returns
/// its whole reply.
fn run(prompt: &[i32], sampling: worker::Sampling, max_tokens: i64) -> Reply {
    let (_guard, mut events) = submit(prompt, sampling, max_tokens);
    read_events(&mut events, false)
}

#[test]
fn requests_run_to_their_end_and_report_their_events() {
    let Ok(checkpoint) = std::env::var("EREMITE_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_CHECKPOINT");
        return;
    };
    let mut options = engine::Options {
        model: checkpoint,
        ctx: CTX as i32,
        ..Default::default()
    };
    options.deepseek_mut().unwrap().experts = std::env::var("EREMITE_EXPERTS").ok();
    let settings = worker::ModelSettings {
        id: "deepseek".into(),
        options,
    };
    assert!(worker::start(
        vec![settings],
        config::Budgets::default(),
        Some(keep_line),
        None
    ));
    let tokenizer = worker::list_models()[0].1;
    let prompt = tokenizer.tokenize("The first ten prime numbers are");

    // A greedy reply ends at its token limit, or before it at a stop token.
    let first = run(&prompt, GREEDY, MAX_TOKENS);
    assert_eq!(first.prompt_tokens, Some(prompt.len()));
    match first.last {
        Some(worker::Event::Done {
            reason: worker::FinishReason::Length,
        }) => assert_eq!(first.tokens.len(), MAX_TOKENS as usize),
        Some(worker::Event::Done {
            reason: worker::FinishReason::StopToken,
        }) => assert!(first.tokens.len() < MAX_TOKENS as usize),
        other => panic!("the reply ended with {other:?}"),
    }
    // The same greedy request gives the same tokens.
    assert_eq!(run(&prompt, GREEDY, MAX_TOKENS).tokens, first.tokens);

    // A prompt as long as the context leaves no room for a reply.
    let long_prompt = vec![prompt[0]; CTX];
    let refused = run(&long_prompt, GREEDY, MAX_TOKENS).last;
    let expected = format!("the prompt has {CTX} tokens; the context holds {CTX}");
    assert!(
        matches!(&refused, Some(worker::Event::Refused(message)) if *message == expected),
        "{refused:?}"
    );
    let negative = worker::Sampling {
        temperature: -1.0,
        ..GREEDY
    };
    let refused = run(&prompt, negative, MAX_TOKENS).last;
    assert!(
        matches!(&refused, Some(worker::Event::Refused(message)) if message == "invalid sampling settings"),
        "{refused:?}"
    );

    // Dropping the guard cancels a decoding request.
    let (guard, mut events) = submit(&prompt, GREEDY, LONG_REPLY_TOKENS);
    read_events(&mut events, true);
    drop(guard);
    let last = read_events(&mut events, false).last;
    assert!(
        matches!(
            last,
            Some(worker::Event::Done {
                reason: worker::FinishReason::Cancelled
            })
        ),
        "{last:?}"
    );

    // The stop ends a decoding request. No request is queued after it.
    let (_guard, mut events) = submit(&prompt, GREEDY, LONG_REPLY_TOKENS);
    read_events(&mut events, true);
    worker::stop();
    let last = read_events(&mut events, false).last;
    assert!(
        matches!(
            last,
            Some(worker::Event::Done {
                reason: worker::FinishReason::ServerStop
            })
        ),
        "{last:?}"
    );
    let request = worker::Request {
        model_index: 0,
        prompt,
        sampling: GREEDY,
        max_tokens: MAX_TOKENS,
        note: NOTE.into(),
    };
    let refused = worker::submit(request).err();
    assert!(
        matches!(&refused, Some(worker::SubmitError::Busy(message)) if message == "the server is stopping"),
        "the worker queued a request after the stop"
    );

    // The first request is number 1. The log shows its note and its end.
    let lines = LINES.lock().unwrap();
    for wanted in [format!("req 1 {NOTE}"), "req 1: queued for deepseek".into()] {
        assert!(
            lines.iter().any(|line| line.contains(&wanted)),
            "no log line with {wanted:?}"
        );
    }
    assert!(
        lines
            .iter()
            .any(|line| line.contains("req 1: length: ") || line.contains("req 1: stop: ")),
        "no end line of request 1"
    );
}
