//! The worker runs a decision model's decisions and reports their events. A
//! decision runs its passes one after another, and its last event holds each
//! pass's logits and the tokens they read. A decision about an image puts
//! the image's span in place of its place in each pass. The model unloads
//! when idle and loads again for the next decision, its vision weights
//! too. The worker refuses a reply request for a decision model, and a
//! pass longer than the context. A dropped guard cancels a decision, and the
//! stop ends one: a running decision before its next pass, a waiting one
//! before its first. The log shows each decision's note when it is queued,
//! and its counts at its end. Set EREMITE_D1_CHECKPOINT to d1-2B's directory.
//! Without it, the test is skipped.
use std::ffi::{CStr, c_char};
use std::sync::{Condvar, Mutex};

use eremite_server::decision::d1;
use eremite_server::{config, engine, worker};
use tokio::runtime::Runtime;
use tokio::sync::mpsc::UnboundedReceiver;

mod common;
use common::make_test_bmp;

/// The tokens in the test model's context.
const CTX: usize = 4096;
/// The seconds without requests before the model unloads: it unloads when
/// the worker next keeps the memory reserve free.
const IDLE_UNLOAD_S: f64 = 1e-3;
/// A JSON state and one yes-or-no question, as d1's prompt holds them.
const TOKENS: [i32; 89] = [
    124894, 124899, 5922, 207, 2718, 229, 496, 99489, 6380, 827, 360, 496, 6929, 6380, 496, 16308,
    52966, 278, 48505, 7285, 1917, 229, 10016, 229, 496, 3166, 6380, 827, 360, 496, 16414, 273,
    6380, 19843, 733, 229, 3255, 920, 733, 229, 3255, 22, 29, 207, 360, 40439, 360, 496, 37507,
    6380, 2846, 920, 360, 496, 14700, 6380, 3495, 207, 229, 1130, 64024, 63509, 833, 29088, 278,
    29936, 7986, 267, 48505, 7285, 7450, 467, 278, 2064, 64929, 57922, 415, 18171, 432, 794, 1049,
    22, 124900, 207, 124899, 63514, 207, 124901, 124902,
];
/// The state's tokens at the start of TOKENS.
const STATE: usize = 63;
/// The tokens of yes, Yes, no, No and NO, read at the question's last row,
/// their logits in LiquidAI's own code in FP32, and the largest difference of
/// its BF16 run from them.
const IDS: [i32; 5] = [11683, 12447, 2243, 4547, 19598];
const REFERENCE: [f32; 5] = [8.003815, 10.504749, 5.083753, 7.219336, 1.851632];
const BOUND: f32 = 0.0619;
/// The passes of the decisions that the test cancels and stops. They outlast
/// the checks of the decisions' status.
const LONG_PASSES: usize = 6;
/// The tokens of each long pass's question, after its state.
const QUESTION_TOKENS: usize = 8;
/// The note of every decision that the test submits.
const NOTE: &str = "from the decision test";
/// A text state after a 400 x 208 image's place (the image token), and three
/// questions: yes or no, a choice of three, and a score of three levels.
const CARD: [i32; 114] = [
    124894, 124899, 5922, 207, 125009, 125030, 125010, 41, 1426, 4559, 34, 3599, 376, 11775, 309,
    267, 5322, 10590, 6542, 20242, 64673, 63509, 833, 4439, 278, 7628, 8286, 6542, 64929, 57922,
    415, 18171, 432, 794, 1049, 22, 124900, 207, 124899, 63514, 207, 124901, 124902, 63509, 833,
    28327, 3599, 355, 31211, 462, 278, 1583, 8527, 64929, 13314, 833, 41, 2959, 207, 42, 4752, 207,
    43, 6542, 8, 57922, 415, 278, 7331, 4713, 1049, 22, 124900, 207, 124899, 63514, 207, 124901,
    124902, 63509, 833, 3522, 34880, 355, 278, 7628, 64929, 24, 37787, 207, 25, 15175, 207, 26,
    9398, 8, 57922, 415, 267, 2936, 19763, 229, 24, 21, 26, 1049, 22, 124900, 207, 124899, 63514,
    207, 124901, 124902,
];
/// CARD's state, its questions' ends and the rows of its answers, before the
/// image's span takes its place; the tokens with the span; the tokens that
/// its questions read; their logits in LiquidAI's own code in FP32; and the
/// largest difference of its BF16 run from them.
const CARD_STATE: usize = 23;
const CARD_ENDS: [i32; 3] = [43, 79, 114];
const CARD_EXPANDED: usize = 185;
const CARD_IDS: [(i32, [i32; 6]); 3] = [
    (42, [11683, 12447, 2243, 4547, 19598, -1]),
    (78, [41, 334, 42, 378, 43, 340]),
    (113, [24, 25, 26, -1, -1, -1]),
];
const CARD_LOGITS: [f32; 14] = [
    2.908662, 7.315761, 5.76309, 10.504194, 3.416532, 13.300821, 8.844008, 7.259619, 2.441981,
    8.001862, 3.481334, 6.407514, 7.811986, 12.584948,
];
const CARD_BOUND: f32 = 0.0583;

/// The lines that the log hook received, and the condition that each new line
/// notifies.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static ARRIVED: Condvar = Condvar::new();

extern "C" fn keep_line(line: *const c_char) {
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    LINES.lock().unwrap().push(line);
    ARRIVED.notify_all();
}

/// Waits for a log line from line `from` on that contains `text`, and returns
/// the number of lines after it.
fn wait_for_line(from: usize, text: &str) -> usize {
    let mut lines = LINES.lock().unwrap();
    loop {
        if let Some(at) = lines[from..].iter().position(|line| line.contains(text)) {
            return from + at + 1;
        }
        lines = ARRIVED.wait(lines).unwrap();
    }
}

/// The pass of the JSON state and its question, which reads IDS at the
/// question's last row.
fn json_pass() -> d1::Pass {
    d1::Pass {
        tokens: TOKENS.to_vec(),
        state_tokens: STATE,
        question_ends: vec![TOKENS.len() as i32],
        rows: vec![TOKENS.len() as i32 - 1; IDS.len()],
        ids: IDS.to_vec(),
    }
}

/// A pass of `rows` tokens, the JSON state's first token repeated: a state,
/// then one question. It reads IDS at its last row.
fn long_pass(rows: usize) -> d1::Pass {
    d1::Pass {
        tokens: vec![TOKENS[0]; rows],
        state_tokens: rows - QUESTION_TOKENS,
        question_ends: vec![rows as i32],
        rows: vec![rows as i32 - 1; IDS.len()],
        ids: IDS.to_vec(),
    }
}

/// The pass of CARD, with its image's place.
fn card_pass() -> d1::Pass {
    let (mut rows, mut ids) = (Vec::new(), Vec::new());
    for (row, tokens) in CARD_IDS {
        for id in tokens.into_iter().filter(|&id| id >= 0) {
            rows.push(row);
            ids.push(id);
        }
    }
    d1::Pass {
        tokens: CARD.to_vec(),
        state_tokens: CARD_STATE,
        question_ends: CARD_ENDS.to_vec(),
        rows,
        ids,
    }
}

/// Runs the decision of CARD about the 400 x 208 test image, and returns its
/// logits and the tokens that it read.
fn decide_card() -> (Vec<f32>, usize) {
    let request = worker::DecisionRequest {
        model_index: 0,
        passes: vec![card_pass()],
        media: vec![make_test_bmp(400, 208)],
        note: NOTE.into(),
    };
    let Ok((_guard, mut events)) = worker::submit_decision(request) else {
        panic!("the worker is busy");
    };
    match read_last(&mut events) {
        worker::Event::Decided { mut logits, tokens } => (logits.remove(0), tokens),
        other => panic!("the decision ended with {other:?}"),
    }
}

/// Submits a decision of `passes` to the first model.
fn submit(passes: Vec<d1::Pass>) -> (worker::CancelGuard, UnboundedReceiver<worker::Event>) {
    let request = worker::DecisionRequest {
        model_index: 0,
        passes,
        media: Vec::new(),
        note: NOTE.into(),
    };
    match worker::submit_decision(request) {
        Ok(submitted) => submitted,
        Err(worker::SubmitError::Busy(message)) => panic!("the worker is busy: {message}"),
    }
}

/// Waits for the last event of a request.
fn read_last(events: &mut UnboundedReceiver<worker::Event>) -> worker::Event {
    events
        .blocking_recv()
        .expect("the worker sends a last event")
}

/// Submits a decision of the long passes.
fn submit_long() -> (worker::CancelGuard, UnboundedReceiver<worker::Event>) {
    submit((0..LONG_PASSES).map(|_| long_pass(CTX)).collect())
}

/// Asks the worker to publish its status until the request `id` runs its
/// passes.
fn wait_until_running(runtime: &Runtime, id: u64) {
    loop {
        let status = worker::build_model_status_json(0);
        if status["requests"].as_array().is_some_and(|requests| {
            requests
                .iter()
                .any(|request| request["request"] == id && request["status"] == "prefill")
        }) {
            return;
        }
        runtime.block_on(worker::refresh_status());
    }
}

/// Checks that `events` end with Done for `reason`.
fn check_done(events: &mut UnboundedReceiver<worker::Event>, reason: worker::FinishReason) {
    let last = read_last(events);
    assert!(
        matches!(&last, worker::Event::Done { reason: ended } if *ended == reason),
        "{last:?}"
    );
}

/// Runs a decision of `passes` and returns its logits.
fn decide(passes: Vec<d1::Pass>) -> Vec<Vec<f32>> {
    let (_guard, mut events) = submit(passes);
    match read_last(&mut events) {
        worker::Event::Decided { logits, .. } => logits,
        other => panic!("the decision ended with {other:?}"),
    }
}

#[test]
fn decisions_run_their_passes_and_return_their_logits() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
        return;
    };
    let settings = worker::ModelSettings {
        id: "d1".into(),
        options: engine::Options {
            model: checkpoint,
            ctx: CTX as i32,
            ..Default::default()
        },
        idle_unload: IDLE_UNLOAD_S,
        max_cached_states: 256,
    };
    assert!(worker::start(
        vec![settings],
        config::Budgets::default(),
        None,
        Some(keep_line),
        None,
        worker::ViewOptions::default()
    ));

    // A decision's logits are as close to FP32 as LiquidAI's BF16 run.
    let first = decide(vec![json_pass()]);
    assert_eq!(first.len(), 1, "one pass, one list of logits");
    let difference = first[0]
        .iter()
        .zip(REFERENCE)
        .map(|(logit, reference)| (logit - reference).abs())
        .fold(0.0, f32::max);
    println!("the logits differ from FP32 by {difference:.4}");
    assert!(difference <= BOUND, "{:?}", first[0]);
    // Each pass of a decision gives the same bits as alone.
    let both = decide(vec![json_pass(), json_pass()]);
    assert_eq!(both, [first[0].clone(), first[0].clone()]);

    // The model unloads when idle, and loads again for the next decision,
    // which gives the same bits.
    let unloaded = wait_for_line(0, "d1: unloaded after");
    assert_eq!(decide(vec![json_pass()]), first);
    let loaded = wait_for_line(unloaded, "d1: loaded");

    // A decision about an image reads the image's span in place of its place,
    // as close to FP32 as LiquidAI's BF16 run. After the model unloads, its
    // vision weights load again, and the same decision gives the same bits.
    let (card, tokens) = decide_card();
    assert_eq!(tokens, CARD_EXPANDED, "the image's span counts whole");
    let difference = card
        .iter()
        .zip(CARD_LOGITS)
        .map(|(logit, reference)| (logit - reference).abs())
        .fold(0.0, f32::max);
    println!("the image decision's logits differ from FP32 by {difference:.4}");
    assert!(difference <= CARD_BOUND, "{card:?}");
    let unloaded = wait_for_line(loaded, "d1: unloaded after");
    assert_eq!(decide_card(), (card, tokens));
    wait_for_line(unloaded, "d1: loaded");

    // A reply request, and a pass longer than the context, are refused.
    let reply = worker::Request {
        model_index: 0,
        prompt: TOKENS.to_vec(),
        media: Vec::new(),
        sampling: worker::Sampling::default(),
        max_tokens: 1,
        note: NOTE.into(),
    };
    let Ok((_guard, mut events)) = worker::submit(reply) else {
        panic!("the worker is busy");
    };
    let refused = read_last(&mut events);
    assert!(
        matches!(&refused, worker::Event::Refused(message) if message == "the model is a decision model: it generates no text"),
        "{refused:?}"
    );
    let (_guard, mut events) = submit(vec![long_pass(CTX + 1)]);
    let refused = read_last(&mut events);
    let expected = format!("a pass has {} tokens; the context holds {CTX}", CTX + 1);
    assert!(
        matches!(&refused, worker::Event::Refused(message) if *message == expected),
        "{refused:?}"
    );

    // A dropped guard cancels a running decision before its next pass, and a
    // waiting one before its first.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    let from = LINES.lock().unwrap().len();
    let (running, mut running_events) = submit_long();
    let running_id = running.id();
    wait_until_running(&runtime, running_id);
    let (waiting, mut waiting_events) = submit_long();
    let waiting_id = waiting.id();
    drop(waiting);
    drop(running);
    check_done(&mut running_events, worker::FinishReason::Cancelled);
    check_done(&mut waiting_events, worker::FinishReason::Cancelled);
    wait_for_line(from, &format!("req {running_id}: client gone: "));
    wait_for_line(
        from,
        &format!("req {waiting_id}: ended before its first pass (cancelled)"),
    );

    // The stop ends a running decision before its next pass, and a waiting
    // one before its first. No decision is queued after it.
    let from = LINES.lock().unwrap().len();
    let (running, mut running_events) = submit_long();
    let running_id = running.id();
    wait_until_running(&runtime, running_id);
    let (waiting, mut waiting_events) = submit_long();
    let waiting_id = waiting.id();
    worker::stop();
    check_done(&mut running_events, worker::FinishReason::ServerStop);
    check_done(&mut waiting_events, worker::FinishReason::ServerStop);
    wait_for_line(
        from,
        &format!("req {running_id}: stopped by the server's stop: "),
    );
    wait_for_line(
        from,
        &format!("req {waiting_id}: ended before its first pass (shutdown)"),
    );
    let refused = worker::submit_decision(worker::DecisionRequest {
        model_index: 0,
        passes: vec![json_pass()],
        media: Vec::new(),
        note: NOTE.into(),
    })
    .err();
    assert!(
        matches!(&refused, Some(worker::SubmitError::Busy(message)) if message == "the server is stopping"),
        "the worker queued a decision after the stop"
    );

    // The first decision is number 1. The log shows its note, its queueing
    // and its end.
    let lines = LINES.lock().unwrap();
    for wanted in [
        format!("req 1 {NOTE}"),
        "req 1: queued for d1, waiting=1, tokens=89, passes=1".into(),
        "req 1: decided: 1 of 1 passes, 89 of 89 tokens".into(),
    ] {
        assert!(
            lines.iter().any(|line| line.contains(&wanted)),
            "no log line with {wanted:?}"
        );
    }
}
