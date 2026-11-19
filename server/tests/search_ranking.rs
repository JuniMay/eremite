//! A decision model ranks a query's search results for the web tool: one
//! decision for each result, with the query and the result's title and snippet
//! as its state and Liquid AI's question for reranking. The results that it
//! finds likeliest to be relevant show, best first. When its decision takes
//! longer than the time limit, is refused, or cannot be queued, the first
//! results show in their given order, and the log says why. Set
//! EREMITE_D1_CHECKPOINT to d1-2B's directory; without it, the test is
//! skipped.
use std::ffi::{CStr, c_char};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eremite_server::codex::search::{Passage, Ranker};
use eremite_server::decision::d1;
use eremite_server::{config, engine, worker};
use tokio::runtime::Runtime;

/// The tokens in the test model's context.
const CTX: i32 = 4096;
/// The results that show.
const SHOWN: usize = 3;
/// A query, and its results in a mixed order: whether each one is relevant,
/// its title and its snippet.
const QUERY: &str = "git undo last commit but keep changes";
const RESULTS: [(bool, &str, &str); 8] = [
    (
        false,
        "Undo and redo in Microsoft Word",
        "Press Ctrl+Z to undo your last action and Ctrl+Y to redo it.",
    ),
    (
        true,
        "How do I undo the most recent local commits in Git? - Stack Overflow",
        "Use git reset --soft HEAD~1 to undo the commit and keep your changes staged, or git \
         reset HEAD~1 to keep them unstaged in the working tree.",
    ),
    (
        false,
        "How to commit to a fitness routine",
        "Set small goals, schedule your workouts and track your progress.",
    ),
    (
        false,
        "Git (slang) - meaning",
        "A mild insult in British English meaning a foolish or contemptible person.",
    ),
    (
        true,
        "Git - git-reset Documentation",
        "--soft: Does not touch the index file or the working tree at all, but resets the head \
         to <commit>. This leaves all your changed files as changes to be committed.",
    ),
    (
        false,
        "Last Commit Coffee Roasters - Menu",
        "Single-origin espresso, pour-over and pastries in downtown Portland.",
    ),
    (
        true,
        "How to undo the last commit without losing changes",
        "Run git reset HEAD~ to move the branch back one commit; your edits stay in the working \
         directory so you can commit them again.",
    ),
    (
        false,
        "Factory reset an Android phone",
        "Back up your data, then open Settings, System, Reset options.",
    ),
];
/// The words of a snippet longer than the context.
const LONG_SNIPPET_WORDS: usize = 2 * CTX as usize;

/// The log lines that the log hook received.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

extern "C" fn keep_line(line: *const c_char) {
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    LINES.lock().unwrap().push(line);
}

/// Returns RESULTS as passages.
fn list_passages() -> Vec<Passage> {
    RESULTS
        .iter()
        .map(|(_, title, snippet)| Passage {
            title: title.to_string(),
            snippet: snippet.to_string(),
        })
        .collect()
}

/// Ranks `passages` for QUERY within `timeout`, and returns the indices that
/// show and the log lines of the ranking.
fn rank(
    runtime: &Runtime,
    ranker: &Arc<Ranker>,
    passages: Vec<Passage>,
    timeout: Duration,
) -> (Vec<usize>, Vec<String>) {
    let from = LINES.lock().unwrap().len();
    let order = runtime.block_on(ranker.rank(QUERY, passages, SHOWN, timeout));
    (order, LINES.lock().unwrap()[from..].to_vec())
}

#[test]
fn a_decision_model_ranks_search_results_or_they_keep_their_order() {
    let Ok(checkpoint) = std::env::var("EREMITE_D1_CHECKPOINT") else {
        eprintln!("skipped: set EREMITE_D1_CHECKPOINT");
        return;
    };
    let settings = worker::ModelSettings {
        id: "d1".into(),
        options: engine::Options {
            model: checkpoint.clone(),
            ctx: CTX,
            ..Default::default()
        },
        idle_unload: config::Budgets::default().idle_unload_s,
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
    let tokenizer = worker::list_models()[0].1;
    let ranker = Arc::new(Ranker {
        id: "d1".into(),
        model_index: 0,
        checkpoint: d1::Checkpoint::read(&checkpoint, tokenizer).expect("the checkpoint reads"),
        tokenizer,
    });
    let runtime = Runtime::new().unwrap();
    let limit = Duration::from_secs_f64(config::Server::default().search_timeout_s);
    let in_order: Vec<usize> = (0..SHOWN).collect();

    // The relevant results show.
    let (order, _) = rank(&runtime, &ranker, list_passages(), limit);
    println!("shown: {order:?}");
    let mut shown = order.clone();
    shown.sort();
    let relevant: Vec<usize> = (0..RESULTS.len()).filter(|&i| RESULTS[i].0).collect();
    assert_eq!(shown, relevant, "the relevant results show");

    // A decision slower than the time limit leaves the given order.
    let (order, lines) = rank(&runtime, &ranker, list_passages(), Duration::ZERO);
    assert_eq!(order, in_order);
    assert!(
        lines
            .iter()
            .any(|line| line.contains("did not rank") && line.contains("search_timeout")),
        "{lines:?}"
    );

    // A refused decision leaves the given order: one passage is longer than the
    // context.
    let mut passages = list_passages();
    passages[RESULTS.len() - 1].snippet = "word ".repeat(LONG_SNIPPET_WORDS);
    let (order, lines) = rank(&runtime, &ranker, passages, limit);
    assert_eq!(order, in_order);
    assert!(
        lines
            .iter()
            .any(|line| line.contains("did not rank") && line.contains("the context holds")),
        "{lines:?}"
    );

    // A decision that cannot be queued leaves the given order: the worker has
    // stopped.
    worker::stop();
    let (order, lines) = rank(&runtime, &ranker, list_passages(), limit);
    assert_eq!(order, in_order);
    assert!(
        lines
            .iter()
            .any(|line| line.contains("did not rank") && line.contains("stopping")),
        "{lines:?}"
    );
}
