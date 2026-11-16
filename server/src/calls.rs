//! How a model's decoding requests split into calls.
//!
//! The rows of one call share the GPU's work for each layer. So one call needs
//! the least GPU time per token. But while the call's rows wait for their
//! experts to be read, the GPU is idle. The GPU work of a second call can fill
//! that wait.
//!
//! The worker runs the decoding requests in one of two arrangements:
//! - Full calls: the requests fill as many full calls as they can. The other
//!   requests wait, and the request that waited longest goes first.
//! - Split calls: the requests split evenly over at least two calls, and every
//!   request runs.
//!
//! Which one is faster depends on the expert format, on the memory and on the
//! requests. So the chooser measures both. It measures the gaps between each
//! request's tokens. The mean gap is how long a request waits for each token.
//! The requests divided by the mean gap are the model's tokens per second.
//!
//! A round lasts until every decoding request has a new token. An arrangement
//! runs ROUNDS_PER_RUN rounds in a row. For each number of decoding requests,
//! the chooser first runs each arrangement once. Then it keeps the one whose
//! recent rounds had the shorter mean gap. Every EXPLORE_EVERY-th run tries the
//! other one again. So the choice follows changes in memory and load.
//!
//! An arrangement's figure is the gaps of its recent rounds. The figure starts
//! afresh each time the arrangement runs after the other one. So slow rounds
//! from long ago do not hold the arrangement back. For example, rounds are slow
//! while the expert cache fills.

use std::collections::{HashMap, VecDeque};

/// The rounds that an arrangement runs in a row. After a switch, a request's
/// first token only starts its next gap. So a run needs more than one round.
pub const ROUNDS_PER_RUN: usize = 3;
/// Every EXPLORE_EVERY-th run of a request count runs the arrangement that is
/// not the faster.
pub const EXPLORE_EVERY: usize = 8;
/// The most recent rounds that each arrangement's figure keeps.
pub const ROUNDS_KEPT: usize = 8;
/// Split calls are at least two. A second call's GPU work fills the first
/// call's waits for reads.
const MIN_SPLIT_CALLS: usize = 2;

/// Arranges `decoding_count` requests in calls of at most `max_rows_per_call`
/// rows. Returns the number of calls and the rows of each. Full calls, when
/// `is_split` is false: the requests fill as many calls as they can, and there
/// is at least one call. The other requests wait. Split calls, when `is_split`
/// is true: there are at least two calls, and as few as `max_rows_per_call`
/// allows. Every request gets a row. The rows are as even as the calls allow.
/// There are never more calls than requests.
pub fn arrange_calls(
    decoding_count: usize,
    max_rows_per_call: usize,
    is_split: bool,
) -> (usize, usize) {
    let max_rows_per_call = max_rows_per_call.max(1);
    if !is_split {
        return (
            (decoding_count / max_rows_per_call).max(1),
            max_rows_per_call,
        );
    }
    let calls = decoding_count
        .div_ceil(max_rows_per_call)
        .max(MIN_SPLIT_CALLS)
        .min(decoding_count)
        .max(1);
    (
        calls,
        decoding_count.div_ceil(calls).clamp(1, max_rows_per_call),
    )
}

/// Whether the two arrangements match: full calls run every one of
/// `decoding_count` requests, in as many calls as split calls.
fn arrangements_match(decoding_count: usize, max_rows_per_call: usize) -> bool {
    let (full_calls, rows) = arrange_calls(decoding_count, max_rows_per_call, false);
    let (split_calls, _) = arrange_calls(decoding_count, max_rows_per_call, true);
    full_calls == split_calls && full_calls * rows >= decoding_count
}

/// The gaps between tokens in the recent rounds of one arrangement, for one
/// request count.
#[derive(Clone, Debug, Default)]
struct TokenGaps {
    /// The last ROUNDS_KEPT rounds' gaps: the number of gaps and their seconds
    /// in all, oldest first.
    rounds: VecDeque<(usize, f64)>,
    /// The rounds recorded in all.
    recorded: usize,
}

impl TokenGaps {
    /// Records a round: `gaps` gaps that took `gap_seconds` seconds in all.
    /// Drops the oldest round when ROUNDS_KEPT rounds are kept.
    fn record(&mut self, gaps: usize, gap_seconds: f64) {
        if self.rounds.len() == ROUNDS_KEPT {
            self.rounds.pop_front();
        }
        self.rounds.push_back((gaps, gap_seconds));
        self.recorded += 1;
    }

    /// Forgets the kept rounds. The figure starts again from the next recorded
    /// round.
    fn forget_rounds(&mut self) {
        self.rounds.clear();
    }

    /// Returns the mean gap between two tokens of a request, in seconds, if the
    /// rounds hold a gap.
    fn mean_gap(&self) -> Option<f64> {
        let gaps: usize = self.rounds.iter().map(|&(gaps, _)| gaps).sum();
        let gap_seconds: f64 = self.rounds.iter().map(|&(_, seconds)| seconds).sum();
        (gaps > 0).then(|| gap_seconds / gaps as f64)
    }
}

/// The figures of one request count.
#[derive(Clone, Debug, Default)]
struct CountFigures {
    /// The token gaps of full calls, then of split calls.
    gaps: [TokenGaps; 2],
    /// Whether the current run's calls split.
    is_split: bool,
}

/// Chooses between full calls and split calls for one model's decoding
/// requests.
#[derive(Debug, Default)]
pub struct CallChooser {
    /// The figures of each request count.
    figures: HashMap<usize, CountFigures>,
    /// Whether the current round's requests split evenly over at least two
    /// calls.
    is_split: bool,
}

impl CallChooser {
    /// Returns a new chooser, with no rounds recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begins a round of `decoding_count` requests, with calls of at most
    /// `max_rows_per_call` rows. Chooses the round's arrangement and returns
    /// whether it splits. The arrangement stays until the next round. A run's
    /// first round chooses it for the whole run. A run that switches
    /// arrangements forgets the kept rounds of the arrangement it starts.
    /// Arrangements that give the same calls are not measured.
    pub fn begin_round(&mut self, decoding_count: usize, max_rows_per_call: usize) -> bool {
        if arrangements_match(decoding_count, max_rows_per_call) {
            self.is_split = false;
            return false;
        }
        let figures = self.figures.entry(decoding_count).or_default();
        let [full, split] = &figures.gaps;
        let recorded = full.recorded + split.recorded;
        if recorded.is_multiple_of(ROUNDS_PER_RUN) {
            let is_split = match (full.mean_gap(), split.mean_gap()) {
                // Each arrangement runs once before the chooser compares them.
                (None, _) => false,
                (_, None) => true,
                (Some(full_gap), Some(split_gap)) => {
                    let run = recorded / ROUNDS_PER_RUN;
                    // Every EXPLORE_EVERY-th run runs the slower arrangement.
                    (split_gap < full_gap) != run.is_multiple_of(EXPLORE_EVERY)
                }
            };
            if is_split != figures.is_split {
                figures.gaps[is_split as usize].forget_rounds();
            }
            figures.is_split = is_split;
        }
        self.is_split = figures.is_split;
        self.is_split
    }

    /// Records a steady round of `decoding_count` requests, in split calls when
    /// `is_split` is true. `gaps` gaps between two tokens of a request ended in
    /// the round. They took `gap_seconds` seconds in all. A gap counts only
    /// when both of its tokens came in steady rounds of this arrangement: no
    /// prompt ran and no other model's calls ran during them.
    pub fn record_round(
        &mut self,
        decoding_count: usize,
        is_split: bool,
        gaps: usize,
        gap_seconds: f64,
    ) {
        if let Some(figures) = self.figures.get_mut(&decoding_count) {
            figures.gaps[is_split as usize].record(gaps, gap_seconds);
        }
    }

    /// Chooses how many calls `decoding_count` requests run in at once, and the
    /// rows of each, in the current round's arrangement.
    pub fn choose_calls(&self, decoding_count: usize, max_rows_per_call: usize) -> (usize, usize) {
        arrange_calls(decoding_count, max_rows_per_call, self.is_split)
    }

    /// Returns whether the current round's requests split evenly over at least
    /// two calls.
    pub fn is_split(&self) -> bool {
        self.is_split
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek V4.1 Flash's most rows per call.
    const MAX_ROWS_PER_CALL: usize = 6;
    /// The rounds from one run that tries the slower arrangement to the next.
    const CYCLE: usize = EXPLORE_EVERY * ROUNDS_PER_RUN;

    /// Records a round of `decoding_count` requests that gave `tokens` tokens
    /// in `seconds`. Each request's gaps cover the round. So the round's gaps
    /// take `decoding_count` times its seconds.
    fn record(
        chooser: &mut CallChooser,
        decoding_count: usize,
        is_split: bool,
        seconds: f64,
        tokens: usize,
    ) {
        chooser.record_round(
            decoding_count,
            is_split,
            tokens,
            decoding_count as f64 * seconds,
        );
    }

    /// Runs `rounds` rounds of `decoding_count` requests. Each round gives
    /// every request one token. A round takes `full_seconds` with full calls
    /// and `split_seconds` with split calls. Returns how many rounds split.
    fn run_rounds(
        chooser: &mut CallChooser,
        rounds: usize,
        decoding_count: usize,
        full_seconds: f64,
        split_seconds: f64,
    ) -> usize {
        let mut split_rounds = 0;
        for _ in 0..rounds {
            let is_split = chooser.begin_round(decoding_count, MAX_ROWS_PER_CALL);
            split_rounds += is_split as usize;
            let seconds = if is_split {
                split_seconds
            } else {
                full_seconds
            };
            record(chooser, decoding_count, is_split, seconds, decoding_count);
        }
        split_rounds
    }

    #[test]
    fn full_calls_leave_the_rest_waiting_and_split_calls_take_every_request() {
        // Full calls of 6 rows. The other requests wait.
        assert_eq!(arrange_calls(1, MAX_ROWS_PER_CALL, false), (1, 6));
        assert_eq!(arrange_calls(4, MAX_ROWS_PER_CALL, false), (1, 6));
        assert_eq!(
            arrange_calls(7, MAX_ROWS_PER_CALL, false),
            (1, 6),
            "one 6-row call, the seventh waits"
        );
        assert_eq!(arrange_calls(12, MAX_ROWS_PER_CALL, false), (2, 6));
        assert_eq!(arrange_calls(13, MAX_ROWS_PER_CALL, false), (2, 6));
        // Split calls: every request gets a row. The rows are as even as the
        // calls allow.
        assert_eq!(arrange_calls(1, MAX_ROWS_PER_CALL, true), (1, 1));
        assert_eq!(arrange_calls(2, MAX_ROWS_PER_CALL, true), (2, 1));
        assert_eq!(arrange_calls(3, MAX_ROWS_PER_CALL, true), (2, 2));
        assert_eq!(arrange_calls(6, MAX_ROWS_PER_CALL, true), (2, 3));
        assert_eq!(arrange_calls(7, MAX_ROWS_PER_CALL, true), (2, 4), "4 + 3");
        assert_eq!(arrange_calls(12, MAX_ROWS_PER_CALL, true), (2, 6));
        assert_eq!(
            arrange_calls(13, MAX_ROWS_PER_CALL, true),
            (3, 5),
            "5 + 5 + 3"
        );
        assert_eq!(arrange_calls(4, 1, true), (4, 1), "one row per call");
    }

    #[test]
    fn each_arrangement_runs_once_then_the_faster_one_runs() {
        let mut chooser = CallChooser::new();
        // A run of full calls, then a run of split calls.
        for round in 0..2 * ROUNDS_PER_RUN {
            let is_split = chooser.begin_round(4, MAX_ROWS_PER_CALL);
            assert_eq!(is_split, round >= ROUNDS_PER_RUN, "round {round}");
            assert_eq!(chooser.is_split(), is_split, "round {round}");
            record(
                &mut chooser,
                4,
                is_split,
                if is_split { 0.05 } else { 0.08 },
                4,
            );
        }
        // Split calls are faster: they run, and every EXPLORE_EVERY-th run
        // tries full calls.
        let rounds = 4 * CYCLE;
        let split_rounds = run_rounds(&mut chooser, rounds, 4, 0.08, 0.05);
        assert_eq!(split_rounds, rounds - rounds / EXPLORE_EVERY);
        assert_eq!(chooser.choose_calls(4, MAX_ROWS_PER_CALL), (2, 2));
    }

    #[test]
    fn a_run_keeps_its_arrangement() {
        let mut chooser = CallChooser::new();
        run_rounds(&mut chooser, 2 * ROUNDS_PER_RUN, 4, 0.10, 0.05);
        // Split calls run. Full calls become faster during a run of split
        // calls. But the run keeps split calls to its end.
        for round in 0..ROUNDS_PER_RUN {
            assert!(chooser.begin_round(4, MAX_ROWS_PER_CALL), "round {round}");
            record(&mut chooser, 4, true, 1.0, 4);
        }
        assert!(!chooser.begin_round(4, MAX_ROWS_PER_CALL));
    }

    #[test]
    fn full_calls_win_when_their_rounds_give_more_tokens() {
        let mut chooser = CallChooser::new();
        // 7 requests and calls of 6 rows. A round of full calls takes two
        // calls: 12 tokens in 0.32 s. A round of split calls gives 7 tokens in
        // 0.30 s. Full calls give more tokens per second.
        let rounds = 8 * CYCLE;
        let mut split_rounds = 0;
        for _ in 0..rounds {
            let is_split = chooser.begin_round(7, MAX_ROWS_PER_CALL);
            split_rounds += is_split as usize;
            if is_split {
                record(&mut chooser, 7, true, 0.30, 7);
            } else {
                record(&mut chooser, 7, false, 0.32, 12);
            }
        }
        assert!(
            split_rounds < rounds / 4,
            "{split_rounds} of {rounds} rounds split"
        );
    }

    #[test]
    fn the_choice_follows_when_the_other_arrangement_becomes_faster() {
        let mut chooser = CallChooser::new();
        // Split calls are faster at first.
        assert!(run_rounds(&mut chooser, 4 * CYCLE, 7, 0.10, 0.06) > 3 * CYCLE);
        // Then they slow down, as when the SSD saturates. A few slow rounds
        // bring full calls back. Then only the runs that try split calls split.
        run_rounds(&mut chooser, CYCLE, 7, 0.10, 0.20);
        assert_eq!(
            run_rounds(&mut chooser, 4 * CYCLE, 7, 0.10, 0.20),
            4 * CYCLE / EXPLORE_EVERY
        );
        // Then full calls slow down more, and split calls come back.
        run_rounds(&mut chooser, CYCLE, 7, 0.30, 0.20);
        assert_eq!(
            run_rounds(&mut chooser, 4 * CYCLE, 7, 0.30, 0.20),
            4 * CYCLE - 4 * CYCLE / EXPLORE_EVERY
        );
    }

    #[test]
    fn a_slow_run_is_forgotten_when_its_arrangement_runs_again() {
        let mut chooser = CallChooser::new();
        // 9 requests. The first run of split calls is slow, as while the expert
        // cache fills.
        run_rounds(&mut chooser, 2 * ROUNDS_PER_RUN, 9, 0.25, 0.40);
        // Then split calls take 0.18 s a round and full calls 0.25 s. Full
        // calls run until the run that tries split calls again.
        let to_next_try = CYCLE - 2 * ROUNDS_PER_RUN;
        assert_eq!(run_rounds(&mut chooser, to_next_try, 9, 0.25, 0.18), 0);
        // That run measures split calls faster. They run from then on, except
        // in the next run that tries full calls.
        let split_rounds = run_rounds(&mut chooser, CYCLE + ROUNDS_PER_RUN, 9, 0.25, 0.18);
        assert_eq!(split_rounds, CYCLE);
    }

    #[test]
    fn each_request_count_has_its_own_figures() {
        let mut chooser = CallChooser::new();
        let mut split_rounds = [0, 0];
        // Rounds of 4 and of 7 requests alternate. Split calls are faster for
        // 4, full calls for 7.
        for round in 0..4 * CYCLE {
            let (count, full_seconds, split_seconds) = if round % 2 == 0 {
                (4, 0.08, 0.05)
            } else {
                (7, 0.10, 0.20)
            };
            let is_split = chooser.begin_round(count, MAX_ROWS_PER_CALL);
            split_rounds[round % 2] += is_split as usize;
            let seconds = if is_split {
                split_seconds
            } else {
                full_seconds
            };
            record(&mut chooser, count, is_split, seconds, count);
        }
        let each = 2 * CYCLE;
        assert!(split_rounds[0] > each * 3 / 4, "4 requests mostly split");
        assert!(
            split_rounds[1] < each / 4,
            "7 requests mostly run full calls"
        );
    }

    #[test]
    fn counts_whose_arrangements_give_the_same_calls_are_not_measured() {
        let mut chooser = CallChooser::new();
        // 12 requests run as two calls of 6 either way, and one request as one
        // call.
        assert_eq!(run_rounds(&mut chooser, 3 * CYCLE, 12, 0.10, 0.05), 0);
        assert_eq!(run_rounds(&mut chooser, 3 * CYCLE, 1, 0.10, 0.05), 0);
        assert!(chooser.figures.is_empty());
    }
}
