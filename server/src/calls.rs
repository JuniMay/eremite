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

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek V4.1 Flash's most rows per call.
    const MAX_ROWS_PER_CALL: usize = 6;

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
}
