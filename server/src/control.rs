//! The server's yield and resume control. A client asks the server to yield:
//! to free memory down to a lower limit and keep to it. Later the client asks
//! it to resume the configured limit. One operation owns a yield at a time.
//! The client names the operation with an operation_id.
//!
//! This module records each operation and its outcome. The worker carries out
//! each yield and resume between engine calls.

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

/// The bytes in one GiB.
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// The most bytes that an operation_id may have. Why 128 is unknown.
const MAX_OPERATION_ID_BYTES: usize = 128;

/// A client's request to yield. The field names are the request's JSON keys.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YieldRequest {
    /// The client's name for the operation. A retry sends the same name.
    pub operation_id: String,
    /// The models' memory limit in GiB, their states included. 0 asks for the
    /// least. None keeps the configured limit.
    pub memory_gib: Option<f64>,
}

/// A client's request to resume.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeRequest {
    /// The operation that owns the yield.
    pub operation_id: String,
}

/// The bytes that the models hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MemoryBytes {
    /// The models' memory, their states included.
    pub model_bytes: u64,
    /// The models' states, part of model_bytes.
    pub state_bytes: u64,
}

/// A yield's target for one kind of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YieldTarget {
    /// The server frees all the memory it can. The limit becomes what is left.
    Minimum,
    /// A limit of this many bytes.
    Bytes(u64),
}

/// Where the server is in a yield.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// No yield: the configured limit holds.
    Running,
    /// The worker carries out a yield, or starts it next.
    Yielding,
    /// The yield is done. The server keeps to the lower limit.
    Yielded,
    /// The worker carries out a resume, or starts it next.
    Resuming,
}

impl Phase {
    /// Returns the phase's name, as the status shows it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Yielding => "yielding",
            Self::Yielded => "yielded",
            Self::Resuming => "resuming",
        }
    }
}

/// A control request's error.
#[derive(Clone, Debug)]
pub struct Error {
    /// The HTTP status code.
    pub http_status: u16,
    /// The error's code in the response.
    pub code: &'static str,
    /// The error's message in the response.
    pub message: String,
}

impl Error {
    /// Makes a new error for an invalid request, with HTTP status 400.
    fn new_invalid_request(message: &str) -> Self {
        Self {
            http_status: StatusCode::BAD_REQUEST.as_u16(),
            code: "invalid_control_request",
            message: message.into(),
        }
    }

    /// Makes a new error for a conflict, with HTTP status 409 and `code`.
    fn new_conflict(code: &'static str, message: &str) -> Self {
        Self {
            http_status: StatusCode::CONFLICT.as_u16(),
            code,
            message: message.into(),
        }
    }
}

/// Checks `operation_id`. Returns an error when it is empty, only white space,
/// longer than MAX_OPERATION_ID_BYTES, or holds a control character.
fn check_operation_id(operation_id: &str) -> Result<(), Error> {
    if operation_id.is_empty()
        || operation_id.len() > MAX_OPERATION_ID_BYTES
        || operation_id.trim().is_empty()
        || operation_id.chars().any(char::is_control)
    {
        return Err(Error::new_invalid_request(&format!(
            "operation_id must contain 1 to {MAX_OPERATION_ID_BYTES} bytes without control characters"
        )));
    }
    Ok(())
}

/// Converts `requested_gib` to a target. None keeps `configured_bytes`. 0 asks
/// for the minimum. Returns an error for a value that is not finite, below 0,
/// or above `configured_bytes`.
fn convert_gib_to_target(
    requested_gib: Option<f64>,
    configured_bytes: u64,
) -> Result<YieldTarget, Error> {
    let Some(gib) = requested_gib else {
        return Ok(YieldTarget::Bytes(configured_bytes));
    };
    if !gib.is_finite() || gib < 0.0 || gib > configured_bytes as f64 / GIB {
        return Err(Error::new_invalid_request(
            "memory_gib must be finite, nonnegative and no larger than the configured limit",
        ));
    }
    Ok(if gib == 0.0 {
        YieldTarget::Minimum
    } else {
        YieldTarget::Bytes((gib * GIB) as u64)
    })
}

/// Where an operation stands.
enum Outcome {
    /// The operation owns the yield. It has not resumed or failed.
    Ongoing,
    /// The operation resumed.
    Resumed,
    /// The yield failed. The string says why.
    Failed(String),
}

/// One operation, as the control records it.
struct Operation {
    /// The limit that the yield asked for.
    target: YieldTarget,
    /// Where the operation stands.
    outcome: Outcome,
}

/// What the worker carries out next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// A yield to this target for the models' memory. The server frees memory
    /// down to it and keeps to it.
    Yield(YieldTarget),
    /// A resume. The configured limit holds again.
    Resume,
}

/// The yield and resume control of the server.
pub struct Control {
    /// Where the server is in a yield.
    pub phase: Phase,
    /// The models' memory limit from the configuration, in bytes.
    pub configured_limit: u64,
    /// The models' memory limit in effect. A yield lowers it.
    pub effective_limit: u64,
    /// The memory the models hold, as the worker last read it.
    pub retained: MemoryBytes,
    /// The operation that owns the yield, if any.
    owner: Option<String>,
    /// Every operation the control has seen, by operation_id.
    operations: HashMap<String, Operation>,
    /// Whether the worker carries out an action now.
    is_action_running: bool,
    /// The last failed yield's error, as the status shows it. A new yield
    /// clears it.
    error: Option<Value>,
}

impl Default for Control {
    /// Returns a control whose configured limit is 0 bytes.
    fn default() -> Self {
        Self::new(0)
    }
}

impl Control {
    /// Returns a new control with the models' memory limit `configured_limit`
    /// in bytes, and no yield.
    pub fn new(configured_limit: u64) -> Self {
        Self {
            phase: Phase::Running,
            configured_limit,
            effective_limit: configured_limit,
            retained: MemoryBytes::default(),
            owner: None,
            operations: HashMap::new(),
            is_action_running: false,
            error: None,
        }
    }

    /// Builds the control's status as JSON.
    pub fn build_status_json(&self) -> Value {
        json!({"state": self.phase, "operation_id": self.owner,
            "configured_limit_bytes": self.configured_limit,
            "effective_limit_bytes": self.effective_limit,
            "retained": self.retained, "error": self.error})
    }

    /// Starts a yield for `request`, and returns HTTP 202. A retry with the
    /// same operation_id and limit gets 202 while the yield goes on. It gets
    /// 200 once the yield is done. Returns an error for invalid input, for
    /// another limit under the same operation_id, for a yield that another
    /// operation owns, and for an operation that ended.
    pub fn start_yield(&mut self, request: YieldRequest) -> Result<u16, Error> {
        check_operation_id(&request.operation_id)?;
        let target = convert_gib_to_target(request.memory_gib, self.configured_limit)?;
        if let Some(operation) = self.operations.get(&request.operation_id) {
            if operation.target != target {
                return Err(Error::new_conflict(
                    "operation_conflict",
                    "this operation_id was used with a different limit",
                ));
            }
            return match &operation.outcome {
                Outcome::Ongoing => Ok(if self.phase == Phase::Yielded {
                    StatusCode::OK.as_u16()
                } else {
                    StatusCode::ACCEPTED.as_u16()
                }),
                Outcome::Resumed => Err(Error::new_conflict(
                    "operation_completed",
                    "this operation was already resumed; use a new operation_id",
                )),
                Outcome::Failed(message) => Err(Error::new_conflict("yield_failed", message)),
            };
        }
        if self.owner.is_some() {
            return Err(Error::new_conflict(
                "operation_conflict",
                "another yield operation owns the server",
            ));
        }
        self.operations.insert(
            request.operation_id.clone(),
            Operation {
                target,
                outcome: Outcome::Ongoing,
            },
        );
        self.owner = Some(request.operation_id);
        self.phase = Phase::Yielding;
        self.error = None;
        Ok(StatusCode::ACCEPTED.as_u16())
    }

    /// Starts the resume of `operation_id`. Returns true when the operation
    /// ended already. The call then changes nothing. A late retry thus cannot
    /// resume a later owner's yield. Returns an error for an operation_id that
    /// never yielded.
    pub fn start_resume(&mut self, operation_id: &str) -> Result<bool, Error> {
        check_operation_id(operation_id)?;
        match self
            .operations
            .get(operation_id)
            .map(|operation| &operation.outcome)
        {
            Some(Outcome::Resumed | Outcome::Failed(_)) => return Ok(true),
            Some(Outcome::Ongoing) => {}
            None => {
                return Err(Error::new_conflict(
                    "operation_conflict",
                    "operation_id does not own a yield",
                ));
            }
        }
        self.phase = Phase::Resuming;
        Ok(false)
    }

    /// Takes the next action for the worker, and marks it running. Returns None
    /// while an action runs, or with nothing to do.
    pub fn take_next_action(&mut self) -> Option<Action> {
        if self.is_action_running {
            return None;
        }
        let action = match self.phase {
            Phase::Yielding => Action::Yield(self.operations[self.owner.as_ref().unwrap()].target),
            Phase::Resuming => Action::Resume,
            Phase::Running | Phase::Yielded => return None,
        };
        self.is_action_running = true;
        Some(action)
    }

    /// Finishes the yield. `effective_limit` takes effect, and `retained` is
    /// the memory the models hold. A resume that arrived meanwhile runs next.
    pub fn finish_yield(&mut self, effective_limit: u64, retained: MemoryBytes) {
        self.effective_limit = effective_limit;
        self.retained = retained;
        self.is_action_running = false;
        if self.phase != Phase::Resuming {
            self.phase = Phase::Yielded;
        }
    }

    /// Fails the yield with `message`. The configured limit holds again.
    /// `retained` is the memory the models hold.
    pub fn fail_yield(&mut self, message: String, retained: MemoryBytes) {
        let operation_id = self.owner.take().unwrap();
        self.operations.get_mut(&operation_id).unwrap().outcome = Outcome::Failed(message.clone());
        self.error = Some(
            json!({"operation_id":operation_id, "code":"limits_unreachable", "message":message}),
        );
        self.phase = Phase::Running;
        self.effective_limit = self.configured_limit;
        self.retained = retained;
        self.is_action_running = false;
    }

    /// Finishes the resume. The configured limit holds again. `retained` is the
    /// memory the models hold.
    pub fn finish_resume(&mut self, retained: MemoryBytes) {
        let operation_id = self.owner.take().unwrap();
        self.operations.get_mut(&operation_id).unwrap().outcome = Outcome::Resumed;
        self.phase = Phase::Running;
        self.effective_limit = self.configured_limit;
        self.retained = retained;
        self.is_action_running = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes in one GiB.
    const GIB_BYTES: u64 = 1 << 30;

    /// Makes a control with a limit of 64 GiB for the models.
    fn make_control() -> Control {
        Control::new(64 * GIB_BYTES)
    }

    /// Makes a yield request named `operation_id`, for 4 GiB of model memory:
    /// below the configured 64 GiB.
    fn make_yield_request(operation_id: &str) -> YieldRequest {
        YieldRequest {
            operation_id: operation_id.into(),
            memory_gib: Some(4.0),
        }
    }

    #[test]
    fn a_yield_goes_on_until_the_worker_finishes_it() {
        let mut control = make_control();
        assert_eq!(
            control.start_yield(make_yield_request("a")).unwrap(),
            StatusCode::ACCEPTED.as_u16()
        );
        assert_eq!(control.phase, Phase::Yielding);
        assert_eq!(control.effective_limit, control.configured_limit);
        assert!(matches!(control.take_next_action(), Some(Action::Yield(_))));
        assert_eq!(control.take_next_action(), None);
        let retained = MemoryBytes {
            model_bytes: 3 * GIB_BYTES,
            state_bytes: GIB_BYTES,
        };
        control.finish_yield(4 * GIB_BYTES, retained);
        assert_eq!(control.effective_limit, 4 * GIB_BYTES);
        assert_eq!(control.phase, Phase::Yielded);
        assert_eq!(
            control.start_yield(make_yield_request("a")).unwrap(),
            StatusCode::OK.as_u16()
        );
        assert_eq!(control.take_next_action(), None);
    }

    #[test]
    fn conflicting_requests_leave_the_current_owner_unchanged() {
        let mut control = make_control();
        control.start_yield(make_yield_request("a")).unwrap();
        let before = control.build_status_json();
        assert!(control.start_yield(make_yield_request("b")).is_err());
        assert!(control.start_resume("b").is_err());
        let mut changed = make_yield_request("a");
        // 3 GiB differs from the first request's 4 GiB.
        changed.memory_gib = Some(3.0);
        assert!(control.start_yield(changed).is_err());
        assert_eq!(before, control.build_status_json());
    }

    #[test]
    fn a_resume_during_a_yield_waits_for_the_running_action() {
        let mut control = make_control();
        control.start_yield(make_yield_request("a")).unwrap();
        control.take_next_action();
        assert!(!control.start_resume("a").unwrap());
        assert_eq!(control.take_next_action(), None);
        control.finish_yield(0, MemoryBytes::default());
        assert_eq!(control.phase, Phase::Resuming);
        assert_eq!(control.take_next_action(), Some(Action::Resume));
        control.finish_resume(MemoryBytes::default());
        assert_eq!(control.phase, Phase::Running);
        assert_eq!(control.effective_limit, control.configured_limit);
    }

    #[test]
    fn a_retry_of_an_ended_operation_never_yields_again_or_resumes_a_new_owner() {
        let mut control = make_control();
        control.start_yield(make_yield_request("a")).unwrap();
        control.start_resume("a").unwrap();
        control.take_next_action();
        control.finish_resume(MemoryBytes::default());
        assert!(control.start_yield(make_yield_request("a")).is_err());
        control.start_yield(make_yield_request("b")).unwrap();
        let before = control.build_status_json();
        assert!(control.start_resume("a").unwrap());
        assert_eq!(before, control.build_status_json());
    }

    #[test]
    fn an_omitted_limit_keeps_the_configured_limit_and_zero_asks_for_the_minimum() {
        for (memory_gib, target) in [
            (None, YieldTarget::Bytes(64 * GIB_BYTES)),
            (Some(0.0), YieldTarget::Minimum),
        ] {
            let mut control = make_control();
            control
                .start_yield(YieldRequest {
                    operation_id: "a".into(),
                    memory_gib,
                })
                .unwrap();
            assert_eq!(control.take_next_action(), Some(Action::Yield(target)));
        }
    }

    #[test]
    fn a_failed_yield_shows_its_error_and_a_retry_does_not_yield_again() {
        let mut control = make_control();
        control.start_yield(make_yield_request("a")).unwrap();
        control.take_next_action();
        control.fail_yield("active states do not fit".into(), MemoryBytes::default());
        assert_eq!(control.phase, Phase::Running);
        assert_eq!(control.build_status_json()["error"]["operation_id"], "a");
        assert!(control.start_yield(make_yield_request("a")).is_err());
        assert!(control.start_resume("a").unwrap());
    }

    #[test]
    fn invalid_limits_and_operation_ids_are_rejected() {
        // -1 is below 0, and 65 is above the configured 64 GiB.
        for value in [-1.0, 65.0, f64::INFINITY, f64::NAN] {
            let mut control = make_control();
            let mut request = make_yield_request("a");
            request.memory_gib = Some(value);
            assert_eq!(
                control.start_yield(request).unwrap_err().http_status,
                StatusCode::BAD_REQUEST.as_u16()
            );
            assert_eq!(control.phase, Phase::Running);
        }
        for operation_id in ["", " \t", "bad\nline"] {
            assert!(
                make_control()
                    .start_yield(make_yield_request(operation_id))
                    .is_err()
            );
        }
    }

    #[test]
    fn a_default_control_runs_with_a_limit_of_zero_bytes() {
        let control = Control::default();
        assert_eq!(control.phase, Phase::Running);
        assert_eq!(control.configured_limit, 0);
    }

    #[test]
    fn each_phase_is_named_in_lower_case() {
        let phases = [
            Phase::Running,
            Phase::Yielding,
            Phase::Yielded,
            Phase::Resuming,
        ];
        assert_eq!(
            phases.map(Phase::as_str),
            ["running", "yielding", "yielded", "resuming"]
        );
    }
}
