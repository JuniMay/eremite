//! The server's yield and resume control. A client asks the server to yield:
//! to free memory down to lower limits and keep to them. Later the client asks
//! it to resume the configured limits. One operation owns a yield at a time.
//! The client names the operation with an operation_id.

use serde::{Deserialize, Serialize};

/// A client's request to yield. The field names are the request's JSON keys.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YieldRequest {
    /// The client's name for the operation. A retry sends the same name.
    pub operation_id: String,
    /// The models' memory limit in GiB, their states included. 0 asks for the
    /// least. None keeps the configured limit.
    pub memory_gib: Option<f64>,
    /// The state cache's limit in GiB. 0 asks for the least. None keeps the
    /// configured limit.
    pub state_cache_gib: Option<f64>,
}

/// A client's request to resume.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeRequest {
    /// The operation that owns the yield.
    pub operation_id: String,
}

/// Bytes of the models' memory and of the state cache: a pair of limits, or
/// the bytes in use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MemoryBytes {
    /// The models' memory, their states included.
    pub model_bytes: u64,
    /// The state cache's memory.
    pub state_cache_bytes: u64,
}

/// A yield's target for one kind of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YieldTarget {
    /// The server frees all the memory it can. The limit becomes what is left.
    Minimum,
    /// A limit of this many bytes.
    Bytes(u64),
}

/// A yield's targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YieldTargets {
    /// The target for the models' memory, their states included.
    pub model: YieldTarget,
    /// The target for the state cache.
    pub state_cache: YieldTarget,
}

/// Where the server is in a yield.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// No yield: the configured limits hold.
    Running,
    /// The worker carries out a yield, or starts it next.
    Yielding,
    /// The yield is done. The server keeps to the lower limits.
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

/// What the worker carries out next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// A yield to these targets. The server frees memory down to them and keeps
    /// to them.
    Yield(YieldTargets),
    /// A resume. The configured limits hold again.
    Resume,
}

#[cfg(test)]
mod tests {
    use super::*;

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
