//! The chat formats of the served models.
//! A chat format turns a request's JSON into prompt tokens and sampling
//! settings.
//! It also turns the worker's events into the API's response.
//! Each model family has its own chat format.
//! A family's format renders the family's chat template.
//! A chat template is the text form in which a model reads a conversation.
//! The format also reads the reasoning and tool calls in the model's output.
//! deepseek.rs holds the DeepSeek V4.1 format.
//! That format uses the deepseek-recipe crate.
//! inkling2.rs holds the Inkling-2-Mini format.
//! This file holds what every format shares: request checks, errors and
//! response ids.

pub mod deepseek;
pub mod inkling2;

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// The most tokens a response may have when its request sets no limit.
/// A response also ends where the model's context ends.
/// The reason for this value is unknown.
pub const DEFAULT_MAX_TOKENS: i64 = 262_144;
/// The number of random letters and digits in an id.
/// The reason for this value is unknown.
const RANDOM_ID_CHARS: usize = 24;

/// An error that ends a request before its response starts.
#[derive(Debug)]
pub struct Error {
    /// The HTTP status of the error's response.
    pub status: StatusCode,
    /// The text that tells the client what went wrong.
    pub message: String,
}

impl Error {
    /// Creates a new error with status 400 Bad Request and the text `message`.
    pub fn new_bad_request(message: impl Into<String>) -> Error {
        Error {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

/// Decodes the bytes of a base64 data URL.
pub fn decode_data_url(url: &str) -> Result<Vec<u8>, Error> {
    let data = data_url::DataUrl::process(url)
        .map_err(|_| Error::new_bad_request("media require a base64 data URL"))?;
    data.decode_to_vec()
        .map(|(bytes, _)| bytes)
        .map_err(|_| Error::new_bad_request("media data is not valid base64"))
}

/// Reads a request's sampling settings from its JSON object `root`.
/// A value that the request sets replaces the model's default in `defaults`.
/// A missing or null value keeps the default.
/// Returns an error for a value of the wrong type or outside its range.
pub fn read_sampling(
    root: &serde_json::Map<String, serde_json::Value>,
    defaults: crate::worker::Sampling,
) -> Result<crate::worker::Sampling, Error> {
    use serde_json::Value;
    let read_finite_number = |key: &str, default: f64| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::new_bad_request(format!("{key} must be a finite number"))),
    };
    let read_nonnegative_integer = |key: &str, default: Option<u64>| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| Error::new_bad_request(format!("{key} must be a nonnegative integer"))),
    };
    let temperature = read_finite_number("temperature", defaults.temperature)?;
    let top_p = read_finite_number("top_p", defaults.top_p)?;
    // temperature is at least 0. top_p is a probability from 0 to 1.
    if temperature < 0.0 || !(0.0..=1.0).contains(&top_p) {
        return Err(Error::new_bad_request(
            "temperature must be nonnegative and top_p must be in [0, 1]",
        ));
    }
    let top_k = read_nonnegative_integer("top_k", Some(defaults.top_k as u64))?.unwrap_or(0);
    if top_k > i32::MAX as u64 {
        return Err(Error::new_bad_request(
            "top_k exceeds the engine's integer range",
        ));
    }
    Ok(crate::worker::Sampling {
        temperature,
        top_p,
        top_k: top_k as u32,
        seed: read_nonnegative_integer("seed", defaults.seed)?,
    })
}

/// Reads a Chat Completions request's token limit from its JSON object `root`.
/// It takes `max_completion_tokens` when the request sets it.
/// Otherwise it takes `max_tokens`.
/// Both must be positive integers when set.
/// Returns `default` when the request sets neither.
pub fn read_token_limit(
    root: &serde_json::Map<String, serde_json::Value>,
    default: i64,
) -> Result<i64, Error> {
    use serde_json::Value;
    let read_positive_integer = |key: &str| match root.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .filter(|&n| n > 0)
            .map(Some)
            .ok_or_else(|| Error::new_bad_request(format!("{key} must be a positive integer"))),
    };
    let completion = read_positive_integer("max_completion_tokens")?;
    Ok(completion
        .or(read_positive_integer("max_tokens")?)
        .unwrap_or(default))
}

/// Reads a request's stop sequences from its `stop` value.
/// The value may be missing, one string, or an array of strings.
/// Returns an error for any other value.
pub fn read_stop_sequences(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<String>, Error> {
    use serde_json::Value;
    match root.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(String::from)
                    .ok_or_else(|| Error::new_bad_request("stop must contain only strings"))
            })
            .collect(),
        _ => Err(Error::new_bad_request(
            "stop must be a string or an array of strings",
        )),
    }
}

/// Formats an OpenAI error object as JSON text, with `error_type` and
/// `message`.
pub fn format_error_json(error_type: &str, message: &str) -> String {
    json!({"error": {"message": message, "type": error_type, "code": null}}).to_string()
}

/// Builds an HTTP error response with `status` and an OpenAI error object as
/// its JSON body.
pub fn build_error_response(status: StatusCode, error_type: &str, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        format_error_json(error_type, message),
    )
        .into_response()
}

/// The OpenAI API that a request came through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// The Chat Completions API, at /v1/chat/completions and /chat/completions.
    ChatCompletions,
    /// The Responses API, at /v1/responses and /responses.
    Responses,
}

/// Generates an id from `prefix` and 24 random ASCII letters and digits.
pub fn generate_random_id(prefix: &str) -> String {
    use rand::Rng;
    let random_chars: String = rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(RANDOM_ID_CHARS)
        .map(char::from)
        .collect();
    format!("{prefix}{random_chars}")
}

pub use crate::engine::Architecture as Family;

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::{engine::Tokenizer, worker::Sampling};

    #[test]
    fn the_worker_request_takes_the_requested_top_k_and_the_default_seed() {
        let Ok(dir) = std::env::var("EREMITE_CHECKPOINT") else {
            return;
        };
        let tokenizer = Tokenizer::open(&dir, Some(deepseek::IMAGE_TOKEN));
        let defaults = deepseek::Defaults {
            model_id: "deepseek".into(),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 20,
                seed: Some(73),
            },
            is_thinking_on: false,
            effort: deepseek::EFFORT_HIGH,
        };
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "top_k": 31});
        let prepared = deepseek::prepare_request(
            &serde_json::to_vec(&body).unwrap(),
            &defaults,
            tokenizer,
            Api::ChatCompletions,
        )
        .unwrap();
        let request = prepared.build_worker_request(0, "test".into()).unwrap();
        assert_eq!(request.sampling.top_k, 31);
        assert_eq!(request.sampling.seed, Some(73));
    }

    #[test]
    fn prepare_request_rejects_malformed_token_limits_and_stop_sequences() {
        let Ok(dir) = std::env::var("EREMITE_CHECKPOINT") else {
            return;
        };
        let tokenizer = Tokenizer::open(&dir, Some(deepseek::IMAGE_TOKEN));
        let defaults = deepseek::Defaults {
            model_id: "deepseek".into(),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
            },
            is_thinking_on: false,
            effort: deepseek::EFFORT_HIGH,
        };
        let mut accepted = Vec::new();
        for (key, value) in [
            ("max_tokens", json!(0)),
            ("max_tokens", json!(-1)),
            ("max_tokens", json!(1.5)),
            ("max_tokens", json!("12")),
            ("max_completion_tokens", json!(u64::MAX)),
            ("stop", json!(["end", 5])),
        ] {
            let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
            body[key] = value;
            let bytes = serde_json::to_vec(&body).unwrap();
            if deepseek::prepare_request(&bytes, &defaults, tokenizer, Api::ChatCompletions).is_ok()
            {
                accepted.push(format!("DeepSeek accepted {body}"));
            }
        }
        assert!(accepted.is_empty(), "{}", accepted.join("; "));
    }
}
