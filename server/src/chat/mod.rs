//! The chat formats of the served models. A chat format turns a request's JSON
//! into prompt tokens and sampling settings. It also turns the worker's events
//! into the API's response. Each model family has its own chat format. A
//! family's format renders the family's chat template. A chat template is the
//! text form in which a model reads a conversation. The format also reads the
//! reasoning and tool calls in the model's output. This file holds what every
//! format shares: request checks, errors and response ids.

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
    /// OpenAI's Chat Completions API.
    ChatCompletions,
    /// OpenAI's Responses API.
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

/// A model family. Each family's requests take its own chat format.
pub use crate::engine::Architecture as Family;

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn a_bad_request_has_status_400_and_its_message() {
        let error = Error::new_bad_request("stop must be a string");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "stop must be a string");
    }

    #[test]
    fn the_error_json_is_an_openai_error_object() {
        let text = format_error_json("invalid_request_error", "no such model");
        let object: serde_json::Value = serde_json::from_str(&text).unwrap();
        let expected = json!({
            "error": {"message": "no such model", "type": "invalid_request_error", "code": null}
        });
        assert_eq!(object, expected);
    }

    #[tokio::test]
    async fn an_error_response_has_its_status_and_the_error_object() {
        let response = build_error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "no such model",
        );
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            body,
            format_error_json("invalid_request_error", "no such model")
        );
    }
}

#[cfg(test)]
mod check_tests {
    use super::*;
    use crate::worker::Sampling;
    use serde_json::{Map, Value};

    /// The defaults of a model that samples with temperature 1, top_p 1 and
    /// every token.
    const DEFAULTS: Sampling = Sampling {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: None,
    };

    /// Returns the JSON object of `request`.
    fn object(request: Value) -> Map<String, Value> {
        let Value::Object(root) = request else {
            panic!("a JSON object")
        };
        root
    }

    #[test]
    fn sampling_takes_the_request_s_values_and_the_defaults_for_the_rest() {
        let sampling = read_sampling(
            &object(json!({"temperature": 0.5, "top_k": 7, "seed": null})),
            DEFAULTS,
        )
        .unwrap();
        let expected = Sampling {
            temperature: 0.5,
            top_k: 7,
            ..DEFAULTS
        };
        assert_eq!(sampling, expected);
        for (key, value) in [
            ("temperature", json!(-1)),
            ("temperature", json!("hot")),
            ("top_p", json!(1.5)),
            ("top_k", json!(-1)),
            ("top_k", json!(u64::MAX)),
            ("seed", json!("7")),
        ] {
            let request = object(json!({key: value}));
            assert!(read_sampling(&request, DEFAULTS).is_err(), "{key} {value}");
        }
    }

    #[test]
    fn the_token_limit_takes_max_completion_tokens_first() {
        let limit = |request: Value| read_token_limit(&object(request), DEFAULT_MAX_TOKENS);
        assert_eq!(
            limit(json!({"max_tokens": 5, "max_completion_tokens": 9})).unwrap(),
            9
        );
        assert_eq!(limit(json!({"max_tokens": 5})).unwrap(), 5);
        assert_eq!(limit(json!({})).unwrap(), DEFAULT_MAX_TOKENS);
        for value in [
            json!(0),
            json!(-1),
            json!(1.5),
            json!("12"),
            json!(u64::MAX),
        ] {
            assert!(limit(json!({"max_tokens": value})).is_err(), "{value}");
        }
    }

    #[test]
    fn stop_sequences_are_one_string_or_an_array_of_strings() {
        let stop = |request: Value| read_stop_sequences(&object(request));
        assert!(stop(json!({})).unwrap().is_empty());
        assert_eq!(stop(json!({"stop": "end"})).unwrap(), ["end"]);
        assert_eq!(stop(json!({"stop": ["a", "b"]})).unwrap(), ["a", "b"]);
        assert!(stop(json!({"stop": ["end", 5]})).is_err());
        assert!(stop(json!({"stop": 5})).is_err());
    }

    #[test]
    fn a_data_url_gives_its_bytes() {
        assert_eq!(
            decode_data_url("data:image/png;base64,AAEC").unwrap(),
            [0, 1, 2]
        );
        assert!(decode_data_url("https://example.com/image.png").is_err());
        assert!(decode_data_url("data:image/png;base64,!!!").is_err());
    }

    #[test]
    fn a_random_id_has_its_prefix_and_letters_and_digits() {
        let id = generate_random_id("chatcmpl-");
        let random_chars = id.strip_prefix("chatcmpl-").unwrap();
        assert_eq!(random_chars.len(), RANDOM_ID_CHARS);
        assert!(
            random_chars
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
        assert_ne!(generate_random_id("chatcmpl-"), id);
    }
}
