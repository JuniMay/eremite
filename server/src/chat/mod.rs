//! The errors that end requests.
//! The server reports each error as an OpenAI error object in JSON.

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

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
