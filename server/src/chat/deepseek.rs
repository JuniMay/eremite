//! The DeepSeek V4.1 chat format for the Chat Completions and Responses APIs.
//! The server first rewrites a request's JSON into forms that deepseek-recipe
//! accepts.
//! It reads the sampling settings, the token limit and the reasoning effort
//! itself.
//! deepseek-recipe then converts the request. It renders the V4.1 prompt.
//! The model's tokenizer turns the prompt into tokens.

use std::ops::Range;
use std::sync::OnceLock;

use axum::http::StatusCode;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest,
};
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use super::{Api, DEFAULT_MAX_TOKENS, Error};
use crate::engine;
use crate::worker::{self, Sampling};

/// The model id of a DeepSeek model whose configuration gives none.
/// Responses and GET /v1/models show this id.
pub const DEFAULT_MODEL_ID: &str = "deepseek-v4.1-flash";
/// The spelling of the image token in the DeepSeek V4.1 tokenizer.
/// Its id is 129264.
/// deepseek-recipe's default spelling, `<｜image｜>`, is not in this tokenizer.
pub const IMAGE_TOKEN: &str = "<｜deepseek_image｜>";
/// The model name that the server writes into a request for deepseek-recipe.
/// With this name, deepseek-recipe converts the request for V4.1.
/// It then renders a V4.1 prompt.
const RECIPE_MODEL_NAME: &str = "DeepSeek-V4.1-Flash";
/// The lowest effort. The V4.1 effort line gives the range 1 to 100.
const EFFORT_MIN: u32 = 1;
/// The effort that deepseek-recipe renders for the level low.
pub const EFFORT_LOW: u32 = 50;
/// The effort that deepseek-recipe renders for the level high.
/// deepseek-recipe also renders it when a request gives no level.
pub const EFFORT_HIGH: u32 = 75;
/// The effort that deepseek-recipe renders for the level max.
pub const EFFORT_MAX: u32 = 100;
/// The efforts that deepseek-recipe renders itself.
const RENDERED_EFFORTS: &[u32] = &[EFFORT_LOW, EFFORT_HIGH, EFFORT_MAX];
/// The start of the effort line that deepseek-recipe renders for the level
/// high.
const RENDERED_EFFORT_LINE: &str = "Reasoning Effort: 75 (range 1-100";
/// The text before the effort's number in that line.
const EFFORT_LINE_PREFIX: &str = "Reasoning Effort: ";
/// The tool types that deepseek-recipe keeps in a Responses request.
const KEPT_TOOL_TYPES: &[&str] = &["function", "namespace", "custom", "tool_search"];

/// The V4.1 prompt encoding with the server's special tokens.
/// It is made at its first use.
static ENCODING: OnceLock<DeepseekV41Encoding> = OnceLock::new();

/// A served DeepSeek model's id and the settings for values that a request does
/// not set.
#[derive(Clone)]
pub struct Defaults {
    /// The model id that responses name when a request names no model.
    pub model_id: String,
    /// The sampling settings for values that a request does not set.
    pub sampling: Sampling,
    /// Whether thinking is on when a request does not say.
    pub is_thinking_on: bool,
    /// The reasoning effort from 1 to 100 when a request gives no valid one.
    pub effort: u32,
}

/// Builds deepseek-recipe's special tokens with the V4.1 tokenizer's image
/// token.
fn build_special_tokens() -> SpecialTokensMap {
    SpecialTokensMap {
        image: IMAGE_TOKEN.to_string(),
        ..SpecialTokensMap::default()
    }
}

/// Gets the V4.1 prompt encoding with the server's special tokens.
/// The first call creates it.
fn get_encoding() -> &'static DeepseekV41Encoding {
    ENCODING.get_or_init(|| DeepseekV41Encoding::new().with_special_tokens(build_special_tokens()))
}

// ---- rewriting a request for deepseek-recipe ----

/// The settings that the server reads from a request itself.
/// deepseek-recipe converts the rest of the request.
struct RequestSettings {
    /// How the request samples its tokens.
    sampling: Sampling,
    /// The most tokens that the response may have.
    max_tokens: i64,
    /// The reasoning effort from 1 to 100, or None when thinking is off.
    effort: Option<u32>,
}

/// Parses the integer at the start of `text`.
/// Leading whitespace and one + or - sign may come before the digits.
/// Parsing stops at the first character that is not a digit.
/// Returns 0 when there are no digits.
fn parse_leading_integer(text: &str) -> i64 {
    let text = text.trim_start();
    let (sign, digits) = match text.as_bytes().first() {
        Some(b'-') => (-1, &text[1..]),
        Some(b'+') => (1, &text[1..]),
        _ => (1, text),
    };
    let number = digits
        .bytes()
        .take_while(u8::is_ascii_digit)
        .fold(0i64, |number, digit| {
            // Each digit moves the number one decimal place.
            number
                .saturating_mul(10)
                .saturating_add((digit - b'0') as i64)
        });
    sign * number
}

/// Reads a reasoning effort from 1 to 100 from a request's effort `value`.
/// A number gives the effort without its fraction.
/// A string is a level, or a number as text.
/// The levels minimal and low give 50.
/// The levels medium, high and xhigh give 75.
/// The levels ultra and max give 100.
/// deepseek-recipe renders these same efforts for these levels.
/// Returns `default` for a missing value, an unknown level or a number outside
/// 1 to 100.
fn read_effort(value: Option<&Value>, default: u32) -> u32 {
    // The effort 0 is outside 1 to 100. It gives the default.
    let effort = match value {
        Some(Value::Number(n)) => n.as_f64().map_or(0, |f| f.trunc() as i64),
        Some(Value::String(s)) => match s.as_str() {
            "minimal" | "low" => i64::from(EFFORT_LOW),
            "medium" | "high" | "xhigh" => i64::from(EFFORT_HIGH),
            "ultra" | "max" => i64::from(EFFORT_MAX),
            s => parse_leading_integer(s),
        },
        _ => 0,
    };
    if (i64::from(EFFORT_MIN)..=i64::from(EFFORT_MAX)).contains(&effort) {
        effort as u32
    } else {
        default
    }
}

/// Rewrites a Chat Completions request's JSON object `root` for
/// deepseek-recipe.
/// Removes the settings that the server reads itself. Returns those settings.
/// Writes the model name, thinking and the effort level in deepseek-recipe's
/// forms.
/// Turns developer messages into system messages.
/// Gives each tool call an id, the type "function" and its arguments as JSON
/// text.
/// Gives tool messages without a `tool_call_id` the ids of the calls before
/// them, in order.
/// Removes `stream_options` when the request does not stream.
fn rewrite_chat_request(
    root: &mut Map<String, Value>,
    defaults: &Defaults,
) -> Result<RequestSettings, Error> {
    let sampling = super::read_sampling(root, defaults.sampling)?;
    for key in ["temperature", "top_p", "top_k", "seed"] {
        root.remove(key);
    }
    let max_tokens = super::read_token_limit(root, DEFAULT_MAX_TOKENS)?;
    super::read_stop_sequences(root)?;
    root.remove("max_completion_tokens");
    root.remove("max_tokens");
    root.insert("model".into(), RECIPE_MODEL_NAME.into());

    // A `thinking` value decides whether thinking is on.
    // Without one, the effort "none" turns thinking off.
    // Any other effort turns it on.
    // Without either, the model's default decides.
    let thinking_field = root.remove("thinking");
    let effort_field = root.remove("reasoning_effort");
    let is_thinking_on = match &thinking_field {
        Some(Value::Bool(on)) => *on,
        Some(Value::Object(object)) => {
            object.get("type").and_then(Value::as_str) != Some("disabled")
        }
        Some(Value::Null) | None => match &effort_field {
            Some(Value::String(s)) if s == "none" => false,
            Some(Value::Null) | None => defaults.is_thinking_on,
            Some(_) => true,
        },
        Some(_) => true,
    };
    let effort = read_effort(effort_field.as_ref(), defaults.effort);
    root.insert(
        "thinking".into(),
        json!({"type": if is_thinking_on { "enabled" } else { "disabled" }}),
    );
    if is_thinking_on {
        // deepseek-recipe renders the efforts 50, 75 and 100 for the levels
        // low, high and max.
        // Another effort renders as high.
        // The server writes the effort's number into the prompt after
        // rendering.
        let level = match effort {
            EFFORT_LOW => "low",
            EFFORT_MAX => "max",
            _ => "high",
        };
        root.insert("reasoning_effort".into(), level.into());
    }
    if root.get("stream") != Some(&Value::Bool(true)) {
        root.remove("stream_options");
    }

    if let Some(Value::Array(messages)) = root.get_mut("messages") {
        // The ids of the tool calls in the last assistant message.
        let mut call_ids: Vec<String> = Vec::new();
        // The number of tool messages since that assistant message.
        let mut answered_calls = 0;
        for message in messages.iter_mut().filter_map(Value::as_object_mut) {
            match message.get("role").and_then(Value::as_str) {
                Some("developer") => {
                    message.insert("role".into(), "system".into());
                }
                Some("assistant") => {
                    call_ids.clear();
                    answered_calls = 0;
                    match message.get_mut("tool_calls") {
                        Some(Value::Array(tool_calls)) if !tool_calls.is_empty() => {
                            for (call_index, call) in tool_calls
                                .iter_mut()
                                .filter_map(Value::as_object_mut)
                                .enumerate()
                            {
                                if !call.get("id").is_some_and(Value::is_string) {
                                    call.insert("id".into(), format!("call_{call_index}").into());
                                }
                                if !call.contains_key("type") {
                                    call.insert("type".into(), "function".into());
                                }
                                if let Some(Value::Object(function)) = call.get_mut("function")
                                    && let Some(arguments) =
                                        function.get("arguments").filter(|a| !a.is_string())
                                {
                                    let text = arguments.to_string();
                                    function.insert("arguments".into(), text.into());
                                }
                                call_ids.push(call["id"].as_str().unwrap_or_default().to_string());
                            }
                        }
                        Some(Value::Array(_)) | Some(Value::Null) => {
                            message.remove("tool_calls");
                        }
                        _ => {}
                    }
                }
                Some("tool") => {
                    if !message.get("tool_call_id").is_some_and(Value::is_string)
                        && let Some(id) = call_ids.get(answered_calls)
                    {
                        message.insert("tool_call_id".into(), id.clone().into());
                    }
                    answered_calls += 1;
                }
                _ => {}
            }
        }
    }
    Ok(RequestSettings {
        sampling,
        max_tokens,
        effort: is_thinking_on.then_some(effort),
    })
}

/// Rewrites a Responses request's JSON object `root` for deepseek-recipe.
/// Reads the settings that the server handles itself: sampling,
/// `max_output_tokens` and the reasoning effort.
/// Writes the model name for deepseek-recipe.
/// When thinking is on, writes the effort back as the level that
/// deepseek-recipe renders.
/// Returns an error for a request with `previous_response_id` or
/// `conversation`.
/// The server stores no responses. It cannot continue one.
fn rewrite_responses_request(
    root: &mut Map<String, Value>,
    defaults: &Defaults,
) -> Result<RequestSettings, Error> {
    for key in ["previous_response_id", "conversation"] {
        if root.get(key).is_some_and(|v| !v.is_null()) {
            return Err(Error::new_bad_request(format!(
                "{key} is not supported: the server keeps no responses, so send the whole conversation in input"
            )));
        }
    }
    let sampling = super::read_sampling(root, defaults.sampling)?;
    let max_tokens = match root.get("max_output_tokens") {
        None | Some(Value::Null) => DEFAULT_MAX_TOKENS,
        Some(value) => value.as_i64().filter(|&n| n > 0).ok_or_else(|| {
            Error::new_bad_request("max_output_tokens must be a positive integer")
        })?,
    };
    root.insert("model".into(), RECIPE_MODEL_NAME.into());

    // The effort "none" turns thinking off.
    // Any other effort turns it on: a level, or a number from 1 to 100.
    // Without an effort, the model's default decides.
    let effort_field = root
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .filter(|e| !e.is_null())
        .cloned();
    let is_thinking_on = match &effort_field {
        None => defaults.is_thinking_on,
        Some(Value::String(s)) => s != "none",
        Some(_) => true,
    };
    let effort = read_effort(effort_field.as_ref(), defaults.effort);
    if is_thinking_on {
        let level = match effort {
            EFFORT_LOW => "low",
            EFFORT_MAX => "max",
            _ => "high",
        };
        match root.get_mut("reasoning") {
            Some(Value::Object(reasoning)) => {
                reasoning.insert("effort".into(), level.into());
            }
            _ => {
                root.insert("reasoning".into(), json!({"effort": level}));
            }
        }
    }
    Ok(RequestSettings {
        sampling,
        max_tokens,
        effort: is_thinking_on.then_some(effort),
    })
}

/// Counts the declared tools that deepseek-recipe drops from a Responses
/// request.
/// It drops the hosted web search tool. The server cannot run that tool.
/// It also drops tool types that it does not know.
fn count_dropped_tools(root: &Map<String, Value>) -> usize {
    root.get("tools")
        .and_then(Value::as_array)
        .map_or(0, |tools| {
            tools
                .iter()
                .filter(|t| {
                    !t.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|tool_type| KEPT_TOOL_TYPES.contains(&tool_type))
                })
                .count()
        })
}

// ---- conversion and rendering ----

/// A request that deepseek-recipe has converted and rendered.
struct RenderedRequest {
    /// The request as deepseek-recipe converted it.
    converted_request: ConversationRequest,
    /// The rendered V4.1 prompt.
    prompt: String,
    /// The byte ranges in `prompt` that hold special tokens.
    special_ranges: Vec<Range<usize>>,
    /// The request's images, in the order of their places in `prompt`.
    image_sources: Vec<ImageSource>,
    /// The settings that the server read from the request itself.
    settings: RequestSettings,
}

/// Turns a deepseek-recipe conversion error into a request error.
/// A bad request gets status 400. An internal failure gets status 500.
impl From<ConversionError> for Error {
    fn from(error: ConversionError) -> Error {
        let status = match error {
            ConversionError::BadRequest { .. } => StatusCode::BAD_REQUEST,
            ConversionError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Error {
            status,
            message: error.to_string(),
        }
    }
}

/// Writes `effort` into the effort line of a prompt that deepseek-recipe
/// rendered with 75.
/// The effort line follows the begin and System tokens at the prompt's start.
/// Shifts the special token ranges after the number by the change in length.
/// Leaves the prompt unchanged when it does not start with that line.
fn write_effort_into_prompt(prompt: &mut String, special_ranges: &mut [Range<usize>], effort: u32) {
    let tokens = build_special_tokens();
    let at = tokens.token(SpecialToken::Bos).len() + tokens.token(SpecialToken::System).len();
    if !prompt
        .get(at..)
        .is_some_and(|rest| rest.starts_with(RENDERED_EFFORT_LINE))
    {
        return;
    }
    let from = at + EFFORT_LINE_PREFIX.len();
    // The length of the rendered 75.
    let rendered_effort_length = EFFORT_HIGH.to_string().len();
    let number = effort.to_string();
    prompt.replace_range(from..from + rendered_effort_length, &number);
    let shift =
        |i: usize| (i as isize + number.len() as isize - rendered_effort_length as isize) as usize;
    for range in special_ranges
        .iter_mut()
        .filter(|r| r.start >= from + rendered_effort_length)
    {
        *range = shift(range.start)..shift(range.end);
    }
}

/// Renders a request's JSON object `root` as a V4.1 prompt.
/// It first rewrites the request for deepseek-recipe.
/// deepseek-recipe then converts the request. It renders the prompt.
fn render_request(
    root: Map<String, Value>,
    defaults: &Defaults,
    api: Api,
) -> Result<RenderedRequest, Error> {
    let mut root = root;
    match api {
        Api::ChatCompletions => {
            let settings = rewrite_chat_request(&mut root, defaults)?;
            render_as::<ChatCompletionRequest>(root, settings, defaults)
        }
        Api::Responses => {
            let settings = rewrite_responses_request(&mut root, defaults)?;
            render_as::<ResponsesRequest>(root, settings, defaults)
        }
    }
}

/// Renders the rewritten JSON object `root` of an `ApiRequest` request.
/// It parses the JSON as `ApiRequest`.
/// deepseek-recipe then converts the request. It renders the V4.1 prompt.
/// An effort other than 50, 75 or 100 is then written into the prompt.
fn render_as<ApiRequest: ProtocolRequest + DeserializeOwned>(
    root: Map<String, Value>,
    settings: RequestSettings,
    defaults: &Defaults,
) -> Result<RenderedRequest, Error> {
    let request: ApiRequest = serde_json::from_value(Value::Object(root))
        .map_err(|e| Error::new_bad_request(format!("invalid request: {e}")))?;
    let options = ConversionOptions::default()
        .with_special_tokens(build_special_tokens())
        .with_default_thinking_mode(defaults.is_thinking_on);
    let converted_request = request.convert(options)?;
    let rendered = get_encoding().render_conversation(&converted_request.conversation);
    let (mut prompt, mut special_ranges) = (rendered.prompt, rendered.special_ranges);
    if let Some(effort) = settings.effort.filter(|e| !RENDERED_EFFORTS.contains(e)) {
        write_effort_into_prompt(&mut prompt, &mut special_ranges, effort);
    }
    Ok(RenderedRequest {
        converted_request,
        prompt,
        special_ranges,
        image_sources: rendered.image_sources,
        settings,
    })
}

/// A request that is ready for the worker.
pub struct PreparedRequest {
    /// The model's tokenizer.
    tokenizer: engine::Tokenizer,
    /// The rendered prompt's bytes, with the byte 0xFF at each image's place.
    prompt: Vec<u8>,
    /// The file bytes of each image, in prompt order.
    images: Vec<Vec<u8>>,
    /// The settings that the server read from the request itself.
    settings: RequestSettings,
    /// The API that the request came through.
    pub api: Api,
    /// Whether the response streams as server-sent events.
    pub is_streamed: bool,
    /// The number of the request's messages, or of its input items for
    /// Responses.
    pub message_count: usize,
    /// The number of tools that the request declares.
    pub tool_count: usize,
    /// The number of declared tools that deepseek-recipe drops.
    pub dropped_tool_count: usize,
}

impl PreparedRequest {
    /// Builds the worker's request for the model at `model_index`, with `note`
    /// for the log.
    /// It tokenizes the prompt. Each 0xFF byte becomes the image token.
    /// Returns an error when the prompt is empty.
    /// Returns an error when the prompt's image tokens and the images differ in
    /// number.
    pub fn build_worker_request(
        &self,
        model_index: usize,
        note: String,
    ) -> Result<worker::Request, Error> {
        let prompt = self.tokenizer.tokenize_prompt(&self.prompt);
        let image_token = self.tokenizer.find_added_token(IMAGE_TOKEN);
        let image_token_count = prompt.iter().filter(|&&t| Some(t) == image_token).count();
        if prompt.is_empty() {
            return Err(Error::new_bad_request("the prompt is empty"));
        }
        if image_token_count != self.images.len() {
            return Err(Error::new_bad_request(format!(
                "the prompt has {image_token_count} image places and {} images",
                self.images.len()
            )));
        }
        let sampling = self.settings.sampling;
        Ok(worker::Request {
            model_index,
            prompt,
            media: self.images.clone(),
            sampling,
            max_tokens: self.settings.max_tokens,
            note,
        })
    }

    /// Describes the thinking setting for the log: "thinking effort N" or
    /// "thinking off".
    pub fn describe_thinking(&self) -> String {
        self.settings
            .effort
            .map_or("thinking off".into(), |e| format!("thinking effort {e}"))
    }
}

/// Prepares a request's `body` for the worker.
/// `defaults` holds the model's id and default settings.
/// `tokenizer` is the model's tokenizer.
/// It checks the JSON. It renders the prompt.
/// It also decodes each image from its data URL.
pub fn prepare_request(
    body: &[u8],
    defaults: &Defaults,
    tokenizer: engine::Tokenizer,
    api: Api,
) -> Result<PreparedRequest, Error> {
    let Ok(Value::Object(root)) = serde_json::from_slice::<Value>(body) else {
        return Err(Error::new_bad_request("the body is not a JSON object"));
    };
    let tool_count = root
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let (message_count, dropped_tool_count) = match api {
        Api::ChatCompletions => {
            let message_count = root
                .get("messages")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if message_count == 0 {
                return Err(Error::new_bad_request("messages must be a non-empty array"));
            }
            let last = root["messages"]
                .as_array()
                .and_then(|m| m.last())
                .and_then(|m| m.get("role"));
            if last.and_then(Value::as_str) == Some("assistant") {
                return Err(Error::new_bad_request(
                    "the last message must not be the assistant's",
                ));
            }
            (message_count, 0)
        }
        Api::Responses => {
            let items = match root.get("input") {
                // A string input is one item.
                Some(Value::String(s)) if !s.is_empty() => 1,
                Some(Value::Array(items)) if !items.is_empty() => items.len(),
                _ => {
                    return Err(Error::new_bad_request(
                        "input must be a non-empty string or array",
                    ));
                }
            };
            (items, count_dropped_tools(&root))
        }
    };
    let rendered = render_request(root, defaults, api)?;
    let mut images = Vec::new();
    for source in &rendered.image_sources {
        match source {
            ImageSource::DataUrl { data_url, .. } => {
                images.push(super::decode_data_url(data_url)?);
            }
            ImageSource::Bytes { data, .. } => images.push(data.clone()),
            ImageSource::Url { .. } => {
                return Err(Error::new_bad_request(
                    "only data: URLs with base64 data are read (the server fetches no URLs and reads no files)",
                ));
            }
        }
    }
    let mut prompt = Vec::with_capacity(rendered.prompt.len());
    let mut at = 0;
    for range in rendered
        .special_ranges
        .iter()
        .filter(|s| &rendered.prompt[(*s).clone()] == IMAGE_TOKEN)
    {
        prompt.extend_from_slice(&rendered.prompt.as_bytes()[at..range.start]);
        prompt.push(0xFF);
        at = range.end;
    }
    prompt.extend_from_slice(&rendered.prompt.as_bytes()[at..]);
    Ok(PreparedRequest {
        tokenizer,
        is_streamed: rendered.converted_request.stream,
        prompt,
        images,
        settings: rendered.settings,
        api,
        message_count,
        tool_count,
        dropped_tool_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the defaults that the tests use: thinking on at effort 75.
    fn build_test_defaults() -> Defaults {
        Defaults {
            model_id: DEFAULT_MODEL_ID.into(),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
            },
            is_thinking_on: true,
            effort: EFFORT_HIGH,
        }
    }

    /// Rewrites the Chat Completions request `request` with the tests'
    /// defaults.
    /// Returns the rewritten object and the settings that the server read.
    fn rewrite_chat_json(request: Value) -> (Map<String, Value>, RequestSettings) {
        let Value::Object(mut root) = request else {
            panic!("an object")
        };
        let s = rewrite_chat_request(&mut root, &build_test_defaults()).expect("compatible");
        (root, s)
    }

    /// Prepares the request `request` of `api` with the tests' defaults.
    fn prepare_test_request(
        request: &Value,
        api: Api,
        tokenizer: engine::Tokenizer,
    ) -> Result<PreparedRequest, Error> {
        let body = request.to_string();
        prepare_request(body.as_bytes(), &build_test_defaults(), tokenizer, api)
    }

    #[test]
    fn chat_requests_are_rewritten_for_deepseek_recipe() {
        let (root, s) = rewrite_chat_json(json!({
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                {"role": "tool", "content": "done"},
                {"role": "assistant", "content": "ok", "tool_calls": []},
                {"role": "user", "content": "more"},
            ],
            "thinking": true, "reasoning_effort": 62, "temperature": 0, "top_p": 0, "seed": 7,
        }));
        assert_eq!(
            (
                s.sampling.temperature,
                s.sampling.top_p,
                s.sampling.seed,
                s.effort
            ),
            (0.0, 0.0, Some(7), Some(62))
        );
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(root["model"], RECIPE_MODEL_NAME);
        assert_eq!(root["thinking"], json!({"type": "enabled"}));
        assert_eq!(root["reasoning_effort"], "high");
        assert!(
            !root.contains_key("temperature")
                && !root.contains_key("top_p")
                && !root.contains_key("seed")
        );
        let m = &root["messages"];
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[2]["tool_calls"][0]["id"], "call_0");
        assert_eq!(m[2]["tool_calls"][0]["type"], "function");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], r#"{"a":1}"#);
        assert_eq!(m[3]["tool_call_id"], "call_0");
        assert!(m[4].get("tool_calls").is_none());
    }

    #[test]
    fn effort_none_turns_thinking_off() {
        let (root, s) = rewrite_chat_json(
            json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}),
        );
        assert_eq!(s.effort, None);
        assert_eq!(root["thinking"], json!({"type": "disabled"}));
        let (_, s) =
            rewrite_chat_json(json!({"messages": [], "reasoning_effort": "max", "max_tokens": 5}));
        assert_eq!((s.effort, s.max_tokens), (Some(100), 5));
    }

    #[test]
    fn an_effort_is_a_level_a_number_or_a_number_in_text() {
        // The effort that a missing or invalid value gives.
        let default = 30;
        let read = |value: Value| read_effort(Some(&value), default);
        assert_eq!(read(json!("minimal")), EFFORT_LOW);
        assert_eq!(read(json!("medium")), EFFORT_HIGH);
        assert_eq!(read(json!("ultra")), EFFORT_MAX);
        assert_eq!(read(json!(62.9)), 62);
        assert_eq!(read(json!(" +42 steps")), 42);
        for invalid in [
            json!("-3"),
            json!("101"),
            json!("much"),
            json!(0),
            json!(true),
        ] {
            assert_eq!(read(invalid.clone()), default, "{invalid}");
        }
        assert_eq!(read_effort(None, default), default);
        assert_eq!(parse_leading_integer("99999999999999999999"), i64::MAX);
    }

    #[test]
    fn a_numeric_effort_is_written_into_the_prompt() {
        let request =
            json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": 62});
        let Value::Object(root) = request else {
            unreachable!()
        };
        let r =
            render_request(root, &build_test_defaults(), Api::ChatCompletions).expect("rendered");
        assert!(
            r.prompt.starts_with(
                "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 62 (range 1-100"
            ),
            "{}",
            r.prompt
        );
        for s in &r.special_ranges {
            assert!(
                r.prompt[s.clone()].starts_with('<'),
                "{:?} in {}",
                s,
                r.prompt
            );
        }
    }

    #[test]
    fn conversion_errors_give_status_400_or_500() {
        let error = Error::from(ConversionError::bad_request("no messages"));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "no messages");
        let error = Error::from(ConversionError::internal("no encoding"));
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// Prepares requests with images for the worker.
    /// The test uses the checkpoint's tokenizer.
    /// EREMITE_CHECKPOINT names its directory.
    /// Without EREMITE_CHECKPOINT, the test returns at once without checking.
    #[test]
    fn each_image_becomes_an_image_token_and_its_bytes() {
        let Ok(dir) = std::env::var("EREMITE_CHECKPOINT") else {
            return;
        };
        let tokenizer = engine::Tokenizer::open(&dir, Some(IMAGE_TOKEN));
        // "aGk=" is "hi" in base64.
        let request = json!({"input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,aGk="},
        ]}]});
        let prepared = prepare_test_request(&request, Api::Responses, tokenizer).expect("prepared");
        let request = prepared
            .build_worker_request(0, String::new())
            .expect("a worker request");
        assert_eq!(request.media, [b"hi"]);
        let image_token = tokenizer.find_added_token(IMAGE_TOKEN);
        let image_places = request
            .prompt
            .iter()
            .filter(|&&t| Some(t) == image_token)
            .count();
        assert_eq!(image_places, 1);
        let request = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}]});
        let e = prepare_test_request(&request, Api::ChatCompletions, tokenizer)
            .err()
            .expect("refused");
        assert!(e.message.contains("only data: URLs"), "{}", e.message);
        let not_an_object = json!(["hi"]);
        assert!(prepare_test_request(&not_an_object, Api::ChatCompletions, tokenizer).is_err());
    }

    /// Rewrites the Responses request `request` with the tests' defaults.
    /// Returns the rewritten object and the settings, or the request's error.
    fn rewrite_responses_json(
        request: Value,
    ) -> Result<(Map<String, Value>, RequestSettings), Error> {
        let Value::Object(mut root) = request else {
            panic!("an object")
        };
        let s = rewrite_responses_request(&mut root, &build_test_defaults())?;
        Ok((root, s))
    }

    #[test]
    fn the_server_reads_its_settings_from_responses_requests() {
        let (root, s) = rewrite_responses_json(json!({
            "model": "deepseek-v4.1-flash", "input": "hi", "max_output_tokens": 300,
            "temperature": 0.5, "reasoning": {"effort": "xhigh", "summary": "auto"},
        }))
        .expect("settings");
        assert_eq!(root["model"], RECIPE_MODEL_NAME);
        assert_eq!(
            (s.effort, &root["reasoning"]["effort"]),
            (Some(75), &json!("high"))
        );
        assert_eq!(root["reasoning"]["summary"], "auto");
        assert_eq!(s.max_tokens, 300);
        assert_eq!(s.sampling.temperature, 0.5);

        let effort = |v: Value| {
            let (root, s) = rewrite_responses_json(v).expect("settings");
            (s.effort, root.get("reasoning").map(|r| r["effort"].clone()))
        };
        assert_eq!(
            effort(json!({"input": "hi", "reasoning": {"effort": "none"}})),
            (None, Some(json!("none")))
        );
        assert_eq!(
            effort(json!({"input": "hi", "reasoning": {"effort": "low"}})),
            (Some(50), Some(json!("low")))
        );
        assert_eq!(
            effort(json!({"input": "hi", "reasoning": {"effort": "max"}})),
            (Some(100), Some(json!("max")))
        );
        assert_eq!(
            effort(json!({"input": "hi", "reasoning": {"effort": 62}})),
            (Some(62), Some(json!("high")))
        );
        // Without an effort, the model's default applies: thinking at effort
        // 75.
        assert_eq!(
            effort(json!({"input": "hi"})),
            (Some(75), Some(json!("high")))
        );
        assert_eq!(
            effort(json!({"input": "hi", "reasoning": null})),
            (Some(75), Some(json!("high")))
        );

        let (_, s) = rewrite_responses_json(json!({"input": "hi"})).expect("settings");
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        for key in ["previous_response_id", "conversation"] {
            let mut root = Map::new();
            root.insert("input".into(), "hi".into());
            root.insert(key.into(), "resp_1".into());
            let e = rewrite_responses_request(&mut root, &build_test_defaults())
                .err()
                .expect("refused");
            assert!(e.message.contains(key), "{}", e.message);
        }
        assert!(rewrite_responses_json(json!({"input": "hi", "max_output_tokens": 0})).is_err());
    }

    /// A request in the shape that Codex 0.167 sends.
    /// It has developer and user messages, a function and a namespace of
    /// functions.
    /// It also declares the hosted web search tool.
    /// deepseek-recipe drops that tool.
    /// It replays a tool call with its reasoning.
    #[test]
    fn codex_requests_render_without_the_web_search_tool() {
        let request = json!({
            "model": "deepseek-v4.1-flash", "stream": true, "store": false, "tool_choice": "auto",
            "parallel_tool_calls": true, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "client_metadata": {"thread_id": "t"},
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "tools": [
                {"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": false,
                 "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"], "additionalProperties": false}},
                {"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
                    {"type": "function", "name": "spawn_agent", "description": "Starts an agent.", "strict": false,
                     "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}]},
                {"type": "web_search", "external_web_access": false},
            ],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are a coding agent."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List the files."}]},
                {"type": "reasoning", "summary": [], "encrypted_content": null,
                 "content": [{"type": "reasoning_text", "text": "I should run ls."}]},
                {"type": "function_call", "call_id": "call_0", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_0", "output": "a.txt\nb.txt"},
            ],
        });
        let Value::Object(root) = request else {
            unreachable!()
        };
        assert_eq!(count_dropped_tools(&root), 1);
        let r = render_request(root, &build_test_defaults(), Api::Responses).expect("rendered");
        assert!(r.converted_request.stream);
        for text in [
            "You are a coding agent.",
            "exec_command",
            "multi_agent_v1::spawn_agent",
            "I should run ls.",
            "a.txt",
        ] {
            assert!(
                r.prompt.contains(text),
                "{text} is not in the prompt:\n{}",
                r.prompt
            );
        }
        assert!(!r.prompt.contains("web_search"), "{}", r.prompt);
    }

    /// A part of the web.run tool of Codex 0.167.
    /// The tool is a namespace with one function.
    /// That function has a generated schema.
    /// The test keeps three of the schema's eleven properties.
    #[test]
    fn codex_web_run_renders() {
        let tool = json!({
            "type": "namespace", "name": "web", "description": "Tools in the web namespace.",
            "tools": [{
                "type": "function", "name": "run", "strict": false,
                "description": "Tool for accessing the internet.",
                "parameters": {"type": "object", "properties": {
                    "open": {
                        "type": "array", "description": "Open pages by reference id or URL.",
                        "items": {"type": "object", "required": ["ref_id"], "properties": {
                            "lineno": {"type": "integer",
                                       "description": "Line number to position the page at."},
                            "ref_id": {"type": "string",
                                       "description": "Reference id or URL to open."}}}},
                    "response_length": {
                        "type": "string", "enum": ["short", "medium", "long"],
                        "description": "Set the length of the response to be returned."},
                    "search_query": {
                        "type": "array",
                        "description": "Query the internet search engine for a given list of queries.",
                        "items": {"type": "object", "required": ["q"], "properties": {
                            "domains": {"type": "array", "items": {"type": "string"},
                                        "description": "Whether to filter by a specific list of domains."},
                            "q": {"type": "string", "description": "Search query."},
                            "recency": {"type": "integer",
                                        "description": "Whether to filter by recency, as a number of recent days."}}}}}}
            }]
        });
        let request = json!({
            "model": "deepseek-v4.1-flash", "stream": true, "tools": [tool],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Search the web."}]},
                {"type": "function_call", "call_id": "call_1", "name": "run", "namespace": "web",
                 "arguments": "{\"search_query\": [{\"q\": \"eremite\"}]}"},
                {"type": "function_call_output", "call_id": "call_1",
                 "output": [{"type": "input_text", "text": "[turn0search0] Example https://example.com/"}]},
            ],
        });
        let Value::Object(root) = request else {
            unreachable!()
        };
        let r = render_request(root, &build_test_defaults(), Api::Responses).expect("rendered");
        for text in ["web::run", "search_query", "turn0search0"] {
            assert!(r.prompt.contains(text), "{text} is not in the prompt");
        }
    }

    /// Checks and counts the messages, the input items and the tools of
    /// prepared requests.
    /// The test uses the checkpoint's tokenizer.
    /// EREMITE_CHECKPOINT names its directory.
    /// Without EREMITE_CHECKPOINT, the test returns at once without checking.
    #[test]
    fn prepared_requests_count_their_messages_and_tools() {
        let Ok(dir) = std::env::var("EREMITE_CHECKPOINT") else {
            return;
        };
        let tokenizer = engine::Tokenizer::open(&dir, Some(IMAGE_TOKEN));
        let user = json!({"role": "user", "content": "hi"});
        let assistant = json!({"role": "assistant", "content": "hello"});
        for (request, api) in [
            (json!({"messages": []}), Api::ChatCompletions),
            (json!({"messages": [user, assistant]}), Api::ChatCompletions),
            (json!({"input": ""}), Api::Responses),
            (json!({"input": []}), Api::Responses),
        ] {
            let prepared = prepare_test_request(&request, api, tokenizer);
            assert!(prepared.is_err(), "{request}");
        }
        let request = json!({
            "messages": [user], "reasoning_effort": 62, "stream": true,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        });
        let prepared =
            prepare_test_request(&request, Api::ChatCompletions, tokenizer).expect("prepared");
        let counts = (
            prepared.message_count,
            prepared.tool_count,
            prepared.dropped_tool_count,
        );
        assert_eq!(counts, (1, 1, 0));
        assert!(prepared.is_streamed);
        assert_eq!(prepared.describe_thinking(), "thinking effort 62");
        let request = json!({
            "input": "hi", "reasoning": {"effort": "none"}, "tools": [{"type": "web_search"}],
        });
        let prepared = prepare_test_request(&request, Api::Responses, tokenizer).expect("prepared");
        let counts = (
            prepared.message_count,
            prepared.tool_count,
            prepared.dropped_tool_count,
        );
        assert_eq!(counts, (1, 1, 1));
        assert_eq!(prepared.describe_thinking(), "thinking off");
    }
}
