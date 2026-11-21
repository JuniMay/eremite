//! The Inkling-2-Mini chat format for the Chat Completions and Responses APIs.
//! It renders conversations in Inkling-2-Mini's message template, decodes
//! base64 image inputs for the vision tower, and parses the model's thinking,
//! text and JSON tool-call blocks into either API's response or event stream.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedReceiver;

use super::{
    Api, DEFAULT_MAX_TOKENS, Error, build_error_response, format_error_json, generate_random_id,
};
use crate::engine;
use crate::worker::{self, Event, FinishReason, Sampling};

/// The model id of an Inkling-2-Mini model whose configuration gives none.
pub const DEFAULT_MODEL_ID: &str = "inkling-2-mini";
/// The display name shown in model lists and Codex's catalog.
pub const DISPLAY_NAME: &str = "Inkling-2-Mini (eremite)";
/// The spelling of the image placeholder token in Inkling-2-Mini's tokenizer
/// (id 200054 in tokenizer_config.json).
pub const IMAGE_TOKEN: &str = "<|unused_200054|>";

/// Special token spellings from Inkling-2-Mini's tokenizer_config.json.
const TOKEN_USER: &str = "<|message_user|>";
/// Special token starting an assistant message block.
const TOKEN_MODEL: &str = "<|message_model|>";
/// Special token starting a system message block.
const TOKEN_SYSTEM: &str = "<|message_system|>";
/// Special token starting a tool response block.
const TOKEN_TOOL: &str = "<|message_tool|>";
/// Special token starting plain text content inside a message block.
const TOKEN_TEXT: &str = "<|content_text|>";
/// Special token starting image content inside a message block.
const TOKEN_IMAGE: &str = "<|content_image|>";
/// Special token starting reasoning content inside a model block.
const TOKEN_THINKING: &str = "<|content_thinking|>";
/// Special token starting XML-wrapped tool declarations.
const TOKEN_XML: &str = "<|content_xml|>";
/// Special token starting tool error content inside a model block.
const TOKEN_TOOL_ERROR: &str = "<|content_tool_error|>";
/// Special token starting a JSON tool invocation block.
const TOKEN_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
/// Special token starting a text tool invocation block.
const TOKEN_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
/// Special token ending one message block.
const TOKEN_END_MESSAGE: &str = "<|end_message|>";
/// Special token ending one assistant sampling turn (id 200006).
const TOKEN_END_SAMPLING: &str = "<|content_model_end_sampling|>";
/// Special end-of-text padding token (id 199999).
const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";

/// Sentinel byte that eremite_tokenize turns into the checkpoint's image token.
const MEDIA_MARK_BYTE: u8 = 0xFF;
/// Header name emitted after the system token for tool declarations.
const TOOL_DECLARE_HEADER: &str = "tool_declare";
/// Prefix of the system message that sets the thinking effort level.
const THINKING_EFFORT_PREFIX: &str = "Thinking effort level: ";
/// Separator placed between a Responses API namespace name and tool name.
const NAMESPACE_SEPARATOR: &str = "::";
/// Prefix of an HTTP data URL carrying inline media bytes.
const DATA_URL_PREFIX: &str = "data:";

/// Reasoning effort value that disables thinking (chat_template.jinja).
pub const EFFORT_NONE: f64 = 0.0;
/// Reasoning effort value for level minimal (chat_template.jinja).
pub const EFFORT_MINIMAL: f64 = 0.1;
/// Reasoning effort value for level low (chat_template.jinja).
pub const EFFORT_LOW: f64 = 0.2;
/// Reasoning effort value for level medium (chat_template.jinja).
pub const EFFORT_MEDIUM: f64 = 0.7;
/// Reasoning effort value for level high and the default (chat_template.jinja).
pub const EFFORT_HIGH: f64 = 0.9;
/// Reasoning effort value for levels xhigh and max (chat_template.jinja).
pub const EFFORT_MAX: f64 = 0.99;

/// Named reasoning effort levels and their numeric values from
/// chat_template.jinja.
const EFFORT_LEVELS: &[(&str, f64)] = &[
    ("none", EFFORT_NONE),
    ("minimal", EFFORT_MINIMAL),
    ("low", EFFORT_LOW),
    ("medium", EFFORT_MEDIUM),
    ("high", EFFORT_HIGH),
    ("xhigh", EFFORT_MAX),
    ("max", EFFORT_MAX),
];

/// A served Inkling-2-Mini model's id and defaults for omitted request fields.
#[derive(Clone)]
pub struct Defaults {
    /// Model id reported in responses when a request omits `model`.
    pub model_id: String,
    /// Sampling settings for fields that a request leaves unset.
    pub sampling: Sampling,
    /// Thinking effort in 0.0..=0.99 when a request leaves effort unset.
    pub effort: f64,
}

impl Defaults {
    /// Returns defaults for `model_id`: temperature 1.0, top_p 1.0, and high
    /// thinking effort (0.9).
    pub fn new(model_id: String) -> Defaults {
        Defaults {
            model_id,
            sampling: Sampling::default(),
            effort: EFFORT_HIGH,
        }
    }
}

/// Returns the conversation role whose block `token` begins, or None for any
/// other token.
pub fn find_role_of_token(token: &str) -> Option<&'static str> {
    match token {
        TOKEN_SYSTEM => Some("system"),
        TOKEN_USER => Some("user"),
        TOKEN_MODEL => Some("assistant"),
        TOKEN_TOOL => Some("tool"),
        _ => None,
    }
}

// ---- conversation model and JSON canonicalization ----

/// Message role recognized by Inkling-2-Mini's template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// System or developer instruction message.
    System,
    /// End-user prompt message.
    User,
    /// Model reply turn.
    Assistant,
    /// Tool result message.
    Tool,
}

impl Role {
    /// Returns the special token spelling that opens this role's blocks.
    fn token(self) -> &'static str {
        match self {
            Role::System => TOKEN_SYSTEM,
            Role::User => TOKEN_USER,
            Role::Assistant => TOKEN_MODEL,
            Role::Tool => TOKEN_TOOL,
        }
    }
}

/// One part inside a multipart message content array.
#[derive(Clone, Debug)]
enum ContentPart {
    /// Plain UTF-8 text part.
    Text(String),
    /// Decoded encoded-image file bytes (PNG, JPEG, WebP, etc.).
    Image(Vec<u8>),
}

/// Content body of one conversation message.
#[derive(Clone, Debug)]
enum MessageContent {
    /// No content field or null content.
    None,
    /// Single string content.
    Text(String),
    /// Ordered sequence of text and image parts.
    Parts(Vec<ContentPart>),
}

/// One historical assistant tool call to render into the prompt.
#[derive(Clone, Debug)]
struct ToolCallInput {
    /// Optional call id used to resolve subsequent tool response names.
    id: Option<String>,
    /// Function name invoked by the assistant.
    name: String,
    /// Parsed JSON object of arguments.
    arguments: Map<String, Value>,
}

/// One message in the normalized conversation.
#[derive(Clone, Debug)]
struct Message {
    /// Conversation role of this message.
    role: Role,
    /// Optional explicit sender or tool name appended to the role header.
    name: Option<String>,
    /// Optional tool call id linking a tool message to its assistant call.
    tool_call_id: Option<String>,
    /// Optional assistant reasoning text rendered before content.
    reasoning_content: Option<String>,
    /// Message content body.
    content: MessageContent,
    /// Historical tool calls attached to an assistant turn.
    tool_calls: Vec<ToolCallInput>,
}

/// One tool declaration rendered into the leading `tool_declare` block.
#[derive(Clone, Debug)]
struct ToolSpec {
    /// Function name visible to the model.
    name: String,
    /// Human-readable description of the tool.
    description: String,
    /// JSON Schema describing the tool parameters.
    parameters: Value,
    /// Tool type string (typically "function").
    tool_type: String,
}

/// Metadata kept from Responses API tool declarations so tool calls round-trip
/// back into Responses custom or namespaced tool items.
#[derive(Clone, Debug, Default)]
struct ResponsesToolMeta {
    /// Names of tools declared with `type: "custom"` (such as `apply_patch`).
    custom_tools: HashSet<String>,
    /// Map from rendered `"namespace::name"` back to `(namespace, name)`.
    namespaced_tools: HashMap<String, (String, String)>,
}

/// Normalized conversation ready to render into Inkling-2-Mini's template.
#[derive(Clone, Debug)]
struct Conversation {
    /// Declared tools rendered at the top of the prompt.
    tools: Vec<ToolSpec>,
    /// Ordered conversation messages.
    messages: Vec<Message>,
    /// Thinking effort in 0.0..=0.99.
    effort: f64,
}

/// Request settings extracted alongside the normalized conversation.
#[derive(Clone, Debug)]
struct RequestSettings {
    /// Sampling parameters for the engine worker.
    sampling: Sampling,
    /// Maximum output tokens to generate.
    max_tokens: i64,
    /// Model id echoed in responses.
    model_id: String,
    /// Client stop sequences (Chat Completions `stop`).
    stop: Vec<String>,
    /// Thinking effort in 0.0..=0.99.
    effort: f64,
    /// Whether the response streams as server-sent events.
    is_streamed: bool,
    /// Whether Chat Completions streaming should include a final usage object.
    include_usage: bool,
    /// Responses API tool round-trip metadata.
    responses_tools: ResponsesToolMeta,
}

/// Returns a copy of `value` with every JSON object's keys sorted in
/// lexicographical order, matching Python's `json.dumps(..., sort_keys=True)`.
fn sort_json_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            let mut out = Map::with_capacity(map.len());
            for (k, v) in sorted {
                out.insert(k.clone(), sort_json_keys(v));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_json_keys).collect()),
        other => other.clone(),
    }
}

/// Formats `effort` as Inkling-2-Mini's template writes it: `"0"` for zero and
/// the shortest decimal representation otherwise.
fn format_effort(effort: f64) -> String {
    if effort == 0.0 {
        "0".into()
    } else {
        format!("{effort}")
    }
}

/// Parses a `reasoning_effort` value into `[0.0, 0.99]`.
fn parse_effort_value(value: &Value) -> Result<f64, Error> {
    let effort = match value {
        Value::String(s) => {
            let key = s.trim();
            EFFORT_LEVELS
                .iter()
                .find(|&&(name, _)| name == key)
                .map(|&(_, v)| v)
                .ok_or_else(|| Error::new_bad_request(format!("Unknown reasoning_effort: {s}")))?
        }
        Value::Number(n) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| Error::new_bad_request("reasoning_effort must be a finite number"))?,
        _ => {
            return Err(Error::new_bad_request(
                "reasoning_effort must be a string or a number in [0.0, 0.99]",
            ));
        }
    };
    if !(EFFORT_NONE..=EFFORT_MAX).contains(&effort) {
        return Err(Error::new_bad_request(
            "reasoning_effort must be in [0.0, 0.99]",
        ));
    }
    Ok(effort)
}

/// Parses a tool call's `arguments` field (JSON string or JSON object) into a
/// JSON object map.
fn parse_tool_arguments(value: Option<&Value>) -> Result<Map<String, Value>, Error> {
    match value {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(Value::String(text)) => {
            if text.trim().is_empty() {
                return Ok(Map::new());
            }
            match serde_json::from_str::<Value>(text) {
                Ok(Value::Object(map)) => Ok(map),
                Ok(_) => Err(Error::new_bad_request(
                    "tool call arguments must be a JSON object",
                )),
                Err(error) => Err(Error::new_bad_request(format!(
                    "tool call arguments must be valid JSON: {error}"
                ))),
            }
        }
        Some(_) => Err(Error::new_bad_request(
            "tool call arguments must be an object",
        )),
    }
}

/// Decodes an image source URL into raw encoded image bytes, refusing remote
/// URLs and file paths.
fn decode_image_source(url: &str) -> Result<Vec<u8>, Error> {
    if !url.starts_with(DATA_URL_PREFIX) {
        return Err(Error::new_bad_request(
            "only data: URLs with base64 data are read (the server fetches no URLs and reads no files)",
        ));
    }
    super::decode_data_url(url)
}

/// Extracts the image URL string from a content part object (`image`,
/// `input_image` or `image_url`).
fn extract_image_url(part: &Map<String, Value>) -> Result<&str, Error> {
    if let Some(image_url) = part.get("image_url") {
        if let Some(url) = image_url.as_str() {
            return Ok(url);
        }
        if let Some(url) = image_url.get("url").and_then(Value::as_str) {
            return Ok(url);
        }
    }
    if let Some(url) = part.get("url").and_then(Value::as_str) {
        return Ok(url);
    }
    if let Some(url) = part.get("image").and_then(Value::as_str) {
        return Ok(url);
    }
    Err(Error::new_bad_request(
        "image content part requires an image_url data URL",
    ))
}

/// Parses an array of multipart content items for `role`.
fn parse_content_parts(items: &[Value], role: Role) -> Result<Vec<ContentPart>, Error> {
    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        if let Some(text) = item.as_str() {
            parts.push(ContentPart::Text(text.to_string()));
            continue;
        }
        let Some(obj) = item.as_object() else {
            return Err(Error::new_bad_request(
                "content parts must be strings or objects",
            ));
        };
        let part_type = obj.get("type").and_then(Value::as_str);
        match part_type {
            None | Some("text" | "input_text" | "output_text") => {
                let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
                parts.push(ContentPart::Text(text.to_string()));
            }
            Some("image" | "input_image" | "image_url") => {
                if role != Role::User && role != Role::Tool {
                    return Err(Error::new_bad_request(
                        "Image content is only supported for user and tool messages",
                    ));
                }
                let url = extract_image_url(obj)?;
                parts.push(ContentPart::Image(decode_image_source(url)?));
            }
            Some("audio" | "input_audio" | "audio_url") => {
                return Err(Error::new_bad_request(
                    "Inkling 2 does not accept audio input",
                ));
            }
            Some(other) => {
                return Err(Error::new_bad_request(format!(
                    "Unsupported content part type: {other}"
                )));
            }
        }
    }
    Ok(parts)
}

/// Parses a message's `content` value (absent, null, string or part array) for
/// `role`.
fn parse_message_content(value: Option<&Value>, role: Role) -> Result<MessageContent, Error> {
    match value {
        None | Some(Value::Null) => Ok(MessageContent::None),
        Some(Value::String(text)) => Ok(MessageContent::Text(text.clone())),
        Some(Value::Array(items)) => Ok(MessageContent::Parts(parse_content_parts(items, role)?)),
        Some(_) => Err(Error::new_bad_request(
            "message content must be a string or an array of parts",
        )),
    }
}

/// Returns whether a request's `tool_choice` disables tool calls (`"none"`).
fn tool_choice_is_none(root: &Map<String, Value>) -> bool {
    root.get("tool_choice").and_then(Value::as_str) == Some("none")
}

/// Parses Chat Completions tool definitions into normalized `ToolSpec` items.
fn parse_chat_tools(root: &Map<String, Value>) -> Result<(Vec<ToolSpec>, usize), Error> {
    let Some(raw) = root.get("tools") else {
        return Ok((Vec::new(), 0));
    };
    if raw.is_null() {
        return Ok((Vec::new(), 0));
    }
    let Some(list) = raw.as_array() else {
        return Err(Error::new_bad_request("tools must be an array"));
    };
    let declared = list.len();
    if tool_choice_is_none(root) {
        return Ok((Vec::new(), declared));
    }
    let mut specs = Vec::with_capacity(list.len());
    for entry in list {
        let Some(tool) = entry.as_object() else {
            return Err(Error::new_bad_request("each tool must be an object"));
        };
        let fn_obj = tool
            .get("function")
            .and_then(Value::as_object)
            .unwrap_or(tool);
        let name = fn_obj
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::new_bad_request("tool function name must be a nonempty string"))?
            .to_string();
        let description = fn_obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let parameters = fn_obj
            .get("parameters")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let tool_type = tool
            .get("type")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("function")
            .to_string();
        specs.push(ToolSpec {
            name,
            description,
            parameters,
            tool_type,
        });
    }
    Ok((specs, declared))
}

/// Builds a normalized `ToolSpec` from one Responses API `function` tool
/// object, optionally prefixed by `namespace`.
fn parse_responses_function_tool(
    tool: &Map<String, Value>,
    namespace: Option<&str>,
) -> Result<(ToolSpec, Option<(String, String)>), Error> {
    let raw_name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::new_bad_request("function tool name must be a nonempty string"))?;
    let (name, mapping) = match namespace {
        Some(ns) => (
            format!("{ns}{NAMESPACE_SEPARATOR}{raw_name}"),
            Some((ns.to_string(), raw_name.to_string())),
        ),
        None => (raw_name.to_string(), None),
    };
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let parameters = tool
        .get("parameters")
        .filter(|v| !v.is_null())
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    Ok((
        ToolSpec {
            name,
            description,
            parameters,
            tool_type: "function".into(),
        },
        mapping,
    ))
}

/// Parses Responses API tool declarations (`function`, `namespace` and
/// `custom`), dropping unsupported hosted tool types and returning their count.
fn parse_responses_tools(
    root: &Map<String, Value>,
) -> Result<(Vec<ToolSpec>, ResponsesToolMeta, usize, usize), Error> {
    let Some(raw) = root.get("tools") else {
        return Ok((Vec::new(), ResponsesToolMeta::default(), 0, 0));
    };
    if raw.is_null() {
        return Ok((Vec::new(), ResponsesToolMeta::default(), 0, 0));
    }
    let Some(list) = raw.as_array() else {
        return Err(Error::new_bad_request("tools must be an array"));
    };
    let declared = list.len();
    let suppress = tool_choice_is_none(root);
    let mut specs = Vec::new();
    let mut meta = ResponsesToolMeta::default();
    let mut dropped = 0;
    for entry in list {
        let Some(tool) = entry.as_object() else {
            return Err(Error::new_bad_request("each tool must be an object"));
        };
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => {
                let (spec, _) = parse_responses_function_tool(tool, None)?;
                if !suppress {
                    specs.push(spec);
                }
            }
            Some("namespace") => {
                let ns = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        Error::new_bad_request("namespace tool requires a nonempty name")
                    })?;
                let inner = tool.get("tools").and_then(Value::as_array).ok_or_else(|| {
                    Error::new_bad_request("namespace tool requires a tools array")
                })?;
                for child in inner {
                    let Some(child_obj) = child.as_object() else {
                        return Err(Error::new_bad_request(
                            "each namespace tool entry must be an object",
                        ));
                    };
                    if child_obj.get("type").and_then(Value::as_str) != Some("function") {
                        dropped += 1;
                        continue;
                    }
                    let (spec, mapping) = parse_responses_function_tool(child_obj, Some(ns))?;
                    if let Some(pair) = mapping {
                        meta.namespaced_tools.insert(spec.name.clone(), pair);
                    }
                    if !suppress {
                        specs.push(spec);
                    }
                }
            }
            Some("custom") => {
                let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| Error::new_bad_request("custom tool requires a nonempty name"))?
                    .to_string();
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let input_desc = tool
                    .get("format")
                    .and_then(Value::as_object)
                    .and_then(|fmt| fmt.get("definition"))
                    .and_then(Value::as_str)
                    .unwrap_or("Raw text input for the tool.");
                let parameters = json!({
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": input_desc,
                        }
                    },
                    "required": ["input"],
                });
                meta.custom_tools.insert(name.clone());
                if !suppress {
                    specs.push(ToolSpec {
                        name,
                        description,
                        parameters,
                        tool_type: "function".into(),
                    });
                }
            }
            _ => dropped += 1,
        }
    }
    Ok((specs, meta, declared, dropped))
}

/// Parses a Chat Completions JSON request into a normalized conversation and
/// request settings.
fn parse_chat_request(
    root: &Map<String, Value>,
    defaults: &Defaults,
) -> Result<(Conversation, RequestSettings, usize, usize), Error> {
    let sampling = super::read_sampling(root, defaults.sampling)?;
    let max_tokens = super::read_token_limit(root, DEFAULT_MAX_TOKENS)?;
    let stop = super::read_stop_sequences(root)?;
    let model_id = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&defaults.model_id)
        .to_string();
    let is_streamed = match root.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(Error::new_bad_request("stream must be true or false")),
    };
    let include_usage = is_streamed
        && root
            .get("stream_options")
            .and_then(Value::as_object)
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(true);

    let thinking_disabled = match root.get("thinking") {
        Some(Value::Bool(false)) => true,
        Some(Value::Object(obj)) => obj.get("type").and_then(Value::as_str) == Some("disabled"),
        _ => false,
    };
    let effort = match root.get("reasoning_effort") {
        Some(value) if !value.is_null() => parse_effort_value(value)?,
        _ if thinking_disabled => EFFORT_NONE,
        _ => defaults.effort,
    };

    let (tools, tool_count) = parse_chat_tools(root)?;

    let raw_messages = root
        .get("messages")
        .and_then(Value::as_array)
        .filter(|list| !list.is_empty())
        .ok_or_else(|| Error::new_bad_request("messages must be a non-empty array"))?;
    let message_count = raw_messages.len();

    let mut messages = Vec::with_capacity(message_count);
    let mut last_call_ids: Vec<String> = Vec::new();
    let mut answered_calls = 0usize;

    for raw in raw_messages {
        let Some(obj) = raw.as_object() else {
            return Err(Error::new_bad_request("each message must be an object"));
        };
        let role_str = obj
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new_bad_request("message role must be a string"))?;
        let role = match role_str {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(Error::new_bad_request(format!(
                    "Unknown message role: {other}"
                )));
            }
        };
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from);
        let reasoning_content = if role == Role::Assistant {
            obj.get("reasoning_content")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
        } else {
            None
        };
        let content = parse_message_content(obj.get("content"), role)?;
        let mut tool_calls = Vec::new();
        let mut tool_call_id = obj
            .get("tool_call_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from);

        if role == Role::Assistant {
            last_call_ids.clear();
            answered_calls = 0;
            if let Some(raw_calls) = obj.get("tool_calls").and_then(Value::as_array) {
                for (call_index, raw_call) in raw_calls.iter().enumerate() {
                    let Some(call_obj) = raw_call.as_object() else {
                        return Err(Error::new_bad_request("each tool call must be an object"));
                    };
                    let fn_obj = call_obj
                        .get("function")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            Error::new_bad_request("tool call must contain a function object")
                        })?;
                    let fn_name = fn_obj
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            Error::new_bad_request("tool call function name must be a string")
                        })?
                        .to_string();
                    if let Some(ref msg_name) = name
                        && msg_name != &fn_name
                    {
                        return Err(Error::new_bad_request(format!(
                            "assistant name {msg_name} does not match its tool call {fn_name}"
                        )));
                    }
                    let arguments = parse_tool_arguments(fn_obj.get("arguments"))?;
                    let id = call_obj
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .unwrap_or_else(|| format!("call_{call_index}"));
                    last_call_ids.push(id.clone());
                    tool_calls.push(ToolCallInput {
                        id: Some(id),
                        name: fn_name,
                        arguments,
                    });
                }
            }
        } else if role == Role::Tool {
            if tool_call_id.is_none()
                && let Some(id) = last_call_ids.get(answered_calls)
            {
                tool_call_id = Some(id.clone());
            }
            answered_calls += 1;
        }

        messages.push(Message {
            role,
            name,
            tool_call_id,
            reasoning_content,
            content,
            tool_calls,
        });
    }

    if messages.last().is_some_and(|m| m.role == Role::Assistant) {
        return Err(Error::new_bad_request(
            "the last message must not be the assistant's",
        ));
    }

    let settings = RequestSettings {
        sampling,
        max_tokens,
        model_id,
        stop,
        effort,
        is_streamed,
        include_usage,
        responses_tools: ResponsesToolMeta::default(),
    };
    Ok((
        Conversation {
            tools,
            messages,
            effort,
        },
        settings,
        message_count,
        tool_count,
    ))
}

/// Extracts concatenated reasoning text from a Responses API `reasoning` item.
fn extract_responses_reasoning(item: &Map<String, Value>) -> String {
    let mut text = String::new();
    if let Some(parts) = item.get("content").and_then(Value::as_array) {
        for part in parts {
            if let Some(s) = part.get("text").and_then(Value::as_str) {
                text.push_str(s);
            }
        }
    }
    if text.is_empty()
        && let Some(parts) = item.get("summary").and_then(Value::as_array)
    {
        for part in parts {
            if let Some(s) = part.get("text").and_then(Value::as_str) {
                text.push_str(s);
            }
        }
    }
    text
}

/// Returns a mutable reference to `pending`, initializing an empty assistant
/// message when none is in progress.
fn ensure_assistant(pending: &mut Option<Message>) -> &mut Message {
    pending.get_or_insert_with(|| Message {
        role: Role::Assistant,
        name: None,
        tool_call_id: None,
        reasoning_content: None,
        content: MessageContent::None,
        tool_calls: Vec::new(),
    })
}

/// Parses a Responses API JSON request into a normalized conversation,
/// request settings, input item count, declared tool count, and dropped tool
/// count.
fn parse_responses_request(
    root: &Map<String, Value>,
    defaults: &Defaults,
) -> Result<(Conversation, RequestSettings, usize, usize, usize), Error> {
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
    let stop = super::read_stop_sequences(root)?;
    let model_id = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&defaults.model_id)
        .to_string();
    let is_streamed = match root.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(Error::new_bad_request("stream must be true or false")),
    };
    let effort = match root.get("reasoning") {
        None | Some(Value::Null) => defaults.effort,
        Some(Value::Object(obj)) => match obj.get("effort") {
            None | Some(Value::Null) => defaults.effort,
            Some(value) => parse_effort_value(value)?,
        },
        Some(_) => return Err(Error::new_bad_request("reasoning must be an object")),
    };

    let (tools, responses_tools, tool_count, dropped_tool_count) = parse_responses_tools(root)?;

    let mut messages: Vec<Message> = Vec::new();
    if let Some(instructions) = root.get("instructions").and_then(Value::as_str)
        && !instructions.is_empty()
    {
        messages.push(Message {
            role: Role::System,
            name: None,
            tool_call_id: None,
            reasoning_content: None,
            content: MessageContent::Text(instructions.to_string()),
            tool_calls: Vec::new(),
        });
    }

    let item_count = match root.get("input") {
        Some(Value::String(text)) if !text.is_empty() => {
            messages.push(Message {
                role: Role::User,
                name: None,
                tool_call_id: None,
                reasoning_content: None,
                content: MessageContent::Text(text.clone()),
                tool_calls: Vec::new(),
            });
            1
        }
        Some(Value::Array(items)) if !items.is_empty() => {
            let mut pending_assistant: Option<Message> = None;
            let flush_assistant = |pending: &mut Option<Message>, out: &mut Vec<Message>| {
                if let Some(msg) = pending.take() {
                    out.push(msg);
                }
            };

            for raw_item in items {
                let Some(item) = raw_item.as_object() else {
                    return Err(Error::new_bad_request("each input item must be an object"));
                };
                let item_type = item.get("type").and_then(Value::as_str);
                match item_type {
                    Some("message") | None if item.contains_key("role") => {
                        let role_str =
                            item.get("role").and_then(Value::as_str).ok_or_else(|| {
                                Error::new_bad_request("message role must be a string")
                            })?;
                        let role = match role_str {
                            "system" | "developer" => Role::System,
                            "user" => Role::User,
                            "assistant" => Role::Assistant,
                            other => {
                                return Err(Error::new_bad_request(format!(
                                    "Unknown message role: {other}"
                                )));
                            }
                        };
                        let content = parse_message_content(item.get("content"), role)?;
                        if role == Role::Assistant {
                            let target = ensure_assistant(&mut pending_assistant);
                            match (&mut target.content, content) {
                                (MessageContent::None, incoming) => target.content = incoming,
                                (MessageContent::Text(existing), MessageContent::Text(extra)) => {
                                    existing.push_str(&extra);
                                }
                                (MessageContent::Parts(existing), MessageContent::Parts(extra)) => {
                                    existing.extend(extra);
                                }
                                (MessageContent::Text(existing), MessageContent::Parts(extra)) => {
                                    let mut combined =
                                        vec![ContentPart::Text(std::mem::take(existing))];
                                    combined.extend(extra);
                                    target.content = MessageContent::Parts(combined);
                                }
                                (MessageContent::Parts(existing), MessageContent::Text(extra)) => {
                                    existing.push(ContentPart::Text(extra));
                                }
                                (_, MessageContent::None) => {}
                            }
                        } else {
                            flush_assistant(&mut pending_assistant, &mut messages);
                            messages.push(Message {
                                role,
                                name: None,
                                tool_call_id: None,
                                reasoning_content: None,
                                content,
                                tool_calls: Vec::new(),
                            });
                        }
                    }
                    Some("reasoning") => {
                        let text = extract_responses_reasoning(item);
                        if !text.is_empty() {
                            let target = ensure_assistant(&mut pending_assistant);
                            match &mut target.reasoning_content {
                                Some(existing) => existing.push_str(&text),
                                None => target.reasoning_content = Some(text),
                            }
                        }
                    }
                    Some("function_call") => {
                        let raw_name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                Error::new_bad_request("function_call requires a nonempty name")
                            })?;
                        let full_name = match item
                            .get("namespace")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            Some(ns) => format!("{ns}{NAMESPACE_SEPARATOR}{raw_name}"),
                            None => raw_name.to_string(),
                        };
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(String::from);
                        let arguments = parse_tool_arguments(item.get("arguments"))?;
                        let target = ensure_assistant(&mut pending_assistant);
                        target.tool_calls.push(ToolCallInput {
                            id: call_id,
                            name: full_name,
                            arguments,
                        });
                    }
                    Some("custom_tool_call") => {
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                Error::new_bad_request("custom_tool_call requires a nonempty name")
                            })?
                            .to_string();
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(String::from);
                        let input = item
                            .get("input")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let mut arguments = Map::new();
                        arguments.insert("input".into(), Value::String(input));
                        let target = ensure_assistant(&mut pending_assistant);
                        target.tool_calls.push(ToolCallInput {
                            id: call_id,
                            name,
                            arguments,
                        });
                    }
                    Some("function_call_output" | "custom_tool_call_output") => {
                        flush_assistant(&mut pending_assistant, &mut messages);
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(String::from);
                        let content = parse_message_content(item.get("output"), Role::Tool)?;
                        messages.push(Message {
                            role: Role::Tool,
                            name: None,
                            tool_call_id: call_id,
                            reasoning_content: None,
                            content,
                            tool_calls: Vec::new(),
                        });
                    }
                    Some("item_reference") => {
                        return Err(Error::new_bad_request(
                            "item_reference is not supported: the server keeps no responses",
                        ));
                    }
                    Some(other) => {
                        return Err(Error::new_bad_request(format!(
                            "Unsupported Responses input item type: {other}"
                        )));
                    }
                    None => {
                        return Err(Error::new_bad_request(
                            "input item must have a type or role",
                        ));
                    }
                }
            }
            flush_assistant(&mut pending_assistant, &mut messages);
            items.len()
        }
        _ => {
            return Err(Error::new_bad_request(
                "input must be a non-empty string or array",
            ));
        }
    };

    if messages.is_empty() {
        return Err(Error::new_bad_request("input contains no messages"));
    }

    let settings = RequestSettings {
        sampling,
        max_tokens,
        model_id,
        stop,
        effort,
        is_streamed,
        include_usage: false,
        responses_tools,
    };
    Ok((
        Conversation {
            tools,
            messages,
            effort,
        },
        settings,
        item_count,
        tool_count,
        dropped_tool_count,
    ))
}

// ---- template rendering ----

/// Renders `conversation` into Inkling-2-Mini's prompt bytes and collects
/// decoded image byte buffers in prompt order. When `use_media_mark` is true,
/// each image slot carries `MEDIA_MARK_BYTE` for `tokenize_prompt`; when false,
/// it carries `IMAGE_TOKEN` for text inspection.
fn render_conversation(
    conversation: &Conversation,
    use_media_mark: bool,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut out = Vec::new();
    let mut images = Vec::new();
    let push_str = |buf: &mut Vec<u8>, text: &str| buf.extend_from_slice(text.as_bytes());

    if !conversation.tools.is_empty() {
        let specs: Vec<Value> = conversation
            .tools
            .iter()
            .map(|tool| {
                let mut spec = Map::new();
                spec.insert(
                    "description".into(),
                    Value::String(tool.description.clone()),
                );
                spec.insert("name".into(), Value::String(tool.name.clone()));
                spec.insert("parameters".into(), sort_json_keys(&tool.parameters));
                spec.insert("type".into(), Value::String(tool.tool_type.clone()));
                Value::Object(spec)
            })
            .collect();
        let json = serde_json::to_string(&Value::Array(specs)).expect("tool specs serialize");
        push_str(&mut out, TOKEN_SYSTEM);
        push_str(&mut out, TOOL_DECLARE_HEADER);
        push_str(&mut out, TOKEN_XML);
        push_str(&mut out, &json);
        push_str(&mut out, TOKEN_END_MESSAGE);
    }

    let emit_effort = |buf: &mut Vec<u8>| {
        push_str(buf, TOKEN_SYSTEM);
        push_str(buf, TOKEN_TEXT);
        push_str(buf, THINKING_EFFORT_PREFIX);
        push_str(buf, &format_effort(conversation.effort));
        push_str(buf, TOKEN_END_MESSAGE);
    };

    let mut effort_emitted = false;
    for message in &conversation.messages {
        if !effort_emitted && message.role != Role::System {
            emit_effort(&mut out);
            effort_emitted = true;
        }

        let mut header = String::from(message.role.token());
        if message.role == Role::Tool {
            if let Some(ref name) = message.name {
                header.push_str(name);
            } else if let Some(ref call_id) = message.tool_call_id {
                for prev in &conversation.messages {
                    if prev.role == Role::Assistant {
                        for tc in &prev.tool_calls {
                            if tc.id.as_deref() == Some(call_id.as_str()) {
                                header.push_str(&tc.name);
                            }
                        }
                    }
                }
            }
        } else if message.role == Role::Assistant
            && let Some(ref name) = message.name
        {
            header.push_str(name);
        }

        if message.role == Role::Assistant
            && let Some(ref reasoning) = message.reasoning_content
            && !reasoning.is_empty()
        {
            push_str(&mut out, &header);
            push_str(&mut out, TOKEN_THINKING);
            push_str(&mut out, reasoning);
            push_str(&mut out, TOKEN_END_MESSAGE);
        }

        match &message.content {
            MessageContent::Text(text) => {
                let has_assistant_extras = message.role == Role::Assistant
                    && (!message.tool_calls.is_empty()
                        || message
                            .reasoning_content
                            .as_deref()
                            .is_some_and(|r| !r.is_empty()));
                if !text.is_empty() || !has_assistant_extras {
                    push_str(&mut out, &header);
                    push_str(&mut out, TOKEN_TEXT);
                    push_str(&mut out, text);
                    push_str(&mut out, TOKEN_END_MESSAGE);
                }
            }
            MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text(text) => {
                            push_str(&mut out, &header);
                            push_str(&mut out, TOKEN_TEXT);
                            push_str(&mut out, text);
                            push_str(&mut out, TOKEN_END_MESSAGE);
                        }
                        ContentPart::Image(bytes) => {
                            push_str(&mut out, &header);
                            push_str(&mut out, TOKEN_IMAGE);
                            if use_media_mark {
                                out.push(MEDIA_MARK_BYTE);
                            } else {
                                push_str(&mut out, IMAGE_TOKEN);
                            }
                            push_str(&mut out, TOKEN_END_MESSAGE);
                            images.push(bytes.clone());
                        }
                    }
                }
            }
            MessageContent::None => {}
        }

        if message.role == Role::Assistant {
            for tc in &message.tool_calls {
                push_str(&mut out, TOKEN_MODEL);
                push_str(&mut out, &tc.name);
                push_str(&mut out, TOKEN_INVOKE_TOOL_JSON);
                push_str(&mut out, "{\"name\":");
                push_str(
                    &mut out,
                    &serde_json::to_string(&tc.name).expect("tool name serializes"),
                );
                push_str(&mut out, ",\"args\":{");
                let sorted_args: BTreeMap<&String, &Value> = tc.arguments.iter().collect();
                for (index, (key, value)) in sorted_args.into_iter().enumerate() {
                    if index > 0 {
                        out.push(b',');
                    }
                    push_str(
                        &mut out,
                        &serde_json::to_string(key).expect("arg key serializes"),
                    );
                    out.push(b':');
                    push_str(
                        &mut out,
                        &serde_json::to_string(value).expect("arg value serializes"),
                    );
                }
                push_str(&mut out, "}}");
                push_str(&mut out, TOKEN_END_MESSAGE);
            }
            push_str(&mut out, TOKEN_END_SAMPLING);
        }
    }

    if !effort_emitted {
        emit_effort(&mut out);
    }

    (out, images)
}

/// A prepared Inkling-2-Mini request ready for the worker.
pub struct PreparedRequest {
    /// The model's tokenizer.
    tokenizer: engine::Tokenizer,
    /// Rendered prompt bytes with `MEDIA_MARK_BYTE` at each image place.
    prompt: Vec<u8>,
    /// Decoded image file bytes in prompt order.
    images: Vec<Vec<u8>>,
    /// Settings read from the HTTP request.
    settings: RequestSettings,
    /// OpenAI API that received this request.
    pub api: Api,
    /// Whether the response streams as server-sent events.
    pub is_streamed: bool,
    /// Number of messages (Chat Completions) or input items (Responses).
    pub message_count: usize,
    /// Number of tools declared in the request.
    pub tool_count: usize,
    /// Number of declared Responses tools dropped as unsupported.
    pub dropped_tool_count: usize,
}

impl PreparedRequest {
    /// Tokenizes the rendered prompt and builds a worker request for model
    /// `model_index`.
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
        Ok(worker::Request {
            model_index,
            prompt,
            media: self.images.clone(),
            sampling: self.settings.sampling,
            max_tokens: self.settings.max_tokens,
            note,
        })
    }

    /// Describes the thinking effort for the request log.
    pub fn describe_thinking(&self) -> String {
        if self.settings.effort == EFFORT_NONE {
            "thinking off".into()
        } else {
            format!("thinking effort {}", format_effort(self.settings.effort))
        }
    }
}

/// Parses and renders an HTTP request body for Inkling-2-Mini.
pub fn prepare_request(
    body: &[u8],
    defaults: &Defaults,
    tokenizer: engine::Tokenizer,
    api: Api,
) -> Result<PreparedRequest, Error> {
    let Ok(Value::Object(root)) = serde_json::from_slice::<Value>(body) else {
        return Err(Error::new_bad_request("the body is not a JSON object"));
    };
    let (conversation, settings, message_count, tool_count, dropped_tool_count) = match api {
        Api::ChatCompletions => {
            let (conv, settings, messages, tools) = parse_chat_request(&root, defaults)?;
            (conv, settings, messages, tools, 0)
        }
        Api::Responses => parse_responses_request(&root, defaults)?,
    };
    let (prompt, images) = render_conversation(&conversation, true);
    let is_streamed = settings.is_streamed;
    Ok(PreparedRequest {
        tokenizer,
        prompt,
        images,
        settings,
        api,
        is_streamed,
        message_count,
        tool_count,
        dropped_tool_count,
    })
}

// ---- streaming block parser ----

/// Special token ids looked up from Inkling-2-Mini's tokenizer for output
/// parsing.
#[derive(Clone, Copy, Debug)]
struct SpecialIds {
    /// `<|message_model|>`
    model: Option<i32>,
    /// `<|content_thinking|>`
    thinking: Option<i32>,
    /// `<|content_text|>`
    text: Option<i32>,
    /// `<|content_tool_error|>`
    tool_error: Option<i32>,
    /// `<|content_invoke_tool_text|>`
    invoke_tool_text: Option<i32>,
    /// `<|content_invoke_tool_json|>`
    invoke_tool_json: Option<i32>,
    /// `<|end_message|>`
    end_message: Option<i32>,
    /// `<|content_model_end_sampling|>`
    end_sampling: Option<i32>,
    /// `<|endoftext|>`
    endoftext: Option<i32>,
}

impl SpecialIds {
    /// Looks up Inkling-2-Mini's delimiter token ids from `tokenizer`.
    fn from_tokenizer(tokenizer: engine::Tokenizer) -> SpecialIds {
        SpecialIds {
            model: tokenizer.find_added_token(TOKEN_MODEL),
            thinking: tokenizer.find_added_token(TOKEN_THINKING),
            text: tokenizer.find_added_token(TOKEN_TEXT),
            tool_error: tokenizer.find_added_token(TOKEN_TOOL_ERROR),
            invoke_tool_text: tokenizer.find_added_token(TOKEN_INVOKE_TOOL_TEXT),
            invoke_tool_json: tokenizer.find_added_token(TOKEN_INVOKE_TOOL_JSON),
            end_message: tokenizer.find_added_token(TOKEN_END_MESSAGE),
            end_sampling: tokenizer.find_added_token(TOKEN_END_SAMPLING),
            endoftext: tokenizer.find_added_token(TOKEN_ENDOFTEXT),
        }
    }
}

/// Incremental UTF-8 decoder that buffers incomplete multibyte byte sequences
/// across BPE token boundaries.
#[derive(Default)]
struct Utf8Buffer {
    /// Pending incomplete trailing bytes from previous tokens.
    pending: Vec<u8>,
}

impl Utf8Buffer {
    /// Appends raw token `bytes` and returns all newly completed UTF-8 text.
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        let valid =
                            std::str::from_utf8(&self.pending[..valid_up_to]).expect("valid UTF-8");
                        out.push_str(valid);
                        self.pending.drain(..valid_up_to);
                    }
                    if let Some(bad_len) = error.error_len() {
                        out.push('\u{FFFD}');
                        self.pending.drain(..bad_len);
                    } else {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Flushes any trailing incomplete UTF-8 sequence as replacement
    /// characters.
    fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

/// Structured event produced by parsing Inkling-2-Mini's generated tokens.
#[derive(Debug, PartialEq)]
enum ParsedEvent {
    /// Start of a `<|content_thinking|>` block.
    ThinkingStart,
    /// Incremental UTF-8 text inside the active thinking block.
    ThinkingDelta(String),
    /// End of the active thinking block.
    ThinkingEnd,
    /// Start of a text content block.
    TextStart,
    /// Incremental UTF-8 text inside the active text block.
    TextDelta(String),
    /// End of the active text block.
    TextEnd,
    /// Complete parsed tool invocation from `<|content_invoke_tool_json|>`.
    ToolCall {
        /// Generated unique call id (`call_...`).
        call_id: String,
        /// Tool name emitted by the model.
        name: String,
        /// Compact JSON-encoded arguments string.
        arguments: String,
        /// Parsed JSON value of `args` for custom tool input extraction.
        args_value: Value,
    },
}

/// Active block parsing state for one assistant reply.
enum BlockState {
    /// Between blocks or reading the optional header text between
    /// `<|message_model|>` and a content tag.
    Header {
        /// True after `<|message_model|>` until a content tag arrives.
        saw_model: bool,
        /// Raw bytes accumulated after `<|message_model|>`.
        header_bytes: Vec<u8>,
    },
    /// Inside `<|content_thinking|> ... <|end_message|>`.
    Thinking,
    /// Inside `<|content_text|>` (or tool-error/text fallback) until
    /// `<|end_message|>`.
    Text,
    /// Inside `<|content_invoke_tool_json|> ... <|end_message|>`.
    ToolJson {
        /// Optional header text read before `<|content_invoke_tool_json|>`.
        header_name: String,
        /// Raw JSON payload bytes accumulated for this invocation.
        json_bytes: Vec<u8>,
    },
    /// Parsing halted after a stop token or client stop sequence.
    Stopped,
}

/// Streaming token parser for Inkling-2-Mini's response template.
struct OutputParser {
    /// Delimiter token ids for this checkpoint.
    ids: SpecialIds,
    /// Active block state.
    state: BlockState,
    /// Multibyte UTF-8 accumulator for streaming thinking and text deltas.
    utf8: Utf8Buffer,
    /// Non-empty client stop sequences.
    stops: Vec<String>,
    /// Maximum byte length among `stops`.
    max_stop_bytes: usize,
    /// Trailing text held back while checking for partial stop sequences.
    stop_holdback: String,
    /// True once a client stop sequence matched.
    stopped_by_sequence: bool,
    /// Count of tokens generated inside `<|content_thinking|>` blocks.
    reasoning_tokens: usize,
}

impl OutputParser {
    /// Creates a new parser for `ids` and optional client `stops`.
    fn new(ids: SpecialIds, stops: Vec<String>) -> OutputParser {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let max_stop_bytes = stops.iter().map(String::len).max().unwrap_or(0);
        OutputParser {
            ids,
            state: BlockState::Header {
                saw_model: false,
                header_bytes: Vec::new(),
            },
            utf8: Utf8Buffer::default(),
            stops,
            max_stop_bytes,
            stop_holdback: String::new(),
            stopped_by_sequence: false,
            reasoning_tokens: 0,
        }
    }

    /// Feeds text into the active text block, applying client stop sequences
    /// and emitting safe text deltas into `out`.
    fn push_text(&mut self, text: &str, out: &mut Vec<ParsedEvent>) {
        if text.is_empty() {
            return;
        }
        if self.stops.is_empty() {
            out.push(ParsedEvent::TextDelta(text.to_string()));
            return;
        }
        self.stop_holdback.push_str(text);
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|stop| self.stop_holdback.find(stop.as_str()))
            .min()
        {
            let before = self.stop_holdback[..pos].to_string();
            self.stop_holdback.clear();
            if !before.is_empty() {
                out.push(ParsedEvent::TextDelta(before));
            }
            out.push(ParsedEvent::TextEnd);
            self.stopped_by_sequence = true;
            self.state = BlockState::Stopped;
            return;
        }
        let keep = self.max_stop_bytes.saturating_sub(1);
        if self.stop_holdback.len() > keep {
            let mut split = self.stop_holdback.len() - keep;
            while split > 0 && !self.stop_holdback.is_char_boundary(split) {
                split -= 1;
            }
            if split > 0 {
                let emit = self.stop_holdback[..split].to_string();
                self.stop_holdback.drain(..split);
                out.push(ParsedEvent::TextDelta(emit));
            }
        }
    }

    /// Flushes any buffered UTF-8 and stop-sequence holdback when closing the
    /// active text block.
    fn close_text(&mut self, out: &mut Vec<ParsedEvent>) {
        let tail = self.utf8.finish();
        if !tail.is_empty() {
            self.push_text(&tail, out);
            if matches!(self.state, BlockState::Stopped) {
                return;
            }
        }
        if !self.stop_holdback.is_empty() {
            let remaining = std::mem::take(&mut self.stop_holdback);
            out.push(ParsedEvent::TextDelta(remaining));
        }
        out.push(ParsedEvent::TextEnd);
    }

    /// Flushes any buffered UTF-8 and closes the active thinking block.
    fn close_thinking(&mut self, out: &mut Vec<ParsedEvent>) {
        let tail = self.utf8.finish();
        if !tail.is_empty() {
            out.push(ParsedEvent::ThinkingDelta(tail));
        }
        out.push(ParsedEvent::ThinkingEnd);
    }

    /// Parses a completed `<|content_invoke_tool_json|>` payload into a
    /// `ParsedEvent::ToolCall`, or emits raw text if the payload is not valid
    /// JSON.
    fn emit_tool_json(
        &mut self,
        header_name: String,
        json_bytes: Vec<u8>,
        out: &mut Vec<ParsedEvent>,
    ) {
        let raw = String::from_utf8_lossy(&json_bytes);
        if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(raw.trim()) {
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .or_else(|| (!header_name.is_empty()).then_some(header_name));
            if let Some(name) = name {
                let args_value = obj
                    .get("args")
                    .or_else(|| obj.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                let arguments = match &args_value {
                    Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
                };
                out.push(ParsedEvent::ToolCall {
                    call_id: generate_random_id("call_"),
                    name,
                    arguments,
                    args_value,
                });
                return;
            }
        }
        if !raw.is_empty() {
            out.push(ParsedEvent::TextStart);
            self.push_text(&raw, out);
            if !matches!(self.state, BlockState::Stopped) {
                self.close_text(out);
            }
        }
    }

    /// Processes one generated token `(token_id, bytes)` and appends resulting
    /// `ParsedEvent` items to `out`.
    fn feed_token(&mut self, token_id: i32, bytes: &[u8], out: &mut Vec<ParsedEvent>) {
        if matches!(self.state, BlockState::Stopped) {
            return;
        }
        if Some(token_id) == self.ids.end_sampling || Some(token_id) == self.ids.endoftext {
            self.finish(out);
            self.state = BlockState::Stopped;
            return;
        }
        if Some(token_id) == self.ids.end_message {
            match std::mem::replace(
                &mut self.state,
                BlockState::Header {
                    saw_model: false,
                    header_bytes: Vec::new(),
                },
            ) {
                BlockState::Thinking => self.close_thinking(out),
                BlockState::Text => self.close_text(out),
                BlockState::ToolJson {
                    header_name,
                    json_bytes,
                } => self.emit_tool_json(header_name, json_bytes, out),
                BlockState::Header { .. } | BlockState::Stopped => {}
            }
            return;
        }
        if Some(token_id) == self.ids.model {
            match std::mem::replace(
                &mut self.state,
                BlockState::Header {
                    saw_model: true,
                    header_bytes: Vec::new(),
                },
            ) {
                BlockState::Thinking => self.close_thinking(out),
                BlockState::Text => self.close_text(out),
                BlockState::ToolJson {
                    header_name,
                    json_bytes,
                } => self.emit_tool_json(header_name, json_bytes, out),
                BlockState::Header { .. } | BlockState::Stopped => {}
            }
            return;
        }
        if Some(token_id) == self.ids.thinking {
            if matches!(self.state, BlockState::Text) {
                self.close_text(out);
            }
            if !matches!(self.state, BlockState::Thinking) {
                self.state = BlockState::Thinking;
                out.push(ParsedEvent::ThinkingStart);
            }
            return;
        }
        if Some(token_id) == self.ids.text
            || Some(token_id) == self.ids.tool_error
            || Some(token_id) == self.ids.invoke_tool_text
        {
            if matches!(self.state, BlockState::Thinking) {
                self.close_thinking(out);
            }
            if !matches!(self.state, BlockState::Text) {
                self.state = BlockState::Text;
                out.push(ParsedEvent::TextStart);
            }
            return;
        }
        if Some(token_id) == self.ids.invoke_tool_json {
            let header_name = match std::mem::replace(
                &mut self.state,
                BlockState::Header {
                    saw_model: false,
                    header_bytes: Vec::new(),
                },
            ) {
                BlockState::Header { header_bytes, .. } => {
                    String::from_utf8_lossy(&header_bytes).trim().to_string()
                }
                BlockState::Thinking => {
                    self.close_thinking(out);
                    String::new()
                }
                BlockState::Text => {
                    self.close_text(out);
                    String::new()
                }
                BlockState::ToolJson {
                    header_name,
                    json_bytes,
                } => {
                    self.emit_tool_json(header_name, json_bytes, out);
                    String::new()
                }
                BlockState::Stopped => return,
            };
            self.state = BlockState::ToolJson {
                header_name,
                json_bytes: Vec::new(),
            };
            return;
        }

        match &mut self.state {
            BlockState::Thinking => {
                self.reasoning_tokens += 1;
                let text = self.utf8.push(bytes);
                if !text.is_empty() {
                    out.push(ParsedEvent::ThinkingDelta(text));
                }
            }
            BlockState::Text => {
                let text = self.utf8.push(bytes);
                self.push_text(&text, out);
            }
            BlockState::ToolJson { json_bytes, .. } => {
                json_bytes.extend_from_slice(bytes);
            }
            BlockState::Header {
                saw_model,
                header_bytes,
            } => {
                if *saw_model {
                    header_bytes.extend_from_slice(bytes);
                } else {
                    self.state = BlockState::Text;
                    out.push(ParsedEvent::TextStart);
                    let text = self.utf8.push(bytes);
                    self.push_text(&text, out);
                }
            }
            BlockState::Stopped => {}
        }
    }

    /// Closes any open block at the end of the token stream.
    fn finish(&mut self, out: &mut Vec<ParsedEvent>) {
        match std::mem::replace(&mut self.state, BlockState::Stopped) {
            BlockState::Thinking => self.close_thinking(out),
            BlockState::Text => self.close_text(out),
            BlockState::ToolJson {
                header_name,
                json_bytes,
            } => {
                // Only emit a complete JSON object if the stream ended cleanly
                // or truncated right after the closing brace.
                if serde_json::from_slice::<Value>(&json_bytes).is_ok() {
                    self.emit_tool_json(header_name, json_bytes, out);
                }
            }
            BlockState::Header { header_bytes, .. } => {
                if !header_bytes.is_empty() {
                    let text = String::from_utf8_lossy(&header_bytes).into_owned();
                    out.push(ParsedEvent::TextStart);
                    self.push_text(&text, out);
                    if !matches!(self.state, BlockState::Stopped) {
                        self.close_text(out);
                    }
                }
            }
            BlockState::Stopped => {}
        }
    }
}

// ---- response builders and SSE formatters ----

/// Reads the current Unix timestamp in seconds.
fn read_unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Logs the number of tool calls produced by request `request_id`.
fn log_tool_calls(request_id: u64, tool_call_count: usize) {
    if tool_call_count > 0 {
        engine::log(&format!(
            "req {request_id}: {tool_call_count} tool call{}",
            if tool_call_count == 1 { "" } else { "s" }
        ));
    }
}

/// Extracts the freeform string input for a Responses `custom_tool_call` from
/// its parsed JSON arguments.
fn extract_custom_tool_input(args_value: &Value, arguments_json: &str) -> String {
    if let Some(obj) = args_value.as_object() {
        if let Some(input) = obj.get("input").and_then(Value::as_str) {
            return input.to_string();
        }
        if obj.len() == 1
            && let Some(val) = obj.values().next().and_then(Value::as_str)
        {
            return val.to_string();
        }
    }
    arguments_json.to_string()
}

/// Builds a Chat Completions `usage` JSON object.
fn chat_usage_json(prompt_tokens: usize, reused_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": reused_tokens},
        "prompt_cache_hit_tokens": reused_tokens,
    })
}

/// Builds a Responses API `usage` JSON object.
fn responses_usage_json(
    prompt_tokens: usize,
    reused_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
) -> Value {
    json!({
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": reused_tokens},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

/// Formats a Responses API `response.failed` server-sent event.
fn format_responses_failure_event(
    error_type: &str,
    message: &str,
    response_id: &str,
    model_id: &str,
    created_at: u64,
    sequence_number: usize,
) -> String {
    let event = json!({
        "type": "response.failed",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "failed",
            "model": model_id,
            "output": [],
            "error": {"code": error_type, "message": message},
            "incomplete_details": null,
        },
    });
    format!("event: response.failed\ndata: {event}\n\n")
}

/// Formats one Chat Completions streaming chunk as an SSE `data:` frame.
fn format_chat_chunk_sse(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
    usage: Option<Value>,
) -> String {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    format!("data: {chunk}\n\n")
}

/// Mutable state for streaming Responses API SSE frames.
struct ResponsesStreamWriter {
    /// Response id (`resp_...`).
    response_id: String,
    /// Model id echoed in response payloads.
    model_id: String,
    /// Response creation Unix timestamp.
    created_at: u64,
    /// Monotonic SSE sequence number.
    sequence: usize,
    /// Tool round-trip metadata from the request.
    tool_meta: ResponsesToolMeta,
    /// Completed output items in generation order.
    output_items: Vec<Value>,
    /// Active open reasoning or message item `(item_id, output_index, is_thinking, text)`.
    active_text_item: Option<(String, usize, bool, String)>,
    /// Number of tool calls emitted so far.
    tool_call_count: usize,
}

impl ResponsesStreamWriter {
    /// Creates a new writer for one Responses stream.
    fn new(
        response_id: String,
        model_id: String,
        created_at: u64,
        tool_meta: ResponsesToolMeta,
    ) -> ResponsesStreamWriter {
        ResponsesStreamWriter {
            response_id,
            model_id,
            created_at,
            sequence: 0,
            tool_meta,
            output_items: Vec::new(),
            active_text_item: None,
            tool_call_count: 0,
        }
    }

    /// Formats one Responses SSE event with the next sequence number.
    fn emit_event(&mut self, event_type: &str, mut body: Value) -> String {
        let seq = self.sequence;
        self.sequence += 1;
        body["type"] = Value::String(event_type.to_string());
        body["sequence_number"] = json!(seq);
        format!("event: {event_type}\ndata: {body}\n\n")
    }

    /// Builds a Responses API response envelope JSON object.
    fn build_response_object(
        &self,
        status: &str,
        incomplete_reason: Option<&str>,
        usage: Option<Value>,
    ) -> Value {
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": null,
            "incomplete_details": incomplete_reason.map(|r| json!({"reason": r})),
            "model": self.model_id,
            "output": self.output_items,
            "usage": usage,
        })
    }

    /// Emits `response.created` and `response.in_progress`.
    fn start(&mut self, frames: &mut Vec<String>) {
        let resp = self.build_response_object("in_progress", None, None);
        frames.push(self.emit_event("response.created", json!({"response": resp.clone()})));
        frames.push(self.emit_event("response.in_progress", json!({"response": resp})));
    }

    /// Translates one `ParsedEvent` into Responses SSE frames.
    fn handle_parsed(&mut self, event: ParsedEvent, frames: &mut Vec<String>) {
        match event {
            ParsedEvent::ThinkingStart => {
                let item_id = generate_random_id("rs_");
                let output_index = self.output_items.len();
                let item = json!({
                    "id": item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                    "content": [],
                });
                frames.push(self.emit_event(
                    "response.output_item.added",
                    json!({"output_index": output_index, "item": item}),
                ));
                frames.push(self.emit_event(
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": ""},
                    }),
                ));
                self.active_text_item = Some((item_id, output_index, true, String::new()));
            }
            ParsedEvent::ThinkingDelta(delta) => {
                if let Some((item_id, output_index, true, text)) = self.active_text_item.as_mut() {
                    text.push_str(&delta);
                    let (item_id, output_index) = (item_id.clone(), *output_index);
                    frames.push(self.emit_event(
                        "response.reasoning_text.delta",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": delta,
                        }),
                    ));
                }
            }
            ParsedEvent::ThinkingEnd => {
                if let Some((item_id, output_index, true, text)) = self.active_text_item.take() {
                    frames.push(self.emit_event(
                        "response.reasoning_text.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "text": text,
                        }),
                    ));
                    frames.push(self.emit_event(
                        "response.content_part.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": {"type": "reasoning_text", "text": text},
                        }),
                    ));
                    let completed = json!({
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": text}],
                    });
                    self.output_items.push(completed.clone());
                    frames.push(self.emit_event(
                        "response.output_item.done",
                        json!({"output_index": output_index, "item": completed}),
                    ));
                }
            }
            ParsedEvent::TextStart => {
                let item_id = generate_random_id("msg_");
                let output_index = self.output_items.len();
                let item = json!({
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                });
                frames.push(self.emit_event(
                    "response.output_item.added",
                    json!({"output_index": output_index, "item": item}),
                ));
                frames.push(self.emit_event(
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }),
                ));
                self.active_text_item = Some((item_id, output_index, false, String::new()));
            }
            ParsedEvent::TextDelta(delta) => {
                if let Some((item_id, output_index, false, text)) = self.active_text_item.as_mut() {
                    text.push_str(&delta);
                    let (item_id, output_index) = (item_id.clone(), *output_index);
                    frames.push(self.emit_event(
                        "response.output_text.delta",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": delta,
                        }),
                    ));
                }
            }
            ParsedEvent::TextEnd => {
                if let Some((item_id, output_index, false, text)) = self.active_text_item.take() {
                    frames.push(self.emit_event(
                        "response.output_text.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "text": text,
                        }),
                    ));
                    frames.push(self.emit_event(
                        "response.content_part.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": text, "annotations": []},
                        }),
                    ));
                    let completed = json!({
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    });
                    self.output_items.push(completed.clone());
                    frames.push(self.emit_event(
                        "response.output_item.done",
                        json!({"output_index": output_index, "item": completed}),
                    ));
                }
            }
            ParsedEvent::ToolCall {
                call_id,
                name,
                arguments,
                args_value,
            } => {
                self.tool_call_count += 1;
                let output_index = self.output_items.len();
                if self.tool_meta.custom_tools.contains(&name) {
                    let item_id = generate_random_id("ctc_");
                    let input = extract_custom_tool_input(&args_value, &arguments);
                    let added = json!({
                        "id": item_id,
                        "type": "custom_tool_call",
                        "status": "in_progress",
                        "call_id": call_id,
                        "name": name,
                        "input": "",
                    });
                    frames.push(self.emit_event(
                        "response.output_item.added",
                        json!({"output_index": output_index, "item": added}),
                    ));
                    frames.push(self.emit_event(
                        "response.custom_tool_call_input.delta",
                        json!({"item_id": item_id, "output_index": output_index, "delta": input}),
                    ));
                    frames.push(self.emit_event(
                        "response.custom_tool_call_input.done",
                        json!({"item_id": item_id, "output_index": output_index, "input": input}),
                    ));
                    let completed = json!({
                        "id": item_id,
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": name,
                        "input": input,
                    });
                    self.output_items.push(completed.clone());
                    frames.push(self.emit_event(
                        "response.output_item.done",
                        json!({"output_index": output_index, "item": completed}),
                    ));
                } else {
                    let item_id = generate_random_id("fc_");
                    let (fn_name, namespace) = match self.tool_meta.namespaced_tools.get(&name) {
                        Some((ns, bare)) => (bare.clone(), Some(ns.clone())),
                        None => (name.clone(), None),
                    };
                    let mut added = json!({
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": call_id,
                        "name": fn_name,
                        "arguments": "",
                    });
                    if let Some(ref ns) = namespace {
                        added["namespace"] = Value::String(ns.clone());
                    }
                    frames.push(self.emit_event(
                        "response.output_item.added",
                        json!({"output_index": output_index, "item": added}),
                    ));
                    frames.push(self.emit_event(
                        "response.function_call_arguments.delta",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": arguments,
                        }),
                    ));
                    frames.push(self.emit_event(
                        "response.function_call_arguments.done",
                        json!({
                            "item_id": item_id,
                            "output_index": output_index,
                            "name": fn_name,
                            "arguments": arguments,
                        }),
                    ));
                    let mut completed = json!({
                        "id": item_id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": fn_name,
                        "arguments": arguments,
                    });
                    if let Some(ns) = namespace {
                        completed["namespace"] = Value::String(ns);
                    }
                    self.output_items.push(completed.clone());
                    frames.push(self.emit_event(
                        "response.output_item.done",
                        json!({"output_index": output_index, "item": completed}),
                    ));
                }
            }
        }
    }

    /// Emits the terminal `response.completed` or `response.incomplete` event.
    fn finish(
        &mut self,
        incomplete: bool,
        prompt_tokens: usize,
        reused_tokens: usize,
        completion_tokens: usize,
        reasoning_tokens: usize,
        frames: &mut Vec<String>,
    ) {
        let usage = responses_usage_json(
            prompt_tokens,
            reused_tokens,
            completion_tokens,
            reasoning_tokens,
        );
        if incomplete {
            let resp =
                self.build_response_object("incomplete", Some("max_output_tokens"), Some(usage));
            frames.push(self.emit_event("response.incomplete", json!({"response": resp})));
        } else {
            let resp = self.build_response_object("completed", None, Some(usage));
            frames.push(self.emit_event("response.completed", json!({"response": resp})));
        }
    }
}

/// Streams an Inkling-2-Mini response as server-sent events for either API.
pub fn stream_response(
    request: PreparedRequest,
    cancel: worker::CancelGuard,
    events: UnboundedReceiver<Event>,
    keepalive: Duration,
) -> Response {
    match request.api {
        Api::ChatCompletions => stream_chat_response(request, cancel, events, keepalive),
        Api::Responses => stream_responses_api(request, cancel, events, keepalive),
    }
}

/// Formats one `ParsedEvent` as a Chat Completions SSE chunk, updating
/// `sent_role` and `tool_call_count` when it emits a chunk.
fn format_chat_parsed_event(
    item: ParsedEvent,
    response_id: &str,
    created: u64,
    model_id: &str,
    sent_role: &mut bool,
    tool_call_count: &mut usize,
) -> Option<String> {
    let role_field = (!*sent_role).then_some("assistant");
    let mut delta = match item {
        ParsedEvent::ThinkingDelta(text) => json!({"reasoning_content": text}),
        ParsedEvent::TextDelta(text) => json!({"content": text}),
        ParsedEvent::ToolCall {
            call_id,
            name,
            arguments,
            ..
        } => {
            let index = *tool_call_count;
            *tool_call_count += 1;
            json!({
                "tool_calls": [{
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }]
            })
        }
        ParsedEvent::ThinkingStart
        | ParsedEvent::ThinkingEnd
        | ParsedEvent::TextStart
        | ParsedEvent::TextEnd => return None,
    };
    *sent_role = true;
    if let Some(role) = role_field {
        delta["role"] = Value::String(role.into());
    }
    Some(format_chat_chunk_sse(
        response_id,
        created,
        model_id,
        delta,
        None,
        None,
    ))
}

/// Streams Chat Completions SSE chunks for an Inkling-2-Mini request.
fn stream_chat_response(
    request: PreparedRequest,
    cancel: worker::CancelGuard,
    mut events: UnboundedReceiver<Event>,
    keepalive: Duration,
) -> Response {
    let request_id = cancel.id();
    let response_id = generate_random_id("chatcmpl-");
    let created = read_unix_time();
    let model_id = request.settings.model_id.clone();
    let include_usage = request.settings.include_usage;
    let tokenizer = request.tokenizer;
    let ids = SpecialIds::from_tokenizer(tokenizer);
    let mut parser = OutputParser::new(ids, request.settings.stop);

    let body = async_stream::stream! {
        let _cancel = cancel;
        let first = loop {
            match tokio::time::timeout(keepalive, events.recv()).await {
                Ok(event) => break event,
                Err(_) => yield Ok::<String, std::io::Error>(": waiting\n\n".into()),
            }
        };
        let first = match first {
            Some(Event::Failed(message)) => {
                yield Ok(format!("data: {}\n\ndata: [DONE]\n\n", format_error_json("server_error", &message)));
                return;
            }
            Some(Event::Refused(message)) => {
                yield Ok(format!("data: {}\n\ndata: [DONE]\n\n", format_error_json("invalid_request_error", &message)));
                return;
            }
            Some(event) => event,
            None => return,
        };

        let mut prompt_tokens = 0usize;
        let mut reused_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut sent_role = false;
        let mut tool_call_count = 0usize;
        let mut finish_reason = FinishReason::StopToken;
        let mut ended_by_shutdown = false;
        let mut parsed = Vec::new();
        let mut next = Some(first);

        loop {
            let event = match next.take() {
                Some(event) => event,
                None => match events.recv().await {
                    Some(event) => event,
                    None => break,
                },
            };
            match event {
                Event::DecodingStarted { prompt_tokens: pt, reused_tokens: rt } => {
                    prompt_tokens = pt;
                    reused_tokens = rt;
                }
                Event::Tokens(token_ids) => {
                    completion_tokens += token_ids.len();
                    parsed.clear();
                    for id in token_ids {
                        let bytes = tokenizer.token_bytes(id as i32);
                        parser.feed_token(id as i32, bytes, &mut parsed);
                    }
                    for item in parsed.drain(..) {
                        if let Some(frame) = format_chat_parsed_event(
                            item,
                            &response_id,
                            created,
                            &model_id,
                            &mut sent_role,
                            &mut tool_call_count,
                        ) {
                            yield Ok(frame);
                        }
                    }
                    if parser.stopped_by_sequence {
                        finish_reason = FinishReason::StopToken;
                        break;
                    }
                }
                Event::Done { reason } => {
                    if reason == FinishReason::ServerStop {
                        ended_by_shutdown = true;
                        break;
                    }
                    if reason == FinishReason::StopToken {
                        completion_tokens += 1;
                    }
                    finish_reason = reason;
                    break;
                }
                _ => break,
            }
        }

        if ended_by_shutdown {
            yield Ok(format!("data: {}\n\n", format_error_json("server_error", "the server is stopping")));
            return;
        }

        parsed.clear();
        parser.finish(&mut parsed);
        for item in parsed.drain(..) {
            if let Some(frame) = format_chat_parsed_event(
                item,
                &response_id,
                created,
                &model_id,
                &mut sent_role,
                &mut tool_call_count,
            ) {
                yield Ok(frame);
            }
        }

        let reason_str = if finish_reason == FinishReason::Length && !parser.stopped_by_sequence {
            "length"
        } else if tool_call_count > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        let usage = include_usage.then(|| chat_usage_json(prompt_tokens, reused_tokens, completion_tokens));
        yield Ok(format_chat_chunk_sse(&response_id, created, &model_id, json!({}), Some(reason_str), usage));
        log_tool_calls(request_id, tool_call_count);
        yield Ok("data: [DONE]\n\n".into());
    };

    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

/// Streams Responses API SSE events for an Inkling-2-Mini request.
fn stream_responses_api(
    request: PreparedRequest,
    cancel: worker::CancelGuard,
    mut events: UnboundedReceiver<Event>,
    keepalive: Duration,
) -> Response {
    let request_id = cancel.id();
    let response_id = generate_random_id("resp_");
    let created_at = read_unix_time();
    let model_id = request.settings.model_id.clone();
    let tokenizer = request.tokenizer;
    let ids = SpecialIds::from_tokenizer(tokenizer);
    let mut parser = OutputParser::new(ids, request.settings.stop);
    let mut writer = ResponsesStreamWriter::new(
        response_id.clone(),
        model_id.clone(),
        created_at,
        request.settings.responses_tools,
    );

    let body = async_stream::stream! {
        let _cancel = cancel;
        let first = loop {
            match tokio::time::timeout(keepalive, events.recv()).await {
                Ok(event) => break event,
                Err(_) => yield Ok::<String, std::io::Error>(": waiting\n\n".into()),
            }
        };
        let first = match first {
            Some(Event::Failed(message)) => {
                yield Ok(format_responses_failure_event("server_error", &message, &response_id, &model_id, created_at, 0));
                return;
            }
            Some(Event::Refused(message)) => {
                yield Ok(format_responses_failure_event("invalid_request_error", &message, &response_id, &model_id, created_at, 0));
                return;
            }
            Some(event) => event,
            None => return,
        };

        let mut prompt_tokens = 0usize;
        let mut reused_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = FinishReason::StopToken;
        let mut ended_by_shutdown = false;
        let mut started = false;
        let mut parsed = Vec::new();
        let mut frames = Vec::new();
        let mut next = Some(first);

        loop {
            let event = match next.take() {
                Some(event) => event,
                None => match events.recv().await {
                    Some(event) => event,
                    None => break,
                },
            };
            match event {
                Event::DecodingStarted { prompt_tokens: pt, reused_tokens: rt } => {
                    prompt_tokens = pt;
                    reused_tokens = rt;
                    if !started {
                        started = true;
                        writer.start(&mut frames);
                        for frame in frames.drain(..) {
                            yield Ok(frame);
                        }
                    }
                }
                Event::Tokens(token_ids) => {
                    if !started {
                        started = true;
                        writer.start(&mut frames);
                    }
                    completion_tokens += token_ids.len();
                    parsed.clear();
                    for id in token_ids {
                        let bytes = tokenizer.token_bytes(id as i32);
                        parser.feed_token(id as i32, bytes, &mut parsed);
                    }
                    for item in parsed.drain(..) {
                        writer.handle_parsed(item, &mut frames);
                    }
                    for frame in frames.drain(..) {
                        yield Ok(frame);
                    }
                    if parser.stopped_by_sequence {
                        finish_reason = FinishReason::StopToken;
                        break;
                    }
                }
                Event::Done { reason } => {
                    if reason == FinishReason::ServerStop {
                        ended_by_shutdown = true;
                        break;
                    }
                    if reason == FinishReason::StopToken {
                        completion_tokens += 1;
                    }
                    finish_reason = reason;
                    break;
                }
                _ => break,
            }
        }

        if ended_by_shutdown {
            yield Ok(format_responses_failure_event(
                "server_error",
                "the server is stopping",
                &response_id,
                &model_id,
                created_at,
                writer.sequence,
            ));
            return;
        }

        if !started {
            writer.start(&mut frames);
        }
        parsed.clear();
        parser.finish(&mut parsed);
        for item in parsed.drain(..) {
            writer.handle_parsed(item, &mut frames);
        }
        let incomplete = finish_reason == FinishReason::Length && !parser.stopped_by_sequence;
        writer.finish(
            incomplete,
            prompt_tokens,
            reused_tokens,
            completion_tokens,
            parser.reasoning_tokens,
            &mut frames,
        );
        for frame in frames.drain(..) {
            yield Ok(frame);
        }
        log_tool_calls(request_id, writer.tool_call_count);
    };

    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

/// Collects a full non-streaming Inkling-2-Mini response for either API.
pub async fn collect_response(
    request: PreparedRequest,
    cancel: worker::CancelGuard,
    mut events: UnboundedReceiver<Event>,
) -> Response {
    let request_id = cancel.id();
    let first = match events.recv().await {
        Some(Event::Failed(message)) => {
            return build_error_response(StatusCode::SERVICE_UNAVAILABLE, "server_error", &message);
        }
        Some(Event::Refused(message)) => {
            return build_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &message,
            );
        }
        Some(event) => event,
        None => {
            return build_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "the request ended without an answer",
            );
        }
    };

    let tokenizer = request.tokenizer;
    let ids = SpecialIds::from_tokenizer(tokenizer);
    let mut parser = OutputParser::new(ids, request.settings.stop);
    let mut prompt_tokens = 0usize;
    let mut reused_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish_reason = FinishReason::StopToken;
    let mut parsed = Vec::new();
    let mut next = Some(first);

    loop {
        let event = match next.take() {
            Some(event) => event,
            None => match events.recv().await {
                Some(event) => event,
                None => break,
            },
        };
        match event {
            Event::DecodingStarted {
                prompt_tokens: pt,
                reused_tokens: rt,
            } => {
                prompt_tokens = pt;
                reused_tokens = rt;
            }
            Event::Tokens(token_ids) => {
                completion_tokens += token_ids.len();
                for id in token_ids {
                    let bytes = tokenizer.token_bytes(id as i32);
                    parser.feed_token(id as i32, bytes, &mut parsed);
                }
                if parser.stopped_by_sequence {
                    finish_reason = FinishReason::StopToken;
                    break;
                }
            }
            Event::Done { reason } => {
                if reason == FinishReason::ServerStop {
                    return build_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "server_error",
                        "the server is stopping",
                    );
                }
                if reason == FinishReason::StopToken {
                    completion_tokens += 1;
                }
                finish_reason = reason;
                break;
            }
            _ => break,
        }
    }
    parser.finish(&mut parsed);

    let created = read_unix_time();
    let model_id = request.settings.model_id;
    let incomplete = finish_reason == FinishReason::Length && !parser.stopped_by_sequence;

    let body = match request.api {
        Api::ChatCompletions => {
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut tool_calls = Vec::new();
            for item in parsed {
                match item {
                    ParsedEvent::ThinkingDelta(text) => reasoning.push_str(&text),
                    ParsedEvent::TextDelta(text) => content.push_str(&text),
                    ParsedEvent::ToolCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    } => {
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }));
                    }
                    ParsedEvent::ThinkingStart
                    | ParsedEvent::ThinkingEnd
                    | ParsedEvent::TextStart
                    | ParsedEvent::TextEnd => {}
                }
            }
            log_tool_calls(request_id, tool_calls.len());
            let reason_str = if incomplete {
                "length"
            } else if !tool_calls.is_empty() {
                "tool_calls"
            } else {
                "stop"
            };
            let mut message = json!({
                "role": "assistant",
                "content": if content.is_empty() && !tool_calls.is_empty() {
                    Value::Null
                } else {
                    Value::String(content)
                },
            });
            if !reasoning.is_empty() {
                message["reasoning_content"] = Value::String(reasoning);
            }
            if !tool_calls.is_empty() {
                message["tool_calls"] = Value::Array(tool_calls);
            }
            json!({
                "id": generate_random_id("chatcmpl-"),
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": reason_str,
                }],
                "usage": chat_usage_json(prompt_tokens, reused_tokens, completion_tokens),
            })
        }
        Api::Responses => {
            let mut writer = ResponsesStreamWriter::new(
                generate_random_id("resp_"),
                model_id,
                created,
                request.settings.responses_tools,
            );
            let mut ignored_frames = Vec::new();
            for item in parsed {
                writer.handle_parsed(item, &mut ignored_frames);
            }
            log_tool_calls(request_id, writer.tool_call_count);
            let usage = responses_usage_json(
                prompt_tokens,
                reused_tokens,
                completion_tokens,
                parser.reasoning_tokens,
            );
            writer.build_response_object(
                if incomplete {
                    "incomplete"
                } else {
                    "completed"
                },
                incomplete.then_some("max_output_tokens"),
                Some(usage),
            )
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns default settings for unit tests.
    fn test_defaults() -> Defaults {
        Defaults::new(DEFAULT_MODEL_ID.into())
    }

    /// Renders a Chat Completions JSON request as UTF-8 text (with `IMAGE_TOKEN`
    /// in place of `MEDIA_MARK_BYTE`) and returns the prompt and image buffers.
    fn render_chat_text(request: Value) -> Result<(String, Vec<Vec<u8>>), Error> {
        let Value::Object(root) = request else {
            panic!("expected object");
        };
        let (conv, _, _, _) = parse_chat_request(&root, &test_defaults())?;
        let (bytes, images) = render_conversation(&conv, false);
        Ok((String::from_utf8(bytes).expect("UTF-8"), images))
    }

    /// Rendered output of a Responses API test request.
    #[derive(Debug)]
    struct RenderedResponses {
        /// Prompt text with `IMAGE_TOKEN` in place of `MEDIA_MARK_BYTE`.
        prompt: String,
        /// Count of declared tools in the request.
        tool_count: usize,
        /// Count of unsupported tools dropped from the prompt.
        dropped_tool_count: usize,
        /// Custom and namespaced tool metadata kept for output conversion.
        meta: ResponsesToolMeta,
    }

    /// Renders a Responses JSON request as UTF-8 text and returns the prompt,
    /// tool counts and tool metadata.
    fn render_responses_text(request: Value) -> Result<RenderedResponses, Error> {
        let Value::Object(root) = request else {
            panic!("expected object");
        };
        let (conv, settings, _, tool_count, dropped_tool_count) =
            parse_responses_request(&root, &test_defaults())?;
        let (bytes, _) = render_conversation(&conv, false);
        Ok(RenderedResponses {
            prompt: String::from_utf8(bytes).expect("UTF-8"),
            tool_count,
            dropped_tool_count,
            meta: settings.responses_tools,
        })
    }

    #[test]
    fn role_tokens_map_to_their_roles() {
        assert_eq!(find_role_of_token(TOKEN_SYSTEM), Some("system"));
        assert_eq!(find_role_of_token(TOKEN_USER), Some("user"));
        assert_eq!(find_role_of_token(TOKEN_MODEL), Some("assistant"));
        assert_eq!(find_role_of_token(TOKEN_TOOL), Some("tool"));
        assert_eq!(find_role_of_token(TOKEN_THINKING), None);
        assert_eq!(find_role_of_token(TOKEN_END_SAMPLING), None);
    }

    #[test]
    fn chat_template_renders_system_effort_and_user_turn() {
        let (prompt, images) = render_chat_text(json!({
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
            "reasoning_effort": "medium",
        }))
        .unwrap();
        assert!(images.is_empty());
        assert_eq!(
            prompt,
            concat!(
                "<|message_system|><|content_text|>Be concise.<|end_message|>",
                "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>",
                "<|message_user|><|content_text|>Hello<|end_message|>",
            )
        );
    }

    #[test]
    fn effort_none_and_thinking_disabled_render_zero() {
        for req in [
            json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none"}),
            json!({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": 0.0}),
            json!({"messages": [{"role": "user", "content": "Hi"}], "thinking": false}),
            json!({"messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}}),
        ] {
            let (prompt, _) = render_chat_text(req).unwrap();
            assert!(
                prompt.starts_with(
                    "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                ),
                "{prompt}"
            );
        }
        assert!(
            render_chat_text(json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": 1.0,
            }))
            .is_err()
        );
        assert!(
            render_chat_text(json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "ultra",
            }))
            .is_err()
        );
    }

    #[test]
    fn chat_template_renders_tools_sorted_keys_and_tool_round_trip() {
        let (prompt, _) = render_chat_text(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Lookup weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Call get_weather.",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"unit\":\"c\",\"city\":\"Paris\"}",
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "18C sunny",
                },
            ],
        }))
        .unwrap();
        assert_eq!(
            prompt,
            concat!(
                "<|message_system|>tool_declare<|content_xml|>",
                r#"[{"description":"Lookup weather","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]"#,
                "<|end_message|>",
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
                "<|message_user|><|content_text|>Weather in Paris?<|end_message|>",
                "<|message_model|><|content_thinking|>Call get_weather.<|end_message|>",
                "<|message_model|>get_weather<|content_invoke_tool_json|>",
                r#"{"name":"get_weather","args":{"city":"Paris","unit":"c"}}"#,
                "<|end_message|><|content_model_end_sampling|>",
                "<|message_tool|>get_weather<|content_text|>18C sunny<|end_message|>",
            )
        );
    }

    #[test]
    fn multimodal_images_render_for_user_and_tool_and_reject_audio_or_remote_urls() {
        // "aGk=" decodes to "hi".
        let (prompt, images) = render_chat_text(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                ],
            }],
        }))
        .unwrap();
        assert_eq!(images, vec![b"hi".to_vec()]);
        assert_eq!(
            prompt,
            concat!(
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
                "<|message_user|><|content_text|>Look:<|end_message|>",
                "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
            )
        );

        let err = render_chat_text(json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
            }],
        }))
        .unwrap_err();
        assert!(err.message.contains("only data: URLs"), "{}", err.message);

        let err = render_chat_text(json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {}}],
            }],
        }))
        .unwrap_err();
        assert!(err.message.contains("audio"), "{}", err.message);

        let err = render_chat_text(json!({
            "messages": [{
                "role": "system",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,aGk="}],
            }, {"role": "user", "content": "hi"}],
        }))
        .unwrap_err();
        assert!(err.message.contains("user and tool"), "{}", err.message);
    }

    #[test]
    fn codex_responses_request_renders_namespaces_custom_tools_and_drops_web_search() {
        let rendered = render_responses_text(json!({
            "model": "inkling-2-mini",
            "stream": true,
            "instructions": "You are a coding agent.",
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
                {
                    "type": "namespace",
                    "name": "web",
                    "description": "Web tools.",
                    "tools": [{
                        "type": "function",
                        "name": "run",
                        "description": "Search.",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }],
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                },
                {"type": "web_search", "external_web_access": false},
            ],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix main.rs"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Patch main.rs."}]},
                {"type": "custom_tool_call", "call_id": "call_0", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"},
                {"type": "custom_tool_call_output", "call_id": "call_0", "output": "patched"},
            ],
        }))
        .unwrap();
        assert_eq!((rendered.tool_count, rendered.dropped_tool_count), (4, 1));
        assert!(rendered.meta.custom_tools.contains("apply_patch"));
        assert_eq!(
            rendered.meta.namespaced_tools.get("web::run"),
            Some(&("web".into(), "run".into()))
        );
        for expected in [
            "You are a coding agent.",
            "Thinking effort level: 0.99",
            "exec_command",
            "web::run",
            "apply_patch",
            "Patch main.rs.",
            "*** Begin Patch",
            "<|message_tool|>apply_patch<|content_text|>patched<|end_message|>",
        ] {
            assert!(
                rendered.prompt.contains(expected),
                "missing {expected} in:\n{}",
                rendered.prompt
            );
        }
        assert!(
            !rendered.prompt.contains("web_search"),
            "{}",
            rendered.prompt
        );
    }

    #[test]
    fn responses_rejects_stored_conversation_references() {
        for key in ["previous_response_id", "conversation"] {
            let err = render_responses_text(json!({"input": "hi", key: "resp_1"})).unwrap_err();
            assert!(err.message.contains(key), "{}", err.message);
        }
    }

    #[test]
    fn output_parser_streams_thinking_text_multibyte_utf8_and_tool_calls() {
        let ids = SpecialIds {
            model: Some(200_001),
            thinking: Some(200_008),
            text: Some(200_004),
            tool_error: Some(200_022),
            invoke_tool_text: Some(200_057),
            invoke_tool_json: Some(200_049),
            end_message: Some(200_010),
            end_sampling: Some(200_006),
            endoftext: Some(199_999),
        };
        let mut parser = OutputParser::new(ids, Vec::new());
        let mut events = Vec::new();
        let ni_hao = "你好".as_bytes();
        // Feed thinking, split multibyte UTF-8 text, and one JSON tool call.
        let tokens: Vec<(i32, &[u8])> = vec![
            (200_001, b"<|message_model|>"),
            (200_008, b"<|content_thinking|>"),
            (10, b"Plan "),
            (11, b"first."),
            (200_010, b"<|end_message|>"),
            (200_001, b"<|message_model|>"),
            (200_004, b"<|content_text|>"),
            (12, &ni_hao[..2]),
            (13, &ni_hao[2..4]),
            (14, &ni_hao[4..]),
            (200_010, b"<|end_message|>"),
            (200_001, b"<|message_model|>"),
            (15, b"apply_patch"),
            (200_049, b"<|content_invoke_tool_json|>"),
            (16, br#"{"name":"apply_patch","args":{"input":"diff"}}"#),
            (200_010, b"<|end_message|>"),
        ];
        for (id, bytes) in tokens {
            parser.feed_token(id, bytes, &mut events);
        }
        parser.finish(&mut events);
        assert_eq!(parser.reasoning_tokens, 2);
        let thinking: String = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::ThinkingDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "Plan first.");
        assert_eq!(text, "你好");
        let calls: Vec<(&str, &str)> = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::ToolCall {
                    name, arguments, ..
                } => Some((name.as_str(), arguments.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec![("apply_patch", r#"{"input":"diff"}"#)]);
    }

    #[test]
    fn output_parser_applies_client_stop_sequences() {
        let ids = SpecialIds {
            model: Some(200_001),
            thinking: Some(200_008),
            text: Some(200_004),
            tool_error: Some(200_022),
            invoke_tool_text: Some(200_057),
            invoke_tool_json: Some(200_049),
            end_message: Some(200_010),
            end_sampling: Some(200_006),
            endoftext: Some(199_999),
        };
        let mut parser = OutputParser::new(ids, vec!["STOP".into()]);
        let mut events = Vec::new();
        for (id, bytes) in [
            (200_001, b"<|message_model|>" as &[u8]),
            (200_004, b"<|content_text|>"),
            (1, b"hello ST"),
            (2, b"OP world"),
            (3, b" ignored"),
        ] {
            parser.feed_token(id, bytes, &mut events);
        }
        parser.finish(&mut events);
        assert!(parser.stopped_by_sequence);
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParsedEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "hello ");
    }

    #[test]
    fn responses_stream_writer_emits_custom_and_namespaced_tool_calls() {
        let mut meta = ResponsesToolMeta::default();
        meta.custom_tools.insert("apply_patch".into());
        meta.namespaced_tools
            .insert("web::run".into(), ("web".into(), "run".into()));
        let mut writer =
            ResponsesStreamWriter::new("resp_1".into(), DEFAULT_MODEL_ID.into(), 100, meta);
        let mut frames = Vec::new();
        writer.start(&mut frames);
        for event in [
            ParsedEvent::ThinkingStart,
            ParsedEvent::ThinkingDelta("Thinking.".into()),
            ParsedEvent::ThinkingEnd,
            ParsedEvent::ToolCall {
                call_id: "call_a".into(),
                name: "apply_patch".into(),
                arguments: r#"{"input":"*** Begin Patch"}"#.into(),
                args_value: json!({"input": "*** Begin Patch"}),
            },
            ParsedEvent::ToolCall {
                call_id: "call_b".into(),
                name: "web::run".into(),
                arguments: r#"{"q":"eremite"}"#.into(),
                args_value: json!({"q": "eremite"}),
            },
        ] {
            writer.handle_parsed(event, &mut frames);
        }
        writer.finish(false, 20, 5, 12, 4, &mut frames);
        let parsed_frames: Vec<Value> = frames
            .iter()
            .map(|frame| {
                let data = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .expect("SSE data line");
                serde_json::from_str::<Value>(data).expect("valid JSON")
            })
            .collect();
        for (i, frame) in parsed_frames.iter().enumerate() {
            assert_eq!(frame["sequence_number"], i);
        }
        let completed = parsed_frames.last().unwrap();
        assert_eq!(completed["type"], "response.completed");
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "Thinking.");
        assert_eq!(output[1]["type"], "custom_tool_call");
        assert_eq!(output[1]["name"], "apply_patch");
        assert_eq!(output[1]["input"], "*** Begin Patch");
        assert_eq!(output[2]["type"], "function_call");
        assert_eq!(output[2]["name"], "run");
        assert_eq!(output[2]["namespace"], "web");
        assert_eq!(output[2]["arguments"], r#"{"q":"eremite"}"#);
        assert_eq!(completed["response"]["usage"]["input_tokens"], 20);
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            5
        );
        assert_eq!(
            completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
            4
        );
    }

    /// Reads the HTTP status and UTF-8 body from `response`.
    async fn read_http_response(response: Response) -> (StatusCode, String) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");
        (status, String::from_utf8(bytes.to_vec()).expect("UTF-8"))
    }

    /// Returns a closed channel preloaded with `events`.
    fn queue_events(events: Vec<Event>) -> UnboundedReceiver<Event> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        for event in events {
            tx.send(event).expect("open channel");
        }
        rx
    }

    /// Returns the Inkling-2-Mini checkpoint directory from
    /// `EREMITE_INKLING2_CHECKPOINT` or `EREMITE_CHECKPOINT` when it holds an
    /// Inkling-2-Mini `config.json`.
    fn inkling2_checkpoint_dir() -> Option<String> {
        let dir = std::env::var("EREMITE_INKLING2_CHECKPOINT")
            .or_else(|_| std::env::var("EREMITE_CHECKPOINT"))
            .ok()?;
        let config =
            std::fs::read_to_string(std::path::Path::new(&dir).join("config.json")).ok()?;
        let json: Value = serde_json::from_str(&config).ok()?;
        matches!(
            json.get("model_type").and_then(Value::as_str),
            Some("inkling2_loop" | "inkling2_loop_text")
        )
        .then_some(dir)
    }

    #[tokio::test]
    async fn stream_and_collect_round_trip_with_checkpoint_tokenizer() {
        let Some(dir) = inkling2_checkpoint_dir() else {
            return;
        };
        let tokenizer = engine::Tokenizer::open(&dir, Some(IMAGE_TOKEN));
        let encode = |text: &str| -> Vec<u32> {
            tokenizer
                .tokenize(text)
                .into_iter()
                .map(|t| t as u32)
                .collect()
        };
        let sample_reply = concat!(
            "<|message_model|><|content_thinking|>Step 1.<|end_message|>",
            "<|message_model|><|content_text|>Hello!<|end_message|>",
        );
        let reply_ids = encode(sample_reply);

        let chat_body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                ],
            }],
        })
        .to_string();
        let prepared = prepare_request(
            chat_body.as_bytes(),
            &test_defaults(),
            tokenizer,
            Api::ChatCompletions,
        )
        .unwrap();
        let worker_req = prepared.build_worker_request(0, "test".into()).unwrap();
        let img_id = tokenizer.find_added_token(IMAGE_TOKEN).unwrap();
        assert_eq!(worker_req.media, vec![b"hi".to_vec()]);
        assert_eq!(
            worker_req.prompt.iter().filter(|&&t| t == img_id).count(),
            1
        );

        let events = vec![
            Event::DecodingStarted {
                prompt_tokens: 12,
                reused_tokens: 3,
            },
            Event::Tokens(reply_ids.clone()),
            Event::Done {
                reason: FinishReason::StopToken,
            },
        ];
        let response = collect_response(
            prepared,
            worker::CancelGuard::new_unsubmitted(1),
            queue_events(events),
        )
        .await;
        let (status, body) = read_http_response(response).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["choices"][0]["message"]["reasoning_content"],
            "Step 1."
        );
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["usage"]["prompt_tokens"], 12);
        assert_eq!(parsed["usage"]["completion_tokens"], reply_ids.len() + 1);

        let stream_body = json!({"input": "Hi", "stream": true}).to_string();
        let prepared = prepare_request(
            stream_body.as_bytes(),
            &test_defaults(),
            tokenizer,
            Api::Responses,
        )
        .unwrap();
        let events = vec![
            Event::DecodingStarted {
                prompt_tokens: 12,
                reused_tokens: 3,
            },
            Event::Tokens(reply_ids),
            Event::Done {
                reason: FinishReason::StopToken,
            },
        ];
        let response = stream_response(
            prepared,
            worker::CancelGuard::new_unsubmitted(2),
            queue_events(events),
            Duration::from_secs(10),
        );
        let (status, sse) = read_http_response(response).await;
        assert_eq!(status, StatusCode::OK);
        assert!(sse.contains("event: response.created"), "{sse}");
        assert!(
            sse.contains("event: response.reasoning_text.delta"),
            "{sse}"
        );
        assert!(sse.contains("event: response.output_text.delta"), "{sse}");
        assert!(sse.contains("event: response.completed"), "{sse}");
        assert!(!sse.contains("[DONE]"), "{sse}");
    }
}
