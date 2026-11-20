//! Image generation and editing API (POST /v1/images/generations and POST
//! /v1/images/edits) for MiniMax-H3-Image. Parses JSON text-to-image requests
//! and multipart/form-data image editing requests, validates MiniMax-H3-Image
//! parameters, encodes generated PNG, JPEG or WebP bytes as base64, and formats
//! OpenAI image responses and stream events.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::worker::ImageSize;
use crate::{chat, engine, http, worker};

/// Default served model id for MiniMax-H3-Image when `id` is omitted.
pub const DEFAULT_MODEL_ID: &str = "minimax-h3-image";
/// Display name shown in GET /v1/models for MiniMax-H3-Image.
pub const DISPLAY_NAME: &str = "MiniMax-H3-Image (eremite)";
/// Maximum body size for multipart image editing uploads (128 MiB for up to 16
/// multi-megabyte reference images).
pub const MAX_EDIT_BODY_BYTES: usize = 128 * http::MIB;

/// CRLF blank-line separator between multipart headers and part body.
const CRLF_HEADER_END: &[u8] = b"\r\n\r\n";
/// LF blank-line separator between multipart headers and part body.
const LF_HEADER_END: &[u8] = b"\n\n";

/// The most partial images that OpenAI's image API takes in one request.
const MAX_PARTIAL_IMAGES: i64 = 3;

/// RFC 4648 standard base64 alphabet.
const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// A validated image generation or editing request ready for the worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedRequest {
    /// Model name from the request, or None when omitted.
    pub model: Option<String>,
    /// Raw prompt string passed verbatim to MiniMax-H3-Image.
    pub prompt: String,
    /// The output size that the request asks for.
    pub size: ImageSize,
    /// Number of images to produce (`n`).
    pub count: usize,
    /// Deterministic RNG seed for the first image (`seed + i` for later ones).
    pub seed: u64,
    /// Minimum generation canvas area in pixels.
    pub min_generation_area: i32,
    /// Encoded output image format.
    pub format: engine::ImageFormat,
    /// JPEG or WebP quality in 0..=100; PNG ignores it.
    pub quality: i32,
    /// Output background.
    pub background: engine::ImageBackground,
    /// Whether the response streams server-sent events.
    pub stream: bool,
    /// Partial images to stream before each image; 0 sends none.
    pub partial_images: usize,
    /// Reference image bytes in upload order (empty for text-to-image).
    pub references: Vec<Vec<u8>>,
    /// An edit's mask image bytes, whose alpha marks what changes; None for
    /// no mask.
    pub mask: Option<Vec<u8>>,
}

/// Parses a request's `size`: omitted or `auto` asks the server to pick it,
/// and `WIDTHxHEIGHT` gives validated pixel dimensions.
fn parse_size(raw: Option<&str>, limits: &engine::ImageLimits) -> Result<ImageSize, chat::Error> {
    let trimmed = raw.map_or("", str::trim);
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return Ok(ImageSize::Auto);
    }
    let Some((w_str, h_str)) = trimmed.split_once(['x', 'X']) else {
        return Err(chat::Error::new_bad_request(
            "size must be auto or have the form WIDTHxHEIGHT",
        ));
    };
    let width: i32 = w_str
        .parse()
        .map_err(|_| chat::Error::new_bad_request("size width must be a positive integer"))?;
    let height: i32 = h_str
        .parse()
        .map_err(|_| chat::Error::new_bad_request("size height must be a positive integer"))?;
    if !(1..=limits.max_dim).contains(&width) || !(1..=limits.max_dim).contains(&height) {
        return Err(chat::Error::new_bad_request(format!(
            "size dimensions must be between 1 and {}",
            limits.max_dim
        )));
    }
    Ok(ImageSize::Exact { width, height })
}

/// Checks that total output pixels across all requested images fit within the
/// response memory bound.
fn validate_total_pixels(
    width: i32,
    height: i32,
    count: usize,
    limits: &engine::ImageLimits,
) -> Result<(), chat::Error> {
    let total_pixels = (width as i64)
        .saturating_mul(height as i64)
        .saturating_mul(count as i64);
    if total_pixels > limits.max_request_pixels {
        return Err(chat::Error::new_bad_request(format!(
            "total output pixels ({width}x{height} x {count} = {total_pixels}) exceed {} pixel limit",
            limits.max_request_pixels
        )));
    }
    Ok(())
}

/// Validates and parses the `response_format` value (only `b64_json`).
fn parse_response_format(value: Option<&str>) -> Result<(), chat::Error> {
    match value {
        None | Some("b64_json") => Ok(()),
        Some(other) => Err(chat::Error::new_bad_request(format!(
            "response_format {other:?} is not supported; use \"b64_json\""
        ))),
    }
}

/// Validates and parses the `output_format` value (`png`, `jpeg`, `jpg` or
/// `webp`).
fn parse_output_format(value: Option<&str>) -> Result<engine::ImageFormat, chat::Error> {
    match value {
        None => Ok(engine::ImageFormat::Png),
        Some(raw) if raw.eq_ignore_ascii_case("png") => Ok(engine::ImageFormat::Png),
        Some(raw) if raw.eq_ignore_ascii_case("jpeg") || raw.eq_ignore_ascii_case("jpg") => {
            Ok(engine::ImageFormat::Jpeg)
        }
        Some(raw) if raw.eq_ignore_ascii_case("webp") => Ok(engine::ImageFormat::Webp),
        Some(other) => Err(chat::Error::new_bad_request(format!(
            "output_format {other:?} is not supported; use \"png\", \"jpeg\" or \"webp\""
        ))),
    }
}

/// Validates and parses the `background` value (`auto`, `opaque` or
/// `transparent`) of a request whose output is `format`; a transparent
/// background needs a format with alpha.
fn parse_background(
    value: Option<&str>,
    format: engine::ImageFormat,
) -> Result<engine::ImageBackground, chat::Error> {
    let background = match value {
        None => engine::ImageBackground::Auto,
        Some(raw) if raw.eq_ignore_ascii_case("auto") => engine::ImageBackground::Auto,
        Some(raw) if raw.eq_ignore_ascii_case("opaque") => engine::ImageBackground::Opaque,
        Some(raw) if raw.eq_ignore_ascii_case("transparent") => {
            engine::ImageBackground::Transparent
        }
        Some(other) => {
            return Err(chat::Error::new_bad_request(format!(
                "background {other:?} is not supported; use \"auto\", \"opaque\" or \"transparent\""
            )));
        }
    };
    if background == engine::ImageBackground::Transparent && format == engine::ImageFormat::Jpeg {
        return Err(chat::Error::new_bad_request(
            "a transparent background needs output_format png or webp",
        ));
    }
    Ok(background)
}

/// Reads an optional string field from a JSON object, rejecting non-strings.
fn read_optional_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>, chat::Error> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(chat::Error::new_bad_request(format!(
            "{key} must be a string"
        ))),
    }
}

/// Reads an optional integer field in `[min, max]` from a JSON object.
fn read_optional_i64(
    object: &Map<String, Value>,
    key: &str,
    min: i64,
    max: i64,
) -> Result<Option<i64>, chat::Error> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let n = value
                .as_i64()
                .ok_or_else(|| chat::Error::new_bad_request(format!("{key} must be an integer")))?;
            if !(min..=max).contains(&n) {
                return Err(chat::Error::new_bad_request(format!(
                    "{key} must be between {min} and {max}"
                )));
            }
            Ok(Some(n))
        }
    }
}

/// Parses and validates a JSON text-to-image request for POST
/// /v1/images/generations against the resolved image model's `limits`.
pub fn parse_generation_request(
    body: &[u8],
    limits: &engine::ImageLimits,
) -> Result<ParsedRequest, chat::Error> {
    read_json_request(&read_json_object(body)?, limits)
}

/// Reads a request body that must be one JSON object.
fn read_json_object(body: &[u8]) -> Result<Map<String, Value>, chat::Error> {
    let root: Value = serde_json::from_slice(body)
        .map_err(|error| chat::Error::new_bad_request(format!("not valid JSON: {error}")))?;
    match root {
        Value::Object(object) => Ok(object),
        _ => Err(chat::Error::new_bad_request(
            "request body must be a JSON object",
        )),
    }
}

/// Reads and validates the fields of a JSON image request, other than its
/// reference images, against the target model's `limits`.
fn read_json_request(
    object: &Map<String, Value>,
    limits: &engine::ImageLimits,
) -> Result<ParsedRequest, chat::Error> {
    let model = read_optional_string(object, "model")?
        .filter(|s| !s.is_empty())
        .map(String::from);
    let prompt = read_optional_string(object, "prompt")?
        .ok_or_else(|| chat::Error::new_bad_request("prompt is required"))?
        .to_string();
    let size = parse_size(read_optional_string(object, "size")?, limits)?;

    let count = read_optional_i64(object, "n", 1, limits.max_images_per_request as i64)?
        .unwrap_or(1) as usize;
    if let ImageSize::Exact { width, height } = size {
        validate_total_pixels(width, height, count, limits)?;
    }
    let seed = match object.get("seed") {
        None | Some(Value::Null) => rand::random(),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| chat::Error::new_bad_request("seed must be a nonnegative integer"))?,
    };
    let _generator_device = read_optional_string(object, "generator_device")?;
    parse_response_format(read_optional_string(object, "response_format")?)?;
    let format = parse_output_format(read_optional_string(object, "output_format")?)?;
    let background = parse_background(read_optional_string(object, "background")?, format)?;
    let stream = match object.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => return Err(chat::Error::new_bad_request("stream must be a boolean")),
    };
    let partial_images =
        read_optional_i64(object, "partial_images", 0, MAX_PARTIAL_IMAGES)?.unwrap_or(0) as usize;
    let quality = read_optional_i64(
        object,
        "output_compression",
        limits.min_quality as i64,
        limits.max_quality as i64,
    )?
    .unwrap_or(limits.default_quality as i64) as i32;
    let min_generation_area = read_optional_i64(
        object,
        "min_generation_area",
        limits.min_generation_area as i64,
        limits.max_generation_area as i64,
    )?
    .unwrap_or(limits.default_generation_area as i64) as i32;

    if let Some(value) = object.get("num_inference_steps")
        && !value.is_null()
        && value.as_i64() != Some(limits.inference_steps as i64)
    {
        return Err(chat::Error::new_bad_request(format!(
            "num_inference_steps must be omitted or {}",
            limits.inference_steps
        )));
    }
    if let Some(value) = object.get("guidance_scale")
        && !value.is_null()
    {
        let ok = value
            .as_f64()
            .is_some_and(|scale| scale.is_finite() && (scale - limits.guidance_scale).abs() == 0.0);
        if !ok {
            return Err(chat::Error::new_bad_request(format!(
                "guidance_scale must be omitted or {}",
                limits.guidance_scale
            )));
        }
    }
    if let Some(neg) = read_optional_string(object, "negative_prompt")?
        && !neg.is_empty()
    {
        return Err(chat::Error::new_bad_request(
            "negative_prompt is not supported",
        ));
    }

    Ok(ParsedRequest {
        model,
        prompt,
        size,
        count,
        seed,
        min_generation_area,
        format,
        quality,
        background,
        stream,
        partial_images,
        references: Vec::new(),
        mask: None,
    })
}

/// Extracts the multipart boundary token from a Content-Type header value.
fn extract_multipart_boundary(content_type: &str) -> Result<String, chat::Error> {
    let mut parts = content_type.split(';');
    let media_type = parts.next().unwrap_or_default().trim();
    if !media_type.eq_ignore_ascii_case("multipart/form-data") {
        return Err(chat::Error::new_bad_request(
            "Content-Type must be application/json or multipart/form-data",
        ));
    }
    for param in parts {
        let Some((key, val)) = param.split_once('=') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("boundary") {
            let val = val.trim();
            let unquoted = val
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(val);
            if !unquoted.is_empty() {
                return Ok(unquoted.to_string());
            }
        }
    }
    Err(chat::Error::new_bad_request(
        "multipart/form-data Content-Type is missing a boundary parameter",
    ))
}

/// One parsed part from a multipart/form-data body.
struct MultipartPart<'a> {
    /// Form field name from Content-Disposition.
    name: String,
    /// Raw bytes of the part body.
    data: &'a [u8],
}

/// Finds the byte offset of `needle` inside `haystack`, or None.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Extracts the `name` attribute from a Content-Disposition header value.
fn parse_disposition_name(value: &str) -> Option<String> {
    for param in value.split(';').skip(1) {
        let Some((key, val)) = param.split_once('=') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("name") {
            let val = val.trim();
            let unquoted = val
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(val);
            return Some(unquoted.to_string());
        }
    }
    None
}

/// Splits an RFC 7578 multipart/form-data body into ordered parts.
fn parse_multipart<'a>(
    boundary: &str,
    body: &'a [u8],
) -> Result<Vec<MultipartPart<'a>>, chat::Error> {
    let delimiter = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let Some(first) = find_subslice(body, &delimiter) else {
        return Err(chat::Error::new_bad_request(
            "multipart body does not contain its boundary",
        ));
    };
    let mut cursor = first + delimiter.len();
    loop {
        let rest = &body[cursor..];
        if rest.starts_with(b"--") {
            break;
        }
        if rest.starts_with(b"\r\n") {
            cursor += 2;
        } else if rest.starts_with(b"\n") {
            cursor += 1;
        } else {
            return Err(chat::Error::new_bad_request(
                "malformed multipart boundary line",
            ));
        }
        // Locate end of part headers (\r\n\r\n or \n\n).
        let part_slice = &body[cursor..];
        let (header_end, body_start) = if let Some(pos) = find_subslice(part_slice, CRLF_HEADER_END)
        {
            (pos, pos + CRLF_HEADER_END.len())
        } else if let Some(pos) = find_subslice(part_slice, LF_HEADER_END) {
            (pos, pos + LF_HEADER_END.len())
        } else {
            return Err(chat::Error::new_bad_request(
                "multipart part is missing header terminator",
            ));
        };
        let headers_text = std::str::from_utf8(&part_slice[..header_end]).map_err(|_| {
            chat::Error::new_bad_request("multipart part headers are not valid UTF-8")
        })?;
        let mut field_name = None;
        for line in headers_text.lines() {
            if let Some((hname, hval)) = line.split_once(':')
                && hname.trim().eq_ignore_ascii_case("content-disposition")
            {
                field_name = parse_disposition_name(hval.trim());
            }
        }
        let name = field_name.ok_or_else(|| {
            chat::Error::new_bad_request("multipart part is missing Content-Disposition name")
        })?;

        let data_slice = &part_slice[body_start..];
        // The next delimiter starts after a line break (\r\n or \n).
        let mut search_from = 0;
        let (data_end, next_cursor) = loop {
            let Some(rel) = find_subslice(&data_slice[search_from..], &delimiter) else {
                return Err(chat::Error::new_bad_request(
                    "multipart body ended before closing boundary",
                ));
            };
            let pos = search_from + rel;
            let after_delim = &data_slice[pos + delimiter.len()..];
            let valid_delim_tail = after_delim.starts_with(b"--")
                || after_delim.starts_with(b"\r\n")
                || after_delim.starts_with(b"\n");
            if valid_delim_tail && pos >= 2 && &data_slice[pos - 2..pos] == b"\r\n" {
                break (pos - 2, cursor + body_start + pos + delimiter.len());
            }
            if valid_delim_tail && pos >= 1 && data_slice[pos - 1] == b'\n' {
                break (pos - 1, cursor + body_start + pos + delimiter.len());
            }
            search_from = pos + 1;
        };
        parts.push(MultipartPart {
            name,
            data: &data_slice[..data_end],
        });
        cursor = next_cursor;
    }
    Ok(parts)
}

/// Parses an optional text form integer in `[min, max]`.
fn parse_form_i64(
    fields: &HashMap<String, String>,
    key: &str,
    min: i64,
    max: i64,
) -> Result<Option<i64>, chat::Error> {
    let Some(raw) = fields.get(key) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let n: i64 = trimmed
        .parse()
        .map_err(|_| chat::Error::new_bad_request(format!("{key} must be an integer")))?;
    if !(min..=max).contains(&n) {
        return Err(chat::Error::new_bad_request(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(Some(n))
}

/// Unvalidated fields and image attachments parsed from a multipart edit body.
pub struct EditForm {
    /// Model name from the `model` field, or None when omitted.
    pub model: Option<String>,
    /// Non-image UTF-8 form fields.
    fields: HashMap<String, String>,
    /// Uploaded `image` and `image[]` parts in upload order.
    references: Vec<Vec<u8>>,
    /// The uploaded `mask` part, if any.
    mask: Option<Vec<u8>>,
}

/// Parses a multipart/form-data image editing body into fields and reference
/// attachments before model-specific limit validation.
pub fn parse_edit_form(content_type: Option<&str>, body: &[u8]) -> Result<EditForm, chat::Error> {
    let content_type = content_type.ok_or_else(|| {
        chat::Error::new_bad_request("Content-Type must be application/json or multipart/form-data")
    })?;
    let boundary = extract_multipart_boundary(content_type)?;
    let parts = parse_multipart(&boundary, body)?;

    let mut fields = HashMap::<String, String>::new();
    let mut references = Vec::<Vec<u8>>::new();
    let mut mask = None;
    for part in parts {
        if part.name == "mask" {
            if part.data.is_empty() {
                return Err(chat::Error::new_bad_request("the mask part is empty"));
            }
            if mask.replace(part.data.to_vec()).is_some() {
                return Err(chat::Error::new_bad_request("an edit takes one mask"));
            }
            continue;
        }
        if part.name == "image" || part.name == "image[]" {
            if part.data.is_empty() {
                return Err(chat::Error::new_bad_request("an image part is empty"));
            }
            references.push(part.data.to_vec());
        } else {
            let text = std::str::from_utf8(part.data).map_err(|_| {
                chat::Error::new_bad_request(format!("field {:?} is not valid UTF-8", part.name))
            })?;
            fields.insert(part.name, text.to_string());
        }
    }
    let model = fields
        .get("model")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);
    Ok(EditForm {
        model,
        fields,
        references,
        mask,
    })
}

impl EditForm {
    /// Validates the parsed multipart form against the target model's `limits`.
    pub fn validate(self, limits: &engine::ImageLimits) -> Result<ParsedRequest, chat::Error> {
        let EditForm {
            model,
            fields,
            references,
            mask,
        } = self;
        check_reference_count(references.len(), limits)?;

        let prompt = fields
            .get("prompt")
            .ok_or_else(|| chat::Error::new_bad_request("prompt is required"))?
            .clone();
        let size = parse_size(fields.get("size").map(String::as_str), limits)?;

        let count = parse_form_i64(&fields, "n", 1, limits.max_images_per_request as i64)?
            .unwrap_or(1) as usize;
        if let ImageSize::Exact { width, height } = size {
            validate_total_pixels(width, height, count, limits)?;
        }
        let seed = match fields
            .get("seed")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            None => rand::random(),
            Some(raw) => raw
                .parse::<u64>()
                .map_err(|_| chat::Error::new_bad_request("seed must be a nonnegative integer"))?,
        };
        parse_response_format(
            fields
                .get("response_format")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty()),
        )?;
        let format = parse_output_format(
            fields
                .get("output_format")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty()),
        )?;
        let background = parse_background(
            fields
                .get("background")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty()),
            format,
        )?;
        let stream = match fields
            .get("stream")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            None => false,
            Some(raw) if raw.eq_ignore_ascii_case("true") => true,
            Some(raw) if raw.eq_ignore_ascii_case("false") => false,
            Some(_) => {
                return Err(chat::Error::new_bad_request("stream must be true or false"));
            }
        };
        let partial_images =
            parse_form_i64(&fields, "partial_images", 0, MAX_PARTIAL_IMAGES)?.unwrap_or(0) as usize;
        let quality = parse_form_i64(
            &fields,
            "output_compression",
            limits.min_quality as i64,
            limits.max_quality as i64,
        )?
        .unwrap_or(limits.default_quality as i64) as i32;
        let min_generation_area = parse_form_i64(
            &fields,
            "min_generation_area",
            limits.min_generation_area as i64,
            limits.max_generation_area as i64,
        )?
        .unwrap_or(limits.default_generation_area as i64) as i32;

        if let Some(raw) = fields
            .get("num_inference_steps")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            && raw.parse::<i64>().ok() != Some(limits.inference_steps as i64)
        {
            return Err(chat::Error::new_bad_request(format!(
                "num_inference_steps must be omitted or {}",
                limits.inference_steps
            )));
        }
        if let Some(raw) = fields
            .get("guidance_scale")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let ok = raw.parse::<f64>().ok().is_some_and(|scale| {
                scale.is_finite() && (scale - limits.guidance_scale).abs() == 0.0
            });
            if !ok {
                return Err(chat::Error::new_bad_request(format!(
                    "guidance_scale must be omitted or {}",
                    limits.guidance_scale
                )));
            }
        }
        if let Some(neg) = fields.get("negative_prompt")
            && !neg.is_empty()
        {
            return Err(chat::Error::new_bad_request(
                "negative_prompt is not supported",
            ));
        }

        Ok(ParsedRequest {
            model,
            prompt,
            size,
            count,
            seed,
            min_generation_area,
            format,
            quality,
            background,
            stream,
            partial_images,
            references,
            mask,
        })
    }
}

/// Checks that an edit has 1 to the model's most reference images.
fn check_reference_count(count: usize, limits: &engine::ImageLimits) -> Result<(), chat::Error> {
    if count == 0 {
        return Err(chat::Error::new_bad_request(
            "at least one reference image is required",
        ));
    }
    if count > limits.max_references as usize {
        return Err(chat::Error::new_bad_request(format!(
            "at most {} reference images are supported",
            limits.max_references
        )));
    }
    Ok(())
}

/// An edit's body before the target model's limits validate it.
pub enum EditBody {
    /// A multipart/form-data body.
    Form(EditForm),
    /// A JSON body, with the reference images of its `images` and its mask
    /// decoded.
    Json {
        /// The body's object.
        object: Map<String, Value>,
        /// The decoded images in the array's order.
        references: Vec<Vec<u8>>,
        /// The decoded mask, if any.
        mask: Option<Vec<u8>>,
    },
}

impl EditBody {
    /// Returns the model that the body names, or None.
    pub fn model(&self) -> Option<&str> {
        match self {
            EditBody::Form(form) => form.model.as_deref(),
            EditBody::Json { object, .. } => object
                .get("model")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty()),
        }
    }

    /// Validates the body against the target model's `limits`.
    pub fn validate(self, limits: &engine::ImageLimits) -> Result<ParsedRequest, chat::Error> {
        match self {
            EditBody::Form(form) => form.validate(limits),
            EditBody::Json {
                object,
                references,
                mask,
            } => {
                check_reference_count(references.len(), limits)?;
                let mut parsed = read_json_request(&object, limits)?;
                parsed.references = references;
                parsed.mask = mask;
                Ok(parsed)
            }
        }
    }
}

/// Parses the body of POST /v1/images/edits by its Content-Type: JSON with an
/// `images` array, or multipart/form-data with `image` parts.
pub fn parse_edit_body(content_type: Option<&str>, body: &[u8]) -> Result<EditBody, chat::Error> {
    let media_type = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .unwrap_or_default();
    if !media_type.eq_ignore_ascii_case("application/json") {
        return parse_edit_form(content_type, body).map(EditBody::Form);
    }
    let object = read_json_object(body)?;
    let references = read_json_images(&object)?;
    let mask = match object.get("mask") {
        None | Some(Value::Null) => None,
        Some(item) => Some(read_image_object(item, "mask")?),
    };
    Ok(EditBody::Json {
        object,
        references,
        mask,
    })
}

/// Decodes an image object of a JSON edit, named `name` in errors: an object
/// whose `image_url` is a base64 data URL.
fn read_image_object(item: &Value, name: &str) -> Result<Vec<u8>, chat::Error> {
    if item.get("file_id").is_some_and(|id| !id.is_null()) {
        return Err(chat::Error::new_bad_request(format!(
            "{name}.file_id is not supported: this server has no Files API; \
             send the image as a base64 data URL in image_url"
        )));
    }
    let url = item
        .get("image_url")
        .and_then(Value::as_str)
        .ok_or_else(|| chat::Error::new_bad_request(format!("{name} needs an image_url")))?;
    let bytes = chat::decode_data_url(url).map_err(|_| {
        chat::Error::new_bad_request(format!(
            "{name}.image_url must be a base64 data URL; the server fetches no URLs"
        ))
    })?;
    if bytes.is_empty() {
        return Err(chat::Error::new_bad_request(format!("{name} is empty")));
    }
    Ok(bytes)
}

/// Decodes the reference images of a JSON edit's `images` array: objects
/// whose `image_url` is a base64 data URL.
fn read_json_images(object: &Map<String, Value>) -> Result<Vec<Vec<u8>>, chat::Error> {
    let items = match object.get("images") {
        Some(Value::Array(items)) => items,
        None | Some(Value::Null) => {
            return Err(chat::Error::new_bad_request(
                "images is required: an array of objects with an image_url",
            ));
        }
        Some(_) => return Err(chat::Error::new_bad_request("images must be an array")),
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| read_image_object(item, &format!("images[{index}]")))
        .collect()
}

/// Parses and validates an image editing request for POST /v1/images/edits
/// against `limits`. Reference images keep their order (`<Picture 1>`,
/// `<Picture 2>`, ...).
pub fn parse_edit_request(
    content_type: Option<&str>,
    body: &[u8],
    limits: &engine::ImageLimits,
) -> Result<ParsedRequest, chat::Error> {
    parse_edit_body(content_type, body)?.validate(limits)
}

/// Encodes `bytes` into standard RFC 4648 base64 with `=` padding.
pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_TABLE[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_TABLE[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_TABLE[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Builds an OpenAI Images API JSON response containing `images` as base64
/// `b64_json` items, with their `width` by `height` size, their `format` and
/// their `background`.
pub fn build_response(
    images: &[Vec<u8>],
    width: i32,
    height: i32,
    format: engine::ImageFormat,
    background: engine::ImageBackground,
) -> Value {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let data: Vec<Value> = images
        .iter()
        .map(|bytes| json!({"b64_json": encode_base64(bytes)}))
        .collect();
    json!({
        "created": created,
        "data": data,
        "size": format!("{width}x{height}"),
        "output_format": format.name(),
        "background": background.response_name(),
    })
}

/// The endpoint that an image request came to, which names its stream's
/// events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageEndpoint {
    /// POST /v1/images/generations, whose events are image_generation.*.
    Generations,
    /// POST /v1/images/edits, whose events are image_edit.*.
    Edits,
}

/// What the stream events of one image request share.
#[derive(Clone, Copy, Debug)]
pub struct ImageStream {
    /// The endpoint, which names the events.
    pub endpoint: ImageEndpoint,
    /// The Unix time in seconds when the request came.
    pub created_at: u64,
    /// The images' format.
    pub format: engine::ImageFormat,
    /// The images' background.
    pub background: engine::ImageBackground,
}

impl ImageStream {
    /// Formats the server-sent event of a `width` by `height` partial image,
    /// with its 0-based `index`.
    pub fn format_partial_image(
        &self,
        image: &[u8],
        width: i32,
        height: i32,
        index: usize,
    ) -> String {
        let mut event = self.describe_image(image, width, height);
        event["partial_image_index"] = json!(index);
        self.format_event("partial_image", event)
    }

    /// Formats the server-sent event of a finished `width` by `height` image.
    pub fn format_completed_image(&self, image: &[u8], width: i32, height: i32) -> String {
        self.format_event("completed", self.describe_image(image, width, height))
    }

    /// Formats the error event that ends a stream, with the error's `kind`
    /// and `message`.
    pub fn format_error(kind: &str, message: &str) -> String {
        let event = json!({
            "type": "error",
            "error": {"type": kind, "code": null, "message": message, "param": null},
        });
        format!("event: error\ndata: {event}\n\n")
    }

    /// Formats the server-sent event of one of the request's worker events, and
    /// says whether the stream ends with it. ImagesDone ends the stream
    /// without an event.
    pub fn format_worker_event(&self, event: worker::Event) -> (Option<String>, bool) {
        match event {
            worker::Event::PartialImage {
                image,
                width,
                height,
                index,
            } => (
                Some(self.format_partial_image(&image, width, height, index)),
                false,
            ),
            worker::Event::Image {
                image,
                width,
                height,
            } => (
                Some(self.format_completed_image(&image, width, height)),
                false,
            ),
            worker::Event::ImagesDone => (None, true),
            worker::Event::Refused(message) => (
                Some(Self::format_error("invalid_request_error", &message)),
                true,
            ),
            worker::Event::Failed(message) => {
                (Some(Self::format_error("server_error", &message)), true)
            }
            _ => (
                Some(Self::format_error("server_error", "the server is stopping")),
                true,
            ),
        }
    }

    /// Returns the fields of an image's event other than its type.
    fn describe_image(&self, image: &[u8], width: i32, height: i32) -> Value {
        json!({
            "b64_json": encode_base64(image),
            "created_at": self.created_at,
            "size": format!("{width}x{height}"),
            "output_format": self.format.name(),
            "background": self.background.response_name(),
        })
    }

    /// Formats the event named by the endpoint and `name`, with the fields
    /// of `event`.
    fn format_event(&self, name: &str, mut event: Value) -> String {
        let prefix = match self.endpoint {
            ImageEndpoint::Generations => "image_generation",
            ImageEndpoint::Edits => "image_edit",
        };
        let kind = format!("{prefix}.{name}");
        event["type"] = json!(kind);
        format!("event: {kind}\ndata: {event}\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a representative image-model limits fixture for parser tests.
    fn test_limits() -> engine::ImageLimits {
        engine::ImageLimits {
            max_prompt_tokens: 4096,
            max_dim: 8192,
            default_width: 2048,
            default_height: 2048,
            min_generation_area: 1024,
            max_generation_area: 16_777_216,
            default_generation_area: 4_194_304,
            max_canvas_area: 24 * 1024 * 1024,
            max_request_pixels: 32 * 1024 * 1024,
            max_images_per_request: 16,
            max_references: 16,
            min_quality: 0,
            max_quality: 100,
            default_quality: 100,
            inference_steps: 6,
            guidance_scale: 1.0,
        }
    }

    #[test]
    fn base64_matches_rfc_4648_test_vectors() {
        for (input, expected) in [
            (&b""[..], ""),
            (&b"f"[..], "Zg=="),
            (&b"fo"[..], "Zm8="),
            (&b"foo"[..], "Zm9v"),
            (&b"foob"[..], "Zm9vYg=="),
            (&b"fooba"[..], "Zm9vYmE="),
            (&b"foobar"[..], "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(input), expected);
        }
    }

    #[test]
    fn generation_request_reads_valid_t2i_fields_and_defaults() {
        let limits = test_limits();
        let body = serde_json::to_vec(&json!({
            "prompt": "A red ceramic mug on wood.",
            "size": "4736x3552",
            "seed": 42,
            "generator_device": "cpu",
            "response_format": "b64_json",
            "output_format": "png",
            "num_inference_steps": 6,
            "guidance_scale": 1
        }))
        .unwrap();
        let req = parse_generation_request(&body, &limits).unwrap();
        assert_eq!(req.model, None);
        assert_eq!(req.prompt, "A red ceramic mug on wood.");
        assert_eq!(
            req.size,
            ImageSize::Exact {
                width: 4736,
                height: 3552
            }
        );
        assert_eq!(req.count, 1);
        assert_eq!(req.seed, 42);
        assert_eq!(req.min_generation_area, limits.default_generation_area);
        assert_eq!(req.format, engine::ImageFormat::Png);
        assert_eq!(req.quality, limits.default_quality);
        assert!(req.references.is_empty());

        let jpg_body = serde_json::to_vec(&json!({
            "model": "minimax-h3-image",
            "prompt": "blue sky",
            "size": "512X512",
            "n": 2,
            "seed": 7,
            "output_format": "jpg",
            "output_compression": 85,
            "min_generation_area": 1024
        }))
        .unwrap();
        let req = parse_generation_request(&jpg_body, &limits).unwrap();
        assert_eq!(req.model.as_deref(), Some("minimax-h3-image"));
        assert_eq!(
            req.size,
            ImageSize::Exact {
                width: 512,
                height: 512
            }
        );
        assert_eq!(req.count, 2);
        assert_eq!(req.format, engine::ImageFormat::Jpeg);
        assert_eq!(req.quality, 85);
        assert_eq!(req.min_generation_area, 1024);

        // Validation follows the resolved model's limits rather than hardcoded values.
        let custom = engine::ImageLimits {
            max_dim: 1024,
            inference_steps: 4,
            ..limits
        };
        let custom_ok = serde_json::to_vec(&json!({
            "prompt": "small",
            "size": "1024x1024",
            "num_inference_steps": 4
        }))
        .unwrap();
        assert!(parse_generation_request(&custom_ok, &custom).is_ok());
        assert!(parse_generation_request(&custom_ok, &limits).is_err());
        let custom_big = serde_json::to_vec(&json!({
            "prompt": "big",
            "size": "2048x1024"
        }))
        .unwrap();
        assert!(parse_generation_request(&custom_big, &custom).is_err());
    }

    #[test]
    fn generation_request_without_a_size_or_with_auto_leaves_it_to_the_server() {
        let limits = test_limits();
        for body in [
            json!({"prompt": "hi"}),
            json!({"prompt": "hi", "size": "auto"}),
            json!({"prompt": "hi", "size": "AUTO"}),
            json!({"prompt": "hi", "size": "4096x4096", "n": 16, "size": "auto"}),
        ] {
            let bytes = serde_json::to_vec(&body).unwrap();
            let req = parse_generation_request(&bytes, &limits).unwrap();
            assert_eq!(req.size, ImageSize::Auto, "{body}");
        }
    }

    #[test]
    fn generation_request_takes_webp_and_the_lowest_compression() {
        let limits = test_limits();
        let body = serde_json::to_vec(&json!({
            "prompt": "hi",
            "output_format": "webp",
            "output_compression": 0
        }))
        .unwrap();
        let req = parse_generation_request(&body, &limits).unwrap();
        assert_eq!((req.format, req.quality), (engine::ImageFormat::Webp, 0));
    }

    #[test]
    fn generation_request_rejects_invalid_and_unsupported_parameters() {
        let limits = test_limits();
        for bad in [
            json!({"size": "512x512"}),
            json!({"prompt": "hi", "size": "512"}),
            json!({"prompt": "hi", "size": "big"}),
            json!({"prompt": "hi", "size": "0x512"}),
            json!({"prompt": "hi", "size": "9000x512"}),
            json!({"prompt": "hi", "size": "512x512", "n": 0}),
            json!({"prompt": "hi", "size": "512x512", "n": 17}),
            json!({"prompt": "hi", "size": "512x512", "seed": -1}),
            json!({"prompt": "hi", "size": "512x512", "response_format": "url"}),
            json!({"prompt": "hi", "size": "512x512", "output_format": "gif"}),
            json!({"prompt": "hi", "size": "512x512", "output_compression": 101}),
            json!({"prompt": "hi", "size": "512x512", "min_generation_area": 512}),
            json!({"prompt": "hi", "size": "512x512", "min_generation_area": 20_000_000}),
            json!({"prompt": "hi", "size": "512x512", "num_inference_steps": 8}),
            json!({"prompt": "hi", "size": "512x512", "guidance_scale": 7.5}),
            json!({"prompt": "hi", "size": "512x512", "negative_prompt": "blurry"}),
            json!({"prompt": "hi", "size": "4096x4096", "n": 16}),
        ] {
            let bytes = serde_json::to_vec(&bad).unwrap();
            assert!(
                parse_generation_request(&bytes, &limits).is_err(),
                "expected err: {bad}"
            );
        }
    }

    #[test]
    fn edit_request_parses_multipart_fields_and_preserves_image_order() {
        let limits = test_limits();
        let boundary = "test-boundary-123";
        let mut body = Vec::new();
        for (name, val) in [
            ("model", "minimax-h3-image"),
            ("prompt", "Combine <Picture 1> and <Picture 2>."),
            ("size", "1024x1024"),
            ("seed", "1234"),
            ("generator_device", "cpu"),
            ("response_format", "b64_json"),
            ("output_format", "png"),
            ("min_generation_area", "1024"),
        ] {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n"
                )
                .as_bytes(),
            );
        }
        let img1 = b"\x89PNG\r\n\x1a\nfirst-image-bytes";
        let img2 = b"\xff\xd8\xff\xe0second-image-bytes";
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(img1);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"b.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(img2);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let ct = format!("multipart/form-data; boundary=\"{boundary}\"");
        let req = parse_edit_request(Some(&ct), &body, &limits).unwrap();
        assert_eq!(req.model.as_deref(), Some("minimax-h3-image"));
        assert_eq!(req.prompt, "Combine <Picture 1> and <Picture 2>.");
        assert_eq!(
            req.size,
            ImageSize::Exact {
                width: 1024,
                height: 1024
            }
        );
        assert_eq!(req.seed, 1234);
        assert_eq!(req.min_generation_area, 1024);
        assert_eq!(req.references, vec![img1.to_vec(), img2.to_vec()]);
    }

    #[test]
    fn edit_request_rejects_missing_images_and_bad_steps() {
        let limits = test_limits();
        let boundary = "b1";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n512x512\r\n\
             --{boundary}--\r\n"
        );
        let ct = format!("multipart/form-data; boundary={boundary}");
        assert!(parse_edit_request(Some(&ct), body.as_bytes(), &limits).is_err());

        let bad_steps = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n512x512\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"num_inference_steps\"\r\n\r\n20\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
             --{boundary}--\r\n"
        );
        assert!(parse_edit_request(Some(&ct), bad_steps.as_bytes(), &limits).is_err());

        let too_many_pixels = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n4096x4096\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n3\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
             --{boundary}--\r\n"
        );
        assert!(parse_edit_request(Some(&ct), too_many_pixels.as_bytes(), &limits).is_err());
    }

    #[test]
    fn edit_request_reads_json_images_from_data_urls() {
        let limits = test_limits();
        let url = |bytes: &[u8]| format!("data:image/png;base64,{}", encode_base64(bytes));
        let body = serde_json::to_vec(&json!({
            "model": "gpt-image-2",
            "prompt": "Combine <Picture 1> and <Picture 2>.",
            "images": [{"image_url": url(b"first")}, {"image_url": url(b"second")}],
            "mask": {"image_url": url(b"mask")},
            "size": "auto",
            "quality": "auto",
            "background": "opaque"
        }))
        .unwrap();
        let edit = parse_edit_body(Some("application/json"), &body).unwrap();
        assert_eq!(edit.model(), Some("gpt-image-2"));
        let req = edit.validate(&limits).unwrap();
        assert_eq!(req.references, vec![b"first".to_vec(), b"second".to_vec()]);
        assert_eq!(req.mask.as_deref(), Some(&b"mask"[..]));
        assert_eq!(req.size, ImageSize::Auto);
        assert_eq!(req.background, engine::ImageBackground::Opaque);
        assert_eq!(req.prompt, "Combine <Picture 1> and <Picture 2>.");
    }

    #[test]
    fn edit_request_refuses_json_images_it_cannot_read() {
        let limits = test_limits();
        let data = format!("data:image/png;base64,{}", encode_base64(b"x"));
        for (bad, message) in [
            (json!({"prompt": "p"}), "images is required"),
            (json!({"prompt": "p", "images": "x"}), "must be an array"),
            (
                json!({"prompt": "p", "images": []}),
                "at least one reference image",
            ),
            (
                json!({"prompt": "p", "images": [{"file_id": "file-1"}]}),
                "no Files API",
            ),
            (
                json!({"prompt": "p", "images": [{"image_url": "https://example.com/a.png"}]}),
                "fetches no URLs",
            ),
            (json!({"prompt": "p", "images": [{}]}), "needs an image_url"),
            (
                json!({"prompt": "p", "images": [{"image_url": data}], "mask": {"file_id": "f"}}),
                "mask.file_id is not supported",
            ),
            (
                json!({"prompt": "p", "images": [{"image_url": data}], "mask": {"image_url": "https://a"}}),
                "mask.image_url must be a base64 data URL",
            ),
            (
                json!({"images": [{"image_url": data}]}),
                "prompt is required",
            ),
        ] {
            let body = serde_json::to_vec(&bad).unwrap();
            let error = parse_edit_request(Some("application/json; charset=utf-8"), &body, &limits)
                .expect_err("refused");
            assert!(error.message.contains(message), "{bad}: {}", error.message);
        }
        let many: Vec<Value> = (0..=limits.max_references)
            .map(|_| json!({"image_url": data}))
            .collect();
        let body = serde_json::to_vec(&json!({"prompt": "p", "images": many})).unwrap();
        assert!(parse_edit_request(Some("application/json"), &body, &limits).is_err());
        assert!(parse_edit_request(Some("text/plain"), b"{}", &limits).is_err());
    }

    #[test]
    fn edit_form_reads_a_mask() {
        let boundary = "b2";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"mask\"\r\n\r\n\u{0}\u{1}\r\n\
             --{boundary}--\r\n"
        );
        let ct = format!("multipart/form-data; boundary={boundary}");
        let req = parse_edit_request(Some(&ct), body.as_bytes(), &test_limits()).unwrap();
        assert_eq!(req.mask.as_deref(), Some(&[0u8, 1][..]));
        assert_eq!(req.references, vec![b"abc".to_vec()]);
    }

    #[test]
    fn generation_request_reads_background() {
        let limits = test_limits();
        let parse = |body: Value| {
            parse_generation_request(&serde_json::to_vec(&body).unwrap(), &limits)
                .map(|req| req.background)
        };
        assert_eq!(
            parse(json!({"prompt": "p"})).unwrap(),
            engine::ImageBackground::Auto
        );
        assert_eq!(
            parse(json!({"prompt": "p", "background": "opaque", "output_format": "jpeg"})).unwrap(),
            engine::ImageBackground::Opaque
        );
        for format in ["png", "webp"] {
            assert_eq!(
                parse(json!({"prompt": "p", "background": "transparent", "output_format": format}))
                    .unwrap(),
                engine::ImageBackground::Transparent
            );
        }
        for (bad, message) in [
            (
                json!({"prompt": "p", "background": "transparent", "output_format": "jpeg"}),
                "needs output_format png or webp",
            ),
            (
                json!({"prompt": "p", "background": "clear"}),
                "background \"clear\" is not supported",
            ),
            (
                json!({"prompt": "p", "background": 1}),
                "background must be a string",
            ),
        ] {
            let error = parse(bad.clone()).expect_err("refused");
            assert!(error.message.contains(message), "{bad}: {}", error.message);
        }
    }

    #[test]
    fn requests_read_stream_and_partial_images() {
        let limits = test_limits();
        let parse = |body: Value| {
            parse_generation_request(&serde_json::to_vec(&body).unwrap(), &limits)
                .map(|req| (req.stream, req.partial_images))
        };
        assert_eq!(parse(json!({"prompt": "p"})).unwrap(), (false, 0));
        assert_eq!(
            parse(json!({"prompt": "p", "stream": true, "partial_images": 3})).unwrap(),
            (true, 3)
        );
        for (bad, message) in [
            (
                json!({"prompt": "p", "stream": "yes"}),
                "stream must be a boolean",
            ),
            (
                json!({"prompt": "p", "stream": true, "partial_images": 4}),
                "partial_images must be between 0 and 3",
            ),
            (
                json!({"prompt": "p", "stream": true, "partial_images": -1}),
                "partial_images must be between 0 and 3",
            ),
        ] {
            let error = parse(bad.clone()).expect_err("refused");
            assert!(error.message.contains(message), "{bad}: {}", error.message);
        }

        let boundary = "b4";
        let form = |stream: &str| {
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\n{stream}\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"partial_images\"\r\n\r\n2\r\n\
                 --{boundary}--\r\n"
            )
        };
        let ct = format!("multipart/form-data; boundary={boundary}");
        let req = parse_edit_request(Some(&ct), form("true").as_bytes(), &limits).unwrap();
        assert_eq!((req.stream, req.partial_images), (true, 2));
        assert!(parse_edit_request(Some(&ct), form("maybe").as_bytes(), &limits).is_err());
    }

    /// Splits a server-sent event into its event name and its data's JSON.
    fn split_event(event: &str) -> (String, Value) {
        let (head, data) = event.split_once("\ndata: ").expect("an event and its data");
        assert!(data.ends_with("\n\n"), "{event}");
        let name = head.strip_prefix("event: ").expect("an event line");
        (
            name.to_string(),
            serde_json::from_str(data.trim_end()).expect("JSON data"),
        )
    }

    #[test]
    fn stream_events_name_the_endpoint_and_carry_the_image() {
        let edits = ImageStream {
            endpoint: ImageEndpoint::Edits,
            created_at: 7,
            format: engine::ImageFormat::Png,
            background: engine::ImageBackground::Transparent,
        };
        let (name, data) = split_event(&edits.format_partial_image(b"foo", 512, 384, 1));
        assert_eq!(name, "image_edit.partial_image");
        assert_eq!(
            data,
            json!({
                "type": "image_edit.partial_image",
                "b64_json": "Zm9v",
                "created_at": 7,
                "size": "512x384",
                "output_format": "png",
                "background": "transparent",
                "partial_image_index": 1,
            })
        );
        let generations = ImageStream {
            endpoint: ImageEndpoint::Generations,
            background: engine::ImageBackground::Auto,
            ..edits
        };
        let (name, data) = split_event(&generations.format_completed_image(b"bar", 64, 32));
        assert_eq!(name, "image_generation.completed");
        assert_eq!(
            data,
            json!({
                "type": "image_generation.completed",
                "b64_json": "YmFy",
                "created_at": 7,
                "size": "64x32",
                "output_format": "png",
                "background": "opaque",
            })
        );
        let (name, data) = split_event(&ImageStream::format_error(
            "server_error",
            "memory is short",
        ));
        assert_eq!(name, "error");
        assert_eq!(
            data,
            json!({
                "type": "error",
                "error": {"type": "server_error", "code": null, "message": "memory is short", "param": null},
            })
        );
    }

    #[test]
    fn worker_events_become_stream_events_until_the_last() {
        let stream = ImageStream {
            endpoint: ImageEndpoint::Generations,
            created_at: 7,
            format: engine::ImageFormat::Webp,
            background: engine::ImageBackground::Opaque,
        };
        let (partial, ends) = stream.format_worker_event(worker::Event::PartialImage {
            image: b"a".to_vec(),
            width: 8,
            height: 8,
            index: 0,
        });
        assert!(!ends);
        assert_eq!(
            split_event(&partial.unwrap()).0,
            "image_generation.partial_image"
        );
        let (completed, ends) = stream.format_worker_event(worker::Event::Image {
            image: b"b".to_vec(),
            width: 8,
            height: 8,
        });
        assert!(!ends);
        assert_eq!(
            split_event(&completed.unwrap()).0,
            "image_generation.completed"
        );
        assert_eq!(
            stream.format_worker_event(worker::Event::ImagesDone),
            (None, true)
        );
        for (event, kind, message) in [
            (
                worker::Event::Refused("too large".into()),
                "invalid_request_error",
                "too large",
            ),
            (
                worker::Event::Failed("memory is short".into()),
                "server_error",
                "memory is short",
            ),
            (
                worker::Event::Done {
                    reason: worker::FinishReason::ServerStop,
                },
                "server_error",
                "the server is stopping",
            ),
        ] {
            let (text, ends) = stream.format_worker_event(event);
            let (name, data) = split_event(&text.unwrap());
            assert!(ends);
            assert_eq!(name, "error");
            assert_eq!(
                (&data["error"]["type"], &data["error"]["message"]),
                (&json!(kind), &json!(message))
            );
        }
    }

    #[test]
    fn edit_form_reads_a_background() {
        let boundary = "b3";
        let form = |background: &str| {
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"background\"\r\n\r\n{background}\r\n\
                 --{boundary}--\r\n"
            )
        };
        let ct = format!("multipart/form-data; boundary={boundary}");
        let req = parse_edit_request(Some(&ct), form("transparent").as_bytes(), &test_limits());
        assert_eq!(
            req.unwrap().background,
            engine::ImageBackground::Transparent
        );
        assert!(parse_edit_request(Some(&ct), form("clear").as_bytes(), &test_limits()).is_err());
    }

    #[test]
    fn response_formats_b64_json_items() {
        let resp = build_response(
            &[b"foo".to_vec(), b"bar".to_vec()],
            512,
            384,
            engine::ImageFormat::Webp,
            engine::ImageBackground::Transparent,
        );
        assert!(resp["created"].is_u64());
        assert_eq!(resp["size"], "512x384");
        assert_eq!(resp["output_format"], "webp");
        assert_eq!(resp["background"], "transparent");
        let auto = build_response(
            &[],
            512,
            384,
            engine::ImageFormat::Png,
            engine::ImageBackground::Auto,
        );
        assert_eq!(auto["background"], "opaque");
        let data = resp["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["b64_json"], "Zm9v");
        assert_eq!(data[1]["b64_json"], "YmFy");
    }
}
