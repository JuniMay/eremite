//! Image generation and editing API (POST /v1/images/generations and POST
//! /v1/images/edits) for MiniMax-H3-Image. Parses JSON text-to-image requests
//! and multipart/form-data image editing requests, validates MiniMax-H3-Image
//! parameters, encodes generated PNG/JPEG bytes as base64, and formats OpenAI
//! image responses.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::{chat, engine, http};

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

/// RFC 4648 standard base64 alphabet.
const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// A validated image generation or editing request ready for the worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedRequest {
    /// Model name from the request, or None when omitted.
    pub model: Option<String>,
    /// Raw prompt string passed verbatim to MiniMax-H3-Image.
    pub prompt: String,
    /// Exact requested output width in pixels.
    pub width: i32,
    /// Exact requested output height in pixels.
    pub height: i32,
    /// Number of images to produce (`n`).
    pub count: usize,
    /// Deterministic RNG seed for the first image (`seed + i` for later ones).
    pub seed: u64,
    /// Minimum generation canvas area in pixels.
    pub min_generation_area: i32,
    /// Encoded output image format.
    pub format: engine::ImageFormat,
    /// JPEG compression quality in 1..=100.
    pub quality: i32,
    /// Reference image bytes in upload order (empty for text-to-image).
    pub references: Vec<Vec<u8>>,
}

/// Parses a `WIDTHxHEIGHT` size string into validated pixel dimensions.
fn parse_size(raw: &str, limits: &engine::ImageLimits) -> Result<(i32, i32), chat::Error> {
    let trimmed = raw.trim();
    let Some((w_str, h_str)) = trimmed.split_once(['x', 'X']) else {
        return Err(chat::Error::new_bad_request(
            "size must have the form WIDTHxHEIGHT",
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
    Ok((width, height))
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

/// Validates and parses the `output_format` value (`png`, `jpeg` or `jpg`).
fn parse_output_format(value: Option<&str>) -> Result<engine::ImageFormat, chat::Error> {
    match value {
        None => Ok(engine::ImageFormat::Png),
        Some(raw) if raw.eq_ignore_ascii_case("png") => Ok(engine::ImageFormat::Png),
        Some(raw) if raw.eq_ignore_ascii_case("jpeg") || raw.eq_ignore_ascii_case("jpg") => {
            Ok(engine::ImageFormat::Jpeg)
        }
        Some(other) => Err(chat::Error::new_bad_request(format!(
            "output_format {other:?} is not supported; use \"png\" or \"jpeg\""
        ))),
    }
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
/// /v1/images/generations.
pub fn parse_generation_request(body: &[u8]) -> Result<ParsedRequest, chat::Error> {
    let limits = engine::default_image_limits();
    let root: Value = serde_json::from_slice(body)
        .map_err(|error| chat::Error::new_bad_request(format!("not valid JSON: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| chat::Error::new_bad_request("request body must be a JSON object"))?;

    let model = read_optional_string(object, "model")?
        .filter(|s| !s.is_empty())
        .map(String::from);
    let prompt = read_optional_string(object, "prompt")?
        .ok_or_else(|| chat::Error::new_bad_request("prompt is required"))?
        .to_string();
    let size_raw = read_optional_string(object, "size")?
        .ok_or_else(|| chat::Error::new_bad_request("size is required (WIDTHxHEIGHT)"))?;
    let (width, height) = parse_size(size_raw, &limits)?;

    let count = read_optional_i64(object, "n", 1, limits.max_images_per_request as i64)?
        .unwrap_or(1) as usize;
    validate_total_pixels(width, height, count, &limits)?;
    let seed = match object.get("seed") {
        None | Some(Value::Null) => rand::random(),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| chat::Error::new_bad_request("seed must be a nonnegative integer"))?,
    };
    let _generator_device = read_optional_string(object, "generator_device")?;
    parse_response_format(read_optional_string(object, "response_format")?)?;
    let format = parse_output_format(read_optional_string(object, "output_format")?)?;
    let quality = read_optional_i64(
        object,
        "output_compression",
        limits.min_jpeg_quality as i64,
        limits.max_jpeg_quality as i64,
    )?
    .unwrap_or(limits.default_jpeg_quality as i64) as i32;
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
        width,
        height,
        count,
        seed,
        min_generation_area,
        format,
        quality,
        references: Vec::new(),
    })
}

/// Extracts the multipart boundary token from a Content-Type header value.
fn extract_multipart_boundary(content_type: &str) -> Result<String, chat::Error> {
    let mut parts = content_type.split(';');
    let media_type = parts.next().unwrap_or_default().trim();
    if !media_type.eq_ignore_ascii_case("multipart/form-data") {
        return Err(chat::Error::new_bad_request(
            "Content-Type must be multipart/form-data",
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

/// Parses and validates a multipart/form-data image editing request for POST
/// /v1/images/edits. Reference images are collected from `image` and `image[]`
/// parts in upload order (`<Picture 1>`, `<Picture 2>`, ...).
pub fn parse_edit_request(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<ParsedRequest, chat::Error> {
    let limits = engine::default_image_limits();
    let content_type = content_type.ok_or_else(|| {
        chat::Error::new_bad_request("Content-Type: multipart/form-data is required")
    })?;
    let boundary = extract_multipart_boundary(content_type)?;
    let parts = parse_multipart(&boundary, body)?;

    let mut fields = HashMap::<String, String>::new();
    let mut references = Vec::<Vec<u8>>::new();
    for part in parts {
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
    if references.is_empty() {
        return Err(chat::Error::new_bad_request(
            "at least one reference image is required",
        ));
    }
    if references.len() > limits.max_references as usize {
        return Err(chat::Error::new_bad_request(format!(
            "at most {} reference images are supported",
            limits.max_references
        )));
    }

    let model = fields
        .get("model")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);
    let prompt = fields
        .get("prompt")
        .ok_or_else(|| chat::Error::new_bad_request("prompt is required"))?
        .clone();
    let size_raw = fields
        .get("size")
        .ok_or_else(|| chat::Error::new_bad_request("size is required (WIDTHxHEIGHT)"))?;
    let (width, height) = parse_size(size_raw, &limits)?;

    let count = parse_form_i64(&fields, "n", 1, limits.max_images_per_request as i64)?.unwrap_or(1)
        as usize;
    validate_total_pixels(width, height, count, &limits)?;
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
    let quality = parse_form_i64(
        &fields,
        "output_compression",
        limits.min_jpeg_quality as i64,
        limits.max_jpeg_quality as i64,
    )?
    .unwrap_or(limits.default_jpeg_quality as i64) as i32;
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
        let ok = raw
            .parse::<f64>()
            .ok()
            .is_some_and(|scale| scale.is_finite() && (scale - limits.guidance_scale).abs() == 0.0);
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
        width,
        height,
        count,
        seed,
        min_generation_area,
        format,
        quality,
        references,
    })
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
/// `b64_json` items.
pub fn build_response(images: &[Vec<u8>]) -> Value {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let req = parse_generation_request(&body).unwrap();
        assert_eq!(req.model, None);
        assert_eq!(req.prompt, "A red ceramic mug on wood.");
        assert_eq!((req.width, req.height), (4736, 3552));
        assert_eq!(req.count, 1);
        assert_eq!(req.seed, 42);
        let limits = engine::default_image_limits();
        assert_eq!(req.min_generation_area, limits.default_generation_area);
        assert_eq!(req.format, engine::ImageFormat::Png);
        assert_eq!(req.quality, limits.default_jpeg_quality);
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
        let req = parse_generation_request(&jpg_body).unwrap();
        assert_eq!(req.model.as_deref(), Some("minimax-h3-image"));
        assert_eq!((req.width, req.height), (512, 512));
        assert_eq!(req.count, 2);
        assert_eq!(req.format, engine::ImageFormat::Jpeg);
        assert_eq!(req.quality, 85);
        assert_eq!(req.min_generation_area, 1024);
    }

    #[test]
    fn generation_request_rejects_invalid_and_unsupported_parameters() {
        for bad in [
            json!({"size": "512x512"}),
            json!({"prompt": "hi"}),
            json!({"prompt": "hi", "size": "512"}),
            json!({"prompt": "hi", "size": "0x512"}),
            json!({"prompt": "hi", "size": "9000x512"}),
            json!({"prompt": "hi", "size": "512x512", "n": 0}),
            json!({"prompt": "hi", "size": "512x512", "n": 17}),
            json!({"prompt": "hi", "size": "512x512", "seed": -1}),
            json!({"prompt": "hi", "size": "512x512", "response_format": "url"}),
            json!({"prompt": "hi", "size": "512x512", "output_format": "webp"}),
            json!({"prompt": "hi", "size": "512x512", "output_compression": 0}),
            json!({"prompt": "hi", "size": "512x512", "min_generation_area": 512}),
            json!({"prompt": "hi", "size": "512x512", "min_generation_area": 20_000_000}),
            json!({"prompt": "hi", "size": "512x512", "num_inference_steps": 8}),
            json!({"prompt": "hi", "size": "512x512", "guidance_scale": 7.5}),
            json!({"prompt": "hi", "size": "512x512", "negative_prompt": "blurry"}),
            json!({"prompt": "hi", "size": "4096x4096", "n": 16}),
        ] {
            let bytes = serde_json::to_vec(&bad).unwrap();
            assert!(
                parse_generation_request(&bytes).is_err(),
                "expected err: {bad}"
            );
        }
    }

    #[test]
    fn edit_request_parses_multipart_fields_and_preserves_image_order() {
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
        let req = parse_edit_request(Some(&ct), &body).unwrap();
        assert_eq!(req.model.as_deref(), Some("minimax-h3-image"));
        assert_eq!(req.prompt, "Combine <Picture 1> and <Picture 2>.");
        assert_eq!((req.width, req.height), (1024, 1024));
        assert_eq!(req.seed, 1234);
        assert_eq!(req.min_generation_area, 1024);
        assert_eq!(req.references, vec![img1.to_vec(), img2.to_vec()]);
    }

    #[test]
    fn edit_request_rejects_missing_images_and_bad_steps() {
        let boundary = "b1";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n512x512\r\n\
             --{boundary}--\r\n"
        );
        let ct = format!("multipart/form-data; boundary={boundary}");
        assert!(parse_edit_request(Some(&ct), body.as_bytes()).is_err());

        let bad_steps = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n512x512\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"num_inference_steps\"\r\n\r\n20\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
             --{boundary}--\r\n"
        );
        assert!(parse_edit_request(Some(&ct), bad_steps.as_bytes()).is_err());

        let too_many_pixels = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n4096x4096\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n3\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nabc\r\n\
             --{boundary}--\r\n"
        );
        assert!(parse_edit_request(Some(&ct), too_many_pixels.as_bytes()).is_err());
    }

    #[test]
    fn response_formats_b64_json_items() {
        let resp = build_response(&[b"foo".to_vec(), b"bar".to_vec()]);
        assert!(resp["created"].is_u64());
        let data = resp["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["b64_json"], "Zm9v");
        assert_eq!(data[1]["b64_json"], "YmFy");
    }
}
