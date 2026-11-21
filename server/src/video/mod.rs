//! OpenAI Videos API (POST /v1/videos, GET /v1/videos, GET /v1/videos/{id},
//! GET /v1/videos/{id}/content and DELETE /v1/videos/{id}) for MiniMax-H3
//! Video. Parses JSON and multipart/form-data creation requests, validates them
//! against the selected model's `engine::VideoLimits`, tracks asynchronous
//! video jobs with live step progress, and stores finished MP4 clips on disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::{chat, engine, http, worker};

/// Default served model id for MiniMax-H3 Video when `id` is omitted.
pub const DEFAULT_MODEL_ID: &str = "minimax-h3-video";
/// Display name shown in GET /v1/models for MiniMax-H3 Video.
pub const DISPLAY_NAME: &str = "MiniMax-H3 Video (eremite)";
/// Maximum body size for video creation requests (256 MiB for up to 3
/// reference videos plus reference images and audio clips).
pub const MAX_VIDEO_BODY_BYTES: usize = 256 * http::MIB;

/// CRLF blank-line separator between multipart headers and part body.
const CRLF_HEADER_END: &[u8] = b"\r\n\r\n";
/// LF blank-line separator between multipart headers and part body.
const LF_HEADER_END: &[u8] = b"\n\n";

/// Default and maximum page sizes for GET /v1/videos.
const DEFAULT_LIST_LIMIT: usize = 20;
/// Maximum videos returned in one GET /v1/videos page.
const MAX_LIST_LIMIT: usize = 100;

/// Progress percentage reported while the prompt encoder prefills.
const PREFILL_PROGRESS_PCT: u32 = 5;
/// Base percentage when DiT denoising begins.
const GEN_PROGRESS_BASE_PCT: u32 = 10;
/// Percentage span allocated across DiT denoising steps.
const GEN_PROGRESS_SPAN_PCT: u32 = 85;
/// Upper percentage cap before the finished MP4 is written to disk.
const MAX_IN_PROGRESS_PCT: u32 = 95;
/// Completed job percentage.
const COMPLETED_PROGRESS_PCT: u32 = 100;

/// Unvalidated video creation payload parsed from JSON or multipart/form-data.
/// Validated against the selected model's `engine::VideoLimits` via
/// [`RawVideoSubmission::validate`].
#[derive(Debug)]
pub struct RawVideoSubmission {
    /// Requested model id, or None when omitted.
    model: Option<String>,
    /// Raw prompt text, or None if missing.
    prompt: Option<String>,
    /// Raw `size` string (`WIDTHxHEIGHT` or `auto`).
    size: Option<String>,
    /// Raw `seconds` value as string or integer.
    seconds: Option<String>,
    /// Raw `frames` value as string or integer.
    frames: Option<String>,
    /// Raw `seed` value as string or integer.
    seed: Option<String>,
    /// Raw `steps` or `num_inference_steps` value.
    steps: Option<String>,
    /// Raw `shift_video` value.
    shift_video: Option<String>,
    /// Raw `shift_audio` value.
    shift_audio: Option<String>,
    /// Raw `guidance_scale` value.
    guidance_scale: Option<String>,
    /// Raw `negative_prompt` value.
    negative_prompt: Option<String>,
    /// Raw `ref_image_size` value (`auto` or `max`).
    ref_image_size: Option<String>,
    /// OpenAI standard `input_reference` media bytes.
    input_reference: Option<Vec<u8>>,
    /// Explicit FL2VA first keyframe bytes.
    first_frame: Option<Vec<u8>>,
    /// Explicit FL2VA last keyframe bytes.
    last_frame: Option<Vec<u8>>,
    /// Explicit REF2VA reference images in upload order.
    ref_images: Vec<Vec<u8>>,
    /// Explicit REF2VA reference videos in upload order.
    ref_videos: Vec<Vec<u8>>,
    /// Optional explicit soundtracks paired with `ref_videos`.
    ref_video_audios: Vec<Option<Vec<u8>>>,
    /// Explicit REF2VA standalone reference audios in upload order.
    ref_audios: Vec<Vec<u8>>,
}

impl RawVideoSubmission {
    /// Returns the requested model name, if supplied.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Validates this submission against the selected video model's `limits`.
    pub fn validate(
        mut self,
        limits: &engine::VideoLimits,
    ) -> Result<ParsedVideoRequest, chat::Error> {
        let prompt = self
            .prompt
            .filter(|p| !p.is_empty())
            .ok_or_else(|| chat::Error::new_bad_request("prompt is required"))?;

        let (width, height) = parse_size(self.size.as_deref(), limits)?;
        let seconds = parse_optional_i32(
            "seconds",
            self.seconds.as_deref(),
            limits.min_seconds,
            limits.max_seconds,
        )?
        .unwrap_or(0);
        let frames = parse_optional_i32("frames", self.frames.as_deref(), 1, limits.max_frames)?
            .unwrap_or(0);
        if seconds > 0 && frames > 0 {
            return Err(chat::Error::new_bad_request(
                "specify seconds or frames, not both",
            ));
        }

        let seed = match self
            .seed
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            None => rand::random(),
            Some(raw) => raw
                .parse::<u64>()
                .map_err(|_| chat::Error::new_bad_request("seed must be a nonnegative integer"))?,
        };
        let steps = parse_optional_i32(
            "steps",
            self.steps.as_deref(),
            limits.min_steps,
            limits.max_steps,
        )?
        .unwrap_or(0);
        let shift_video =
            parse_optional_positive_f64("shift_video", self.shift_video.as_deref())?.unwrap_or(0.0);
        let shift_audio =
            parse_optional_positive_f64("shift_audio", self.shift_audio.as_deref())?.unwrap_or(0.0);

        if let Some(raw) = self
            .guidance_scale
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let ok = raw
                .parse::<f64>()
                .ok()
                .is_some_and(|v| v.is_finite() && (v - limits.guidance_scale).abs() == 0.0);
            if !ok {
                return Err(chat::Error::new_bad_request(format!(
                    "guidance_scale must be omitted or {}",
                    limits.guidance_scale
                )));
            }
        }
        if let Some(neg) = self.negative_prompt.as_deref()
            && !neg.is_empty()
        {
            return Err(chat::Error::new_bad_request(
                "negative_prompt is not supported",
            ));
        }

        let ref_image_max = match self
            .ref_image_size
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            None => false,
            Some(raw) if raw.eq_ignore_ascii_case("auto") => false,
            Some(raw) if raw.eq_ignore_ascii_case("max") => true,
            Some(other) => {
                return Err(chat::Error::new_bad_request(format!(
                    "ref_image_size {other:?} is not supported; use \"auto\" or \"max\""
                )));
            }
        };

        let variant = limits.task_variant();
        if let Some(input_ref) = self.input_reference.take() {
            match variant {
                engine::VideoVariant::Fl2va => {
                    if self.first_frame.is_some() {
                        return Err(chat::Error::new_bad_request(
                            "specify input_reference or first_frame, not both",
                        ));
                    }
                    self.first_frame = Some(input_ref);
                }
                engine::VideoVariant::Ref2va => {
                    self.ref_images.insert(0, input_ref);
                }
            }
        }

        let has_fl2va = self.first_frame.is_some() || self.last_frame.is_some();
        let has_refs = !self.ref_images.is_empty()
            || !self.ref_videos.is_empty()
            || !self.ref_audios.is_empty()
            || !self.ref_video_audios.is_empty();
        if variant == engine::VideoVariant::Fl2va && has_refs {
            return Err(chat::Error::new_bad_request(
                "reference images, videos and audio require a REF2VA model",
            ));
        }
        if variant == engine::VideoVariant::Ref2va && has_fl2va {
            return Err(chat::Error::new_bad_request(
                "first_frame and last_frame require an FL2VA model",
            ));
        }
        if self.ref_video_audios.len() > self.ref_videos.len() {
            return Err(chat::Error::new_bad_request(
                "ref_video_audios cannot have more items than ref_videos",
            ));
        }
        let total_refs = self.ref_images.len() + self.ref_videos.len() + self.ref_audios.len();
        if self.ref_images.len() > limits.max_ref_images as usize
            || self.ref_videos.len() > limits.max_ref_videos as usize
            || self.ref_audios.len() > limits.max_ref_audios as usize
            || total_refs > limits.max_total_refs as usize
        {
            return Err(chat::Error::new_bad_request(format!(
                "reference counts exceed limits (images <= {}, videos <= {}, audios <= {}, total <= {})",
                limits.max_ref_images,
                limits.max_ref_videos,
                limits.max_ref_audios,
                limits.max_total_refs
            )));
        }

        let display_size = if width > 0 && height > 0 {
            format!("{width}x{height}")
        } else {
            format!("{}x{}", limits.default_width, limits.default_height)
        };
        let display_seconds = if seconds > 0 {
            seconds.to_string()
        } else if frames > 0 && limits.fps > 0 {
            ((frames as f64 / limits.fps as f64).round() as i32)
                .max(1)
                .to_string()
        } else {
            limits.default_seconds.to_string()
        };

        Ok(ParsedVideoRequest {
            prompt,
            width,
            height,
            seconds,
            frames,
            steps,
            shift_video,
            shift_audio,
            seed,
            first_frame: self.first_frame,
            last_frame: self.last_frame,
            ref_images: self.ref_images,
            ref_videos: self.ref_videos,
            ref_video_audios: self.ref_video_audios,
            ref_audios: self.ref_audios,
            ref_image_max,
            display_size,
            display_seconds,
        })
    }
}

/// Validated video generation request ready for the worker.
#[derive(Debug)]
pub struct ParsedVideoRequest {
    /// Raw prompt string passed verbatim to MiniMax-H3 Video.
    pub prompt: String,
    /// Requested width in pixels (0 adapts or uses model default).
    pub width: i32,
    /// Requested height in pixels (0 adapts or uses model default).
    pub height: i32,
    /// Requested duration in whole seconds (0 uses frames or default).
    pub seconds: i32,
    /// Requested explicit frame count (0 uses seconds or default).
    pub frames: i32,
    /// Requested denoising steps (0 uses model default).
    pub steps: i32,
    /// Video shift override (0.0 uses model default).
    pub shift_video: f64,
    /// Audio shift override (0.0 uses model default).
    pub shift_audio: f64,
    /// Deterministic RNG seed.
    pub seed: u64,
    /// Optional FL2VA first keyframe bytes.
    pub first_frame: Option<Vec<u8>>,
    /// Optional FL2VA last keyframe bytes.
    pub last_frame: Option<Vec<u8>>,
    /// Optional REF2VA reference images.
    pub ref_images: Vec<Vec<u8>>,
    /// Optional REF2VA reference videos.
    pub ref_videos: Vec<Vec<u8>>,
    /// Optional explicit soundtracks paired with `ref_videos`.
    pub ref_video_audios: Vec<Option<Vec<u8>>>,
    /// Optional REF2VA standalone audios.
    pub ref_audios: Vec<Vec<u8>>,
    /// True in REF2VA to size reference images to a 2048 short edge.
    pub ref_image_max: bool,
    /// Initial `size` text shown in the queued job JSON.
    pub display_size: String,
    /// Initial `seconds` text shown in the queued job JSON.
    pub display_seconds: String,
}

impl ParsedVideoRequest {
    /// Converts this validated request into a worker job payload for
    /// `model_index`.
    pub fn into_worker_request(self, model_index: usize, note: String) -> worker::VideoRequest {
        worker::VideoRequest {
            model_index,
            prompt: self.prompt,
            width: self.width,
            height: self.height,
            seconds: self.seconds,
            frames: self.frames,
            steps: self.steps,
            shift_video: self.shift_video,
            shift_audio: self.shift_audio,
            seed: self.seed,
            first_frame: self.first_frame,
            last_frame: self.last_frame,
            ref_images: self.ref_images,
            ref_videos: self.ref_videos,
            ref_video_audios: self.ref_video_audios,
            ref_audios: self.ref_audios,
            ref_image_max: self.ref_image_max,
            note,
        }
    }
}

/// Parses a POST /v1/videos request body (application/json or
/// multipart/form-data) without validating model-specific limits yet.
pub fn parse_create_body(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<RawVideoSubmission, chat::Error> {
    let media_type = content_type
        .unwrap_or("application/json")
        .split(';')
        .next()
        .unwrap_or_default()
        .trim();
    if media_type.is_empty() || media_type.eq_ignore_ascii_case("application/json") {
        return parse_json_body(body);
    }
    if media_type.eq_ignore_ascii_case("multipart/form-data") {
        let boundary = extract_multipart_boundary(content_type.unwrap_or_default())?;
        return parse_multipart_body(body, &boundary);
    }
    Err(chat::Error::new_bad_request(
        "Content-Type must be application/json or multipart/form-data",
    ))
}

/// Parses an application/json POST /v1/videos body into a raw submission.
fn parse_json_body(body: &[u8]) -> Result<RawVideoSubmission, chat::Error> {
    let root: Value = serde_json::from_slice(body)
        .map_err(|error| chat::Error::new_bad_request(format!("not valid JSON: {error}")))?;
    let Value::Object(object) = root else {
        return Err(chat::Error::new_bad_request(
            "request body must be a JSON object",
        ));
    };
    let steps =
        read_json_scalar(&object, "steps")?.or(read_json_scalar(&object, "num_inference_steps")?);
    let mut ref_images = read_json_media_list(&object, "ref_images")?;
    if let Some(single) = read_json_media_opt(&object, "ref_image")? {
        ref_images.insert(0, single);
    }
    let mut ref_videos = read_json_media_list(&object, "ref_videos")?;
    if let Some(single) = read_json_media_opt(&object, "ref_video")? {
        ref_videos.insert(0, single);
    }
    let mut ref_audios = read_json_media_list(&object, "ref_audios")?;
    if let Some(single) = read_json_media_opt(&object, "ref_audio")? {
        ref_audios.insert(0, single);
    }
    let mut ref_video_audios = read_json_opt_media_list(&object, "ref_video_audios")?;
    if let Some(single) = read_json_media_opt(&object, "ref_video_audio")? {
        ref_video_audios.insert(0, Some(single));
    }

    Ok(RawVideoSubmission {
        model: read_json_string(&object, "model")?.filter(|s| !s.is_empty()),
        prompt: read_json_string(&object, "prompt")?,
        size: read_json_string(&object, "size")?,
        seconds: read_json_scalar(&object, "seconds")?,
        frames: read_json_scalar(&object, "frames")?,
        seed: read_json_scalar(&object, "seed")?,
        steps,
        shift_video: read_json_scalar(&object, "shift_video")?,
        shift_audio: read_json_scalar(&object, "shift_audio")?,
        guidance_scale: read_json_scalar(&object, "guidance_scale")?,
        negative_prompt: read_json_string(&object, "negative_prompt")?,
        ref_image_size: read_json_string(&object, "ref_image_size")?,
        input_reference: read_json_media_opt(&object, "input_reference")?,
        first_frame: read_json_media_opt(&object, "first_frame")?,
        last_frame: read_json_media_opt(&object, "last_frame")?,
        ref_images,
        ref_videos,
        ref_video_audios,
        ref_audios,
    })
}

/// Parses a multipart/form-data POST /v1/videos body into a raw submission.
fn parse_multipart_body(body: &[u8], boundary: &str) -> Result<RawVideoSubmission, chat::Error> {
    let parts = parse_multipart(body, boundary)?;
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut input_reference = None;
    let mut first_frame = None;
    let mut last_frame = None;
    let mut ref_images = Vec::new();
    let mut ref_videos = Vec::new();
    let mut ref_video_audios = Vec::new();
    let mut ref_audios = Vec::new();

    for part in parts {
        match part.name.as_str() {
            "input_reference" => {
                if !part.data.is_empty() {
                    input_reference = Some(part.data.to_vec());
                }
            }
            "first_frame" => {
                if !part.data.is_empty() {
                    first_frame = Some(part.data.to_vec());
                }
            }
            "last_frame" => {
                if !part.data.is_empty() {
                    last_frame = Some(part.data.to_vec());
                }
            }
            "ref_image" | "ref_images" | "ref_images[]" => {
                if !part.data.is_empty() {
                    ref_images.push(part.data.to_vec());
                }
            }
            "ref_video" | "ref_videos" | "ref_videos[]" => {
                if !part.data.is_empty() {
                    ref_videos.push(part.data.to_vec());
                }
            }
            "ref_video_audio" | "ref_video_audios" | "ref_video_audios[]" => {
                if part.data.is_empty() {
                    ref_video_audios.push(None);
                } else {
                    ref_video_audios.push(Some(part.data.to_vec()));
                }
            }
            "ref_audio" | "ref_audios" | "ref_audios[]" => {
                if !part.data.is_empty() {
                    ref_audios.push(part.data.to_vec());
                }
            }
            _ => {
                let text = std::str::from_utf8(part.data).map_err(|_| {
                    chat::Error::new_bad_request(format!("form field {} must be UTF-8", part.name))
                })?;
                fields.insert(part.name, text.to_string());
            }
        }
    }

    let steps = fields
        .remove("steps")
        .or_else(|| fields.remove("num_inference_steps"));
    Ok(RawVideoSubmission {
        model: fields.remove("model").filter(|s| !s.is_empty()),
        prompt: fields.remove("prompt"),
        size: fields.remove("size"),
        seconds: fields.remove("seconds"),
        frames: fields.remove("frames"),
        seed: fields.remove("seed"),
        steps,
        shift_video: fields.remove("shift_video"),
        shift_audio: fields.remove("shift_audio"),
        guidance_scale: fields.remove("guidance_scale"),
        negative_prompt: fields.remove("negative_prompt"),
        ref_image_size: fields.remove("ref_image_size"),
        input_reference,
        first_frame,
        last_frame,
        ref_images,
        ref_videos,
        ref_video_audios,
        ref_audios,
    })
}

/// Parses `size` against `limits` (`auto` or omitted returns `(0, 0)`).
fn parse_size(raw: Option<&str>, limits: &engine::VideoLimits) -> Result<(i32, i32), chat::Error> {
    let trimmed = raw.map_or("", str::trim);
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return Ok((0, 0));
    }
    let Some((w_str, h_str)) = trimmed.split_once(['x', 'X']) else {
        return Err(chat::Error::new_bad_request(
            "size must be auto or have the form WIDTHxHEIGHT",
        ));
    };
    let width: i32 = w_str
        .trim()
        .parse()
        .map_err(|_| chat::Error::new_bad_request("size width must be a positive integer"))?;
    let height: i32 = h_str
        .trim()
        .parse()
        .map_err(|_| chat::Error::new_bad_request("size height must be a positive integer"))?;
    if !(1..=limits.max_dim).contains(&width) || !(1..=limits.max_dim).contains(&height) {
        return Err(chat::Error::new_bad_request(format!(
            "size dimensions must be between 1 and {}",
            limits.max_dim
        )));
    }
    if (width as i64).saturating_mul(height as i64) > limits.max_canvas_area {
        return Err(chat::Error::new_bad_request(format!(
            "size {width}x{height} exceeds {} pixel limit",
            limits.max_canvas_area
        )));
    }
    Ok((width, height))
}

/// Parses an optional integer field within `[min, max]`.
fn parse_optional_i32(
    name: &str,
    raw: Option<&str>,
    min: i32,
    max: i32,
) -> Result<Option<i32>, chat::Error> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let v: i32 = s
        .parse()
        .map_err(|_| chat::Error::new_bad_request(format!("{name} must be an integer")))?;
    if !(min..=max).contains(&v) {
        return Err(chat::Error::new_bad_request(format!(
            "{name} must be between {min} and {max}"
        )));
    }
    Ok(Some(v))
}

/// Parses an optional positive finite floating-point field.
fn parse_optional_positive_f64(name: &str, raw: Option<&str>) -> Result<Option<f64>, chat::Error> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let v: f64 = s
        .parse()
        .map_err(|_| chat::Error::new_bad_request(format!("{name} must be a number")))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(chat::Error::new_bad_request(format!(
            "{name} must be a positive number"
        )));
    }
    Ok(Some(v))
}

/// Reads an optional string field from a JSON object.
fn read_json_string(object: &Map<String, Value>, key: &str) -> Result<Option<String>, chat::Error> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(chat::Error::new_bad_request(format!(
            "{key} must be a string"
        ))),
    }
}

/// Reads an optional JSON string or numeric scalar as text.
fn read_json_scalar(object: &Map<String, Value>, key: &str) -> Result<Option<String>, chat::Error> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(Value::Number(n)) => Ok(Some(n.to_string())),
        Some(_) => Err(chat::Error::new_bad_request(format!(
            "{key} must be a number or string"
        ))),
    }
}

/// Reads one optional base64 or data-URL encoded media item from JSON.
fn read_json_media_opt(
    object: &Map<String, Value>,
    key: &str,
) -> Result<Option<Vec<u8>>, chat::Error> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let text = value.as_str().ok_or_else(|| {
        chat::Error::new_bad_request(format!("{key} must be a base64 string or data URL"))
    })?;
    decode_media_base64(key, text).map(Some)
}

/// Reads a JSON array (or single string) of base64/data-URL media items.
fn read_json_media_list(
    object: &Map<String, Value>,
    key: &str,
) -> Result<Vec<Vec<u8>>, chat::Error> {
    let Some(value) = object.get(key) else {
        return Ok(Vec::new());
    };
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => Ok(vec![decode_media_base64(key, text)?]),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let text = item.as_str().ok_or_else(|| {
                    chat::Error::new_bad_request(format!(
                        "{key}[{i}] must be a base64 string or data URL"
                    ))
                })?;
                decode_media_base64(&format!("{key}[{i}]"), text)
            })
            .collect(),
        _ => Err(chat::Error::new_bad_request(format!(
            "{key} must be an array of base64 strings or data URLs"
        ))),
    }
}

/// Reads a JSON array of optional base64/data-URL media items (null keeps a
/// slot empty).
fn read_json_opt_media_list(
    object: &Map<String, Value>,
    key: &str,
) -> Result<Vec<Option<Vec<u8>>>, chat::Error> {
    let Some(value) = object.get(key) else {
        return Ok(Vec::new());
    };
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(i, item)| match item {
                Value::Null => Ok(None),
                Value::String(text) if text.is_empty() => Ok(None),
                Value::String(text) => decode_media_base64(&format!("{key}[{i}]"), text).map(Some),
                _ => Err(chat::Error::new_bad_request(format!(
                    "{key}[{i}] must be null or a base64 string"
                ))),
            })
            .collect(),
        _ => Err(chat::Error::new_bad_request(format!(
            "{key} must be an array of base64 strings or nulls"
        ))),
    }
}

/// Decodes a standard base64 string or `data:<mime>;base64,<payload>` URL.
fn decode_media_base64(field: &str, raw: &str) -> Result<Vec<u8>, chat::Error> {
    let trimmed = raw.trim();
    let payload = if let Some(rest) = trimmed.strip_prefix("data:") {
        let Some((meta, data)) = rest.split_once(',') else {
            return Err(chat::Error::new_bad_request(format!(
                "{field} has an invalid data URL"
            )));
        };
        if !meta.ends_with(";base64") {
            return Err(chat::Error::new_bad_request(format!(
                "{field} data URL must use base64 encoding"
            )));
        }
        data
    } else {
        trimmed
    };
    let bytes = decode_base64(payload)
        .ok_or_else(|| chat::Error::new_bad_request(format!("{field} is not valid base64")))?;
    if bytes.is_empty() {
        return Err(chat::Error::new_bad_request(format!(
            "{field} must not be empty"
        )));
    }
    Ok(bytes)
}

/// Decodes RFC 4648 base64 ignoring ASCII whitespace, or returns None on
/// malformed input.
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    let mut vals = Vec::with_capacity(input.len());
    let mut pad = 0usize;
    for b in input.bytes() {
        if b.is_ascii_whitespace() {
            continue;
        }
        if b == b'=' {
            pad += 1;
            continue;
        }
        if pad > 0 {
            return None;
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        };
        vals.push(v);
    }
    let total = vals.len() + pad;
    if pad > 2 || !total.is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity((total / 4) * 3);
    let mut i = 0;
    while i + 4 <= vals.len() {
        let n = ((vals[i] as u32) << 18)
            | ((vals[i + 1] as u32) << 12)
            | ((vals[i + 2] as u32) << 6)
            | (vals[i + 3] as u32);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
        i += 4;
    }
    let rem = vals.len() - i;
    if rem == 2 {
        let n = ((vals[i] as u32) << 18) | ((vals[i + 1] as u32) << 12);
        out.push((n >> 16) as u8);
    } else if rem == 3 {
        let n =
            ((vals[i] as u32) << 18) | ((vals[i + 1] as u32) << 12) | ((vals[i + 2] as u32) << 6);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
    } else if rem != 0 {
        return None;
    }
    Some(out)
}

/// Extracts the multipart boundary token from a Content-Type header value.
fn extract_multipart_boundary(content_type: &str) -> Result<String, chat::Error> {
    for param in content_type.split(';').skip(1) {
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

/// Extracts the quoted or bare `name=` attribute from Content-Disposition.
fn extract_disposition_name(headers: &str) -> Option<String> {
    for line in headers.lines() {
        let Some((header_name, rest)) = line.split_once(':') else {
            continue;
        };
        if !header_name
            .trim()
            .eq_ignore_ascii_case("content-disposition")
        {
            continue;
        }
        for param in rest.split(';').skip(1) {
            let Some((k, v)) = param.split_once('=') else {
                continue;
            };
            if k.trim().eq_ignore_ascii_case("name") {
                let v = v.trim();
                let unquoted = v
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .unwrap_or(v);
                return Some(unquoted.to_string());
            }
        }
    }
    None
}

/// Parses a multipart/form-data body delimited by `boundary`.
fn parse_multipart<'a>(
    body: &'a [u8],
    boundary: &str,
) -> Result<Vec<MultipartPart<'a>>, chat::Error> {
    let delimiter = format!("--{boundary}");
    let delim = delimiter.as_bytes();
    let Some(first) = find_subslice(body, delim) else {
        return Err(chat::Error::new_bad_request(
            "multipart body does not contain the declared boundary",
        ));
    };
    let mut cursor = first + delim.len();
    let mut parts = Vec::new();
    loop {
        if body[cursor..].starts_with(b"--") {
            break;
        }
        if body[cursor..].starts_with(b"\r\n") {
            cursor += 2;
        } else if body[cursor..].starts_with(b"\n") {
            cursor += 1;
        } else {
            return Err(chat::Error::new_bad_request("malformed multipart boundary"));
        }
        let Some(next_rel) = find_subslice(&body[cursor..], delim) else {
            return Err(chat::Error::new_bad_request(
                "unterminated multipart/form-data body",
            ));
        };
        let raw_part = &body[cursor..cursor + next_rel];
        cursor += next_rel + delim.len();
        let trimmed_part = raw_part
            .strip_suffix(b"\r\n")
            .or_else(|| raw_part.strip_suffix(b"\n"))
            .unwrap_or(raw_part);
        let Some((header_bytes, payload)) = find_subslice(trimmed_part, CRLF_HEADER_END)
            .map(|pos| {
                (
                    &trimmed_part[..pos],
                    &trimmed_part[pos + CRLF_HEADER_END.len()..],
                )
            })
            .or_else(|| {
                find_subslice(trimmed_part, LF_HEADER_END).map(|pos| {
                    (
                        &trimmed_part[..pos],
                        &trimmed_part[pos + LF_HEADER_END.len()..],
                    )
                })
            })
        else {
            return Err(chat::Error::new_bad_request(
                "multipart part is missing header separator",
            ));
        };
        let headers = std::str::from_utf8(header_bytes)
            .map_err(|_| chat::Error::new_bad_request("multipart headers must be UTF-8"))?;
        if let Some(name) = extract_disposition_name(headers) {
            parts.push(MultipartPart {
                name,
                data: payload,
            });
        }
    }
    Ok(parts)
}

/// Lifecycle state of a video generation job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    /// Queued in the worker waiting for the GPU.
    Queued,
    /// Running prompt encoding, DiT denoising, VAE decode or MP4 muxing.
    InProgress,
    /// Finished and MP4 saved on disk.
    Completed,
    /// Failed, refused or cancelled.
    Failed,
}

impl JobStatus {
    /// Returns the OpenAI Videos API status name.
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::InProgress => "in_progress",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        }
    }
}

/// Structured error attached to a failed video job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobError {
    /// Machine-readable error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
}

/// Public snapshot of one stored video generation job.
#[derive(Clone, Debug)]
pub struct VideoJob {
    /// Unique OpenAI-style video job identifier (`video_<hex>`).
    pub id: String,
    /// Served model id that owns this job.
    pub model: String,
    /// Served model index in the worker.
    pub model_index: usize,
    /// Worker request id used to poll live step progress.
    pub worker_request_id: u64,
    /// Prompt text submitted by the client.
    pub prompt: String,
    /// Current lifecycle status.
    pub status: JobStatus,
    /// Completion percentage in `0..=100`.
    pub progress: u32,
    /// Video dimensions formatted as `WIDTHxHEIGHT`.
    pub size: String,
    /// Video duration in whole seconds as a string.
    pub seconds: String,
    /// Actual generated frame count once completed.
    pub frames: Option<i32>,
    /// Actual generated frame rate once completed.
    pub fps: Option<i32>,
    /// Unix timestamp in seconds when the job was created.
    pub created_at: u64,
    /// Unix timestamp in seconds when the job completed.
    pub completed_at: Option<u64>,
    /// Error object when `status == JobStatus::Failed`.
    pub error: Option<JobError>,
    /// Disk path of the finished MP4 file once completed.
    pub content_path: Option<PathBuf>,
}

impl VideoJob {
    /// Formats this video job as an OpenAI Videos API JSON object.
    pub fn to_json(&self) -> Value {
        let error = self.error.as_ref().map(|err| {
            json!({
                "code": err.code,
                "message": err.message,
            })
        });
        json!({
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status.as_str(),
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": Value::Null,
            "size": self.size,
            "seconds": self.seconds,
            "frames": self.frames,
            "fps": self.fps,
            "prompt": self.prompt,
            "error": error,
        })
    }
}

/// Computes OpenAI-style `0..=95` live progress percentage from engine video
/// stage and denoising step counters.
pub fn compute_progress(stage: engine::VideoStage, step: usize, steps_total: usize) -> u32 {
    match stage {
        engine::VideoStage::None => 0,
        engine::VideoStage::Prefill => PREFILL_PROGRESS_PCT,
        engine::VideoStage::Generating => {
            if steps_total == 0 {
                GEN_PROGRESS_BASE_PCT
            } else {
                let done = step.min(steps_total) as u64;
                let pct = GEN_PROGRESS_BASE_PCT as u64
                    + (done * GEN_PROGRESS_SPAN_PCT as u64) / steps_total as u64;
                (pct as u32).min(MAX_IN_PROGRESS_PCT)
            }
        }
    }
}

/// Internal entry stored inside [`VideoStore`], pairing metadata with the
/// active cancellation guard while queued or running.
struct StoredEntry {
    /// Public job record.
    job: VideoJob,
    /// Dropping this cancels a queued or running worker job.
    cancel: Option<worker::CancelGuard>,
}

/// Thread-safe store of active and completed video jobs backed by MP4 files in
/// `video_dir`.
#[derive(Clone)]
pub struct VideoStore {
    /// Directory where finished `{id}.mp4` files are written.
    dir: PathBuf,
    /// Monotonic counter combined with timestamp to produce unique video ids.
    next_seq: Arc<AtomicU64>,
    /// Jobs by id plus insertion order for deterministic pagination.
    inner: Arc<Mutex<StoreInner>>,
}

/// Guarded job index inside [`VideoStore`].
#[derive(Default)]
struct StoreInner {
    /// Creation-ordered job ids (oldest first).
    order: Vec<String>,
    /// Active and finished jobs keyed by id.
    jobs: HashMap<String, StoredEntry>,
}

/// Returns current Unix epoch time in whole seconds.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl VideoStore {
    /// Creates a new store writing completed MP4 clips inside `dir`.
    pub fn new(dir: PathBuf) -> VideoStore {
        VideoStore {
            dir,
            next_seq: Arc::new(AtomicU64::new(1)),
            inner: Arc::new(Mutex::new(StoreInner::default())),
        }
    }

    /// Allocates a new unique `video_<hex>` identifier.
    pub fn next_id(&self) -> String {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let ts = unix_now();
        format!("video_{ts:010x}{seq:06x}")
    }

    /// Inserts a newly queued job with its worker cancellation guard and
    /// returns the initial job snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_queued(
        &self,
        id: String,
        model: String,
        model_index: usize,
        prompt: String,
        size: String,
        seconds: String,
        cancel: worker::CancelGuard,
    ) -> VideoJob {
        let worker_request_id = cancel.id();
        let job = VideoJob {
            id: id.clone(),
            model,
            model_index,
            worker_request_id,
            prompt,
            status: JobStatus::Queued,
            progress: 0,
            size,
            seconds,
            frames: None,
            fps: None,
            created_at: unix_now(),
            completed_at: None,
            error: None,
            content_path: None,
        };
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.order.push(id.clone());
        inner.jobs.insert(
            id,
            StoredEntry {
                job: job.clone(),
                cancel: Some(cancel),
            },
        );
        job
    }

    /// Marks `id` as `in_progress` if still present and queued.
    pub fn mark_in_progress(&self, id: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = inner.jobs.get_mut(id)
            && entry.job.status == JobStatus::Queued
        {
            entry.job.status = JobStatus::InProgress;
        }
    }

    /// Writes `output.bytes` atomically to `{dir}/{id}.mp4` and marks `id`
    /// completed, releasing its cancel guard. If the job was deleted while
    /// running, the file is discarded.
    pub fn complete(&self, id: &str, output: engine::VideoCallOutput) {
        let still_exists = {
            let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            inner.jobs.contains_key(id)
        };
        if !still_exists {
            return;
        }
        let path = self.dir.join(format!("{id}.mp4"));
        let write_result = write_file_atomic(&self.dir, &path, &output.bytes);
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = inner.jobs.get_mut(id) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        entry.cancel = None;
        match write_result {
            Ok(()) => {
                let sec = if output.fps > 0 {
                    ((output.frames as f64 / output.fps as f64).round() as i32).max(1)
                } else {
                    1
                };
                entry.job.status = JobStatus::Completed;
                entry.job.progress = COMPLETED_PROGRESS_PCT;
                entry.job.size = format!("{}x{}", output.width, output.height);
                entry.job.seconds = sec.to_string();
                entry.job.frames = Some(output.frames);
                entry.job.fps = Some(output.fps);
                entry.job.completed_at = Some(unix_now());
                entry.job.content_path = Some(path);
            }
            Err(err) => {
                entry.job.status = JobStatus::Failed;
                entry.job.error = Some(JobError {
                    code: "storage_error".into(),
                    message: format!("failed to write video file: {err}"),
                });
            }
        }
    }

    /// Marks `id` as failed with `code` and `message`, releasing its cancel
    /// guard.
    pub fn fail(&self, id: &str, code: &str, message: String) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = inner.jobs.get_mut(id) {
            entry.cancel = None;
            entry.job.status = JobStatus::Failed;
            entry.job.error = Some(JobError {
                code: code.into(),
                message,
            });
        }
    }

    /// Returns a snapshot of `id`, updating live progress from `live_progress`
    /// when `status == JobStatus::InProgress`.
    pub fn get(
        &self,
        id: &str,
        live_progress: impl Fn(usize, u64) -> Option<(engine::VideoStage, usize, usize)>,
    ) -> Option<VideoJob> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let mut job = inner.jobs.get(id)?.job.clone();
        if job.status == JobStatus::InProgress
            && let Some((stage, step, total)) =
                live_progress(job.model_index, job.worker_request_id)
        {
            job.progress = job.progress.max(compute_progress(stage, step, total));
        }
        Some(job)
    }

    /// Returns true if any stored job is currently `in_progress`.
    pub fn has_in_progress(&self) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner
            .jobs
            .values()
            .any(|e| e.job.status == JobStatus::InProgress)
    }

    /// Returns a paginated list of jobs matching `order` (`asc` or `desc`),
    /// cursor `after`, and `limit` (clamped to `1..=100`), plus `has_more`.
    pub fn list(
        &self,
        limit: Option<usize>,
        after: Option<&str>,
        order: &str,
        live_progress: impl Fn(usize, u64) -> Option<(engine::VideoStage, usize, usize)>,
    ) -> (Vec<VideoJob>, bool) {
        let limit = limit.unwrap_or(DEFAULT_LIST_LIMIT).clamp(1, MAX_LIST_LIMIT);
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let mut ids: Vec<&str> = inner.order.iter().map(String::as_str).collect();
        if !order.eq_ignore_ascii_case("asc") {
            ids.reverse();
        }
        let start = match after {
            Some(cursor) => ids
                .iter()
                .position(|&id| id == cursor)
                .map_or(ids.len(), |pos| pos + 1),
            None => 0,
        };
        let slice = &ids[start..];
        let has_more = slice.len() > limit;
        let data = slice
            .iter()
            .take(limit)
            .filter_map(|&id| inner.jobs.get(id))
            .map(|entry| {
                let mut job = entry.job.clone();
                if job.status == JobStatus::InProgress
                    && let Some((stage, step, total)) =
                        live_progress(job.model_index, job.worker_request_id)
                {
                    job.progress = job.progress.max(compute_progress(stage, step, total));
                }
                job
            })
            .collect();
        (data, has_more)
    }

    /// Deletes `id`, cooperatively cancelling it if queued or running and
    /// removing its MP4 file from disk. Returns true when `id` existed.
    pub fn delete(&self, id: &str) -> bool {
        let removed = {
            let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            let entry = inner.jobs.remove(id);
            if entry.is_some() {
                inner.order.retain(|existing| existing != id);
            }
            entry
        };
        let Some(mut entry) = removed else {
            return false;
        };
        // Drop cancel guard outside the store lock to avoid lock inversion.
        drop(entry.cancel.take());
        if let Some(path) = entry.job.content_path {
            let _ = std::fs::remove_file(path);
        }
        true
    }
}

/// Writes `bytes` into `path` via a sibling temporary file and atomic rename.
fn write_file_atomic(dir: &Path, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("mp4.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns MiniMax-H3 FL2VA or REF2VA test limits matching C constants.
    fn test_limits(variant: engine::VideoVariant) -> engine::VideoLimits {
        engine::VideoLimits {
            max_prompt_tokens: 4096,
            patch_px: 32,
            max_dim: 2048,
            default_width: 1344,
            default_height: 768,
            max_canvas_area: 2048 * 1152,
            fps: 24,
            min_frames: 5,
            max_frames: 362,
            default_frames: 124,
            min_seconds: 1,
            max_seconds: 15,
            default_seconds: 5,
            min_steps: 1,
            max_steps: 100,
            default_steps: 4,
            max_ref_images: 9,
            max_ref_videos: 3,
            max_ref_audios: 3,
            max_total_refs: 12,
            variant: variant as i32,
            guidance_scale: 1.0,
            default_shift_video: 12.0,
            default_shift_audio: 3.0,
        }
    }

    #[test]
    fn json_requests_validate_against_model_limits() {
        let fl2va = test_limits(engine::VideoVariant::Fl2va);
        let body = json!({
            "model": "minimax-h3-video",
            "prompt": "ocean waves at dusk",
            "size": "1280x720",
            "seconds": "4",
            "steps": 4,
            "seed": 42,
            "input_reference": "data:image/png;base64,iVBORw0K"
        })
        .to_string();
        let raw = parse_create_body(Some("application/json"), body.as_bytes()).unwrap();
        assert_eq!(raw.model(), Some("minimax-h3-video"));
        let parsed = raw.validate(&fl2va).unwrap();
        assert_eq!((parsed.width, parsed.height), (1280, 720));
        assert_eq!(parsed.seconds, 4);
        assert_eq!(parsed.steps, 4);
        assert_eq!(parsed.seed, 42);
        assert_eq!(parsed.first_frame.as_deref(), Some(&b"\x89PNG\r\n"[..]));
        assert!(parsed.ref_images.is_empty());

        // Same input_reference maps to ref_images[0] on REF2VA.
        let ref2va = test_limits(engine::VideoVariant::Ref2va);
        let raw = parse_create_body(Some("application/json"), body.as_bytes()).unwrap();
        let parsed = raw.validate(&ref2va).unwrap();
        assert!(parsed.first_frame.is_none());
        assert_eq!(parsed.ref_images.len(), 1);

        // FL2VA rejects REF2VA fields, and REF2VA rejects first_frame.
        let bad_ref = json!({"prompt": "hi", "ref_images": ["AQID"]}).to_string();
        assert!(
            parse_create_body(None, bad_ref.as_bytes())
                .unwrap()
                .validate(&fl2va)
                .is_err()
        );
        let bad_fl = json!({"prompt": "hi", "first_frame": "AQID"}).to_string();
        assert!(
            parse_create_body(None, bad_fl.as_bytes())
                .unwrap()
                .validate(&ref2va)
                .is_err()
        );

        // seconds and frames together are rejected.
        let both = json!({"prompt": "hi", "seconds": 4, "frames": 22}).to_string();
        assert!(
            parse_create_body(None, both.as_bytes())
                .unwrap()
                .validate(&fl2va)
                .is_err()
        );

        // Non-1.0 guidance_scale and negative_prompt are rejected.
        let bad_cfg = json!({"prompt": "hi", "guidance_scale": 5.0}).to_string();
        assert!(
            parse_create_body(None, bad_cfg.as_bytes())
                .unwrap()
                .validate(&fl2va)
                .is_err()
        );
        let bad_neg = json!({"prompt": "hi", "negative_prompt": "blurry"}).to_string();
        assert!(
            parse_create_body(None, bad_neg.as_bytes())
                .unwrap()
                .validate(&fl2va)
                .is_err()
        );
    }

    #[test]
    fn multipart_requests_parse_ref2va_media_arrays_and_limits() {
        let boundary = "----VideoBoundary42";
        let mut body = Vec::new();
        let mut add = |name: &str, data: &[u8]| {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        };
        add("prompt", b"<Picture 1> walking while <Audio 1> plays");
        add("frames", b"22");
        add("ref_image_size", b"max");
        add("ref_images", b"img1bytes");
        add("ref_videos", b"vid1bytes");
        add("ref_video_audios", b"aud1bytes");
        add("ref_audios", b"sound1bytes");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let ct = format!("multipart/form-data; boundary={boundary}");
        let raw = parse_create_body(Some(&ct), &body).unwrap();
        let parsed = raw
            .validate(&test_limits(engine::VideoVariant::Ref2va))
            .unwrap();
        assert_eq!(parsed.frames, 22);
        assert!(parsed.ref_image_max);
        assert_eq!(parsed.ref_images, vec![b"img1bytes".to_vec()]);
        assert_eq!(parsed.ref_videos, vec![b"vid1bytes".to_vec()]);
        assert_eq!(parsed.ref_video_audios, vec![Some(b"aud1bytes".to_vec())]);
        assert_eq!(parsed.ref_audios, vec![b"sound1bytes".to_vec()]);
    }

    #[test]
    fn store_tracks_lifecycle_progress_pagination_and_atomic_mp4_cleanup() {
        let dir = engine::TestDir::new("video-store");
        let store = VideoStore::new(dir.0.clone());
        let id1 = store.next_id();
        let id2 = store.next_id();
        assert_ne!(id1, id2);

        let j1 = store.insert_queued(
            id1.clone(),
            "minimax-h3-video".into(),
            0,
            "first".into(),
            "512x288".into(),
            "1".into(),
            worker::CancelGuard::new_unsubmitted(1),
        );
        assert_eq!(j1.status, JobStatus::Queued);
        store.mark_in_progress(&id1);
        assert!(store.has_in_progress());

        let live = store
            .get(&id1, |_, _| Some((engine::VideoStage::Generating, 2, 4)))
            .unwrap();
        assert_eq!(live.status, JobStatus::InProgress);
        assert_eq!(live.progress, 52);

        store.complete(
            &id1,
            engine::VideoCallOutput {
                bytes: b"fake-mp4-bytes".to_vec(),
                width: 512,
                height: 288,
                frames: 22,
                fps: 24,
                prompt_tokens: 23,
                prefill_seconds: 0.3,
                video_seconds: 5.0,
            },
        );
        let done = store.get(&id1, |_, _| None).unwrap();
        assert_eq!(done.status, JobStatus::Completed);
        assert_eq!(done.progress, 100);
        let mp4_path = done.content_path.clone().unwrap();
        assert_eq!(std::fs::read(&mp4_path).unwrap(), b"fake-mp4-bytes");

        let _j2 = store.insert_queued(
            id2.clone(),
            "minimax-h3-video".into(),
            0,
            "second".into(),
            "512x288".into(),
            "1".into(),
            worker::CancelGuard::new_unsubmitted(2),
        );
        let (page, has_more) = store.list(Some(1), None, "desc", |_, _| None);
        assert!(has_more);
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].id, id2);

        let (page2, has_more2) = store.list(Some(10), Some(&id2), "desc", |_, _| None);
        assert!(!has_more2);
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].id, id1);

        assert!(store.delete(&id1));
        assert!(!mp4_path.exists(), "delete removes the MP4 from disk");
        assert!(store.get(&id1, |_, _| None).is_none());
    }
}
