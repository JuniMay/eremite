//! OpenAI Audio API (POST /v1/audio/transcriptions and POST
//! /v1/audio/translations) for Nemotron-3.5-Transcribe. Parses
//! multipart/form-data uploads, maps ISO-639-1 language codes and context terms
//! into the nemotron3p5 prompt template, runs transcription or English
//! translation through the shared worker, aligns CTC word and segment
//! timestamps when requested, and formats JSON, plain text, verbose JSON,
//! SubRip (SRT), WebVTT (VTT) and server-sent event responses.

use std::collections::HashMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::{chat, engine, http, worker};

/// Default served model id for Nemotron-3.5-Transcribe when `id` is omitted.
pub const DEFAULT_MODEL_ID: &str = "nemotron-3.5-transcribe";
/// Display name shown in GET /v1/models for Nemotron-3.5-Transcribe.
pub const DISPLAY_NAME: &str = "Nemotron-3.5-Transcribe (eremite)";
/// Maximum body size for multipart audio uploads (256 MiB covers 1-hour WAV,
/// FLAC, MP3 and M4A recordings).
pub const MAX_AUDIO_BODY_BYTES: usize = 256 * http::MIB;
/// Default maximum output tokens for one transcription or translation call
/// (matches Nemotron-3.5-Transcribe's recommended transcription limit).
pub const DEFAULT_MAX_TOKENS: i64 = 16_384;
/// Duration represented by one audio perception token in seconds (1280 samples
/// at 16 kHz = 80 ms, or 12.5 Hz).
const AUDIO_TOKEN_SECONDS: f64 = 0.08;
/// Scale factor for rounding reported audio duration to two decimal places.
const DURATION_ROUND_SCALE: f64 = 100.0;
/// Scale factor for rounding word and segment timestamps to milliseconds.
const TIMESTAMP_ROUND_SCALE: f64 = 1000.0;
/// Pause duration in seconds that starts a new transcript or subtitle segment.
const SEGMENT_PAUSE_SECONDS: f64 = 0.75;
/// Target segment duration in seconds after which clause punctuation or the
/// soft character cap splits the segment.
const SEGMENT_SOFT_MAX_SECONDS: f64 = 7.0;
/// Maximum segment duration in seconds for long unpunctuated speech.
const SEGMENT_HARD_MAX_SECONDS: f64 = 12.0;
/// Target character count per segment (two standard 42-character subtitle
/// lines).
const SEGMENT_SOFT_MAX_CHARS: usize = 84;
/// Hard character cap per segment for long unpunctuated speech.
const SEGMENT_HARD_MAX_CHARS: usize = 120;
/// Milliseconds per second for SRT and WebVTT timestamp formatting.
const MILLISECONDS_PER_SECOND: u64 = 1000;
/// Seconds per minute for SRT and WebVTT timestamp formatting.
const SECONDS_PER_MINUTE: u64 = 60;
/// Seconds per hour for SRT and WebVTT timestamp formatting.
const SECONDS_PER_HOUR: u64 = 3600;

/// Sentinel byte that eremite_tokenize turns into the checkpoint's audio token.
const MEDIA_MARK_BYTE: u8 = 0xFF;
/// CRLF blank-line separator between multipart headers and part body.
const CRLF_HEADER_END: &[u8] = b"\r\n\r\n";
/// LF blank-line separator between multipart headers and part body.
const LF_HEADER_END: &[u8] = b"\n\n";

/// Minimum consecutive identical characters collapsed by the loop guardrail
/// (matches Nemotron-3.5-Transcribe's documented 21-character threshold).
const LOOP_MIN_CHAR_RUN: usize = 21;
/// Minimum consecutive identical substrings collapsed by the loop guardrail
/// (matches Nemotron-3.5-Transcribe's documented 20-repeat threshold).
const LOOP_MIN_BLOCK_REPEATS: usize = 20;
/// Maximum substring length in Unicode scalar values checked by the loop
/// guardrail (matches Nemotron-3.5-Transcribe's documented 4096-char cap).
const LOOP_MAX_BLOCK_CHARS: usize = 4096;

/// ISO-639-1 and common subtag codes mapped to the English language names used
/// by Nemotron-3.5-Transcribe's prompt instructions.
const LANGUAGE_CODES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("cmn", "Chinese Mandarin"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fil", "Tagalog"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("in", "Indonesian"),
    ("it", "Italian"),
    ("iw", "Hebrew"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mr", "Marathi"),
    ("mt", "Maltese"),
    ("nb", "Norwegian"),
    ("nl", "Dutch"),
    ("nn", "Norwegian"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sv", "Swedish"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese Mandarin"),
];

/// OpenAI Audio API endpoint being served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEndpoint {
    /// POST /v1/audio/transcriptions: transcribe audio in the spoken or
    /// requested language.
    Transcriptions,
    /// POST /v1/audio/translations: translate speech into English.
    Translations,
}

impl AudioEndpoint {
    /// Returns the task label reported in `verbose_json` responses and logs.
    pub fn task_name(self) -> &'static str {
        match self {
            AudioEndpoint::Transcriptions => "transcribe",
            AudioEndpoint::Translations => "translate",
        }
    }
}

/// Output representation requested in `response_format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseFormat {
    /// JSON object with `text` and token `usage`.
    Json,
    /// Plain UTF-8 transcript body with `text/plain; charset=utf-8`.
    Text,
    /// JSON object with `task`, `language`, `duration`, `text`, `usage` and
    /// optional `words` and `segments`.
    VerboseJson,
    /// SubRip subtitle cues with `text/plain; charset=utf-8`.
    Srt,
    /// WebVTT subtitle cues with `text/vtt; charset=utf-8`.
    Vtt,
}

/// Requested timestamp granularities for `response_format=verbose_json`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimestampGranularities {
    /// Include word-level timestamps in `words`.
    pub word: bool,
    /// Include segment-level timestamps in `segments`.
    pub segment: bool,
    /// True when the request explicitly provided `timestamp_granularities`.
    pub explicit: bool,
}

/// One grouped caption or transcript segment derived from aligned words.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedSegment {
    /// Zero-based sequential segment index.
    pub id: usize,
    /// Segment start time in seconds.
    pub start: f64,
    /// Segment end time in seconds.
    pub end: f64,
    /// Verbatim UTF-8 segment text sliced from the cleaned transcript.
    pub text: String,
}

/// Validated audio transcription or translation request parsed from a
/// multipart/form-data upload.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedAudioRequest {
    /// Requested model id, or None when omitted.
    pub model: Option<String>,
    /// Raw uploaded audio file bytes.
    pub file: Vec<u8>,
    /// Optional uploaded filename from Content-Disposition.
    pub filename: Option<String>,
    /// Optional spoken language hint (transcriptions only).
    pub language: Option<String>,
    /// Optional domain vocabulary or context terms hint.
    pub prompt: Option<String>,
    /// Requested output response format.
    pub response_format: ResponseFormat,
    /// Requested `timestamp_granularities` for `verbose_json`.
    pub timestamp_granularities: TimestampGranularities,
    /// Sampling temperature (0.0 for greedy decoding).
    pub temperature: f64,
    /// Optional deterministic RNG seed.
    pub seed: Option<u64>,
    /// True to stream OpenAI audio transcription server-sent events.
    pub stream: bool,
    /// Target audio endpoint.
    pub endpoint: AudioEndpoint,
}

/// Tokenized audio request ready to submit to the worker and format on return.
pub struct PreparedAudioRequest {
    /// Target worker model index.
    pub model_index: usize,
    /// Tokenized prompt containing one audio placeholder token.
    pub prompt_tokens: Vec<i32>,
    /// Number of tokens in `prompt_tokens` before expanding the audio span.
    pub unexpanded_prompt_tokens: usize,
    /// Uploaded audio bytes passed to the engine media decoder.
    pub file: Vec<u8>,
    /// Retained copy of uploaded audio bytes when post-decode CTC alignment is
    /// required, or None when timestamps are not requested.
    pub align_audio: Option<Vec<u8>>,
    /// Sampling settings for the worker.
    pub sampling: worker::Sampling,
    /// Maximum output tokens to generate.
    pub max_tokens: i64,
    /// Requested response format.
    pub response_format: ResponseFormat,
    /// Include word-level timestamps in `verbose_json`.
    pub include_words: bool,
    /// Include segment-level timestamps in `verbose_json`.
    pub include_segments: bool,
    /// Resolved language label if known ahead of decoding.
    pub language_label: Option<String>,
    /// Target audio endpoint.
    pub endpoint: AudioEndpoint,
    /// True to stream server-sent events.
    pub stream: bool,
    /// Model tokenizer used to decode emitted token ids.
    pub tokenizer: engine::Tokenizer,
}

impl PreparedAudioRequest {
    /// Converts this prepared request into a worker job for `model_index`.
    pub fn build_worker_request(&mut self, model_index: usize, note: String) -> worker::Request {
        worker::Request {
            model_index,
            prompt: std::mem::take(&mut self.prompt_tokens),
            media: vec![std::mem::take(&mut self.file)],
            sampling: self.sampling,
            max_tokens: self.max_tokens,
            note,
        }
    }
}

/// One parsed part from a multipart/form-data request body.
struct MultipartPart<'a> {
    /// Form field name from Content-Disposition.
    name: String,
    /// Optional filename parameter from Content-Disposition.
    filename: Option<String>,
    /// Raw bytes of the part body.
    data: &'a [u8],
}

/// Finds the first byte offset of `needle` inside `haystack`, or None.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
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

/// Extracts the `name` and optional `filename` parameters from a
/// Content-Disposition header value.
fn parse_disposition(value: &str) -> (Option<String>, Option<String>) {
    let mut name = None;
    let mut filename = None;
    for param in value.split(';').skip(1) {
        let Some((key, val)) = param.split_once('=') else {
            continue;
        };
        let val = val.trim();
        let unquoted = val
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(val);
        let key = key.trim();
        if key.eq_ignore_ascii_case("name") {
            name = Some(unquoted.to_string());
        } else if key.eq_ignore_ascii_case("filename") && !unquoted.is_empty() {
            filename = Some(unquoted.to_string());
        }
    }
    (name, filename)
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
        let mut filename = None;
        for line in headers_text.lines() {
            if let Some((hname, hval)) = line.split_once(':')
                && hname.trim().eq_ignore_ascii_case("content-disposition")
            {
                (field_name, filename) = parse_disposition(hval.trim());
            }
        }
        let name = field_name.ok_or_else(|| {
            chat::Error::new_bad_request("multipart part is missing Content-Disposition name")
        })?;

        let data_slice = &part_slice[body_start..];
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
            filename,
            data: &data_slice[..data_end],
        });
        cursor = next_cursor;
    }
    Ok(parts)
}

/// Maps an ISO-639-1 or BCP-47 language tag to the English language name used
/// in Nemotron-3.5-Transcribe's prompt instructions. Unknown values pass
/// through trimmed.
pub fn resolve_language_name(raw: &str) -> String {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let primary = lower.split(['-', '_']).next().unwrap_or(&lower);
    if let Some(&(_, name)) = LANGUAGE_CODES.iter().find(|&&(code, _)| code == primary) {
        return name.to_string();
    }
    trimmed.to_string()
}

/// Parses one `timestamp_granularities` form value (single token,
/// comma-separated tokens, or JSON array of strings) into `out`.
fn parse_timestamp_granularity_value(
    raw: &str,
    out: &mut TimestampGranularities,
) -> Result<(), chat::Error> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let items: Vec<String> = if trimmed.starts_with('[') {
        serde_json::from_str::<Vec<String>>(trimmed).map_err(|_| {
            chat::Error::new_bad_request(
                "timestamp_granularities must contain \"word\" and/or \"segment\"",
            )
        })?
    } else {
        trimmed
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    if items.is_empty() {
        return Err(chat::Error::new_bad_request(
            "timestamp_granularities must contain \"word\" and/or \"segment\"",
        ));
    }
    for item in items {
        if item.eq_ignore_ascii_case("word") {
            out.word = true;
            out.explicit = true;
        } else if item.eq_ignore_ascii_case("segment") {
            out.segment = true;
            out.explicit = true;
        } else {
            return Err(chat::Error::new_bad_request(format!(
                "timestamp_granularities value {item:?} is not supported; use \"word\" or \"segment\""
            )));
        }
    }
    Ok(())
}

/// Parses and validates a multipart/form-data audio request for `endpoint`.
pub fn parse_audio_request(
    endpoint: AudioEndpoint,
    content_type: Option<&str>,
    body: &[u8],
) -> Result<ParsedAudioRequest, chat::Error> {
    let content_type = content_type
        .ok_or_else(|| chat::Error::new_bad_request("Content-Type must be multipart/form-data"))?;
    let boundary = extract_multipart_boundary(content_type)?;
    let parts = parse_multipart(&boundary, body)?;

    let mut file: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut fields = HashMap::<String, String>::new();
    let mut timestamp_granularities = TimestampGranularities::default();

    for part in parts {
        if part.name == "file" {
            if part.data.is_empty() {
                return Err(chat::Error::new_bad_request("file is empty"));
            }
            if file.replace(part.data.to_vec()).is_some() {
                return Err(chat::Error::new_bad_request("only one file is supported"));
            }
            filename = part.filename;
            continue;
        }
        let text = std::str::from_utf8(part.data).map_err(|_| {
            chat::Error::new_bad_request(format!("field {} is not valid UTF-8", part.name))
        })?;
        if part.name == "timestamp_granularities" || part.name == "timestamp_granularities[]" {
            parse_timestamp_granularity_value(text, &mut timestamp_granularities)?;
            continue;
        }
        fields.insert(part.name, text.to_string());
    }

    let file = file.ok_or_else(|| chat::Error::new_bad_request("file is required"))?;

    let model = fields
        .get("model")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let language = fields
        .get("language")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);
    if endpoint == AudioEndpoint::Translations && language.is_some() {
        return Err(chat::Error::new_bad_request(
            "language is not supported for translations; translations always output English",
        ));
    }

    let prompt = fields
        .get("prompt")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let response_format = match fields
        .get("response_format")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        None => ResponseFormat::Json,
        Some(raw) if raw.eq_ignore_ascii_case("json") => ResponseFormat::Json,
        Some(raw) if raw.eq_ignore_ascii_case("text") => ResponseFormat::Text,
        Some(raw) if raw.eq_ignore_ascii_case("verbose_json") => ResponseFormat::VerboseJson,
        Some(raw) if raw.eq_ignore_ascii_case("srt") => ResponseFormat::Srt,
        Some(raw) if raw.eq_ignore_ascii_case("vtt") => ResponseFormat::Vtt,
        Some(other) => {
            return Err(chat::Error::new_bad_request(format!(
                "response_format {other:?} is not supported; use \"json\", \"text\", \"verbose_json\", \"srt\" or \"vtt\""
            )));
        }
    };

    if timestamp_granularities.explicit {
        if endpoint == AudioEndpoint::Translations {
            return Err(chat::Error::new_bad_request(
                "timestamp_granularities is only supported for transcriptions",
            ));
        }
        if response_format != ResponseFormat::VerboseJson {
            return Err(chat::Error::new_bad_request(
                "timestamp_granularities requires response_format \"verbose_json\"",
            ));
        }
    }

    let temperature = match fields
        .get("temperature")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        None => 0.0,
        Some(raw) => {
            let value = raw.parse::<f64>().map_err(|_| {
                chat::Error::new_bad_request("temperature must be a nonnegative finite number")
            })?;
            if !value.is_finite() || value < 0.0 {
                return Err(chat::Error::new_bad_request(
                    "temperature must be a nonnegative finite number",
                ));
            }
            value
        }
    };

    let seed = match fields
        .get("seed")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        None => None,
        Some(raw) => Some(
            raw.parse::<u64>()
                .map_err(|_| chat::Error::new_bad_request("seed must be a nonnegative integer"))?,
        ),
    };

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
    if stream && response_format != ResponseFormat::Json {
        return Err(chat::Error::new_bad_request(
            "stream requires response_format \"json\"",
        ));
    }

    Ok(ParsedAudioRequest {
        model,
        file,
        filename,
        language,
        prompt,
        response_format,
        timestamp_granularities,
        temperature,
        seed,
        stream,
        endpoint,
    })
}

/// Builds the user instruction text for Nemotron-3.5-Transcribe from the
/// requested endpoint, optional language hint and optional context terms.
pub fn build_instruction(
    endpoint: AudioEndpoint,
    language: Option<&str>,
    prompt: Option<&str>,
) -> String {
    match endpoint {
        AudioEndpoint::Transcriptions => match (language, prompt) {
            (None, None) => {
                "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only."
                    .into()
            }
            (Some(lang), None) => {
                let name = resolve_language_name(lang);
                format!(
                    "Audio transcription. Language: {name}. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only."
                )
            }
            (None, Some(context)) => {
                format!("Transcribe with punctuation and capitalization. Context terms: {context}.")
            }
            (Some(lang), Some(context)) => {
                let name = resolve_language_name(lang);
                format!(
                    "Audio transcription. Language: {name}. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Context terms: {context}. Output: transcript only."
                )
            }
        },
        AudioEndpoint::Translations => match prompt {
            None => "Please translate the given speech to English.".into(),
            Some(context) => {
                format!("Please translate the given speech to English. Context terms: {context}.")
            }
        },
    }
}

/// Renders the full nemotron3p5 prompt bytes (empty system turn, user
/// instruction followed by one space and the audio sentinel byte 0xFF, and
/// assistant turn with thinking disabled).
pub fn render_prompt_bytes(
    endpoint: AudioEndpoint,
    language: Option<&str>,
    prompt: Option<&str>,
) -> Vec<u8> {
    let instruction = build_instruction(endpoint, language, prompt);
    let mut out = Vec::with_capacity(instruction.len() + 96);
    out.extend_from_slice(b"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n");
    out.extend_from_slice(instruction.as_bytes());
    out.push(b' ');
    out.push(MEDIA_MARK_BYTE);
    out.extend_from_slice(b"<|im_end|>\n<|im_start|>assistant\n<think></think>");
    out
}

/// Tokenizes a validated audio request for submission to the worker and checks
/// whether requested timestamp formats are supported by the model.
pub fn prepare_request(
    parsed: ParsedAudioRequest,
    model_index: usize,
    can_align_audio: bool,
    tokenizer: engine::Tokenizer,
) -> Result<PreparedAudioRequest, chat::Error> {
    let requires_alignment = matches!(
        parsed.response_format,
        ResponseFormat::Srt | ResponseFormat::Vtt
    ) || parsed.timestamp_granularities.explicit;
    if requires_alignment && !can_align_audio {
        return Err(chat::Error::new_bad_request(
            "word and segment timestamps require model-ctc-bf16.safetensors in the checkpoint",
        ));
    }
    let (include_words, include_segments, needs_alignment) = match parsed.response_format {
        ResponseFormat::Srt | ResponseFormat::Vtt => (false, true, true),
        ResponseFormat::VerboseJson => {
            if parsed.timestamp_granularities.explicit {
                (
                    parsed.timestamp_granularities.word,
                    parsed.timestamp_granularities.segment,
                    parsed.timestamp_granularities.word || parsed.timestamp_granularities.segment,
                )
            } else if can_align_audio {
                // OpenAI's default verbose_json granularity is ["segment"].
                (false, true, true)
            } else {
                (false, false, false)
            }
        }
        ResponseFormat::Json | ResponseFormat::Text => (false, false, false),
    };

    let rendered = render_prompt_bytes(
        parsed.endpoint,
        parsed.language.as_deref(),
        parsed.prompt.as_deref(),
    );
    let prompt_tokens = tokenizer.tokenize_prompt(&rendered);
    if prompt_tokens.is_empty() {
        return Err(chat::Error::new_bad_request(
            "failed to tokenize audio prompt",
        ));
    }
    let unexpanded_prompt_tokens = prompt_tokens.len();
    let language_label = match parsed.endpoint {
        AudioEndpoint::Translations => Some("english".to_string()),
        AudioEndpoint::Transcriptions => parsed
            .language
            .as_deref()
            .map(|l| resolve_language_name(l).to_ascii_lowercase()),
    };
    let align_audio = if needs_alignment {
        Some(parsed.file.clone())
    } else {
        None
    };
    Ok(PreparedAudioRequest {
        model_index,
        prompt_tokens,
        unexpanded_prompt_tokens,
        file: parsed.file,
        align_audio,
        sampling: worker::Sampling {
            temperature: parsed.temperature,
            top_p: 1.0,
            top_k: 0,
            seed: parsed.seed,
        },
        max_tokens: DEFAULT_MAX_TOKENS,
        response_format: parsed.response_format,
        include_words,
        include_segments,
        language_label,
        endpoint: parsed.endpoint,
        stream: parsed.stream,
        tokenizer,
    })
}

/// Incrementally decodes token byte chunks into valid UTF-8 text without
/// splitting multi-byte code points across token boundaries.
#[derive(Default)]
struct Utf8StreamDecoder {
    /// Incomplete trailing UTF-8 bytes waiting for the next token.
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    /// Appends raw token `bytes` and returns the longest valid UTF-8 prefix
    /// ready to emit immediately.
    fn push_bytes(&mut self, bytes: &[u8]) -> String {
        if bytes.is_empty() {
            return String::new();
        }
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid = std::str::from_utf8(&self.pending[..valid_up_to])
                            .expect("validated UTF-8 prefix");
                        out.push_str(valid);
                        self.pending.drain(..valid_up_to);
                        continue;
                    }
                    match err.error_len() {
                        Some(bad_len) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..bad_len);
                        }
                        None => break,
                    }
                }
            }
        }
        out
    }

    /// Flushes any trailing incomplete bytes as replacement characters.
    fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        tail
    }
}

/// Decodes `ids` through `tokenizer` into `decoder`, skipping special control
/// tokens.
fn decode_token_chunk(
    tokenizer: engine::Tokenizer,
    ids: &[u32],
    decoder: &mut Utf8StreamDecoder,
) -> String {
    let mut out = String::new();
    for &id in ids {
        let token_id = id as i32;
        if tokenizer.is_special(token_id) {
            continue;
        }
        let piece = decoder.push_bytes(tokenizer.token_bytes(token_id));
        out.push_str(&piece);
    }
    out
}

/// Collapses pathological generation loops recommended by
/// Nemotron-3.5-Transcribe: runs of at least 21 identical characters collapse to
/// one character, and runs of at least 20 consecutive identical substrings (up
/// to 4096 characters) collapse to one copy.
pub fn cleanup_generation_loops(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if chars.len() < LOOP_MIN_BLOCK_REPEATS {
        return text.to_string();
    }

    // Pass 1: collapse runs of >= LOOP_MIN_CHAR_RUN identical characters.
    let mut collapsed = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        let mut end = i + 1;
        while end < chars.len() && chars[end] == ch {
            end += 1;
        }
        if end - i >= LOOP_MIN_CHAR_RUN {
            collapsed.push(ch);
        } else {
            collapsed.extend_from_slice(&chars[i..end]);
        }
        i = end;
    }
    chars = collapsed;

    // Pass 2: collapse >= LOOP_MIN_BLOCK_REPEATS consecutive identical blocks.
    let mut out = Vec::with_capacity(chars.len());
    let mut pos = 0;
    while pos < chars.len() {
        let remaining = chars.len() - pos;
        let max_len = (remaining / LOOP_MIN_BLOCK_REPEATS).min(LOOP_MAX_BLOCK_CHARS);
        let mut matched: Option<(usize, usize)> = None;
        for len in 1..=max_len {
            // Fast rejection before full slice comparison.
            if chars[pos] != chars[pos + len] {
                continue;
            }
            let block = &chars[pos..pos + len];
            let mut repeats = 1usize;
            while pos + (repeats + 1) * len <= chars.len()
                && &chars[pos + repeats * len..pos + (repeats + 1) * len] == block
            {
                repeats += 1;
            }
            if repeats >= LOOP_MIN_BLOCK_REPEATS {
                matched = Some((len, repeats));
                break;
            }
        }
        if let Some((len, repeats)) = matched {
            out.extend_from_slice(&chars[pos..pos + len]);
            pos += len * repeats;
        } else {
            out.push(chars[pos]);
            pos += 1;
        }
    }
    out.into_iter().collect()
}

/// Computes estimated audio duration in seconds from expanded and unexpanded
/// prompt token counts, rounded to two decimal places.
fn estimate_duration_seconds(
    expanded_prompt_tokens: usize,
    unexpanded_prompt_tokens: usize,
) -> f64 {
    let text_tokens = unexpanded_prompt_tokens.saturating_sub(1);
    let audio_tokens = expanded_prompt_tokens.saturating_sub(text_tokens);
    let raw = audio_tokens as f64 * AUDIO_TOKEN_SECONDS;
    (raw * DURATION_ROUND_SCALE).round() / DURATION_ROUND_SCALE
}

/// Builds the OpenAI Audio API `usage` JSON object from input and output token
/// counts.
fn build_usage_json(input_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "type": "tokens",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })
}

/// Rounds a timestamp in seconds to millisecond precision without floating
/// point representation noise.
fn round_timestamp(seconds: f64) -> f64 {
    let clamped = if seconds.is_finite() && seconds > 0.0 {
        seconds
    } else {
        0.0
    };
    (clamped * TIMESTAMP_ROUND_SCALE).round() / TIMESTAMP_ROUND_SCALE
}

/// Strips outer sentence punctuation from a word slice for `verbose_json`
/// `words[].word`, preserving internal apostrophes, hyphens and decimals, and
/// returning the original slice if stripping would leave an empty string.
pub fn clean_word_text(raw: &str) -> &str {
    const OUTER_PUNCT: &[char] = &[
        '.', ',', '?', '!', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '“', '”', '‘', '’',
        '。', '，', '、', '？', '！', '；', '：', '（', '）', '「', '」', '《', '》',
    ];
    let trimmed = raw.trim_matches(OUTER_PUNCT);
    if trimmed.is_empty() { raw } else { trimmed }
}

/// Returns whether `word` ends with sentence-terminating punctuation, ignoring
/// trailing closing quotes or parentheses.
fn ends_sentence(word: &str) -> bool {
    const CLOSERS: &[char] = &['"', '\'', ')', ']', '”', '’', '」', '》', '）'];
    let core = word.trim_end_matches(CLOSERS);
    matches!(
        core.chars().last(),
        Some('.' | '!' | '?' | '。' | '！' | '？')
    )
}

/// Returns whether `word` ends with clause-separating punctuation.
fn ends_clause(word: &str) -> bool {
    matches!(
        word.chars().last(),
        Some(',' | ';' | ':' | '，' | '；' | '：' | '、')
    )
}

/// Groups CTC-aligned `words` into sentence and subtitle-sized segments slicing
/// verbatim text from `transcript`.
pub fn build_segments(
    transcript: &str,
    words: &[engine::AudioWord],
    duration: f64,
) -> Vec<AlignedSegment> {
    if words.is_empty() {
        let trimmed = transcript.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        return vec![AlignedSegment {
            id: 0,
            start: 0.0,
            end: round_timestamp(duration),
            text: trimmed.to_string(),
        }];
    }
    let mut segments = Vec::new();
    let mut seg_start_idx = 0usize;
    for idx in 0..words.len() {
        let is_last = idx + 1 == words.len();
        let word_slice = &transcript[words[idx].byte_start..words[idx].byte_end];
        let seg_slice = &transcript[words[seg_start_idx].byte_start..words[idx].byte_end];
        let seg_dur = words[idx].end - words[seg_start_idx].start;
        let seg_chars = seg_slice.chars().count();
        let pause = if is_last {
            0.0
        } else {
            words[idx + 1].start - words[idx].end
        };
        let split = is_last
            || ends_sentence(word_slice)
            || pause >= SEGMENT_PAUSE_SECONDS
            || (seg_dur >= SEGMENT_SOFT_MAX_SECONDS
                && (ends_clause(word_slice) || seg_chars >= SEGMENT_SOFT_MAX_CHARS))
            || seg_dur >= SEGMENT_HARD_MAX_SECONDS
            || seg_chars >= SEGMENT_HARD_MAX_CHARS;
        if split {
            let text = seg_slice.trim();
            if !text.is_empty() {
                segments.push(AlignedSegment {
                    id: segments.len(),
                    start: round_timestamp(words[seg_start_idx].start),
                    end: round_timestamp(words[idx].end),
                    text: text.to_string(),
                });
            }
            seg_start_idx = idx + 1;
        }
    }
    segments
}

/// Formats `seconds` as `HH:MM:SS<sep>mmm` for SRT (`','`) or WebVTT (`'.'`).
fn format_caption_timestamp(seconds: f64, sep: char) -> String {
    let clamped = if seconds.is_finite() && seconds > 0.0 {
        seconds
    } else {
        0.0
    };
    let total_ms = (clamped * MILLISECONDS_PER_SECOND as f64).round() as u64;
    let ms = total_ms % MILLISECONDS_PER_SECOND;
    let total_s = total_ms / MILLISECONDS_PER_SECOND;
    let hours = total_s / SECONDS_PER_HOUR;
    let minutes = (total_s % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let secs = total_s % SECONDS_PER_MINUTE;
    format!("{hours:02}:{minutes:02}:{secs:02}{sep}{ms:03}")
}

/// Formats `segments` as SubRip (`srt`) captions.
pub fn format_srt(segments: &[AlignedSegment]) -> String {
    let mut out = String::new();
    for (idx, seg) in segments.iter().enumerate() {
        let start = format_caption_timestamp(seg.start, ',');
        let end = format_caption_timestamp(seg.end, ',');
        out.push_str(&format!("{}\n{start} --> {end}\n{}\n\n", idx + 1, seg.text));
    }
    out
}

/// Formats `segments` as WebVTT (`vtt`) captions.
pub fn format_vtt(segments: &[AlignedSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        let start = format_caption_timestamp(seg.start, '.');
        let end = format_caption_timestamp(seg.end, '.');
        out.push_str(&format!("{start} --> {end}\n{}\n\n", seg.text));
    }
    out
}

/// Builds a non-streaming response body for `format`.
#[allow(clippy::too_many_arguments)]
fn build_audio_response(
    format: ResponseFormat,
    endpoint: AudioEndpoint,
    language_label: Option<&str>,
    duration: f64,
    text: &str,
    alignment: Option<&engine::AudioAlignment>,
    include_words: bool,
    include_segments: bool,
    input_tokens: usize,
    output_tokens: usize,
) -> Response {
    match format {
        ResponseFormat::Json => {
            let body = json!({
                "text": text,
                "usage": build_usage_json(input_tokens, output_tokens),
            })
            .to_string();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response()
        }
        ResponseFormat::Text => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text.to_string(),
        )
            .into_response(),
        ResponseFormat::VerboseJson => {
            let mut body = json!({
                "task": endpoint.task_name(),
                "duration": duration,
                "text": text,
                "usage": build_usage_json(input_tokens, output_tokens),
            });
            if let Some(lang) = language_label {
                body["language"] = json!(lang);
            }
            if let Some(aligned) = alignment {
                if include_words {
                    let words_json: Vec<Value> = aligned
                        .words
                        .iter()
                        .map(|w| {
                            let raw = &text[w.byte_start..w.byte_end];
                            json!({
                                "word": clean_word_text(raw),
                                "start": round_timestamp(w.start),
                                "end": round_timestamp(w.end),
                            })
                        })
                        .collect();
                    body["words"] = json!(words_json);
                }
                if include_segments {
                    let segments = build_segments(text, &aligned.words, aligned.duration);
                    let segs_json: Vec<Value> = segments
                        .iter()
                        .map(|seg| {
                            json!({
                                "id": seg.id,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                            })
                        })
                        .collect();
                    body["segments"] = json!(segs_json);
                }
            }
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body.to_string(),
            )
                .into_response()
        }
        ResponseFormat::Srt => {
            let words = alignment.map_or(&[][..], |a| &a.words[..]);
            let segments = build_segments(text, words, duration);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                format_srt(&segments),
            )
                .into_response()
        }
        ResponseFormat::Vtt => {
            let words = alignment.map_or(&[][..], |a| &a.words[..]);
            let segments = build_segments(text, words, duration);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
                format_vtt(&segments),
            )
                .into_response()
        }
    }
}

/// Formats one OpenAI `transcript.text.delta` server-sent event.
fn format_delta_sse(delta: &str) -> String {
    let payload = json!({
        "type": "transcript.text.delta",
        "delta": delta,
    });
    format!("event: transcript.text.delta\ndata: {payload}\n\n")
}

/// Formats the final OpenAI `transcript.text.done` server-sent event.
fn format_done_sse(text: &str, input_tokens: usize, output_tokens: usize) -> String {
    let payload = json!({
        "type": "transcript.text.done",
        "text": text,
        "usage": build_usage_json(input_tokens, output_tokens),
    });
    format!("event: transcript.text.done\ndata: {payload}\n\n")
}

/// Formats a mid-stream SSE `error` event.
fn format_error_sse(kind: &str, message: &str) -> String {
    let payload = json!({
        "type": "error",
        "error": {"type": kind, "code": null, "message": message, "param": null},
    });
    format!("event: error\ndata: {payload}\n\n")
}

/// Collects a full non-streaming transcription or translation response from
/// worker `events`.
pub async fn collect_response(
    request: PreparedAudioRequest,
    _cancel: worker::CancelGuard,
    mut events: UnboundedReceiver<worker::Event>,
) -> Response {
    let first = match events.recv().await {
        Some(worker::Event::Failed(message)) => {
            return chat::build_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                &message,
            );
        }
        Some(worker::Event::Refused(message)) => {
            return chat::build_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &message,
            );
        }
        Some(event) => event,
        None => {
            return chat::build_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "the request ended without an answer",
            );
        }
    };

    let mut decoder = Utf8StreamDecoder::default();
    let mut text = String::new();
    let mut prompt_tokens = request.unexpanded_prompt_tokens;
    let mut output_tokens = 0usize;
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
            worker::Event::DecodingStarted {
                prompt_tokens: pt, ..
            } => {
                prompt_tokens = pt;
            }
            worker::Event::Tokens(token_ids) => {
                output_tokens += token_ids.len();
                text.push_str(&decode_token_chunk(
                    request.tokenizer,
                    &token_ids,
                    &mut decoder,
                ));
            }
            worker::Event::Done { reason } => {
                if reason == worker::FinishReason::ServerStop {
                    return chat::build_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "server_error",
                        "the server is stopping",
                    );
                }
                break;
            }
            worker::Event::Failed(message) => {
                return chat::build_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    &message,
                );
            }
            worker::Event::Refused(message) => {
                return chat::build_error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    &message,
                );
            }
            _ => break,
        }
    }
    text.push_str(&decoder.finish());
    let cleaned = cleanup_generation_loops(text.trim());
    let alignment = if let Some(audio_bytes) = request.align_audio {
        match worker::align_audio(request.model_index, audio_bytes, cleaned.clone()).await {
            Ok(aligned) => Some(aligned),
            Err(message) => {
                return chat::build_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    &message,
                );
            }
        }
    } else {
        None
    };
    let duration = match alignment.as_ref().filter(|a| a.duration > 0.0) {
        Some(aligned) => (aligned.duration * DURATION_ROUND_SCALE).round() / DURATION_ROUND_SCALE,
        None => estimate_duration_seconds(prompt_tokens, request.unexpanded_prompt_tokens),
    };
    build_audio_response(
        request.response_format,
        request.endpoint,
        request.language_label.as_deref(),
        duration,
        &cleaned,
        alignment.as_ref(),
        request.include_words,
        request.include_segments,
        prompt_tokens,
        output_tokens,
    )
}

/// Streams OpenAI Audio API server-sent events (`transcript.text.delta` and
/// `transcript.text.done`). Awaits the first worker event up to `keepalive` so
/// immediate refusals or load errors return plain HTTP 400 or 503 before the
/// event stream begins.
pub async fn stream_response(
    request: PreparedAudioRequest,
    cancel: worker::CancelGuard,
    mut events: UnboundedReceiver<worker::Event>,
    keepalive: Duration,
) -> Response {
    let initial = match tokio::time::timeout(keepalive, events.recv()).await {
        Ok(Some(worker::Event::Refused(message))) => {
            return chat::build_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &message,
            );
        }
        Ok(Some(worker::Event::Failed(message))) => {
            return chat::build_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                &message,
            );
        }
        Ok(None) => {
            return chat::build_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "the request ended without an answer",
            );
        }
        Ok(Some(event)) => Some(event),
        Err(_) => None,
    };

    let tokenizer = request.tokenizer;
    let unexpanded_prompt_tokens = request.unexpanded_prompt_tokens;
    let body = async_stream::stream! {
        let _cancel = cancel;
        let mut decoder = Utf8StreamDecoder::default();
        let mut full_text = String::new();
        let mut prompt_tokens = unexpanded_prompt_tokens;
        let mut output_tokens = 0usize;
        let mut started = false;
        let mut pending = initial;
        if pending.is_none() {
            yield Ok::<String, std::io::Error>(": waiting\n\n".into());
        }
        loop {
            let event = match pending.take() {
                Some(event) => Some(event),
                None if !started => loop {
                    match tokio::time::timeout(keepalive, events.recv()).await {
                        Ok(event) => break event,
                        Err(_) => yield Ok(": waiting\n\n".into()),
                    }
                },
                None => events.recv().await,
            };
            let Some(event) = event else {
                yield Ok(format_error_sse("server_error", "the server is stopping"));
                return;
            };
            match event {
                worker::Event::DecodingStarted { prompt_tokens: pt, .. } => {
                    prompt_tokens = pt;
                    started = true;
                }
                worker::Event::Tokens(token_ids) => {
                    started = true;
                    output_tokens += token_ids.len();
                    let chunk = decode_token_chunk(tokenizer, &token_ids, &mut decoder);
                    if !chunk.is_empty() {
                        full_text.push_str(&chunk);
                        yield Ok(format_delta_sse(&chunk));
                    }
                }
                worker::Event::Done { reason } => {
                    let tail = decoder.finish();
                    if !tail.is_empty() {
                        full_text.push_str(&tail);
                        yield Ok(format_delta_sse(&tail));
                    }
                    if reason == worker::FinishReason::ServerStop {
                        yield Ok(format_error_sse("server_error", "the server is stopping"));
                    } else {
                        yield Ok(format_done_sse(&full_text, prompt_tokens, output_tokens));
                    }
                    return;
                }
                worker::Event::Refused(message) => {
                    yield Ok(format_error_sse("invalid_request_error", &message));
                    return;
                }
                worker::Event::Failed(message) => {
                    yield Ok(format_error_sse("server_error", &message));
                    return;
                }
                _ => return,
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes `parts` into a valid RFC 7578 multipart/form-data payload using
    /// `boundary`.
    fn build_multipart(boundary: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(name, filename, data) in parts {
            out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            match filename {
                Some(fname) => out.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n\r\n"
                    )
                    .as_bytes(),
                ),
                None => out.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            out.extend_from_slice(data);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        out
    }

    #[test]
    fn multipart_transcription_request_parses_all_supported_fields() {
        let boundary = "eremite-audio-boundary";
        let ct = format!("multipart/form-data; boundary=\"{boundary}\"");
        let audio = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        let body = build_multipart(
            boundary,
            &[
                ("file", Some("clip.wav"), audio),
                ("model", None, b"whisper-1"),
                ("language", None, b"en-US"),
                ("prompt", None, b"eremite, Apple silicon"),
                ("response_format", None, b"verbose_json"),
                ("temperature", None, b"0.25"),
                ("seed", None, b"42"),
                ("stream", None, b"false"),
            ],
        );
        let parsed = parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &body).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("whisper-1"));
        assert_eq!(parsed.file, audio);
        assert_eq!(parsed.filename.as_deref(), Some("clip.wav"));
        assert_eq!(parsed.language.as_deref(), Some("en-US"));
        assert_eq!(parsed.prompt.as_deref(), Some("eremite, Apple silicon"));
        assert_eq!(parsed.response_format, ResponseFormat::VerboseJson);
        assert_eq!(parsed.temperature, 0.25);
        assert_eq!(parsed.seed, Some(42));
        assert!(!parsed.stream);
        assert_eq!(parsed.endpoint, AudioEndpoint::Transcriptions);
    }

    #[test]
    fn multipart_validation_rejects_missing_or_unsupported_fields() {
        let boundary = "b1";
        let ct = format!("multipart/form-data; boundary={boundary}");
        let wav = b"fake-audio-bytes";

        assert!(
            parse_audio_request(
                AudioEndpoint::Transcriptions,
                Some("application/json"),
                b"{}"
            )
            .is_err()
        );
        let no_file = build_multipart(boundary, &[("model", None, b"whisper-1")]);
        assert!(parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &no_file).is_err());

        let empty_file = build_multipart(boundary, &[("file", Some("a.wav"), b"")]);
        assert!(
            parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &empty_file).is_err()
        );

        let two_files = build_multipart(
            boundary,
            &[("file", Some("a.wav"), wav), ("file", Some("b.wav"), wav)],
        );
        assert!(parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &two_files).is_err());

        let bad_fmt = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("response_format", None, b"xml"),
            ],
        );
        assert!(parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &bad_fmt).is_err());

        // timestamp_granularities requires verbose_json and transcriptions.
        let ts_json = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("timestamp_granularities[]", None, b"word"),
            ],
        );
        assert!(parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &ts_json).is_err());

        let ts_bad_val = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("response_format", None, b"verbose_json"),
                ("timestamp_granularities[]", None, b"character"),
            ],
        );
        assert!(
            parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &ts_bad_val).is_err()
        );

        let ts_translate = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("response_format", None, b"verbose_json"),
                ("timestamp_granularities[]", None, b"word"),
            ],
        );
        assert!(
            parse_audio_request(AudioEndpoint::Translations, Some(&ct), &ts_translate).is_err()
        );

        for fmt in [b"text".as_slice(), b"srt", b"vtt", b"verbose_json"] {
            let bad_stream = build_multipart(
                boundary,
                &[
                    ("file", Some("a.wav"), wav),
                    ("response_format", None, fmt),
                    ("stream", None, b"true"),
                ],
            );
            assert!(
                parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &bad_stream).is_err()
            );
        }

        let neg_temp = build_multipart(
            boundary,
            &[("file", Some("a.wav"), wav), ("temperature", None, b"-0.1")],
        );
        assert!(parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &neg_temp).is_err());

        let lang_on_translate = build_multipart(
            boundary,
            &[("file", Some("a.wav"), wav), ("language", None, b"fr")],
        );
        assert!(
            parse_audio_request(AudioEndpoint::Translations, Some(&ct), &lang_on_translate)
                .is_err()
        );
    }

    #[test]
    fn language_codes_map_to_nemotron_names() {
        assert_eq!(resolve_language_name("en"), "English");
        assert_eq!(resolve_language_name("en-GB"), "English");
        assert_eq!(resolve_language_name("ZH_cn"), "Chinese Mandarin");
        assert_eq!(resolve_language_name("ja"), "Japanese");
        assert_eq!(resolve_language_name("pt-BR"), "Portuguese");
        assert_eq!(resolve_language_name("nb"), "Norwegian");
        assert_eq!(resolve_language_name("fil"), "Tagalog");
        assert_eq!(resolve_language_name("French"), "French");
    }

    #[test]
    fn prompt_rendering_places_audio_sentinel_and_disables_thinking() {
        let transcribe = render_prompt_bytes(AudioEndpoint::Transcriptions, None, None);
        assert!(transcribe.starts_with(b"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"));
        assert!(transcribe.ends_with(b" \xff<|im_end|>\n<|im_start|>assistant\n<think></think>"));
        let count_sentinel = transcribe.iter().filter(|&&b| b == MEDIA_MARK_BYTE).count();
        assert_eq!(count_sentinel, 1);

        let with_lang = build_instruction(AudioEndpoint::Transcriptions, Some("de"), None);
        assert!(with_lang.contains("Language: German."));

        let with_ctx = build_instruction(AudioEndpoint::Transcriptions, None, Some("CUDA, Metal"));
        assert_eq!(
            with_ctx,
            "Transcribe with punctuation and capitalization. Context terms: CUDA, Metal."
        );

        let translate = build_instruction(AudioEndpoint::Translations, None, Some("Nemotron"));
        assert_eq!(
            translate,
            "Please translate the given speech to English. Context terms: Nemotron."
        );
    }

    #[test]
    fn utf8_stream_decoder_preserves_multibyte_chars_across_token_splits() {
        let mut decoder = Utf8StreamDecoder::default();
        let sample = "Hello 世界 🌍!";
        let bytes = sample.as_bytes();
        let mut reconstructed = String::new();
        for chunk in bytes.chunks(1) {
            reconstructed.push_str(&decoder.push_bytes(chunk));
        }
        reconstructed.push_str(&decoder.finish());
        assert_eq!(reconstructed, sample);
    }

    #[test]
    fn loop_guardrail_collapses_char_and_block_runs_only() {
        let normal = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(cleanup_generation_loops(normal), normal);

        let char_loop = format!("hello {} world", "a".repeat(30));
        assert_eq!(cleanup_generation_loops(&char_loop), "hello a world");

        let block_loop = format!("start {}end", "um, ".repeat(25));
        assert_eq!(cleanup_generation_loops(&block_loop), "start um, end");

        let short_repeat = "ha ".repeat(5);
        assert_eq!(cleanup_generation_loops(&short_repeat), short_repeat);
    }

    #[test]
    fn duration_and_sse_formatting_match_openai_schema() {
        // 125 audio tokens * 0.08 s = 10.00 s; raw prompt had 20 tokens (19 text + 1 audio).
        assert_eq!(estimate_duration_seconds(144, 20), 10.0);

        let delta = format_delta_sse("Hello");
        assert!(delta.starts_with("event: transcript.text.delta\ndata: "));
        assert!(delta.contains(r#""type":"transcript.text.delta""#));
        assert!(delta.contains(r#""delta":"Hello""#));

        let done = format_done_sse("Hello world.", 144, 4);
        assert!(done.starts_with("event: transcript.text.done\ndata: "));
        assert!(done.contains(r#""text":"Hello world.""#));
        assert!(done.contains(r#""total_tokens":148"#));
    }

    #[test]
    fn timestamp_granularities_and_caption_formats_parse_correctly() {
        let boundary = "ts-boundary";
        let ct = format!("multipart/form-data; boundary={boundary}");
        let wav = b"fake-audio-bytes";

        // Repeated bracketed fields (curl / OpenAI SDK convention).
        let repeated = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("response_format", None, b"verbose_json"),
                ("timestamp_granularities[]", None, b"word"),
                ("timestamp_granularities[]", None, b"segment"),
            ],
        );
        let parsed = parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &repeated)
            .expect("repeated granularities");
        assert!(parsed.timestamp_granularities.word);
        assert!(parsed.timestamp_granularities.segment);
        assert!(parsed.timestamp_granularities.explicit);

        // JSON array form.
        let json_arr = build_multipart(
            boundary,
            &[
                ("file", Some("a.wav"), wav),
                ("response_format", None, b"verbose_json"),
                ("timestamp_granularities", None, br#"["word"]"#),
            ],
        );
        let parsed = parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &json_arr)
            .expect("json granularities");
        assert!(parsed.timestamp_granularities.word);
        assert!(!parsed.timestamp_granularities.segment);

        // SRT and VTT formats parse on both transcriptions and translations.
        for &(raw, expected) in &[
            (b"srt".as_slice(), ResponseFormat::Srt),
            (b"vtt".as_slice(), ResponseFormat::Vtt),
        ] {
            let body = build_multipart(
                boundary,
                &[("file", Some("a.wav"), wav), ("response_format", None, raw)],
            );
            let p1 = parse_audio_request(AudioEndpoint::Transcriptions, Some(&ct), &body).unwrap();
            let p2 = parse_audio_request(AudioEndpoint::Translations, Some(&ct), &body).unwrap();
            assert_eq!(p1.response_format, expected);
            assert_eq!(p2.response_format, expected);
        }
    }

    #[test]
    fn word_cleaning_segments_srt_and_vtt_format_accurately() {
        assert_eq!(clean_word_text("dog."), "dog");
        assert_eq!(clean_word_text("\"don't!\""), "don't");
        assert_eq!(clean_word_text("3.5,"), "3.5");
        assert_eq!(clean_word_text("..."), "...");

        let transcript = "Hello world. Second sentence here!";
        let words = vec![
            engine::AudioWord {
                byte_start: 0,
                byte_end: 5,
                start: 0.0,
                end: 0.4,
            },
            engine::AudioWord {
                byte_start: 6,
                byte_end: 12,
                start: 0.44,
                end: 1.12,
            },
            engine::AudioWord {
                byte_start: 13,
                byte_end: 19,
                start: 1.6,
                end: 2.05,
            },
            engine::AudioWord {
                byte_start: 20,
                byte_end: 28,
                start: 2.1,
                end: 2.6,
            },
            engine::AudioWord {
                byte_start: 29,
                byte_end: 34,
                start: 2.64,
                end: 3.125,
            },
        ];
        let segments = build_segments(transcript, &words, 3.2);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].id, 0);
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 1.12);
        assert_eq!(segments[0].text, "Hello world.");
        assert_eq!(segments[1].id, 1);
        assert_eq!(segments[1].start, 1.6);
        assert_eq!(segments[1].end, 3.125);
        assert_eq!(segments[1].text, "Second sentence here!");

        let srt = format_srt(&segments);
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:01,120\nHello world.\n\n\
             2\n00:00:01,600 --> 00:00:03,125\nSecond sentence here!\n\n"
        );
        let vtt = format_vtt(&segments);
        assert_eq!(
            vtt,
            "WEBVTT\n\n\
             00:00:00.000 --> 00:00:01.120\nHello world.\n\n\
             00:00:01.600 --> 00:00:03.125\nSecond sentence here!\n\n"
        );
    }
}
