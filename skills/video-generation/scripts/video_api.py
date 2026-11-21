#!/usr/bin/env python3
"""Creates, polls, downloads, lists, deletes and batches videos through the
server's video API, supporting text-to-video, first/last keyframes and
full-reference images, videos and audio.

usage: video_api.py create --prompt TEXT [options]
       video_api.py create-and-poll --prompt TEXT [--download] [--out FILE] [options]
       video_api.py status --id JOB_ID [options]
       video_api.py poll --id JOB_ID [--download] [--out FILE] [options]
       video_api.py download --id JOB_ID [--out FILE] [options]
       video_api.py list [--limit N] [--after JOB_ID] [--order asc|desc] [options]
       video_api.py delete --id JOB_ID [options]
       video_api.py create-batch --input JOBS.jsonl --out-dir DIR [options]

The server's address comes from --base-url or VIDEO_API_URL, and its API key
from the file that --auth-file or VIDEO_API_AUTH_FILE names.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# Environment variable holding the server's base URL, ending in /v1.
URL_VARIABLE = "VIDEO_API_URL"

# Environment variable naming the file that holds the server's API key.
AUTH_FILE_VARIABLE = "VIDEO_API_AUTH_FILE"

# Default output directory for downloaded MP4 clips.
DEFAULT_OUTPUT_DIR = Path("output/videos")

# Default output path for create-and-poll when --download has no --out.
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "video.mp4"

# Default per-request HTTP timeout in seconds (allows uploading up to 256 MiB).
DEFAULT_REQUEST_TIMEOUT = 120.0

# Default seconds between job status polls.
DEFAULT_POLL_INTERVAL = 5.0

# Default maximum seconds to wait for a video job to complete (60 minutes).
DEFAULT_POLL_TIMEOUT = 3600.0

# Chunk size in bytes when streaming downloaded MP4 files to disk (1 MiB).
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# Maximum HTTP request body accepted by POST /v1/videos (256 MiB).
MAX_BODY_BYTES = 256 * 1024 * 1024

# Maximum output width or height in pixels accepted by the video API.
MAX_DIM = 2048

# Maximum output frame area in pixels (2048 by 1152).
MAX_CANVAS_AREA = 2048 * 1152

# Allowed clip duration bounds in whole seconds.
MIN_SECONDS = 1
MAX_SECONDS = 15

# Allowed explicit frame count bounds.
MIN_FRAMES = 1
MAX_FRAMES = 362

# Allowed denoising step bounds.
MIN_STEPS = 1
MAX_STEPS = 100

# Full-reference media count limits enforced by the video API.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_TOTAL_REFS = 12

# Allowed GET /v1/videos page size bounds.
MIN_LIST_LIMIT = 1
MAX_LIST_LIMIT = 100

# Accepted GET /v1/videos sort directions.
LIST_ORDERS = {"asc", "desc"}

# Accepted ref_image_size values.
REF_IMAGE_SIZES = {"auto", "max"}

# Accepted content download variant values.
DOWNLOAD_VARIANTS = {"video"}

# Accepted image file extensions and MIME types.
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
}

# Accepted video file extensions and MIME types.
VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
}

# Accepted audio file extensions and MIME types.
AUDIO_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".caf": "audio/x-caf",
}

# Pattern matching auto or WIDTHxHEIGHT dimensions.
SIZE_PATTERN = re.compile(r"auto|([1-9][0-9]*)x([1-9][0-9]*)", re.IGNORECASE)

# Optional keys accepted in each JSONL object passed to create-batch.
BATCH_JOB_KEYS = {
    "size",
    "seconds",
    "frames",
    "steps",
    "shift_video",
    "shift_audio",
    "seed",
    "model",
    "ref_image_size",
    "input_reference",
    "first_frame",
    "last_frame",
    "ref_images",
    "ref_videos",
    "ref_video_audios",
    "ref_audios",
    "out",
}


class VideoError(Exception):
    """A problem that stops a video API call, with what to fix."""


def check_int(value, name, minimum, maximum=None):
    """Checks that value is an integer (not a boolean) within [minimum, maximum]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise VideoError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise VideoError(f"{name} must be at least {minimum}")
        raise VideoError(f"{name} must be between {minimum} and {maximum}")
    return value


def check_positive_float(value, name):
    """Checks that value is a finite positive number (not a boolean)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise VideoError(f"{name} must be a positive number")
    return result


def check_job_id(raw):
    """Validates a video job identifier before placing it into a URL path."""
    if not isinstance(raw, str) or not raw.strip():
        raise VideoError("--id must be a nonempty job identifier")
    job_id = raw.strip()
    if "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        raise VideoError(f"invalid job id {raw!r}")
    return job_id


def read_prompt(options):
    """Returns the prompt from --prompt or from the file named by --prompt-file."""
    if options.prompt and options.prompt_file:
        raise VideoError("give --prompt or --prompt-file, not both")
    if options.prompt_file:
        path = Path(options.prompt_file)
        if not path.is_file():
            raise VideoError(f"{options.prompt_file}: prompt file does not exist")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise VideoError(f"{options.prompt_file} holds an empty prompt")
        return text
    if options.prompt and options.prompt.strip():
        return options.prompt
    raise VideoError("give the prompt with --prompt or --prompt-file")


def parse_size(raw):
    """Validates size (auto or WIDTHxHEIGHT) against width, height and area limits."""
    if not isinstance(raw, str):
        raise VideoError("size must be a string: auto or WIDTHxHEIGHT")
    trimmed = raw.strip()
    match = SIZE_PATTERN.fullmatch(trimmed)
    if not match:
        raise VideoError(f"--size must be auto or WIDTHxHEIGHT, not {raw!r}")
    if match.group(1) is None:
        return "auto"
    width, height = int(match.group(1)), int(match.group(2))
    if width > MAX_DIM or height > MAX_DIM:
        raise VideoError(f"size dimensions must be between 1 and {MAX_DIM}")
    if width * height > MAX_CANVAS_AREA:
        raise VideoError(f"size {width}x{height} exceeds {MAX_CANVAS_AREA} pixel limit")
    return f"{width}x{height}"


def load_media_file(path_text, media_types, kind_label):
    """Reads and validates a media file, returning (filename, content_type, bytes)."""
    if not isinstance(path_text, str) or not path_text.strip():
        raise VideoError(f"{kind_label} path must be a nonempty string")
    path = Path(path_text)
    suffix = path.suffix.lower()
    content_type = media_types.get(suffix)
    if content_type is None:
        allowed = ", ".join(sorted(media_types))
        raise VideoError(f"{path}: unsupported {kind_label} extension; use {allowed}")
    if not path.is_file():
        raise VideoError(f"{path}: {kind_label} file does not exist")
    raw = path.read_bytes()
    if not raw:
        raise VideoError(f"{path}: {kind_label} file is empty")
    return path.name, content_type, raw


def normalize_media_list(raw, name, allow_empty=False):
    """Normalizes an optional string or list of media paths into a list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise VideoError(f"{name} must be a list of file paths")
    normalized = []
    for item in items:
        if item is None or (isinstance(item, str) and item == ""):
            if allow_empty:
                normalized.append(None)
                continue
            raise VideoError(f"{name} items must be nonempty file paths")
        if not isinstance(item, str):
            raise VideoError(f"{name} items must be file path strings")
        normalized.append(item)
    return normalized


def build_submission(options, prompt):
    """Validates creation options and returns (fields_dict, file_parts_list).

    Each item in file_parts_list is (field_name, filename, content_type, bytes).
    Empty bytes with filename None represent a skipped ref_video_audios slot.
    """
    fields = {"prompt": prompt, "size": parse_size(options.size)}

    if options.seconds is not None and options.frames is not None:
        raise VideoError("specify --seconds or --frames, not both")
    if options.seconds is not None:
        fields["seconds"] = check_int(options.seconds, "seconds", MIN_SECONDS, MAX_SECONDS)
    if options.frames is not None:
        fields["frames"] = check_int(options.frames, "frames", MIN_FRAMES, MAX_FRAMES)
    if options.steps is not None:
        fields["steps"] = check_int(options.steps, "steps", MIN_STEPS, MAX_STEPS)
    if options.shift_video is not None:
        fields["shift_video"] = check_positive_float(options.shift_video, "shift_video")
    if options.shift_audio is not None:
        fields["shift_audio"] = check_positive_float(options.shift_audio, "shift_audio")
    if options.seed is not None:
        fields["seed"] = check_int(options.seed, "seed", 0)
    if options.model is not None:
        if not isinstance(options.model, str) or not options.model.strip():
            raise VideoError("model must be a nonempty string")
        fields["model"] = options.model.strip()

    ref_image_size = None
    if options.ref_image_size is not None:
        if not isinstance(options.ref_image_size, str):
            raise VideoError("ref_image_size must be a string")
        ref_image_size = options.ref_image_size.strip().lower()
        if ref_image_size not in REF_IMAGE_SIZES:
            raise VideoError(
                f"ref_image_size {options.ref_image_size!r} is not supported; use auto or max"
            )
        fields["ref_image_size"] = ref_image_size

    input_reference = options.input_reference
    first_frame = options.first_frame
    last_frame = options.last_frame
    ref_images = normalize_media_list(options.ref_image, "ref_images")
    ref_videos = normalize_media_list(options.ref_video, "ref_videos")
    ref_video_audios = normalize_media_list(
        options.ref_video_audio, "ref_video_audios", allow_empty=True
    )
    ref_audios = normalize_media_list(options.ref_audio, "ref_audios")

    if input_reference and first_frame:
        raise VideoError("specify --input-reference or --first-frame, not both")
    has_keyframes = bool(first_frame or last_frame)
    has_refs = bool(
        ref_images or ref_videos or ref_video_audios or ref_audios or ref_image_size is not None
    )
    if has_keyframes and has_refs:
        raise VideoError(
            "cannot mix keyframe options (--first-frame, --last-frame) with "
            "full-reference options (--ref-image, --ref-video, --ref-audio)"
        )
    if len(ref_video_audios) > len(ref_videos):
        raise VideoError("ref_video_audios cannot have more items than ref_videos")

    effective_ref_images = len(ref_images) + (1 if input_reference and has_refs else 0)
    total_refs = effective_ref_images + len(ref_videos) + len(ref_audios)
    if (
        effective_ref_images > MAX_REF_IMAGES
        or len(ref_videos) > MAX_REF_VIDEOS
        or len(ref_audios) > MAX_REF_AUDIOS
        or total_refs > MAX_TOTAL_REFS
    ):
        raise VideoError(
            f"reference counts exceed limits (images <= {MAX_REF_IMAGES}, "
            f"videos <= {MAX_REF_VIDEOS}, audios <= {MAX_REF_AUDIOS}, total <= {MAX_TOTAL_REFS})"
        )

    files = []
    if input_reference:
        name, ctype, raw = load_media_file(input_reference, IMAGE_MEDIA_TYPES, "input reference")
        files.append(("input_reference", name, ctype, raw))
    if first_frame:
        name, ctype, raw = load_media_file(first_frame, IMAGE_MEDIA_TYPES, "first frame")
        files.append(("first_frame", name, ctype, raw))
    if last_frame:
        name, ctype, raw = load_media_file(last_frame, IMAGE_MEDIA_TYPES, "last frame")
        files.append(("last_frame", name, ctype, raw))
    for item in ref_images:
        name, ctype, raw = load_media_file(item, IMAGE_MEDIA_TYPES, "reference image")
        files.append(("ref_images", name, ctype, raw))
    for item in ref_videos:
        name, ctype, raw = load_media_file(item, VIDEO_MEDIA_TYPES, "reference video")
        files.append(("ref_videos", name, ctype, raw))
    for item in ref_video_audios:
        if item is None:
            files.append(("ref_video_audios", None, "application/octet-stream", b""))
        else:
            name, ctype, raw = load_media_file(item, AUDIO_MEDIA_TYPES, "reference video audio")
            files.append(("ref_video_audios", name, ctype, raw))
    for item in ref_audios:
        name, ctype, raw = load_media_file(item, AUDIO_MEDIA_TYPES, "reference audio")
        files.append(("ref_audios", name, ctype, raw))

    return fields, files


def sanitize_header_value(text):
    """Removes characters that could split or corrupt a multipart header parameter."""
    return text.replace("\\", "_").replace('"', "_").replace("\r", "").replace("\n", "")


def encode_multipart(fields, files):
    """Encodes text fields and binary files into a multipart/form-data payload."""
    payloads = [raw for _, _, _, raw in files]
    while True:
        boundary = f"----video-api-{uuid.uuid4().hex}"
        marker = boundary.encode("ascii")
        if all(marker not in data for data in payloads):
            break

    chunks = []
    delim = f"--{boundary}\r\n".encode("ascii")
    for key, value in fields.items():
        safe_key = sanitize_header_value(key)
        chunks.append(delim)
        chunks.append(f'Content-Disposition: form-data; name="{safe_key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for field_name, filename, content_type, raw in files:
        safe_field = sanitize_header_value(field_name)
        chunks.append(delim)
        if filename is None:
            chunks.append(
                f'Content-Disposition: form-data; name="{safe_field}"\r\n\r\n'.encode("utf-8")
            )
        else:
            safe_file = sanitize_header_value(filename)
            chunks.append(
                f'Content-Disposition: form-data; name="{safe_field}"; '
                f'filename="{safe_file}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
            )
        chunks.append(raw)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(chunks)
    if len(body) > MAX_BODY_BYTES:
        raise VideoError(
            f"request body ({len(body)} bytes) exceeds {MAX_BODY_BYTES} byte (256 MiB) limit"
        )
    return f"multipart/form-data; boundary={boundary}", body


def encode_submission(fields, files):
    """Encodes a creation request as JSON (text-only) or multipart/form-data."""
    if not files:
        body = json.dumps(fields).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise VideoError(
                f"request body ({len(body)} bytes) exceeds {MAX_BODY_BYTES} byte (256 MiB) limit"
            )
        return "application/json", body
    return encode_multipart(fields, files)


def check_output_path(path, force):
    """Checks that path does not already exist unless force is set."""
    if not force and path.exists():
        raise VideoError(f"{path} already exists; pass --force to replace it")
    return path


def resolve_batch_output(out_dir, name):
    """Resolves a batch job's relative output path inside out_dir, refusing escapes."""
    if not isinstance(name, str) or not name.strip():
        raise VideoError("job out must be a nonempty relative file name")
    candidate = Path(name)
    if candidate.is_absolute():
        raise VideoError(f"job out {name!r} must be relative to --out-dir")
    base = Path(out_dir).resolve()
    target = (base / candidate).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise VideoError(f"job out {name!r} escapes --out-dir") from None
    return Path(out_dir) / candidate


def read_settings(options):
    """Returns the server's base URL and API key after validating timeouts."""
    check_positive_float(options.timeout, "--timeout")
    base_url = options.base_url or os.environ.get(URL_VARIABLE)
    auth_file = options.auth_file or os.environ.get(AUTH_FILE_VARIABLE)
    if not base_url:
        raise VideoError(f"set {URL_VARIABLE} or pass --base-url with the server's URL")
    if not auth_file:
        raise VideoError(
            f"set {AUTH_FILE_VARIABLE} or pass --auth-file with the file that holds the API key"
        )
    key_path = Path(auth_file).expanduser()
    if not key_path.is_file():
        raise VideoError(f"{auth_file} does not exist")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise VideoError(f"{auth_file} holds no key")
    return base_url.rstrip("/"), key


def format_http_error(error):
    """Extracts a readable message from an HTTP error response."""
    try:
        payload = json.load(error)
        message = payload["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = error.reason
    return f"the server answered {error.code}: {message}"


def call_json(options, method, path, content_type=None, body=None):
    """Sends an HTTP request to path and parses the JSON response."""
    base_url, key = read_settings(options)
    headers = {"Authorization": f"Bearer {key}"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{base_url}/{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=options.timeout) as answer:
            return json.load(answer)
    except urllib.error.HTTPError as error:
        raise VideoError(format_http_error(error)) from None
    except urllib.error.URLError as error:
        raise VideoError(
            f"cannot reach {base_url}: {error.reason}. A sandbox without network "
            "access blocks the call: ask the user to approve it, or to allow network."
        ) from None
    except TimeoutError:
        raise VideoError(f"request to {base_url}/{path} timed out after {options.timeout}s") from None


def download_content(options, job_id, out_path, variant="video"):
    """Downloads a completed job's MP4 clip atomically into out_path."""
    base_url, key = read_settings(options)
    quoted_id = urllib.parse.quote(job_id, safe="")
    query = urllib.parse.urlencode({"variant": variant})
    url = f"{base_url}/videos/{quoted_id}/content?{query}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.part")
    try:
        with urllib.request.urlopen(request, timeout=options.timeout) as answer:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = answer.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
        os.replace(tmp_path, out_path)
    except urllib.error.HTTPError as error:
        tmp_path.unlink(missing_ok=True)
        raise VideoError(format_http_error(error)) from None
    except urllib.error.URLError as error:
        tmp_path.unlink(missing_ok=True)
        raise VideoError(
            f"cannot reach {base_url}: {error.reason}. A sandbox without network "
            "access blocks the call: ask the user to approve it, or to allow network."
        ) from None
    except (TimeoutError, OSError) as error:
        tmp_path.unlink(missing_ok=True)
        raise VideoError(f"failed to save {out_path}: {error}") from None


def describe_files_for_dry_run(files):
    """Formats attached media parts as concise summaries for --dry-run output."""
    summary = []
    for field_name, filename, content_type, raw in files:
        if filename is None:
            summary.append({"field": field_name, "skipped": True})
        else:
            summary.append(
                {
                    "field": field_name,
                    "filename": filename,
                    "content_type": content_type,
                    "bytes": len(raw),
                }
            )
    return summary


def format_job_failure(job):
    """Formats a failed job's error code and message into a single sentence."""
    err = job.get("error")
    if isinstance(err, dict):
        code = err.get("code") or "failed"
        message = err.get("message") or "unknown error"
        return f"video {job.get('id', '')} failed ({code}): {message}"
    return f"video {job.get('id', '')} failed"


def wait_for_job(options, job_id):
    """Polls GET /v1/videos/{job_id} until completed or failed, printing progress
    changes to stderr."""
    interval = check_positive_float(options.poll_interval, "--poll-interval")
    limit = check_positive_float(options.poll_timeout, "--poll-timeout")
    quoted_id = urllib.parse.quote(job_id, safe="")
    deadline = time.monotonic() + limit
    last_seen = None

    while True:
        job = call_json(options, "GET", f"videos/{quoted_id}")
        status = job.get("status", "unknown")
        progress = job.get("progress", 0)
        state = (status, progress)
        if state != last_seen:
            print(f"{job_id}: {status} ({progress}%)", file=sys.stderr)
            last_seen = state
        if status == "completed":
            return job
        if status == "failed":
            raise VideoError(format_job_failure(job))
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise VideoError(
                f"timed out after {limit:g}s waiting for {job_id} ({status}, {progress}%); "
                f"resume with: poll --id {job_id}"
            )
        time.sleep(min(interval, remaining))


def format_saved_summary(out_path, job):
    """Formats a one-line summary after saving a downloaded MP4 clip."""
    details = [job.get("id", "unknown id")]
    if job.get("size"):
        details.append(str(job["size"]))
    if job.get("seconds"):
        details.append(f"{job['seconds']}s")
    if job.get("frames") is not None and job.get("fps") is not None:
        details.append(f"{job['frames']} frames @ {job['fps']} fps")
    return f"saved {out_path} ({', '.join(details)})"


def cmd_create(options):
    """Queues one video job and prints the server's job object."""
    fields, files = build_submission(options, read_prompt(options))
    content_type, body = encode_submission(fields, files)
    if options.dry_run:
        shown = {
            "method": "POST",
            "path": "videos",
            "content_type": content_type.split(";", 1)[0],
            "body_bytes": len(body),
            "fields": fields,
            "files": describe_files_for_dry_run(files),
        }
        print(json.dumps(shown, indent=2))
        return
    job = call_json(options, "POST", "videos", content_type=content_type, body=body)
    print(json.dumps(job, indent=2))


def cmd_create_and_poll(options):
    """Queues one video job, waits until it finishes, and optionally saves the MP4."""
    check_positive_float(options.poll_interval, "--poll-interval")
    check_positive_float(options.poll_timeout, "--poll-timeout")
    should_download = bool(options.download or options.out)
    out_path = check_output_path(Path(options.out or DEFAULT_OUTPUT), options.force) if should_download else None
    fields, files = build_submission(options, read_prompt(options))
    content_type, body = encode_submission(fields, files)
    if options.dry_run:
        shown = {
            "method": "POST",
            "path": "videos",
            "content_type": content_type.split(";", 1)[0],
            "body_bytes": len(body),
            "fields": fields,
            "files": describe_files_for_dry_run(files),
            "poll": {
                "interval": options.poll_interval,
                "timeout": options.poll_timeout,
            },
            "download": should_download,
            "output": str(out_path) if out_path else None,
        }
        print(json.dumps(shown, indent=2))
        return
    queued = call_json(options, "POST", "videos", content_type=content_type, body=body)
    job_id = queued.get("id")
    if not job_id:
        raise VideoError("the server returned no job id")
    final_job = wait_for_job(options, job_id)
    if out_path is not None:
        download_content(options, job_id, out_path)
        print(format_saved_summary(out_path, final_job))
    else:
        print(json.dumps(final_job, indent=2))


def cmd_status(options):
    """Fetches and prints one video job's status and live progress."""
    job_id = check_job_id(options.id)
    quoted_id = urllib.parse.quote(job_id, safe="")
    if options.dry_run:
        print(json.dumps({"method": "GET", "path": f"videos/{quoted_id}"}, indent=2))
        return
    job = call_json(options, "GET", f"videos/{quoted_id}")
    print(json.dumps(job, indent=2))


def cmd_poll(options):
    """Polls an existing video job until completion, optionally downloading the MP4."""
    job_id = check_job_id(options.id)
    check_positive_float(options.poll_interval, "--poll-interval")
    check_positive_float(options.poll_timeout, "--poll-timeout")
    should_download = bool(options.download or options.out)
    out_path = (
        check_output_path(Path(options.out or (DEFAULT_OUTPUT_DIR / f"{job_id}.mp4")), options.force)
        if should_download
        else None
    )
    if options.dry_run:
        quoted_id = urllib.parse.quote(job_id, safe="")
        shown = {
            "method": "GET",
            "path": f"videos/{quoted_id}",
            "poll": {
                "interval": options.poll_interval,
                "timeout": options.poll_timeout,
            },
            "download": should_download,
            "output": str(out_path) if out_path else None,
        }
        print(json.dumps(shown, indent=2))
        return
    final_job = wait_for_job(options, job_id)
    if out_path is not None:
        download_content(options, job_id, out_path)
        print(format_saved_summary(out_path, final_job))
    else:
        print(json.dumps(final_job, indent=2))


def cmd_download(options):
    """Downloads a completed job's MP4 file."""
    job_id = check_job_id(options.id)
    variant = options.variant.strip().lower()
    if variant not in DOWNLOAD_VARIANTS:
        raise VideoError(f"variant {options.variant!r} is not supported; use video")
    out_path = check_output_path(
        Path(options.out or (DEFAULT_OUTPUT_DIR / f"{job_id}.mp4")),
        options.force,
    )
    quoted_id = urllib.parse.quote(job_id, safe="")
    if options.dry_run:
        print(
            json.dumps(
                {
                    "method": "GET",
                    "path": f"videos/{quoted_id}/content?variant={variant}",
                    "output": str(out_path),
                },
                indent=2,
            )
        )
        return
    download_content(options, job_id, out_path, variant=variant)
    print(f"saved {out_path} ({job_id})")


def cmd_list(options):
    """Lists stored video jobs with optional pagination and ordering."""
    params = {}
    if options.limit is not None:
        params["limit"] = check_int(options.limit, "limit", MIN_LIST_LIMIT, MAX_LIST_LIMIT)
    if options.after is not None:
        params["after"] = check_job_id(options.after)
    if options.order is not None:
        order = options.order.strip().lower()
        if order not in LIST_ORDERS:
            raise VideoError(f"order {options.order!r} is not supported; use asc or desc")
        params["order"] = order
    path = "videos"
    if params:
        path = f"{path}?{urllib.parse.urlencode(params)}"
    if options.dry_run:
        print(json.dumps({"method": "GET", "path": path}, indent=2))
        return
    print(json.dumps(call_json(options, "GET", path), indent=2))


def cmd_delete(options):
    """Cancels an active job or deletes a finished job and its MP4 file."""
    job_id = check_job_id(options.id)
    quoted_id = urllib.parse.quote(job_id, safe="")
    if options.dry_run:
        print(json.dumps({"method": "DELETE", "path": f"videos/{quoted_id}"}, indent=2))
        return
    print(json.dumps(call_json(options, "DELETE", f"videos/{quoted_id}"), indent=2))


def apply_batch_job_overrides(base_options, job):
    """Applies per-job JSONL fields onto a copy of base_options."""
    job_options = argparse.Namespace(**vars(base_options))
    job_options.prompt = job["prompt"]
    job_options.prompt_file = None
    job_options.input_reference = None
    job_options.first_frame = None
    job_options.last_frame = None
    job_options.ref_image = None
    job_options.ref_video = None
    job_options.ref_video_audio = None
    job_options.ref_audio = None

    if "seconds" in job and "frames" not in job:
        job_options.frames = None
    if "frames" in job and "seconds" not in job:
        job_options.seconds = None

    media_map = {
        "ref_images": "ref_image",
        "ref_videos": "ref_video",
        "ref_video_audios": "ref_video_audio",
        "ref_audios": "ref_audio",
    }
    for key in BATCH_JOB_KEYS - {"out"}:
        if key in job:
            target = media_map.get(key, key)
            setattr(job_options, target, job[key])
    return job_options


def cmd_create_batch(options):
    """Runs each job in a JSONL file sequentially, saving finished MP4 clips
    under --out-dir unless --no-wait is passed."""
    if not options.no_wait and not options.out_dir:
        raise VideoError("create-batch needs --out-dir (or pass --no-wait)")
    if not options.no_wait:
        check_positive_float(options.poll_interval, "--poll-interval")
        check_positive_float(options.poll_timeout, "--poll-timeout")

    input_path = Path(options.input)
    if not input_path.is_file():
        raise VideoError(f"{options.input}: batch file does not exist")
    lines = input_path.read_text(encoding="utf-8").splitlines()
    failures = 0

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, str):
                job = {"prompt": parsed}
            elif isinstance(parsed, dict) and isinstance(parsed.get("prompt"), str):
                job = parsed
            else:
                raise VideoError("a job is a prompt string or an object with a prompt")
            unknown = sorted(set(job) - BATCH_JOB_KEYS - {"prompt"})
            if unknown:
                raise VideoError(f"unknown job keys: {', '.join(unknown)}")

            job_options = apply_batch_job_overrides(options, job)
            print(f"job {number}:")
            if options.no_wait:
                cmd_create(job_options)
            else:
                if "out" in job:
                    out_path = resolve_batch_output(options.out_dir, job["out"])
                else:
                    out_path = Path(options.out_dir) / f"job_{number}.mp4"
                job_options.download = True
                job_options.out = str(out_path)
                cmd_create_and_poll(job_options)
        except (VideoError, ValueError, TypeError, OSError) as error:
            failures += 1
            print(f"job {number} failed: {error}", file=sys.stderr)
            if options.fail_fast:
                break

    if failures:
        raise VideoError(f"{failures} jobs failed")


def add_connection_options(parser):
    """Adds server URL, auth file, HTTP timeout and dry-run options."""
    parser.add_argument("--base-url", help=f"server base URL ending in /v1 (default ${URL_VARIABLE})")
    parser.add_argument(
        "--auth-file", help=f"file holding the API key (default ${AUTH_FILE_VARIABLE})"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="seconds to wait per HTTP request",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print request only")


def add_creation_options(parser, include_prompt_and_media=True):
    """Adds video generation parameter options to a command parser."""
    if include_prompt_and_media:
        parser.add_argument("--prompt", help="the full structured audiovisual prompt")
        parser.add_argument("--prompt-file", help="UTF-8 text file holding the prompt")
    parser.add_argument("--model", help="target video model when the server serves several")
    parser.add_argument("--size", default="auto", help="auto or WIDTHxHEIGHT (up to 2048x1152 area)")
    parser.add_argument("--seconds", type=int, help="clip length in whole seconds (1 to 15)")
    parser.add_argument("--frames", type=int, help="explicit frame count (1 to 362)")
    parser.add_argument("--steps", type=int, help="denoising steps (1 to 100)")
    parser.add_argument("--shift-video", type=float, help="positive video flow-shift override")
    parser.add_argument("--shift-audio", type=float, help="positive audio flow-shift override")
    parser.add_argument("--seed", type=int, help="nonnegative seed for reproducible output")
    parser.add_argument(
        "--ref-image-size",
        choices=sorted(REF_IMAGE_SIZES),
        help="full-reference image scaling: auto or max (2048px short edge)",
    )
    if include_prompt_and_media:
        parser.add_argument(
            "--input-reference",
            help="single reference image (first frame on base models, first ref image on full-ref)",
        )
        parser.add_argument("--first-frame", help="base-mode opening keyframe image")
        parser.add_argument("--last-frame", help="base-mode closing keyframe image")
        parser.add_argument(
            "--ref-image", action="append", help="full-reference image file (repeatable, up to 9)"
        )
        parser.add_argument(
            "--ref-video", action="append", help="full-reference video file (repeatable, up to 3)"
        )
        parser.add_argument(
            "--ref-video-audio",
            action="append",
            help="explicit soundtrack paired with --ref-video (pass empty string to skip a slot)",
        )
        parser.add_argument(
            "--ref-audio", action="append", help="full-reference audio file (repeatable, up to 3)"
        )


def add_poll_options(parser, include_download=True):
    """Adds polling interval, wait limit and optional MP4 download options."""
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="seconds between status checks",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=DEFAULT_POLL_TIMEOUT,
        help="maximum seconds to wait for job completion",
    )
    if include_download:
        parser.add_argument(
            "--download",
            action="store_true",
            help="download the finished MP4 when completed (implied by --out)",
        )
        parser.add_argument("--out", help="output MP4 file path")
        parser.add_argument("--force", action="store_true", help="replace existing output files")


def build_parser():
    """Builds and returns the CLI argument parser for all video API commands."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p_create = commands.add_parser("create", help="queue a video generation job")
    add_creation_options(p_create)
    add_connection_options(p_create)

    p_run = commands.add_parser(
        "create-and-poll", help="queue a job, wait for completion and optionally download MP4"
    )
    add_creation_options(p_run)
    add_poll_options(p_run)
    add_connection_options(p_run)

    p_status = commands.add_parser("status", help="fetch a video job's status and progress")
    p_status.add_argument("--id", required=True, help="video job identifier")
    add_connection_options(p_status)

    p_poll = commands.add_parser("poll", help="poll an existing job until completion")
    p_poll.add_argument("--id", required=True, help="video job identifier")
    add_poll_options(p_poll)
    add_connection_options(p_poll)

    p_download = commands.add_parser("download", help="download a completed job's MP4 file")
    p_download.add_argument("--id", required=True, help="video job identifier")
    p_download.add_argument(
        "--variant", default="video", choices=sorted(DOWNLOAD_VARIANTS), help="content variant"
    )
    p_download.add_argument("--out", help="output MP4 path (default output/videos/<id>.mp4)")
    p_download.add_argument("--force", action="store_true", help="replace existing output file")
    add_connection_options(p_download)

    p_list = commands.add_parser("list", help="list stored video jobs")
    p_list.add_argument("--limit", type=int, help="maximum jobs to return (1 to 100)")
    p_list.add_argument("--after", help="pagination cursor job identifier")
    p_list.add_argument("--order", choices=sorted(LIST_ORDERS), help="sort order: asc or desc")
    add_connection_options(p_list)

    p_delete = commands.add_parser("delete", help="cancel or delete a video job and its MP4")
    p_delete.add_argument("--id", required=True, help="video job identifier")
    add_connection_options(p_delete)

    p_batch = commands.add_parser("create-batch", help="run multiple jobs from a JSONL file")
    p_batch.add_argument("--input", required=True, help="JSONL file of video jobs")
    p_batch.add_argument("--out-dir", help="folder for finished MP4 clips")
    p_batch.add_argument(
        "--no-wait", action="store_true", help="queue every job without waiting or downloading"
    )
    p_batch.add_argument("--fail-fast", action="store_true", help="stop at the first failed job")
    p_batch.add_argument("--force", action="store_true", help="replace existing output files")
    add_creation_options(p_batch, include_prompt_and_media=False)
    add_poll_options(p_batch, include_download=False)
    add_connection_options(p_batch)

    return parser


def main(arguments):
    """Dispatches the selected CLI subcommand and returns an exit code."""
    options = build_parser().parse_args(arguments)
    handlers = {
        "create": cmd_create,
        "create-and-poll": cmd_create_and_poll,
        "status": cmd_status,
        "poll": cmd_poll,
        "download": cmd_download,
        "list": cmd_list,
        "delete": cmd_delete,
        "create-batch": cmd_create_batch,
    }
    try:
        handlers[options.command](options)
        return 0
    except VideoError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
