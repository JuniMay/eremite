#!/usr/bin/env python3
"""Makes or edits images through the server's image API, with the options
that the image tool leaves to the server: an exact size, several images, the
background, the file format and its compression, a seed, a canvas area floor,
an edit's mask, and JSONL batch jobs.

usage: image_api.py generate --prompt TEXT [options]
       image_api.py edit --image FILE [--image FILE ...] [--mask FILE] --prompt TEXT [options]
       image_api.py generate-batch --input JOBS.jsonl --out-dir DIR [options]

The server's address comes from --base-url or IMAGE_API_URL, and its API key
from the file that --auth-file or IMAGE_API_AUTH_FILE names.
"""

import argparse
import base64
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The environment variable that gives the server's base URL, ending in /v1.
URL_VARIABLE = "IMAGE_API_URL"

# The environment variable that names the file holding the server's API key.
AUTH_FILE_VARIABLE = "IMAGE_API_AUTH_FILE"

# The file name that a request's output gets when the call names none.
DEFAULT_OUTPUT = Path("output/images/image.png")

# Seconds that a call may wait for the server's answer. A large image takes
# minutes, and a waiting queue adds more.
DEFAULT_TIMEOUT = 900.0

# Maximum width or height in pixels accepted by the image API.
MAX_DIM = 8192

# Maximum number of generated images per request.
MAX_IMAGES_PER_REQUEST = 16

# Maximum number of reference images per edit request.
MAX_REFERENCES = 16

# Maximum total output pixels across all images in one request (32 Mi pixels).
MAX_REQUEST_PIXELS = 32 * 1024 * 1024

# Lowest JPEG or WebP compression quality value.
MIN_COMPRESSION = 0

# Highest JPEG or WebP compression quality value.
MAX_COMPRESSION = 100

# Lowest minimum generation canvas area in pixels (one 32x32 patch).
MIN_GENERATION_AREA = 1024

# Highest minimum generation canvas area in pixels (4096x4096).
MAX_GENERATION_AREA = 4096 * 4096

# Output format aliases normalized to the canonical API value.
FORMAT_ALIASES = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg", "webp": "webp"}

# The file extension of each canonical output format.
EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}

# Output extensions that output_paths may replace when the format changes.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Accepted background values.
BACKGROUNDS = {"auto", "opaque", "transparent"}

# The media type of each image file extension that an edit may send.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
}

# Mask file extensions that carry an alpha channel.
MASK_SUFFIXES = {".png", ".webp"}

# A size: auto, or a width and a height in pixels.
SIZE_PATTERN = re.compile(r"auto|([1-9][0-9]*)x([1-9][0-9]*)", re.IGNORECASE)

# The keys that a batch job may set, beside its prompt.
JOB_KEYS = {
    "size",
    "n",
    "background",
    "output_format",
    "output_compression",
    "seed",
    "min_generation_area",
    "model",
    "out",
}


class ImageError(Exception):
    """A problem that stops a call, with what to fix."""


def check_int(value, name, minimum, maximum=None):
    """Checks that value is an integer (not a boolean) within [minimum, maximum]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImageError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ImageError(f"{name} must be at least {minimum}")
        raise ImageError(f"{name} must be between {minimum} and {maximum}")
    return value


def read_prompt(options):
    """Returns the prompt from --prompt or from the file that --prompt-file
    names."""
    if options.prompt and options.prompt_file:
        raise ImageError("give --prompt or --prompt-file, not both")
    if options.prompt_file:
        text = Path(options.prompt_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ImageError(f"{options.prompt_file} holds an empty prompt")
        return text
    if options.prompt:
        return options.prompt
    raise ImageError("give the prompt with --prompt or --prompt-file")


def parse_size(raw, count):
    """Validates size (auto or WIDTHxHEIGHT) and total pixels across count."""
    if not isinstance(raw, str):
        raise ImageError("size must be a string: auto or WIDTHxHEIGHT")
    trimmed = raw.strip()
    match = SIZE_PATTERN.fullmatch(trimmed)
    if not match:
        raise ImageError(f"--size must be auto or WIDTHxHEIGHT, not {raw!r}")
    if match.group(1) is None:
        return "auto"
    width, height = int(match.group(1)), int(match.group(2))
    if width > MAX_DIM or height > MAX_DIM:
        raise ImageError(f"size dimensions must be between 1 and {MAX_DIM}")
    total_pixels = width * height * count
    if total_pixels > MAX_REQUEST_PIXELS:
        raise ImageError(
            f"total output pixels ({width}x{height} x {count} = {total_pixels}) "
            f"exceed {MAX_REQUEST_PIXELS} pixel limit"
        )
    return f"{width}x{height}"


def build_body(options, prompt):
    """Returns the validated JSON request body of a generation, from the
    options that were given; the server picks what is left out."""
    count = check_int(options.n, "n", 1, MAX_IMAGES_PER_REQUEST)
    size = parse_size(options.size, count)
    body = {"prompt": prompt, "size": size, "n": count}

    output_format = None
    if options.output_format is not None:
        if not isinstance(options.output_format, str):
            raise ImageError("output_format must be a string")
        output_format = FORMAT_ALIASES.get(options.output_format.strip().lower())
        if output_format is None:
            raise ImageError(
                f"output_format {options.output_format!r} is not supported; use png, jpeg or webp"
            )
        body["output_format"] = output_format

    if options.background is not None:
        if not isinstance(options.background, str):
            raise ImageError("background must be a string")
        background = options.background.strip().lower()
        if background not in BACKGROUNDS:
            raise ImageError(
                f"background {options.background!r} is not supported; use auto, opaque or transparent"
            )
        if background == "transparent" and output_format == "jpeg":
            raise ImageError("a transparent background needs output_format png or webp")
        body["background"] = background

    if options.output_compression is not None:
        body["output_compression"] = check_int(
            options.output_compression,
            "output_compression",
            MIN_COMPRESSION,
            MAX_COMPRESSION,
        )
    if options.seed is not None:
        body["seed"] = check_int(options.seed, "seed", 0)
    if options.min_generation_area is not None:
        body["min_generation_area"] = check_int(
            options.min_generation_area,
            "min_generation_area",
            MIN_GENERATION_AREA,
            MAX_GENERATION_AREA,
        )
    if options.model is not None:
        if not isinstance(options.model, str) or not options.model.strip():
            raise ImageError("model must be a nonempty string")
        body["model"] = options.model.strip()
    return body


def data_url(path, allowed_suffixes=None):
    """Returns an image file as a base64 data URL after checking it exists."""
    suffix = path.suffix.lower()
    if allowed_suffixes is not None and suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ImageError(f"{path}: mask must use an alpha format ({allowed})")
    media_type = MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ImageError(f"{path}: use a PNG, JPEG, WebP, GIF or HEIC file")
    if not path.is_file():
        raise ImageError(f"{path}: image file does not exist")
    raw = path.read_bytes()
    if not raw:
        raise ImageError(f"{path}: image file is empty")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def apply_extension(path, extension):
    """Sets path's file extension to extension, replacing an existing image
    suffix or appending extension when the name has no image suffix."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return path.with_suffix(extension)
    return path.with_name(f"{path.name}{extension}")


def output_paths(options, count, extension):
    """Returns the files that count images go to: --out-dir's image_1 and so
    on, --out itself for one image, or --out with -1, -2 and so on for more.
    The extension follows the output format. Refuses files that exist,
    unless --force allows it."""
    if options.out and options.out_dir:
        raise ImageError("give --out or --out-dir, not both")
    if options.out_dir:
        folder = Path(options.out_dir)
        paths = [folder / f"image_{i}{extension}" for i in range(1, count + 1)]
    else:
        out = apply_extension(Path(options.out or DEFAULT_OUTPUT), extension)
        if count == 1:
            paths = [out]
        else:
            paths = [out.with_name(f"{out.stem}-{i}{extension}") for i in range(1, count + 1)]
    if not options.force:
        taken = [str(path) for path in paths if path.exists()]
        if taken:
            raise ImageError(f"{', '.join(taken)} already exists; pass --force to replace it")
    return paths


def resolve_batch_output(out_dir, name):
    """Resolves a batch job's relative output name inside out_dir, refusing
    paths that leave out_dir."""
    if not isinstance(name, str) or not name.strip():
        raise ImageError("job out must be a nonempty relative file name")
    candidate = Path(name)
    if candidate.is_absolute():
        raise ImageError(f"job out {name!r} must be relative to --out-dir")
    base = Path(out_dir).resolve()
    target = (base / candidate).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ImageError(f"job out {name!r} escapes --out-dir") from None
    return Path(out_dir) / candidate


def read_settings(options):
    """Returns the server's base URL and its API key."""
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        raise ImageError("--timeout must be a positive number of seconds")
    base_url = options.base_url or os.environ.get(URL_VARIABLE)
    auth_file = options.auth_file or os.environ.get(AUTH_FILE_VARIABLE)
    if not base_url:
        raise ImageError(f"set {URL_VARIABLE} or pass --base-url with the server's URL")
    if not auth_file:
        raise ImageError(
            f"set {AUTH_FILE_VARIABLE} or pass --auth-file with the file that holds the API key"
        )
    key_path = Path(auth_file).expanduser()
    if not key_path.is_file():
        raise ImageError(f"{auth_file} does not exist")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ImageError(f"{auth_file} holds no key")
    return base_url.rstrip("/"), key


def call_server(options, path, body):
    """Sends body to the server's image API at path and returns the JSON
    answer. Raises ImageError with the server's message when it refuses."""
    base_url, key = read_settings(options)
    request = urllib.request.Request(
        f"{base_url}/{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=options.timeout) as answer:
            return json.load(answer)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = error.reason
        raise ImageError(f"the server answered {error.code}: {message}") from None
    except urllib.error.URLError as error:
        raise ImageError(
            f"cannot reach {base_url}: {error.reason}. A sandbox without network "
            "access blocks the call: ask the user to approve it, or to allow network."
        ) from None


def save_images(answer, paths):
    """Writes the images of the server's answer to paths, and prints each
    path with the image's size."""
    images = [item.get("b64_json") for item in answer.get("data", [])]
    if len(images) != len(paths) or not all(images):
        raise ImageError(f"the server returned {len(images)} images, not {len(paths)}")
    for image, path in zip(images, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(image))
        print(f"saved {path} ({answer.get('size', 'unknown size')})")


def run(options, path, body):
    """Sends one request, or prints it with --dry-run, and saves its images."""
    extension = EXTENSIONS[body.get("output_format", "png")]
    paths = output_paths(options, body["n"], extension)
    if options.dry_run:
        shown = dict(body)
        if "images" in shown:
            shown["images"] = [f"<{len(shown['images'])} images>"]
        if "mask" in shown:
            shown["mask"] = "<mask>"
        print(json.dumps({"path": path, "body": shown, "outputs": [str(p) for p in paths]}, indent=2))
        return
    save_images(call_server(options, path, body), paths)


def generate(options):
    """Makes new images from a prompt."""
    run(options, "images/generations", build_body(options, read_prompt(options)))


def edit(options):
    """Edits images from a prompt: the images in order, and an optional mask
    whose clear pixels mark what may change."""
    if len(options.image) > MAX_REFERENCES:
        raise ImageError(f"at most {MAX_REFERENCES} reference images are supported")
    body = build_body(options, read_prompt(options))
    body["images"] = [{"image_url": data_url(Path(image))} for image in options.image]
    if options.mask:
        body["mask"] = {"image_url": data_url(Path(options.mask), MASK_SUFFIXES)}
    run(options, "images/edits", body)


def generate_batch(options):
    """Makes the images of each job in a JSONL file, one job at a time. Each
    line is a prompt string or a JSON object with prompt and optional size, n,
    background, output_format, output_compression, seed, min_generation_area,
    model and out (a relative file name under --out-dir)."""
    if not options.out_dir:
        raise ImageError("generate-batch needs --out-dir")
    failures = 0
    lines = Path(options.input).read_text(encoding="utf-8").splitlines()
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
                raise ImageError("a job is a prompt string or an object with a prompt")
            unknown = sorted(set(job) - JOB_KEYS - {"prompt"})
            if unknown:
                raise ImageError(f"unknown job keys: {', '.join(unknown)}")
            job_options = argparse.Namespace(**vars(options))
            for key in JOB_KEYS:
                if key in job:
                    setattr(job_options, key, job[key])
            job_options.prompt, job_options.prompt_file = job["prompt"], None
            if "out" in job:
                out_path = resolve_batch_output(options.out_dir, job["out"])
            else:
                out_path = Path(options.out_dir) / f"job_{number}.png"
            job_options.out, job_options.out_dir = str(out_path), None
            print(f"job {number}:")
            generate(job_options)
        except (ImageError, ValueError, TypeError, OSError) as error:
            failures += 1
            print(f"job {number} failed: {error}", file=sys.stderr)
            if options.fail_fast:
                break
    if failures:
        raise ImageError(f"{failures} jobs failed")


def add_shared_options(parser):
    """Adds the options that every command takes."""
    parser.add_argument("--prompt", help="the full description of the image")
    parser.add_argument("--prompt-file", help="a file that holds the prompt")
    parser.add_argument("--size", default="auto", help="WIDTHxHEIGHT, or auto for the server's choice")
    parser.add_argument("--n", type=int, default=1, help="how many images to make (1 to 16)")
    parser.add_argument("--background", choices=sorted(BACKGROUNDS))
    parser.add_argument("--output-format", choices=sorted(FORMAT_ALIASES))
    parser.add_argument("--output-compression", type=int, help="JPEG or WebP quality, 0 to 100")
    parser.add_argument("--seed", type=int, help="a seed, to make the same image again")
    parser.add_argument(
        "--min-generation-area",
        type=int,
        help="minimum canvas area in pixels (1024 to 16777216; 1024 disables upscaling)",
    )
    parser.add_argument("--model", help="the image model, when the server has several")
    parser.add_argument("--out", help=f"the output file (default {DEFAULT_OUTPUT})")
    parser.add_argument("--out-dir", help="a folder for the outputs: image_1, image_2 and so on")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument("--dry-run", action="store_true", help="print the request and send nothing")
    parser.add_argument("--base-url", help=f"the server's base URL (default ${URL_VARIABLE})")
    parser.add_argument("--auth-file", help=f"the file that holds the API key (default ${AUTH_FILE_VARIABLE})")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds to wait for the server"
    )


def main(arguments):
    """Runs the command that arguments name, and returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, helped in (
        ("generate", "make new images"),
        ("edit", "edit images"),
        ("generate-batch", "make the images of many jobs"),
    ):
        command = commands.add_parser(name, help=helped)
        add_shared_options(command)
        if name == "edit":
            command.add_argument("--image", action="append", required=True, help="an image to edit, in order")
            command.add_argument("--mask", help="a PNG or WebP image whose clear pixels mark what may change")
        if name == "generate-batch":
            command.add_argument("--input", required=True, help="a JSONL file of jobs")
            command.add_argument("--fail-fast", action="store_true", help="stop at the first failed job")
    options = parser.parse_args(arguments)
    try:
        {"generate": generate, "edit": edit, "generate-batch": generate_batch}[options.command](options)
    except (ImageError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
