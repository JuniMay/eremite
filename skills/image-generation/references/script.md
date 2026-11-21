# The image API script

`scripts/image_api.py` calls the server's image API directly. Use it instead
of the built-in image tool when the user asks for options the tool leaves to
the server: an exact size or aspect ratio, several variants of one prompt, a
JPEG or WebP file, a compression level, a seed, a custom minimum canvas area,
a mask for an edit, or a batch of prompts (or when the built-in image tool is
unavailable). For ordinary single generations or edits where the default size
and PNG work, prefer the built-in image tool.

Treat `scripts/image_api.py` as read-only: run it as provided rather than
modifying it or writing ad-hoc API scripts.

## Settings

- `IMAGE_API_URL`: the server's base URL, ending in `/v1` (or `--base-url`).
- `IMAGE_API_AUTH_FILE`: path to a file that holds the server's API key (or
  `--auth-file`).

When one is missing, tell the user which one to set. Never ask for the key
itself.

## Network

The script needs network access to reach the server, even when the server
runs on this computer. A sandbox without network access blocks it: then ask
the user to approve running the command outside the sandbox. The user can
allow sandbox network access in the agent's settings to stop repeated
approvals.

## Commands

Run the script by its path in this skill's folder. Write a long prompt to a
file and pass `--prompt-file`, so that the shell leaves it as it is.

- New images:
  `python3 scripts/image_api.py generate --prompt-file prompt.txt --size 1536x1024 --out assets/hero.png`
- An edit:
  `python3 scripts/image_api.py edit --image room.png --image chair.png --prompt-file edit.txt --out assets/room-chair.png`.
  The images are `<Picture 1>`, `<Picture 2>` and so on, in the order given
  (up to 16 images). `--mask FILE` takes a PNG or WebP image with an alpha
  channel, the size of the first image: its clear pixels mark what may change,
  and its opaque pixels keep the first image's pixels. With a mask, `--size`
  must be `auto` or the first image's size.
- Many prompts:
  `python3 scripts/image_api.py generate-batch --input jobs.jsonl --out-dir assets/icons`.
  Each line is a JSON prompt string or a JSON object with `prompt`, and
  optionally `size`, `n`, `background`, `output_format`, `output_compression`,
  `seed`, `min_generation_area`, `model` and `out` (a relative file name under
  `--out-dir`). Pass `--fail-fast` to stop at the first failure. The jobs run
  one at a time.

## Options

- `--size WIDTHxHEIGHT`: the exact size of the output, each side 1 to 8192
  pixels and at most 33,554,432 pixels in all the images of one call. `auto`,
  the default, lets the server choose.
- `--n N`: how many images of one prompt to make (1 to 16), saved as `NAME-1`,
  `NAME-2` and so on.
- `--background auto|opaque|transparent`: a transparent background needs PNG
  or WebP; [transparency.md](transparency.md) says how to write its prompt.
- `--output-format png|jpeg|jpg|webp`, and `--output-compression 0..100` for
  JPEG and WebP.
- `--seed N`: nonnegative integer; the same seed and prompt give the same
  image again.
- `--min-generation-area AREA`: minimum generation canvas area in pixels
  (1024 to 16777216; server default 4194304). Pass `1024` to disable automatic
  2048x2048 upscaling on smaller `--size` calls.
- `--model ID`: target image model when the server serves more than one.
- `--out FILE`, by default `output/images/image.png`, or `--out-dir DIR` for
  `image_1`, `image_2` and so on (do not pass both). `--force` replaces
  existing files, which the script otherwise refuses.
- `--dry-run` validates arguments, prints the request and output paths, and
  sends nothing.

After the script saves the images, look at each one, and tell the user its
path and the prompt that made it.
