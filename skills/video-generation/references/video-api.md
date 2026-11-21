# The video API script

`scripts/video_api.py` calls the server's video API directly to create, poll,
download, list, delete and batch video generation jobs.

Treat `scripts/video_api.py` as read-only: run it as provided rather than
modifying it or writing ad-hoc API scripts.

## Settings

- `VIDEO_API_URL`: the server's base URL, ending in `/v1` (or `--base-url`).
- `VIDEO_API_AUTH_FILE`: path to a file that holds the server's API key (or
  `--auth-file`).

When one is missing, tell the user which one to set. Never ask for the key
itself.

## Network

The script needs network access to reach the server, even when the server runs
on this computer. A sandbox without network access blocks the call: ask the
user to approve running the command outside the sandbox. The user can allow
sandbox network access in the agent's settings to stop repeated approvals.

## Commands

Run the script by its path in this skill's folder. Write long structured prompts
to a text file and pass `--prompt-file` so the shell preserves quotes, tags and
line breaks.

- Create, wait and download (default workflow):
  `python3 scripts/video_api.py create-and-poll --prompt-file prompt.txt --seconds 5 --size 1344x768 --out assets/intro.mp4`
- First-and-last-frame animation (base mode):
  `python3 scripts/video_api.py create-and-poll --prompt-file fl2va.txt --first-frame start.png --last-frame end.png --seconds 6 --out assets/transition.mp4`
- Full-reference generation or video editing:
  `python3 scripts/video_api.py create-and-poll --prompt-file ref.txt --ref-image hero.png --ref-video clip.mp4 --ref-audio voice.wav --seconds 6 --out assets/scene.mp4`
- Queue without waiting:
  `python3 scripts/video_api.py create --prompt-file prompt.txt --seconds 5`
- Check one job's status and live progress:
  `python3 scripts/video_api.py status --id video_...`
- Wait for an existing job and save its MP4:
  `python3 scripts/video_api.py poll --id video_... --out assets/intro.mp4`
- Download a completed job's MP4:
  `python3 scripts/video_api.py download --id video_... --out assets/intro.mp4`
- List stored jobs:
  `python3 scripts/video_api.py list --limit 20 --order desc`
- Cancel an active job or remove a finished job and its MP4:
  `python3 scripts/video_api.py delete --id video_...`
- Run a JSONL batch sequentially:
  `python3 scripts/video_api.py create-batch --input jobs.jsonl --out-dir assets/clips`

## Creation options

- `--prompt TEXT` or `--prompt-file FILE`: required prompt, passed verbatim
  (truncated at 4,096 tokens by the model).
- `--model ID`: target video model when the server serves more than one.
- `--size auto|WIDTHxHEIGHT`: output resolution (`auto` by default). Each side
  is 1 to 2048 pixels, total area at most 2,359,296 pixels (2048 by 1152),
  rounded to multiples of 32.
- `--seconds 1..15`: clip length in whole seconds (default 5). Mutually
  exclusive with `--frames`.
- `--frames 1..362`: explicit frame count (aligned to the `17k + 5` grid,
  default 124). Mutually exclusive with `--seconds`.
- `--steps 1..100`: denoising steps (omitted uses the server's configured
  default).
- `--shift-video FLOAT` and `--shift-audio FLOAT`: optional positive flow-shift
  overrides.
- `--seed N`: nonnegative integer for reproducible output.
- `--input-reference FILE`: single reference image; acts as `--first-frame` on
  base models or prepends to `--ref-image` on full-reference models. Cannot be
  combined with `--first-frame`.
- `--first-frame FILE` and `--last-frame FILE`: base-mode keyframe images.
  Cannot be combined with full-reference flags.
- `--ref-image FILE` (repeatable, up to 9), `--ref-video FILE` (repeatable, up
  to 3), `--ref-video-audio FILE` (repeatable, index-paired with `--ref-video`;
  pass `""` to keep a video slot's own audio track while overriding a later
  slot), `--ref-audio FILE` (repeatable, up to 3), and `--ref-image-size
  auto|max` (`max` scales reference images to a 2048-pixel short edge):
  full-reference mode only. Total images, videos and standalone audios may not
  exceed 12.
- `--poll-interval SECONDS`: seconds between status checks (default `5.0`).
- `--poll-timeout SECONDS`: maximum wait while polling (default `3600.0`).
- `--timeout SECONDS`: HTTP timeout per request (default `120.0`).
- `--download` and `--out FILE`: save the completed MP4. Passing `--out` on
  `create-and-poll` or `poll` enables downloading automatically. Without
  `--out`, `--download` saves to `output/videos/video.mp4` (`create-and-poll`)
  or `output/videos/<id>.mp4` (`poll` and `download`).
- `--force`: replace an existing destination file, which the script refuses by
  default.
- `--dry-run`: validate arguments, check input files and overwrite rules, print
  the planned request as JSON and send nothing.

## Batch file format (`create-batch`)

Each non-empty line of `--input JOBS.jsonl` is either a JSON prompt string or a
JSON object with `prompt` and optional keys: `size`, `seconds`, `frames`,
`steps`, `shift_video`, `shift_audio`, `seed`, `model`, `ref_image_size`,
`input_reference`, `first_frame`, `last_frame`, `ref_images`, `ref_videos`,
`ref_video_audios`, `ref_audios` and `out` (a relative file name inside
`--out-dir`; defaults to `job_<line>.mp4`).

Jobs run one at a time. Pass `--no-wait` to queue every job without waiting for
completion, or `--fail-fast` to stop at the first failure.

## Troubleshooting

- **HTTP 400 (`first_frame and last_frame require...` or `reference images...
  require...`)**: the flags do not match the selected model's mode. Pass
  `--model` for the matching model or switch between base-mode and
  full-reference flags.
- **HTTP 400 (`negative_prompt` or `guidance_scale`)**: guidance scale is fixed
  at 1 and negative prompts are unsupported; describe what to show positively.
- **HTTP 409 on download**: the job is still `queued`, `in_progress` or
  `failed`. Run `poll --id <id>` or `status --id <id>` first.
- **HTTP 413**: total upload size exceeds 256 MiB. Trim or re-encode large
  reference videos before retrying.
- **HTTP 503**: memory is temporarily busy with another request. Wait briefly
  and retry, or reduce `--size` or `--seconds`.
- **Job `status: "failed"`**: read the `error.code` and `error.message` fields
  from `status --id <id>` (for example an unreadable media file, a reference
  video shorter than 5 frames, or insufficient memory for the requested canvas).
- **Poll timed out**: the job may still be running on the server. Resume with
  `poll --id <id> --out <file>` or cancel it with `delete --id <id>`.
