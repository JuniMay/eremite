---
name: video-generation
description: "Make or edit short videos with synchronized sound through the video API, from text, first or last keyframes, or reference images, videos and audio, using structured audiovisual prompts. Use for new clips, keyframe animation, video editing, video continuation, or reference-guided video generation. Do not use for static images, animations that code or CSS draws better, or trims and conversions that ffmpeg handles directly."
---

# Video generation

The video API makes an MP4 clip with synchronized video and stereo sound from a
prompt and optional reference media. No second model rewrites or lengthens the
prompt on the way, so the prompt you send is the whole audiovisual script.

## When to use this skill

Use it for generated or edited audiovisual clips. When the result should be a
static image, use image generation instead. When the result is a UI animation,
chart or motion graphic that code, CSS or SVG draws better, write the code.
When the user only wants to trim, resize, splice or convert existing media, run
`ffmpeg` directly.

## Pick the mode

The server serves models in one of two modes. Match the prompt format and CLI
flags to the task and target model:

- **Base mode** (text only, first frame, first and last frames, or last frame):
  read [references/prompting.md](references/prompting.md).
- **Full-reference mode** (reference images, reference videos, reference audio,
  video editing, or video continuation): read
  [references/references-mode.md](references/references-mode.md).

Read [references/video-api.md](references/video-api.md) for settings,
subcommands, batch files and error handling.

## Workflow

1. Inspect every input media file before writing the prompt. Look at each input
   image with the image-viewing tool. For an input video, check its size,
   duration, audio tracks and sample frames before describing it.
2. Choose Base mode or Full-reference mode and write the full structured prompt
   to a UTF-8 text file. Pass it with `--prompt-file` so the shell leaves
   quotes, line breaks and tags intact.
3. Run `scripts/video_api.py create-and-poll --download` (or `create`, then
   `poll` and `download`) by its path in this skill's folder. Treat
   `scripts/video_api.py` as read-only rather than editing it or writing ad-hoc
   API scripts.
4. For several unrelated clips, run one job per clip or use `create-batch` with
   a JSONL file. Jobs run one at a time on the server.
5. Inspect the saved MP4: verify its resolution, duration, audio stream and a
   few extracted frames (for example with `ffmpeg` and the image-viewing tool).
   Check subject continuity, motion, on-screen text spelling and sound timing.
   Change one thing per retry and keep the rest of the prompt steady.
6. Put clips that the project uses inside the project. Do not overwrite an
   existing file unless the user asked for that; choose a new name such as
   `intro-v2.mp4`. Report each saved path, job ID, actual size and duration,
   and the prompt that made it.

## Server behavior and limits

- Jobs run asynchronously (`queued`, `in_progress`, `completed` or `failed`)
  and output 24 fps H.264 video with 32 kHz stereo AAC audio in MP4.
- Duration is `--seconds` from 1 to 15 (default 5) or `--frames` from 1 to 362
  (aligned to the `17k + 5` frame grid, default 124). Pass one or the other,
  not both.
- Size is `auto` (default: `1344x768` for text only, or matched to the first
  frame or first reference image at a 768-pixel short edge) or `WIDTHxHEIGHT`
  with each side 1 to 2048 pixels and total area at most 2,359,296 pixels
  (2048 by 1152), rounded to multiples of 32.
- Denoising steps run from 1 to 100 (`--steps`; omitted uses the server's
  configured default).
- The model reads the first 4,096 tokens of a prompt and drops the rest. Put
  what matters first.
- Guidance scale is fixed at 1 and there is no negative prompt. Describe what
  appears and sounds positively.
- One request body may be at most 256 MiB. Keyframe options (`--first-frame`,
  `--last-frame`) cannot be combined with full-reference options
  (`--ref-image`, `--ref-video`, `--ref-video-audio`, `--ref-audio`,
  `--ref-image-size`).
