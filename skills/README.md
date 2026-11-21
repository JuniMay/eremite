# Skills

`skills/` holds agent skills for eremite. Each skill is written without the
name of any agent harness, vendor or model, and follows the standard skill
layout: a short `SKILL.md`, `references/` and `scripts/` where a task needs
them, and `agents/openai.yaml` for interface metadata (display name, short
description and default prompt).

## `image-generation`

`skills/image-generation` guides an agent through generating and editing raster
images against eremite's image models (such as MiniMax-H3-Image):

- `SKILL.md`: when to use raster generation instead of vector or code, the
  workflow for generating, inspecting and saving assets, what the server
  controls, and prose prompt rules (no labeled lines or keyword lists, quoted
  text, no GIFs).
- `agents/openai.yaml`: display name, short description and default prompt.
- `references/editing.md`: multi-reference `<Picture N>` editing rules.
- `references/transparency.md`: prompt guidance for green-key transparent
  cutouts.
- `references/script.md` and `scripts/image_api.py`: direct API CLI aligned
  with the `generate`, `edit`, and `generate-batch` workflow for exact sizes
  (`--size WIDTHxHEIGHT`), multiple variants (`--n`), formats (`png`, `jpeg`,
  `webp`), compression, seeds, canvas area (`--min-generation-area`), masks,
  and JSONL batches.

## `video-generation`

`skills/video-generation` guides an agent through generating and editing
synchronized audio-video MP4 clips against eremite's video models (such as
MiniMax-H3 Video) through the `/v1/videos` API. It is a standalone skill, not a
built-in Codex compatibility tool:

- `SKILL.md`: when to use video generation, choosing base mode or
  full-reference mode, inspecting input and output clips, server limits (24 fps
  H.264 and 32 kHz stereo AAC, 1 to 15 seconds or 1 to 362 frames, up to
  2048x1152 area, 4,096 prompt tokens, no negative prompt, 256 MiB upload
  limit) and workflow rules.
- `agents/openai.yaml`: display name, short description and default prompt.
- `references/prompting.md`: MiniMax-H3 Video base-mode prompt format (T2VA,
  I2VA, FL2VA and L2VA alignment lines, `integrated_multimodal_description`,
  `overall_soundscape`, `non_diegetic_music`, `[Shot N]` cuts, three-part
  camera motion, `(S1)` speakers, `<d>[Language] ...</d>`, voiceover,
  `<scenetrans>`, `<cutoff>` and quoted on-screen text).
- `references/references-mode.md`: MiniMax-H3 Video full-reference prompt
  format (`subject_definitions`, `summary` task types, `retention_analysis`
  markers, `detailed_description`, `<Subject N>` / `<Picture N>` / `<Video N>`
  / `<Audio N>` upload label numbering and soundtrack attribution).
- `references/video-api.md` and `scripts/video_api.py`: direct API CLI with
  `create`, `create-and-poll`, `status`, `poll`, `download`, `list`, `delete`
  and `create-batch` subcommands, supporting JSON and `multipart/form-data`
  uploads for keyframes and full-reference media.

## Install

Copy the skills you want into your agent's skills folder. For Codex:

```sh
cp -R skills/image-generation skills/video-generation "${CODEX_HOME:-$HOME/.codex}/skills/"
```

To let the agent run `scripts/image_api.py` directly, set two environment
variables:

- `IMAGE_API_URL`: the server's base URL ending in `/v1` (for example
  `http://127.0.0.1:8008/v1`).
- `IMAGE_API_AUTH_FILE`: path to the file holding the server's API key.

To let the agent run `scripts/video_api.py` directly, set two environment
variables:

- `VIDEO_API_URL`: the server's base URL ending in `/v1` (for example
  `http://127.0.0.1:8008/v1`).
- `VIDEO_API_AUTH_FILE`: path to the file holding the server's API key.

See `docs/codex.md` for the Codex `config.toml` lines that enable the image
tool with eremite.
