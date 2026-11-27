# The server

`build/eremite-server` serves the models that its configuration file lists
(README.md, Serve). This document lists every setting, and explains how the
server handles requests, memory, decision models, image models, video models
and audio transcription models. `docs/codex.md` describes Codex configuration,
the Codex model catalog, Codex's image tool and Codex's web search.

## Settings

The file's root holds the server-wide settings:

| Key | Meaning | Default |
| --- | --- | --- |
| `port` | the port to serve on | required |
| `host` | the IPv4 address to serve on, or `localhost` | `127.0.0.1` |
| `api_key_file` | the file that holds the API key | `EREMITE_API_KEY` |
| `memory` | GiB of memory for all models together, their states included; 0: the memory that other programs leave, less `reserve` | 0 |
| `reserve` | GiB of memory kept available to other programs, 2 to 1024 | 10 |
| `state_disk` | GiB on the SSD for states evicted from memory or idle at quit; a prompt that starts like a saved state continues from it; 0: off | 32 |
| `state_dir` | the folder of the saved states | `~/.cache/eremite/states` |
| `state_disk_tokens` | a state is saved only when it holds at least this many tokens that no other kept state holds | 8192 |
| `snapshot_interval` | positions between the snapshots of a long prompt, for a model that resumes only from snapshots | 2048 |
| `idle_unload` | seconds without requests before a model's weights and experts unload; its states stay; 0: never | 600 |
| `max_waiting_requests` | the most requests that wait for one model; another gets HTTP 429 | 16 |
| `gpu_timeout` | seconds that a wait for the GPU may last before the server ends | 60 |
| `progress_log` | seconds between a decoding request's progress lines in the log; 0: none | 10 |
| `http_timeout` | seconds that a request's head, each part of its body and a stalled write may take | 60 |
| `keepalive` | seconds between the comment lines of a streamed response that waits for its first token | 15 |
| `searxng_dir` | the folder of eremite's SearXNG, for Codex's web tool | `~/.local/share/eremite/searxng` |
| `searxng_socket` | the Unix socket that SearXNG runs on | `eremite.sock` in `searxng_dir` |
| `search_dir` | the folder of the web tool's turn counters | `~/.local/share/eremite/web-search` when `HOME` is set, otherwise memory only |
| `video_dir` | the folder where completed video MP4 clips are stored | `~/.local/share/eremite/videos` |
| `search_timeout` | seconds that each fetch, search and health check of the web tool may take, and SearXNG's start | 30 |
| `searxng_idle` | seconds without a search before SearXNG stops | 600 |
| `search_session_idle` | seconds without a web call before the server discards the session's cached references | 7200 |
| `search_ranking_model` | the `id` of a served decision model that ranks each search query's results | none: SearXNG's order |
| `tui` | `true` shows the terminal UI on stderr's terminal instead of the log: the server's status, its requests, its memory and its log; `q` stops the server, and the log prints on exit | `false` |
| `tui_theme` | the terminal UI's colors: `auto` (by the terminal's background), `light` or `dark` | `auto` |
| `tui_background_answer_wait` | seconds that the terminal UI waits for the terminal to tell its background color | 1 |
| `tui_now_window` | seconds that the terminal UI's "now" figures average | 5 |
| `tui_recent_window` | seconds that its "recent" figures average | 60 |
| `tui_history_window` | seconds that its history charts span, one sample a column | 60 |
| `tui_expert_map_tokens` | the latest tokens of a decoding request whose experts its expert map shows | 12 |

Each entry under `models` holds one model's settings:

| Key | Meaning | Default |
| --- | --- | --- |
| `model` | the checkpoint's folder | required |
| `id` | the name that requests give in `model` | DeepSeek: `deepseek-v4.1-flash`; Inkling-2-Mini: `inkling-2-mini`; a d1 model: its folder's name in lower case; MiniMax-H3-Image: `minimax-h3-image`; MiniMax-H3 Video: `minimax-h3-video`; Nemotron-3.5-Transcribe: `nemotron-3.5-transcribe` |
| `experts` | DeepSeek: a GGUF file, or the first shard of a set, whose routed experts replace the checkpoint's | the checkpoint's |
| `text_encoder` | MiniMax-H3: the safetensors file of the Qwen3-VL NVFP4 text and vision encoder | required for MiniMax-H3 image and video models |
| `dit` | MiniMax-H3: the safetensors file of the pruned INT8-ConvRot DiT | required for MiniMax-H3 image and video models |
| `vae` | MiniMax-H3: the safetensors file of the FP16 image VAE or INT8-ConvRot 3D video VAE | required for MiniMax-H3 image and video models |
| `audio_vae` | MiniMax-H3 Video: the safetensors file of the FP32 stereo Audio VAE | required for MiniMax-H3 Video |
| `lora` | MiniMax-H3 Video: optional safetensors file of a BF16 LoRA adapter | none |
| `lora_strength` | MiniMax-H3 Video: positive scaling multiplier for `lora` | `1.0` |
| `steps` | MiniMax-H3 Video: default Euler denoising step count (1 to 100) | `50` |
| `shift_video` | MiniMax-H3 Video: default video flow-matching shift factor | `12.0` |
| `shift_audio` | MiniMax-H3 Video: default audio flow-matching shift factor | `3.0` |
| `variant` | MiniMax-H3 Video: checkpoint task variant, `"fl2va"` or `"ref2va"` | `"fl2va"` |
| `default_size` | MiniMax-H3-Image: the `WIDTHxHEIGHT` of a text-to-image request whose `size` is `auto` or omitted; it must fit the model as it is | `2048x2048`, the default generation canvas |
| `ctx` | the context in tokens, at most 1048576 and the model's own | the most; a d1 model: its checkpoint's; MiniMax-H3 image and video: 4096; Nemotron-3.5-Transcribe: 65536 |
| `draft` | the most proposals of the model's draft head per pass: 0 to 5 for DeepSeek, 0 to 7 for Inkling-2-Mini, 0 to 8 for Nemotron-3.5-Transcribe; each pass runs the count that has given the most tokens per second | 0 |
| `lookup` | `off`, or the most context-copy proposals per pass, 1 to 15 | 15 |
| `ced` | DeepSeek: `off`, or an approximate prefill that replays this many decoder tokens, 128 to 1048576, plus the rewind margin: 8, or the most draft or lookup proposals when more (15 with the default lookup) | `off` |
| `fp8_kv` | Inkling-2-Mini: approximate state storing global, sliding-window and draft-head KV caches in FP8 E4M3 with per-head scales instead of BF16, in about 51% of BF16's KV bytes | `false` |
| `memory` | GiB of the model's own memory: its weights, expert cache, workspace and states; 0: none of its own | 0 |
| `max_cached_states` | the most states of the model kept in memory | 256 |
| `idle_unload` | the model's own `idle_unload` | the server-wide one |

## Requests

A request names its model by its `id` in `model`. A request that names no
served model goes to the only model that answers its endpoint: a chat request
to the model that generates text, a decision to the decision model, an image
request to the image model, a video request to the video model, and an audio
request to the audio transcription model.

Each request sets its own sampling and thinking effort. A request that sets
none samples with temperature 1, top_p 1 and every token, and thinks at high
effort (75 for DeepSeek; 0.9 for Inkling-2-Mini).

Inkling-2-Mini checkpoints (`model_type` `inkling2_loop` or
`inkling2_loop_text`) serve `POST /v1/chat/completions`, `POST /v1/responses`,
and `GET /v1/codex/models`. Their entry takes `model`, `id`, `ctx` (up to
1,048,576), `draft` (0 to 7, using `mtp.safetensors` when present), `lookup`
(`off` or 1 to 15), `fp8_kv` (approximate FP8 E4M3 KV caches, about 51% of
BF16's KV bytes), `memory`, `max_cached_states` and `idle_unload`, and
refuses DeepSeek-only and MiniMax-H3 settings. Its dense backbone stays
resident in GPU memory (about 3.6 GiB, or about 4.3 GiB with its draft head),
while its 3,072 routed experts across 24 physical layers live in an SSD-backed
expert cache (preloaded into GPU memory — about 15.1 GiB — when memory fits,
or streamed on demand from SSD on smaller-memory Macs). Thinking effort takes
`"none"` (0), `"minimal"` (0.1), `"low"` (0.2), `"medium"` (0.7), `"high"`
(0.9), `"xhigh"` or `"max"` (0.99), or any number in `0.0` to `0.99`. Images
are accepted as base64 `data:` URLs in `user` and `tool` messages; audio parts
return HTTP 400. Inkling-2-Mini resumes cached conversations from saved
snapshots (including its sliding-window KV rings, short-convolution rings and
draft-head state).

## Memory

A model's states take an adaptive part of its memory: all that its weights and
a full-speed prompt pass leave of its own `memory`, or else of the root
`memory` and the free memory above `reserve`. Its expert cache grows into what
the states leave. A new state takes room from the expert cache first, down to
the smallest cache that a prompt pass needs. Then the model's least recently
used idle state goes. It is saved on the SSD first when `state_disk` is on and
it holds `state_disk_tokens` tokens of its own. Saved branches that share at
least `state_disk_tokens` prefix tokens with an existing saved state reference
the parent's prefix part files on disk and write only their divergent suffix,
counting each shared part file once toward `state_disk`.

In the terminal UI, 4 Models: States deletes the chosen session's state from
memory when `d` is pressed twice. The state is not saved on the SSD. A session
that a request runs in keeps its state.

Free memory means the free pages and the file cache above the pre-existing
floor that macOS already held. That floor is the smallest file cache read since
the server started, so pre-existing file cache stays respected until macOS
drains or purges it, while file cache growing above the floor counts as
available. On macOS unified memory, the server also holds released and idle
headroom above `reserve` (and within `memory`) in a releasable guard pool of
incompressible anonymous pages so OS file cache cannot grow into it. The guard
pool yields synchronously whenever eremite allocates memory, and sheds chunks
immediately when another program pushes available memory below `reserve`, when
macOS signals memory pressure, or when eremite's own pages compress. The memory
view shows the pre-existing floor as file cache, the rest as free file cache,
and active guard memory as guard pool.

DeepSeek V4.1 Flash always enables lossless packed state rows (exact FP4 +
scales in about 28% of BF16's bytes) and dependency-pruned prefills whenever
`ced` is off.

## GPU sharing

When one model runs alone, the engine runs it unthrottled at full GPU capacity
without slicing.

When two or more distinct models have active GPU requests at the same time, the
engine measures each model's GPU time and balances GPU turns evenly across them
(for example 50/50 across two models or 33/33/33 across three). Heavy
MiniMax-H3 DiT stages dynamically slice to the peer's measured turn budget, and
off-GPU stages such as MP4 and image encoding run on background threads so
other models keep full GPU capacity while encoding finishes. Once peer models
finish, the remaining model returns dynamically to full uncontended speed.

## Decision models

LiquidAI's d1 models, such as d1-2B, are decision models: they answer
questions about an input and generate no text. A d1 checkpoint serves
`POST /v1/decisions` only: the chat endpoints refuse it, and Codex's catalog
leaves it out. Its entry takes `model`, `id`, `ctx`, `memory` and
`idle_unload`, and refuses the other model settings. A request gives its
`input` and 1 to 400 `questions`. Each question has `instructions`, an
optional `name` and a `type`:

- `predicate`: a yes-or-no question.
- `choice`: 2 to 255 `choices`, each a `value`, a string or a boolean, with an
  optional `description`.
- `score`: 2 to 10 `levels`, each a `label` with an optional `description`.

The input is a string, or a list of messages and tool items. Each `input_image`
part of a user message gives an image, from its `image_url`, a base64 data URL;
the images come first, in the input's order. Of the rest, a string, or one user
message whose content is a string or one `input_text` or `input_json` part,
gives the model that text or JSON. One user message that held only images gives
the model the images alone. Another list gives the model the list as JSON,
without its images. Each of the model's passes reads the images, the rest of
the input and up to the checkpoint's `d1_set_size` questions (16 for d1-2B). An
image over 1,638,400 pixels shrinks to that size first, and an image takes one
token for each 32 x 32 pixels after LFM3-VL's processor resizes it. The answers
come back in the questions' order: a predicate's `probability` of yes; a
choice's likeliest `choice`, its `confidence` and every option's
`probabilities`; a score's probability-weighted `score`, its likeliest level's
`confidence` and every level's `probabilities`. `usage` counts the tokens that
the passes read, each image's tokens included.

## Image models

MiniMax-H3-Image generates images from text and edits one or more reference
images. Its `model` folder holds `model_index.json` (`_class_name`
`MiniMaxH3ImagePipeline`) and `tokenizer/tokenizer.json`, and its entry also
sets `text_encoder`, `dit` and `vae` to the quantized Comfy-Org safetensors
files (Qwen3-VL-32B NVFP4, pruned INT8-ConvRot DiT and FP16 VAE). Its entry
also takes `id`, `ctx` (at most 4096), `memory`, `idle_unload` and
`default_size`, and refuses text-generation settings.

An image model serves `POST /v1/images/generations` (JSON) and
`POST /v1/images/edits` (JSON or `multipart/form-data`, up to 128 MiB). The chat and
decision endpoints refuse it, and Codex's catalog leaves it out. Both endpoints
return `{"created": ..., "data": [{"b64_json": "..."}], "size": "WIDTHxHEIGHT",
"output_format": "png", "background": "opaque"}`, where `size` is the size of
the returned images, `output_format` is their format, and `background` is
`transparent` or `opaque`:

- `prompt`: required string, passed verbatim and truncated to 4096 tokens.
- `size`: `auto` (the default) or `WIDTHxHEIGHT` (each 1 to 8192). The
  returned image has exactly the size given. With `auto`, a text-to-image
  request gets the model's `default_size`, and an edit gets the size of its
  first image. A first image larger than the model generates is scaled down
  to the largest size of its shape that fits; an image whose shape fits no
  canvas gets HTTP 400. Total output pixels across all images in one request
  (`width * height * n`) may be at most 33,554,432 (32 Mi pixels).
- `image` (or `image[]`): `/v1/images/edits` with `multipart/form-data`
  only; 1 to 16 image files in upload order, defining `<Picture 1>`,
  `<Picture 2>`, and so on in `prompt`.
- `images`: `/v1/images/edits` with JSON only; 1 to 16 objects, each with an
  `image_url` that is a base64 data URL, in the same order. The server
  fetches no URLs and has no Files API, so `file_id` gets HTTP 400. An image
  that does not decode gets HTTP 400, and so does a PNG file that is cut
  short or damaged: one whose chunks run past its end, fail their CRC, or
  end before IEND. A JPEG file whose data ends before the image does too.
- `mask`: `/v1/images/edits` only, optional: an image with an alpha channel,
  the size of the first image, as a file part or in JSON as an object with an
  `image_url`. Its transparent pixels mark what the edit changes; its opaque
  pixels keep the first image's, and partly transparent pixels mix the two.
  MiniMax-H3-Image takes no mask itself: it edits the whole image, and the
  server then keeps the first image's pixels under the mask. With a mask,
  `size` must be `auto` or the first image's size.
- `n`: optional image count, 1 to 16 (default 1); image `i` uses `seed + i`.
- `seed`: optional nonnegative integer (random when omitted).
- `generator_device`: accepted and ignored (initial noise always uses the
  deterministic CPU RNG).
- `response_format`: omit or `"b64_json"`.
- `output_format`: `"png"` (default), `"jpeg"` (or `"jpg"`) or `"webp"`.
  Other formats get HTTP 400, and so does `"gif"`: the server makes no GIFs.
- `output_compression`: optional JPEG or WebP quality, 0 to 100 (default
  100, as in OpenAI's API). PNG ignores it.
- `background`: `"auto"` (default), `"opaque"` or `"transparent"`.
  MiniMax-H3-Image draws no alpha, so `auto` gives an opaque background. For
  `transparent`, eremite puts a sentence before the prompt that asks for a
  flat pure green (#00FF00) background; the token limit cuts a long prompt's
  end, so the sentence stays. It then makes that background clear. The key
  is the median color of the image's border, and a pixel's alpha comes from
  how far its green leads its red and blue. A pixel is clear when its lead is
  within the background's noise of the key's: the noise is measured on the
  border, and widened to how far it reaches among all the image's pixels. A
  pixel with no lead stays opaque. With a mask, the key comes from the edit's own border, and the
  mask then keeps the first image's pixels, opaque. This is an
  approximation: the edges of hair, glass, smoke and soft shadows suffer, and
  green parts of the subject turn clear. An image whose border has no green
  key fails with HTTP 503. A transparent background needs `output_format`
  `"png"` or `"webp"`; JPEG gets HTTP 400.
- `stream`: optional boolean (default `false`); a form field takes `true` or
  `false`. With `true`, the response is a stream of server-sent events. Each
  image's partial images come as `image_generation.partial_image` events
  (`image_edit.partial_image` for edits), and then the image comes as an
  `image_generation.completed` event (`image_edit.completed`). Each event's
  data is a JSON object with `type`, `b64_json`, `created_at`, `size`,
  `output_format` and `background`, and a partial image adds
  `partial_image_index`, which starts at 0 for each image. The events carry
  no `usage` or `quality`. A request refused before its first event gets a
  plain HTTP 400 or 503; an error after that comes as an `error` event whose
  data holds an `error` object, and it ends the stream.
- `partial_images`: optional, 0 to 3 (default 0), used only with `stream`.
  A partial image is the clean image that a denoising step predicts,
  decoded and encoded like the final image. The partial images spread evenly
  over the 6 steps: after step 3 for one, after steps 2 and 4 for two, and
  after steps 1, 3 and 4 for three. Each costs one more VAE decode, and
  during sampling the request also holds the VAE decoder and its workspace.
  With a transparent background, a partial image whose border has no green
  key yet is skipped. Partial images leave the final image as it is.
- `min_generation_area`: optional integer from 1024 to 16777216 (default
  4194304, 2048 x 2048). When `size` has fewer pixels, the model generates on a
  larger 32-pixel-aligned canvas with the same aspect ratio and resizes back
  with Lanczos-3; pass `1024` to disable automatic upscaling.
- `num_inference_steps`: omit or `6` (other values return HTTP 400).
- `guidance_scale`: omit or `1` (other values return HTTP 400).
- `negative_prompt`: unsupported; nonempty values return HTTP 400.
- `frame_count` and `frame_duration_ms`: GIF options, unsupported; any value
  returns HTTP 400.

Full residence is the primary mode: when enough memory fits, the text encoder
(13.5 GiB), DiT (19.5 GiB) and VAE decoder (0.8 GiB) stay resident together
(33.8 GiB), and the vision tower (1.1 GiB) and VAE encoder (0.1 GiB) load on
demand at the first edit (35.0 GiB total). When available memory cannot hold
all three core groups at once, the model runs staged: each stage loads its own
weights in turn. The model then keeps its largest stage's memory reserved: 21.8
GiB for the default canvas and a full prompt.

Before its text encoder runs, each request reserves the memory that its
canvas, prompt, reference images and partial images need. A request that
needs more than the model may ever use gets HTTP 400: more than the model's
`memory`, the server's `memory`, or the machine's RAM less `reserve`. A
request that does not fit now gets HTTP 503.

When an image model runs alone, it uses full GPU throughput without slicing.
When another model runs alongside it, the engine balances GPU time evenly
between active models, slicing heavy DiT stages to match the peer's measured
turn duration and running final image encoding on a background thread.

## Video models

MiniMax-H3 Video generates synchronized 24 fps H.264 video and 32 kHz stereo
AAC audio muxed into MP4 using macOS AVFoundation and VideoToolbox. Its `model`
folder holds `model_index.json` (`_class_name` `MiniMaxH3ModularPipeline` or
`MiniMaxH3Pipeline`) and `processor/tokenizer.json`. Its entry sets
`text_encoder`, `dit`, `vae` and `audio_vae` to the quantized Comfy-Org
safetensors files, with optional `lora`, `lora_strength`, `steps`,
`shift_video`, `shift_audio` and `variant` (`"fl2va"` or `"ref2va"`).

A video model serves the OpenAI Videos API (and refuses chat, decision and
image endpoints; Codex's catalog leaves it out):

- `POST /v1/videos` (JSON or `multipart/form-data`, up to 256 MiB): queues a
  video generation job and returns the job object immediately (`id`, `status`:
  `"queued"`, `progress`: `0`, `size`, `seconds`, `created_at`).
- `GET /v1/videos`: lists stored video jobs (`limit` 1 to 100, default 20;
  `after`; `order` `"desc"` or `"asc"`).
- `GET /v1/videos/{id}`: returns the job's current `status` (`"queued"`,
  `"in_progress"`, `"completed"` or `"failed"`), live `progress` (0 to 100),
  actual `size`, `seconds`, `frames`, `fps`, `completed_at` and `error`.
- `GET /v1/videos/{id}/content`: downloads the finished MP4 (`video/mp4`) once
  completed. Returns HTTP 409 before completion and HTTP 400 for unsupported
  `variant` values other than `"video"`.
- `DELETE /v1/videos/{id}`: cancels the job cooperatively if queued or running,
  deletes its metadata and removes its MP4 file from `video_dir`.

Request fields accepted by `POST /v1/videos`:

- `prompt`: required string, passed verbatim and truncated to 4096 tokens. In
  REF2VA requests, reference tags `<Picture 1>`, `<Video 1>`, `<Audio 1>`, and
  so on address uploaded conditioning media in upload order.
- `size`: `auto` (default) or `WIDTHxHEIGHT` (each 1 to 2048, area at most
  2048 x 1152 = 2,359,296 pixels; rounded to 32-pixel multiples). With `auto`,
  a request adapts its aspect ratio to `first_frame` or the first reference
  image at 768p short edge, or uses `1344x768` for text-to-video.
- `seconds`: optional duration in whole seconds, 1 to 15 (string or integer,
  default 5). Mutually exclusive with `frames`.
- `frames`: optional explicit frame count, 1 to 362 (aligned to the Video VAE
  17k + 5 grid, default 124). Mutually exclusive with `seconds`.
- `steps` (or `num_inference_steps`): optional Euler denoising steps, 1 to 100
  (default is the model's `steps`, or 50).
- `shift_video` and `shift_audio`: optional positive flow-matching shift
  overrides.
- `seed`: optional nonnegative integer (random when omitted).
- `guidance_scale`: omit or `1` (released MiniMax-H3 video checkpoints are
  CFG-distilled). `negative_prompt` is unsupported.
- `input_reference`: optional OpenAI standard reference file (base64/data URL
  in JSON or file part in multipart); maps to `first_frame` on FL2VA models and
  prepends to `ref_images` on REF2VA models.
- `first_frame` and `last_frame`: optional FL2VA keyframe images (refused on
  REF2VA models).
- `ref_images` (up to 9), `ref_videos` (up to 3), `ref_video_audios` (optional
  explicit soundtracks paired with `ref_videos`) and `ref_audios` (up to 3),
  total references at most 12, plus `ref_image_size` (`"auto"` or `"max"` for
  2048-pixel short edge): REF2VA only (refused on FL2VA models).

In full residence, the text encoder, DiT, optional LoRA, Video VAE decoder and
Audio VAE decoder stay resident together, and the vision tower and Video/Audio
VAE encoders load on demand. When available memory cannot hold all core groups
at once, the model runs staged and pins only the active stage's weights in
turn. Each request reserves its planned memory before Qwen3-VL runs, uses full
GPU throughput when running alone, balances GPU time evenly with other active
models by slicing heavy DiT stages to the peer's measured turn duration and
offloading MP4 encoding to a background thread, and reports prompt encoding as
prefill and denoising/VAE/MP4 work in dedicated video counters.

## Audio transcription models

NVIDIA Nemotron-3.5-Transcribe (`model_type` `nemo_speechlm`) transcribes
speech and translates speech into English. Its `model` folder holds
`config.json`, `tokenizer.json`, `model.safetensors.index.json`, its
safetensors weight shards, `perception.safetensors`, and optionally
`model-mtp-fp8.safetensors` for speculative drafting and
`model-ctc-bf16.safetensors` for CTC word-level timestamps. Its entry takes `model`,
`id`, `ctx` (default 65,536, fitting up to 1-hour recordings plus transcripts),
`draft` (0 to 8, default 0), `lookup` (`off` or 1 to 15, default `off`),
`memory`, `max_cached_states` and `idle_unload`, and refuses DeepSeek-only and
MiniMax-H3 settings.

An audio transcription model serves OpenAI's Audio API (`multipart/form-data`,
up to 256 MiB per upload) and refuses chat, decision, image and video
endpoints; Codex's catalog leaves it out:

- `POST /v1/audio/transcriptions`: transcribes an uploaded recording in its
  spoken language or a requested language.
- `POST /v1/audio/translations`: translates an uploaded recording into English.

Request form fields accepted by the audio endpoints:

- `file`: required non-empty audio file (WAV, FLAC, MP3, M4A, AAC, OGG, MP4 or
  any format macOS AVFoundation decodes; resampled to 16 kHz mono, up to 1 hour
  long).
- `model`: optional model id. When omitted or unrecognized and only one audio
  model is served (for example clients sending `whisper-1`), the request routes
  to that audio model.
- `language`: `/v1/audio/transcriptions` only; optional ISO-639-1 or BCP-47
  language code (for example `en`, `zh`, `ja`, `de`) or English language name.
  Refused on `/v1/audio/translations`.
- `prompt`: optional context terms or domain vocabulary passed as biasing hints
  in the transcription or translation prompt.
- `response_format`: `"json"` (default: `{"text": "...", "usage": ...}`),
  `"text"` (`text/plain; charset=utf-8`), `"verbose_json"` (`task`, `duration`,
  `text`, `usage`, `language` when known, plus `segments` and/or `words` when
  aligned), `"srt"` (SubRip captions, `text/plain; charset=utf-8`), or `"vtt"`
  (WebVTT captions, `text/vtt; charset=utf-8`). `"srt"` and `"vtt"` require
  `model-ctc-bf16.safetensors` in the checkpoint.
- `timestamp_granularities[]` (or `timestamp_granularities`):
  `/v1/audio/transcriptions` only; requires `response_format` `"verbose_json"`
  and `model-ctc-bf16.safetensors`. Accepts `"word"`, `"segment"`, or both
  (default `["segment"]` for `"verbose_json"` when `model-ctc-bf16.safetensors`
  is present).
- `temperature`: optional nonnegative finite number (default `0`, greedy).
- `seed`: optional nonnegative integer random seed.
- `stream`: optional `"true"` or `"false"` (default `false`); requires
  `response_format` `"json"`. Streams `transcript.text.delta` and
  `transcript.text.done` server-sent events with keepalive comments while the
  audio encoder and prefill run.

Nemotron-3.5-Transcribe keeps its dense hybrid Mamba-2 and attention backbone
resident in GPU memory (~3.3 GiB) and loads its 815 MiB Parallel Expert
Perception audio encoder and 96.6 MiB BF16 CTC timestamp head on demand. Its
2,944 NVFP4 routed experts across 23 MoE layers live in the shared SSD-backed
expert cache (~15.7 GiB when all experts stay resident). Setting `draft` to `1`
through `8` loads the checkpoint's repeated `*E` FP8 Multi-Token Prediction head
(`model-mtp-fp8.safetensors`, ~1.25 GiB) and accelerates decoding without
changing outputs; `draft: 7` gives the fastest measured decode speed in both
full-resident and SSD-streaming modes because verification stays within the
8-row multi-vector kernels. Setting `memory` (for example `5` GiB without
`draft`, or `6` GiB with `draft: 7`)
streams routed experts from SSD into a bounded cache so transcription runs
alongside a large language model such as DeepSeek without holding all experts
in RAM.
