# The server

`build/eremite-server` serves the models that its configuration file lists
(README.md, Serve). This document lists every setting, and explains how the
server handles requests, memory, decision models and image models.
`docs/codex.md` describes Codex configuration, the Codex model catalog, Codex's
image tool and Codex's web search.

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
| `id` | the name that requests give in `model` | `deepseek-v4.1-flash`; a d1 model: its folder's name in lower case; MiniMax-H3-Image: `minimax-h3-image` |
| `experts` | DeepSeek: a GGUF file, or the first shard of a set, whose routed experts replace the checkpoint's | the checkpoint's |
| `text_encoder` | MiniMax-H3-Image: the safetensors file of the Qwen3-VL NVFP4 text and vision encoder | required for MiniMax-H3-Image |
| `dit` | MiniMax-H3-Image: the safetensors file of the pruned INT8-ConvRot DiT | required for MiniMax-H3-Image |
| `vae` | MiniMax-H3-Image: the safetensors file of the FP16 image VAE | required for MiniMax-H3-Image |
| `default_size` | MiniMax-H3-Image: the `WIDTHxHEIGHT` of a text-to-image request whose `size` is `auto` or omitted; it must fit the model as it is | `2048x2048`, the default generation canvas |
| `ctx` | the context in tokens, at most 1048576 and the model's own | the most; a d1 model: its checkpoint's; MiniMax-H3-Image: 4096 |
| `draft` | the most proposals of the model's draft head per pass, 0 to 5; each pass runs the count that has given the most tokens per second | 0 |
| `lookup` | `off`, or the most context-copy proposals per pass, 1 to 15 | 15 |
| `ced` | DeepSeek: `off`, or an approximate prefill that replays this many decoder tokens, 128 to 1048576, plus the rewind margin: 8, or the most draft or lookup proposals when more (15 with the default lookup) | `off` |
| `prefill_prune` | DeepSeek: a prefill skips the decoder batches that no later row needs; it changes no result, and needs `ced` off | `true` without `ced` |
| `packed_state_rows` | DeepSeek: a state keeps its compressed rows and index keys packed: the same values in about 28% of BF16's bytes | `false` |
| `memory` | GiB of the model's own memory: its weights, expert cache, workspace and states; 0: none of its own | 0 |
| `max_cached_states` | the most states of the model kept in memory | 256 |
| `idle_unload` | the model's own `idle_unload` | the server-wide one |

## Requests

A request names its model by its `id` in `model`. A request that names no
served model goes to the only model that answers its endpoint: a chat request
to the model that generates text, a decision to the decision model, and an
image request to the image model.

Each request sets its own sampling and thinking effort. A request that sets
none samples with temperature 1, top_p 1 and every token, and DeepSeek thinks
at high effort.

## Memory

A model's states take an adaptive part of its memory: all that its weights and
a full-speed prompt pass leave of its own `memory`, or else of the root
`memory` and the free memory above `reserve`. Its expert cache grows into what
the states leave. A new state takes room from the expert cache first, down to
the smallest cache that a prompt pass needs. Then the model's least recently
used idle state goes. It is saved on the SSD first when `state_disk` is on and
it holds `state_disk_tokens` tokens of its own.

In the terminal UI, 4 Models: States deletes the chosen session's state from
memory when `d` is pressed twice. The state is not saved on the SSD. A session
that a request runs in keeps its state.

Free memory means the free pages and the file cache above the part that macOS
keeps. That part is the smallest file cache read since the server started:
when memory runs short, macOS drops the file cache above it first. If macOS
compresses other memory instead while that file cache counts as free, the file
cache at that moment becomes the part that macOS keeps. So a growing file cache
does not make the models' memory shrink. The memory view shows the part that
macOS keeps as file cache, and the rest as free file cache.

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

During a pass, the image model yields between layers to any running
text-decoding calls so decoding latency is not degraded.
