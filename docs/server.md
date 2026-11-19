# The server

`build/eremite-server` serves the models that its configuration file lists
(README.md, Serve). This document lists every setting, and explains how the
server handles requests, memory, decision models and Codex's web search.

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
| `search_timeout` | seconds that each fetch, search and health check of the web tool may take, and SearXNG's start | 30 |
| `searxng_idle` | seconds without a search before SearXNG stops | 600 |
| `search_session_idle` | seconds without a call before the web tool forgets a session's references | 7200 |
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
| `id` | the name that requests give in `model` | `deepseek-v4.1-flash`; a d1 model: its folder's name in lower case |
| `experts` | DeepSeek: a GGUF file, or the first shard of a set, whose routed experts replace the checkpoint's | the checkpoint's |
| `ctx` | the context in tokens, at most 1048576 and the model's own | the most; a d1 model: its checkpoint's |
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
to the model that generates text, a decision to the decision model.

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

## Codex's web search

Codex's standalone web tool sends its session `id` and `commands` to the
search endpoint. The server runs `search_query` through SearXNG. Each query
shows up to 5, 8 or 12 results, by Codex's `response_length`. With
`search_ranking_model` set, that decision model ranks the query's first 20
results from SearXNG, one decision each. A decision's input is `Query:` and
the query, a blank line, then `Passage:` and the result's title, with its
snippet on the next line. Its one predicate is "Is this passage relevant to
answering the query?", Liquid AI's prompt for reranking. The results with the
highest probability of yes show, ties in SearXNG's order. Without the
setting, or when the decision is refused, fails, cannot be queued or takes
longer than `search_timeout`, the query's first results show in SearXNG's
order, and the log says why. The `output` text lists each query's results
under the query: title, URL, citation reference and snippet. For each result
in `output`, `results` holds a `text_result` with the same `ref_id`, `url`,
`title` and `snippet`, in the same order. A successful query with no result
shown gives an empty array, and a call without a successful query gives
`null`. `encrypted_output` is always `null`.

Within a session, `open` accepts a search reference, a page reference or a
URL. Page views number each line and link. `find` says how many lines match.
It returns the whole page when the page fits the view's line and output
limits, and otherwise the matching lines with up to two lines on either side.
`click` follows a numbered link of the page. These commands do not use the
model.
