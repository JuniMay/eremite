# Codex

`build/eremite-server` serves the endpoints that Codex uses: OpenAI's Responses
API (`POST /v1/responses`), Codex's model catalog (`GET /v1/codex/models`),
OpenAI's Images API (`POST /v1/images/generations` and `POST /v1/images/edits`),
and Codex's standalone web tool (`POST /v1/alpha/search`). Each route also
works without the `/v1` prefix.

## Setup

Start `build/eremite-server` with a configuration file that lists your models
(`README.md`, Serve; `docs/server.md`). Set the environment variable
`EREMITE_API_KEY` in the shell that starts Codex to the key from `api_key_file`
(or the server's `EREMITE_API_KEY`).

Configure eremite as a model provider in `~/.codex/config.toml`:

```toml
model_provider = "eremite"
model = "deepseek-v4.1-flash"
web_search = "live"

[model_providers.eremite]
name = "eremite"
base_url = "http://127.0.0.1:8000/v1"
env_key = "EREMITE_API_KEY"
wire_api = "responses"
model_catalog_url = "http://127.0.0.1:8000/v1/codex/models"
supports_standalone_web_search = true
http_headers = { "x-openai-actor-authorization" = "eremite" }

[features]
image_generation = true
```

Adjust `model` and `base_url` to match the served model's `id` and the server's
`host` and `port`:

- `model_catalog_url` tells Codex to read the model's context window, reasoning
  levels, instructions, and `apply_patch` support from eremite.
- `supports_standalone_web_search` and `web_search = "live"` route Codex's
  `web.run` tool to `POST /v1/alpha/search`. Set `web_search = "disabled"` and
  omit `supports_standalone_web_search` when you do not use web search.
- `http_headers` and `[features] image_generation = true` turn on Codex's
  `image_gen` tool. Omit them when the server serves no image model.

## Responses API and tools

DeepSeek and Inkling-2-Mini models answer `POST /v1/responses`:

- Send the full conversation in `input`. The server keeps no stored responses,
  so `previous_response_id`, `conversation`, and `item_reference` return HTTP
  400. Prompt prefix reuse happens automatically through eremite's state cache
  (`docs/server.md`, Memory).
- `reasoning.effort` sets the model's thinking effort:
  - DeepSeek: `"none"` turns thinking off; `"minimal"` and `"low"` give effort
    50; `"medium"`, `"high"` and `"xhigh"` give 75; `"ultra"` and `"max"` give
    100; and a number from 1 to 100 sets that effort directly. Without an
    effort, the model thinks at high effort (75).
  - Inkling-2-Mini: `"none"` gives 0 (thinking off), `"minimal"` gives 0.1,
    `"low"` gives 0.2, `"medium"` gives 0.7, `"high"` gives 0.9, `"xhigh"` and
    `"max"` give 0.99, and a number from 0.0 to 0.99 sets that effort directly.
    Without an effort, the model thinks at high effort (0.9).
  - Both families return reasoning as `reasoning_text` rather than summaries.
- Declared tools of types `function`, `namespace`, and `custom` stay in the
  request (plus `tool_search` on DeepSeek). The server exposes Codex's freeform
  `apply_patch` custom tool as a function the model knows and returns its calls
  as `custom_tool_call` items. Unsupported hosted tool types, such as
  `web_search`, are dropped; use Codex's standalone web tool instead.

## Model catalog

`GET /v1/codex/models` returns Codex's model catalog (`{"models": [...]}`) for
the served DeepSeek and Inkling-2-Mini models. Decision models (d1), image
models (MiniMax-H3-Image) and video models (MiniMax-H3 Video) are left out.

Without catalog metadata, Codex assumes a 272,000-token context window, caps
`model_context_window` there, and offers no `apply_patch` tool. Each entry in
eremite's catalog carries:

- `slug` (the model's `id`), `display_name`, and `context_window` /
  `max_context_window` set to the model's served `ctx`.
- `supported_reasoning_levels` (`none`, `low`, `high`, `max` for DeepSeek;
  `none`, `minimal`, `low`, `medium`, `high`, `xhigh` for Inkling-2-Mini) and
  `default_reasoning_level` from the model's default effort.
- `apply_patch_tool_type` set to `"freeform"`, `web_search_tool_type` set to
  `"text"`, `input_modalities` set to `["text", "image"]`, and
  `supports_parallel_tool_calls` set to `true`.
- Codex's default instructions in `model_messages.instructions_template`, read
  from `server/assets/codex/prompt.md`. Codex rejects a catalog entry without
  instructions.

After a Codex update, update `server/assets/codex/prompt.md` as described in
`server/assets/codex/README.md` and rebuild the server.

## Image tool (`image_gen`)

Codex's image tool, `image_gen`, calls the provider's
`POST /v1/images/generations` and `POST /v1/images/edits` when the provider is
OpenAI, signs in with OpenAI, or sends an `x-openai-actor-authorization`
header. Setting that header in `[model_providers.eremite]` and enabling
`image_generation = true` under `[features]` turns the tool on for eremite.

The server ignores the header's value. When Codex calls `image_gen`:

- It always sends `size` and `quality` as `auto`: a new image gets the image
  model's `default_size` (`2048x2048` by default), and an edit gets the size of
  its first image.
- It sends `background` as `transparent` or `opaque`, sends edits as JSON with
  base64 data URLs, and sends no mask.
- It names the model `gpt-image-2`, which no served model uses, so eremite
  routes the request to the served image model when the server serves only one.
- Its `gif` option sends `output_format: "gif"`, which gets HTTP 400: eremite
  makes no GIFs.

Install eremite's `image-generation` skill into Codex's skills folder so the
agent writes prose prompts that fit MiniMax-H3-Image and knows when to call the
image API directly for exact sizes, multiple variants, JPEG/WebP compression,
seeds, masks, or JSONL batches:

```sh
cp -R skills/image-generation "${CODEX_HOME:-$HOME/.codex}/skills/"
```

See `skills/README.md` for `IMAGE_API_URL` and `IMAGE_API_AUTH_FILE` used by
`skills/image-generation/scripts/image_api.py`, and `docs/server.md` (Image
models) for every image endpoint parameter and limit.

## Web search (`web.run`)

Codex's standalone web tool (`web.run`) sends each call to
`POST /v1/alpha/search` with its session `id` and the model's `commands`.

### Setup and server settings

Before searching, install eremite's local SearXNG instance into
`~/.local/share/eremite/searxng` (the default `searxng_dir`):

```sh
scripts/setup-searxng.sh
```

The script requires `git` and Python 3.10 or later and downloads about 100 MB.
The server starts SearXNG on demand over a Unix socket (`searxng_socket`, so it
uses no TCP port) and stops it after `searxng_idle` seconds without a search.

The root settings in `server.json` that control web search are (`docs/server.md`,
Settings):

| Key | Purpose | Default |
| --- | --- | --- |
| `searxng_dir` | folder of eremite's SearXNG installation | `~/.local/share/eremite/searxng` |
| `searxng_socket` | Unix socket path that SearXNG listens on | `eremite.sock` in `searxng_dir` |
| `search_dir` | folder storing per-session turn counters (`turns.json`) | `~/.local/share/eremite/web-search` when `HOME` is set, else memory only |
| `search_timeout` | seconds allowed for each fetch, search, health check, and SearXNG start | `30` |
| `searxng_idle` | seconds without a search before SearXNG stops | `600` |
| `search_session_idle` | seconds without a web call before dropping a session's cached references | `7200` |
| `search_ranking_model` | `id` of a served decision model (such as `d1-2b`) that reranks search results | none (SearXNG's order) |

### Commands and access rules

The server runs these `web.run` commands:

- `search_query`: searches through SearXNG, supporting query text `q`, optional
  `recency` (days, rounded up to a day, week, month, or year), and `domains`.
  Each query shows up to 5, 8, or 12 results according to Codex's
  `response_length` (`short`, `medium`, or `long`).
- `open`: opens a search reference, page reference, or URL (starting at optional
  `lineno`). HTML pages become numbered lines with numbered links.
- `click`: follows a numbered link `id` on a referenced page or URL.
- `find`: searches a page for a case-insensitive text `pattern`, reports the
  number of matching lines, and returns either the whole page (when within the
  view limits) or each match with up to 2 lines of context on either side.
- `time`: answers the current time at a given `utc_offset`.

Commands that the server does not implement (`image_query`,
`literature_search_query`, `screenshot`, `finance`, `weather`, and `sports`)
answer that they are not available.

Live web access follows Codex's `web_search` setting:

- With live or indexed access (`web_search = "live"` or `"indexed"`), the server
  runs SearXNG queries and fetches pages with `/usr/bin/curl`. Fetches follow at
  most 5 redirects, cap each page at 5 MiB, and refuse loopback, private,
  link-local, and other non-public addresses at every redirect. When Codex sends
  `filters.allowed_domains`, searches and page fetches stay within those
  domains.
- With `web_search = "cached"`, commands that require live web access refuse
  without contacting SearXNG or remote hosts; already cached page references
  remain readable.

### Reranking search results

With `search_ranking_model` set to the `id` of a served decision model (for
example `d1-2b`), that model ranks each query's first 20 results from SearXNG,
one decision each. A decision's input is `Query:` and the query, a blank line,
then `Passage:` and the result's title, with its snippet on the next line. Its
one predicate is "Is this passage relevant to answering the query?", Liquid AI's
prompt for reranking. The results with the highest probability of yes show, ties
in SearXNG's order. Without the setting, or when the decision is refused, fails,
cannot be queued, or takes longer than `search_timeout`, the query's first
results show in SearXNG's order and the log says why.

The `output` text lists each query's results under the query: title, URL,
citation reference, and snippet. For each result in `output`, `results` holds a
`text_result` with the same `ref_id`, `url`, `title`, and `snippet`, in the same
order. A successful query with no result shown gives an empty array, and a call
without a successful query gives `null`. `encrypted_output` is always `null`.

### Sessions, turns, and forks

Each call takes the next turn number `N` of its session. Its references carry
`N`: a search result is `turnNsearchM`, and an opened page is `turnNviewM`. No
two calls of a session share a reference.

A session `id` that the server has not seen starts at turn 0, unless it is a
fork. For a new `id`, the server reads the string `forked_from_thread_id` from
the JSON object in the `x-codex-turn-metadata` header. The fork then starts at
its parent's next turn. When the server does not know the parent, the fork
starts after the highest turn that any session has used. A fork whose header
names no readable parent starts at 0 and can reuse references from the history
it copied.

The server keeps each session's next turn in `turns.json` in `search_dir`. The
file holds one entry per session and keeps it forever. Each call saves the file
before issuing references: the server syncs a new copy, replaces the old file,
and syncs the folder. Only one server at a time can use the folder. When the
file cannot be saved, the call fails with HTTP 500 and its turn number stays
unused. Without `search_dir` and without a nonempty `HOME`, the counters stay in
memory, so a restart can give a resumed chat references that it already holds.

To move `search_dir`, stop the server first and copy `turns.json` along. A lost
or older copy of the file can also give a chat references that it already holds.

The pages and search results behind the references stay in memory per session
(up to 400 references per session, dropping the oldest first). The server drops
a session's references after `search_session_idle` without a call and on
restart. A call that names a dropped reference tells the model to open its URL
instead.
