# eremite

eremite runs large language models locally on Apple silicon. It streams the
weights of mixture-of-experts models from the SSD into an expert cache that
grows and shrinks with free memory, schedules the requests of several models
on one GPU, and serves an OpenAI- and Codex-compatible API.

## Build

Requirements: macOS 27 with Xcode 27, and `pkgconf`, `yyjson` and
`jpeg-turbo` from Homebrew. The GPU code compiles its kernels at run time as
Metal 4.1, which needs macOS 27. JPEG images are decoded with jpeg-turbo, and
images in other formats with ImageIO.

```sh
make            # the library, the command-line program and the test programs
make test       # tests that need neither a GPU nor model files
make gpu-test   # tests that need a GPU but no model files
```

`make THREADS=pthread` builds the threads layer on POSIX threads instead of
Grand Central Dispatch. `make SANITIZE=1` builds with AddressSanitizer and
UndefinedBehaviorSanitizer. Each test ends after `TEST_SECONDS` (default 600),
so that a hang fails the run; `TEST_SECONDS=0` sets no limit.

The tests in `tests/model` need a checkpoint's files and the GPU, and run by
hand. Each one's usage is at the top of its source. For example,
`build/tests/model-tokenizer CHECKPOINT` checks DeepSeek V4.1 Flash's tokenizer
against the reference tokenizer's ids.

## Models

`scripts/download-models.sh FOLDER` downloads DeepSeek V4.1 Flash from Hugging
Face: its checkpoint (510 GB) into `FOLDER/deepseek-ai/DeepSeek-V4.1-Flash-1017`,
and the routed experts of the UD-IQ2_XXS GGUF quantization (165 GB) into
`FOLDER/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS`. Other
quantizations, such as `UD-IQ1_M`, can follow FOLDER. `--dry-run` lists the
files without downloading them. It needs Python 3.10 or later, and runs again
from where an interrupted download stopped.

## Run

`build/eremite -m CHECKPOINT -p PROMPT` generates a reply with DeepSeek V4.1
Flash, whose checkpoint is the directory CHECKPOINT. `--experts GGUF` reads
the routed experts from a GGUF file set instead, and `--memory GB` limits the
model memory, state memory included. `build/eremite` alone lists the options.

## Serve

`make server` builds `build/eremite-server`. It serves the OpenAI Chat
Completions and Responses APIs at `http://HOST:PORT/v1`, and Codex's model
catalog at `/v1/codex/models`.

Its settings are in a JSON file: `build/eremite-server --config FILE`, or
`~/.config/eremite/server.json` without `--config`. For example:

```json
{
  "port": 8000,
  "api_key_file": "/path/to/api-key",
  "models": [
    {
      "model": "/path/to/DeepSeek-V4.1-Flash-1017",
      "experts": "/path/to/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf"
    }
  ]
}
```

The API key file holds the key that clients send, 16 to 256 characters.
Without `api_key_file`, the server takes the key from the environment variable
`EREMITE_API_KEY`.

The file's root holds the server-wide settings:

| Key | Meaning | Default |
| --- | --- | --- |
| `port` | the port to serve on | required |
| `host` | the IPv4 address to serve on, or `localhost` | `127.0.0.1` |
| `api_key_file` | the file that holds the API key | `EREMITE_API_KEY` |
| `memory` | GiB of memory for all models together, their states included; 0: the memory that other programs leave, less `reserve` | 0 |
| `reserve` | GiB of memory kept available to other programs, 2 to 1024 | 10 |
| `state_cache` | GiB of states and snapshots kept in memory, for all models together | 8 |
| `max_cached_states` | the most states kept in memory, for all models together | 256 |
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
| `id` | the name that requests give in `model` | `deepseek-v4.1-flash` |
| `experts` | DeepSeek: a GGUF file, or the first shard of a set, whose routed experts replace the checkpoint's | the checkpoint's |
| `ctx` | the context in tokens, at most 1048576 and the model's own | the most |
| `draft` | the most proposals of the model's draft head per pass, 0 to 5; each pass runs the count that has given the most tokens per second | 0 |
| `lookup` | `off`, or the most context-copy proposals per pass, 1 to 15 | `off` |
| `ced` | DeepSeek: `off`, or an approximate prefill that replays this many decoder tokens, 128 to 1048576, plus the rewind margin (8) | `off` |
| `prefill_prune` | DeepSeek: a prefill skips the decoder batches that no later row needs; it changes no result, and needs `ced` off | `true` without `ced` |
| `packed_state_rows` | DeepSeek: a state keeps its compressed rows and index keys packed: the same values in about 28% of BF16's bytes | `false` |
| `idle_unload` | the model's own `idle_unload` | the server-wide one |

Each request sets its own sampling and thinking effort. A request that sets
none samples with temperature 1, top_p 1 and every token, and DeepSeek thinks
at high effort.

Codex's web tool searches through eremite's own SearXNG.
`scripts/setup-searxng.sh` sets it up in `~/.local/share/eremite/searxng`, the
server's default `searxng_dir`. It needs git and Python 3.10 or later, and
downloads about 100 MB.
