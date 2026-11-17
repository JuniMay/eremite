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
catalog at `/v1/codex/models`. `--tui` shows its status in the terminal.

Its settings are in a JSON file: `--config FILE`, or else
`~/.config/eremite/server.json`. For example:

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

The API key file holds the key that clients send, 16 to 256 characters. Each
option that `build/eremite-server --help` lists is a key of the same name, with
`_` for `-`. A model's own options go in its entry under `models`.

Codex's web tool searches through eremite's own SearXNG.
`scripts/setup-searxng.sh` sets it up in `~/.local/share/eremite/searxng`, the
server's default `searxng_dir`. It needs git and Python 3.10 or later, and
downloads about 100 MB.
