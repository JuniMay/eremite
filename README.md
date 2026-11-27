# eremite

eremite runs large language models locally on Apple silicon. It streams the
weights of mixture-of-experts models from the SSD into an expert cache that
grows and shrinks with free memory, schedules the requests of several models
on one GPU, and serves an OpenAI- and Codex-compatible API.

## Performance

DeepSeek V4.1 Flash on an Apple M5 Max with 128 GiB, with 64 GiB of its own
memory, the default settings and one request at a time. Prefill reads a prompt
of 8,921 tokens. Decoding speed depends on the text, so it is a range.

| Experts | Prefill | Decoding |
|---|---|---|
| UD-IQ1_M | 688 token/s | 23~27 token/s |
| UD-IQ2_XXS | 664 token/s | 21~25 token/s |
| MXFP4 | 448 token/s | 10~15 token/s |

`docs/performance/deepseek-v4.1-flash.md` gives the setup and the measured
runs.

Inkling-2-Mini on the same Apple M5 Max with 128 GiB, all weights resident
(~18.5 to 20.3 GiB footprint), `lookup` 15, and one request at a time. Prefill
reads a prompt of 8,784 tokens of source code.

| Mode | Prefill | Decoding |
|---|---|---|
| Plain (`draft` 0) | 1,146 token/s | 46.4~61.5 token/s |
| MTP draft head (`draft` 7) | 1,132 token/s | 48.8~60.9 token/s |

`docs/performance/inkling-2-mini.md` gives the setup and the measured runs.

## Build

Requirements: macOS 27 with Xcode 27, and `pkgconf`, `yyjson`, `jpeg-turbo`
and `webp` from Homebrew. The GPU code compiles its kernels at run time as
Metal 4.1, which needs macOS 27. JPEG images are decoded with jpeg-turbo, and
images in other formats with ImageIO. WebP images are encoded with libwebp,
because ImageIO does not write them.

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

`scripts/download-models.sh FOLDER` downloads the models from Hugging Face:
DeepSeek V4.1 Flash's checkpoint (510 GB) into
`FOLDER/deepseek-ai/DeepSeek-V4.1-Flash-1017`, the routed experts of its
UD-IQ2_XXS GGUF quantization (165 GB) into
`FOLDER/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS`, Thinking Machines
Lab's Inkling-2-Mini checkpoint (20.2 GB, including `mtp.safetensors`) into
`FOLDER/thinkingmachines/Inkling-2-Mini`, LiquidAI's
d1-2B decision model (3.9 GB) into `FOLDER/LiquidAI/d1-2B`, and
MiniMax-H3's image and video metadata and tokenizers into
`FOLDER/MiniMaxAI/MiniMax-H3-Image` and `FOLDER/MiniMaxAI/MiniMax-H3` with
Comfy-Org's quantized checkpoints into `FOLDER/Comfy-Org/MiniMax-H3` (15.7 GB
text and vision encoder, 21.0 GB FL2VA and REF2VA video DiTs, 2.8 GB Video VAE,
and 0.6 GB Audio VAE) and
`FOLDER/Comfy-Org/MiniMax-H3-Image` (21.0 GB image DiT and 1.2 GB image VAE).
Other quantizations, such as `UD-IQ1_M`, can follow FOLDER. `--dry-run` lists
the files without downloading them. It needs Python 3.10 or later, and runs
again from where an interrupted download stopped.

## Run

`build/eremite -m CHECKPOINT -p PROMPT` generates a reply with DeepSeek V4.1
Flash or Inkling-2-Mini, whose checkpoint is the directory CHECKPOINT. For
DeepSeek, `--experts GGUF` reads the routed experts from a GGUF file set
instead. `--draft N` turns on the model's draft head (up to 5 for DeepSeek, up
to 7 for Inkling-2-Mini), `--fp8-kv` stores Inkling-2-Mini's KV caches in
approximate FP8 E4M3 instead of BF16, and `--memory GB` limits the model
memory, state memory included. `build/eremite` alone lists the options.

## Serve

`make server` builds `build/eremite-server`. It serves the OpenAI Chat
Completions and Responses APIs at `http://HOST:PORT/v1`, OpenAI's Decisions API
at `POST /v1/decisions`, OpenAI's Images API at `POST /v1/images/generations`
and `POST /v1/images/edits`, OpenAI's Videos API at `/v1/videos`, OpenAI's
Audio API at `POST /v1/audio/transcriptions` and `POST /v1/audio/translations`,
Codex's model catalog at `/v1/codex/models`, and standalone web search at
`POST /v1/alpha/search`.

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
    },
    {
      "model": "/path/to/Inkling-2-Mini",
      "draft": 7
    },
    {
      "model": "/path/to/d1-2B"
    },
    {
      "model": "/path/to/MiniMax-H3-Image",
      "text_encoder": "/path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      "dit": "/path/to/minimax_h3_image_pruned_int8_convrot.safetensors",
      "vae": "/path/to/minimax_h3_image_vae_fp16.safetensors"
    },
    {
      "model": "/path/to/MiniMax-H3",
      "text_encoder": "/path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      "dit": "/path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      "vae": "/path/to/minimax_h3_video_vae_int8_convrot.safetensors",
      "audio_vae": "/path/to/minimax_h3_audio_vae_fp32.safetensors",
      "variant": "fl2va"
    },
    {
      "model": "/path/to/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
      "memory": 6,
      "draft": 7
    }
  ]
}
```

The API key file holds the key that clients send, 16 to 256 characters.
Without `api_key_file`, the server takes the key from the environment variable
`EREMITE_API_KEY`.

`docs/server.md` lists every setting, and explains how the server handles
requests, memory, decision models, image models, video models and audio
transcription models. `docs/codex.md` explains
how to use eremite with Codex: its configuration, model catalog, image tool
and web search. `skills/README.md` describes eremite's agent skills.

`scripts/setup-searxng.sh` installs SearXNG in
`~/.local/share/eremite/searxng`, the server's default `searxng_dir`. It needs
git and Python 3.10 or later, and downloads about 100 MB.

## Credits

eremite was inspired by [ds4](https://github.com/antirez/ds4), antirez's
local inference engine for DeepSeek 4 Flash and Pro. It also builds on:

- [DeepSeek](https://huggingface.co/deepseek-ai)'s V4.1 Flash,
  [Thinking Machines Lab](https://huggingface.co/thinkingmachines)'s
  Inkling-2-Mini, [Liquid AI](https://www.liquid.ai)'s d1 decision models
  (with Liquid AI's prompt for reranking),
  [MiniMax](https://huggingface.co/MiniMaxAI)'s MiniMax-H3-Image and
  MiniMax-H3 Video with [Comfy-Org](https://huggingface.co/Comfy-Org)'s
  quantized checkpoints, and [NVIDIA](https://huggingface.co/nvidia)'s
  Nemotron-3.5-Transcribe.
- The ggml library of [llama.cpp](https://github.com/ggml-org/llama.cpp): the
  GGUF format and the codebooks of its i-quants. Their MIT license is in
  `src/kernels/gguf_grids.metal`.
- [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)'s
  GGUF expert files.
- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe), which
  renders DeepSeek's prompts and reads its output.
- [SearXNG](https://github.com/searxng/searxng), the search engine behind
  Codex's web tool.
