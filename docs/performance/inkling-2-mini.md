# Inkling-2-Mini: speed

Measured on 2026-11-21 on an Apple M5 Max with 128 GiB of unified memory,
under macOS 27.2.1.

## Setup

- The model: Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini`), a looped
  24-layer MoE checkpoint (48 logical layer visits, NVFP4 experts on layers
  0 to 19, MXFP8 experts on layers 20 to 23), with its multi-token prediction
  draft head (`mtp.safetensors`) enabled when `draft` is nonzero.
- All weights stay resident in GPU memory. The server's footprint was 18.5 to
  19.3 GiB with `draft` 0, and 19.2 to 20.3 GiB with `draft` 7.
- Settings: `ctx` 32,768, `lookup` 15, `max_cached_states` 1, `state_disk` 0,
  `reserve` 2.
- One request at a time, greedy (temperature 0), with `reasoning_effort` set
  to `"none"`.
- Each run started a fresh server and answered one short warm-up prompt first
  so that all weights were loaded.
- Prompt reading: an 8,784-token source code prompt answered with one token,
  two runs per setting.
- Decoding: three prompts asking for prose, new C code, and a C-to-Rust
  rewrite, two runs each, with replies of 400 tokens. Each speed is the
  reply's tokens divided by its decoding time from the server's log.

## Results

| Drafts (`draft`) | Prefill of 8,784 tokens | Prose | New C code | C-to-Rust rewrite | Footprint |
|---|---|---|---|---|---|
| 0 (`lookup` 15) | 1,146 token/s (1,146 and 1,145), 7.66 s | 48.1 token/s (48.1, 48.1) | 49.6 token/s (49.5, 49.6) | 46.4~61.5 token/s (61.5, 46.4) | 18.5~19.3 GiB |
| 7 (`lookup` 15) | 1,132 token/s (1,144 and 1,120), 7.76 s | 50.0 token/s (51.1, 48.8) | 59.7 token/s (59.8, 59.6) | 59.4 token/s (60.9, 57.9) | 19.2~20.3 GiB |

- Each setting's prefill figure is the mean of two runs, with both runs in
  parentheses. With `draft` 7, prefill also updates the draft head's context
  caches and Markov feature ring.
- On new code, `draft` 7 raised decoding speed from 49.6 to 59.7 token/s
  (+20%). On the second C-to-Rust rewrite run (where context lookup accepted
  only 1 token), `draft` 7 raised decoding speed from 46.4 to 57.9 token/s
  (+25%). On the first rewrite run, context lookup accepted 129 tokens under
  `draft` 0 because the run's prompt prefix led the model to copy more text
  verbatim.
- On prose, adaptive draft-count selection kept `draft` 7 slightly above
  `draft` 0 (48.8 to 51.1 token/s vs 48.1 token/s).

## Limits

- One request at a time. Concurrent batched decoding (up to 8 requests per
  call) was not included in this table.
- Greedy decoding only. Non-zero temperature changes draft acceptance rates.
- Vision encoding (160 x 160 PNG, 16 visual tokens) was measured separately
  in `build/tests/model-inkling2` at 0.19 s.
