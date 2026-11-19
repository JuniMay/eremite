# DeepSeek V4.1 Flash: speed

Measured on 2026-11-18 with eremite at commit 28ff0bb2, on an Apple M5 Max
(6 Super and 12 Performance cores) with 128 GiB, under macOS 27.2.1.

## Setup

- The model: DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash-1017`),
  with three sets of routed experts: Unsloth's UD-IQ1_M and UD-IQ2_XXS GGUF
  files, and the checkpoint's own MXFP4 experts.
- The model's own `memory` was 64 GiB, and its footprint 64.1 to 65.7 GiB.
  About 77 GiB was free before each run.
- Every other setting had its default: `lookup` 15, `draft` 0 and the most
  context.
- One request at a time, greedy (temperature 0), with thinking off.
- Each run was a fresh server with no saved states. It first answered a short
  request, so that the weights were loaded.
- Decoding: three prompts, which ask for prose, for new code, and for code
  rewritten from the prompt, two runs each, with replies of up to 400 tokens.
  The speed is the reply's tokens over its decoding time, from the prompt's
  end to the reply's end, as the server's log gives them.
- Prompt reading: a prompt of 8,921 tokens of source code, answered with one
  token, two runs. Each run started with no experts in memory.

## Results

| Experts | Prefill of 8,921 tokens | Decoding | Experts found in memory |
|---|---|---|---|
| UD-IQ1_M | 688 token/s (637 and 738), 13.0 s | 22.9~27.4 token/s | 93%~97% |
| UD-IQ2_XXS | 664 token/s (654 and 675), 13.4 s | 20.9~24.5 token/s | 93%~96% |
| MXFP4 | 448 token/s (446 and 449), 19.9 s | 9.6~14.7 token/s | 87%~93% |

- Prefill gives the median of two runs, with both runs in parentheses.
- Decoding speed depends on the text. In each set, prose decoded fastest and
  code rewritten from the prompt slowest. That reply needed the most experts
  that were not in memory.
- The two runs of a decoding case differed by under 1%, except one case of
  UD-IQ1_M (6%).

## Limits

- One request at a time. Several requests at once were not measured.
- One memory size. With more memory, more experts stay in memory, and
  decoding is faster.
- Each expert set gives its own replies, so their lengths differ.
