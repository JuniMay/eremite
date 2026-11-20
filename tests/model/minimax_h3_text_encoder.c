#include "check.h"
#include "base/clock.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/text_encoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Tests MiniMax-H3's Qwen3-VL-32B NVFP4 text encoder on a real checkpoint.
// Pass the weights file and tokenizer directory as arguments, or set
// EREMITE_MINIMAX_H3_TEXT_ENCODER and EREMITE_MINIMAX_H3_TOKENIZER.

// Maximum token count inspected in this test.
static constexpr int MAX_TEST_TOKENS = 256;

// Prints summary statistics and checks finiteness for an encoded hidden buffer.
static int check_hidden(const float *data, int seq_len, double *mean_abs_out) {
  size_t n = (size_t)seq_len * (size_t)MINIMAX_H3_TEXT_DIM;
  double sum_abs = 0.0;
  float min_v = data[0], max_v = data[0];
  for (size_t i = 0; i < n; i++) {
    float v = data[i];
    if (!isfinite(v)) return 0;
    sum_abs += fabsf(v);
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
  }
  *mean_abs_out = sum_abs / (double)n;
  printf("hidden [%d, %d]: min=%.4f max=%.4f mean_abs=%.4f first4=(%.4f, %.4f, "
         "%.4f, %.4f)\n",
         seq_len, MINIMAX_H3_TEXT_DIM, min_v, max_v, *mean_abs_out, data[0],
         data[1], data[2], data[3]);
  return *mean_abs_out > 0.0;
}

int main(int argc, char **argv) {
  const char *weights =
      argc > 1 ? argv[1] : getenv("EREMITE_MINIMAX_H3_TEXT_ENCODER");
  const char *tokenizer =
      argc > 2 ? argv[2] : getenv("EREMITE_MINIMAX_H3_TOKENIZER");
  if (!weights || !tokenizer) {
    printf("usage: %s <qwen3vl_nvfp4.safetensors> <tokenizer_dir>\n", argv[0]);
    return 1;
  }

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-text-test", 0);
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  MinimaxH3TextEncoder *enc =
      minimax_h3_text_encoder_open(gpu, &account, weights, tokenizer);
  uint64_t required = minimax_h3_text_encoder_required_bytes(enc);
  printf("required weights: %.2f GiB\n",
         (double)required / (double)(1ull << 30));

  double t0 = now();
  int loaded = minimax_h3_text_encoder_load(enc);
  double load_s = now() - t0;
  printf("loaded in %.2f s (resident %.2f GiB)\n", load_s,
         (double)minimax_h3_text_encoder_resident_bytes(enc) /
             (double)(1ull << 30));
  check(loaded, "Qwen3-VL-32B NVFP4 text encoder loads cleanly");

  // Empty prompt falls back to the pad token (151643).
  int empty_ids[MAX_TEST_TOKENS];
  int empty_n =
      minimax_h3_text_encoder_tokenize(enc, "", empty_ids, MAX_TEST_TOKENS);
  check(empty_n == 1 && empty_ids[0] == MINIMAX_H3_TEXT_PAD_TOKEN,
        "empty prompt tokenizes to a single pad token");

  // Raw text tokenization must not treat special token spellings as special.
  const char *prompt = "A red apple on a wooden table in soft morning light.";
  int ids[MAX_TEST_TOKENS];
  int tok_n =
      minimax_h3_text_encoder_tokenize(enc, prompt, ids, MAX_TEST_TOKENS);
  printf("prompt tokens (%d):", tok_n);
  for (int i = 0; i < tok_n; i++) printf(" %d", ids[i]);
  printf("\n");
  check(tok_n > 0 && tok_n < MAX_TEST_TOKENS, "prompt tokenizes cleanly");

  Buf hidden = {0};
  int seq_len = 0;
  double t1 = now();
  int encoded = minimax_h3_text_encoder_encode_prompt(enc, prompt, slot,
                                                      &hidden, &seq_len);
  double enc_ms = (now() - t1) * 1000.0;
  printf("encoded %d tokens in %.1f ms\n", seq_len, enc_ms);
  double mean_abs = 0.0;
  check(encoded && seq_len == tok_n &&
            check_hidden((const float *)hidden.ptr, seq_len, &mean_abs),
        "text encoder produces finite nonzero layer-49 hidden states");

  // Determinism: encoding the same prompt a second time gives identical values.
  Buf hidden2 = {0};
  int seq_len2 = 0;
  double t2 = now();
  int encoded2 = minimax_h3_text_encoder_encode_prompt(enc, prompt, slot,
                                                       &hidden2, &seq_len2);
  printf("second encode of %d tokens in %.1f ms\n", seq_len2,
         (now() - t2) * 1000.0);
  float max_diff = 0.0f;
  if (encoded && encoded2 && seq_len == seq_len2) {
    const float *a = (const float *)hidden.ptr, *b = (const float *)hidden2.ptr;
    size_t total = (size_t)seq_len * (size_t)MINIMAX_H3_TEXT_DIM;
    for (size_t i = 0; i < total; i++) {
      float d = fabsf(a[i] - b[i]);
      if (d > max_diff) max_diff = d;
    }
  }
  check(encoded2 && seq_len2 == seq_len && max_diff == 0.0f,
        "repeated encoding is bit-exact deterministic");

  gpu_free(&hidden2);
  gpu_free(&hidden);
  minimax_h3_text_encoder_close(enc);
  return failures != 0;
}
