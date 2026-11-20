#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                                                      &hidden, &seq_len, NULL);
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
  int encoded2 = minimax_h3_text_encoder_encode_prompt(
      enc, prompt, slot, &hidden2, &seq_len2, NULL);
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

  // Regression check 1: GPU embedding lookup on aligned embed_w must match
  // reading the same embedding row directly from the checkpoint on CPU.
  int single_id = ids[0];
  int neg_id = -1;
  Buf h_gpu_emb = gpu_alloc(gpu, slot, (size_t)MINIMAX_H3_TEXT_DIM * 4);
  Buf h_cpu_emb = gpu_alloc(gpu, slot, (size_t)MINIMAX_H3_TEXT_DIM * 4);
  Buf emb_override = gpu_alloc(gpu, slot, (size_t)MINIMAX_H3_TEXT_DIM * 4);
  const TensorIndex *ti = minimax_h3_text_encoder_tensors(enc);
  Tensor ew = tensor_index_get(ti, "model.embed_tokens.weight");
  Tensor es = tensor_index_get(ti, "model.embed_tokens.weight_scale");
  Tensor row_t = ew;
  row_t.off += (uint64_t)single_id * (uint64_t)MINIMAX_H3_TEXT_DIM;
  row_t.size = (uint64_t)MINIMAX_H3_TEXT_DIM;
  int8_t *row_i8 = xmalloc((size_t)MINIMAX_H3_TEXT_DIM);
  minimax_h3_read_tensor_bytes(row_t, row_i8);
  Tensor scale_t = es;
  scale_t.off += (uint64_t)single_id * sizeof(float);
  scale_t.size = sizeof(float);
  float row_scale = 0.0f;
  minimax_h3_read_tensor_bytes(scale_t, &row_scale);
  float *ov = (float *)emb_override.ptr;
  for (int d = 0; d < MINIMAX_H3_TEXT_DIM; d++)
    ov[d] = (float)row_i8[d] * row_scale;
  free(row_i8);

  int ok_g = minimax_h3_text_encoder_encode_tokens(
      enc, &single_id, NULL, NULL, NULL, NULL, 0, 1, &h_gpu_emb, NULL);
  int ok_c = minimax_h3_text_encoder_encode_tokens(
      enc, &neg_id, NULL, &emb_override, NULL, NULL, 0, 1, &h_cpu_emb, NULL);
  float emb_diff = 0.0f;
  for (int d = 0; d < MINIMAX_H3_TEXT_DIM; d++) {
    float d0 = fabsf(((const float *)h_gpu_emb.ptr)[d] -
                     ((const float *)h_cpu_emb.ptr)[d]);
    if (d0 > emb_diff) emb_diff = d0;
  }
  printf("embed lookup vs CPU row max_diff: %.6f\n", emb_diff);
  check(ok_g && ok_c && emb_diff == 0.0f,
        "GPU aligned embedding lookup matches CPU row bit for bit");
  gpu_free(&emb_override);
  gpu_free(&h_cpu_emb);
  gpu_free(&h_gpu_emb);

  // Regression check 2: near-identical prompts differing in one word must yield
  // highly correlated last-token representations (corrupted or misaligned layer
  // weights collapse cosine similarity near zero).
  Buf h_s1 = {0}, h_s2 = {0};
  int n_s1 = 0, n_s2 = 0;
  int ok_s1 = minimax_h3_text_encoder_encode_prompt(
      enc, "The cat sat quietly on the warm wooden mat.", slot, &h_s1, &n_s1,
      NULL);
  int ok_s2 = minimax_h3_text_encoder_encode_prompt(
      enc, "The dog sat quietly on the warm wooden mat.", slot, &h_s2, &n_s2,
      NULL);
  double cos_sim = 0.0;
  if (ok_s1 && ok_s2 && n_s1 > 0 && n_s1 == n_s2) {
    const float *a =
        (const float *)h_s1.ptr + (size_t)(n_s1 - 1) * MINIMAX_H3_TEXT_DIM;
    const float *b =
        (const float *)h_s2.ptr + (size_t)(n_s2 - 1) * MINIMAX_H3_TEXT_DIM;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int d = 0; d < MINIMAX_H3_TEXT_DIM; d++) {
      dot += (double)a[d] * (double)b[d];
      na += (double)a[d] * (double)a[d];
      nb += (double)b[d] * (double)b[d];
    }
    cos_sim = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 0.0;
  }
  printf("near-duplicate prompt last-token cosine similarity: %.4f\n", cos_sim);
  check(ok_s1 && ok_s2 && cos_sim > 0.85,
        "near-identical prompts produce correlated layer-49 hidden states");
  gpu_free(&h_s2);
  gpu_free(&h_s1);

  gpu_free(&hidden2);
  gpu_free(&hidden);
  minimax_h3_text_encoder_close(enc);
  return failures != 0;
}
