#include "model/deepseek/model.h"
#include "model/deepseek/model.c"
#include "load.h"

// Tests that the DSpark rings after a verify pass, which keeps some of its
// rows, equal those of decoding the same tokens one at a time.
//
// usage: model-dspark_state CHECKPOINT [EXPERT_GGUF]

static Tokenizer *tk;  // the checkpoint's tokenizer

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: model-dspark_state CHECKPOINT [EXPERT_GGUF]");
  tk = tokenizer_open(argv[1], deepseek_ops.media_token);
  int tokens[2048];
  int n = tokenizer_encode_prompt(
      tk,
      "<｜begin▁of▁sentence｜><｜User｜>Write a short "
      "deterministic task scheduler. "
      "Preserve dependency order and cancel queued work "
      "when a required input fails."
      "<｜Assistant｜></think>",
      tokens, 2048);
  double model_memory_gib = 26;  // the draft stages preload their 384 experts
  double model_reserve_gib = 4;
  Options o = {.ctx = 2048,
               .draft = 1,
               .deepseek.experts = argc == 3 ? argv[2] : NULL,
               .dump_fd = -1,
               .routes_fd = -1,
               .quiet = 1};
  if (!load_deepseek(argv[1], &o, model_memory_gib, model_reserve_gib))
    die("the model does not fit in %g GiB of model memory", model_memory_gib);
  const float *lg = deepseek_forward(tokens, n, 0);
  int rows[BLOCK + 1];
  rows[0] = sample_greedy(lg, VOCAB);
  deepseek_save_rewind_point();
  size_t bytes = D[0].ring.size;
  void *expected = xmalloc((size_t)(BLOCK + 1) * DRAFTS * bytes);
  for (int r = 0; r <= BLOCK; r++) {
    lg = deepseek_forward(rows + r, 1, n + r);
    if (r < BLOCK) rows[r + 1] = sample_greedy(lg, VOCAB);
    for (int d = 0; d < DRAFTS; d++)
      memcpy((char *)expected + ((size_t)r * DRAFTS + d) * bytes, D[d].ring.ptr,
             bytes);
  }
  int failed = 0;
  for (int kept = 1; kept <= BLOCK + 1; kept++) {
    deepseek_restore_rewind_point();
    deepseek_verify(rows, BLOCK + 1, n);
    deepseek_accept(rows, kept, n);
    uint64_t different = 0;
    for (int d = 0; d < DRAFTS; d++) {
      const uint16_t *ref =
          (const uint16_t *)((const char *)expected +
                             ((size_t)(kept - 1) * DRAFTS + d) * bytes);
      const uint16_t *actual = (const uint16_t *)D[d].ring.ptr;
      for (size_t j = 0; j < bytes / sizeof(uint16_t); j++)
        different += actual[j] != ref[j];
    }
    printf("{\"dspark_state_check\":{\"accepted_rows\":%d,\"compared_values\":%"
           "zu,\"different_values\":%llu}}\n",
           kept, DRAFTS * bytes / sizeof(uint16_t),
           (unsigned long long)different);
    fflush(stdout);
    failed |= different != 0;
  }
  free(expected);
  return failed;
}
