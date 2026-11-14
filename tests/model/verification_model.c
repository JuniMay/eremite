#include "base/error.h"
#include "engine/sampling.h"
#include "engine/tokenizer.h"
#include "model/deepseek/model.h"
#include "load.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests a verify pass of 16 rows, and the rollback of the rows it rejects,
// against decoding the same tokens one at a time: every logit, and with
// --draft every draft, must be the same.
//
// usage: model-verification_model CHECKPOINT [--long] [--draft]
//        [--experts GGUF]

static Tokenizer *tk;  // the checkpoint's tokenizer

static int compare(const float *a, const float *b) {
  int errors = 0;
  for (int i = 0; i < VOCAB; i++)
    errors += !isfinite(a[i]) || !isfinite(b[i]) ||
              memcmp(a + i, b + i, sizeof(float)) != 0;
  return errors;
}

int main(int argc, char **argv) {
  if (argc < 2)
    die("usage: model-verification_model CHECKPOINT [--long] [--draft] "
        "[--experts GGUF]");
  double model_memory_gib = 20;
  double model_reserve_gib = 4;
  Options o = {.ctx = 2048,
               .lookup = VERIFY_DRAFTS,
               .dump_fd = -1,
               .routes_fd = -1,
               .quiet = 1};
  int longer = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long"))
      longer = 1, o.ctx = 32768;
    else if (!strcmp(argv[i], "--draft"))
      o.draft = 2, model_memory_gib += 6;  // the stages' 384 experts
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc)
      o.deepseek.experts = argv[++i];
    else
      die("invalid verification check option");
  }
  tk = tokenizer_open(argv[1], deepseek_ops.media_token);
  int seed[128], n = longer ? 16521 : 257;
  int *tokens = xmalloc((size_t)n * sizeof *tokens);
  int length =
      tokenizer_encode_with_added(tk,
                                  " Review the queue, keep duplicate jobs, and "
                                  "preserve cancellation order.\n",
                                  seed, 128);
  if (!length) die("empty fixture prompt");
  for (int i = 0; i < n; i++) tokens[i] = seed[i % length];
  if (!load_deepseek(argv[1], &o, model_memory_gib, model_reserve_gib))
    die("the model does not fit in %g GiB of model memory", model_memory_gib);
  int rows[VERIFY_DRAFTS + 2];
  rows[0] = sample_greedy(deepseek_forward(tokens, n, 0), VOCAB);
  for (int i = 1; i < VERIFY_DRAFTS + 2; i++) rows[i] = seed[(i * 7) % length];
  deepseek_save_rewind_point();
  float *reference =
      xmalloc((size_t)(VERIFY_DRAFTS + 2) * VOCAB * sizeof(float));
  int draft_reference[VERIFY_DRAFTS + 2][2];
  for (int r = 0; r < VERIFY_DRAFTS + 2; r++) {
    if (o.draft) deepseek_propose_drafts(rows[r], n + r, draft_reference[r], 2);
    memcpy(reference + (size_t)r * VOCAB, deepseek_forward(rows + r, 1, n + r),
           VOCAB * sizeof(float));
  }
  deepseek_restore_rewind_point();
  const float *got = deepseek_verify(rows, VERIFY_DRAFTS + 1, n);
  int failed = 0;
  for (int r = 0; r <= VERIFY_DRAFTS; r++) {
    int different =
        compare(got + (size_t)r * VOCAB, reference + (size_t)r * VOCAB);
    printf("{\"verify_row\":%d,\"different_logits\":%d}\n", r, different);
    failed |= different != 0;
  }
  fflush(stdout);
  const int kept_cases[] = {1, 2, 6, 15, 16};
  for (size_t i = 0; i < sizeof kept_cases / sizeof *kept_cases; i++) {
    int kept = kept_cases[i], drafts[2], draft_diff = 0;
    deepseek_restore_rewind_point();
    deepseek_verify(rows, VERIFY_DRAFTS + 1, n);
    deepseek_accept(rows, kept, n);
    if (o.draft) {
      deepseek_propose_drafts(rows[kept], n + kept, drafts, 2);
      draft_diff = memcmp(drafts, draft_reference[kept], sizeof drafts) != 0;
    }
    got = deepseek_forward(rows + kept, 1, n + kept);
    int different = compare(got, reference + (size_t)kept * VOCAB);
    printf("{\"rollback_kept\":%d,\"different_logits\":%d,\"different_drafts\":"
           "%d}\n",
           kept, different, draft_diff);
    fflush(stdout);
    failed |= different != 0 || draft_diff;
  }
  free(reference), free(tokens);
  return failed;
}
