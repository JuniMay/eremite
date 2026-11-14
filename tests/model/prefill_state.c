#include "model/deepseek/model.h"
#include "model/deepseek/model.c"
#include "load.h"

// Tests that a prefill keeps each layer's window and the rewind margin in its
// ring: the rows of the last layer's ring equal the prompt's rows.
//
// usage: model-prefill_state CHECKPOINT [EXPERT_GGUF]

static Tokenizer *tk;  // the checkpoint's tokenizer

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: model-prefill_state CHECKPOINT [EXPERT_GGUF]");
  tk = tokenizer_open(argv[1], deepseek_ops.media_token);
  int pattern[64], tokens[256];
  int pn = tokenizer_encode_with_added(tk, " alpha beta gamma delta epsilon",
                                       pattern, 64);
  if (!pn) die("empty pattern");
  for (int i = 0; i < 256; i++) tokens[i] = pattern[i % pn];
  double model_memory_gib = 20;
  double model_reserve_gib = 4;
  Options o = {.ctx = 2048,
               .deepseek.experts = argc == 3 ? argv[2] : NULL,
               .dump_fd = -1,
               .routes_fd = -1,
               .quiet = 1};
  if (!load_deepseek(argv[1], &o, model_memory_gib, model_reserve_gib))
    die("the model does not fit in %g GiB of model memory", model_memory_gib);
  deepseek_forward(tokens, 256, 0);
  const uint16_t *all = (const uint16_t *)WIN.ptr;
  const uint16_t *ring = (const uint16_t *)L[LAYERS - 1].ring.ptr;
  uint64_t bad_window = 0, bad_margin = 0;
  for (int p = 256 - RING; p < 256; p++)
    for (int d = 0; d < HEAD_DIM; d++) {
      int different = ring[(p % RING) * HEAD_DIM + d] != all[p * HEAD_DIM + d];
      if (p < 256 - WINDOW)
        bad_margin += different;
      else
        bad_window += different;
    }
  printf("{\"prefill_state_check\":{\"window_rows\":%d,\"rewind_rows\":%d,"
         "\"window_differences\":%llu,\"rewind_differences\":%llu}}\n",
         WINDOW, BLOCK, (unsigned long long)bad_window,
         (unsigned long long)bad_margin);
  return bad_window || bad_margin;
}
