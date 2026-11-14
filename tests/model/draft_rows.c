#include "model/deepseek/model.h"
#include "model/deepseek/model.c"
#include "load.h"

// Tests that decoding several sequences together writes each one's draft
// window keys as decoding it alone does; otherwise a sequence that drafts
// again after decoding in a batch attends over a window that lacks those
// positions. Two states decode their next token alone, their draft rings
// kept, then together: each state's rings must be the same, byte for byte.
//
// usage: model-draft_rows CHECKPOINT GGUF (one model process, about a minute)

static Tokenizer *tk;  // the checkpoint's tokenizer

int main(int argc, char **argv) {
  if (argc != 3) die("usage: model-draft_rows CHECKPOINT GGUF");
  double model_memory_gib = 46;  // the draft stages preload their 384 experts
  double model_reserve_gib = 4;
  Options o = {.ctx = 4096,
               .dump_fd = -1,
               .routes_fd = -1,
               .quiet = 1,
               .deepseek.experts = argv[2],
               .draft = 2};
  tk = tokenizer_open(argv[1], deepseek_ops.media_token);
  const char *prompt = "Write a short story about a lighthouse keeper who "
                       "discovers a message in a bottle.";
  int tokens[256], n = tokenizer_encode_with_added(tk, prompt, tokens, 256);
  if (!load_deepseek(argv[1], &o, model_memory_gib, model_reserve_gib))
    die("the model does not fit in %g GiB of model memory", model_memory_gib);
  State *st[2] = {deepseek_state_new(), deepseek_state_new()};
  int len[2] = {n, n - 3}, next[2];
  uint8_t *alone[2][DRAFTS];
  for (int s = 0; s < 2; s++) {
    deepseek_set_state_in_use(st[s]);
    next[s] = sample_greedy(deepseek_forward(tokens, len[s], 0), VOCAB);
    deepseek_save_rewind_point();
    deepseek_forward(&next[s], 1, len[s]);  // alone: its draft keys at len[s]
    for (int d = 0; d < DRAFTS; d++)
      alone[s][d] = xmalloc(D[d].ring.size),
      memcpy(alone[s][d], D[d].ring.ptr, D[d].ring.size);
    deepseek_restore_rewind_point();
  }
  deepseek_decode_rows(st, next, len, 2);
  int bad = 0;
  for (int s = 0; s < 2; s++)
    for (int d = 0; d < DRAFTS; d++) {
      size_t differ = 0;
      const uint8_t *got = st[s]->dring[d].ptr;
      for (size_t i = 0; i < D[d].ring.size; i++)
        differ += got[i] != alone[s][d][i];
      if (differ)
        fprintf(
            stderr,
            "FAIL: sequence %d, stage %d: %zu of %zu draft ring bytes differ\n",
            s, d, differ, D[d].ring.size);
      bad += differ != 0;
    }
  printf("draft rings after decoding together against alone: %s\n",
         bad ? "FAIL" : "PASS");
  return bad != 0;
}
