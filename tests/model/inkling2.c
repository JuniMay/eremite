#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "eremite.h"
#include "platform/image.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Live checkpoint test for thinkingmachines/Inkling-2-Mini:
// 1. Verifies first-step logits and 30-token greedy completion against the
//    published B200 reference from Hugging Face transformers PR #51050.
// 2. Verifies prefill-vs-stepwise-decode agreement, batched multi-state decode
//    agreement, snapshot take/restore + state_copy fork equivalence, and
//    speculative verify/accept equivalence.
// 3. Reports measured prefill and single-stream decode speeds.
//
// usage: model-inkling2 CHECKPOINT

// Context capacity, memory limit, greedy check length and timing steps.
static constexpr int CONTEXT = 4096;
static constexpr double MEMORY_GB = 32.0;
static constexpr double RESERVE_GB = 2.0;
static constexpr int GOLDEN_STEPS = 30;
static constexpr int GOLDEN_LOGIT_COUNT = 5;
static constexpr int BENCH_PROMPT_TOKENS = 512;
static constexpr int BENCH_DECODE_STEPS = 32;
static constexpr int MAX_TOKENS = 1024;
static constexpr int BATCH_M = 2;

// Tolerances matching Hugging Face PR #51050 (atol=1.5 for 48-visit looped MoE
// BF16 cross-hardware and cross-schedule comparisons) and bit-exact snapshot
// equivalence.
static constexpr float GOLDEN_LOGIT_TOL = 1.5f;
static constexpr float EQUIV_TOL = 1.5f;
static constexpr float EXACT_TOL = 1e-5f;

// Golden first-step logits[:5] and 30-token greedy completion from
// tests/models/inkling2_loop/test_modeling_inkling2_loop.py (HF PR #51050).
static const float GOLDEN_LOGITS[GOLDEN_LOGIT_COUNT] = {
    -12.755271f, -11.535194f, -16.171505f, -17.790428f, -19.750546f};
static const char GOLDEN_PROMPT[] =
    "<|message_system|><|content_text|>Thinking effort level: 0"
    "<|end_message|><|message_user|><|content_text|>"
    "Explain in one sentence why the sky is usually blue.<|end_message|>";
static const char GOLDEN_COMPLETION[] =
    "<|message_model|><|content_thinking|>The user asks for one sentence "
    "explanation why sky is usually blue. Need answer concise. Need ensure "
    "accuracy: Rayleigh scattering: shorter blue";
static const char IMAGE_PROMPT[] =
    "<|message_system|><|content_text|>Thinking effort level: 0"
    "<|end_message|><|message_user|><|content_image|><|unused_200054|>"
    "<|end_message|><|message_user|><|content_text|>"
    "What color is this image? Answer in one word.<|end_message|>";
static constexpr int IMAGE_SIDE = 160;
static constexpr int IMAGE_GEN_STEPS = 48;

// Returns whether haystack contains needle case-insensitively.
static int contains_ci(const char *haystack, const char *needle) {
  size_t nlen = strlen(needle);
  for (const char *p = haystack; *p; p++) {
    size_t k = 0;
    while (k < nlen &&
           tolower((unsigned char)p[k]) == tolower((unsigned char)needle[k]))
      k++;
    if (k == nlen) return 1;
  }
  return 0;
}

// Returns the token ID with the largest logit.
static int32_t argmax(const float *logits, int vocab) {
  int32_t best = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[best]) best = i;
  return best;
}

// Returns the maximum absolute difference between two logit vectors.
static float max_abs_diff(const float *a, const float *b, int n) {
  float worst = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-inkling2 CHECKPOINT");
  check(eremite_init(&(eremite_process){.abi = EREMITE_ABI,
                                        .memory_gb = MEMORY_GB,
                                        .reserve_gb = RESERVE_GB}) == 0,
        "engine initializes");
  eremite_model *m = eremite_open(
      &(eremite_config){.abi = EREMITE_ABI, .model = argv[1], .ctx = CONTEXT});
  double t_load0 = now();
  int loaded = m && eremite_load(m) && eremite_is_loaded(m);
  double t_load = now() - t_load0;
  check(loaded, "Inkling-2-Mini loads");
  if (!loaded) return 1;
  printf("loaded Inkling-2-Mini in %.2f s\n", t_load);

  int vocab = eremite_vocab_size(m);
  eremite_tokenizer *tk = eremite_model_tokenizer(m);

  // 1. Golden first-step logits and 30-token greedy completion.
  int32_t prompt_ids[MAX_TOKENS];
  int n_prompt = eremite_tokenize(tk, GOLDEN_PROMPT, strlen(GOLDEN_PROMPT), 1,
                                  prompt_ids, MAX_TOKENS);
  check(n_prompt > 0, "golden prompt tokenizes");

  eremite_state *s = eremite_state_new(m);
  const float *lg =
      eremite_prefill(m, s, prompt_ids, n_prompt, 0, 0, -1, NULL, NULL, NULL);
  float *first_logits = xmalloc((size_t)vocab * sizeof(float));
  memcpy(first_logits, lg, (size_t)vocab * sizeof(float));

  float worst_golden = 0.0f;
  printf("first logits[:5]:");
  for (int i = 0; i < GOLDEN_LOGIT_COUNT; i++) {
    printf(" %.4f (ref %.4f)", first_logits[i], GOLDEN_LOGITS[i]);
    float d = fabsf(first_logits[i] - GOLDEN_LOGITS[i]);
    if (d > worst_golden) worst_golden = d;
  }
  printf("\n");
  char msg[160];
  snprintf(msg, sizeof msg,
           "first-step logits[:5] match B200 golden (max diff %.4f < %.4f)",
           worst_golden, GOLDEN_LOGIT_TOL);
  check(worst_golden < GOLDEN_LOGIT_TOL, msg);

  int32_t gen[GOLDEN_STEPS];
  char decoded[2048] = "";
  size_t dec_len = 0;
  for (int step = 0; step < GOLDEN_STEPS; step++) {
    gen[step] = argmax(lg, vocab);
    size_t plen = 0;
    const char *piece = eremite_token_bytes(tk, gen[step], &plen);
    if (piece && plen > 0 && dec_len + plen < sizeof decoded) {
      memcpy(decoded + dec_len, piece, plen);
      dec_len += plen;
      decoded[dec_len] = '\0';
    }
    if (step + 1 < GOLDEN_STEPS) {
      int32_t pos = n_prompt + step;
      lg = eremite_decode_rows(m, &s, &gen[step], &pos, 1);
    }
  }
  printf("greedy %d tokens: %s\n", GOLDEN_STEPS, decoded);
  check(!strcmp(decoded, GOLDEN_COMPLETION),
        "30-token greedy completion matches B200 golden verbatim");

  // 2. Stepwise decode vs prefill agreement on the prompt.
  eremite_state *s_step = eremite_state_new(m);
  const float *step_lg = NULL;
  for (int i = 0; i < n_prompt; i++)
    step_lg = eremite_decode_rows(m, &s_step, &prompt_ids[i], &i, 1);
  float *step_copy = xmalloc((size_t)vocab * sizeof(float));
  memcpy(step_copy, step_lg, (size_t)vocab * sizeof(float));
  printf("step  logits[:5]:");
  for (int i = 0; i < GOLDEN_LOGIT_COUNT; i++) printf(" %.4f", step_copy[i]);
  printf(" (argmax %d vs prefill %d)\n", argmax(step_copy, vocab), gen[0]);
  float diff_step = max_abs_diff(first_logits, step_copy, vocab);
  snprintf(msg, sizeof msg,
           "stepwise decode matches prefill (max diff %.4f < %.4f)", diff_step,
           EQUIV_TOL);
  check(diff_step < EQUIV_TOL && argmax(step_copy, vocab) == gen[0], msg);

  // 3. Snapshot capture during prefill + state_copy + snapshot_restore.
  int cap_pos = n_prompt / 2;
  eremite_state *s_cap = eremite_state_new(m);
  eremite_snapshot *snap = NULL;
  const float *cap_full = eremite_prefill(m, s_cap, prompt_ids, n_prompt, 0, 0,
                                          cap_pos, &snap, NULL, NULL);
  float *cap_copy = xmalloc((size_t)vocab * sizeof(float));
  memcpy(cap_copy, cap_full, (size_t)vocab * sizeof(float));
  check(snap != NULL, "prefill captures snapshot at mid-prompt");
  float diff_cap = max_abs_diff(first_logits, cap_copy, vocab);
  printf("cap   logits[:5]:");
  for (int i = 0; i < GOLDEN_LOGIT_COUNT; i++) printf(" %.4f", cap_copy[i]);
  printf(" (diff vs full %.4f)\n", diff_cap);
  check(diff_cap < EQUIV_TOL && argmax(cap_copy, vocab) == gen[0],
        "prefill with capture matches full prefill");

  eremite_state *s_fork = eremite_state_new(m);
  eremite_state_copy(m, s_fork, s_cap, cap_pos);
  eremite_snapshot_restore(m, s_fork, snap);
  const float *fork_lg = eremite_prefill(m, s_fork, prompt_ids, n_prompt,
                                         cap_pos, 0, -1, NULL, NULL, NULL);
  float diff_fork_cap = max_abs_diff(cap_copy, fork_lg, vocab);
  printf("fork  logits[:5]:");
  for (int i = 0; i < GOLDEN_LOGIT_COUNT; i++) printf(" %.4f", fork_lg[i]);
  printf(" (diff vs cap %.6f)\n", diff_fork_cap);
  snprintf(
      msg, sizeof msg,
      "state_copy + snapshot_restore matches capture exactly (%.6f < %.6f)",
      diff_fork_cap, EXACT_TOL);
  check(diff_fork_cap < EXACT_TOL, msg);
  free(step_copy);
  free(cap_copy);

  // 4. Batched multi-state decode (m = 2) vs single-state decode.
  eremite_state *pair[BATCH_M] = {s_step, s_fork};
  int32_t pair_tok[BATCH_M] = {gen[0], gen[0]};
  int32_t pair_pos[BATCH_M] = {n_prompt, n_prompt};
  const float *batched_lg =
      eremite_decode_rows(m, pair, pair_tok, pair_pos, BATCH_M);
  int32_t next0 = argmax(batched_lg, vocab);
  int32_t next1 = argmax(batched_lg + vocab, vocab);
  check(next0 == gen[1] && next1 == gen[1],
        "batched multi-state decode matches single-state argmax");

  // 5. Speculative verify + accept vs sequential decode.
  eremite_state *s_ver = eremite_state_new(m);
  eremite_prefill(m, s_ver, prompt_ids, n_prompt, 0, 0, -1, NULL, NULL, NULL);
  int32_t vtoks[4] = {gen[0], gen[1], gen[2], gen[3]};
  const float *vlg = eremite_verify(m, s_ver, vtoks, 4, n_prompt);
  int ver_ok = 1;
  for (int i = 0; i < 4; i++)
    if (argmax(vlg + (size_t)i * vocab, vocab) != gen[i + 1]) ver_ok = 0;
  check(ver_ok, "verify produces matching argmax for 4 speculative tokens");
  eremite_accept(m, s_ver, vtoks, 2, n_prompt);
  int32_t p2 = n_prompt + 2;
  const float *after_acc = eremite_decode_rows(m, &s_ver, &gen[2], &p2, 1);
  check(argmax(after_acc, vocab) == gen[3],
        "accept + rewind continues with matching next token");

  // 6. Live multimodal vision check on a synthetic 160x160 solid red PNG.
  uint8_t *red_rgb = xmalloc((size_t)IMAGE_SIDE * IMAGE_SIDE * 3);
  for (int i = 0; i < IMAGE_SIDE * IMAGE_SIDE; i++) {
    red_rgb[i * 3 + 0] = 255;
    red_rgb[i * 3 + 1] = 0;
    red_rgb[i * 3 + 2] = 0;
  }
  size_t png_len = 0;
  const char *ierr = NULL;
  uint8_t *png =
      image_encode_png(red_rgb, IMAGE_SIDE, IMAGE_SIDE, &png_len, &ierr);
  free(red_rgb);
  check(png != NULL, "synthetic red PNG encodes");
  const uint8_t *png_arr[1] = {png};
  size_t len_arr[1] = {png_len};
  char merr[128] = "";
  eremite_media *med =
      eremite_media_open(m, png_arr, len_arr, 1, merr, sizeof merr);
  check(med != NULL, "eremite_media_open accepts PNG for Inkling-2-Mini");
  int32_t img_raw[MAX_TOKENS], img_exp[MAX_TOKENS];
  int n_iraw = eremite_tokenize(tk, IMAGE_PROMPT, strlen(IMAGE_PROMPT), 1,
                                img_raw, MAX_TOKENS);
  int n_iexp = eremite_media_expand(med, img_raw, n_iraw, img_exp, MAX_TOKENS);
  check(n_iexp == n_iraw + 15,
        "160x160 image expands 1 placeholder into 16 (4x4) image tokens");
  double tv0 = now();
  int encoded = eremite_media_encode(med, 0);
  double tv = now() - tv0;
  check(encoded == 1, "eremite_media_encode runs 23-layer ViT + projector");
  eremite_state *s_vis = eremite_state_new(m);
  const float *v_lg =
      eremite_prefill(m, s_vis, img_exp, n_iexp, 0, 0, -1, NULL, NULL, med);
  char vtext[1024] = "";
  size_t vlen = 0;
  for (int step = 0; step < IMAGE_GEN_STEPS; step++) {
    int32_t id = argmax(v_lg, vocab);
    if (eremite_is_stop_token(m, id)) break;
    size_t plen = 0;
    const char *piece = eremite_token_bytes(tk, id, &plen);
    if (piece && plen > 0 && vlen + plen < sizeof vtext) {
      memcpy(vtext + vlen, piece, plen);
      vlen += plen;
      vtext[vlen] = '\0';
    }
    int32_t pos = n_iexp + step;
    v_lg = eremite_decode_rows(m, &s_vis, &id, &pos, 1);
  }
  printf("vision encode %.3f s, reply: %s\n", tv, vtext);
  check(contains_ci(vtext, "red"),
        "multimodal vision reply identifies solid red image");
  eremite_state_free(m, s_vis);
  eremite_media_close(med);
  free(png);

  // 7. Measure 512-token prefill and 32-token single-stream decode speed.
  int32_t bench_ids[BENCH_PROMPT_TOKENS];
  for (int i = 0; i < BENCH_PROMPT_TOKENS; i++)
    bench_ids[i] = prompt_ids[i % n_prompt];
  eremite_state *s_bench = eremite_state_new(m);
  double t0 = now();
  const float *blg = eremite_prefill(m, s_bench, bench_ids, BENCH_PROMPT_TOKENS,
                                     0, 0, -1, NULL, NULL, NULL);
  double t_prefill = now() - t0;
  int32_t tok = argmax(blg, vocab);
  double td0 = now();
  for (int step = 0; step < BENCH_DECODE_STEPS; step++) {
    int32_t pos = BENCH_PROMPT_TOKENS + step;
    blg = eremite_decode_rows(m, &s_bench, &tok, &pos, 1);
    tok = argmax(blg, vocab);
  }
  double t_decode = now() - td0;
  printf("measured speed: prefill %d tok in %.3f s (%.1f tok/s), "
         "decode %d tok in %.3f s (%.1f tok/s)\n",
         BENCH_PROMPT_TOKENS, t_prefill, BENCH_PROMPT_TOKENS / t_prefill,
         BENCH_DECODE_STEPS, t_decode, BENCH_DECODE_STEPS / t_decode);

  free(first_logits);
  eremite_snapshot_free(m, snap);
  eremite_state_free(m, s);
  eremite_state_free(m, s_step);
  eremite_state_free(m, s_cap);
  eremite_state_free(m, s_fork);
  eremite_state_free(m, s_ver);
  eremite_state_free(m, s_bench);
  return failures != 0;
}
