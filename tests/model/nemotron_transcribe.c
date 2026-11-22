#include "check.h"
#include "base/clock.h"
#include "base/file.h"
#include "engine/tokenizer.h"
#include "eremite.h"
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Formats a check description and records the result.
static void checkf(int ok, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void checkf(int ok, const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  check(ok, buf);
}

// Live end-to-end speech transcription test for Nemotron-3.5-Transcribe:
// 1. Loads the checkpoint under an optional --memory-gb limit (allowing both
//    full-resident and SSD-streaming verification).
// 2. Decodes an audio file, runs the Parallel Expert Perception encoder,
//    expands the audio span, prefills the 52-layer hybrid backbone, and
//    greedily decodes the verbatim transcript.
// 3. Verifies snapshot capture, state copy and snapshot restore reproduce
//    identical next-token logits.
//
// usage: model-nemotron_transcribe CHECKPOINT [AUDIO_FILE] [--say TEXT]
//        [--memory-gb N] [--draft N] [--expect SUBSTRING]

static constexpr int MAX_PROMPT_TOKENS = 65536;
static constexpr int MAX_GEN_TOKENS = 256;
static constexpr int MODEL_CTX = 8192;
static constexpr int MAX_VERIFY_ROWS = 16;
static constexpr size_t TRANSCRIPT_CAP = 16384;
static constexpr float SNAP_LOGIT_TOL = 1e-4f;

// Returns nonzero when needle appears in haystack ignoring ASCII case.
static int contains_ci(const char *haystack, const char *needle) {
  size_t nlen = strlen(needle);
  if (nlen == 0) return 1;
  for (const char *p = haystack; *p; p++) {
    size_t i = 0;
    while (i < nlen && p[i] &&
           tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i]))
      i++;
    if (i == nlen) return 1;
  }
  return 0;
}

// Returns the argmax token index of logits[0..vocab-1].
static int32_t argmax_token(const float *logits, int vocab) {
  int32_t best = 0;
  float bv = logits[0];
  for (int i = 1; i < vocab; i++) {
    if (logits[i] > bv) {
      bv = logits[i];
      best = i;
    }
  }
  return best;
}

static constexpr char DEFAULT_SAY_TEXT[] =
    "The quick brown fox jumps over the lazy dog. "
    "Local speech recognition on Apple silicon runs reliably.";

// Synthesizes spoken English WAV audio into memory using macOS /usr/bin/say
// and removes the temporary file immediately after reading.
static uint8_t *synthesize_wav(const char *phrase, size_t *out_len) {
  char tmpl[] = "/tmp/eremite-nemo-XXXXXX.wav";
  int fd = mkstemps(tmpl, 4);
  if (fd < 0) return NULL;
  close(fd);
  pid_t pid = fork();
  if (pid == 0) {
    execl("/usr/bin/say", "say", "-o", tmpl, "--file-format=WAVE",
          "--data-format=LEI16@16000", phrase, (char *)NULL);
    _exit(127);
  }
  int status = 0;
  if (pid < 0 || waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    unlink(tmpl);
    return NULL;
  }
  uint8_t *buf = (uint8_t *)read_file(tmpl, out_len);
  unlink(tmpl);
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s CHECKPOINT [AUDIO_FILE] [--say TEXT] "
            "[--memory-gb N] [--draft N] [--expect SUBSTRING]\n",
            argv[0]);
    return 2;
  }
  const char *dir = argv[1];
  const char *audio_path = NULL;
  const char *say_text = NULL;
  double memory_gb = 0.0;
  int draft = 0;
  const char *expect = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--memory-gb") && i + 1 < argc) {
      memory_gb = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--draft") && i + 1 < argc) {
      draft = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--expect") && i + 1 < argc) {
      expect = argv[++i];
    } else if (!strcmp(argv[i], "--say") && i + 1 < argc) {
      say_text = argv[++i];
    } else if (argv[i][0] != '-' && !audio_path) {
      audio_path = argv[i];
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (!audio_path && !say_text) {
    say_text = DEFAULT_SAY_TEXT;
    if (!expect) expect = "quick brown fox";
  }

  eremite_process proc = {.abi = EREMITE_ABI, .memory_gb = memory_gb};
  checkf(eremite_init(&proc) == 0, "initialized eremite (memory_gb=%.1f)",
         memory_gb);

  eremite_config cfg = {.abi = EREMITE_ABI,
                        .model = dir,
                        .ctx = MODEL_CTX,
                        .deepseek = {.draft = draft}};
  eremite_model *m = eremite_open(&cfg);
  checkf(m != NULL, "opened Nemotron-3.5-Transcribe model (draft=%d)", draft);

  double t_load0 = now();
  checkf(eremite_load(m) == 1, "loaded model weights (%.2fs)", now() - t_load0);
  checkf(eremite_vocab_size(m) == 131072, "vocab_size is 131072 (got %d)",
         eremite_vocab_size(m));
  checkf(eremite_drafts_per_pass(m) == draft,
         "drafts_per_pass matches requested draft=%d (got %d)", draft,
         eremite_drafts_per_pass(m));
  checkf(eremite_replay_rows(m, 10) == -1,
         "replay_rows reports snapshot-only resume (-1)");

  size_t audio_len = 0;
  uint8_t *audio_bytes = audio_path
                             ? (uint8_t *)read_file(audio_path, &audio_len)
                             : synthesize_wav(say_text, &audio_len);
  checkf(audio_bytes != NULL && audio_len > 0, "loaded audio (%zu bytes)",
         audio_len);

  char err[256] = {0};
  const uint8_t *data_arr[1] = {audio_bytes};
  size_t len_arr[1] = {audio_len};
  eremite_media *media =
      eremite_media_open(m, data_arr, len_arr, 1, err, sizeof err);
  checkf(media != NULL, "decoded audio media (err=%s)", err);

  const char *prompt =
      "<|im_start|>system\n<|im_end|>\n"
      "<|im_start|>user\nTranscribe the following audio. " MEDIA_MARK
      "<|im_end|>\n"
      "<|im_start|>assistant\n<think></think>";
  eremite_tokenizer *tk = eremite_model_tokenizer(m);
  int32_t *raw_toks = malloc(MAX_PROMPT_TOKENS * sizeof(int32_t));
  int32_t *exp_toks = malloc(MAX_PROMPT_TOKENS * sizeof(int32_t));
  int raw_n = eremite_tokenize(tk, prompt, strlen(prompt), 2, raw_toks,
                               MAX_PROMPT_TOKENS);
  checkf(raw_n > 0, "tokenized prompt (%d base tokens)", raw_n);
  int exp_n =
      eremite_media_expand(media, raw_toks, raw_n, exp_toks, MAX_PROMPT_TOKENS);
  checkf(exp_n > raw_n, "expanded audio span (%d total prompt tokens)", exp_n);

  double t_enc0 = now();
  int encoded = eremite_media_encode(media, 0);
  double enc_s = now() - t_enc0;
  checkf(encoded == 1, "encoded audio with Perception (%.3fs)", enc_s);

  // Prefill full prompt while capturing a snapshot at exp_n - 1 to verify
  // snapshot restore + state copy produces identical logits.
  eremite_state *st = eremite_state_new(m);
  checkf(st != NULL, "allocated hybrid state");
  eremite_snapshot *snap = NULL;
  int snap_pos = exp_n - 1;
  double t_pre0 = now();
  const float *logits = eremite_prefill(m, st, exp_toks, exp_n, 0, 0, snap_pos,
                                        &snap, NULL, media);
  double pre_s = now() - t_pre0;
  checkf(logits != NULL && snap != NULL,
         "prefilled %d tokens in %.3fs (%.1f tok/s)", exp_n, pre_s,
         (double)exp_n / pre_s);

  int vocab = eremite_vocab_size(m);
  int32_t first_tok = argmax_token(logits, vocab);
  float *ref_logits = malloc((size_t)vocab * sizeof(float));
  memcpy(ref_logits, logits, (size_t)vocab * sizeof(float));

  // Verify snapshot restore into a forked state produces identical logits for
  // the last prompt token.
  eremite_state *st2 = eremite_state_new(m);
  checkf(st2 != NULL, "allocated second state for snapshot check");
  eremite_state_copy(m, st2, st, snap_pos);
  eremite_snapshot_restore(m, st2, snap);
  int32_t last_id = exp_toks[snap_pos];
  int32_t last_pos = snap_pos;
  const float *logits2 = eremite_decode_rows(m, &st2, &last_id, &last_pos, 1);
  checkf(logits2 != NULL, "re-decoded last token from restored snapshot");
  float snap_max = 0.0f;
  for (int i = 0; i < vocab; i++) {
    float d = fabsf(logits2[i] - ref_logits[i]);
    if (d > snap_max) snap_max = d;
  }
  int32_t first_tok2 = argmax_token(logits2, vocab);
  checkf(first_tok == first_tok2 && snap_max <= SNAP_LOGIT_TOL,
         "snapshot restore matches prefill logits (tok=%d vs %d, max=%.2e)",
         first_tok, first_tok2, snap_max);
  eremite_snapshot_free(m, snap);
  eremite_state_free(m, st2);
  free(ref_logits);

  // Capture prompt-end snapshot before greedy decoding so verify()+accept() can
  // be checked against sequential 1-token decode from the identical state.
  eremite_snapshot *prompt_end_snap = eremite_snapshot_take(m, st, exp_n, NULL);
  checkf(prompt_end_snap != NULL, "captured prompt-end snapshot");

  // Greedy decode full transcript from st, recording generated tokens and
  // per-step logits for verify()+accept() equivalence checks.
  char text[TRANSCRIPT_CAP];
  size_t text_len = 0;
  text[0] = '\0';
  int32_t gen_toks[MAX_GEN_TOKENS];
  int32_t cur_tok = first_tok;
  int32_t cur_pos = exp_n;
  int gen_count = 0;
  double t_dec0 = now();
  while (gen_count < MAX_GEN_TOKENS && !eremite_is_stop_token(m, cur_tok)) {
    gen_toks[gen_count] = cur_tok;
    size_t blen = 0;
    const char *bytes = eremite_token_bytes(tk, cur_tok, &blen);
    if (bytes && blen > 0 && text_len + blen < sizeof text - 1) {
      memcpy(text + text_len, bytes, blen);
      text_len += blen;
      text[text_len] = '\0';
    }
    const float *dl = eremite_decode_rows(m, &st, &cur_tok, &cur_pos, 1);
    if (!dl) break;
    cur_tok = argmax_token(dl, vocab);
    cur_pos++;
    gen_count++;
  }
  double dec_s = now() - t_dec0;

  // Verify multi-token verify() + accept() on both MV (count=5 <= 8) and MM
  // (count=12 > 8) paths, including partial Mamba-2 state rollback after
  // rejected garbage draft tokens.
  if (gen_count >= 12) {
    // 1. MV verify (count=5): 3 true tokens + 2 wrong tokens -> accept(3) ->
    // verify next decoded token matches gen_toks[3] and its logits match
    // sequential decode.
    eremite_state *sv = eremite_state_new(m);
    eremite_state_copy(m, sv, st, exp_n);
    eremite_snapshot_restore(m, sv, prompt_end_snap);
    int32_t vrows5[5] = {gen_toks[0], gen_toks[1], gen_toks[2], 999, 1000};
    const float *vl5 = eremite_verify(m, sv, vrows5, 5, exp_n);
    checkf(vl5 != NULL, "verify(count=5) MV path succeeded");
    int mv_preds_ok =
        (argmax_token(vl5, vocab) == gen_toks[1] &&
         argmax_token(vl5 + (size_t)vocab, vocab) == gen_toks[2] &&
         argmax_token(vl5 + (size_t)2 * vocab, vocab) == gen_toks[3]);
    checkf(mv_preds_ok,
           "verify(count=5) predicted exact next tokens for rows 0..2");
    eremite_accept(m, sv, vrows5, 3, exp_n);
    int32_t next_tok = gen_toks[3];
    int32_t next_pos = exp_n + 3;
    const float *after_rb =
        eremite_decode_rows(m, &sv, &next_tok, &next_pos, 1);
    checkf(
        after_rb != NULL && argmax_token(after_rb, vocab) == gen_toks[4],
        "partial accept(3/5) rewound Mamba-2 state accurately (next=%d vs %d)",
        after_rb ? argmax_token(after_rb, vocab) : -1, gen_toks[4]);

    // 2. MM verify (count=12 > 8): 4 true tokens + 8 wrong tokens -> accept(4)
    // -> verify subsequent decode matches gen_toks[5], then full accept(7/7).
    eremite_state_copy(m, sv, st, exp_n);
    eremite_snapshot_restore(m, sv, prompt_end_snap);
    int32_t vrows12[12];
    for (int i = 0; i < 4; i++) vrows12[i] = gen_toks[i];
    for (int i = 4; i < 12; i++) vrows12[i] = 500 + i;
    const float *vl12 = eremite_verify(m, sv, vrows12, 12, exp_n);
    checkf(vl12 != NULL, "verify(count=12) MM path succeeded");
    int mm_preds_ok =
        (argmax_token(vl12, vocab) == gen_toks[1] &&
         argmax_token(vl12 + (size_t)vocab, vocab) == gen_toks[2] &&
         argmax_token(vl12 + (size_t)2 * vocab, vocab) == gen_toks[3] &&
         argmax_token(vl12 + (size_t)3 * vocab, vocab) == gen_toks[4]);
    checkf(mm_preds_ok,
           "verify(count=12) predicted exact next tokens for rows 0..3");
    eremite_accept(m, sv, vrows12, 4, exp_n);
    int32_t p4 = exp_n + 4;
    const float *vl_full = eremite_verify(m, sv, &gen_toks[4], 7, p4);
    checkf(vl_full != NULL &&
               argmax_token(vl_full + (size_t)6 * vocab, vocab) == gen_toks[11],
           "post-MM-rewind full verify(7) matches sequential decode (tok=%d vs "
           "%d)",
           vl_full ? argmax_token(vl_full + (size_t)6 * vocab, vocab) : -1,
           gen_toks[11]);
    eremite_accept(m, sv, &gen_toks[4], 7, p4);
    int32_t t11 = gen_toks[11];
    int32_t p11 = exp_n + 11;
    const float *after_full = eremite_decode_rows(m, &sv, &t11, &p11, 1);
    int32_t expect12 = gen_count > 12 ? gen_toks[12] : cur_tok;
    checkf(after_full != NULL && argmax_token(after_full, vocab) == expect12,
           "full accept(7/7) leaves Mamba-2+KV state exact (tok=%d vs %d)",
           after_full ? argmax_token(after_full, vocab) : -1, expect12);
    eremite_state_free(m, sv);
  }

  // When MTP drafting is enabled (--draft N), decode the full transcript using
  // speculative propose_drafts + verify + sample_verify_drafts + accept from
  // the restored prompt-end snapshot, asserting exact token-by-token match
  // against greedy single-token decoding and reporting acceptance rate + speed.
  if (draft > 0 && gen_count > 0) {
    eremite_state *sm = eremite_state_new(m);
    checkf(sm != NULL, "allocated state for MTP speculative decode");
    eremite_state_copy(m, sm, st, exp_n);
    eremite_snapshot_restore(m, sm, prompt_end_snap);

    eremite_sampler *greedy = eremite_sampler_new(0.0, 1.0, 0, 0, 0);
    int32_t spec_toks[MAX_GEN_TOKENS];
    int spec_count = 0;
    int32_t tok = first_tok;
    int pos = exp_n;
    int total_proposed = 0, total_accepted = 0, passes = 0;
    int exact_match = 1;
    double t_prop = 0.0, t_ver = 0.0, t_acc = 0.0;
    double t_spec0 = now();
    while (spec_count < MAX_GEN_TOKENS && !eremite_is_stop_token(m, tok)) {
      int32_t vrows[MAX_VERIFY_ROWS];
      vrows[0] = tok;
      int nd = draft;
      if (spec_count + 1 + nd > MAX_GEN_TOKENS)
        nd = MAX_GEN_TOKENS - spec_count - 1;
      if (nd <= 0) {
        spec_toks[spec_count++] = tok;
        break;
      }
      double ts0 = now();
      const double *q =
          eremite_propose_drafts(m, sm, tok, pos, vrows + 1, nd, greedy);
      double ts1 = now();
      const float *vl = eremite_verify(m, sm, vrows, nd + 1, pos);
      double ts2 = now();
      if (!vl) break;
      int accepted = 0;
      int32_t next = eremite_sample_verify_drafts(m, vl, vrows + 1, nd, q,
                                                  greedy, &accepted);
      eremite_accept(m, sm, vrows, accepted + 1, pos);
      double ts3 = now();
      t_prop += ts1 - ts0;
      t_ver += ts2 - ts1;
      t_acc += ts3 - ts2;
      total_proposed += nd;
      total_accepted += accepted;
      passes++;
      for (int k = 0; k <= accepted && spec_count < MAX_GEN_TOKENS; k++)
        spec_toks[spec_count++] = vrows[k];
      pos += accepted + 1;
      tok = next;
    }
    double spec_s = now() - t_spec0;
    if (spec_count != gen_count || tok != cur_tok) {
      exact_match = 0;
    } else {
      for (int i = 0; i < gen_count; i++) {
        if (spec_toks[i] != gen_toks[i]) {
          exact_match = 0;
          break;
        }
      }
    }
    double accept_pct =
        total_proposed > 0 ? 100.0 * total_accepted / total_proposed : 0.0;
    double tok_per_pass = passes > 0 ? (double)spec_count / passes : 0.0;
    double spec_tps = spec_s > 0.0 ? (double)spec_count / spec_s : 0.0;
    double base_tps = dec_s > 0.0 ? (double)gen_count / dec_s : 0.0;
    printf("MTP (draft=%d): %d tokens in %.3fs (%.1f tok/s vs %.1f tok/s "
           "baseline = %.2fx), %d/%d drafts accepted (%.1f%%, %.2f tok/pass) "
           "[propose=%.3fs verify=%.3fs accept=%.3fs]\n",
           draft, spec_count, spec_s, spec_tps, base_tps,
           base_tps > 0.0 ? spec_tps / base_tps : 0.0, total_accepted,
           total_proposed, accept_pct, tok_per_pass, t_prop, t_ver, t_acc);
    checkf(exact_match,
           "MTP speculative decode (draft=%d) matches baseline token-for-token "
           "(%d vs %d tokens, stop=%d vs %d)",
           draft, spec_count, gen_count, tok, cur_tok);
    checkf(total_accepted > 0,
           "MTP draft head achieved non-zero draft acceptance (%d/%d = %.1f%%)",
           total_accepted, total_proposed, accept_pct);
    eremite_sampler_free(greedy);
    eremite_state_free(m, sm);
  }
  eremite_snapshot_free(m, prompt_end_snap);

  eremite_refresh_stats();
  eremite_stats stats = {0};
  eremite_read_stats(m, &stats);
  printf("TRANSCRIPT (%d tokens, %.3fs, %.1f tok/s): %s\n", gen_count, dec_s,
         gen_count > 0 && dec_s > 0.0 ? (double)gen_count / dec_s : 0.0, text);
  printf("STATS: dense=%.2f GiB cache=%.2f GiB (%d/%d slots, %d uncached) "
         "decode_hits=%llu decode_misses=%llu ssd=%.2f GiB read_s=%.3fs\n",
         (double)stats.dense_bytes / (1024.0 * 1024.0 * 1024.0),
         (double)stats.cache_bytes / (1024.0 * 1024.0 * 1024.0), stats.slots,
         stats.experts, stats.uncached, (unsigned long long)stats.decode.hits,
         (unsigned long long)stats.decode.misses,
         (double)stats.cache.bytes / (1024.0 * 1024.0 * 1024.0),
         stats.cache.read_s);

  checkf(gen_count > 0 && eremite_is_stop_token(m, cur_tok),
         "decoded non-empty transcript ending cleanly at EOS (tok=%d)",
         cur_tok);
  if (expect) {
    checkf(contains_ci(text, expect), "transcript contains expected text '%s'",
           expect);
  }

  // Verify CTC word-level timestamp alignment through the public engine ABI
  // (reusing cached pre-fusion ASR states from media encoding).
  if (eremite_can_align_audio(m)) {
    eremite_audio_word *words = NULL;
    int n_words = 0;
    double clip_dur = 0.0;
    char align_err[256] = {0};
    double t_al0 = now();
    int al_ok =
        eremite_align_audio(m, audio_bytes, audio_len, text, text_len, &words,
                            &n_words, &clip_dur, align_err, sizeof align_err);
    double al_s = now() - t_al0;
    int mono = 1;
    for (int i = 0; i < n_words; i++) {
      if (words[i].byte_start < 0 || words[i].byte_end <= words[i].byte_start ||
          (size_t)words[i].byte_end > text_len || words[i].start < 0.0 ||
          words[i].end < words[i].start || words[i].end > clip_dur + 1e-6) {
        mono = 0;
      }
      if (i > 0 && words[i].start < words[i - 1].end - 1e-9) mono = 0;
    }
    checkf(al_ok == 1 && n_words > 0 && mono && clip_dur > 0.0,
           "eremite_align_audio aligned %d words over %.2fs in %.4fs (err=%s)",
           n_words, clip_dur, al_s, align_err);
    eremite_audio_words_free(words);
  }

  free(audio_bytes);
  eremite_media_close(media);
  eremite_state_free(m, st);
  free(raw_toks);
  free(exp_toks);
  return failures != 0;
}
