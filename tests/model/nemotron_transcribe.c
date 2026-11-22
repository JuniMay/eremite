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
// usage: model-nemotron_transcribe CHECKPOINT AUDIO_FILE
//        [--memory-gb N] [--expect SUBSTRING]

static constexpr int MAX_PROMPT_TOKENS = 65536;
static constexpr int MAX_GEN_TOKENS = 256;
static constexpr int MODEL_CTX = 8192;
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
            "[--memory-gb N] [--expect SUBSTRING]\n",
            argv[0]);
    return 2;
  }
  const char *dir = argv[1];
  const char *audio_path = NULL;
  const char *say_text = NULL;
  double memory_gb = 0.0;
  const char *expect = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--memory-gb") && i + 1 < argc) {
      memory_gb = atof(argv[++i]);
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

  eremite_config cfg = {.abi = EREMITE_ABI, .model = dir, .ctx = MODEL_CTX};
  eremite_model *m = eremite_open(&cfg);
  checkf(m != NULL, "opened Nemotron-3.5-Transcribe model");

  double t_load0 = now();
  checkf(eremite_load(m) == 1, "loaded model weights (%.2fs)", now() - t_load0);
  checkf(eremite_vocab_size(m) == 131072, "vocab_size is 131072 (got %d)",
         eremite_vocab_size(m));
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
  free(audio_bytes);

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

  // Greedy decode full transcript from st.
  char text[TRANSCRIPT_CAP];
  size_t text_len = 0;
  text[0] = '\0';
  int32_t cur_tok = first_tok;
  int32_t cur_pos = exp_n;
  int gen_count = 0;
  double t_dec0 = now();
  while (gen_count < MAX_GEN_TOKENS && !eremite_is_stop_token(m, cur_tok)) {
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

  eremite_media_close(media);
  eremite_state_free(m, st);
  free(raw_toks);
  free(exp_toks);
  return failures != 0;
}
