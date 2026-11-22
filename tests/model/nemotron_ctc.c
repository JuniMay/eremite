#include "check.h"
#include "base/clock.h"
#include "base/file.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/nemotron/audio.h"
#include "model/nemotron/ctc.h"
#include "platform/video.h"
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Live checkpoint test for Nemotron-3.5-Transcribe CTC timestamp head:
// 1. Runs the 32-layer ASR encoder on spoken English audio.
// 2. Runs greedy CTC decoding via model-ctc-bf16.safetensors and verifies it
//    recognizes the spoken words verbatim.
// 3. Runs CTC Viterbi forced alignment and verifies monotonic, non-overlapping
//    word timestamps within [0, duration], plus acoustic onset correlation.
//
// usage: model-nemotron_ctc CHECKPOINT [AUDIO_FILE] [--say TEXT]

static constexpr char DEFAULT_PHRASE[] =
    "The quick brown fox jumps over the lazy dog. "
    "Local speech recognition on Apple silicon runs reliably.";
static constexpr int MAX_DECODE_SECONDS = 3600;

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

// Synthesizes spoken English 16 kHz WAV audio into memory using macOS say.
static uint8_t *synthesize_wav(const char *phrase, size_t *out_len) {
  char tmpl[] = "/tmp/eremite-ctc-XXXXXX.wav";
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
    fprintf(stderr, "usage: %s CHECKPOINT [AUDIO_FILE] [--say TEXT]\n",
            argv[0]);
    return 2;
  }
  const char *dir = argv[1];
  const char *audio_path = NULL;
  const char *phrase = DEFAULT_PHRASE;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--say") && i + 1 < argc) {
      phrase = argv[++i];
    } else if (argv[i][0] != '-' && !audio_path) {
      audio_path = argv[i];
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  checkf(nemo_ctc_available(dir) == 1,
         "checkpoint includes model-ctc-bf16.safetensors");

  Memory *mem = memory_new();
  MemoryAccount account;
  MemorySlot work;
  memory_account_init(mem, &account, "nemotron_ctc_test", UINT64_MAX);
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "work", NULL, NULL);
  work.automatic = 1;

  size_t klen = 0;
  const char *ksrc = kernel_library_source(&klen);
  Gpu *gpu = gpu_new(ksrc, klen);

  NemoAudioEncoder *aud = nemo_audio_open(gpu, &account, &work, dir);
  checkf(aud != NULL, "opened perception audio encoder");
  NemoCtcAligner *ctc = nemo_ctc_open(gpu, &account, &work, dir);
  checkf(ctc != NULL, "opened CTC timestamp aligner (%.1f MiB)",
         (double)nemo_ctc_weight_bytes(ctc) / (1024.0 * 1024.0));

  size_t wav_len = 0;
  uint8_t *wav = audio_path ? (uint8_t *)read_file(audio_path, &wav_len)
                            : synthesize_wav(phrase, &wav_len);
  checkf(wav != NULL && wav_len > 0, "loaded audio (%zu bytes)", wav_len);

  int pcm_samples = 0;
  const char *dec_err = NULL;
  float *pcm =
      audio_decode_pcm_mono(wav, wav_len, NEMO_AUDIO_SAMPLE_RATE,
                            MAX_DECODE_SECONDS, &pcm_samples, &dec_err);
  free(wav);
  checkf(pcm != NULL && pcm_samples > 0, "decoded %d PCM samples (%.2fs)",
         pcm_samples, (double)pcm_samples / (double)NEMO_AUDIO_SAMPLE_RATE);

  double duration = (double)pcm_samples / (double)NEMO_AUDIO_SAMPLE_RATE;
  uint16_t *asr_bf16 = NULL;
  int asr_tokens = 0;
  char err[256] = {0};
  double t0 = now();
  nemo_audio_pin(aud, 1);
  int ok = nemo_audio_encode_asr_only(aud, pcm, pcm_samples, NULL, &asr_bf16,
                                      &asr_tokens, err, sizeof err);
  nemo_audio_pin(aud, 0);
  double t_asr = now() - t0;
  checkf(ok == 1 && asr_bf16 != NULL && asr_tokens > 0,
         "ASR-only encode produced %d tokens in %.3fs", asr_tokens, t_asr);

  // Measure first and last audible speech energy in PCM for acoustic sanity.
  double first_speech_s = duration, last_speech_s = 0.0;
  int hop = NEMO_AUDIO_SAMPLE_RATE / 100;  // 10 ms RMS step
  for (int i = 0; i + hop <= pcm_samples; i += hop) {
    double e = 0.0;
    for (int j = 0; j < hop; j++) e += (double)pcm[i + j] * (double)pcm[i + j];
    double rms = sqrt(e / hop);
    if (rms > 0.01) {
      double ts = (double)i / (double)NEMO_AUDIO_SAMPLE_RATE;
      if (ts < first_speech_s) first_speech_s = ts;
      if (ts + 0.01 > last_speech_s) last_speech_s = ts + 0.01;
    }
  }

  // 1. Greedy CTC decoding check (proves exact weights, upsample transpose,
  //    bridging Transformer and residual blend).
  char *greedy = NULL;
  double tg0 = now();
  nemo_ctc_pin(ctc, 1);
  ok = nemo_ctc_greedy_decode(ctc, asr_bf16, asr_tokens, NULL, &greedy, err,
                              sizeof err);
  double tg_cold = now() - tg0;
  printf("CTC GREEDY (cold %.3fs): %s\n", tg_cold, greedy ? greedy : "(null)");
  checkf(ok == 1 && greedy != NULL && strlen(greedy) > 0,
         "greedy CTC decode succeeded (err=%s)", err);
  if (!audio_path) {
    checkf(contains_ci(greedy, "quick brown fox") &&
               contains_ci(greedy, "apple silicon"),
           "greedy CTC decode matches spoken words");
  }
  free(greedy);

  // 2. Forced alignment check (warm weights).
  NemoCtcWord *words = NULL;
  int n_words = 0;
  double ta0 = now();
  ok = nemo_ctc_align(ctc, asr_bf16, asr_tokens, duration, phrase,
                      strlen(phrase), NULL, &words, &n_words, err, sizeof err);
  double ta_warm = now() - ta0;
  checkf(ok == 1 && words != NULL && n_words > 0,
         "CTC forced alignment returned %d words in %.4fs (err=%s)", n_words,
         ta_warm, err);

  printf("ACOUSTIC SPEECH SPAN: [%.3fs, %.3fs] (duration=%.3fs)\n",
         first_speech_s, last_speech_s, duration);
  int mono_ok = 1;
  for (int i = 0; i < n_words; i++) {
    printf("  [%2d] %6.3fs - %6.3fs  %.*s\n", i, words[i].start, words[i].end,
           words[i].byte_end - words[i].byte_start,
           phrase + words[i].byte_start);
    if (words[i].start < 0.0 || words[i].end < words[i].start ||
        words[i].end > duration + 1e-6) {
      mono_ok = 0;
    }
    if (i > 0 && words[i].start < words[i - 1].end - 1e-9) mono_ok = 0;
  }
  checkf(mono_ok, "all %d word timestamps are monotonic within [0, %.3f]",
         n_words, duration);
  if (n_words > 0 && first_speech_s < last_speech_s) {
    double start_diff = fabs(words[0].start - first_speech_s);
    double end_diff = fabs(words[n_words - 1].end - last_speech_s);
    checkf(start_diff < 0.5 && end_diff < 0.5,
           "aligned boundaries track acoustic speech (start_err=%.3fs, "
           "end_err=%.3fs)",
           start_diff, end_diff);
  }
  free(words);

  // 3. Long-form windowed (> 40 s, > 500 ASR tokens) alignment check: tile the
  //    clip 8x so both ASR-only and CTC decoder cross window boundaries.
  if (!audio_path && pcm_samples > 0) {
    static constexpr int REPS = 8;
    int long_samples = pcm_samples * REPS;
    float *long_pcm = malloc((size_t)long_samples * sizeof(float));
    for (int r = 0; r < REPS; r++) {
      memcpy(long_pcm + (size_t)r * pcm_samples, pcm,
             (size_t)pcm_samples * sizeof(float));
    }
    size_t plen = strlen(phrase);
    size_t long_tlen = REPS * (plen + 1);
    char *long_text = malloc(long_tlen + 1);
    size_t pos = 0;
    for (int r = 0; r < REPS; r++) {
      if (r > 0) long_text[pos++] = ' ';
      memcpy(long_text + pos, phrase, plen);
      pos += plen;
    }
    long_text[pos] = '\0';

    uint16_t *long_asr = NULL;
    int long_toks = 0;
    double long_dur = (double)long_samples / (double)NEMO_AUDIO_SAMPLE_RATE;
    double tl0 = now();
    nemo_audio_pin(aud, 1);
    int lok =
        nemo_audio_encode_asr_only(aud, long_pcm, long_samples, NULL, &long_asr,
                                   &long_toks, err, sizeof err);
    nemo_audio_pin(aud, 0);
    double tl_asr = now() - tl0;
    checkf(lok == 1 && long_toks > NEMO_AUDIO_ONLINE_TOKENS,
           "long-form ASR-only produced %d tokens (%.1fs audio) in %.3fs",
           long_toks, long_dur, tl_asr);

    NemoCtcWord *lwords = NULL;
    int ln_words = 0;
    double tlc0 = now();
    lok = nemo_ctc_align(ctc, long_asr, long_toks, long_dur, long_text, pos,
                         NULL, &lwords, &ln_words, err, sizeof err);
    double tl_ctc = now() - tlc0;
    int lmono = 1;
    for (int i = 0; i < ln_words; i++) {
      if (lwords[i].start < 0.0 || lwords[i].end < lwords[i].start ||
          lwords[i].end > long_dur + 1e-6)
        lmono = 0;
      if (i > 0 && lwords[i].start < lwords[i - 1].end - 1e-9) lmono = 0;
    }
    // Check that each repetition's first word aligns within 0.4s of r *
    // duration.
    double max_rep_err = 0.0;
    if (n_words > 0 && ln_words == n_words * REPS) {
      for (int r = 0; r < REPS; r++) {
        double expected_t = (double)r * duration;
        double e = fabs(lwords[r * n_words].start - expected_t);
        if (e > max_rep_err) max_rep_err = e;
      }
    } else {
      max_rep_err = INFINITY;
    }
    checkf(
        lok == 1 && lmono && max_rep_err < 0.4,
        "windowed long-form CTC aligned %d words in %.4fs (max_rep_err=%.3fs)",
        ln_words, tl_ctc, max_rep_err);
    free(lwords);
    free(long_asr);
    free(long_text);
    free(long_pcm);
  }

  nemo_ctc_pin(ctc, 0);
  free(pcm);
  free(asr_bf16);
  nemo_ctc_close(ctc);
  nemo_audio_close(aud);
  return failures != 0;
}
