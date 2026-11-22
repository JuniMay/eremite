#include "check.h"
#include "base/clock.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/nemotron/audio.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// Formats a check description and records the result.
static void checkf(int ok, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void checkf(int ok, const char *fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  check(ok, buf);
}

// Live checkpoint test for the Nemotron-3.5-Transcribe Parallel Expert
// Perception Encoder (perception.safetensors):
// 1. Loads perception.safetensors and verifies single-pass encoding (5 s and
//    30 s audio) produces finite, non-degenerate, deterministic embeddings.
// 2. Verifies long-form windowed + streaming Sortformer encoding (45 s audio)
//    produces finite, non-degenerate embeddings matching expected token count.
// 3. Reports measured perception latency and audio real-time factor.
//
// usage: model-nemotron_audio CHECKPOINT

// Synthetic speech-like multi-tone waveform parameters and test durations.
static constexpr double TWO_PI = 6.2831853071795864769;
static constexpr int SHORT_SECONDS = 5;
static constexpr int MEDIUM_SECONDS = 30;
static constexpr int LONG_SECONDS = 45;
static constexpr float MIN_RMS = 1e-3f;
static constexpr float MAX_ABS_LIMIT = 1e4f;
static constexpr float DETERMINISM_TOL = 1e-5f;

// Generates a deterministic modulated multi-tone test waveform of seconds at
// 16 kHz.
static float *make_speech_like_pcm(int seconds, int *out_samples) {
  int n = seconds * NEMO_AUDIO_SAMPLE_RATE;
  float *pcm = malloc((size_t)n * sizeof(float));
  checkf(pcm != NULL, "allocated %ds PCM buffer", seconds);
  for (int i = 0; i < n; i++) {
    double t = (double)i / (double)NEMO_AUDIO_SAMPLE_RATE;
    double env = 0.5 + 0.5 * sin(TWO_PI * 2.5 * t);
    double sig =
        0.35 * sin(TWO_PI * (180.0 + 40.0 * sin(TWO_PI * 1.1 * t)) * t) +
        0.25 * sin(TWO_PI * 620.0 * t) + 0.15 * sin(TWO_PI * 1850.0 * t);
    pcm[i] = (float)(env * sig);
  }
  *out_samples = n;
  return pcm;
}

// Verifies all values in x[0..n-1] are finite, bounded and non-degenerate, and
// returns the RMS value.
static float check_embeddings(const float *x, size_t n, const char *label) {
  double sq = 0.0;
  float max_abs = 0.0f;
  int all_finite = 1;
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    if (!isfinite(v)) {
      all_finite = 0;
      break;
    }
    float a = fabsf(v);
    if (a > max_abs) max_abs = a;
    sq += (double)v * (double)v;
  }
  float rms = (float)sqrt(sq / (double)n);
  checkf(all_finite && rms > MIN_RMS && max_abs < MAX_ABS_LIMIT,
         "%s embeddings finite and well-scaled (rms=%.4f max=%.3f)", label, rms,
         max_abs);
  return rms;
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : getenv("EREMITE_CHECKPOINT");
  if (!dir || !dir[0]) {
    fprintf(stderr, "usage: %s CHECKPOINT\n", argv[0]);
    return 2;
  }

  Memory *mem = memory_new();
  MemoryAccount account;
  MemorySlot work;
  memory_account_init(mem, &account, dir, UINT64_MAX);
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "perception work", NULL,
                   NULL);
  work.automatic = 1;

  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *gpu = gpu_new(src, len);

  double t0 = now();
  NemoAudioEncoder *enc = nemo_audio_open(gpu, &account, &work, dir);
  checkf(enc != NULL, "opened perception encoder (%.1f MiB weights)",
         (double)nemo_audio_weight_bytes(enc) / (1024.0 * 1024.0));
  nemo_audio_pin(enc, 1);

  // 1. Short 5 s clip (single-pass) + determinism check.
  int n5 = 0;
  float *pcm5 = make_speech_like_pcm(SHORT_SECONDS, &n5);
  float *emb5a = NULL, *emb5b = NULL;
  int tok5a = 0, tok5b = 0;
  char err[256] = {0};
  double s0 = now();
  checkf(nemo_audio_encode(enc, pcm5, n5, NULL, &emb5a, &tok5a, err,
                           sizeof err) == 1,
         "encoded 5s clip (%d tokens, %.3fs incl load, err=%s)", tok5a,
         now() - s0, err);
  checkf(tok5a == nemo_audio_token_count(n5), "5s token count exact (%d)",
         tok5a);
  check_embeddings(emb5a, (size_t)tok5a * NEMO_AUDIO_OUT_DIM, "5s");

  double s1 = now();
  checkf(nemo_audio_encode(enc, pcm5, n5, NULL, &emb5b, &tok5b, err,
                           sizeof err) == 1,
         "re-encoded 5s clip warm (%.3fs, %.1fx realtime)", now() - s1,
         (double)SHORT_SECONDS / (now() - s1));
  float max_det = 0.0f;
  for (size_t i = 0; i < (size_t)tok5a * NEMO_AUDIO_OUT_DIM; i++) {
    float d = fabsf(emb5a[i] - emb5b[i]);
    if (d > max_det) max_det = d;
  }
  checkf(tok5a == tok5b && max_det <= DETERMINISM_TOL,
         "5s encoding deterministic (max_diff=%.2e)", max_det);
  free(emb5b);
  free(emb5a);
  free(pcm5);

  // 2. Medium 30 s clip (single-pass, 375 tokens).
  int n30 = 0;
  float *pcm30 = make_speech_like_pcm(MEDIUM_SECONDS, &n30);
  float *emb30 = NULL;
  int tok30 = 0;
  double s30 = now();
  checkf(nemo_audio_encode(enc, pcm30, n30, NULL, &emb30, &tok30, err,
                           sizeof err) == 1,
         "encoded 30s clip (%d tokens, %.3fs, %.1fx realtime)", tok30,
         now() - s30, (double)MEDIUM_SECONDS / (now() - s30));
  checkf(tok30 == 375, "30s token count is 375 (got %d)", tok30);
  check_embeddings(emb30, (size_t)tok30 * NEMO_AUDIO_OUT_DIM, "30s");
  free(emb30);
  free(pcm30);

  // 3. Long 45 s clip (windowed ASR + streaming Sortformer speaker cache).
  int n45 = 0;
  float *pcm45 = make_speech_like_pcm(LONG_SECONDS, &n45);
  float *emb45 = NULL;
  int tok45 = 0;
  double s45 = now();
  checkf(nemo_audio_encode(enc, pcm45, n45, NULL, &emb45, &tok45, err,
                           sizeof err) == 1,
         "encoded 45s long-form clip (%d tokens, %.3fs, %.1fx realtime)", tok45,
         now() - s45, (double)LONG_SECONDS / (now() - s45));
  checkf(tok45 == nemo_audio_token_count(n45), "45s token count exact (%d)",
         tok45);
  check_embeddings(emb45, (size_t)tok45 * NEMO_AUDIO_OUT_DIM, "45s");
  free(emb45);
  free(pcm45);

  printf("total perception test time: %.2fs\n", now() - t0);
  nemo_audio_pin(enc, 0);
  nemo_audio_close(enc);
  return failures != 0;
}
