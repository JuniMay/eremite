#include "check.h"
#include "model/nemotron/audio.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

// Unit tests for CPU-side Nemotron-3.5-Transcribe audio frontend and
// Sortformer streaming speaker-cache compression.

// Constants matching src/model/nemotron/audio.c frontend math.
static constexpr double TWO_PI = 6.2831853071795864769;
static constexpr float PREEMPH = 0.97f;
static constexpr float LOG_GUARD = 5.9604644775390625e-8f;
static constexpr float NORM_EPS = 1e-5f;
static constexpr float TOL_MEL = 1e-4f;
static constexpr float TOL_NORM = 1e-5f;
static constexpr float TOL_SIN = 1e-2f;
static constexpr int STFT_PAD = NEMO_AUDIO_N_FFT / 2;
static constexpr int WIN_OFFSET = (NEMO_AUDIO_N_FFT - NEMO_AUDIO_WIN_LEN) / 2;

// Computes an independent slow DFT reference for one frame's log-mel bins.
static void ref_mel_frame(const float *padded, int frame, const float *window,
                          const float *fb, float *out_mels) {
  float x[NEMO_AUDIO_N_FFT] = {0};
  const float *base = padded + (size_t)frame * NEMO_AUDIO_HOP_LEN;
  for (int i = 0; i < NEMO_AUDIO_WIN_LEN; i++)
    x[WIN_OFFSET + i] = base[WIN_OFFSET + i] * window[i];

  double power[NEMO_AUDIO_N_BINS];
  for (int k = 0; k < NEMO_AUDIO_N_BINS; k++) {
    double re = 0.0, im = 0.0;
    for (int n = 0; n < NEMO_AUDIO_N_FFT; n++) {
      double ang = -TWO_PI * (double)k * (double)n / (double)NEMO_AUDIO_N_FFT;
      re += (double)x[n] * cos(ang);
      im += (double)x[n] * sin(ang);
    }
    power[k] = re * re + im * im;
  }
  for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
    double e = 0.0;
    for (int k = 0; k < NEMO_AUDIO_N_BINS; k++)
      e += (double)fb[m * NEMO_AUDIO_N_BINS + k] * power[k];
    out_mels[m] = (float)log(e + (double)LOG_GUARD);
  }
}

// Verifies mel frame counts, FFT log-mel accuracy against direct DFT,
// per-feature normalization and 8x FeatureStacking padding.
static void test_frontend(void) {
  check(nemo_audio_mel_frames(0) == 0 && nemo_audio_token_count(0) == 0,
        "zero samples produce zero frames");
  check(nemo_audio_mel_frames(159) == 0 && nemo_audio_token_count(159) == 0,
        "sub-hop samples produce zero frames");
  check(nemo_audio_mel_frames(160) == 1 && nemo_audio_token_count(160) == 1,
        "one hop produces one frame and one token");
  check(nemo_audio_mel_frames(16000) == 100 &&
            nemo_audio_token_count(16000) == 13,
        "1 s produces 100 mel frames and 13 tokens");
  check(nemo_audio_mel_frames(480000) == 3000 &&
            nemo_audio_token_count(480000) == 375,
        "30 s produces 3000 mel frames and 375 tokens");

  // Build deterministic Hann window, sparse mel filterbank and 0.1 s signal.
  static constexpr int SAMPLES = 1600;
  float pcm[SAMPLES];
  for (int i = 0; i < SAMPLES; i++) {
    double t = (double)i / (double)NEMO_AUDIO_SAMPLE_RATE;
    pcm[i] = 0.4f * (float)sin(TWO_PI * 440.0 * t) +
             0.25f * (float)cos(TWO_PI * 1320.0 * t);
  }
  float window[NEMO_AUDIO_WIN_LEN];
  for (int i = 0; i < NEMO_AUDIO_WIN_LEN; i++)
    window[i] = 0.5f * (1.0f - (float)cos(TWO_PI * (double)i /
                                          (double)(NEMO_AUDIO_WIN_LEN - 1)));
  float *fb =
      calloc((size_t)NEMO_AUDIO_N_MELS * NEMO_AUDIO_N_BINS, sizeof(float));
  for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
    int bin = 1 + (m * (NEMO_AUDIO_N_BINS - 2)) / NEMO_AUDIO_N_MELS;
    fb[m * NEMO_AUDIO_N_BINS + bin] = 0.5f;
    fb[m * NEMO_AUDIO_N_BINS + bin + 1] = 0.5f;
  }

  int frames = nemo_audio_mel_frames(SAMPLES);
  float *mel = calloc((size_t)frames * NEMO_AUDIO_N_MELS, sizeof(float));
  nemo_audio_mel_spectrogram(pcm, SAMPLES, window, fb, mel);

  float padded[SAMPLES + 2 * STFT_PAD];
  memset(padded, 0, sizeof padded);
  padded[STFT_PAD] = pcm[0];
  for (int i = 1; i < SAMPLES; i++)
    padded[STFT_PAD + i] = pcm[i] - PREEMPH * pcm[i - 1];

  float max_err = 0.0f;
  for (int f = 0; f < frames; f += 3) {
    float ref[NEMO_AUDIO_N_MELS];
    ref_mel_frame(padded, f, window, fb, ref);
    for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
      float err = fabsf(mel[f * NEMO_AUDIO_N_MELS + m] - ref[m]);
      if (err > max_err) max_err = err;
    }
  }
  checkf(max_err < TOL_MEL, "log-mel FFT matches direct DFT (max=%.2e)",
         max_err);

  // Per-feature normalization: mean ~0, std ~1 on varying features.
  float *norm = calloc((size_t)frames * NEMO_AUDIO_N_MELS, sizeof(float));
  nemo_audio_normalize_per_feature(mel, frames, norm);
  float norm_mean_err = 0.0f;
  for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
    double s = 0.0;
    for (int f = 0; f < frames; f++) s += norm[f * NEMO_AUDIO_N_MELS + m];
    float me = (float)fabs(s / (double)frames);
    if (me > norm_mean_err) norm_mean_err = me;
  }
  checkf(norm_mean_err < TOL_NORM, "per-feature norm zero mean (max=%.2e)",
         norm_mean_err);

  // Constant feature must normalize to exact zero (no NaN).
  float const_mel[4 * NEMO_AUDIO_N_MELS];
  float const_norm[4 * NEMO_AUDIO_N_MELS];
  for (int i = 0; i < 4 * NEMO_AUDIO_N_MELS; i++) const_mel[i] = -7.25f;
  nemo_audio_normalize_per_feature(const_mel, 4, const_norm);
  int const_ok = 1;
  for (int i = 0; i < 4 * NEMO_AUDIO_N_MELS; i++)
    if (!isfinite(const_norm[i]) || fabsf(const_norm[i]) > NORM_EPS)
      const_ok = 0;
  check(const_ok, "constant mel normalizes cleanly without NaN");

  // FeatureStacking: 10 frames -> 2 stacked tokens, tail 6*128 zeroed.
  float *stacked = calloc(2u * NEMO_AUDIO_STACK_DIM, sizeof(float));
  nemo_audio_stack_frames(norm, frames, stacked);
  int stack_ok = 1;
  for (int s = 0; s < 8; s++)
    if (memcmp(stacked + s * NEMO_AUDIO_N_MELS, norm + s * NEMO_AUDIO_N_MELS,
               NEMO_AUDIO_N_MELS * sizeof(float)) != 0)
      stack_ok = 0;
  for (int s = 0; s < 2; s++)
    if (memcmp(stacked + NEMO_AUDIO_STACK_DIM + s * NEMO_AUDIO_N_MELS,
               norm + (8 + s) * NEMO_AUDIO_N_MELS,
               NEMO_AUDIO_N_MELS * sizeof(float)) != 0)
      stack_ok = 0;
  for (int s = 2; s < 8; s++)
    for (int c = 0; c < NEMO_AUDIO_N_MELS; c++)
      if (stacked[NEMO_AUDIO_STACK_DIM + s * NEMO_AUDIO_N_MELS + c] != 0.0f)
        stack_ok = 0;
  check(stack_ok, "FeatureStacking packs 8 frames and zero-pads tail");

  free(stacked);
  free(norm);
  free(mel);
  free(fb);
}

// Verifies sinusoidal speaker fusion kernel values at speaker 0 and 1.
static void test_sinusoid_kernel(void) {
  float *k =
      calloc((size_t)NEMO_AUDIO_NUM_SPK * NEMO_AUDIO_ASR_DIM, sizeof(float));
  nemo_audio_build_sinusoid_kernel(k, NEMO_AUDIO_NUM_SPK, NEMO_AUDIO_ASR_DIM);
  int spk0_ok = 1;
  for (int d = 0; d < NEMO_AUDIO_ASR_DIM; d += 2) {
    if (k[d] != 0.0f || k[d + 1] != 1.0f) spk0_ok = 0;
  }
  check(spk0_ok, "sinusoid kernel speaker 0 is (sin 0, cos 0) = (0, 1)");
  check(fabsf(k[NEMO_AUDIO_ASR_DIM + 0] - sinf(1.0f)) < TOL_SIN &&
            fabsf(k[NEMO_AUDIO_ASR_DIM + 1] - cosf(1.0f)) < TOL_SIN,
        "sinusoid kernel speaker 1 col 0..1 matches (sin 1, cos 1)");
  free(k);
}

// Verifies Sortformer speaker-cache compression retains active speaker frames
// in chronological order and fills silence slots with learnable_sil_emb.
static void test_sortformer_compress(void) {
  static constexpr int N_FRAMES = 240;
  static constexpr int EMB_DIM = 8;
  static constexpr int N_SPK = 8;
  static constexpr int CACHE_LEN = 188;
  static constexpr int SIL_PER_SPK = 1;

  float *embs = calloc((size_t)N_FRAMES * EMB_DIM, sizeof(float));
  float *preds = calloc((size_t)N_FRAMES * N_SPK, sizeof(float));
  float sil[EMB_DIM];
  for (int d = 0; d < EMB_DIM; d++) sil[d] = -99.0f;

  // Assign frames 0..99 to speaker 0, 100..199 to speaker 1, 200..239 silence.
  for (int f = 0; f < N_FRAMES; f++) {
    for (int d = 0; d < EMB_DIM; d++) embs[f * EMB_DIM + d] = (float)f;
    if (f < 100)
      preds[f * N_SPK + 0] = 0.95f;
    else if (f < 200)
      preds[f * N_SPK + 1] = 0.92f;
  }

  float *out_e = calloc((size_t)CACHE_LEN * EMB_DIM, sizeof(float));
  float *out_p = calloc((size_t)CACHE_LEN * N_SPK, sizeof(float));
  nemo_audio_sortformer_compress_spkcache(embs, preds, N_FRAMES, EMB_DIM, N_SPK,
                                          CACHE_LEN, SIL_PER_SPK, sil, out_e,
                                          out_p);

  int spk0_cnt = 0, spk1_cnt = 0, sil_cnt = 0, ordered = 1;
  float prev0 = -1.0f, prev1 = -1.0f;
  for (int i = 0; i < CACHE_LEN; i++) {
    float tag = out_e[i * EMB_DIM];
    if (tag == -99.0f) {
      sil_cnt++;
      for (int s = 0; s < N_SPK; s++)
        if (out_p[i * N_SPK + s] != 0.0f) ordered = 0;
    } else if (out_p[i * N_SPK + 0] > 0.5f) {
      spk0_cnt++;
      if (tag <= prev0) ordered = 0;
      prev0 = tag;
    } else if (out_p[i * N_SPK + 1] > 0.5f) {
      spk1_cnt++;
      if (tag <= prev1) ordered = 0;
      prev1 = tag;
    }
  }
  checkf(spk0_cnt > 50 && spk1_cnt > 50 && sil_cnt >= N_SPK,
         "spkcache keeps both speakers and silence markers (%d, %d, %d)",
         spk0_cnt, spk1_cnt, sil_cnt);
  check(ordered, "spkcache preserves per-speaker frame order and zero silence");

  free(out_p);
  free(out_e);
  free(preds);
  free(embs);
}

int main(void) {
  test_frontend();
  test_sinusoid_kernel();
  test_sortformer_compress();
  return failures != 0;
}
