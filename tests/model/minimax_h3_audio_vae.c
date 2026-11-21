// Tests MiniMax-H3 Audio VAE (BigVGAN decoder + DAC encoder) on a real
// safetensors checkpoint. Pass minimax_h3_audio_vae_fp32.safetensors as
// argv[1] or EREMITE_MINIMAX_H3_AUDIO_VAE.
#include "base/clock.h"
#include "base/error.h"
#include "check.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/audio_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Synthetic stereo test duration (1 second = 32000 samples = 40 latent frames).
static constexpr int TEST_SECONDS = 1,
                     TEST_SAMPLES = TEST_SECONDS * MINIMAX_H3_AUDIO_SAMPLE_RATE,
                     TEST_LATENT_T =
                         TEST_SECONDS * MINIMAX_H3_AUDIO_LATENT_RATE;

// Left and right test tone frequencies in Hz, amplitude and full cycle 2 * pi.
static constexpr double FREQ_LEFT = 440.0, FREQ_RIGHT = 880.0,
                        TWO_PI = 6.28318530717958647692;
static constexpr float TONE_AMP = 0.4f;

// Conversion constants and minimum reconstruction thresholds.
static constexpr double MIB_BYTES = (double)(1ull << 20), MS_PER_SEC = 1000.0,
                        MIN_PCM_RMS = 0.10, MIN_CHANNEL_DIFF = 0.10,
                        MIN_TONE_CORR = 0.50;

// Computes root-mean-square energy of x[0..n-1].
static double rms_f32(const float *x, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
  return n > 0 ? sqrt(sum / (double)n) : 0.0;
}

// Computes normalized single-frequency DFT magnitude (cosine + sine energy)
// relative to total RMS so phase shift through the codec does not penalize it.
static double tone_purity(const float *x, int n, double freq_hz, int sr) {
  double re = 0.0, im = 0.0, energy = 0.0;
  for (int i = 0; i < n; i++) {
    double ang = TWO_PI * freq_hz * (double)i / (double)sr;
    double v = (double)x[i];
    re += v * cos(ang);
    im += v * sin(ang);
    energy += v * v;
  }
  if (energy <= 1e-12) return 0.0;
  double tone_energy = 2.0 * (re * re + im * im) / (double)n;
  return sqrt(tone_energy / energy);
}

// Runs Audio VAE weight loading, encode->decode round-trip and latent decode.
int main(int argc, char **argv) {
  const char *weights =
      argc > 1 ? argv[1] : getenv("EREMITE_MINIMAX_H3_AUDIO_VAE");
  if (!weights) {
    printf("usage: %s <minimax_h3_audio_vae_fp32.safetensors>\n", argv[0]);
    return 1;
  }

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-audio-vae-test", 0);
  size_t source_bytes = 0;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  MinimaxH3AudioVae *vae = minimax_h3_audio_vae_open(gpu, &account, weights);
  check(vae != NULL, "audio VAE opened");
  printf("required decoder: %.2f MiB, encoder: %.2f MiB\n",
         (double)minimax_h3_audio_vae_required_bytes(vae) / MIB_BYTES,
         (double)minimax_h3_audio_vae_encoder_required_bytes(vae) / MIB_BYTES);

  double t0 = now();
  check(minimax_h3_audio_vae_load(vae) == 1, "decoder weights loaded");
  check(minimax_h3_audio_vae_load_encoder(vae) == 1, "encoder weights loaded");
  printf("loaded both groups in %.1f ms (dec %.2f MiB, enc %.2f MiB)\n",
         (now() - t0) * MS_PER_SEC,
         (double)minimax_h3_audio_vae_resident_bytes(vae) / MIB_BYTES,
         (double)minimax_h3_audio_vae_encoder_resident_bytes(vae) / MIB_BYTES);

  // Build 1 s stereo test tone (440 Hz left, 880 Hz right).
  float *pcm_in = xmalloc((size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                          (size_t)TEST_SAMPLES * sizeof(float));
  for (int i = 0; i < TEST_SAMPLES; i++) {
    double t = (double)i / (double)MINIMAX_H3_AUDIO_SAMPLE_RATE;
    pcm_in[i] = TONE_AMP * (float)sin(TWO_PI * FREQ_LEFT * t);
    pcm_in[TEST_SAMPLES + i] = TONE_AMP * (float)sin(TWO_PI * FREQ_RIGHT * t);
  }

  int lat_t = minimax_h3_audio_vae_encoded_frames(TEST_SAMPLES);
  check(lat_t == TEST_LATENT_T, "encoded frame count matches 40 Hz rate");
  size_t lat_elems = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                     (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS * (size_t)lat_t;
  float *latents = xmalloc(lat_elems * sizeof(float));

  t0 = now();
  int enc_ok =
      minimax_h3_audio_vae_encode(vae, pcm_in, TEST_SAMPLES, latents, NULL);
  double enc_ms = (now() - t0) * MS_PER_SEC;
  check(enc_ok == 1, "stereo audio encoded");
  int lat_finite = 1;
  for (size_t i = 0; i < lat_elems; i++)
    if (!isfinite(latents[i])) lat_finite = 0;
  double lat_rms = rms_f32(latents, lat_elems);
  printf("encode %.1f ms: latent rms=%.4f z[0..3]=%.5f,%.5f,%.5f,%.5f\n",
         enc_ms, lat_rms, latents[0], latents[1], latents[2], latents[3]);
  check(lat_finite && lat_rms > 0.1, "encoded latents finite and non-trivial");

  float *pcm_out = xmalloc((size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                           (size_t)TEST_SAMPLES * sizeof(float));
  t0 = now();
  int dec_ok = minimax_h3_audio_vae_decode(vae, latents, lat_t, pcm_out, NULL);
  double dec_ms = (now() - t0) * MS_PER_SEC;
  check(dec_ok == 1, "stereo audio decoded");

  int pcm_ok = 1;
  for (size_t i = 0;
       i < (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS * (size_t)TEST_SAMPLES;
       i++) {
    if (!isfinite(pcm_out[i]) || pcm_out[i] < -1.0f || pcm_out[i] > 1.0f)
      pcm_ok = 0;
  }
  check(pcm_ok, "decoded PCM finite and clamped to [-1, 1]");

  double rms_l = rms_f32(pcm_out, (size_t)TEST_SAMPLES);
  double rms_r = rms_f32(pcm_out + TEST_SAMPLES, (size_t)TEST_SAMPLES);
  double pur_l_440 = tone_purity(pcm_out, TEST_SAMPLES, FREQ_LEFT,
                                 MINIMAX_H3_AUDIO_SAMPLE_RATE);
  double pur_l_880 = tone_purity(pcm_out, TEST_SAMPLES, FREQ_RIGHT,
                                 MINIMAX_H3_AUDIO_SAMPLE_RATE);
  double pur_r_880 = tone_purity(pcm_out + TEST_SAMPLES, TEST_SAMPLES,
                                 FREQ_RIGHT, MINIMAX_H3_AUDIO_SAMPLE_RATE);
  double pur_r_440 = tone_purity(pcm_out + TEST_SAMPLES, TEST_SAMPLES,
                                 FREQ_LEFT, MINIMAX_H3_AUDIO_SAMPLE_RATE);
  double diff_sum = 0.0;
  for (int i = 0; i < TEST_SAMPLES; i++) {
    double d = (double)pcm_out[i] - (double)pcm_out[TEST_SAMPLES + i];
    diff_sum += d * d;
  }
  double diff_rms = sqrt(diff_sum / (double)TEST_SAMPLES);
  printf("decode %.1f ms: rms_l=%.4f rms_r=%.4f diff_rms=%.4f "
         "L(440=%.3f,880=%.3f) R(880=%.3f,440=%.3f) "
         "pcm[0..3]=%.5f,%.5f,%.5f,%.5f\n",
         dec_ms, rms_l, rms_r, diff_rms, pur_l_440, pur_l_880, pur_r_880,
         pur_r_440, pcm_out[0], pcm_out[1], pcm_out[2], pcm_out[3]);

  check(rms_l > MIN_PCM_RMS && rms_r > MIN_PCM_RMS,
        "decoded stereo channels carry signal");
  check(diff_rms > MIN_CHANNEL_DIFF, "decoded stereo channels stay distinct");
  check(pur_l_440 > MIN_TONE_CORR && pur_l_440 > pur_l_880 &&
            pur_r_880 > MIN_TONE_CORR && pur_r_880 > pur_r_440,
        "decoded tones preserve 440 Hz left and 880 Hz right frequencies");

  free(pcm_out);
  free(latents);
  free(pcm_in);
  minimax_h3_audio_vae_close(vae);
  return failures != 0;
}
