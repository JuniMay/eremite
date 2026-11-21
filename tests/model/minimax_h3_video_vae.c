// Tests MiniMax-H3 Video VAE (36-layer INT8-ConvRot 3D ViT decoder + 3D causal
// CNN encoder) on a real checkpoint. Pass
// minimax_h3_video_vae_int8_convrot.safetensors as argv[1] or
// EREMITE_MINIMAX_H3_VIDEO_VAE.
#include "base/clock.h"
#include "base/error.h"
#include "check.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/video_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Test dimensions (> 256 px to exercise 2x2 spatial tiling and multi-frame
// 22-frame temporal chunking: 22 frames <-> 7 latents).
static constexpr int TEST_W = 320, TEST_H = 320, TEST_FRAMES = 22,
                     TEST_LATENT_T = 7;

// Constants for synthetic animation, timing display and PSNR assertions.
static constexpr double MIB_BYTES = (double)(1ull << 20), MS_PER_SEC = 1000.0,
                        PI = 3.14159265358979323846, MIN_STILL_PSNR = 24.0,
                        MIN_VIDEO_PSNR = 24.0, MOTION_PSNR_MARGIN = 4.0;

// Computes PSNR in dB between two equal-sized 8-bit RGB buffers.
static double rgb_psnr(const uint8_t *a, const uint8_t *b, size_t bytes) {
  double mse = 0.0;
  for (size_t i = 0; i < bytes; i++) {
    double d = (double)a[i] - (double)b[i];
    mse += d * d;
  }
  mse /= (double)bytes;
  if (mse <= 1e-12) return 99.0;
  return 10.0 * log10((255.0 * 255.0) / mse);
}

// Fills frame f of total frames [height, width, 3] with a smooth colorful
// scene containing a disk that moves horizontally across time.
static void fill_video_frame(uint8_t *rgb, int f, int total, int w, int h) {
  double phase = total > 1 ? (double)f / (double)(total - 1) : 0.0;
  double cx = 0.25 + 0.50 * phase;
  double cy = 0.50 + 0.15 * sin(PI * phase);
  for (int y = 0; y < h; y++) {
    double fy = (double)y / (double)h;
    for (int x = 0; x < w; x++) {
      double fx = (double)x / (double)w;
      double dx = fx - cx, dy = fy - cy;
      double spot = exp(-28.0 * (dx * dx + dy * dy));
      double r = 0.15 + 0.35 * fx + 0.50 * spot;
      double g = 0.20 + 0.45 * fy + 0.25 * (1.0 - spot) * sin(PI * fx);
      double b = 0.70 - 0.45 * fy + 0.20 * spot * (1.0 - phase);
      uint8_t *px = rgb + ((size_t)y * (size_t)w + (size_t)x) *
                              (size_t)MINIMAX_H3_RGB_CHANNELS;
      px[0] = (uint8_t)lround(fmin(fmax(r, 0.0), 1.0) * 255.0);
      px[1] = (uint8_t)lround(fmin(fmax(g, 0.0), 1.0) * 255.0);
      px[2] = (uint8_t)lround(fmin(fmax(b, 0.0), 1.0) * 255.0);
    }
  }
}

// Runs Video VAE weight loading, still-image and 22-frame video round-trips.
int main(int argc, char **argv) {
  const char *weights =
      argc > 1 ? argv[1] : getenv("EREMITE_MINIMAX_H3_VIDEO_VAE");
  if (!weights) {
    printf("usage: %s <minimax_h3_video_vae_int8_convrot.safetensors>\n",
           argv[0]);
    return 1;
  }

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-video-vae-test", 0);
  size_t source_bytes = 0;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  MinimaxH3VideoVae *vae = minimax_h3_video_vae_open(gpu, &account, weights);
  check(vae != NULL, "video VAE opened");
  printf("required decoder: %.2f MiB, encoder: %.2f MiB\n",
         (double)minimax_h3_video_vae_required_bytes(vae) / MIB_BYTES,
         (double)minimax_h3_video_vae_encoder_required_bytes(vae) / MIB_BYTES);

  double t0 = now();
  check(minimax_h3_video_vae_load(vae) == 1, "decoder weights loaded");
  check(minimax_h3_video_vae_load_encoder(vae) == 1, "encoder weights loaded");
  printf("loaded both groups in %.1f ms (dec %.2f MiB, enc %.2f MiB)\n",
         (now() - t0) * MS_PER_SEC,
         (double)minimax_h3_video_vae_resident_bytes(vae) / MIB_BYTES,
         (double)minimax_h3_video_vae_encoder_resident_bytes(vae) / MIB_BYTES);

  int h_lat = TEST_H / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = TEST_W / MINIMAX_H3_VAE_SPATIAL_RATIO;
  size_t frame_bytes =
      (size_t)TEST_H * (size_t)TEST_W * (size_t)MINIMAX_H3_RGB_CHANNELS;

  // Test 1: still-frame (T=1) 320x320 tiled round-trip.
  uint8_t *still_in = xmalloc(frame_bytes);
  uint8_t *still_out = xmalloc(frame_bytes);
  float *still_z = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                           (size_t)h_lat * (size_t)w_lat * sizeof(float));
  fill_video_frame(still_in, 0, 1, TEST_W, TEST_H);

  t0 = now();
  check(minimax_h3_video_vae_encode(vae, still_in, 1, TEST_W, TEST_H, still_z,
                                    NULL) == 1,
        "still frame encoded");
  double still_enc_ms = (now() - t0) * MS_PER_SEC;
  t0 = now();
  check(minimax_h3_video_vae_decode(vae, still_z, 1, h_lat, w_lat, still_out,
                                    NULL) == 1,
        "still frame decoded");
  double still_dec_ms = (now() - t0) * MS_PER_SEC;
  double still_psnr = rgb_psnr(still_in, still_out, frame_bytes);
  printf("still 320x320: enc %.1f ms, dec %.1f ms, PSNR %.2f dB\n",
         still_enc_ms, still_dec_ms, still_psnr);
  check(still_psnr >= MIN_STILL_PSNR, "still frame PSNR >= 24 dB");

  // Test 2: 22-frame (t_lat=7) 320x320 moving video round-trip.
  check(minimax_h3_video_vae_encode_latent_t(TEST_FRAMES) == TEST_LATENT_T,
        "22 frames map to 7 latents");
  check(minimax_h3_video_vae_decode_plan(TEST_LATENT_T).out_frames ==
            TEST_FRAMES,
        "7 latents decode back to 22 frames");
  uint8_t *vid_in = xmalloc((size_t)TEST_FRAMES * frame_bytes);
  uint8_t *vid_out = xmalloc((size_t)TEST_FRAMES * frame_bytes);
  float *vid_z =
      xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)TEST_LATENT_T *
              (size_t)h_lat * (size_t)w_lat * sizeof(float));
  for (int f = 0; f < TEST_FRAMES; f++)
    fill_video_frame(vid_in + (size_t)f * frame_bytes, f, TEST_FRAMES, TEST_W,
                     TEST_H);

  t0 = now();
  check(minimax_h3_video_vae_encode(vae, vid_in, TEST_FRAMES, TEST_W, TEST_H,
                                    vid_z, NULL) == 1,
        "22-frame video encoded");
  double vid_enc_ms = (now() - t0) * MS_PER_SEC;
  t0 = now();
  check(minimax_h3_video_vae_decode(vae, vid_z, TEST_LATENT_T, h_lat, w_lat,
                                    vid_out, NULL) == 1,
        "22-frame video decoded");
  double vid_dec_ms = (now() - t0) * MS_PER_SEC;

  double vid_psnr =
      rgb_psnr(vid_in, vid_out, (size_t)TEST_FRAMES * frame_bytes);
  double f0_match = rgb_psnr(vid_in, vid_out, frame_bytes);
  double f_last_match =
      rgb_psnr(vid_in + (size_t)(TEST_FRAMES - 1) * frame_bytes,
               vid_out + (size_t)(TEST_FRAMES - 1) * frame_bytes, frame_bytes);
  double f_cross = rgb_psnr(
      vid_in, vid_out + (size_t)(TEST_FRAMES - 1) * frame_bytes, frame_bytes);
  printf("video 22x320x320: enc %.1f ms, dec %.1f ms, PSNR %.2f dB "
         "(f0=%.2f, f21=%.2f, cross=%.2f)\n",
         vid_enc_ms, vid_dec_ms, vid_psnr, f0_match, f_last_match, f_cross);
  check(vid_psnr >= MIN_VIDEO_PSNR, "22-frame video PSNR >= 24 dB");
  check(f0_match > f_cross + MOTION_PSNR_MARGIN &&
            f_last_match > f_cross + MOTION_PSNR_MARGIN,
        "temporal motion preserved between first and last frame");

  free(vid_z);
  free(vid_out);
  free(vid_in);
  free(still_z);
  free(still_out);
  free(still_in);
  minimax_h3_video_vae_close(vae);
  return failures != 0;
}
