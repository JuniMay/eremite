#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/image_vae.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests MiniMax-H3-Image's FP16 2D ViT VAE decoder on a real checkpoint.
// Pass the VAE safetensors path as argv[1] or EREMITE_MINIMAX_H3_IMAGE_VAE.

// Fills a smooth, deterministic synthetic latent [24, h_lat, w_lat].
static void fill_smooth_latent(float *z, int h_lat, int w_lat) {
  size_t plane = (size_t)h_lat * (size_t)w_lat;
  for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
    for (int y = 0; y < h_lat; y++) {
      float fy = (float)y / (float)h_lat;
      for (int x = 0; x < w_lat; x++) {
        float fx = (float)x / (float)w_lat;
        z[(size_t)c * plane + (size_t)y * (size_t)w_lat + (size_t)x] =
            0.3f * sinf(3.14159f * (fx + 0.1f * (float)c)) *
            cosf(3.14159f * (fy + 0.07f * (float)c));
      }
    }
  }
}

int main(int argc, char **argv) {
  const char *weights =
      argc > 1 ? argv[1] : getenv("EREMITE_MINIMAX_H3_IMAGE_VAE");
  if (!weights) {
    printf("usage: %s <minimax_h3_image_vae_fp16.safetensors>\n", argv[0]);
    return 1;
  }

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-vae-test", 0);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  MinimaxH3ImageVae *vae = minimax_h3_image_vae_open(gpu, &account, weights);
  printf("required decoder weights: %.2f MiB\n",
         (double)minimax_h3_image_vae_required_bytes(vae) /
             (double)(1ull << 20));
  double t0 = now();
  int loaded = minimax_h3_image_vae_load(vae);
  printf("loaded in %.1f ms (resident %.2f MiB)\n", (now() - t0) * 1000.0,
         (double)minimax_h3_image_vae_resident_bytes(vae) /
             (double)(1ull << 20));
  check(loaded, "MiniMax-H3-Image VAE decoder loads cleanly");

  // Single-tile decode (16x16 latent -> 256x256 RGB) + determinism check.
  int h1 = 16, w1 = 16;
  float *z1 = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h1 *
                      (size_t)w1 * sizeof(float));
  fill_smooth_latent(z1, h1, w1);
  size_t rgb1_bytes = (size_t)(h1 * 16) * (size_t)(w1 * 16) * 3;
  uint8_t *rgb1 = xmalloc(rgb1_bytes), *rgb1b = xmalloc(rgb1_bytes);
  double t1 = now();
  int ok1 = minimax_h3_image_vae_decode(vae, z1, h1, w1, rgb1);
  printf("decoded 256x256 (1 tile) in %.1f ms\n", (now() - t1) * 1000.0);
  int ok1b = minimax_h3_image_vae_decode(vae, z1, h1, w1, rgb1b);
  int min_v = 255, max_v = 0;
  for (size_t i = 0; i < rgb1_bytes; i++) {
    if (rgb1[i] < min_v) min_v = rgb1[i];
    if (rgb1[i] > max_v) max_v = rgb1[i];
  }
  printf("256x256 pixel range: [%d, %d]\n", min_v, max_v);
  check(ok1 && ok1b && !memcmp(rgb1, rgb1b, rgb1_bytes) && max_v > min_v,
        "256x256 VAE decode is deterministic and spans non-trivial RGB values");

  // PNG encoding round-trip on the decoded image.
  size_t png_len = 0;
  const char *err = NULL;
  uint8_t *png = image_encode_png(rgb1, w1 * 16, h1 * 16, &png_len, &err);
  check(png && png_len > 0 && !err, "decoded VAE image encodes cleanly to PNG");
  free(png);
  free(rgb1b);
  free(rgb1);
  free(z1);

  // Full 1024x1024 multi-tile decode (64x64 latent -> 25 tiles with blending).
  int h2 = 64, w2 = 64;
  float *z2 = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h2 *
                      (size_t)w2 * sizeof(float));
  fill_smooth_latent(z2, h2, w2);
  size_t rgb2_bytes = (size_t)(h2 * 16) * (size_t)(w2 * 16) * 3;
  uint8_t *rgb2 = xmalloc(rgb2_bytes);
  // Check seam continuity on a constant zero latent (equal to LATENTS_MEAN).
  memset(z2, 0,
         (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h2 * (size_t)w2 *
             sizeof(float));
  double t2 = now();
  int ok2 = minimax_h3_image_vae_decode(vae, z2, h2, w2, rgb2);
  printf("decoded 1024x1024 (25 tiles) in %.1f ms\n", (now() - t2) * 1000.0);
  int width2 = w2 * 16, height2 = h2 * 16;
  static const int cols[] = {185, 191, 224, 255, 511};
  double max_seam_diff = 0.0;
  for (size_t ci = 0; ci < sizeof cols / sizeof cols[0]; ci++) {
    int col = cols[ci];
    double step = 0.0;
    for (int y = 0; y < height2; y++) {
      const uint8_t *a = rgb2 + ((size_t)y * (size_t)width2 + (size_t)col) * 3;
      const uint8_t *b =
          rgb2 + ((size_t)y * (size_t)width2 + (size_t)(col + 1)) * 3;
      for (int c = 0; c < 3; c++) step += abs((int)a[c] - (int)b[c]);
    }
    step /= (double)(height2 * 3);
    printf("1024x1024 mean pixel step at x=%d->%d: %.2f\n", col, col + 1, step);
    if (col == 191 || col == 255)
      if (step > max_seam_diff) max_seam_diff = step;
  }
  check(ok2 && max_seam_diff < 10.0,
        "1024x1024 tiled VAE decode blends smoothly across tile seams");

  free(rgb2);
  free(z2);
  minimax_h3_image_vae_close(vae);
  return failures != 0;
}
