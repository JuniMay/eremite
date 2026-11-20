#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "engine/resample.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/image_vae.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests MiniMax-H3-Image's FP16 2D ViT VAE decoder and single-frame 3D causal
// CNN encoder on a real checkpoint.
// Pass the VAE safetensors path as argv[1] or EREMITE_MINIMAX_H3_IMAGE_VAE,
// and an optional reference image path as argv[2] or
// EREMITE_MINIMAX_H3_REF_IMAGE for real-photo encode->decode round-trip PSNR.

// Computes PSNR (dB) between two 8-bit RGB images of identical dimensions.
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

// Fills a smooth colorful test image [height, width, 3] for VAE round-trip.
static void fill_test_image(uint8_t *rgb, int width, int height) {
  for (int y = 0; y < height; y++) {
    double fy = (double)y / (double)height;
    for (int x = 0; x < width; x++) {
      double fx = (double)x / (double)width;
      double dx = fx - 0.5, dy = fy - 0.5;
      double ring = exp(-18.0 * (dx * dx + dy * dy));
      double r = 0.2 + 0.6 * fx + 0.2 * ring;
      double g =
          0.15 + 0.55 * fy + 0.25 * sin(3.14159 * fx) * sin(3.14159 * fy);
      double b = 0.75 - 0.5 * fx * fy + 0.2 * ring;
      uint8_t *px = rgb + ((size_t)y * (size_t)width + (size_t)x) * 3u;
      px[0] = (uint8_t)lround(fmin(fmax(r, 0.0), 1.0) * 255.0);
      px[1] = (uint8_t)lround(fmin(fmax(g, 0.0), 1.0) * 255.0);
      px[2] = (uint8_t)lround(fmin(fmax(b, 0.0), 1.0) * 255.0);
    }
  }
}

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

  // Single-frame 3D causal CNN encoder load and 512x512 tiled round-trip.
  printf("required encoder weights: %.2f MiB\n",
         (double)minimax_h3_image_vae_encoder_required_bytes(vae) /
             (double)(1ull << 20));
  double te0 = now();
  int enc_loaded = minimax_h3_image_vae_load_encoder(vae);
  printf("encoder loaded in %.1f ms (resident %.2f MiB)\n",
         (now() - te0) * 1000.0,
         (double)minimax_h3_image_vae_encoder_resident_bytes(vae) /
             (double)(1ull << 20));
  check(enc_loaded, "MiniMax-H3-Image VAE encoder loads cleanly");

  int rt_w = 512, rt_h = 512;
  int rt_lw = rt_w / 16, rt_lh = rt_h / 16;
  size_t rt_rgb_bytes = (size_t)rt_w * (size_t)rt_h * 3u;
  size_t rt_lat_floats =
      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)rt_lh * (size_t)rt_lw;
  uint8_t *rt_in = xmalloc(rt_rgb_bytes);
  uint8_t *rt_out = xmalloc(rt_rgb_bytes);
  float *rt_z1 = xmalloc(rt_lat_floats * sizeof(float));
  float *rt_z2 = xmalloc(rt_lat_floats * sizeof(float));
  fill_test_image(rt_in, rt_w, rt_h);

  double te1 = now();
  int ok_e1 = minimax_h3_image_vae_encode(vae, rt_in, rt_w, rt_h, rt_z1);
  printf("encoded 512x512 (9 tiles) in %.1f ms\n", (now() - te1) * 1000.0);
  int ok_e2 = minimax_h3_image_vae_encode(vae, rt_in, rt_w, rt_h, rt_z2);
  int ok_rt_dec = minimax_h3_image_vae_decode(vae, rt_z1, rt_lh, rt_lw, rt_out);
  double psnr = rgb_psnr(rt_in, rt_out, rt_rgb_bytes);
  printf("512x512 synthetic encode->decode PSNR: %.2f dB\n", psnr);
  check(ok_e1 && ok_e2 && !memcmp(rt_z1, rt_z2, rt_lat_floats * sizeof(float)),
        "512x512 tiled VAE encode is bit-deterministic");
  check(ok_rt_dec && psnr > 28.0,
        "512x512 VAE encode->decode round-trip preserves image (PSNR > 28 dB)");
  free(rt_z2), free(rt_z1), free(rt_out), free(rt_in);

  // Optional real-photo round-trip check when argv[2] / env var is provided.
  const char *ref_path =
      argc > 2 ? argv[2] : getenv("EREMITE_MINIMAX_H3_REF_IMAGE");
  if (ref_path) {
    FILE *f = fopen(ref_path, "rb");
    if (!f) die("cannot open reference image %s", ref_path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *raw = xmalloc((size_t)sz);
    if (fread(raw, 1, (size_t)sz, f) != (size_t)sz)
      die("read failed for %s", ref_path);
    fclose(f);
    int dw = 0, dh = 0;
    const char *ierr = NULL;
    uint8_t *decoded = image_decode_rgb(raw, (size_t)sz, &dw, &dh, &ierr);
    check(decoded != NULL && !ierr, "reference image decodes cleanly");
    free(raw);
    if (decoded) {
      int rw = (dw / 16) * 16;
      int rh = (dh / 16) * 16;
      if (rw < 16) rw = 16;
      if (rh < 16) rh = 16;
      uint8_t *resized = image_resize_rgb_lanczos(decoded, dw, dh, rw, rh);
      free(decoded);
      int rlw = rw / 16, rlh = rh / 16;
      float *rz = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)rlh *
                          (size_t)rlw * sizeof(float));
      uint8_t *recon = xmalloc((size_t)rw * (size_t)rh * 3u);
      double te_r = now();
      int ok_re = minimax_h3_image_vae_encode(vae, resized, rw, rh, rz);
      printf("encoded %dx%d reference image in %.1f ms\n", rw, rh,
             (now() - te_r) * 1000.0);
      int ok_rd = minimax_h3_image_vae_decode(vae, rz, rlh, rlw, recon);
      double r_psnr = rgb_psnr(resized, recon, (size_t)rw * (size_t)rh * 3u);
      printf("reference image (%dx%d) encode->decode PSNR: %.2f dB\n", rw, rh,
             r_psnr);
      check(ok_re && ok_rd && r_psnr > 26.0,
            "real reference image VAE round-trip PSNR > 26 dB");
      free(recon), free(rz), free(resized);
    }
  }

  minimax_h3_image_vae_close(vae);
  return failures != 0;
}
