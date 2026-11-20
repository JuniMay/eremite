#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/text_encoder.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// End-to-end MiniMax-H3-Image text-to-image test across the real NVFP4 Qwen3-VL
// text encoder, INT8-ConvRot 6-step PDD DiT, and FP16 2D ViT VAE decoder.
// Pass paths via argv[1..4] or environment variables:
//   EREMITE_MINIMAX_H3_TEXT_ENCODER
//   EREMITE_MINIMAX_H3_TOKENIZER
//   EREMITE_MINIMAX_H3_IMAGE_DIT
//   EREMITE_MINIMAX_H3_IMAGE_VAE
// Optional flags: --prompt <text>, --out <file.png>.

// Computes mean, standard deviation and finite check over count floats.
static int latent_stats(const float *x, size_t count, double *mean_out,
                        double *std_out) {
  double sum = 0.0, sq = 0.0;
  for (size_t i = 0; i < count; i++) {
    if (!isfinite(x[i])) return 0;
    sum += (double)x[i];
    sq += (double)x[i] * (double)x[i];
  }
  double mean = sum / (double)count;
  double var = sq / (double)count - mean * mean;
  *mean_out = mean;
  *std_out = sqrt(var > 0.0 ? var : 0.0);
  return 1;
}

// Computes mean adjacent horizontal pixel step and RGB channel std over an
// RGB image to distinguish structured images from uniform fields or pure noise.
static void image_stats(const uint8_t *rgb, int width, int height,
                        double *std_out, double *step_out) {
  size_t n = (size_t)width * (size_t)height * 3u;
  double sum = 0.0, sq = 0.0;
  for (size_t i = 0; i < n; i++) {
    double v = (double)rgb[i];
    sum += v;
    sq += v * v;
  }
  double mean = sum / (double)n;
  double var = sq / (double)n - mean * mean;
  *std_out = sqrt(var > 0.0 ? var : 0.0);

  double step = 0.0;
  size_t pairs = 0;
  for (int y = 0; y < height; y++) {
    const uint8_t *row = rgb + (size_t)y * (size_t)width * 3u;
    for (int x = 0; x + 1 < width; x++) {
      for (int c = 0; c < 3; c++)
        step += fabs((double)row[x * 3 + c] - (double)row[(x + 1) * 3 + c]);
      pairs += 3;
    }
  }
  *step_out = step / (double)pairs;
}

// Computes mean R, G and B channel values across an RGB image.
static void mean_rgb(const uint8_t *rgb, int width, int height, double out[3]) {
  size_t pixels = (size_t)width * (size_t)height;
  double sum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < pixels; i++) {
    sum[0] += (double)rgb[i * 3];
    sum[1] += (double)rgb[i * 3 + 1];
    sum[2] += (double)rgb[i * 3 + 2];
  }
  for (int c = 0; c < 3; c++) out[c] = sum[c] / (double)pixels;
}

int main(int argc, char **argv) {
  const char *te_path = getenv("EREMITE_MINIMAX_H3_TEXT_ENCODER");
  const char *tok_path = getenv("EREMITE_MINIMAX_H3_TOKENIZER");
  const char *dit_path = getenv("EREMITE_MINIMAX_H3_IMAGE_DIT");
  const char *vae_path = getenv("EREMITE_MINIMAX_H3_IMAGE_VAE");
  const char *prompt =
      "A red apple sitting on a clean white wooden table in soft morning "
      "sunlight.";
  const char *out_path = NULL;

  int pos_arg = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
      prompt = argv[++i];
    } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      if (pos_arg == 0)
        te_path = argv[i];
      else if (pos_arg == 1)
        tok_path = argv[i];
      else if (pos_arg == 2)
        dit_path = argv[i];
      else if (pos_arg == 3)
        vae_path = argv[i];
      pos_arg++;
    }
  }
  if (!te_path || !tok_path || !dit_path || !vae_path) {
    printf("usage: %s <text_encoder.safetensors> <tokenizer_dir> "
           "<dit.safetensors> <vae.safetensors> [--prompt text] [--out png]\n",
           argv[0]);
    return 1;
  }

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-t2i-test", 0);
  MemorySlot *shared = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  double t0 = now();
  MinimaxH3TextEncoder *te =
      minimax_h3_text_encoder_open(gpu, &account, te_path, tok_path);
  MinimaxH3Dit *dit = minimax_h3_dit_open(gpu, &account, dit_path);
  MinimaxH3ImageVae *vae = minimax_h3_image_vae_open(gpu, &account, vae_path);
  check(minimax_h3_dit_pdd_steps(dit) == 6,
        "DiT checkpoint configures 6 PDD steps");

  double te_gib =
      (double)minimax_h3_text_encoder_required_bytes(te) / (double)(1ull << 30);
  double dit_gib =
      (double)minimax_h3_dit_required_bytes(dit) / (double)(1ull << 30);
  double vae_gib =
      (double)minimax_h3_image_vae_required_bytes(vae) / (double)(1ull << 30);
  printf("required weights: text=%.2f GiB, dit=%.2f GiB, vae=%.2f GiB, "
         "total=%.2f GiB\n",
         te_gib, dit_gib, vae_gib, te_gib + dit_gib + vae_gib);

  int ok_load = minimax_h3_text_encoder_load(te) && minimax_h3_dit_load(dit) &&
                minimax_h3_image_vae_load(vae);
  printf("all three models loaded in %.2f s (resident %.2f GiB)\n", now() - t0,
         (double)(minimax_h3_text_encoder_resident_bytes(te) +
                  minimax_h3_dit_resident_bytes(dit) +
                  minimax_h3_image_vae_resident_bytes(vae)) /
             (double)(1ull << 30));
  check(ok_load, "text encoder, DiT and VAE load together in full residence");

  // Encode prompt once.
  double t_enc = now();
  Buf text_hidden = {0};
  int text_len = 0;
  int ok_enc = minimax_h3_text_encoder_encode_prompt(te, prompt, shared,
                                                     &text_hidden, &text_len);
  printf("text encode (%d tokens): %.1f ms\n", text_len,
         (now() - t_enc) * 1000.0);
  check(ok_enc && text_len > 0, "prompt encodes to Qwen3-VL layer-49 states");

  // 512x512 (32x32 latent) sampling twice with identical seed for determinism.
  int h512 = 32, w512 = 32;
  size_t lat512_n =
      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h512 * (size_t)w512;
  float *z512_a = xmalloc(lat512_n * sizeof(float));
  float *z512_b = xmalloc(lat512_n * sizeof(float));
  double t_s1 = now();
  int ok_s1 = minimax_h3_dit_sample(dit, &text_hidden, text_len, NULL, NULL, 0,
                                    h512, w512, 42, z512_a);
  printf("DiT 512x512 (6 steps): %.2f s\n", now() - t_s1);
  int ok_s2 = minimax_h3_dit_sample(dit, &text_hidden, text_len, NULL, NULL, 0,
                                    h512, w512, 42, z512_b);
  double mean512 = 0.0, std512 = 0.0;
  int fin512 = latent_stats(z512_a, lat512_n, &mean512, &std512);
  printf("512x512 latent stats: mean=%.4f std=%.4f\n", mean512, std512);
  check(
      ok_s1 && ok_s2 && fin512 &&
          !memcmp(z512_a, z512_b, lat512_n * sizeof(float)),
      "512x512 DiT sampling is finite and bit-deterministic for a fixed seed");
  check(std512 > 0.2 && std512 < 5.0,
        "512x512 DiT latent has plausible normalized spread");

  size_t rgb512_bytes = (size_t)(h512 * 16) * (size_t)(w512 * 16) * 3u;
  uint8_t *rgb512 = xmalloc(rgb512_bytes);
  double t_v1 = now();
  int ok_v1 = minimax_h3_image_vae_decode(vae, z512_a, h512, w512, rgb512);
  printf("VAE decode 512x512: %.1f ms\n", (now() - t_v1) * 1000.0);
  double std_px512 = 0.0, step_px512 = 0.0;
  image_stats(rgb512, w512 * 16, h512 * 16, &std_px512, &step_px512);
  printf("512x512 image stats: pixel_std=%.2f mean_adj_step=%.2f\n", std_px512,
         step_px512);
  check(ok_v1 && std_px512 > 15.0 && step_px512 < 25.0,
        "512x512 decoded image has natural contrast and smooth local pixels");

  // Semantic prompt-following check at 512x512 using identical seed: a bright
  // red prompt must shift mean R above B, and a bright blue prompt must shift
  // mean B above R.
  Buf red_h = {0}, blue_h = {0};
  int red_len = 0, blue_len = 0;
  int ok_red = minimax_h3_text_encoder_encode_prompt(
                   te,
                   "A vibrant crimson red wall filling the entire frame, solid "
                   "bright red color.",
                   shared, &red_h, &red_len) &&
               minimax_h3_dit_sample(dit, &red_h, red_len, NULL, NULL, 0, h512,
                                     w512, 7, z512_a) &&
               minimax_h3_image_vae_decode(vae, z512_a, h512, w512, rgb512);
  double red_rgb[3] = {0};
  if (ok_red) mean_rgb(rgb512, w512 * 16, h512 * 16, red_rgb);
  int ok_blue = minimax_h3_text_encoder_encode_prompt(
                    te,
                    "A vibrant deep sapphire blue wall filling the entire "
                    "frame, solid bright blue color.",
                    shared, &blue_h, &blue_len) &&
                minimax_h3_dit_sample(dit, &blue_h, blue_len, NULL, NULL, 0,
                                      h512, w512, 7, z512_a) &&
                minimax_h3_image_vae_decode(vae, z512_a, h512, w512, rgb512);
  double blue_rgb[3] = {0};
  if (ok_blue) mean_rgb(rgb512, w512 * 16, h512 * 16, blue_rgb);
  printf("red prompt mean RGB=(%.1f, %.1f, %.1f), blue prompt mean RGB=(%.1f, "
         "%.1f, %.1f)\n",
         red_rgb[0], red_rgb[1], red_rgb[2], blue_rgb[0], blue_rgb[1],
         blue_rgb[2]);
  check(ok_red && ok_blue && red_rgb[0] > red_rgb[2] + 30.0 &&
            blue_rgb[2] > blue_rgb[0] + 30.0,
        "end-to-end T2I follows red vs blue prompt conditioning");
  gpu_free(&blue_h);
  gpu_free(&red_h);
  free(rgb512);
  free(z512_b);
  free(z512_a);

  // 1024x1024 (64x64 latent) full-resolution sampling + decode + PNG encode.
  int h1024 = 64, w1024 = 64;
  size_t lat1024_n =
      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h1024 * (size_t)w1024;
  float *z1024 = xmalloc(lat1024_n * sizeof(float));
  double t_s3 = now();
  int ok_s3 = minimax_h3_dit_sample(dit, &text_hidden, text_len, NULL, NULL, 0,
                                    h1024, w1024, 42, z1024);
  printf("DiT 1024x1024 (6 steps): %.2f s\n", now() - t_s3);
  double mean1024 = 0.0, std1024 = 0.0;
  int fin1024 = latent_stats(z1024, lat1024_n, &mean1024, &std1024);
  printf("1024x1024 latent stats: mean=%.4f std=%.4f\n", mean1024, std1024);
  check(ok_s3 && fin1024 && std1024 > 0.2 && std1024 < 5.0,
        "1024x1024 DiT sampling produces finite, well-scaled latents");

  size_t rgb1024_bytes = (size_t)(h1024 * 16) * (size_t)(w1024 * 16) * 3u;
  uint8_t *rgb1024 = xmalloc(rgb1024_bytes);
  double t_v2 = now();
  int ok_v2 = minimax_h3_image_vae_decode(vae, z1024, h1024, w1024, rgb1024);
  printf("VAE decode 1024x1024: %.1f ms\n", (now() - t_v2) * 1000.0);
  double std_px1024 = 0.0, step_px1024 = 0.0;
  image_stats(rgb1024, w1024 * 16, h1024 * 16, &std_px1024, &step_px1024);
  printf("1024x1024 image stats: pixel_std=%.2f mean_adj_step=%.2f\n",
         std_px1024, step_px1024);
  check(ok_v2 && std_px1024 > 15.0 && step_px1024 < 20.0,
        "1024x1024 decoded image has natural contrast and smooth local pixels");

  size_t png_len = 0;
  const char *err = NULL;
  uint8_t *png =
      image_encode_png(rgb1024, w1024 * 16, h1024 * 16, &png_len, &err);
  check(png && png_len > 0 && !err, "1024x1024 generated image encodes to PNG");
  if (out_path && png) {
    FILE *f = fopen(out_path, "wb");
    if (!f) die("cannot open %s for writing", out_path);
    if (fwrite(png, 1, png_len, f) != png_len)
      die("write failed for %s", out_path);
    fclose(f);
    printf("wrote %s (%zu bytes)\n", out_path, png_len);
  }

  free(png);
  free(rgb1024);
  free(z1024);
  gpu_free(&text_hidden);
  minimax_h3_image_vae_close(vae);
  minimax_h3_dit_close(dit);
  minimax_h3_text_encoder_close(te);
  return failures != 0;
}
