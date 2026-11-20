#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "engine/resample.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/text_encoder.h"
#include "model/minimax_h3/vision_encoder.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// End-to-end MiniMax-H3-Image reference image editing test across the real
// Qwen3-VL text+vision encoder, 3D causal CNN VAE encoder, INT8-ConvRot 6-step
// PDD DiT, and FP16 2D ViT VAE decoder.
// Pass paths via argv[1..5] or environment variables:
//   EREMITE_MINIMAX_H3_TEXT_ENCODER
//   EREMITE_MINIMAX_H3_TOKENIZER
//   EREMITE_MINIMAX_H3_IMAGE_DIT
//   EREMITE_MINIMAX_H3_IMAGE_VAE
//   EREMITE_MINIMAX_H3_REF_IMAGE
// Optional flags: --prompt <text>, --out <file.png>.

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

// Computes Pearson correlation between two 8-bit RGB images of equal size.
static double rgb_corr(const uint8_t *a, const uint8_t *b, size_t n) {
  double ma = 0.0, mb = 0.0;
  for (size_t i = 0; i < n; i++) {
    ma += (double)a[i];
    mb += (double)b[i];
  }
  ma /= (double)n;
  mb /= (double)n;
  double num = 0.0, da = 0.0, db = 0.0;
  for (size_t i = 0; i < n; i++) {
    double xa = (double)a[i] - ma, xb = (double)b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  if (da <= 0.0 || db <= 0.0) return 0.0;
  return num / sqrt(da * db);
}

int main(int argc, char **argv) {
  const char *te_path = getenv("EREMITE_MINIMAX_H3_TEXT_ENCODER");
  const char *tok_path = getenv("EREMITE_MINIMAX_H3_TOKENIZER");
  const char *dit_path = getenv("EREMITE_MINIMAX_H3_IMAGE_DIT");
  const char *vae_path = getenv("EREMITE_MINIMAX_H3_IMAGE_VAE");
  const char *ref_path = getenv("EREMITE_MINIMAX_H3_REF_IMAGE");
  const char *prompt =
      "Keep the exact scene, composition, and subjects of <Picture 1> "
      "unchanged.";
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
      else if (pos_arg == 4)
        ref_path = argv[i];
      pos_arg++;
    }
  }
  if (!te_path || !tok_path || !dit_path || !vae_path || !ref_path) {
    printf("usage: %s <text_encoder.safetensors> <tokenizer_dir> "
           "<dit.safetensors> <vae.safetensors> <ref.png> [--prompt text] "
           "[--out png]\n",
           argv[0]);
    return 1;
  }

  // Load and decode the reference image.
  FILE *f = fopen(ref_path, "rb");
  if (!f) die("cannot open reference image %s", ref_path);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *raw = xmalloc((size_t)sz);
  if (fread(raw, 1, (size_t)sz, f) != (size_t)sz)
    die("read failed for %s", ref_path);
  fclose(f);
  int ref_w = 0, ref_h = 0;
  const char *ierr = NULL;
  uint8_t *ref_rgb = image_decode_rgb(raw, (size_t)sz, &ref_w, &ref_h, &ierr);
  free(raw);
  check(ref_rgb != NULL && !ierr, "reference image decodes cleanly");
  if (!ref_rgb) return 1;

  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-edit-test", 0);
  MemorySlot *shared = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  double t0 = now();
  MinimaxH3TextEncoder *te =
      minimax_h3_text_encoder_open(gpu, &account, te_path, tok_path);
  MinimaxH3VisionEncoder *vis = minimax_h3_vision_encoder_open(te, &account);

  // Sanity-check vision encoder output alone first (finite, non-zero spread).
  {
    int vw = 0, vh = 0;
    minimax_h3_vision_ref_size(ref_w, ref_h, &vw, &vh);
    uint8_t *vrgb =
        (vw == ref_w && vh == ref_h)
            ? ref_rgb
            : image_resize_rgb_lanczos(ref_rgb, ref_w, ref_h, vw, vh);
    Buf vm = {0}, vds[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS] = {0};
    int vgh = 0, vgw = 0;
    int ok_v = minimax_h3_vision_encoder_encode_image(vis, vrgb, vw, vh, shared,
                                                      &vm, vds, &vgh, &vgw);
    if (vrgb != ref_rgb) free(vrgb);
    size_t vn =
        (size_t)(vgh / 2) * (size_t)(vgw / 2) * (size_t)MINIMAX_H3_TEXT_DIM;
    size_t mnans = 0, d0nans = 0;
    float mmx = 0.0f, d0mx = 0.0f;
    if (ok_v) {
      const float *mp = (const float *)vm.ptr;
      const float *d0 = (const float *)vds[0].ptr;
      for (size_t i = 0; i < vn; i++) {
        if (!isfinite(mp[i]))
          mnans++;
        else if (fabsf(mp[i]) > mmx)
          mmx = fabsf(mp[i]);
        if (!isfinite(d0[i]))
          d0nans++;
        else if (fabsf(d0[i]) > d0mx)
          d0mx = fabsf(d0[i]);
      }
      printf("vision alone (%dx%d): merged nans=%zu max=%.4f; ds0 nans=%zu "
             "max=%.4f\n",
             vgw, vgh, mnans, mmx, d0nans, d0mx);
    }
    gpu_free(&vm);
    for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
      gpu_free(&vds[d]);
    check(ok_v && mnans == 0 && d0nans == 0 && mmx > 0.1f,
          "vision tower produces finite non-trivial merged and DeepStack "
          "features");
    if (!ok_v || mnans > 0 || d0nans > 0) return 1;
  }

  MinimaxH3Dit *dit = minimax_h3_dit_open(gpu, &account, dit_path);
  MinimaxH3ImageVae *vae = minimax_h3_image_vae_open(gpu, &account, vae_path);

  int ok_load = minimax_h3_text_encoder_load(te) &&
                minimax_h3_vision_encoder_load(vis) &&
                minimax_h3_dit_load(dit) && minimax_h3_image_vae_load(vae) &&
                minimax_h3_image_vae_load_encoder(vae);
  uint64_t total_res = minimax_h3_text_encoder_resident_bytes(te) +
                       minimax_h3_vision_encoder_resident_bytes(vis) +
                       minimax_h3_dit_resident_bytes(dit) +
                       minimax_h3_image_vae_resident_bytes(vae) +
                       minimax_h3_image_vae_encoder_resident_bytes(vae);
  printf("all 5 weight groups loaded in %.2f s (resident %.2f GiB; "
         "vision=%.2f GiB)\n",
         now() - t0, (double)total_res / (double)(1ull << 30),
         (double)minimax_h3_vision_encoder_resident_bytes(vis) /
             (double)(1ull << 30));
  check(ok_load, "full editing stack loads together in full residence");

  // Encode reference image with VAE encoder at its reference size.
  int vae_w = 0, vae_h = 0;
  minimax_h3_vae_ref_size(ref_w, ref_h, &vae_w, &vae_h);
  uint8_t *vae_rgb =
      (vae_w == ref_w && vae_h == ref_h)
          ? ref_rgb
          : image_resize_rgb_lanczos(ref_rgb, ref_w, ref_h, vae_w, vae_h);
  int ref_lh = vae_h / 16, ref_lw = vae_w / 16;
  float *ref_lat = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                           (size_t)ref_lh * (size_t)ref_lw * sizeof(float));
  double t_ve = now();
  int ok_ve = minimax_h3_image_vae_encode(vae, vae_rgb, vae_w, vae_h, ref_lat);
  printf("VAE encode reference (%dx%d -> %dx%d lat): %.1f ms\n", vae_w, vae_h,
         ref_lw, ref_lh, (now() - t_ve) * 1000.0);
  check(ok_ve, "reference image encodes cleanly through VAE encoder");
  MinimaxH3RefLatent dit_ref = {ref_lat, ref_lh, ref_lw};
  MinimaxH3RefImage vis_ref = {ref_rgb, ref_w, ref_h};

  // 1. Preserve-reference edit at 1024x1024 (64x64 latent) and check spatial
  //    correlation against reference image!
  double t_mm = now();
  Buf text_h = {0};
  int seq_len = 0;
  int *tags = NULL;
  int ok_mm = minimax_h3_encode_multimodal_prompt(
      te, vis, prompt, &vis_ref, 1, shared, &text_h, &seq_len, &tags);
  printf("multimodal encode (%d tokens): %.2f s\n", seq_len, now() - t_mm);
  if (ok_mm) {
    const float *hp = (const float *)text_h.ptr;
    size_t nans = 0;
    float mx = 0.0f;
    for (size_t i = 0; i < (size_t)seq_len * MINIMAX_H3_TEXT_DIM; i++) {
      if (!isfinite(hp[i]))
        nans++;
      else if (fabsf(hp[i]) > mx)
        mx = fabsf(hp[i]);
    }
    int r_vis = 4, r_last = seq_len - 1;
    printf("text_h: nans=%zu max_abs=%.4f row0[0]=%.4f row4[0]=%.4f "
           "row_last[0]=%.4f\n",
           nans, mx, hp[0], hp[(size_t)r_vis * MINIMAX_H3_TEXT_DIM],
           hp[(size_t)r_last * MINIMAX_H3_TEXT_DIM]);
  }
  check(ok_mm && seq_len > 1024 && tags != NULL,
        "multimodal prompt encodes vision tokens, DeepStack and MRoPE");

  int out_lh = 64, out_lw = 64;
  int out_h = out_lh * 16, out_w = out_lw * 16;
  float *out_lat = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                           (size_t)out_lh * (size_t)out_lw * sizeof(float));
  uint8_t *out_rgb = xmalloc((size_t)out_h * (size_t)out_w * 3u);
  double t_dit = now();
  int ok_dit = minimax_h3_dit_sample(dit, &text_h, seq_len, tags, &dit_ref, 1,
                                     out_lh, out_lw, 42, out_lat);
  printf("DiT edit 1024x1024 (6 steps, seq=%d + 2048 vis): %.2f s\n", seq_len,
         now() - t_dit);
  int ok_dec = ok_dit && minimax_h3_image_vae_decode(vae, out_lat, out_lh,
                                                     out_lw, out_rgb);
  uint8_t *ref_1024 =
      (ref_w == out_w && ref_h == out_h)
          ? ref_rgb
          : image_resize_rgb_lanczos(ref_rgb, ref_w, ref_h, out_w, out_h);
  double corr_keep =
      ok_dec ? rgb_corr(ref_1024, out_rgb, (size_t)out_w * (size_t)out_h * 3u)
             : 0.0;
  printf("preserve-reference edit correlation with reference1: %.4f\n",
         corr_keep);
  check(ok_dec && corr_keep > 0.70,
        "preserve-reference edit closely reproduces reference image "
        "(corr > 0.70)");

  if (out_path && ok_dec) {
    size_t png_len = 0;
    const char *perr = NULL;
    uint8_t *png = image_encode_png(out_rgb, out_w, out_h, &png_len, &perr);
    if (png) {
      FILE *of = fopen(out_path, "wb");
      if (!of) die("cannot open %s", out_path);
      if (fwrite(png, 1, png_len, of) != png_len) die("write failed");
      fclose(of);
      printf("wrote %s (%zu bytes)\n", out_path, png_len);
      free(png);
    }
  }
  gpu_free(&text_h);
  free(tags);

  double ref_m[3] = {0};
  mean_rgb(ref_rgb, ref_w, ref_h, ref_m);
  printf("reference1 (%dx%d) mean RGB=(%.1f, %.1f, %.1f)\n", ref_w, ref_h,
         ref_m[0], ref_m[1], ref_m[2]);

  // 2. Instruction-following check on reference edit at 512x512: red vs blue
  //    lighting edit on the same reference image and seed.
  int h512 = 32, w512 = 32;
  float *lat512_r = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                            (size_t)h512 * (size_t)w512 * sizeof(float));
  float *lat512_b = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                            (size_t)h512 * (size_t)w512 * sizeof(float));
  uint8_t *rgb512 = xmalloc((size_t)(h512 * 16) * (size_t)(w512 * 16) * 3u);
  double red_m[3] = {0}, blue_m[3] = {0};

  Buf rh = {0}, bh = {0};
  int rlen = 0, blen = 0, *rtags = NULL, *btags = NULL;
  int ok_r =
      minimax_h3_encode_multimodal_prompt(
          te, vis,
          "Apply an intense crimson red color filter and bright red lighting "
          "over the entire scene of <Picture 1>.",
          &vis_ref, 1, shared, &rh, &rlen, &rtags) &&
      minimax_h3_dit_sample(dit, &rh, rlen, rtags, &dit_ref, 1, h512, w512, 9,
                            lat512_r) &&
      minimax_h3_image_vae_decode(vae, lat512_r, h512, w512, rgb512);
  if (ok_r) mean_rgb(rgb512, w512 * 16, h512 * 16, red_m);

  int ok_b =
      minimax_h3_encode_multimodal_prompt(
          te, vis,
          "Apply an intense deep sapphire blue color filter and bright blue "
          "lighting over the entire scene of <Picture 1>.",
          &vis_ref, 1, shared, &bh, &blen, &btags) &&
      minimax_h3_dit_sample(dit, &bh, blen, btags, &dit_ref, 1, h512, w512, 9,
                            lat512_b) &&
      minimax_h3_image_vae_decode(vae, lat512_b, h512, w512, rgb512);
  if (ok_b) mean_rgb(rgb512, w512 * 16, h512 * 16, blue_m);

  // Diagnostic: check hidden state & latent differences between red and blue.
  float h_last_diff = 0.0f, lat_diff = 0.0f;
  if (ok_r && ok_b) {
    const float *rp =
        (const float *)rh.ptr + (size_t)(rlen - 1) * MINIMAX_H3_TEXT_DIM;
    const float *bp =
        (const float *)bh.ptr + (size_t)(blen - 1) * MINIMAX_H3_TEXT_DIM;
    for (int i = 0; i < MINIMAX_H3_TEXT_DIM; i++) {
      float d = fabsf(rp[i] - bp[i]);
      if (d > h_last_diff) h_last_diff = d;
    }
    size_t ln =
        (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h512 * (size_t)w512;
    for (size_t i = 0; i < ln; i++) {
      float d = fabsf(lat512_r[i] - lat512_b[i]);
      if (d > lat_diff) lat_diff = d;
    }
  }
  printf("rlen=%d blen=%d h_last_max_diff=%.4f lat_max_diff=%.4f\n", rlen, blen,
         h_last_diff, lat_diff);
  printf("red-lit edit mean RGB=(%.1f, %.1f, %.1f), blue-lit edit mean "
         "RGB=(%.1f, %.1f, %.1f)\n",
         red_m[0], red_m[1], red_m[2], blue_m[0], blue_m[1], blue_m[2]);
  check(ok_r && ok_b && (red_m[0] - red_m[2]) > (blue_m[0] - blue_m[2]) + 40.0,
        "reference image edit shifts scene color according to red vs blue "
        "prompt");

  gpu_free(&bh), gpu_free(&rh);
  free(btags), free(rtags);
  free(rgb512), free(lat512_b), free(lat512_r);
  if (ref_1024 != ref_rgb) free(ref_1024);
  if (vae_rgb != ref_rgb) free(vae_rgb);
  free(out_rgb), free(out_lat), free(ref_lat), free(ref_rgb);
  minimax_h3_image_vae_close(vae);
  minimax_h3_dit_close(dit);
  minimax_h3_vision_encoder_close(vis);
  minimax_h3_text_encoder_close(te);
  return failures != 0;
}
