#include "check.h"
#include "model/inkling2/vision.h"
#include "platform/image.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Unit tests for Inkling-2-Mini's vision image preprocessing, checkerboard
// alpha compositing, bilinear 32x32 position embedding taps, and 2D axial
// RoPE frequency layout (no GPU or checkpoint required).

// Test image dimensions and tolerances.
static constexpr int TEST_W = 16;
static constexpr int TEST_H = 16;
static constexpr int PAD_W = 25;
static constexpr int PAD_H = 15;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr float TOL = 1e-5f;

// Decodes one BF16 code to FP32.
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << BF16_SHIFT;
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Verifies 8x8 checkerboard alpha compositing matches
// Inkling2ImageProcessor.convert_to_rgb.
static void test_checkerboard(void) {
  uint8_t rgb[TEST_W * TEST_H * 3];
  uint8_t alpha[TEST_W * TEST_H];
  memset(rgb, 0, sizeof rgb);
  // Fully transparent everywhere -> pure 8x8 checkerboard (204 light, 153
  // dark).
  memset(alpha, 0, sizeof alpha);
  inkling2_vision_composite_checkerboard(rgb, alpha, TEST_W, TEST_H);
  check(rgb[0] == 204 && rgb[1] == 204 && rgb[2] == 204,
        "top-left 8x8 cell composites to light gray 204");
  check(rgb[8 * 3] == 153 && rgb[8 * 3 + 1] == 153,
        "adjacent horizontal 8x8 cell composites to dark gray 153");
  check(rgb[(8 * TEST_W) * 3] == 153,
        "adjacent vertical 8x8 cell composites to dark gray 153");
  check(rgb[(8 * TEST_W + 8) * 3] == 204,
        "diagonal 8x8 cell composites to light gray 204");

  // Half alpha (128) over pure red (255, 0, 0) in light cell (204).
  rgb[0] = 255;
  rgb[1] = 0;
  rgb[2] = 0;
  alpha[0] = 128;
  inkling2_vision_composite_checkerboard(rgb, alpha, 1, 1);
  int exp_r = (255 * 128 + 204 * 127 + 127) / 255;
  int exp_g = (0 * 128 + 204 * 127 + 127) / 255;
  check(rgb[0] == exp_r && rgb[1] == exp_g && rgb[2] == exp_g,
        "partial alpha blends with exact rounding");
}

// Verifies PNG decode, -1.0 fill padding to 40x40 cells, normalization and
// padded 1216-wide patch layout.
static void test_prepare_media(void) {
  uint8_t rgb[PAD_W * PAD_H * 3];
  for (int y = 0; y < PAD_H; y++) {
    for (int x = 0; x < PAD_W; x++) {
      uint8_t *px = rgb + (y * PAD_W + x) * 3;
      px[0] = 255;
      px[1] = 0;
      px[2] = 128;
    }
  }
  size_t png_len = 0;
  const char *err = NULL;
  uint8_t *png = image_encode_png(rgb, PAD_W, PAD_H, &png_len, &err);
  check(png != NULL, "test PNG encodes");

  Media im = {0};
  int ok = inkling2_prepare_media(NULL, &im, png, png_len, &err);
  check(ok, "inkling2_prepare_media decodes and packs PNG");
  check(im.th == 1 && im.tw == 1 && im.vh == 2 && im.vw == 2 && im.ntokens == 1,
        "25x15 image pads to 1x1 token (2x2 patches = 40x40)");

  // Top-left patch (0) pixel (0, 0) is (255, 0, 128) -> (1.0, -1.0, ~0.004).
  float r0 = bf16_to_f32(im.patch[0]);
  float g0 = bf16_to_f32(im.patch[1]);
  check(fabsf(r0 - 1.0f) < 1e-2f && fabsf(g0 - (-1.0f)) < 1e-2f,
        "in-bounds pixels normalize to [-1, 1]");

  // Bottom-right sub-patch (mr=1, mc=1 -> patch 3) starts at (y=20, x=20),
  // which is outside 25x15 -> filled with -1.0 on 0..255 scale -> -1.0078125.
  float pad_v = bf16_to_f32(im.patch[3 * INK2_VIS_PATCH_K_PAD]);
  float expected_pad = (-1.0f / 255.0f - 0.5f) / 0.5f;
  check(fabsf(pad_v - expected_pad) < 1e-2f,
        "out-of-bounds pixels use fill_value=-1.0 before normalization");

  // Padding columns 1200..1215 are zero.
  int tail_zero = 1;
  for (int d = INK2_VIS_PATCH_K; d < INK2_VIS_PATCH_K_PAD; d++)
    if (im.patch[d] != 0) tail_zero = 0;
  check(tail_zero, "patch row tail columns 1200..1215 are zero-padded");

  media_free(&im);
  free(png);
}

// Verifies 4-tap bilinear position embedding taps and weights.
static void test_bilinear_taps(void) {
  // Exact 32x32 grid must sample identity index (row*32 + col) with weight 1.
  int vh = INK2_VIS_POS_GRID, vw = INK2_VIS_POS_GRID;
  int np = vh * vw;
  uint32_t *idx = malloc((size_t)np * INK2_VIS_TAPS * sizeof(uint32_t));
  float *w = malloc((size_t)np * INK2_VIS_TAPS * sizeof(float));
  inkling2_vision_bilinear_taps(vh, vw, idx, w);

  int id_ok = 1;
  size_t p = 0;
  for (int tr = 0; tr < vh / 2; tr++) {
    for (int tc = 0; tc < vw / 2; tc++) {
      for (int mr = 0; mr < 2; mr++) {
        for (int mc = 0; mc < 2; mc++, p++) {
          uint32_t want = (uint32_t)((tr * 2 + mr) * 32 + (tc * 2 + mc));
          if (idx[p * 4] != want || fabsf(w[p * 4] - 1.0f) > TOL) id_ok = 0;
          float sum = w[p * 4] + w[p * 4 + 1] + w[p * 4 + 2] + w[p * 4 + 3];
          if (fabsf(sum - 1.0f) > TOL) id_ok = 0;
        }
      }
    }
  }
  check(id_ok, "32x32 bilinear taps match identity in 2x2 merge-block order");
  free(idx);
  free(w);

  // Non-square 4x6 grid: weights sum to 1 and indices stay within 0..1023.
  uint32_t small_idx[24 * 4];
  float small_w[24 * 4];
  inkling2_vision_bilinear_taps(4, 6, small_idx, small_w);
  int small_ok = 1;
  for (int i = 0; i < 24; i++) {
    float sum = 0.0f;
    for (int t = 0; t < 4; t++) {
      if (small_idx[i * 4 + t] >= (uint32_t)INK2_VIS_POS_ENTRIES) small_ok = 0;
      sum += small_w[i * 4 + t];
    }
    if (fabsf(sum - 1.0f) > TOL) small_ok = 0;
  }
  check(small_ok, "non-square 4x6 bilinear weights sum to 1 and stay in range");
}

// Verifies 2D axial RoPE puts CLS at (0, 0) and width freqs before height freqs
// with 1-indexed patch coordinates.
static void test_rope_table(void) {
  int vh = 2, vw = 4;
  int seq = 1 + vh * vw;
  float *cs = malloc((size_t)seq * INK2_VIS_ROPE_PAIRS * 2 * sizeof(float));
  inkling2_vision_rope_table(vh, vw, cs);

  int cls_ok = 1;
  for (int pair = 0; pair < INK2_VIS_ROPE_PAIRS; pair++) {
    if (fabsf(cs[pair * 2] - 1.0f) > TOL || fabsf(cs[pair * 2 + 1]) > TOL)
      cls_ok = 0;
  }
  check(cls_ok, "CLS row 0 uses identity rotation (0, 0)");

  // Patch index 1 (tr=0, tc=0, mr=0, mc=1) has row=0 -> ph=1, col=1 -> pw=2.
  // Sequence index = 1 + 1 = 2. Pair 0 uses pw=2 * inv_freq[0]=1 -> angle 2.0.
  // Pair 16 uses ph=1 * inv_freq[0]=1 -> angle 1.0.
  const float *r2 = cs + 2 * INK2_VIS_ROPE_PAIRS * 2;
  check(fabsf(r2[0] - cosf(2.0f)) < TOL && fabsf(r2[1] - sinf(2.0f)) < TOL &&
            fabsf(r2[16 * 2] - cosf(1.0f)) < TOL &&
            fabsf(r2[16 * 2 + 1] - sinf(1.0f)) < TOL,
        "2D axial RoPE places 1-indexed width freqs before height freqs");
  free(cs);
}

int main(void) {
  test_checkerboard();
  test_prepare_media();
  test_bilinear_taps();
  test_rope_table();
  return failures != 0;
}
