#include "model/deepseek/image.h"
#include "base/error.h"
#include "engine/memory.h"
#include "engine/resample.h"
#include "model/deepseek/model.h"
#include "platform/image.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// inference/config.json: vision_patch_size, vision_downsample_ratio,
// vision_max_n_token, vision_min_pixels (vision_max_wh_ratio is unset).
static constexpr int PATCH = 14, RATIO = 3, MAX_TOKENS = 1024,
                     MIN_PIXELS = 295936;

typedef struct {
  int w, h;
  uint8_t *p;
} Rgb;  // rows of w RGB pixels

// ---- the grid (plan_image_grid, safe_resize, solve_resize_ratio, llm_grid)
// ----

static int tokens_of(int th, int tw) { return th * (tw + 1) + 2; }

static void grid(int bh, int bw, int *th, int *tw) {
  *th = (int)ceil((double)(bh / PATCH) / RATIO),
  *tw = (int)ceil((double)(bw / PATCH) / RATIO);
}

// The largest aspect-preserving size whose grid fits in MAX_TOKENS.
static void fit(int height, int width, int *bh, int *bw) {
  double r = (double)height / width,
         mw = sqrt((MAX_TOKENS - 2) / r + 0.25) - 0.5, mh = mw * r;
  int cell = PATCH * RATIO;
  if (mw < 1.0) {  // very tall: a single column
    *bh = (MAX_TOKENS - 2) / 2 * cell, *bw = cell;
    return;
  }
  if (mh < 1.0) {  // very wide: a single row
    *bh = cell, *bw = (MAX_TOKENS - 3) * cell;
    return;
  }
  double beta = fmin(floor(mw) * cell / width, floor(mh) * cell / height);
  *bh = (int)floor(height * beta / PATCH) * PATCH,
  *bw = (int)floor(width * beta / PATCH) * PATCH;
}

// The pixel size an image is padded to (bw x bh) and its token grid; images
// of fewer than MIN_PIXELS are scaled up first.
static void plan(int width, int height, int *bw, int *bh, int *th, int *tw) {
  if ((double)width * height < MIN_PIXELS) {
    double ratio = pow((double)MIN_PIXELS / ((double)width * height), 0.5);
    width = (int)(width * ratio), height = (int)(height * ratio);
  }
  *bw = (int)ceil((double)width / PATCH) * PATCH,
  *bh = (int)ceil((double)height / PATCH) * PATCH;
  grid(*bh, *bw, th, tw);
  if (tokens_of(*th, *tw) > MAX_TOKENS) {
    fit(height, width, bh, bw);
    grid(*bh, *bw, th, tw);
  }
}

// ImageOps.pad(image, (bw, bh), color=(127, 127, 127)): contain() resizes
// to fit (Python's round(): halves to even), then the result is centered.
static Rgb pad(Rgb s, int bw, int bh) {
  double ir = (double)s.w / s.h, dr = (double)bw / bh;
  int w = bw, h = bh;
  if (ir > dr && (int)rint((double)s.h / s.w * bw) != bh)
    h = (int)rint((double)s.h / s.w * bw);
  if (ir < dr && (int)rint((double)s.w / s.h * bh) != bw)
    w = (int)rint((double)s.w / s.h * bh);
  Rgb r = {w, h, image_resize_rgb(s.p, s.w, s.h, w, h)};
  if (r.w == bw && r.h == bh) return r;
  Rgb o = {bw, bh, xmalloc((size_t)bw * bh * 3)};
  memset(o.p, 127, (size_t)bw * bh * 3);
  int x = r.w != bw ? (int)rint((bw - r.w) * 0.5) : 0,
      y = r.w != bw ? 0 : (int)rint((bh - r.h) * 0.5);
  for (int i = 0; i < r.h; i++)
    memcpy(o.p + ((size_t)(y + i) * bw + x) * 3, r.p + (size_t)i * r.w * 3,
           (size_t)r.w * 3);
  free(r.p);
  return o;
}

// ---- the image's patches, positions and tokens ----

// The image padded to the size the ViT takes (bw x bh RGB pixels) and its
// token grid; null with *err if it cannot be decoded.
static uint8_t *image_rgb(const void *data, size_t len, int *bw, int *bh,
                          int *th, int *tw, const char **err) {
  Rgb s;
  s.p = image_decode_rgb(data, len, &s.w, &s.h, err);
  if (!s.p) return 0;
  plan(s.w, s.h, bw, bh, th, tw);
  Rgb o = pad(s, *bw, *bh);
  free(s.p);
  return o.p;
}

static uint16_t bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

int image_open(Media *im, const void *data, size_t len, const char **err) {
  int bw, bh;
  uint8_t *rgb = image_rgb(data, len, &bw, &bh, &im->th, &im->tw, err);
  if (!rgb) return 0;
  im->vh = bh / PATCH, im->vw = bw / PATCH, im->rows = 0, im->start = 0;
  im->ntokens = tokens_of(im->th, im->tw);
  im->hash = 1469598103934665603ull;  // FNV-1a of the file's bytes
  for (size_t i = 0; i < len; i++)
    im->hash = (im->hash ^ ((const uint8_t *)data)[i]) * 1099511628211ull;
  // the pixels as float32 in [-1, 1], rounded to BF16; a patch's channel c
  // at 196 c + 14 i + j (x.reshape(3, vh, 14, vw, 14).permute(1, 3, 0, 2, 4))
  size_t n = (size_t)im->vh * im->vw;
  im->patch = calloc(n * VIT_K, 2);
  if (!im->patch) {
    free(rgb);
    *err = "the image patches do not fit in memory";
    return 0;
  }
  for (int y = 0; y < bh; y++)
    for (int x = 0; x < bw; x++)
      for (int c = 0; c < 3; c++) {
        float v =
            ((float)rgb[((size_t)y * bw + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
        size_t p = (size_t)(y / PATCH) * im->vw + x / PATCH;
        im->patch[p * VIT_K + c * PATCH * PATCH + (y % PATCH) * PATCH +
                  x % PATCH] = bf16(v);
      }
  free(rgb);
  return 1;
}

// vision.py get_vision_cos_sin: patch (ph, pw) turns pairs 0-15 by ph and
// 16-31 by pw times the frequencies theta^(-2i/32), theta = 10000, all in
// float32 as the reference computes them.
void image_rope(const Media *im, float *cs) {
  float inv[16];
  for (int i = 0; i < 16; i++)
    inv[i] = 1.0f / powf(10000.0f, (float)(2 * i) / 32.0f);
  for (int ph = 0; ph < im->vh; ph++)
    for (int pw = 0; pw < im->vw; pw++) {
      float *row = cs + ((size_t)ph * im->vw + pw) * 64;
      for (int i = 0; i < 32; i++) {
        float a = (float)(i < 16 ? ph : pw) * inv[i % 16];
        row[i] = (float)cos((double)a), row[32 + i] = (float)sin((double)a);
      }
    }
}
