#include "engine/resample.h"
#include "base/error.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int w, h;
  uint8_t *p;
} Rgb;  // rows of w RGB pixels

// Pillow's PRECISION_BITS: the fraction bits of its weights.
#define FRAC 22
// The largest weight of torch's 16-bit weights, and the most fraction bits
// that it gives them (_compute_index_ranges_int16_weights).
#define TORCH_WEIGHT_LIMIT (1 << 15)
#define TORCH_MOST_FRAC 22

// How the weights become integers.
typedef enum {
  // Pillow's: FRAC fraction bits.
  WEIGHTS_PILLOW,
  // torch's for uint8 pixels: the most fraction bits, up to TORCH_MOST_FRAC,
  // that keep twice the axis's largest weight below TORCH_WEIGHT_LIMIT.
  WEIGHTS_TORCH,
} Weights;

// Resampling kernel and its half-width support radius.
typedef struct {
  // Kernel evaluation function at distance x from sample center.
  double (*eval)(double);
  // Filter support half-width in output-normalized units.
  double support;
} Filter;

// Pillow's bicubic filter (a = -0.5, support 2).
static double cubic(double x) {
  const double a = -0.5;
  x = fabs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Normalized cardinal sine sin(pi*x)/(pi*x) used by Lanczos-3.
static double sinc(double x) {
  static constexpr double PI = 3.14159265358979323846;
  if (x == 0.0) return 1.0;
  x *= PI;
  return sin(x) / x;
}

// Pillow's Lanczos-3 filter (support 3).
static double lanczos(double x) {
  x = fabs(x);
  if (x < 3.0) return sinc(x) * sinc(x / 3.0);
  return 0.0;
}

// Pillow's bilinear triangle filter (support 1).
static double bilinear(double x) {
  x = fabs(x);
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

// Built-in resampling filters matching Pillow's libImaging/Resample.c.
static const Filter FILTER_BICUBIC = {cubic, 2.0};
static const Filter FILTER_LANCZOS = {lanczos, 3.0};
static const Filter FILTER_BILINEAR = {bilinear, 1.0};

// Each output pixel's taps along one axis (precompute_coeffs and
// normalize_coeffs_8bpc): the first input pixel and their count in bounds,
// their weights in fixed point in k, ksize per pixel. The weights get the
// fraction bits that `weights` gives them, returned in *frac.
static int taps(int in, int out, Filter filter, Weights weights, int **bounds,
                int32_t **k, int *frac) {
  double scale = (double)in / out, fs = scale < 1 ? 1 : scale,
         support = filter.support * fs;
  int ksize = (int)ceil(support) * 2 + 1;
  double *w = xmalloc((size_t)out * ksize * sizeof *w), largest = 0;
  *bounds = xmalloc((size_t)out * 2 * sizeof **bounds);
  *k = xmalloc((size_t)out * ksize * sizeof **k);
  for (int xx = 0; xx < out; xx++) {
    double center = (xx + 0.5) * scale, ww = 0, ss = 1.0 / fs,
           *kx = w + (size_t)xx * ksize;
    int xmin = (int)(center - support + 0.5),
        xmax = (int)(center + support + 0.5);
    if (xmin < 0) xmin = 0;
    if (xmax > in) xmax = in;
    xmax -= xmin;
    for (int x = 0; x < xmax; x++)
      ww += kx[x] = filter.eval((x + xmin - center + 0.5) * ss);
    for (int x = 0; x < ksize; x++) {
      kx[x] = x >= xmax ? 0 : ww != 0 ? kx[x] / ww : kx[x];
      largest = fmax(largest, kx[x]);
    }
    (*bounds)[2 * xx] = xmin, (*bounds)[2 * xx + 1] = xmax;
  }
  *frac = FRAC;
  if (weights == WEIGHTS_TORCH)
    for (*frac = 0; *frac < TORCH_MOST_FRAC; ++*frac)
      if ((int)(0.5 + largest * (1 << (*frac + 1))) >= TORCH_WEIGHT_LIMIT)
        break;
  for (size_t i = 0; i < (size_t)out * ksize; i++)
    (*k)[i] = (int32_t)(w[i] < 0 ? -0.5 + w[i] * (1 << *frac)
                                 : 0.5 + w[i] * (1 << *frac));
  free(w);
  return ksize;
}

// Returns the sum v of weights with frac fraction bits as 8 bits, clipped.
static uint8_t clip8(int32_t v, int frac) {
  v >>= frac;
  return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

// ImagingResampleInner: along rows first, over the rows the column pass
// reads, then along columns; each pass rounds to 8 bits. The weights become
// integers as `weights` says.
static Rgb resample(Rgb s, int w, int h, Filter filter, Weights weights) {
  int *bx = 0, *by, fx = 0, fy;
  int32_t *kx = 0, *ky;
  int ksx = w != s.w ? taps(s.w, w, filter, weights, &bx, &kx, &fx) : 0,
      ksy = taps(s.h, h, filter, weights, &by, &ky, &fy);
  int y0 = by[0], y1 = by[2 * (h - 1)] + by[2 * (h - 1) + 1];
  Rgb t = s;
  if (w != s.w) {
    for (int y = 0; y < h; y++) by[2 * y] -= y0;
    t = (Rgb){w, y1 - y0, xmalloc((size_t)w * (y1 - y0) * 3)};
    for (int y = y0; y < y1; y++)
      for (int x = 0; x < w; x++) {
        const uint8_t *in = s.p + ((size_t)y * s.w + bx[2 * x]) * 3;
        const int32_t *k = kx + (size_t)x * ksx;
        int32_t a[3] = {1 << (fx - 1), 1 << (fx - 1), 1 << (fx - 1)};
        for (int i = 0; i < bx[2 * x + 1]; i++)
          for (int c = 0; c < 3; c++) a[c] += in[3 * i + c] * k[i];
        for (int c = 0; c < 3; c++)
          t.p[((size_t)(y - y0) * w + x) * 3 + c] = clip8(a[c], fx);
      }
  }
  Rgb o = t;
  if (h != s.h) {
    o = (Rgb){t.w, h, xmalloc((size_t)t.w * h * 3)};
    for (int y = 0; y < h; y++) {
      const int32_t *k = ky + (size_t)y * ksy;
      for (int x = 0; x < t.w; x++) {
        int32_t a[3] = {1 << (fy - 1), 1 << (fy - 1), 1 << (fy - 1)};
        for (int i = 0; i < by[2 * y + 1]; i++)
          for (int c = 0; c < 3; c++)
            a[c] += t.p[((size_t)(by[2 * y] + i) * t.w + x) * 3 + c] * k[i];
        for (int c = 0; c < 3; c++)
          o.p[((size_t)y * t.w + x) * 3 + c] = clip8(a[c], fy);
      }
    }
    if (t.p != s.p) free(t.p);
  } else if (t.p == s.p) {
    o.p = xmalloc((size_t)s.w * s.h * 3);
    memcpy(o.p, s.p, (size_t)s.w * s.h * 3);
  }
  free(bx), free(kx), free(by), free(ky);
  return o;
}

// Media.resize: a copy at the same size; an image over 100 times taller
// than wide shrinks along its height first.
static Rgb resize(Rgb s, int w, int h, Filter filter) {
  if (s.h > s.w * 100 && h < s.h) {
    Rgb t = resample(s, s.w, h, filter, WEIGHTS_PILLOW),
        o = resample(t, w, h, filter, WEIGHTS_PILLOW);
    free(t.p);
    return o;
  }
  return resample(s, w, h, filter, WEIGHTS_PILLOW);
}

uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height,
                          int out_width, int out_height) {
  return resize((Rgb){width, height, (uint8_t *)pixels}, out_width, out_height,
                FILTER_BICUBIC)
      .p;
}

uint8_t *image_resize_rgb_lanczos(const uint8_t *pixels, int width, int height,
                                  int out_width, int out_height) {
  return resize((Rgb){width, height, (uint8_t *)pixels}, out_width, out_height,
                FILTER_LANCZOS)
      .p;
}

uint8_t *image_resize_rgb_bilinear(const uint8_t *pixels, int width, int height,
                                   int out_width, int out_height) {
  return resize((Rgb){width, height, (uint8_t *)pixels}, out_width, out_height,
                FILTER_BILINEAR)
      .p;
}

uint8_t *image_resize_rgb_torchvision(const uint8_t *pixels, int width,
                                      int height, int out_width,
                                      int out_height) {
  return resample((Rgb){width, height, (uint8_t *)pixels}, out_width,
                  out_height, FILTER_BICUBIC, WEIGHTS_TORCH)
      .p;
}
