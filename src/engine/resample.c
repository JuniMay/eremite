#include "engine/resample.h"
#include "base/error.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int w, h;
  uint8_t *p;
} Rgb;  // rows of w RGB pixels

#define FRAC 22  // PRECISION_BITS: the weights' fraction bits

static double cubic(double x) {  // a = -0.5, support 2
  const double a = -0.5;
  x = fabs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Each output pixel's taps along one axis (precompute_coeffs and
// normalize_coeffs_8bpc): the first input pixel and their count in bounds,
// their weights in fixed point in k, ksize per pixel.
static int taps(int in, int out, int **bounds, int32_t **k) {
  double scale = (double)in / out, fs = scale < 1 ? 1 : scale, support = 2 * fs;
  int ksize = (int)ceil(support) * 2 + 1;
  double *w = xmalloc((size_t)out * ksize * sizeof *w);
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
      ww += kx[x] = cubic((x + xmin - center + 0.5) * ss);
    for (int x = 0; x < ksize; x++)
      kx[x] = x >= xmax ? 0 : ww != 0 ? kx[x] / ww : kx[x];
    (*bounds)[2 * xx] = xmin, (*bounds)[2 * xx + 1] = xmax;
  }
  for (size_t i = 0; i < (size_t)out * ksize; i++)
    (*k)[i] = (int32_t)(w[i] < 0 ? -0.5 + w[i] * (1 << FRAC)
                                 : 0.5 + w[i] * (1 << FRAC));
  free(w);
  return ksize;
}

static uint8_t clip8(int32_t v) {
  v >>= FRAC;
  return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

// ImagingResampleInner: along rows first, over the rows the column pass
// reads, then along columns; each pass rounds to 8 bits.
static Rgb resample(Rgb s, int w, int h) {
  int *bx = 0, *by;
  int32_t *kx = 0, *ky;
  int ksx = w != s.w ? taps(s.w, w, &bx, &kx) : 0, ksy = taps(s.h, h, &by, &ky);
  int y0 = by[0], y1 = by[2 * (h - 1)] + by[2 * (h - 1) + 1];
  Rgb t = s;
  if (w != s.w) {
    for (int y = 0; y < h; y++) by[2 * y] -= y0;
    t = (Rgb){w, y1 - y0, xmalloc((size_t)w * (y1 - y0) * 3)};
    for (int y = y0; y < y1; y++)
      for (int x = 0; x < w; x++) {
        const uint8_t *in = s.p + ((size_t)y * s.w + bx[2 * x]) * 3;
        const int32_t *k = kx + (size_t)x * ksx;
        int32_t a[3] = {1 << (FRAC - 1), 1 << (FRAC - 1), 1 << (FRAC - 1)};
        for (int i = 0; i < bx[2 * x + 1]; i++)
          for (int c = 0; c < 3; c++) a[c] += in[3 * i + c] * k[i];
        for (int c = 0; c < 3; c++)
          t.p[((size_t)(y - y0) * w + x) * 3 + c] = clip8(a[c]);
      }
  }
  Rgb o = t;
  if (h != s.h) {
    o = (Rgb){t.w, h, xmalloc((size_t)t.w * h * 3)};
    for (int y = 0; y < h; y++) {
      const int32_t *k = ky + (size_t)y * ksy;
      for (int x = 0; x < t.w; x++) {
        int32_t a[3] = {1 << (FRAC - 1), 1 << (FRAC - 1), 1 << (FRAC - 1)};
        for (int i = 0; i < by[2 * y + 1]; i++)
          for (int c = 0; c < 3; c++)
            a[c] += t.p[((size_t)(by[2 * y] + i) * t.w + x) * 3 + c] * k[i];
        for (int c = 0; c < 3; c++)
          o.p[((size_t)y * t.w + x) * 3 + c] = clip8(a[c]);
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
static Rgb resize(Rgb s, int w, int h) {
  if (s.h > s.w * 100 && h < s.h) {
    Rgb t = resample(s, s.w, h), o = resample(t, w, h);
    free(t.p);
    return o;
  }
  return resample(s, w, h);
}

uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height,
                          int out_width, int out_height) {
  return resize((Rgb){width, height, (uint8_t *)pixels}, out_width, out_height)
      .p;
}
