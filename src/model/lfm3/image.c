#include "model/lfm3/image.h"
#include "engine/resample.h"
#include "model/lfm3/model.h"
#include "platform/image.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// d1's pixel budget (config.json's d1_vision_max_pixels), the fewest and the
// most pixels of the processor's resize (processor_config.json's
// size.shortest_edge and size.longest_edge), and the largest ratio of an
// image's sides that smart_resize takes.
static constexpr int BUDGET_PIXELS = 1638400, RESIZE_MIN_PIXELS = 65536,
                     RESIZE_MAX_PIXELS = 16777216, MOST_SIDE_RATIO = 200;

// The side in pixels of a token's square of patches: the processor's sides
// are multiples of it.
static constexpr int SQUARE = LFM3_PATCH * LFM3_MERGE;

// The processor's pixels as values from -1 to 1: (value - CENTER) / CENTER.
static constexpr float CENTER = 127.5f;

// FNV-1a's 64-bit offset basis and prime: the hash of a file's bytes.
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ull,
                          FNV_PRIME = 1099511628211ull;

typedef struct {
  int w, h;
  uint8_t *p;
} Rgb;  // rows of w RGB pixels

// media.cap_pixels: an image of more than BUDGET_PIXELS pixels shrinks to fit,
// keeping its shape, with Pillow's bicubic resize. It frees s's pixels then.
static Rgb cap_pixels(Rgb s) {
  if ((double)s.w * s.h <= BUDGET_PIXELS) return s;
  double scale = sqrt(BUDGET_PIXELS / ((double)s.w * s.h));
  int w = (int)(s.w * scale), h = (int)(s.h * scale);
  w = w < 1 ? 1 : w, h = h < 1 ? 1 : h;
  Rgb o = {w, h, image_resize_rgb(s.p, s.w, s.h, w, h)};
  free(s.p);
  return o;
}

// smart_resize: the sides on multiples of SQUARE nearest to the image's
// width x height, as Python's round() rounds (halves to even), then scaled
// down to at most RESIZE_MAX_PIXELS or up to at least RESIZE_MIN_PIXELS.
// Returns 0 for sides that differ more than MOST_SIDE_RATIO times.
static int fit_sides(int width, int height, int *fit_w, int *fit_h) {
  int longer = width > height ? width : height,
      shorter = width > height ? height : width;
  if ((double)longer / shorter > MOST_SIDE_RATIO) return 0;
  int w = (int)rint((double)width / SQUARE) * SQUARE,
      h = (int)rint((double)height / SQUARE) * SQUARE;
  if ((double)w * h > RESIZE_MAX_PIXELS) {
    double beta = sqrt((double)height * width / RESIZE_MAX_PIXELS);
    w = (int)floor(width / beta / SQUARE) * SQUARE;
    h = (int)floor(height / beta / SQUARE) * SQUARE;
    w = w < SQUARE ? SQUARE : w, h = h < SQUARE ? SQUARE : h;
  } else if ((double)w * h < RESIZE_MIN_PIXELS) {
    double beta = sqrt(RESIZE_MIN_PIXELS / ((double)height * width));
    w = (int)ceil(width * beta / SQUARE) * SQUARE;
    h = (int)ceil(height * beta / SQUARE) * SQUARE;
  }
  *fit_w = w, *fit_h = h;
  return 1;
}

// Returns f rounded to BF16, to the nearest with ties to even.
static uint16_t round_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

int lfm3_image_open(Media *im, const void *data, size_t len,
                    const char **error) {
  Rgb s;
  s.p = image_decode_rgb(data, len, &s.w, &s.h, error);
  if (!s.p) return 0;
  s = cap_pixels(s);
  int w, h;
  if (!fit_sides(s.w, s.h, &w, &h)) {
    free(s.p);
    *error = "the image's sides differ more than 200 times";
    return 0;
  }
  uint8_t *rgb = w == s.w && h == s.h
                     ? s.p
                     : image_resize_rgb_torchvision(s.p, s.w, s.h, w, h);
  if (rgb != s.p) free(s.p);
  im->vh = h / LFM3_PATCH, im->vw = w / LFM3_PATCH;
  im->th = im->vh / LFM3_MERGE, im->tw = im->vw / LFM3_MERGE;
  im->ntokens = im->th * im->tw, im->rows = 0, im->start = 0;
  im->hash = FNV_OFFSET;
  for (size_t i = 0; i < len; i++)
    im->hash = (im->hash ^ ((const uint8_t *)data)[i]) * FNV_PRIME;
  size_t n = (size_t)im->vh * im->vw;
  im->patch = calloc(n * LFM3_PATCH_VALUES, sizeof *im->patch);
  if (!im->patch) {
    free(rgb);
    *error = "the image patches do not fit in memory";
    return 0;
  }
  // Patch p of the grid's row gy and column gx is in square
  // (gy / LFM3_MERGE, gx / LFM3_MERGE) of the rows of squares, at
  // (gy % LFM3_MERGE, gx % LFM3_MERGE) in it. A patch holds its channels in
  // turn, each as LFM3_FRAMES frames of LFM3_PATCH rows of LFM3_PATCH pixels.
  int squares_w = im->vw / LFM3_MERGE, frame = LFM3_PATCH * LFM3_PATCH;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int gy = y / LFM3_PATCH, gx = x / LFM3_PATCH;
      size_t p = ((size_t)(gy / LFM3_MERGE) * squares_w + gx / LFM3_MERGE) *
                     LFM3_MERGE * LFM3_MERGE +
                 (size_t)(gy % LFM3_MERGE) * LFM3_MERGE + gx % LFM3_MERGE;
      int at = (y % LFM3_PATCH) * LFM3_PATCH + x % LFM3_PATCH;
      for (int c = 0; c < 3; c++) {
        uint16_t v = round_to_bf16(
            ((float)rgb[((size_t)y * w + x) * 3 + c] - CENTER) / CENTER);
        for (int f = 0; f < LFM3_FRAMES; f++)
          im->patch[p * LFM3_PATCH_VALUES +
                    (size_t)(c * LFM3_FRAMES + f) * frame + at] = v;
      }
    }
  free(rgb);
  return 1;
}
