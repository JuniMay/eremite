#include "check.h"
#include "model/lfm3/image.h"
#include "model/lfm3/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests the images for LFM3-VL's vision tower. An image over d1's pixel
// budget shrinks to it, a small one grows to the processor's fewest pixels,
// and one on the grid keeps its size. A side halfway between two multiples of
// the grid rounds to the even one, as Python rounds. The patches hold the
// processor's pixels: their FNV-1a hashes are those of LiquidAI's processor
// (Pillow 13, torchvision 0.29), each square of patches together, each frame
// the same picture. Sides that differ more than 200 times, and bytes that
// hold no image, are refused.

// The BMP file's header and its information header, in bytes.
static constexpr int BMP_HEADER = 14, BMP_INFO = 40;

// FNV-1a's 64-bit offset basis and prime.
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ull,
                          FNV_PRIME = 1099511628211ull;

// Each test image: its size, its grid of patches after the processor's
// resize, and the FNV-1a hash of its resized RGB pixels in LiquidAI's code.
static const struct {
  int width, height, grid_w, grid_h;
  uint64_t pixels_hash;
} images[] = {
    // over the budget: capped to 1810 x 905, then resized to 1824 x 896
    {2000, 1000, 114, 56, 14463176760598286730ull},
    // under the processor's fewest pixels: resized up to 320 x 224
    {200, 150, 20, 14, 12191839527666251017ull},
    // on the grid already: kept
    {512, 384, 32, 24, 10642681553009516710ull},
    // 12.5 and 6.5 squares of 32 pixels: rounded to 12 and 6
    {400, 208, 24, 12, 5321428952103445155ull},
};

// Writes v to p in little-endian order, in bytes bytes.
static void put_le(uint8_t *p, uint32_t v, int bytes) {
  for (int i = 0; i < bytes; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// Returns a 24-bit BMP file of width x height pixels and sets *len: red rises
// from left to right, green from top to bottom, and blue is a texture of
// sharp edges.
static uint8_t *make_bmp(int width, int height, size_t *len) {
  int row = (width * 3 + 3) / 4 * 4;
  size_t data = (size_t)row * height, start = BMP_HEADER + BMP_INFO;
  *len = start + data;
  uint8_t *f = calloc(*len, 1);
  f[0] = 'B', f[1] = 'M';
  put_le(f + 2, (uint32_t)*len, 4), put_le(f + 10, (uint32_t)start, 4);
  uint8_t *info = f + BMP_HEADER;
  put_le(info, BMP_INFO, 4), put_le(info + 4, (uint32_t)width, 4);
  put_le(info + 8, (uint32_t)height, 4), put_le(info + 12, 1, 2);
  put_le(info + 14, 24, 2), put_le(info + 20, (uint32_t)data, 4);
  // 2835 pixels per meter: 72 per inch.
  put_le(info + 24, 2835, 4), put_le(info + 28, 2835, 4);
  for (int y = 0; y < height; y++) {
    // The rows go from the bottom up, each pixel blue, green, red.
    uint8_t *line = f + start + (size_t)(height - 1 - y) * row;
    for (int x = 0; x < width; x++) {
      line[3 * x] = (uint8_t)((x / 7) * 37 + (y / 5) * 91 + (x * y) % 13 * 11);
      line[3 * x + 1] = (uint8_t)(y * 255 / (height - 1));
      line[3 * x + 2] = (uint8_t)(x * 255 / (width - 1));
    }
  }
  return f;
}

// Returns the pixel level that a patch's BF16 value holds.
static uint8_t level_of(uint16_t value) {
  uint32_t bits = (uint32_t)value << 16;
  float f;
  memcpy(&f, &bits, sizeof f);
  return (uint8_t)lrintf(f * 127.5f + 127.5f);
}

// Returns the FNV-1a hash of the image's pixels, read back from its patches
// in rows of RGB pixels, and sets *frames_agree to whether every frame of each
// patch holds the same picture.
static uint64_t hash_pixels(const Media *im, int *frames_agree) {
  int w = im->vw * LFM3_PATCH, h = im->vh * LFM3_PATCH,
      frame = LFM3_PATCH * LFM3_PATCH, squares_w = im->vw / LFM3_MERGE;
  uint64_t hash = FNV_OFFSET;
  *frames_agree = 1;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int gy = y / LFM3_PATCH, gx = x / LFM3_PATCH;
      size_t p = ((size_t)(gy / LFM3_MERGE) * squares_w + gx / LFM3_MERGE) *
                     LFM3_MERGE * LFM3_MERGE +
                 (size_t)(gy % LFM3_MERGE) * LFM3_MERGE + gx % LFM3_MERGE;
      const uint16_t *patch = im->patch + p * LFM3_PATCH_VALUES;
      int at = (y % LFM3_PATCH) * LFM3_PATCH + x % LFM3_PATCH;
      for (int c = 0; c < 3; c++) {
        uint16_t first = patch[c * LFM3_FRAMES * frame + at];
        for (int f = 1; f < LFM3_FRAMES; f++)
          *frames_agree &= patch[(c * LFM3_FRAMES + f) * frame + at] == first;
        hash = (hash ^ level_of(first)) * FNV_PRIME;
      }
    }
  return hash;
}

// Opens a BMP image of width x height pixels in im.
static int open_bmp(Media *im, int width, int height) {
  size_t len;
  uint8_t *bmp = make_bmp(width, height, &len);
  const char *error = 0;
  int opened = lfm3_image_open(im, bmp, len, &error);
  free(bmp);
  return opened && !error;
}

int main(void) {
  uint64_t file_hashes[sizeof images / sizeof *images];
  for (size_t i = 0; i < sizeof images / sizeof *images; i++) {
    Media im = {0};
    printf("a %d x %d image:\n", images[i].width, images[i].height);
    int opened = open_bmp(&im, images[i].width, images[i].height);
    check(opened, "it opens");
    if (!opened) continue;
    check(im.vw == images[i].grid_w && im.vh == images[i].grid_h,
          "its grid of patches is the processor's");
    check(im.tw == im.vw / LFM3_MERGE && im.th == im.vh / LFM3_MERGE &&
              im.ntokens == im.tw * im.th,
          "it takes a token for each square of patches");
    int frames_agree;
    uint64_t hash = hash_pixels(&im, &frames_agree);
    check(hash == images[i].pixels_hash,
          "its patches hold the pixels of LiquidAI's processor");
    check(frames_agree, "each frame of a patch holds the same picture");
    file_hashes[i] = im.hash;
    media_free(&im);
  }
  check(file_hashes[0] != file_hashes[1] && file_hashes[1] != file_hashes[2] &&
            file_hashes[2] != file_hashes[3],
        "the images' hashes follow their files' bytes");

  Media im = {0};
  check(!open_bmp(&im, 2010, 10), "sides that differ 201 times are refused");
  const char *error = 0;
  const uint8_t junk[] = "not an image";
  check(!lfm3_image_open(&im, junk, sizeof junk, &error) && error,
        "bytes that hold no image are refused, with a message");
  return failures != 0;
}
