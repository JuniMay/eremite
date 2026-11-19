#ifndef EREMITE_TESTS_BMP_H
#define EREMITE_TESTS_BMP_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Test images as BMP files: 24-bit pixels, which Pillow and ImageIO decode to
// the same values. Red rises from left to right, green from top to bottom,
// and blue is a texture of sharp edges.

// Writes v to p in little-endian order, in bytes bytes.
static inline void put_le(uint8_t *p, uint32_t v, int bytes) {
  for (int i = 0; i < bytes; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// Returns a BMP file of the test pattern at width x height pixels, and sets
// *len to its bytes.
static inline uint8_t *make_test_bmp(int width, int height, size_t *len) {
  // The file's header and its information header, in bytes.
  constexpr int header = 14, info = 40;
  int row = (width * 3 + 3) / 4 * 4;
  size_t data = (size_t)row * height, start = header + info;
  *len = start + data;
  uint8_t *f = calloc(*len, 1);
  f[0] = 'B', f[1] = 'M';
  put_le(f + 2, (uint32_t)*len, 4), put_le(f + 10, (uint32_t)start, 4);
  uint8_t *head = f + header;
  put_le(head, info, 4), put_le(head + 4, (uint32_t)width, 4);
  put_le(head + 8, (uint32_t)height, 4), put_le(head + 12, 1, 2);
  put_le(head + 14, 24, 2), put_le(head + 20, (uint32_t)data, 4);
  // 2835 pixels per meter: 72 per inch.
  put_le(head + 24, 2835, 4), put_le(head + 28, 2835, 4);
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

#endif
