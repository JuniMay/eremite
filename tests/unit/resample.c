#include "check.h"
#include "engine/resample.h"
#include <stdlib.h>
#include <string.h>

// Tests the bicubic resizes of RGB pixels. A resize to the same size copies
// the pixels. One color stays that color at any size. A resize along one axis
// keeps the lines that do not change along that axis. Near an edge, the
// filter's overshoot is clipped to 0 and 255, never wrapped around. Pillow's
// resize and torchvision's give their own libraries' pixels, which differ
// where their weights' precisions round differently.

// Pixels that differ everywhere: 7 x 5, and the same shrunk to 5 x 3 by
// Pillow 13 and by torchvision 0.29's antialiased resize of uint8 tensors.
static constexpr int MIXED_W = 7, MIXED_H = 5, SMALL_W = 5, SMALL_H = 3;
static const uint8_t pillow_small[SMALL_H * SMALL_W * 3] = {
    40, 91,  128, 198, 79,  121, 142, 130, 116, 135, 182, 63,  128, 165, 222,
    58, 106, 143, 176, 95,  136, 145, 133, 131, 151, 160, 86,  143, 183, 87,
    84, 121, 158, 67,  114, 151, 119, 107, 146, 170, 51,  103, 158, 209, 58};
static const uint8_t torchvision_small[SMALL_H * SMALL_W * 3] = {
    40, 91,  128, 198, 79,  121, 143, 130, 116, 135, 182, 63,  128, 165, 222,
    58, 106, 143, 176, 95,  136, 146, 133, 131, 151, 160, 85,  143, 183, 87,
    84, 121, 158, 67,  114, 151, 119, 107, 146, 170, 51,  102, 158, 209, 58};

// Returns whether all count pixels at p are the color c.
static int all_color(const uint8_t *p, int count, const uint8_t c[3]) {
  for (int i = 0; i < count; i++)
    if (memcmp(p + 3 * i, c, 3)) return 0;
  return 1;
}

int main(void) {
  // Pixels that differ everywhere.
  uint8_t mixed[MIXED_H * MIXED_W * 3];
  for (size_t i = 0; i < sizeof mixed; i++) mixed[i] = (uint8_t)(i * 37 + 11);
  uint8_t *same = image_resize_rgb(mixed, MIXED_W, MIXED_H, MIXED_W, MIXED_H),
          *same_torchvision = image_resize_rgb_torchvision(
              mixed, MIXED_W, MIXED_H, MIXED_W, MIXED_H);
  check(same != mixed && !memcmp(same, mixed, sizeof mixed) &&
            same_torchvision != mixed &&
            !memcmp(same_torchvision, mixed, sizeof mixed),
        "a resize to the same size copies the pixels into a new buffer");
  free(same), free(same_torchvision);
  uint8_t *small = image_resize_rgb(mixed, MIXED_W, MIXED_H, SMALL_W, SMALL_H),
          *small_torchvision = image_resize_rgb_torchvision(
              mixed, MIXED_W, MIXED_H, SMALL_W, SMALL_H);
  check(!memcmp(small, pillow_small, sizeof pillow_small),
        "Pillow's resize gives Pillow's pixels");
  check(!memcmp(small_torchvision, torchvision_small, sizeof torchvision_small),
        "torchvision's resize gives torchvision's pixels, 1 apart from "
        "Pillow's at 4 of them");
  free(small), free(small_torchvision);

  // One color.
  const uint8_t color[3] = {200, 30, 90};
  constexpr int PLAIN_W = 9, PLAIN_H = 7;
  uint8_t plain[PLAIN_H * PLAIN_W * 3];
  for (int i = 0; i < PLAIN_W * PLAIN_H; i++) memcpy(plain + 3 * i, color, 3);
  const int sizes[][2] = {{4, 3}, {13, 11}, {1, 1}, {9, 2}, {2, 7}};
  int kept = 1;
  for (size_t s = 0; s < sizeof sizes / sizeof *sizes; s++) {
    int w = sizes[s][0], h = sizes[s][1];
    uint8_t *out = image_resize_rgb(plain, PLAIN_W, PLAIN_H, w, h);
    kept &= all_color(out, w * h, color);
    free(out);
  }
  check(kept, "one color stays that color at smaller, larger and mixed sizes");

  // Columns and rows of their own colors.
  constexpr int LINES = 6, LENGTH = 8;
  uint8_t columns[LENGTH * LINES * 3], rows[LINES * LENGTH * 3];
  for (int i = 0; i < LENGTH; i++)
    for (int j = 0; j < LINES; j++) {
      const uint8_t line[3] = {(uint8_t)(40 * j), (uint8_t)(255 - 40 * j),
                               (uint8_t)(20 * j + 5)};
      memcpy(columns + 3 * (i * LINES + j), line, 3);
      memcpy(rows + 3 * (j * LENGTH + i), line, 3);
    }
  const int lengths[] = {3, 13};
  int columns_kept = 1, rows_kept = 1;
  for (size_t n = 0; n < sizeof lengths / sizeof *lengths; n++) {
    int length = lengths[n];
    uint8_t *tall = image_resize_rgb(columns, LINES, LENGTH, LINES, length);
    for (int i = 0; i < length; i++)
      columns_kept &= !memcmp(tall + 3 * i * LINES, columns, 3 * LINES);
    free(tall);
    uint8_t *wide = image_resize_rgb(rows, LENGTH, LINES, length, LINES);
    for (int j = 0; j < LINES; j++)
      rows_kept &=
          all_color(wide + 3 * j * length, length, rows + 3 * j * LENGTH);
    free(wide);
  }
  check(columns_kept, "a resize in height keeps columns of one color each");
  check(rows_kept, "a resize in width keeps rows of one color each");

  // An image over 100 times taller than wide shrinks along its height first.
  constexpr int NARROW_W = 2, NARROW_H = 300;
  uint8_t narrow[NARROW_H * NARROW_W * 3];
  for (int i = 0; i < NARROW_W * NARROW_H; i++)
    memcpy(narrow + 3 * i, color, 3);
  uint8_t *line = image_resize_rgb(narrow, NARROW_W, NARROW_H, 1, 10);
  check(all_color(line, 10, color),
        "a very tall image of one color keeps its color as it shrinks");
  free(line);

  // A black-to-white edge, resized to four times its width.
  constexpr int EDGE_W = 8, SCALE = 4;
  uint8_t edge[EDGE_W * 3];
  for (int x = 0; x < EDGE_W; x++)
    memset(edge + 3 * x, x < EDGE_W / 2 ? 0 : 255, 3);
  uint8_t *wide = image_resize_rgb(edge, EDGE_W, 1, EDGE_W * SCALE, 1);
  int dark = 1, bright = 1;
  for (int x = 0; x < EDGE_W * SCALE; x++) {
    // Where the output pixel's center falls among the input pixels.
    double at = (x + 0.5) / SCALE;
    if (at < EDGE_W / 2 - 1) dark &= wide[3 * x] < 64;
    if (at > EDGE_W / 2 + 1) bright &= wide[3 * x] > 191;
  }
  check(dark && bright,
        "near an edge, the overshoot is clipped to 0 and 255, not wrapped");
  free(wide);
  return failures != 0;
}
