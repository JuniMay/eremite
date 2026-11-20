#include "engine/matte.h"
#include <stdio.h>
#include <string.h>

// Alpha mattes: keeping an original's pixels where a mask is opaque.

// The number of failed checks.
static int failures;

// Prints label with ok or FAIL by cond, and counts the failures.
static void check(int cond, const char *label) {
  printf("%s: %s\n", cond ? "ok" : "FAIL", label);
  if (!cond) failures++;
}

int main(void) {
  // Three pixels: the mask keeps the first, changes the second, and mixes the
  // third halfway.
  uint8_t edited[] = {10, 20, 30, 40, 50, 60, 200, 100, 0};
  const uint8_t original[] = {110, 120, 130, 140, 150, 160, 0, 100, 255};
  const uint8_t mask[] = {255, 0, 128};
  matte_keep_masked(edited, original, mask, 3);
  const uint8_t expected[] = {110, 120, 130, 40, 50, 60, 100, 100, 128};
  check(!memcmp(edited, expected, sizeof expected),
        "an opaque mask keeps the original, a clear one keeps the edit, and "
        "half alpha mixes them");

  // A 16 by 16 image: a green background with a few levels of noise, a red
  // square in the middle, and one pixel half green and half red at its edge.
  enum { SIDE = 16, SQUARE_FROM = 5, SQUARE_TO = 11 };

  uint8_t keyed[SIDE * SIDE * 3], rgba[SIDE * SIDE * 4];
  for (int y = 0; y < SIDE; y++)
    for (int x = 0; x < SIDE; x++) {
      uint8_t *px = keyed + (y * SIDE + x) * 3;
      int noise = (x * 7 + y * 3) % 5;
      int inside = x >= SQUARE_FROM && x < SQUARE_TO && y >= SQUARE_FROM &&
                   y < SQUARE_TO;
      px[0] = (uint8_t)(inside ? 200 : 10 + noise);
      px[1] = (uint8_t)(inside ? 30 : 236 + noise);
      px[2] = (uint8_t)(inside ? 40 : 12 + noise);
    }
  uint8_t *edge = keyed + (SQUARE_FROM * SIDE + SQUARE_FROM - 1) * 3;
  edge[0] = 105, edge[1] = 135, edge[2] = 26;
  const uint8_t green[3] = {0, 255, 0};
  int keyed_out = matte_remove_key(keyed, SIDE, SIDE, green, rgba);
  int background_clear = 1;
  for (int i = 0; i < SIDE * SIDE; i++) {
    int x = i % SIDE, y = i / SIDE;
    int inside =
        x >= SQUARE_FROM && x < SQUARE_TO && y >= SQUARE_FROM && y < SQUARE_TO;
    if (!inside && !(x == SQUARE_FROM - 1 && y == SQUARE_FROM))
      background_clear &= rgba[i * 4 + 3] == 0;
  }
  const uint8_t *center = rgba + ((SIDE / 2) * SIDE + SIDE / 2) * 4;
  const uint8_t *edge_out = rgba + (SQUARE_FROM * SIDE + SQUARE_FROM - 1) * 4;
  check(keyed_out && background_clear, "the key-colored background is clear");
  check(center[3] == 255 && center[0] == 200 && center[1] == 30 &&
            center[2] == 40,
        "the subject stays opaque with its own color");
  check(edge_out[3] > 0 && edge_out[3] < 255 &&
            edge_out[1] <=
                (edge_out[0] > edge_out[2] ? edge_out[0] : edge_out[2]),
        "a mixed edge pixel is partly clear and loses the key's lead");

  // A border without the key's color has nothing to remove.
  uint8_t red[4 * 4 * 3], red_rgba[4 * 4 * 4];
  for (int i = 0; i < 4 * 4; i++)
    red[i * 3] = 200, red[i * 3 + 1] = 30, red[i * 3 + 2] = 40;
  check(!matte_remove_key(red, 4, 4, green, red_rgba),
        "a border without the requested key color gives no matte");
  return failures != 0;
}
