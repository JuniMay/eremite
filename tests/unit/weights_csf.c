#include "check.h"
#include "engine/weights.h"
#include <string.h>

// Tests the compressed scale format's decoding: row bases, the order of the
// nibbles, exceptions, and malformed streams.

int main(void) {
  enum { ROWS = 32, COLS = 40, STRIDE = 16 + 8 * COLS };

  uint8_t fixed[2 * STRIDE] = {0}, expected[ROWS * COLS], decoded[ROWS * COLS];
  for (int row = 0; row < ROWS; row++) {
    uint8_t *slab = fixed + row / 16 * STRIDE;
    slab[row % 16] = (uint8_t)(80 + row);
    for (int col = 0; col < COLS; col++) {
      unsigned code = (unsigned)(row * 7 + col * 3) % 16;
      slab[16 + row % 16 * (COLS / 2) + col / 2] |= code << (4 * (col % 2));
      expected[row * COLS + col] = (uint8_t)(80 + row + code);
    }
  }
  uint32_t exceptions[] = {(254u << 24), (3u << 24) | (17 * COLS + 39)};
  expected[0] = 254;
  expected[17 * COLS + 39] = 3;
  check(weight_group_decode_compressed_scales(decoded, fixed, sizeof fixed,
                                              exceptions, 2, ROWS, COLS) &&
            !memcmp(decoded, expected, sizeof expected),
        "row bases plus nibble offsets, with exceptions, decode to the scales");
  check(!weight_group_decode_compressed_scales(decoded, fixed, sizeof fixed - 1,
                                               exceptions, 2, ROWS, COLS),
        "a stream of the wrong size is refused");
  exceptions[1] = ROWS * COLS;
  check(!weight_group_decode_compressed_scales(decoded, fixed, sizeof fixed,
                                               exceptions, 2, ROWS, COLS),
        "an exception past the scales is refused");
  fixed[0] = 241;
  check(!weight_group_decode_compressed_scales(decoded, fixed, sizeof fixed,
                                               NULL, 0, ROWS, COLS),
        "a row base above 240 is refused");
  return failures != 0;
}
