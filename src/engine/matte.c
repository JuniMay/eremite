#include "engine/matte.h"

// Channels of an RGB pixel.
static constexpr size_t RGB_CHANNELS = 3;

// The alpha of an opaque pixel.
static constexpr unsigned OPAQUE = 255;

void matte_keep_masked(uint8_t *edited, const uint8_t *original,
                       const uint8_t *mask, size_t pixels) {
  for (size_t i = 0; i < pixels; i++) {
    unsigned keep = mask[i], change = OPAQUE - keep;
    for (size_t c = 0; c < RGB_CHANNELS; c++) {
      size_t at = i * RGB_CHANNELS + c;
      // Rounds to the nearest level.
      edited[at] =
          (uint8_t)((edited[at] * change + original[at] * keep + OPAQUE / 2) /
                    OPAQUE);
    }
  }
}
