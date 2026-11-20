#ifndef EREMITE_ENGINE_MATTE_H
#define EREMITE_ENGINE_MATTE_H
#include <stddef.h>
#include <stdint.h>

// Alpha mattes of image outputs: keeping an original's pixels where a mask is
// opaque.

// Blends each of the pixels RGB pixels of edited toward original by the
// pixel's alpha in mask: 255 keeps the original, 0 keeps the edit, and values
// between mix the two.
void matte_keep_masked(uint8_t *edited, const uint8_t *original,
                       const uint8_t *mask, size_t pixels);

#endif
