#ifndef EREMITE_ENGINE_MATTE_H
#define EREMITE_ENGINE_MATTE_H
#include <stddef.h>
#include <stdint.h>

// Alpha mattes of image outputs: keeping an original's pixels where a mask is
// opaque, and making a flat key-colored background transparent.

// Blends each of the pixels RGB pixels of edited toward original by the
// pixel's alpha in mask: 255 keeps the original, 0 keeps the edit, and values
// between mix the two.
void matte_keep_masked(uint8_t *edited, const uint8_t *original,
                       const uint8_t *mask, size_t pixels);

// Blends each of the pixels RGBA pixels of edited, with straight alpha, toward
// the opaque RGB pixels of original by the pixel's alpha in mask: 255 keeps the
// original, opaque, 0 keeps the edit, and values between mix the two.
void matte_keep_masked_rgba(uint8_t *edited, const uint8_t *original,
                            const uint8_t *mask, size_t pixels);

// Writes into rgba the width by height RGBA pixels, with straight alpha, of
// rgb: an image generated on a flat background of the color requested, which
// becomes transparent. The key is the median color of the image's border, and
// it must lead in requested's strongest channel. A pixel's alpha comes from
// how far its color leads in that channel: pixels within the border's spread
// of the key's lead are transparent, and pixels with no lead are opaque. A
// partly transparent pixel's color is unmixed from the key. Returns 0 when the
// border shows no such key.
int matte_remove_key(const uint8_t *rgb, int width, int height,
                     const uint8_t requested[3], uint8_t *rgba);

#endif
