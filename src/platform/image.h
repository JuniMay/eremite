#ifndef EREMITE_PLATFORM_IMAGE_H
#define EREMITE_PLATFORM_IMAGE_H
#include <stddef.h>
#include <stdint.h>

// The smallest and largest JPEG encoding quality values.
static constexpr int IMAGE_JPEG_QUALITY_MIN = 1, IMAGE_JPEG_QUALITY_MAX = 100;

// Decodes an image file's bytes into rows of 8-bit RGB pixels, as Pillow's
// convert("RGB") gives them, and sets *width and *height; returns null with
// *error set when it cannot.
uint8_t *image_decode_rgb(const void *data, size_t len, int *width, int *height,
                          const char **error);

// Encodes width by height rows of 8-bit sRGB pixels as a PNG file in a new
// buffer and sets *length; returns null with *error set when it cannot.
uint8_t *image_encode_png(const uint8_t *pixels, int width, int height,
                          size_t *length, const char **error);

// Encodes width by height rows of 8-bit RGB pixels as a JPEG file at quality
// (IMAGE_JPEG_QUALITY_MIN to IMAGE_JPEG_QUALITY_MAX) in a new buffer and sets
// *length; returns null with *error set when it cannot.
uint8_t *image_encode_jpeg(const uint8_t *pixels, int width, int height,
                           int quality, size_t *length, const char **error);

#endif
