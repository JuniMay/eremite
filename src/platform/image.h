#ifndef EREMITE_PLATFORM_IMAGE_H
#define EREMITE_PLATFORM_IMAGE_H
#include <stddef.h>
#include <stdint.h>

// Decodes an image file's bytes into rows of 8-bit RGB pixels, as Pillow's
// convert("RGB") gives them, and sets *width and *height; returns null with
// *error set when it cannot.
uint8_t *image_decode_rgb(const void *data, size_t len, int *width, int *height,
                          const char **error);

#endif
