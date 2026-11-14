#ifndef EREMITE_ENGINE_RESAMPLE_H
#define EREMITE_ENGINE_RESAMPLE_H
#include <stdint.h>

// Pillow's bicubic resize of 8-bit RGB pixels (libImaging/Resample.c).

// Returns the width x height RGB pixels resized to out_width x out_height, in
// a new buffer.
uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height,
                          int out_width, int out_height);

#endif
