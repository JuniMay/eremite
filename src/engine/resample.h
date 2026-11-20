#ifndef EREMITE_ENGINE_RESAMPLE_H
#define EREMITE_ENGINE_RESAMPLE_H
#include <stdint.h>

// Resizes of 8-bit RGB pixels: Pillow's bicubic, Lanczos-3 and bilinear
// filters (libImaging/Resample.c), and torchvision's bicubic for uint8 tensors
// (torch's upsample_bicubic2d_aa).

// Returns the width x height RGB pixels resized to out_width x out_height, in
// a new buffer, as Pillow resizes them with its bicubic filter.
uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height,
                          int out_width, int out_height);

// Returns the width x height RGB pixels resized to out_width x out_height, in
// a new buffer, as Pillow resizes them with its Lanczos-3 filter.
uint8_t *image_resize_rgb_lanczos(const uint8_t *pixels, int width, int height,
                                  int out_width, int out_height);

// Returns the width x height RGB pixels resized to out_width x out_height, in
// a new buffer, as Pillow resizes them with its bilinear filter.
uint8_t *image_resize_rgb_bilinear(const uint8_t *pixels, int width, int height,
                                   int out_width, int out_height);

// Returns the width x height RGB pixels resized to out_width x out_height, in
// a new buffer, as torchvision resizes a uint8 tensor with antialiasing:
// Pillow's bicubic filter, with 16-bit weights.
uint8_t *image_resize_rgb_torchvision(const uint8_t *pixels, int width,
                                      int height, int out_width,
                                      int out_height);

#endif
