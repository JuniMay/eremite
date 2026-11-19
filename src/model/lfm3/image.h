#ifndef EREMITE_MODEL_LFM3_IMAGE_H
#define EREMITE_MODEL_LFM3_IMAGE_H
#include "engine/media.h"
#include <stddef.h>

// Images for LFM3-VL's vision tower, as d1's media.cap_pixels and
// Lfm3VlImageProcessor prepare them: within d1's pixel budget (Pillow's
// bicubic resize), sides on multiples of a token's square of patches
// (torchvision's bicubic resize), and patches from -1 to 1, each square of
// patches together.

// Prepares the image file's len bytes at data in im: its grids, patches and
// hash. Returns 0, with *error set, if it cannot.
int lfm3_image_open(Media *im, const void *data, size_t len,
                    const char **error);

#endif
