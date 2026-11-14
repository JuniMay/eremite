#ifndef EREMITE_MODEL_DEEPSEEK_IMAGE_H
#define EREMITE_MODEL_DEEPSEEK_IMAGE_H
#include "engine/media.h"
#include <stddef.h>

// Images for the vision encoder, as inference/image_processor.py prepares
// them: the grid an image gets, gray padding (ImageOps.pad), the ViT's patches
// and 2D RoPE angles, and the tokens of the image's span in a prompt.

// Prepares the image file's len bytes at data in im: its grids, patches and
// hash. Returns 0, with *error set, if it cannot.
int image_open(Media *im, const void *data, size_t len, const char **error);

// Writes each patch's RoPE angles to cs: their 32 cosines, then their 32
// sines.
void image_rope(const Media *im, float *cs);

#endif
