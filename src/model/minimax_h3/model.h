#ifndef EREMITE_MODEL_MINIMAX_H3_MODEL_H
#define EREMITE_MODEL_MINIMAX_H3_MODEL_H
#include "model/model.h"
#include <stddef.h>

// MiniMax-H3-Image model integration: wraps the Qwen3-VL-32B NVFP4 text and
// vision encoders, pruned INT8-ConvRot 50-layer DiT, and FP16 24-channel VAE
// behind ModelOps for text-to-image generation and multi-reference editing.

// Maximum prompt tokens retained from the text prompt before encoding.
static constexpr int MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS = 4096;

// Spatial multiple in pixels required by the VAE (16x) and DiT (2x2 patch).
static constexpr int MINIMAX_H3_IMAGE_PATCH_PX = 32;

// Minimum and maximum values accepted for min_generation_area in pixels.
static constexpr int MINIMAX_H3_IMAGE_MIN_AREA = 1024,
                     MINIMAX_H3_IMAGE_MAX_AREA = 16777216;

// Default minimum generation canvas area in pixels (2048 by 2048).
static constexpr int MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA = 4194304;

// Maximum single-axis output or canvas dimension in pixels.
static constexpr int MINIMAX_H3_IMAGE_MAX_DIM = 8192;

// Maximum generation canvas area in pixels (24 megapixels).
static constexpr int64_t MINIMAX_H3_IMAGE_MAX_CANVAS_AREA = 24ll * 1024 * 1024;

// Maximum number of reference images accepted by one edit request.
static constexpr int MINIMAX_H3_IMAGE_MAX_REFS = 16;

// Default JPEG quality when an image request leaves quality unset.
static constexpr int MINIMAX_H3_IMAGE_DEFAULT_JPEG_QUALITY = 95;

// Vocabulary size of the Qwen3-VL tokenizer used by MiniMax-H3.
static constexpr int MINIMAX_H3_IMAGE_VOCAB = 151936;

// Computes the generation canvas dimensions (positive multiples of 32 pixels)
// for a requested output size and minimum generation area (0 uses the default).
// Returns 1 on success or 0 with a message in error when arguments are invalid.
int minimax_h3_image_plan_canvas(int width, int height, int min_area,
                                 int *out_canvas_w, int *out_canvas_h,
                                 char *error, size_t error_cap);

#endif
