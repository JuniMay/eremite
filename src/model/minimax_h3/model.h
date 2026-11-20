#ifndef EREMITE_MODEL_MINIMAX_H3_MODEL_H
#define EREMITE_MODEL_MINIMAX_H3_MODEL_H
#include "model/model.h"
#include <stddef.h>

// MiniMax-H3-Image model integration: wraps the Qwen3-VL-32B NVFP4 text and
// vision encoders, pruned INT8-ConvRot 50-layer DiT, and FP16 24-channel VAE
// behind ModelOps for text-to-image generation and multi-reference editing.

// Maximum prompt tokens retained from the text prompt before encoding.
static constexpr int MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS = 4096;

// Spatial multiple in pixels required by the VAE (16x) and DiT (2x2 patch);
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:32 (CANVAS_MULTIPLE = 32).
static constexpr int MINIMAX_H3_IMAGE_PATCH_PX = 32;

// Minimum and maximum values accepted for min_generation_area in pixels (one
// 32x32 patch up to 4096x4096).
static constexpr int MINIMAX_H3_IMAGE_MIN_AREA = 1024,
                     MINIMAX_H3_IMAGE_MAX_AREA = 16777216;

// Default minimum generation canvas area in pixels (2048 by 2048);
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:39 (IMAGE_SIZE = 2048).
static constexpr int MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA = 4194304;

// Maximum single-axis output or canvas dimension in pixels.
static constexpr int MINIMAX_H3_IMAGE_MAX_DIM = 8192;

// Maximum generation canvas area in pixels (24 megapixels).
static constexpr int64_t MINIMAX_H3_IMAGE_MAX_CANVAS_AREA = 24ll * 1024 * 1024;

// Maximum total output pixels across all images of one request (32 Mi pixels),
// bounding peak response PNG and base64 memory to ~224 MiB.
static constexpr int64_t MINIMAX_H3_IMAGE_MAX_REQUEST_PIXELS =
    32ll * 1024 * 1024;

// Maximum number of images returned by one request (n).
static constexpr int MINIMAX_H3_IMAGE_MAX_IMAGES = 16;

// Maximum number of reference images accepted by one edit request.
static constexpr int MINIMAX_H3_IMAGE_MAX_REFS = 16;

// Default JPEG quality when an image request leaves quality unset.
static constexpr int MINIMAX_H3_IMAGE_DEFAULT_JPEG_QUALITY = 95;

// Required inference step count of the distilled PDD checkpoint;
// MiniMaxAI/MiniMax-H3-Image pdd_config.json:2 ("num_inference_steps": 6).
static constexpr int MINIMAX_H3_IMAGE_STEPS = 6;

// Required guidance scale of the CFG-free distilled checkpoint.
static constexpr double MINIMAX_H3_IMAGE_GUIDANCE_SCALE = 1.0;

// Vocabulary size of the Qwen3-VL tokenizer used by MiniMax-H3;
// MiniMaxAI/MiniMax-H3-Image text_encoder/config.json:35 ("vocab_size").
static constexpr int MINIMAX_H3_IMAGE_VOCAB = 151936;

// Resident GPU byte requirements of the five MiniMax-H3-Image weight groups.
typedef struct {
  // Qwen3-VL text encoder weight bytes.
  uint64_t text;
  // Qwen3-VL vision tower weight bytes.
  uint64_t vision;
  // Pruned INT8-ConvRot DiT weight bytes.
  uint64_t dit;
  // FP16 2D ViT VAE decoder weight bytes.
  uint64_t vae;
  // FP16 VAE encoder weight bytes.
  uint64_t vae_encoder;
} MinimaxH3WeightBytes;

// Peak GPU bytes needed by one MiniMax-H3-Image request in full-residence and
// staged execution modes.
typedef struct {
  // Peak GPU bytes keeping all three core weight groups pinned together.
  uint64_t full_bytes;
  // Peak GPU bytes when weight groups swap between stages.
  uint64_t staged_bytes;
} MinimaxH3RequestMemory;

// Computes the generation canvas dimensions (positive multiples of 32 pixels)
// for a requested output size and minimum generation area (0 uses the default).
// Returns 1 on success or 0 with a message in error when arguments are invalid.
int minimax_h3_image_plan_canvas(int width, int height, int min_area,
                                 int *out_canvas_w, int *out_canvas_h,
                                 char *error, size_t error_cap);

// Computes full-residence and staged GPU memory peaks for a planned canvas
// (canvas_w by canvas_h), prompt token length text_len, reference vision token
// count vis_tokens, reference latent patch count ref_patches and reference
// count ref_count.
MinimaxH3RequestMemory
minimax_h3_image_plan_request_memory(MinimaxH3WeightBytes weights, int canvas_w,
                                     int canvas_h, int text_len, int vis_tokens,
                                     int ref_patches, int ref_count);

// Reserves GPU memory in account for a planned canvas before running the text
// encoder. Prefers full residence when full_resident is nonzero, falling back
// to staged execution if only staged_bytes fits. On failure writes a message
// naming the canvas and required GiB into error and leaves account->minimum
// unchanged. Returns 1 on success and sets *out_use_full, or 0 on failure.
int minimax_h3_image_admit_request(MemoryAccount *account, int full_resident,
                                   int canvas_w, int canvas_h,
                                   MinimaxH3RequestMemory req_mem,
                                   int *out_use_full, char *error,
                                   size_t error_cap);

#endif
