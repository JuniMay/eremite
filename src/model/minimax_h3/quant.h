#ifndef EREMITE_MODEL_MINIMAX_H3_QUANT_H
#define EREMITE_MODEL_MINIMAX_H3_QUANT_H
#include "engine/pass.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include <stdint.h>

// Quantization helpers for MiniMax-H3 checkpoints: unswizzling cuBLASLt
// blocked NVFP4 E4M3 scale tables and reading small checkpoint tensors without
// filling the OS page cache.

// Block height and width of cuBLASLt's swizzled NVFP4 scale tile.
static constexpr int MINIMAX_H3_SCALE_TILE_ROWS = 128,
                     MINIMAX_H3_SCALE_TILE_COLS = 4;

// Number of weights that share one NVFP4 block scale.
static constexpr int MINIMAX_H3_NVFP4_GROUP = 16;

// Number of input channels rotated together by ConvRot's regular Hadamard.
static constexpr int MINIMAX_H3_CONVROT_GROUP = 256;

// Color channels in one 8-bit sRGB pixel (red, green, blue).
static constexpr int MINIMAX_H3_RGB_CHANNELS = 3;

// Spatial-temporal position axes for 3D RoPE and M-RoPE (time, height, width).
static constexpr int MINIMAX_H3_POS_AXES = 3;

// Projections packed in one fused QKV linear output (query, key, value).
static constexpr int MINIMAX_H3_QKV_PARTS = 3;

// Maximum value of one unsigned 8-bit pixel channel.
static constexpr float MINIMAX_H3_PIXEL_MAX = 255.0f;

// Stack buffer length in bytes for formatting checkpoint tensor names.
static constexpr size_t MINIMAX_H3_NAME_CAP = 128;

// Elements processed per vectorized float4/bfloat4/half4 load or store.
static constexpr uint32_t MINIMAX_H3_VEC4_WIDTH = 4;

// Fills count floats with deterministic standard normal N(0, 1) draws seeded
// by seed, using SplitMix64 uniform draws and the Box-Muller transform.
void minimax_h3_fill_gaussian(float *out, size_t count, uint64_t seed);

// Unswizzles a cuBLASLt blocked E4M3 scale matrix into row-major order of rows
// by cols bytes. blocked must hold ceil(rows / 128) * 128 by
// ceil(cols / 4) * 4 bytes.
void minimax_h3_unswizzle_nvfp4_scales(uint8_t *out, const uint8_t *blocked,
                                       int rows, int cols);

// Reads all bytes of tensor into out using sector-aligned uncached reads.
void minimax_h3_read_tensor_bytes(Tensor tensor, void *out);

// Reads a single F32 scalar tensor from disk and returns its value.
float minimax_h3_read_scalar_f32(Tensor tensor);

// Adds a swizzled E4M3 NVFP4 scale tensor to group, unswizzled into row-major
// rows by cols bytes in target when loaded.
void minimax_h3_nvfp4_scales_add(WeightGroup *group, Buf *target, Tensor tensor,
                                 int rows, int cols);

// Cooperative scheduling, cancellation and progress hooks for MiniMax-H3 GPU
// loops.
typedef struct {
  // Optional wait checked between GPU layers to let decoding calls run first.
  const PassWait *yield_wait;
  // Optional pointer to a nonzero flag set on the engine thread to cancel.
  const int *cancelled;
  // Optional pointer updated with the running 1-based text-encoder layer.
  int *prefill_layer;
  // Optional pointer updated with the running text-encoder token count.
  int *prefill_rows_now;
  // Optional counter increased by the encoded tokens after the text encoder
  // finishes.
  uint64_t *prefill_rows_total;
  // Optional pointer updated with the running 1-based denoising step.
  int *image_step;
  // Optional pointer updated with the active denoising step count.
  int *image_steps_total;
  // Optional counter increased after each finished denoising step.
  uint64_t *image_steps_completed;
  // Optional: runs after each denoising step but the last, with the latent
  // that the step predicts for the run's end and the number of finished
  // steps. It returns 0 to stop the run.
  int (*on_estimate)(void *context, const float *latents, int steps_done);
  // The context passed to on_estimate.
  void *estimate_context;
} MinimaxH3RunHooks;

// Returns nonzero when process shutdown was requested or hooks request cancel.
int minimax_h3_should_stop(const MinimaxH3RunHooks *hooks);

// Waits for the GPU command buffer numbered number to finish, yielding when
// called inside a cooperative engine pass.
void minimax_h3_gpu_wait(Gpu *gpu, uint64_t number);

// Waits for command buffer number to finish and then yields to
// hooks->yield_wait when running inside an engine pass without a stop request.
void minimax_h3_gpu_wait_hooks(Gpu *gpu, uint64_t number,
                               const MinimaxH3RunHooks *hooks);

#endif
