#ifndef EREMITE_MODEL_MINIMAX_H3_QUANT_H
#define EREMITE_MODEL_MINIMAX_H3_QUANT_H
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

// Waits for the GPU command buffer numbered number to finish, yielding when
// called inside a cooperative engine pass.
void minimax_h3_gpu_wait(Gpu *gpu, uint64_t number);

#endif
