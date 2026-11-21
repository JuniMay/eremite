#ifndef EREMITE_MODEL_MINIMAX_H3_LORA_H
#define EREMITE_MODEL_MINIMAX_H3_LORA_H
#include "engine/memory.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <stdint.h>

// Optional BF16 LoRA adapter for MiniMax-H3 DiT and TokenRefiner projections.
// Deltas run at inference time as low-rank BF16 residuals
// y += (strength * alpha / rank) * (B @ (A @ x)) on unrotated activations so
// INT8-ConvRot base weights are never re-quantized.

// Number of TokenRefiner and DiT blocks that a MiniMax-H3 LoRA may patch.
static constexpr int MINIMAX_H3_LORA_REFINER_LAYERS = 2,
                     MINIMAX_H3_LORA_DIT_LAYERS = 50;

// Low-rank BF16 adapter weights for one linear projection.
typedef struct {
  // Down-projection BF16 matrix [rank, k], zero-padded along rank to a
  // multiple of 64.
  Buf a;
  // Up-projection BF16 matrix [n, rank], zero-padded along rank to a multiple
  // of 64.
  Buf b;
  // Padded rank, output width and input width.
  int rank, n, k;
  // Non-zero when this projection is present in the LoRA file.
  int present;
  // Checkpoint alpha divided by unpadded training rank.
  float scale;
} MinimaxH3LoraTarget;

// Adapter targets for one TokenRefiner or DiT transformer block.
typedef struct {
  // Fused QKV projection adapter.
  MinimaxH3LoraTarget qkv_proj;
  // Attention output projection adapter.
  MinimaxH3LoraTarget out_proj;
  // SwiGLU MLP gate/value projection adapter.
  MinimaxH3LoraTarget fc1;
  // SwiGLU MLP down-projection adapter.
  MinimaxH3LoraTarget fc2;
} MinimaxH3LoraBlock;

// Loaded MiniMax-H3 BF16 LoRA adapter handle.
typedef struct MinimaxH3Lora MinimaxH3Lora;

// Opens a ComfyUI-format MiniMax-H3 BF16 LoRA safetensors file at path with
// runtime multiplier strength (typically 1.0). Does not load GPU weight
// buffers until minimax_h3_lora_load is called.
MinimaxH3Lora *minimax_h3_lora_open(Gpu *gpu, MemoryAccount *account,
                                    const char *path, float strength);

// Frees the LoRA handle, its weight group and open safetensors file.
void minimax_h3_lora_close(MinimaxH3Lora *lora);

// Returns the LoRA's weight group.
WeightGroup *minimax_h3_lora_weights(MinimaxH3Lora *lora);

// Returns the GPU bytes required to keep the LoRA weights resident.
uint64_t minimax_h3_lora_required_bytes(const MinimaxH3Lora *lora);

// Returns the GPU bytes currently resident for the LoRA weights.
uint64_t minimax_h3_lora_resident_bytes(const MinimaxH3Lora *lora);

// Returns the maximum padded rank across all present targets, or 0 when empty.
int minimax_h3_lora_max_rank(const MinimaxH3Lora *lora);

// Returns the GPU bytes needed for the BF16 low-rank intermediate buffer when
// applying lora to seq_len rows (0 when lora is null or empty).
uint64_t minimax_h3_lora_workspace_bytes(const MinimaxH3Lora *lora,
                                         int seq_len);

// Loads the LoRA weights into GPU memory if not already resident. Returns 1 on
// success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_lora_load(MinimaxH3Lora *lora);

// Returns the adapter targets for TokenRefiner block index (0..1), or null
// when lora is null.
const MinimaxH3LoraBlock *
minimax_h3_lora_refiner_block(const MinimaxH3Lora *lora, int index);

// Returns the adapter targets for DiT block index (0..49), or null when lora
// is null.
const MinimaxH3LoraBlock *minimax_h3_lora_dit_block(const MinimaxH3Lora *lora,
                                                    int index);

// Encodes the low-rank BF16 residual y += (strength * scale) * (B @ (A @ x))
// into cmd for rows starting at byte offsets x_off in x_f32 and y_off in y.
// When xb_ready is non-zero, xb already holds padded BF16 rows of x; otherwise
// x_f32 is converted into xb first. lora_mid must hold padded_rows by
// target->rank BF16 values. No-op when lora or target is null or absent.
void minimax_h3_lora_apply_at(const MinimaxH3Lora *lora, GpuCommandBuffer *cmd,
                              const MinimaxH3LoraTarget *target,
                              const Buf *x_f32, size_t x_off, const Buf *xb,
                              const Buf *lora_mid, const Buf *y, size_t y_off,
                              int rows, int xb_ready);

// Encodes the low-rank BF16 residual y += (strength * scale) * (B @ (A @ x))
// into cmd at zero buffer offsets.
void minimax_h3_lora_apply(const MinimaxH3Lora *lora, GpuCommandBuffer *cmd,
                           const MinimaxH3LoraTarget *target, const Buf *x_f32,
                           const Buf *xb, const Buf *lora_mid, const Buf *y,
                           int rows, int xb_ready);

#endif
