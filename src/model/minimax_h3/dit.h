#ifndef EREMITE_MODEL_MINIMAX_H3_DIT_H
#define EREMITE_MODEL_MINIMAX_H3_DIT_H
#include "engine/memory.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/minimax_h3/quant.h"
#include <stdint.h>

// MiniMax-H3-Image DiT: a 50-layer INT8-ConvRot single-stream transformer with
// a 2-layer BF16 token refiner, pruned curve-form AdaLN, partial 3D split-half
// RoPE and a 6-step Progressive Distillation (PDD) head bank.

// Maximum number of PDD sampling steps supported by the head bank.
static constexpr int MINIMAX_H3_DIT_MAX_PDD_STEPS = 16;

// One optional encoded reference image latent passed during image editing.
typedef struct {
  // Normalized channel-first FP32 latent [24, h_lat, w_lat].
  const float *latents;
  // Reference latent height and width in VAE cells (1/16 pixel dimensions).
  int h_lat, w_lat;
} MinimaxH3RefLatent;

// Loaded MiniMax-H3-Image DiT handle.
typedef struct MinimaxH3Dit MinimaxH3Dit;

// Opens the pruned INT8-ConvRot DiT safetensors file at weights_path and
// registers all weights in account. Does not load GPU buffers until requested.
MinimaxH3Dit *minimax_h3_dit_open(Gpu *gpu, MemoryAccount *account,
                                  const char *weights_path);

// Frees the DiT handle, its weight group and open safetensors file.
void minimax_h3_dit_close(MinimaxH3Dit *dit);

// Returns the DiT's weight group.
WeightGroup *minimax_h3_dit_weights(MinimaxH3Dit *dit);

// Returns the GPU bytes required to keep the DiT weights resident.
uint64_t minimax_h3_dit_required_bytes(const MinimaxH3Dit *dit);

// Returns the GPU bytes currently resident for the DiT weights.
uint64_t minimax_h3_dit_resident_bytes(const MinimaxH3Dit *dit);

// Returns the number of PDD sampling steps configured by the checkpoint.
int minimax_h3_dit_pdd_steps(const MinimaxH3Dit *dit);

// Loads the DiT weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_dit_load(MinimaxH3Dit *dit);

// Runs full PDD flow-matching sampling to produce a normalized channel-first
// latent [24, h_lat, w_lat] in out_latents.
// - text_hidden: FP32 GPU buffer of text_len by 5120 Qwen3-VL layer-49 states.
// - text_tags: optional text_len array of AdaLN modality tags (0=visual,
// 1=text);
//   when null, all text tokens receive tag 1.
// - refs: optional array of ref_count encoded reference image latents.
// - seed: deterministic RNG seed for initial latent noise (and ref noise aug).
// - hooks: optional cooperative yield and cancellation hooks; may be null.
// Returns 1 on success or 0 if loading, workspace allocation or stop fails.
int minimax_h3_dit_sample(MinimaxH3Dit *dit, const Buf *text_hidden,
                          int text_len, const int *text_tags,
                          const MinimaxH3RefLatent *refs, int ref_count,
                          int h_lat, int w_lat, uint64_t seed,
                          float *out_latents, const MinimaxH3RunHooks *hooks);

#endif
