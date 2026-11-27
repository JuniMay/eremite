#ifndef EREMITE_MODEL_NEMOTRON_MODEL_H
#define EREMITE_MODEL_NEMOTRON_MODEL_H

#include <stdint.h>

// NVIDIA Nemotron-3.5-Transcribe / Nemotron-H 52-layer hybrid backbone
// (model_type nemo_speechlm and nemotron_h):
// - 23 Mamba-2 SSM layers (per-tensor FP8 in_proj/out_proj, k=4 causal conv1d
//   + SiLU, FP32 selective SSM recurrence, group-512 gated RMSNorm)
// - 6 NoPE GQA 32/2 Attention layers (NVFP4 q/k/v/o_proj, head_dim=128)
// - 23 MoE layers (FP32 sigmoid+bias top-6 router, NVFP4 non-gated relu2
//   shared expert + 128 NVFP4 routed experts streamed via ExpertStore)
// - Optional Parallel Expert Perception audio encoder (perception.safetensors).

// Backbone geometry constants matching llm_backbone/config.json.
static constexpr int NEMO_LAYERS = 52;
static constexpr int NEMO_MAMBA_LAYERS = 23;
static constexpr int NEMO_ATTN_LAYERS = 6;
static constexpr int NEMO_MOE_LAYERS = 23;
static constexpr int NEMO_DIM = 2688;
static constexpr int NEMO_EH_DIM = 2 * NEMO_DIM;
static constexpr int NEMO_VOCAB = 131072;
static constexpr int NEMO_HEADS = 32;
static constexpr int NEMO_KV_HEADS = 2;
static constexpr int NEMO_HEAD_DIM = 128;
static constexpr int NEMO_Q_DIM = NEMO_HEADS * NEMO_HEAD_DIM;
static constexpr int NEMO_KV_DIM = NEMO_KV_HEADS * NEMO_HEAD_DIM;
static constexpr int NEMO_MAMBA_HEADS = 64;
static constexpr int NEMO_MAMBA_HEAD_DIM = 64;
static constexpr int NEMO_MAMBA_INNER = NEMO_MAMBA_HEADS * NEMO_MAMBA_HEAD_DIM;
static constexpr int NEMO_MAMBA_GROUPS = 8;
static constexpr int NEMO_MAMBA_STATE = 128;
static constexpr int NEMO_MAMBA_BC = NEMO_MAMBA_GROUPS * NEMO_MAMBA_STATE;
static constexpr int NEMO_MAMBA_CONV_DIM = NEMO_MAMBA_INNER + 2 * NEMO_MAMBA_BC;
static constexpr int NEMO_MAMBA_IN_DIM =
    NEMO_MAMBA_INNER + NEMO_MAMBA_CONV_DIM + NEMO_MAMBA_HEADS;
static constexpr int NEMO_MAMBA_CONV_K = 4;
static constexpr int NEMO_MAMBA_CONV_HIST = NEMO_MAMBA_CONV_K - 1;
static constexpr int NEMO_EXPERTS = 128;
static constexpr int NEMO_TOPK = 6;
static constexpr int NEMO_MOE_INTER = 1856;
static constexpr int NEMO_SHARED_INTER = 3712;
static constexpr int NEMO_NVFP4_BLOCK = 16;
static constexpr int NEMO_MAX_BATCH_ROWS = 8;
static constexpr int NEMO_MAX_VERIFY = 16;
static constexpr int NEMO_MTP_MAX_DRAFTS = 8;
// Fixed power-of-two token count per shared KV page (multiple of the 64-row
// FlashAttention tile so every tile stays contiguous inside one page).
static constexpr int NEMO_KV_PAGE_TOKENS = 256;
// Packed KV head slices per attention layer inside a page (k0, k1, v0, v1).
static constexpr int NEMO_KV_SLICES_PER_LAYER = 2 * NEMO_KV_HEADS;

// One grouped-expert work tile matching NemoMoeTile in nemotron.metal.
typedef struct {
  // Expert index (0..127), first pair index, pair count in tile, alignment pad.
  uint32_t expert, start, count, pad;
} NemoMoeTile;

// Groups `rows * NEMO_TOPK` routed (token, slot) pairs by expert index and
// splits each expert's pairs into tiles of at most `max_tile` pairs. Writes
// per-pair token index `ptok`, routing weight `pw`, inverse token-slot map
// `tp`, and tile list `tiles`. Returns the number of tiles written.
int nemotron_group_expert_tiles(const uint32_t *route_ids, const float *route_w,
                                int rows, int max_tile, int *ptok, float *pw,
                                int *tp, NemoMoeTile *tiles);

#endif
