#ifndef EREMITE_MODEL_INKLING2_MODEL_H
#define EREMITE_MODEL_INKLING2_MODEL_H

#include "model/model.h"
#include <stddef.h>
#include <stdint.h>

// thinkingmachines/Inkling-2-Mini (model_type inkling2_loop and
// inkling2_loop_text): looped 24-layer MoE language backbone with Block
// Attention Residuals, cross-pass message gating, static and dynamic short
// convolutions, hidden-conditioned relative position bias, per-token sinks,
// and mixed NVFP4/MXFP8 routed experts.

// Hidden dimension and layer topology (config.json text_config and
// loop_config).
static constexpr int INK2_DIM = 2304;
static constexpr int INK2_PHYS_LAYERS = 24;
static constexpr int INK2_CORE_START = 6;
static constexpr int INK2_CORE_END = 18;
static constexpr int INK2_CORE_LAYERS = INK2_CORE_END - INK2_CORE_START;
static constexpr int INK2_LOOP_PASSES = 3;
static constexpr int INK2_TOTAL_LAYERS =
    INK2_PHYS_LAYERS + (INK2_LOOP_PASSES - 1) * INK2_CORE_LAYERS;

// Block Attention Residuals span and maximum completed blocks
// (config.json text_config.n_layers_per_attn_res_block).
static constexpr int INK2_BLOCK_SPAN = 6;
static constexpr int INK2_MAX_BLOCKS = INK2_PHYS_LAYERS / INK2_BLOCK_SPAN;

// Vocabulary and special token IDs (config.json and tokenizer_config.json).
static constexpr int INK2_VOCAB_PADDED = 201024;
static constexpr int INK2_VOCAB = 200058;
static constexpr int INK2_EOS_TOKEN = 200006;
static constexpr int INK2_IMAGE_TOKEN = 200054;

// Attention, relative bias and short convolution geometry (config.json
// text_config).
static constexpr int INK2_HEADS = 24;
static constexpr int INK2_KV_HEADS = 8;
static constexpr int INK2_GQA = INK2_HEADS / INK2_KV_HEADS;
static constexpr int INK2_HEAD_DIM = 128;
static constexpr int INK2_Q_DIM = INK2_HEADS * INK2_HEAD_DIM;
static constexpr int INK2_KV_DIM = INK2_KV_HEADS * INK2_HEAD_DIM;
static constexpr int INK2_D_REL = 16;
static constexpr int INK2_R_DIM = INK2_HEADS * INK2_D_REL;
static constexpr int INK2_REL_EXTENT = 1024;
static constexpr int INK2_WINDOW = 1024;
static constexpr int INK2_SCONV_K = 4;
static constexpr int INK2_SCONV_HIST = INK2_SCONV_K - 1;
static constexpr int INK2_SCONV_DYN_DIM = INK2_KV_HEADS * INK2_SCONV_K;
static constexpr int INK2_GATE_RANK = 64;
static constexpr int INK2_MSG_COUNT = 4;

// Routed MoE geometry and quantization layout (config.json text_config and
// hf_quant_config.json: layers 0..19 NVFP4, layers 20..23 MXFP8).
static constexpr int INK2_EXPERTS = 128;
static constexpr int INK2_TOPK = 8;
static constexpr int INK2_INTER = 1152;
static constexpr int INK2_W13_ROWS = 2 * INK2_INTER;
static constexpr int INK2_NVFP4_LAYERS = 20;
static constexpr int INK2_NVFP4_BLOCK = 16;
static constexpr int INK2_MXFP8_BLOCK = 32;

// Logical attention cache counts across all 48 visits: 8 global visits
// (physical 5, 11x3, 17x3, 23) and 40 sliding-window visits.
static constexpr int INK2_GLOBAL_COUNT = 8;
static constexpr int INK2_SLIDE_COUNT = INK2_TOTAL_LAYERS - INK2_GLOBAL_COUNT;

// Batching, speculative verification and rolling-ring rewind capacities.
static constexpr int INK2_MAX_BATCH_ROWS = 8;
static constexpr int INK2_MAX_VERIFY = 16;
static constexpr int INK2_REWIND_MARGIN = 16;
static constexpr int INK2_SLIDE_RING = INK2_WINDOW + INK2_REWIND_MARGIN;
static constexpr int INK2_SCONV_RING = INK2_SCONV_HIST + INK2_REWIND_MARGIN;

// One logical layer visit in the 48-step unrolled execution schedule.
typedef struct {
  // Physical layer index in 0..23.
  int phys;
  // Loop pass index (0 outside the loop or on first core pass; 1 or 2 on
  // subsequent core passes).
  int loop_pass;
  // Index inside the core loop (0..11), or -1 for prelude and postlude layers.
  int loop_idx;
  // Nonzero for full-attention layers (physical 5, 11, 17, 23).
  int is_global;
  // Nonzero when this layer starts a new AttnRes block (physical 0, 6, 12, 18).
  int is_boundary;
  // Index into the 8 global KV buffers when is_global is nonzero, else -1.
  int global_slot;
  // Index into the 40 sliding KV rings when is_global is zero, else -1.
  int slide_slot;
} Ink2Visit;

// One grouped-expert work tile matching Ink2MoeTile in inkling2.metal.
typedef struct {
  // Expert index in 0..127.
  uint32_t expert;
  // First pair index in the grouped ptok/pw arrays.
  uint32_t start;
  // Number of pairs in this tile (1..64).
  uint32_t count;
  // Alignment padding.
  uint32_t pad;
} Ink2MoeTile;

// Populates the 48-entry unrolled execution schedule for Inkling-2-Mini.
void inkling2_build_schedule(Ink2Visit out[INK2_TOTAL_LAYERS]);

// Groups `rows * INK2_TOPK` routed (token, expert) pairs by expert into tiles
// of at most `max_tile` pairs. Writes per-pair token indices (`ptok`), routing
// weights (`pw`), inverse token-slot map (`tp`), and tile descriptors
// (`tiles`). Returns the number of tiles emitted.
int inkling2_group_expert_tiles(const uint32_t *route_ids, const float *route_w,
                                int rows, int max_tile, int *ptok, float *pw,
                                int *tp, Ink2MoeTile *tiles);

// ModelOps table for thinkingmachines/Inkling-2-Mini.
extern const ModelOps inkling2_ops;

#endif
