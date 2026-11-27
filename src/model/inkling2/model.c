#include "model/inkling2/model.h"
#include "model/inkling2/mtp.h"
#include "model/inkling2/vision.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "base/shutdown.h"
#include "engine/expert_store.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// thinkingmachines/Inkling-2-Mini (model_type inkling2_loop and
// inkling2_loop_text): weights, SSD-backed expert store, KV and
// short-convolution state, snapshots and forward passes.
//
// The checkpoint stores 24 physical decoder layers. Prelude layers 0..5 run
// once, core layers 6..17 run 3 times with cross-pass message retrieval and
// low-rank message gates on passes 1 and 2, and postlude layers 18..23 run
// once, yielding 48 logical layer visits. Each logical visit owns its own KV
// cache (8 growing global caches, 40 rolling sliding-window rings) and its own
// 4 short-convolution rings. Dense backbone weights stay resident in
// text_weights, while the 24 physical layers' 3,072 routed experts live in an
// ExpertStore (preloaded when memory fits, or streamed from SSD on demand).

// Numerical parameters from config.json text_config and modeling_inkling2.py.
static constexpr float RMS_EPS = 1e-6f;
static constexpr float GATE_SOFTCAP = 4.0f;
static constexpr float UP_SOFTCAP = 25.0f;
static constexpr float ROUTE_SCALE = 8.0f;
static constexpr float LOGIT_SOFTCAP = 30.0f;
static constexpr double LOG_SCALING_ALPHA = 0.1;
static constexpr int LOG_SCALING_FLOOR = 8192;
static constexpr double QK_L2_CLIP = 22.627416997969522;
static constexpr double REL_L2_CLIP = 8.0;

// Prefill chunk size, KV allocation growth, FlashDecoding split bounds and
// aligned direct-read threshold.
static constexpr int PREFILL_CHUNK = 512;
static constexpr int START_CAP = 256;
static constexpr int MAX_DECODE_PARTS = 64;
static constexpr int DECODE_PART_TARGET = 128;
static constexpr int DECODE_PART_FLOATS = INK2_HEAD_DIM + 2;
static constexpr int PREFILL_Q_PAD = 64;
static constexpr int PREFILL_Q_GROUP = 64;
// Rows dequantized per FP8 prefill slab: 2,048 rows of BF16 K and V take 8 MiB
// total (hot in SLC/L2) and hold an entire 1,040-row sliding ring in one pass.
static constexpr int FP8_PREFILL_SLAB_ROWS = 2048;
static constexpr int MM_TILE_M = 64;
static constexpr int MM_TILE_N = 64;
static constexpr int MOE_MM_TILE = 64;
static constexpr int MOE_MV_MAX_M = 8;
static constexpr int MOE_MV_ROWS_PER_TG = 32;
static constexpr int LOGITS_STRIDE = 200064;
static constexpr int DEFAULT_CTX = 8192;
static constexpr size_t ALIGN_READ_MIN_BYTES = 1u << 20;
static constexpr size_t ALIGN_MASK = 15u;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr size_t PASS_STACK_BYTES = 4u << 20;
static constexpr size_t PATH_CAP = 1024;

// Metal threadgroup sizes matching src/kernels/inkling2.metal and
// src/kernels/kernels.metal.
static constexpr uint32_t ROW_THREADS = 256;
static constexpr uint32_t QKVR_THREADS = 128;
static constexpr uint32_t QKVR_HEADS =
    INK2_HEADS + 2 * INK2_KV_HEADS + INK2_HEADS;
static constexpr uint32_t DECODE_THREADS = 128;
static constexpr uint32_t COMBINE_THREADS = 32;
static constexpr uint32_t PREFILL_THREADS = 128;
static constexpr uint32_t ROUTE_THREADS = 128;
static constexpr uint32_t MV_THREADS = 256;
static constexpr uint32_t MM_THREADS = 128;
static constexpr uint32_t ELEM_THREADS = 256;
static constexpr uint32_t SCONV_THREADS = 64;
static constexpr uint32_t VEC4_WIDTH = 4;
static constexpr uint32_t BF16_MV_SPLIT_TINY_N = 128;
static constexpr uint32_t BF16_MV_SPLIT_SMALL_N = 256;
static constexpr uint32_t BF16_MV_SPLIT_MID_N = 1024;
static constexpr uint32_t BF16_MV_SPLIT_MIN_K = 2048;
static constexpr uint32_t BF16_MV_SPLIT_TINY = 8;
static constexpr uint32_t BF16_MV_SPLIT_SMALL = 4;
static constexpr uint32_t BF16_MV_SPLIT_MID = 2;
static constexpr uint32_t BF16_MV_ROWS_PER_SG = 4;
static constexpr uint32_t BF16_MV_SGS = 8;
static constexpr uint32_t ATTNRES_GLOBAL_QUERIES = 3;
static constexpr int MAX_CHAT_IMAGES = 16;
static constexpr int MAX_THINK_PERCENT = 99;
static constexpr double THINK_PERCENT_SCALE = 100.0;

// Expert store layout, packing ratio, prefetch threshold and route inspection
// source codes matching src/eremite.h.
static constexpr int EXPERT_PARTS = 4;
static constexpr int NVFP4_VALUES_PER_BYTE = 2;
static constexpr int NEAR_MISSES = INK2_TOPK;
static constexpr int PREFILL_BULK_ROWS = 64;
static constexpr int ROUTE_DECODE_CAP = INK2_MAX_VERIFY;
static constexpr uint8_t ROUTE_SOURCE_MEMORY = 0;
static constexpr uint8_t ROUTE_SOURCE_AHEAD = 1;
static constexpr uint8_t ROUTE_SOURCE_LATE = 2;
static constexpr uint8_t ROUTE_SOURCE_SSD = 4;

// Byte layout of fixed FP32 short-convolution rings and logits inside one
// Ink2State.
static constexpr size_t CONV_KV_RING_BYTES =
    (size_t)INK2_SCONV_RING * INK2_KV_DIM * sizeof(float);
static constexpr size_t CONV_DIM_RING_BYTES =
    (size_t)INK2_SCONV_RING * INK2_DIM * sizeof(float);
static constexpr size_t CONV_K_OFF = 0;
static constexpr size_t CONV_V_OFF = CONV_KV_RING_BYTES;
static constexpr size_t CONV_ATTN_OFF = 2 * CONV_KV_RING_BYTES;
static constexpr size_t CONV_MLP_OFF =
    2 * CONV_KV_RING_BYTES + CONV_DIM_RING_BYTES;
static constexpr size_t CONV_LAYER_BYTES =
    2 * CONV_KV_RING_BYTES + 2 * CONV_DIM_RING_BYTES;
static constexpr size_t CONV_TOTAL_BYTES =
    (size_t)INK2_TOTAL_LAYERS * CONV_LAYER_BYTES;
static constexpr size_t LOGITS_BYTES = (size_t)INK2_VOCAB * sizeof(float);
static constexpr int SCONV_RINGS_PER_LAYER = 4;
// Snapshot metadata flags stored in meta[1] alongside the snapshot position.
static constexpr int32_t SNAP_FLAG_MTP = 1;
static constexpr int32_t SNAP_FLAG_FP8_KV = 2;

// Kernel parameter structs matching src/kernels/kernels.metal.
typedef struct {
  // Output rows, input columns, group row count, group column offset, BF16
  // rounding flag and split-K factor.
  uint32_t n, k, group_rows, group_k, round, split;
} MvParam;

typedef struct {
  // Base matrix-vector parameters.
  MvParam mv;
  // Batch rows, input row stride and output row stride in floats.
  uint32_t m, lda, ldy;
} MvmParam;

typedef struct {
  // Active rows, output width, inner width, BF16 rounding flag and strides.
  uint32_t m, n, k, round, lda, ldy;
} MmParam;

// Kernel parameter structs matching src/kernels/inkling2.metal.
typedef struct {
  // Row width in floats and BF16 rounding flag.
  uint32_t dim, round;
  // RMSNorm epsilon.
  float eps;
} NormParam;

typedef struct {
  // Active rows, block row stride, hidden width, completed blocks, queries.
  uint32_t rows, block_stride, dim, nblocks, nqueries;
  // Output row stride between query results.
  uint32_t out_row_stride;
  // Unweighted RMSNorm epsilon.
  float eps;
} AttnResParam;

typedef struct {
  // Active rows, hidden width, gate rank and row strides for 3-way Q/K/V gate.
  uint32_t rows, dim, rank, h_stride_rows, m_stride_rows, distinct_h;
  // Unweighted RMSNorm epsilon.
  float eps;
} Msg3Param;

typedef struct {
  // Active rows, hidden width and bottleneck gate rank.
  uint32_t rows, dim, rank;
  // Unweighted RMSNorm epsilon.
  float eps;
} MsgGateParam;

typedef struct {
  // Active rows, channel width, ring capacity, row-0 pos, BF16 round flag.
  uint32_t rows, dim, ring_rows, pos0, round;
} SconvParam;

typedef struct {
  // Device GPU address of the state's FP32 ring buffer.
  uint64_t ring;
  // Token position of this row and alignment padding.
  uint32_t pos, pad;
} SconvTarget;

typedef struct {
  // Device GPU addresses of the state's K and V buffers (BF16 or packed FP8).
  uint64_t k, v;
  // Token position and ring capacity (0 for global linear KV).
  uint32_t pos, ring_rows;
} KvTarget;

typedef struct {
  // Active rows, head-major Q stride, row-0 pos, ring capacity, flags.
  uint32_t rows, q_stride, pos0, ring_rows, is_global, batched;
} QkvrParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, ring capacity, window, partitions, flag.
  uint32_t rows, q_stride, pos0, ring_rows, window, nparts, part_stride,
      batched;
} AttendDecodeParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, ring capacity and sliding window.
  uint32_t rows, q_stride, pos0, ring_rows, window;
} AttendPrefillParam;

typedef struct {
  // First source FP8 KV row and number of rows to unpack to BF16.
  uint32_t start_row, n_rows;
} Fp8DequantParam;

typedef struct {
  // Query geometry, global key slab bounds and first/last slab flags.
  uint32_t rows, q_stride, pos0, k_start, k_count, is_first, is_last;
} AttendSlabParam;

typedef struct {
  // Active rows, total experts and top-k count.
  uint32_t rows, experts, topk;
  // Routing weight multiplier.
  float scale;
} RouteParam;

// Output channels processed per threadgroup in single-token fused MoE down.
static constexpr uint32_t MOE_MV_DOWN_M1_ROWS = 4;

typedef struct {
  // Output width, inner width, current token position and ring capacity.
  uint32_t n, k, pos, ring_rows;
} MoeDownM1Param;

typedef struct {
  // Matrix output rows, inner columns, active pair count and tile count.
  uint32_t n, k, npairs, ntiles;
  // Gate and up softcap bounds.
  float gate_cap, up_cap;
} MoeParam;

typedef struct {
  // Active rows, hidden width and top-k count.
  uint32_t rows, dim, topk;
} MoeCombineParam;

typedef struct {
  // Batch rows (1..8), unpadded vocab size, input width and output stride.
  uint32_t m, vocab, k, ldy;
  // Logit softcap bound and its reciprocal.
  float cap, inv_cap;
} UnembedParam;

typedef struct {
  // Active rows, unpadded vocab size and row stride in floats.
  uint32_t rows, vocab, stride;
  // Logit softcap bound.
  float cap;
} SoftcapParam;

// One BF16 linear weight matrix of shape [n, k].
typedef struct {
  // Raw BF16 weight buffer in GPU memory.
  Buf w;
  // Output rows and input columns.
  int n, k;
} Bf16Lin;

// Cross-pass message controller weights for one loop sub-stream.
typedef struct {
  // Low-rank down projection [64, 4608] in BF16.
  Bf16Lin down;
  // Forget gate projection [2304, 64] in BF16.
  Buf forget_w;
  // Gain gate projection [2304, 64] in BF16.
  Buf gain_w;
} MsgController;

// All weights for one of the 24 physical decoder layers.
typedef struct {
  // Nonzero for full-attention layers (5, 11, 17, 23).
  int is_global;
  // Nonzero for core looped layers (6..17).
  int is_loop;
  // Nonzero for NVFP4 layers (0..19); zero for MXFP8 layers (20..23).
  int is_nvfp4;

  // AttnRes queries: attn_query ([2304] or [3, 2304] FP32; unused on layer 0)
  // and mlp_query ([2304] FP32).
  Buf attn_query, mlp_query;
  // Pre-attention RMSNorm(s): attn_norm on sliding layers, or q/k/v norms on
  // global layers, plus mlp_norm (all FP32 [2304]).
  Buf attn_norm, q_norm, k_norm, v_norm, mlp_norm;
  // Cross-pass message controllers (valid when is_loop is nonzero).
  MsgController ctrl_q, ctrl_k, ctrl_v, ctrl_mlp;

  // Attention linear projections in BF16.
  Bf16Lin wq, wk, wv, wr, gate_attn, sink_proj, wo;
  // Relative position bias projection transposed to BF16 [1024, 16] and clip
  // scales (FP32).
  Buf rel_proj_t, q_clip_scale, rel_clip_scale;
  // Key/value short convolution weights: static [1024, 4] FP32 on sliding
  // layers, or static_w [4, 1024] FP32 + weight_proj [32, 2304] BF16 on global.
  Buf k_sconv_w, v_sconv_w;
  Bf16Lin k_sconv_proj, v_sconv_proj;
  // Post-attention and post-MLP static short convolution filters FP32 [2304,4].
  Buf attn_sconv_w, mlp_sconv_w;

  // Router weights (BF16 [128, 2304]) and FP32 bias ([128]).
  Bf16Lin router_w;
  Buf router_bias;
  // Precomputed FP32 per-expert scale factors for NVFP4 layers (0..19). The
  // packed expert weights and block scales live in experts_store.
  Buf f13, f2;
} PhysLayer;

// Per-expert GPU device addresses matching Ink2ExpertRef in inkling2.metal.
typedef struct {
  // Packed w13 weights, w13 scales, w2 weights and w2 scales.
  uint64_t w13, s13, w2, s2;
} ExpertRef;

// Routed experts of one physical layer held by the active pass until its GPU
// work completes.
typedef struct {
  // Physical layer index and count of held experts.
  int layer, n;
  // Held expert indices within `layer`.
  int experts[INK2_EXPERTS];
} HeldExperts;

// Per-sequence attention and short-convolution state.
typedef struct {
  // Allocated global KV capacity in tokens.
  int cap;
  // Growing global K and V caches (BF16 or packed FP8) for the 8 global visits.
  Buf gk[INK2_GLOBAL_COUNT], gv[INK2_GLOBAL_COUNT];
  // Rolling K and V rings (BF16 or packed FP8) for all 40 sliding visits.
  Buf slide_kv;
  // Rolling FP32 short-convolution rings for all 48 logical visits.
  Buf conv_ring;
  // Optional MTP draft head context KV and Markov feature caches.
  Ink2MtpState mtp;
} Ink2State;

// Saved rolling state (sliding KV rings, sconv rings, optional MTP rolling
// rings and optional logits) at one sequence position.
typedef struct {
  // Number of positions already processed when this snapshot was taken.
  int pos;
  // Host buffer holding slide_kv, conv_ring and optional MTP rolling rings.
  uint8_t *rows;
  // Optional host buffer of INK2_VOCAB FP32 logits, or null.
  float *logits;
} Ink2Snapshot;

// Per-pass reusable GPU scratch buffers.
typedef struct {
  // Row capacity currently allocated across row-scaled buffers.
  int cap_rows;
  // Padded head-major Q row stride (multiple of 16).
  int q_stride;
  // Row capacity currently allocated in `logits`.
  int logits_rows;
  // Token IDs, residual streams, AttnRes block stacks and cross-pass states.
  Buf ids, cur, cur2, residual, pre_loop_cur, prev_blk, prev_cur;
  Buf blk, msg, attn_in, q_in, k_in, v_in;
  // Loop message gate scratch buffers.
  Buf msg_cat, msg_g, msg_gated;
  // Attention projections, short-conv outputs and FlashDecoding partials.
  Buf q_raw, k_raw, k_conv, v_raw, v_conv, dyn_w, r_raw, r_out, gate, sink;
  Buf q_hm, part, attn_out;
  // Reusable BF16 K/V dequantization slabs and multi-slab softmax state when
  // fp8_kv is enabled.
  Buf deq_k, deq_v, slab_ml;
  // MoE routing, tile grouping, per-pass expert table and intermediates.
  Buf route_logits, route_ids, route_w, ptok, pw, tp, tiles, expert_table;
  Buf x_bf16, h_bf16, y_pair;
  // Final output logits buffer (stride LOGITS_STRIDE floats per row).
  Buf logits;
  // Batched per-row device-address target tables for multi-state decoding.
  Buf sconv_tgt, kv_tgt;
  // Optional MTP tap capture, context injection and block drafting buffers.
  Ink2MtpWork mtp;
} Ink2Work;

// Parameters passed to the NVFP4 per-expert scale factor fill callback.
typedef struct {
  // Per-expert amax FP32 [128] and outer scale denominator FP32 [1].
  Tensor amax, denom;
} FactorFillParam;

// Parameters passed to the cross-pass message query merge callback.
typedef struct {
  // Original [24, 2304] Q/MLP and K/V cross-pass query tensors.
  Tensor q_mlp, kv;
} MsgQueryFillParam;

// Global model handles, loaded dense weights and SSD-backed expert store.
static Gpu *gpu;
static MemorySlot *work_slot, state_slot;
static TensorIndex *checkpoint;
static WeightGroup *text_weights;
static ExpertStore *experts_store;
static FileSpan expert_spans[INK2_PHYS_LAYERS][INK2_EXPERTS][EXPERT_PARTS];
static int expert_span_counts[INK2_PHYS_LAYERS];
static Buf layer_table[INK2_PHYS_LAYERS];
static uint64_t layer_table_gen[INK2_PHYS_LAYERS];
static int store_started;
static Options opt;
static int eos_token = INK2_EOS_TOKEN;
static Ink2Visit schedule[INK2_TOTAL_LAYERS];
static PhysLayer layers[INK2_PHYS_LAYERS];
static Buf embed_w, embed_norm, msg_queries, output_query, final_norm,
    unembed_w;
static uint64_t required_bytes;
static int prepare_prev_rows;
static uint64_t prepare_prev_payload;

// Runtime counters and latest decode routes for server status reporting.
static uint64_t decode_rows_count, prefill_rows_count;
static double decode_seconds_total, decode_gpu_seconds_total;
static int prefill_layer_now, prefill_rows_now;
static ExpertStoreCounts decode_stats, prefill_stats;
static uint16_t route_id[ROUTE_DECODE_CAP][INK2_TOTAL_LAYERS][INK2_TOPK];
static uint8_t route_from[ROUTE_DECODE_CAP][INK2_TOTAL_LAYERS][INK2_TOPK];
static uint64_t route_step[ROUTE_DECODE_CAP][INK2_TOTAL_LAYERS], route_steps;

// Pass-local state swapped automatically across concurrent passes.
static Ink2Work work;
static uint64_t last_committed;
static float *logits_compact;
static int logits_compact_rows;
static const Media *prefill_media;
static int prefill_media_count;
static HeldExperts held;
static uint32_t held_layers;

// Compiled Metal kernel handles.
static int K_mv_bf16, K_mv_bf16_2, K_mv_bf16_3, K_mvm_bf16, K_mvm_bf16_2,
    K_mvm_bf16_3, K_mm_bf16;
static int K_embed_norm, K_rmsnorm, K_rmsnorm_3, K_attn_res, K_attn_res_rmsnorm,
    K_msg_cat_norm, K_msg_cat_norm_3, K_msg_down_3, K_msg_gate_apply,
    K_msg_gate_apply_3;
static int K_sconv_static, K_sconv_static_batched, K_sconv_static_post,
    K_sconv_static_post_batched, K_sconv_dynamic, K_sconv_dynamic_batched,
    K_sconv_ring_save;
static int K_qkvr_prep, K_qkvr_prep_fp8, K_sconv_qkvr_m1, K_sconv_qkvr_fp8_m1,
    K_attend_part_rel, K_attend_part_fp8_rel, K_attend_combine,
    K_attend_prefill_rel, K_fp8_dequant_kv, K_attend_prefill_slab_rel;
static int K_route_topk, K_route_topk_m1, K_moe_group, K_moe_gather_bf16;
static int K_nv_up_m1, K_nv_up_mv, K_nv_dn_mv, K_nv_dn_comb_m1, K_nv_up_mm,
    K_nv_dn_mm;
static int K_mx_up_m1, K_mx_up_mv, K_mx_dn_mv, K_mx_dn_comb_m1, K_mx_up_mm,
    K_mx_dn_mm;
static int K_moe_combine, K_copy, K_add, K_unembed_softcap_m1,
    K_unembed_softcap, K_logit_softcap;

// ---- schedule and tile grouping ----

// Builds the 48-visit unrolled schedule: 6 prelude layers, 3 passes over core
// layers 6..17, and 6 postlude layers.
void inkling2_build_schedule(Ink2Visit out[INK2_TOTAL_LAYERS]) {
  int v = 0, gslot = 0, sslot = 0;
  for (int p = 0; p < INK2_CORE_START; p++) {
    int is_g = (p % INK2_BLOCK_SPAN) == (INK2_BLOCK_SPAN - 1);
    out[v++] = (Ink2Visit){.phys = p,
                           .loop_pass = 0,
                           .loop_idx = -1,
                           .is_global = is_g,
                           .is_boundary = (p % INK2_BLOCK_SPAN) == 0,
                           .global_slot = is_g ? gslot++ : -1,
                           .slide_slot = is_g ? -1 : sslot++};
  }
  for (int pass = 0; pass < INK2_LOOP_PASSES; pass++) {
    for (int li = 0; li < INK2_CORE_LAYERS; li++) {
      int p = INK2_CORE_START + li;
      int is_g = (p % INK2_BLOCK_SPAN) == (INK2_BLOCK_SPAN - 1);
      out[v++] = (Ink2Visit){.phys = p,
                             .loop_pass = pass,
                             .loop_idx = li,
                             .is_global = is_g,
                             .is_boundary = (p % INK2_BLOCK_SPAN) == 0,
                             .global_slot = is_g ? gslot++ : -1,
                             .slide_slot = is_g ? -1 : sslot++};
    }
  }
  for (int p = INK2_CORE_END; p < INK2_PHYS_LAYERS; p++) {
    int is_g = (p % INK2_BLOCK_SPAN) == (INK2_BLOCK_SPAN - 1);
    out[v++] = (Ink2Visit){.phys = p,
                           .loop_pass = 0,
                           .loop_idx = -1,
                           .is_global = is_g,
                           .is_boundary = (p % INK2_BLOCK_SPAN) == 0,
                           .global_slot = is_g ? gslot++ : -1,
                           .slide_slot = is_g ? -1 : sslot++};
  }
}

// Groups routed (token, expert) pairs by expert into tiles of at most max_tile
// pairs.
int inkling2_group_expert_tiles(const uint32_t *route_ids, const float *route_w,
                                int rows, int max_tile, int *ptok, float *pw,
                                int *tp, Ink2MoeTile *tiles) {
  int counts[INK2_EXPERTS] = {0};
  int offsets[INK2_EXPERTS];
  int npairs = rows * INK2_TOPK;
  for (int p = 0; p < npairs; p++) {
    uint32_t e = route_ids[p];
    if (e >= (uint32_t)INK2_EXPERTS) die("router selected expert %u", e);
    counts[e]++;
  }
  int cursor = 0, ntiles = 0;
  for (int e = 0; e < INK2_EXPERTS; e++) {
    offsets[e] = cursor;
    int left = counts[e], start = cursor;
    while (left > 0) {
      int take = left < max_tile ? left : max_tile;
      tiles[ntiles++] = (Ink2MoeTile){.expert = (uint32_t)e,
                                      .start = (uint32_t)start,
                                      .count = (uint32_t)take,
                                      .pad = 0};
      start += take;
      left -= take;
    }
    cursor += counts[e];
  }
  for (int p = 0; p < npairs; p++) {
    uint32_t e = route_ids[p];
    int pos = offsets[e]++;
    ptok[pos] = p / INK2_TOPK;
    pw[pos] = route_w[p];
    tp[p] = pos;
  }
  return ntiles;
}

// ---- synchronization and buffer helpers ----

// Commits a command buffer and records it as this pass's latest submission.
static void model_commit(GpuCommandBuffer *c) {
  last_committed = c->number;
  gpu_commit(c);
}

// Returns whether the command buffer with the given sequence number finished.
static int committed_finished(void *number) {
  return gpu_has_finished(gpu, (uint64_t)(uintptr_t)number);
}

// Blocks until the command buffer with the given sequence number finishes.
static void committed_wait(void *number) {
  gpu_wait_until_finished(gpu, (uint64_t)(uintptr_t)number);
}

// Waits for this pass's GPU work, yielding when inside a cooperative pass.
static void model_sync(void) {
  if (!pass_current()) {
    gpu_sync(gpu);
    return;
  }
  if (last_committed)
    pass_wait(&(PassWait){committed_finished, committed_wait,
                          (void *)(uintptr_t)last_committed, PASS_WAIT_GPU,
                          NULL});
  gpu_check_finished(gpu);
}

// Ends the program if a pass yields with an uncommitted command buffer open in
// the queue, since it would block every command buffer begun after it.
static void before_yield(void) {
  if (gpu && gpu_count_uncommitted_in_queue(gpu))
    die("a pass yields with a command buffer open in the queue");
}

// Allocates a resident buffer counted in the workspace slot.
static Buf work_alloc(size_t size) {
  Buf b = gpu_alloc(gpu, work_slot, size);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Allocates a resident buffer counted in the state slot, returning empty on
// memory refusal.
static Buf state_try_alloc(size_t size) {
  Buf b = gpu_try_alloc_required(gpu, &state_slot, size);
  if (!b.obj) return b;
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Rounds n up to a multiple of align (a power of two).
static int align_up(int n, int align) { return (n + align - 1) & ~(align - 1); }

// Returns the byte width of one KV cache row for the active format.
static size_t kv_row_bytes(void) {
  return opt.inkling2.fp8_kv ? INK2_FP8_KV_ROW_BYTES : INK2_BF16_KV_ROW_BYTES;
}

// Returns the byte size of one sliding-window K or V ring.
static size_t slide_kv_one_bytes(void) {
  return (size_t)INK2_SLIDE_RING * kv_row_bytes();
}

// Returns the total byte size of all 40 sliding-window K and V rings.
static size_t slide_kv_bytes(void) {
  return (size_t)INK2_SLIDE_COUNT * 2 * slide_kv_one_bytes();
}

// Returns the byte offset of sliding visit slot's K or V ring in slide_kv.
static size_t slide_ring_off(int slide_slot, int is_v) {
  return ((size_t)slide_slot * 2 + (size_t)is_v) * slide_kv_one_bytes();
}

// Returns the byte offset of logical visit v's sub-ring in conv_ring.
static size_t conv_ring_off(int visit, size_t sub_off) {
  return (size_t)visit * CONV_LAYER_BYTES + sub_off;
}

// Reads all bytes of a small checkpoint tensor directly via pread.
static void read_tensor_exact(Tensor t, void *dst) {
  size_t done = 0;
  while (done < t.size) {
    ssize_t n = pread(t.fd, (uint8_t *)dst + done, t.size - done,
                      (off_t)(t.off + done));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("cannot read Inkling-2-Mini tensor bytes");
    done += (size_t)n;
  }
}

// ---- config validation and weight loading ----

// Reads a required JSON number field from object, ending the process if absent.
static double json_req_num(const Json *obj, const char *key, const char *path) {
  const Json *v = json_get(obj, key);
  if (!v || v->type != J_NUM) die("%s: missing number %s", path, key);
  return v->num;
}

// Validates that config.json matches Inkling-2-Mini's expected geometry.
static void check_config(const char *dir) {
  char path[PATH_CAP];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *tc = json_get(root, "text_config");
  if (!tc) tc = root;
  const Json *eos = json_get(root, "eos_token_id");
  if (eos && eos->type == J_NUM) eos_token = (int)eos->num;

  if ((int)json_req_num(tc, "hidden_size", path) != INK2_DIM ||
      (int)json_req_num(tc, "num_hidden_layers", path) != INK2_PHYS_LAYERS ||
      (int)json_req_num(tc, "vocab_size", path) != INK2_VOCAB_PADDED ||
      (int)json_req_num(tc, "unpadded_vocab_size", path) != INK2_VOCAB ||
      (int)json_req_num(tc, "num_attention_heads", path) != INK2_HEADS ||
      (int)json_req_num(tc, "num_key_value_heads", path) != INK2_KV_HEADS ||
      (int)json_req_num(tc, "head_dim", path) != INK2_HEAD_DIM ||
      (int)json_req_num(tc, "d_rel", path) != INK2_D_REL ||
      (int)json_req_num(tc, "rel_extent", path) != INK2_REL_EXTENT ||
      (int)json_req_num(tc, "sliding_window_size", path) != INK2_WINDOW ||
      (int)json_req_num(tc, "sconv_kernel_size", path) != INK2_SCONV_K ||
      (int)json_req_num(tc, "n_layers_per_attn_res_block", path) !=
          INK2_BLOCK_SPAN ||
      (int)json_req_num(tc, "n_routed_experts", path) != INK2_EXPERTS ||
      (int)json_req_num(tc, "num_experts_per_tok", path) != INK2_TOPK ||
      (int)json_req_num(tc, "intermediate_size", path) != INK2_INTER ||
      (int)json_req_num(tc, "log_scaling_n_floor", path) != LOG_SCALING_FLOOR)
    die("%s: Inkling-2-Mini text_config dimensions do not match", path);

  if (fabs(json_req_num(tc, "log_scaling_alpha", path) - LOG_SCALING_ALPHA) >
          1e-6 ||
      fabs(json_req_num(tc, "qk_l2_clip_threshold", path) - QK_L2_CLIP) >
          1e-5 ||
      fabs(json_req_num(tc, "rel_l2_clip_threshold", path) - REL_L2_CLIP) >
          1e-5 ||
      fabs(json_req_num(tc, "mlp_gate_softcap", path) - GATE_SOFTCAP) > 1e-5 ||
      fabs(json_req_num(tc, "mlp_up_softcap", path) - UP_SOFTCAP) > 1e-5 ||
      fabs(json_req_num(tc, "route_scale", path) - ROUTE_SCALE) > 1e-5 ||
      fabs(json_req_num(tc, "final_logit_softcapping", path) - LOGIT_SOFTCAP) >
          1e-5)
    die("%s: Inkling-2-Mini text_config scaling constants do not match", path);

  const Json *lc = json_get(tc, "loop_config");
  if (!lc || (int)json_req_num(lc, "core_start", path) != INK2_CORE_START ||
      (int)json_req_num(lc, "core_end", path) != INK2_CORE_END ||
      (int)json_req_num(lc, "num_passes", path) != INK2_LOOP_PASSES ||
      (int)json_req_num(lc, "gate_rank", path) != INK2_GATE_RANK)
    die("%s: Inkling-2-Mini loop_config does not match", path);

  json_free(root);
  free(text);
}

// Adds a raw tensor to text_weights, enabling aligned direct reads when large
// and 16-byte aligned in the safetensors file.
static void add_raw(Buf *dst, Tensor t) {
  int aligned =
      t.size >= ALIGN_READ_MIN_BYTES && (t.off & ALIGN_MASK) == 0 ? 1 : 0;
  weight_group_add_tensor(text_weights, dst, t, aligned);
}

// Adds a BF16 linear projection tensor [n, k] to text_weights and validates its
// dtype and shape.
static void add_bf16_lin(Bf16Lin *lin, Tensor t, int n, int k) {
  if (t.dtype != 'b' || t.ndim != 2 || t.shape[0] != n || t.shape[1] != k)
    die("unexpected BF16 linear tensor shape (%lldx%lld, expected %dx%d)",
        (long long)t.shape[0], (long long)t.shape[1], n, k);
  lin->n = n;
  lin->k = k;
  add_raw(&lin->w, t);
}

// Computes the per-expert NVFP4 FP32 scale factors amax[e] / denom[0].
static void fill_nvfp4_factor(Buf *buf, const void *ctx) {
  const FactorFillParam *p = ctx;
  float amax[INK2_EXPERTS], denom = 1.0f;
  read_tensor_exact(p->amax, amax);
  read_tensor_exact(p->denom, &denom);
  float *dst = (float *)buf->ptr;
  for (int e = 0; e < INK2_EXPERTS; e++) dst[e] = amax[e] / denom;
}

// Merges middle_cycle_cross_pass_queries ([12, 2, 2304] Q/MLP) and
// middle_cycle_cross_pass_kv_queries ([12, 2, 2304] K/V) into [12, 4, 2304]
// in [Q, K, V, MLP] order matching Inkling2LoopMergeMessageQueries.
static void fill_msg_queries(Buf *buf, const void *ctx) {
  const MsgQueryFillParam *p = ctx;
  size_t half_floats = (size_t)INK2_CORE_LAYERS * 2 * INK2_DIM;
  float *qm = xmalloc(half_floats * sizeof(float));
  float *kv = xmalloc(half_floats * sizeof(float));
  read_tensor_exact(p->q_mlp, qm);
  read_tensor_exact(p->kv, kv);
  float *dst = (float *)buf->ptr;
  for (int li = 0; li < INK2_CORE_LAYERS; li++) {
    float *out = dst + (size_t)li * INK2_MSG_COUNT * INK2_DIM;
    const float *q_src = qm + (size_t)(li * 2) * INK2_DIM;
    const float *mlp_src = qm + (size_t)(li * 2 + 1) * INK2_DIM;
    const float *k_src = kv + (size_t)(li * 2) * INK2_DIM;
    const float *v_src = kv + (size_t)(li * 2 + 1) * INK2_DIM;
    memcpy(out + (size_t)0 * INK2_DIM, q_src, INK2_DIM * sizeof(float));
    memcpy(out + (size_t)1 * INK2_DIM, k_src, INK2_DIM * sizeof(float));
    memcpy(out + (size_t)2 * INK2_DIM, v_src, INK2_DIM * sizeof(float));
    memcpy(out + (size_t)3 * INK2_DIM, mlp_src, INK2_DIM * sizeof(float));
  }
  free(qm);
  free(kv);
}

// Transposes rel_logits_proj.proj from BF16 [16, 1024] to BF16 [1024, 16] so
// attention kernels compute dot(r, rel_proj_t[dist]) with coalesced 32-byte
// reads without materializing a [rows, 24, 1024] bias tensor.
static void fill_rel_proj_t(Buf *buf, const void *ctx) {
  const Tensor *src_t = ctx;
  size_t count = (size_t)INK2_D_REL * INK2_REL_EXTENT;
  uint16_t *raw = xmalloc(count * sizeof(uint16_t));
  read_tensor_exact(*src_t, raw);
  uint16_t *dst = (uint16_t *)buf->ptr;
  for (int j = 0; j < INK2_D_REL; j++) {
    for (int d = 0; d < INK2_REL_EXTENT; d++) {
      dst[(size_t)d * INK2_D_REL + (size_t)j] =
          raw[(size_t)j * INK2_REL_EXTENT + (size_t)d];
    }
  }
  free(raw);
}

// Registers one loop layer's message controller tensors in text_weights.
static void add_controller(MsgController *c, int l, const char *name) {
  add_bf16_lin(&c->down,
               tensor_index_get(
                   checkpoint, "model.llm.layers.%d.controllers.%s.down_weight",
                   l, name),
               INK2_GATE_RANK, 2 * INK2_DIM);
  add_raw(&c->forget_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.controllers.%s.forget_weight",
                           l, name));
  add_raw(&c->gain_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.controllers.%s.gain_weight", l,
                           name));
}

// Registers every text backbone weight tensor in text_weights.
static void register_text_weights(void) {
  add_raw(&embed_w, tensor_index_get(checkpoint, "model.llm.embed.weight"));
  weight_group_add_vector(
      text_weights, &embed_norm,
      tensor_index_get(checkpoint, "model.llm.embed_norm.weight"), 0.0f, 0);
  weight_group_add_vector(
      text_weights, &output_query,
      tensor_index_get(checkpoint, "model.llm.attn_res.output_query"), 0.0f, 0);
  weight_group_add_vector(text_weights, &final_norm,
                          tensor_index_get(checkpoint, "model.llm.norm.weight"),
                          0.0f, 0);
  add_raw(&unembed_w, tensor_index_get(checkpoint, "model.llm.unembed.weight"));

  MsgQueryFillParam mqp = {
      .q_mlp = tensor_index_get(checkpoint,
                                "model.llm.middle_cycle_cross_pass_queries"),
      .kv = tensor_index_get(checkpoint,
                             "model.llm.middle_cycle_cross_pass_kv_queries")};
  weight_group_add_generated(text_weights, &msg_queries,
                             (size_t)INK2_CORE_LAYERS * INK2_MSG_COUNT *
                                 INK2_DIM * sizeof(float),
                             fill_msg_queries, &mqp, sizeof mqp);

  for (int l = 0; l < INK2_PHYS_LAYERS; l++) {
    PhysLayer *pl = &layers[l];
    pl->is_global = (l % INK2_BLOCK_SPAN) == (INK2_BLOCK_SPAN - 1);
    pl->is_loop = (l >= INK2_CORE_START && l < INK2_CORE_END);
    pl->is_nvfp4 = (l < INK2_NVFP4_LAYERS);

    // attn_queries.{l - 1} maps to physical layer l (layer 0 has none).
    if (l > 0)
      weight_group_add_vector(
          text_weights, &pl->attn_query,
          tensor_index_get(checkpoint, "model.llm.attn_res.attn_queries.%d",
                           l - 1),
          0.0f, 0);
    weight_group_add_vector(
        text_weights, &pl->mlp_query,
        tensor_index_get(checkpoint, "model.llm.attn_res.mlp_queries.%d", l),
        0.0f, 0);

    if (pl->is_global) {
      weight_group_add_vector(
          text_weights, &pl->q_norm,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.q_input_norm.weight", l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &pl->k_norm,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.k_input_norm.weight", l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &pl->v_norm,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.v_input_norm.weight", l),
          0.0f, 0);
    } else {
      weight_group_add_vector(
          text_weights, &pl->attn_norm,
          tensor_index_get(checkpoint, "model.llm.layers.%d.attn_norm.weight",
                           l),
          0.0f, 0);
    }
    weight_group_add_vector(
        text_weights, &pl->mlp_norm,
        tensor_index_get(checkpoint, "model.llm.layers.%d.mlp_norm.weight", l),
        0.0f, 0);

    if (pl->is_loop) {
      add_controller(&pl->ctrl_q, l, "attn");
      add_controller(&pl->ctrl_k, l, "k");
      add_controller(&pl->ctrl_v, l, "v");
      add_controller(&pl->ctrl_mlp, l, "mlp");
    }

    add_bf16_lin(&pl->wq,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.wq_du.weight", l),
                 INK2_Q_DIM, INK2_DIM);
    add_bf16_lin(&pl->wk,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.wk_dv.weight", l),
                 INK2_KV_DIM, INK2_DIM);
    add_bf16_lin(&pl->wv,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.wv_dv.weight", l),
                 INK2_KV_DIM, INK2_DIM);
    add_bf16_lin(&pl->wr,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.wr_du.weight", l),
                 INK2_R_DIM, INK2_DIM);
    add_bf16_lin(&pl->gate_attn,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.gate_attn.weight",
                                  l),
                 INK2_Q_DIM, INK2_DIM);
    add_bf16_lin(
        &pl->sink_proj,
        tensor_index_get(checkpoint,
                         "model.llm.layers.%d.attn.attn_sink_proj.weight", l),
        INK2_HEADS, INK2_DIM);
    add_bf16_lin(&pl->wo,
                 tensor_index_get(checkpoint,
                                  "model.llm.layers.%d.attn.wo_ud.weight", l),
                 INK2_DIM, INK2_Q_DIM);

    Tensor rel_t = tensor_index_get(
        checkpoint, "model.llm.layers.%d.attn.rel_logits_proj.proj", l);
    if (rel_t.dtype != 'b' || rel_t.ndim != 2 || rel_t.shape[0] != INK2_D_REL ||
        rel_t.shape[1] != INK2_REL_EXTENT)
      die("unexpected rel_logits_proj.proj shape on layer %d", l);
    weight_group_add_generated(text_weights, &pl->rel_proj_t,
                               (size_t)INK2_REL_EXTENT * INK2_D_REL *
                                   sizeof(uint16_t),
                               fill_rel_proj_t, &rel_t, sizeof rel_t);
    weight_group_add_vector(
        text_weights, &pl->q_clip_scale,
        tensor_index_get(checkpoint, "model.llm.layers.%d.attn.q_l2_clip.scale",
                         l),
        0.0f, 0);
    weight_group_add_vector(
        text_weights, &pl->rel_clip_scale,
        tensor_index_get(checkpoint,
                         "model.llm.layers.%d.attn.rel_l2_clip.scale", l),
        0.0f, 0);

    if (pl->is_global) {
      weight_group_add_vector(
          text_weights, &pl->k_sconv_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.k_sconv.static_w", l),
          0.0f, 0);
      add_bf16_lin(
          &pl->k_sconv_proj,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.k_sconv.weight_proj", l),
          INK2_SCONV_DYN_DIM, INK2_DIM);
      weight_group_add_vector(
          text_weights, &pl->v_sconv_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.v_sconv.static_w", l),
          0.0f, 0);
      add_bf16_lin(
          &pl->v_sconv_proj,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.v_sconv.weight_proj", l),
          INK2_SCONV_DYN_DIM, INK2_DIM);
    } else {
      weight_group_add_vector(
          text_weights, &pl->k_sconv_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.k_sconv.weight", l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &pl->v_sconv_w,
          tensor_index_get(checkpoint,
                           "model.llm.layers.%d.attn.v_sconv.weight", l),
          0.0f, 0);
    }

    weight_group_add_vector(
        text_weights, &pl->attn_sconv_w,
        tensor_index_get(checkpoint, "model.llm.layers.%d.attn_sconv.weight",
                         l),
        0.0f, 0);
    weight_group_add_vector(
        text_weights, &pl->mlp_sconv_w,
        tensor_index_get(checkpoint, "model.llm.layers.%d.mlp_sconv.weight", l),
        0.0f, 0);

    add_bf16_lin(
        &pl->router_w,
        tensor_index_get(checkpoint, "model.llm.layers.%d.mlp.gate.weight", l),
        INK2_EXPERTS, INK2_DIM);
    weight_group_add_vector(
        text_weights, &pl->router_bias,
        tensor_index_get(checkpoint, "model.llm.layers.%d.mlp.gate.bias", l),
        0.0f, 0);

    if (pl->is_nvfp4) {
      FactorFillParam f13 = {
          .amax = tensor_index_get(
              checkpoint, "model.llm.layers.%d.mlp.experts.w13_weight_amax", l),
          .denom = tensor_index_get(checkpoint,
                                    "model.llm.layers.%d.mlp.experts.w13_"
                                    "weight_outer_scale_denominator",
                                    l)};
      FactorFillParam f2 = {
          .amax = tensor_index_get(
              checkpoint, "model.llm.layers.%d.mlp.experts.w2_weight_amax", l),
          .denom = tensor_index_get(checkpoint,
                                    "model.llm.layers.%d.mlp.experts.w2_weight_"
                                    "outer_scale_denominator",
                                    l)};
      weight_group_add_generated(text_weights, &pl->f13,
                                 (size_t)INK2_EXPERTS * sizeof(float),
                                 fill_nvfp4_factor, &f13, sizeof f13);
      weight_group_add_generated(text_weights, &pl->f2,
                                 (size_t)INK2_EXPERTS * sizeof(float),
                                 fill_nvfp4_factor, &f2, sizeof f2);
    }
  }
}

// Validates one 3D routed-expert tensor [128, rows, cols] and records each
// expert's 16-byte-aligned FileSpan in expert_spans[layer][e][part].
static void fill_expert_part_spans(int layer, int part, Tensor t, int64_t rows,
                                   int64_t cols, int dtype_a, int dtype_b) {
  if ((t.dtype != dtype_a && t.dtype != dtype_b) || t.ndim != 3 ||
      t.shape[0] != INK2_EXPERTS || t.shape[1] != rows || t.shape[2] != cols)
    die("layer %d expert part %d has unexpected shape or dtype", layer, part);
  uint64_t per_expert = (uint64_t)rows * (uint64_t)cols;
  if (t.size != (uint64_t)INK2_EXPERTS * per_expert)
    die("layer %d expert part %d byte size mismatch", layer, part);
  if ((t.off & ALIGN_MASK) != 0 || (per_expert & ALIGN_MASK) != 0)
    die("layer %d expert part %d is not 16-byte aligned", layer, part);
  for (int e = 0; e < INK2_EXPERTS; e++)
    expert_spans[layer][e][part] =
        (FileSpan){t.fd, t.off + (uint64_t)e * per_expert, per_expert};
}

// Registers the 4 checkpoint file spans per expert for all 24 physical layers
// so the expert store streams them from SSD into GPU slots.
static void register_expert_spans(void) {
  for (int l = 0; l < INK2_PHYS_LAYERS; l++) {
    Tensor w13 = tensor_index_get(
        checkpoint, "model.llm.layers.%d.mlp.experts.w13_weight", l);
    Tensor s13 = tensor_index_get(
        checkpoint, "model.llm.layers.%d.mlp.experts.w13_weight_scale", l);
    Tensor w2 = tensor_index_get(
        checkpoint, "model.llm.layers.%d.mlp.experts.w2_weight", l);
    Tensor s2 = tensor_index_get(
        checkpoint, "model.llm.layers.%d.mlp.experts.w2_weight_scale", l);
    if (l < INK2_NVFP4_LAYERS) {
      fill_expert_part_spans(l, 0, w13, INK2_W13_ROWS,
                             INK2_DIM / NVFP4_VALUES_PER_BYTE, 'u', 'u');
      fill_expert_part_spans(l, 1, s13, INK2_W13_ROWS,
                             INK2_DIM / INK2_NVFP4_BLOCK, '8', '8');
      fill_expert_part_spans(l, 2, w2, INK2_DIM,
                             INK2_INTER / NVFP4_VALUES_PER_BYTE, 'u', 'u');
      fill_expert_part_spans(l, 3, s2, INK2_DIM, INK2_INTER / INK2_NVFP4_BLOCK,
                             '8', '8');
    } else {
      fill_expert_part_spans(l, 0, w13, INK2_W13_ROWS, INK2_DIM, '8', '8');
      fill_expert_part_spans(l, 1, s13, INK2_W13_ROWS,
                             INK2_DIM / INK2_MXFP8_BLOCK, 'u', 'e');
      fill_expert_part_spans(l, 2, w2, INK2_DIM, INK2_INTER, '8', '8');
      fill_expert_part_spans(l, 3, s2, INK2_DIM, INK2_INTER / INK2_MXFP8_BLOCK,
                             'u', 'e');
    }
    expert_span_counts[l] = EXPERT_PARTS;
  }
}

// ---- workspace management ----

// Returns the GPU workspace byte count required for `rows` tokens.
static uint64_t estimate_work_bytes(int rows) {
  int cap = rows <= INK2_MAX_VERIFY ? INK2_MAX_VERIFY : rows;
  int q_stride = align_up(cap, PREFILL_Q_PAD);
  int npairs = cap * INK2_TOPK;
  int max_tiles = npairs + INK2_EXPERTS;
  int part_rows = cap < INK2_MAX_VERIFY ? cap : INK2_MAX_VERIFY;
  int route_id_count = INK2_TOTAL_LAYERS * INK2_MAX_VERIFY * INK2_TOPK;
  if (npairs > route_id_count) route_id_count = npairs;
  uint64_t bytes = 0;
  bytes += gpu_round_up_to_pages((size_t)cap * sizeof(int32_t));
  bytes += 6 * gpu_round_up_to_pages((size_t)cap * INK2_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)INK2_MAX_BLOCKS * cap * INK2_DIM *
                                 sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)INK2_MSG_COUNT * cap * INK2_DIM *
                                 sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)ATTNRES_GLOBAL_QUERIES * cap *
                                 INK2_DIM * sizeof(float));
  bytes += 3 * gpu_round_up_to_pages((size_t)cap * INK2_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)ATTNRES_GLOBAL_QUERIES * cap * 2 *
                                 INK2_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)ATTNRES_GLOBAL_QUERIES * cap *
                                 INK2_GATE_RANK * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)ATTNRES_GLOBAL_QUERIES * cap *
                                 INK2_DIM * sizeof(float));
  bytes += 3 * gpu_round_up_to_pages((size_t)cap * INK2_Q_DIM * sizeof(float));
  bytes += 4 * gpu_round_up_to_pages((size_t)cap * INK2_KV_DIM * sizeof(float));
  bytes +=
      gpu_round_up_to_pages((size_t)cap * INK2_SCONV_DYN_DIM * sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)cap * INK2_R_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)cap * INK2_HEADS * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)INK2_HEADS * q_stride * INK2_HEAD_DIM *
                                 sizeof(uint16_t));
  bytes +=
      gpu_round_up_to_pages((size_t)part_rows * INK2_HEADS * MAX_DECODE_PARTS *
                            DECODE_PART_FLOATS * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)cap * INK2_EXPERTS * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)route_id_count * sizeof(uint32_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)npairs * sizeof(int32_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)max_tiles * sizeof(Ink2MoeTile));
  bytes += gpu_round_up_to_pages((size_t)INK2_EXPERTS * sizeof(ExpertRef));
  bytes += gpu_round_up_to_pages((size_t)(npairs + MOE_MM_TILE) * INK2_DIM *
                                 sizeof(uint16_t));
  bytes += gpu_round_up_to_pages((size_t)(npairs + MOE_MM_TILE) * INK2_INTER *
                                 sizeof(uint16_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * INK2_DIM * sizeof(float));
  bytes +=
      gpu_round_up_to_pages((size_t)INK2_TOTAL_LAYERS * SCONV_RINGS_PER_LAYER *
                            INK2_MAX_BATCH_ROWS * sizeof(SconvTarget));
  bytes += gpu_round_up_to_pages((size_t)INK2_TOTAL_LAYERS *
                                 INK2_MAX_BATCH_ROWS * sizeof(KvTarget));
  bytes += gpu_round_up_to_pages((size_t)LOGITS_STRIDE * sizeof(float));
  if (opt.inkling2.fp8_kv) {
    bytes += 2 * gpu_round_up_to_pages((size_t)FP8_PREFILL_SLAB_ROWS *
                                       INK2_BF16_KV_ROW_BYTES);
    bytes +=
        gpu_round_up_to_pages((size_t)cap * INK2_HEADS * 2 * sizeof(float));
  }
  bytes += inkling2_mtp_estimate_work_bytes(cap);
  return bytes;
}

// Returns the bytes currently allocated in this pass's resizable workspace
// buffers (excluding work.logits, which persists across row-capacity changes).
static uint64_t work_resizable_bytes(void) {
  const Buf *list[] = {&work.ids,
                       &work.cur,
                       &work.cur2,
                       &work.residual,
                       &work.pre_loop_cur,
                       &work.prev_blk,
                       &work.prev_cur,
                       &work.blk,
                       &work.msg,
                       &work.attn_in,
                       &work.q_in,
                       &work.k_in,
                       &work.v_in,
                       &work.msg_cat,
                       &work.msg_g,
                       &work.msg_gated,
                       &work.q_raw,
                       &work.k_raw,
                       &work.k_conv,
                       &work.v_raw,
                       &work.v_conv,
                       &work.dyn_w,
                       &work.r_raw,
                       &work.r_out,
                       &work.gate,
                       &work.sink,
                       &work.q_hm,
                       &work.part,
                       &work.attn_out,
                       &work.deq_k,
                       &work.deq_v,
                       &work.slab_ml,
                       &work.route_logits,
                       &work.route_ids,
                       &work.route_w,
                       &work.ptok,
                       &work.pw,
                       &work.tp,
                       &work.tiles,
                       &work.expert_table,
                       &work.x_bf16,
                       &work.h_bf16,
                       &work.y_pair,
                       &work.sconv_tgt,
                       &work.kv_tgt,
                       &work.mtp.taps,
                       &work.mtp.ctx,
                       &work.mtp.k_ctx_raw,
                       &work.mtp.v_ctx_raw,
                       &work.mtp.kv_tgt,
                       &work.mtp.frnn_tgt,
                       &work.mtp.cur,
                       &work.mtp.xn,
                       &work.mtp.q_raw,
                       &work.mtp.k_raw,
                       &work.mtp.v_raw,
                       &work.mtp.q_blk,
                       &work.mtp.k_blk,
                       &work.mtp.v_blk,
                       &work.mtp.part,
                       &work.mtp.attn_out,
                       &work.mtp.o_out,
                       &work.mtp.gate,
                       &work.mtp.up,
                       &work.mtp.h_mlp,
                       &work.mtp.mlp_out,
                       &work.mtp.h_final,
                       &work.mtp.rnn_state,
                       &work.mtp.joint,
                       &work.mtp.proj,
                       &work.mtp.latent,
                       &work.mtp.corr,
                       &work.mtp.h_eff,
                       &work.mtp.argmax_parts,
                       &work.mtp.tok};
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    bytes += list[i]->allocated;
  return bytes;
}

// Marks all buffers in w to be freed once prior GPU commands finish.
static void free_work(Ink2Work *w) {
  Buf *list[] = {
      &w->ids,          &w->cur,       &w->cur2,     &w->residual,
      &w->pre_loop_cur, &w->prev_blk,  &w->prev_cur, &w->blk,
      &w->msg,          &w->attn_in,   &w->q_in,     &w->k_in,
      &w->v_in,         &w->msg_cat,   &w->msg_g,    &w->msg_gated,
      &w->q_raw,        &w->k_raw,     &w->k_conv,   &w->v_raw,
      &w->v_conv,       &w->dyn_w,     &w->r_raw,    &w->r_out,
      &w->gate,         &w->sink,      &w->q_hm,     &w->part,
      &w->attn_out,     &w->deq_k,     &w->deq_v,    &w->slab_ml,
      &w->route_logits, &w->route_ids, &w->route_w,  &w->ptok,
      &w->pw,           &w->tp,        &w->tiles,    &w->expert_table,
      &w->x_bf16,       &w->h_bf16,    &w->y_pair,   &w->logits,
      &w->sconv_tgt,    &w->kv_tgt};
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    if (list[i]->obj) gpu_mark_to_free(list[i]);
  inkling2_mtp_free_work(&w->mtp);
  *w = (Ink2Work){0};
}

// Ensures the pass-local workspace holds `rows` active rows (shrinking back to
// INK2_MAX_VERIFY for decode passes) and `logits_rows` rows in work.logits.
static void ensure_work(int rows, int logits_rows) {
  int new_cap = rows <= INK2_MAX_VERIFY ? INK2_MAX_VERIFY : rows;
  int shrink_to_small =
      (new_cap == INK2_MAX_VERIFY && work.cap_rows > INK2_MAX_VERIFY);
  if (rows > work.cap_rows || shrink_to_small) {
    int old_logits_rows = work.logits_rows;
    Buf old_logits = work.logits;
    work.logits = (Buf){0};
    free_work(&work);
    work.cap_rows = new_cap;
    work.q_stride = align_up(new_cap, PREFILL_Q_PAD);
    int npairs = new_cap * INK2_TOPK;
    int max_tiles = npairs + INK2_EXPERTS;
    int part_rows = new_cap < INK2_MAX_VERIFY ? new_cap : INK2_MAX_VERIFY;
    int route_id_count = INK2_TOTAL_LAYERS * INK2_MAX_VERIFY * INK2_TOPK;
    if (npairs > route_id_count) route_id_count = npairs;

    work.ids = work_alloc((size_t)new_cap * sizeof(int32_t));
    work.cur = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.cur2 = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.residual = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.pre_loop_cur = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.prev_blk = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.prev_cur = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.blk = work_alloc((size_t)INK2_MAX_BLOCKS * new_cap * INK2_DIM *
                          sizeof(float));
    work.msg =
        work_alloc((size_t)INK2_MSG_COUNT * new_cap * INK2_DIM * sizeof(float));
    work.attn_in = work_alloc((size_t)ATTNRES_GLOBAL_QUERIES * new_cap *
                              INK2_DIM * sizeof(float));
    work.q_in = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.k_in = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.v_in = work_alloc((size_t)new_cap * INK2_DIM * sizeof(float));
    work.msg_cat = work_alloc((size_t)ATTNRES_GLOBAL_QUERIES * new_cap * 2 *
                              INK2_DIM * sizeof(float));
    work.msg_g = work_alloc((size_t)ATTNRES_GLOBAL_QUERIES * new_cap *
                            INK2_GATE_RANK * sizeof(float));
    work.msg_gated = work_alloc((size_t)ATTNRES_GLOBAL_QUERIES * new_cap *
                                INK2_DIM * sizeof(float));
    work.q_raw = work_alloc((size_t)new_cap * INK2_Q_DIM * sizeof(float));
    work.k_raw = work_alloc((size_t)new_cap * INK2_KV_DIM * sizeof(float));
    work.k_conv = work_alloc((size_t)new_cap * INK2_KV_DIM * sizeof(float));
    work.v_raw = work_alloc((size_t)new_cap * INK2_KV_DIM * sizeof(float));
    work.v_conv = work_alloc((size_t)new_cap * INK2_KV_DIM * sizeof(float));
    work.dyn_w =
        work_alloc((size_t)new_cap * INK2_SCONV_DYN_DIM * sizeof(float));
    work.r_raw = work_alloc((size_t)new_cap * INK2_R_DIM * sizeof(float));
    work.r_out = work_alloc((size_t)new_cap * INK2_R_DIM * sizeof(float));
    work.gate = work_alloc((size_t)new_cap * INK2_Q_DIM * sizeof(float));
    work.sink = work_alloc((size_t)new_cap * INK2_HEADS * sizeof(float));
    work.q_hm = work_alloc((size_t)INK2_HEADS * work.q_stride * INK2_HEAD_DIM *
                           sizeof(uint16_t));
    work.part = work_alloc((size_t)part_rows * INK2_HEADS * MAX_DECODE_PARTS *
                           DECODE_PART_FLOATS * sizeof(float));
    work.attn_out = work_alloc((size_t)new_cap * INK2_Q_DIM * sizeof(float));
    if (opt.inkling2.fp8_kv) {
      work.deq_k =
          work_alloc((size_t)FP8_PREFILL_SLAB_ROWS * INK2_BF16_KV_ROW_BYTES);
      work.deq_v =
          work_alloc((size_t)FP8_PREFILL_SLAB_ROWS * INK2_BF16_KV_ROW_BYTES);
      work.slab_ml =
          work_alloc((size_t)new_cap * INK2_HEADS * 2 * sizeof(float));
    }
    work.route_logits =
        work_alloc((size_t)new_cap * INK2_EXPERTS * sizeof(float));
    work.route_ids = work_alloc((size_t)route_id_count * sizeof(uint32_t));
    work.route_w = work_alloc((size_t)npairs * sizeof(float));
    work.ptok = work_alloc((size_t)npairs * sizeof(int32_t));
    work.pw = work_alloc((size_t)npairs * sizeof(float));
    work.tp = work_alloc((size_t)npairs * sizeof(int32_t));
    work.tiles = work_alloc((size_t)max_tiles * sizeof(Ink2MoeTile));
    work.expert_table = work_alloc((size_t)INK2_EXPERTS * sizeof(ExpertRef));
    work.x_bf16 = work_alloc((size_t)(npairs + MOE_MM_TILE) * INK2_DIM *
                             sizeof(uint16_t));
    work.h_bf16 = work_alloc((size_t)(npairs + MOE_MM_TILE) * INK2_INTER *
                             sizeof(uint16_t));
    work.y_pair = work_alloc((size_t)npairs * INK2_DIM * sizeof(float));
    work.sconv_tgt =
        work_alloc((size_t)INK2_TOTAL_LAYERS * SCONV_RINGS_PER_LAYER *
                   INK2_MAX_BATCH_ROWS * sizeof(SconvTarget));
    work.kv_tgt = work_alloc((size_t)INK2_TOTAL_LAYERS * INK2_MAX_BATCH_ROWS *
                             sizeof(KvTarget));
    work.logits = old_logits;
    work.logits_rows = old_logits_rows;
  }
  if (logits_rows > work.logits_rows) {
    if (work.logits.obj) gpu_mark_to_free(&work.logits);
    work.logits =
        work_alloc((size_t)logits_rows * LOGITS_STRIDE * sizeof(float));
    work.logits_rows = logits_rows;
  }
  inkling2_mtp_ensure_work(gpu, work_slot, &work.mtp, work.cap_rows);
}

// ---- expert store integration helpers ----

// Returns the expert store's live counter structure.
static ExpertStoreCounts *expert_stats(void) {
  return expert_store_counts(experts_store);
}

// Adds the expert store deltas accumulated since `before` into `sum`.
static void stats_add(ExpertStoreCounts *sum, const ExpertStoreCounts *before) {
  const ExpertStoreCounts *c = expert_stats();
  sum->hits += c->hits - before->hits;
  sum->misses += c->misses - before->misses;
  sum->prefetch_hits += c->prefetch_hits - before->prefetch_hits;
  sum->prefetches += c->prefetches - before->prefetches;
  sum->bytes += c->bytes - before->bytes;
  sum->evictions += c->evictions - before->evictions;
  sum->late += c->late - before->late;
  sum->cancelled += c->cancelled - before->cancelled;
  sum->substituted += c->substituted - before->substituted;
  sum->remaps += c->remaps - before->remaps;
  sum->given_back += c->given_back - before->given_back;
  sum->decode_uses += c->decode_uses - before->decode_uses;
  sum->row_own += c->row_own - before->row_own;
  sum->row_ahead += c->row_ahead - before->row_ahead;
  sum->row_late += c->row_late - before->row_late;
  sum->row_sub += c->row_sub - before->row_sub;
  sum->row_ssd += c->row_ssd - before->row_ssd;
  sum->row_unique += c->row_unique - before->row_unique;
  sum->read_s += c->read_s - before->read_s;
}

// Reserves account memory for a pass of up to `rows` tokens, halving `rows`
// down to 1 for prompt passes when available memory is tight. Returns the
// admitted row count, or 0 if even the minimum pass does not fit.
static int prepare_rows(int rows, int prompt) {
  if (!prompt && work_slot && rows == prepare_prev_rows &&
      work_slot->payload == prepare_prev_payload)
    return rows;
  if (!work_slot || !text_weights || !experts_store) return rows;
  int candidate = prompt && rows > PREFILL_CHUNK ? PREFILL_CHUNK : rows;
  for (;;) {
    uint64_t fixed = work_slot->payload - work_resizable_bytes();
    expert_store_set_pass_rows(experts_store, candidate);
    uint64_t required =
        weight_group_required_bytes(text_weights) + fixed +
        estimate_work_bytes(candidate) +
        expert_store_minimum_bytes(experts_store, INK2_PHYS_LAYERS);
    MemoryAdmission mode =
        candidate > 1 && prompt ? MEMORY_WORKSPACE : MEMORY_REQUIRED;
    if (memory_set_minimum(work_slot->account, required, mode)) {
      prepare_prev_rows = prompt ? 0 : rows;
      prepare_prev_payload = work_slot->payload;
      return candidate;
    }
    if (!prompt || candidate == 1) return 0;
    candidate = (candidate + 1) / 2;
    if (work.cap_rows > candidate) {
      model_sync();
      free_work(&work);
    }
  }
}

// Releases all individual expert holds and whole-layer holds owned by this
// pass after its GPU work has completed.
static void held_release_all(void) {
  if (!experts_store) return;
  if (held.n) {
    expert_store_release(experts_store, held.layer, held.experts, held.n);
    held.n = 0;
  }
  if (held_layers) {
    for (int l = 0; l < INK2_PHYS_LAYERS; l++) {
      if (held_layers & (1u << l)) expert_store_release_layer(experts_store, l);
    }
    held_layers = 0;
  }
}

// Requests and holds `n` routed experts of physical layer `l` until released.
static void hold_experts(int l, const int *experts, int n) {
  if (held.n)
    die("Inkling-2-Mini experts of layer %d are still held", held.layer);
  expert_store_hold(experts_store, l, experts, n);
  held.layer = l;
  held.n = n;
  memcpy(held.experts, experts, (size_t)n * sizeof *experts);
}

// Holds all experts of physical layer `l` once for this pass's open GPU work.
static void hold_layer_if_needed(int l) {
  uint32_t bit = 1u << l;
  if (!(held_layers & bit)) {
    expert_store_hold_layer(experts_store, l);
    held_layers |= bit;
  }
}

// Rebuilds physical layer `l`'s resident address table when its slot mapping
// generation has changed.
static void refresh_layer_table(int l, uint64_t gen) {
  if (layer_table_gen[l] == gen) return;
  ExpertRef *ref = (ExpertRef *)layer_table[l].ptr;
  for (int e = 0; e < INK2_EXPERTS; e++) {
    uint32_t at[EXPERT_STORE_PARTS];
    int fresh = 0;
    const Buf *b = expert_store_get_buffer(experts_store, l, e, at, &fresh);
    if (!b)
      die("reachable Inkling-2-Mini layer %d expert %d has no slot", l, e);
    uint64_t base = b->address + b->off;
    ref[e] =
        (ExpertRef){base + at[0], base + at[1], base + at[2], base + at[3]};
  }
  layer_table_gen[l] = gen;
}

// Populates work.expert_table for the active experts `need[0..nn-1]` of layer
// `l`, declaring GPU reads on slots not yet in committed residency sets.
static void fill_streaming_table(GpuCommandBuffer *cmd, int l, const int *need,
                                 int nn) {
  ExpertRef *ref = (ExpertRef *)work.expert_table.ptr;
  for (int i = 0; i < nn; i++) {
    int e = need[i];
    uint32_t at[EXPERT_STORE_PARTS];
    int fresh = 0;
    const Buf *b = expert_store_get_buffer(experts_store, l, e, at, &fresh);
    if (!b) die("held Inkling-2-Mini layer %d expert %d has no slot", l, e);
    uint64_t base = b->address + b->off;
    ref[e] =
        (ExpertRef){base + at[0], base + at[1], base + at[2], base + at[3]};
    if (fresh) gpu_declare_read(cmd, b);
  }
}

// Credits each decoded row's top NEAR_MISSES unselected experts in layer `l`,
// ranked by sigmoid(logit) + bias matching ink2_route_topk.
static void credit_near_misses(int l, int rows, const uint8_t *used) {
  const float *bias = (const float *)layers[l].router_bias.ptr;
  int list[INK2_EXPERTS], n = 0;
  uint8_t listed[INK2_EXPERTS] = {0};
  for (int r = 0; r < rows; r++) {
    const float *lg =
        (const float *)work.route_logits.ptr + (size_t)r * INK2_EXPERTS;
    float key[INK2_EXPERTS];
    for (int e = 0; e < INK2_EXPERTS; e++)
      key[e] = 1.0f / (1.0f + expf(-lg[e])) + bias[e];
    int top[NEAR_MISSES], nt = 0;
    for (int e = 0; e < INK2_EXPERTS; e++) {
      if (used[e] || (nt == NEAR_MISSES && !(key[e] > key[top[nt - 1]])))
        continue;
      int k = nt < NEAR_MISSES ? nt++ : nt - 1;
      for (; k > 0 && key[e] > key[top[k - 1]]; k--) top[k] = top[k - 1];
      top[k] = e;
    }
    for (int k = 0; k < nt; k++) {
      if (!listed[top[k]]) {
        listed[top[k]] = 1;
        list[n++] = top[k];
      }
    }
  }
  expert_store_credit_near_misses(experts_store, l, list, n);
}

// Converts an ExpertPlace location into the route inspection source code.
static uint8_t expert_place_to_route_source(ExpertPlace place) {
  switch (place) {
  case EXPERT_IN_MEMORY: return ROUTE_SOURCE_MEMORY;
  case EXPERT_READ_AHEAD: return ROUTE_SOURCE_AHEAD;
  case EXPERT_BEING_READ_AHEAD: return ROUTE_SOURCE_LATE;
  case EXPERT_ON_SSD: return ROUTE_SOURCE_SSD;
  }
  return ROUTE_SOURCE_SSD;
}

// Records hit counts, expert-store uses and decode route history for every
// resident fast-path visit executed before the latest model_sync.
static void flush_fast_visits(uint8_t fast_visits[INK2_TOTAL_LAYERS],
                              int rows) {
  ExpertStoreCounts *st = expert_stats();
  for (int v = 0; v < INK2_TOTAL_LAYERS; v++) {
    if (!fast_visits[v]) continue;
    fast_visits[v] = 0;
    int l = schedule[v].phys;
    const uint32_t *rids = (const uint32_t *)work.route_ids.ptr +
                           (size_t)v * (size_t)rows * INK2_TOPK;
    uint8_t used[INK2_EXPERTS] = {0};
    int need[INK2_EXPERTS], nn = 0;
    for (int r = 0; r < rows; r++) {
      for (int k = 0; k < INK2_TOPK; k++) {
        uint32_t e = rids[(size_t)r * INK2_TOPK + (size_t)k];
        if (e >= (uint32_t)INK2_EXPERTS) die("router selected expert %u", e);
        used[e] = 1;
        ExpertPlace place = expert_store_locate(experts_store, l, (int)e);
        if (place == EXPERT_READ_AHEAD)
          st->row_ahead++;
        else
          st->row_own++;
        if (r < ROUTE_DECODE_CAP) {
          route_id[r][v][k] = (uint16_t)e;
          route_from[r][v][k] = place == EXPERT_READ_AHEAD
                                    ? ROUTE_SOURCE_AHEAD
                                    : ROUTE_SOURCE_MEMORY;
          route_step[r][v] = route_steps;
        }
      }
    }
    for (int e = 0; e < INK2_EXPERTS; e++)
      if (used[e]) need[nn++] = e;
    st->row_unique += (uint64_t)nn;
    expert_store_count_use(experts_store, l, need, nn);
  }
}

// Starts background read-ahead for all 128 experts of physical layer `l`
// during large prefill batches that touch nearly every expert.
static void prefetch_all_layer_experts(int l) {
  int all[INK2_EXPERTS];
  for (int e = 0; e < INK2_EXPERTS; e++) all[e] = e;
  expert_store_read_ahead(experts_store, l, all, INK2_EXPERTS,
                          2 * INK2_EXPERTS);
}

// ---- state and snapshot management ----

// Returns the next power-of-two capacity holding `positions` tokens, capped at
// opt.ctx.
static int grown_cap(int positions) {
  int cap = START_CAP;
  while (cap < positions && cap < opt.ctx) cap *= 2;
  if (cap > opt.ctx) cap = opt.ctx;
  if (cap < positions)
    die("sequence length %d exceeds context limit %d", positions, opt.ctx);
  return cap;
}

// Returns the rolling snapshot byte size, including MTP rings when active.
static size_t snap_rolling_bytes(void) {
  return slide_kv_bytes() + CONV_TOTAL_BYTES +
         inkling2_mtp_snap_rolling_bytes();
}

// Returns the snapshot format flag bitmask matching the active MTP and FP8 KV
// settings.
static int32_t snap_flags(void) {
  return (inkling2_mtp_active_drafts() ? SNAP_FLAG_MTP : 0) |
         (opt.inkling2.fp8_kv ? SNAP_FLAG_FP8_KV : 0);
}

// Resizes the 8 global KV buffer pairs (and MTP global KV when active) of s
// to hold `new_cap` rows, preserving the first `keep` rows. Returns 1 on
// success or 0 on memory refusal.
static int state_resize_global(Ink2State *s, int new_cap, int keep) {
  size_t row_bytes = kv_row_bytes();
  size_t new_bytes = (size_t)new_cap * row_bytes;
  size_t copy_bytes = (size_t)keep * row_bytes;
  Buf nk[INK2_GLOBAL_COUNT] = {0}, nv[INK2_GLOBAL_COUNT] = {0};
  for (int i = 0; i < INK2_GLOBAL_COUNT; i++) {
    nk[i] = state_try_alloc(new_bytes);
    nv[i] = nk[i].obj ? state_try_alloc(new_bytes) : (Buf){0};
    if (!nv[i].obj) {
      for (int j = 0; j <= i; j++) {
        if (nk[j].obj) gpu_free(&nk[j]);
        if (nv[j].obj) gpu_free(&nv[j]);
      }
      return 0;
    }
    if (copy_bytes && s->gk[i].obj) {
      memcpy(nk[i].ptr, s->gk[i].ptr, copy_bytes);
      memcpy(nv[i].ptr, s->gv[i].ptr, copy_bytes);
    }
  }
  if (!inkling2_mtp_state_resize_global(gpu, &state_slot, &s->mtp, new_cap,
                                        keep)) {
    for (int i = 0; i < INK2_GLOBAL_COUNT; i++) {
      gpu_free(&nk[i]);
      gpu_free(&nv[i]);
    }
    return 0;
  }
  for (int i = 0; i < INK2_GLOBAL_COUNT; i++) {
    if (s->gk[i].obj) gpu_mark_to_free(&s->gk[i]);
    if (s->gv[i].obj) gpu_mark_to_free(&s->gv[i]);
    s->gk[i] = nk[i];
    s->gv[i] = nv[i];
  }
  s->cap = new_cap;
  return 1;
}

// Ensures s holds at least `need` positions in its global KV caches.
static void state_ensure_cap(Ink2State *s, int need) {
  if (need <= s->cap) return;
  model_sync();
  int next = grown_cap(need);
  if (!state_resize_global(s, next, s->cap))
    die("out of memory growing Inkling-2-Mini KV cache to %d tokens", next);
}

// Allocates a new Ink2State, or returns null when memory does not fit.
static ModelState *ink2_state_new(void) {
  Ink2State *s = xmalloc(sizeof *s);
  *s = (Ink2State){0};
  int init_cap = grown_cap(1);
  size_t slide_bytes = slide_kv_bytes();
  s->slide_kv = state_try_alloc(slide_bytes);
  s->conv_ring = s->slide_kv.obj ? state_try_alloc(CONV_TOTAL_BYTES) : (Buf){0};
  if (!s->conv_ring.obj ||
      !inkling2_mtp_state_init(gpu, &state_slot, &s->mtp) ||
      !state_resize_global(s, init_cap, 0)) {
    if (s->slide_kv.obj) gpu_free(&s->slide_kv);
    if (s->conv_ring.obj) gpu_free(&s->conv_ring);
    inkling2_mtp_state_free(&s->mtp);
    free(s);
    return NULL;
  }
  memset(s->slide_kv.ptr, 0, slide_bytes);
  memset(s->conv_ring.ptr, 0, CONV_TOTAL_BYTES);
  return (ModelState *)s;
}

// Frees an Ink2State and its GPU buffers.
static void ink2_state_free(ModelState *ms) {
  Ink2State *s = (Ink2State *)ms;
  if (!s) return;
  for (int i = 0; i < INK2_GLOBAL_COUNT; i++) {
    if (s->gk[i].obj) gpu_mark_to_free(&s->gk[i]);
    if (s->gv[i].obj) gpu_mark_to_free(&s->gv[i]);
  }
  if (s->slide_kv.obj) gpu_mark_to_free(&s->slide_kv);
  if (s->conv_ring.obj) gpu_mark_to_free(&s->conv_ring);
  inkling2_mtp_state_free(&s->mtp);
  free(s);
}

// Returns the byte footprint of state s (or a new state when s is null) at
// `positions` tokens.
static size_t ink2_state_bytes(const ModelState *ms, int positions) {
  const Ink2State *s = (const Ink2State *)ms;
  int cap = s ? s->cap : grown_cap(positions > 0 ? positions : 1);
  if (positions > cap) cap = grown_cap(positions);
  size_t gk_one = gpu_round_up_to_pages((size_t)cap * kv_row_bytes());
  return (size_t)INK2_GLOBAL_COUNT * 2 * gk_one +
         gpu_round_up_to_pages(slide_kv_bytes()) +
         gpu_round_up_to_pages(CONV_TOTAL_BYTES) +
         inkling2_mtp_state_bytes(cap);
}

// Writes the growing global KV regions of s (16 base, plus 2 for MTP when
// active). When fill is nonzero, grows s to hold `positions` tokens first.
static int ink2_write_state_regions(ModelState *ms, int positions, int fill,
                                    StateRegion *out) {
  Ink2State *s = (Ink2State *)ms;
  if (positions < 0 || (!fill && positions > s->cap))
    die("invalid Inkling-2-Mini state regions for %d positions", positions);
  model_sync();
  if (fill) {
    int cap = grown_cap(positions > 0 ? positions : 1);
    if (s->cap != cap && !state_resize_global(s, cap, 0))
      die("out of memory sizing Inkling-2-Mini state to %d tokens", cap);
  }
  uint32_t row_bytes = (uint32_t)kv_row_bytes();
  int n = 0;
  for (int i = 0; i < INK2_GLOBAL_COUNT; i++) {
    out[n++] = (StateRegion){s->gk[i].ptr, row_bytes, 1, 0, 0};
    out[n++] = (StateRegion){s->gv[i].ptr, row_bytes, 1, 0, 0};
  }
  if (inkling2_mtp_active_drafts()) {
    uint32_t mtp_row_bytes = (uint32_t)inkling2_mtp_kv_row_bytes();
    out[n++] = (StateRegion){s->mtp.gk.ptr, mtp_row_bytes, 1, 0, 0};
    out[n++] = (StateRegion){s->mtp.gv.ptr, mtp_row_bytes, 1, 0, 0};
  }
  return n;
}

// Copies the growing global KV rows before `positions` from src into dst.
static void ink2_state_copy(ModelState *dst_ms, const ModelState *src_ms,
                            int positions) {
  if (dst_ms == src_ms) die("cannot copy an Inkling-2-Mini state onto itself");
  StateRegion to[STATE_REGIONS_MAX], from[STATE_REGIONS_MAX];
  int n = ink2_write_state_regions(dst_ms, positions, 1, to);
  ink2_write_state_regions((ModelState *)src_ms, positions, 0, from);
  regions_copy(to, from, n, positions);
}

// Captures the rolling sliding-window KV, short-convolution and optional MTP
// rings of s at `pos`, along with optional logits.
static ModelSnapshot *ink2_snapshot_take(ModelState *ms, int pos,
                                         const float *lg) {
  Ink2State *s = (Ink2State *)ms;
  model_sync();
  size_t slide_bytes = slide_kv_bytes();
  size_t base_rolling = slide_bytes + CONV_TOTAL_BYTES;
  size_t rolling = snap_rolling_bytes();
  Ink2Snapshot *x = xmalloc(sizeof *x);
  *x = (Ink2Snapshot){.pos = pos,
                      .rows = memory_host_alloc_required(&state_slot, rolling)};
  memcpy(x->rows, s->slide_kv.ptr, slide_bytes);
  memcpy(x->rows + slide_bytes, s->conv_ring.ptr, CONV_TOTAL_BYTES);
  if (inkling2_mtp_active_drafts()) {
    size_t mtp_slide = inkling2_mtp_slide_kv_bytes();
    memcpy(x->rows + base_rolling, s->mtp.slide_kv.ptr, mtp_slide);
    memcpy(x->rows + base_rolling + mtp_slide, s->mtp.frnn.ptr,
           INK2_MTP_FRNN_BYTES);
  }
  if (lg) {
    x->logits = memory_host_alloc_required(&state_slot, LOGITS_BYTES);
    memcpy(x->logits, lg, LOGITS_BYTES);
  }
  return (ModelSnapshot *)x;
}

// Restores rolling sliding-window KV, short-convolution and optional MTP rings
// from x into s.
static void ink2_snapshot_restore(ModelState *ms, const ModelSnapshot *mx) {
  Ink2State *s = (Ink2State *)ms;
  const Ink2Snapshot *x = (const Ink2Snapshot *)mx;
  model_sync();
  size_t slide_bytes = slide_kv_bytes();
  size_t base_rolling = slide_bytes + CONV_TOTAL_BYTES;
  memcpy(s->slide_kv.ptr, x->rows, slide_bytes);
  memcpy(s->conv_ring.ptr, x->rows + slide_bytes, CONV_TOTAL_BYTES);
  if (inkling2_mtp_active_drafts()) {
    size_t mtp_slide = inkling2_mtp_slide_kv_bytes();
    memcpy(s->mtp.slide_kv.ptr, x->rows + base_rolling, mtp_slide);
    memcpy(s->mtp.frnn.ptr, x->rows + base_rolling + mtp_slide,
           INK2_MTP_FRNN_BYTES);
  }
}

// Frees a snapshot and its host buffers.
static void ink2_snapshot_free(ModelSnapshot *mx) {
  Ink2Snapshot *x = (Ink2Snapshot *)mx;
  if (!x) return;
  memory_host_free(&state_slot, x->rows);
  memory_host_free(&state_slot, x->logits);
  free(x);
}

// Returns the byte footprint of one snapshot.
static size_t ink2_snapshot_bytes(int with_logits) {
  return snap_rolling_bytes() + (with_logits ? LOGITS_BYTES : 0);
}

// Returns the saved logits of snapshot x, or null if none were saved.
static const float *ink2_snapshot_logits(const ModelSnapshot *mx) {
  const Ink2Snapshot *x = (const Ink2Snapshot *)mx;
  return x->logits;
}

// Writes the serialized parts and metadata of snapshot x, or of state s's
// live rolling rings when x is null.
static int ink2_write_snapshot_parts(ModelSnapshot *mx, ModelState *ms, int pos,
                                     SnapshotPart *out, int32_t *meta,
                                     int *nmeta) {
  model_sync();
  Ink2Snapshot *x = (Ink2Snapshot *)mx;
  Ink2State *s = (Ink2State *)ms;
  int mtp_on = inkling2_mtp_active_drafts() ? 1 : 0;
  *nmeta = 2;
  meta[1] = snap_flags();
  int n = 0;
  if (x) {
    meta[0] = x->pos;
    out[n++] = (SnapshotPart){x->rows, snap_rolling_bytes()};
    if (x->logits) out[n++] = (SnapshotPart){x->logits, LOGITS_BYTES};
    return n;
  }
  meta[0] = pos;
  out[n++] = (SnapshotPart){s->slide_kv.ptr, slide_kv_bytes()};
  out[n++] = (SnapshotPart){s->conv_ring.ptr, CONV_TOTAL_BYTES};
  if (mtp_on) {
    out[n++] =
        (SnapshotPart){s->mtp.slide_kv.ptr, inkling2_mtp_slide_kv_bytes()};
    out[n++] = (SnapshotPart){s->mtp.frnn.ptr, INK2_MTP_FRNN_BYTES};
  }
  return n;
}

// Allocates an empty snapshot matching the serialized metadata.
static ModelSnapshot *ink2_snapshot_new_empty(const int32_t *meta, int nmeta,
                                              int with_logits) {
  if (nmeta != 2 || meta[0] < 0 || meta[1] != snap_flags()) return NULL;
  Ink2Snapshot *x = xmalloc(sizeof *x);
  *x = (Ink2Snapshot){
      .pos = meta[0],
      .rows = memory_host_alloc_required(&state_slot, snap_rolling_bytes())};
  if (with_logits)
    x->logits = memory_host_alloc_required(&state_slot, LOGITS_BYTES);
  return (ModelSnapshot *)x;
}

// ---- GPU kernel dispatch helpers ----

// Chooses split-K for mv_bf16 / mvm_bf16 to saturate the GPU on narrow output
// projections while keeping m = 1 and m = 2..8 identical in reduction order.
static uint32_t pick_mv_split(int n, int k) {
  if (k < (int)BF16_MV_SPLIT_MIN_K) return 1u;
  if (n <= (int)BF16_MV_SPLIT_TINY_N) return BF16_MV_SPLIT_TINY;
  if (n <= (int)BF16_MV_SPLIT_SMALL_N) return BF16_MV_SPLIT_SMALL;
  if (n <= (int)BF16_MV_SPLIT_MID_N) return BF16_MV_SPLIT_MID;
  return 1u;
}

// Dispatches a single BF16 linear projection y = x @ w^T starting at byte
// offset x_off in x, choosing mv_bf16 (m = 1), mvm_bf16 (2 <= m <= 8,
// non-prefill), or tiled mm_bf16 (m > 8 or prefill).
static void run_bf16_linear_off(GpuCommandBuffer *c, const Buf *w, const Buf *x,
                                size_t x_off, const Buf *y, int m, int n, int k,
                                int ldy, int round, int is_prefill) {
  if (!is_prefill && m == 1) {
    uint32_t split = pick_mv_split(n, k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    MvParam p = {(uint32_t)n, (uint32_t)k,     (uint32_t)n,
                 0,           (uint32_t)round, split};
    gpu_select_kernel(c, K_mv_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, x_off);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   MV_THREADS, 1, 1);
  } else if (!is_prefill && m <= INK2_MAX_BATCH_ROWS) {
    uint32_t split = pick_mv_split(n, k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    MvmParam p = {
        {(uint32_t)n, (uint32_t)k, (uint32_t)n, 0, (uint32_t)round, split},
        (uint32_t)m,
        (uint32_t)k,
        (uint32_t)ldy};
    gpu_select_kernel(c, K_mvm_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, x_off);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   MV_THREADS, 1, 1);
  } else {
    MmParam p = {(uint32_t)m,     (uint32_t)n, (uint32_t)k,
                 (uint32_t)round, (uint32_t)k, (uint32_t)ldy};
    gpu_select_kernel(c, K_mm_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, x_off);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    uint32_t tiles_n = ((uint32_t)n + MM_TILE_N - 1u) / MM_TILE_N;
    uint32_t tiles_m = ((uint32_t)m + MM_TILE_M - 1u) / MM_TILE_M;
    gpu_run_kernel(c, tiles_n * tiles_m, 1, 1, MM_THREADS, 1, 1);
  }
}

// Dispatches a single BF16 linear projection y = x @ w^T at offset 0 of x.
static void run_bf16_linear(GpuCommandBuffer *c, const Buf *w, const Buf *x,
                            const Buf *y, int m, int n, int k, int ldy,
                            int round, int is_prefill) {
  run_bf16_linear_off(c, w, x, 0, y, m, n, k, ldy, round, is_prefill);
}

// Dispatches fused unembed + softcap for 1..8 rows, or mm_bf16 + softcap for
// larger batches (e.g. --perplexity with every-row prefill logits).
static void run_unembed_softcap(GpuCommandBuffer *c, const Buf *x, size_t x_off,
                                const Buf *y, int m, int is_prefill) {
  if (!is_prefill && m <= INK2_MAX_BATCH_ROWS) {
    uint32_t rows_per_tg = BF16_MV_SGS * BF16_MV_ROWS_PER_SG;
    UnembedParam up = {(uint32_t)m,        (uint32_t)INK2_VOCAB,
                       (uint32_t)INK2_DIM, (uint32_t)LOGITS_STRIDE,
                       LOGIT_SOFTCAP,      1.0f / LOGIT_SOFTCAP};
    gpu_select_kernel(c, m == 1 ? K_unembed_softcap_m1 : K_unembed_softcap);
    gpu_bind_buffer(c, 0, &unembed_w, 0);
    gpu_bind_buffer(c, 1, x, x_off);
    gpu_bind_buffer(c, 2, y, 0);
    gpu_copy_bytes(c, 3, &up, sizeof up);
    gpu_run_kernel(c, ((uint32_t)INK2_VOCAB + rows_per_tg - 1u) / rows_per_tg,
                   1, 1, MV_THREADS, 1, 1);
    return;
  }
  run_bf16_linear_off(c, &unembed_w, x, x_off, y, m, INK2_VOCAB, INK2_DIM,
                      LOGITS_STRIDE, 0, is_prefill);
  SoftcapParam sp = {(uint32_t)m, (uint32_t)INK2_VOCAB, (uint32_t)LOGITS_STRIDE,
                     LOGIT_SOFTCAP};
  gpu_select_kernel(c, K_logit_softcap);
  gpu_bind_buffer(c, 0, y, 0);
  gpu_copy_bytes(c, 1, &sp, sizeof sp);
  gpu_run_kernel(c, ((uint32_t)INK2_VOCAB + ELEM_THREADS - 1u) / ELEM_THREADS,
                 (uint32_t)m, 1, ELEM_THREADS, 1, 1);
}

// Dispatches 2 BF16 linear projections sharing the same input x and width k in
// one fused dispatch during decode/verify (mv_bf16_2 / mvm_bf16_2), or 2
// mm_bf16 calls during prefill.
static void run_bf16_linear2(GpuCommandBuffer *c, const Bf16Lin *a,
                             const Buf *ya, const Bf16Lin *b, const Buf *yb,
                             const Buf *x, int m, int round, int is_prefill) {
  if (!is_prefill && m <= INK2_MAX_BATCH_ROWS) {
    int max_n = a->n > b->n ? a->n : b->n;
    uint32_t split = pick_mv_split(max_n, a->k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    uint32_t dims[2][2] = {{(uint32_t)a->n, (uint32_t)a->n},
                           {(uint32_t)b->n, (uint32_t)b->n}};
    if (m == 1) {
      MvParam p = {(uint32_t)max_n, (uint32_t)a->k, (uint32_t)max_n, 0,
                   (uint32_t)round, split};
      gpu_select_kernel(c, K_mv_bf16_2);
      gpu_bind_buffer(c, 0, &a->w, 0);
      gpu_bind_buffer(c, 2, x, 0);
      gpu_bind_buffer(c, 3, ya, 0);
      gpu_copy_bytes(c, 4, &p, sizeof p);
      gpu_bind_buffer(c, 5, &b->w, 0);
      gpu_bind_buffer(c, 7, yb, 0);
      gpu_copy_bytes(c, 11, dims, sizeof dims);
      gpu_run_kernel(c, ((uint32_t)max_n + rows_per_tg - 1u) / rows_per_tg, 2,
                     1, MV_THREADS, 1, 1);
    } else {
      MvmParam p = {{(uint32_t)max_n, (uint32_t)a->k, (uint32_t)max_n, 0,
                     (uint32_t)round, split},
                    (uint32_t)m,
                    (uint32_t)a->k,
                    (uint32_t)max_n};
      gpu_select_kernel(c, K_mvm_bf16_2);
      gpu_bind_buffer(c, 0, &a->w, 0);
      gpu_bind_buffer(c, 2, x, 0);
      gpu_bind_buffer(c, 3, ya, 0);
      gpu_copy_bytes(c, 4, &p, sizeof p);
      gpu_bind_buffer(c, 5, &b->w, 0);
      gpu_bind_buffer(c, 7, yb, 0);
      gpu_copy_bytes(c, 11, dims, sizeof dims);
      gpu_run_kernel(c, ((uint32_t)max_n + rows_per_tg - 1u) / rows_per_tg, 2,
                     1, MV_THREADS, 1, 1);
    }
    return;
  }
  run_bf16_linear(c, &a->w, x, ya, m, a->n, a->k, a->n, round, is_prefill);
  run_bf16_linear(c, &b->w, x, yb, m, b->n, b->k, b->n, round, is_prefill);
}

// Dispatches 3 BF16 linear projections sharing the same input x and width k in
// one fused dispatch during decode (mv_bf16_3 / mvm_bf16_3), or 3 mm_bf16
// calls during prefill or verify with m > 8.
static void run_bf16_linear3(GpuCommandBuffer *c, const Bf16Lin *a,
                             const Buf *ya, const Bf16Lin *b, const Buf *yb,
                             const Bf16Lin *d, const Buf *yd, const Buf *x,
                             int m, int round, int is_prefill) {
  if (!is_prefill && m <= INK2_MAX_BATCH_ROWS) {
    int max_n = a->n;
    if (b->n > max_n) max_n = b->n;
    if (d->n > max_n) max_n = d->n;
    uint32_t rows_per_tg = BF16_MV_SGS * BF16_MV_ROWS_PER_SG;
    uint32_t dims[3][2] = {{(uint32_t)a->n, (uint32_t)a->n},
                           {(uint32_t)b->n, (uint32_t)b->n},
                           {(uint32_t)d->n, (uint32_t)d->n}};
    if (m == 1) {
      MvParam p = {(uint32_t)max_n, (uint32_t)a->k,
                   (uint32_t)max_n, 0,
                   (uint32_t)round, 1u};
      gpu_select_kernel(c, K_mv_bf16_3);
      gpu_bind_buffer(c, 0, &a->w, 0);
      gpu_bind_buffer(c, 2, x, 0);
      gpu_bind_buffer(c, 3, ya, 0);
      gpu_copy_bytes(c, 4, &p, sizeof p);
      gpu_bind_buffer(c, 5, &b->w, 0);
      gpu_bind_buffer(c, 7, yb, 0);
      gpu_bind_buffer(c, 8, &d->w, 0);
      gpu_bind_buffer(c, 10, yd, 0);
      gpu_copy_bytes(c, 11, dims, sizeof dims);
      gpu_run_kernel(c, ((uint32_t)max_n + rows_per_tg - 1u) / rows_per_tg, 3,
                     1, MV_THREADS, 1, 1);
    } else {
      MvmParam p = {{(uint32_t)max_n, (uint32_t)a->k, (uint32_t)max_n, 0,
                     (uint32_t)round, 1u},
                    (uint32_t)m,
                    (uint32_t)a->k,
                    (uint32_t)max_n};
      gpu_select_kernel(c, K_mvm_bf16_3);
      gpu_bind_buffer(c, 0, &a->w, 0);
      gpu_bind_buffer(c, 2, x, 0);
      gpu_bind_buffer(c, 3, ya, 0);
      gpu_copy_bytes(c, 4, &p, sizeof p);
      gpu_bind_buffer(c, 5, &b->w, 0);
      gpu_bind_buffer(c, 7, yb, 0);
      gpu_bind_buffer(c, 8, &d->w, 0);
      gpu_bind_buffer(c, 10, yd, 0);
      gpu_copy_bytes(c, 11, dims, sizeof dims);
      gpu_run_kernel(c, ((uint32_t)max_n + rows_per_tg - 1u) / rows_per_tg, 3,
                     1, MV_THREADS, 1, 1);
    }
    return;
  }
  run_bf16_linear(c, &a->w, x, ya, m, a->n, a->k, a->n, round, is_prefill);
  run_bf16_linear(c, &b->w, x, yb, m, b->n, b->k, b->n, round, is_prefill);
  run_bf16_linear(c, &d->w, x, yd, m, d->n, d->k, d->n, round, is_prefill);
}

// Dispatches weighted RMSNorm on `rows` rows of INK2_DIM floats.
static void run_rmsnorm(GpuCommandBuffer *c, const Buf *x, size_t x_off,
                        const Buf *w, const Buf *y, int rows, int round) {
  NormParam p = {(uint32_t)INK2_DIM, (uint32_t)round, RMS_EPS};
  gpu_select_kernel(c, K_rmsnorm);
  gpu_bind_buffer(c, 0, x, x_off);
  gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, y, 0);
  gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
}

// Dispatches Block Attention Residuals over `nblocks` stacked blocks and
// `partial` using `nqueries` FP32 queries.
static void run_attn_res(GpuCommandBuffer *c, const Buf *blocks,
                         const Buf *partial, const Buf *queries, size_t q_off,
                         const Buf *out, int rows, int block_stride,
                         int nblocks, int nqueries, int out_stride) {
  AttnResParam p = {(uint32_t)rows,
                    (uint32_t)block_stride,
                    (uint32_t)INK2_DIM,
                    (uint32_t)nblocks,
                    (uint32_t)nqueries,
                    (uint32_t)out_stride,
                    RMS_EPS};
  gpu_select_kernel(c, K_attn_res);
  gpu_bind_buffer(c, 0, blocks, 0);
  gpu_bind_buffer(c, 1, partial, 0);
  gpu_bind_buffer(c, 2, queries, q_off);
  gpu_bind_buffer(c, 3, out, 0);
  gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
}

// Dispatches fused Block Attention Residuals + weighted BF16-rounded RMSNorm
// for 1 query (out0, w0) or 3 queries (out0/1/2, w0/1/2).
static void run_attn_res_rmsnorm(GpuCommandBuffer *c, const Buf *blocks,
                                 size_t blk_off, const Buf *partial,
                                 size_t part_off, const Buf *queries,
                                 size_t q_off, const Buf *w0, const Buf *w1,
                                 const Buf *w2, const Buf *out0,
                                 const Buf *out1, const Buf *out2, int rows,
                                 int block_stride, int nblocks, int nqueries) {
  AttnResParam p = {(uint32_t)rows,
                    (uint32_t)block_stride,
                    (uint32_t)INK2_DIM,
                    (uint32_t)nblocks,
                    (uint32_t)nqueries,
                    (uint32_t)rows,
                    RMS_EPS};
  gpu_select_kernel(c, K_attn_res_rmsnorm);
  gpu_bind_buffer(c, 0, blocks, blk_off);
  gpu_bind_buffer(c, 1, partial, part_off);
  gpu_bind_buffer(c, 2, queries, q_off);
  gpu_bind_buffer(c, 3, w0, 0);
  gpu_bind_buffer(c, 4, w1 ? w1 : w0, 0);
  gpu_bind_buffer(c, 5, w2 ? w2 : w0, 0);
  gpu_bind_buffer(c, 6, out0, 0);
  gpu_bind_buffer(c, 7, out1 ? out1 : out0, 0);
  gpu_bind_buffer(c, 8, out2 ? out2 : out0, 0);
  gpu_copy_bytes(c, 9, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
}

// Dispatches 3-way Q/K/V loop message gates followed by 3-way weighted
// BF16-rounded RMSNorm.
static void run_msg_gate_3_rmsnorm(GpuCommandBuffer *c, const Buf *h,
                                   const Buf *m, const MsgController *cq,
                                   const MsgController *ck,
                                   const MsgController *cv, const Buf *nw0,
                                   const Buf *nw1, const Buf *nw2, int rows,
                                   int distinct_h, int is_prefill) {
  Msg3Param p = {(uint32_t)rows,
                 (uint32_t)INK2_DIM,
                 (uint32_t)INK2_GATE_RANK,
                 (uint32_t)work.cap_rows,
                 (uint32_t)work.cap_rows,
                 (uint32_t)distinct_h,
                 RMS_EPS};
  gpu_select_kernel(c, K_msg_cat_norm_3);
  gpu_bind_buffer(c, 0, h, 0);
  gpu_bind_buffer(c, 1, m, 0);
  gpu_bind_buffer(c, 2, &work.msg_cat, 0);
  gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 3, 1, ROW_THREADS, 1, 1);

  if (!is_prefill && rows <= INK2_MAX_BATCH_ROWS) {
    uint32_t split = pick_mv_split(cq->down.n, cq->down.k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    MvmParam mp = {{(uint32_t)cq->down.n, (uint32_t)cq->down.k,
                    (uint32_t)cq->down.n, 0, 1u, split},
                   (uint32_t)rows,
                   (uint32_t)cq->down.k,
                   (uint32_t)cq->down.n};
    gpu_select_kernel(c, K_msg_down_3);
    gpu_bind_buffer(c, 0, &cq->down.w, 0);
    gpu_bind_buffer(c, 1, &ck->down.w, 0);
    gpu_bind_buffer(c, 2, &cv->down.w, 0);
    gpu_bind_buffer(c, 3, &work.msg_cat, 0);
    gpu_bind_buffer(c, 4, &work.msg_g, 0);
    gpu_copy_bytes(c, 5, &mp, sizeof mp);
    gpu_run_kernel(c, ((uint32_t)cq->down.n + rows_per_tg - 1u) / rows_per_tg,
                   3, 1, MV_THREADS, 1, 1);
  } else {
    size_t cat_bytes = (size_t)rows * 2 * INK2_DIM * sizeof(float);
    size_t g_bytes = (size_t)rows * INK2_GATE_RANK * sizeof(float);
    const MsgController *ctrls[3] = {cq, ck, cv};
    for (int s = 0; s < 3; s++) {
      MmParam mp = {(uint32_t)rows,
                    (uint32_t)ctrls[s]->down.n,
                    (uint32_t)ctrls[s]->down.k,
                    1u,
                    (uint32_t)ctrls[s]->down.k,
                    (uint32_t)ctrls[s]->down.n};
      gpu_select_kernel(c, K_mm_bf16);
      gpu_bind_buffer(c, 0, &ctrls[s]->down.w, 0);
      gpu_bind_buffer(c, 2, &work.msg_cat, (size_t)s * cat_bytes);
      gpu_bind_buffer(c, 3, &work.msg_g, (size_t)s * g_bytes);
      gpu_copy_bytes(c, 4, &mp, sizeof mp);
      uint32_t tn = ((uint32_t)ctrls[s]->down.n + MM_TILE_N - 1u) / MM_TILE_N;
      uint32_t tm = ((uint32_t)rows + MM_TILE_M - 1u) / MM_TILE_M;
      gpu_run_kernel(c, tn * tm, 1, 1, MM_THREADS, 1, 1);
    }
  }

  gpu_select_kernel(c, K_msg_gate_apply_3);
  gpu_bind_buffer(c, 0, h, 0);
  gpu_bind_buffer(c, 1, m, 0);
  gpu_bind_buffer(c, 2, &work.msg_g, 0);
  gpu_bind_buffer(c, 3, &cq->forget_w, 0);
  gpu_bind_buffer(c, 4, &cq->gain_w, 0);
  gpu_bind_buffer(c, 5, &ck->forget_w, 0);
  gpu_bind_buffer(c, 6, &ck->gain_w, 0);
  gpu_bind_buffer(c, 7, &cv->forget_w, 0);
  gpu_bind_buffer(c, 8, &cv->gain_w, 0);
  gpu_bind_buffer(c, 9, &work.msg_gated, 0);
  gpu_copy_bytes(c, 10, &p, sizeof p);
  gpu_run_kernel(
      c, ((uint32_t)INK2_DIM + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG,
      ((uint32_t)rows + MOE_MV_MAX_M - 1u) / MOE_MV_MAX_M, 3, ROW_THREADS, 1,
      1);

  size_t mg_bytes = (size_t)rows * INK2_DIM * sizeof(float);
  NormParam np = {(uint32_t)INK2_DIM, 1u, RMS_EPS};
  gpu_select_kernel(c, K_rmsnorm_3);
  gpu_bind_buffer(c, 0, &work.msg_gated, 0);
  gpu_bind_buffer(c, 1, nw0, 0);
  gpu_bind_buffer(c, 2, &work.q_in, 0);
  gpu_bind_buffer(c, 3, &work.msg_gated, mg_bytes);
  gpu_bind_buffer(c, 4, nw1, 0);
  gpu_bind_buffer(c, 5, &work.k_in, 0);
  gpu_bind_buffer(c, 6, &work.msg_gated, 2 * mg_bytes);
  gpu_bind_buffer(c, 7, nw2, 0);
  gpu_bind_buffer(c, 8, &work.v_in, 0);
  gpu_copy_bytes(c, 9, &np, sizeof np);
  gpu_run_kernel(c, (uint32_t)rows, 3, 1, ROW_THREADS, 1, 1);
}

// Dispatches post-attention static short convolution fused with residual add
// (is_boundary == 0) or block boundary copy to blk (is_boundary == 1).
static void run_sconv_static_post(GpuCommandBuffer *c, const Buf *x,
                                  const Buf *w, const Buf *cur, const Buf *out,
                                  const Buf *blk, size_t blk_off,
                                  const Ink2State *s, int visit,
                                  size_t ring_sub_off, int ring_kind, int rows,
                                  int pos0, int batched, int is_boundary) {
  SconvParam p = {(uint32_t)rows, (uint32_t)INK2_DIM, (uint32_t)INK2_SCONV_RING,
                  (uint32_t)pos0, (uint32_t)is_boundary};
  uint32_t gx = ((uint32_t)INK2_DIM + SCONV_THREADS - 1u) / SCONV_THREADS;
  if (batched) {
    size_t tgt_off =
        ((size_t)visit * SCONV_RINGS_PER_LAYER + (size_t)ring_kind) *
        INK2_MAX_BATCH_ROWS * sizeof(SconvTarget);
    gpu_select_kernel(c, K_sconv_static_post_batched);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &work.sconv_tgt, tgt_off);
    gpu_bind_buffer(c, 2, w, 0);
    gpu_bind_buffer(c, 3, cur, 0);
    gpu_bind_buffer(c, 4, out, 0);
    gpu_bind_buffer(c, 5, blk, blk_off);
    gpu_copy_bytes(c, 6, &p, sizeof p);
    gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);
    return;
  }
  size_t roff = conv_ring_off(visit, ring_sub_off);
  gpu_select_kernel(c, K_sconv_static_post);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, &s->conv_ring, roff);
  gpu_bind_buffer(c, 2, w, 0);
  gpu_bind_buffer(c, 3, cur, 0);
  gpu_bind_buffer(c, 4, out, 0);
  gpu_bind_buffer(c, 5, blk, blk_off);
  gpu_copy_bytes(c, 6, &p, sizeof p);
  gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);
  if (rows > 1) {
    uint32_t keep = (uint32_t)(rows < INK2_SCONV_RING ? rows : INK2_SCONV_RING);
    gpu_select_kernel(c, K_sconv_ring_save);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &s->conv_ring, roff);
    gpu_copy_bytes(c, 2, &p, sizeof p);
    gpu_run_kernel(c, gx, keep, 1, SCONV_THREADS, 1, 1);
  }
}

// Dispatches one loop message gate (h, m) -> out (FP32) using `ctrl`.
static void run_msg_gate(GpuCommandBuffer *c, const Buf *h, size_t h_off,
                         const Buf *m, size_t m_off, const MsgController *ctrl,
                         const Buf *out, int rows, int is_prefill) {
  MsgGateParam p = {(uint32_t)rows, (uint32_t)INK2_DIM,
                    (uint32_t)INK2_GATE_RANK, RMS_EPS};
  gpu_select_kernel(c, K_msg_cat_norm);
  gpu_bind_buffer(c, 0, h, h_off);
  gpu_bind_buffer(c, 1, m, m_off);
  gpu_bind_buffer(c, 2, &work.msg_cat, 0);
  gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);

  run_bf16_linear(c, &ctrl->down.w, &work.msg_cat, &work.msg_g, rows,
                  ctrl->down.n, ctrl->down.k, ctrl->down.n, 1, is_prefill);

  gpu_select_kernel(c, K_msg_gate_apply);
  gpu_bind_buffer(c, 0, h, h_off);
  gpu_bind_buffer(c, 1, m, m_off);
  gpu_bind_buffer(c, 2, &work.msg_g, 0);
  gpu_bind_buffer(c, 3, &ctrl->forget_w, 0);
  gpu_bind_buffer(c, 4, &ctrl->gain_w, 0);
  gpu_bind_buffer(c, 5, out, 0);
  gpu_copy_bytes(c, 6, &p, sizeof p);
  gpu_run_kernel(
      c, ((uint32_t)INK2_DIM + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG,
      ((uint32_t)rows + MOE_MV_MAX_M - 1u) / MOE_MV_MAX_M, 1, ROW_THREADS, 1,
      1);
}

// Dispatches a float buffer copy of `count` floats on the GPU.
static void run_copy(GpuCommandBuffer *c, const Buf *src, size_t src_off,
                     const Buf *dst, size_t dst_off, uint32_t count) {
  gpu_select_kernel(c, K_copy);
  gpu_bind_buffer(c, 0, src, src_off);
  gpu_bind_buffer(c, 1, dst, dst_off);
  gpu_copy_bytes(c, 2, &count, sizeof count);
  gpu_run_kernel(c, (count + ELEM_THREADS - 1u) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
}

// Dispatches elementwise residual addition out = a + b with optional BF16
// rounding.
static void run_add(GpuCommandBuffer *c, const Buf *a, const Buf *b,
                    const Buf *out, uint32_t count, int round) {
  uint32_t args[2] = {count, (uint32_t)round};
  gpu_select_kernel(c, K_add);
  gpu_bind_buffer(c, 0, a, 0);
  gpu_bind_buffer(c, 1, b, 0);
  gpu_bind_buffer(c, 2, out, 0);
  gpu_copy_bytes(c, 3, args, sizeof args);
  gpu_run_kernel(c, (count + ELEM_THREADS - 1u) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
}

// Dispatches static short convolution and updates the ring buffer.
static void run_sconv_static(GpuCommandBuffer *c, const Buf *x, const Buf *w,
                             const Buf *y, const Ink2State *s, int visit,
                             size_t ring_sub_off, int ring_kind, int rows,
                             int dim, int pos0, int batched) {
  SconvParam p = {(uint32_t)rows, (uint32_t)dim, (uint32_t)INK2_SCONV_RING,
                  (uint32_t)pos0, 1u};
  uint32_t gx = ((uint32_t)dim + SCONV_THREADS - 1u) / SCONV_THREADS;
  if (batched) {
    size_t tgt_off =
        ((size_t)visit * SCONV_RINGS_PER_LAYER + (size_t)ring_kind) *
        INK2_MAX_BATCH_ROWS * sizeof(SconvTarget);
    gpu_select_kernel(c, K_sconv_static_batched);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &work.sconv_tgt, tgt_off);
    gpu_bind_buffer(c, 2, w, 0);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);
    return;
  }
  size_t roff = conv_ring_off(visit, ring_sub_off);
  gpu_select_kernel(c, K_sconv_static);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, &s->conv_ring, roff);
  gpu_bind_buffer(c, 2, w, 0);
  gpu_bind_buffer(c, 3, y, 0);
  gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);

  if (rows > 1) {
    uint32_t keep = (uint32_t)(rows < INK2_SCONV_RING ? rows : INK2_SCONV_RING);
    gpu_select_kernel(c, K_sconv_ring_save);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &s->conv_ring, roff);
    gpu_copy_bytes(c, 2, &p, sizeof p);
    gpu_run_kernel(c, gx, keep, 1, SCONV_THREADS, 1, 1);
  }
}

// Dispatches dynamic short convolution (global layers' K and V) and updates
// the ring buffer.
static void run_sconv_dynamic(GpuCommandBuffer *c, const Buf *x,
                              const Buf *static_w, const Buf *dyn_w,
                              const Buf *y, const Ink2State *s, int visit,
                              size_t ring_sub_off, int ring_kind, int rows,
                              int pos0, int batched) {
  SconvParam p = {(uint32_t)rows, (uint32_t)INK2_KV_DIM,
                  (uint32_t)INK2_SCONV_RING, (uint32_t)pos0, 1u};
  uint32_t gx = ((uint32_t)INK2_KV_DIM + SCONV_THREADS - 1u) / SCONV_THREADS;
  if (batched) {
    size_t tgt_off =
        ((size_t)visit * SCONV_RINGS_PER_LAYER + (size_t)ring_kind) *
        INK2_MAX_BATCH_ROWS * sizeof(SconvTarget);
    gpu_select_kernel(c, K_sconv_dynamic_batched);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &work.sconv_tgt, tgt_off);
    gpu_bind_buffer(c, 2, static_w, 0);
    gpu_bind_buffer(c, 3, dyn_w, 0);
    gpu_bind_buffer(c, 4, y, 0);
    gpu_copy_bytes(c, 5, &p, sizeof p);
    gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);
    return;
  }
  size_t roff = conv_ring_off(visit, ring_sub_off);
  gpu_select_kernel(c, K_sconv_dynamic);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, &s->conv_ring, roff);
  gpu_bind_buffer(c, 2, static_w, 0);
  gpu_bind_buffer(c, 3, dyn_w, 0);
  gpu_bind_buffer(c, 4, y, 0);
  gpu_copy_bytes(c, 5, &p, sizeof p);
  gpu_run_kernel(c, gx, (uint32_t)rows, 1, SCONV_THREADS, 1, 1);

  if (rows > 1) {
    uint32_t keep = (uint32_t)(rows < INK2_SCONV_RING ? rows : INK2_SCONV_RING);
    gpu_select_kernel(c, K_sconv_ring_save);
    gpu_bind_buffer(c, 0, x, 0);
    gpu_bind_buffer(c, 1, &s->conv_ring, roff);
    gpu_copy_bytes(c, 2, &p, sizeof p);
    gpu_run_kernel(c, gx, keep, 1, SCONV_THREADS, 1, 1);
  }
}

// Encodes the routed MoE expert kernels given already-populated tile metadata
// and an `expert_table` of per-expert device addresses.
static void encode_moe_experts(GpuCommandBuffer *c, const PhysLayer *pl,
                               const Buf *expert_table, int rows, int ntiles,
                               int run_mv, int run_mm) {
  int npairs = rows * INK2_TOPK;
  if (run_mm) {
    uint32_t dim4 = INK2_DIM / VEC4_WIDTH;
    gpu_select_kernel(c, K_moe_gather_bf16);
    gpu_bind_buffer(c, 0, &work.q_in, 0);
    gpu_bind_buffer(c, 1, &work.ptok, 0);
    gpu_bind_buffer(c, 2, &work.x_bf16, 0);
    gpu_copy_bytes(c, 3, &dim4, sizeof dim4);
    gpu_run_kernel(c, (dim4 + COMBINE_THREADS - 1u) / COMBINE_THREADS,
                   (uint32_t)npairs, 1, COMBINE_THREADS, 1, 1);
  }
  MoeParam mp13 = {
      (uint32_t)INK2_W13_ROWS, (uint32_t)INK2_DIM, (uint32_t)npairs,
      (uint32_t)ntiles,        GATE_SOFTCAP,       UP_SOFTCAP};
  MoeParam mp2 = {(uint32_t)INK2_DIM, (uint32_t)INK2_INTER, (uint32_t)npairs,
                  (uint32_t)ntiles,   GATE_SOFTCAP,         UP_SOFTCAP};
  uint32_t gx13_mv =
      ((uint32_t)INK2_W13_ROWS + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG;
  uint32_t gx13_mm = ((uint32_t)INK2_W13_ROWS + MOE_MM_TILE - 1u) / MOE_MM_TILE;
  uint32_t gx2_mv =
      ((uint32_t)INK2_DIM + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG;
  uint32_t gx2_mm = ((uint32_t)INK2_DIM + MOE_MM_TILE - 1u) / MOE_MM_TILE;

  if (pl->is_nvfp4) {
    if (run_mv) {
      gpu_select_kernel(c, K_nv_up_mv);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &pl->f13, 0);
      gpu_bind_buffer(c, 2, &work.q_in, 0);
      gpu_bind_buffer(c, 3, &work.ptok, 0);
      gpu_bind_buffer(c, 4, &work.tiles, 0);
      gpu_bind_buffer(c, 5, &work.h_bf16, 0);
      gpu_copy_bytes(c, 6, &mp13, sizeof mp13);
      gpu_run_kernel(c, gx13_mv, (uint32_t)ntiles, 1, MV_THREADS, 1, 1);
    }
    if (run_mm) {
      gpu_select_kernel(c, K_nv_up_mm);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &pl->f13, 0);
      gpu_bind_buffer(c, 2, &work.x_bf16, 0);
      gpu_bind_buffer(c, 3, &work.tiles, 0);
      gpu_bind_buffer(c, 4, &work.h_bf16, 0);
      gpu_copy_bytes(c, 5, &mp13, sizeof mp13);
      gpu_run_kernel(c, gx13_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
    }
    if (run_mv) {
      gpu_select_kernel(c, K_nv_dn_mv);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &pl->f2, 0);
      gpu_bind_buffer(c, 2, &work.h_bf16, 0);
      gpu_bind_buffer(c, 3, &work.pw, 0);
      gpu_bind_buffer(c, 4, &work.tiles, 0);
      gpu_bind_buffer(c, 5, &work.y_pair, 0);
      gpu_copy_bytes(c, 6, &mp2, sizeof mp2);
      gpu_run_kernel(c, gx2_mv, (uint32_t)ntiles, 1, MV_THREADS, 1, 1);
    }
    if (run_mm) {
      gpu_select_kernel(c, K_nv_dn_mm);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &pl->f2, 0);
      gpu_bind_buffer(c, 2, &work.h_bf16, 0);
      gpu_bind_buffer(c, 3, &work.pw, 0);
      gpu_bind_buffer(c, 4, &work.tiles, 0);
      gpu_bind_buffer(c, 5, &work.y_pair, 0);
      gpu_copy_bytes(c, 6, &mp2, sizeof mp2);
      gpu_run_kernel(c, gx2_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
    }
  } else {
    if (run_mv) {
      gpu_select_kernel(c, K_mx_up_mv);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &work.q_in, 0);
      gpu_bind_buffer(c, 2, &work.ptok, 0);
      gpu_bind_buffer(c, 3, &work.tiles, 0);
      gpu_bind_buffer(c, 4, &work.h_bf16, 0);
      gpu_copy_bytes(c, 5, &mp13, sizeof mp13);
      gpu_run_kernel(c, gx13_mv, (uint32_t)ntiles, 1, MV_THREADS, 1, 1);
    }
    if (run_mm) {
      gpu_select_kernel(c, K_mx_up_mm);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &work.x_bf16, 0);
      gpu_bind_buffer(c, 2, &work.tiles, 0);
      gpu_bind_buffer(c, 3, &work.h_bf16, 0);
      gpu_copy_bytes(c, 4, &mp13, sizeof mp13);
      gpu_run_kernel(c, gx13_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
    }
    if (run_mv) {
      gpu_select_kernel(c, K_mx_dn_mv);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &work.h_bf16, 0);
      gpu_bind_buffer(c, 2, &work.pw, 0);
      gpu_bind_buffer(c, 3, &work.tiles, 0);
      gpu_bind_buffer(c, 4, &work.y_pair, 0);
      gpu_copy_bytes(c, 5, &mp2, sizeof mp2);
      gpu_run_kernel(c, gx2_mv, (uint32_t)ntiles, 1, MV_THREADS, 1, 1);
    }
    if (run_mm) {
      gpu_select_kernel(c, K_mx_dn_mm);
      gpu_bind_buffer(c, 0, expert_table, 0);
      gpu_bind_buffer(c, 1, &work.h_bf16, 0);
      gpu_bind_buffer(c, 2, &work.pw, 0);
      gpu_bind_buffer(c, 3, &work.tiles, 0);
      gpu_bind_buffer(c, 4, &work.y_pair, 0);
      gpu_copy_bytes(c, 5, &mp2, sizeof mp2);
      gpu_run_kernel(c, gx2_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
    }
  }

  MoeCombineParam mcp = {(uint32_t)rows, (uint32_t)INK2_DIM,
                         (uint32_t)INK2_TOPK};
  gpu_select_kernel(c, K_moe_combine);
  gpu_bind_buffer(c, 0, &work.y_pair, 0);
  gpu_bind_buffer(c, 1, &work.tp, 0);
  gpu_bind_buffer(c, 2, &work.cur2, 0);
  gpu_copy_bytes(c, 3, &mcp, sizeof mcp);
  gpu_run_kernel(c, ((uint32_t)INK2_DIM + SCONV_THREADS - 1u) / SCONV_THREADS,
                 (uint32_t)rows, 1, SCONV_THREADS, 1, 1);
}

// Encodes m = 1 resident MoE experts with fused down + combine + post-MLP
// static short convolution + residual add.
static void encode_moe_m1_fused(GpuCommandBuffer *c, const PhysLayer *pl,
                                const Buf *expert_table, const Ink2State *s,
                                int visit, int pos0, size_t route_ids_off) {
  MoeParam mp13 = {
      (uint32_t)INK2_W13_ROWS, (uint32_t)INK2_DIM, (uint32_t)INK2_TOPK,
      (uint32_t)INK2_TOPK,     GATE_SOFTCAP,       UP_SOFTCAP};
  MoeDownM1Param dp = {(uint32_t)INK2_DIM, (uint32_t)INK2_INTER, (uint32_t)pos0,
                       (uint32_t)INK2_SCONV_RING};
  uint32_t gx13_mv =
      ((uint32_t)INK2_W13_ROWS + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG;
  uint32_t gx_dn =
      ((uint32_t)INK2_DIM + MOE_MV_DOWN_M1_ROWS - 1u) / MOE_MV_DOWN_M1_ROWS;
  size_t ring_off = conv_ring_off(visit, CONV_MLP_OFF);
  if (pl->is_nvfp4) {
    gpu_select_kernel(c, K_nv_up_m1);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &pl->f13, 0);
    gpu_bind_buffer(c, 2, &work.q_in, 0);
    gpu_bind_buffer(c, 3, &work.route_ids, route_ids_off);
    gpu_bind_buffer(c, 4, &work.h_bf16, 0);
    gpu_copy_bytes(c, 5, &mp13, sizeof mp13);
    gpu_run_kernel(c, gx13_mv, (uint32_t)INK2_TOPK, 1, MV_THREADS, 1, 1);

    gpu_select_kernel(c, K_nv_dn_comb_m1);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &pl->f2, 0);
    gpu_bind_buffer(c, 2, &work.h_bf16, 0);
    gpu_bind_buffer(c, 3, &work.route_ids, route_ids_off);
    gpu_bind_buffer(c, 4, &work.route_w, 0);
    gpu_bind_buffer(c, 5, &s->conv_ring, ring_off);
    gpu_bind_buffer(c, 6, &pl->mlp_sconv_w, 0);
    gpu_bind_buffer(c, 7, &work.residual, 0);
    gpu_bind_buffer(c, 8, &work.cur, 0);
    gpu_copy_bytes(c, 9, &dp, sizeof dp);
    gpu_run_kernel(c, gx_dn, 1, 1, MV_THREADS, 1, 1);
  } else {
    gpu_select_kernel(c, K_mx_up_m1);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &work.q_in, 0);
    gpu_bind_buffer(c, 2, &work.route_ids, route_ids_off);
    gpu_bind_buffer(c, 3, &work.h_bf16, 0);
    gpu_copy_bytes(c, 4, &mp13, sizeof mp13);
    gpu_run_kernel(c, gx13_mv, (uint32_t)INK2_TOPK, 1, MV_THREADS, 1, 1);

    gpu_select_kernel(c, K_mx_dn_comb_m1);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &work.h_bf16, 0);
    gpu_bind_buffer(c, 2, &work.route_ids, route_ids_off);
    gpu_bind_buffer(c, 3, &work.route_w, 0);
    gpu_bind_buffer(c, 4, &s->conv_ring, ring_off);
    gpu_bind_buffer(c, 5, &pl->mlp_sconv_w, 0);
    gpu_bind_buffer(c, 6, &work.residual, 0);
    gpu_bind_buffer(c, 7, &work.cur, 0);
    gpu_copy_bytes(c, 8, &dp, sizeof dp);
    gpu_run_kernel(c, gx_dn, 1, 1, MV_THREADS, 1, 1);
  }
}

// ---- forward pass ----

// Runs one logical visit v (0..47) over `rows` tokens. Prefill visits sync
// around routing, check `wait` while holding no experts, and stream active
// experts through experts_store. Decode and verify visits stay on GPU in *cmd
// when the physical layer is fully resident and committed, or sync and stream
// missing experts from SSD otherwise.
static void run_visit(GpuCommandBuffer *cmd, int v, int rows, int pos0,
                      int max_pos, const Ink2State *s, int batched,
                      int is_prefill, int *nblocks, const PassWait *wait,
                      uint8_t fast_visits[INK2_TOTAL_LAYERS]) {
  const Ink2Visit *vis = &schedule[v];
  int l = vis->phys;
  const PhysLayer *pl = &layers[l];
  size_t row_floats = (size_t)rows * INK2_DIM;
  size_t stride_bytes = (size_t)work.cap_rows * INK2_DIM * sizeof(float);
  int use_msg = (vis->loop_pass > 0);
  if (is_prefill) expert_store_set_prompt_layer(experts_store, l);

  if (use_msg) {
    size_t q_off =
        (size_t)vis->loop_idx * INK2_MSG_COUNT * INK2_DIM * sizeof(float);
    run_attn_res(cmd, &work.prev_blk, &work.prev_cur, &msg_queries, q_off,
                 &work.msg, rows, work.cap_rows, 1, INK2_MSG_COUNT,
                 work.cap_rows);
  }

  // Attention input via AttnRes, optional loop message gates, and input norm.
  const Buf *q_src = &work.q_in;
  const Buf *k_src = &work.q_in;
  const Buf *v_src = &work.q_in;
  if (vis->phys == 0) {
    run_rmsnorm(cmd, &work.cur, 0, &pl->attn_norm, &work.q_in, rows, 1);
  } else if (pl->is_global) {
    if (use_msg) {
      run_attn_res(cmd, &work.blk, &work.cur, &pl->attn_query, 0, &work.attn_in,
                   rows, work.cap_rows, *nblocks, ATTNRES_GLOBAL_QUERIES,
                   work.cap_rows);
      run_msg_gate_3_rmsnorm(cmd, &work.attn_in, &work.msg, &pl->ctrl_q,
                             &pl->ctrl_k, &pl->ctrl_v, &pl->q_norm, &pl->k_norm,
                             &pl->v_norm, rows, 1, is_prefill);
    } else {
      run_attn_res_rmsnorm(cmd, &work.blk, 0, &work.cur, 0, &pl->attn_query, 0,
                           &pl->q_norm, &pl->k_norm, &pl->v_norm, &work.q_in,
                           &work.k_in, &work.v_in, rows, work.cap_rows,
                           *nblocks, ATTNRES_GLOBAL_QUERIES);
    }
    k_src = &work.k_in;
    v_src = &work.v_in;
  } else {
    if (use_msg) {
      run_attn_res(cmd, &work.blk, &work.cur, &pl->attn_query, 0, &work.attn_in,
                   rows, work.cap_rows, *nblocks, 1, work.cap_rows);
      run_msg_gate_3_rmsnorm(
          cmd, &work.attn_in, &work.msg, &pl->ctrl_q, &pl->ctrl_k, &pl->ctrl_v,
          &pl->attn_norm, &pl->attn_norm, &pl->attn_norm, rows, 0, is_prefill);
      k_src = &work.k_in;
      v_src = &work.v_in;
    } else {
      run_attn_res_rmsnorm(cmd, &work.blk, 0, &work.cur, 0, &pl->attn_query, 0,
                           &pl->attn_norm, NULL, NULL, &work.q_in, NULL, NULL,
                           rows, work.cap_rows, *nblocks, 1);
    }
  }

  // Q, R, gate, sink, K and V projections and K/V short convolutions.
  int fuse_kv_prep_m1 = (!pl->is_global && rows == 1 && !batched);
  run_bf16_linear3(cmd, &pl->wq, &work.q_raw, &pl->gate_attn, &work.gate,
                   &pl->wr, &work.r_raw, q_src, rows, 1, is_prefill);
  if (!pl->is_global && !use_msg) {
    run_bf16_linear3(cmd, &pl->wk, &work.k_raw, &pl->wv, &work.v_raw,
                     &pl->sink_proj, &work.sink, q_src, rows, 1, is_prefill);
    if (!fuse_kv_prep_m1) {
      run_sconv_static(cmd, &work.k_raw, &pl->k_sconv_w, &work.k_conv, s, v,
                       CONV_K_OFF, 0, rows, INK2_KV_DIM, pos0, batched);
      run_sconv_static(cmd, &work.v_raw, &pl->v_sconv_w, &work.v_conv, s, v,
                       CONV_V_OFF, 1, rows, INK2_KV_DIM, pos0, batched);
    }
  } else {
    run_bf16_linear(cmd, &pl->sink_proj.w, q_src, &work.sink, rows,
                    pl->sink_proj.n, pl->sink_proj.k, pl->sink_proj.n, 1,
                    is_prefill);
    if (pl->is_global) {
      run_bf16_linear2(cmd, &pl->wk, &work.k_raw, &pl->k_sconv_proj,
                       &work.dyn_w, k_src, rows, 1, is_prefill);
      run_sconv_dynamic(cmd, &work.k_raw, &pl->k_sconv_w, &work.dyn_w,
                        &work.k_conv, s, v, CONV_K_OFF, 0, rows, pos0, batched);

      run_bf16_linear2(cmd, &pl->wv, &work.v_raw, &pl->v_sconv_proj,
                       &work.dyn_w, v_src, rows, 1, is_prefill);
      run_sconv_dynamic(cmd, &work.v_raw, &pl->v_sconv_w, &work.dyn_w,
                        &work.v_conv, s, v, CONV_V_OFF, 1, rows, pos0, batched);
    } else {
      run_bf16_linear(cmd, &pl->wk.w, k_src, &work.k_raw, rows, pl->wk.n,
                      pl->wk.k, pl->wk.n, 1, is_prefill);
      run_bf16_linear(cmd, &pl->wv.w, v_src, &work.v_raw, rows, pl->wv.n,
                      pl->wv.k, pl->wv.n, 1, is_prefill);
      if (!fuse_kv_prep_m1) {
        run_sconv_static(cmd, &work.k_raw, &pl->k_sconv_w, &work.k_conv, s, v,
                         CONV_K_OFF, 0, rows, INK2_KV_DIM, pos0, batched);
        run_sconv_static(cmd, &work.v_raw, &pl->v_sconv_w, &work.v_conv, s, v,
                         CONV_V_OFF, 1, rows, INK2_KV_DIM, pos0, batched);
      }
    }
  }

  // Q/K/V/R head prep and KV cache store (fused with K/V sconv on m = 1
  // sliding visits; relative position bias is computed on the fly in
  // attention).
  const Buf *k_buf = pl->is_global ? &s->gk[vis->global_slot] : &s->slide_kv;
  const Buf *v_buf = pl->is_global ? &s->gv[vis->global_slot] : &s->slide_kv;
  size_t k_off = pl->is_global ? 0 : slide_ring_off(vis->slide_slot, 0);
  size_t v_off = pl->is_global ? 0 : slide_ring_off(vis->slide_slot, 1);
  uint32_t ring_rows = pl->is_global ? 0u : (uint32_t)INK2_SLIDE_RING;
  size_t kv_tgt_off = (size_t)v * INK2_MAX_BATCH_ROWS * sizeof(KvTarget);

  QkvrParam qp = {(uint32_t)rows, (uint32_t)work.q_stride, (uint32_t)pos0,
                  ring_rows,      (uint32_t)pl->is_global, (uint32_t)batched};
  if (fuse_kv_prep_m1) {
    gpu_select_kernel(cmd, opt.inkling2.fp8_kv ? K_sconv_qkvr_fp8_m1
                                               : K_sconv_qkvr_m1);
    gpu_bind_buffer(cmd, 0, &work.q_raw, 0);
    gpu_bind_buffer(cmd, 1, &work.k_raw, 0);
    gpu_bind_buffer(cmd, 2, &work.v_raw, 0);
    gpu_bind_buffer(cmd, 3, &work.r_raw, 0);
    gpu_bind_buffer(cmd, 4, &s->conv_ring, conv_ring_off(v, CONV_K_OFF));
    gpu_bind_buffer(cmd, 5, &pl->k_sconv_w, 0);
    gpu_bind_buffer(cmd, 6, &s->conv_ring, conv_ring_off(v, CONV_V_OFF));
    gpu_bind_buffer(cmd, 7, &pl->v_sconv_w, 0);
    gpu_bind_buffer(cmd, 8, &pl->q_clip_scale, 0);
    gpu_bind_buffer(cmd, 9, &pl->rel_clip_scale, 0);
    gpu_bind_buffer(cmd, 10, &work.q_hm, 0);
    gpu_bind_buffer(cmd, 11, k_buf, k_off);
    gpu_bind_buffer(cmd, 12, v_buf, v_off);
    gpu_bind_buffer(cmd, 13, &work.r_out, 0);
    gpu_copy_bytes(cmd, 14, &qp, sizeof qp);
    gpu_run_kernel(cmd, QKVR_HEADS, 1, 1, QKVR_THREADS, 1, 1);
  } else {
    gpu_select_kernel(cmd, opt.inkling2.fp8_kv ? K_qkvr_prep_fp8 : K_qkvr_prep);
    gpu_bind_buffer(cmd, 0, &work.q_raw, 0);
    gpu_bind_buffer(cmd, 1, &work.k_conv, 0);
    gpu_bind_buffer(cmd, 2, &work.v_conv, 0);
    gpu_bind_buffer(cmd, 3, &work.r_raw, 0);
    gpu_bind_buffer(cmd, 4, &pl->q_clip_scale, 0);
    gpu_bind_buffer(cmd, 5, &pl->rel_clip_scale, 0);
    gpu_bind_buffer(cmd, 6, &work.q_hm, 0);
    gpu_bind_buffer(cmd, 7, k_buf, k_off);
    gpu_bind_buffer(cmd, 8, v_buf, v_off);
    gpu_bind_buffer(cmd, 9, &work.r_out, 0);
    gpu_bind_buffer(cmd, 10, &work.kv_tgt, kv_tgt_off);
    gpu_copy_bytes(cmd, 11, &qp, sizeof qp);
    gpu_run_kernel(cmd, QKVR_HEADS, (uint32_t)rows, 1, QKVR_THREADS, 1, 1);
  }

  // Attention: tiled MPP FlashAttention for prefill, FlashDecoding otherwise.
  uint32_t window = pl->is_global ? 0u : (uint32_t)INK2_WINDOW;
  if (is_prefill) {
    uint32_t q_tiles =
        ((uint32_t)rows + PREFILL_Q_GROUP - 1u) / PREFILL_Q_GROUP;
    if (!opt.inkling2.fp8_kv) {
      AttendPrefillParam app = {(uint32_t)rows, (uint32_t)work.q_stride,
                                (uint32_t)pos0, ring_rows, window};
      gpu_select_kernel(cmd, K_attend_prefill_rel);
      gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
      gpu_bind_buffer(cmd, 1, k_buf, k_off);
      gpu_bind_buffer(cmd, 2, v_buf, v_off);
      gpu_bind_buffer(cmd, 3, &work.r_out, 0);
      gpu_bind_buffer(cmd, 4, &pl->rel_proj_t, 0);
      gpu_bind_buffer(cmd, 5, &work.sink, 0);
      gpu_bind_buffer(cmd, 6, &work.gate, 0);
      gpu_bind_buffer(cmd, 7, &work.attn_out, 0);
      gpu_copy_bytes(cmd, 8, &app, sizeof app);
      gpu_run_kernel(cmd, q_tiles * (uint32_t)INK2_HEADS, 1, 1, PREFILL_THREADS,
                     1, 1);
    } else if (!pl->is_global || pos0 + rows <= FP8_PREFILL_SLAB_ROWS) {
      uint32_t n_deq =
          pl->is_global ? (uint32_t)(pos0 + rows) : (uint32_t)INK2_SLIDE_RING;
      if (!pl->is_global && (uint32_t)(pos0 + rows) < n_deq)
        n_deq = (uint32_t)(pos0 + rows);
      Fp8DequantParam dp = {0u, n_deq};
      gpu_select_kernel(cmd, K_fp8_dequant_kv);
      gpu_bind_buffer(cmd, 0, k_buf, k_off);
      gpu_bind_buffer(cmd, 1, v_buf, v_off);
      gpu_bind_buffer(cmd, 2, &work.deq_k, 0);
      gpu_bind_buffer(cmd, 3, &work.deq_v, 0);
      gpu_copy_bytes(cmd, 4, &dp, sizeof dp);
      gpu_run_kernel(cmd, n_deq, 1, 1, ROW_THREADS, 1, 1);

      AttendPrefillParam app = {(uint32_t)rows, (uint32_t)work.q_stride,
                                (uint32_t)pos0, ring_rows, window};
      gpu_select_kernel(cmd, K_attend_prefill_rel);
      gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
      gpu_bind_buffer(cmd, 1, &work.deq_k, 0);
      gpu_bind_buffer(cmd, 2, &work.deq_v, 0);
      gpu_bind_buffer(cmd, 3, &work.r_out, 0);
      gpu_bind_buffer(cmd, 4, &pl->rel_proj_t, 0);
      gpu_bind_buffer(cmd, 5, &work.sink, 0);
      gpu_bind_buffer(cmd, 6, &work.gate, 0);
      gpu_bind_buffer(cmd, 7, &work.attn_out, 0);
      gpu_copy_bytes(cmd, 8, &app, sizeof app);
      gpu_run_kernel(cmd, q_tiles * (uint32_t)INK2_HEADS, 1, 1, PREFILL_THREADS,
                     1, 1);
    } else {
      uint32_t total_k = (uint32_t)(pos0 + rows);
      for (uint32_t ks = 0; ks < total_k;
           ks += (uint32_t)FP8_PREFILL_SLAB_ROWS) {
        uint32_t kc = total_k - ks;
        if (kc > (uint32_t)FP8_PREFILL_SLAB_ROWS)
          kc = (uint32_t)FP8_PREFILL_SLAB_ROWS;
        Fp8DequantParam dp = {ks, kc};
        gpu_select_kernel(cmd, K_fp8_dequant_kv);
        gpu_bind_buffer(cmd, 0, k_buf, k_off);
        gpu_bind_buffer(cmd, 1, v_buf, v_off);
        gpu_bind_buffer(cmd, 2, &work.deq_k, 0);
        gpu_bind_buffer(cmd, 3, &work.deq_v, 0);
        gpu_copy_bytes(cmd, 4, &dp, sizeof dp);
        gpu_run_kernel(cmd, kc, 1, 1, ROW_THREADS, 1, 1);

        AttendSlabParam sp = {(uint32_t)rows,
                              (uint32_t)work.q_stride,
                              (uint32_t)pos0,
                              ks,
                              kc,
                              ks == 0u ? 1u : 0u,
                              ks + kc == total_k ? 1u : 0u};
        gpu_select_kernel(cmd, K_attend_prefill_slab_rel);
        gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
        gpu_bind_buffer(cmd, 1, &work.deq_k, 0);
        gpu_bind_buffer(cmd, 2, &work.deq_v, 0);
        gpu_bind_buffer(cmd, 3, &work.r_out, 0);
        gpu_bind_buffer(cmd, 4, &pl->rel_proj_t, 0);
        gpu_bind_buffer(cmd, 5, &work.sink, 0);
        gpu_bind_buffer(cmd, 6, &work.gate, 0);
        gpu_bind_buffer(cmd, 7, &work.attn_out, 0);
        gpu_bind_buffer(cmd, 8, &work.slab_ml, 0);
        gpu_copy_bytes(cmd, 9, &sp, sizeof sp);
        gpu_run_kernel(cmd, q_tiles * (uint32_t)INK2_HEADS, 1, 1,
                       PREFILL_THREADS, 1, 1);
      }
    }
  } else {
    int total_k = max_pos + 1;
    if (window && total_k > (int)window) total_k = (int)window;
    int nparts = (total_k + DECODE_PART_TARGET - 1) / DECODE_PART_TARGET;
    if (nparts < 1) nparts = 1;
    if (nparts > MAX_DECODE_PARTS) nparts = MAX_DECODE_PARTS;
    AttendDecodeParam adp = {(uint32_t)rows,
                             (uint32_t)work.q_stride,
                             (uint32_t)pos0,
                             ring_rows,
                             window,
                             (uint32_t)nparts,
                             (uint32_t)MAX_DECODE_PARTS,
                             (uint32_t)batched};
    gpu_select_kernel(cmd, opt.inkling2.fp8_kv ? K_attend_part_fp8_rel
                                               : K_attend_part_rel);
    gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
    gpu_bind_buffer(cmd, 1, k_buf, k_off);
    gpu_bind_buffer(cmd, 2, v_buf, v_off);
    gpu_bind_buffer(cmd, 3, &work.r_out, 0);
    gpu_bind_buffer(cmd, 4, &pl->rel_proj_t, 0);
    gpu_bind_buffer(cmd, 5, &work.kv_tgt, kv_tgt_off);
    gpu_bind_buffer(cmd, 6, &work.part, 0);
    gpu_copy_bytes(cmd, 7, &adp, sizeof adp);
    gpu_run_kernel(cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                   (uint32_t)rows, DECODE_THREADS, 1, 1);

    gpu_select_kernel(cmd, K_attend_combine);
    gpu_bind_buffer(cmd, 0, &work.part, 0);
    gpu_bind_buffer(cmd, 1, &work.sink, 0);
    gpu_bind_buffer(cmd, 2, &work.gate, 0);
    gpu_bind_buffer(cmd, 3, &work.attn_out, 0);
    gpu_copy_bytes(cmd, 4, &adp, sizeof adp);
    gpu_run_kernel(cmd, (uint32_t)INK2_HEADS, (uint32_t)rows, 1,
                   COMBINE_THREADS, 1, 1);
  }

  // Output projection and fused post-attention static short convolution +
  // residual update (or block boundary push into blk).
  run_bf16_linear(cmd, &pl->wo.w, &work.attn_out, &work.cur2, rows, pl->wo.n,
                  pl->wo.k, pl->wo.n, 1, is_prefill);
  size_t blk_off = vis->is_boundary ? (size_t)(*nblocks) * stride_bytes : 0;
  run_sconv_static_post(cmd, &work.cur2, &pl->attn_sconv_w, &work.cur,
                        &work.residual, &work.blk, blk_off, s, v, CONV_ATTN_OFF,
                        2, rows, pos0, batched, vis->is_boundary);
  if (vis->is_boundary) (*nblocks)++;

  // MLP input via AttnRes, optional loop message gate, and mlp_norm.
  if (use_msg) {
    run_attn_res(cmd, &work.blk, &work.residual, &pl->mlp_query, 0,
                 &work.attn_in, rows, work.cap_rows, *nblocks, 1,
                 work.cap_rows);
    run_msg_gate(cmd, &work.attn_in, 0, &work.msg, 3 * stride_bytes,
                 &pl->ctrl_mlp, &work.msg_gated, rows, is_prefill);
    run_rmsnorm(cmd, &work.msg_gated, 0, &pl->mlp_norm, &work.q_in, rows, 1);
  } else {
    run_attn_res_rmsnorm(cmd, &work.blk, 0, &work.residual, 0, &pl->mlp_query,
                         0, &pl->mlp_norm, NULL, NULL, &work.q_in, NULL, NULL,
                         rows, work.cap_rows, *nblocks, 1);
  }

  // Router logits (FP32, unrounded) and biased top-8 selection.
  size_t route_ids_off =
      is_prefill ? 0 : (size_t)v * (size_t)rows * INK2_TOPK * sizeof(uint32_t);
  run_bf16_linear(cmd, &pl->router_w.w, &work.q_in, &work.route_logits, rows,
                  pl->router_w.n, pl->router_w.k, pl->router_w.n, 0,
                  is_prefill);
  RouteParam rp = {(uint32_t)rows, (uint32_t)INK2_EXPERTS, (uint32_t)INK2_TOPK,
                   ROUTE_SCALE};
  uint64_t fast_gen = 0;
  int fast_m1 = !is_prefill && rows == 1 && !batched &&
                expert_store_layer_is_reachable(experts_store, l, &fast_gen);
  if (fast_m1) {
    hold_layer_if_needed(l);
    refresh_layer_table(l, fast_gen);
    gpu_select_kernel(cmd, K_route_topk_m1);
    gpu_bind_buffer(cmd, 0, &work.route_logits, 0);
    gpu_bind_buffer(cmd, 1, &pl->router_bias, 0);
    gpu_bind_buffer(cmd, 2, &work.route_ids, route_ids_off);
    gpu_bind_buffer(cmd, 3, &work.route_w, 0);
    gpu_bind_buffer(cmd, 4, &work.ptok, 0);
    gpu_bind_buffer(cmd, 5, &work.pw, 0);
    gpu_bind_buffer(cmd, 6, &work.tp, 0);
    gpu_bind_buffer(cmd, 7, &work.tiles, 0);
    gpu_copy_bytes(cmd, 8, &rp, sizeof rp);
    gpu_run_kernel(cmd, 1, 1, 1, ROUTE_THREADS, 1, 1);
    encode_moe_m1_fused(cmd, pl, &layer_table[l], s, v, pos0, route_ids_off);
    fast_visits[v] = 1;
    return;
  }

  gpu_select_kernel(cmd, K_route_topk);
  gpu_bind_buffer(cmd, 0, &work.route_logits, 0);
  gpu_bind_buffer(cmd, 1, &pl->router_bias, 0);
  gpu_bind_buffer(cmd, 2, &work.route_ids, route_ids_off);
  gpu_bind_buffer(cmd, 3, &work.route_w, 0);
  gpu_copy_bytes(cmd, 4, &rp, sizeof rp);
  gpu_run_kernel(cmd, (uint32_t)rows, 1, 1, ROUTE_THREADS, 1, 1);

  // Tile grouping and expert execution.
  if (is_prefill) {
    model_commit(cmd);
    model_sync();
    held_release_all();
    expert_store_resize_in_prompt(experts_store, l);
    if (wait && wait->ready && !wait->ready(wait->context)) pass_wait(wait);
    int ntiles = inkling2_group_expert_tiles(
        (const uint32_t *)work.route_ids.ptr, (const float *)work.route_w.ptr,
        rows, MOE_MM_TILE, (int *)work.ptok.ptr, (float *)work.pw.ptr,
        (int *)work.tp.ptr, (Ink2MoeTile *)work.tiles.ptr);
    int has_mv = 0, has_mm = 0, need[INK2_EXPERTS], nn = 0;
    uint8_t used[INK2_EXPERTS] = {0};
    const Ink2MoeTile *tl = (const Ink2MoeTile *)work.tiles.ptr;
    for (int t = 0; t < ntiles; t++) {
      used[tl[t].expert] = 1;
      if (tl[t].count <= (uint32_t)MOE_MV_MAX_M)
        has_mv = 1;
      else
        has_mm = 1;
    }
    for (int e = 0; e < INK2_EXPERTS; e++)
      if (used[e]) need[nn++] = e;
    hold_experts(l, need, nn);
    expert_store_cancel_read_aheads(experts_store, l, need, nn);
    expert_store_unprotect_read_aheads(experts_store, l);
    if (rows >= PREFILL_BULK_ROWS && v + 1 < INK2_TOTAL_LAYERS &&
        schedule[v + 1].phys == l + 1)
      prefetch_all_layer_experts(l + 1);
    expert_store_wait_for_reads(experts_store, l, need, nn);
    *cmd = gpu_begin(gpu);
    fill_streaming_table(cmd, l, need, nn);
    encode_moe_experts(cmd, pl, &work.expert_table, rows, ntiles, has_mv,
                       has_mm);
  } else {
    uint64_t gen = 0;
    if (expert_store_layer_is_reachable(experts_store, l, &gen)) {
      hold_layer_if_needed(l);
      refresh_layer_table(l, gen);
      uint32_t m_topk[2] = {(uint32_t)rows, (uint32_t)INK2_TOPK};
      gpu_select_kernel(cmd, K_moe_group);
      gpu_bind_buffer(cmd, 0, &work.route_ids, route_ids_off);
      gpu_bind_buffer(cmd, 1, &work.route_w, 0);
      gpu_bind_buffer(cmd, 2, &work.ptok, 0);
      gpu_bind_buffer(cmd, 3, &work.pw, 0);
      gpu_bind_buffer(cmd, 4, &work.tp, 0);
      gpu_bind_buffer(cmd, 5, &work.tiles, 0);
      gpu_copy_bytes(cmd, 6, m_topk, sizeof m_topk);
      gpu_run_kernel(cmd, 1, 1, 1, ROUTE_THREADS, 1, 1);
      int ntiles = rows * INK2_TOPK;
      int run_mm = rows > MOE_MV_MAX_M ? 1 : 0;
      encode_moe_experts(cmd, pl, &layer_table[l], rows, ntiles, 1, run_mm);
      fast_visits[v] = 1;
    } else {
      model_commit(cmd);
      model_sync();
      flush_fast_visits(fast_visits, rows);
      held_release_all();

      const uint32_t *rids = (const uint32_t *)work.route_ids.ptr +
                             (size_t)v * (size_t)rows * INK2_TOPK;
      ExpertStoreCounts *st = expert_stats();
      uint8_t used[INK2_EXPERTS] = {0};
      int need[INK2_EXPERTS], nn = 0;
      for (int r = 0; r < rows; r++) {
        for (int k = 0; k < INK2_TOPK; k++) {
          uint32_t e = rids[(size_t)r * INK2_TOPK + (size_t)k];
          if (e >= (uint32_t)INK2_EXPERTS) die("router selected expert %u", e);
          used[e] = 1;
          ExpertPlace place = expert_store_locate(experts_store, l, (int)e);
          if (place == EXPERT_IN_MEMORY)
            st->row_own++;
          else if (place == EXPERT_READ_AHEAD)
            st->row_ahead++;
          else if (place == EXPERT_BEING_READ_AHEAD)
            st->row_late++;
          else
            st->row_ssd++;
          if (r < ROUTE_DECODE_CAP) {
            route_id[r][v][k] = (uint16_t)e;
            route_from[r][v][k] = expert_place_to_route_source(place);
            route_step[r][v] = route_steps;
          }
        }
      }
      for (int e = 0; e < INK2_EXPERTS; e++)
        if (used[e]) need[nn++] = e;
      st->row_unique += (uint64_t)nn;

      hold_experts(l, need, nn);
      expert_store_cancel_read_aheads(experts_store, l, need, nn);
      expert_store_unprotect_read_aheads(experts_store, l);
      credit_near_misses(l, rows, used);
      expert_store_wait_for_reads(experts_store, l, need, nn);

      int ntiles = inkling2_group_expert_tiles(
          rids, (const float *)work.route_w.ptr, rows, MOE_MM_TILE,
          (int *)work.ptok.ptr, (float *)work.pw.ptr, (int *)work.tp.ptr,
          (Ink2MoeTile *)work.tiles.ptr);
      int has_mv = 0, has_mm = 0;
      const Ink2MoeTile *tl = (const Ink2MoeTile *)work.tiles.ptr;
      for (int t = 0; t < ntiles; t++) {
        if (tl[t].count <= (uint32_t)MOE_MV_MAX_M)
          has_mv = 1;
        else
          has_mm = 1;
      }
      *cmd = gpu_begin(gpu);
      fill_streaming_table(cmd, l, need, nn);
      encode_moe_experts(cmd, pl, &work.expert_table, rows, ntiles, has_mv,
                         has_mm);
    }
  }

  // Post-MLP static short convolution and BF16-rounded residual update.
  run_sconv_static(cmd, &work.cur2, &pl->mlp_sconv_w, &work.cur, s, v,
                   CONV_MLP_OFF, 3, rows, INK2_DIM, pos0, batched);
  run_add(cmd, &work.residual, &work.cur, &work.cur, (uint32_t)row_floats, 1);
}

// Returns the encoded BF16 feature row for prompt position `pos` inside an
// image span.
static const uint16_t *find_media_row(int pos) {
  for (int i = 0; i < prefill_media_count; i++) {
    const Media *im = &prefill_media[i];
    int j = pos - im->start;
    if (j < 0 || j >= im->ntokens) continue;
    if (!im->rows) die("image at position %d was not encoded", im->start);
    return im->rows + (size_t)j * INK2_DIM;
  }
  die("position %d: negative media token outside registered images", pos);
}

// Unpacks encoded BF16 image feature rows into work.cur for any negative
// placeholder tokens in [pos0, pos0 + rows). ink2_embed_norm skips id < 0.
static void inject_media_rows(const int *tokens, int pos0, int rows) {
  float *cur = (float *)work.cur.ptr;
  for (int r = 0; r < rows; r++) {
    if (tokens[r] >= 0) continue;
    const uint16_t *src = find_media_row(pos0 + r);
    float *dst = cur + (size_t)r * INK2_DIM;
    for (int d = 0; d < INK2_DIM; d++) {
      uint32_t bits = (uint32_t)src[d] << BF16_SHIFT;
      memcpy(&dst[d], &bits, sizeof(float));
    }
  }
}

// Runs the complete 48-visit Inkling-2-Mini forward pass over `rows` tokens.
// Computes logits for all `rows` rows when `logits_mode == 2`, for the last
// row only when `logits_mode == 1`, or skips unembed when `logits_mode == 0`.
// Returns 1 on completion, or 0 if shutdown interrupted a prefill pass.
static int run_forward(Ink2State *const *states, const int *tokens,
                       const int *pos, int rows, int batched, int is_prefill,
                       int logits_mode, const PassWait *wait) {
  int need_logits = logits_mode == 2 ? rows : (logits_mode == 1 ? 1 : 0);
  ensure_work(rows, need_logits);
  memcpy(work.ids.ptr, tokens, (size_t)rows * sizeof(int32_t));

  int pos0 = pos[0], max_pos = pos[0];
  if (batched) {
    for (int r = 0; r < rows; r++) {
      state_ensure_cap(states[r], pos[r] + 1);
      if (pos[r] > max_pos) max_pos = pos[r];
    }
    SconvTarget *st = (SconvTarget *)work.sconv_tgt.ptr;
    KvTarget *kt = (KvTarget *)work.kv_tgt.ptr;
    for (int v = 0; v < INK2_TOTAL_LAYERS; v++) {
      const Ink2Visit *vis = &schedule[v];
      for (int r = 0; r < rows; r++) {
        const Ink2State *sr = states[r];
        uint64_t conv_base = sr->conv_ring.address + sr->conv_ring.off +
                             (size_t)v * CONV_LAYER_BYTES;
        size_t base_idx =
            (size_t)v * SCONV_RINGS_PER_LAYER * INK2_MAX_BATCH_ROWS;
        st[base_idx + 0 * INK2_MAX_BATCH_ROWS + r] =
            (SconvTarget){conv_base + CONV_K_OFF, (uint32_t)pos[r], 0};
        st[base_idx + 1 * INK2_MAX_BATCH_ROWS + r] =
            (SconvTarget){conv_base + CONV_V_OFF, (uint32_t)pos[r], 0};
        st[base_idx + 2 * INK2_MAX_BATCH_ROWS + r] =
            (SconvTarget){conv_base + CONV_ATTN_OFF, (uint32_t)pos[r], 0};
        st[base_idx + 3 * INK2_MAX_BATCH_ROWS + r] =
            (SconvTarget){conv_base + CONV_MLP_OFF, (uint32_t)pos[r], 0};
        if (vis->is_global) {
          const Buf *kb = &sr->gk[vis->global_slot];
          const Buf *vb = &sr->gv[vis->global_slot];
          kt[(size_t)v * INK2_MAX_BATCH_ROWS + r] =
              (KvTarget){kb->address + kb->off, vb->address + vb->off,
                         (uint32_t)pos[r], 0u};
        } else {
          uint64_t sb = sr->slide_kv.address + sr->slide_kv.off;
          kt[(size_t)v * INK2_MAX_BATCH_ROWS + r] =
              (KvTarget){sb + slide_ring_off(vis->slide_slot, 0),
                         sb + slide_ring_off(vis->slide_slot, 1),
                         (uint32_t)pos[r], (uint32_t)INK2_SLIDE_RING};
        }
      }
    }
  } else {
    state_ensure_cap(states[0], pos0 + rows);
    max_pos = pos0 + rows - 1;
    inject_media_rows(tokens, pos0, rows);
  }

  if (is_prefill && rows >= PREFILL_BULK_ROWS) {
    expert_store_set_prompt_layer(experts_store, 0);
    prefetch_all_layer_experts(0);
  }

  size_t row_floats = (size_t)rows * INK2_DIM;
  size_t stride_bytes = (size_t)work.cap_rows * INK2_DIM * sizeof(float);
  uint8_t fast_visits[INK2_TOTAL_LAYERS] = {0};
  int stopped = 0;
  GpuCommandBuffer cmd = gpu_begin(gpu);

  // Embedding lookup + embed_norm (BF16-rounded).
  NormParam ep = {(uint32_t)INK2_DIM, 1u, RMS_EPS};
  gpu_select_kernel(&cmd, K_embed_norm);
  gpu_bind_buffer(&cmd, 0, &embed_w, 0);
  gpu_bind_buffer(&cmd, 1, &work.ids, 0);
  gpu_bind_buffer(&cmd, 2, &embed_norm, 0);
  gpu_bind_buffer(&cmd, 3, &work.cur, 0);
  gpu_copy_bytes(&cmd, 4, &ep, sizeof ep);
  gpu_run_kernel(&cmd, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);

  int nblocks = 0;
  // Prelude visits 0..5.
  for (int v = 0; v < INK2_CORE_START; v++) {
    if (is_prefill && shutdown_requested()) {
      stopped = 1;
      break;
    }
    if (is_prefill) prefill_layer_now = v + 1;
    if (v == INK2_MTP_TAP_0)
      inkling2_mtp_capture_tap(&cmd, &work.cur, &work.mtp, 0, rows);
    run_visit(&cmd, v, rows, pos0, max_pos, states[0], batched, is_prefill,
              &nblocks, wait, fast_visits);
  }

  int pre_loop_blocks = nblocks;
  if (!stopped) {
    // Save pre-loop hidden state; blk[0] (h_embed) stays untouched across
    // loops.
    run_copy(&cmd, &work.cur, 0, &work.pre_loop_cur, 0, (uint32_t)row_floats);
  }

  // Core loop passes 0, 1, 2 (visits 6..41).
  int visit = INK2_CORE_START;
  for (int pass = 0; pass < INK2_LOOP_PASSES && !stopped; pass++) {
    if (pass > 0) {
      // Previous pass's blocks past pre_loop_blocks + 1 is blk[2]; restart
      // hidden state from pre_loop_cur and blk[0..pre_loop_blocks-1].
      run_copy(&cmd, &work.blk, (size_t)(pre_loop_blocks + 1) * stride_bytes,
               &work.prev_blk, 0, (uint32_t)row_floats);
      run_copy(&cmd, &work.cur, 0, &work.prev_cur, 0, (uint32_t)row_floats);
      run_copy(&cmd, &work.pre_loop_cur, 0, &work.cur, 0, (uint32_t)row_floats);
      nblocks = pre_loop_blocks;
    }
    for (int li = 0; li < INK2_CORE_LAYERS; li++, visit++) {
      if (is_prefill && shutdown_requested()) {
        stopped = 1;
        break;
      }
      if (is_prefill) prefill_layer_now = visit + 1;
      if (visit == INK2_MTP_TAP_1)
        inkling2_mtp_capture_tap(&cmd, &work.cur, &work.mtp, 1, rows);
      if (visit == INK2_MTP_TAP_2)
        inkling2_mtp_capture_tap(&cmd, &work.cur, &work.mtp, 2, rows);
      run_visit(&cmd, visit, rows, pos0, max_pos, states[0], batched,
                is_prefill, &nblocks, wait, fast_visits);
    }
  }

  // Postlude visits 42..47.
  for (; visit < INK2_TOTAL_LAYERS && !stopped; visit++) {
    if (is_prefill && shutdown_requested()) {
      stopped = 1;
      break;
    }
    if (is_prefill) prefill_layer_now = visit + 1;
    run_visit(&cmd, visit, rows, pos0, max_pos, states[0], batched, is_prefill,
              &nblocks, wait, fast_visits);
  }

  int mtp_on = inkling2_mtp_active_drafts() > 0;
  if (!stopped && (logits_mode > 0 || mtp_on)) {
    if (mtp_on) {
      run_attn_res_rmsnorm(&cmd, &work.blk, 0, &work.cur, 0, &output_query, 0,
                           &final_norm, NULL, NULL, &work.q_in, NULL, NULL,
                           rows, work.cap_rows, nblocks, 1);
      inkling2_mtp_capture_tap(&cmd, &work.q_in, &work.mtp, 3, rows);
      Ink2MtpState *mtp_states[INK2_MAX_BATCH_ROWS];
      int mcount = batched ? rows : 1;
      for (int r = 0; r < mcount; r++) mtp_states[r] = &states[r]->mtp;
      inkling2_mtp_inject(&cmd, &work.mtp, mtp_states, pos, rows, batched,
                          is_prefill);
      if (logits_mode == 1) {
        size_t last_off = (size_t)(rows - 1) * INK2_DIM * sizeof(float);
        run_unembed_softcap(&cmd, &work.q_in, last_off, &work.logits, 1, 0);
      } else if (logits_mode == 2) {
        run_unembed_softcap(&cmd, &work.q_in, 0, &work.logits, rows,
                            is_prefill);
      }
    } else if (logits_mode == 1) {
      size_t last_off = (size_t)(rows - 1) * INK2_DIM * sizeof(float);
      run_attn_res_rmsnorm(&cmd, &work.blk, last_off, &work.cur, last_off,
                           &output_query, 0, &final_norm, NULL, NULL,
                           &work.q_in, NULL, NULL, 1, work.cap_rows, nblocks,
                           1);
      run_unembed_softcap(&cmd, &work.q_in, 0, &work.logits, 1, 0);
    } else {
      run_attn_res_rmsnorm(&cmd, &work.blk, 0, &work.cur, 0, &output_query, 0,
                           &final_norm, NULL, NULL, &work.q_in, NULL, NULL,
                           rows, work.cap_rows, nblocks, 1);
      run_unembed_softcap(&cmd, &work.q_in, 0, &work.logits, rows, is_prefill);
    }
  }

  if (is_prefill) expert_store_set_prompt_layer(experts_store, -1);
  model_commit(&cmd);
  model_sync();
  if (!is_prefill) flush_fast_visits(fast_visits, rows);
  held_release_all();
  return !stopped;
}

// Copies `rows` padded logits rows (stride LOGITS_STRIDE) into the tightly
// packed pass-local `logits_compact` buffer (stride INK2_VOCAB).
static const float *pack_logits(int rows) {
  if (rows == 1) return (const float *)work.logits.ptr;
  if (rows > logits_compact_rows) {
    free(logits_compact);
    logits_compact = xmalloc((size_t)rows * LOGITS_BYTES);
    logits_compact_rows = rows;
  }
  const float *src = (const float *)work.logits.ptr;
  for (int r = 0; r < rows; r++)
    memcpy(logits_compact + (size_t)r * INK2_VOCAB,
           src + (size_t)r * LOGITS_STRIDE, LOGITS_BYTES);
  return logits_compact;
}

// ---- ModelOps entry points ----

// Loads the Inkling-2-Mini dense backbone on model_gpu and starts its
// SSD-backed expert store, preloading all routed experts when memory fits.
static int ink2_load(Gpu *model_gpu, MemorySlot *work_mem, const char *dir,
                     const Options *o) {
  opt = *o;
  if (opt.ctx <= 0) opt.ctx = DEFAULT_CTX;
  if (!checkpoint) {
    gpu = model_gpu;
    work_slot = work_mem;
    memory_slot_init(&state_slot, work_mem->account, MEMORY_STATE, NULL,
                     "Inkling-2-Mini states", NULL, NULL);
    state_slot.automatic = 1;
    pass_before_yield = before_yield;
    check_config(dir);
    inkling2_build_schedule(schedule);

#define K(field, name) field = gpu_find_kernel(gpu, name)
    K(K_mv_bf16, "mv_bf16");
    K(K_mv_bf16_2, "mv_bf16_2");
    K(K_mv_bf16_3, "mv_bf16_3");
    K(K_mvm_bf16, "mvm_bf16");
    K(K_mvm_bf16_2, "mvm_bf16_2");
    K(K_mvm_bf16_3, "mvm_bf16_3");
    K(K_mm_bf16, "mm_bf16");
    K(K_embed_norm, "ink2_embed_norm");
    K(K_rmsnorm, "ink2_rmsnorm");
    K(K_rmsnorm_3, "ink2_rmsnorm_3");
    K(K_attn_res, "ink2_attn_res");
    K(K_attn_res_rmsnorm, "ink2_attn_res_rmsnorm");
    K(K_msg_cat_norm, "ink2_msg_cat_norm");
    K(K_msg_cat_norm_3, "ink2_msg_cat_norm_3");
    K(K_msg_down_3, "ink2_msg_down_3");
    K(K_msg_gate_apply, "ink2_msg_gate_apply");
    K(K_msg_gate_apply_3, "ink2_msg_gate_apply_3");
    K(K_sconv_static, "ink2_sconv_static");
    K(K_sconv_static_batched, "ink2_sconv_static_batched");
    K(K_sconv_static_post, "ink2_sconv_static_post");
    K(K_sconv_static_post_batched, "ink2_sconv_static_post_batched");
    K(K_sconv_dynamic, "ink2_sconv_dynamic");
    K(K_sconv_dynamic_batched, "ink2_sconv_dynamic_batched");
    K(K_sconv_ring_save, "ink2_sconv_ring_save");
    K(K_qkvr_prep, "ink2_qkvr_prep");
    K(K_qkvr_prep_fp8, "ink2_qkvr_prep_fp8");
    K(K_sconv_qkvr_m1, "ink2_sconv_qkvr_prep_m1");
    K(K_sconv_qkvr_fp8_m1, "ink2_sconv_qkvr_prep_fp8_m1");
    K(K_attend_part_rel, "ink2_attend_part_rel");
    K(K_attend_part_fp8_rel, "ink2_attend_part_fp8_rel");
    K(K_attend_combine, "ink2_attend_combine");
    K(K_attend_prefill_rel, "ink2_attend_prefill_rel");
    K(K_fp8_dequant_kv, "ink2_fp8_dequant_kv");
    K(K_attend_prefill_slab_rel, "ink2_attend_prefill_slab_rel");
    K(K_route_topk, "ink2_route_topk");
    K(K_route_topk_m1, "ink2_route_topk_m1");
    K(K_moe_group, "ink2_moe_group");
    K(K_moe_gather_bf16, "ink2_moe_gather_bf16");
    K(K_nv_up_m1, "ink2_moe_nvfp4_up_m1");
    K(K_nv_up_mv, "ink2_moe_nvfp4_up_mv");
    K(K_nv_dn_mv, "ink2_moe_nvfp4_down_mv");
    K(K_nv_dn_comb_m1, "ink2_moe_nvfp4_down_combine_m1");
    K(K_nv_up_mm, "ink2_moe_nvfp4_up_mm");
    K(K_nv_dn_mm, "ink2_moe_nvfp4_down_mm");
    K(K_mx_up_m1, "ink2_moe_mxfp8_up_m1");
    K(K_mx_up_mv, "ink2_moe_mxfp8_up_mv");
    K(K_mx_dn_mv, "ink2_moe_mxfp8_down_mv");
    K(K_mx_dn_comb_m1, "ink2_moe_mxfp8_down_combine_m1");
    K(K_mx_up_mm, "ink2_moe_mxfp8_up_mm");
    K(K_mx_dn_mm, "ink2_moe_mxfp8_down_mm");
    K(K_moe_combine, "ink2_moe_combine");
    K(K_copy, "ink2_copy");
    K(K_add, "ink2_add");
    K(K_unembed_softcap_m1, "ink2_unembed_softcap_m1");
    K(K_unembed_softcap, "ink2_unembed_softcap");
    K(K_logit_softcap, "ink2_logit_softcap");
#undef K

    checkpoint = tensor_index_open(dir);
    text_weights =
        weight_group_new(gpu, work_mem->account, "Inkling-2-Mini weights");
    register_text_weights();
    register_expert_spans();
    inkling2_mtp_init(gpu, text_weights, dir, opt.draft, opt.inkling2.fp8_kv);
    inkling2_vision_init(gpu, work_mem, checkpoint, text_weights);

    ExpertLayout layout = {.layers = INK2_PHYS_LAYERS,
                           .scored_layers = INK2_PHYS_LAYERS,
                           .experts = INK2_EXPERTS,
                           .chosen = INK2_TOPK,
                           .stage_experts = 0,
                           .parts = EXPERT_PARTS,
                           .span = &expert_spans[0][0][0],
                           .spans = expert_span_counts};
    experts_store = expert_store_new(gpu, &layout);
    expert_store_set_backbone(experts_store, weight_group_slot(text_weights));
  }

  MemorySlot *slot = weight_group_slot(text_weights);
  memory_pin(slot, 1);
  expert_store_set_request_running(experts_store, 1);
  if (work.cap_rows > INK2_MAX_VERIFY) {
    model_sync();
    free_work(&work);
  }
  expert_store_set_pass_rows(experts_store, 1);
  uint64_t fixed = work_slot->payload - work_resizable_bytes();
  uint64_t minimum =
      weight_group_required_bytes(text_weights) + fixed +
      estimate_work_bytes(INK2_MAX_VERIFY) +
      expert_store_minimum_bytes(experts_store, INK2_PHYS_LAYERS);
  if (!memory_wait_for_minimum(slot->account, minimum) ||
      !weight_group_load(text_weights)) {
    memory_pin(slot, 0);
    expert_store_set_request_running(experts_store, 0);
    return 0;
  }
  weight_group_set_resident(text_weights, 1);
  gpu_commit_residency(gpu);

  if (!store_started) {
    for (int l = 0; l < INK2_PHYS_LAYERS; l++) {
      layer_table[l] = work_alloc((size_t)INK2_EXPERTS * sizeof(ExpertRef));
      layer_table_gen[l] = UINT64_MAX;
    }
    expert_store_start(experts_store, 0, INK2_PHYS_LAYERS);
    store_started = 1;
  }
  expert_store_preload_if_fits(experts_store);
  if (shutdown_requested()) {
    memory_pin(slot, 0);
    expert_store_set_request_running(experts_store, 0);
    return 0;
  }

  uint64_t load_fixed = work_slot->payload - work_resizable_bytes();
  required_bytes = weight_group_required_bytes(text_weights) + load_fixed +
                   estimate_work_bytes(PREFILL_CHUNK) +
                   expert_store_pass_minimum_bytes(experts_store, PREFILL_CHUNK,
                                                   INK2_PHYS_LAYERS);
  prepare_prev_rows = 0;
  prepare_prev_payload = 0;
  return 1;
}

// Returns the memory slot of the loaded text weights.
static MemorySlot *ink2_dense_weight_slot(void) {
  return text_weights ? weight_group_slot(text_weights) : NULL;
}

// Marks whether a request is running so idle expert slots stay protected while
// active.
static void ink2_set_request_running(int running) {
  if (experts_store) expert_store_set_request_running(experts_store, running);
}

// Returns the bytes currently held by the routed expert store, or 0 before
// initialization.
static uint64_t ink2_cache_bytes(void) {
  return experts_store ? expert_store_bytes(experts_store) : 0;
}

// Writes the routed expert IDs and source codes chosen for `row` in the latest
// decoding pass across all 48 logical visits. Returns 1 on success or 0 when
// `row` has no complete routing record.
static int ink2_read_routes(int row, uint16_t *id, uint8_t *from) {
  if (row < 0 || row >= ROUTE_DECODE_CAP || !route_steps) return 0;
  for (int v = 0; v < INK2_TOTAL_LAYERS; v++)
    if (route_step[row][v] != route_steps) return 0;
  memcpy(id, route_id[row], sizeof route_id[row]);
  memcpy(from, route_from[row], sizeof route_from[row]);
  return 1;
}

// Writes runtime counters and expert-cache statistics into c.
static void ink2_read_counters(ModelCounters *c) {
  *c = (ModelCounters){
      .decode_rows = decode_rows_count,
      .prefill_rows = prefill_rows_count,
      .decode_seconds = decode_seconds_total,
      .decode_gpu_seconds = decode_gpu_seconds_total,
      .prefill_layer = prefill_layer_now,
      .prefill_rows_now = prefill_rows_now,
      .layers = INK2_TOTAL_LAYERS,
      .experts_per_row = INK2_TOTAL_LAYERS * INK2_TOPK,
      .routed_layers = INK2_TOTAL_LAYERS,
      .experts_per_layer = INK2_EXPERTS,
      .dense_bytes =
          (text_weights ? weight_group_resident_bytes(text_weights) : 0) +
          inkling2_vision_resident_bytes(),
      .required_bytes = required_bytes,
      .decode = decode_stats,
      .prefill = prefill_stats,
      .experts = INK2_PHYS_LAYERS * INK2_EXPERTS};
  if (!experts_store) return;
  c->cache = *expert_stats();
  c->cache_bytes = expert_store_bytes(experts_store);
  c->slot_bytes = expert_store_slot_bytes(experts_store);
  c->cache_slots = expert_store_slot_count(experts_store);
  c->mapped = expert_store_mapped_count(experts_store);
  c->cap = expert_store_slot_limit(experts_store);
  c->max_slots = expert_store_slot_maximum(experts_store);
  c->uncached_bytes =
      expert_store_bytes_without_slot(experts_store, &c->uncached);
}

// Returns the unpadded output vocabulary size.
static int ink2_vocab_size(void) { return INK2_VOCAB; }

// Returns the maximum distinct states decoded in one batched pass.
static int ink2_max_rows(void) { return INK2_MAX_BATCH_ROWS; }

// Returns the maximum rows verified in one speculative verification pass.
static int ink2_max_verify(void) { return INK2_MAX_VERIFY; }

// Returns the active MTP draft count per pass, or 0 when MTP is disabled.
static int ink2_drafts_per_pass(void) { return inkling2_mtp_active_drafts(); }

// Returns how many positions a state can rewind safely without overwriting its
// sliding KV or short-convolution rings.
static int ink2_rewind_margin(void) { return INK2_REWIND_MARGIN; }

// Returns whether `token` ends a model reply.
static int ink2_is_stop_token(int token) { return token == eos_token; }

// Indicates that Inkling-2-Mini resumes only from a saved snapshot, not replay.
static int ink2_replay_rows(int k) {
  (void)k;
  return -1;
}

// Prefills positions `from` through `n - 1` of state `ms` in memory-admitted
// chunks of up to PREFILL_CHUNK tokens, capturing a snapshot at `capture` if
// requested. Returns null if memory cannot fit or shutdown interrupts prefill.
static const float *ink2_prefill(ModelState *ms, const int *tokens, int n,
                                 int from, int replay, int capture,
                                 ModelSnapshot **captured, float *every,
                                 const PassWait *wait) {
  if (replay && from > 0)
    die("Inkling-2-Mini does not support replay without a snapshot");
  if (from < 0 || n <= from || n > opt.ctx)
    die("invalid Inkling-2-Mini prefill range %d..%d (ctx %d)", from, n,
        opt.ctx);
  if (capture >= 0 && (capture <= from || capture >= n || !captured))
    die("invalid Inkling-2-Mini capture %d in %d..%d", capture, from, n);
  if (captured) *captured = NULL;

  Ink2State *s = (Ink2State *)ms;
  prefill_rows_now = n - from;
  ExpertStoreCounts c0 = *expert_stats();
  int stopped = 0;
  for (int pos = from; pos < n;) {
    if (shutdown_requested()) {
      stopped = 1;
      break;
    }
    int rem = n - pos;
    if (capture > pos && pos + rem > capture) rem = capture - pos;
    int chunk = prepare_rows(rem, 1);
    if (!chunk) {
      stopped = 1;
      break;
    }
    int is_last = (pos + chunk == n);
    int logits_mode = every ? 2 : (is_last ? 1 : 0);
    if (!run_forward(&s, tokens + pos, &pos, chunk, 0, 1, logits_mode, wait)) {
      stopped = 1;
      break;
    }
    if (every) {
      const float *src = (const float *)work.logits.ptr;
      for (int r = 0; r < chunk; r++)
        memcpy(every + (size_t)(pos - from + r) * INK2_VOCAB,
               src + (size_t)r * LOGITS_STRIDE, LOGITS_BYTES);
    }
    pos += chunk;
    prefill_rows_count += (uint64_t)chunk;
    if (capture >= 0 && pos == capture && captured)
      *captured = ink2_snapshot_take(ms, capture, NULL);
  }
  expert_store_age_and_resize(experts_store);
  stats_add(&prefill_stats, &c0);
  prefill_layer_now = 0;
  prefill_rows_now = 0;
  if (stopped) {
    if (captured && *captured) {
      ink2_snapshot_free(*captured);
      *captured = NULL;
    }
    return NULL;
  }
  return every ? every + (size_t)(n - from - 1) * INK2_VOCAB
               : (const float *)work.logits.ptr;
}

// Decodes one token for each of `m` distinct states. Returns null if the
// pass's minimum memory does not fit.
static const float *ink2_decode_rows(ModelState *const *states,
                                     const int *tokens, const int *pos, int m) {
  if (m < 1 || m > INK2_MAX_BATCH_ROWS)
    die("invalid Inkling-2-Mini decode batch size %d", m);
  if (!prepare_rows(m, 0)) return NULL;
  ExpertStoreCounts c0 = *expert_stats();
  double t0 = now();
  GpuTimes g0 = gpu_times(gpu);
  route_steps++;
  Ink2State *s_arr[INK2_MAX_BATCH_ROWS];
  for (int i = 0; i < m; i++) s_arr[i] = (Ink2State *)states[i];
  int batched = m > 1 ? 1 : 0;
  run_forward(s_arr, tokens, pos, m, batched, 0, m == 1 ? 1 : 2, NULL);
  expert_store_age_and_resize(experts_store);
  stats_add(&decode_stats, &c0);
  GpuTimes g1 = gpu_times(gpu);
  decode_rows_count += (uint64_t)m;
  decode_seconds_total += now() - t0;
  decode_gpu_seconds_total += g1.busy_seconds - g0.busy_seconds;
  return pack_logits(m);
}

// Proposes `n` draft token IDs after `token` at `pos` via the MTP DSpark head.
static const double *ink2_propose_drafts(ModelState *ms, int token, int pos,
                                         int *drafts, int n, Sampler *sampler) {
  (void)sampler;
  const Ink2State *s = (const Ink2State *)ms;
  inkling2_mtp_propose(gpu, work_slot, &work.mtp, &s->mtp, &embed_w,
                       &embed_norm, &unembed_w, token, pos, drafts, n,
                       &last_committed, model_sync);
  return NULL;
}

// Verifies `m` contiguous candidate tokens starting at `pos` on state `ms`.
// Returns null if the pass's minimum memory does not fit.
static const float *ink2_verify(ModelState *ms, const int *tokens, int m,
                                int pos) {
  if (m < 1 || m > INK2_MAX_VERIFY)
    die("invalid Inkling-2-Mini verify length %d", m);
  if (!prepare_rows(m, 0)) return NULL;
  ExpertStoreCounts c0 = *expert_stats();
  double t0 = now();
  GpuTimes g0 = gpu_times(gpu);
  route_steps++;
  Ink2State *s = (Ink2State *)ms;
  run_forward(&s, tokens, &pos, m, 0, 0, 2, NULL);
  expert_store_age_and_resize(experts_store);
  stats_add(&decode_stats, &c0);
  GpuTimes g1 = gpu_times(gpu);
  decode_rows_count += (uint64_t)m;
  decode_seconds_total += now() - t0;
  decode_gpu_seconds_total += g1.busy_seconds - g0.busy_seconds;
  return pack_logits(m);
}

// Accepts the first `n` verified tokens; rejected ring slots are overwritten on
// subsequent writes and need no rollback.
static void ink2_accept(ModelState *ms, const int *tokens, int n, int pos) {
  (void)ms;
  (void)tokens;
  (void)n;
  (void)pos;
}

// Resizes the routed expert cache while no request runs and returns the
// remaining slot count.
static int ink2_run_idle_work(void) {
  if (!experts_store) return 0;
  expert_store_resize_when_idle(experts_store);
  return expert_store_slot_count(experts_store);
}

// Informs the expert store how many other passes currently hold GPU work.
static void ink2_set_other_passes(int passes) {
  if (experts_store) expert_store_set_other_passes(experts_store, passes);
}

// Formats one CLI user turn matching chat_template.jinja.
static void ink2_render_chat_turn(char *text, size_t cap, const char *prompt,
                                  int media_count, int think) {
  char effort[16];
  if (think <= 0) {
    snprintf(effort, sizeof effort, "0");
  } else {
    int clamped = think > MAX_THINK_PERCENT ? MAX_THINK_PERCENT : think;
    snprintf(effort, sizeof effort, "%.2g",
             (double)clamped / THINK_PERCENT_SCALE);
  }
  int off =
      snprintf(text, cap,
               "<|message_system|><|content_text|>Thinking effort level: %s"
               "<|end_message|>",
               effort);
  if (off < 0 || (size_t)off >= cap) return;
  for (int i = 0; i < media_count && i < MAX_CHAT_IMAGES; i++) {
    int n = snprintf(
        text + off, cap - (size_t)off,
        "<|message_user|><|content_image|><|unused_200054|><|end_message|>");
    if (n < 0 || (size_t)n >= cap - (size_t)off) return;
    off += n;
  }
  snprintf(text + off, cap - (size_t)off,
           "<|message_user|><|content_text|>%s<|end_message|>", prompt);
}

// Records the encoded media items whose spans appear in subsequent prefills.
static void ink2_set_prefill_media(const Media *im, int n) {
  prefill_media = im;
  prefill_media_count = n;
}

// Creates a cooperative pass with its own workspace, submission sequence,
// packed logits buffer, held expert state and expert-store pass variables.
static Pass *ink2_pass_new(void (*body)(void *), void *argument) {
  const PassVariable own[] = {
      {&work, sizeof work},
      {&last_committed, sizeof last_committed},
      {&logits_compact, sizeof logits_compact},
      {&logits_compact_rows, sizeof logits_compact_rows},
      {&prefill_media, sizeof prefill_media},
      {&prefill_media_count, sizeof prefill_media_count},
      {&held, sizeof held},
      {&held_layers, sizeof held_layers}};
  constexpr int OWN = sizeof own / sizeof *own;
  PassVariable variables[OWN + EXPERT_STORE_PASS_VARIABLES];
  memcpy(variables, own, sizeof own);
  int count = OWN;
  if (experts_store) {
    expert_store_write_pass_variables(experts_store, variables + OWN);
    count += EXPERT_STORE_PASS_VARIABLES;
  }
  Pass *pass = pass_new(body, argument, PASS_STACK_BYTES);
  pass_set_variables(pass, variables, count);
  return pass;
}

// Initializes pass-local scratch handles and held-expert state when a pass
// begins.
static void ink2_pass_begin(void) {
  work = (Ink2Work){0};
  last_committed = 0;
  logits_compact = NULL;
  logits_compact_rows = 0;
  prefill_media = NULL;
  prefill_media_count = 0;
  held.n = 0;
  held_layers = 0;
  if (experts_store) expert_store_set_prompt_layer(experts_store, -1);
}

// Syncs, releases held experts and frees pass-local scratch buffers when a
// pass ends.
static void ink2_pass_end(void) {
  model_sync();
  held_release_all();
  free_work(&work);
  free(logits_compact);
  logits_compact = NULL;
  logits_compact_rows = 0;
}

// config.json model_type values served by this family.
static const char *const ink2_model_types[] = {"inkling2_loop",
                                               "inkling2_loop_text", NULL};

const ModelOps inkling2_ops = {.family = "Inkling-2-Mini",
                               .model_types = ink2_model_types,
                               .media_token = "<|unused_200054|>",
                               .max_drafts = INK2_MTP_MAX_DRAFTS,
                               .load = ink2_load,
                               .dense_weight_slot = ink2_dense_weight_slot,
                               .set_request_running = ink2_set_request_running,
                               .read_counters = ink2_read_counters,
                               .expert_store_bytes = ink2_cache_bytes,
                               .read_routes = ink2_read_routes,
                               .vocab_size = ink2_vocab_size,
                               .max_rows = ink2_max_rows,
                               .max_verify = ink2_max_verify,
                               .drafts_per_pass = ink2_drafts_per_pass,
                               .rewind_margin = ink2_rewind_margin,
                               .is_stop_token = ink2_is_stop_token,
                               .state_new = ink2_state_new,
                               .state_free = ink2_state_free,
                               .state_bytes = ink2_state_bytes,
                               .state_copy = ink2_state_copy,
                               .snapshot_take = ink2_snapshot_take,
                               .snapshot_restore = ink2_snapshot_restore,
                               .snapshot_free = ink2_snapshot_free,
                               .snapshot_bytes = ink2_snapshot_bytes,
                               .snapshot_logits = ink2_snapshot_logits,
                               .state_version = 1,
                               .write_state_regions = ink2_write_state_regions,
                               .write_snapshot_parts =
                                   ink2_write_snapshot_parts,
                               .snapshot_new_empty = ink2_snapshot_new_empty,
                               .prefill = ink2_prefill,
                               .replay_rows = ink2_replay_rows,
                               .decode_rows = ink2_decode_rows,
                               .propose_drafts = ink2_propose_drafts,
                               .verify = ink2_verify,
                               .accept = ink2_accept,
                               .run_idle_work = ink2_run_idle_work,
                               .prepare_media = inkling2_prepare_media,
                               .encode_media = inkling2_encode_media,
                               .set_prefill_media = ink2_set_prefill_media,
                               .render_chat_turn = ink2_render_chat_turn,
                               .pass_new = ink2_pass_new,
                               .pass_begin = ink2_pass_begin,
                               .pass_end = ink2_pass_end,
                               .set_other_passes = ink2_set_other_passes};
