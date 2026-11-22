#include "model/nemotron/model.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "base/shutdown.h"
#include "engine/expert_store.h"
#include "engine/media.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/model.h"
#include "model/nemotron/audio.h"
#include "platform/video.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// NVIDIA Nemotron-3.5-Transcribe / Nemotron-H 52-layer hybrid language model
// (model_type nemo_speechlm and nemotron_h): weights, SSD-backed expert store,
// Mamba-2 and NoPE GQA state, snapshots, audio perception and forward passes.

// Numerical parameters from llm_backbone/config.json and
// modeling_nemotron_h.py.
static constexpr float RMS_EPS = 1e-5f;
static constexpr float ROUTE_SCALE = 2.5f;
static constexpr float FP8_UNDO = 256.0f;

// Prefill chunking, KV allocation growth, FlashDecoding split bounds and
// aligned direct-read thresholds.
static constexpr int PREFILL_CHUNK = 512;
static constexpr int START_CAP = 256;
static constexpr int KV_CAP_ALIGN = 64;
static constexpr int MAX_DECODE_PARTS = 64;
static constexpr int DECODE_PART_TARGET = 128;
static constexpr int DECODE_PART_FLOATS = NEMO_HEAD_DIM + 2;
static constexpr int PREFILL_Q_PAD = 64;
static constexpr int MM_TILE_M = 64;
static constexpr int MM_TILE_M2 = 128;
static constexpr int MM_TILE_N = 64;
static constexpr int MOE_MM_TILE = 64;
static constexpr int MOE_MV_MAX_M = 8;
static constexpr int MOE_MV_ROWS_PER_TG = 32;
static constexpr int FP8_MV_ROWS_PER_TG = 32;
static constexpr int NV_MV_ROWS_PER_TG = 16;
static constexpr int DEFAULT_CTX = 16384;
static constexpr int MAX_EOS_TOKENS = 8;
static constexpr int DEFAULT_SPEECH_EOS = 11;
static constexpr int DEFAULT_TEXT_EOS = 2;
static constexpr size_t ALIGN_READ_MIN_BYTES = 1u << 20;
static constexpr size_t ALIGN_16_MASK = 15u;
static constexpr size_t ALIGN_8_MASK = 7u;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_MASK = 0xFFFF0000u;
static constexpr size_t PASS_STACK_BYTES = 4u << 20;
static constexpr size_t PATH_CAP = 1024;
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ull;
static constexpr uint64_t FNV_PRIME = 1099511628211ull;

// Metal threadgroup sizes matching src/kernels/nemotron.metal and
// src/kernels/kernels.metal.
static constexpr uint32_t ROW_THREADS = 256;
static constexpr uint32_t FP8_MV_THREADS = 256;
static constexpr uint32_t NV_MV_THREADS = 128;
static constexpr uint32_t MM_THREADS = 128;
static constexpr uint32_t CONV_THREADS = 256;
static constexpr uint32_t SSM_THREADS = 128;
static constexpr uint32_t SSM_P_PER_TG = 4;
static constexpr uint32_t QKV_SLOTS = NEMO_HEADS + 2 * NEMO_KV_HEADS;
static constexpr uint32_t QKV_THREADS = NEMO_HEAD_DIM;
static constexpr uint32_t PREFILL_THREADS = 128;
static constexpr uint32_t DECODE_PART_THREADS = 512;
static constexpr uint32_t DECODE_COMBINE_THREADS = 32;
static constexpr uint32_t ROUTE_THREADS = 128;
static constexpr uint32_t MOE_MV_THREADS = 256;
static constexpr uint32_t MOE_COMBINE_THREADS = 64;
static constexpr uint32_t PACK_THREADS = 64;
static constexpr uint32_t VEC4_WIDTH = 4;
static constexpr uint32_t BF16_MV_THREADS = 256;
static constexpr uint32_t BF16_MV_ROWS_PER_SG = 4;
static constexpr uint32_t BF16_MV_SGS = 8;
static constexpr uint32_t BF16_MV_SPLIT_SMALL_N = 256;
static constexpr uint32_t BF16_MV_SPLIT_MIN_K = 2048;
static constexpr uint32_t BF16_MV_SPLIT_FACTOR = 4;
static constexpr uint32_t ACT_NONE = 0;
static constexpr uint32_t ACT_RELU2 = 1;

// Expert store layout, packing ratio, prefetch threshold and route inspection
// source codes matching src/eremite.h.
static constexpr int EXPERT_PARTS = 4;
static constexpr int NVFP4_VALUES_PER_BYTE = 2;
static constexpr int NEAR_MISSES = NEMO_TOPK;
static constexpr int PREFILL_BULK_ROWS = 64;
static constexpr int ROUTE_DECODE_CAP = NEMO_MAX_BATCH_ROWS;
static constexpr uint8_t ROUTE_SOURCE_MEMORY = 0;
static constexpr uint8_t ROUTE_SOURCE_AHEAD = 1;
static constexpr uint8_t ROUTE_SOURCE_LATE = 2;
static constexpr uint8_t ROUTE_SOURCE_SSD = 4;

// Byte layout of per-sequence KV and Mamba-2 rolling state buffers.
static constexpr size_t KV_HEAD_ROW_BYTES =
    (size_t)NEMO_HEAD_DIM * sizeof(uint16_t);
static constexpr size_t CONV_LAYER_BYTES =
    (size_t)NEMO_MAMBA_CONV_HIST * NEMO_MAMBA_CONV_DIM * sizeof(float);
static constexpr size_t CONV_TOTAL_BYTES =
    (size_t)NEMO_MAMBA_LAYERS * CONV_LAYER_BYTES;
static constexpr size_t SSM_LAYER_BYTES = (size_t)NEMO_MAMBA_HEADS *
                                          NEMO_MAMBA_HEAD_DIM *
                                          NEMO_MAMBA_STATE * sizeof(float);
static constexpr size_t SSM_TOTAL_BYTES =
    (size_t)NEMO_MAMBA_LAYERS * SSM_LAYER_BYTES;
static constexpr size_t SNAP_ROLLING_BYTES = CONV_TOTAL_BYTES + SSM_TOTAL_BYTES;
static constexpr size_t LOGITS_BYTES = (size_t)NEMO_VOCAB * sizeof(float);

// Layer block type in the 52-layer Nemotron-H hybrid schedule.
typedef enum {
  // Mamba-2 selective state-space layer.
  LAYER_MAMBA = 0,
  // NoPE grouped-query attention layer.
  LAYER_ATTN = 1,
  // Mixture-of-experts layer with shared and routed experts.
  LAYER_MOE = 2
} LayerKind;

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

// Kernel parameter structs matching src/kernels/nemotron.metal.
typedef struct {
  // Row width in floats and BF16 rounding flag.
  uint32_t dim, round;
  // RMSNorm epsilon.
  float eps;
} NormParam;

typedef struct {
  // Active rows, output width, inner width, residual flag and BF16 round flag.
  uint32_t m, n, k, has_res, round;
  // Per-tensor weight scale multiplied by 256.
  float scale;
} Fp8LinParam;

typedef struct {
  // Active rows, output width, inner width, activation, residual and round.
  uint32_t m, n, k, act, has_res, round;
  // Per-tensor NVFP4 weight_scale_2 scalar.
  float scale2;
} Nvfp4LinParam;

typedef struct {
  // Active token count.
  uint32_t rows;
  // RMSNorm epsilon for gated group norm.
  float eps;
} MambaParam;

typedef struct {
  // Device GPU pointers to one state's Mamba conv_state and ssm_state.
  uint64_t conv, ssm;
} MambaTarget;

typedef struct {
  // Device GPU pointers to one state's per-KV-head BF16 K and V buffers.
  uint64_t k0, k1, v0, v1;
  // Token position of this state and alignment padding.
  uint32_t pos, pad;
} KvTarget;

typedef struct {
  // Active rows, head-major Q stride, row-0 pos and batched-decode flag.
  uint32_t rows, q_stride, pos0, batched;
} QkvParam;

typedef struct {
  // Active query rows, head-major Q stride, row-0 pos and KV buffer cap.
  uint32_t rows, q_stride, pos0, kv_cap;
} AttnPrefillParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, partition count, stride and batch flag.
  uint32_t rows, q_stride, pos0, nparts, part_stride, batched;
} AttnDecodeParam;

typedef struct {
  // Active rows, total experts and top-k count.
  uint32_t rows, experts, topk;
  // Routed weight scaling multiplier.
  float scale;
} RouteParam;

typedef struct {
  // Matrix output rows, inner columns, active pair count and tile count.
  uint32_t n, k, npairs, ntiles;
} MoeParam;

typedef struct {
  // Active rows, hidden width and top-k count.
  uint32_t rows, dim, topk;
} MoeCombineParam;

// Per-expert GPU device addresses matching NemoExpertRef in nemotron.metal.
typedef struct {
  // Packed up_proj weights, up_proj scales, down_proj weights and scales.
  uint64_t w_up, s_up, w_down, s_down;
} ExpertRef;

// One per-tensor FP8 linear projection [n, k].
typedef struct {
  // Raw F8_E4M3 weight buffer in GPU memory.
  Buf w;
  // Output rows and input columns.
  int n, k;
  // Per-tensor scale multiplied by 256.
  float scale;
} Fp8Lin;

// One dense NVFP4 linear projection [n, k].
typedef struct {
  // Packed U8 weight buffer [n, k / 2] and E4M3 block scale buffer [n, k / 16].
  Buf w, sc;
  // Output rows and input columns.
  int n, k;
  // Per-tensor weight_scale_2 scalar.
  float scale2;
} Nvfp4Lin;

// Weights for one Mamba-2 SSM layer.
typedef struct {
  // Input projection [10304, 2688] and output projection [2688, 4096].
  Fp8Lin in_proj, out_proj;
  // Causal conv1d filter [6144, 4] and bias [6144] in FP32.
  Buf conv_w, conv_b;
  // Exponentiated A_log [64], skip D [64], dt_bias [64] and norm [4096] FP32.
  Buf a_exp, d_skip, dt_bias, norm_w;
} MambaLayer;

// Weights for one NoPE GQA 32/2 attention layer.
typedef struct {
  // NVFP4 query, key, value and output projections.
  Nvfp4Lin q, k, v, o;
} AttnLayer;

// Weights for one MoE layer's router, shared expert and routed scale factors.
typedef struct {
  // Router weights BF16 [128, 2688] and FP32 correction bias [128].
  Buf router_w, router_bias;
  // Always-resident NVFP4 shared expert up [3712, 2688] and down [2688, 3712].
  Nvfp4Lin shared_up, shared_down;
  // Per-expert FP32 weight_scale_2 arrays [128] for up_proj and down_proj.
  Buf factor_up, factor_down;
} MoeLayer;

// One layer of the 52-layer hybrid backbone.
typedef struct {
  // Block type (Mamba, Attention or MoE).
  LayerKind kind;
  // Index into the kind-specific array (0..22, 0..5, or 0..22).
  int sub_idx;
  // Pre-mixer RMSNorm weights [2688] in FP32.
  Buf norm_w;
} LayerDesc;

// Context passed to weight_group_add_generated to read 128 expert scale_2
// scalars into one FP32 GPU buffer.
typedef struct {
  // Safetensors descriptors for the 128 scalar weight_scale_2 tensors.
  Tensor t[NEMO_EXPERTS];
} Scale2FillParam;

// Routed experts of one MoE layer held by the active pass until GPU finishes.
typedef struct {
  // MoE layer index (0..22) and count of held experts.
  int layer, n;
  // Held expert indices within layer.
  int experts[NEMO_EXPERTS];
} HeldExperts;

// Per-sequence hybrid KV and Mamba-2 recurrent state.
typedef struct {
  // Allocated KV capacity in tokens (multiple of KV_CAP_ALIGN).
  int cap;
  // Exact sequence position matching the live Mamba conv_state and ssm_state.
  int pos;
  // Growing BF16 K and V buffers for the 6 attention layers and 2 KV heads.
  Buf k[NEMO_ATTN_LAYERS][NEMO_KV_HEADS];
  Buf v[NEMO_ATTN_LAYERS][NEMO_KV_HEADS];
  // Rolling FP32 Mamba-2 conv1d history [23, 3, 6144].
  Buf conv_state;
  // Rolling FP32 Mamba-2 SSM state [23, 64, 64, 128].
  Buf ssm_state;
} NemoState;

// Snapshot of rolling Mamba-2 state at one position, plus optional logits.
typedef struct {
  // Sequence position at capture time.
  int pos;
  // Host copy of conv_state followed by ssm_state.
  uint8_t *rows;
  // Optional host copy of vocabulary logits [131072].
  float *logits;
} NemoSnapshot;

// Pass-local scratch GPU buffers resized for prefill chunks or decode batches.
typedef struct {
  // Current allocated token capacity, head-major Q stride and logits rows.
  int cap_rows, q_stride, logits_rows;
  // Token IDs, residual stream, normed input and pre-packed BF16 activations.
  Buf ids, cur, normed, x_pack;
  // Mamba-2 projections [rows, 10304], conv output [rows, 6144], SSM output
  // [rows, 4096] and gated-norm output [rows, 4096].
  Buf mamba_zxbcdt, mamba_xbc, mamba_y, mamba_normed;
  // Attention Q [rows, 4096], K [rows, 256], V [rows, 256], head-major BF16 Q,
  // FlashDecoding partials and attention output [rows, 4096].
  Buf q_raw, k_raw, v_raw, q_hm, attn_part, attn_out;
  // MoE shared-expert intermediate [rows, 3712] and output [rows, 2688].
  Buf shared_h, shared_out;
  // MoE routing logits, IDs, weights, tile metadata and streaming table.
  Buf route_logits, route_ids, route_w, ptok, pw, tp, tiles, expert_table;
  // MoE gathered BF16 inputs, BF16 relu2 intermediates and FP32 pair outputs.
  Buf x_bf16, h_bf16, y_pair;
  // Per-state target arrays for batched multi-state Mamba and KV decode.
  Buf mamba_tgt, kv_tgt;
  // Pre-verify Mamba-2 state backups and per-layer FP32 zxbcdt inputs for
  // exact partial-acceptance rewind.
  Buf verify_conv, verify_ssm, verify_zxbcdt;
  // Allocated verify_zxbcdt row capacity, active verify row count and start
  // position.
  int verify_cap, verify_count, verify_pos;
  // State currently awaiting accept() after a verify() pass.
  NemoState *verify_state;
  // Output logits [logits_rows, 131072].
  Buf logits;
} NemoWork;

// ---- Model-global state (one loaded Nemotron model per process) ----

static Gpu *gpu;
static MemorySlot *work_slot;
static MemorySlot state_slot;
static int state_slot_ready;
static char model_dir[PATH_CAP];
static Options opt;
static int has_audio;
static int eos_tokens[MAX_EOS_TOKENS];
static int n_eos_tokens;

static TensorIndex *checkpoint;
static WeightGroup *text_weights;
static ExpertStore *experts_store;
static int store_started;
static NemoAudioEncoder *audio_enc;

static Buf embed_w, final_norm_w, lm_head_w;
static LayerDesc layer_desc[NEMO_LAYERS];
static MambaLayer mamba_layers[NEMO_MAMBA_LAYERS];
static AttnLayer attn_layers[NEMO_ATTN_LAYERS];
static MoeLayer moe_layers[NEMO_MOE_LAYERS];
static FileSpan expert_spans[NEMO_MOE_LAYERS][NEMO_EXPERTS][EXPERT_PARTS];
static int expert_span_counts[NEMO_MOE_LAYERS];
static Buf layer_table[NEMO_MOE_LAYERS];
static uint64_t layer_table_gen[NEMO_MOE_LAYERS];

// Compiled Metal kernel handles.
static int K_mv_bf16, K_mvm_bf16, K_mm_bf16;
static int K_embed, K_rmsnorm, K_pack_bf16;
static int K_fp8_mv, K_fp8_mm;
static int K_nvfp4_mv, K_nvfp4_mm;
static int K_mamba_conv_prefill, K_mamba_conv_save, K_mamba_conv_decode;
static int K_mamba_ssm_prefill, K_mamba_ssm_decode, K_mamba_norm_gated;
static int K_qkv_store, K_attend_prefill, K_attend_part, K_attend_combine;
static int K_route_topk, K_moe_group, K_moe_gather_bf16;
static int K_moe_up_mv, K_moe_dn_mv, K_moe_up_mm, K_moe_dn_mm,
    K_moe_combine_add;

// Runtime counters and latest decode route history.
static uint64_t decode_rows_count, prefill_rows_count;
static double decode_seconds_total, decode_gpu_seconds_total;
static int prefill_layer_now, prefill_rows_now;
static uint64_t required_bytes;
static int prepare_prev_rows;
static uint64_t prepare_prev_payload;
static ExpertStoreCounts decode_stats, prefill_stats;
static uint64_t route_steps;
static uint64_t route_step[ROUTE_DECODE_CAP][NEMO_MOE_LAYERS];
static uint16_t route_id[ROUTE_DECODE_CAP][NEMO_MOE_LAYERS][NEMO_TOPK];
static uint8_t route_from[ROUTE_DECODE_CAP][NEMO_MOE_LAYERS][NEMO_TOPK];

// Pass-local variables swapped per cooperative Pass.
static NemoWork work;
static uint64_t last_committed;
static const Media *prefill_media;
static int prefill_media_count;
static HeldExperts held;
static uint32_t held_layers;

// ---- Helpers ----

// Rounds n up to a multiple of align.
static inline int align_up(int n, int align) {
  return ((n + align - 1) / align) * align;
}

// Converts one IEEE FP32 value to BF16 bits (nearest-even).
static inline uint16_t f32_to_bf16(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  return (uint16_t)(u >> BF16_SHIFT);
}

// Reads exact byte count of tensor t from disk into dst.
static void read_tensor_exact(Tensor t, void *dst) {
  uint8_t *p = dst;
  uint64_t off = t.off, left = t.size;
  while (left > 0) {
    ssize_t n = pread(t.fd, p, (size_t)left, (off_t)off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("nemotron: short read on safetensors weight");
    p += n;
    off += (uint64_t)n;
    left -= (uint64_t)n;
  }
}

// Reads one scalar F32 tensor value from disk.
static float read_scalar_f32(Tensor t) {
  if (t.dtype != 'f' || t.size != sizeof(float))
    die("nemotron: expected scalar F32 scale tensor");
  float v = 0.0f;
  read_tensor_exact(t, &v);
  return v;
}

// Allocates a resident GPU workspace buffer counted in work_slot.
static Buf work_alloc(size_t bytes) {
  Buf b = gpu_alloc(gpu, work_slot, bytes);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Tries to allocate a resident GPU state buffer counted in state_slot.
static Buf state_try_alloc(size_t bytes) {
  Buf b = gpu_try_alloc_required(gpu, &state_slot, bytes);
  if (!b.obj) return b;
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Commits a command buffer and records its completion sequence.
static void model_commit(GpuCommandBuffer *cmd) {
  last_committed = cmd->number;
  gpu_commit(cmd);
}

// Returns whether the command buffer with the given sequence number finished.
static int committed_finished(void *number) {
  return gpu_has_finished(gpu, (uint64_t)(uintptr_t)number);
}

// Blocks until the command buffer with the given sequence number finishes.
static void committed_wait(void *number) {
  gpu_wait_until_finished(gpu, (uint64_t)(uintptr_t)number);
}

// Waits for the pass's GPU work, yielding when inside a cooperative pass.
static void model_sync(void) {
  if (!pass_current()) {
    gpu_sync(gpu);
    last_committed = 0;
    return;
  }
  if (last_committed) {
    pass_wait(&(PassWait){committed_finished, committed_wait,
                          (void *)(uintptr_t)last_committed, PASS_WAIT_GPU,
                          NULL});
    last_committed = 0;
  }
  gpu_check_finished(gpu);
}

// Returns the expert store's live counter structure.
static ExpertStoreCounts *expert_stats(void) {
  return expert_store_counts(experts_store);
}

// Adds the delta since c0 in the expert store's counts into dst.
static void stats_add(ExpertStoreCounts *dst, const ExpertStoreCounts *c0) {
  const ExpertStoreCounts *now_c = expert_stats();
  dst->hits += now_c->hits - c0->hits;
  dst->misses += now_c->misses - c0->misses;
  dst->prefetch_hits += now_c->prefetch_hits - c0->prefetch_hits;
  dst->prefetches += now_c->prefetches - c0->prefetches;
  dst->bytes += now_c->bytes - c0->bytes;
  dst->evictions += now_c->evictions - c0->evictions;
  dst->late += now_c->late - c0->late;
  dst->cancelled += now_c->cancelled - c0->cancelled;
  dst->substituted += now_c->substituted - c0->substituted;
  dst->remaps += now_c->remaps - c0->remaps;
  dst->given_back += now_c->given_back - c0->given_back;
  dst->decode_uses += now_c->decode_uses - c0->decode_uses;
  dst->row_own += now_c->row_own - c0->row_own;
  dst->row_ahead += now_c->row_ahead - c0->row_ahead;
  dst->row_late += now_c->row_late - c0->row_late;
  dst->row_sub += now_c->row_sub - c0->row_sub;
  dst->row_ssd += now_c->row_ssd - c0->row_ssd;
  dst->row_unique += now_c->row_unique - c0->row_unique;
  dst->read_s += now_c->read_s - c0->read_s;
}

// ---- Tile grouping ----

int nemotron_group_expert_tiles(const uint32_t *route_ids, const float *route_w,
                                int rows, int max_tile, int *ptok, float *pw,
                                int *tp, NemoMoeTile *tiles) {
  int counts[NEMO_EXPERTS] = {0};
  int offsets[NEMO_EXPERTS] = {0};
  int npairs = rows * NEMO_TOPK;
  for (int i = 0; i < npairs; i++) {
    uint32_t e = route_ids[i];
    if (e >= (uint32_t)NEMO_EXPERTS) die("nemotron: invalid expert ID %u", e);
    counts[e]++;
  }
  int off = 0, ntiles = 0;
  for (int e = 0; e < NEMO_EXPERTS; e++) {
    offsets[e] = off;
    int cnt = counts[e];
    int pos = off;
    while (cnt > 0) {
      int take = cnt > max_tile ? max_tile : cnt;
      tiles[ntiles++] =
          (NemoMoeTile){(uint32_t)e, (uint32_t)pos, (uint32_t)take, 0u};
      pos += take;
      cnt -= take;
    }
    off += counts[e];
  }
  int cursor[NEMO_EXPERTS];
  memcpy(cursor, offsets, sizeof cursor);
  for (int r = 0; r < rows; r++) {
    for (int k = 0; k < NEMO_TOPK; k++) {
      int src = r * NEMO_TOPK + k;
      int dst = cursor[route_ids[src]]++;
      ptok[dst] = r;
      pw[dst] = route_w[src];
      tp[src] = dst;
    }
  }
  return ntiles;
}

// ---- Config parsing and weight registration ----

// Reads a required JSON number field from obj.
static double json_req_num(const Json *obj, const char *key, const char *path) {
  const Json *v = json_get(obj, key);
  if (!v || v->type != J_NUM) die("%s: missing numeric field %s", path, key);
  return v->num;
}

// Records one EOS token ID if not already present.
static void add_eos_token(int id) {
  if (id < 0 || id >= NEMO_VOCAB) return;
  for (int i = 0; i < n_eos_tokens; i++)
    if (eos_tokens[i] == id) return;
  if (n_eos_tokens < MAX_EOS_TOKENS) eos_tokens[n_eos_tokens++] = id;
}

// Parses config.json, validates backbone geometry and fills layer_desc.
static void read_config(const char *dir) {
  char path[PATH_CAP];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);

  const Json *mt = json_get(root, "model_type");
  has_audio = (mt && mt->type == J_STR && !strcmp(mt->s, "nemo_speechlm"));
  const Json *llm = has_audio ? json_get(root, "llm_config") : root;
  if (!llm || llm->type != J_OBJ) die("%s: missing llm_config", path);

  n_eos_tokens = 0;
  const Json *eos = json_get(root, "eos_token_id");
  if (eos && eos->type == J_NUM) {
    add_eos_token((int)eos->num);
  } else if (eos && eos->type == J_ARR) {
    for (int i = 0; i < eos->n; i++)
      if (eos->kid[i].type == J_NUM) add_eos_token((int)eos->kid[i].num);
  }
  if (n_eos_tokens == 0)
    add_eos_token(has_audio ? DEFAULT_SPEECH_EOS : DEFAULT_TEXT_EOS);

  if ((int)json_req_num(llm, "hidden_size", path) != NEMO_DIM ||
      (int)json_req_num(llm, "vocab_size", path) != NEMO_VOCAB ||
      (int)json_req_num(llm, "num_attention_heads", path) != NEMO_HEADS ||
      (int)json_req_num(llm, "num_key_value_heads", path) != NEMO_KV_HEADS ||
      (int)json_req_num(llm, "head_dim", path) != NEMO_HEAD_DIM ||
      (int)json_req_num(llm, "mamba_num_heads", path) != NEMO_MAMBA_HEADS ||
      (int)json_req_num(llm, "mamba_head_dim", path) != NEMO_MAMBA_HEAD_DIM ||
      (int)json_req_num(llm, "ssm_state_size", path) != NEMO_MAMBA_STATE ||
      (int)json_req_num(llm, "n_groups", path) != NEMO_MAMBA_GROUPS ||
      (int)json_req_num(llm, "conv_kernel", path) != NEMO_MAMBA_CONV_K ||
      (int)json_req_num(llm, "n_routed_experts", path) != NEMO_EXPERTS ||
      (int)json_req_num(llm, "num_experts_per_tok", path) != NEMO_TOPK ||
      (int)json_req_num(llm, "moe_intermediate_size", path) != NEMO_MOE_INTER ||
      (int)json_req_num(llm, "moe_shared_expert_intermediate_size", path) !=
          NEMO_SHARED_INTER)
    die("%s: Nemotron-H config dimensions do not match", path);

  const Json *bt = json_get(llm, "layers_block_type");
  if (!bt || bt->type != J_ARR || bt->n != NEMO_LAYERS)
    die("%s: layers_block_type must have %d entries", path, NEMO_LAYERS);

  int nm = 0, na = 0, ne = 0;
  for (int l = 0; l < NEMO_LAYERS; l++) {
    if (bt->kid[l].type != J_STR)
      die("%s: invalid layers_block_type[%d]", path, l);
    const char *s = bt->kid[l].s;
    if (!strcmp(s, "mamba") || !strcmp(s, "linear_attention")) {
      layer_desc[l].kind = LAYER_MAMBA;
      layer_desc[l].sub_idx = nm++;
    } else if (!strcmp(s, "attention") || !strcmp(s, "full_attention")) {
      layer_desc[l].kind = LAYER_ATTN;
      layer_desc[l].sub_idx = na++;
    } else if (!strcmp(s, "moe")) {
      layer_desc[l].kind = LAYER_MOE;
      layer_desc[l].sub_idx = ne++;
    } else {
      die("%s: unsupported block type %s at layer %d", path, s, l);
    }
  }
  if (nm != NEMO_MAMBA_LAYERS || na != NEMO_ATTN_LAYERS ||
      ne != NEMO_MOE_LAYERS)
    die("%s: unexpected block counts (mamba=%d attn=%d moe=%d)", path, nm, na,
        ne);

  json_free(root);
  free(text);
}

// Adds a raw tensor to text_weights, using aligned direct reads only when large
// and 16-byte aligned in the safetensors file.
static void add_raw(Buf *dst, Tensor t) {
  int aligned =
      t.size >= ALIGN_READ_MIN_BYTES && (t.off & ALIGN_16_MASK) == 0 ? 1 : 0;
  weight_group_add_tensor(text_weights, dst, t, aligned);
}

// Registers one per-tensor FP8 linear projection in text_weights.
static void add_fp8_lin(Fp8Lin *lin, Tensor w, Tensor sc, int n, int k) {
  if (w.dtype != '8' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k)
    die("nemotron: unexpected FP8 matrix shape (%lldx%lld, expected %dx%d)",
        (long long)w.shape[0], (long long)w.shape[1], n, k);
  lin->n = n;
  lin->k = k;
  lin->scale = read_scalar_f32(sc) * FP8_UNDO;
  add_raw(&lin->w, w);
}

// Registers one dense NVFP4 linear projection in text_weights.
static void add_nvfp4_lin(Nvfp4Lin *lin, Tensor w, Tensor sc, Tensor sc2, int n,
                          int k) {
  int w_cols = k / NVFP4_VALUES_PER_BYTE;
  int sc_cols = k / NEMO_NVFP4_BLOCK;
  if (w.dtype != 'u' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != w_cols)
    die("nemotron: unexpected NVFP4 weight shape (%lldx%lld, expected %dx%d)",
        (long long)w.shape[0], (long long)w.shape[1], n, w_cols);
  if (sc.dtype != '8' || sc.ndim != 2 || sc.shape[0] != n ||
      sc.shape[1] != sc_cols)
    die("nemotron: unexpected NVFP4 scale shape (%lldx%lld, expected %dx%d)",
        (long long)sc.shape[0], (long long)sc.shape[1], n, sc_cols);
  lin->n = n;
  lin->k = k;
  lin->scale2 = read_scalar_f32(sc2);
  add_raw(&lin->w, w);
  add_raw(&lin->sc, sc);
}

// WeightFill callback reading 128 per-expert FP32 weight_scale_2 scalars.
static void fill_expert_scale2(Buf *buf, const void *ctx) {
  const Scale2FillParam *p = ctx;
  float *dst = (float *)buf->ptr;
  for (int e = 0; e < NEMO_EXPERTS; e++) dst[e] = read_scalar_f32(p->t[e]);
}

// Registers all dense backbone weights in text_weights and all 23 * 128 routed
// expert file spans in expert_spans.
static void register_weights(void) {
  add_raw(&embed_w, tensor_index_get(checkpoint, "backbone.embeddings.weight"));
  weight_group_add_vector(
      text_weights, &final_norm_w,
      tensor_index_get(checkpoint, "backbone.norm_f.weight"), 0.0f, 0);
  add_raw(&lm_head_w, tensor_index_get(checkpoint, "lm_head.weight"));

  for (int l = 0; l < NEMO_LAYERS; l++) {
    LayerDesc *ld = &layer_desc[l];
    weight_group_add_vector(
        text_weights, &ld->norm_w,
        tensor_index_get(checkpoint, "backbone.layers.%d.norm.weight", l), 0.0f,
        0);

    if (ld->kind == LAYER_MAMBA) {
      MambaLayer *ml = &mamba_layers[ld->sub_idx];
      add_fp8_lin(
          &ml->in_proj,
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.in_proj.weight", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.in_proj.weight_scale", l),
          NEMO_MAMBA_IN_DIM, NEMO_DIM);
      add_fp8_lin(
          &ml->out_proj,
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.out_proj.weight", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.out_proj.weight_scale", l),
          NEMO_DIM, NEMO_MAMBA_INNER);
      weight_group_add_vector(
          text_weights, &ml->conv_w,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.conv1d.weight",
                           l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &ml->conv_b,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.conv1d.bias",
                           l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &ml->a_exp,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.A_log", l),
          0.0f, WEIGHT_EXP);
      weight_group_add_vector(
          text_weights, &ml->d_skip,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.D", l), 0.0f,
          0);
      weight_group_add_vector(
          text_weights, &ml->dt_bias,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.dt_bias", l),
          0.0f, 0);
      weight_group_add_vector(
          text_weights, &ml->norm_w,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.norm.weight",
                           l),
          0.0f, 0);
    } else if (ld->kind == LAYER_ATTN) {
      AttnLayer *al = &attn_layers[ld->sub_idx];
      add_nvfp4_lin(
          &al->q,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.q_proj.weight",
                           l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.q_proj.weight_scale", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.q_proj.weight_scale_2", l),
          NEMO_Q_DIM, NEMO_DIM);
      add_nvfp4_lin(
          &al->k,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.k_proj.weight",
                           l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.k_proj.weight_scale", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.k_proj.weight_scale_2", l),
          NEMO_KV_DIM, NEMO_DIM);
      add_nvfp4_lin(
          &al->v,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.v_proj.weight",
                           l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.v_proj.weight_scale", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.v_proj.weight_scale_2", l),
          NEMO_KV_DIM, NEMO_DIM);
      add_nvfp4_lin(
          &al->o,
          tensor_index_get(checkpoint, "backbone.layers.%d.mixer.o_proj.weight",
                           l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.o_proj.weight_scale", l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.o_proj.weight_scale_2", l),
          NEMO_DIM, NEMO_Q_DIM);
    } else {
      int m = ld->sub_idx;
      MoeLayer *el = &moe_layers[m];
      add_raw(&el->router_w,
              tensor_index_get(checkpoint,
                               "backbone.layers.%d.mixer.gate.weight", l));
      weight_group_add_vector(
          text_weights, &el->router_bias,
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.gate.e_score_correction_bias", l),
          0.0f, 0);
      add_nvfp4_lin(
          &el->shared_up,
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.shared_experts.up_proj.weight", l),
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.shared_experts.up_proj.weight_scale",
              l),
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.shared_experts.up_proj.weight_scale_2",
              l),
          NEMO_SHARED_INTER, NEMO_DIM);
      add_nvfp4_lin(
          &el->shared_down,
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.shared_experts.down_proj.weight", l),
          tensor_index_get(
              checkpoint,
              "backbone.layers.%d.mixer.shared_experts.down_proj.weight_scale",
              l),
          tensor_index_get(checkpoint,
                           "backbone.layers.%d.mixer.shared_experts.down_proj."
                           "weight_scale_2",
                           l),
          NEMO_DIM, NEMO_SHARED_INTER);

      Scale2FillParam up_s2, dn_s2;
      uint64_t up_w_bytes =
          (uint64_t)NEMO_MOE_INTER * (NEMO_DIM / NVFP4_VALUES_PER_BYTE);
      uint64_t up_s_bytes =
          (uint64_t)NEMO_MOE_INTER * (NEMO_DIM / NEMO_NVFP4_BLOCK);
      uint64_t dn_w_bytes =
          (uint64_t)NEMO_DIM * (NEMO_MOE_INTER / NVFP4_VALUES_PER_BYTE);
      uint64_t dn_s_bytes =
          (uint64_t)NEMO_DIM * (NEMO_MOE_INTER / NEMO_NVFP4_BLOCK);
      for (int e = 0; e < NEMO_EXPERTS; e++) {
        Tensor uw = tensor_index_get(
            checkpoint, "backbone.layers.%d.mixer.experts.%d.up_proj.weight", l,
            e);
        Tensor us = tensor_index_get(
            checkpoint,
            "backbone.layers.%d.mixer.experts.%d.up_proj.weight_scale", l, e);
        Tensor dw = tensor_index_get(
            checkpoint, "backbone.layers.%d.mixer.experts.%d.down_proj.weight",
            l, e);
        Tensor ds = tensor_index_get(
            checkpoint,
            "backbone.layers.%d.mixer.experts.%d.down_proj.weight_scale", l, e);
        if (uw.size != up_w_bytes || us.size != up_s_bytes ||
            dw.size != dn_w_bytes || ds.size != dn_s_bytes)
          die("nemotron: layer %d expert %d byte size mismatch", l, e);
        if ((uw.off & ALIGN_8_MASK) || (us.off & ALIGN_8_MASK) ||
            (dw.off & ALIGN_8_MASK) || (ds.off & ALIGN_8_MASK))
          die("nemotron: layer %d expert %d is not 8-byte aligned", l, e);
        expert_spans[m][e][0] = (FileSpan){uw.fd, uw.off, uw.size};
        expert_spans[m][e][1] = (FileSpan){us.fd, us.off, us.size};
        expert_spans[m][e][2] = (FileSpan){dw.fd, dw.off, dw.size};
        expert_spans[m][e][3] = (FileSpan){ds.fd, ds.off, ds.size};
        up_s2.t[e] = tensor_index_get(
            checkpoint,
            "backbone.layers.%d.mixer.experts.%d.up_proj.weight_scale_2", l, e);
        dn_s2.t[e] = tensor_index_get(
            checkpoint,
            "backbone.layers.%d.mixer.experts.%d.down_proj.weight_scale_2", l,
            e);
      }
      expert_span_counts[m] = EXPERT_PARTS;
      weight_group_add_generated(text_weights, &el->factor_up,
                                 (size_t)NEMO_EXPERTS * sizeof(float),
                                 fill_expert_scale2, &up_s2, sizeof up_s2);
      weight_group_add_generated(text_weights, &el->factor_down,
                                 (size_t)NEMO_EXPERTS * sizeof(float),
                                 fill_expert_scale2, &dn_s2, sizeof dn_s2);
    }
  }
}

// ---- Workspace management ----

// Returns the GPU workspace byte footprint needed for `rows` active tokens.
static uint64_t estimate_work_bytes(int rows) {
  int cap = rows <= NEMO_MAX_BATCH_ROWS
                ? NEMO_MAX_BATCH_ROWS
                : (rows <= NEMO_MAX_VERIFY ? NEMO_MAX_VERIFY : rows);
  int q_stride = align_up(cap, PREFILL_Q_PAD);
  int padded_m2 = align_up(cap, MM_TILE_M2);
  int npairs = cap * NEMO_TOPK;
  int max_tiles = npairs + NEMO_EXPERTS;
  int part_rows = cap < NEMO_MAX_BATCH_ROWS ? cap : NEMO_MAX_BATCH_ROWS;
  int route_id_count = NEMO_MOE_LAYERS * NEMO_MAX_VERIFY * NEMO_TOPK;
  if (npairs > route_id_count) route_id_count = npairs;
  uint64_t bytes = 0;
  bytes += gpu_round_up_to_pages((size_t)cap * sizeof(int32_t));
  bytes += 2 * gpu_round_up_to_pages((size_t)cap * NEMO_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)padded_m2 * NEMO_MAMBA_INNER *
                                 sizeof(uint16_t));
  bytes +=
      gpu_round_up_to_pages((size_t)cap * NEMO_MAMBA_IN_DIM * sizeof(float));
  bytes +=
      gpu_round_up_to_pages((size_t)cap * NEMO_MAMBA_CONV_DIM * sizeof(float));
  bytes +=
      2 * gpu_round_up_to_pages((size_t)cap * NEMO_MAMBA_INNER * sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)cap * NEMO_Q_DIM * sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)cap * NEMO_KV_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)NEMO_HEADS * q_stride * NEMO_HEAD_DIM *
                                 sizeof(uint16_t));
  bytes +=
      gpu_round_up_to_pages((size_t)part_rows * NEMO_HEADS * MAX_DECODE_PARTS *
                            DECODE_PART_FLOATS * sizeof(float));
  bytes +=
      gpu_round_up_to_pages((size_t)cap * NEMO_SHARED_INTER * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)cap * NEMO_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)cap * NEMO_EXPERTS * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)route_id_count * sizeof(uint32_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)npairs * sizeof(int32_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)max_tiles * sizeof(NemoMoeTile));
  bytes += gpu_round_up_to_pages((size_t)NEMO_EXPERTS * sizeof(ExpertRef));
  bytes += gpu_round_up_to_pages((size_t)(npairs + MOE_MM_TILE) * NEMO_DIM *
                                 sizeof(uint16_t));
  bytes += gpu_round_up_to_pages((size_t)(npairs + MOE_MM_TILE) *
                                 NEMO_MOE_INTER * sizeof(uint16_t));
  bytes += gpu_round_up_to_pages((size_t)npairs * NEMO_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)NEMO_MAMBA_LAYERS *
                                 NEMO_MAX_BATCH_ROWS * sizeof(MambaTarget));
  bytes += gpu_round_up_to_pages((size_t)NEMO_ATTN_LAYERS *
                                 NEMO_MAX_BATCH_ROWS * sizeof(KvTarget));
  bytes += gpu_round_up_to_pages(LOGITS_BYTES);
  return bytes;
}

// Returns the extra workspace bytes needed by multi-token verify() rollback
// buffers and multi-row logits.
static uint64_t estimate_verify_bytes(int rows) {
  int vcap = rows > NEMO_MAX_VERIFY ? rows : NEMO_MAX_VERIFY;
  uint64_t bytes = gpu_round_up_to_pages(CONV_TOTAL_BYTES);
  bytes += gpu_round_up_to_pages(SSM_TOTAL_BYTES);
  bytes += gpu_round_up_to_pages((size_t)NEMO_MAMBA_LAYERS * (size_t)vcap *
                                 NEMO_MAMBA_IN_DIM * sizeof(float));
  if (rows > 1)
    bytes += gpu_round_up_to_pages((size_t)(rows - 1) * LOGITS_BYTES);
  return bytes;
}

// Returns the bytes currently allocated in pass-local resizable workspace
// buffers (excluding work.logits).
static uint64_t work_resizable_bytes(void) {
  const Buf *list[] = {&work.ids,        &work.cur,          &work.normed,
                       &work.x_pack,     &work.mamba_zxbcdt, &work.mamba_xbc,
                       &work.mamba_y,    &work.mamba_normed, &work.q_raw,
                       &work.k_raw,      &work.v_raw,        &work.q_hm,
                       &work.attn_part,  &work.attn_out,     &work.shared_h,
                       &work.shared_out, &work.route_logits, &work.route_ids,
                       &work.route_w,    &work.ptok,         &work.pw,
                       &work.tp,         &work.tiles,        &work.expert_table,
                       &work.x_bf16,     &work.h_bf16,       &work.y_pair,
                       &work.mamba_tgt,  &work.kv_tgt,       &work.verify_conv,
                       &work.verify_ssm, &work.verify_zxbcdt};
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    bytes += list[i]->allocated;
  return bytes;
}

// Marks all buffers in w to be freed once prior GPU commands complete.
static void free_work(NemoWork *w) {
  Buf *list[] = {
      &w->ids,          &w->cur,         &w->normed,     &w->x_pack,
      &w->mamba_zxbcdt, &w->mamba_xbc,   &w->mamba_y,    &w->mamba_normed,
      &w->q_raw,        &w->k_raw,       &w->v_raw,      &w->q_hm,
      &w->attn_part,    &w->attn_out,    &w->shared_h,   &w->shared_out,
      &w->route_logits, &w->route_ids,   &w->route_w,    &w->ptok,
      &w->pw,           &w->tp,          &w->tiles,      &w->expert_table,
      &w->x_bf16,       &w->h_bf16,      &w->y_pair,     &w->mamba_tgt,
      &w->kv_tgt,       &w->verify_conv, &w->verify_ssm, &w->verify_zxbcdt,
      &w->logits};
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    if (list[i]->obj) gpu_mark_to_free(list[i]);
  *w = (NemoWork){0};
}

// Ensures pass-local workspace buffers hold `rows` active rows and
// `logits_rows` rows in work.logits, shrinking back after prefill.
static void ensure_work(int rows, int logits_rows) {
  int new_cap = rows <= NEMO_MAX_BATCH_ROWS
                    ? NEMO_MAX_BATCH_ROWS
                    : (rows <= NEMO_MAX_VERIFY ? NEMO_MAX_VERIFY : rows);
  int shrink = (new_cap <= NEMO_MAX_VERIFY && work.cap_rows > NEMO_MAX_VERIFY);
  if (rows > work.cap_rows || shrink) {
    int old_lr = work.logits_rows;
    Buf old_logits = work.logits;
    work.logits = (Buf){0};
    free_work(&work);
    work.cap_rows = new_cap;
    work.q_stride = align_up(new_cap, PREFILL_Q_PAD);
    int padded_m2 = align_up(new_cap, MM_TILE_M2);
    int npairs = new_cap * NEMO_TOPK;
    int max_tiles = npairs + NEMO_EXPERTS;
    int part_rows =
        new_cap < NEMO_MAX_BATCH_ROWS ? new_cap : NEMO_MAX_BATCH_ROWS;
    int route_id_count = NEMO_MOE_LAYERS * NEMO_MAX_VERIFY * NEMO_TOPK;
    if (npairs > route_id_count) route_id_count = npairs;

    work.ids = work_alloc((size_t)new_cap * sizeof(int32_t));
    work.cur = work_alloc((size_t)new_cap * NEMO_DIM * sizeof(float));
    work.normed = work_alloc((size_t)new_cap * NEMO_DIM * sizeof(float));
    work.x_pack =
        work_alloc((size_t)padded_m2 * NEMO_MAMBA_INNER * sizeof(uint16_t));
    work.mamba_zxbcdt =
        work_alloc((size_t)new_cap * NEMO_MAMBA_IN_DIM * sizeof(float));
    work.mamba_xbc =
        work_alloc((size_t)new_cap * NEMO_MAMBA_CONV_DIM * sizeof(float));
    work.mamba_y =
        work_alloc((size_t)new_cap * NEMO_MAMBA_INNER * sizeof(float));
    work.mamba_normed =
        work_alloc((size_t)new_cap * NEMO_MAMBA_INNER * sizeof(float));
    work.q_raw = work_alloc((size_t)new_cap * NEMO_Q_DIM * sizeof(float));
    work.k_raw = work_alloc((size_t)new_cap * NEMO_KV_DIM * sizeof(float));
    work.v_raw = work_alloc((size_t)new_cap * NEMO_KV_DIM * sizeof(float));
    work.q_hm = work_alloc((size_t)NEMO_HEADS * work.q_stride * NEMO_HEAD_DIM *
                           sizeof(uint16_t));
    work.attn_part =
        work_alloc((size_t)part_rows * NEMO_HEADS * MAX_DECODE_PARTS *
                   DECODE_PART_FLOATS * sizeof(float));
    work.attn_out = work_alloc((size_t)new_cap * NEMO_Q_DIM * sizeof(float));
    work.shared_h =
        work_alloc((size_t)new_cap * NEMO_SHARED_INTER * sizeof(float));
    work.shared_out = work_alloc((size_t)new_cap * NEMO_DIM * sizeof(float));
    work.route_logits =
        work_alloc((size_t)new_cap * NEMO_EXPERTS * sizeof(float));
    work.route_ids = work_alloc((size_t)route_id_count * sizeof(uint32_t));
    work.route_w = work_alloc((size_t)npairs * sizeof(float));
    work.ptok = work_alloc((size_t)npairs * sizeof(int32_t));
    work.pw = work_alloc((size_t)npairs * sizeof(float));
    work.tp = work_alloc((size_t)npairs * sizeof(int32_t));
    work.tiles = work_alloc((size_t)max_tiles * sizeof(NemoMoeTile));
    work.expert_table = work_alloc((size_t)NEMO_EXPERTS * sizeof(ExpertRef));
    work.x_bf16 = work_alloc((size_t)(npairs + MOE_MM_TILE) * NEMO_DIM *
                             sizeof(uint16_t));
    work.h_bf16 = work_alloc((size_t)(npairs + MOE_MM_TILE) * NEMO_MOE_INTER *
                             sizeof(uint16_t));
    work.y_pair = work_alloc((size_t)npairs * NEMO_DIM * sizeof(float));
    work.mamba_tgt = work_alloc((size_t)NEMO_MAMBA_LAYERS *
                                NEMO_MAX_BATCH_ROWS * sizeof(MambaTarget));
    work.kv_tgt = work_alloc((size_t)NEMO_ATTN_LAYERS * NEMO_MAX_BATCH_ROWS *
                             sizeof(KvTarget));
    work.logits = old_logits;
    work.logits_rows = old_lr;
  }
  if (logits_rows > work.logits_rows) {
    if (work.logits.obj) gpu_mark_to_free(&work.logits);
    work.logits = work_alloc((size_t)logits_rows * LOGITS_BYTES);
    work.logits_rows = logits_rows;
  }
}

// Allocates Mamba-2 pre-verify backup and zxbcdt rollback buffers for up to
// `rows` verified tokens.
static void ensure_verify_work(int rows) {
  int need = rows > NEMO_MAX_VERIFY ? rows : NEMO_MAX_VERIFY;
  if (!work.verify_conv.obj) work.verify_conv = work_alloc(CONV_TOTAL_BYTES);
  if (!work.verify_ssm.obj) work.verify_ssm = work_alloc(SSM_TOTAL_BYTES);
  if (need > work.verify_cap) {
    if (work.verify_zxbcdt.obj) gpu_mark_to_free(&work.verify_zxbcdt);
    work.verify_zxbcdt = work_alloc((size_t)NEMO_MAMBA_LAYERS * (size_t)need *
                                    NEMO_MAMBA_IN_DIM * sizeof(float));
    work.verify_cap = need;
  }
}

// Reserves memory for a pass of up to `rows` tokens, halving down to 1 for
// prompt passes when memory is constrained. Returns admitted rows or 0.
static int prepare_rows(int rows, int prompt, int verify) {
  if (!prompt && !verify && work_slot && rows == prepare_prev_rows &&
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
        (verify ? estimate_verify_bytes(candidate) : 0) +
        expert_store_minimum_bytes(experts_store, NEMO_MOE_LAYERS);
    MemoryAdmission mode =
        candidate > 1 && prompt ? MEMORY_WORKSPACE : MEMORY_REQUIRED;
    if (memory_set_minimum(work_slot->account, required, mode)) {
      prepare_prev_rows = (prompt || verify) ? 0 : rows;
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

// ---- Expert store holds and address tables ----

// Releases all individual expert holds and whole-layer holds owned by this
// pass after GPU completion.
static void held_release_all(void) {
  if (!experts_store) return;
  if (held.n) {
    expert_store_release(experts_store, held.layer, held.experts, held.n);
    held.n = 0;
  }
  if (held_layers) {
    for (int m = 0; m < NEMO_MOE_LAYERS; m++)
      if (held_layers & (1u << m)) expert_store_release_layer(experts_store, m);
    held_layers = 0;
  }
}

// Holds `n` routed experts of MoE layer `m` until released.
static void hold_experts(int m, const int *experts, int n) {
  if (held.n)
    die("nemotron: experts of MoE layer %d are still held", held.layer);
  expert_store_hold(experts_store, m, experts, n);
  held.layer = m;
  held.n = n;
  memcpy(held.experts, experts, (size_t)n * sizeof *experts);
}

// Holds all 128 experts of MoE layer `m` once for the open GPU pass.
static void hold_layer_if_needed(int m) {
  uint32_t bit = 1u << m;
  if (!(held_layers & bit)) {
    expert_store_hold_layer(experts_store, m);
    held_layers |= bit;
  }
}

// Rebuilds MoE layer `m`'s resident expert address table on generation change.
static void refresh_layer_table(int m, uint64_t gen) {
  if (layer_table_gen[m] == gen) return;
  ExpertRef *ref = (ExpertRef *)layer_table[m].ptr;
  for (int e = 0; e < NEMO_EXPERTS; e++) {
    uint32_t at[EXPERT_STORE_PARTS];
    int fresh = 0;
    const Buf *b = expert_store_get_buffer(experts_store, m, e, at, &fresh);
    if (!b) die("nemotron: reachable MoE layer %d expert %d has no slot", m, e);
    uint64_t base = b->address + b->off;
    ref[e] =
        (ExpertRef){base + at[0], base + at[1], base + at[2], base + at[3]};
  }
  layer_table_gen[m] = gen;
}

// Fills work.expert_table for active experts `need[0..nn-1]` of MoE layer `m`,
// declaring GPU reads on freshly mapped slots.
static void fill_streaming_table(GpuCommandBuffer *cmd, int m, const int *need,
                                 int nn) {
  ExpertRef *ref = (ExpertRef *)work.expert_table.ptr;
  for (int i = 0; i < nn; i++) {
    int e = need[i];
    uint32_t at[EXPERT_STORE_PARTS];
    int fresh = 0;
    const Buf *b = expert_store_get_buffer(experts_store, m, e, at, &fresh);
    if (!b) die("nemotron: held MoE layer %d expert %d has no slot", m, e);
    uint64_t base = b->address + b->off;
    ref[e] =
        (ExpertRef){base + at[0], base + at[1], base + at[2], base + at[3]};
    if (fresh) gpu_declare_read(cmd, b);
  }
}

// Credits each decoded row's top NEAR_MISSES unselected experts in MoE layer m.
static void credit_near_misses(int m, int rows, const uint8_t *used) {
  const float *bias = (const float *)moe_layers[m].router_bias.ptr;
  int list[NEMO_EXPERTS], n = 0;
  uint8_t listed[NEMO_EXPERTS] = {0};
  for (int r = 0; r < rows; r++) {
    const float *lg =
        (const float *)work.route_logits.ptr + (size_t)r * NEMO_EXPERTS;
    float key[NEMO_EXPERTS];
    for (int e = 0; e < NEMO_EXPERTS; e++)
      key[e] = 1.0f / (1.0f + expf(-lg[e])) + bias[e];
    int top[NEAR_MISSES], nt = 0;
    for (int e = 0; e < NEMO_EXPERTS; e++) {
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
  expert_store_credit_near_misses(experts_store, m, list, n);
}

// Converts an ExpertPlace value to the eremite route source code.
static uint8_t place_to_source(ExpertPlace place) {
  switch (place) {
  case EXPERT_IN_MEMORY: return ROUTE_SOURCE_MEMORY;
  case EXPERT_READ_AHEAD: return ROUTE_SOURCE_AHEAD;
  case EXPERT_BEING_READ_AHEAD: return ROUTE_SOURCE_LATE;
  case EXPERT_ON_SSD: return ROUTE_SOURCE_SSD;
  }
  return ROUTE_SOURCE_SSD;
}

// Records hit counts, expert-store uses and route history for resident
// fast-path MoE layers executed before the latest model_sync.
static void flush_fast_layers(uint8_t fast_moe[NEMO_MOE_LAYERS], int rows) {
  ExpertStoreCounts *st = expert_stats();
  for (int m = 0; m < NEMO_MOE_LAYERS; m++) {
    if (!fast_moe[m]) continue;
    fast_moe[m] = 0;
    const uint32_t *rids = (const uint32_t *)work.route_ids.ptr +
                           (size_t)m * (size_t)rows * NEMO_TOPK;
    uint8_t used[NEMO_EXPERTS] = {0};
    int need[NEMO_EXPERTS], nn = 0;
    for (int r = 0; r < rows; r++) {
      for (int k = 0; k < NEMO_TOPK; k++) {
        uint32_t e = rids[(size_t)r * NEMO_TOPK + (size_t)k];
        if (e >= (uint32_t)NEMO_EXPERTS)
          die("nemotron: router selected expert %u", e);
        used[e] = 1;
        ExpertPlace place = expert_store_locate(experts_store, m, (int)e);
        if (place == EXPERT_READ_AHEAD)
          st->row_ahead++;
        else
          st->row_own++;
        if (r < ROUTE_DECODE_CAP) {
          route_id[r][m][k] = (uint16_t)e;
          route_from[r][m][k] = place == EXPERT_READ_AHEAD
                                    ? ROUTE_SOURCE_AHEAD
                                    : ROUTE_SOURCE_MEMORY;
          route_step[r][m] = route_steps;
        }
      }
    }
    for (int e = 0; e < NEMO_EXPERTS; e++)
      if (used[e]) need[nn++] = e;
    st->row_unique += (uint64_t)nn;
    expert_store_count_use(experts_store, m, need, nn);
  }
}

// Starts read-ahead for all 128 experts of MoE layer `m` during large prefills.
static void prefetch_all_layer_experts(int m) {
  int all[NEMO_EXPERTS];
  for (int e = 0; e < NEMO_EXPERTS; e++) all[e] = e;
  expert_store_read_ahead(experts_store, m, all, NEMO_EXPERTS,
                          2 * NEMO_EXPERTS);
}

// ---- State and snapshot management ----

// Returns the next power-of-two KV capacity >= positions (multiple of 64,
// capped at opt.ctx).
static int grown_cap(int positions) {
  int need = align_up(positions > 0 ? positions : 1, KV_CAP_ALIGN);
  int cap = START_CAP;
  while (cap < need && cap < opt.ctx) cap *= 2;
  int max_cap = align_up(opt.ctx, KV_CAP_ALIGN);
  if (cap > max_cap) cap = max_cap;
  if (positions > opt.ctx)
    die("nemotron: sequence length %d exceeds context limit %d", positions,
        opt.ctx);
  return cap;
}

// Resizes the 24 growing KV buffers of s to `new_cap` rows, preserving the
// first `keep` rows and zero-initializing the newly added tail. Returns 1 on
// success or 0 on memory refusal.
static int state_resize_kv(NemoState *s, int new_cap, int keep) {
  size_t new_bytes = (size_t)new_cap * KV_HEAD_ROW_BYTES;
  size_t copy_bytes = (size_t)keep * KV_HEAD_ROW_BYTES;
  Buf nk[NEMO_ATTN_LAYERS][NEMO_KV_HEADS] = {0};
  Buf nv[NEMO_ATTN_LAYERS][NEMO_KV_HEADS] = {0};
  for (int a = 0; a < NEMO_ATTN_LAYERS; a++) {
    for (int h = 0; h < NEMO_KV_HEADS; h++) {
      nk[a][h] = state_try_alloc(new_bytes);
      nv[a][h] = nk[a][h].obj ? state_try_alloc(new_bytes) : (Buf){0};
      if (!nv[a][h].obj) {
        for (int pa = 0; pa <= a; pa++) {
          for (int ph = 0; ph < NEMO_KV_HEADS; ph++) {
            if (nk[pa][ph].obj) gpu_free(&nk[pa][ph]);
            if (nv[pa][ph].obj) gpu_free(&nv[pa][ph]);
          }
        }
        return 0;
      }
      if (copy_bytes && s->k[a][h].obj) {
        memcpy(nk[a][h].ptr, s->k[a][h].ptr, copy_bytes);
        memcpy(nv[a][h].ptr, s->v[a][h].ptr, copy_bytes);
      }
      if (new_bytes > copy_bytes) {
        memset((uint8_t *)nk[a][h].ptr + copy_bytes, 0, new_bytes - copy_bytes);
        memset((uint8_t *)nv[a][h].ptr + copy_bytes, 0, new_bytes - copy_bytes);
      }
    }
  }
  for (int a = 0; a < NEMO_ATTN_LAYERS; a++) {
    for (int h = 0; h < NEMO_KV_HEADS; h++) {
      if (s->k[a][h].obj) gpu_mark_to_free(&s->k[a][h]);
      if (s->v[a][h].obj) gpu_mark_to_free(&s->v[a][h]);
      s->k[a][h] = nk[a][h];
      s->v[a][h] = nv[a][h];
    }
  }
  s->cap = new_cap;
  return 1;
}

// Ensures s holds at least `need` positions in its KV buffers.
static void state_ensure_cap(NemoState *s, int need) {
  if (need <= s->cap) return;
  model_sync();
  int next = grown_cap(need);
  if (!state_resize_kv(s, next, s->cap))
    die("nemotron: out of memory growing KV cache to %d tokens", next);
}

// Allocates a new NemoState, or returns null when memory does not fit.
static ModelState *nemo_state_new(void) {
  NemoState *s = xmalloc(sizeof *s);
  *s = (NemoState){0};
  int init_cap = grown_cap(1);
  s->conv_state = state_try_alloc(CONV_TOTAL_BYTES);
  s->ssm_state =
      s->conv_state.obj ? state_try_alloc(SSM_TOTAL_BYTES) : (Buf){0};
  if (!s->ssm_state.obj || !state_resize_kv(s, init_cap, 0)) {
    if (s->conv_state.obj) gpu_free(&s->conv_state);
    if (s->ssm_state.obj) gpu_free(&s->ssm_state);
    free(s);
    return NULL;
  }
  memset(s->conv_state.ptr, 0, CONV_TOTAL_BYTES);
  memset(s->ssm_state.ptr, 0, SSM_TOTAL_BYTES);
  return (ModelState *)s;
}

// Frees a NemoState and its GPU buffers.
static void nemo_state_free(ModelState *ms) {
  NemoState *s = (NemoState *)ms;
  if (!s) return;
  for (int a = 0; a < NEMO_ATTN_LAYERS; a++) {
    for (int h = 0; h < NEMO_KV_HEADS; h++) {
      if (s->k[a][h].obj) gpu_mark_to_free(&s->k[a][h]);
      if (s->v[a][h].obj) gpu_mark_to_free(&s->v[a][h]);
    }
  }
  if (s->conv_state.obj) gpu_mark_to_free(&s->conv_state);
  if (s->ssm_state.obj) gpu_mark_to_free(&s->ssm_state);
  free(s);
}

// Returns the byte footprint of state s (or a new state when s is null) at
// `positions` tokens.
static size_t nemo_state_bytes(const ModelState *ms, int positions) {
  const NemoState *s = (const NemoState *)ms;
  int cap = s ? s->cap : grown_cap(positions > 0 ? positions : 1);
  if (positions > cap) cap = grown_cap(positions);
  size_t kv_one = gpu_round_up_to_pages((size_t)cap * KV_HEAD_ROW_BYTES);
  return (size_t)(NEMO_ATTN_LAYERS * NEMO_KV_HEADS * 2) * kv_one +
         gpu_round_up_to_pages(CONV_TOTAL_BYTES) +
         gpu_round_up_to_pages(SSM_TOTAL_BYTES);
}

// Writes the 24 growing KV regions of s. When fill is nonzero, resizes s to
// hold `positions` tokens first.
static int nemo_write_state_regions(ModelState *ms, int positions, int fill,
                                    StateRegion *out) {
  NemoState *s = (NemoState *)ms;
  if (positions < 0 || (!fill && positions > s->cap))
    die("nemotron: invalid state regions for %d positions", positions);
  model_sync();
  if (fill) {
    int cap = grown_cap(positions > 0 ? positions : 1);
    if (s->cap != cap && !state_resize_kv(s, cap, 0))
      die("nemotron: out of memory sizing state to %d tokens", cap);
  }
  int n = 0;
  for (int a = 0; a < NEMO_ATTN_LAYERS; a++) {
    for (int h = 0; h < NEMO_KV_HEADS; h++) {
      out[n++] =
          (StateRegion){s->k[a][h].ptr, (uint32_t)KV_HEAD_ROW_BYTES, 1, 0, 0};
      out[n++] =
          (StateRegion){s->v[a][h].ptr, (uint32_t)KV_HEAD_ROW_BYTES, 1, 0, 0};
    }
  }
  return n;
}

// Copies the growing KV rows before `positions` from src into dst.
static void nemo_state_copy(ModelState *dst_ms, const ModelState *src_ms,
                            int positions) {
  if (dst_ms == src_ms) die("nemotron: cannot copy a state onto itself");
  StateRegion to[STATE_REGIONS_MAX], from[STATE_REGIONS_MAX];
  int n = nemo_write_state_regions(dst_ms, positions, 1, to);
  nemo_write_state_regions((ModelState *)src_ms, positions, 0, from);
  regions_copy(to, from, n, positions);
}

// Captures the rolling Mamba-2 conv and SSM states of s at `pos`, plus
// optional logits.
static ModelSnapshot *nemo_snapshot_take(ModelState *ms, int pos,
                                         const float *lg) {
  NemoState *s = (NemoState *)ms;
  model_sync();
  NemoSnapshot *x = xmalloc(sizeof *x);
  *x = (NemoSnapshot){
      .pos = pos,
      .rows = memory_host_alloc_required(&state_slot, SNAP_ROLLING_BYTES)};
  memcpy(x->rows, s->conv_state.ptr, CONV_TOTAL_BYTES);
  memcpy(x->rows + CONV_TOTAL_BYTES, s->ssm_state.ptr, SSM_TOTAL_BYTES);
  if (lg) {
    x->logits = memory_host_alloc_required(&state_slot, LOGITS_BYTES);
    memcpy(x->logits, lg, LOGITS_BYTES);
  }
  return (ModelSnapshot *)x;
}

// Restores rolling Mamba-2 conv and SSM states from x into s.
static void nemo_snapshot_restore(ModelState *ms, const ModelSnapshot *mx) {
  NemoState *s = (NemoState *)ms;
  const NemoSnapshot *x = (const NemoSnapshot *)mx;
  model_sync();
  memcpy(s->conv_state.ptr, x->rows, CONV_TOTAL_BYTES);
  memcpy(s->ssm_state.ptr, x->rows + CONV_TOTAL_BYTES, SSM_TOTAL_BYTES);
  s->pos = x->pos;
}

// Frees a snapshot and its host buffers.
static void nemo_snapshot_free(ModelSnapshot *mx) {
  NemoSnapshot *x = (NemoSnapshot *)mx;
  if (!x) return;
  memory_host_free(&state_slot, x->rows);
  memory_host_free(&state_slot, x->logits);
  free(x);
}

// Returns the byte footprint of one snapshot.
static size_t nemo_snapshot_bytes(int with_logits) {
  return SNAP_ROLLING_BYTES + (with_logits ? LOGITS_BYTES : 0);
}

// Returns the saved logits of snapshot x, or null if none were saved.
static const float *nemo_snapshot_logits(const ModelSnapshot *mx) {
  return ((const NemoSnapshot *)mx)->logits;
}

// Writes the serialized parts and metadata of snapshot x, or of live state s
// at `pos` when x is null.
static int nemo_write_snapshot_parts(ModelSnapshot *mx, ModelState *ms, int pos,
                                     SnapshotPart *out, int32_t *meta,
                                     int *nmeta) {
  model_sync();
  NemoSnapshot *x = (NemoSnapshot *)mx;
  NemoState *s = (NemoState *)ms;
  *nmeta = 1;
  int n = 0;
  if (x) {
    meta[0] = x->pos;
    out[n++] = (SnapshotPart){x->rows, SNAP_ROLLING_BYTES};
    if (x->logits) out[n++] = (SnapshotPart){x->logits, LOGITS_BYTES};
    return n;
  }
  if (s->pos != pos) return -1;
  meta[0] = pos;
  out[n++] = (SnapshotPart){s->conv_state.ptr, CONV_TOTAL_BYTES};
  out[n++] = (SnapshotPart){s->ssm_state.ptr, SSM_TOTAL_BYTES};
  return n;
}

// Allocates an empty snapshot matching serialized metadata.
static ModelSnapshot *nemo_snapshot_new_empty(const int32_t *meta, int nmeta,
                                              int with_logits) {
  if (nmeta != 1 || meta[0] < 0) return NULL;
  NemoSnapshot *x = xmalloc(sizeof *x);
  *x = (NemoSnapshot){
      .pos = meta[0],
      .rows = memory_host_alloc_required(&state_slot, SNAP_ROLLING_BYTES)};
  if (with_logits)
    x->logits = memory_host_alloc_required(&state_slot, LOGITS_BYTES);
  return (ModelSnapshot *)x;
}

// ---- GPU dispatch helpers ----

// Chooses split-K for mv_bf16 / mvm_bf16 on narrow projections.
static uint32_t pick_bf16_split(int n, int k) {
  return (n < (int)BF16_MV_SPLIT_SMALL_N && k >= (int)BF16_MV_SPLIT_MIN_K)
             ? BF16_MV_SPLIT_FACTOR
             : 1u;
}

// Dispatches a BF16 linear projection y = x @ w^T starting at byte offset x_off
// of x, selecting mv_bf16 (m == 1), mvm_bf16 (2..8) or mm_bf16 (m > 8).
static void run_bf16_linear(GpuCommandBuffer *c, const Buf *w, const Buf *x,
                            size_t x_off, const Buf *y, int m, int n, int k,
                            int round) {
  if (m == 1) {
    uint32_t split = pick_bf16_split(n, k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    MvParam p = {(uint32_t)n, (uint32_t)k,     (uint32_t)n,
                 0,           (uint32_t)round, split};
    gpu_select_kernel(c, K_mv_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, x_off);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   BF16_MV_THREADS, 1, 1);
  } else if (m <= NEMO_MAX_BATCH_ROWS) {
    uint32_t split = pick_bf16_split(n, k);
    uint32_t rows_per_tg = (BF16_MV_SGS / split) * BF16_MV_ROWS_PER_SG;
    MvmParam p = {
        {(uint32_t)n, (uint32_t)k, (uint32_t)n, 0, (uint32_t)round, split},
        (uint32_t)m,
        (uint32_t)k,
        (uint32_t)n};
    gpu_select_kernel(c, K_mvm_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, x_off);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   BF16_MV_THREADS, 1, 1);
  } else {
    MmParam p = {(uint32_t)m,     (uint32_t)n, (uint32_t)k,
                 (uint32_t)round, (uint32_t)k, (uint32_t)n};
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

// Dispatches NemotronHRMSNorm on `rows` rows of width NEMO_DIM.
static void run_rmsnorm(GpuCommandBuffer *c, const Buf *x, size_t x_off,
                        const Buf *w, const Buf *y, int rows) {
  NormParam p = {(uint32_t)NEMO_DIM, 1u, RMS_EPS};
  gpu_select_kernel(c, K_rmsnorm);
  gpu_bind_buffer(c, 0, x, x_off);
  gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, y, 0);
  gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
}

// Packs FP32 activations [rows, cols] into BF16 work.x_pack padded to
// MM_TILE_M2 rows for 128x64 MPP GEMM kernels.
static void run_pack_bf16(GpuCommandBuffer *c, const Buf *x, int rows,
                          int cols) {
  uint32_t padded = (uint32_t)align_up(rows, MM_TILE_M2);
  uint32_t cols4 = (uint32_t)cols / VEC4_WIDTH;
  uint32_t rc[3] = {(uint32_t)rows, padded, cols4};
  gpu_select_kernel(c, K_pack_bf16);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, &work.x_pack, 0);
  gpu_copy_bytes(c, 2, rc, sizeof rc);
  gpu_run_kernel(c, (cols4 + PACK_THREADS - 1u) / PACK_THREADS, padded, 1,
                 PACK_THREADS, 1, 1);
}

// Dispatches one per-tensor FP8 linear projection y = x @ w^T * scale (+ res).
// When prepacked is nonzero and m > MOE_MV_MAX_M, uses work.x_pack directly.
static void run_fp8_linear(GpuCommandBuffer *c, const Fp8Lin *lin, const Buf *x,
                           const Buf *res, const Buf *y, size_t y_off, int m,
                           int prepacked) {
  Fp8LinParam p = {
      (uint32_t)m, (uint32_t)lin->n, (uint32_t)lin->k, res ? 1u : 0u,
      1u,          lin->scale};
  const Buf *res_buf = res ? res : x;
  if (m <= MOE_MV_MAX_M) {
    gpu_select_kernel(c, K_fp8_mv);
    gpu_bind_buffer(c, 0, &lin->w, 0);
    gpu_bind_buffer(c, 1, x, 0);
    gpu_bind_buffer(c, 2, res_buf, 0);
    gpu_bind_buffer(c, 3, y, y_off);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    uint32_t tgs =
        ((uint32_t)lin->n + FP8_MV_ROWS_PER_TG - 1u) / FP8_MV_ROWS_PER_TG;
    gpu_run_kernel(c, tgs, 1, 1, FP8_MV_THREADS, 1, 1);
  } else {
    if (!prepacked) run_pack_bf16(c, x, m, lin->k);
    gpu_select_kernel(c, K_fp8_mm);
    gpu_bind_buffer(c, 0, &lin->w, 0);
    gpu_bind_buffer(c, 1, &work.x_pack, 0);
    gpu_bind_buffer(c, 2, res_buf, 0);
    gpu_bind_buffer(c, 3, y, y_off);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    uint32_t tiles_m = ((uint32_t)m + MM_TILE_M2 - 1u) / MM_TILE_M2;
    uint32_t tiles_n = ((uint32_t)lin->n + MM_TILE_N - 1u) / MM_TILE_N;
    gpu_run_kernel(c, tiles_n * tiles_m, 1, 1, MM_THREADS, 1, 1);
  }
}

// Dispatches one dense NVFP4 linear projection y = act(x @ w^T * scale2) + res.
// When prepacked is nonzero and m > MOE_MV_MAX_M, uses work.x_pack directly.
static void run_nvfp4_linear(GpuCommandBuffer *c, const Nvfp4Lin *lin,
                             const Buf *x, const Buf *res, const Buf *y, int m,
                             uint32_t act, int prepacked) {
  Nvfp4LinParam p = {(uint32_t)m, (uint32_t)lin->n, (uint32_t)lin->k,
                     act,         res ? 1u : 0u,    1u,
                     lin->scale2};
  const Buf *res_buf = res ? res : x;
  if (m <= MOE_MV_MAX_M) {
    gpu_select_kernel(c, K_nvfp4_mv);
    gpu_bind_buffer(c, 0, &lin->w, 0);
    gpu_bind_buffer(c, 1, &lin->sc, 0);
    gpu_bind_buffer(c, 2, x, 0);
    gpu_bind_buffer(c, 3, res_buf, 0);
    gpu_bind_buffer(c, 4, y, 0);
    gpu_copy_bytes(c, 5, &p, sizeof p);
    uint32_t tgs =
        ((uint32_t)lin->n + NV_MV_ROWS_PER_TG - 1u) / NV_MV_ROWS_PER_TG;
    gpu_run_kernel(c, tgs, 1, 1, NV_MV_THREADS, 1, 1);
  } else {
    if (!prepacked) run_pack_bf16(c, x, m, lin->k);
    gpu_select_kernel(c, K_nvfp4_mm);
    gpu_bind_buffer(c, 0, &lin->w, 0);
    gpu_bind_buffer(c, 1, &lin->sc, 0);
    gpu_bind_buffer(c, 2, &work.x_pack, 0);
    gpu_bind_buffer(c, 3, res_buf, 0);
    gpu_bind_buffer(c, 4, y, 0);
    gpu_copy_bytes(c, 5, &p, sizeof p);
    uint32_t tiles_m = ((uint32_t)m + MM_TILE_M2 - 1u) / MM_TILE_M2;
    uint32_t tiles_n = ((uint32_t)lin->n + MM_TILE_N - 1u) / MM_TILE_N;
    gpu_run_kernel(c, tiles_n * tiles_m, 1, 1, MM_THREADS, 1, 1);
  }
}

// Encodes routed NVFP4 MoE up/down projections and combine+shared+residual add.
static void encode_moe_experts(GpuCommandBuffer *c, const MoeLayer *el,
                               const Buf *expert_table, int rows, int ntiles,
                               int run_mv, int run_mm) {
  int npairs = rows * NEMO_TOPK;
  if (run_mm) {
    uint32_t dim4 = NEMO_DIM / VEC4_WIDTH;
    gpu_select_kernel(c, K_moe_gather_bf16);
    gpu_bind_buffer(c, 0, &work.normed, 0);
    gpu_bind_buffer(c, 1, &work.ptok, 0);
    gpu_bind_buffer(c, 2, &work.x_bf16, 0);
    gpu_copy_bytes(c, 3, &dim4, sizeof dim4);
    gpu_run_kernel(
        c, (dim4 + DECODE_COMBINE_THREADS - 1u) / DECODE_COMBINE_THREADS,
        (uint32_t)npairs, 1, DECODE_COMBINE_THREADS, 1, 1);
  }
  MoeParam mp_up = {(uint32_t)NEMO_MOE_INTER, (uint32_t)NEMO_DIM,
                    (uint32_t)npairs, (uint32_t)ntiles};
  MoeParam mp_dn = {(uint32_t)NEMO_DIM, (uint32_t)NEMO_MOE_INTER,
                    (uint32_t)npairs, (uint32_t)ntiles};
  uint32_t gx_up_mv =
      ((uint32_t)NEMO_MOE_INTER + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG;
  uint32_t gx_up_mm =
      ((uint32_t)NEMO_MOE_INTER + MOE_MM_TILE - 1u) / MOE_MM_TILE;
  uint32_t gx_dn_mv =
      ((uint32_t)NEMO_DIM + MOE_MV_ROWS_PER_TG - 1u) / MOE_MV_ROWS_PER_TG;
  uint32_t gx_dn_mm = ((uint32_t)NEMO_DIM + MOE_MM_TILE - 1u) / MOE_MM_TILE;

  if (run_mv) {
    gpu_select_kernel(c, K_moe_up_mv);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &el->factor_up, 0);
    gpu_bind_buffer(c, 2, &work.normed, 0);
    gpu_bind_buffer(c, 3, &work.ptok, 0);
    gpu_bind_buffer(c, 4, &work.tiles, 0);
    gpu_bind_buffer(c, 5, &work.h_bf16, 0);
    gpu_copy_bytes(c, 6, &mp_up, sizeof mp_up);
    gpu_run_kernel(c, gx_up_mv, (uint32_t)ntiles, 1, MOE_MV_THREADS, 1, 1);
  }
  if (run_mm) {
    gpu_select_kernel(c, K_moe_up_mm);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &el->factor_up, 0);
    gpu_bind_buffer(c, 2, &work.x_bf16, 0);
    gpu_bind_buffer(c, 3, &work.tiles, 0);
    gpu_bind_buffer(c, 4, &work.h_bf16, 0);
    gpu_copy_bytes(c, 5, &mp_up, sizeof mp_up);
    gpu_run_kernel(c, gx_up_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
  }
  if (run_mv) {
    gpu_select_kernel(c, K_moe_dn_mv);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &el->factor_down, 0);
    gpu_bind_buffer(c, 2, &work.h_bf16, 0);
    gpu_bind_buffer(c, 3, &work.pw, 0);
    gpu_bind_buffer(c, 4, &work.tiles, 0);
    gpu_bind_buffer(c, 5, &work.y_pair, 0);
    gpu_copy_bytes(c, 6, &mp_dn, sizeof mp_dn);
    gpu_run_kernel(c, gx_dn_mv, (uint32_t)ntiles, 1, MOE_MV_THREADS, 1, 1);
  }
  if (run_mm) {
    gpu_select_kernel(c, K_moe_dn_mm);
    gpu_bind_buffer(c, 0, expert_table, 0);
    gpu_bind_buffer(c, 1, &el->factor_down, 0);
    gpu_bind_buffer(c, 2, &work.h_bf16, 0);
    gpu_bind_buffer(c, 3, &work.pw, 0);
    gpu_bind_buffer(c, 4, &work.tiles, 0);
    gpu_bind_buffer(c, 5, &work.y_pair, 0);
    gpu_copy_bytes(c, 6, &mp_dn, sizeof mp_dn);
    gpu_run_kernel(c, gx_dn_mm, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);
  }

  MoeCombineParam mcp = {(uint32_t)rows, (uint32_t)NEMO_DIM,
                         (uint32_t)NEMO_TOPK};
  gpu_select_kernel(c, K_moe_combine_add);
  gpu_bind_buffer(c, 0, &work.y_pair, 0);
  gpu_bind_buffer(c, 1, &work.tp, 0);
  gpu_bind_buffer(c, 2, &work.shared_out, 0);
  gpu_bind_buffer(c, 3, &work.cur, 0);
  gpu_copy_bytes(c, 4, &mcp, sizeof mcp);
  gpu_run_kernel(
      c, ((uint32_t)NEMO_DIM + MOE_COMBINE_THREADS - 1u) / MOE_COMBINE_THREADS,
      (uint32_t)rows, 1, MOE_COMBINE_THREADS, 1, 1);
}

// ---- Forward pass ----

// Returns the encoded BF16 audio feature row for prompt position `pos`.
static const uint16_t *find_media_row(int pos) {
  for (int i = 0; i < prefill_media_count; i++) {
    const Media *im = &prefill_media[i];
    int j = pos - im->start;
    if (j < 0 || j >= im->ntokens) continue;
    if (!im->rows)
      die("nemotron: audio at position %d was not encoded", im->start);
    return im->rows + (size_t)j * NEMO_DIM;
  }
  die("nemotron: negative media token at position %d outside registered audio",
      pos);
}

// Unpacks encoded BF16 audio feature rows into work.cur for any negative
// placeholder tokens in [pos0, pos0 + rows).
static void inject_media_rows(const int *tokens, int pos0, int rows) {
  float *cur = (float *)work.cur.ptr;
  for (int r = 0; r < rows; r++) {
    if (tokens[r] >= 0) continue;
    const uint16_t *src = find_media_row(pos0 + r);
    float *dst = cur + (size_t)r * NEMO_DIM;
    for (int d = 0; d < NEMO_DIM; d++) {
      uint32_t bits = (uint32_t)src[d] << BF16_SHIFT;
      memcpy(&dst[d], &bits, sizeof(float));
    }
  }
}

// Runs one Mamba-2 layer (m_idx in 0..22) and adds its output to work.cur.
// When capture_verify is nonzero, stores in_proj output directly into
// work.verify_zxbcdt so accept() can rewind Mamba-2 state without rerunning
// linear projections.
static void run_mamba_layer(GpuCommandBuffer *cmd, int m_idx, int rows,
                            const NemoState *s, int batched,
                            int capture_verify) {
  const MambaLayer *ml = &mamba_layers[m_idx];
  const Buf *zbuf = capture_verify ? &work.verify_zxbcdt : &work.mamba_zxbcdt;
  size_t zoff = capture_verify ? (size_t)m_idx * (size_t)rows *
                                     NEMO_MAMBA_IN_DIM * sizeof(float)
                               : 0;
  run_fp8_linear(cmd, &ml->in_proj, &work.normed, NULL, zbuf, zoff, rows, 0);

  MambaParam mp = {(uint32_t)rows, RMS_EPS};
  uint32_t conv_gx =
      ((uint32_t)NEMO_MAMBA_CONV_DIM + CONV_THREADS - 1u) / CONV_THREADS;
  uint32_t ssm_gx = (uint32_t)NEMO_MAMBA_HEAD_DIM / SSM_P_PER_TG;

  if (batched) {
    size_t tgt_off = (size_t)m_idx * NEMO_MAX_BATCH_ROWS * sizeof(MambaTarget);
    gpu_select_kernel(cmd, K_mamba_conv_decode);
    gpu_bind_buffer(cmd, 0, zbuf, zoff);
    gpu_bind_buffer(cmd, 1, &work.mamba_tgt, tgt_off);
    gpu_bind_buffer(cmd, 2, &ml->conv_w, 0);
    gpu_bind_buffer(cmd, 3, &ml->conv_b, 0);
    gpu_bind_buffer(cmd, 4, &work.mamba_xbc, 0);
    gpu_copy_bytes(cmd, 5, &mp, sizeof mp);
    gpu_run_kernel(cmd, conv_gx, (uint32_t)rows, 1, CONV_THREADS, 1, 1);

    gpu_select_kernel(cmd, K_mamba_ssm_decode);
    gpu_bind_buffer(cmd, 0, zbuf, zoff);
    gpu_bind_buffer(cmd, 1, &work.mamba_xbc, 0);
    gpu_bind_buffer(cmd, 2, &work.mamba_tgt, tgt_off);
    gpu_bind_buffer(cmd, 3, &ml->a_exp, 0);
    gpu_bind_buffer(cmd, 4, &ml->d_skip, 0);
    gpu_bind_buffer(cmd, 5, &ml->dt_bias, 0);
    gpu_bind_buffer(cmd, 6, &work.mamba_y, 0);
    gpu_copy_bytes(cmd, 7, &mp, sizeof mp);
    gpu_run_kernel(cmd, ssm_gx, (uint32_t)NEMO_MAMBA_HEADS, (uint32_t)rows,
                   SSM_THREADS, 1, 1);
  } else {
    size_t conv_off = (size_t)m_idx * CONV_LAYER_BYTES;
    size_t ssm_off = (size_t)m_idx * SSM_LAYER_BYTES;

    gpu_select_kernel(cmd, K_mamba_conv_prefill);
    gpu_bind_buffer(cmd, 0, zbuf, zoff);
    gpu_bind_buffer(cmd, 1, &s->conv_state, conv_off);
    gpu_bind_buffer(cmd, 2, &ml->conv_w, 0);
    gpu_bind_buffer(cmd, 3, &ml->conv_b, 0);
    gpu_bind_buffer(cmd, 4, &work.mamba_xbc, 0);
    gpu_copy_bytes(cmd, 5, &mp, sizeof mp);
    gpu_run_kernel(cmd, conv_gx, (uint32_t)rows, 1, CONV_THREADS, 1, 1);

    gpu_select_kernel(cmd, K_mamba_conv_save);
    gpu_bind_buffer(cmd, 0, zbuf, zoff);
    gpu_bind_buffer(cmd, 1, &s->conv_state, conv_off);
    gpu_copy_bytes(cmd, 2, &mp, sizeof mp);
    gpu_run_kernel(cmd, conv_gx, 1, 1, CONV_THREADS, 1, 1);

    gpu_select_kernel(cmd, K_mamba_ssm_prefill);
    gpu_bind_buffer(cmd, 0, zbuf, zoff);
    gpu_bind_buffer(cmd, 1, &work.mamba_xbc, 0);
    gpu_bind_buffer(cmd, 2, &s->ssm_state, ssm_off);
    gpu_bind_buffer(cmd, 3, &ml->a_exp, 0);
    gpu_bind_buffer(cmd, 4, &ml->d_skip, 0);
    gpu_bind_buffer(cmd, 5, &ml->dt_bias, 0);
    gpu_bind_buffer(cmd, 6, &work.mamba_y, 0);
    gpu_copy_bytes(cmd, 7, &mp, sizeof mp);
    gpu_run_kernel(cmd, ssm_gx, (uint32_t)NEMO_MAMBA_HEADS, 1, SSM_THREADS, 1,
                   1);
  }

  gpu_select_kernel(cmd, K_mamba_norm_gated);
  gpu_bind_buffer(cmd, 0, &work.mamba_y, 0);
  gpu_bind_buffer(cmd, 1, zbuf, zoff);
  gpu_bind_buffer(cmd, 2, &ml->norm_w, 0);
  gpu_bind_buffer(cmd, 3, &work.mamba_normed, 0);
  gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(cmd, (uint32_t)NEMO_MAMBA_GROUPS, (uint32_t)rows, 1,
                 ROW_THREADS, 1, 1);

  run_fp8_linear(cmd, &ml->out_proj, &work.mamba_normed, &work.cur, &work.cur,
                 0, rows, 0);
}

// Runs one NoPE GQA 32/2 attention layer (a_idx in 0..5) and adds its output
// to work.cur.
static void run_attn_layer(GpuCommandBuffer *cmd, int a_idx, int rows, int pos0,
                           int max_pos, const NemoState *s, int batched) {
  const AttnLayer *al = &attn_layers[a_idx];
  int use_tiled = (!batched && rows > NEMO_MAX_BATCH_ROWS);
  int prepacked = 0;
  if (rows > MOE_MV_MAX_M) {
    run_pack_bf16(cmd, &work.normed, rows, NEMO_DIM);
    prepacked = 1;
  }
  run_nvfp4_linear(cmd, &al->q, &work.normed, NULL, &work.q_raw, rows, ACT_NONE,
                   prepacked);
  run_nvfp4_linear(cmd, &al->k, &work.normed, NULL, &work.k_raw, rows, ACT_NONE,
                   prepacked);
  run_nvfp4_linear(cmd, &al->v, &work.normed, NULL, &work.v_raw, rows, ACT_NONE,
                   prepacked);

  size_t kv_tgt_off = (size_t)a_idx * NEMO_MAX_BATCH_ROWS * sizeof(KvTarget);
  int q_stride = use_tiled ? align_up(rows, PREFILL_Q_PAD) : rows;
  QkvParam qp = {(uint32_t)rows, (uint32_t)q_stride, (uint32_t)pos0,
                 (uint32_t)batched};
  gpu_select_kernel(cmd, K_qkv_store);
  gpu_bind_buffer(cmd, 0, &work.q_raw, 0);
  gpu_bind_buffer(cmd, 1, &work.k_raw, 0);
  gpu_bind_buffer(cmd, 2, &work.v_raw, 0);
  gpu_bind_buffer(cmd, 3, &work.q_hm, 0);
  gpu_bind_buffer(cmd, 4, &s->k[a_idx][0], 0);
  gpu_bind_buffer(cmd, 5, &s->k[a_idx][1], 0);
  gpu_bind_buffer(cmd, 6, &s->v[a_idx][0], 0);
  gpu_bind_buffer(cmd, 7, &s->v[a_idx][1], 0);
  gpu_bind_buffer(cmd, 8, &work.kv_tgt, kv_tgt_off);
  gpu_copy_bytes(cmd, 9, &qp, sizeof qp);
  gpu_run_kernel(cmd, QKV_SLOTS, (uint32_t)q_stride, 1, QKV_THREADS, 1, 1);

  if (use_tiled) {
    AttnPrefillParam app = {(uint32_t)rows, (uint32_t)q_stride, (uint32_t)pos0,
                            (uint32_t)s->cap};
    gpu_select_kernel(cmd, K_attend_prefill);
    gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
    gpu_bind_buffer(cmd, 1, &s->k[a_idx][0], 0);
    gpu_bind_buffer(cmd, 2, &s->k[a_idx][1], 0);
    gpu_bind_buffer(cmd, 3, &s->v[a_idx][0], 0);
    gpu_bind_buffer(cmd, 4, &s->v[a_idx][1], 0);
    gpu_bind_buffer(cmd, 5, &work.attn_out, 0);
    gpu_copy_bytes(cmd, 6, &app, sizeof app);
    uint32_t q_tiles = ((uint32_t)rows + PREFILL_Q_PAD - 1u) / PREFILL_Q_PAD;
    gpu_run_kernel(cmd, q_tiles, (uint32_t)NEMO_HEADS, 1, PREFILL_THREADS, 1,
                   1);
  } else {
    int total_k = max_pos + 1;
    int nparts = (total_k + DECODE_PART_TARGET - 1) / DECODE_PART_TARGET;
    if (nparts < 1) nparts = 1;
    if (nparts > MAX_DECODE_PARTS) nparts = MAX_DECODE_PARTS;
    AttnDecodeParam adp = {
        (uint32_t)rows,   (uint32_t)q_stride,         (uint32_t)pos0,
        (uint32_t)nparts, (uint32_t)MAX_DECODE_PARTS, (uint32_t)batched};
    gpu_select_kernel(cmd, K_attend_part);
    gpu_bind_buffer(cmd, 0, &work.q_hm, 0);
    gpu_bind_buffer(cmd, 1, &s->k[a_idx][0], 0);
    gpu_bind_buffer(cmd, 2, &s->k[a_idx][1], 0);
    gpu_bind_buffer(cmd, 3, &s->v[a_idx][0], 0);
    gpu_bind_buffer(cmd, 4, &s->v[a_idx][1], 0);
    gpu_bind_buffer(cmd, 5, &work.kv_tgt, kv_tgt_off);
    gpu_bind_buffer(cmd, 6, &work.attn_part, 0);
    gpu_copy_bytes(cmd, 7, &adp, sizeof adp);
    gpu_run_kernel(cmd, (uint32_t)NEMO_KV_HEADS, (uint32_t)nparts,
                   (uint32_t)rows, DECODE_PART_THREADS, 1, 1);

    gpu_select_kernel(cmd, K_attend_combine);
    gpu_bind_buffer(cmd, 0, &work.attn_part, 0);
    gpu_bind_buffer(cmd, 1, &work.attn_out, 0);
    gpu_copy_bytes(cmd, 2, &adp, sizeof adp);
    gpu_run_kernel(cmd, (uint32_t)NEMO_HEADS, (uint32_t)rows, 1,
                   DECODE_COMBINE_THREADS, 1, 1);
  }

  run_nvfp4_linear(cmd, &al->o, &work.attn_out, &work.cur, &work.cur, rows,
                   ACT_NONE, 0);
}

// Runs one MoE layer (m_idx in 0..22): shared expert + top-6 router + routed
// NVFP4 experts (via ExpertStore resident fast path or SSD streaming) +
// residual add into work.cur.
static void run_moe_layer(GpuCommandBuffer *cmd, int m_idx, int rows,
                          int is_prefill, const PassWait *wait,
                          uint8_t fast_moe[NEMO_MOE_LAYERS]) {
  const MoeLayer *el = &moe_layers[m_idx];
  if (is_prefill) expert_store_set_prompt_layer(experts_store, m_idx);

  // Shared expert runs on GPU while router computes top-6 selections.
  run_nvfp4_linear(cmd, &el->shared_up, &work.normed, NULL, &work.shared_h,
                   rows, ACT_RELU2, 0);
  run_nvfp4_linear(cmd, &el->shared_down, &work.shared_h, NULL,
                   &work.shared_out, rows, ACT_NONE, 0);

  size_t route_ids_off =
      is_prefill ? 0
                 : (size_t)m_idx * (size_t)rows * NEMO_TOPK * sizeof(uint32_t);
  run_bf16_linear(cmd, &el->router_w, &work.normed, 0, &work.route_logits, rows,
                  NEMO_EXPERTS, NEMO_DIM, 0);
  RouteParam rp = {(uint32_t)rows, (uint32_t)NEMO_EXPERTS, (uint32_t)NEMO_TOPK,
                   ROUTE_SCALE};
  gpu_select_kernel(cmd, K_route_topk);
  gpu_bind_buffer(cmd, 0, &work.route_logits, 0);
  gpu_bind_buffer(cmd, 1, &el->router_bias, 0);
  gpu_bind_buffer(cmd, 2, &work.route_ids, route_ids_off);
  gpu_bind_buffer(cmd, 3, &work.route_w, 0);
  gpu_copy_bytes(cmd, 4, &rp, sizeof rp);
  gpu_run_kernel(cmd, (uint32_t)rows, 1, 1, ROUTE_THREADS, 1, 1);

  if (is_prefill) {
    model_commit(cmd);
    model_sync();
    held_release_all();
    expert_store_resize_in_prompt(experts_store, m_idx);
    if (wait && wait->ready && !wait->ready(wait->context)) pass_wait(wait);
    int ntiles = nemotron_group_expert_tiles(
        (const uint32_t *)work.route_ids.ptr, (const float *)work.route_w.ptr,
        rows, MOE_MM_TILE, (int *)work.ptok.ptr, (float *)work.pw.ptr,
        (int *)work.tp.ptr, (NemoMoeTile *)work.tiles.ptr);
    int has_mv = 0, has_mm = 0, need[NEMO_EXPERTS], nn = 0;
    uint8_t used[NEMO_EXPERTS] = {0};
    const NemoMoeTile *tl = (const NemoMoeTile *)work.tiles.ptr;
    for (int t = 0; t < ntiles; t++) {
      used[tl[t].expert] = 1;
      if (tl[t].count <= (uint32_t)MOE_MV_MAX_M)
        has_mv = 1;
      else
        has_mm = 1;
    }
    for (int e = 0; e < NEMO_EXPERTS; e++)
      if (used[e]) need[nn++] = e;
    hold_experts(m_idx, need, nn);
    expert_store_cancel_read_aheads(experts_store, m_idx, need, nn);
    expert_store_unprotect_read_aheads(experts_store, m_idx);
    if (rows >= PREFILL_BULK_ROWS && m_idx + 1 < NEMO_MOE_LAYERS)
      prefetch_all_layer_experts(m_idx + 1);
    expert_store_wait_for_reads(experts_store, m_idx, need, nn);
    *cmd = gpu_begin(gpu);
    fill_streaming_table(cmd, m_idx, need, nn);
    encode_moe_experts(cmd, el, &work.expert_table, rows, ntiles, has_mv,
                       has_mm);
  } else {
    uint64_t gen = 0;
    if (expert_store_layer_is_reachable(experts_store, m_idx, &gen)) {
      hold_layer_if_needed(m_idx);
      refresh_layer_table(m_idx, gen);
      uint32_t m_topk[2] = {(uint32_t)rows, (uint32_t)NEMO_TOPK};
      gpu_select_kernel(cmd, K_moe_group);
      gpu_bind_buffer(cmd, 0, &work.route_ids, route_ids_off);
      gpu_bind_buffer(cmd, 1, &work.route_w, 0);
      gpu_bind_buffer(cmd, 2, &work.ptok, 0);
      gpu_bind_buffer(cmd, 3, &work.pw, 0);
      gpu_bind_buffer(cmd, 4, &work.tp, 0);
      gpu_bind_buffer(cmd, 5, &work.tiles, 0);
      gpu_copy_bytes(cmd, 6, m_topk, sizeof m_topk);
      gpu_run_kernel(cmd, 1, 1, 1, ROUTE_THREADS, 1, 1);
      int ntiles = rows * NEMO_TOPK;
      encode_moe_experts(cmd, el, &layer_table[m_idx], rows, ntiles, 1,
                         rows > MOE_MV_MAX_M ? 1 : 0);
      fast_moe[m_idx] = 1;
    } else {
      model_commit(cmd);
      model_sync();
      flush_fast_layers(fast_moe, rows);
      held_release_all();

      const uint32_t *rids = (const uint32_t *)work.route_ids.ptr +
                             (size_t)m_idx * (size_t)rows * NEMO_TOPK;
      ExpertStoreCounts *st = expert_stats();
      uint8_t used[NEMO_EXPERTS] = {0};
      int need[NEMO_EXPERTS], nn = 0;
      for (int r = 0; r < rows; r++) {
        for (int k = 0; k < NEMO_TOPK; k++) {
          uint32_t e = rids[(size_t)r * NEMO_TOPK + (size_t)k];
          if (e >= (uint32_t)NEMO_EXPERTS)
            die("nemotron: router selected expert %u", e);
          used[e] = 1;
          ExpertPlace place = expert_store_locate(experts_store, m_idx, (int)e);
          if (place == EXPERT_IN_MEMORY)
            st->row_own++;
          else if (place == EXPERT_READ_AHEAD)
            st->row_ahead++;
          else if (place == EXPERT_BEING_READ_AHEAD)
            st->row_late++;
          else
            st->row_ssd++;
          if (r < ROUTE_DECODE_CAP) {
            route_id[r][m_idx][k] = (uint16_t)e;
            route_from[r][m_idx][k] = place_to_source(place);
            route_step[r][m_idx] = route_steps;
          }
        }
      }
      for (int e = 0; e < NEMO_EXPERTS; e++)
        if (used[e]) need[nn++] = e;
      st->row_unique += (uint64_t)nn;

      hold_experts(m_idx, need, nn);
      expert_store_cancel_read_aheads(experts_store, m_idx, need, nn);
      expert_store_unprotect_read_aheads(experts_store, m_idx);
      credit_near_misses(m_idx, rows, used);
      expert_store_wait_for_reads(experts_store, m_idx, need, nn);

      int ntiles = nemotron_group_expert_tiles(
          rids, (const float *)work.route_w.ptr, rows, MOE_MM_TILE,
          (int *)work.ptok.ptr, (float *)work.pw.ptr, (int *)work.tp.ptr,
          (NemoMoeTile *)work.tiles.ptr);
      int has_mv = 0, has_mm = 0;
      const NemoMoeTile *tl = (const NemoMoeTile *)work.tiles.ptr;
      for (int t = 0; t < ntiles; t++) {
        if (tl[t].count <= (uint32_t)MOE_MV_MAX_M)
          has_mv = 1;
        else
          has_mm = 1;
      }
      *cmd = gpu_begin(gpu);
      fill_streaming_table(cmd, m_idx, need, nn);
      encode_moe_experts(cmd, el, &work.expert_table, rows, ntiles, has_mv,
                         has_mm);
    }
  }
}

// Runs the complete 52-layer Nemotron-H forward pass over `rows` tokens.
// Computes logits for all rows (logits_mode == 2), last row only (1), or skips
// lm_head (0). When capture_verify is nonzero, captures Mamba-2 in_proj outputs
// into work.verify_zxbcdt for exact partial-acceptance rollback. Returns 1 on
// completion, or 0 if shutdown stopped a prefill.
static int run_forward(NemoState *const *states, const int *tokens,
                       const int *pos, int rows, int batched, int is_prefill,
                       int logits_mode, int capture_verify,
                       const PassWait *wait) {
  int need_logits = logits_mode == 2 ? rows : (logits_mode == 1 ? 1 : 0);
  ensure_work(rows, need_logits);
  if (capture_verify) ensure_verify_work(rows);
  memcpy(work.ids.ptr, tokens, (size_t)rows * sizeof(int32_t));

  int pos0 = pos[0], max_pos = pos[0];
  if (batched) {
    for (int r = 0; r < rows; r++) {
      state_ensure_cap(states[r], pos[r] + 1);
      if (pos[r] > max_pos) max_pos = pos[r];
    }
    MambaTarget *mt = (MambaTarget *)work.mamba_tgt.ptr;
    for (int m = 0; m < NEMO_MAMBA_LAYERS; m++) {
      for (int r = 0; r < rows; r++) {
        const NemoState *sr = states[r];
        mt[(size_t)m * NEMO_MAX_BATCH_ROWS + r] =
            (MambaTarget){sr->conv_state.address + sr->conv_state.off +
                              (size_t)m * CONV_LAYER_BYTES,
                          sr->ssm_state.address + sr->ssm_state.off +
                              (size_t)m * SSM_LAYER_BYTES};
      }
    }
    KvTarget *kt = (KvTarget *)work.kv_tgt.ptr;
    for (int a = 0; a < NEMO_ATTN_LAYERS; a++) {
      for (int r = 0; r < rows; r++) {
        const NemoState *sr = states[r];
        kt[(size_t)a * NEMO_MAX_BATCH_ROWS + r] =
            (KvTarget){sr->k[a][0].address + sr->k[a][0].off,
                       sr->k[a][1].address + sr->k[a][1].off,
                       sr->v[a][0].address + sr->v[a][0].off,
                       sr->v[a][1].address + sr->v[a][1].off,
                       (uint32_t)pos[r],
                       0u};
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

  uint8_t fast_moe[NEMO_MOE_LAYERS] = {0};
  int stopped = 0;
  GpuCommandBuffer cmd = gpu_begin(gpu);

  NormParam ep = {(uint32_t)NEMO_DIM, 0u, RMS_EPS};
  gpu_select_kernel(&cmd, K_embed);
  gpu_bind_buffer(&cmd, 0, &embed_w, 0);
  gpu_bind_buffer(&cmd, 1, &work.ids, 0);
  gpu_bind_buffer(&cmd, 2, &work.cur, 0);
  gpu_copy_bytes(&cmd, 3, &ep, sizeof ep);
  gpu_run_kernel(&cmd, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);

  for (int l = 0; l < NEMO_LAYERS; l++) {
    if (is_prefill) {
      prefill_layer_now = l + 1;
      if (shutdown_requested()) {
        stopped = 1;
        break;
      }
    }
    const LayerDesc *ld = &layer_desc[l];
    run_rmsnorm(&cmd, &work.cur, 0, &ld->norm_w, &work.normed, rows);
    if (ld->kind == LAYER_MAMBA) {
      run_mamba_layer(&cmd, ld->sub_idx, rows, states[0], batched,
                      capture_verify);
    } else if (ld->kind == LAYER_ATTN) {
      run_attn_layer(&cmd, ld->sub_idx, rows, pos0, max_pos, states[0],
                     batched);
    } else {
      run_moe_layer(&cmd, ld->sub_idx, rows, is_prefill, wait, fast_moe);
    }
  }

  if (!stopped && logits_mode > 0) {
    if (logits_mode == 1) {
      size_t last_off = (size_t)(rows - 1) * NEMO_DIM * sizeof(float);
      run_rmsnorm(&cmd, &work.cur, last_off, &final_norm_w, &work.normed, 1);
      run_bf16_linear(&cmd, &lm_head_w, &work.normed, 0, &work.logits, 1,
                      NEMO_VOCAB, NEMO_DIM, 0);
    } else {
      run_rmsnorm(&cmd, &work.cur, 0, &final_norm_w, &work.normed, rows);
      run_bf16_linear(&cmd, &lm_head_w, &work.normed, 0, &work.logits, rows,
                      NEMO_VOCAB, NEMO_DIM, 0);
    }
  }

  model_commit(&cmd);
  model_sync();
  if (!is_prefill) flush_fast_layers(fast_moe, rows);
  held_release_all();
  if (is_prefill) expert_store_set_prompt_layer(experts_store, -1);

  if (!stopped) {
    if (batched) {
      for (int r = 0; r < rows; r++) states[r]->pos = pos[r] + 1;
    } else {
      states[0]->pos = pos0 + rows;
    }
  }
  return !stopped;
}

// ---- Audio media preparation and encoding ----

// Computes a 64-bit FNV-1a hash of encoded audio bytes for prefix-cache keys.
static uint64_t fnv1a_bytes(const void *data, size_t len) {
  const uint8_t *p = data;
  uint64_t h = FNV_OFFSET;
  for (size_t i = 0; i < len; i++) {
    h ^= p[i];
    h *= FNV_PRIME;
  }
  return h;
}

// Decodes an encoded audio file (WAV, MP3, M4A, FLAC, OGG, etc.) to 16 kHz mono
// PCM and populates `im` for prompt expansion. Returns 1 on success or 0 with
// `*error` set.
static int nemo_prepare_media(const char *dir, Media *im, const void *data,
                              size_t len, const char **error) {
  (void)dir;
  *im = (Media){0};
  if (!data || len == 0) {
    *error = "empty audio input";
    return 0;
  }
  int n_samples = 0;
  float *pcm =
      audio_decode_pcm_mono(data, len, NEMO_AUDIO_SAMPLE_RATE,
                            AUDIO_MAX_DECODE_SECONDS, &n_samples, error);
  if (!pcm) return 0;
  int ntokens = nemo_audio_token_count(n_samples);
  if (ntokens <= 0) {
    free(pcm);
    *error = "audio clip is too short to produce any tokens";
    return 0;
  }
  im->ntokens = ntokens;
  im->pcm = pcm;
  im->pcm_samples = n_samples;
  im->hash = fnv1a_bytes(data, len);
  return 1;
}

// Runs the Parallel Expert Perception audio encoder on `im->pcm` and stores
// `im->ntokens` BF16 feature rows of width 2688 in `im->rows`. Returns 1 on
// success or 0 when memory does not fit or perception is unavailable.
static int nemo_encode_media(Media *im) {
  if (!has_audio || !im || !im->pcm || im->pcm_samples <= 0) return 0;
  if (!audio_enc) {
    audio_enc = nemo_audio_open(gpu, work_slot->account, work_slot, model_dir);
    if (!audio_enc) return 0;
  }
  nemo_audio_pin(audio_enc, 1);
  float *emb = NULL;
  int ntokens = 0;
  char err[256] = {0};
  int ok = nemo_audio_encode(audio_enc, im->pcm, im->pcm_samples, NULL, &emb,
                             &ntokens, err, sizeof err);
  nemo_audio_pin(audio_enc, 0);
  if (!ok || !emb || ntokens != im->ntokens) {
    free(emb);
    return 0;
  }
  size_t count = (size_t)ntokens * NEMO_DIM;
  uint16_t *rows = memory_host_alloc(work_slot, count * sizeof(uint16_t));
  if (!rows) {
    free(emb);
    return 0;
  }
  for (size_t i = 0; i < count; i++) rows[i] = f32_to_bf16(emb[i]);
  free(emb);
  free(im->pcm);
  im->pcm = NULL;
  im->pcm_samples = 0;
  im->rows = rows;
  im->memory = work_slot;
  return 1;
}

// Records encoded audio media items whose spans appear in subsequent prefills.
static void nemo_set_prefill_media(const Media *im, int n) {
  prefill_media = im;
  prefill_media_count = n;
}

// ---- ModelOps implementation ----

// Loads the Nemotron-3.5 / Nemotron-H checkpoint into GPU memory and starts
// its routed expert store. Returns 1 on success or 0 on memory refusal/stop.
static int nemo_load(Gpu *g, MemorySlot *work_mem, const char *dir,
                     const Options *o) {
  gpu = g;
  work_slot = work_mem;
  opt = *o;
  if (opt.ctx <= 0) opt.ctx = DEFAULT_CTX;
  snprintf(model_dir, sizeof model_dir, "%s", dir);

  if (!state_slot_ready) {
    memory_slot_init(&state_slot, work_mem->account, MEMORY_STATE, NULL,
                     "nemotron states", NULL, NULL);
    state_slot.automatic = 1;
    state_slot_ready = 1;
  }

  if (!text_weights) {
    read_config(dir);

#define K(dst, name) ((dst) = gpu_find_kernel(gpu, name))
    K(K_mv_bf16, "mv_bf16");
    K(K_mvm_bf16, "mvm_bf16");
    K(K_mm_bf16, "mm_bf16");
    K(K_embed, "nemo_embed");
    K(K_rmsnorm, "nemo_rmsnorm");
    K(K_pack_bf16, "nemo_pack_bf16");
    K(K_fp8_mv, "nemo_fp8_mv");
    K(K_fp8_mm, "nemo_fp8_mm");
    K(K_nvfp4_mv, "nemo_nvfp4_mv");
    K(K_nvfp4_mm, "nemo_nvfp4_mm");
    K(K_mamba_conv_prefill, "nemo_mamba_conv_prefill");
    K(K_mamba_conv_save, "nemo_mamba_conv_save");
    K(K_mamba_conv_decode, "nemo_mamba_conv_decode");
    K(K_mamba_ssm_prefill, "nemo_mamba_ssm_prefill");
    K(K_mamba_ssm_decode, "nemo_mamba_ssm_decode");
    K(K_mamba_norm_gated, "nemo_mamba_norm_gated");
    K(K_qkv_store, "nemo_qkv_store");
    K(K_attend_prefill, "nemo_attend_prefill");
    K(K_attend_part, "nemo_attend_part");
    K(K_attend_combine, "nemo_attend_combine");
    K(K_route_topk, "nemo_route_topk");
    K(K_moe_group, "nemo_moe_group");
    K(K_moe_gather_bf16, "nemo_moe_gather_bf16");
    K(K_moe_up_mv, "nemo_moe_nvfp4_up_mv");
    K(K_moe_dn_mv, "nemo_moe_nvfp4_down_mv");
    K(K_moe_up_mm, "nemo_moe_nvfp4_up_mm");
    K(K_moe_dn_mm, "nemo_moe_nvfp4_down_mm");
    K(K_moe_combine_add, "nemo_moe_combine_add");
#undef K

    checkpoint = tensor_index_open(dir);
    text_weights =
        weight_group_new(gpu, work_mem->account, "Nemotron-H weights");
    register_weights();

    ExpertLayout layout = {.layers = NEMO_MOE_LAYERS,
                           .scored_layers = NEMO_MOE_LAYERS,
                           .experts = NEMO_EXPERTS,
                           .chosen = NEMO_TOPK,
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
  if (work.cap_rows > NEMO_MAX_BATCH_ROWS) {
    model_sync();
    free_work(&work);
  }
  expert_store_set_pass_rows(experts_store, 1);
  uint64_t fixed = work_slot->payload - work_resizable_bytes();
  uint64_t minimum = weight_group_required_bytes(text_weights) + fixed +
                     estimate_work_bytes(NEMO_MAX_BATCH_ROWS) +
                     expert_store_minimum_bytes(experts_store, NEMO_MOE_LAYERS);
  if (!memory_wait_for_minimum(slot->account, minimum) ||
      !weight_group_load(text_weights)) {
    memory_pin(slot, 0);
    expert_store_set_request_running(experts_store, 0);
    return 0;
  }
  weight_group_set_resident(text_weights, 1);
  gpu_commit_residency(gpu);

  if (!store_started) {
    for (int m = 0; m < NEMO_MOE_LAYERS; m++) {
      layer_table[m] = work_alloc((size_t)NEMO_EXPERTS * sizeof(ExpertRef));
      layer_table_gen[m] = UINT64_MAX;
    }
    expert_store_start(experts_store, 0, NEMO_MOE_LAYERS);
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
                                                   NEMO_MOE_LAYERS);
  prepare_prev_rows = 0;
  prepare_prev_payload = 0;
  return 1;
}

// Returns the dense backbone weight slot.
static MemorySlot *nemo_dense_weight_slot(void) {
  return text_weights ? weight_group_slot(text_weights) : NULL;
}

// Marks whether a request is active so expert slots stay protected.
static void nemo_set_request_running(int running) {
  if (experts_store) expert_store_set_request_running(experts_store, running);
}

// Returns the bytes currently held by the routed expert store.
static uint64_t nemo_cache_bytes(void) {
  return experts_store ? expert_store_bytes(experts_store) : 0;
}

// Writes the routed expert IDs and source codes chosen for `row` in the latest
// decoding pass across all 23 MoE layers.
static int nemo_read_routes(int row, uint16_t *id, uint8_t *from) {
  if (row < 0 || row >= ROUTE_DECODE_CAP || !route_steps) return 0;
  for (int m = 0; m < NEMO_MOE_LAYERS; m++)
    if (route_step[row][m] != route_steps) return 0;
  memcpy(id, route_id[row], sizeof route_id[row]);
  memcpy(from, route_from[row], sizeof route_from[row]);
  return 1;
}

// Writes runtime counters and expert-store statistics into c.
static void nemo_read_counters(ModelCounters *c) {
  *c = (ModelCounters){
      .decode_rows = decode_rows_count,
      .prefill_rows = prefill_rows_count,
      .decode_seconds = decode_seconds_total,
      .decode_gpu_seconds = decode_gpu_seconds_total,
      .prefill_layer = prefill_layer_now,
      .prefill_rows_now = prefill_rows_now,
      .layers = NEMO_LAYERS,
      .experts_per_row = NEMO_MOE_LAYERS * NEMO_TOPK,
      .routed_layers = NEMO_MOE_LAYERS,
      .experts_per_layer = NEMO_EXPERTS,
      .dense_bytes =
          (text_weights ? weight_group_resident_bytes(text_weights) : 0) +
          (audio_enc && nemo_audio_is_loaded(audio_enc)
               ? nemo_audio_weight_bytes(audio_enc)
               : 0),
      .required_bytes = required_bytes,
      .decode = decode_stats,
      .prefill = prefill_stats,
      .experts = NEMO_MOE_LAYERS * NEMO_EXPERTS};
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

// Returns the output vocabulary size (131072).
static int nemo_vocab_size(void) { return NEMO_VOCAB; }

// Returns the maximum distinct states decoded in one batched pass.
static int nemo_max_rows(void) { return NEMO_MAX_BATCH_ROWS; }

// Returns the maximum verify rows (committed token + up to 15 proposals).
static int nemo_max_verify(void) { return NEMO_MAX_VERIFY; }

// Returns active drafts per pass (0 while speculative drafting is disabled).
static int nemo_drafts_per_pass(void) { return 0; }

// Returns safe rewind margin (0 for Mamba-2 SSM states).
static int nemo_rewind_margin(void) { return 0; }

// Returns whether `token` ends a model reply.
static int nemo_is_stop_token(int token) {
  for (int i = 0; i < n_eos_tokens; i++)
    if (eos_tokens[i] == token) return 1;
  return 0;
}

// Indicates that Nemotron-H resumes from snapshots rather than replay.
static int nemo_replay_rows(int k) {
  (void)k;
  return -1;
}

// Prefills positions `from` through `n - 1` of state `ms` in memory-admitted
// chunks, capturing a snapshot at `capture` if requested.
static const float *nemo_prefill(ModelState *ms, const int *tokens, int n,
                                 int from, int replay, int capture,
                                 ModelSnapshot **captured, float *every,
                                 const PassWait *wait) {
  (void)replay;
  if (from < 0 || from >= n)
    die("nemotron: invalid prefill range [%d, %d)", from, n);
  NemoState *s = (NemoState *)ms;
  if (from == 0) {
    memset(s->conv_state.ptr, 0, CONV_TOTAL_BYTES);
    memset(s->ssm_state.ptr, 0, SSM_TOTAL_BYTES);
    s->pos = 0;
  } else if (s->pos != from) {
    die("nemotron: prefill from %d does not match Mamba state position %d",
        from, s->pos);
  }

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
    int chunk = prepare_rows(rem, 1, 0);
    if (!chunk) {
      stopped = 1;
      break;
    }
    int is_last = (pos + chunk == n);
    int logits_mode = every ? 2 : (is_last ? 1 : 0);
    if (!run_forward(&s, tokens + pos, &pos, chunk, 0, 1, logits_mode, 0,
                     wait)) {
      stopped = 1;
      break;
    }
    if (every) {
      memcpy(every + (size_t)(pos - from) * NEMO_VOCAB, work.logits.ptr,
             (size_t)chunk * LOGITS_BYTES);
    }
    pos += chunk;
    prefill_rows_count += (uint64_t)chunk;
    if (capture >= 0 && pos == capture && captured)
      *captured = nemo_snapshot_take(ms, capture, NULL);
  }
  expert_store_age_and_resize(experts_store);
  stats_add(&prefill_stats, &c0);
  prefill_layer_now = 0;
  prefill_rows_now = 0;
  if (stopped) {
    if (captured && *captured) {
      nemo_snapshot_free(*captured);
      *captured = NULL;
    }
    return NULL;
  }
  return every ? every + (size_t)(n - from - 1) * NEMO_VOCAB
               : (const float *)work.logits.ptr;
}

// Decodes one token for each of `m` distinct states.
static const float *nemo_decode_rows(ModelState *const *states,
                                     const int *tokens, const int *pos, int m) {
  if (m < 1 || m > NEMO_MAX_BATCH_ROWS)
    die("nemotron: invalid decode batch size %d", m);
  if (!prepare_rows(m, 0, 0)) return NULL;
  ExpertStoreCounts c0 = *expert_stats();
  double t0 = now();
  GpuTimes g0 = gpu_times(gpu);
  route_steps++;
  NemoState *s_arr[NEMO_MAX_BATCH_ROWS];
  for (int i = 0; i < m; i++) {
    s_arr[i] = (NemoState *)states[i];
    if (s_arr[i]->pos != pos[i])
      die("nemotron: decode pos %d does not match Mamba state pos %d", pos[i],
          s_arr[i]->pos);
  }
  int batched = m > 1 ? 1 : 0;
  run_forward(s_arr, tokens, pos, m, batched, 0, m == 1 ? 1 : 2, 0, NULL);
  expert_store_age_and_resize(experts_store);
  stats_add(&decode_stats, &c0);
  GpuTimes g1 = gpu_times(gpu);
  decode_rows_count += (uint64_t)m;
  decode_seconds_total += now() - t0;
  decode_gpu_seconds_total += g1.busy_seconds - g0.busy_seconds;
  return (const float *)work.logits.ptr;
}

// Verifies `count` sequential tokens (1 committed token + up to 15 drafts) on
// state `ms` starting at `pos`, returning `count` rows of logits and saving
// pre-verify Mamba-2 state and zxbcdt projections for exact partial accept.
static const float *nemo_verify(ModelState *ms, const int *rows, int count,
                                int pos) {
  if (count < 1 || count > NEMO_MAX_VERIFY)
    die("nemotron: invalid verify length %d", count);
  NemoState *s = (NemoState *)ms;
  if (s->pos != pos)
    die("nemotron: verify pos %d does not match Mamba state pos %d", pos,
        s->pos);
  if (!prepare_rows(count, 0, 1)) return NULL;
  ensure_work(count, count);
  ensure_verify_work(count);
  model_sync();
  memcpy(work.verify_conv.ptr, s->conv_state.ptr, CONV_TOTAL_BYTES);
  memcpy(work.verify_ssm.ptr, s->ssm_state.ptr, SSM_TOTAL_BYTES);
  work.verify_state = s;
  work.verify_pos = pos;
  work.verify_count = count;

  ExpertStoreCounts c0 = *expert_stats();
  double t0 = now();
  GpuTimes g0 = gpu_times(gpu);
  route_steps++;
  run_forward(&s, rows, &pos, count, 0, 0, 2, 1, NULL);
  expert_store_age_and_resize(experts_store);
  stats_add(&decode_stats, &c0);
  GpuTimes g1 = gpu_times(gpu);
  decode_rows_count += (uint64_t)count;
  decode_seconds_total += now() - t0;
  decode_gpu_seconds_total += g1.busy_seconds - g0.busy_seconds;
  return (const float *)work.logits.ptr;
}

// Commits the first `n` verified tokens (1 <= n <= count) on state `ms`.
// Full acceptance needs no rollback; partial acceptance restores pre-verify
// Mamba-2 conv/SSM state and replays only the 23 lightweight conv1d + SSM scan
// updates over the saved FP32 zxbcdt prefix without rerunning any projections.
static void nemo_accept(ModelState *ms, const int *rows, int n, int pos) {
  (void)rows;
  NemoState *s = (NemoState *)ms;
  if (s != work.verify_state || pos != work.verify_pos || n < 1 ||
      n > work.verify_count)
    die("nemotron: invalid accept (n=%d, verify_count=%d, pos=%d, "
        "verify_pos=%d)",
        n, work.verify_count, pos, work.verify_pos);
  if (n < work.verify_count) {
    model_sync();
    memcpy(s->conv_state.ptr, work.verify_conv.ptr, CONV_TOTAL_BYTES);
    memcpy(s->ssm_state.ptr, work.verify_ssm.ptr, SSM_TOTAL_BYTES);

    MambaParam mp = {(uint32_t)n, RMS_EPS};
    uint32_t conv_gx =
        ((uint32_t)NEMO_MAMBA_CONV_DIM + CONV_THREADS - 1u) / CONV_THREADS;
    uint32_t ssm_gx = (uint32_t)NEMO_MAMBA_HEAD_DIM / SSM_P_PER_TG;

    GpuCommandBuffer cmd = gpu_begin(gpu);
    for (int m = 0; m < NEMO_MAMBA_LAYERS; m++) {
      const MambaLayer *ml = &mamba_layers[m];
      size_t zoff = (size_t)m * (size_t)work.verify_count * NEMO_MAMBA_IN_DIM *
                    sizeof(float);
      size_t conv_off = (size_t)m * CONV_LAYER_BYTES;
      size_t ssm_off = (size_t)m * SSM_LAYER_BYTES;

      gpu_select_kernel(&cmd, K_mamba_conv_prefill);
      gpu_bind_buffer(&cmd, 0, &work.verify_zxbcdt, zoff);
      gpu_bind_buffer(&cmd, 1, &s->conv_state, conv_off);
      gpu_bind_buffer(&cmd, 2, &ml->conv_w, 0);
      gpu_bind_buffer(&cmd, 3, &ml->conv_b, 0);
      gpu_bind_buffer(&cmd, 4, &work.mamba_xbc, 0);
      gpu_copy_bytes(&cmd, 5, &mp, sizeof mp);
      gpu_run_kernel(&cmd, conv_gx, (uint32_t)n, 1, CONV_THREADS, 1, 1);

      gpu_select_kernel(&cmd, K_mamba_conv_save);
      gpu_bind_buffer(&cmd, 0, &work.verify_zxbcdt, zoff);
      gpu_bind_buffer(&cmd, 1, &s->conv_state, conv_off);
      gpu_copy_bytes(&cmd, 2, &mp, sizeof mp);
      gpu_run_kernel(&cmd, conv_gx, 1, 1, CONV_THREADS, 1, 1);

      gpu_select_kernel(&cmd, K_mamba_ssm_prefill);
      gpu_bind_buffer(&cmd, 0, &work.verify_zxbcdt, zoff);
      gpu_bind_buffer(&cmd, 1, &work.mamba_xbc, 0);
      gpu_bind_buffer(&cmd, 2, &s->ssm_state, ssm_off);
      gpu_bind_buffer(&cmd, 3, &ml->a_exp, 0);
      gpu_bind_buffer(&cmd, 4, &ml->d_skip, 0);
      gpu_bind_buffer(&cmd, 5, &ml->dt_bias, 0);
      gpu_bind_buffer(&cmd, 6, &work.mamba_y, 0);
      gpu_copy_bytes(&cmd, 7, &mp, sizeof mp);
      gpu_run_kernel(&cmd, ssm_gx, (uint32_t)NEMO_MAMBA_HEADS, 1, SSM_THREADS,
                     1, 1);
    }
    model_commit(&cmd);
    model_sync();
    s->pos = pos + n;
  }
  work.verify_state = NULL;
  work.verify_count = 0;
}

// Resizes the expert cache while idle and returns remaining slot count.
static int nemo_run_idle_work(void) {
  if (!experts_store) return 0;
  expert_store_resize_when_idle(experts_store);
  return expert_store_slot_count(experts_store);
}

// Updates concurrent GPU pass count in the expert store.
static void nemo_set_other_passes(int passes) {
  if (experts_store) expert_store_set_other_passes(experts_store, passes);
}

// Formats one CLI user turn matching chat_template.jinja (nemotron3p5).
static void nemo_render_chat_turn(char *text, size_t cap, const char *prompt,
                                  int media_count, int think) {
  int off = snprintf(text, cap,
                     "<|im_start|>system\n<|im_end|>\n"
                     "<|im_start|>user\n%s",
                     prompt ? prompt : "");
  if (off < 0 || (size_t)off >= cap) return;
  for (int i = 0; i < media_count; i++) {
    int n = snprintf(text + off, cap - (size_t)off, " <|audio|>");
    if (n < 0 || (size_t)n >= cap - (size_t)off) return;
    off += n;
  }
  snprintf(text + off, cap - (size_t)off,
           think ? "<|im_end|>\n<|im_start|>assistant\n<think>\n"
                 : "<|im_end|>\n<|im_start|>assistant\n<think></think>");
}

// Creates a cooperative pass with its own workspace, submission sequence, media
// pointers, held experts and expert-store pass variables.
static Pass *nemo_pass_new(void (*body)(void *), void *argument) {
  const PassVariable own[] = {
      {&work, sizeof work},
      {&last_committed, sizeof last_committed},
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

// Initializes pass-local scratch handles when a pass begins.
static void nemo_pass_begin(void) {
  work = (NemoWork){0};
  last_committed = 0;
  prefill_media = NULL;
  prefill_media_count = 0;
  held.n = 0;
  held_layers = 0;
  if (experts_store) expert_store_set_prompt_layer(experts_store, -1);
}

// Syncs, releases held experts and frees pass-local buffers when a pass ends.
static void nemo_pass_end(void) {
  model_sync();
  held_release_all();
  free_work(&work);
}

// config.json model_type values served by this family.
static const char *const nemo_model_types[] = {"nemo_speechlm", "nemotron_h",
                                               NULL};

const ModelOps nemotron_ops = {.family = "Nemotron-H",
                               .model_types = nemo_model_types,
                               .media_token = "<|audio|>",
                               .max_drafts = 0,
                               .load = nemo_load,
                               .dense_weight_slot = nemo_dense_weight_slot,
                               .set_request_running = nemo_set_request_running,
                               .read_counters = nemo_read_counters,
                               .expert_store_bytes = nemo_cache_bytes,
                               .read_routes = nemo_read_routes,
                               .vocab_size = nemo_vocab_size,
                               .max_rows = nemo_max_rows,
                               .max_verify = nemo_max_verify,
                               .drafts_per_pass = nemo_drafts_per_pass,
                               .rewind_margin = nemo_rewind_margin,
                               .is_stop_token = nemo_is_stop_token,
                               .state_new = nemo_state_new,
                               .state_free = nemo_state_free,
                               .state_bytes = nemo_state_bytes,
                               .state_copy = nemo_state_copy,
                               .snapshot_take = nemo_snapshot_take,
                               .snapshot_restore = nemo_snapshot_restore,
                               .snapshot_free = nemo_snapshot_free,
                               .snapshot_bytes = nemo_snapshot_bytes,
                               .snapshot_logits = nemo_snapshot_logits,
                               .state_version = 1,
                               .write_state_regions = nemo_write_state_regions,
                               .write_snapshot_parts =
                                   nemo_write_snapshot_parts,
                               .snapshot_new_empty = nemo_snapshot_new_empty,
                               .prefill = nemo_prefill,
                               .replay_rows = nemo_replay_rows,
                               .decode_rows = nemo_decode_rows,
                               .verify = nemo_verify,
                               .accept = nemo_accept,
                               .run_idle_work = nemo_run_idle_work,
                               .prepare_media = nemo_prepare_media,
                               .encode_media = nemo_encode_media,
                               .set_prefill_media = nemo_set_prefill_media,
                               .render_chat_turn = nemo_render_chat_turn,
                               .pass_new = nemo_pass_new,
                               .pass_begin = nemo_pass_begin,
                               .pass_end = nemo_pass_end,
                               .set_other_passes = nemo_set_other_passes};
