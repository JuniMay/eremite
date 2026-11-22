#include "model/inkling2/mtp.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "engine/safetensors.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Inkling-2-Mini multi-token prediction (DSpark/DFlare + Markov RNN) weight
// loading, per-state context caches, target tap injection and block drafting.

// Numerical and tiling constants matching config.json mtp_config and
// src/kernels/inkling2_mtp.metal.
static constexpr float MTP_RMS_EPS = 1e-6f;
static constexpr float MTP_ROPE_THETA = 1000000.0f;
static constexpr float MTP_LOGIT_SOFTCAP = 30.0f;
static constexpr int MTP_MAX_PARTS = 64;
static constexpr int MTP_PART_TARGET = 128;
static constexpr int MTP_PART_FLOATS = INK2_HEAD_DIM + 2;
static constexpr size_t MTP_GLOBAL_KV_ROW_BYTES =
    (size_t)INK2_KV_DIM * sizeof(uint16_t);
static constexpr size_t MTP_ALIGN_READ_MIN_BYTES = 1u << 20;
static constexpr size_t MTP_ALIGN_MASK = 15u;
static constexpr size_t MTP_PATH_CAP = 1024;
static constexpr uint32_t MTP_BF16_SHIFT = 16;
static constexpr uint32_t MTP_ROW_THREADS = 256;
static constexpr uint32_t MTP_HEAD_THREADS = 64;
static constexpr uint32_t MTP_ATTEND_THREADS = 128;
static constexpr uint32_t MTP_COMBINE_THREADS = 32;
static constexpr uint32_t MTP_ELEM_THREADS = 256;
static constexpr uint32_t MTP_MV_THREADS = 256;
static constexpr uint32_t MTP_MM_THREADS = 128;
static constexpr uint32_t MTP_MM_TILE_M = 64;
static constexpr uint32_t MTP_MM_TILE_N = 64;
static constexpr uint32_t MTP_BF16_ROWS_PER_TG = 32;
static constexpr uint32_t MTP_BF16_MV_SGS = 8;
static constexpr uint32_t MTP_BF16_MV_ROWS_PER_SG = 4;
static constexpr uint32_t MTP_RNN_SPLIT_K = 2;
static constexpr uint32_t MTP_UNEMBED_PARTS =
    ((uint32_t)INK2_VOCAB + MTP_BF16_ROWS_PER_TG - 1u) / MTP_BF16_ROWS_PER_TG;

// Kernel parameter structs matching src/kernels/kernels.metal.
typedef struct {
  // Output rows, input columns, group row count, group column offset, BF16
  // rounding flag and split-K factor.
  uint32_t n, k, group_rows, group_k, round, split;
} MtpMvParam;

typedef struct {
  // Base matrix-vector parameters.
  MtpMvParam mv;
  // Batch rows, input row stride and output row stride in floats.
  uint32_t m, lda, ldy;
} MtpMvmParam;

typedef struct {
  // Active rows, output width, inner width, BF16 rounding flag and strides.
  uint32_t m, n, k, round, lda, ldy;
} MtpMmParam;

// Kernel parameter structs matching src/kernels/inkling2.metal.
typedef struct {
  // Row width in floats and BF16 rounding flag.
  uint32_t dim, round;
  // RMSNorm epsilon.
  float eps;
} MtpNormParam;

// Kernel parameter structs matching src/kernels/inkling2_mtp.metal.
typedef struct {
  // Device GPU address of the state's FP32 f_rnn ring buffer.
  uint64_t ring;
  // Token position of this row and alignment padding.
  uint32_t pos, pad;
} MtpFrnnTarget;

typedef struct {
  // Device GPU addresses of the state's BF16 K and V buffers.
  uint64_t k, v;
  // Token position and ring capacity (0 for global linear KV).
  uint32_t pos, ring_rows;
} MtpKvTarget;

typedef struct {
  // Precomputed softmax weights over the 4 target layer taps.
  float w0, w1, w2, w3;
  // Active rows, tap buffer row stride, hidden width, row-0 pos, ring size and
  // batch flag.
  uint32_t rows, tap_stride, dim, pos0, frnn_ring, batched;
  // RMSNorm epsilon.
  float eps;
} MtpFuseParam;

typedef struct {
  // Active rows, row-0 pos, ring capacity (0 for global) and batch flag.
  uint32_t rows, pos0, ring_rows, batched;
  // NeoX RoPE base frequency and per-head RMSNorm epsilon.
  float rope_theta, eps;
} MtpCtxKvParam;

typedef struct {
  // Anchor token ID and hidden width.
  uint32_t anchor, dim;
  // Embedding RMSNorm epsilon.
  float eps;
} MtpBlockInitParam;

typedef struct {
  // Starting position of the 8-token draft block.
  uint32_t pos0;
  // NeoX RoPE base frequency and per-head RMSNorm epsilon.
  float rope_theta, eps;
} MtpBlockQkParam;

typedef struct {
  // Block start position, block length, ring capacity, SWA window, partition
  // count and partition buffer stride.
  uint32_t pos0, block, ring_rows, window, nparts, part_stride;
} MtpAttendParam;

typedef struct {
  // Draft slot index (1..7), block start position, f_rnn ring size, hidden
  // width and Markov GRU rank.
  uint32_t step_idx, pos0, frnn_ring, dim, rank;
} MtpMarkovParam;

typedef struct {
  // Unpadded vocabulary size and hidden width.
  uint32_t vocab, k;
  // Final logit softcap bound.
  float cap;
} MtpUnembedParam;

typedef struct {
  // Winning softcapped logit and token index for one unembed threadgroup.
  float val;
  int32_t idx;
} MtpArgmaxPair;

// One BF16 linear projection weight matrix [n, k].
typedef struct {
  // Raw BF16 weight buffer in GPU memory.
  Buf w;
  // Output rows and input columns.
  int n, k;
} MtpBf16Lin;

// All weights for one of the 3 MTP draft transformer layers.
typedef struct {
  // Nonzero for the full-attention draft layer (layer 1); zero for SWA (0, 2).
  int is_global;
  // Local SWA slot index (0 or 1), or -1 for the global layer.
  int local_slot;
  // Context RMSNorm, pre-attention RMSNorm, and post-attention RMSNorm (FP32).
  Buf dflare_norm, input_norm, post_attn_norm;
  // Per-head RMSNorm weights for Q, block K, and context K (FP32 [128]).
  Buf q_norm, k_norm, k_norm_ctx;
  // Self-attention and context K/V linear projections (BF16).
  MtpBf16Lin wq, wk, wv, wo, wk_ctx, wv_ctx;
  // Standard SwiGLU feed-forward projections (BF16).
  MtpBf16Lin gate_proj, up_proj, down_proj;
} MtpLayer;

// Loaded MTP metadata, weights and kernel handles.
static int active_drafts;
static TensorIndex *mtp_index;
static float fuse_probs[INK2_MTP_LAYERS + 1][INK2_MTP_TAPS];
static MtpLayer mtp_layers[INK2_MTP_LAYERS];
static Buf mask_embed, final_norm;
static Buf markov_dflare_norm, markov_w1, rnn_bias, hidden_bias;
static MtpBf16Lin rnn_proj, hidden_out;

// Compiled Metal kernel handles.
static int K_mv_bf16, K_mvm_bf16, K_mvm_bf16_3, K_mm_bf16;
static int K_rmsnorm, K_copy, K_add;
static int K_fuse_norm, K_ctx_kv_store, K_block_init, K_block_qk_prep;
static int K_attend_part, K_attend_combine, K_swiglu;
static int K_markov_concat, K_markov_gru, K_markov_add;
static int K_unembed_argmax_part, K_unembed_argmax_final;

// Reads all bytes of a small safetensors tensor directly via pread.
static void read_exact(Tensor t, void *dst) {
  size_t done = 0;
  while (done < t.size) {
    ssize_t n = pread(t.fd, (uint8_t *)dst + done, t.size - done,
                      (off_t)(t.off + done));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("cannot read Inkling-2-Mini MTP tensor bytes");
    done += (size_t)n;
  }
}

// Converts one BF16 bit pattern to FP32.
static float bf16_to_f32(uint16_t bits) {
  uint32_t u = (uint32_t)bits << MTP_BF16_SHIFT;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Returns the byte offset of SWA draft layer local_slot's K or V ring.
static size_t mtp_slide_off(int local_slot, int is_v) {
  return ((size_t)local_slot * 2 + (size_t)is_v) * INK2_MTP_SLIDE_KV_ONE_BYTES;
}

// Allocates a resident GPU buffer counted in `slot`.
static Buf mtp_work_alloc(Gpu *gpu, MemorySlot *slot, size_t size) {
  Buf b = gpu_alloc(gpu, slot, size);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Allocates a resident GPU buffer counted in `slot`, returning empty on
// refusal.
static Buf mtp_state_try_alloc(Gpu *gpu, MemorySlot *slot, size_t size) {
  Buf b = gpu_try_alloc_required(gpu, slot, size);
  if (!b.obj) return b;
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Reads a required JSON number field from obj, ending the process if absent.
static double req_num(const Json *obj, const char *key, const char *path) {
  const Json *v = json_get(obj, key);
  if (!v || v->type != J_NUM) die("%s: missing MTP number %s", path, key);
  return v->num;
}

// Validates mtp_config in dir/config.json against Inkling-2-Mini's geometry.
static void check_mtp_config(const char *dir) {
  char path[MTP_PATH_CAP];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *mc = json_get(root, "mtp_config");
  if (!mc || mc->type != J_OBJ) die("%s: missing mtp_config object", path);

  if ((int)req_num(mc, "hidden_size", path) != INK2_DIM ||
      (int)req_num(mc, "num_hidden_layers", path) != INK2_MTP_LAYERS ||
      (int)req_num(mc, "num_attention_heads", path) != INK2_HEADS ||
      (int)req_num(mc, "num_key_value_heads", path) != INK2_KV_HEADS ||
      (int)req_num(mc, "head_dim", path) != INK2_HEAD_DIM ||
      (int)req_num(mc, "intermediate_size", path) != INK2_MTP_INTER ||
      (int)req_num(mc, "sliding_window_size", path) != INK2_MTP_WINDOW ||
      (int)req_num(mc, "block_size", path) != INK2_MTP_BLOCK_SIZE ||
      (int)req_num(mc, "markov_rank", path) != INK2_MTP_MARKOV_RANK ||
      fabs(req_num(mc, "rope_theta", path) - MTP_ROPE_THETA) > 1.0 ||
      fabs(req_num(mc, "rms_norm_eps", path) - MTP_RMS_EPS) > 1e-9)
    die("%s: Inkling-2-Mini mtp_config dimensions do not match", path);

  const Json *tids = json_get(mc, "target_layer_ids");
  const int want_taps[INK2_MTP_TAPS] = {INK2_MTP_TAP_0, INK2_MTP_TAP_1,
                                        INK2_MTP_TAP_2, INK2_MTP_TAP_3};
  if (!tids || tids->type != J_ARR || tids->n != INK2_MTP_TAPS)
    die("%s: unexpected mtp_config target_layer_ids count", path);
  for (int i = 0; i < INK2_MTP_TAPS; i++) {
    if (tids->kid[i].type != J_NUM || (int)tids->kid[i].num != want_taps[i])
      die("%s: unexpected mtp_config target_layer_ids[%d]", path, i);
  }

  const Json *lids = json_get(mc, "local_layer_ids");
  if (!lids || lids->type != J_ARR || lids->n != INK2_MTP_LOCAL_COUNT ||
      lids->kid[0].type != J_NUM || lids->kid[1].type != J_NUM ||
      (int)lids->kid[0].num != 0 || (int)lids->kid[1].num != 2)
    die("%s: unexpected mtp_config local_layer_ids", path);

  json_free(root);
  free(text);
}

// Adds a raw tensor to weights, enabling aligned direct reads when large.
static void add_raw(WeightGroup *weights, Buf *dst, Tensor t) {
  int aligned =
      t.size >= MTP_ALIGN_READ_MIN_BYTES && (t.off & MTP_ALIGN_MASK) == 0 ? 1
                                                                          : 0;
  weight_group_add_tensor(weights, dst, t, aligned);
}

// Adds a BF16 [n, k] linear projection weight to weights and validates shape.
static void add_bf16_lin(WeightGroup *weights, MtpBf16Lin *lin, Tensor t, int n,
                         int k) {
  if (t.dtype != 'b' || t.ndim != 2 || t.shape[0] != n || t.shape[1] != k)
    die("unexpected Inkling-2-Mini MTP linear shape (%lldx%lld, expected "
        "%dx%d)",
        (long long)t.shape[0], (long long)t.shape[1], n, k);
  lin->n = n;
  lin->k = k;
  add_raw(weights, &lin->w, t);
}

// Reads fuse_weight [4, 4] BF16 and precomputes per-row FP32 softmax weights.
static void load_fuse_probs(Tensor t) {
  if (t.dtype != 'b' || t.ndim != 2 || t.shape[0] != INK2_MTP_LAYERS + 1 ||
      t.shape[1] != INK2_MTP_TAPS)
    die("unexpected Inkling-2-Mini MTP fuse_weight shape");
  uint16_t raw[INK2_MTP_LAYERS + 1][INK2_MTP_TAPS];
  read_exact(t, raw);
  for (int r = 0; r < INK2_MTP_LAYERS + 1; r++) {
    float v[INK2_MTP_TAPS];
    float mx = -INFINITY;
    for (int j = 0; j < INK2_MTP_TAPS; j++) {
      v[j] = bf16_to_f32(raw[r][j]);
      if (v[j] > mx) mx = v[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < INK2_MTP_TAPS; j++) {
      v[j] = expf(v[j] - mx);
      sum += v[j];
    }
    for (int j = 0; j < INK2_MTP_TAPS; j++) fuse_probs[r][j] = v[j] / sum;
  }
}

int inkling2_mtp_init(Gpu *gpu, WeightGroup *weights, const char *dir,
                      int requested_drafts) {
  active_drafts = 0;
  if (requested_drafts <= 0) return 0;

  char path[MTP_PATH_CAP];
  snprintf(path, sizeof path, "%s/mtp.safetensors", dir);
  if (access(path, R_OK) != 0) return 0;

  check_mtp_config(dir);
  mtp_index = tensor_index_open_file(path);
  load_fuse_probs(tensor_index_get(mtp_index, "fuse_weight"));

  weight_group_add_vector(weights, &mask_embed,
                          tensor_index_get(mtp_index, "mask_embed"), 0.0f, 0);
  weight_group_add_vector(weights, &final_norm,
                          tensor_index_get(mtp_index, "norm.weight"), 0.0f, 0);

  int local_slot = 0;
  for (int l = 0; l < INK2_MTP_LAYERS; l++) {
    MtpLayer *ml = &mtp_layers[l];
    ml->is_global = (l == INK2_MTP_GLOBAL_LAYER);
    ml->local_slot = ml->is_global ? -1 : local_slot++;

    weight_group_add_vector(
        weights, &ml->dflare_norm,
        tensor_index_get(mtp_index, "layers.%d.dflare_norm.weight", l), 0.0f,
        0);
    weight_group_add_vector(
        weights, &ml->input_norm,
        tensor_index_get(mtp_index, "layers.%d.input_layernorm.weight", l),
        0.0f, 0);
    weight_group_add_vector(
        weights, &ml->post_attn_norm,
        tensor_index_get(mtp_index, "layers.%d.post_attention_layernorm.weight",
                         l),
        0.0f, 0);
    weight_group_add_vector(
        weights, &ml->q_norm,
        tensor_index_get(mtp_index, "layers.%d.self_attn.q_norm.weight", l),
        0.0f, 0);
    weight_group_add_vector(
        weights, &ml->k_norm,
        tensor_index_get(mtp_index, "layers.%d.self_attn.k_norm.weight", l),
        0.0f, 0);
    weight_group_add_vector(
        weights, &ml->k_norm_ctx,
        tensor_index_get(mtp_index, "layers.%d.self_attn.k_norm_ctx.weight", l),
        0.0f, 0);

    add_bf16_lin(
        weights, &ml->wq,
        tensor_index_get(mtp_index, "layers.%d.self_attn.q_proj.weight", l),
        INK2_Q_DIM, INK2_DIM);
    add_bf16_lin(
        weights, &ml->wk,
        tensor_index_get(mtp_index, "layers.%d.self_attn.k_proj.weight", l),
        INK2_KV_DIM, INK2_DIM);
    add_bf16_lin(
        weights, &ml->wv,
        tensor_index_get(mtp_index, "layers.%d.self_attn.v_proj.weight", l),
        INK2_KV_DIM, INK2_DIM);
    add_bf16_lin(
        weights, &ml->wo,
        tensor_index_get(mtp_index, "layers.%d.self_attn.o_proj.weight", l),
        INK2_DIM, INK2_Q_DIM);
    add_bf16_lin(
        weights, &ml->wk_ctx,
        tensor_index_get(mtp_index, "layers.%d.self_attn.k_proj_ctx.weight", l),
        INK2_KV_DIM, INK2_DIM);
    add_bf16_lin(
        weights, &ml->wv_ctx,
        tensor_index_get(mtp_index, "layers.%d.self_attn.v_proj_ctx.weight", l),
        INK2_KV_DIM, INK2_DIM);

    add_bf16_lin(
        weights, &ml->gate_proj,
        tensor_index_get(mtp_index, "layers.%d.mlp.gate_proj.weight", l),
        INK2_MTP_INTER, INK2_DIM);
    add_bf16_lin(weights, &ml->up_proj,
                 tensor_index_get(mtp_index, "layers.%d.mlp.up_proj.weight", l),
                 INK2_MTP_INTER, INK2_DIM);
    add_bf16_lin(
        weights, &ml->down_proj,
        tensor_index_get(mtp_index, "layers.%d.mlp.down_proj.weight", l),
        INK2_DIM, INK2_MTP_INTER);
  }

  weight_group_add_vector(
      weights, &markov_dflare_norm,
      tensor_index_get(mtp_index, "markov_head.dflare_norm.weight"), 0.0f, 0);
  add_raw(weights, &markov_w1,
          tensor_index_get(mtp_index, "markov_head.markov_w1.weight"));
  add_bf16_lin(weights, &rnn_proj,
               tensor_index_get(mtp_index, "markov_head.rnn_proj.weight"),
               INK2_MTP_RNN_OUT, INK2_MTP_RNN_IN);
  weight_group_add_vector(
      weights, &rnn_bias,
      tensor_index_get(mtp_index, "markov_head.rnn_proj.bias"), 0.0f, 0);
  add_bf16_lin(weights, &hidden_out,
               tensor_index_get(mtp_index, "markov_head.hidden_out.weight"),
               INK2_DIM, INK2_MTP_MARKOV_RANK);
  weight_group_add_vector(
      weights, &hidden_bias,
      tensor_index_get(mtp_index, "markov_head.hidden_out.bias"), 0.0f, 0);

#define K(field, name) field = gpu_find_kernel(gpu, name)
  K(K_mv_bf16, "mv_bf16");
  K(K_mvm_bf16, "mvm_bf16");
  K(K_mvm_bf16_3, "mvm_bf16_3");
  K(K_mm_bf16, "mm_bf16");
  K(K_rmsnorm, "ink2_rmsnorm");
  K(K_copy, "ink2_copy");
  K(K_add, "ink2_add");
  K(K_fuse_norm, "ink2_mtp_fuse_norm");
  K(K_ctx_kv_store, "ink2_mtp_ctx_kv_store");
  K(K_block_init, "ink2_mtp_block_init");
  K(K_block_qk_prep, "ink2_mtp_block_qk_prep");
  K(K_attend_part, "ink2_mtp_attend_part");
  K(K_attend_combine, "ink2_mtp_attend_combine");
  K(K_swiglu, "ink2_mtp_swiglu");
  K(K_markov_concat, "ink2_mtp_markov_concat");
  K(K_markov_gru, "ink2_mtp_markov_gru");
  K(K_markov_add, "ink2_mtp_markov_add");
  K(K_unembed_argmax_part, "ink2_mtp_unembed_argmax_part");
  K(K_unembed_argmax_final, "ink2_mtp_unembed_argmax_final");
#undef K

  active_drafts = requested_drafts < INK2_MTP_MAX_DRAFTS ? requested_drafts
                                                         : INK2_MTP_MAX_DRAFTS;
  return active_drafts;
}

int inkling2_mtp_active_drafts(void) { return active_drafts; }

// ---- workspace and state allocation ----

uint64_t inkling2_mtp_estimate_work_bytes(int rows) {
  if (!active_drafts) return 0;
  uint64_t bytes = 0;
  bytes += gpu_round_up_to_pages((size_t)INK2_MTP_TAPS * rows * INK2_DIM *
                                 sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)rows * INK2_DIM * sizeof(float));
  bytes +=
      2 * gpu_round_up_to_pages((size_t)rows * INK2_KV_DIM * sizeof(float));
  bytes += 5 * gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM *
                                     sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM *
                                     sizeof(float));
  bytes += 2 * gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                     sizeof(float));
  bytes += gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM *
                                 sizeof(uint16_t));
  bytes += 2 * gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                     sizeof(uint16_t));
  bytes +=
      gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE * INK2_HEADS *
                            MTP_MAX_PARTS * MTP_PART_FLOATS * sizeof(float));
  bytes += 3 * gpu_round_up_to_pages((size_t)INK2_MTP_BLOCK_SIZE *
                                     INK2_MTP_INTER * sizeof(float));
  bytes +=
      gpu_round_up_to_pages((size_t)MTP_UNEMBED_PARTS * sizeof(MtpArgmaxPair));
  return bytes;
}

void inkling2_mtp_free_work(Ink2MtpWork *w) {
  Buf *list[] = {
      &w->taps,     &w->ctx,     &w->k_ctx_raw, &w->v_ctx_raw,    &w->kv_tgt,
      &w->frnn_tgt, &w->cur,     &w->xn,        &w->q_raw,        &w->k_raw,
      &w->v_raw,    &w->q_blk,   &w->k_blk,     &w->v_blk,        &w->part,
      &w->attn_out, &w->o_out,   &w->gate,      &w->up,           &w->h_mlp,
      &w->mlp_out,  &w->h_final, &w->rnn_state, &w->joint,        &w->proj,
      &w->latent,   &w->corr,    &w->h_eff,     &w->argmax_parts, &w->tok};
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    if (list[i]->obj) gpu_mark_to_free(list[i]);
  *w = (Ink2MtpWork){0};
}

void inkling2_mtp_ensure_work(Gpu *gpu, MemorySlot *work_slot, Ink2MtpWork *w,
                              int rows) {
  if (!active_drafts) return;
  if (!w->cur.obj) {
    w->kv_tgt = mtp_work_alloc(gpu, work_slot,
                               (size_t)INK2_MTP_LAYERS * INK2_MAX_BATCH_ROWS *
                                   sizeof(MtpKvTarget));
    w->frnn_tgt = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MAX_BATCH_ROWS * sizeof(MtpFrnnTarget));
    w->cur = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
    w->xn = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
    w->q_raw = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM *
                                  sizeof(float));
    w->k_raw = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                  sizeof(float));
    w->v_raw = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                  sizeof(float));
    w->q_blk = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM *
                                  sizeof(uint16_t));
    w->k_blk = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                  sizeof(uint16_t));
    w->v_blk = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM *
                                  sizeof(uint16_t));
    w->part =
        mtp_work_alloc(gpu, work_slot,
                       (size_t)INK2_MTP_BLOCK_SIZE * INK2_HEADS *
                           MTP_MAX_PARTS * MTP_PART_FLOATS * sizeof(float));
    w->attn_out = mtp_work_alloc(gpu, work_slot,
                                 (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM *
                                     sizeof(float));
    w->o_out = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
    w->gate = mtp_work_alloc(gpu, work_slot,
                             (size_t)INK2_MTP_BLOCK_SIZE * INK2_MTP_INTER *
                                 sizeof(float));
    w->up = mtp_work_alloc(gpu, work_slot,
                           (size_t)INK2_MTP_BLOCK_SIZE * INK2_MTP_INTER *
                               sizeof(float));
    w->h_mlp = mtp_work_alloc(gpu, work_slot,
                              (size_t)INK2_MTP_BLOCK_SIZE * INK2_MTP_INTER *
                                  sizeof(float));
    w->mlp_out = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
    w->h_final = mtp_work_alloc(
        gpu, work_slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
    w->rnn_state = mtp_work_alloc(gpu, work_slot,
                                  (size_t)INK2_MTP_MARKOV_RANK * sizeof(float));
    w->joint =
        mtp_work_alloc(gpu, work_slot, (size_t)INK2_MTP_RNN_IN * sizeof(float));
    w->proj = mtp_work_alloc(gpu, work_slot,
                             (size_t)INK2_MTP_RNN_OUT * sizeof(float));
    w->latent = mtp_work_alloc(gpu, work_slot,
                               (size_t)INK2_MTP_MARKOV_RANK * sizeof(float));
    w->corr = mtp_work_alloc(gpu, work_slot, (size_t)INK2_DIM * sizeof(float));
    w->h_eff = mtp_work_alloc(gpu, work_slot, (size_t)INK2_DIM * sizeof(float));
    w->argmax_parts = mtp_work_alloc(
        gpu, work_slot, (size_t)MTP_UNEMBED_PARTS * sizeof(MtpArgmaxPair));
    w->tok = mtp_work_alloc(gpu, work_slot,
                            (size_t)INK2_MTP_BLOCK_SIZE * sizeof(int32_t));
  }
  if (rows > w->cap_rows) {
    if (w->taps.obj) gpu_mark_to_free(&w->taps);
    if (w->ctx.obj) gpu_mark_to_free(&w->ctx);
    if (w->k_ctx_raw.obj) gpu_mark_to_free(&w->k_ctx_raw);
    if (w->v_ctx_raw.obj) gpu_mark_to_free(&w->v_ctx_raw);
    w->cap_rows = rows < INK2_MAX_VERIFY ? INK2_MAX_VERIFY : rows;
    w->taps = mtp_work_alloc(gpu, work_slot,
                             (size_t)INK2_MTP_TAPS * w->cap_rows * INK2_DIM *
                                 sizeof(float));
    w->ctx = mtp_work_alloc(gpu, work_slot,
                            (size_t)w->cap_rows * INK2_DIM * sizeof(float));
    w->k_ctx_raw = mtp_work_alloc(
        gpu, work_slot, (size_t)w->cap_rows * INK2_KV_DIM * sizeof(float));
    w->v_ctx_raw = mtp_work_alloc(
        gpu, work_slot, (size_t)w->cap_rows * INK2_KV_DIM * sizeof(float));
  }
}

int inkling2_mtp_state_init(Gpu *gpu, MemorySlot *state_slot,
                            Ink2MtpState *ms) {
  *ms = (Ink2MtpState){0};
  if (!active_drafts) return 1;
  ms->slide_kv = mtp_state_try_alloc(gpu, state_slot, INK2_MTP_SLIDE_KV_BYTES);
  ms->frnn = ms->slide_kv.obj
                 ? mtp_state_try_alloc(gpu, state_slot, INK2_MTP_FRNN_BYTES)
                 : (Buf){0};
  if (!ms->frnn.obj) {
    if (ms->slide_kv.obj) gpu_free(&ms->slide_kv);
    *ms = (Ink2MtpState){0};
    return 0;
  }
  memset(ms->slide_kv.ptr, 0, INK2_MTP_SLIDE_KV_BYTES);
  memset(ms->frnn.ptr, 0, INK2_MTP_FRNN_BYTES);
  return 1;
}

int inkling2_mtp_state_resize_global(Gpu *gpu, MemorySlot *state_slot,
                                     Ink2MtpState *ms, int new_cap, int keep) {
  if (!active_drafts) return 1;
  size_t new_bytes = (size_t)new_cap * MTP_GLOBAL_KV_ROW_BYTES;
  size_t copy_bytes = (size_t)keep * MTP_GLOBAL_KV_ROW_BYTES;
  Buf nk = mtp_state_try_alloc(gpu, state_slot, new_bytes);
  Buf nv = nk.obj ? mtp_state_try_alloc(gpu, state_slot, new_bytes) : (Buf){0};
  if (!nv.obj) {
    if (nk.obj) gpu_free(&nk);
    return 0;
  }
  if (copy_bytes && ms->gk.obj) {
    memcpy(nk.ptr, ms->gk.ptr, copy_bytes);
    memcpy(nv.ptr, ms->gv.ptr, copy_bytes);
  }
  if (ms->gk.obj) gpu_mark_to_free(&ms->gk);
  if (ms->gv.obj) gpu_mark_to_free(&ms->gv);
  ms->gk = nk;
  ms->gv = nv;
  return 1;
}

void inkling2_mtp_state_free(Ink2MtpState *ms) {
  if (ms->gk.obj) gpu_mark_to_free(&ms->gk);
  if (ms->gv.obj) gpu_mark_to_free(&ms->gv);
  if (ms->slide_kv.obj) gpu_mark_to_free(&ms->slide_kv);
  if (ms->frnn.obj) gpu_mark_to_free(&ms->frnn);
  *ms = (Ink2MtpState){0};
}

size_t inkling2_mtp_state_bytes(int cap) {
  if (!active_drafts) return 0;
  size_t gk_one = gpu_round_up_to_pages((size_t)cap * MTP_GLOBAL_KV_ROW_BYTES);
  return 2 * gk_one + gpu_round_up_to_pages(INK2_MTP_SLIDE_KV_BYTES) +
         gpu_round_up_to_pages(INK2_MTP_FRNN_BYTES);
}

// ---- GPU dispatch helpers ----

// Dispatches a single BF16 linear projection y = x @ w^T (unrounded FP32).
static void run_linear(GpuCommandBuffer *c, const Buf *w, const Buf *x,
                       const Buf *y, int m, int n, int k, int ldy,
                       uint32_t split, int is_prefill) {
  if (!is_prefill && m == 1) {
    uint32_t rows_per_tg = (MTP_BF16_MV_SGS / split) * MTP_BF16_MV_ROWS_PER_SG;
    MtpMvParam p = {(uint32_t)n, (uint32_t)k, (uint32_t)n, 0, 0, split};
    gpu_select_kernel(c, K_mv_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, 0);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   MTP_MV_THREADS, 1, 1);
  } else if (!is_prefill && m <= INK2_MAX_BATCH_ROWS) {
    uint32_t rows_per_tg = (MTP_BF16_MV_SGS / split) * MTP_BF16_MV_ROWS_PER_SG;
    MtpMvmParam p = {{(uint32_t)n, (uint32_t)k, (uint32_t)n, 0, 0, split},
                     (uint32_t)m,
                     (uint32_t)k,
                     (uint32_t)ldy};
    gpu_select_kernel(c, K_mvm_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, 0);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    gpu_run_kernel(c, ((uint32_t)n + rows_per_tg - 1u) / rows_per_tg, 1, 1,
                   MTP_MV_THREADS, 1, 1);
  } else {
    MtpMmParam p = {(uint32_t)m, (uint32_t)n, (uint32_t)k,
                    0,           (uint32_t)k, (uint32_t)ldy};
    gpu_select_kernel(c, K_mm_bf16);
    gpu_bind_buffer(c, 0, w, 0);
    gpu_bind_buffer(c, 2, x, 0);
    gpu_bind_buffer(c, 3, y, 0);
    gpu_copy_bytes(c, 4, &p, sizeof p);
    uint32_t tiles_n = ((uint32_t)n + MTP_MM_TILE_N - 1u) / MTP_MM_TILE_N;
    uint32_t tiles_m = ((uint32_t)m + MTP_MM_TILE_M - 1u) / MTP_MM_TILE_M;
    gpu_run_kernel(c, tiles_n * tiles_m, 1, 1, MTP_MM_THREADS, 1, 1);
  }
}

// Dispatches fused Q, K, V BF16 projections for the 8-row draft block.
static void run_qkv_block(GpuCommandBuffer *c, const MtpLayer *ml, const Buf *x,
                          const Buf *yq, const Buf *yk, const Buf *yv) {
  uint32_t max_n = (uint32_t)ml->wq.n;
  uint32_t dims[3][2] = {{(uint32_t)ml->wq.n, (uint32_t)ml->wq.n},
                         {(uint32_t)ml->wk.n, (uint32_t)ml->wk.n},
                         {(uint32_t)ml->wv.n, (uint32_t)ml->wv.n}};
  MtpMvmParam p = {{max_n, (uint32_t)ml->wq.k, max_n, 0, 0, 1u},
                   (uint32_t)INK2_MTP_BLOCK_SIZE,
                   (uint32_t)ml->wq.k,
                   max_n};
  gpu_select_kernel(c, K_mvm_bf16_3);
  gpu_bind_buffer(c, 0, &ml->wq.w, 0);
  gpu_bind_buffer(c, 2, x, 0);
  gpu_bind_buffer(c, 3, yq, 0);
  gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_bind_buffer(c, 5, &ml->wk.w, 0);
  gpu_bind_buffer(c, 7, yk, 0);
  gpu_bind_buffer(c, 8, &ml->wv.w, 0);
  gpu_bind_buffer(c, 10, yv, 0);
  gpu_copy_bytes(c, 11, dims, sizeof dims);
  gpu_run_kernel(c, (max_n + MTP_BF16_ROWS_PER_TG - 1u) / MTP_BF16_ROWS_PER_TG,
                 3, 1, MTP_MV_THREADS, 1, 1);
}

// Dispatches unrounded weighted RMSNorm over `rows` rows of INK2_DIM floats.
static void run_rmsnorm(GpuCommandBuffer *c, const Buf *x, const Buf *w,
                        const Buf *y, int rows) {
  MtpNormParam p = {(uint32_t)INK2_DIM, 0u, MTP_RMS_EPS};
  gpu_select_kernel(c, K_rmsnorm);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, y, 0);
  gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, MTP_ROW_THREADS, 1, 1);
}

// Dispatches elementwise unrounded residual add out = a + b.
static void run_add(GpuCommandBuffer *c, const Buf *a, const Buf *b,
                    const Buf *out, uint32_t count) {
  uint32_t args[2] = {count, 0u};
  gpu_select_kernel(c, K_add);
  gpu_bind_buffer(c, 0, a, 0);
  gpu_bind_buffer(c, 1, b, 0);
  gpu_bind_buffer(c, 2, out, 0);
  gpu_copy_bytes(c, 3, args, sizeof args);
  gpu_run_kernel(c, (count + MTP_ELEM_THREADS - 1u) / MTP_ELEM_THREADS, 1, 1,
                 MTP_ELEM_THREADS, 1, 1);
}

// ---- tap capture, context injection and block drafting ----

void inkling2_mtp_capture_tap(GpuCommandBuffer *cmd, const Buf *src,
                              const Ink2MtpWork *w, int tap_idx, int rows) {
  if (!active_drafts) return;
  size_t dst_off =
      (size_t)tap_idx * (size_t)w->cap_rows * INK2_DIM * sizeof(float);
  uint32_t count = (uint32_t)((size_t)rows * INK2_DIM);
  gpu_select_kernel(cmd, K_copy);
  gpu_bind_buffer(cmd, 0, src, 0);
  gpu_bind_buffer(cmd, 1, &w->taps, dst_off);
  gpu_copy_bytes(cmd, 2, &count, sizeof count);
  gpu_run_kernel(cmd, (count + MTP_ELEM_THREADS - 1u) / MTP_ELEM_THREADS, 1, 1,
                 MTP_ELEM_THREADS, 1, 1);
}

void inkling2_mtp_inject(GpuCommandBuffer *cmd, Ink2MtpWork *w,
                         Ink2MtpState *const *states, const int *pos, int rows,
                         int batched, int is_prefill) {
  if (!active_drafts) return;
  if (batched) {
    MtpKvTarget *kt = (MtpKvTarget *)w->kv_tgt.ptr;
    MtpFrnnTarget *ft = (MtpFrnnTarget *)w->frnn_tgt.ptr;
    for (int r = 0; r < rows; r++) {
      const Ink2MtpState *sr = states[r];
      ft[r] =
          (MtpFrnnTarget){sr->frnn.address + sr->frnn.off, (uint32_t)pos[r], 0};
      for (int l = 0; l < INK2_MTP_LAYERS; l++) {
        const MtpLayer *ml = &mtp_layers[l];
        size_t idx = (size_t)l * INK2_MAX_BATCH_ROWS + (size_t)r;
        if (ml->is_global) {
          kt[idx] =
              (MtpKvTarget){sr->gk.address + sr->gk.off,
                            sr->gv.address + sr->gv.off, (uint32_t)pos[r], 0u};
        } else {
          uint64_t sb = sr->slide_kv.address + sr->slide_kv.off;
          kt[idx] =
              (MtpKvTarget){sb + mtp_slide_off(ml->local_slot, 0),
                            sb + mtp_slide_off(ml->local_slot, 1),
                            (uint32_t)pos[r], (uint32_t)INK2_MTP_SLIDE_RING};
        }
      }
    }
  }

  const Ink2MtpState *s0 = states[0];
  int pos0 = pos[0];
  for (int l = 0; l < INK2_MTP_LAYERS; l++) {
    const MtpLayer *ml = &mtp_layers[l];
    // For single-sequence SWA chunks exceeding the ring size, only the final
    // INK2_MTP_SLIDE_RING rows survive in the ring; slice to avoid slot races.
    int row_off = 0, run_rows = rows;
    if (!batched && !ml->is_global && rows > INK2_MTP_SLIDE_RING) {
      row_off = rows - INK2_MTP_SLIDE_RING;
      run_rows = INK2_MTP_SLIDE_RING;
    }
    size_t tap_off = (size_t)row_off * INK2_DIM * sizeof(float);
    MtpFuseParam fp = {fuse_probs[l][0],
                       fuse_probs[l][1],
                       fuse_probs[l][2],
                       fuse_probs[l][3],
                       (uint32_t)run_rows,
                       (uint32_t)w->cap_rows,
                       (uint32_t)INK2_DIM,
                       (uint32_t)(pos0 + row_off),
                       0u,
                       0u,
                       MTP_RMS_EPS};
    gpu_select_kernel(cmd, K_fuse_norm);
    gpu_bind_buffer(cmd, 0, &w->taps, tap_off);
    gpu_bind_buffer(cmd, 1, &ml->dflare_norm, 0);
    gpu_bind_buffer(cmd, 2, &w->ctx, 0);
    gpu_bind_buffer(cmd, 3, &w->frnn_tgt, 0);
    gpu_copy_bytes(cmd, 4, &fp, sizeof fp);
    gpu_run_kernel(cmd, (uint32_t)run_rows, 1, 1, MTP_ROW_THREADS, 1, 1);

    run_linear(cmd, &ml->wk_ctx.w, &w->ctx, &w->k_ctx_raw, run_rows,
               ml->wk_ctx.n, ml->wk_ctx.k, ml->wk_ctx.n, 1u, is_prefill);
    run_linear(cmd, &ml->wv_ctx.w, &w->ctx, &w->v_ctx_raw, run_rows,
               ml->wv_ctx.n, ml->wv_ctx.k, ml->wv_ctx.n, 1u, is_prefill);

    const Buf *k_dst = ml->is_global ? &s0->gk : &s0->slide_kv;
    const Buf *v_dst = ml->is_global ? &s0->gv : &s0->slide_kv;
    size_t k_off = ml->is_global ? 0 : mtp_slide_off(ml->local_slot, 0);
    size_t v_off = ml->is_global ? 0 : mtp_slide_off(ml->local_slot, 1);
    uint32_t ring_rows = ml->is_global ? 0u : (uint32_t)INK2_MTP_SLIDE_RING;
    size_t kv_tgt_off = (size_t)l * INK2_MAX_BATCH_ROWS * sizeof(MtpKvTarget);
    MtpCtxKvParam kp = {(uint32_t)run_rows, (uint32_t)(pos0 + row_off),
                        ring_rows,          (uint32_t)batched,
                        MTP_ROPE_THETA,     MTP_RMS_EPS};
    gpu_select_kernel(cmd, K_ctx_kv_store);
    gpu_bind_buffer(cmd, 0, &w->k_ctx_raw, 0);
    gpu_bind_buffer(cmd, 1, &w->v_ctx_raw, 0);
    gpu_bind_buffer(cmd, 2, &ml->k_norm_ctx, 0);
    gpu_bind_buffer(cmd, 3, k_dst, k_off);
    gpu_bind_buffer(cmd, 4, v_dst, v_off);
    gpu_bind_buffer(cmd, 5, &w->kv_tgt, kv_tgt_off);
    gpu_copy_bytes(cmd, 6, &kp, sizeof kp);
    gpu_run_kernel(cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)run_rows, 1,
                   MTP_HEAD_THREADS, 1, 1);
  }

  // Write f_rnn into the rolling ring. For single-sequence chunks larger than
  // INK2_MTP_FRNN_RING, slice to the trailing rows so no slot collides.
  int fr_off = 0, fr_rows = rows;
  if (!batched && rows > INK2_MTP_FRNN_RING) {
    fr_off = rows - INK2_MTP_FRNN_RING;
    fr_rows = INK2_MTP_FRNN_RING;
  }
  size_t fr_tap_off = (size_t)fr_off * INK2_DIM * sizeof(float);
  MtpFuseParam ffp = {fuse_probs[INK2_MTP_LAYERS][0],
                      fuse_probs[INK2_MTP_LAYERS][1],
                      fuse_probs[INK2_MTP_LAYERS][2],
                      fuse_probs[INK2_MTP_LAYERS][3],
                      (uint32_t)fr_rows,
                      (uint32_t)w->cap_rows,
                      (uint32_t)INK2_DIM,
                      (uint32_t)(pos0 + fr_off),
                      (uint32_t)INK2_MTP_FRNN_RING,
                      (uint32_t)batched,
                      MTP_RMS_EPS};
  gpu_select_kernel(cmd, K_fuse_norm);
  gpu_bind_buffer(cmd, 0, &w->taps, fr_tap_off);
  gpu_bind_buffer(cmd, 1, &markov_dflare_norm, 0);
  gpu_bind_buffer(cmd, 2, &s0->frnn, 0);
  gpu_bind_buffer(cmd, 3, &w->frnn_tgt, 0);
  gpu_copy_bytes(cmd, 4, &ffp, sizeof ffp);
  gpu_run_kernel(cmd, (uint32_t)fr_rows, 1, 1, MTP_ROW_THREADS, 1, 1);
}

void inkling2_mtp_propose(Gpu *gpu, MemorySlot *work_slot, Ink2MtpWork *w,
                          const Ink2MtpState *ms, const Buf *embed_w,
                          const Buf *embed_norm, const Buf *unembed_w,
                          int token, int pos, int *drafts, int n,
                          uint64_t *last_committed, void (*sync_fn)(void)) {
  if (!active_drafts) die("Inkling-2-Mini MTP draft head is not active");
  if (n < 1 || n > active_drafts)
    die("Inkling-2-Mini MTP propose needs 1 to %d drafts (got %d)",
        active_drafts, n);
  if (pos < 1) die("Inkling-2-Mini MTP propose requires pos >= 1");
  if (token < 0 || token >= INK2_VOCAB_PADDED)
    die("Inkling-2-Mini MTP propose got invalid token %d", token);

  inkling2_mtp_ensure_work(gpu, work_slot, w, 1);
  ((int32_t *)w->tok.ptr)[0] = token;

  GpuCommandBuffer cmd = gpu_begin(gpu);

  // Build the 8-row block input: normalized anchor embedding + 7 mask rows.
  MtpBlockInitParam bip = {(uint32_t)token, (uint32_t)INK2_DIM, MTP_RMS_EPS};
  gpu_select_kernel(&cmd, K_block_init);
  gpu_bind_buffer(&cmd, 0, embed_w, 0);
  gpu_bind_buffer(&cmd, 1, embed_norm, 0);
  gpu_bind_buffer(&cmd, 2, &mask_embed, 0);
  gpu_bind_buffer(&cmd, 3, &w->cur, 0);
  gpu_copy_bytes(&cmd, 4, &bip, sizeof bip);
  gpu_run_kernel(&cmd, (uint32_t)INK2_MTP_BLOCK_SIZE, 1, 1, MTP_ROW_THREADS, 1,
                 1);

  // Run all 3 draft transformer layers across the full 8-row block (later mask
  // rows affect earlier slots through noncausal self-attention).
  uint32_t block_dim_floats = (uint32_t)(INK2_MTP_BLOCK_SIZE * INK2_DIM);
  uint32_t block_mlp_floats = (uint32_t)(INK2_MTP_BLOCK_SIZE * INK2_MTP_INTER);
  for (int l = 0; l < INK2_MTP_LAYERS; l++) {
    const MtpLayer *ml = &mtp_layers[l];
    run_rmsnorm(&cmd, &w->cur, &ml->input_norm, &w->xn, INK2_MTP_BLOCK_SIZE);
    run_qkv_block(&cmd, ml, &w->xn, &w->q_raw, &w->k_raw, &w->v_raw);

    MtpBlockQkParam qkp = {(uint32_t)pos, MTP_ROPE_THETA, MTP_RMS_EPS};
    gpu_select_kernel(&cmd, K_block_qk_prep);
    gpu_bind_buffer(&cmd, 0, &w->q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &w->k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &w->v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &ml->q_norm, 0);
    gpu_bind_buffer(&cmd, 4, &ml->k_norm, 0);
    gpu_bind_buffer(&cmd, 5, &w->q_blk, 0);
    gpu_bind_buffer(&cmd, 6, &w->k_blk, 0);
    gpu_bind_buffer(&cmd, 7, &w->v_blk, 0);
    gpu_copy_bytes(&cmd, 8, &qkp, sizeof qkp);
    gpu_run_kernel(&cmd, (uint32_t)(INK2_HEADS + INK2_KV_HEADS),
                   (uint32_t)INK2_MTP_BLOCK_SIZE, 1, MTP_HEAD_THREADS, 1, 1);

    const Buf *k_ctx = ml->is_global ? &ms->gk : &ms->slide_kv;
    const Buf *v_ctx = ml->is_global ? &ms->gv : &ms->slide_kv;
    size_t k_off = ml->is_global ? 0 : mtp_slide_off(ml->local_slot, 0);
    size_t v_off = ml->is_global ? 0 : mtp_slide_off(ml->local_slot, 1);
    uint32_t ring_rows = ml->is_global ? 0u : (uint32_t)INK2_MTP_SLIDE_RING;
    uint32_t window = ml->is_global ? 0u : (uint32_t)INK2_MTP_WINDOW;
    int ctx_len = (window > 0u && pos > (int)window) ? (int)window : pos;
    int total_k = ctx_len + INK2_MTP_BLOCK_SIZE;
    int nparts = (total_k + MTP_PART_TARGET - 1) / MTP_PART_TARGET;
    if (nparts < 1) nparts = 1;
    if (nparts > MTP_MAX_PARTS) nparts = MTP_MAX_PARTS;
    MtpAttendParam ap = {(uint32_t)pos,    (uint32_t)INK2_MTP_BLOCK_SIZE,
                         ring_rows,        window,
                         (uint32_t)nparts, (uint32_t)MTP_MAX_PARTS};

    gpu_select_kernel(&cmd, K_attend_part);
    gpu_bind_buffer(&cmd, 0, &w->q_blk, 0);
    gpu_bind_buffer(&cmd, 1, k_ctx, k_off);
    gpu_bind_buffer(&cmd, 2, v_ctx, v_off);
    gpu_bind_buffer(&cmd, 3, &w->k_blk, 0);
    gpu_bind_buffer(&cmd, 4, &w->v_blk, 0);
    gpu_bind_buffer(&cmd, 5, &w->part, 0);
    gpu_copy_bytes(&cmd, 6, &ap, sizeof ap);
    gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                   (uint32_t)INK2_MTP_BLOCK_SIZE, MTP_ATTEND_THREADS, 1, 1);

    gpu_select_kernel(&cmd, K_attend_combine);
    gpu_bind_buffer(&cmd, 0, &w->part, 0);
    gpu_bind_buffer(&cmd, 1, &w->attn_out, 0);
    gpu_copy_bytes(&cmd, 2, &ap, sizeof ap);
    gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                   MTP_COMBINE_THREADS, 1, 1);

    run_linear(&cmd, &ml->wo.w, &w->attn_out, &w->o_out, INK2_MTP_BLOCK_SIZE,
               ml->wo.n, ml->wo.k, ml->wo.n, 1u, 0);
    run_add(&cmd, &w->cur, &w->o_out, &w->cur, block_dim_floats);

    run_rmsnorm(&cmd, &w->cur, &ml->post_attn_norm, &w->xn,
                INK2_MTP_BLOCK_SIZE);
    run_linear(&cmd, &ml->gate_proj.w, &w->xn, &w->gate, INK2_MTP_BLOCK_SIZE,
               ml->gate_proj.n, ml->gate_proj.k, ml->gate_proj.n, 1u, 0);
    run_linear(&cmd, &ml->up_proj.w, &w->xn, &w->up, INK2_MTP_BLOCK_SIZE,
               ml->up_proj.n, ml->up_proj.k, ml->up_proj.n, 1u, 0);

    gpu_select_kernel(&cmd, K_swiglu);
    gpu_bind_buffer(&cmd, 0, &w->gate, 0);
    gpu_bind_buffer(&cmd, 1, &w->up, 0);
    gpu_bind_buffer(&cmd, 2, &w->h_mlp, 0);
    gpu_copy_bytes(&cmd, 3, &block_mlp_floats, sizeof block_mlp_floats);
    gpu_run_kernel(
        &cmd, (block_mlp_floats + MTP_ELEM_THREADS - 1u) / MTP_ELEM_THREADS, 1,
        1, MTP_ELEM_THREADS, 1, 1);

    run_linear(&cmd, &ml->down_proj.w, &w->h_mlp, &w->mlp_out,
               INK2_MTP_BLOCK_SIZE, ml->down_proj.n, ml->down_proj.k,
               ml->down_proj.n, 1u, 0);
    run_add(&cmd, &w->cur, &w->mlp_out, &w->cur, block_dim_floats);
  }

  run_rmsnorm(&cmd, &w->cur, &final_norm, &w->h_final, INK2_MTP_BLOCK_SIZE);

  // Sequential Markov RNN head over draft slots 1..n. Combines h[step] + corr
  // before the fused unembed + softcap + argmax so each slot reads unembed_w
  // once and avoids writing full vocabulary logits to DRAM.
  MtpUnembedParam up = {(uint32_t)INK2_VOCAB, (uint32_t)INK2_DIM,
                        MTP_LOGIT_SOFTCAP};
  uint32_t nparts = MTP_UNEMBED_PARTS;
  for (int step = 1; step <= n; step++) {
    MtpMarkovParam mp = {(uint32_t)step, (uint32_t)pos,
                         (uint32_t)INK2_MTP_FRNN_RING, (uint32_t)INK2_DIM,
                         (uint32_t)INK2_MTP_MARKOV_RANK};

    gpu_select_kernel(&cmd, K_markov_concat);
    gpu_bind_buffer(&cmd, 0, &w->rnn_state, 0);
    gpu_bind_buffer(&cmd, 1, &markov_w1, 0);
    gpu_bind_buffer(&cmd, 2, &w->tok, 0);
    gpu_bind_buffer(&cmd, 3, &w->h_final, 0);
    gpu_bind_buffer(&cmd, 4, &ms->frnn, 0);
    gpu_bind_buffer(&cmd, 5, &w->joint, 0);
    gpu_copy_bytes(&cmd, 6, &mp, sizeof mp);
    gpu_run_kernel(&cmd,
                   ((uint32_t)INK2_MTP_RNN_IN + MTP_ELEM_THREADS - 1u) /
                       MTP_ELEM_THREADS,
                   1, 1, MTP_ELEM_THREADS, 1, 1);

    run_linear(&cmd, &rnn_proj.w, &w->joint, &w->proj, 1, rnn_proj.n,
               rnn_proj.k, rnn_proj.n, MTP_RNN_SPLIT_K, 0);

    gpu_select_kernel(&cmd, K_markov_gru);
    gpu_bind_buffer(&cmd, 0, &w->proj, 0);
    gpu_bind_buffer(&cmd, 1, &rnn_bias, 0);
    gpu_bind_buffer(&cmd, 2, &w->rnn_state, 0);
    gpu_bind_buffer(&cmd, 3, &w->latent, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(&cmd, 1, 1, 1, (uint32_t)INK2_MTP_MARKOV_RANK, 1, 1);

    run_linear(&cmd, &hidden_out.w, &w->latent, &w->corr, 1, hidden_out.n,
               hidden_out.k, hidden_out.n, 1u, 0);

    gpu_select_kernel(&cmd, K_markov_add);
    gpu_bind_buffer(&cmd, 0, &w->h_final, 0);
    gpu_bind_buffer(&cmd, 1, &w->corr, 0);
    gpu_bind_buffer(&cmd, 2, &hidden_bias, 0);
    gpu_bind_buffer(&cmd, 3, &w->h_eff, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(
        &cmd, ((uint32_t)INK2_DIM + MTP_ELEM_THREADS - 1u) / MTP_ELEM_THREADS,
        1, 1, MTP_ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, K_unembed_argmax_part);
    gpu_bind_buffer(&cmd, 0, unembed_w, 0);
    gpu_bind_buffer(&cmd, 1, &w->h_eff, 0);
    gpu_bind_buffer(&cmd, 2, &w->argmax_parts, 0);
    gpu_copy_bytes(&cmd, 3, &up, sizeof up);
    gpu_run_kernel(&cmd, nparts, 1, 1, MTP_MV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, K_unembed_argmax_final);
    gpu_bind_buffer(&cmd, 0, &w->argmax_parts, 0);
    gpu_bind_buffer(&cmd, 1, &w->tok, (size_t)step * sizeof(int32_t));
    gpu_copy_bytes(&cmd, 2, &nparts, sizeof nparts);
    gpu_run_kernel(&cmd, 1, 1, 1, MTP_ROW_THREADS, 1, 1);
  }

  *last_committed = cmd.number;
  gpu_commit(&cmd);
  sync_fn();
  memcpy(drafts, (const int32_t *)w->tok.ptr + 1, (size_t)n * sizeof(int32_t));
}
