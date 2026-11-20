#include "model/minimax_h3/text_encoder.h"
#include "base/error.h"
#include "engine/tokenizer.h"
#include "model/minimax_h3/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Attention and feed-forward dimensions of Qwen3-VL-32B.
static constexpr int QWEN_VOCAB = 151936, QWEN_HEADS = 64, QWEN_KV_HEADS = 8,
                     QWEN_HEAD_DIM = 128,
                     QWEN_Q_WIDTH = QWEN_HEADS * QWEN_HEAD_DIM,
                     QWEN_KV_WIDTH = QWEN_KV_HEADS * QWEN_HEAD_DIM,
                     QWEN_FFN_DIM = 25600, QWEN_ROPE_PAIRS = QWEN_HEAD_DIM / 2;

// RMSNorm epsilon and RoPE base frequency for Qwen3-VL-32B.
static constexpr float QWEN_RMS_EPS = 1e-6f;
static constexpr double QWEN_ROPE_THETA = 5000000.0;

// Maximum token count accepted when tokenizing a text prompt.
static constexpr int QWEN_MAX_PROMPT_TOKENS = 4096;

// Tile and threadgroup sizes matching minimax_h3_quant.metal and
// minimax_h3_qwen.metal.
static constexpr uint32_t TILE_M = 128, TILE_N = 64, ATTN_PAD = 64,
                          MM_THREADS = 128, NORM_THREADS = 256,
                          ELEM_THREADS = 256, BF16X_THREADS = 32,
                          EMBED_THREADS = 64;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Active rows, inner width, input row stride and pre_quant_scale flag.
  uint32_t m, k, lda, has_scale;
} Bf16XParam;

// Parameters passed to minimax_h3_mm_nvfp4.
typedef struct {
  // Active rows, output width, inner width and output row stride.
  uint32_t m, n, k, ldy;
  // Per-tensor NVFP4 weight_scale_2 scalar.
  float scale2;
} Nvfp4MMParam;

// Parameters passed to minimax_h3_rmsnorm.
typedef struct {
  // Row length in floats.
  uint32_t n;
  // RMSNorm epsilon.
  float eps;
} RmsNormParam;

// Parameters passed to minimax_h3_add_indexed.
typedef struct {
  // Number of indexed rows and row width in floats.
  uint32_t count, dim;
} AddIndexedParam;

// Parameters passed to minimax_h3_qwen_heads.
typedef struct {
  // Active sequence length and padded head stride in rows.
  uint32_t rows, padded;
  // Per-head QK RMSNorm epsilon.
  float eps;
} QwenHeadsParam;

// Parameters passed to minimax_h3_qwen_attend.
typedef struct {
  // Active sequence length and padded head stride in rows.
  uint32_t rows, padded;
} QwenAttendParam;

// One NVFP4 + optional AWQ pre_quant_scale linear projection.
typedef struct {
  // Packed hi-nibble-first FP4 E2M1 weight bytes (n by k / 2), sector-aligned.
  Buf w;
  // Unswizzled row-major FP8 E4M3 block-16 scales (n by k / 16).
  Buf scale;
  // Optional BF16 per-channel activation pre_quant_scale (k).
  Buf pre_scale;
  // Per-tensor FP32 global weight scale.
  float scale2;
  // Output width, inner width and whether pre_scale is present.
  int n, k, has_pre_scale;
} Nvfp4Lin;

// Weights for one Qwen3-VL-32B transformer layer.
typedef struct {
  // Pre-attention and pre-MLP RMSNorm weights (FP32, 5120).
  Buf input_norm, post_norm;
  // Per-head Q and K RMSNorm weights (FP32, 128).
  Buf q_norm, k_norm;
  // Attention projections (q: 8192x5120, k/v: 1024x5120, o: 5120x8192).
  Nvfp4Lin q_proj, k_proj, v_proj, o_proj;
  // SwiGLU MLP projections (gate/up: 25600x5120, down: 5120x25600).
  Nvfp4Lin gate_proj, up_proj, down_proj;
} QwenLayer;

// Transient workspace buffers allocated for one encode_tokens call.
typedef struct {
  // Token IDs (seq_len) and 3D MRoPE position IDs (seq_len by 3).
  Buf ids, pos_ids;
  // RMSNorm and output projection staging (seq_len by 5120).
  Buf norm;
  // Padded BF16 activation staging (padded_m by 25600).
  Buf xb;
  // Attention Q/O (seq_len by 8192), K and V (seq_len by 1024).
  Buf q, k, v;
  // Head-major BF16 Q (64 heads), K and V (8 heads each).
  Buf qh, kh, vh;
  // SwiGLU MLP intermediates (seq_len by 25600 each).
  Buf gate, up;
  // Optional DeepStack row indices (visual_count).
  Buf visual_indices;
} TextWorkspace;

// MiniMax-H3 Qwen3-VL-32B text encoder state and weights.
struct MinimaxH3TextEncoder {
  // GPU device and memory account owning the encoder's buffers.
  Gpu *gpu;
  // Memory account that bounds both weights and transient workspace.
  MemoryAccount *account;
  // Non-evictable slot tracking transient encode buffers.
  MemorySlot workspace_slot;
  // Open safetensors checkpoint file.
  TensorIndex *tensors;
  // Byte-level BPE tokenizer from the model's tokenizer directory.
  Tokenizer *tokenizer;
  // Full-residence weight group for embeddings, RoPE table and 50 layers.
  WeightGroup *weights;
  // Row-scaled INT8 token embedding table (151936 by 5120) and FP32 scales.
  Buf embed_w, embed_scale;
  // Precomputed RoPE inverse frequencies (FP32, 64).
  Buf inv_freq;
  // Transformer layers 0 through 49.
  QwenLayer layers[MINIMAX_H3_TEXT_LAYERS];
  // Compiled Metal kernel indices.
  int k_embed, k_rmsnorm, k_swiglu, k_add, k_add_indexed;
  int k_x_bf16, k_mm_nvfp4, k_heads, k_attend;
};

// Generates the 64 FP32 RoPE inverse frequencies for theta = 5,000,000.
static void fill_inv_freq(Buf *buffer, const void *context) {
  (void)context;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < QWEN_ROPE_PAIRS; i++)
    out[i] = (float)(1.0 / pow(QWEN_ROPE_THETA,
                               (double)(2 * i) / (double)QWEN_HEAD_DIM));
}

// Registers one NVFP4 linear projection from tensors into group.
static void add_nvfp4_lin(WeightGroup *group, const TensorIndex *tensors,
                          Nvfp4Lin *lin, int n, int k, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor s = tensor_index_get(tensors, "%s.weight_scale", prefix);
  Tensor s2 = tensor_index_get(tensors, "%s.weight_scale_2", prefix);
  if (w.dtype != 'u' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k / 2)
    die("unexpected NVFP4 weight shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  lin->scale2 = minimax_h3_read_scalar_f32(s2);
  weight_group_add_tensor(group, &lin->w, w, 0);
  minimax_h3_nvfp4_scales_add(group, &lin->scale, s, n,
                              k / MINIMAX_H3_NVFP4_GROUP);
  lin->has_pre_scale = tensor_index_has(tensors, "%s.pre_quant_scale", prefix);
  if (lin->has_pre_scale) {
    Tensor ps = tensor_index_get(tensors, "%s.pre_quant_scale", prefix);
    if (ps.dtype != 'b' || ps.ndim != 1 || ps.shape[0] != k)
      die("unexpected pre_quant_scale shape for %s", prefix);
    weight_group_add_tensor(group, &lin->pre_scale, ps, 0);
  }
}

MinimaxH3TextEncoder *minimax_h3_text_encoder_open(Gpu *gpu,
                                                   MemoryAccount *account,
                                                   const char *weights_path,
                                                   const char *tokenizer_dir) {
  MinimaxH3TextEncoder *enc = xmalloc(sizeof *enc);
  enc->gpu = gpu;
  enc->account = account;
  memory_slot_init(&enc->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-text-workspace", NULL, NULL);
  enc->tensors = tensor_index_open_file(weights_path);
  enc->tokenizer = tokenizer_open(tokenizer_dir, NULL);
  enc->weights = weight_group_new(gpu, account, "minimax-h3-text-encoder");

  enc->k_embed = gpu_find_kernel(gpu, "minimax_h3_embed_int8");
  enc->k_rmsnorm = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  enc->k_swiglu = gpu_find_kernel(gpu, "minimax_h3_swiglu");
  enc->k_add = gpu_find_kernel(gpu, "minimax_h3_add");
  enc->k_add_indexed = gpu_find_kernel(gpu, "minimax_h3_add_indexed");
  enc->k_x_bf16 = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  enc->k_mm_nvfp4 = gpu_find_kernel(gpu, "minimax_h3_mm_nvfp4");
  enc->k_heads = gpu_find_kernel(gpu, "minimax_h3_qwen_heads");
  enc->k_attend = gpu_find_kernel(gpu, "minimax_h3_qwen_attend");

  Tensor ew = tensor_index_get(enc->tensors, "model.embed_tokens.weight");
  Tensor es = tensor_index_get(enc->tensors, "model.embed_tokens.weight_scale");
  if (ew.dtype != 'i' || ew.ndim != 2 || ew.shape[0] != QWEN_VOCAB ||
      ew.shape[1] != MINIMAX_H3_TEXT_DIM)
    die("unexpected Qwen3-VL embed_tokens.weight shape");
  if (es.dtype != 'f' || es.shape[0] != QWEN_VOCAB)
    die("unexpected Qwen3-VL embed_tokens.weight_scale shape");
  weight_group_add_tensor(enc->weights, &enc->embed_w, ew, 0);
  weight_group_add_tensor(enc->weights, &enc->embed_scale, es, 0);
  weight_group_add_generated(enc->weights, &enc->inv_freq,
                             (size_t)QWEN_ROPE_PAIRS * sizeof(float),
                             fill_inv_freq, NULL, 0);

  for (int i = 0; i < MINIMAX_H3_TEXT_LAYERS; i++) {
    QwenLayer *l = &enc->layers[i];
    weight_group_add_vector(
        enc->weights, &l->input_norm,
        tensor_index_get(enc->tensors, "model.layers.%d.input_layernorm.weight",
                         i),
        0.0f, 0);
    weight_group_add_vector(
        enc->weights, &l->post_norm,
        tensor_index_get(enc->tensors,
                         "model.layers.%d.post_attention_layernorm.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        enc->weights, &l->q_norm,
        tensor_index_get(enc->tensors,
                         "model.layers.%d.self_attn.q_norm.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        enc->weights, &l->k_norm,
        tensor_index_get(enc->tensors,
                         "model.layers.%d.self_attn.k_norm.weight", i),
        0.0f, 0);

    char name[128];
    snprintf(name, sizeof name, "model.layers.%d.self_attn.q_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->q_proj, QWEN_Q_WIDTH,
                  MINIMAX_H3_TEXT_DIM, name);
    snprintf(name, sizeof name, "model.layers.%d.self_attn.k_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->k_proj, QWEN_KV_WIDTH,
                  MINIMAX_H3_TEXT_DIM, name);
    snprintf(name, sizeof name, "model.layers.%d.self_attn.v_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->v_proj, QWEN_KV_WIDTH,
                  MINIMAX_H3_TEXT_DIM, name);
    snprintf(name, sizeof name, "model.layers.%d.self_attn.o_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->o_proj, MINIMAX_H3_TEXT_DIM,
                  QWEN_Q_WIDTH, name);
    snprintf(name, sizeof name, "model.layers.%d.mlp.gate_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->gate_proj, QWEN_FFN_DIM,
                  MINIMAX_H3_TEXT_DIM, name);
    snprintf(name, sizeof name, "model.layers.%d.mlp.up_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->up_proj, QWEN_FFN_DIM,
                  MINIMAX_H3_TEXT_DIM, name);
    snprintf(name, sizeof name, "model.layers.%d.mlp.down_proj", i);
    add_nvfp4_lin(enc->weights, enc->tensors, &l->down_proj,
                  MINIMAX_H3_TEXT_DIM, QWEN_FFN_DIM, name);
  }
  return enc;
}

void minimax_h3_text_encoder_close(MinimaxH3TextEncoder *enc) {
  if (!enc) return;
  weight_group_free(enc->weights);
  memory_slot_remove(&enc->workspace_slot);
  tokenizer_close(enc->tokenizer);
  tensor_index_close(enc->tensors);
  free(enc);
}

const TensorIndex *
minimax_h3_text_encoder_tensors(const MinimaxH3TextEncoder *enc) {
  return enc->tensors;
}

Gpu *minimax_h3_text_encoder_gpu(const MinimaxH3TextEncoder *enc) {
  return enc->gpu;
}

WeightGroup *minimax_h3_text_encoder_weights(MinimaxH3TextEncoder *enc) {
  return enc->weights;
}

uint64_t
minimax_h3_text_encoder_required_bytes(const MinimaxH3TextEncoder *enc) {
  return weight_group_required_bytes(enc->weights);
}

uint64_t
minimax_h3_text_encoder_resident_bytes(const MinimaxH3TextEncoder *enc) {
  return weight_group_resident_bytes(enc->weights);
}

int minimax_h3_text_encoder_load(MinimaxH3TextEncoder *enc) {
  return weight_group_load(enc->weights);
}

int minimax_h3_text_encoder_tokenize(const MinimaxH3TextEncoder *enc,
                                     const char *prompt, int *ids,
                                     int max_ids) {
  if (max_ids <= 0) return 0;
  int n = prompt && prompt[0]
              ? tokenizer_encode_text(enc->tokenizer, prompt, ids, max_ids)
              : 0;
  if (n == 0) {
    ids[0] = MINIMAX_H3_TEXT_PAD_TOKEN;
    n = 1;
  }
  return n;
}

// Frees all buffers held by a transient workspace.
static void free_workspace(TextWorkspace *ws) {
  gpu_free(&ws->visual_indices);
  gpu_free(&ws->up);
  gpu_free(&ws->gate);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->v);
  gpu_free(&ws->k);
  gpu_free(&ws->q);
  gpu_free(&ws->xb);
  gpu_free(&ws->norm);
  gpu_free(&ws->pos_ids);
  gpu_free(&ws->ids);
}

// Allocates transient GPU buffers for encoding seq_len tokens. Returns 0 if
// memory does not fit.
static int alloc_workspace(MinimaxH3TextEncoder *enc, int seq_len,
                           int visual_count, TextWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  uint32_t padded_m = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t padded_attn =
      ((uint32_t)seq_len + ATTN_PAD - 1) / ATTN_PAD * ATTN_PAD;
  MemorySlot *slot = &enc->workspace_slot;
  ws->ids =
      gpu_try_alloc_required(enc->gpu, slot, (size_t)seq_len * sizeof(int));
  ws->pos_ids =
      gpu_try_alloc_required(enc->gpu, slot, (size_t)seq_len * 3 * sizeof(int));
  ws->norm = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * MINIMAX_H3_TEXT_DIM * sizeof(float));
  ws->xb = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)padded_m * QWEN_FFN_DIM * sizeof(uint16_t));
  ws->q = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * QWEN_Q_WIDTH * sizeof(float));
  ws->k = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * QWEN_KV_WIDTH * sizeof(float));
  ws->v = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * QWEN_KV_WIDTH * sizeof(float));
  ws->qh = gpu_try_alloc_required(enc->gpu, slot,
                                  (size_t)QWEN_HEADS * padded_attn *
                                      QWEN_HEAD_DIM * sizeof(uint16_t));
  ws->kh = gpu_try_alloc_required(enc->gpu, slot,
                                  (size_t)QWEN_KV_HEADS * padded_attn *
                                      QWEN_HEAD_DIM * sizeof(uint16_t));
  ws->vh = gpu_try_alloc_required(enc->gpu, slot,
                                  (size_t)QWEN_KV_HEADS * padded_attn *
                                      QWEN_HEAD_DIM * sizeof(uint16_t));
  ws->gate = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * QWEN_FFN_DIM * sizeof(float));
  ws->up = gpu_try_alloc_required(
      enc->gpu, slot, (size_t)seq_len * QWEN_FFN_DIM * sizeof(float));
  if (visual_count > 0)
    ws->visual_indices = gpu_try_alloc_required(
        enc->gpu, slot, (size_t)visual_count * sizeof(int));
  if (!ws->ids.obj || !ws->pos_ids.obj || !ws->norm.obj || !ws->xb.obj ||
      !ws->q.obj || !ws->k.obj || !ws->v.obj || !ws->qh.obj || !ws->kh.obj ||
      !ws->vh.obj || !ws->gate.obj || !ws->up.obj ||
      (visual_count > 0 && !ws->visual_indices.obj)) {
    free_workspace(ws);
    return 0;
  }
  return 1;
}

// Encodes FP32 activations x into padded BF16 rows in xb (applying optional
// pre_quant_scale when lin->has_pre_scale is set).
static void stage_bf16_x(const MinimaxH3TextEncoder *enc, GpuCommandBuffer *cmd,
                         const Buf *x, const Buf *xb, const Nvfp4Lin *lin,
                         int seq_len) {
  uint32_t padded_m = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  Bf16XParam xp = {(uint32_t)seq_len, (uint32_t)lin->k, (uint32_t)lin->k,
                   (uint32_t)lin->has_pre_scale};
  const Buf *scale_buf = lin->has_pre_scale ? &lin->pre_scale : &enc->inv_freq;
  gpu_select_kernel(cmd, enc->k_x_bf16);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, scale_buf, 0);
  gpu_bind_buffer(cmd, 2, xb, 0);
  gpu_copy_bytes(cmd, 3, &xp, sizeof xp);
  gpu_run_kernel(cmd,
                 ((uint32_t)lin->k / 4u + BF16X_THREADS - 1) / BF16X_THREADS,
                 padded_m, 1, BF16X_THREADS, 1, 1);
}

// Dispatches an NVFP4 linear projection from pre-staged BF16 activations xb.
static void run_nvfp4_from_xb(const MinimaxH3TextEncoder *enc,
                              GpuCommandBuffer *cmd, const Buf *xb,
                              const Nvfp4Lin *lin, const Buf *y, int seq_len) {
  Nvfp4MMParam mp = {(uint32_t)seq_len, (uint32_t)lin->n, (uint32_t)lin->k,
                     (uint32_t)lin->n, lin->scale2};
  gpu_select_kernel(cmd, enc->k_mm_nvfp4);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, &lin->scale, 0);
  gpu_bind_buffer(cmd, 2, xb, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N,
                 ((uint32_t)seq_len + TILE_M - 1) / TILE_M, 1, MM_THREADS, 1,
                 1);
}

// Encodes RMSNorm of seq_len rows of MINIMAX_H3_TEXT_DIM floats.
static void run_rmsnorm(const MinimaxH3TextEncoder *enc, GpuCommandBuffer *cmd,
                        const Buf *x, const Buf *w, const Buf *y, int seq_len) {
  RmsNormParam p = {(uint32_t)MINIMAX_H3_TEXT_DIM, QWEN_RMS_EPS};
  gpu_select_kernel(cmd, enc->k_rmsnorm);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, w, 0);
  gpu_bind_buffer(cmd, 2, y, 0);
  gpu_copy_bytes(cmd, 3, &p, sizeof p);
  gpu_run_kernel(cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);
}

// Encodes in-place residual addition x += y for count floats.
static void run_add(const MinimaxH3TextEncoder *enc, GpuCommandBuffer *cmd,
                    const Buf *x, const Buf *y, uint32_t count) {
  gpu_select_kernel(cmd, enc->k_add);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, y, 0);
  gpu_copy_bytes(cmd, 2, &count, sizeof count);
  gpu_run_kernel(cmd, (count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
}

int minimax_h3_text_encoder_encode_tokens(
    MinimaxH3TextEncoder *enc, const int *ids, const int *pos_ids,
    const Buf *embeds_override,
    const Buf *const deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS],
    const int *visual_indices, int visual_count, int seq_len, Buf *out_hidden) {
  if (seq_len <= 0) die("text encoder requires at least one token");
  size_t hidden_bytes =
      (size_t)seq_len * (size_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  if (!out_hidden || out_hidden->size < hidden_bytes)
    die("text encoder output buffer is too small");
  if (!minimax_h3_text_encoder_load(enc)) return 0;

  MemorySlot *wslot = weight_group_slot(enc->weights);
  memory_pin(wslot, 1);
  TextWorkspace ws;
  if (!alloc_workspace(enc, seq_len, visual_count, &ws)) {
    memory_pin(wslot, 0);
    return 0;
  }

  memcpy(ws.ids.ptr, ids, (size_t)seq_len * sizeof(int));
  int *pos_dst = (int *)ws.pos_ids.ptr;
  if (pos_ids) {
    memcpy(pos_dst, pos_ids, (size_t)seq_len * 3 * sizeof(int));
  } else {
    for (int i = 0; i < seq_len; i++) {
      pos_dst[3 * i] = i;
      pos_dst[3 * i + 1] = i;
      pos_dst[3 * i + 2] = i;
    }
  }
  if (visual_count > 0 && visual_indices)
    memcpy(ws.visual_indices.ptr, visual_indices,
           (size_t)visual_count * sizeof(int));
  if (embeds_override)
    memcpy(out_hidden->ptr, embeds_override->ptr, hidden_bytes);

  uint32_t padded_attn =
      ((uint32_t)seq_len + ATTN_PAD - 1) / ATTN_PAD * ATTN_PAD;
  uint32_t dim = (uint32_t)MINIMAX_H3_TEXT_DIM;
  uint32_t hidden_count = (uint32_t)seq_len * dim;
  uint32_t ffn_count = (uint32_t)seq_len * (uint32_t)QWEN_FFN_DIM;

  // Token embedding lookup (skips negative IDs pre-filled by embeds_override).
  GpuCommandBuffer cmd = gpu_begin(enc->gpu);
  gpu_select_kernel(&cmd, enc->k_embed);
  gpu_bind_buffer(&cmd, 0, &enc->embed_w, 0);
  gpu_bind_buffer(&cmd, 1, &enc->embed_scale, 0);
  gpu_bind_buffer(&cmd, 2, &ws.ids, 0);
  gpu_bind_buffer(&cmd, 3, out_hidden, 0);
  gpu_copy_bytes(&cmd, 4, &dim, sizeof dim);
  gpu_run_kernel(&cmd, dim / EMBED_THREADS, (uint32_t)seq_len, 1, EMBED_THREADS,
                 1, 1);
  gpu_commit(&cmd);
  uint64_t prev_cmd = cmd.number;

  // Run layers 0 through 49, yielding cooperatively between layers.
  for (int i = 0; i < MINIMAX_H3_TEXT_LAYERS; i++) {
    const QwenLayer *l = &enc->layers[i];
    cmd = gpu_begin(enc->gpu);

    // Self-attention: RMSNorm -> Q/K/V -> QK-Norm + MRoPE -> causal GQA -> O.
    run_rmsnorm(enc, &cmd, out_hidden, &l->input_norm, &ws.norm, seq_len);
    stage_bf16_x(enc, &cmd, &ws.norm, &ws.xb, &l->q_proj, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->q_proj, &ws.q, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->k_proj, &ws.k, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->v_proj, &ws.v, seq_len);

    QwenHeadsParam hp = {(uint32_t)seq_len, padded_attn, QWEN_RMS_EPS};
    gpu_select_kernel(&cmd, enc->k_heads);
    gpu_bind_buffer(&cmd, 0, &ws.q, 0);
    gpu_bind_buffer(&cmd, 1, &ws.k, 0);
    gpu_bind_buffer(&cmd, 2, &ws.v, 0);
    gpu_bind_buffer(&cmd, 3, &l->q_norm, 0);
    gpu_bind_buffer(&cmd, 4, &l->k_norm, 0);
    gpu_bind_buffer(&cmd, 5, &enc->inv_freq, 0);
    gpu_bind_buffer(&cmd, 6, &ws.pos_ids, 0);
    gpu_bind_buffer(&cmd, 7, &ws.qh, 0);
    gpu_bind_buffer(&cmd, 8, &ws.kh, 0);
    gpu_bind_buffer(&cmd, 9, &ws.vh, 0);
    gpu_copy_bytes(&cmd, 10, &hp, sizeof hp);
    gpu_run_kernel(&cmd, (uint32_t)(QWEN_HEADS + 2 * QWEN_KV_HEADS),
                   padded_attn, 1, (uint32_t)QWEN_ROPE_PAIRS, 1, 1);

    QwenAttendParam ap = {(uint32_t)seq_len, padded_attn};
    gpu_select_kernel(&cmd, enc->k_attend);
    gpu_bind_buffer(&cmd, 0, &ws.qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws.kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws.vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws.q, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, ((uint32_t)seq_len + ATTN_PAD - 1) / ATTN_PAD,
                   (uint32_t)QWEN_HEADS, 1, MM_THREADS, 1, 1);

    stage_bf16_x(enc, &cmd, &ws.q, &ws.xb, &l->o_proj, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->o_proj, &ws.norm, seq_len);
    run_add(enc, &cmd, out_hidden, &ws.norm, hidden_count);

    // Feed-forward: RMSNorm -> gate/up -> SwiGLU -> down -> residual add.
    run_rmsnorm(enc, &cmd, out_hidden, &l->post_norm, &ws.norm, seq_len);
    stage_bf16_x(enc, &cmd, &ws.norm, &ws.xb, &l->gate_proj, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->gate_proj, &ws.gate, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->up_proj, &ws.up, seq_len);

    gpu_select_kernel(&cmd, enc->k_swiglu);
    gpu_bind_buffer(&cmd, 0, &ws.gate, 0);
    gpu_bind_buffer(&cmd, 1, &ws.up, 0);
    gpu_bind_buffer(&cmd, 2, &ws.gate, 0);
    gpu_copy_bytes(&cmd, 3, &ffn_count, sizeof ffn_count);
    gpu_run_kernel(&cmd, (ffn_count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);

    stage_bf16_x(enc, &cmd, &ws.gate, &ws.xb, &l->down_proj, seq_len);
    run_nvfp4_from_xb(enc, &cmd, &ws.xb, &l->down_proj, &ws.norm, seq_len);
    run_add(enc, &cmd, out_hidden, &ws.norm, hidden_count);

    // DeepStack: add visual features at image token positions after layers
    // 0..2.
    if (i < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS && deepstack && deepstack[i] &&
        visual_count > 0) {
      AddIndexedParam dp = {(uint32_t)visual_count, dim};
      gpu_select_kernel(&cmd, enc->k_add_indexed);
      gpu_bind_buffer(&cmd, 0, out_hidden, 0);
      gpu_bind_buffer(&cmd, 1, deepstack[i], 0);
      gpu_bind_buffer(&cmd, 2, &ws.visual_indices, 0);
      gpu_copy_bytes(&cmd, 3, &dp, sizeof dp);
      gpu_run_kernel(&cmd, dim / EMBED_THREADS, (uint32_t)visual_count, 1,
                     EMBED_THREADS, 1, 1);
    }

    gpu_commit(&cmd);
    minimax_h3_gpu_wait(enc->gpu, prev_cmd);
    prev_cmd = cmd.number;
  }

  minimax_h3_gpu_wait(enc->gpu, prev_cmd);
  free_workspace(&ws);
  memory_pin(wslot, 0);
  memory_touch(wslot);
  return 1;
}

int minimax_h3_text_encoder_encode_prompt(MinimaxH3TextEncoder *enc,
                                          const char *prompt,
                                          MemorySlot *out_slot, Buf *out_hidden,
                                          int *out_seq_len) {
  int ids[QWEN_MAX_PROMPT_TOKENS];
  int seq_len = minimax_h3_text_encoder_tokenize(enc, prompt, ids,
                                                 QWEN_MAX_PROMPT_TOKENS);
  size_t bytes = (size_t)seq_len * (size_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  Buf hidden = gpu_try_alloc_required(enc->gpu, out_slot, bytes);
  if (!hidden.obj) return 0;
  if (!minimax_h3_text_encoder_encode_tokens(enc, ids, NULL, NULL, NULL, NULL,
                                             0, seq_len, &hidden)) {
    gpu_free(&hidden);
    return 0;
  }
  *out_hidden = hidden;
  *out_seq_len = seq_len;
  return 1;
}
