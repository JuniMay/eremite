#include "model/minimax_h3/lora.h"
#include "base/error.h"
#include "engine/safetensors.h"
#include "model/minimax_h3/quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MiniMax-H3 DiT and TokenRefiner projection dimensions across 2 TokenRefiner
// blocks and 50 DiT blocks (624 adapter tensors).
static constexpr int LORA_DIM = 5376, LORA_INNER = 7168,
                     LORA_QKV_WIDTH = MINIMAX_H3_QKV_PARTS * LORA_INNER,
                     LORA_FFN_DIM = 14336, LORA_FC1_WIDTH = 2 * LORA_FFN_DIM;

// Tile and threadgroup sizes matching minimax_h3_dit.metal and
// minimax_h3_quant.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          GRID2D_THREADS = 32;

// Alignment of the LoRA rank dimension required by 64x64 BF16 matmul tiles.
static constexpr int RANK_ALIGN = 64;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Active rows, inner width, input row stride and scale flag.
  uint32_t m, k, lda, has_scale;
} LoraBf16XParam;

// Parameters passed to minimax_h3_mm_bf16_out_bf16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} LoraBf16MMParam;

// Parameters passed to minimax_h3_mm_bf16_add.
typedef struct {
  // Active rows, output width, inner width and output row stride.
  uint32_t m, n, k, ldy;
  // Runtime multiplier applied to the low-rank product before adding into y.
  float scale;
} LoraBf16MMAddParam;

// Parameters for zero-padding a BF16 LoRA matrix to a 64-aligned rank.
typedef struct {
  // Source BF16 tensor [rows_in, cols_in].
  Tensor tensor;
  // Unpadded row and column counts.
  int rows_in, cols_in;
  // Padded row and column counts.
  int rows_out, cols_out;
} LoraPadParam;

// Loaded MiniMax-H3 BF16 LoRA adapter handle.
struct MinimaxH3Lora {
  // Borrowed GPU device.
  Gpu *gpu;
  // Open safetensors file backing the LoRA weights.
  TensorIndex *tensors;
  // Full-residence weight group holding all loaded LoRA matrices.
  WeightGroup *weights;
  // Adapter blocks for the two TokenRefiner layers.
  MinimaxH3LoraBlock refiner[MINIMAX_H3_LORA_REFINER_LAYERS];
  // Adapter blocks for the 50 DiT layers.
  MinimaxH3LoraBlock blocks[MINIMAX_H3_LORA_DIT_LAYERS];
  // Maximum padded rank across all loaded projections.
  int max_rank;
  // Global runtime adapter strength multiplier.
  float strength;
  // Compiled Metal kernel indices.
  int k_x_bf16, k_mm_bf16_out_bf16, k_mm_bf16_add;
};

// Loads a BF16 matrix [rows_in, cols_in] into a zero-padded BF16 buffer
// [rows_out, cols_out].
static void fill_padded_bf16(Buf *buffer, const void *context) {
  const LoraPadParam *p = context;
  uint16_t *raw = xmalloc(p->tensor.size);
  minimax_h3_read_tensor_bytes(p->tensor, raw);
  memset(buffer->ptr, 0,
         (size_t)p->rows_out * (size_t)p->cols_out * sizeof(uint16_t));
  uint16_t *dst = (uint16_t *)buffer->ptr;
  for (int r = 0; r < p->rows_in; r++)
    memcpy(dst + (size_t)r * (size_t)p->cols_out,
           raw + (size_t)r * (size_t)p->cols_in,
           (size_t)p->cols_in * sizeof(uint16_t));
  free(raw);
}

// Reads a scalar alpha tensor (FP32 or BF16) from the LoRA safetensors file.
static float read_alpha_scalar(Tensor t) {
  if (t.dtype == 'f' && t.size == sizeof(float)) {
    float v = 0.0f;
    minimax_h3_read_tensor_bytes(t, &v);
    return v;
  }
  if (t.dtype == 'b' && t.size == sizeof(uint16_t)) {
    uint16_t h = 0;
    minimax_h3_read_tensor_bytes(t, &h);
    uint32_t u = (uint32_t)h << 16;
    float v = 0.0f;
    memcpy(&v, &u, sizeof v);
    return v;
  }
  die("unsupported LoRA alpha tensor dtype %d size %llu", t.dtype,
      (unsigned long long)t.size);
}

// Registers one optional LoRA target (lora_A, lora_B and alpha) under prefix,
// checking both "diffusion_model.<prefix>" and bare "<prefix>" keys.
static int add_lora_target(MinimaxH3Lora *lora, MinimaxH3LoraTarget *target,
                           int n, int k, const char *prefix) {
  memset(target, 0, sizeof *target);
  char full[MINIMAX_H3_NAME_CAP];
  if (tensor_index_has(lora->tensors, "diffusion_model.%s.lora_A.weight",
                       prefix))
    snprintf(full, sizeof full, "diffusion_model.%s", prefix);
  else if (tensor_index_has(lora->tensors, "%s.lora_A.weight", prefix))
    snprintf(full, sizeof full, "%s", prefix);
  else
    return 0;

  Tensor ta = tensor_index_get(lora->tensors, "%s.lora_A.weight", full);
  Tensor tb = tensor_index_get(lora->tensors, "%s.lora_B.weight", full);
  if (ta.dtype != 'b' || ta.ndim != 2 || ta.shape[1] != k || ta.shape[0] <= 0)
    die("unexpected LoRA A shape for %s", full);
  int rank_raw = (int)ta.shape[0];
  if (tb.dtype != 'b' || tb.ndim != 2 || tb.shape[0] != n ||
      tb.shape[1] != rank_raw)
    die("unexpected LoRA B shape for %s", full);

  float alpha = (float)rank_raw;
  if (tensor_index_has(lora->tensors, "%s.alpha", full))
    alpha =
        read_alpha_scalar(tensor_index_get(lora->tensors, "%s.alpha", full));

  int rank_pad = (rank_raw + RANK_ALIGN - 1) / RANK_ALIGN * RANK_ALIGN;
  target->rank = rank_pad;
  target->n = n;
  target->k = k;
  target->present = 1;
  target->scale = alpha / (float)rank_raw;

  if (rank_raw == rank_pad) {
    weight_group_add_tensor(lora->weights, &target->a, ta, 0);
    weight_group_add_tensor(lora->weights, &target->b, tb, 0);
  } else {
    LoraPadParam pa = {ta, rank_raw, k, rank_pad, k};
    LoraPadParam pb = {tb, n, rank_raw, n, rank_pad};
    weight_group_add_generated(lora->weights, &target->a,
                               (size_t)rank_pad * (size_t)k * sizeof(uint16_t),
                               fill_padded_bf16, &pa, sizeof pa);
    weight_group_add_generated(lora->weights, &target->b,
                               (size_t)n * (size_t)rank_pad * sizeof(uint16_t),
                               fill_padded_bf16, &pb, sizeof pb);
  }
  if (rank_pad > lora->max_rank) lora->max_rank = rank_pad;
  return 1;
}

// Registers the four linear targets of one TokenRefiner or DiT block.
static int add_lora_block(MinimaxH3Lora *lora, MinimaxH3LoraBlock *blk,
                          const char *block_prefix) {
  char name[MINIMAX_H3_NAME_CAP];
  int found = 0;
  snprintf(name, sizeof name, "%s.attn.qkv_proj", block_prefix);
  found +=
      add_lora_target(lora, &blk->qkv_proj, LORA_QKV_WIDTH, LORA_DIM, name);
  snprintf(name, sizeof name, "%s.attn.out_proj", block_prefix);
  found += add_lora_target(lora, &blk->out_proj, LORA_DIM, LORA_INNER, name);
  snprintf(name, sizeof name, "%s.mlp.fc1", block_prefix);
  found += add_lora_target(lora, &blk->fc1, LORA_FC1_WIDTH, LORA_DIM, name);
  snprintf(name, sizeof name, "%s.mlp.fc2", block_prefix);
  found += add_lora_target(lora, &blk->fc2, LORA_DIM, LORA_FFN_DIM, name);
  return found;
}

MinimaxH3Lora *minimax_h3_lora_open(Gpu *gpu, MemoryAccount *account,
                                    const char *path, float strength) {
  if (!gpu || !account || !path) die("invalid MiniMax-H3 LoRA open arguments");
  MinimaxH3Lora *lora = xmalloc(sizeof *lora);
  memset(lora, 0, sizeof *lora);
  lora->gpu = gpu;
  lora->strength = strength;
  lora->tensors = tensor_index_open_file(path);
  lora->weights = weight_group_new(gpu, account, "minimax-h3-dit-lora");

  lora->k_x_bf16 = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  lora->k_mm_bf16_out_bf16 =
      gpu_find_kernel(gpu, "minimax_h3_mm_bf16_out_bf16");
  lora->k_mm_bf16_add = gpu_find_kernel(gpu, "minimax_h3_mm_bf16_add");

  int total_found = 0;
  char prefix[MINIMAX_H3_NAME_CAP];
  for (int i = 0; i < MINIMAX_H3_LORA_REFINER_LAYERS; i++) {
    snprintf(prefix, sizeof prefix, "token_refiner.blocks.%d", i);
    total_found += add_lora_block(lora, &lora->refiner[i], prefix);
  }
  for (int i = 0; i < MINIMAX_H3_LORA_DIT_LAYERS; i++) {
    snprintf(prefix, sizeof prefix, "blocks.%d", i);
    total_found += add_lora_block(lora, &lora->blocks[i], prefix);
  }
  if (total_found == 0) die("%s: no MiniMax-H3 LoRA targets found", path);
  return lora;
}

void minimax_h3_lora_close(MinimaxH3Lora *lora) {
  if (!lora) return;
  weight_group_free(lora->weights);
  tensor_index_close(lora->tensors);
  free(lora);
}

WeightGroup *minimax_h3_lora_weights(MinimaxH3Lora *lora) {
  return lora->weights;
}

uint64_t minimax_h3_lora_required_bytes(const MinimaxH3Lora *lora) {
  return lora ? weight_group_required_bytes(lora->weights) : 0;
}

uint64_t minimax_h3_lora_resident_bytes(const MinimaxH3Lora *lora) {
  return lora ? weight_group_resident_bytes(lora->weights) : 0;
}

int minimax_h3_lora_max_rank(const MinimaxH3Lora *lora) {
  return lora ? lora->max_rank : 0;
}

uint64_t minimax_h3_lora_workspace_bytes(const MinimaxH3Lora *lora,
                                         int seq_len) {
  if (!lora || lora->max_rank <= 0 || seq_len <= 0) return 0;
  uint64_t padded =
      ((uint64_t)seq_len + (uint64_t)TILE_M - 1) / (uint64_t)TILE_M * TILE_M;
  return padded * (uint64_t)lora->max_rank * sizeof(uint16_t);
}

int minimax_h3_lora_load(MinimaxH3Lora *lora) {
  return lora ? weight_group_load(lora->weights) : 1;
}

const MinimaxH3LoraBlock *
minimax_h3_lora_refiner_block(const MinimaxH3Lora *lora, int index) {
  if (!lora || index < 0 || index >= MINIMAX_H3_LORA_REFINER_LAYERS)
    return NULL;
  return &lora->refiner[index];
}

const MinimaxH3LoraBlock *minimax_h3_lora_dit_block(const MinimaxH3Lora *lora,
                                                    int index) {
  if (!lora || index < 0 || index >= MINIMAX_H3_LORA_DIT_LAYERS) return NULL;
  return &lora->blocks[index];
}

void minimax_h3_lora_apply(const MinimaxH3Lora *lora, GpuCommandBuffer *cmd,
                           const MinimaxH3LoraTarget *target, const Buf *x_f32,
                           const Buf *xb, const Buf *lora_mid, const Buf *y,
                           int rows, int xb_ready) {
  if (!lora || !target || !target->present || rows <= 0 ||
      lora->strength == 0.0f)
    return;
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;

  if (!xb_ready) {
    LoraBf16XParam xp = {(uint32_t)rows, (uint32_t)target->k,
                         (uint32_t)target->k, 0};
    gpu_select_kernel(cmd, lora->k_x_bf16);
    gpu_bind_buffer(cmd, 0, x_f32, 0);
    gpu_bind_buffer(cmd, 1, &target->a, 0);
    gpu_bind_buffer(cmd, 2, xb, 0);
    gpu_copy_bytes(cmd, 3, &xp, sizeof xp);
    gpu_run_kernel(
        cmd,
        ((uint32_t)target->k / MINIMAX_H3_VEC4_WIDTH + GRID2D_THREADS - 1) /
            GRID2D_THREADS,
        padded, 1, GRID2D_THREADS, 1, 1);
  }

  LoraBf16MMParam ap = {(uint32_t)rows, (uint32_t)target->rank,
                        (uint32_t)target->k, (uint32_t)target->rank, 0};
  gpu_select_kernel(cmd, lora->k_mm_bf16_out_bf16);
  gpu_bind_buffer(cmd, 0, &target->a, 0);
  gpu_bind_buffer(cmd, 1, xb, 0);
  gpu_bind_buffer(cmd, 2, lora_mid, 0);
  gpu_copy_bytes(cmd, 3, &ap, sizeof ap);
  gpu_run_kernel(cmd, ((uint32_t)target->rank + TILE_N - 1) / TILE_N,
                 padded / TILE_M, 1, MM_THREADS, 1, 1);

  LoraBf16MMAddParam bp = {(uint32_t)rows, (uint32_t)target->n,
                           (uint32_t)target->rank, (uint32_t)target->n,
                           target->scale * lora->strength};
  gpu_select_kernel(cmd, lora->k_mm_bf16_add);
  gpu_bind_buffer(cmd, 0, &target->b, 0);
  gpu_bind_buffer(cmd, 1, lora_mid, 0);
  gpu_bind_buffer(cmd, 2, y, 0);
  gpu_copy_bytes(cmd, 3, &bp, sizeof bp);
  gpu_run_kernel(cmd, ((uint32_t)target->n + TILE_N - 1) / TILE_N,
                 padded / TILE_M, 1, MM_THREADS, 1, 1);
}
