#include "model/minimax_h3/dit.h"
#include "base/error.h"
#include "base/json.h"
#include "engine/safetensors.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MiniMax-H3-Image DiT architecture constants matching transformer/config.json.
static constexpr int DIT_DIM = 5376, DIT_LAYERS = 50, DIT_REFINER_LAYERS = 2,
                     DIT_HEADS = 56, DIT_HEAD_DIM = 128,
                     DIT_INNER = DIT_HEADS * DIT_HEAD_DIM,
                     DIT_QKV_WIDTH = 3 * DIT_INNER, DIT_FFN_DIM = 14336,
                     DIT_FC1_WIDTH = 2 * DIT_FFN_DIM, DIT_PATCH_SIDE = 2,
                     DIT_PATCH_DIM = MINIMAX_H3_VAE_LATENT_CHANNELS *
                                     DIT_PATCH_SIDE * DIT_PATCH_SIDE,
                     DIT_ROPE_FREQS = 16, DIT_ADALN_GRID = 1025,
                     DIT_ADALN_RANK = 8, DIT_ADALN_MODALITIES = 3,
                     DIT_ADALN_CHUNKS = 6,
                     DIT_ADALN_STRIDE = DIT_ADALN_CHUNKS * DIT_DIM,
                     DIT_ADALN_OUT = DIT_ADALN_MODALITIES * DIT_ADALN_STRIDE,
                     DIT_FINAL_ADALN_OUT = 2 * DIT_DIM;

// Default PDD step count when checkpoint metadata omits one.
static constexpr int DEFAULT_PDD_STEPS = 6;

// Default 6-step PDD decreasing sigma schedule when metadata omits one.
static const double DEFAULT_PDD_SIGMAS[DEFAULT_PDD_STEPS + 1] = {
    1.0,  0.9675236940383911, 0.8875485062599182,
    0.75, 0.5327739119529724, 0.23200887441635132,
    0.0,
};

// RMSNorm epsilon matching transformer/config.json.
static constexpr float DIT_NORM_EPS = 1e-5f;

// Spatial RoPE coordinate scale over area-normalized patch grids.
static constexpr double DIT_ROPE_SPATIAL_SCALE = 32.0;

// Noise-augmentation timestep pinned on conditioning reference images.
static constexpr double DIT_VISUAL_COND_TIMESTEP = 0.999;

// AdaLN modality tags: visual (0) and text (1).
static constexpr uint32_t TAG_VISUAL = 0, TAG_TEXT = 1;

// Offsets of the 6 modulation vectors inside one modality's AdaLN slice.
static constexpr uint32_t OFF_SHIFT_MSA = 0 * DIT_DIM,
                          OFF_SCALE_MSA = 1 * DIT_DIM,
                          OFF_GATE_MSA = 2 * DIT_DIM,
                          OFF_SHIFT_MLP = 3 * DIT_DIM,
                          OFF_SCALE_MLP = 4 * DIT_DIM,
                          OFF_GATE_MLP = 5 * DIT_DIM;

// Threadgroup and tile sizes matching minimax_h3_quant.metal and
// minimax_h3_dit.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          CONVROT_THREADS = 256, NORM_THREADS = 256,
                          HEADS_THREADS = 64, F32MM_THREADS = 32,
                          ELEM_THREADS = 256, GRID2D_THREADS = 32;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Active rows, inner width, input row stride and scale flag.
  uint32_t m, k, lda, has_scale;
} Bf16XParam;

// Parameters passed to minimax_h3_mm_bf16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} Bf16MMParam;

// Parameters passed to minimax_h3_mm_f32.
typedef struct {
  // Active rows, output width, inner width and output row stride.
  uint32_t m, n, k, ldy;
} F32MMParam;

// Parameters passed to minimax_h3_convrot_quant_i8.
typedef struct {
  // Active rows, inner width and input row stride.
  uint32_t m, k, lda;
} ConvRotParam;

// Parameters passed to minimax_h3_mm_int8.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} Int8MMParam;

// Parameters passed to minimax_h3_rmsnorm.
typedef struct {
  // Row width and epsilon.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} RmsNormParam;

// Parameters passed to minimax_h3_swiglu_chunked.
typedef struct {
  // Rows and output half-width.
  uint32_t rows, inner;
} SwigluChunkedParam;

// Parameters passed to minimax_h3_dit_adaln.
typedef struct {
  // Number of distinct timesteps and output vector width per timestep.
  uint32_t num_t, out_dim;
} DitAdalnParam;

// Parameters passed to minimax_h3_dit_norm_mod.
typedef struct {
  // Row width, modulation stride, shift offset and scale offset.
  uint32_t dim, mod_stride, shift_off, scale_off;
  // RMSNorm epsilon.
  float eps;
} DitNormModParam;

// Parameters passed to minimax_h3_dit_gate_add.
typedef struct {
  // Rows, row width, modulation stride and gate offset.
  uint32_t rows, dim, mod_stride, gate_off;
} DitGateAddParam;

// Parameters passed to minimax_h3_dit_heads.
typedef struct {
  // Active sequence length, padded head stride and RoPE flag.
  uint32_t rows, padded, use_rope;
  // Per-head affine QK RMSNorm epsilon.
  float eps;
} DitHeadsParam;

// Parameters passed to minimax_h3_dit_attend.
typedef struct {
  // Active sequence length and padded head stride.
  uint32_t rows, padded;
} DitAttendParam;

// One BF16 linear projection (with optional FP32 bias).
typedef struct {
  // BF16 weight matrix (n by k), loaded page-aligned.
  Buf w;
  // Optional FP32 bias vector (n).
  Buf bias;
  // Output width, inner width and whether bias is present.
  int n, k, has_bias;
} Bf16Lin;

// One INT8-ConvRot linear projection (bias-free in MiniMax-H3 DiT).
typedef struct {
  // INT8 rotated weight matrix (n by k), loaded page-aligned.
  Buf w;
  // Per-output-row FP32 weight scale (n).
  Buf scale;
  // Output width and inner width.
  int n, k;
} Int8Lin;

// Weights for one BF16 TokenRefiner block.
typedef struct {
  // Pre-attention and pre-MLP RMSNorm weights (FP32, 5376).
  Buf norm1, norm2;
  // Per-head Q and K RMSNorm weights (FP32, 128).
  Buf q_norm, k_norm;
  // Attention QKV (21504x5376) and output (5376x7168) BF16 projections.
  Bf16Lin qkv_proj, out_proj;
  // SwiGLU MLP fc1 (28672x5376) and fc2 (5376x14336) BF16 projections.
  Bf16Lin fc1, fc2;
} RefinerBlock;

// Weights for one INT8-ConvRot DiT transformer block.
typedef struct {
  // Pre-attention and pre-MLP RMSNorm weights (FP32, 5376).
  Buf norm1, norm2;
  // Per-head Q and K RMSNorm weights (FP32, 128).
  Buf q_norm, k_norm;
  // Pruned curve-form AdaLN linear (F16 [96768, 8] weight + BF16 [96768] bias).
  Buf adaln_w, adaln_b;
  // Attention QKV and output INT8-ConvRot projections.
  Int8Lin qkv_proj, out_proj;
  // SwiGLU MLP fc1 and fc2 INT8-ConvRot projections.
  Int8Lin fc1, fc2;
} DitBlock;

// Parameters stored by a WeightGroup entry that unfolds the PDD delta heads.
typedef struct {
  // Checkpoint tensor holding [pdd_steps * rows_per_step, cols] F32 values.
  Tensor tensor;
  // Number of PDD steps stacked along axis 0.
  int steps;
  // Number of FP32 values in one PDD step slice.
  size_t slice_floats;
} PddHeadParam;

// Transient workspace buffers allocated for one sampling call.
typedef struct {
  // Refined text tokens [text_len, 5376] and packed stream h [seq, 5376].
  Buf text_refined, h;
  // Modulated / normalized row staging [seq, 5376].
  Buf norm;
  // Padded BF16 staging (for refiner) and INT8 staging (for DiT ConvRot).
  Buf xb, xq, xscale;
  // Contiguous QKV [seq, 21504] and attention output [seq, 7168].
  Buf qkv, attn_out;
  // Head-major BF16 Q, K, V [56, padded_seq, 128].
  Buf qh, kh, vh;
  // MLP fc1 output [seq, 28672] and SwiGLU inner [seq, 14336].
  Buf fc1_out, fc1_mid;
  // AdaLN inputs [2, 8], block modulation outputs [2, 96768] and per-token
  // tags.
  Buf t_emb, mod, mod_rows, final_mod_rows;
  // 3D MM-RoPE positions [seq, 3].
  Buf pos;
  // Patchified visual inputs [total_vis, 96] and target velocity out.
  Buf patch_in, patch_out;
} DitWorkspace;

// MiniMax-H3-Image DiT handle.
struct MinimaxH3Dit {
  // GPU device and memory account owning the DiT buffers.
  Gpu *gpu;
  // Memory account that bounds weights and transient workspace.
  MemoryAccount *account;
  // Non-evictable slot tracking transient sampling buffers.
  MemorySlot workspace_slot;
  // Open safetensors file for the DiT checkpoint.
  TensorIndex *tensors;
  // Full-residence weight group for all DiT weights.
  WeightGroup *weights;
  // Number of PDD sampling steps.
  int pdd_steps;
  // Decreasing PDD sigma grid [pdd_steps + 1].
  double pdd_sigmas[MINIMAX_H3_DIT_MAX_PDD_STEPS + 1];
  // Curve-form AdaLN time-embedding lookup table [1025][8] on CPU.
  float adaln_t_table[DIT_ADALN_GRID][DIT_ADALN_RANK];
  // 3D MM-RoPE inverse frequencies (FP32, 16).
  Buf rope_inv_freq;
  // Text condition projection (BF16 5376x5120 + bias).
  Bf16Lin condition_proj;
  // Two BF16 token refiner transformer blocks.
  RefinerBlock refiner[DIT_REFINER_LAYERS];
  // Token refiner final RMSNorm weights (FP32, 5376).
  Buf refiner_final_norm;
  // FP32 video patch input projection (5376x96 + bias).
  Buf patch_proj_w, patch_proj_b;
  // 50 INT8-ConvRot DiT blocks.
  DitBlock blocks[DIT_LAYERS];
  // Final RMSNorm and curve-form AdaLN weights (10752x8).
  Buf final_norm, final_adaln_w, final_adaln_b;
  // Unfolded FP32 PDD video_out head weights and biases.
  Buf pdd_w, pdd_b;
  // Compiled Metal kernel indices for linear, norm, AdaLN and attention ops.
  int k_x_bf16, k_mm_bf16, k_mm_f32, k_convrot, k_mm_int8, k_rmsnorm, k_add,
      k_swiglu_chunked, k_adaln, k_norm_mod, k_gate_add, k_heads, k_attend;
};

// Reads PDD step count and sigma schedule from checkpoint __metadata__.config.
static void parse_pdd_config(MinimaxH3Dit *dit) {
  dit->pdd_steps = DEFAULT_PDD_STEPS;
  memcpy(dit->pdd_sigmas, DEFAULT_PDD_SIGMAS, sizeof DEFAULT_PDD_SIGMAS);
  const char *meta = tensor_index_metadata(dit->tensors, "config");
  if (!meta) return;
  char *copy = strdup(meta);
  if (!copy) die("out of memory for DiT config metadata");
  Json *root = json_parse(copy);
  Json *tr = json_get(root, "transformer");
  if (tr && tr->type == J_OBJ) {
    Json *steps = json_get(tr, "pdd_steps");
    Json *sigmas = json_get(tr, "pdd_sigmas");
    if (steps && steps->type == J_NUM && sigmas && sigmas->type == J_ARR) {
      int n = (int)steps->num;
      if (n >= 1 && n <= MINIMAX_H3_DIT_MAX_PDD_STEPS && sigmas->n == n + 1) {
        dit->pdd_steps = n;
        for (int i = 0; i <= n; i++) dit->pdd_sigmas[i] = sigmas->kid[i].num;
      }
    }
  }
  json_free(root);
  free(copy);
}

// Loads a PDD delta head bank tensor and unfolds blocks 1..steps-1 by adding
// base block 0 into each delta slice.
static void fill_pdd_head(Buf *buffer, const void *context) {
  const PddHeadParam *p = context;
  minimax_h3_read_tensor_bytes(p->tensor, buffer->ptr);
  float *data = (float *)buffer->ptr;
  for (int s = 1; s < p->steps; s++) {
    float *dst = data + (size_t)s * p->slice_floats;
    for (size_t i = 0; i < p->slice_floats; i++) dst[i] += data[i];
  }
}

// Registers a BF16 linear projection (page-aligned, aligned = 0) into group.
static void add_bf16_lin(WeightGroup *group, const TensorIndex *tensors,
                         Bf16Lin *lin, int n, int k, int with_bias,
                         const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  if (w.dtype != 'b' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k)
    die("unexpected BF16 weight shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  lin->has_bias = with_bias;
  weight_group_add_tensor(group, &lin->w, w, 0);
  if (with_bias) {
    Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
    if (b.ndim != 1 || b.shape[0] != n)
      die("unexpected bias shape for %s", prefix);
    weight_group_add_vector(group, &lin->bias, b, 0.0f, 0);
  }
}

// Registers an INT8-ConvRot linear projection (page-aligned, aligned = 0).
static void add_int8_lin(WeightGroup *group, const TensorIndex *tensors,
                         Int8Lin *lin, int n, int k, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor s = tensor_index_get(tensors, "%s.weight_scale", prefix);
  if (w.dtype != 'i' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k)
    die("unexpected INT8 weight shape for %s", prefix);
  if (s.dtype != 'f' || s.shape[0] != n)
    die("unexpected INT8 scale shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  weight_group_add_tensor(group, &lin->w, w, 0);
  weight_group_add_tensor(group, &lin->scale, s, 0);
}

MinimaxH3Dit *minimax_h3_dit_open(Gpu *gpu, MemoryAccount *account,
                                  const char *weights_path) {
  MinimaxH3Dit *dit = xmalloc(sizeof *dit);
  dit->gpu = gpu;
  dit->account = account;
  memory_slot_init(&dit->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-dit-workspace", NULL, NULL);
  dit->tensors = tensor_index_open_file(weights_path);
  parse_pdd_config(dit);
  dit->weights = weight_group_new(gpu, account, "minimax-h3-image-dit");

  dit->k_x_bf16 = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  dit->k_mm_bf16 = gpu_find_kernel(gpu, "minimax_h3_mm_bf16");
  dit->k_mm_f32 = gpu_find_kernel(gpu, "minimax_h3_mm_f32");
  dit->k_convrot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  dit->k_mm_int8 = gpu_find_kernel(gpu, "minimax_h3_mm_int8");
  dit->k_rmsnorm = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  dit->k_add = gpu_find_kernel(gpu, "minimax_h3_add");
  dit->k_swiglu_chunked = gpu_find_kernel(gpu, "minimax_h3_swiglu_chunked");
  dit->k_adaln = gpu_find_kernel(gpu, "minimax_h3_dit_adaln");
  dit->k_norm_mod = gpu_find_kernel(gpu, "minimax_h3_dit_norm_mod");
  dit->k_gate_add = gpu_find_kernel(gpu, "minimax_h3_dit_gate_add");
  dit->k_heads = gpu_find_kernel(gpu, "minimax_h3_dit_heads");
  dit->k_attend = gpu_find_kernel(gpu, "minimax_h3_dit_attend");

  // Curve-form AdaLN table lives on CPU (32 KiB) for timestep interpolation.
  Tensor t_table = tensor_index_get(dit->tensors, "adaln_t_table");
  if (t_table.dtype != 'f' || t_table.ndim != 2 ||
      t_table.shape[0] != DIT_ADALN_GRID || t_table.shape[1] != DIT_ADALN_RANK)
    die("unexpected adaln_t_table shape");
  minimax_h3_read_tensor_bytes(t_table, dit->adaln_t_table);

  Tensor inv = tensor_index_get(dit->tensors, "rope.inv_freq");
  if (inv.dtype != 'f' || inv.shape[0] != DIT_ROPE_FREQS)
    die("unexpected rope.inv_freq shape");
  weight_group_add_tensor(dit->weights, &dit->rope_inv_freq, inv, 0);

  // Text condition projection and 2-layer BF16 token refiner.
  add_bf16_lin(dit->weights, dit->tensors, &dit->condition_proj, DIT_DIM,
               MINIMAX_H3_TEXT_DIM, 1, "condition_proj");
  for (int i = 0; i < DIT_REFINER_LAYERS; i++) {
    RefinerBlock *r = &dit->refiner[i];
    weight_group_add_vector(
        dit->weights, &r->norm1,
        tensor_index_get(dit->tensors, "token_refiner.blocks.%d.norm1.weight",
                         i),
        0.0f, 0);
    weight_group_add_vector(
        dit->weights, &r->norm2,
        tensor_index_get(dit->tensors, "token_refiner.blocks.%d.norm2.weight",
                         i),
        0.0f, 0);
    weight_group_add_vector(
        dit->weights, &r->q_norm,
        tensor_index_get(dit->tensors,
                         "token_refiner.blocks.%d.attn.q_norm.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        dit->weights, &r->k_norm,
        tensor_index_get(dit->tensors,
                         "token_refiner.blocks.%d.attn.k_norm.weight", i),
        0.0f, 0);
    char name[128];
    snprintf(name, sizeof name, "token_refiner.blocks.%d.attn.qkv_proj", i);
    add_bf16_lin(dit->weights, dit->tensors, &r->qkv_proj, DIT_QKV_WIDTH,
                 DIT_DIM, 0, name);
    snprintf(name, sizeof name, "token_refiner.blocks.%d.attn.out_proj", i);
    add_bf16_lin(dit->weights, dit->tensors, &r->out_proj, DIT_DIM, DIT_INNER,
                 0, name);
    snprintf(name, sizeof name, "token_refiner.blocks.%d.mlp.fc1", i);
    add_bf16_lin(dit->weights, dit->tensors, &r->fc1, DIT_FC1_WIDTH, DIT_DIM, 0,
                 name);
    snprintf(name, sizeof name, "token_refiner.blocks.%d.mlp.fc2", i);
    add_bf16_lin(dit->weights, dit->tensors, &r->fc2, DIT_DIM, DIT_FFN_DIM, 0,
                 name);
  }
  weight_group_add_vector(
      dit->weights, &dit->refiner_final_norm,
      tensor_index_get(dit->tensors, "token_refiner.final_norm.weight"), 0.0f,
      0);

  // FP32 video patch embedder.
  Tensor pw = tensor_index_get(dit->tensors, "video_patch_proj.weight");
  Tensor pb = tensor_index_get(dit->tensors, "video_patch_proj.bias");
  if (pw.dtype != 'f' || pw.shape[0] != DIT_DIM || pw.shape[1] != DIT_PATCH_DIM)
    die("unexpected video_patch_proj.weight shape");
  if (pb.dtype != 'f' || pb.shape[0] != DIT_DIM)
    die("unexpected video_patch_proj.bias shape");
  weight_group_add_tensor(dit->weights, &dit->patch_proj_w, pw, 0);
  weight_group_add_tensor(dit->weights, &dit->patch_proj_b, pb, 0);

  // 50 INT8-ConvRot DiT blocks.
  for (int i = 0; i < DIT_LAYERS; i++) {
    DitBlock *b = &dit->blocks[i];
    weight_group_add_vector(
        dit->weights, &b->norm1,
        tensor_index_get(dit->tensors, "blocks.%d.norm1.weight", i), 0.0f, 0);
    weight_group_add_vector(
        dit->weights, &b->norm2,
        tensor_index_get(dit->tensors, "blocks.%d.norm2.weight", i), 0.0f, 0);
    weight_group_add_vector(
        dit->weights, &b->q_norm,
        tensor_index_get(dit->tensors, "blocks.%d.attn.q_norm.weight", i), 0.0f,
        0);
    weight_group_add_vector(
        dit->weights, &b->k_norm,
        tensor_index_get(dit->tensors, "blocks.%d.attn.k_norm.weight", i), 0.0f,
        0);
    Tensor aw =
        tensor_index_get(dit->tensors, "blocks.%d.adaln_proj.linear.weight", i);
    Tensor ab =
        tensor_index_get(dit->tensors, "blocks.%d.adaln_proj.linear.bias", i);
    if (aw.dtype != 'h' || aw.shape[0] != DIT_ADALN_OUT ||
        aw.shape[1] != DIT_ADALN_RANK)
      die("unexpected blocks.%d.adaln_proj.linear.weight shape", i);
    if (ab.dtype != 'b' || ab.shape[0] != DIT_ADALN_OUT)
      die("unexpected blocks.%d.adaln_proj.linear.bias shape", i);
    weight_group_add_tensor(dit->weights, &b->adaln_w, aw, 0);
    weight_group_add_tensor(dit->weights, &b->adaln_b, ab, 0);

    char name[128];
    snprintf(name, sizeof name, "blocks.%d.attn.qkv_proj", i);
    add_int8_lin(dit->weights, dit->tensors, &b->qkv_proj, DIT_QKV_WIDTH,
                 DIT_DIM, name);
    snprintf(name, sizeof name, "blocks.%d.attn.out_proj", i);
    add_int8_lin(dit->weights, dit->tensors, &b->out_proj, DIT_DIM, DIT_INNER,
                 name);
    snprintf(name, sizeof name, "blocks.%d.mlp.fc1", i);
    add_int8_lin(dit->weights, dit->tensors, &b->fc1, DIT_FC1_WIDTH, DIT_DIM,
                 name);
    snprintf(name, sizeof name, "blocks.%d.mlp.fc2", i);
    add_int8_lin(dit->weights, dit->tensors, &b->fc2, DIT_DIM, DIT_FFN_DIM,
                 name);
  }

  // Final layer: norm, curve-form AdaLN, and unfolded PDD video_out head bank.
  weight_group_add_vector(
      dit->weights, &dit->final_norm,
      tensor_index_get(dit->tensors, "final_layer.norm.weight"), 0.0f, 0);
  Tensor faw =
      tensor_index_get(dit->tensors, "final_layer.adaln_proj.linear.weight");
  Tensor fab =
      tensor_index_get(dit->tensors, "final_layer.adaln_proj.linear.bias");
  if (faw.dtype != 'h' || faw.shape[0] != DIT_FINAL_ADALN_OUT ||
      faw.shape[1] != DIT_ADALN_RANK)
    die("unexpected final_layer.adaln_proj.linear.weight shape");
  if (fab.dtype != 'b' || fab.shape[0] != DIT_FINAL_ADALN_OUT)
    die("unexpected final_layer.adaln_proj.linear.bias shape");
  weight_group_add_tensor(dit->weights, &dit->final_adaln_w, faw, 0);
  weight_group_add_tensor(dit->weights, &dit->final_adaln_b, fab, 0);

  Tensor vow = tensor_index_get(dit->tensors, "final_layer.video_out.weight");
  Tensor vob = tensor_index_get(dit->tensors, "final_layer.video_out.bias");
  int expected_out = dit->pdd_steps * DIT_PATCH_DIM;
  if (vow.dtype != 'f' || vow.shape[0] != expected_out ||
      vow.shape[1] != DIT_DIM)
    die("unexpected final_layer.video_out.weight shape");
  if (vob.dtype != 'f' || vob.shape[0] != expected_out)
    die("unexpected final_layer.video_out.bias shape");
  PddHeadParam pw_param = {
      .tensor = vow,
      .steps = dit->pdd_steps,
      .slice_floats = (size_t)DIT_PATCH_DIM * (size_t)DIT_DIM,
  };
  PddHeadParam pb_param = {
      .tensor = vob,
      .steps = dit->pdd_steps,
      .slice_floats = (size_t)DIT_PATCH_DIM,
  };
  weight_group_add_generated(dit->weights, &dit->pdd_w, vow.size, fill_pdd_head,
                             &pw_param, sizeof pw_param);
  weight_group_add_generated(dit->weights, &dit->pdd_b, vob.size, fill_pdd_head,
                             &pb_param, sizeof pb_param);
  return dit;
}

void minimax_h3_dit_close(MinimaxH3Dit *dit) {
  if (!dit) return;
  weight_group_free(dit->weights);
  memory_slot_remove(&dit->workspace_slot);
  tensor_index_close(dit->tensors);
  free(dit);
}

WeightGroup *minimax_h3_dit_weights(MinimaxH3Dit *dit) { return dit->weights; }

uint64_t minimax_h3_dit_required_bytes(const MinimaxH3Dit *dit) {
  return weight_group_required_bytes(dit->weights);
}

uint64_t minimax_h3_dit_resident_bytes(const MinimaxH3Dit *dit) {
  return weight_group_resident_bytes(dit->weights);
}

int minimax_h3_dit_pdd_steps(const MinimaxH3Dit *dit) { return dit->pdd_steps; }

int minimax_h3_dit_load(MinimaxH3Dit *dit) {
  return weight_group_load(dit->weights);
}

// Deterministic SplitMix64 generator for Gaussian latent noise.
static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Fills count floats with standard normal N(0, 1) draws via Box-Muller.
static void fill_gaussian(float *out, size_t count, uint64_t seed) {
  uint64_t state = seed;
  static constexpr double TWO_PI = 6.28318530717958647692;
  static constexpr double INV_2_53 = 1.0 / 9007199254740992.0;
  for (size_t i = 0; i < count; i += 2) {
    double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) * INV_2_53;
    double u2 = (double)(splitmix64_next(&state) >> 11) * INV_2_53;
    double r = sqrt(-2.0 * log(u1));
    double theta = TWO_PI * u2;
    out[i] = (float)(r * cos(theta));
    if (i + 1 < count) out[i + 1] = (float)(r * sin(theta));
  }
}

// Interpolates the rank-8 curve-form AdaLN time embedding for t in [0, 1].
static void interp_t_emb(const MinimaxH3Dit *dit, double t, float out[8]) {
  double clamped = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
  double pos = clamped * (double)(DIT_ADALN_GRID - 1);
  int i0 = (int)floor(pos);
  if (i0 > DIT_ADALN_GRID - 2) i0 = DIT_ADALN_GRID - 2;
  float frac = (float)(pos - (double)i0);
  for (int k = 0; k < DIT_ADALN_RANK; k++) {
    float a = dit->adaln_t_table[i0][k];
    float b = dit->adaln_t_table[i0 + 1][k];
    out[k] = a + frac * (b - a);
  }
}

// Fills area-normalized (t, h, w) coordinates for one latent frame's 2x2
// patches.
static void fill_frame_positions(float *dst_pos, int lat_h, int lat_w,
                                 double t_coord) {
  int nh = lat_h / DIT_PATCH_SIDE, nw = lat_w / DIT_PATCH_SIDE;
  double sqrt_area = sqrt((double)lat_h * (double)lat_w);
  double ratio_h = (double)lat_h / sqrt_area;
  double ratio_w = (double)lat_w / sqrt_area;
  for (int py = 0; py < nh; py++) {
    double ch = ((double)py * (ratio_h / (double)nh) + (1.0 - ratio_h) * 0.5) *
                DIT_ROPE_SPATIAL_SCALE;
    for (int px = 0; px < nw; px++) {
      double cw =
          ((double)px * (ratio_w / (double)nw) + (1.0 - ratio_w) * 0.5) *
          DIT_ROPE_SPATIAL_SCALE;
      float *p = dst_pos + ((size_t)py * (size_t)nw + (size_t)px) * 3u;
      p[0] = (float)t_coord;
      p[1] = (float)ch;
      p[2] = (float)cw;
    }
  }
}

// Patchifies channel-first latent [24, orig_h, orig_w] (zero-padded to even
// lat_h by lat_w) into [lat_h/2 * lat_w/2, 96] patch rows.
static void patchify_latent(const float *z, int orig_h, int orig_w, int lat_h,
                            int lat_w, float *out_rows) {
  int nh = lat_h / DIT_PATCH_SIDE, nw = lat_w / DIT_PATCH_SIDE;
  size_t plane = (size_t)orig_h * (size_t)orig_w;
  for (int py = 0; py < nh; py++) {
    for (int px = 0; px < nw; px++) {
      float *row =
          out_rows + ((size_t)py * (size_t)nw + (size_t)px) * DIT_PATCH_DIM;
      for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
        const float *zc = z + (size_t)c * plane;
        for (int dy = 0; dy < DIT_PATCH_SIDE; dy++) {
          int y = py * DIT_PATCH_SIDE + dy;
          for (int dx = 0; dx < DIT_PATCH_SIDE; dx++) {
            int x = px * DIT_PATCH_SIDE + dx;
            float v = (y < orig_h && x < orig_w)
                          ? zc[(size_t)y * (size_t)orig_w + (size_t)x]
                          : 0.0f;
            row[c * 4 + dy * 2 + dx] = v;
          }
        }
      }
    }
  }
}

// Unpatchifies [lat_h/2 * lat_w/2, 96] velocity rows and applies the Euler
// update z += dt * out onto channel-first [24, orig_h, orig_w].
static void unpatchify_euler_step(const float *rows, int orig_h, int orig_w,
                                  int lat_h, int lat_w, float dt, float *z) {
  int nh = lat_h / DIT_PATCH_SIDE, nw = lat_w / DIT_PATCH_SIDE;
  size_t plane = (size_t)orig_h * (size_t)orig_w;
  for (int py = 0; py < nh; py++) {
    for (int px = 0; px < nw; px++) {
      const float *row =
          rows + ((size_t)py * (size_t)nw + (size_t)px) * DIT_PATCH_DIM;
      for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
        float *zc = z + (size_t)c * plane;
        for (int dy = 0; dy < DIT_PATCH_SIDE; dy++) {
          int y = py * DIT_PATCH_SIDE + dy;
          if (y >= orig_h) continue;
          for (int dx = 0; dx < DIT_PATCH_SIDE; dx++) {
            int x = px * DIT_PATCH_SIDE + dx;
            if (x >= orig_w) continue;
            zc[(size_t)y * (size_t)orig_w + (size_t)x] +=
                dt * row[c * 4 + dy * 2 + dx];
          }
        }
      }
    }
  }
}

// Frees all transient sampling workspace buffers.
static void free_dit_workspace(DitWorkspace *ws) {
  gpu_free(&ws->patch_out);
  gpu_free(&ws->patch_in);
  gpu_free(&ws->pos);
  gpu_free(&ws->final_mod_rows);
  gpu_free(&ws->mod_rows);
  gpu_free(&ws->mod);
  gpu_free(&ws->t_emb);
  gpu_free(&ws->fc1_mid);
  gpu_free(&ws->fc1_out);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->attn_out);
  gpu_free(&ws->qkv);
  gpu_free(&ws->xscale);
  gpu_free(&ws->xq);
  gpu_free(&ws->xb);
  gpu_free(&ws->norm);
  gpu_free(&ws->h);
  gpu_free(&ws->text_refined);
}

// Allocates transient sampling workspace buffers for the packed sequence.
static int alloc_dit_workspace(MinimaxH3Dit *dit, int text_len, int total_vis,
                               int target_patches, int seq_len,
                               DitWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  uint32_t padded_seq = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t padded_text = ((uint32_t)text_len + TILE_M - 1) / TILE_M * TILE_M;
  MemorySlot *slot = &dit->workspace_slot;

  ws->text_refined = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)text_len * DIT_DIM * sizeof(float));
  ws->h = gpu_try_alloc_required(dit->gpu, slot,
                                 (size_t)seq_len * DIT_DIM * sizeof(float));
  ws->norm = gpu_try_alloc_required(dit->gpu, slot,
                                    (size_t)seq_len * DIT_DIM * sizeof(float));
  ws->xb = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)padded_text * DIT_FFN_DIM * sizeof(uint16_t));
  ws->xq =
      gpu_try_alloc_required(dit->gpu, slot, (size_t)padded_seq * DIT_FFN_DIM);
  ws->xscale =
      gpu_try_alloc_required(dit->gpu, slot, (size_t)seq_len * sizeof(float));
  ws->qkv = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)seq_len * DIT_QKV_WIDTH * sizeof(float));
  ws->attn_out = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)seq_len * DIT_INNER * sizeof(float));
  ws->qh = gpu_try_alloc_required(dit->gpu, slot,
                                  (size_t)DIT_HEADS * padded_seq *
                                      DIT_HEAD_DIM * sizeof(uint16_t));
  ws->kh = gpu_try_alloc_required(dit->gpu, slot,
                                  (size_t)DIT_HEADS * padded_seq *
                                      DIT_HEAD_DIM * sizeof(uint16_t));
  ws->vh = gpu_try_alloc_required(dit->gpu, slot,
                                  (size_t)DIT_HEADS * padded_seq *
                                      DIT_HEAD_DIM * sizeof(uint16_t));
  ws->fc1_out = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)seq_len * DIT_FC1_WIDTH * sizeof(float));
  ws->fc1_mid = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)seq_len * DIT_FFN_DIM * sizeof(float));
  ws->t_emb = gpu_try_alloc_required(
      dit->gpu, slot, 2 * (size_t)DIT_ADALN_RANK * sizeof(float));
  ws->mod = gpu_try_alloc_required(dit->gpu, slot,
                                   2 * (size_t)DIT_ADALN_OUT * sizeof(float));
  ws->mod_rows = gpu_try_alloc_required(dit->gpu, slot,
                                        (size_t)seq_len * sizeof(uint32_t));
  ws->final_mod_rows = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)target_patches * sizeof(uint32_t));
  ws->pos = gpu_try_alloc_required(dit->gpu, slot,
                                   (size_t)seq_len * 3 * sizeof(float));
  ws->patch_in = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)total_vis * DIT_PATCH_DIM * sizeof(float));
  ws->patch_out = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)target_patches * DIT_PATCH_DIM * sizeof(float));

  if (!ws->text_refined.obj || !ws->h.obj || !ws->norm.obj || !ws->xb.obj ||
      !ws->xq.obj || !ws->xscale.obj || !ws->qkv.obj || !ws->attn_out.obj ||
      !ws->qh.obj || !ws->kh.obj || !ws->vh.obj || !ws->fc1_out.obj ||
      !ws->fc1_mid.obj || !ws->t_emb.obj || !ws->mod.obj || !ws->mod_rows.obj ||
      !ws->final_mod_rows.obj || !ws->pos.obj || !ws->patch_in.obj ||
      !ws->patch_out.obj) {
    free_dit_workspace(ws);
    return 0;
  }
  return 1;
}

// Encodes FP32 x -> padded BF16 xb -> native BF16 matmul (+ optional bias).
static void run_bf16_linear(const MinimaxH3Dit *dit, GpuCommandBuffer *cmd,
                            const Buf *x, const Buf *xb, const Bf16Lin *lin,
                            const Buf *y, int rows) {
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  Bf16XParam xp = {(uint32_t)rows, (uint32_t)lin->k, (uint32_t)lin->k, 0};
  gpu_select_kernel(cmd, dit->k_x_bf16);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &dit->rope_inv_freq, 0);
  gpu_bind_buffer(cmd, 2, xb, 0);
  gpu_copy_bytes(cmd, 3, &xp, sizeof xp);
  gpu_run_kernel(cmd,
                 ((uint32_t)lin->k / 4u + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 padded, 1, GRID2D_THREADS, 1, 1);

  Bf16MMParam mp = {(uint32_t)rows, (uint32_t)lin->n, (uint32_t)lin->k,
                    (uint32_t)lin->n, (uint32_t)lin->has_bias};
  const Buf *bias_buf = lin->has_bias ? &lin->bias : &dit->rope_inv_freq;
  gpu_select_kernel(cmd, dit->k_mm_bf16);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, xb, 0);
  gpu_bind_buffer(cmd, 2, bias_buf, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N, padded / TILE_M,
                 1, MM_THREADS, 1, 1);
}

// Encodes ConvRot Hadamard + per-row INT8 quantization + native INT8 matmul.
static void run_convrot_linear(const MinimaxH3Dit *dit, GpuCommandBuffer *cmd,
                               const Buf *x, const Buf *xq, const Buf *xscale,
                               const Int8Lin *lin, const Buf *y, int rows) {
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  ConvRotParam rp = {(uint32_t)rows, (uint32_t)lin->k, (uint32_t)lin->k};
  gpu_select_kernel(cmd, dit->k_convrot);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, xq, 0);
  gpu_bind_buffer(cmd, 2, xscale, 0);
  gpu_copy_bytes(cmd, 3, &rp, sizeof rp);
  gpu_run_kernel(cmd, padded, 1, 1, CONVROT_THREADS, 1, 1);

  Int8MMParam mp = {(uint32_t)rows, (uint32_t)lin->n, (uint32_t)lin->k,
                    (uint32_t)lin->n, 0};
  gpu_select_kernel(cmd, dit->k_mm_int8);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, &lin->scale, 0);
  gpu_bind_buffer(cmd, 2, xq, 0);
  gpu_bind_buffer(cmd, 3, xscale, 0);
  gpu_bind_buffer(cmd, 4, &dit->rope_inv_freq, 0);
  gpu_bind_buffer(cmd, 5, y, 0);
  gpu_copy_bytes(cmd, 6, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N, padded / TILE_M,
                 1, MM_THREADS, 1, 1);
}

// Encodes condition_proj and the 2-layer BF16 token refiner once per sample.
static void run_token_refiner(MinimaxH3Dit *dit, const Buf *text_hidden,
                              int text_len, const DitWorkspace *ws) {
  uint32_t padded = ((uint32_t)text_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t count = (uint32_t)text_len * (uint32_t)DIT_DIM;
  RmsNormParam np = {(uint32_t)DIT_DIM, DIT_NORM_EPS};
  SwigluChunkedParam swp = {(uint32_t)text_len, (uint32_t)DIT_FFN_DIM};
  DitHeadsParam hp = {(uint32_t)text_len, padded, 0, DIT_NORM_EPS};
  DitAttendParam ap = {(uint32_t)text_len, padded};

  GpuCommandBuffer cmd = gpu_begin(dit->gpu);
  run_bf16_linear(dit, &cmd, text_hidden, &ws->xb, &dit->condition_proj,
                  &ws->text_refined, text_len);

  for (int i = 0; i < DIT_REFINER_LAYERS; i++) {
    const RefinerBlock *r = &dit->refiner[i];

    gpu_select_kernel(&cmd, dit->k_rmsnorm);
    gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
    gpu_bind_buffer(&cmd, 1, &r->norm1, 0);
    gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 3, &np, sizeof np);
    gpu_run_kernel(&cmd, (uint32_t)text_len, 1, 1, NORM_THREADS, 1, 1);

    run_bf16_linear(dit, &cmd, &ws->norm, &ws->xb, &r->qkv_proj, &ws->qkv,
                    text_len);

    gpu_select_kernel(&cmd, dit->k_heads);
    gpu_bind_buffer(&cmd, 0, &ws->qkv, 0);
    gpu_bind_buffer(&cmd, 1, &r->q_norm, 0);
    gpu_bind_buffer(&cmd, 2, &r->k_norm, 0);
    gpu_bind_buffer(&cmd, 3, &ws->pos, 0);
    gpu_bind_buffer(&cmd, 4, &dit->rope_inv_freq, 0);
    gpu_bind_buffer(&cmd, 5, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 6, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 7, &ws->vh, 0);
    gpu_copy_bytes(&cmd, 8, &hp, sizeof hp);
    gpu_run_kernel(&cmd, (uint32_t)(3 * DIT_HEADS), padded, 1, HEADS_THREADS, 1,
                   1);

    gpu_select_kernel(&cmd, dit->k_attend);
    gpu_bind_buffer(&cmd, 0, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws->vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->attn_out, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, padded / TILE_M, (uint32_t)DIT_HEADS, 1, MM_THREADS, 1,
                   1);

    run_bf16_linear(dit, &cmd, &ws->attn_out, &ws->xb, &r->out_proj, &ws->norm,
                    text_len);

    gpu_select_kernel(&cmd, dit->k_add);
    gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 2, &count, sizeof count);
    gpu_run_kernel(&cmd, (count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, dit->k_rmsnorm);
    gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
    gpu_bind_buffer(&cmd, 1, &r->norm2, 0);
    gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 3, &np, sizeof np);
    gpu_run_kernel(&cmd, (uint32_t)text_len, 1, 1, NORM_THREADS, 1, 1);

    run_bf16_linear(dit, &cmd, &ws->norm, &ws->xb, &r->fc1, &ws->fc1_out,
                    text_len);

    gpu_select_kernel(&cmd, dit->k_swiglu_chunked);
    gpu_bind_buffer(&cmd, 0, &ws->fc1_out, 0);
    gpu_bind_buffer(&cmd, 1, &ws->fc1_mid, 0);
    gpu_copy_bytes(&cmd, 2, &swp, sizeof swp);
    gpu_run_kernel(&cmd, (uint32_t)DIT_FFN_DIM / GRID2D_THREADS,
                   (uint32_t)text_len, 1, GRID2D_THREADS, 1, 1);

    run_bf16_linear(dit, &cmd, &ws->fc1_mid, &ws->xb, &r->fc2, &ws->norm,
                    text_len);

    gpu_select_kernel(&cmd, dit->k_add);
    gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 2, &count, sizeof count);
    gpu_run_kernel(&cmd, (count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);
  }

  // token_refiner.final_norm writes into norm, then copy back into
  // text_refined.
  gpu_select_kernel(&cmd, dit->k_rmsnorm);
  gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
  gpu_bind_buffer(&cmd, 1, &dit->refiner_final_norm, 0);
  gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
  gpu_copy_bytes(&cmd, 3, &np, sizeof np);
  gpu_run_kernel(&cmd, (uint32_t)text_len, 1, 1, NORM_THREADS, 1, 1);

  gpu_commit(&cmd);
  minimax_h3_gpu_wait(dit->gpu, cmd.number);
  memcpy(ws->text_refined.ptr, ws->norm.ptr, (size_t)count * sizeof(float));
}

int minimax_h3_dit_sample(MinimaxH3Dit *dit, const Buf *text_hidden,
                          int text_len, const int *text_tags,
                          const MinimaxH3RefLatent *refs, int ref_count,
                          int h_lat, int w_lat, uint64_t seed,
                          float *out_latents, const MinimaxH3RunHooks *hooks) {
  if (text_len <= 0 || h_lat <= 0 || w_lat <= 0 || !text_hidden || !out_latents)
    die("invalid DiT sampling arguments");
  if (ref_count < 0 || (ref_count > 0 && !refs))
    die("invalid reference image list");
  if (minimax_h3_should_stop(hooks)) return 0;
  if (!minimax_h3_dit_load(dit)) return 0;

  int lat_h = (h_lat + 1) / 2 * 2;
  int lat_w = (w_lat + 1) / 2 * 2;
  int target_patches = (lat_h / DIT_PATCH_SIDE) * (lat_w / DIT_PATCH_SIDE);

  int ref_patches = 0;
  for (int i = 0; i < ref_count; i++) {
    int rh = (refs[i].h_lat + 1) / 2 * 2;
    int rw = (refs[i].w_lat + 1) / 2 * 2;
    ref_patches += (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
  }
  int total_vis = ref_patches + target_patches;
  int seq_len = text_len + total_vis;

  MemorySlot *wslot = weight_group_slot(dit->weights);
  unsigned was_pinned = wslot->pins;
  memory_pin(wslot, 1);
  DitWorkspace ws;
  if (!alloc_dit_workspace(dit, text_len, total_vis, target_patches, seq_len,
                           &ws)) {
    memory_pin(wslot, (int)was_pinned);
    return 0;
  }

  // Build 3D MM-RoPE coordinates [seq_len, 3] and per-token AdaLN row indices.
  float *pos = (float *)ws.pos.ptr;
  uint32_t *mod_rows = (uint32_t *)ws.mod_rows.ptr;
  for (int i = 0; i < text_len; i++) {
    pos[3 * i] = (float)i;
    pos[3 * i + 1] = 0.0f;
    pos[3 * i + 2] = 0.0f;
    uint32_t tag = text_tags ? (uint32_t)text_tags[i] : TAG_TEXT;
    mod_rows[i] = tag;
  }

  double cursor = (double)text_len;
  int row_off = text_len;
  int vis_off = 0;
  float *patch_in = (float *)ws.patch_in.ptr;

  // Patchify reference images once with VISUAL_COND_TIMESTEP noise
  // augmentation.
  for (int i = 0; i < ref_count; i++) {
    int rh = (refs[i].h_lat + 1) / 2 * 2;
    int rw = (refs[i].w_lat + 1) / 2 * 2;
    int n = (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
    fill_frame_positions(pos + (size_t)row_off * 3u, rh, rw, cursor);
    for (int j = 0; j < n; j++) mod_rows[row_off + j] = 3u + TAG_VISUAL;

    float *dst = patch_in + (size_t)vis_off * DIT_PATCH_DIM;
    patchify_latent(refs[i].latents, refs[i].h_lat, refs[i].w_lat, rh, rw, dst);
    size_t count = (size_t)n * DIT_PATCH_DIM;
    float *noise = xmalloc(count * sizeof(float));
    fill_gaussian(noise, count, seed);
    float aug = (float)DIT_VISUAL_COND_TIMESTEP;
    for (size_t k = 0; k < count; k++)
      dst[k] = aug * dst[k] + (1.0f - aug) * noise[k];
    free(noise);

    cursor += 1.0;
    row_off += n;
    vis_off += n;
  }

  fill_frame_positions(pos + (size_t)row_off * 3u, lat_h, lat_w, cursor);
  for (int j = 0; j < target_patches; j++) mod_rows[row_off + j] = TAG_VISUAL;
  memset(ws.final_mod_rows.ptr, 0, (size_t)target_patches * sizeof(uint32_t));

  // Refine text embeddings once for all PDD denoising steps.
  run_token_refiner(dit, text_hidden, text_len, &ws);

  // Initialize target latent with standard normal noise from seed.
  size_t latent_floats =
      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h_lat * (size_t)w_lat;
  fill_gaussian(out_latents, latent_floats, seed);

  uint32_t padded_seq = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t num_t = ref_count > 0 ? 2u : 1u;
  DitAdalnParam adaln_p = {num_t, (uint32_t)DIT_ADALN_OUT};
  DitAdalnParam final_adaln_p = {1u, (uint32_t)DIT_FINAL_ADALN_OUT};
  DitNormModParam msa_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MSA, OFF_SCALE_MSA, DIT_NORM_EPS};
  DitNormModParam mlp_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MLP, OFF_SCALE_MLP, DIT_NORM_EPS};
  DitNormModParam final_norm_p = {(uint32_t)DIT_DIM,
                                  (uint32_t)DIT_FINAL_ADALN_OUT, 0,
                                  (uint32_t)DIT_DIM, DIT_NORM_EPS};
  DitGateAddParam msa_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MSA};
  DitGateAddParam mlp_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MLP};
  DitHeadsParam heads_p = {(uint32_t)seq_len, padded_seq, 1u, DIT_NORM_EPS};
  DitAttendParam attend_p = {(uint32_t)seq_len, padded_seq};
  SwigluChunkedParam swiglu_p = {(uint32_t)seq_len, (uint32_t)DIT_FFN_DIM};

  size_t text_bytes = (size_t)text_len * (size_t)DIT_DIM * sizeof(float);
  size_t target_h_off =
      (size_t)(text_len + ref_patches) * (size_t)DIT_DIM * sizeof(float);

  // 6-step PDD Euler flow-matching loop.
  int stopped = 0;
  for (int step = 0; step < dit->pdd_steps && !stopped; step++) {
    if (minimax_h3_should_stop(hooks)) {
      stopped = 1;
      break;
    }
    double sigma = dit->pdd_sigmas[step];
    double sigma_next = dit->pdd_sigmas[step + 1];
    double t_main = 1.0 - sigma;
    double t_ref =
        t_main > DIT_VISUAL_COND_TIMESTEP ? t_main : DIT_VISUAL_COND_TIMESTEP;

    float *te = (float *)ws.t_emb.ptr;
    interp_t_emb(dit, t_main, te);
    if (ref_count > 0) interp_t_emb(dit, t_ref, te + DIT_ADALN_RANK);

    // Patchify current noisy target latent and embed all visual rows on GPU.
    patchify_latent(out_latents, h_lat, w_lat, lat_h, lat_w,
                    patch_in + (size_t)ref_patches * DIT_PATCH_DIM);
    memcpy(ws.h.ptr, ws.text_refined.ptr, text_bytes);

    GpuCommandBuffer cmd = gpu_begin(dit->gpu);
    F32MMParam ep = {(uint32_t)total_vis, (uint32_t)DIT_DIM,
                     (uint32_t)DIT_PATCH_DIM, (uint32_t)DIT_DIM};
    gpu_select_kernel(&cmd, dit->k_mm_f32);
    gpu_bind_buffer(&cmd, 0, &dit->patch_proj_w, 0);
    gpu_bind_buffer(&cmd, 1, &ws.patch_in, 0);
    gpu_bind_buffer(&cmd, 2, &dit->patch_proj_b, 0);
    gpu_bind_buffer(&cmd, 3, &ws.h, text_bytes);
    gpu_copy_bytes(&cmd, 4, &ep, sizeof ep);
    gpu_run_kernel(&cmd, (uint32_t)DIT_DIM, (uint32_t)total_vis, 1,
                   F32MM_THREADS, 1, 1);
    gpu_commit(&cmd);
    uint64_t prev_cmd = cmd.number;

    // 50 INT8-ConvRot DiT blocks, yielding cooperatively between blocks.
    for (int i = 0; i < DIT_LAYERS; i++) {
      if (minimax_h3_should_stop(hooks)) {
        stopped = 1;
        break;
      }
      const DitBlock *b = &dit->blocks[i];
      cmd = gpu_begin(dit->gpu);

      gpu_select_kernel(&cmd, dit->k_adaln);
      gpu_bind_buffer(&cmd, 0, &b->adaln_w, 0);
      gpu_bind_buffer(&cmd, 1, &b->adaln_b, 0);
      gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
      gpu_copy_bytes(&cmd, 4, &adaln_p, sizeof adaln_p);
      gpu_run_kernel(
          &cmd, ((uint32_t)DIT_ADALN_OUT + ELEM_THREADS - 1) / ELEM_THREADS,
          num_t, 1, ELEM_THREADS, 1, 1);

      // Attention branch: norm1 + AdaLN -> ConvRot QKV -> QK-Norm + 3D RoPE ->
      // bidirectional 56-head FlashAttention -> ConvRot out_proj -> gated add.
      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, 0);
      gpu_bind_buffer(&cmd, 1, &b->norm1, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &msa_norm_p, sizeof msa_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);

      run_convrot_linear(dit, &cmd, &ws.norm, &ws.xq, &ws.xscale, &b->qkv_proj,
                         &ws.qkv, seq_len);

      gpu_select_kernel(&cmd, dit->k_heads);
      gpu_bind_buffer(&cmd, 0, &ws.qkv, 0);
      gpu_bind_buffer(&cmd, 1, &b->q_norm, 0);
      gpu_bind_buffer(&cmd, 2, &b->k_norm, 0);
      gpu_bind_buffer(&cmd, 3, &ws.pos, 0);
      gpu_bind_buffer(&cmd, 4, &dit->rope_inv_freq, 0);
      gpu_bind_buffer(&cmd, 5, &ws.qh, 0);
      gpu_bind_buffer(&cmd, 6, &ws.kh, 0);
      gpu_bind_buffer(&cmd, 7, &ws.vh, 0);
      gpu_copy_bytes(&cmd, 8, &heads_p, sizeof heads_p);
      gpu_run_kernel(&cmd, (uint32_t)(3 * DIT_HEADS), padded_seq, 1,
                     HEADS_THREADS, 1, 1);

      gpu_select_kernel(&cmd, dit->k_attend);
      gpu_bind_buffer(&cmd, 0, &ws.qh, 0);
      gpu_bind_buffer(&cmd, 1, &ws.kh, 0);
      gpu_bind_buffer(&cmd, 2, &ws.vh, 0);
      gpu_bind_buffer(&cmd, 3, &ws.attn_out, 0);
      gpu_copy_bytes(&cmd, 4, &attend_p, sizeof attend_p);
      gpu_run_kernel(&cmd, padded_seq / TILE_M, (uint32_t)DIT_HEADS, 1,
                     MM_THREADS, 1, 1);

      run_convrot_linear(dit, &cmd, &ws.attn_out, &ws.xq, &ws.xscale,
                         &b->out_proj, &ws.norm, seq_len);

      gpu_select_kernel(&cmd, dit->k_gate_add);
      gpu_bind_buffer(&cmd, 0, &ws.h, 0);
      gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod_rows, 0);
      gpu_copy_bytes(&cmd, 4, &msa_gate_p, sizeof msa_gate_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                     (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);

      // MLP branch: norm2 + AdaLN -> ConvRot fc1 -> SwiGLU -> ConvRot fc2 ->
      // gated residual add.
      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, 0);
      gpu_bind_buffer(&cmd, 1, &b->norm2, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &mlp_norm_p, sizeof mlp_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);

      run_convrot_linear(dit, &cmd, &ws.norm, &ws.xq, &ws.xscale, &b->fc1,
                         &ws.fc1_out, seq_len);

      gpu_select_kernel(&cmd, dit->k_swiglu_chunked);
      gpu_bind_buffer(&cmd, 0, &ws.fc1_out, 0);
      gpu_bind_buffer(&cmd, 1, &ws.fc1_mid, 0);
      gpu_copy_bytes(&cmd, 2, &swiglu_p, sizeof swiglu_p);
      gpu_run_kernel(&cmd, (uint32_t)DIT_FFN_DIM / GRID2D_THREADS,
                     (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);

      run_convrot_linear(dit, &cmd, &ws.fc1_mid, &ws.xq, &ws.xscale, &b->fc2,
                         &ws.norm, seq_len);

      gpu_select_kernel(&cmd, dit->k_gate_add);
      gpu_bind_buffer(&cmd, 0, &ws.h, 0);
      gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod_rows, 0);
      gpu_copy_bytes(&cmd, 4, &mlp_gate_p, sizeof mlp_gate_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                     (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);

      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(dit->gpu, prev_cmd, hooks);
      prev_cmd = cmd.number;
    }

    if (stopped) {
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      break;
    }

    // FinalLayer on target image slice using PDD head for step.
    cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_adaln);
    gpu_bind_buffer(&cmd, 0, &dit->final_adaln_w, 0);
    gpu_bind_buffer(&cmd, 1, &dit->final_adaln_b, 0);
    gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
    gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
    gpu_copy_bytes(&cmd, 4, &final_adaln_p, sizeof final_adaln_p);
    gpu_run_kernel(
        &cmd, ((uint32_t)DIT_FINAL_ADALN_OUT + ELEM_THREADS - 1) / ELEM_THREADS,
        1, 1, ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, dit->k_norm_mod);
    gpu_bind_buffer(&cmd, 0, &ws.h, target_h_off);
    gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
    gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
    gpu_bind_buffer(&cmd, 3, &ws.final_mod_rows, 0);
    gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
    gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
    gpu_run_kernel(&cmd, (uint32_t)target_patches, 1, 1, NORM_THREADS, 1, 1);

    size_t pdd_w_off =
        (size_t)step * (size_t)DIT_PATCH_DIM * (size_t)DIT_DIM * sizeof(float);
    size_t pdd_b_off = (size_t)step * (size_t)DIT_PATCH_DIM * sizeof(float);
    F32MMParam fp = {(uint32_t)target_patches, (uint32_t)DIT_PATCH_DIM,
                     (uint32_t)DIT_DIM, (uint32_t)DIT_PATCH_DIM};
    gpu_select_kernel(&cmd, dit->k_mm_f32);
    gpu_bind_buffer(&cmd, 0, &dit->pdd_w, pdd_w_off);
    gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
    gpu_bind_buffer(&cmd, 2, &dit->pdd_b, pdd_b_off);
    gpu_bind_buffer(&cmd, 3, &ws.patch_out, 0);
    gpu_copy_bytes(&cmd, 4, &fp, sizeof fp);
    gpu_run_kernel(&cmd, (uint32_t)DIT_PATCH_DIM, (uint32_t)target_patches, 1,
                   F32MM_THREADS, 1, 1);

    gpu_commit(&cmd);
    minimax_h3_gpu_wait(dit->gpu, prev_cmd);
    minimax_h3_gpu_wait_hooks(dit->gpu, cmd.number, hooks);

    // Flow-matching Euler step: model predicts v = x_data - noise, so
    // x_next = x + (sigma - sigma_next) * v.
    float dt = (float)(sigma - sigma_next);
    unpatchify_euler_step((const float *)ws.patch_out.ptr, h_lat, w_lat, lat_h,
                          lat_w, dt, out_latents);
  }

  free_dit_workspace(&ws);
  memory_pin(wslot, (int)was_pinned);
  memory_touch(wslot);
  return !stopped;
}
