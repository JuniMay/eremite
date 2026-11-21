#include "model/minimax_h3/dit.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/json.h"
#include "engine/safetensors.h"
#include "model/minimax_h3/audio_vae.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MiniMax-H3-Image DiT architecture constants from MiniMaxAI/MiniMax-H3-Image
// transformer/config.json:4-26 (hidden_size 5376, num_layers 50,
// token_refiner_num_layers 2, num_attention_heads 56, attention_head_dim 128,
// ffn_hidden_size 14336, patch_size [1, 2, 2], adaln_out_features 96768,
// final_adaln_out_features 10752, rope_inv_freq_len 16).
static constexpr int DIT_DIM = 5376, DIT_LAYERS = 50, DIT_REFINER_LAYERS = 2,
                     DIT_HEADS = 56, DIT_HEAD_DIM = 128,
                     DIT_INNER = DIT_HEADS * DIT_HEAD_DIM,
                     DIT_QKV_WIDTH = MINIMAX_H3_QKV_PARTS * DIT_INNER,
                     DIT_FFN_DIM = 14336, DIT_FC1_WIDTH = 2 * DIT_FFN_DIM,
                     DIT_PATCH_SIDE = 2,
                     DIT_PATCH_AREA = DIT_PATCH_SIDE * DIT_PATCH_SIDE,
                     DIT_PATCH_DIM =
                         MINIMAX_H3_VAE_LATENT_CHANNELS * DIT_PATCH_AREA,
                     DIT_ROPE_FREQS = 16, DIT_ADALN_GRID = 1025,
                     DIT_ADALN_RANK = 8, DIT_ADALN_MODALITIES = 3,
                     DIT_ADALN_CHUNKS = 6,
                     DIT_ADALN_STRIDE = DIT_ADALN_CHUNKS * DIT_DIM,
                     DIT_ADALN_OUT = DIT_ADALN_MODALITIES * DIT_ADALN_STRIDE,
                     DIT_FINAL_ADALN_OUT = 2 * DIT_DIM;

// Default PDD step count when checkpoint metadata omits one;
// MiniMaxAI/MiniMax-H3-Image pdd_config.json:2 ("num_inference_steps": 6).
static constexpr int DEFAULT_PDD_STEPS = 6;

// Default 6-step PDD decreasing sigma schedule when metadata omits one;
// MiniMaxAI/MiniMax-H3-Image pdd_config.json:3-11 ("sigmas") and
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:833-835 (MiniMaxH3ImageModel).
static const double DEFAULT_PDD_SIGMAS[DEFAULT_PDD_STEPS + 1] = {
    1.0,  0.9675236940383911, 0.8875485062599182,
    0.75, 0.5327739119529724, 0.23200887441635132,
    0.0,
};

// RMSNorm epsilon from MiniMaxAI/MiniMax-H3-Image transformer/config.json:24-26
// (norm_eps, qk_norm_eps, final_norm_eps).
static constexpr float DIT_NORM_EPS = 1e-5f;

// Spatial RoPE coordinate scale over area-normalized patch grids;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:81 (_axis_from_sqrt_area).
static constexpr double DIT_ROPE_SPATIAL_SCALE = 32.0;

// Half-unit offset centering normalized spatial coordinates within a unit span;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:81 ((1.0 - ratio) / 2.0).
static constexpr double CELL_CENTER = 0.5;

// Noise-augmentation timesteps pinned on visual and audio conditioning rows;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:42-43 (VISUAL_COND_TIMESTEP,
// AUDIO_COND_TIMESTEP).
static constexpr double DIT_VISUAL_COND_TIMESTEP = 0.999,
                        DIT_AUDIO_COND_TIMESTEP = 1.0;

// Video latent temporal stride constants: every 5 latent tokens span 17 pixel
// frames (1 + 4 + 4 + 4 + 4), and 1 pixel frame at 24 fps equals 5/3 audio
// latent frames at 40 Hz; Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:40-41
// and comfy_extras/nodes_minimax_h3.py:955-962.
static constexpr int VIDEO_FRAME_GRID_STEP = 17, VIDEO_FRAME_GRID_BASE = 5,
                     VIDEO_LATENT_GRID_STEP = 5, VIDEO_LATENT_GRID_BASE = 2,
                     VIDEO_TOKEN_CYCLE = 5;
static const int FRAME_PER_TOKEN[VIDEO_TOKEN_CYCLE] = {1, 4, 4, 4, 4};
static constexpr double FRAME_RESCALE = 5.0 / 3.0;

// Seed offset separating audio latent noise from video latent noise;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:593.
static constexpr uint64_t AUDIO_NOISE_SEED_OFFSET = 1;

// AdaLN modality tags: visual (0), text (1) and audio (2);
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:247 (SEG_TAG).
static constexpr uint32_t TAG_VISUAL = 0, TAG_TEXT = 1, TAG_AUDIO = 2;

// Row index of the conditioning reference timestep in the 2-row image AdaLN
// table (row 0 holds main denoising timestep t_main; row 1 holds t_ref).
static constexpr uint32_t DIT_REF_T_ROW = 1;

// Fixed AdaLN timestep row slots for joint audio-video denoising:
// 0 = video/text (t_v), 1 = audio (t_a), 2 = visual cond (t_cv),
// 3 = audio cond (t_ca).
static constexpr uint32_t DIT_VID_T_ROW_VIDEO = 0, DIT_VID_T_ROW_AUDIO = 1,
                          DIT_VID_T_ROW_COND_VIS = 2,
                          DIT_VID_T_ROW_COND_AUD = 3, DIT_IMAGE_MAX_T_ROWS = 2,
                          DIT_VIDEO_MAX_T_ROWS = 4;

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
  // Number of distinct timesteps, output width per timestep and F16 bias flag.
  uint32_t num_t, out_dim, bias_f16;
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
  // Active sequence length, padded head stride and query tile offset.
  uint32_t rows, padded, tile_offset;
} DitAttendParam;

// Measured GPU seconds per 64-row tile for each heavy DiT block stage during
// contended execution, reused across blocks and steps of one sample.
typedef struct {
  // Seconds per 64-row tile for QKV, attention, out_proj, fc1, fc2, patch_in
  // and patch_out projections.
  double qkv, attend, out_proj, fc1, fc2, patch_in, patch_out;
  // Last measured whole-block seconds when one block fit inside a turn budget.
  double block;
  // Nonzero once every stage has at least one measured tile duration.
  int measured;
} DitSlicePace;

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
  // Pruned curve-form AdaLN linear (F16 [96768, 8] weight + BF16/F16 bias).
  Buf adaln_w, adaln_b;
  // Non-zero when adaln_b is stored as IEEE FP16 instead of BF16.
  uint32_t adaln_bias_f16;
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
  // Padded BF16 staging (refiner or LoRA) and INT8 staging (DiT ConvRot).
  Buf xb, xq, xscale;
  // Optional BF16 low-rank intermediate buffer when a runtime LoRA is active.
  Buf lora_mid;
  // Aliases into fc1_out and fc1_mid (attention and MLP branches are disjoint).
  Buf qkv, attn_out;
  // Head-major BF16 Q, K, V [56, padded_seq, 128].
  Buf qh, kh, vh;
  // MLP fc1 output [seq, 28672] and SwiGLU inner [seq, 14336].
  Buf fc1_out, fc1_mid;
  // AdaLN inputs [max_t, 8], block modulation outputs [max_t, 96768] and
  // per-token modulation row indices.
  Buf t_emb, mod, mod_rows, final_mod_rows, final_audio_mod_rows;
  // 3D MM-RoPE positions [seq, 3].
  Buf pos;
  // Patchified visual inputs [total_vis, 96] and target velocity out.
  Buf patch_in, patch_out;
  // Packed stereo audio inputs [total_audio, 32] and target audio velocity out.
  Buf audio_in, audio_out;
} DitWorkspace;

// MiniMax-H3 DiT handle (image-only or joint audio-video).
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
  // Non-zero when the checkpoint includes stereo audio input and output heads.
  int has_audio;
  // Number of PDD sampling steps (1 for base video models, 6 for image).
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
  // Optional FP32 audio patch input projection (5376x32 + bias).
  Buf audio_patch_proj_w, audio_patch_proj_b;
  // 50 INT8-ConvRot DiT blocks.
  DitBlock blocks[DIT_LAYERS];
  // Final RMSNorm and curve-form AdaLN weights (10752x8).
  Buf final_norm, final_adaln_w, final_adaln_b;
  // Non-zero when final_adaln_b is stored as IEEE FP16 instead of BF16.
  uint32_t final_adaln_bias_f16;
  // Unfolded FP32 PDD video_out head weights and biases.
  Buf pdd_w, pdd_b;
  // Optional unfolded FP32 audio_out head weights and biases.
  Buf audio_out_w, audio_out_b;
  // Compiled Metal kernel indices for linear, norm, AdaLN and attention ops.
  int k_x_bf16, k_mm_bf16, k_mm_f32, k_convrot, k_mm_int8, k_rmsnorm, k_add,
      k_swiglu_chunked, k_adaln, k_norm_mod, k_gate_add, k_heads, k_attend;
};

// Reads PDD step count and sigma schedule from checkpoint metadata and
// final_layer.video_out shape (1 step for video checkpoints, 6 for image).
static void parse_pdd_config(MinimaxH3Dit *dit) {
  Tensor vow = tensor_index_get(dit->tensors, "final_layer.video_out.weight");
  if (vow.dtype != 'f' || vow.ndim != 2 || vow.shape[1] != DIT_DIM ||
      vow.shape[0] <= 0 || vow.shape[0] % DIT_PATCH_DIM != 0)
    die("unexpected final_layer.video_out.weight shape");
  int weight_steps = (int)(vow.shape[0] / DIT_PATCH_DIM);
  if (weight_steps > MINIMAX_H3_DIT_MAX_PDD_STEPS)
    die("unsupported DiT PDD step count %d", weight_steps);

  dit->pdd_steps = weight_steps;
  if (weight_steps == DEFAULT_PDD_STEPS) {
    memcpy(dit->pdd_sigmas, DEFAULT_PDD_SIGMAS, sizeof DEFAULT_PDD_SIGMAS);
  } else {
    for (int i = 0; i <= weight_steps; i++)
      dit->pdd_sigmas[i] = (double)(weight_steps - i) / (double)weight_steps;
  }

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
      if (n == weight_steps && sigmas->n == n + 1) {
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
  memset(dit, 0, sizeof *dit);
  dit->gpu = gpu;
  dit->account = account;
  memory_slot_init(&dit->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-dit-workspace", NULL, NULL);
  // The workspace counts only its buffers in use, so freed buffers stop
  // counting at once.
  dit->workspace_slot.automatic = 1;
  dit->tensors = tensor_index_open_file(weights_path);
  dit->has_audio =
      tensor_index_has(dit->tensors, "audio_patch_proj.weight") &&
      tensor_index_has(dit->tensors, "final_layer.audio_out.weight");
  parse_pdd_config(dit);
  dit->weights = weight_group_new(gpu, account,
                                  dit->has_audio ? "minimax-h3-video-dit"
                                                 : "minimax-h3-image-dit");

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
    char name[MINIMAX_H3_NAME_CAP];
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

  // Optional FP32 audio patch embedder (present on FL2VA / REF2VA video DiT).
  if (dit->has_audio) {
    Tensor apw = tensor_index_get(dit->tensors, "audio_patch_proj.weight");
    Tensor apb = tensor_index_get(dit->tensors, "audio_patch_proj.bias");
    if (apw.dtype != 'f' || apw.shape[0] != DIT_DIM ||
        apw.shape[1] != MINIMAX_H3_AUDIO_LATENT_CHANNELS)
      die("unexpected audio_patch_proj.weight shape");
    if (apb.dtype != 'f' || apb.shape[0] != DIT_DIM)
      die("unexpected audio_patch_proj.bias shape");
    weight_group_add_tensor(dit->weights, &dit->audio_patch_proj_w, apw, 0);
    weight_group_add_tensor(dit->weights, &dit->audio_patch_proj_b, apb, 0);
  }

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
    if ((ab.dtype != 'b' && ab.dtype != 'h') || ab.shape[0] != DIT_ADALN_OUT)
      die("unexpected blocks.%d.adaln_proj.linear.bias shape", i);
    b->adaln_bias_f16 = (uint32_t)(ab.dtype == 'h');
    weight_group_add_tensor(dit->weights, &b->adaln_w, aw, 0);
    weight_group_add_tensor(dit->weights, &b->adaln_b, ab, 0);

    char name[MINIMAX_H3_NAME_CAP];
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
  if ((fab.dtype != 'b' && fab.dtype != 'h') ||
      fab.shape[0] != DIT_FINAL_ADALN_OUT)
    die("unexpected final_layer.adaln_proj.linear.bias shape");
  dit->final_adaln_bias_f16 = (uint32_t)(fab.dtype == 'h');
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

  // Optional unfolded FP32 audio_out head (present on FL2VA / REF2VA video
  // DiT).
  if (dit->has_audio) {
    Tensor aow = tensor_index_get(dit->tensors, "final_layer.audio_out.weight");
    Tensor aob = tensor_index_get(dit->tensors, "final_layer.audio_out.bias");
    int expected_aout = dit->pdd_steps * MINIMAX_H3_AUDIO_LATENT_CHANNELS;
    if (aow.dtype != 'f' || aow.shape[0] != expected_aout ||
        aow.shape[1] != DIT_DIM)
      die("unexpected final_layer.audio_out.weight shape");
    if (aob.dtype != 'f' || aob.shape[0] != expected_aout)
      die("unexpected final_layer.audio_out.bias shape");
    PddHeadParam aw_param = {
        .tensor = aow,
        .steps = dit->pdd_steps,
        .slice_floats =
            (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS * (size_t)DIT_DIM,
    };
    PddHeadParam ab_param = {
        .tensor = aob,
        .steps = dit->pdd_steps,
        .slice_floats = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    };
    weight_group_add_generated(dit->weights, &dit->audio_out_w, aow.size,
                               fill_pdd_head, &aw_param, sizeof aw_param);
    weight_group_add_generated(dit->weights, &dit->audio_out_b, aob.size,
                               fill_pdd_head, &ab_param, sizeof ab_param);
  }
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

int minimax_h3_dit_has_audio(const MinimaxH3Dit *dit) { return dit->has_audio; }

int minimax_h3_video_align_frames(int frame_count) {
  int n =
      frame_count < VIDEO_FRAME_GRID_BASE ? VIDEO_FRAME_GRID_BASE : frame_count;
  while (n % VIDEO_FRAME_GRID_STEP != VIDEO_FRAME_GRID_BASE) n++;
  return n;
}

int minimax_h3_video_latent_t(int frame_count) {
  if (frame_count <= VIDEO_FRAME_GRID_BASE) return VIDEO_LATENT_GRID_BASE;
  return ((frame_count - VIDEO_FRAME_GRID_BASE) / VIDEO_FRAME_GRID_STEP) *
             VIDEO_LATENT_GRID_STEP +
         VIDEO_LATENT_GRID_BASE;
}

int minimax_h3_video_audio_t(int frame_count) {
  int aligned = minimax_h3_video_align_frames(frame_count);
  double duration = (double)aligned / (double)MINIMAX_H3_VIDEO_FPS;
  return (int)llround(duration * (double)MINIMAX_H3_AUDIO_LATENT_RATE);
}

double minimax_h3_time_shift_sigma(double sigma, double from_shift,
                                   double to_shift) {
  double base = sigma / (from_shift + sigma * (1.0 - from_shift));
  return to_shift * base / (1.0 + (to_shift - 1.0) * base);
}

int minimax_h3_dit_load(MinimaxH3Dit *dit) {
  return weight_group_load(dit->weights);
}

// Interpolates the rank-8 curve-form AdaLN time embedding for t in [0, 1].
static void interp_t_emb(const MinimaxH3Dit *dit, double t,
                         float out[DIT_ADALN_RANK]) {
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
    double ch =
        ((double)py * (ratio_h / (double)nh) + (1.0 - ratio_h) * CELL_CENTER) *
        DIT_ROPE_SPATIAL_SCALE;
    for (int px = 0; px < nw; px++) {
      double cw = ((double)px * (ratio_w / (double)nw) +
                   (1.0 - ratio_w) * CELL_CENTER) *
                  DIT_ROPE_SPATIAL_SCALE;
      float *p = dst_pos + ((size_t)py * (size_t)nw + (size_t)px) *
                               (size_t)MINIMAX_H3_POS_AXES;
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
            row[c * DIT_PATCH_AREA + dy * DIT_PATCH_SIDE + dx] = v;
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
                dt * row[c * DIT_PATCH_AREA + dy * DIT_PATCH_SIDE + dx];
          }
        }
      }
    }
  }
}

// Frees all transient sampling workspace buffers (qkv and attn_out are aliases
// into fc1_out and fc1_mid and are not freed separately).
static void free_dit_workspace(DitWorkspace *ws) {
  gpu_free(&ws->audio_out);
  gpu_free(&ws->audio_in);
  gpu_free(&ws->patch_out);
  gpu_free(&ws->patch_in);
  gpu_free(&ws->pos);
  gpu_free(&ws->final_audio_mod_rows);
  gpu_free(&ws->final_mod_rows);
  gpu_free(&ws->mod_rows);
  gpu_free(&ws->mod);
  gpu_free(&ws->t_emb);
  gpu_free(&ws->fc1_mid);
  gpu_free(&ws->fc1_out);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->lora_mid);
  gpu_free(&ws->xscale);
  gpu_free(&ws->xq);
  gpu_free(&ws->xb);
  gpu_free(&ws->norm);
  gpu_free(&ws->h);
  gpu_free(&ws->text_refined);
  ws->qkv = (Buf){0};
  ws->attn_out = (Buf){0};
}

// Computes exact transient workspace bytes for either image or video sampling.
static uint64_t dit_workspace_bytes_impl(int text_len, int total_vis,
                                         int target_video_rows, int total_audio,
                                         int target_audio_rows, int max_t_rows,
                                         const MinimaxH3Lora *lora) {
  int seq_len = text_len + total_vis + total_audio;
  uint64_t padded_seq =
      ((uint64_t)seq_len + (uint64_t)TILE_M - 1) / (uint64_t)TILE_M * TILE_M;
  uint64_t padded_text =
      ((uint64_t)text_len + (uint64_t)TILE_M - 1) / (uint64_t)TILE_M * TILE_M;
  uint64_t xb_rows = lora ? padded_seq : padded_text;
  uint64_t bytes = 0;
  bytes += (uint64_t)text_len * (uint64_t)DIT_DIM * sizeof(float);
  bytes += 2u * (uint64_t)seq_len * (uint64_t)DIT_DIM * sizeof(float);
  bytes += xb_rows * (uint64_t)DIT_FFN_DIM * sizeof(uint16_t);
  bytes += padded_seq * (uint64_t)DIT_FFN_DIM;
  bytes += (uint64_t)seq_len * sizeof(float);
  bytes += minimax_h3_lora_workspace_bytes(lora, seq_len);
  bytes += (uint64_t)MINIMAX_H3_QKV_PARTS * (uint64_t)DIT_HEADS * padded_seq *
           (uint64_t)DIT_HEAD_DIM * sizeof(uint16_t);
  bytes += (uint64_t)seq_len * (uint64_t)DIT_FC1_WIDTH * sizeof(float);
  bytes += (uint64_t)seq_len * (uint64_t)DIT_FFN_DIM * sizeof(float);
  bytes += (uint64_t)max_t_rows * (uint64_t)DIT_ADALN_RANK * sizeof(float);
  bytes += (uint64_t)max_t_rows * (uint64_t)DIT_ADALN_OUT * sizeof(float);
  bytes += (uint64_t)seq_len * sizeof(uint32_t);
  bytes += (uint64_t)target_video_rows * sizeof(uint32_t);
  bytes += (uint64_t)target_audio_rows * sizeof(uint32_t);
  bytes += (uint64_t)seq_len * (uint64_t)MINIMAX_H3_POS_AXES * sizeof(float);
  bytes += (uint64_t)total_vis * (uint64_t)DIT_PATCH_DIM * sizeof(float);
  bytes +=
      (uint64_t)target_video_rows * (uint64_t)DIT_PATCH_DIM * sizeof(float);
  bytes += (uint64_t)total_audio * (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
           sizeof(float);
  bytes += (uint64_t)target_audio_rows *
           (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS * sizeof(float);
  return bytes;
}

uint64_t minimax_h3_dit_workspace_bytes(int text_len, int ref_patches,
                                        int target_patches) {
  return dit_workspace_bytes_impl(text_len, ref_patches + target_patches,
                                  target_patches, 0, 0,
                                  (int)DIT_IMAGE_MAX_T_ROWS, NULL);
}

uint64_t minimax_h3_dit_video_workspace_bytes(int text_len, int total_vis,
                                              int target_video_rows,
                                              int total_audio,
                                              int target_audio_rows,
                                              const MinimaxH3Lora *lora) {
  return dit_workspace_bytes_impl(text_len, total_vis, target_video_rows,
                                  total_audio, target_audio_rows,
                                  (int)DIT_VIDEO_MAX_T_ROWS, lora);
}

// Allocates transient sampling workspace buffers for the packed sequence.
// Attention qkv and attn_out alias fc1_out and fc1_mid because attention and
// MLP branches never overlap in liveness.
static int alloc_dit_workspace(MinimaxH3Dit *dit, int text_len, int total_vis,
                               int target_video_rows, int total_audio,
                               int target_audio_rows, int max_t_rows,
                               const MinimaxH3Lora *lora, DitWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  int seq_len = text_len + total_vis + total_audio;
  uint32_t padded_seq = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t padded_text = ((uint32_t)text_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t xb_rows = lora ? padded_seq : padded_text;
  uint64_t lora_mid_bytes = minimax_h3_lora_workspace_bytes(lora, seq_len);
  MemorySlot *slot = &dit->workspace_slot;

  ws->text_refined = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)text_len * DIT_DIM * sizeof(float));
  ws->h = gpu_try_alloc_required(dit->gpu, slot,
                                 (size_t)seq_len * DIT_DIM * sizeof(float));
  ws->norm = gpu_try_alloc_required(dit->gpu, slot,
                                    (size_t)seq_len * DIT_DIM * sizeof(float));
  ws->xb = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)xb_rows * DIT_FFN_DIM * sizeof(uint16_t));
  ws->xq =
      gpu_try_alloc_required(dit->gpu, slot, (size_t)padded_seq * DIT_FFN_DIM);
  ws->xscale =
      gpu_try_alloc_required(dit->gpu, slot, (size_t)seq_len * sizeof(float));
  if (lora_mid_bytes > 0)
    ws->lora_mid = gpu_try_alloc_required(dit->gpu, slot, lora_mid_bytes);
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
  ws->qkv = ws->fc1_out;
  ws->attn_out = ws->fc1_mid;
  ws->t_emb = gpu_try_alloc_required(
      dit->gpu, slot,
      (size_t)max_t_rows * (size_t)DIT_ADALN_RANK * sizeof(float));
  ws->mod = gpu_try_alloc_required(dit->gpu, slot,
                                   (size_t)max_t_rows * (size_t)DIT_ADALN_OUT *
                                       sizeof(float));
  ws->mod_rows = gpu_try_alloc_required(dit->gpu, slot,
                                        (size_t)seq_len * sizeof(uint32_t));
  ws->final_mod_rows = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)target_video_rows * sizeof(uint32_t));
  if (target_audio_rows > 0)
    ws->final_audio_mod_rows = gpu_try_alloc_required(
        dit->gpu, slot, (size_t)target_audio_rows * sizeof(uint32_t));
  ws->pos = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)seq_len * MINIMAX_H3_POS_AXES * sizeof(float));
  ws->patch_in = gpu_try_alloc_required(
      dit->gpu, slot, (size_t)total_vis * DIT_PATCH_DIM * sizeof(float));
  ws->patch_out = gpu_try_alloc_required(dit->gpu, slot,
                                         (size_t)target_video_rows *
                                             DIT_PATCH_DIM * sizeof(float));
  if (total_audio > 0)
    ws->audio_in = gpu_try_alloc_required(
        dit->gpu, slot,
        (size_t)total_audio * (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
            sizeof(float));
  if (target_audio_rows > 0)
    ws->audio_out = gpu_try_alloc_required(
        dit->gpu, slot,
        (size_t)target_audio_rows * (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
            sizeof(float));

  if (!ws->text_refined.obj || !ws->h.obj || !ws->norm.obj || !ws->xb.obj ||
      !ws->xq.obj || !ws->xscale.obj ||
      (lora_mid_bytes > 0 && !ws->lora_mid.obj) || !ws->qh.obj || !ws->kh.obj ||
      !ws->vh.obj || !ws->fc1_out.obj || !ws->fc1_mid.obj || !ws->t_emb.obj ||
      !ws->mod.obj || !ws->mod_rows.obj || !ws->final_mod_rows.obj ||
      (target_audio_rows > 0 && !ws->final_audio_mod_rows.obj) ||
      !ws->pos.obj || !ws->patch_in.obj || !ws->patch_out.obj ||
      (total_audio > 0 && !ws->audio_in.obj) ||
      (target_audio_rows > 0 && !ws->audio_out.obj)) {
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
  gpu_run_kernel(
      cmd,
      ((uint32_t)lin->k / MINIMAX_H3_VEC4_WIDTH + GRID2D_THREADS - 1) /
          GRID2D_THREADS,
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

// Encodes ConvRot Hadamard + per-row INT8 quantization + native INT8 matmul
// for rows starting at byte offsets x_off in x and y_off in y.
static void run_convrot_linear_at(const MinimaxH3Dit *dit,
                                  GpuCommandBuffer *cmd, const Buf *x,
                                  size_t x_off, const Buf *xq,
                                  const Buf *xscale, const Int8Lin *lin,
                                  const Buf *y, size_t y_off, int rows) {
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  ConvRotParam rp = {(uint32_t)rows, (uint32_t)lin->k, (uint32_t)lin->k};
  gpu_select_kernel(cmd, dit->k_convrot);
  gpu_bind_buffer(cmd, 0, x, x_off);
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
  gpu_bind_buffer(cmd, 5, y, y_off);
  gpu_copy_bytes(cmd, 6, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N, padded / TILE_M,
                 1, MM_THREADS, 1, 1);
}

// Encodes ConvRot Hadamard + per-row INT8 quantization + native INT8 matmul
// at zero buffer offsets.
static void run_convrot_linear(const MinimaxH3Dit *dit, GpuCommandBuffer *cmd,
                               const Buf *x, const Buf *xq, const Buf *xscale,
                               const Int8Lin *lin, const Buf *y, int rows) {
  run_convrot_linear_at(dit, cmd, x, 0, xq, xscale, lin, y, 0, rows);
}

// Picks how many 64-row tiles to dispatch in one contended command buffer from
// remaining tiles, the remaining measured turn budget in seconds, and the last
// measured GPU seconds per tile. Runs at least 1 tile and at most remaining.
static int pick_slice_tiles(int remaining, double budget, double sec_per_tile) {
  if (remaining <= 1) return remaining;
  if (sec_per_tile <= 0.0 || budget <= 0.0) return 1;
  double fit = floor(budget / sec_per_tile);
  if (fit < 1.0) return 1;
  if (fit >= (double)remaining) return remaining;
  return (int)fit;
}

// Runs one INT8-ConvRot linear projection (plus optional BF16 LoRA) in
// adaptive 64-row tile slices while contending, yielding between slices and
// updating *sec_per_tile from measured GPU execution time. Returns 1 on
// completion or 0 if stopped.
static int run_convrot_linear_sliced(
    const MinimaxH3Dit *dit, const DitWorkspace *ws, const Buf *x,
    const Int8Lin *lin, const MinimaxH3Lora *lora,
    const MinimaxH3LoraTarget *ltarget, const Buf *y, int seq_len,
    const MinimaxH3RunHooks *hooks, double *sec_per_tile) {
  int total_tiles = (seq_len + (int)TILE_M - 1) / (int)TILE_M;
  for (int tile_off = 0; tile_off < total_tiles;) {
    if (minimax_h3_should_stop(hooks)) return 0;
    int rem_tiles = total_tiles - tile_off;
    int slice_tiles =
        minimax_h3_hooks_contended(hooks)
            ? pick_slice_tiles(rem_tiles, minimax_h3_hooks_turn_budget(hooks),
                               *sec_per_tile)
            : rem_tiles;
    int row0 = tile_off * (int)TILE_M;
    int slice_rows = slice_tiles * (int)TILE_M;
    if (row0 + slice_rows > seq_len) slice_rows = seq_len - row0;
    size_t x_off = (size_t)row0 * (size_t)lin->k * sizeof(float);
    size_t y_off = (size_t)row0 * (size_t)lin->n * sizeof(float);

    GpuCommandBuffer cmd = gpu_begin(dit->gpu);
    run_convrot_linear_at(dit, &cmd, x, x_off, &ws->xq, &ws->xscale, lin, y,
                          y_off, slice_rows);
    if (ltarget)
      minimax_h3_lora_apply_at(lora, &cmd, ltarget, x, x_off, &ws->xb,
                               &ws->lora_mid, y, y_off, slice_rows, 0);
    double t0 = now();
    gpu_commit(&cmd);
    minimax_h3_gpu_wait(dit->gpu, cmd.number);
    double dt = now() - t0;
    if (dt > 0.0) *sec_per_tile = dt / (double)slice_tiles;
    minimax_h3_yield_hooks(hooks);
    tile_off += slice_tiles;
  }
  return 1;
}

// Runs one FP32 linear projection in adaptive 64-row tile slices while
// contending, yielding between slices and updating *sec_per_tile from measured
// GPU execution time. Returns 1 on completion or 0 if stopped.
static int run_f32_linear_sliced(const MinimaxH3Dit *dit, const Buf *w,
                                 size_t w_off, const Buf *x, size_t x_base_off,
                                 const Buf *b, size_t b_off, const Buf *y,
                                 size_t y_base_off, int rows, int out_dim,
                                 int in_dim, const MinimaxH3RunHooks *hooks,
                                 double *sec_per_tile) {
  int total_tiles = (rows + (int)TILE_M - 1) / (int)TILE_M;
  for (int tile_off = 0; tile_off < total_tiles;) {
    if (minimax_h3_should_stop(hooks)) return 0;
    int rem_tiles = total_tiles - tile_off;
    int slice_tiles =
        minimax_h3_hooks_contended(hooks)
            ? pick_slice_tiles(rem_tiles, minimax_h3_hooks_turn_budget(hooks),
                               *sec_per_tile)
            : rem_tiles;
    int row0 = tile_off * (int)TILE_M;
    int slice_rows = slice_tiles * (int)TILE_M;
    if (row0 + slice_rows > rows) slice_rows = rows - row0;
    size_t x_off = x_base_off + (size_t)row0 * (size_t)in_dim * sizeof(float);
    size_t y_off = y_base_off + (size_t)row0 * (size_t)out_dim * sizeof(float);
    F32MMParam ep = {(uint32_t)slice_rows, (uint32_t)out_dim, (uint32_t)in_dim,
                     (uint32_t)out_dim};
    GpuCommandBuffer cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_mm_f32);
    gpu_bind_buffer(&cmd, 0, w, w_off);
    gpu_bind_buffer(&cmd, 1, x, x_off);
    gpu_bind_buffer(&cmd, 2, b, b_off);
    gpu_bind_buffer(&cmd, 3, y, y_off);
    gpu_copy_bytes(&cmd, 4, &ep, sizeof ep);
    gpu_run_kernel(&cmd, (uint32_t)out_dim, (uint32_t)slice_rows, 1,
                   F32MM_THREADS, 1, 1);
    double t0 = now();
    gpu_commit(&cmd);
    minimax_h3_gpu_wait(dit->gpu, cmd.number);
    double dt = now() - t0;
    if (dt > 0.0) *sec_per_tile = dt / (double)slice_tiles;
    minimax_h3_yield_hooks(hooks);
    tile_off += slice_tiles;
  }
  return 1;
}

// Runs bidirectional 56-head DiT FlashAttention in adaptive 64-query-row tile
// slices while contending, yielding between slices and updating *sec_per_tile.
// Returns 1 on completion or 0 if stopped.
static int run_dit_attend_sliced(const MinimaxH3Dit *dit,
                                 const DitWorkspace *ws, int seq_len,
                                 uint32_t padded_seq,
                                 const MinimaxH3RunHooks *hooks,
                                 double *sec_per_tile) {
  int total_tiles = (int)(padded_seq / TILE_M);
  for (int tile_off = 0; tile_off < total_tiles;) {
    if (minimax_h3_should_stop(hooks)) return 0;
    int rem_tiles = total_tiles - tile_off;
    int slice_tiles =
        minimax_h3_hooks_contended(hooks)
            ? pick_slice_tiles(rem_tiles, minimax_h3_hooks_turn_budget(hooks),
                               *sec_per_tile)
            : rem_tiles;
    DitAttendParam ap = {(uint32_t)seq_len, padded_seq, (uint32_t)tile_off};
    GpuCommandBuffer cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_attend);
    gpu_bind_buffer(&cmd, 0, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws->vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->attn_out, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, (uint32_t)slice_tiles, (uint32_t)DIT_HEADS, 1,
                   MM_THREADS, 1, 1);
    double t0 = now();
    gpu_commit(&cmd);
    minimax_h3_gpu_wait(dit->gpu, cmd.number);
    double dt = now() - t0;
    if (dt > 0.0) *sec_per_tile = dt / (double)slice_tiles;
    minimax_h3_yield_hooks(hooks);
    tile_off += slice_tiles;
  }
  return 1;
}

// Encodes condition_proj and the 2-layer BF16 token refiner once per sample,
// applying optional BF16 LoRA deltas on the four projections per block.
static void run_token_refiner(MinimaxH3Dit *dit, const Buf *text_hidden,
                              int text_len, const DitWorkspace *ws,
                              const MinimaxH3Lora *lora) {
  uint32_t padded = ((uint32_t)text_len + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t count = (uint32_t)text_len * (uint32_t)DIT_DIM;
  RmsNormParam np = {(uint32_t)DIT_DIM, DIT_NORM_EPS};
  SwigluChunkedParam swp = {(uint32_t)text_len, (uint32_t)DIT_FFN_DIM};
  DitHeadsParam hp = {(uint32_t)text_len, padded, 0, DIT_NORM_EPS};
  DitAttendParam ap = {(uint32_t)text_len, padded, 0u};

  GpuCommandBuffer cmd = gpu_begin(dit->gpu);
  run_bf16_linear(dit, &cmd, text_hidden, &ws->xb, &dit->condition_proj,
                  &ws->text_refined, text_len);

  for (int i = 0; i < DIT_REFINER_LAYERS; i++) {
    const RefinerBlock *r = &dit->refiner[i];
    const MinimaxH3LoraBlock *lb = minimax_h3_lora_refiner_block(lora, i);

    gpu_select_kernel(&cmd, dit->k_rmsnorm);
    gpu_bind_buffer(&cmd, 0, &ws->text_refined, 0);
    gpu_bind_buffer(&cmd, 1, &r->norm1, 0);
    gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 3, &np, sizeof np);
    gpu_run_kernel(&cmd, (uint32_t)text_len, 1, 1, NORM_THREADS, 1, 1);

    run_bf16_linear(dit, &cmd, &ws->norm, &ws->xb, &r->qkv_proj, &ws->qkv,
                    text_len);
    if (lb)
      minimax_h3_lora_apply(lora, &cmd, &lb->qkv_proj, &ws->norm, &ws->xb,
                            &ws->lora_mid, &ws->qkv, text_len, 1);

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
    gpu_run_kernel(&cmd, (uint32_t)(MINIMAX_H3_QKV_PARTS * DIT_HEADS), padded,
                   1, HEADS_THREADS, 1, 1);

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
    if (lb)
      minimax_h3_lora_apply(lora, &cmd, &lb->out_proj, &ws->attn_out, &ws->xb,
                            &ws->lora_mid, &ws->norm, text_len, 1);

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
    if (lb)
      minimax_h3_lora_apply(lora, &cmd, &lb->fc1, &ws->norm, &ws->xb,
                            &ws->lora_mid, &ws->fc1_out, text_len, 1);

    gpu_select_kernel(&cmd, dit->k_swiglu_chunked);
    gpu_bind_buffer(&cmd, 0, &ws->fc1_out, 0);
    gpu_bind_buffer(&cmd, 1, &ws->fc1_mid, 0);
    gpu_copy_bytes(&cmd, 2, &swp, sizeof swp);
    gpu_run_kernel(&cmd, (uint32_t)DIT_FFN_DIM / GRID2D_THREADS,
                   (uint32_t)text_len, 1, GRID2D_THREADS, 1, 1);

    run_bf16_linear(dit, &cmd, &ws->fc1_mid, &ws->xb, &r->fc2, &ws->norm,
                    text_len);
    if (lb)
      minimax_h3_lora_apply(lora, &cmd, &lb->fc2, &ws->fc1_mid, &ws->xb,
                            &ws->lora_mid, &ws->norm, text_len, 1);

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

// Encodes one complete INT8-ConvRot DiT block (with optional BF16 LoRA) into
// cmd as a single command buffer.
static void encode_dit_block_full(const MinimaxH3Dit *dit,
                                  GpuCommandBuffer *cmd, const DitWorkspace *ws,
                                  const DitBlock *b, const MinimaxH3Lora *lora,
                                  const MinimaxH3LoraBlock *lb, int seq_len,
                                  uint32_t padded_seq, uint32_t num_t) {
  DitNormModParam msa_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MSA, OFF_SCALE_MSA, DIT_NORM_EPS};
  DitNormModParam mlp_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MLP, OFF_SCALE_MLP, DIT_NORM_EPS};
  DitGateAddParam msa_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MSA};
  DitGateAddParam mlp_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MLP};
  DitHeadsParam heads_p = {(uint32_t)seq_len, padded_seq, 1u, DIT_NORM_EPS};
  DitAttendParam attend_p = {(uint32_t)seq_len, padded_seq, 0u};
  SwigluChunkedParam swiglu_p = {(uint32_t)seq_len, (uint32_t)DIT_FFN_DIM};
  DitAdalnParam block_adaln_p = {num_t, (uint32_t)DIT_ADALN_OUT,
                                 b->adaln_bias_f16};

  gpu_select_kernel(cmd, dit->k_adaln);
  gpu_bind_buffer(cmd, 0, &b->adaln_w, 0);
  gpu_bind_buffer(cmd, 1, &b->adaln_b, 0);
  gpu_bind_buffer(cmd, 2, &ws->t_emb, 0);
  gpu_bind_buffer(cmd, 3, &ws->mod, 0);
  gpu_copy_bytes(cmd, 4, &block_adaln_p, sizeof block_adaln_p);
  gpu_run_kernel(cmd,
                 ((uint32_t)DIT_ADALN_OUT + ELEM_THREADS - 1) / ELEM_THREADS,
                 num_t, 1, ELEM_THREADS, 1, 1);

  gpu_select_kernel(cmd, dit->k_norm_mod);
  gpu_bind_buffer(cmd, 0, &ws->h, 0);
  gpu_bind_buffer(cmd, 1, &b->norm1, 0);
  gpu_bind_buffer(cmd, 2, &ws->mod, 0);
  gpu_bind_buffer(cmd, 3, &ws->mod_rows, 0);
  gpu_bind_buffer(cmd, 4, &ws->norm, 0);
  gpu_copy_bytes(cmd, 5, &msa_norm_p, sizeof msa_norm_p);
  gpu_run_kernel(cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);

  run_convrot_linear(dit, cmd, &ws->norm, &ws->xq, &ws->xscale, &b->qkv_proj,
                     &ws->qkv, seq_len);
  if (lb)
    minimax_h3_lora_apply(lora, cmd, &lb->qkv_proj, &ws->norm, &ws->xb,
                          &ws->lora_mid, &ws->qkv, seq_len, 0);

  gpu_select_kernel(cmd, dit->k_heads);
  gpu_bind_buffer(cmd, 0, &ws->qkv, 0);
  gpu_bind_buffer(cmd, 1, &b->q_norm, 0);
  gpu_bind_buffer(cmd, 2, &b->k_norm, 0);
  gpu_bind_buffer(cmd, 3, &ws->pos, 0);
  gpu_bind_buffer(cmd, 4, &dit->rope_inv_freq, 0);
  gpu_bind_buffer(cmd, 5, &ws->qh, 0);
  gpu_bind_buffer(cmd, 6, &ws->kh, 0);
  gpu_bind_buffer(cmd, 7, &ws->vh, 0);
  gpu_copy_bytes(cmd, 8, &heads_p, sizeof heads_p);
  gpu_run_kernel(cmd, (uint32_t)(MINIMAX_H3_QKV_PARTS * DIT_HEADS), padded_seq,
                 1, HEADS_THREADS, 1, 1);

  gpu_select_kernel(cmd, dit->k_attend);
  gpu_bind_buffer(cmd, 0, &ws->qh, 0);
  gpu_bind_buffer(cmd, 1, &ws->kh, 0);
  gpu_bind_buffer(cmd, 2, &ws->vh, 0);
  gpu_bind_buffer(cmd, 3, &ws->attn_out, 0);
  gpu_copy_bytes(cmd, 4, &attend_p, sizeof attend_p);
  gpu_run_kernel(cmd, padded_seq / TILE_M, (uint32_t)DIT_HEADS, 1, MM_THREADS,
                 1, 1);

  run_convrot_linear(dit, cmd, &ws->attn_out, &ws->xq, &ws->xscale,
                     &b->out_proj, &ws->norm, seq_len);
  if (lb)
    minimax_h3_lora_apply(lora, cmd, &lb->out_proj, &ws->attn_out, &ws->xb,
                          &ws->lora_mid, &ws->norm, seq_len, 0);

  gpu_select_kernel(cmd, dit->k_gate_add);
  gpu_bind_buffer(cmd, 0, &ws->h, 0);
  gpu_bind_buffer(cmd, 1, &ws->norm, 0);
  gpu_bind_buffer(cmd, 2, &ws->mod, 0);
  gpu_bind_buffer(cmd, 3, &ws->mod_rows, 0);
  gpu_copy_bytes(cmd, 4, &msa_gate_p, sizeof msa_gate_p);
  gpu_run_kernel(cmd, ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);

  gpu_select_kernel(cmd, dit->k_norm_mod);
  gpu_bind_buffer(cmd, 0, &ws->h, 0);
  gpu_bind_buffer(cmd, 1, &b->norm2, 0);
  gpu_bind_buffer(cmd, 2, &ws->mod, 0);
  gpu_bind_buffer(cmd, 3, &ws->mod_rows, 0);
  gpu_bind_buffer(cmd, 4, &ws->norm, 0);
  gpu_copy_bytes(cmd, 5, &mlp_norm_p, sizeof mlp_norm_p);
  gpu_run_kernel(cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);

  run_convrot_linear(dit, cmd, &ws->norm, &ws->xq, &ws->xscale, &b->fc1,
                     &ws->fc1_out, seq_len);
  if (lb)
    minimax_h3_lora_apply(lora, cmd, &lb->fc1, &ws->norm, &ws->xb,
                          &ws->lora_mid, &ws->fc1_out, seq_len, 0);

  gpu_select_kernel(cmd, dit->k_swiglu_chunked);
  gpu_bind_buffer(cmd, 0, &ws->fc1_out, 0);
  gpu_bind_buffer(cmd, 1, &ws->fc1_mid, 0);
  gpu_copy_bytes(cmd, 2, &swiglu_p, sizeof swiglu_p);
  gpu_run_kernel(cmd, (uint32_t)DIT_FFN_DIM / GRID2D_THREADS, (uint32_t)seq_len,
                 1, GRID2D_THREADS, 1, 1);

  run_convrot_linear(dit, cmd, &ws->fc1_mid, &ws->xq, &ws->xscale, &b->fc2,
                     &ws->norm, seq_len);
  if (lb)
    minimax_h3_lora_apply(lora, cmd, &lb->fc2, &ws->fc1_mid, &ws->xb,
                          &ws->lora_mid, &ws->norm, seq_len, 0);

  gpu_select_kernel(cmd, dit->k_gate_add);
  gpu_bind_buffer(cmd, 0, &ws->h, 0);
  gpu_bind_buffer(cmd, 1, &ws->norm, 0);
  gpu_bind_buffer(cmd, 2, &ws->mod, 0);
  gpu_bind_buffer(cmd, 3, &ws->mod_rows, 0);
  gpu_copy_bytes(cmd, 4, &mlp_gate_p, sizeof mlp_gate_p);
  gpu_run_kernel(cmd, ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);
}

// Runs the 50 INT8-ConvRot DiT blocks (with optional runtime BF16 LoRA deltas).
// Runs full pipelined single-command-buffer blocks when alone, and drains and
// slices heavy stages to the measured turn budget when sharing the GPU.
// Updates *prev_cmd to the last committed unawaited command buffer number (0
// when drained). Returns 1 on completion or 0 if stopped.
static int run_dit_blocks(const MinimaxH3Dit *dit, const DitWorkspace *ws,
                          int seq_len, uint32_t num_t,
                          const MinimaxH3Lora *lora,
                          const MinimaxH3RunHooks *hooks, DitSlicePace *pace,
                          uint64_t *prev_cmd) {
  uint32_t padded_seq = ((uint32_t)seq_len + TILE_M - 1) / TILE_M * TILE_M;
  int total_tiles = (int)(padded_seq / TILE_M);
  DitNormModParam msa_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MSA, OFF_SCALE_MSA, DIT_NORM_EPS};
  DitNormModParam mlp_norm_p = {(uint32_t)DIT_DIM, (uint32_t)DIT_ADALN_STRIDE,
                                OFF_SHIFT_MLP, OFF_SCALE_MLP, DIT_NORM_EPS};
  DitGateAddParam msa_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MSA};
  DitGateAddParam mlp_gate_p = {(uint32_t)seq_len, (uint32_t)DIT_DIM,
                                (uint32_t)DIT_ADALN_STRIDE, OFF_GATE_MLP};
  DitHeadsParam heads_p = {(uint32_t)seq_len, padded_seq, 1u, DIT_NORM_EPS};
  SwigluChunkedParam swiglu_p = {(uint32_t)seq_len, (uint32_t)DIT_FFN_DIM};

  for (int i = 0; i < DIT_LAYERS; i++) {
    if (minimax_h3_should_stop(hooks)) return 0;
    const DitBlock *b = &dit->blocks[i];
    const MinimaxH3LoraBlock *lb = minimax_h3_lora_dit_block(lora, i);
    if (!minimax_h3_hooks_contended(hooks)) {
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      encode_dit_block_full(dit, &cmd, ws, b, lora, lb, seq_len, padded_seq,
                            num_t);
      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(dit->gpu, *prev_cmd, hooks);
      *prev_cmd = cmd.number;
      continue;
    }

    if (*prev_cmd) {
      minimax_h3_gpu_wait_hooks(dit->gpu, *prev_cmd, hooks);
      *prev_cmd = 0;
      if (minimax_h3_should_stop(hooks)) return 0;
      if (!minimax_h3_hooks_contended(hooks)) {
        GpuCommandBuffer cmd = gpu_begin(dit->gpu);
        encode_dit_block_full(dit, &cmd, ws, b, lora, lb, seq_len, padded_seq,
                              num_t);
        gpu_commit(&cmd);
        *prev_cmd = cmd.number;
        continue;
      }
    }

    // When a measured whole block already fits within the peer's measured turn
    // budget, submit it as one drained command buffer to avoid intra-block
    // synchronization overhead.
    double budget = minimax_h3_hooks_turn_budget(hooks);
    double est_block =
        pace->block > 0.0
            ? pace->block
            : (double)total_tiles * (pace->qkv + pace->attend + pace->out_proj +
                                     pace->fc1 + pace->fc2);
    if (pace->measured && est_block > 0.0 && budget > 0.0 &&
        est_block <= budget) {
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      encode_dit_block_full(dit, &cmd, ws, b, lora, lb, seq_len, padded_seq,
                            num_t);
      double t0 = now();
      gpu_commit(&cmd);
      minimax_h3_gpu_wait(dit->gpu, cmd.number);
      double dt = now() - t0;
      if (dt > 0.0) pace->block = dt;
      minimax_h3_yield_hooks(hooks);
      continue;
    }

    // Contended heavy block: slice each heavy matmul and FlashAttention pass
    // dynamically to match the measured turn budget.
    DitAdalnParam block_adaln_p = {num_t, (uint32_t)DIT_ADALN_OUT,
                                   b->adaln_bias_f16};
    GpuCommandBuffer cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_adaln);
    gpu_bind_buffer(&cmd, 0, &b->adaln_w, 0);
    gpu_bind_buffer(&cmd, 1, &b->adaln_b, 0);
    gpu_bind_buffer(&cmd, 2, &ws->t_emb, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mod, 0);
    gpu_copy_bytes(&cmd, 4, &block_adaln_p, sizeof block_adaln_p);
    gpu_run_kernel(&cmd,
                   ((uint32_t)DIT_ADALN_OUT + ELEM_THREADS - 1) / ELEM_THREADS,
                   num_t, 1, ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, dit->k_norm_mod);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm1, 0);
    gpu_bind_buffer(&cmd, 2, &ws->mod, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mod_rows, 0);
    gpu_bind_buffer(&cmd, 4, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 5, &msa_norm_p, sizeof msa_norm_p);
    gpu_run_kernel(&cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);
    gpu_commit(&cmd);
    uint64_t pending = cmd.number;

    if (!run_convrot_linear_sliced(dit, ws, &ws->norm, &b->qkv_proj, lora,
                                   lb ? &lb->qkv_proj : NULL, &ws->qkv, seq_len,
                                   hooks, &pace->qkv)) {
      minimax_h3_gpu_wait(dit->gpu, pending);
      return 0;
    }

    if (minimax_h3_should_stop(hooks)) return 0;
    cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_heads);
    gpu_bind_buffer(&cmd, 0, &ws->qkv, 0);
    gpu_bind_buffer(&cmd, 1, &b->q_norm, 0);
    gpu_bind_buffer(&cmd, 2, &b->k_norm, 0);
    gpu_bind_buffer(&cmd, 3, &ws->pos, 0);
    gpu_bind_buffer(&cmd, 4, &dit->rope_inv_freq, 0);
    gpu_bind_buffer(&cmd, 5, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 6, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 7, &ws->vh, 0);
    gpu_copy_bytes(&cmd, 8, &heads_p, sizeof heads_p);
    gpu_run_kernel(&cmd, (uint32_t)(MINIMAX_H3_QKV_PARTS * DIT_HEADS),
                   padded_seq, 1, HEADS_THREADS, 1, 1);
    gpu_commit(&cmd);
    pending = cmd.number;

    if (!run_dit_attend_sliced(dit, ws, seq_len, padded_seq, hooks,
                               &pace->attend)) {
      minimax_h3_gpu_wait(dit->gpu, pending);
      return 0;
    }

    if (!run_convrot_linear_sliced(dit, ws, &ws->attn_out, &b->out_proj, lora,
                                   lb ? &lb->out_proj : NULL, &ws->norm,
                                   seq_len, hooks, &pace->out_proj))
      return 0;

    if (minimax_h3_should_stop(hooks)) return 0;
    cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_gate_add);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &ws->mod, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mod_rows, 0);
    gpu_copy_bytes(&cmd, 4, &msa_gate_p, sizeof msa_gate_p);
    gpu_run_kernel(&cmd,
                   ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                   (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);

    gpu_select_kernel(&cmd, dit->k_norm_mod);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm2, 0);
    gpu_bind_buffer(&cmd, 2, &ws->mod, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mod_rows, 0);
    gpu_bind_buffer(&cmd, 4, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 5, &mlp_norm_p, sizeof mlp_norm_p);
    gpu_run_kernel(&cmd, (uint32_t)seq_len, 1, 1, NORM_THREADS, 1, 1);
    gpu_commit(&cmd);
    pending = cmd.number;

    if (!run_convrot_linear_sliced(dit, ws, &ws->norm, &b->fc1, lora,
                                   lb ? &lb->fc1 : NULL, &ws->fc1_out, seq_len,
                                   hooks, &pace->fc1)) {
      minimax_h3_gpu_wait(dit->gpu, pending);
      return 0;
    }

    if (minimax_h3_should_stop(hooks)) return 0;
    cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_swiglu_chunked);
    gpu_bind_buffer(&cmd, 0, &ws->fc1_out, 0);
    gpu_bind_buffer(&cmd, 1, &ws->fc1_mid, 0);
    gpu_copy_bytes(&cmd, 2, &swiglu_p, sizeof swiglu_p);
    gpu_run_kernel(&cmd, (uint32_t)DIT_FFN_DIM / GRID2D_THREADS,
                   (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);
    gpu_commit(&cmd);
    pending = cmd.number;

    if (!run_convrot_linear_sliced(dit, ws, &ws->fc1_mid, &b->fc2, lora,
                                   lb ? &lb->fc2 : NULL, &ws->norm, seq_len,
                                   hooks, &pace->fc2)) {
      minimax_h3_gpu_wait(dit->gpu, pending);
      return 0;
    }

    if (minimax_h3_should_stop(hooks)) return 0;
    cmd = gpu_begin(dit->gpu);
    gpu_select_kernel(&cmd, dit->k_gate_add);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &ws->mod, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mod_rows, 0);
    gpu_copy_bytes(&cmd, 4, &mlp_gate_p, sizeof mlp_gate_p);
    gpu_run_kernel(&cmd,
                   ((uint32_t)DIT_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                   (uint32_t)seq_len, 1, GRID2D_THREADS, 1, 1);
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(dit->gpu, cmd.number, hooks);
    pace->measured = 1;
  }
  return 1;
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
  if (!alloc_dit_workspace(dit, text_len, total_vis, target_patches, 0, 0,
                           (int)DIT_IMAGE_MAX_T_ROWS, NULL, &ws)) {
    memory_pin(wslot, (int)was_pinned);
    return 0;
  }

  // Build 3D MM-RoPE coordinates [seq_len, MINIMAX_H3_POS_AXES] and per-token
  // AdaLN row indices.
  float *pos = (float *)ws.pos.ptr;
  uint32_t *mod_rows = (uint32_t *)ws.mod_rows.ptr;
  for (int i = 0; i < text_len; i++) {
    pos[MINIMAX_H3_POS_AXES * i] = (float)i;
    pos[MINIMAX_H3_POS_AXES * i + 1] = 0.0f;
    pos[MINIMAX_H3_POS_AXES * i + 2] = 0.0f;
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
    fill_frame_positions(pos + (size_t)row_off * (size_t)MINIMAX_H3_POS_AXES,
                         rh, rw, cursor);
    uint32_t ref_mod_row =
        DIT_REF_T_ROW * (uint32_t)DIT_ADALN_MODALITIES + TAG_VISUAL;
    for (int j = 0; j < n; j++) mod_rows[row_off + j] = ref_mod_row;

    float *dst = patch_in + (size_t)vis_off * DIT_PATCH_DIM;
    patchify_latent(refs[i].latents, refs[i].h_lat, refs[i].w_lat, rh, rw, dst);
    size_t count = (size_t)n * DIT_PATCH_DIM;
    float *noise = xmalloc(count * sizeof(float));
    minimax_h3_fill_gaussian(noise, count, seed);
    float aug = (float)DIT_VISUAL_COND_TIMESTEP;
    for (size_t k = 0; k < count; k++)
      dst[k] = aug * dst[k] + (1.0f - aug) * noise[k];
    free(noise);

    cursor += 1.0;
    row_off += n;
    vis_off += n;
  }

  fill_frame_positions(pos + (size_t)row_off * (size_t)MINIMAX_H3_POS_AXES,
                       lat_h, lat_w, cursor);
  for (int j = 0; j < target_patches; j++) mod_rows[row_off + j] = TAG_VISUAL;
  memset(ws.final_mod_rows.ptr, 0, (size_t)target_patches * sizeof(uint32_t));

  // Refine text embeddings once for all PDD denoising steps.
  run_token_refiner(dit, text_hidden, text_len, &ws, NULL);

  // Initialize target latent with standard normal noise from seed.
  size_t latent_floats =
      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h_lat * (size_t)w_lat;
  minimax_h3_fill_gaussian(out_latents, latent_floats, seed);

  uint32_t num_t = ref_count > 0 ? 2u : 1u;
  DitAdalnParam final_adaln_p = {1u, (uint32_t)DIT_FINAL_ADALN_OUT,
                                 dit->final_adaln_bias_f16};
  DitNormModParam final_norm_p = {(uint32_t)DIT_DIM,
                                  (uint32_t)DIT_FINAL_ADALN_OUT, 0,
                                  (uint32_t)DIT_DIM, DIT_NORM_EPS};

  size_t text_bytes = (size_t)text_len * (size_t)DIT_DIM * sizeof(float);
  size_t target_h_off =
      (size_t)(text_len + ref_patches) * (size_t)DIT_DIM * sizeof(float);

  // The latent that a step predicts for the run's end, for on_estimate.
  float *estimate = hooks && hooks->on_estimate
                        ? xmalloc(latent_floats * sizeof(float))
                        : NULL;

  // 6-step PDD Euler flow-matching loop.
  if (hooks && hooks->image_steps_total)
    *hooks->image_steps_total = dit->pdd_steps;
  DitSlicePace pace = {0};
  int stopped = 0;
  for (int step = 0; step < dit->pdd_steps && !stopped; step++) {
    if (minimax_h3_should_stop(hooks)) {
      stopped = 1;
      break;
    }
    if (hooks && hooks->image_step) *hooks->image_step = step + 1;
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

    uint64_t prev_cmd = 0;
    if (!minimax_h3_hooks_contended(hooks)) {
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
      prev_cmd = cmd.number;
    } else if (!run_f32_linear_sliced(dit, &dit->patch_proj_w, 0, &ws.patch_in,
                                      0, &dit->patch_proj_b, 0, &ws.h,
                                      text_bytes, total_vis, DIT_DIM,
                                      DIT_PATCH_DIM, hooks, &pace.patch_in)) {
      stopped = 1;
      break;
    }

    // 50 INT8-ConvRot DiT blocks, yielding cooperatively between blocks.
    if (!run_dit_blocks(dit, &ws, seq_len, num_t, NULL, hooks, &pace,
                        &prev_cmd)) {
      stopped = 1;
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      break;
    }

    // FinalLayer on target image slice using PDD head for step.
    size_t pdd_w_off =
        (size_t)step * (size_t)DIT_PATCH_DIM * (size_t)DIT_DIM * sizeof(float);
    size_t pdd_b_off = (size_t)step * (size_t)DIT_PATCH_DIM * sizeof(float);
    if (!minimax_h3_hooks_contended(hooks)) {
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      gpu_select_kernel(&cmd, dit->k_adaln);
      gpu_bind_buffer(&cmd, 0, &dit->final_adaln_w, 0);
      gpu_bind_buffer(&cmd, 1, &dit->final_adaln_b, 0);
      gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
      gpu_copy_bytes(&cmd, 4, &final_adaln_p, sizeof final_adaln_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_FINAL_ADALN_OUT + ELEM_THREADS - 1) /
                         ELEM_THREADS,
                     1, 1, ELEM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)target_patches, 1, 1, NORM_THREADS, 1, 1);

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
    } else {
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      gpu_select_kernel(&cmd, dit->k_adaln);
      gpu_bind_buffer(&cmd, 0, &dit->final_adaln_w, 0);
      gpu_bind_buffer(&cmd, 1, &dit->final_adaln_b, 0);
      gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
      gpu_copy_bytes(&cmd, 4, &final_adaln_p, sizeof final_adaln_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_FINAL_ADALN_OUT + ELEM_THREADS - 1) /
                         ELEM_THREADS,
                     1, 1, ELEM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)target_patches, 1, 1, NORM_THREADS, 1, 1);
      gpu_commit(&cmd);
      uint64_t norm_cmd = cmd.number;

      if (!run_f32_linear_sliced(dit, &dit->pdd_w, pdd_w_off, &ws.norm, 0,
                                 &dit->pdd_b, pdd_b_off, &ws.patch_out, 0,
                                 target_patches, DIT_PATCH_DIM, DIT_DIM, hooks,
                                 &pace.patch_out)) {
        minimax_h3_gpu_wait(dit->gpu, norm_cmd);
        stopped = 1;
        break;
      }
    }

    // Flow-matching Euler step: model predicts v = x_data - noise, so
    // x_next = x + (sigma - sigma_next) * v.
    float dt = (float)(sigma - sigma_next);
    unpatchify_euler_step((const float *)ws.patch_out.ptr, h_lat, w_lat, lat_h,
                          lat_w, dt, out_latents);
    if (hooks && hooks->image_steps_completed)
      (*hooks->image_steps_completed)++;
    if (estimate && step + 1 < dit->pdd_steps) {
      // The data that the step aims for is x_next + sigma_next * v.
      memcpy(estimate, out_latents, latent_floats * sizeof(float));
      unpatchify_euler_step((const float *)ws.patch_out.ptr, h_lat, w_lat,
                            lat_h, lat_w, (float)sigma_next, estimate);
      if (!hooks->on_estimate(hooks->estimate_context, estimate, step + 1)) {
        stopped = 1;
        break;
      }
    }
  }
  free(estimate);

  if (hooks && hooks->image_step) *hooks->image_step = 0;
  if (hooks && hooks->image_steps_total) *hooks->image_steps_total = 0;
  free_dit_workspace(&ws);
  memory_pin(wslot, (int)was_pinned);
  memory_touch(wslot);
  return !stopped;
}

// Returns the sum of temporal spans across n video latent tokens;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:110-111.
static double video_t_span_sum(int n) {
  double total = 0.0;
  for (int k = 0; k < n; k++)
    total += FRAME_RESCALE * (double)FRAME_PER_TOKEN[k % VIDEO_TOKEN_CYCLE];
  return total;
}

// Returns the time-axis span a REF2VA block occupies ahead of the target
// streams; Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:120-129.
static double ref_block_t_span(const MinimaxH3VideoRef *ref) {
  if (ref->kind == MINIMAX_H3_VIDEO_REF_IMAGE) return 1.0;
  if (ref->kind == MINIMAX_H3_VIDEO_REF_AUDIO) return (double)ref->audio_t;
  double vspan = video_t_span_sum(ref->t_lat);
  double aspan = (double)ref->audio_t;
  return aspan > vspan ? aspan : vspan;
}

// Computes the first and last width RoPE coordinates of a latent frame's 2x2
// patch grid (used to pin left and right stereo audio channels);
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:85-90,103-107.
static void frame_w_bounds(int lat_h, int lat_w, double *out_low,
                           double *out_high) {
  int nw = lat_w / DIT_PATCH_SIDE;
  double sqrt_area = sqrt((double)lat_h * (double)lat_w);
  double ratio_w = (double)lat_w / sqrt_area;
  *out_low = ((1.0 - ratio_w) * CELL_CENTER) * DIT_ROPE_SPATIAL_SCALE;
  *out_high = ((double)(nw - 1) * (ratio_w / (double)nw) +
               (1.0 - ratio_w) * CELL_CENTER) *
              DIT_ROPE_SPATIAL_SCALE;
}

// Fills 3D MM-RoPE coordinates for a multi-frame video segment [t_lat, nh*nw]
// starting at time origin; Comfy-Org/ComfyUI
// comfy/ldm/minimax/model.py:114-145.
static void fill_video_positions(float *dst_pos, int t_lat, int lat_h,
                                 int lat_w, double origin) {
  int frame_rows = (lat_h / DIT_PATCH_SIDE) * (lat_w / DIT_PATCH_SIDE);
  double cur = origin;
  for (int f = 0; f < t_lat; f++) {
    fill_frame_positions(dst_pos + (size_t)f * (size_t)frame_rows *
                                       (size_t)MINIMAX_H3_POS_AXES,
                         lat_h, lat_w, cur);
    cur += FRAME_RESCALE * (double)FRAME_PER_TOKEN[f % VIDEO_TOKEN_CYCLE];
  }
}

// Fills 3D MM-RoPE coordinates for channel-major stereo audio rows [2 * t]
// starting at cursor with left channel at w_low and right channel at w_high;
// Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:132-138.
static void fill_audio_positions(float *dst_pos, int audio_t, double cursor,
                                 double w_low, double w_high) {
  for (int ch = 0; ch < MINIMAX_H3_AUDIO_STEREO_CHANNELS; ch++) {
    float w_coord = (float)(ch == 0 ? w_low : w_high);
    for (int t = 0; t < audio_t; t++) {
      float *p = dst_pos + ((size_t)ch * (size_t)audio_t + (size_t)t) *
                               (size_t)MINIMAX_H3_POS_AXES;
      p[0] = (float)(cursor + (double)t);
      p[1] = 0.0f;
      p[2] = w_coord;
    }
  }
}

// Appends one segment entry to a PackedLayout segment array.
static void push_segment(MinimaxH3PackedLayout *out, int start, int count,
                         MinimaxH3PackedSegKind kind) {
  out->segments[out->seg_count++] =
      (MinimaxH3PackedSeg){.start = start, .count = count, .kind = kind};
}

void minimax_h3_packed_layout_free(MinimaxH3PackedLayout *layout) {
  if (!layout) return;
  free(layout->segments);
  free(layout->pos);
  free(layout->mod_rows);
  memset(layout, 0, sizeof *layout);
}

void minimax_h3_packed_layout_build(
    int text_len, const int *text_tags, int t_lat, int h_lat, int w_lat,
    int audio_t, const MinimaxH3VideoKeyframe *keyframes, int keyframe_count,
    const MinimaxH3VideoRef *refs, int ref_count, MinimaxH3PackedLayout *out) {
  if (text_len <= 0 || t_lat <= 0 || h_lat <= 0 || w_lat <= 0 || audio_t < 0 ||
      keyframe_count < 0 || (keyframe_count > 0 && !keyframes) ||
      ref_count < 0 || (ref_count > 0 && !refs) || !out)
    die("invalid packed layout arguments");
  memset(out, 0, sizeof *out);

  int lat_h = (h_lat + 1) / 2 * 2;
  int lat_w = (w_lat + 1) / 2 * 2;
  int target_frame_rows = (lat_h / DIT_PATCH_SIDE) * (lat_w / DIT_PATCH_SIDE);

  // Pass 1: count segments, visual rows, audio rows and conditioning kinds.
  int max_segs = 1 + keyframe_count * 2 + ref_count * 2 + 2;
  int total_vis = 0, total_audio = 0;
  int has_cond_vis = 0, has_cond_aud = 0;
  for (int i = 0; i < keyframe_count; i++) {
    if (keyframes[i].t_lat > 0) {
      total_vis += keyframes[i].t_lat * target_frame_rows;
      has_cond_vis = 1;
    }
    if (keyframes[i].audio_t > 0) {
      total_audio += keyframes[i].audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
      has_cond_aud = 1;
    }
  }
  for (int i = 0; i < ref_count; i++) {
    const MinimaxH3VideoRef *r = &refs[i];
    if (r->kind == MINIMAX_H3_VIDEO_REF_IMAGE) {
      int rh = (r->h_lat + 1) / 2 * 2, rw = (r->w_lat + 1) / 2 * 2;
      total_vis += (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
      has_cond_vis = 1;
    } else if (r->kind == MINIMAX_H3_VIDEO_REF_AUDIO) {
      if (r->audio_t > 0) {
        total_audio += r->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
        has_cond_aud = 1;
      }
    } else {
      if (r->audio_t > 0) {
        total_audio += r->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
        has_cond_aud = 1;
      }
      int rh = (r->h_lat + 1) / 2 * 2, rw = (r->w_lat + 1) / 2 * 2;
      total_vis += r->t_lat * (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
      has_cond_vis = 1;
    }
  }
  int target_audio_rows = audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
  int target_video_rows = t_lat * target_frame_rows;
  total_audio += target_audio_rows;
  total_vis += target_video_rows;

  int seq_len = text_len + total_vis + total_audio;
  out->segments = xmalloc((size_t)max_segs * sizeof *out->segments);
  out->pos =
      xmalloc((size_t)seq_len * (size_t)MINIMAX_H3_POS_AXES * sizeof(float));
  out->mod_rows = xmalloc((size_t)seq_len * sizeof(uint32_t));
  out->seq_len = seq_len;
  out->total_vis = total_vis;
  out->total_audio = total_audio;
  out->target_audio_rows = target_audio_rows;
  out->target_video_rows = target_video_rows;
  out->num_t = has_cond_aud   ? (int)DIT_VIDEO_MAX_T_ROWS
               : has_cond_vis ? (int)DIT_VID_T_ROW_COND_AUD
                              : (int)DIT_VID_T_ROW_COND_VIS;

  // Pass 2: assemble segments, 3D MM-RoPE positions and AdaLN mod rows in
  // exact ComfyUI PackedLayout order (comfy/ldm/minimax/model.py:394-512).
  int row = 0;
  push_segment(out, 0, text_len, MINIMAX_H3_SEG_TEXT);
  for (int i = 0; i < text_len; i++) {
    out->pos[MINIMAX_H3_POS_AXES * i] = (float)i;
    out->pos[MINIMAX_H3_POS_AXES * i + 1] = 0.0f;
    out->pos[MINIMAX_H3_POS_AXES * i + 2] = 0.0f;
    uint32_t tag = text_tags ? (uint32_t)text_tags[i] : TAG_TEXT;
    out->mod_rows[i] =
        DIT_VID_T_ROW_VIDEO * (uint32_t)DIT_ADALN_MODALITIES + tag;
  }
  row += text_len;

  double target_w_low = 0.0, target_w_high = 0.0;
  frame_w_bounds(lat_h, lat_w, &target_w_low, &target_w_high);

  double target_cursor = (double)text_len;
  for (int i = 0; i < ref_count; i++)
    target_cursor += ref_block_t_span(&refs[i]);

  uint32_t cond_vis_mod =
      DIT_VID_T_ROW_COND_VIS * (uint32_t)DIT_ADALN_MODALITIES + TAG_VISUAL;
  uint32_t cond_aud_mod =
      DIT_VID_T_ROW_COND_AUD * (uint32_t)DIT_ADALN_MODALITIES + TAG_AUDIO;
  uint32_t target_aud_mod =
      DIT_VID_T_ROW_AUDIO * (uint32_t)DIT_ADALN_MODALITIES + TAG_AUDIO;
  uint32_t target_vis_mod =
      DIT_VID_T_ROW_VIDEO * (uint32_t)DIT_ADALN_MODALITIES + TAG_VISUAL;

  // Keyframe anchors (FL2VA / guides) sit right after text, anchored onto
  // target_cursor + FRAME_RESCALE * frame_index.
  for (int i = 0; i < keyframe_count; i++) {
    double cond_t =
        target_cursor + FRAME_RESCALE * (double)keyframes[i].frame_index;
    if (keyframes[i].t_lat > 0) {
      int n = keyframes[i].t_lat * target_frame_rows;
      push_segment(out, row, n, MINIMAX_H3_SEG_COND);
      fill_video_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                           keyframes[i].t_lat, lat_h, lat_w, cond_t);
      for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_vis_mod;
      row += n;
    }
    if (keyframes[i].audio_t > 0) {
      int n = keyframes[i].audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
      push_segment(out, row, n, MINIMAX_H3_SEG_COND_AUDIO);
      fill_audio_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                           keyframes[i].audio_t, cond_t, target_w_low,
                           target_w_high);
      for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_aud_mod;
      row += n;
    }
  }

  // REF2VA blocks pack ahead of the target streams starting at cursor=text_len.
  double ref_cursor = (double)text_len;
  for (int i = 0; i < ref_count; i++) {
    const MinimaxH3VideoRef *r = &refs[i];
    if (r->kind == MINIMAX_H3_VIDEO_REF_IMAGE) {
      int rh = (r->h_lat + 1) / 2 * 2, rw = (r->w_lat + 1) / 2 * 2;
      int n = (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
      push_segment(out, row, n, MINIMAX_H3_SEG_REF_IMG);
      fill_frame_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                           rh, rw, ref_cursor);
      for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_vis_mod;
      row += n;
      ref_cursor += 1.0;
    } else if (r->kind == MINIMAX_H3_VIDEO_REF_AUDIO) {
      if (r->audio_t > 0) {
        int n = r->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
        push_segment(out, row, n, MINIMAX_H3_SEG_REF_AUDIO);
        fill_audio_positions(
            out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES, r->audio_t,
            ref_cursor, target_w_low, target_w_high);
        for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_aud_mod;
        row += n;
      }
      ref_cursor += (double)r->audio_t;
    } else {
      int rh = (r->h_lat + 1) / 2 * 2, rw = (r->w_lat + 1) / 2 * 2;
      double rw_low = 0.0, rw_high = 0.0;
      frame_w_bounds(rh, rw, &rw_low, &rw_high);
      if (r->audio_t > 0) {
        int n = r->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
        push_segment(out, row, n, MINIMAX_H3_SEG_REF_AUDIO);
        fill_audio_positions(out->pos +
                                 (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                             r->audio_t, ref_cursor, rw_low, rw_high);
        for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_aud_mod;
        row += n;
      }
      int n = r->t_lat * (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
      push_segment(out, row, n, MINIMAX_H3_SEG_REF_IMG);
      fill_video_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                           r->t_lat, rh, rw, ref_cursor);
      for (int j = 0; j < n; j++) out->mod_rows[row + j] = cond_vis_mod;
      row += n;
      ref_cursor += ref_block_t_span(r);
    }
  }

  // Target audio (second-to-last) and target video (always last).
  out->target_audio_start = row;
  if (target_audio_rows > 0) {
    push_segment(out, row, target_audio_rows, MINIMAX_H3_SEG_AUDIO);
    fill_audio_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                         audio_t, target_cursor, target_w_low, target_w_high);
    for (int j = 0; j < target_audio_rows; j++)
      out->mod_rows[row + j] = target_aud_mod;
    row += target_audio_rows;
  }

  out->target_video_start = row;
  push_segment(out, row, target_video_rows, MINIMAX_H3_SEG_VIDEO);
  fill_video_positions(out->pos + (size_t)row * (size_t)MINIMAX_H3_POS_AXES,
                       t_lat, lat_h, lat_w, target_cursor);
  for (int j = 0; j < target_video_rows; j++)
    out->mod_rows[row + j] = target_vis_mod;
  row += target_video_rows;

  if (row != seq_len) die("packed layout row count mismatch");
}

// Patchifies channel-first video latent [24, t_lat, orig_h, orig_w] (padded to
// even lat_h by lat_w) into [t_lat * lat_h/2 * lat_w/2, 96] rows.
static void patchify_video_latent(const float *z, int t_lat, int orig_h,
                                  int orig_w, int lat_h, int lat_w,
                                  float *out_rows) {
  int nh = lat_h / DIT_PATCH_SIDE, nw = lat_w / DIT_PATCH_SIDE;
  size_t plane = (size_t)orig_h * (size_t)orig_w;
  size_t frame_stride = (size_t)nh * (size_t)nw;
  for (int f = 0; f < t_lat; f++) {
    for (int py = 0; py < nh; py++) {
      for (int px = 0; px < nw; px++) {
        float *row = out_rows + ((size_t)f * frame_stride +
                                 (size_t)py * (size_t)nw + (size_t)px) *
                                    DIT_PATCH_DIM;
        for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
          const float *zc = z + ((size_t)c * (size_t)t_lat + (size_t)f) * plane;
          for (int dy = 0; dy < DIT_PATCH_SIDE; dy++) {
            int y = py * DIT_PATCH_SIDE + dy;
            for (int dx = 0; dx < DIT_PATCH_SIDE; dx++) {
              int x = px * DIT_PATCH_SIDE + dx;
              float v = (y < orig_h && x < orig_w)
                            ? zc[(size_t)y * (size_t)orig_w + (size_t)x]
                            : 0.0f;
              row[c * DIT_PATCH_AREA + dy * DIT_PATCH_SIDE + dx] = v;
            }
          }
        }
      }
    }
  }
}

// Unpatchifies [t_lat * lat_h/2 * lat_w/2, 96] velocity rows and applies the
// Euler update z += dt * out onto [24, t_lat, orig_h, orig_w].
static void unpatchify_video_euler_step(const float *rows, int t_lat,
                                        int orig_h, int orig_w, int lat_h,
                                        int lat_w, float dt, float *z) {
  int nh = lat_h / DIT_PATCH_SIDE, nw = lat_w / DIT_PATCH_SIDE;
  size_t plane = (size_t)orig_h * (size_t)orig_w;
  size_t frame_stride = (size_t)nh * (size_t)nw;
  for (int f = 0; f < t_lat; f++) {
    for (int py = 0; py < nh; py++) {
      for (int px = 0; px < nw; px++) {
        const float *row = rows + ((size_t)f * frame_stride +
                                   (size_t)py * (size_t)nw + (size_t)px) *
                                      DIT_PATCH_DIM;
        for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
          float *zc = z + ((size_t)c * (size_t)t_lat + (size_t)f) * plane;
          for (int dy = 0; dy < DIT_PATCH_SIDE; dy++) {
            int y = py * DIT_PATCH_SIDE + dy;
            if (y >= orig_h) continue;
            for (int dx = 0; dx < DIT_PATCH_SIDE; dx++) {
              int x = px * DIT_PATCH_SIDE + dx;
              if (x >= orig_w) continue;
              zc[(size_t)y * (size_t)orig_w + (size_t)x] +=
                  dt * row[c * DIT_PATCH_AREA + dy * DIT_PATCH_SIDE + dx];
            }
          }
        }
      }
    }
  }
}

// Packs channel-first stereo audio latent [32, 2, audio_t] into channel-major
// rows [2 * audio_t, 32]; Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:74-77.
static void pack_audio_latent(const float *z, int audio_t, float *out_rows) {
  for (int ch = 0; ch < MINIMAX_H3_AUDIO_STEREO_CHANNELS; ch++) {
    for (int t = 0; t < audio_t; t++) {
      float *row = out_rows + ((size_t)ch * (size_t)audio_t + (size_t)t) *
                                  (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS;
      for (int c = 0; c < MINIMAX_H3_AUDIO_LATENT_CHANNELS; c++)
        row[c] = z[((size_t)c * (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS +
                    (size_t)ch) *
                       (size_t)audio_t +
                   (size_t)t];
    }
  }
}

// Unpacks [2 * audio_t, 32] audio velocity rows and applies the Euler update
// z += dt * out onto [32, 2, audio_t]; comfy/ldm/minimax/model.py:80-82.
static void unpack_audio_euler_step(const float *rows, int audio_t, float dt,
                                    float *z) {
  for (int ch = 0; ch < MINIMAX_H3_AUDIO_STEREO_CHANNELS; ch++) {
    for (int t = 0; t < audio_t; t++) {
      const float *row = rows + ((size_t)ch * (size_t)audio_t + (size_t)t) *
                                    (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS;
      for (int c = 0; c < MINIMAX_H3_AUDIO_LATENT_CHANNELS; c++)
        z[((size_t)c * (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS + (size_t)ch) *
              (size_t)audio_t +
          (size_t)t] += dt * row[c];
    }
  }
}

// Applies VISUAL_COND_TIMESTEP noise augmentation in place on count floats,
// restarting the deterministic RNG at seed per condition block.
static void apply_visual_cond_noise(float *dst, size_t count, uint64_t seed) {
  float *noise = xmalloc(count * sizeof(float));
  minimax_h3_fill_gaussian(noise, count, seed);
  float aug = (float)DIT_VISUAL_COND_TIMESTEP;
  for (size_t k = 0; k < count; k++)
    dst[k] = aug * dst[k] + (1.0f - aug) * noise[k];
  free(noise);
}

int minimax_h3_dit_sample_video(MinimaxH3Dit *dit,
                                const MinimaxH3VideoSampleParams *params) {
  if (!dit || !dit->has_audio || !params || !params->text_hidden ||
      params->text_len <= 0 || params->t_lat <= 0 || params->h_lat <= 0 ||
      params->w_lat <= 0 || params->audio_t <= 0 || params->steps <= 0 ||
      !params->out_video_latents || !params->out_audio_latents)
    die("invalid video DiT sampling arguments");
  if (minimax_h3_should_stop(params->hooks)) return 0;
  if (!minimax_h3_dit_load(dit)) return 0;

  MinimaxH3Lora *lora_mut = (MinimaxH3Lora *)params->lora;
  if (lora_mut && !minimax_h3_lora_load(lora_mut)) return 0;

  MinimaxH3PackedLayout layout;
  minimax_h3_packed_layout_build(
      params->text_len, params->text_tags, params->t_lat, params->h_lat,
      params->w_lat, params->audio_t, params->keyframes, params->keyframe_count,
      params->refs, params->ref_count, &layout);

  MemorySlot *wslot = weight_group_slot(dit->weights);
  unsigned was_pinned = wslot->pins;
  memory_pin(wslot, 1);
  MemorySlot *lslot =
      lora_mut ? weight_group_slot(minimax_h3_lora_weights(lora_mut)) : NULL;
  unsigned l_was_pinned = lslot ? lslot->pins : 0;
  if (lslot) memory_pin(lslot, 1);

  DitWorkspace ws;
  if (!alloc_dit_workspace(dit, params->text_len, layout.total_vis,
                           layout.target_video_rows, layout.total_audio,
                           layout.target_audio_rows, layout.num_t, params->lora,
                           &ws)) {
    if (lslot) memory_pin(lslot, (int)l_was_pinned);
    memory_pin(wslot, (int)was_pinned);
    minimax_h3_packed_layout_free(&layout);
    return 0;
  }

  // Copy static RoPE positions and per-token AdaLN modulation rows to GPU.
  memcpy(ws.pos.ptr, layout.pos,
         (size_t)layout.seq_len * (size_t)MINIMAX_H3_POS_AXES * sizeof(float));
  memcpy(ws.mod_rows.ptr, layout.mod_rows,
         (size_t)layout.seq_len * sizeof(uint32_t));
  memset(ws.final_mod_rows.ptr, 0,
         (size_t)layout.target_video_rows * sizeof(uint32_t));
  uint32_t *fa_rows = (uint32_t *)ws.final_audio_mod_rows.ptr;
  for (int i = 0; i < layout.target_audio_rows; i++)
    fa_rows[i] = DIT_VID_T_ROW_AUDIO;

  // Pack static condition video and audio latents once in exact PackedLayout
  // order: keyframes first, then references.
  int lat_h = (params->h_lat + 1) / 2 * 2;
  int lat_w = (params->w_lat + 1) / 2 * 2;
  int target_frame_rows = (lat_h / DIT_PATCH_SIDE) * (lat_w / DIT_PATCH_SIDE);
  float *patch_in = (float *)ws.patch_in.ptr;
  float *audio_in = (float *)ws.audio_in.ptr;
  int cond_vis_off = 0, cond_aud_off = 0;

  for (int i = 0; i < params->keyframe_count; i++) {
    const MinimaxH3VideoKeyframe *kf = &params->keyframes[i];
    if (kf->t_lat > 0) {
      if (!kf->video_latents) die("missing video_latents in keyframe %d", i);
      int n = kf->t_lat * target_frame_rows;
      float *dst = patch_in + (size_t)cond_vis_off * DIT_PATCH_DIM;
      patchify_video_latent(kf->video_latents, kf->t_lat, params->h_lat,
                            params->w_lat, lat_h, lat_w, dst);
      apply_visual_cond_noise(dst, (size_t)n * DIT_PATCH_DIM, params->seed);
      cond_vis_off += n;
    }
    if (kf->audio_t > 0) {
      if (!kf->audio_latents) die("missing audio_latents in keyframe %d", i);
      pack_audio_latent(kf->audio_latents, kf->audio_t,
                        audio_in + (size_t)cond_aud_off *
                                       MINIMAX_H3_AUDIO_LATENT_CHANNELS);
      cond_aud_off += kf->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
    }
  }

  for (int i = 0; i < params->ref_count; i++) {
    const MinimaxH3VideoRef *r = &params->refs[i];
    if (r->kind != MINIMAX_H3_VIDEO_REF_IMAGE && r->audio_t > 0) {
      if (!r->audio_latents) die("missing audio_latents in ref %d", i);
      pack_audio_latent(r->audio_latents, r->audio_t,
                        audio_in + (size_t)cond_aud_off *
                                       MINIMAX_H3_AUDIO_LATENT_CHANNELS);
      cond_aud_off += r->audio_t * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
    }
    if (r->kind != MINIMAX_H3_VIDEO_REF_AUDIO) {
      if (!r->video_latents) die("missing video_latents in ref %d", i);
      int rh = (r->h_lat + 1) / 2 * 2, rw = (r->w_lat + 1) / 2 * 2;
      int t_ref = r->kind == MINIMAX_H3_VIDEO_REF_IMAGE ? 1 : r->t_lat;
      int n = t_ref * (rh / DIT_PATCH_SIDE) * (rw / DIT_PATCH_SIDE);
      float *dst = patch_in + (size_t)cond_vis_off * DIT_PATCH_DIM;
      patchify_video_latent(r->video_latents, t_ref, r->h_lat, r->w_lat, rh, rw,
                            dst);
      apply_visual_cond_noise(dst, (size_t)n * DIT_PATCH_DIM, params->seed);
      cond_vis_off += n;
    }
  }

  // Refine text embeddings once for all denoising steps.
  run_token_refiner(dit, params->text_hidden, params->text_len, &ws,
                    params->lora);

  // Initialize target video and audio latents with independent Gaussian noise.
  size_t video_floats = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                        (size_t)params->t_lat * (size_t)params->h_lat *
                        (size_t)params->w_lat;
  size_t audio_floats = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                        (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                        (size_t)params->audio_t;
  minimax_h3_fill_gaussian(params->out_video_latents, video_floats,
                           params->seed);
  minimax_h3_fill_gaussian(params->out_audio_latents, audio_floats,
                           params->seed + AUDIO_NOISE_SEED_OFFSET);

  double shift_v = params->shift_video > 0.0 ? params->shift_video
                                             : MINIMAX_H3_VIDEO_SIGMA_SHIFT;
  double shift_a = params->shift_audio > 0.0 ? params->shift_audio
                                             : MINIMAX_H3_AUDIO_SIGMA_SHIFT;
  size_t text_bytes =
      (size_t)params->text_len * (size_t)DIT_DIM * sizeof(float);
  size_t target_video_h_off =
      (size_t)layout.target_video_start * (size_t)DIT_DIM * sizeof(float);
  size_t target_audio_h_off =
      (size_t)layout.target_audio_start * (size_t)DIT_DIM * sizeof(float);
  size_t audio_norm_off =
      (size_t)layout.target_video_rows * (size_t)DIT_DIM * sizeof(float);

  DitAdalnParam final_adaln_p = {2u, (uint32_t)DIT_FINAL_ADALN_OUT,
                                 dit->final_adaln_bias_f16};
  DitNormModParam final_norm_p = {(uint32_t)DIT_DIM,
                                  (uint32_t)DIT_FINAL_ADALN_OUT, 0,
                                  (uint32_t)DIT_DIM, DIT_NORM_EPS};

  const MinimaxH3RunHooks *hooks = params->hooks;
  if (hooks && hooks->image_steps_total)
    *hooks->image_steps_total = params->steps;

  DitSlicePace pace = {0};
  int stopped = 0;
  for (int step = 0; step < params->steps && !stopped; step++) {
    if (minimax_h3_should_stop(hooks)) {
      stopped = 1;
      break;
    }
    if (hooks && hooks->image_step) *hooks->image_step = step + 1;

    double b0 = (double)(params->steps - step) / (double)params->steps;
    double b1 = (double)(params->steps - step - 1) / (double)params->steps;
    double sigma_v = minimax_h3_time_shift_sigma(b0, 1.0, shift_v);
    double sigma_v_next = minimax_h3_time_shift_sigma(b1, 1.0, shift_v);
    double sigma_a = minimax_h3_time_shift_sigma(b0, 1.0, shift_a);
    double sigma_a_next = minimax_h3_time_shift_sigma(b1, 1.0, shift_a);

    double t_v = 1.0 - sigma_v;
    double t_a = 1.0 - sigma_a;
    double t_cv =
        t_v > DIT_VISUAL_COND_TIMESTEP ? t_v : DIT_VISUAL_COND_TIMESTEP;
    double t_ca = t_a > DIT_AUDIO_COND_TIMESTEP ? t_a : DIT_AUDIO_COND_TIMESTEP;

    float *te = (float *)ws.t_emb.ptr;
    interp_t_emb(dit, t_v, te + DIT_VID_T_ROW_VIDEO * DIT_ADALN_RANK);
    interp_t_emb(dit, t_a, te + DIT_VID_T_ROW_AUDIO * DIT_ADALN_RANK);
    if (layout.num_t > (int)DIT_VID_T_ROW_COND_VIS)
      interp_t_emb(dit, t_cv, te + DIT_VID_T_ROW_COND_VIS * DIT_ADALN_RANK);
    if (layout.num_t > (int)DIT_VID_T_ROW_COND_AUD)
      interp_t_emb(dit, t_ca, te + DIT_VID_T_ROW_COND_AUD * DIT_ADALN_RANK);

    // Pack current noisy target video and audio latents and project all
    // segments into ws.h in one command buffer.
    patchify_video_latent(params->out_video_latents, params->t_lat,
                          params->h_lat, params->w_lat, lat_h, lat_w,
                          patch_in + (size_t)cond_vis_off * DIT_PATCH_DIM);
    pack_audio_latent(params->out_audio_latents, params->audio_t,
                      audio_in + (size_t)cond_aud_off *
                                     MINIMAX_H3_AUDIO_LATENT_CHANNELS);
    memcpy(ws.h.ptr, ws.text_refined.ptr, text_bytes);

    uint64_t prev_cmd = 0;
    if (!minimax_h3_hooks_contended(hooks)) {
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      int v_cursor = 0, a_cursor = 0;
      for (int s = 0; s < layout.seg_count; s++) {
        MinimaxH3PackedSeg seg = layout.segments[s];
        size_t h_off = (size_t)seg.start * (size_t)DIT_DIM * sizeof(float);
        if (seg.kind == MINIMAX_H3_SEG_TEXT) continue;
        if (seg.kind == MINIMAX_H3_SEG_COND ||
            seg.kind == MINIMAX_H3_SEG_REF_IMG ||
            seg.kind == MINIMAX_H3_SEG_VIDEO) {
          size_t p_off =
              (size_t)v_cursor * (size_t)DIT_PATCH_DIM * sizeof(float);
          F32MMParam ep = {(uint32_t)seg.count, (uint32_t)DIT_DIM,
                           (uint32_t)DIT_PATCH_DIM, (uint32_t)DIT_DIM};
          gpu_select_kernel(&cmd, dit->k_mm_f32);
          gpu_bind_buffer(&cmd, 0, &dit->patch_proj_w, 0);
          gpu_bind_buffer(&cmd, 1, &ws.patch_in, p_off);
          gpu_bind_buffer(&cmd, 2, &dit->patch_proj_b, 0);
          gpu_bind_buffer(&cmd, 3, &ws.h, h_off);
          gpu_copy_bytes(&cmd, 4, &ep, sizeof ep);
          gpu_run_kernel(&cmd, (uint32_t)DIT_DIM, (uint32_t)seg.count, 1,
                         F32MM_THREADS, 1, 1);
          v_cursor += seg.count;
        } else {
          size_t a_off = (size_t)a_cursor *
                         (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                         sizeof(float);
          F32MMParam ep = {(uint32_t)seg.count, (uint32_t)DIT_DIM,
                           (uint32_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS,
                           (uint32_t)DIT_DIM};
          gpu_select_kernel(&cmd, dit->k_mm_f32);
          gpu_bind_buffer(&cmd, 0, &dit->audio_patch_proj_w, 0);
          gpu_bind_buffer(&cmd, 1, &ws.audio_in, a_off);
          gpu_bind_buffer(&cmd, 2, &dit->audio_patch_proj_b, 0);
          gpu_bind_buffer(&cmd, 3, &ws.h, h_off);
          gpu_copy_bytes(&cmd, 4, &ep, sizeof ep);
          gpu_run_kernel(&cmd, (uint32_t)DIT_DIM, (uint32_t)seg.count, 1,
                         F32MM_THREADS, 1, 1);
          a_cursor += seg.count;
        }
      }
      gpu_commit(&cmd);
      prev_cmd = cmd.number;
    } else {
      int v_cursor = 0, a_cursor = 0;
      for (int s = 0; s < layout.seg_count && !stopped; s++) {
        MinimaxH3PackedSeg seg = layout.segments[s];
        size_t h_off = (size_t)seg.start * (size_t)DIT_DIM * sizeof(float);
        if (seg.kind == MINIMAX_H3_SEG_TEXT) continue;
        if (seg.kind == MINIMAX_H3_SEG_COND ||
            seg.kind == MINIMAX_H3_SEG_REF_IMG ||
            seg.kind == MINIMAX_H3_SEG_VIDEO) {
          size_t p_off =
              (size_t)v_cursor * (size_t)DIT_PATCH_DIM * sizeof(float);
          if (!run_f32_linear_sliced(dit, &dit->patch_proj_w, 0, &ws.patch_in,
                                     p_off, &dit->patch_proj_b, 0, &ws.h, h_off,
                                     seg.count, DIT_DIM, DIT_PATCH_DIM, hooks,
                                     &pace.patch_in))
            stopped = 1;
          v_cursor += seg.count;
        } else {
          size_t a_off = (size_t)a_cursor *
                         (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                         sizeof(float);
          if (!run_f32_linear_sliced(
                  dit, &dit->audio_patch_proj_w, 0, &ws.audio_in, a_off,
                  &dit->audio_patch_proj_b, 0, &ws.h, h_off, seg.count, DIT_DIM,
                  MINIMAX_H3_AUDIO_LATENT_CHANNELS, hooks, &pace.patch_in))
            stopped = 1;
          a_cursor += seg.count;
        }
      }
      if (stopped) break;
    }

    if (!run_dit_blocks(dit, &ws, layout.seq_len, (uint32_t)layout.num_t,
                        params->lora, hooks, &pace, &prev_cmd)) {
      stopped = 1;
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      break;
    }

    // FinalLayer on both target video (t_emb[0]) and target audio (t_emb[1]).
    if (!minimax_h3_hooks_contended(hooks)) {
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      gpu_select_kernel(&cmd, dit->k_adaln);
      gpu_bind_buffer(&cmd, 0, &dit->final_adaln_w, 0);
      gpu_bind_buffer(&cmd, 1, &dit->final_adaln_b, 0);
      gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
      gpu_copy_bytes(&cmd, 4, &final_adaln_p, sizeof final_adaln_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_FINAL_ADALN_OUT + ELEM_THREADS - 1) /
                         ELEM_THREADS,
                     2, 1, ELEM_THREADS, 1, 1);

      // Modulate and project target video velocity.
      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_video_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)layout.target_video_rows, 1, 1,
                     NORM_THREADS, 1, 1);

      F32MMParam fvp = {(uint32_t)layout.target_video_rows,
                        (uint32_t)DIT_PATCH_DIM, (uint32_t)DIT_DIM,
                        (uint32_t)DIT_PATCH_DIM};
      gpu_select_kernel(&cmd, dit->k_mm_f32);
      gpu_bind_buffer(&cmd, 0, &dit->pdd_w, 0);
      gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
      gpu_bind_buffer(&cmd, 2, &dit->pdd_b, 0);
      gpu_bind_buffer(&cmd, 3, &ws.patch_out, 0);
      gpu_copy_bytes(&cmd, 4, &fvp, sizeof fvp);
      gpu_run_kernel(&cmd, (uint32_t)DIT_PATCH_DIM,
                     (uint32_t)layout.target_video_rows, 1, F32MM_THREADS, 1,
                     1);

      // Modulate and project target audio velocity.
      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_audio_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_audio_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, audio_norm_off);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)layout.target_audio_rows, 1, 1,
                     NORM_THREADS, 1, 1);

      F32MMParam fap = {(uint32_t)layout.target_audio_rows,
                        (uint32_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS,
                        (uint32_t)DIT_DIM,
                        (uint32_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS};
      gpu_select_kernel(&cmd, dit->k_mm_f32);
      gpu_bind_buffer(&cmd, 0, &dit->audio_out_w, 0);
      gpu_bind_buffer(&cmd, 1, &ws.norm, audio_norm_off);
      gpu_bind_buffer(&cmd, 2, &dit->audio_out_b, 0);
      gpu_bind_buffer(&cmd, 3, &ws.audio_out, 0);
      gpu_copy_bytes(&cmd, 4, &fap, sizeof fap);
      gpu_run_kernel(&cmd, (uint32_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS,
                     (uint32_t)layout.target_audio_rows, 1, F32MM_THREADS, 1,
                     1);

      gpu_commit(&cmd);
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      minimax_h3_gpu_wait_hooks(dit->gpu, cmd.number, hooks);
    } else {
      minimax_h3_gpu_wait(dit->gpu, prev_cmd);
      GpuCommandBuffer cmd = gpu_begin(dit->gpu);
      gpu_select_kernel(&cmd, dit->k_adaln);
      gpu_bind_buffer(&cmd, 0, &dit->final_adaln_w, 0);
      gpu_bind_buffer(&cmd, 1, &dit->final_adaln_b, 0);
      gpu_bind_buffer(&cmd, 2, &ws.t_emb, 0);
      gpu_bind_buffer(&cmd, 3, &ws.mod, 0);
      gpu_copy_bytes(&cmd, 4, &final_adaln_p, sizeof final_adaln_p);
      gpu_run_kernel(&cmd,
                     ((uint32_t)DIT_FINAL_ADALN_OUT + ELEM_THREADS - 1) /
                         ELEM_THREADS,
                     2, 1, ELEM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_video_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, 0);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)layout.target_video_rows, 1, 1,
                     NORM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, dit->k_norm_mod);
      gpu_bind_buffer(&cmd, 0, &ws.h, target_audio_h_off);
      gpu_bind_buffer(&cmd, 1, &dit->final_norm, 0);
      gpu_bind_buffer(&cmd, 2, &ws.mod, 0);
      gpu_bind_buffer(&cmd, 3, &ws.final_audio_mod_rows, 0);
      gpu_bind_buffer(&cmd, 4, &ws.norm, audio_norm_off);
      gpu_copy_bytes(&cmd, 5, &final_norm_p, sizeof final_norm_p);
      gpu_run_kernel(&cmd, (uint32_t)layout.target_audio_rows, 1, 1,
                     NORM_THREADS, 1, 1);
      gpu_commit(&cmd);
      uint64_t norm_cmd = cmd.number;

      if (!run_f32_linear_sliced(dit, &dit->pdd_w, 0, &ws.norm, 0, &dit->pdd_b,
                                 0, &ws.patch_out, 0, layout.target_video_rows,
                                 DIT_PATCH_DIM, DIT_DIM, hooks,
                                 &pace.patch_out) ||
          !run_f32_linear_sliced(dit, &dit->audio_out_w, 0, &ws.norm,
                                 audio_norm_off, &dit->audio_out_b, 0,
                                 &ws.audio_out, 0, layout.target_audio_rows,
                                 MINIMAX_H3_AUDIO_LATENT_CHANNELS, DIT_DIM,
                                 hooks, &pace.patch_out)) {
        minimax_h3_gpu_wait(dit->gpu, norm_cmd);
        stopped = 1;
        break;
      }
    }

    // Dual-stream flow-matching Euler step on native video and audio latents.
    float dt_v = (float)(sigma_v - sigma_v_next);
    float dt_a = (float)(sigma_a - sigma_a_next);
    unpatchify_video_euler_step((const float *)ws.patch_out.ptr, params->t_lat,
                                params->h_lat, params->w_lat, lat_h, lat_w,
                                dt_v, params->out_video_latents);
    unpack_audio_euler_step((const float *)ws.audio_out.ptr, params->audio_t,
                            dt_a, params->out_audio_latents);
    if (hooks && hooks->image_steps_completed)
      (*hooks->image_steps_completed)++;
  }

  if (hooks && hooks->image_step) *hooks->image_step = 0;
  if (hooks && hooks->image_steps_total) *hooks->image_steps_total = 0;
  free_dit_workspace(&ws);
  if (lslot) {
    memory_pin(lslot, (int)l_was_pinned);
    memory_touch(lslot);
  }
  memory_pin(wslot, (int)was_pinned);
  memory_touch(wslot);
  minimax_h3_packed_layout_free(&layout);
  return !stopped;
}
