#include "model/minimax_h3/vision_encoder.h"
#include "base/error.h"
#include "engine/resample.h"
#include "model/minimax_h3/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Qwen3-VL-32B vision tower constants from MiniMaxAI/MiniMax-H3-Image
// text_encoder/config.json:40-59 (depth 27, hidden_size 1152, num_heads 16,
// intermediate_size 4304, patch_size 16, temporal_patch_size 2,
// spatial_merge_size 2, num_position_embeddings 2304).
static constexpr int VIS_LAYERS = 27, VIS_DIM = 1152, VIS_HEADS = 16,
                     VIS_PAD_HEAD = 128,
                     VIS_QKV_WIDTH = MINIMAX_H3_QKV_PARTS * VIS_DIM,
                     VIS_MLP_RAW = 4304, VIS_MLP_PAD = 4352,
                     VIS_PATCH_SIDE = 16,
                     VIS_PATCH_AREA = VIS_PATCH_SIDE * VIS_PATCH_SIDE,
                     VIS_TEMPORAL_PATCH = 2, VIS_MERGE_SIDE = 2,
                     VIS_MERGE_UNIT = VIS_MERGE_SIDE * VIS_MERGE_SIDE,
                     VIS_PATCH_DIM = MINIMAX_H3_RGB_CHANNELS *
                                     VIS_TEMPORAL_PATCH * VIS_PATCH_AREA,
                     VIS_MERGE_DIM = VIS_DIM * VIS_MERGE_UNIT,
                     VIS_POS_ENTRIES = 2304, VIS_ROPE_FREQS = 18;

// Reference image sizing and prompt truncation constants from Comfy-Org/ComfyUI
// comfy_extras/nodes_minimax_h3.py:32,41-42 (CANVAS_MULTIPLE 32,
// IMAGE_REF_SHORT_EDGE 1024, IMAGE_REF_VAE_MAX_SHORT_EDGE 2048) and
// comfy/text_encoders/minimax.py:43-44 (min_pixels 3136, max_pixels 12845056).
static constexpr int REF_GRID_MULTIPLE = 32, REF_VISION_SHORT_EDGE = 1024,
                     REF_VAE_MIN_SHORT_EDGE = 1024,
                     REF_VAE_MAX_SHORT_EDGE = 2048, REF_SMART_MIN_PIXELS = 3136,
                     REF_SMART_MAX_PIXELS = 12845056, PROMPT_MAX_TOKENS = 4096;

// Reference prompt template budgets for "Picture %d: " plus flanking
// <|vision_start|> and <|vision_end|>: "Picture 16: " is 12 ASCII bytes and at
// most 5 BPE tokens, so 32 bytes / 16 tokens per label and 4 guard tokens
// comfortably cover up to 16 reference images.
static constexpr size_t REF_LABEL_BUF_CAP = 32;
static constexpr int REF_LABEL_MAX_TOKENS = 16,
                     REF_TEMPLATE_TOKENS = REF_LABEL_MAX_TOKENS + 2,
                     REF_SEQ_GUARD_TOKENS = 4;

// DeepStack vision block indices from MiniMaxAI/MiniMax-H3-Image
// text_encoder/config.json:41-45 ("deepstack_visual_indexes": [8, 16, 24]).
static const int DEEPSTACK_LAYERS[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS] = {8, 16,
                                                                       24};

// Vision LayerNorm epsilon and 2D RoPE base frequency from Comfy-Org/ComfyUI
// comfy/text_encoders/qwen35.py:629,680 (theta = 10000.0, eps = 1e-6).
static constexpr float VIS_NORM_EPS = 1e-6f;
static constexpr double VIS_ROPE_THETA = 10000.0;

// Threadgroup and tile sizes matching minimax_h3_vision.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          NORM_THREADS = 256, HEADS_THREADS = 64,
                          ELEM_THREADS = 256, GRID2D_THREADS = 32;

// AdaLN modality tags: visual (0) and text (1).
static constexpr int TAG_VISUAL = 0, TAG_TEXT = 1;

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

// Parameters passed to minimax_h3_layernorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} LayerNormParam;

// Parameters passed to minimax_h3_vis_pos_add.
typedef struct {
  // Patch grid height and width (h_bar / 16 and w_bar / 16).
  uint32_t grid_h, grid_w;
} VisPosAddParam;

// Parameters passed to minimax_h3_vis_heads.
typedef struct {
  // Patch token count, padded head stride and patch grid height/width.
  uint32_t rows, padded, grid_h, grid_w;
} VisHeadsParam;

// Parameters passed to minimax_h3_vis_attend.
typedef struct {
  // Patch token count and padded head stride.
  uint32_t rows, padded;
} VisAttendParam;

// Parameters for zero-padding a BF16 weight matrix along rows or columns.
typedef struct {
  // Source BF16 tensor [rows_in, cols_in].
  Tensor tensor;
  // Source row and column counts.
  int rows_in, cols_in;
  // Padded row and column counts (multiples of 64).
  int rows_out, cols_out;
} PaddedBf16Param;

// Parameters for zero-padding a BF16 bias vector to FP32 of length len_out.
typedef struct {
  // Source BF16 bias tensor [len_in].
  Tensor tensor;
  // Source and padded lengths.
  int len_in, len_out;
} PaddedBiasParam;

// One BF16 linear projection with FP32 bias.
typedef struct {
  // Page-aligned BF16 weight matrix [n, k].
  Buf w;
  // FP32 bias vector [n].
  Buf bias;
  // Output width and inner width (multiples of 64).
  int n, k;
} VisLin;

// Weights for one BF16 Qwen3-VL vision transformer block.
typedef struct {
  // Pre-attention and pre-MLP LayerNorm weights and biases (FP32, 1152).
  Buf norm1_w, norm1_b, norm2_w, norm2_b;
  // Contiguous QKV projection (3456x1152) and attention output (1152x1152).
  VisLin qkv, proj;
  // Tanh-GELU MLP fc1 (4352x1152 padded) and fc2 (1152x4352 padded).
  VisLin fc1, fc2;
} VisBlock;

// Weights for one patch merger (main pre-norm or DeepStack post-norm).
typedef struct {
  // LayerNorm weight and bias (1152 for main merger, 4608 for DeepStack).
  Buf norm_w, norm_b;
  // First (4608x4608) and second (5120x4608) BF16 projections.
  VisLin fc1, fc2;
  // Non-zero when LayerNorm runs after 2x2 merge (DeepStack).
  int post_norm;
} VisMerger;

// Transient workspace buffers allocated for one image encode call.
typedef struct {
  // Patchified pixel rows [num_patches, 1536].
  Buf patch_in;
  // Vision hidden stream [num_patches, 1152] and normalized staging.
  Buf h, norm;
  // Padded BF16 activation staging [padded_patches, 4608].
  Buf xb;
  // Contiguous QKV [num_patches, 3456] and attention output [num_patches,
  // 1152].
  Buf qkv, attn_out;
  // Head-major padded BF16 Q, K, V [16, padded_patches, 128].
  Buf qh, kh, vh;
  // MLP intermediate [num_patches, 4352] and merger hidden [n_merged, 4608].
  Buf mlp_mid, merge_mid;
} VisWorkspace;

// MiniMax-H3 Qwen3-VL vision tower handle.
struct MinimaxH3VisionEncoder {
  // Borrowed GPU device and open safetensors index from MinimaxH3TextEncoder.
  Gpu *gpu;
  // Borrowed safetensors index owning the visual.* tensors.
  const TensorIndex *tensors;
  // Memory account bounding weights and transient workspace.
  MemoryAccount *account;
  // Non-evictable slot tracking transient vision encode buffers.
  MemorySlot workspace_slot;
  // On-demand full-residence weight group for all vision weights.
  WeightGroup *weights;
  // Patch embedding linear (1152x1536), 48x48 BF16 pos_embed and 2D RoPE freqs.
  VisLin patch_embed;
  // Learned 48x48 BF16 position embedding table [2304, 1152].
  Buf pos_embed;
  // 2D vision RoPE inverse frequencies (FP32, 18).
  Buf inv_freq;
  // 27 BF16 vision transformer blocks.
  VisBlock blocks[VIS_LAYERS];
  // Main pre-norm patch merger and 3 DeepStack post-norm patch mergers.
  VisMerger merger;
  // DeepStack mergers attached to vision blocks 8, 16 and 24.
  VisMerger deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS];
  // Compiled Metal kernel indices.
  int k_x_bf16, k_mm_bf16, k_layernorm, k_add, k_vis_pos_add, k_gelu_tanh,
      k_gelu_exact, k_vis_heads, k_vis_attend;
};

// Generates the 18 FP32 per-axis inverse frequencies 10000^(-k / 18).
static void fill_vis_inv_freq(Buf *buffer, const void *context) {
  (void)context;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < VIS_ROPE_FREQS; i++)
    out[i] =
        (float)(1.0 / pow(VIS_ROPE_THETA, (double)i / (double)VIS_ROPE_FREQS));
}

// Loads a BF16 weight tensor [rows_in, cols_in] into a zero-padded BF16 buffer
// [rows_out, cols_out].
static void fill_padded_bf16(Buf *buffer, const void *context) {
  const PaddedBf16Param *p = context;
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

// Loads a BF16 bias tensor [len_in] into a zero-padded FP32 buffer [len_out].
static void fill_padded_bias(Buf *buffer, const void *context) {
  const PaddedBiasParam *p = context;
  uint16_t *raw = xmalloc(p->tensor.size);
  minimax_h3_read_tensor_bytes(p->tensor, raw);
  float *dst = (float *)buffer->ptr;
  memset(dst, 0, (size_t)p->len_out * sizeof(float));
  for (int i = 0; i < p->len_in; i++) {
    uint32_t u = (uint32_t)raw[i] << 16;
    memcpy(&dst[i], &u, sizeof dst[i]);
  }
  free(raw);
}

// Registers a BF16 linear layer (with optional row/col zero-padding to
// multiples of 64) into group using page-aligned buffers.
static void add_vis_lin(WeightGroup *group, const TensorIndex *tensors,
                        VisLin *lin, int n_raw, int k_raw, int n_pad, int k_pad,
                        const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
  size_t expected_w = (size_t)n_raw * (size_t)k_raw * sizeof(uint16_t);
  if (w.dtype != 'b' || w.size != expected_w)
    die("unexpected vision BF16 weight size for %s", prefix);
  if (b.dtype != 'b' || b.ndim != 1 || b.shape[0] != n_raw)
    die("unexpected vision BF16 bias shape for %s", prefix);
  lin->n = n_pad;
  lin->k = k_pad;
  if (n_raw == n_pad && k_raw == k_pad) {
    weight_group_add_tensor(group, &lin->w, w, 0);
    weight_group_add_vector(group, &lin->bias, b, 0.0f, 0);
  } else {
    PaddedBf16Param wp = {w, n_raw, k_raw, n_pad, k_pad};
    weight_group_add_generated(group, &lin->w,
                               (size_t)n_pad * (size_t)k_pad * sizeof(uint16_t),
                               fill_padded_bf16, &wp, sizeof wp);
    if (n_raw == n_pad) {
      weight_group_add_vector(group, &lin->bias, b, 0.0f, 0);
    } else {
      PaddedBiasParam bp = {b, n_raw, n_pad};
      weight_group_add_generated(group, &lin->bias,
                                 (size_t)n_pad * sizeof(float),
                                 fill_padded_bias, &bp, sizeof bp);
    }
  }
}

// Registers one patch merger (main pre-norm or DeepStack post-norm).
static void add_vis_merger(WeightGroup *group, const TensorIndex *tensors,
                           VisMerger *m, int post_norm, const char *prefix) {
  m->post_norm = post_norm;
  int norm_dim = post_norm ? VIS_MERGE_DIM : VIS_DIM;
  Tensor nw = tensor_index_get(tensors, "%s.norm.weight", prefix);
  Tensor nb = tensor_index_get(tensors, "%s.norm.bias", prefix);
  if (nw.shape[0] != norm_dim || nb.shape[0] != norm_dim)
    die("unexpected vision merger norm shape for %s", prefix);
  weight_group_add_vector(group, &m->norm_w, nw, 0.0f, 0);
  weight_group_add_vector(group, &m->norm_b, nb, 0.0f, 0);
  char name[MINIMAX_H3_NAME_CAP];
  snprintf(name, sizeof name, "%s.linear_fc1", prefix);
  add_vis_lin(group, tensors, &m->fc1, VIS_MERGE_DIM, VIS_MERGE_DIM,
              VIS_MERGE_DIM, VIS_MERGE_DIM, name);
  snprintf(name, sizeof name, "%s.linear_fc2", prefix);
  add_vis_lin(group, tensors, &m->fc2, MINIMAX_H3_TEXT_DIM, VIS_MERGE_DIM,
              MINIMAX_H3_TEXT_DIM, VIS_MERGE_DIM, name);
}

// Scales (width, height) keeping aspect ratio so its short edge falls within
// [low, high], rounding each axis to REF_GRID_MULTIPLE (32).
static void size_short_edge(int width, int height, int low, int high,
                            int *out_w, int *out_h) {
  int short_side = width < height ? width : height;
  int target = short_side < low ? low : short_side > high ? high : short_side;
  double scale = (double)target / (double)short_side;
  int rw = (int)llround((double)width * scale / (double)REF_GRID_MULTIPLE) *
           REF_GRID_MULTIPLE;
  int rh = (int)llround((double)height * scale / (double)REF_GRID_MULTIPLE) *
           REF_GRID_MULTIPLE;
  if (rw < REF_GRID_MULTIPLE) rw = REF_GRID_MULTIPLE;
  if (rh < REF_GRID_MULTIPLE) rh = REF_GRID_MULTIPLE;
  *out_w = rw;
  *out_h = rh;
}

// Qwen-VL smart_resize: rounds to factor (32) and bounds area within
// [min_pixels, max_pixels].
static void smart_resize_32(int width, int height, int *out_w, int *out_h) {
  int factor = REF_GRID_MULTIPLE;
  int h_bar = (int)llround((double)height / (double)factor) * factor;
  int w_bar = (int)llround((double)width / (double)factor) * factor;
  if (h_bar < factor) h_bar = factor;
  if (w_bar < factor) w_bar = factor;
  int64_t area = (int64_t)h_bar * (int64_t)w_bar;
  if (area > REF_SMART_MAX_PIXELS) {
    double beta =
        sqrt((double)height * (double)width / (double)REF_SMART_MAX_PIXELS);
    h_bar = (int)floor((double)height / beta / (double)factor) * factor;
    w_bar = (int)floor((double)width / beta / (double)factor) * factor;
    if (h_bar < factor) h_bar = factor;
    if (w_bar < factor) w_bar = factor;
  } else if (area < REF_SMART_MIN_PIXELS) {
    double beta =
        sqrt((double)REF_SMART_MIN_PIXELS / ((double)height * (double)width));
    h_bar = (int)ceil((double)height * beta / (double)factor) * factor;
    w_bar = (int)ceil((double)width * beta / (double)factor) * factor;
  }
  *out_w = w_bar;
  *out_h = h_bar;
}

void minimax_h3_vision_ref_size(int width, int height, int *out_w, int *out_h) {
  int rw = 0, rh = 0;
  size_short_edge(width, height, REF_VISION_SHORT_EDGE, REF_VISION_SHORT_EDGE,
                  &rw, &rh);
  smart_resize_32(rw, rh, out_w, out_h);
}

void minimax_h3_vae_ref_size(int width, int height, int *out_w, int *out_h) {
  size_short_edge(width, height, REF_VAE_MIN_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE,
                  out_w, out_h);
}

MinimaxH3VisionEncoder *
minimax_h3_vision_encoder_open(MinimaxH3TextEncoder *text_enc,
                               MemoryAccount *account) {
  MinimaxH3VisionEncoder *vis = xmalloc(sizeof *vis);
  vis->gpu = minimax_h3_text_encoder_gpu(text_enc);
  vis->tensors = minimax_h3_text_encoder_tensors(text_enc);
  vis->account = account;
  memory_slot_init(&vis->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-vision-workspace", NULL, NULL);
  vis->weights = weight_group_new(vis->gpu, account, "minimax-h3-qwen-vision");

  vis->k_x_bf16 = gpu_find_kernel(vis->gpu, "minimax_h3_x_bf16");
  vis->k_mm_bf16 = gpu_find_kernel(vis->gpu, "minimax_h3_mm_bf16");
  vis->k_layernorm = gpu_find_kernel(vis->gpu, "minimax_h3_layernorm");
  vis->k_add = gpu_find_kernel(vis->gpu, "minimax_h3_add");
  vis->k_vis_pos_add = gpu_find_kernel(vis->gpu, "minimax_h3_vis_pos_add");
  vis->k_gelu_tanh = gpu_find_kernel(vis->gpu, "minimax_h3_gelu_tanh");
  vis->k_gelu_exact = gpu_find_kernel(vis->gpu, "minimax_h3_gelu_exact");
  vis->k_vis_heads = gpu_find_kernel(vis->gpu, "minimax_h3_vis_heads");
  vis->k_vis_attend = gpu_find_kernel(vis->gpu, "minimax_h3_vis_attend");

  add_vis_lin(vis->weights, vis->tensors, &vis->patch_embed, VIS_DIM,
              VIS_PATCH_DIM, VIS_DIM, VIS_PATCH_DIM, "visual.patch_embed.proj");
  Tensor pe = tensor_index_get(vis->tensors, "visual.pos_embed.weight");
  if (pe.dtype != 'b' || pe.ndim != 2 || pe.shape[0] != VIS_POS_ENTRIES ||
      pe.shape[1] != VIS_DIM)
    die("unexpected visual.pos_embed.weight shape");
  weight_group_add_tensor(vis->weights, &vis->pos_embed, pe, 0);
  weight_group_add_generated(vis->weights, &vis->inv_freq,
                             (size_t)VIS_ROPE_FREQS * sizeof(float),
                             fill_vis_inv_freq, NULL, 0);

  for (int i = 0; i < VIS_LAYERS; i++) {
    VisBlock *b = &vis->blocks[i];
    weight_group_add_vector(
        vis->weights, &b->norm1_w,
        tensor_index_get(vis->tensors, "visual.blocks.%d.norm1.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vis->weights, &b->norm1_b,
        tensor_index_get(vis->tensors, "visual.blocks.%d.norm1.bias", i), 0.0f,
        0);
    weight_group_add_vector(
        vis->weights, &b->norm2_w,
        tensor_index_get(vis->tensors, "visual.blocks.%d.norm2.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vis->weights, &b->norm2_b,
        tensor_index_get(vis->tensors, "visual.blocks.%d.norm2.bias", i), 0.0f,
        0);
    char name[MINIMAX_H3_NAME_CAP];
    snprintf(name, sizeof name, "visual.blocks.%d.attn.qkv", i);
    add_vis_lin(vis->weights, vis->tensors, &b->qkv, VIS_QKV_WIDTH, VIS_DIM,
                VIS_QKV_WIDTH, VIS_DIM, name);
    snprintf(name, sizeof name, "visual.blocks.%d.attn.proj", i);
    add_vis_lin(vis->weights, vis->tensors, &b->proj, VIS_DIM, VIS_DIM, VIS_DIM,
                VIS_DIM, name);
    snprintf(name, sizeof name, "visual.blocks.%d.mlp.linear_fc1", i);
    add_vis_lin(vis->weights, vis->tensors, &b->fc1, VIS_MLP_RAW, VIS_DIM,
                VIS_MLP_PAD, VIS_DIM, name);
    snprintf(name, sizeof name, "visual.blocks.%d.mlp.linear_fc2", i);
    add_vis_lin(vis->weights, vis->tensors, &b->fc2, VIS_DIM, VIS_MLP_RAW,
                VIS_DIM, VIS_MLP_PAD, name);
  }

  add_vis_merger(vis->weights, vis->tensors, &vis->merger, 0, "visual.merger");
  for (int i = 0; i < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; i++) {
    char name[MINIMAX_H3_NAME_CAP];
    snprintf(name, sizeof name, "visual.deepstack_merger_list.%d", i);
    add_vis_merger(vis->weights, vis->tensors, &vis->deepstack[i], 1, name);
  }
  return vis;
}

void minimax_h3_vision_encoder_close(MinimaxH3VisionEncoder *vis) {
  if (!vis) return;
  weight_group_free(vis->weights);
  memory_slot_remove(&vis->workspace_slot);
  free(vis);
}

WeightGroup *minimax_h3_vision_encoder_weights(MinimaxH3VisionEncoder *vis) {
  return vis->weights;
}

uint64_t
minimax_h3_vision_encoder_required_bytes(const MinimaxH3VisionEncoder *vis) {
  return weight_group_required_bytes(vis->weights);
}

uint64_t
minimax_h3_vision_encoder_resident_bytes(const MinimaxH3VisionEncoder *vis) {
  return weight_group_resident_bytes(vis->weights);
}

int minimax_h3_vision_encoder_load(MinimaxH3VisionEncoder *vis) {
  return weight_group_load(vis->weights);
}

// Frees all transient vision encode buffers.
static void free_vis_workspace(VisWorkspace *ws) {
  gpu_free(&ws->merge_mid);
  gpu_free(&ws->mlp_mid);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->attn_out);
  gpu_free(&ws->qkv);
  gpu_free(&ws->xb);
  gpu_free(&ws->norm);
  gpu_free(&ws->h);
  gpu_free(&ws->patch_in);
}

// Allocates transient GPU buffers for num_patches (multiple of 4) tokens.
static int alloc_vis_workspace(MinimaxH3VisionEncoder *vis, int num_patches,
                               VisWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  int n_merged = num_patches / VIS_MERGE_UNIT;
  uint32_t padded = ((uint32_t)num_patches + TILE_M - 1) / TILE_M * TILE_M;
  MemorySlot *slot = &vis->workspace_slot;

  ws->patch_in = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)num_patches * VIS_PATCH_DIM * sizeof(float));
  ws->h = gpu_try_alloc_required(vis->gpu, slot,
                                 (size_t)num_patches * VIS_DIM * sizeof(float));
  ws->norm = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)num_patches * VIS_DIM * sizeof(float));
  ws->xb = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)padded * VIS_MERGE_DIM * sizeof(uint16_t));
  ws->qkv = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)num_patches * VIS_QKV_WIDTH * sizeof(float));
  ws->attn_out = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)num_patches * VIS_DIM * sizeof(float));
  ws->qh = gpu_try_alloc_required(vis->gpu, slot,
                                  (size_t)VIS_HEADS * padded * VIS_PAD_HEAD *
                                      sizeof(uint16_t));
  ws->kh = gpu_try_alloc_required(vis->gpu, slot,
                                  (size_t)VIS_HEADS * padded * VIS_PAD_HEAD *
                                      sizeof(uint16_t));
  ws->vh = gpu_try_alloc_required(vis->gpu, slot,
                                  (size_t)VIS_HEADS * padded * VIS_PAD_HEAD *
                                      sizeof(uint16_t));
  ws->mlp_mid = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)num_patches * VIS_MLP_PAD * sizeof(float));
  ws->merge_mid = gpu_try_alloc_required(
      vis->gpu, slot, (size_t)n_merged * VIS_MERGE_DIM * sizeof(float));

  if (!ws->patch_in.obj || !ws->h.obj || !ws->norm.obj || !ws->xb.obj ||
      !ws->qkv.obj || !ws->attn_out.obj || !ws->qh.obj || !ws->kh.obj ||
      !ws->vh.obj || !ws->mlp_mid.obj || !ws->merge_mid.obj) {
    free_vis_workspace(ws);
    return 0;
  }
  return 1;
}

// Encodes FP32 x -> padded BF16 xb -> native BF16 matmul + FP32 bias into y.
static void run_vis_linear(const MinimaxH3VisionEncoder *vis,
                           GpuCommandBuffer *cmd, const Buf *x, const Buf *xb,
                           const VisLin *lin, const Buf *y, int rows) {
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  Bf16XParam xp = {(uint32_t)rows, (uint32_t)lin->k, (uint32_t)lin->k, 0};
  gpu_select_kernel(cmd, vis->k_x_bf16);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &vis->inv_freq, 0);
  gpu_bind_buffer(cmd, 2, xb, 0);
  gpu_copy_bytes(cmd, 3, &xp, sizeof xp);
  gpu_run_kernel(cmd,
                 ((uint32_t)lin->k / 4u + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 padded, 1, GRID2D_THREADS, 1, 1);

  Bf16MMParam mp = {(uint32_t)rows, (uint32_t)lin->n, (uint32_t)lin->k,
                    (uint32_t)lin->n, 1};
  gpu_select_kernel(cmd, vis->k_mm_bf16);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, xb, 0);
  gpu_bind_buffer(cmd, 2, &lin->bias, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N, padded / TILE_M,
                 1, MM_THREADS, 1, 1);
}

// Encodes one patch merger (main pre-norm or DeepStack post-norm) from ws->h
// into out_buf [n_merged, 5120].
static void run_vis_merger(const MinimaxH3VisionEncoder *vis,
                           GpuCommandBuffer *cmd, const VisWorkspace *ws,
                           const VisMerger *m, int num_patches,
                           const Buf *out_buf) {
  int n_merged = num_patches / VIS_MERGE_UNIT;
  if (m->post_norm) {
    LayerNormParam lnp = {(uint32_t)VIS_MERGE_DIM, VIS_NORM_EPS};
    gpu_select_kernel(cmd, vis->k_layernorm);
    gpu_bind_buffer(cmd, 0, &ws->h, 0);
    gpu_bind_buffer(cmd, 1, &m->norm_w, 0);
    gpu_bind_buffer(cmd, 2, &m->norm_b, 0);
    gpu_bind_buffer(cmd, 3, &ws->norm, 0);
    gpu_copy_bytes(cmd, 4, &lnp, sizeof lnp);
    gpu_run_kernel(cmd, (uint32_t)n_merged, 1, 1, NORM_THREADS, 1, 1);
  } else {
    LayerNormParam lnp = {(uint32_t)VIS_DIM, VIS_NORM_EPS};
    gpu_select_kernel(cmd, vis->k_layernorm);
    gpu_bind_buffer(cmd, 0, &ws->h, 0);
    gpu_bind_buffer(cmd, 1, &m->norm_w, 0);
    gpu_bind_buffer(cmd, 2, &m->norm_b, 0);
    gpu_bind_buffer(cmd, 3, &ws->norm, 0);
    gpu_copy_bytes(cmd, 4, &lnp, sizeof lnp);
    gpu_run_kernel(cmd, (uint32_t)num_patches, 1, 1, NORM_THREADS, 1, 1);
  }
  run_vis_linear(vis, cmd, &ws->norm, &ws->xb, &m->fc1, &ws->merge_mid,
                 n_merged);
  uint32_t gelu_n = (uint32_t)n_merged * (uint32_t)VIS_MERGE_DIM;
  gpu_select_kernel(cmd, vis->k_gelu_exact);
  gpu_bind_buffer(cmd, 0, &ws->merge_mid, 0);
  gpu_copy_bytes(cmd, 1, &gelu_n, sizeof gelu_n);
  gpu_run_kernel(cmd, (gelu_n + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
  run_vis_linear(vis, cmd, &ws->merge_mid, &ws->xb, &m->fc2, out_buf, n_merged);
}

int minimax_h3_vision_encoder_encode_image(
    MinimaxH3VisionEncoder *vis, const uint8_t *rgb, int width, int height,
    MemorySlot *out_slot, Buf *out_merged,
    Buf out_deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS], int *out_grid_h,
    int *out_grid_w, const MinimaxH3RunHooks *hooks) {
  if (!vis || !rgb || width <= 0 || height <= 0 ||
      width % REF_GRID_MULTIPLE != 0 || height % REF_GRID_MULTIPLE != 0 ||
      !out_merged || !out_deepstack || !out_grid_h || !out_grid_w)
    die("invalid vision encoder arguments");
  if (minimax_h3_should_stop(hooks)) return 0;
  if (!minimax_h3_vision_encoder_load(vis)) return 0;

  int grid_h = height / VIS_PATCH_SIDE, grid_w = width / VIS_PATCH_SIDE;
  int mh = grid_h / VIS_MERGE_SIDE, mw = grid_w / VIS_MERGE_SIDE;
  int num_patches = grid_h * grid_w;
  int n_merged = mh * mw;
  uint32_t padded = ((uint32_t)num_patches + TILE_M - 1) / TILE_M * TILE_M;

  MemorySlot *wslot = weight_group_slot(vis->weights);
  unsigned was_pinned = wslot->pins;
  memory_pin(wslot, 1);
  VisWorkspace ws;
  if (!alloc_vis_workspace(vis, num_patches, &ws)) {
    memory_pin(wslot, (int)was_pinned);
    return 0;
  }

  size_t out_bytes =
      (size_t)n_merged * (size_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  *out_merged = gpu_try_alloc_required(vis->gpu, out_slot, out_bytes);
  for (int i = 0; i < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; i++)
    out_deepstack[i] = gpu_try_alloc_required(vis->gpu, out_slot, out_bytes);
  if (!out_merged->obj || !out_deepstack[0].obj || !out_deepstack[1].obj ||
      !out_deepstack[2].obj) {
    gpu_free(out_merged);
    for (int i = 0; i < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; i++)
      gpu_free(&out_deepstack[i]);
    free_vis_workspace(&ws);
    memory_pin(wslot, (int)was_pinned);
    return 0;
  }

  // Patchify single RGB image into 2x2 merge-ordered rows [num_patches, 1536]
  // with mean=0.5, std=0.5 normalization (processor/preprocessor_config.json:
  // 9-18) and temporal repeat t=0,1.
  float *pin = (float *)ws.patch_in.ptr;
  for (int by = 0; by < mh; by++) {
    for (int bx = 0; bx < mw; bx++) {
      for (int my = 0; my < VIS_MERGE_SIDE; my++) {
        for (int mx = 0; mx < VIS_MERGE_SIDE; mx++) {
          int tok =
              ((by * mw + bx) * VIS_MERGE_SIDE + my) * VIS_MERGE_SIDE + mx;
          float *row = pin + (size_t)tok * (size_t)VIS_PATCH_DIM;
          int y0 = (by * VIS_MERGE_SIDE + my) * VIS_PATCH_SIDE;
          int x0 = (bx * VIS_MERGE_SIDE + mx) * VIS_PATCH_SIDE;
          for (int py = 0; py < VIS_PATCH_SIDE; py++) {
            const uint8_t *srow =
                rgb + ((size_t)(y0 + py) * (size_t)width + (size_t)x0) *
                          (size_t)MINIMAX_H3_RGB_CHANNELS;
            for (int px = 0; px < VIS_PATCH_SIDE; px++) {
              int sp = py * VIS_PATCH_SIDE + px;
              for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
                float v = (float)srow[px * MINIMAX_H3_RGB_CHANNELS + c] *
                              (2.0f / MINIMAX_H3_PIXEL_MAX) -
                          1.0f;
                for (int t = 0; t < VIS_TEMPORAL_PATCH; t++)
                  row[(c * VIS_TEMPORAL_PATCH + t) * VIS_PATCH_AREA + sp] = v;
              }
            }
          }
        }
      }
    }
  }

  // Patch embed + bilinear 48x48 position embedding.
  GpuCommandBuffer cmd = gpu_begin(vis->gpu);
  run_vis_linear(vis, &cmd, &ws.patch_in, &ws.xb, &vis->patch_embed, &ws.h,
                 num_patches);
  VisPosAddParam pos_p = {(uint32_t)grid_h, (uint32_t)grid_w};
  gpu_select_kernel(&cmd, vis->k_vis_pos_add);
  gpu_bind_buffer(&cmd, 0, &ws.h, 0);
  gpu_bind_buffer(&cmd, 1, &vis->pos_embed, 0);
  gpu_copy_bytes(&cmd, 2, &pos_p, sizeof pos_p);
  gpu_run_kernel(&cmd,
                 ((uint32_t)VIS_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 (uint32_t)num_patches, 1, GRID2D_THREADS, 1, 1);
  gpu_commit(&cmd);
  uint64_t prev_cmd = cmd.number;

  LayerNormParam lnp = {(uint32_t)VIS_DIM, VIS_NORM_EPS};
  VisHeadsParam hp = {(uint32_t)num_patches, padded, (uint32_t)grid_h,
                      (uint32_t)grid_w};
  VisAttendParam ap = {(uint32_t)num_patches, padded};
  uint32_t h_count = (uint32_t)num_patches * (uint32_t)VIS_DIM;
  uint32_t mlp_count = (uint32_t)num_patches * (uint32_t)VIS_MLP_PAD;

  int stopped = 0;
  for (int i = 0; i < VIS_LAYERS; i++) {
    if (minimax_h3_should_stop(hooks)) {
      stopped = 1;
      break;
    }
    const VisBlock *b = &vis->blocks[i];
    cmd = gpu_begin(vis->gpu);

    // Attention branch: norm1 -> qkv -> 2D RoPE + head pad -> FlashAttention
    // -> proj -> residual add.
    gpu_select_kernel(&cmd, vis->k_layernorm);
    gpu_bind_buffer(&cmd, 0, &ws.h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm1_w, 0);
    gpu_bind_buffer(&cmd, 2, &b->norm1_b, 0);
    gpu_bind_buffer(&cmd, 3, &ws.norm, 0);
    gpu_copy_bytes(&cmd, 4, &lnp, sizeof lnp);
    gpu_run_kernel(&cmd, (uint32_t)num_patches, 1, 1, NORM_THREADS, 1, 1);

    run_vis_linear(vis, &cmd, &ws.norm, &ws.xb, &b->qkv, &ws.qkv, num_patches);

    gpu_select_kernel(&cmd, vis->k_vis_heads);
    gpu_bind_buffer(&cmd, 0, &ws.qkv, 0);
    gpu_bind_buffer(&cmd, 1, &vis->inv_freq, 0);
    gpu_bind_buffer(&cmd, 2, &ws.qh, 0);
    gpu_bind_buffer(&cmd, 3, &ws.kh, 0);
    gpu_bind_buffer(&cmd, 4, &ws.vh, 0);
    gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
    gpu_run_kernel(&cmd, (uint32_t)(MINIMAX_H3_QKV_PARTS * VIS_HEADS), padded,
                   1, HEADS_THREADS, 1, 1);

    gpu_select_kernel(&cmd, vis->k_vis_attend);
    gpu_bind_buffer(&cmd, 0, &ws.qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws.kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws.vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws.attn_out, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, padded / TILE_M, (uint32_t)VIS_HEADS, 1, MM_THREADS, 1,
                   1);

    run_vis_linear(vis, &cmd, &ws.attn_out, &ws.xb, &b->proj, &ws.norm,
                   num_patches);

    gpu_select_kernel(&cmd, vis->k_add);
    gpu_bind_buffer(&cmd, 0, &ws.h, 0);
    gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
    gpu_copy_bytes(&cmd, 2, &h_count, sizeof h_count);
    gpu_run_kernel(&cmd, (h_count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);

    // MLP branch: norm2 -> fc1 -> gelu_tanh -> fc2 -> residual add.
    gpu_select_kernel(&cmd, vis->k_layernorm);
    gpu_bind_buffer(&cmd, 0, &ws.h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm2_w, 0);
    gpu_bind_buffer(&cmd, 2, &b->norm2_b, 0);
    gpu_bind_buffer(&cmd, 3, &ws.norm, 0);
    gpu_copy_bytes(&cmd, 4, &lnp, sizeof lnp);
    gpu_run_kernel(&cmd, (uint32_t)num_patches, 1, 1, NORM_THREADS, 1, 1);

    run_vis_linear(vis, &cmd, &ws.norm, &ws.xb, &b->fc1, &ws.mlp_mid,
                   num_patches);

    gpu_select_kernel(&cmd, vis->k_gelu_tanh);
    gpu_bind_buffer(&cmd, 0, &ws.mlp_mid, 0);
    gpu_copy_bytes(&cmd, 1, &mlp_count, sizeof mlp_count);
    gpu_run_kernel(&cmd, (mlp_count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);

    run_vis_linear(vis, &cmd, &ws.mlp_mid, &ws.xb, &b->fc2, &ws.norm,
                   num_patches);

    gpu_select_kernel(&cmd, vis->k_add);
    gpu_bind_buffer(&cmd, 0, &ws.h, 0);
    gpu_bind_buffer(&cmd, 1, &ws.norm, 0);
    gpu_copy_bytes(&cmd, 2, &h_count, sizeof h_count);
    gpu_run_kernel(&cmd, (h_count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                   ELEM_THREADS, 1, 1);

    // Extract DeepStack features after vision blocks 8, 16 and 24.
    for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++) {
      if (i == DEEPSTACK_LAYERS[d])
        run_vis_merger(vis, &cmd, &ws, &vis->deepstack[d], num_patches,
                       &out_deepstack[d]);
    }

    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vis->gpu, prev_cmd, hooks);
    prev_cmd = cmd.number;
  }

  if (stopped) {
    minimax_h3_gpu_wait(vis->gpu, prev_cmd);
    gpu_free(out_merged);
    for (int i = 0; i < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; i++)
      gpu_free(&out_deepstack[i]);
    free_vis_workspace(&ws);
    memory_pin(wslot, (int)was_pinned);
    return 0;
  }

  // Final pre-norm patch merger.
  cmd = gpu_begin(vis->gpu);
  run_vis_merger(vis, &cmd, &ws, &vis->merger, num_patches, out_merged);
  gpu_commit(&cmd);
  minimax_h3_gpu_wait(vis->gpu, prev_cmd);
  minimax_h3_gpu_wait_hooks(vis->gpu, cmd.number, hooks);

  free_vis_workspace(&ws);
  memory_pin(wslot, (int)was_pinned);
  memory_touch(wslot);
  *out_grid_h = grid_h;
  *out_grid_w = grid_w;
  return 1;
}

// Encoded vision embeddings and grid dimensions for one reference image.
typedef struct {
  // Merged visual embeddings [n_merged, 5120] on GPU.
  Buf merged;
  // Three DeepStack feature tensors [n_merged, 5120] on GPU.
  Buf deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS];
  // Patch grid dimensions and merged token count.
  int grid_h, grid_w, n_merged;
  // Start token index of this image's visual embeddings in the full sequence.
  int start_idx;
} EncodedRefVis;

int minimax_h3_encode_multimodal_prompt(
    MinimaxH3TextEncoder *text_enc, MinimaxH3VisionEncoder *vis,
    const char *prompt, const MinimaxH3RefImage *refs, int ref_count,
    MemorySlot *out_slot, Buf *out_hidden, int *out_seq_len, int **out_tags,
    const MinimaxH3RunHooks *hooks) {
  if (!text_enc || ref_count < 0 || (ref_count > 0 && (!vis || !refs)) ||
      !out_hidden || !out_seq_len || !out_tags)
    die("invalid multimodal prompt encoding arguments");
  *out_tags = NULL;

  // Fast path for text-only generation: no vision tower load needed.
  if (ref_count == 0) {
    if (!minimax_h3_text_encoder_encode_prompt(text_enc, prompt ? prompt : "",
                                               out_slot, out_hidden,
                                               out_seq_len, hooks))
      return 0;
    int *tags = xmalloc((size_t)(*out_seq_len) * sizeof(int));
    for (int i = 0; i < *out_seq_len; i++) tags[i] = TAG_TEXT;
    *out_tags = tags;
    return 1;
  }

  Gpu *gpu = minimax_h3_text_encoder_gpu(text_enc);
  EncodedRefVis *enc_refs = xmalloc((size_t)ref_count * sizeof *enc_refs);
  memset(enc_refs, 0, (size_t)ref_count * sizeof *enc_refs);
  int total_vis = 0;

  // Encode each reference image through the vision tower at its Qwen3-VL size.
  for (int i = 0; i < ref_count; i++) {
    int short_w = 0, short_h = 0, vis_w = 0, vis_h = 0;
    size_short_edge(refs[i].width, refs[i].height, REF_VISION_SHORT_EDGE,
                    REF_VISION_SHORT_EDGE, &short_w, &short_h);
    smart_resize_32(short_w, short_h, &vis_w, &vis_h);

    const uint8_t *src_rgb = refs[i].rgb;
    uint8_t *tmp1 = NULL, *tmp2 = NULL;
    if (short_w != refs[i].width || short_h != refs[i].height) {
      tmp1 = image_resize_rgb_lanczos(refs[i].rgb, refs[i].width,
                                      refs[i].height, short_w, short_h);
      src_rgb = tmp1;
    }
    if (vis_w != short_w || vis_h != short_h) {
      tmp2 = image_resize_rgb_bilinear(src_rgb, short_w, short_h, vis_w, vis_h);
      src_rgb = tmp2;
    }

    int ok = minimax_h3_vision_encoder_encode_image(
        vis, src_rgb, vis_w, vis_h, out_slot, &enc_refs[i].merged,
        enc_refs[i].deepstack, &enc_refs[i].grid_h, &enc_refs[i].grid_w, hooks);
    free(tmp2);
    free(tmp1);
    if (!ok) {
      for (int k = 0; k <= i; k++) {
        gpu_free(&enc_refs[k].merged);
        for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
          gpu_free(&enc_refs[k].deepstack[d]);
      }
      free(enc_refs);
      return 0;
    }
    enc_refs[i].n_merged = (enc_refs[i].grid_h / VIS_MERGE_SIDE) *
                           (enc_refs[i].grid_w / VIS_MERGE_SIDE);
    total_vis += enc_refs[i].n_merged;
  }

  // Tokenize reference labels ("Picture %d: ", Comfy-Org/ComfyUI
  // comfy/text_encoders/minimax.py:8,233 and
  // comfy_extras/nodes_minimax_h3.py:768) and raw prompt text.
  int *prompt_ids = xmalloc((size_t)PROMPT_MAX_TOKENS * sizeof(int));
  int prompt_len = 0;
  if (prompt && prompt[0] != '\0')
    prompt_len = minimax_h3_text_encoder_tokenize(text_enc, prompt, prompt_ids,
                                                  PROMPT_MAX_TOKENS);

  int max_seq = total_vis + prompt_len + ref_count * REF_TEMPLATE_TOKENS +
                REF_SEQ_GUARD_TOKENS;
  int *ids = xmalloc((size_t)max_seq * sizeof(int));
  int *tags = xmalloc((size_t)max_seq * sizeof(int));
  int *vis_indices = xmalloc((size_t)total_vis * sizeof(int));
  int seq_len = 0, vis_cursor = 0;

  for (int i = 0; i < ref_count; i++) {
    char label[REF_LABEL_BUF_CAP];
    snprintf(label, sizeof label, "Picture %d: ", i + 1);
    int label_ids[REF_LABEL_MAX_TOKENS];
    int l_len = minimax_h3_text_encoder_tokenize(text_enc, label, label_ids,
                                                 REF_LABEL_MAX_TOKENS);
    for (int j = 0; j < l_len; j++) {
      ids[seq_len] = label_ids[j];
      tags[seq_len] = TAG_TEXT;
      seq_len++;
    }
    ids[seq_len] = MINIMAX_H3_TEXT_VISION_START_TOKEN;
    tags[seq_len] = TAG_VISUAL;
    seq_len++;

    enc_refs[i].start_idx = seq_len;
    for (int j = 0; j < enc_refs[i].n_merged; j++) {
      ids[seq_len] = -1;
      tags[seq_len] = TAG_VISUAL;
      vis_indices[vis_cursor++] = seq_len;
      seq_len++;
    }

    ids[seq_len] = MINIMAX_H3_TEXT_VISION_END_TOKEN;
    tags[seq_len] = TAG_VISUAL;
    seq_len++;
  }
  for (int j = 0; j < prompt_len; j++) {
    ids[seq_len] = prompt_ids[j];
    tags[seq_len] = TAG_TEXT;
    seq_len++;
  }
  free(prompt_ids);

  // Build qwen2vl_mrope_position_ids [seq_len, MINIMAX_H3_POS_AXES] (t, h, w).
  int *pos_ids =
      xmalloc((size_t)seq_len * (size_t)MINIMAX_H3_POS_AXES * sizeof(int));
  for (int i = 0; i < seq_len; i++) {
    pos_ids[MINIMAX_H3_POS_AXES * i] = i;
    pos_ids[MINIMAX_H3_POS_AXES * i + 1] = i;
    pos_ids[MINIMAX_H3_POS_AXES * i + 2] = i;
  }
  int offset = 0;
  for (int i = 0; i < ref_count; i++) {
    int start = enc_refs[i].start_idx;
    int size = enc_refs[i].n_merged;
    int end = start + size;
    int mh = enc_refs[i].grid_h / VIS_MERGE_SIDE;
    int mw = enc_refs[i].grid_w / VIS_MERGE_SIDE;
    int len_max = mh > mw ? mh : mw;
    int base = start + offset;
    for (int k = 0; k < size; k++) {
      int row = k / mw, col = k % mw;
      int idx = start + k;
      pos_ids[MINIMAX_H3_POS_AXES * idx] = base;
      pos_ids[MINIMAX_H3_POS_AXES * idx + 1] = base + row;
      pos_ids[MINIMAX_H3_POS_AXES * idx + 2] = base + col;
    }
    int start_next = len_max + start + offset;
    for (int idx = end; idx < seq_len; idx++) {
      int p = start_next + (idx - end);
      pos_ids[MINIMAX_H3_POS_AXES * idx] = p;
      pos_ids[MINIMAX_H3_POS_AXES * idx + 1] = p;
      pos_ids[MINIMAX_H3_POS_AXES * idx + 2] = p;
    }
    offset += len_max - size;
  }

  // Assemble embeds_override [seq_len, 5120] and concatenated deepstack[0..2].
  size_t row_bytes = (size_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  Buf embeds_override =
      gpu_try_alloc_required(gpu, out_slot, (size_t)seq_len * row_bytes);
  Buf cat_ds[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS] = {0};
  for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
    cat_ds[d] =
        gpu_try_alloc_required(gpu, out_slot, (size_t)total_vis * row_bytes);
  *out_hidden =
      gpu_try_alloc_required(gpu, out_slot, (size_t)seq_len * row_bytes);

  int ok = embeds_override.obj && cat_ds[0].obj && cat_ds[1].obj &&
           cat_ds[2].obj && out_hidden->obj;
  if (ok) {
    memset(embeds_override.ptr, 0, (size_t)seq_len * row_bytes);
    size_t ds_off = 0;
    for (int i = 0; i < ref_count; i++) {
      size_t n_bytes = (size_t)enc_refs[i].n_merged * row_bytes;
      memcpy((uint8_t *)embeds_override.ptr +
                 (size_t)enc_refs[i].start_idx * row_bytes,
             enc_refs[i].merged.ptr, n_bytes);
      for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
        memcpy((uint8_t *)cat_ds[d].ptr + ds_off, enc_refs[i].deepstack[d].ptr,
               n_bytes);
      ds_off += n_bytes;
    }
  }

  // Free per-image vision outputs before running the 32B text encoder.
  for (int i = 0; i < ref_count; i++) {
    gpu_free(&enc_refs[i].merged);
    for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
      gpu_free(&enc_refs[i].deepstack[d]);
  }
  free(enc_refs);

  if (ok) {
    const Buf *ds_ptrs[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS] = {
        &cat_ds[0], &cat_ds[1], &cat_ds[2]};
    ok = minimax_h3_text_encoder_encode_tokens(
        text_enc, ids, pos_ids, &embeds_override, ds_ptrs, vis_indices,
        total_vis, seq_len, out_hidden, hooks);
  }

  for (int d = 0; d < MINIMAX_H3_TEXT_DEEPSTACK_LAYERS; d++)
    gpu_free(&cat_ds[d]);
  gpu_free(&embeds_override);
  free(pos_ids);
  free(vis_indices);
  free(ids);

  if (!ok) {
    gpu_free(out_hidden);
    free(tags);
    return 0;
  }
  *out_seq_len = seq_len;
  *out_tags = tags;
  return 1;
}
