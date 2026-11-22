#include "model/inkling2/vision.h"
#include "base/error.h"
#include "engine/memory.h"
#include "engine/weights.h"
#include "model/inkling2/model.h"
#include "platform/image.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Inkling-2-Mini vision preprocessing, lazy weight loading, and GPU forward
// pass (image_processing_inkling2.py and modeling_inkling2.py
// Inkling2VisionModel).

// Preprocessing and numerical constants from processor_config.json and
// config.json vision_config.
static constexpr int CHECKER_CELL = 8;
static constexpr int CHECKER_LIGHT = 204;
static constexpr int CHECKER_DARK = 153;
static constexpr int BYTE_MAX = 255;
static constexpr int BYTE_HALF = 127;
static constexpr float PIXEL_PAD_RAW = -1.0f;
static constexpr float PIXEL_SCALE = 255.0f;
static constexpr float PIXEL_MEAN = 0.5f;
static constexpr float PIXEL_STD = 0.5f;
static constexpr float LAYER_NORM_EPS = 1e-5f;
static constexpr float ROPE_THETA = 10000.0f;
static constexpr int ROPE_SPATIAL_DIM = INK2_VIS_HEAD_DIM / 2;
static constexpr int ROPE_AXIS_PAIRS = ROPE_SPATIAL_DIM / 2;
static constexpr float CELL_CENTER = 0.5f;
static constexpr int MAX_IMAGE_TOKENS = 16384;
static constexpr uint64_t FNV_OFFSET = 1469598103934665603ull;
static constexpr uint64_t FNV_PRIME = 1099511628211ull;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_SHIFT = 16;

// Metal dispatch constants matching src/kernels/inkling2_vision.metal and
// src/kernels/kernels.metal.
static constexpr int ATTN_PAD = 64;
static constexpr int ATTN_Q_GROUP = 64;
static constexpr int MM_TILE_M = 64;
static constexpr int MM_TILE_N = 64;
static constexpr uint32_t MM_THREADS = 128;
static constexpr uint32_t LN_THREADS = 256;
static constexpr uint32_t ELEM_THREADS = 256;
static constexpr uint32_t GRID2D_THREADS = 64;
static constexpr uint32_t HEADS_THREADS = INK2_VIS_ROPE_PAIRS;
static constexpr uint32_t ATTN_THREADS = 128;
static constexpr size_t ALIGN_READ_MIN_BYTES = 1u << 20;
static constexpr size_t ALIGN_MASK = 15u;

// Kernel parameter structs matching src/kernels/inkling2_vision.metal and
// src/kernels/kernels.metal.
typedef struct {
  // Active rows, output width, inner width, BF16 rounding flag and strides.
  uint32_t m, n, k, round, lda, ldy;
} VisMmParam;

typedef struct {
  // Element count for BF16<->FP32 conversion.
  uint32_t count;
} VisCastParam;

typedef struct {
  // Patch count (excluding CLS) and vision width.
  uint32_t npatches, dim;
} VisEmbedParam;

typedef struct {
  // Row width and LayerNorm epsilon.
  uint32_t dim;
  float eps;
} VisLnParam;

typedef struct {
  // Row count, row width, add-bias flag and GELU flag.
  uint32_t rows, dim, has_bias, apply_gelu;
} VisBiasActParam;

typedef struct {
  // Sequence length (1 + npatches) and padded head row stride.
  uint32_t seq, padded;
} VisHeadsParam;

typedef struct {
  // Sequence length and padded head row stride.
  uint32_t seq, padded;
} VisAttendParam;

typedef struct {
  // Row count and row width for LayerScale residual add.
  uint32_t rows, dim;
} VisScaleAddParam;

// Weights for one of the 23 Inkling2 Vision Transformer layers.
typedef struct {
  // Pre-attention and pre-MLP LayerNorm weights and biases (FP32 [1024]).
  Buf ln1_w, ln1_b, ln2_w, ln2_b;
  // LayerScale gammas after attention and MLP (FP32 [1024]).
  Buf ls1_gamma, ls2_gamma;
  // Attention BF16 weight matrices [1024, 1024] and FP32 biases [1024].
  Buf q_w, k_w, v_w, o_w;
  Buf q_b, k_b, v_b, o_b;
  // MLP BF16 weight matrices [4096, 1024] / [1024, 4096] and FP32 biases.
  Buf fc1_w, fc2_w;
  Buf fc1_b, fc2_b;
} VisLayer;

// Module-level handles initialized by inkling2_vision_init.
static Gpu *vis_gpu;
static MemorySlot *vis_work_slot;
static TensorIndex *vis_checkpoint;
static WeightGroup *vis_weights;
static int vis_registered;

// Vision embedding, pre-norm, 23 transformer layers, and 2x2 merge projector.
static Buf vis_cls_emb, vis_cls_pos, vis_pos_table, vis_patch_w;
static Buf vis_ln_pre_w, vis_ln_pre_b;
static VisLayer vis_layers[INK2_VIS_LAYERS];
static Buf vis_proj_in_w, vis_proj_out_w;

// Compiled Metal kernel handles.
static int K_mm_bf16, K_bf16_to_f32, K_f32_to_bf16, K_embed_assemble;
static int K_layernorm, K_bias_act, K_heads_rope, K_attend, K_layerscale_add;

// Rounds one FP32 value to BF16 ties-to-even.
static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return (uint16_t)((u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) >> BF16_SHIFT);
}

// Clamps integer x to [lo, hi].
static int clamp_int(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Rounds n up to a multiple of align (a power of two).
static int align_up(int n, int align) { return (n + align - 1) & ~(align - 1); }

// Composites an 8-bit RGB buffer with straight alpha onto an 8x8 checkerboard.
void inkling2_vision_composite_checkerboard(uint8_t *rgb, const uint8_t *alpha,
                                            int width, int height) {
  for (int y = 0; y < height; y++) {
    int cy = y / CHECKER_CELL;
    for (int x = 0; x < width; x++) {
      int bg =
          (((cy + x / CHECKER_CELL) & 1) == 0) ? CHECKER_LIGHT : CHECKER_DARK;
      size_t idx = (size_t)y * (size_t)width + (size_t)x;
      int a = alpha[idx];
      int inv_a = BYTE_MAX - a;
      uint8_t *px = rgb + idx * INK2_VIS_CHANNELS;
      for (int c = 0; c < INK2_VIS_CHANNELS; c++)
        px[c] = (uint8_t)((px[c] * a + bg * inv_a + BYTE_HALF) / BYTE_MAX);
    }
  }
}

// Computes 4-tap bilinear (align_corners=False, border clamp) indices and
// weights into the 32x32 position embedding table in 2x2 merge-block order.
void inkling2_vision_bilinear_taps(int vh, int vw, uint32_t *indices,
                                   float *weights) {
  int th = vh / INK2_VIS_MERGE, tw = vw / INK2_VIS_MERGE;
  int side = INK2_VIS_POS_GRID;
  size_t p = 0;
  for (int tr = 0; tr < th; tr++) {
    for (int tc = 0; tc < tw; tc++) {
      for (int mr = 0; mr < INK2_VIS_MERGE; mr++) {
        for (int mc = 0; mc < INK2_VIS_MERGE; mc++, p++) {
          int row = tr * INK2_VIS_MERGE + mr;
          int col = tc * INK2_VIS_MERGE + mc;
          float sh = ((float)row + CELL_CENTER) * (float)side / (float)vh -
                     CELL_CENTER;
          float sw = ((float)col + CELL_CENTER) * (float)side / (float)vw -
                     CELL_CENTER;
          float fh = floorf(sh), fw = floorf(sw);
          float dh = sh - fh, dw = sw - fw;
          int h0 = clamp_int((int)fh, 0, side - 1);
          int h1 = clamp_int((int)fh + 1, 0, side - 1);
          int w0 = clamp_int((int)fw, 0, side - 1);
          int w1 = clamp_int((int)fw + 1, 0, side - 1);
          float wh0 = 1.0f - dh, wh1 = dh;
          float ww0 = 1.0f - dw, ww1 = dw;
          uint32_t *ip = indices + p * INK2_VIS_TAPS;
          float *wp = weights + p * INK2_VIS_TAPS;
          ip[0] = (uint32_t)(h0 * side + w0);
          ip[1] = (uint32_t)(h0 * side + w1);
          ip[2] = (uint32_t)(h1 * side + w0);
          ip[3] = (uint32_t)(h1 * side + w1);
          wp[0] = wh0 * ww0;
          wp[1] = wh0 * ww1;
          wp[2] = wh1 * ww0;
          wp[3] = wh1 * ww1;
        }
      }
    }
  }
}

// Computes 2D axial pairwise RoPE (cos, sin) pairs for CLS at (0, 0) followed
// by 1-indexed patches in merge-block order (width pairs 0..15, height 16..31).
void inkling2_vision_rope_table(int vh, int vw, float *cos_sin) {
  float inv_freq[ROPE_AXIS_PAIRS];
  for (int k = 0; k < ROPE_AXIS_PAIRS; k++)
    inv_freq[k] = 1.0f / powf(ROPE_THETA, (float)(2 * k) / ROPE_SPATIAL_DIM);

  // Row 0 is the CLS token at (h=0, w=0).
  for (int pair = 0; pair < INK2_VIS_ROPE_PAIRS; pair++) {
    cos_sin[pair * 2] = 1.0f;
    cos_sin[pair * 2 + 1] = 0.0f;
  }

  int th = vh / INK2_VIS_MERGE, tw = vw / INK2_VIS_MERGE;
  size_t seq_idx = 1;
  for (int tr = 0; tr < th; tr++) {
    for (int tc = 0; tc < tw; tc++) {
      for (int mr = 0; mr < INK2_VIS_MERGE; mr++) {
        for (int mc = 0; mc < INK2_VIS_MERGE; mc++, seq_idx++) {
          float ph = (float)(tr * INK2_VIS_MERGE + mr + 1);
          float pw = (float)(tc * INK2_VIS_MERGE + mc + 1);
          float *row = cos_sin + seq_idx * INK2_VIS_ROPE_PAIRS * 2;
          for (int k = 0; k < ROPE_AXIS_PAIRS; k++) {
            float aw = pw * inv_freq[k];
            row[k * 2] = cosf(aw);
            row[k * 2 + 1] = sinf(aw);
            float ah = ph * inv_freq[k];
            row[(ROPE_AXIS_PAIRS + k) * 2] = cosf(ah);
            row[(ROPE_AXIS_PAIRS + k) * 2 + 1] = sinf(ah);
          }
        }
      }
    }
  }
}

// Adds a raw vision weight tensor to vis_weights with aligned direct read when
// large and 16-byte aligned.
static void add_vis_raw(Buf *dst, Tensor t) {
  int aligned =
      t.size >= ALIGN_READ_MIN_BYTES && (t.off & ALIGN_MASK) == 0 ? 1 : 0;
  weight_group_add_tensor(vis_weights, dst, t, aligned);
}

// Registers all vision tower and projector weights into the lazy dependent
// WeightGroup on first use.
static void register_vision_weights(void) {
  if (vis_registered) return;
  weight_group_add_vector(
      vis_weights, &vis_cls_emb,
      tensor_index_get(vis_checkpoint,
                       "model.visual.backbone.embeddings.class_embedding"),
      0.0f, 0);
  weight_group_add_vector(
      vis_weights, &vis_cls_pos,
      tensor_index_get(
          vis_checkpoint,
          "model.visual.backbone.embeddings.cls_position_embedding"),
      0.0f, 0);
  weight_group_add_vector(
      vis_weights, &vis_pos_table,
      tensor_index_get(vis_checkpoint,
                       "model.visual.backbone.embeddings.position_embedding"),
      0.0f, 0);
  weight_group_add_padded(
      vis_weights, &vis_patch_w,
      tensor_index_get(
          vis_checkpoint,
          "model.visual.backbone.embeddings.patch_embedding.weight"),
      INK2_VIS_PATCH_K_PAD);

  weight_group_add_vector(
      vis_weights, &vis_ln_pre_w,
      tensor_index_get(vis_checkpoint, "model.visual.backbone.ln_pre.weight"),
      0.0f, 0);
  weight_group_add_vector(
      vis_weights, &vis_ln_pre_b,
      tensor_index_get(vis_checkpoint, "model.visual.backbone.ln_pre.bias"),
      0.0f, 0);

  for (int l = 0; l < INK2_VIS_LAYERS; l++) {
    VisLayer *vl = &vis_layers[l];
    weight_group_add_vector(
        vis_weights, &vl->ln1_w,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ln_1.weight", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->ln1_b,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ln_1.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->ln2_w,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ln_2.weight", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->ln2_b,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ln_2.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->ls1_gamma,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ls_1.gamma", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->ls2_gamma,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.ls_2.gamma", l),
        0.0f, 0);

    add_vis_raw(&vl->q_w,
                tensor_index_get(
                    vis_checkpoint,
                    "model.visual.backbone.layers.%d.self_attn.q_proj.weight",
                    l));
    add_vis_raw(&vl->k_w,
                tensor_index_get(
                    vis_checkpoint,
                    "model.visual.backbone.layers.%d.self_attn.k_proj.weight",
                    l));
    add_vis_raw(&vl->v_w,
                tensor_index_get(
                    vis_checkpoint,
                    "model.visual.backbone.layers.%d.self_attn.v_proj.weight",
                    l));
    add_vis_raw(&vl->o_w,
                tensor_index_get(
                    vis_checkpoint,
                    "model.visual.backbone.layers.%d.self_attn.out_proj.weight",
                    l));
    weight_group_add_vector(
        vis_weights, &vl->q_b,
        tensor_index_get(
            vis_checkpoint,
            "model.visual.backbone.layers.%d.self_attn.q_proj.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->k_b,
        tensor_index_get(
            vis_checkpoint,
            "model.visual.backbone.layers.%d.self_attn.k_proj.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->v_b,
        tensor_index_get(
            vis_checkpoint,
            "model.visual.backbone.layers.%d.self_attn.v_proj.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->o_b,
        tensor_index_get(
            vis_checkpoint,
            "model.visual.backbone.layers.%d.self_attn.out_proj.bias", l),
        0.0f, 0);

    add_vis_raw(
        &vl->fc1_w,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.mlp.c_fc.weight", l));
    add_vis_raw(&vl->fc2_w,
                tensor_index_get(
                    vis_checkpoint,
                    "model.visual.backbone.layers.%d.mlp.c_proj.weight", l));
    weight_group_add_vector(
        vis_weights, &vl->fc1_b,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.mlp.c_fc.bias", l),
        0.0f, 0);
    weight_group_add_vector(
        vis_weights, &vl->fc2_b,
        tensor_index_get(vis_checkpoint,
                         "model.visual.backbone.layers.%d.mlp.c_proj.bias", l),
        0.0f, 0);
  }

  add_vis_raw(&vis_proj_in_w,
              tensor_index_get(vis_checkpoint, "model.visual.proj_in.weight"));
  add_vis_raw(&vis_proj_out_w,
              tensor_index_get(vis_checkpoint, "model.visual.proj_out.weight"));
  vis_registered = 1;
}

// Initializes the vision module's GPU handles and dependent weight group.
void inkling2_vision_init(Gpu *gpu, MemorySlot *work_slot,
                          TensorIndex *checkpoint, WeightGroup *text_weights) {
  vis_gpu = gpu;
  vis_work_slot = work_slot;
  vis_checkpoint = checkpoint;
  if (!vis_weights)
    vis_weights =
        weight_group_new_dependent(text_weights, "Inkling-2-Mini vision");
  K_mm_bf16 = gpu_find_kernel(gpu, "mm_bf16");
  K_bf16_to_f32 = gpu_find_kernel(gpu, "ink2_vis_bf16_to_f32");
  K_f32_to_bf16 = gpu_find_kernel(gpu, "ink2_vis_f32_to_bf16");
  K_embed_assemble = gpu_find_kernel(gpu, "ink2_vis_embed_assemble");
  K_layernorm = gpu_find_kernel(gpu, "ink2_vis_layernorm");
  K_bias_act = gpu_find_kernel(gpu, "ink2_vis_bias_act");
  K_heads_rope = gpu_find_kernel(gpu, "ink2_vis_heads_rope");
  K_attend = gpu_find_kernel(gpu, "ink2_vis_attend");
  K_layerscale_add = gpu_find_kernel(gpu, "ink2_vis_layerscale_add");
}

// Decodes an image file and packs padded BF16 patches into `im`.
int inkling2_prepare_media(const char *dir, Media *im, const void *data,
                           size_t len, const char **error) {
  (void)dir;
  int w = 0, h = 0;
  uint8_t *rgb = image_decode_rgb(data, len, &w, &h, error);
  if (!rgb) return 0;

  int aw = 0, ah = 0;
  const char *aerr = NULL;
  uint8_t *alpha = image_decode_alpha(data, len, &aw, &ah, &aerr);
  if (alpha) {
    if (aw == w && ah == h)
      inkling2_vision_composite_checkerboard(rgb, alpha, w, h);
    free(alpha);
  }

  int th = (h + INK2_VIS_CELL - 1) / INK2_VIS_CELL;
  int tw = (w + INK2_VIS_CELL - 1) / INK2_VIS_CELL;
  int64_t ntokens = (int64_t)th * (int64_t)tw;
  if (ntokens <= 0 || ntokens > MAX_IMAGE_TOKENS) {
    free(rgb);
    *error = "image exceeds maximum Inkling-2-Mini token grid";
    return 0;
  }

  int vh = th * INK2_VIS_MERGE, vw = tw * INK2_VIS_MERGE;
  size_t npatches = (size_t)vh * (size_t)vw;
  uint16_t *patches = calloc(npatches * INK2_VIS_PATCH_K_PAD, sizeof(uint16_t));
  if (!patches) {
    free(rgb);
    *error = "image patches do not fit in memory";
    return 0;
  }

  float pad_norm = (PIXEL_PAD_RAW / PIXEL_SCALE - PIXEL_MEAN) / PIXEL_STD;
  uint16_t pad_bf16 = f32_to_bf16(pad_norm);

  size_t p = 0;
  for (int tr = 0; tr < th; tr++) {
    for (int tc = 0; tc < tw; tc++) {
      for (int mr = 0; mr < INK2_VIS_MERGE; mr++) {
        for (int mc = 0; mc < INK2_VIS_MERGE; mc++, p++) {
          uint16_t *dst = patches + p * INK2_VIS_PATCH_K_PAD;
          int y0 = (tr * INK2_VIS_MERGE + mr) * INK2_VIS_PATCH;
          int x0 = (tc * INK2_VIS_MERGE + mc) * INK2_VIS_PATCH;
          for (int py = 0; py < INK2_VIS_PATCH; py++) {
            int y = y0 + py;
            for (int px = 0; px < INK2_VIS_PATCH; px++) {
              int x = x0 + px;
              uint16_t *cell =
                  dst + (py * INK2_VIS_PATCH + px) * INK2_VIS_CHANNELS;
              if (y < h && x < w) {
                const uint8_t *src = rgb + ((size_t)y * (size_t)w + (size_t)x) *
                                               INK2_VIS_CHANNELS;
                for (int c = 0; c < INK2_VIS_CHANNELS; c++) {
                  float v =
                      ((float)src[c] / PIXEL_SCALE - PIXEL_MEAN) / PIXEL_STD;
                  cell[c] = f32_to_bf16(v);
                }
              } else {
                for (int c = 0; c < INK2_VIS_CHANNELS; c++) cell[c] = pad_bf16;
              }
            }
          }
        }
      }
    }
  }
  free(rgb);

  uint64_t hash = FNV_OFFSET;
  const uint8_t *bytes = (const uint8_t *)data;
  for (size_t i = 0; i < len; i++) hash = (hash ^ bytes[i]) * FNV_PRIME;

  *im = (Media){.ntokens = (int)ntokens,
                .vh = vh,
                .vw = vw,
                .th = th,
                .tw = tw,
                .patch = patches,
                .hash = hash};
  return 1;
}

// Dispatches mm_bf16 on vision matrices (y = x @ w^T with optional BF16
// rounding).
static void vis_mm(GpuCommandBuffer *c, const Buf *w, const Buf *x,
                   size_t x_off, const Buf *y, int m, int n, int k, int round) {
  VisMmParam p = {(uint32_t)m,     (uint32_t)n, (uint32_t)k,
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

// Dispatches vision LayerNorm on `rows` rows of INK2_VIS_DIM floats.
static void vis_ln(GpuCommandBuffer *c, const Buf *x, const Buf *w,
                   const Buf *b, const Buf *y, int rows) {
  VisLnParam p = {(uint32_t)INK2_VIS_DIM, LAYER_NORM_EPS};
  gpu_select_kernel(c, K_layernorm);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, b, 0);
  gpu_bind_buffer(c, 3, y, 0);
  gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)rows, 1, 1, LN_THREADS, 1, 1);
}

// Encodes `im->patch` through the 23-layer ViT and 2x2 spatial-merge projector.
int inkling2_encode_media(Media *im) {
  if (im->rows) return 1;
  if (!tensor_index_has(vis_checkpoint, "model.visual.proj_out.weight"))
    die("checkpoint does not include Inkling-2-Mini vision weights");
  register_vision_weights();
  MemorySlot *vslot = weight_group_slot(vis_weights);
  memory_pin(vslot, 1);
  if (!weight_group_load(vis_weights)) {
    memory_pin(vslot, 0);
    return 0;
  }
  weight_group_set_resident(vis_weights, 1);
  gpu_commit_residency(vis_gpu);

  int npatches = im->vh * im->vw;
  int seq = 1 + npatches;
  int padded = align_up(seq, ATTN_PAD);
  int ntokens = im->ntokens;

  Buf patch_bf16 = {0}, patch_f32 = {0}, patch_emb = {0};
  Buf pos_idx = {0}, pos_w = {0}, rope_cs = {0};
  Buf x = {0}, xn = {0}, q = {0}, k = {0}, v = {0}, qh = {0}, kh = {0},
      vh_buf = {0};
  Buf attn_out = {0}, proj_tmp = {0}, mlp_h = {0}, out_f32 = {0},
      out_bf16 = {0};

  Buf *all[] = {&patch_bf16, &patch_f32, &patch_emb, &pos_idx, &pos_w,
                &rope_cs,    &x,         &xn,        &q,       &k,
                &v,          &qh,        &kh,        &vh_buf,  &attn_out,
                &proj_tmp,   &mlp_h,     &out_f32,   &out_bf16};

  patch_bf16 = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                      (size_t)npatches * INK2_VIS_PATCH_K_PAD *
                                          sizeof(uint16_t));
  patch_f32 = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                     (size_t)npatches * INK2_VIS_PATCH_K_PAD *
                                         sizeof(float));
  patch_emb = gpu_try_alloc_required(
      vis_gpu, vis_work_slot, (size_t)npatches * INK2_VIS_DIM * sizeof(float));
  pos_idx = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                   (size_t)npatches * INK2_VIS_TAPS *
                                       sizeof(uint32_t));
  pos_w = gpu_try_alloc_required(
      vis_gpu, vis_work_slot, (size_t)npatches * INK2_VIS_TAPS * sizeof(float));
  rope_cs = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                   (size_t)seq * INK2_VIS_ROPE_PAIRS * 2 *
                                       sizeof(float));
  x = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                             (size_t)seq * INK2_VIS_DIM * sizeof(float));
  xn = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                              (size_t)seq * INK2_VIS_DIM * sizeof(float));
  q = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                             (size_t)seq * INK2_VIS_DIM * sizeof(float));
  k = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                             (size_t)seq * INK2_VIS_DIM * sizeof(float));
  v = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                             (size_t)seq * INK2_VIS_DIM * sizeof(float));
  size_t head_bytes = (size_t)INK2_VIS_HEADS * (size_t)padded *
                      INK2_VIS_HEAD_DIM * sizeof(uint16_t);
  qh = gpu_try_alloc_required(vis_gpu, vis_work_slot, head_bytes);
  kh = gpu_try_alloc_required(vis_gpu, vis_work_slot, head_bytes);
  vh_buf = gpu_try_alloc_required(vis_gpu, vis_work_slot, head_bytes);
  attn_out = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                    (size_t)seq * INK2_VIS_DIM * sizeof(float));
  proj_tmp = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                    (size_t)seq * INK2_VIS_DIM * sizeof(float));
  mlp_h = gpu_try_alloc_required(
      vis_gpu, vis_work_slot, (size_t)seq * INK2_VIS_MLP_DIM * sizeof(float));
  out_f32 = gpu_try_alloc_required(vis_gpu, vis_work_slot,
                                   (size_t)ntokens * INK2_DIM * sizeof(float));
  out_bf16 = gpu_try_alloc_required(
      vis_gpu, vis_work_slot, (size_t)ntokens * INK2_DIM * sizeof(uint16_t));

  int alloc_ok = 1;
  for (size_t i = 0; i < sizeof all / sizeof *all; i++)
    if (!all[i]->obj) alloc_ok = 0;
  if (!alloc_ok) {
    for (size_t i = 0; i < sizeof all / sizeof *all; i++)
      if (all[i]->obj) gpu_free(all[i]);
    memory_pin(vslot, 0);
    return 0;
  }

  memcpy(patch_bf16.ptr, im->patch, patch_bf16.size);
  inkling2_vision_bilinear_taps(im->vh, im->vw, (uint32_t *)pos_idx.ptr,
                                (float *)pos_w.ptr);
  inkling2_vision_rope_table(im->vh, im->vw, (float *)rope_cs.ptr);

  GpuCommandBuffer cmd = gpu_begin(vis_gpu);

  // Convert BF16 patches to FP32 and run patch_embedding.
  VisCastParam cp_in = {(uint32_t)(npatches * INK2_VIS_PATCH_K_PAD)};
  gpu_select_kernel(&cmd, K_bf16_to_f32);
  gpu_bind_buffer(&cmd, 0, &patch_bf16, 0);
  gpu_bind_buffer(&cmd, 1, &patch_f32, 0);
  gpu_copy_bytes(&cmd, 2, &cp_in, sizeof cp_in);
  gpu_run_kernel(&cmd, (cp_in.count + ELEM_THREADS - 1u) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  vis_mm(&cmd, &vis_patch_w, &patch_f32, 0, &patch_emb, npatches, INK2_VIS_DIM,
         INK2_VIS_PATCH_K_PAD, 1);

  // Assemble [CLS, patches + bilinear pos_emb] and run ln_pre.
  VisEmbedParam ep = {(uint32_t)npatches, (uint32_t)INK2_VIS_DIM};
  gpu_select_kernel(&cmd, K_embed_assemble);
  gpu_bind_buffer(&cmd, 0, &patch_emb, 0);
  gpu_bind_buffer(&cmd, 1, &vis_pos_table, 0);
  gpu_bind_buffer(&cmd, 2, &vis_cls_emb, 0);
  gpu_bind_buffer(&cmd, 3, &vis_cls_pos, 0);
  gpu_bind_buffer(&cmd, 4, &pos_idx, 0);
  gpu_bind_buffer(&cmd, 5, &pos_w, 0);
  gpu_bind_buffer(&cmd, 6, &xn, 0);
  gpu_copy_bytes(&cmd, 7, &ep, sizeof ep);
  gpu_run_kernel(
      &cmd, ((uint32_t)INK2_VIS_DIM + GRID2D_THREADS - 1u) / GRID2D_THREADS,
      (uint32_t)seq, 1, GRID2D_THREADS, 1, 1);

  vis_ln(&cmd, &xn, &vis_ln_pre_w, &vis_ln_pre_b, &x, seq);

  // 23 Vision Transformer encoder layers.
  VisHeadsParam hp = {(uint32_t)seq, (uint32_t)padded};
  VisAttendParam ap = {(uint32_t)seq, (uint32_t)padded};
  VisScaleAddParam sp = {(uint32_t)seq, (uint32_t)INK2_VIS_DIM};
  VisBiasActParam mlp_act = {(uint32_t)seq, (uint32_t)INK2_VIS_MLP_DIM, 1u, 1u};
  uint32_t gx_dim =
      ((uint32_t)INK2_VIS_DIM + GRID2D_THREADS - 1u) / GRID2D_THREADS;
  uint32_t gx_mlp =
      ((uint32_t)INK2_VIS_MLP_DIM + GRID2D_THREADS - 1u) / GRID2D_THREADS;

  for (int l = 0; l < INK2_VIS_LAYERS; l++) {
    const VisLayer *vl = &vis_layers[l];
    vis_ln(&cmd, &x, &vl->ln1_w, &vl->ln1_b, &xn, seq);
    vis_mm(&cmd, &vl->q_w, &xn, 0, &q, seq, INK2_VIS_DIM, INK2_VIS_DIM, 1);
    vis_mm(&cmd, &vl->k_w, &xn, 0, &k, seq, INK2_VIS_DIM, INK2_VIS_DIM, 1);
    vis_mm(&cmd, &vl->v_w, &xn, 0, &v, seq, INK2_VIS_DIM, INK2_VIS_DIM, 1);

    gpu_select_kernel(&cmd, K_heads_rope);
    gpu_bind_buffer(&cmd, 0, &q, 0);
    gpu_bind_buffer(&cmd, 1, &k, 0);
    gpu_bind_buffer(&cmd, 2, &v, 0);
    gpu_bind_buffer(&cmd, 3, &vl->q_b, 0);
    gpu_bind_buffer(&cmd, 4, &vl->k_b, 0);
    gpu_bind_buffer(&cmd, 5, &vl->v_b, 0);
    gpu_bind_buffer(&cmd, 6, &rope_cs, 0);
    gpu_bind_buffer(&cmd, 7, &qh, 0);
    gpu_bind_buffer(&cmd, 8, &kh, 0);
    gpu_bind_buffer(&cmd, 9, &vh_buf, 0);
    gpu_copy_bytes(&cmd, 10, &hp, sizeof hp);
    gpu_run_kernel(&cmd, (uint32_t)INK2_VIS_HEADS, (uint32_t)padded, 1,
                   HEADS_THREADS, 1, 1);

    gpu_select_kernel(&cmd, K_attend);
    gpu_bind_buffer(&cmd, 0, &qh, 0);
    gpu_bind_buffer(&cmd, 1, &kh, 0);
    gpu_bind_buffer(&cmd, 2, &vh_buf, 0);
    gpu_bind_buffer(&cmd, 3, &attn_out, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    uint32_t q_tiles = ((uint32_t)seq + ATTN_Q_GROUP - 1u) / ATTN_Q_GROUP;
    gpu_run_kernel(&cmd, q_tiles * (uint32_t)INK2_VIS_HEADS, 1, 1, ATTN_THREADS,
                   1, 1);

    vis_mm(&cmd, &vl->o_w, &attn_out, 0, &proj_tmp, seq, INK2_VIS_DIM,
           INK2_VIS_DIM, 1);
    gpu_select_kernel(&cmd, K_layerscale_add);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &proj_tmp, 0);
    gpu_bind_buffer(&cmd, 2, &vl->o_b, 0);
    gpu_bind_buffer(&cmd, 3, &vl->ls1_gamma, 0);
    gpu_copy_bytes(&cmd, 4, &sp, sizeof sp);
    gpu_run_kernel(&cmd, gx_dim, (uint32_t)seq, 1, GRID2D_THREADS, 1, 1);

    vis_ln(&cmd, &x, &vl->ln2_w, &vl->ln2_b, &xn, seq);
    vis_mm(&cmd, &vl->fc1_w, &xn, 0, &mlp_h, seq, INK2_VIS_MLP_DIM,
           INK2_VIS_DIM, 1);
    gpu_select_kernel(&cmd, K_bias_act);
    gpu_bind_buffer(&cmd, 0, &mlp_h, 0);
    gpu_bind_buffer(&cmd, 1, &vl->fc1_b, 0);
    gpu_copy_bytes(&cmd, 2, &mlp_act, sizeof mlp_act);
    gpu_run_kernel(&cmd, gx_mlp, (uint32_t)seq, 1, GRID2D_THREADS, 1, 1);

    vis_mm(&cmd, &vl->fc2_w, &mlp_h, 0, &proj_tmp, seq, INK2_VIS_DIM,
           INK2_VIS_MLP_DIM, 1);
    gpu_select_kernel(&cmd, K_layerscale_add);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &proj_tmp, 0);
    gpu_bind_buffer(&cmd, 2, &vl->fc2_b, 0);
    gpu_bind_buffer(&cmd, 3, &vl->ls2_gamma, 0);
    gpu_copy_bytes(&cmd, 4, &sp, sizeof sp);
    gpu_run_kernel(&cmd, gx_dim, (uint32_t)seq, 1, GRID2D_THREADS, 1, 1);
  }

  // Drop CLS row 0; remaining 4 * ntokens rows of 1024 are contiguous as
  // [ntokens, 4096]. Run proj_in -> GELU -> proj_out -> BF16.
  size_t patch_byte_off = (size_t)INK2_VIS_DIM * sizeof(float);
  vis_mm(&cmd, &vis_proj_in_w, &x, patch_byte_off, &mlp_h, ntokens,
         INK2_VIS_PROJ_DIM, INK2_VIS_PROJ_DIM, 1);
  VisBiasActParam proj_act = {(uint32_t)ntokens, (uint32_t)INK2_VIS_PROJ_DIM,
                              0u, 1u};
  gpu_select_kernel(&cmd, K_bias_act);
  gpu_bind_buffer(&cmd, 0, &mlp_h, 0);
  gpu_bind_buffer(&cmd, 1, &vis_ln_pre_b, 0);
  gpu_copy_bytes(&cmd, 2, &proj_act, sizeof proj_act);
  gpu_run_kernel(&cmd, gx_mlp, (uint32_t)ntokens, 1, GRID2D_THREADS, 1, 1);

  vis_mm(&cmd, &vis_proj_out_w, &mlp_h, 0, &out_f32, ntokens, INK2_DIM,
         INK2_VIS_PROJ_DIM, 1);
  VisCastParam cp_out = {(uint32_t)(ntokens * INK2_DIM)};
  gpu_select_kernel(&cmd, K_f32_to_bf16);
  gpu_bind_buffer(&cmd, 0, &out_f32, 0);
  gpu_bind_buffer(&cmd, 1, &out_bf16, 0);
  gpu_copy_bytes(&cmd, 2, &cp_out, sizeof cp_out);
  gpu_run_kernel(&cmd, (cp_out.count + ELEM_THREADS - 1u) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  gpu_commit(&cmd);
  gpu_sync(vis_gpu);

  size_t out_bytes = (size_t)ntokens * INK2_DIM * sizeof(uint16_t);
  im->rows = memory_host_alloc_required(vis_work_slot, out_bytes);
  memcpy(im->rows, out_bf16.ptr, out_bytes);
  im->memory = vis_work_slot;

  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
  memory_pin(vslot, 0);
  return 1;
}

// Returns the resident bytes of the lazy vision WeightGroup, or 0 while
// unloaded.
uint64_t inkling2_vision_resident_bytes(void) {
  return vis_weights ? weight_group_resident_bytes(vis_weights) : 0;
}
