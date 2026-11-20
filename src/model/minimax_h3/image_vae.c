#include "model/minimax_h3/image_vae.h"
#include "base/error.h"
#include "model/minimax_h3/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 2D ViT decoder architecture constants matching vae/config.json.
static constexpr int VAE_LAYERS = 24, VAE_DIM = 1024, VAE_HEADS = 16,
                     VAE_HEAD_DIM = 64, VAE_FFN_INNER = 4096,
                     VAE_QKV_WIDTH = 3 * VAE_DIM,
                     VAE_W1_WIDTH = 2 * VAE_FFN_INNER,
                     VAE_PATCH_OUT = 3 * MINIMAX_H3_VAE_SPATIAL_RATIO *
                                     MINIMAX_H3_VAE_SPATIAL_RATIO,
                     VAE_REGISTERS = 4, VAE_SUFFIX_TOKENS = VAE_REGISTERS + 1,
                     VAE_ROPE_FREQS = 8;

// Spatial tiling constants in pixels.
static constexpr int VAE_TILE_SIZE = 256, VAE_TILE_OVERLAP_MIN = 64;

// Normalization epsilon and RoPE base frequency for the 2D ViT decoder.
static constexpr float VAE_NORM_EPS = 1e-5f;
static constexpr double VAE_ROPE_THETA = 100.0;

// Threadgroup and tile sizes matching src/kernels/minimax_h3_vae.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          NORM_THREADS = 256, EMBED_THREADS = 256,
                          HEADS_THREADS = 32, ELEM_THREADS = 32;

// Per-channel latent mean and standard deviation from MiniMax-H3-Image
// vae/config.json.
static const float LATENTS_MEAN[MINIMAX_H3_VAE_LATENT_CHANNELS] = {
    0.858090341091156f,   -0.9606591463088989f,  1.0661640167236328f,
    -0.5090325474739075f, -0.2727581858634949f,  -1.3675414323806763f,
    -0.2553254961967468f, -0.26907554268836975f, -0.5376840829849243f,
    -0.0464097298681736f, 0.6657370328903198f,   0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f,  -0.23683024942874908f,
    0.25928452610969543f, -0.30133944749832153f, 0.211341992020607f,
    -1.1206848621368408f, 0.3581933379173279f,   -0.04225143790245056f,
    0.2604829967021942f,  0.22864092886447906f,  0.7056031823158264f,
};

static const float LATENTS_STD[MINIMAX_H3_VAE_LATENT_CHANNELS] = {
    1.2223774194717407f, 1.2767263650894165f,  1.6831774711608887f,
    1.7549455165863037f, 1.5636216402053833f,  2.194143533706665f,
    0.9653137922286987f, 1.0569885969161987f,  0.841948926448822f,
    0.7729952931404114f, 1.8955937623977661f,  0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f,
    0.6936293244361877f, 2.961095094680786f,   2.7694199085235596f,
    3.0496184825897217f, 2.1088054180145264f,  3.276226282119751f,
    3.1627357006073f,    2.2816812992095947f,  2.6127843856811523f,
};

// ImageNet RGB mean and standard deviation used to finalize decoded pixels.
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

// Parameters passed to minimax_h3_x_f16.
typedef struct {
  // Active rows, columns and input row stride.
  uint32_t m, k, lda;
} F16XParam;

// Parameters passed to minimax_h3_mm_f16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} F16MMParam;

// Parameters passed to minimax_h3_rmsnorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} RmsNormParam;

// Parameters passed to minimax_h3_add_scaled.
typedef struct {
  // Rows and channels.
  uint32_t rows, dim;
} AddScaledParam;

// Parameters passed to minimax_h3_swiglu_chunked.
typedef struct {
  // Rows and output inner width (half of input width).
  uint32_t rows, inner;
} SwigluChunkedParam;

// Parameters passed to minimax_h3_layernorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} LayerNormParam;

// Parameters passed to minimax_h3_vae_embed.
typedef struct {
  // Tile height and width in latent cells.
  uint32_t h_lat, w_lat;
} VaeEmbedParam;

// Parameters passed to minimax_h3_vae_heads.
typedef struct {
  // Total tokens (patches + 5), padded head stride and latent tile dimensions.
  uint32_t rows, padded, h_lat, w_lat;
  // Per-head non-affine QK RMSNorm epsilon.
  float eps;
} VaeHeadsParam;

// Parameters passed to minimax_h3_vae_attend.
typedef struct {
  // Total tokens and padded head stride.
  uint32_t rows, padded;
} VaeAttendParam;

// One FP16 linear projection with FP32 bias.
typedef struct {
  // FP16 weight matrix (n by k), sector-aligned.
  Buf w;
  // FP32 bias vector (n).
  Buf bias;
  // Output width and inner width.
  int n, k;
} F16Lin;

// Weights for one 2D ViT decoder TransformerBlock.
typedef struct {
  // Pre-attention and pre-FFN RMSNorm weights (FP32, 1024).
  Buf norm1, norm2;
  // Residual branch scales (FP32, 1024).
  Buf scale1, scale2;
  // Head-interleaved QKV projection (3072 by 1024) and output projection.
  F16Lin to_qkv, to_out;
  // SwiGLU FFN up/gate projection (8192 by 1024) and down projection.
  F16Lin ff_w1, ff_w2;
} VaeBlock;

// Transient workspace buffers reused across all tiles of one decode call.
typedef struct {
  // Channel-first tile latent input (24 by max_patches).
  Buf z_tile;
  // Hidden state stream (max_seq by 1024) and normalized staging.
  Buf h, norm;
  // Padded FP16 activation staging (max_padded by 4096).
  Buf xh;
  // QKV projection output (max_seq by 3072) and attention output (max_seq by
  // 1024).
  Buf qkv, attn_out;
  // Head-major FP16 Q, K and V (16 by max_padded by 64).
  Buf qh, kh, vh;
  // FFN gate+up (max_seq by 8192) and SwiGLU inner activation (max_seq by
  // 4096).
  Buf ff_up, ff_mid;
  // Patch projection output (max_patches by 768).
  Buf proj_out;
} VaeWorkspace;

// MiniMax-H3-Image VAE handle.
struct MinimaxH3ImageVae {
  // GPU device and memory account owning the VAE buffers.
  Gpu *gpu;
  // Memory account that bounds both weights and transient workspace.
  MemoryAccount *account;
  // Non-evictable slot tracking transient decode buffers.
  MemorySlot workspace_slot;
  // Open safetensors file for the FP16 VAE checkpoint.
  TensorIndex *tensors;
  // Full-residence weight group for all 2D ViT decoder weights.
  WeightGroup *weights;
  // post_quant_conv (24x24) and x_embedder (1024x24) weights and biases (FP32).
  Buf pqc_w, pqc_b, emb_w, emb_b;
  // Learned register tokens (FP32, 4 by 1024) and 2D RoPE frequencies (FP32,
  // 8).
  Buf register_tokens, inv_freq;
  // 24 ViT transformer blocks.
  VaeBlock blocks[VAE_LAYERS];
  // Final LayerNorm (FP32, 1024) and patch output projection (768 by 1024).
  Buf norm_out_w, norm_out_b;
  F16Lin proj_out;
  // Compiled Metal kernel indices.
  int k_x_f16, k_mm_f16, k_rmsnorm, k_add_scaled, k_swiglu_chunked;
  int k_layernorm, k_embed, k_heads, k_attend;
};

// Generates the 8 FP32 per-axis inverse frequencies 100^(-k / 8) for 2D RoPE.
static void fill_vae_inv_freq(Buf *buffer, const void *context) {
  (void)context;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < VAE_ROPE_FREQS; i++)
    out[i] =
        (float)(1.0 / pow(VAE_ROPE_THETA, (double)i / (double)VAE_ROPE_FREQS));
}

// Registers an FP16 linear projection (weight + bias) into group.
static void add_f16_lin(WeightGroup *group, const TensorIndex *tensors,
                        F16Lin *lin, int n, int k, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
  if (w.dtype != 'h' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k)
    die("unexpected FP16 weight shape for %s", prefix);
  if (b.dtype != 'h' || b.ndim != 1 || b.shape[0] != n)
    die("unexpected FP16 bias shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  weight_group_add_tensor(group, &lin->w, w, 1);
  weight_group_add_vector(group, &lin->bias, b, 0.0f, 0);
}

MinimaxH3VaeTilePlan minimax_h3_image_vae_split_tiles(int input_len) {
  if (input_len <= 0 || input_len % MINIMAX_H3_VAE_SPATIAL_RATIO != 0)
    die("VAE image dimension %d must be a positive multiple of %d", input_len,
        MINIMAX_H3_VAE_SPATIAL_RATIO);
  MinimaxH3VaeTilePlan plan = {0};
  if (VAE_TILE_SIZE >= input_len) {
    plan.count = 1;
    plan.start[0] = 0;
    plan.len[0] = input_len;
    return plan;
  }
  int n = (input_len + VAE_TILE_SIZE - 1) / VAE_TILE_SIZE;
  int remaining = 0;
  for (;;) {
    remaining = VAE_TILE_SIZE * n - VAE_TILE_OVERLAP_MIN * (n - 1) - input_len;
    if (remaining >= 0) break;
    n++;
  }
  if (n > MINIMAX_H3_VAE_MAX_TILES)
    die("VAE image dimension %d needs too many tiles (%d)", input_len, n);
  plan.count = n;
  for (int i = 0; i < n - 1; i++) plan.overlap[i] = VAE_TILE_OVERLAP_MIN;
  int units = remaining / MINIMAX_H3_VAE_SPATIAL_RATIO;
  for (int i = 0; i < units; i++)
    plan.overlap[i % (n - 1)] += MINIMAX_H3_VAE_SPATIAL_RATIO;
  plan.start[0] = 0;
  plan.len[0] = VAE_TILE_SIZE;
  for (int i = 1; i < n; i++) {
    plan.start[i] = plan.start[i - 1] + VAE_TILE_SIZE - plan.overlap[i - 1];
    plan.len[i] = VAE_TILE_SIZE;
  }
  return plan;
}

MinimaxH3ImageVae *minimax_h3_image_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path) {
  MinimaxH3ImageVae *vae = xmalloc(sizeof *vae);
  vae->gpu = gpu;
  vae->account = account;
  memory_slot_init(&vae->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-vae-workspace", NULL, NULL);
  vae->tensors = tensor_index_open_file(weights_path);
  vae->weights = weight_group_new(gpu, account, "minimax-h3-image-vae-decoder");

  vae->k_x_f16 = gpu_find_kernel(gpu, "minimax_h3_x_f16");
  vae->k_mm_f16 = gpu_find_kernel(gpu, "minimax_h3_mm_f16");
  vae->k_rmsnorm = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  vae->k_add_scaled = gpu_find_kernel(gpu, "minimax_h3_add_scaled");
  vae->k_swiglu_chunked = gpu_find_kernel(gpu, "minimax_h3_swiglu_chunked");
  vae->k_layernorm = gpu_find_kernel(gpu, "minimax_h3_layernorm");
  vae->k_embed = gpu_find_kernel(gpu, "minimax_h3_vae_embed");
  vae->k_heads = gpu_find_kernel(gpu, "minimax_h3_vae_heads");
  vae->k_attend = gpu_find_kernel(gpu, "minimax_h3_vae_attend");

  // Small embedding matrices and register tokens are stored as FP32 vectors.
  Tensor pqc_w = tensor_index_get(vae->tensors, "post_quant_conv.weight");
  Tensor pqc_b = tensor_index_get(vae->tensors, "post_quant_conv.bias");
  Tensor emb_w = tensor_index_get(vae->tensors, "decoder.x_embedder.weight");
  Tensor emb_b = tensor_index_get(vae->tensors, "decoder.x_embedder.bias");
  Tensor reg = tensor_index_get(vae->tensors, "decoder.register_tokens");
  if (pqc_w.shape[0] != MINIMAX_H3_VAE_LATENT_CHANNELS ||
      pqc_w.shape[1] != MINIMAX_H3_VAE_LATENT_CHANNELS)
    die("unexpected post_quant_conv.weight shape");
  if (emb_w.shape[0] != VAE_DIM ||
      emb_w.shape[1] != MINIMAX_H3_VAE_LATENT_CHANNELS)
    die("unexpected decoder.x_embedder.weight shape");
  if (reg.size != (size_t)VAE_REGISTERS * VAE_DIM * 2)
    die("unexpected decoder.register_tokens size");
  weight_group_add_vector(vae->weights, &vae->pqc_w, pqc_w, 0.0f, 0);
  weight_group_add_vector(vae->weights, &vae->pqc_b, pqc_b, 0.0f, 0);
  weight_group_add_vector(vae->weights, &vae->emb_w, emb_w, 0.0f, 0);
  weight_group_add_vector(vae->weights, &vae->emb_b, emb_b, 0.0f, 0);
  weight_group_add_vector(vae->weights, &vae->register_tokens, reg, 0.0f, 0);
  weight_group_add_generated(vae->weights, &vae->inv_freq,
                             (size_t)VAE_ROPE_FREQS * sizeof(float),
                             fill_vae_inv_freq, NULL, 0);

  for (int i = 0; i < VAE_LAYERS; i++) {
    VaeBlock *b = &vae->blocks[i];
    weight_group_add_vector(
        vae->weights, &b->norm1,
        tensor_index_get(vae->tensors,
                         "decoder.transformer_blocks.%d.norm1.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vae->weights, &b->norm2,
        tensor_index_get(vae->tensors,
                         "decoder.transformer_blocks.%d.norm2.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vae->weights, &b->scale1,
        tensor_index_get(vae->tensors, "decoder.transformer_blocks.%d.scale1",
                         i),
        0.0f, 0);
    weight_group_add_vector(
        vae->weights, &b->scale2,
        tensor_index_get(vae->tensors, "decoder.transformer_blocks.%d.scale2",
                         i),
        0.0f, 0);

    char name[128];
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.attn.to_qkv", i);
    add_f16_lin(vae->weights, vae->tensors, &b->to_qkv, VAE_QKV_WIDTH, VAE_DIM,
                name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.attn.to_out", i);
    add_f16_lin(vae->weights, vae->tensors, &b->to_out, VAE_DIM, VAE_DIM, name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.ff.w1", i);
    add_f16_lin(vae->weights, vae->tensors, &b->ff_w1, VAE_W1_WIDTH, VAE_DIM,
                name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.ff.w2", i);
    add_f16_lin(vae->weights, vae->tensors, &b->ff_w2, VAE_DIM, VAE_FFN_INNER,
                name);
  }

  weight_group_add_vector(
      vae->weights, &vae->norm_out_w,
      tensor_index_get(vae->tensors, "decoder.norm_out.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->weights, &vae->norm_out_b,
      tensor_index_get(vae->tensors, "decoder.norm_out.bias"), 0.0f, 0);
  add_f16_lin(vae->weights, vae->tensors, &vae->proj_out, VAE_PATCH_OUT,
              VAE_DIM, "decoder.proj_out");
  return vae;
}

void minimax_h3_image_vae_close(MinimaxH3ImageVae *vae) {
  if (!vae) return;
  weight_group_free(vae->weights);
  memory_slot_remove(&vae->workspace_slot);
  tensor_index_close(vae->tensors);
  free(vae);
}

const TensorIndex *minimax_h3_image_vae_tensors(const MinimaxH3ImageVae *vae) {
  return vae->tensors;
}

Gpu *minimax_h3_image_vae_gpu(const MinimaxH3ImageVae *vae) { return vae->gpu; }

WeightGroup *minimax_h3_image_vae_weights(MinimaxH3ImageVae *vae) {
  return vae->weights;
}

uint64_t minimax_h3_image_vae_required_bytes(const MinimaxH3ImageVae *vae) {
  return weight_group_required_bytes(vae->weights);
}

uint64_t minimax_h3_image_vae_resident_bytes(const MinimaxH3ImageVae *vae) {
  return weight_group_resident_bytes(vae->weights);
}

int minimax_h3_image_vae_load(MinimaxH3ImageVae *vae) {
  return weight_group_load(vae->weights);
}

// Frees all transient GPU buffers of a decode workspace.
static void free_vae_workspace(VaeWorkspace *ws) {
  gpu_free(&ws->proj_out);
  gpu_free(&ws->ff_mid);
  gpu_free(&ws->ff_up);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->attn_out);
  gpu_free(&ws->qkv);
  gpu_free(&ws->xh);
  gpu_free(&ws->norm);
  gpu_free(&ws->h);
  gpu_free(&ws->z_tile);
}

// Allocates transient GPU buffers large enough for max_patches latent cells.
static int alloc_vae_workspace(MinimaxH3ImageVae *vae, int max_patches,
                               VaeWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  int max_seq = max_patches + VAE_SUFFIX_TOKENS;
  uint32_t max_padded = ((uint32_t)max_seq + TILE_M - 1) / TILE_M * TILE_M;
  MemorySlot *slot = &vae->workspace_slot;
  ws->z_tile = gpu_try_alloc_required(vae->gpu, slot,
                                      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                                          (size_t)max_patches * sizeof(float));
  ws->h = gpu_try_alloc_required(vae->gpu, slot,
                                 (size_t)max_seq * VAE_DIM * sizeof(float));
  ws->norm = gpu_try_alloc_required(vae->gpu, slot,
                                    (size_t)max_seq * VAE_DIM * sizeof(float));
  ws->xh = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_padded * VAE_FFN_INNER * sizeof(uint16_t));
  ws->qkv = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_seq * VAE_QKV_WIDTH * sizeof(float));
  ws->attn_out = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_seq * VAE_DIM * sizeof(float));
  ws->qh = gpu_try_alloc_required(vae->gpu, slot,
                                  (size_t)VAE_HEADS * max_padded *
                                      VAE_HEAD_DIM * sizeof(uint16_t));
  ws->kh = gpu_try_alloc_required(vae->gpu, slot,
                                  (size_t)VAE_HEADS * max_padded *
                                      VAE_HEAD_DIM * sizeof(uint16_t));
  ws->vh = gpu_try_alloc_required(vae->gpu, slot,
                                  (size_t)VAE_HEADS * max_padded *
                                      VAE_HEAD_DIM * sizeof(uint16_t));
  ws->ff_up = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_seq * VAE_W1_WIDTH * sizeof(float));
  ws->ff_mid = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_seq * VAE_FFN_INNER * sizeof(float));
  ws->proj_out = gpu_try_alloc_required(
      vae->gpu, slot, (size_t)max_patches * VAE_PATCH_OUT * sizeof(float));
  if (!ws->z_tile.obj || !ws->h.obj || !ws->norm.obj || !ws->xh.obj ||
      !ws->qkv.obj || !ws->attn_out.obj || !ws->qh.obj || !ws->kh.obj ||
      !ws->vh.obj || !ws->ff_up.obj || !ws->ff_mid.obj || !ws->proj_out.obj) {
    free_vae_workspace(ws);
    return 0;
  }
  return 1;
}

// Encodes FP32 x -> padded FP16 xh, then native FP16 matmul + bias into y.
static void run_f16_linear(const MinimaxH3ImageVae *vae, GpuCommandBuffer *cmd,
                           const Buf *x, const Buf *xh, const F16Lin *lin,
                           const Buf *y, int rows) {
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  F16XParam xp = {(uint32_t)rows, (uint32_t)lin->k, (uint32_t)lin->k};
  gpu_select_kernel(cmd, vae->k_x_f16);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, xh, 0);
  gpu_copy_bytes(cmd, 2, &xp, sizeof xp);
  gpu_run_kernel(cmd, ((uint32_t)lin->k / 4u + ELEM_THREADS - 1) / ELEM_THREADS,
                 padded, 1, ELEM_THREADS, 1, 1);

  F16MMParam mp = {(uint32_t)rows, (uint32_t)lin->n, (uint32_t)lin->k,
                   (uint32_t)lin->n, 1};
  gpu_select_kernel(cmd, vae->k_mm_f16);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, xh, 0);
  gpu_bind_buffer(cmd, 2, &lin->bias, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(cmd, ((uint32_t)lin->n + TILE_N - 1) / TILE_N, padded / TILE_M,
                 1, MM_THREADS, 1, 1);
}

// Runs the 24-layer 2D ViT decoder on one tile already copied into ws->z_tile,
// and unpatchifies the result into channel-first FP32 tile_out [3, tile_h,
// tile_w].
static void decode_one_tile(MinimaxH3ImageVae *vae, const VaeWorkspace *ws,
                            int h_lat, int w_lat, float *tile_out) {
  int patches = h_lat * w_lat;
  int seq = patches + VAE_SUFFIX_TOKENS;
  uint32_t padded = ((uint32_t)seq + TILE_M - 1) / TILE_M * TILE_M;

  GpuCommandBuffer cmd = gpu_begin(vae->gpu);
  VaeEmbedParam ep = {(uint32_t)h_lat, (uint32_t)w_lat};
  gpu_select_kernel(&cmd, vae->k_embed);
  gpu_bind_buffer(&cmd, 0, &ws->z_tile, 0);
  gpu_bind_buffer(&cmd, 1, &vae->pqc_w, 0);
  gpu_bind_buffer(&cmd, 2, &vae->pqc_b, 0);
  gpu_bind_buffer(&cmd, 3, &vae->emb_w, 0);
  gpu_bind_buffer(&cmd, 4, &vae->emb_b, 0);
  gpu_bind_buffer(&cmd, 5, &vae->register_tokens, 0);
  gpu_bind_buffer(&cmd, 6, &ws->h, 0);
  gpu_copy_bytes(&cmd, 7, &ep, sizeof ep);
  gpu_run_kernel(&cmd, (uint32_t)seq, 1, 1, EMBED_THREADS, 1, 1);

  RmsNormParam np = {(uint32_t)VAE_DIM, VAE_NORM_EPS};
  AddScaledParam sp = {(uint32_t)seq, (uint32_t)VAE_DIM};
  SwigluChunkedParam swp = {(uint32_t)seq, (uint32_t)VAE_FFN_INNER};
  VaeHeadsParam hp = {(uint32_t)seq, padded, (uint32_t)h_lat, (uint32_t)w_lat,
                      VAE_NORM_EPS};
  VaeAttendParam ap = {(uint32_t)seq, padded};

  for (int i = 0; i < VAE_LAYERS; i++) {
    const VaeBlock *b = &vae->blocks[i];

    // Attention branch: norm1 -> QKV -> QK-Norm + 2D RoPE -> attention ->
    // to_out.
    gpu_select_kernel(&cmd, vae->k_rmsnorm);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm1, 0);
    gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 3, &np, sizeof np);
    gpu_run_kernel(&cmd, (uint32_t)seq, 1, 1, NORM_THREADS, 1, 1);

    run_f16_linear(vae, &cmd, &ws->norm, &ws->xh, &b->to_qkv, &ws->qkv, seq);

    gpu_select_kernel(&cmd, vae->k_heads);
    gpu_bind_buffer(&cmd, 0, &ws->qkv, 0);
    gpu_bind_buffer(&cmd, 1, &vae->inv_freq, 0);
    gpu_bind_buffer(&cmd, 2, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 4, &ws->vh, 0);
    gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
    gpu_run_kernel(&cmd, (uint32_t)(3 * VAE_HEADS), padded, 1, HEADS_THREADS, 1,
                   1);

    gpu_select_kernel(&cmd, vae->k_attend);
    gpu_bind_buffer(&cmd, 0, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws->vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->attn_out, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, padded / TILE_M, (uint32_t)VAE_HEADS, 1, MM_THREADS, 1,
                   1);

    run_f16_linear(vae, &cmd, &ws->attn_out, &ws->xh, &b->to_out, &ws->norm,
                   seq);

    gpu_select_kernel(&cmd, vae->k_add_scaled);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &b->scale1, 0);
    gpu_copy_bytes(&cmd, 3, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (uint32_t)VAE_DIM / ELEM_THREADS, (uint32_t)seq, 1,
                   ELEM_THREADS, 1, 1);

    // Feed-forward branch: norm2 -> w1 -> SwiGLU -> w2 -> scaled residual add.
    gpu_select_kernel(&cmd, vae->k_rmsnorm);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm2, 0);
    gpu_bind_buffer(&cmd, 2, &ws->norm, 0);
    gpu_copy_bytes(&cmd, 3, &np, sizeof np);
    gpu_run_kernel(&cmd, (uint32_t)seq, 1, 1, NORM_THREADS, 1, 1);

    run_f16_linear(vae, &cmd, &ws->norm, &ws->xh, &b->ff_w1, &ws->ff_up, seq);

    gpu_select_kernel(&cmd, vae->k_swiglu_chunked);
    gpu_bind_buffer(&cmd, 0, &ws->ff_up, 0);
    gpu_bind_buffer(&cmd, 1, &ws->ff_mid, 0);
    gpu_copy_bytes(&cmd, 2, &swp, sizeof swp);
    gpu_run_kernel(&cmd, (uint32_t)VAE_FFN_INNER / ELEM_THREADS, (uint32_t)seq,
                   1, ELEM_THREADS, 1, 1);

    run_f16_linear(vae, &cmd, &ws->ff_mid, &ws->xh, &b->ff_w2, &ws->norm, seq);

    gpu_select_kernel(&cmd, vae->k_add_scaled);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &b->scale2, 0);
    gpu_copy_bytes(&cmd, 3, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (uint32_t)VAE_DIM / ELEM_THREADS, (uint32_t)seq, 1,
                   ELEM_THREADS, 1, 1);
  }

  // Final LayerNorm + proj_out on the patch tokens (suffix tokens dropped).
  LayerNormParam lnp = {(uint32_t)VAE_DIM, VAE_NORM_EPS};
  gpu_select_kernel(&cmd, vae->k_layernorm);
  gpu_bind_buffer(&cmd, 0, &ws->h, 0);
  gpu_bind_buffer(&cmd, 1, &vae->norm_out_w, 0);
  gpu_bind_buffer(&cmd, 2, &vae->norm_out_b, 0);
  gpu_bind_buffer(&cmd, 3, &ws->norm, 0);
  gpu_copy_bytes(&cmd, 4, &lnp, sizeof lnp);
  gpu_run_kernel(&cmd, (uint32_t)patches, 1, 1, NORM_THREADS, 1, 1);

  run_f16_linear(vae, &cmd, &ws->norm, &ws->xh, &vae->proj_out, &ws->proj_out,
                 patches);

  gpu_commit(&cmd);
  minimax_h3_gpu_wait(vae->gpu, cmd.number);

  // Unpatchify [h_lat * w_lat, 3 * 16 * 16] -> [3, h_lat * 16, w_lat * 16].
  const float *src = (const float *)ws->proj_out.ptr;
  int p = MINIMAX_H3_VAE_SPATIAL_RATIO;
  int tile_h = h_lat * p, tile_w = w_lat * p;
  size_t plane = (size_t)tile_h * (size_t)tile_w;
  for (int py = 0; py < h_lat; py++) {
    for (int px = 0; px < w_lat; px++) {
      const float *tok =
          src + (size_t)(py * w_lat + px) * (size_t)VAE_PATCH_OUT;
      for (int c = 0; c < 3; c++) {
        float *dst_plane = tile_out + (size_t)c * plane;
        for (int ly = 0; ly < p; ly++) {
          float *dst_row = dst_plane + (size_t)(py * p + ly) * (size_t)tile_w +
                           (size_t)(px * p);
          const float *src_row = tok + (size_t)(c * p + ly) * (size_t)p;
          memcpy(dst_row, src_row, (size_t)p * sizeof(float));
        }
      }
    }
  }
}

int minimax_h3_image_vae_decode(MinimaxH3ImageVae *vae, const float *latents,
                                int h_lat, int w_lat, uint8_t *out_rgb) {
  if (h_lat <= 0 || w_lat <= 0 || !latents || !out_rgb)
    die("invalid VAE decode arguments");
  if (!minimax_h3_image_vae_load(vae)) return 0;

  int height = h_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  int width = w_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  MinimaxH3VaeTilePlan y_plan = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan x_plan = minimax_h3_image_vae_split_tiles(width);

  int max_tile_h_lat = 0, max_tile_w_lat = 0;
  for (int i = 0; i < y_plan.count; i++) {
    int hl = y_plan.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    if (hl > max_tile_h_lat) max_tile_h_lat = hl;
  }
  for (int j = 0; j < x_plan.count; j++) {
    int wl = x_plan.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    if (wl > max_tile_w_lat) max_tile_w_lat = wl;
  }

  MemorySlot *wslot = weight_group_slot(vae->weights);
  memory_pin(wslot, 1);
  VaeWorkspace ws;
  if (!alloc_vae_workspace(vae, max_tile_h_lat * max_tile_w_lat, &ws)) {
    memory_pin(wslot, 0);
    return 0;
  }

  // Host buffers for canvas, tile pixels and row/column overlap strips.
  size_t canvas_plane = (size_t)height * (size_t)width;
  float *canvas = xmalloc(3 * canvas_plane * sizeof(float));
  size_t max_tile_px = (size_t)(max_tile_h_lat * MINIMAX_H3_VAE_SPATIAL_RATIO) *
                       (size_t)(max_tile_w_lat * MINIMAX_H3_VAE_SPATIAL_RATIO);
  float *tile = xmalloc(3 * max_tile_px * sizeof(float));
  float *strip = NULL, *left_tail = NULL;

  int out_y = 0;
  size_t lat_plane = (size_t)h_lat * (size_t)w_lat;
  for (int i = 0; i < y_plan.count; i++) {
    int ty_lat = y_plan.start[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int th_lat = y_plan.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int tile_h = y_plan.len[i];
    int ov_top = i > 0 ? y_plan.overlap[i - 1] : 0;
    int ov_bot = i < y_plan.count - 1 ? y_plan.overlap[i] : 0;
    float *new_strip =
        ov_bot ? xmalloc(3 * (size_t)ov_bot * (size_t)width * sizeof(float))
               : NULL;

    int out_x = 0;
    int keep_h = tile_h - ov_bot;
    for (int j = 0; j < x_plan.count; j++) {
      int tx_lat = x_plan.start[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int tw_lat = x_plan.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int tile_w = x_plan.len[j];
      int ov_left = j > 0 ? x_plan.overlap[j - 1] : 0;
      int ov_right = j < x_plan.count - 1 ? x_plan.overlap[j] : 0;

      // Copy and denormalize the tile's 24-channel latents into ws.z_tile.
      float *z_dst = (float *)ws.z_tile.ptr;
      size_t tile_lat_plane = (size_t)th_lat * (size_t)tw_lat;
      for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
        float mean = LATENTS_MEAN[c], std = LATENTS_STD[c];
        const float *src_c = latents + (size_t)c * lat_plane;
        float *dst_c = z_dst + (size_t)c * tile_lat_plane;
        for (int y = 0; y < th_lat; y++) {
          const float *srow =
              src_c + (size_t)(ty_lat + y) * (size_t)w_lat + (size_t)tx_lat;
          float *drow = dst_c + (size_t)y * (size_t)tw_lat;
          for (int x = 0; x < tw_lat; x++) drow[x] = srow[x] * std + mean;
        }
      }

      decode_one_tile(vae, &ws, th_lat, tw_lat, tile);

      size_t tile_plane = (size_t)tile_h * (size_t)tile_w;
      // Vertical linear ramp blend with the previous tile row's bottom strip.
      if (ov_top > 0) {
        size_t strip_plane = (size_t)ov_top * (size_t)width;
        for (int c = 0; c < 3; c++) {
          float *tp = tile + (size_t)c * tile_plane;
          const float *sp = strip + (size_t)c * strip_plane;
          for (int y = 0; y < ov_top; y++) {
            float wb = (float)y / (float)ov_top, wa = 1.0f - wb;
            float *trow = tp + (size_t)y * (size_t)tile_w;
            const float *srow =
                sp + (size_t)y * (size_t)width + (size_t)x_plan.start[j];
            for (int x = 0; x < tile_w; x++)
              trow[x] = srow[x] * wa + trow[x] * wb;
          }
        }
      }

      // Horizontal linear ramp blend with the previous tile's right edge.
      if (ov_left > 0) {
        size_t tail_plane = (size_t)tile_h * (size_t)ov_left;
        for (int c = 0; c < 3; c++) {
          float *tp = tile + (size_t)c * tile_plane;
          const float *lp = left_tail + (size_t)c * tail_plane;
          for (int y = 0; y < tile_h; y++) {
            float *trow = tp + (size_t)y * (size_t)tile_w;
            const float *lrow = lp + (size_t)y * (size_t)ov_left;
            for (int x = 0; x < ov_left; x++) {
              float wb = (float)x / (float)ov_left, wa = 1.0f - wb;
              trow[x] = lrow[x] * wa + trow[x] * wb;
            }
          }
        }
      }

      free(left_tail);
      left_tail = NULL;
      if (ov_right > 0) {
        size_t tail_plane = (size_t)tile_h * (size_t)ov_right;
        left_tail = xmalloc(3 * tail_plane * sizeof(float));
        for (int c = 0; c < 3; c++) {
          const float *tp = tile + (size_t)c * tile_plane;
          float *lp = left_tail + (size_t)c * tail_plane;
          for (int y = 0; y < tile_h; y++)
            memcpy(lp + (size_t)y * (size_t)ov_right,
                   tp + (size_t)y * (size_t)tile_w +
                       (size_t)(tile_w - ov_right),
                   (size_t)ov_right * sizeof(float));
        }
      }

      int keep_w = tile_w - ov_right;
      if (ov_bot > 0) {
        size_t strip_plane = (size_t)ov_bot * (size_t)width;
        for (int c = 0; c < 3; c++) {
          const float *tp = tile + (size_t)c * tile_plane;
          float *np = new_strip + (size_t)c * strip_plane;
          for (int y = 0; y < ov_bot; y++)
            memcpy(np + (size_t)y * (size_t)width + (size_t)out_x,
                   tp + (size_t)(keep_h + y) * (size_t)tile_w,
                   (size_t)keep_w * sizeof(float));
        }
      }

      for (int c = 0; c < 3; c++) {
        const float *tp = tile + (size_t)c * tile_plane;
        float *cp = canvas + (size_t)c * canvas_plane;
        for (int y = 0; y < keep_h; y++)
          memcpy(cp + (size_t)(out_y + y) * (size_t)width + (size_t)out_x,
                 tp + (size_t)y * (size_t)tile_w,
                 (size_t)keep_w * sizeof(float));
      }
      out_x += keep_w;
    }

    free(left_tail);
    left_tail = NULL;
    free(strip);
    strip = new_strip;
    out_y += keep_h;
  }

  free(strip);
  free(tile);
  free_vae_workspace(&ws);
  memory_pin(wslot, 0);
  memory_touch(wslot);

  // Finalize ImageNet-normalized FP32 canvas to 8-bit interleaved RGB pixels.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      size_t idx = (size_t)y * (size_t)width + (size_t)x;
      uint8_t *px = out_rgb + idx * 3;
      for (int c = 0; c < 3; c++) {
        float v = canvas[(size_t)c * canvas_plane + idx] * IMAGENET_STD[c] +
                  IMAGENET_MEAN[c];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        px[c] = (uint8_t)lroundf(v * 255.0f);
      }
    }
  }
  free(canvas);
  return 1;
}
