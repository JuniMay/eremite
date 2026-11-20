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

// 3D causal CNN encoder architecture constants (single-frame T=1).
static constexpr int ENC_LEVELS = 6, ENC_BLOCKS_PER_LEVEL = 2,
                     ENC_GN_GROUPS = 32,
                     ENC_MOMENT_CH = 2 * MINIMAX_H3_VAE_LATENT_CHANNELS,
                     ENC_BASE_CH = 128;

// Encoder GroupNorm epsilon, posterior sampling seed and logvar clamp range.
static constexpr float ENC_GN_EPS = 1e-6f, ENC_LOGVAR_MIN = -30.0f,
                       ENC_LOGVAR_MAX = 20.0f;
static constexpr uint64_t ENC_POSTERIOR_SEED = 42;

// Cap on FP16 im2col staging elements (16384 x 1152 = 36 MiB) per conv strip.
static constexpr size_t ENC_IM2COL_MAX_HALFS = 16384u * 1152u,
                        ENC_IM2COL_MIN_HALFS = 256u * 9216u;

// Channel counts and spatial downsampling flags for each encoder level.
static const int ENC_CH[ENC_LEVELS] = {128, 256, 256, 512, 512, 1024};
static const int ENC_SPACE_DOWN[ENC_LEVELS] = {2, 2, 2, 2, 1, 1};

// Threadgroup and tile sizes matching src/kernels/minimax_h3_vae.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          NORM_THREADS = 256, EMBED_THREADS = 256,
                          HEADS_THREADS = 32, ELEM_THREADS = 32,
                          ADD_THREADS = 256;

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

// Parameters passed to minimax_h3_groupnorm_silu.
typedef struct {
  // Spatial pixel count, channel count and group count.
  uint32_t hw, channels, groups;
  // GroupNorm epsilon.
  float eps;
} GroupNormParam;

// Parameters passed to minimax_h3_im2col_f16.
typedef struct {
  // Input spatial height, width and channels.
  uint32_t in_h, in_w, cin;
  // Output width, starting output row and strip height.
  uint32_t out_w, out_y0, strip_h;
  // Kernel side (1 or 3), spatial stride, top/left pad and padded K width.
  uint32_t ksize, stride, pad_t, pad_l, k_padded;
} Im2ColParam;

// Parameters for generating a padded 2D FP16 weight matrix from a 5D Conv3d
// checkpoint tensor by slicing its last temporal tap (t = kT - 1).
typedef struct {
  // Source 5D F16 Conv3d weight tensor [cout, cin, kt, ksize, ksize].
  Tensor tensor;
  // Output channels, input channels, temporal kernel size and spatial side.
  int cout, cin, kt, ksize;
  // Output channels and inner K padded to multiples of 64.
  int cout_padded, k_padded;
} EncConvWeightParam;

// One single-frame 2D convolution extracted from a CausalConv3d / Conv3d layer.
typedef struct {
  // Padded FP16 weight matrix [cout_padded, k_padded] in tap-major NHWC layout.
  Buf w;
  // FP32 bias vector [cout].
  Buf bias;
  // Input channels, output channels, spatial kernel side (1 or 3) and padded K.
  int cin, cout, ksize, k_padded;
  // Spatial stride (1 or 2) and top/left reflect padding (0 or 1).
  int stride, pad;
} EncConv;

// Weights for one encoder ResnetBlock3D (single-frame T=1).
typedef struct {
  // Pre-conv1 GroupNorm weight and bias (FP32, cin).
  Buf norm1_w, norm1_b;
  // Pre-conv2 GroupNorm weight and bias (FP32, cout).
  Buf norm2_w, norm2_b;
  // First and second 3x3 convolutions.
  EncConv conv1, conv2;
  // Optional 1x1 channel-change shortcut convolution.
  EncConv shortcut;
  // Non-zero when cin != cout and shortcut is present.
  int has_shortcut;
} EncResBlock;

// Weights for one encoder resolution level.
typedef struct {
  // Two residual blocks at this resolution.
  EncResBlock blocks[ENC_BLOCKS_PER_LEVEL];
  // Optional stride-2 spatial downsampling convolution.
  EncConv downsample;
  // Non-zero when this level downsamples spatially by 2x.
  int has_downsample;
} EncLevel;

// Transient workspace buffers reused across all tiles of one encode call.
typedef struct {
  // Three ping-pong NHWC FP32 activation buffers sized for level 0.
  Buf a, b, c;
  // Padded FP16 im2col staging buffer.
  Buf im2col;
  // Capacity of im2col in 16-bit half elements.
  size_t im2col_halfs;
} EncWorkspace;

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
  // Full-residence weight group for single-frame 3D causal CNN encoder weights.
  WeightGroup *enc_weights;
  // post_quant_conv (24x24) and x_embedder (1024x24) weights and biases (FP32).
  Buf pqc_w, pqc_b, emb_w, emb_b;
  // Learned register tokens (FP32, 4 by 1024) and 2D RoPE frequencies (FP32,
  // 8).
  Buf register_tokens, inv_freq;
  // 24 ViT transformer blocks.
  VaeBlock blocks[VAE_LAYERS];
  // Final LayerNorm (FP32, 1024) and patch output projection (768 by 1024).
  Buf norm_out_w, norm_out_b;
  // Patch projection from decoder width to 16x16 RGB patch outputs.
  F16Lin proj_out;
  // Single-frame encoder input conv, 6 levels, final norm/conv and quant_conv.
  EncConv enc_conv_in;
  // Six encoder resolution levels.
  EncLevel enc_levels[ENC_LEVELS];
  // Encoder output GroupNorm weights and biases (FP32, 1024).
  Buf enc_norm_out_w, enc_norm_out_b;
  // Encoder output conv (1024 -> 48) and 1x1 quant_conv (48 -> 48).
  EncConv enc_conv_out, enc_quant_conv;
  // Compiled Metal kernel indices for decoder and encoder ops.
  int k_x_f16, k_mm_f16, k_rmsnorm, k_add_scaled, k_swiglu_chunked, k_layernorm,
      k_embed, k_heads, k_attend, k_add, k_groupnorm_silu, k_im2col_f16;
};

// Generates the 8 FP32 per-axis inverse frequencies 100^(-k / 8) for 2D RoPE.
static void fill_vae_inv_freq(Buf *buffer, const void *context) {
  (void)context;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < VAE_ROPE_FREQS; i++)
    out[i] =
        (float)(1.0 / pow(VAE_ROPE_THETA, (double)i / (double)VAE_ROPE_FREQS));
}

// Extracts the last temporal slice (t = kt - 1) of a 5D F16 Conv3d weight
// tensor [cout, cin, kt, ksize, ksize] and writes a padded 2D FP16 matrix
// [cout_padded, k_padded] in tap-major NHWC order (ky, kx, ci).
static void fill_enc_conv_weight(Buf *buffer, const void *context) {
  const EncConvWeightParam *p = context;
  uint16_t *raw = xmalloc(p->tensor.size);
  minimax_h3_read_tensor_bytes(p->tensor, raw);
  size_t out_bytes =
      (size_t)p->cout_padded * (size_t)p->k_padded * sizeof(uint16_t);
  memset(buffer->ptr, 0, out_bytes);
  uint16_t *dst = (uint16_t *)buffer->ptr;
  int t_last = p->kt - 1;
  for (int co = 0; co < p->cout; co++) {
    uint16_t *drow = dst + (size_t)co * (size_t)p->k_padded;
    for (int ci = 0; ci < p->cin; ci++) {
      for (int ky = 0; ky < p->ksize; ky++) {
        for (int kx = 0; kx < p->ksize; kx++) {
          size_t sidx =
              (((((size_t)co * (size_t)p->cin + (size_t)ci) * (size_t)p->kt +
                 (size_t)t_last) *
                    (size_t)p->ksize +
                (size_t)ky) *
                   (size_t)p->ksize +
               (size_t)kx);
          size_t didx =
              (size_t)(ky * p->ksize + kx) * (size_t)p->cin + (size_t)ci;
          drow[didx] = raw[sidx];
        }
      }
    }
  }
  free(raw);
}

// Registers one single-frame encoder convolution (5D F16 weight + 1D F16 bias)
// into group.
static void add_enc_conv(WeightGroup *group, const TensorIndex *tensors,
                         EncConv *conv, int cin, int cout, int ksize,
                         int stride, int pad, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
  int kt = ksize;
  if (w.dtype != 'h' || w.ndim != 5 || w.shape[0] != cout ||
      w.shape[1] != cin || w.shape[2] != kt || w.shape[3] != ksize ||
      w.shape[4] != ksize)
    die("unexpected encoder conv weight shape for %s", prefix);
  if (b.dtype != 'h' || b.ndim != 1 || b.shape[0] != cout)
    die("unexpected encoder conv bias shape for %s", prefix);
  int cout_padded = (cout + (int)TILE_N - 1) / (int)TILE_N * (int)TILE_N;
  int k_raw = ksize * ksize * cin;
  int k_padded = (k_raw + (int)TILE_N - 1) / (int)TILE_N * (int)TILE_N;
  conv->cin = cin;
  conv->cout = cout;
  conv->ksize = ksize;
  conv->k_padded = k_padded;
  conv->stride = stride;
  conv->pad = pad;
  EncConvWeightParam param = {
      .tensor = w,
      .cout = cout,
      .cin = cin,
      .kt = kt,
      .ksize = ksize,
      .cout_padded = cout_padded,
      .k_padded = k_padded,
  };
  weight_group_add_generated(group, &conv->w,
                             (size_t)cout_padded * (size_t)k_padded *
                                 sizeof(uint16_t),
                             fill_enc_conv_weight, &param, sizeof param);
  weight_group_add_vector(group, &conv->bias, b, 0.0f, 0);
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
  vae->enc_weights =
      weight_group_new(gpu, account, "minimax-h3-image-vae-encoder");

  vae->k_x_f16 = gpu_find_kernel(gpu, "minimax_h3_x_f16");
  vae->k_mm_f16 = gpu_find_kernel(gpu, "minimax_h3_mm_f16");
  vae->k_rmsnorm = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  vae->k_add_scaled = gpu_find_kernel(gpu, "minimax_h3_add_scaled");
  vae->k_swiglu_chunked = gpu_find_kernel(gpu, "minimax_h3_swiglu_chunked");
  vae->k_layernorm = gpu_find_kernel(gpu, "minimax_h3_layernorm");
  vae->k_embed = gpu_find_kernel(gpu, "minimax_h3_vae_embed");
  vae->k_heads = gpu_find_kernel(gpu, "minimax_h3_vae_heads");
  vae->k_attend = gpu_find_kernel(gpu, "minimax_h3_vae_attend");
  vae->k_add = gpu_find_kernel(gpu, "minimax_h3_add");
  vae->k_groupnorm_silu = gpu_find_kernel(gpu, "minimax_h3_groupnorm_silu");
  vae->k_im2col_f16 = gpu_find_kernel(gpu, "minimax_h3_im2col_f16");

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

  // Single-frame 3D causal CNN encoder weights (loaded on demand for edits).
  add_enc_conv(vae->enc_weights, vae->tensors, &vae->enc_conv_in, 3,
               ENC_BASE_CH, 3, 1, 1, "encoder.conv_in");
  for (int lv = 0; lv < ENC_LEVELS; lv++) {
    EncLevel *el = &vae->enc_levels[lv];
    int c_in_lv = lv == 0 ? ENC_CH[0] : ENC_CH[lv - 1];
    int c_out_lv = ENC_CH[lv];
    for (int bi = 0; bi < ENC_BLOCKS_PER_LEVEL; bi++) {
      EncResBlock *rb = &el->blocks[bi];
      int cin = bi == 0 ? c_in_lv : c_out_lv;
      int cout = c_out_lv;
      weight_group_add_vector(
          vae->enc_weights, &rb->norm1_w,
          tensor_index_get(vae->tensors,
                           "encoder.down.%d.block.%d.norm1.weight", lv, bi),
          0.0f, 0);
      weight_group_add_vector(
          vae->enc_weights, &rb->norm1_b,
          tensor_index_get(vae->tensors, "encoder.down.%d.block.%d.norm1.bias",
                           lv, bi),
          0.0f, 0);
      weight_group_add_vector(
          vae->enc_weights, &rb->norm2_w,
          tensor_index_get(vae->tensors,
                           "encoder.down.%d.block.%d.norm2.weight", lv, bi),
          0.0f, 0);
      weight_group_add_vector(
          vae->enc_weights, &rb->norm2_b,
          tensor_index_get(vae->tensors, "encoder.down.%d.block.%d.norm2.bias",
                           lv, bi),
          0.0f, 0);
      char name[128];
      snprintf(name, sizeof name, "encoder.down.%d.block.%d.conv1", lv, bi);
      add_enc_conv(vae->enc_weights, vae->tensors, &rb->conv1, cin, cout, 3, 1,
                   1, name);
      snprintf(name, sizeof name, "encoder.down.%d.block.%d.conv2", lv, bi);
      add_enc_conv(vae->enc_weights, vae->tensors, &rb->conv2, cout, cout, 3, 1,
                   1, name);
      rb->has_shortcut = (cin != cout);
      if (rb->has_shortcut) {
        snprintf(name, sizeof name, "encoder.down.%d.block.%d.nin_shortcut", lv,
                 bi);
        add_enc_conv(vae->enc_weights, vae->tensors, &rb->shortcut, cin, cout,
                     1, 1, 0, name);
      }
    }
    el->has_downsample = (ENC_SPACE_DOWN[lv] > 1);
    if (el->has_downsample) {
      char name[128];
      snprintf(name, sizeof name, "encoder.down.%d.downsample.conv", lv);
      add_enc_conv(vae->enc_weights, vae->tensors, &el->downsample, c_out_lv,
                   c_out_lv, 3, 2, 0, name);
    }
  }
  weight_group_add_vector(
      vae->enc_weights, &vae->enc_norm_out_w,
      tensor_index_get(vae->tensors, "encoder.norm_out.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->enc_weights, &vae->enc_norm_out_b,
      tensor_index_get(vae->tensors, "encoder.norm_out.bias"), 0.0f, 0);
  add_enc_conv(vae->enc_weights, vae->tensors, &vae->enc_conv_out,
               ENC_CH[ENC_LEVELS - 1], ENC_MOMENT_CH, 3, 1, 1,
               "encoder.conv_out");
  add_enc_conv(vae->enc_weights, vae->tensors, &vae->enc_quant_conv,
               ENC_MOMENT_CH, ENC_MOMENT_CH, 1, 1, 0, "quant_conv");
  return vae;
}

void minimax_h3_image_vae_close(MinimaxH3ImageVae *vae) {
  if (!vae) return;
  weight_group_free(vae->enc_weights);
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

WeightGroup *minimax_h3_image_vae_encoder_weights(MinimaxH3ImageVae *vae) {
  return vae->enc_weights;
}

uint64_t
minimax_h3_image_vae_encoder_required_bytes(const MinimaxH3ImageVae *vae) {
  return weight_group_required_bytes(vae->enc_weights);
}

uint64_t
minimax_h3_image_vae_encoder_resident_bytes(const MinimaxH3ImageVae *vae) {
  return weight_group_resident_bytes(vae->enc_weights);
}

int minimax_h3_image_vae_load_encoder(MinimaxH3ImageVae *vae) {
  return weight_group_load(vae->enc_weights);
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

// Deterministic SplitMix64 generator for posterior latent sampling.
static uint64_t enc_splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Fills count floats with standard normal N(0, 1) draws via Box-Muller.
static void enc_fill_gaussian(float *out, size_t count, uint64_t seed) {
  uint64_t state = seed;
  static constexpr double TWO_PI = 6.28318530717958647692;
  static constexpr double INV_2_53 = 1.0 / 9007199254740992.0;
  for (size_t i = 0; i < count; i += 2) {
    double u1 = ((double)(enc_splitmix64_next(&state) >> 11) + 0.5) * INV_2_53;
    double u2 = (double)(enc_splitmix64_next(&state) >> 11) * INV_2_53;
    double r = sqrt(-2.0 * log(u1));
    double theta = TWO_PI * u2;
    out[i] = (float)(r * cos(theta));
    if (i + 1 < count) out[i + 1] = (float)(r * sin(theta));
  }
}

// Frees all transient GPU buffers of an encoder workspace.
static void free_enc_workspace(EncWorkspace *ws) {
  gpu_free(&ws->im2col);
  gpu_free(&ws->c);
  gpu_free(&ws->b);
  gpu_free(&ws->a);
}

// Allocates transient GPU buffers for single-frame encoder tiles up to
// max_tile_h by max_tile_w pixels.
static int alloc_enc_workspace(MinimaxH3ImageVae *vae, int max_tile_h,
                               int max_tile_w, EncWorkspace *ws) {
  memset(ws, 0, sizeof *ws);
  size_t act_bytes = (size_t)max_tile_h * (size_t)max_tile_w *
                     (size_t)ENC_BASE_CH * sizeof(float);
  size_t wanted_halfs =
      (size_t)max_tile_h * (size_t)max_tile_w * 9u * (size_t)ENC_BASE_CH;
  if (wanted_halfs < ENC_IM2COL_MIN_HALFS) wanted_halfs = ENC_IM2COL_MIN_HALFS;
  if (wanted_halfs > ENC_IM2COL_MAX_HALFS) wanted_halfs = ENC_IM2COL_MAX_HALFS;
  ws->im2col_halfs = wanted_halfs;

  MemorySlot *slot = &vae->workspace_slot;
  ws->a = gpu_try_alloc_required(vae->gpu, slot, act_bytes);
  ws->b = gpu_try_alloc_required(vae->gpu, slot, act_bytes);
  ws->c = gpu_try_alloc_required(vae->gpu, slot, act_bytes);
  ws->im2col =
      gpu_try_alloc_required(vae->gpu, slot, wanted_halfs * sizeof(uint16_t));
  if (!ws->a.obj || !ws->b.obj || !ws->c.obj || !ws->im2col.obj) {
    free_enc_workspace(ws);
    return 0;
  }
  return 1;
}

// Encodes fused NHWC GroupNorm(32, channels, eps=1e-6) + SiLU.
static void run_enc_gn_silu(const MinimaxH3ImageVae *vae, GpuCommandBuffer *cmd,
                            const Buf *x, const Buf *w, const Buf *b,
                            const Buf *y, int h, int w_px, int channels) {
  GroupNormParam p = {(uint32_t)(h * w_px), (uint32_t)channels,
                      (uint32_t)ENC_GN_GROUPS, ENC_GN_EPS};
  gpu_select_kernel(cmd, vae->k_groupnorm_silu);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, w, 0);
  gpu_bind_buffer(cmd, 2, b, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &p, sizeof p);
  gpu_run_kernel(cmd, (uint32_t)ENC_GN_GROUPS, 1, 1, NORM_THREADS, 1, 1);
}

// Encodes a single-frame 2D convolution via strip-chunked FP16 im2col + matmul.
static void run_enc_conv(const MinimaxH3ImageVae *vae, GpuCommandBuffer *cmd,
                         const EncWorkspace *ws, const Buf *x,
                         const EncConv *conv, const Buf *y, int in_h,
                         int in_w) {
  int out_h = in_h / conv->stride, out_w = in_w / conv->stride;
  size_t max_rows = ws->im2col_halfs / (size_t)conv->k_padded;
  size_t max_padded_rows = (max_rows / TILE_M) * TILE_M;
  int strip_h = (int)(max_padded_rows / (size_t)out_w);
  if (strip_h < 1) strip_h = 1;
  if (strip_h > out_h) strip_h = out_h;

  for (int y0 = 0; y0 < out_h; y0 += strip_h) {
    int sh = out_h - y0 < strip_h ? out_h - y0 : strip_h;
    uint32_t m = (uint32_t)(sh * out_w);
    uint32_t padded_m = (m + TILE_M - 1) / TILE_M * TILE_M;
    Im2ColParam ip = {
        (uint32_t)in_h,
        (uint32_t)in_w,
        (uint32_t)conv->cin,
        (uint32_t)out_w,
        (uint32_t)y0,
        (uint32_t)sh,
        (uint32_t)conv->ksize,
        (uint32_t)conv->stride,
        (uint32_t)conv->pad,
        (uint32_t)conv->pad,
        (uint32_t)conv->k_padded,
    };
    gpu_select_kernel(cmd, vae->k_im2col_f16);
    gpu_bind_buffer(cmd, 0, x, 0);
    gpu_bind_buffer(cmd, 1, &ws->im2col, 0);
    gpu_copy_bytes(cmd, 2, &ip, sizeof ip);
    gpu_run_kernel(cmd,
                   ((uint32_t)conv->k_padded + ELEM_THREADS - 1) / ELEM_THREADS,
                   padded_m, 1, ELEM_THREADS, 1, 1);

    F16MMParam mp = {m, (uint32_t)conv->cout, (uint32_t)conv->k_padded,
                     (uint32_t)conv->cout, 1};
    size_t y_off =
        (size_t)y0 * (size_t)out_w * (size_t)conv->cout * sizeof(float);
    gpu_select_kernel(cmd, vae->k_mm_f16);
    gpu_bind_buffer(cmd, 0, &conv->w, 0);
    gpu_bind_buffer(cmd, 1, &ws->im2col, 0);
    gpu_bind_buffer(cmd, 2, &conv->bias, 0);
    gpu_bind_buffer(cmd, 3, y, y_off);
    gpu_copy_bytes(cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(cmd, ((uint32_t)conv->cout + TILE_N - 1) / TILE_N,
                   padded_m / TILE_M, 1, MM_THREADS, 1, 1);
  }
}

// Encodes in-place elementwise addition dst[i] += src[i] over count floats.
static void run_enc_add(const MinimaxH3ImageVae *vae, GpuCommandBuffer *cmd,
                        const Buf *dst, const Buf *src, uint32_t count) {
  gpu_select_kernel(cmd, vae->k_add);
  gpu_bind_buffer(cmd, 0, dst, 0);
  gpu_bind_buffer(cmd, 1, src, 0);
  gpu_copy_bytes(cmd, 2, &count, sizeof count);
  gpu_run_kernel(cmd, (count + ADD_THREADS - 1) / ADD_THREADS, 1, 1,
                 ADD_THREADS, 1, 1);
}

// Runs the single-frame 3D causal CNN encoder + quant_conv on one RGB tile
// [tile_h, tile_w, 3] and writes channel-first FP32 moments [48, h_lat, w_lat].
static void encode_one_tile(MinimaxH3ImageVae *vae, const EncWorkspace *ws,
                            const uint8_t *rgb, int full_w, int y0, int x0,
                            int tile_h, int tile_w, float *out_moments) {
  // Normalize tile RGB pixels into NHWC FP32 in ws->b.
  float *in_px = (float *)ws->b.ptr;
  for (int y = 0; y < tile_h; y++) {
    const uint8_t *srow =
        rgb + ((size_t)(y0 + y) * (size_t)full_w + (size_t)x0) * 3u;
    float *drow = in_px + (size_t)y * (size_t)tile_w * 3u;
    for (int x = 0; x < tile_w; x++) {
      for (int c = 0; c < 3; c++) {
        float p = (float)srow[x * 3 + c] * (1.0f / 255.0f);
        drow[x * 3 + c] = (p - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
    }
  }

  const Buf *cur = &ws->a, *alt = &ws->b, *tmp = &ws->c;
  int h = tile_h, w = tile_w;
  GpuCommandBuffer cmd = gpu_begin(vae->gpu);
  run_enc_conv(vae, &cmd, ws, alt, &vae->enc_conv_in, cur, h, w);
  gpu_commit(&cmd);
  uint64_t prev_cmd = cmd.number;

  for (int lv = 0; lv < ENC_LEVELS; lv++) {
    const EncLevel *el = &vae->enc_levels[lv];
    cmd = gpu_begin(vae->gpu);
    for (int bi = 0; bi < ENC_BLOCKS_PER_LEVEL; bi++) {
      const EncResBlock *rb = &el->blocks[bi];
      run_enc_gn_silu(vae, &cmd, cur, &rb->norm1_w, &rb->norm1_b, tmp, h, w,
                      rb->conv1.cin);
      run_enc_conv(vae, &cmd, ws, tmp, &rb->conv1, alt, h, w);
      if (rb->has_shortcut) {
        run_enc_conv(vae, &cmd, ws, cur, &rb->shortcut, tmp, h, w);
        run_enc_gn_silu(vae, &cmd, alt, &rb->norm2_w, &rb->norm2_b, cur, h, w,
                        rb->conv2.cin);
        run_enc_conv(vae, &cmd, ws, cur, &rb->conv2, alt, h, w);
        run_enc_add(vae, &cmd, alt, tmp,
                    (uint32_t)((size_t)h * (size_t)w * (size_t)rb->conv2.cout));
      } else {
        run_enc_gn_silu(vae, &cmd, alt, &rb->norm2_w, &rb->norm2_b, tmp, h, w,
                        rb->conv2.cin);
        run_enc_conv(vae, &cmd, ws, tmp, &rb->conv2, alt, h, w);
        run_enc_add(vae, &cmd, alt, cur,
                    (uint32_t)((size_t)h * (size_t)w * (size_t)rb->conv2.cout));
      }
      const Buf *swap = cur;
      cur = alt;
      alt = swap;
    }
    if (el->has_downsample) {
      run_enc_conv(vae, &cmd, ws, cur, &el->downsample, alt, h, w);
      h /= 2;
      w /= 2;
      const Buf *swap = cur;
      cur = alt;
      alt = swap;
    }
    gpu_commit(&cmd);
    minimax_h3_gpu_wait(vae->gpu, prev_cmd);
    prev_cmd = cmd.number;
  }

  cmd = gpu_begin(vae->gpu);
  run_enc_gn_silu(vae, &cmd, cur, &vae->enc_norm_out_w, &vae->enc_norm_out_b,
                  tmp, h, w, ENC_CH[ENC_LEVELS - 1]);
  run_enc_conv(vae, &cmd, ws, tmp, &vae->enc_conv_out, alt, h, w);
  run_enc_conv(vae, &cmd, ws, alt, &vae->enc_quant_conv, cur, h, w);
  gpu_commit(&cmd);
  minimax_h3_gpu_wait(vae->gpu, prev_cmd);
  minimax_h3_gpu_wait(vae->gpu, cmd.number);

  // Transpose NHWC [h * w, 48] -> channel-first [48, h, w].
  const float *src = (const float *)cur->ptr;
  size_t plane = (size_t)h * (size_t)w;
  for (size_t p = 0; p < plane; p++) {
    const float *pix = src + p * (size_t)ENC_MOMENT_CH;
    for (int c = 0; c < ENC_MOMENT_CH; c++)
      out_moments[(size_t)c * plane + p] = pix[c];
  }
}

int minimax_h3_image_vae_encode(MinimaxH3ImageVae *vae, const uint8_t *rgb,
                                int width, int height, float *out_latents) {
  if (width <= 0 || height <= 0 || width % MINIMAX_H3_VAE_SPATIAL_RATIO != 0 ||
      height % MINIMAX_H3_VAE_SPATIAL_RATIO != 0 || !rgb || !out_latents)
    die("invalid VAE encode arguments");
  if (!minimax_h3_image_vae_load_encoder(vae)) return 0;

  MinimaxH3VaeTilePlan y_plan = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan x_plan = minimax_h3_image_vae_split_tiles(width);
  int max_tile_h = 0, max_tile_w = 0;
  for (int i = 0; i < y_plan.count; i++)
    if (y_plan.len[i] > max_tile_h) max_tile_h = y_plan.len[i];
  for (int j = 0; j < x_plan.count; j++)
    if (x_plan.len[j] > max_tile_w) max_tile_w = x_plan.len[j];

  MemorySlot *wslot = weight_group_slot(vae->enc_weights);
  memory_pin(wslot, 1);
  EncWorkspace ws;
  if (!alloc_enc_workspace(vae, max_tile_h, max_tile_w, &ws)) {
    memory_pin(wslot, 0);
    return 0;
  }

  // Encode all tiles into raw [48, th_lat, tw_lat] buffers so blending matches
  // tiled_encode's raw-neighbor blend order.
  int total_tiles = y_plan.count * x_plan.count;
  float **raw_tiles = xmalloc((size_t)total_tiles * sizeof *raw_tiles);
  for (int i = 0; i < y_plan.count; i++) {
    int th_lat = y_plan.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    for (int j = 0; j < x_plan.count; j++) {
      int tw_lat = x_plan.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      size_t floats = (size_t)ENC_MOMENT_CH * (size_t)th_lat * (size_t)tw_lat;
      float *buf = xmalloc(floats * sizeof(float));
      encode_one_tile(vae, &ws, rgb, width, y_plan.start[i], x_plan.start[j],
                      y_plan.len[i], x_plan.len[j], buf);
      raw_tiles[i * x_plan.count + j] = buf;
    }
  }

  free_enc_workspace(&ws);
  memory_pin(wslot, 0);
  memory_touch(wslot);

  // Blend tile overlaps along height then width, trim trailing overlaps, and
  // assemble full moments [48, h_lat, w_lat].
  int h_lat = height / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = width / MINIMAX_H3_VAE_SPATIAL_RATIO;
  size_t full_plane = (size_t)h_lat * (size_t)w_lat;
  float *moments = xmalloc((size_t)ENC_MOMENT_CH * full_plane * sizeof(float));
  size_t max_tile_floats = (size_t)ENC_MOMENT_CH *
                           (size_t)(max_tile_h / MINIMAX_H3_VAE_SPATIAL_RATIO) *
                           (size_t)(max_tile_w / MINIMAX_H3_VAE_SPATIAL_RATIO);
  float *blended = xmalloc(max_tile_floats * sizeof(float));

  int out_y = 0;
  for (int i = 0; i < y_plan.count; i++) {
    int th = y_plan.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int ov_top =
        i > 0 ? y_plan.overlap[i - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
    int ov_bot = i < y_plan.count - 1
                     ? y_plan.overlap[i] / MINIMAX_H3_VAE_SPATIAL_RATIO
                     : 0;
    int keep_h = th - ov_bot;
    int out_x = 0;
    for (int j = 0; j < x_plan.count; j++) {
      int tw = x_plan.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int ov_left =
          j > 0 ? x_plan.overlap[j - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
      int ov_right = j < x_plan.count - 1
                         ? x_plan.overlap[j] / MINIMAX_H3_VAE_SPATIAL_RATIO
                         : 0;
      int keep_w = tw - ov_right;
      size_t tplane = (size_t)th * (size_t)tw;
      memcpy(blended, raw_tiles[i * x_plan.count + j],
             (size_t)ENC_MOMENT_CH * tplane * sizeof(float));

      if (ov_top > 0) {
        const float *above = raw_tiles[(i - 1) * x_plan.count + j];
        int ah = y_plan.len[i - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO;
        size_t aplane = (size_t)ah * (size_t)tw;
        for (int c = 0; c < ENC_MOMENT_CH; c++) {
          float *tp = blended + (size_t)c * tplane;
          const float *ap = above + (size_t)c * aplane;
          for (int y = 0; y < ov_top; y++) {
            float wb = (float)y / (float)ov_top, wa = 1.0f - wb;
            float *trow = tp + (size_t)y * (size_t)tw;
            const float *arow = ap + (size_t)(ah - ov_top + y) * (size_t)tw;
            for (int x = 0; x < tw; x++) trow[x] = arow[x] * wa + trow[x] * wb;
          }
        }
      }
      if (ov_left > 0) {
        const float *left = raw_tiles[i * x_plan.count + (j - 1)];
        int lw = x_plan.len[j - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO;
        size_t lplane = (size_t)th * (size_t)lw;
        for (int c = 0; c < ENC_MOMENT_CH; c++) {
          float *tp = blended + (size_t)c * tplane;
          const float *lp = left + (size_t)c * lplane;
          for (int y = 0; y < th; y++) {
            float *trow = tp + (size_t)y * (size_t)tw;
            const float *lrow = lp + (size_t)y * (size_t)lw + (lw - ov_left);
            for (int x = 0; x < ov_left; x++) {
              float wb = (float)x / (float)ov_left, wa = 1.0f - wb;
              trow[x] = lrow[x] * wa + trow[x] * wb;
            }
          }
        }
      }

      for (int c = 0; c < ENC_MOMENT_CH; c++) {
        const float *tp = blended + (size_t)c * tplane;
        float *mp = moments + (size_t)c * full_plane;
        for (int y = 0; y < keep_h; y++)
          memcpy(mp + (size_t)(out_y + y) * (size_t)w_lat + (size_t)out_x,
                 tp + (size_t)y * (size_t)tw, (size_t)keep_w * sizeof(float));
      }
      out_x += keep_w;
    }
    out_y += keep_h;
  }

  for (int t = 0; t < total_tiles; t++) free(raw_tiles[t]);
  free(raw_tiles);
  free(blended);

  // Sample posterior with fixed seed 42 and normalize to unit latent scale.
  size_t lat_floats = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * full_plane;
  float *noise = xmalloc(lat_floats * sizeof(float));
  enc_fill_gaussian(noise, lat_floats, ENC_POSTERIOR_SEED);
  for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
    const float *mean_c = moments + (size_t)c * full_plane;
    const float *lv_c =
        moments + (size_t)(c + MINIMAX_H3_VAE_LATENT_CHANNELS) * full_plane;
    const float *n_c = noise + (size_t)c * full_plane;
    float *out_c = out_latents + (size_t)c * full_plane;
    float lm = LATENTS_MEAN[c], inv_ls = 1.0f / LATENTS_STD[c];
    for (size_t p = 0; p < full_plane; p++) {
      float lv = lv_c[p];
      if (lv < ENC_LOGVAR_MIN) lv = ENC_LOGVAR_MIN;
      if (lv > ENC_LOGVAR_MAX) lv = ENC_LOGVAR_MAX;
      float lat = mean_c[p] + expf(0.5f * lv) * n_c[p];
      out_c[p] = (lat - lm) * inv_ls;
    }
  }
  free(noise);
  free(moments);
  return 1;
}
