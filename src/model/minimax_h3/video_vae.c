#include "model/minimax_h3/video_vae.h"
#include "base/error.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 3D ViT decoder constants from Comfy-Org/ComfyUI comfy/ldm/minimax/vae.py:
// num_layers 36, heads 32, dim_head 64, patch_size_t 4, patch_size 16,
// rope_dim_ratio 0.75, rope_theta 100.0, num_register_tokens 4.
static constexpr int VVAE_LAYERS = 36, VVAE_DIM = 2048, VVAE_HEADS = 32,
                     VVAE_HEAD_DIM = 64, VVAE_FFN_INNER = 8192,
                     VVAE_QKV_WIDTH = MINIMAX_H3_QKV_PARTS * VVAE_DIM,
                     VVAE_W1_WIDTH = 2 * VVAE_FFN_INNER,
                     VVAE_PATCH_OUT = MINIMAX_H3_RGB_CHANNELS *
                                      MINIMAX_H3_VVAE_TEMPORAL_RATIO *
                                      MINIMAX_H3_VAE_SPATIAL_RATIO *
                                      MINIMAX_H3_VAE_SPATIAL_RATIO,
                     VVAE_REGISTERS = 4,
                     VVAE_SUFFIX_TOKENS = VVAE_REGISTERS + 1,
                     VVAE_ROPE_FREQS = 8,
                     VVAE_MAX_CHUNK_LATENTS = MINIMAX_H3_VVAE_CHUNK_TOKENS +
                                              MINIMAX_H3_VVAE_TOKEN_OVERLAP,
                     VVAE_TILE_SIZE = 256,
                     VVAE_TILE_CELLS =
                         VVAE_TILE_SIZE / MINIMAX_H3_VAE_SPATIAL_RATIO,
                     VVAE_SPLIT_COUNT = 2,
                     VVAE_CHUNK_DEC_FRAMES = MINIMAX_H3_VVAE_CHUNK_TOKENS *
                                             MINIMAX_H3_VVAE_TEMPORAL_RATIO;

// Normalization epsilon and RoPE theta (comfy/ldm/minimax/vae.py:421-422).
static constexpr float VVAE_NORM_EPS = 1e-5f, VVAE_ENC_GN_EPS = 1e-6f;
static constexpr double VVAE_ROPE_THETA = 100.0;

// 3D causal CNN encoder constants (comfy/ldm/minimax/vae.py:611-620).
static constexpr int ENC_LEVELS = 6, ENC_BLOCKS = 2, ENC_GN_GROUPS = 32,
                     ENC_MOMENT_CH = 2 * MINIMAX_H3_VAE_LATENT_CHANNELS,
                     ENC_BASE_CH = 128, ENC_CONV_RANK = 5, ENC_CONV_K1 = 1,
                     ENC_CONV_K3 = 3, ENC_FRONT_PAD_K3 = 2, ENC_ACT_BUFS = 3;

// Encoder channel widths, spatial strides and temporal strides per level
// (comfy/ldm/minimax/vae.py:617-620).
static const int ENC_CH[ENC_LEVELS] = {128, 256, 256, 512, 512, 1024};
static const int ENC_SPACE_DOWN[ENC_LEVELS] = {2, 2, 2, 2, 1, 1};
static const int ENC_TIME_DOWN[ENC_LEVELS] = {1, 2, 2, 1, 1, 1};

// FP16 3D im2col staging capacity in halfs (256 rows x 27648 K = 13.5 MiB).
static constexpr size_t ENC_IM2COL_MAX_HALFS = 256u * 27648u;

// ImageNet RGB mean and std (comfy/ldm/minimax/vae.py:59-60).
static const float IMAGENET_MEAN[MINIMAX_H3_RGB_CHANNELS] = {0.485f, 0.456f,
                                                             0.406f};
static const float IMAGENET_STD[MINIMAX_H3_RGB_CHANNELS] = {0.229f, 0.224f,
                                                            0.225f};

// Metal dispatch tile and threadgroup dimensions matching
// src/kernels/minimax_h3_{quant,vae,video_vae}.metal.
static constexpr uint32_t TILE_M = 64, TILE_M_I8 = 128, TILE_N = 64,
                          MM_THREADS = 128, CONVROT_THREADS = 256,
                          NORM_THREADS = 256, EMBED_THREADS = 256,
                          HEADS_THREADS = 32, ATTN_GROUP_ROWS = 64,
                          ATTN_THREADS = 128, ELEM_THREADS = 32,
                          ADD_THREADS = 256, IM2COL_TX = 32, IM2COL_TY = 8;

// Parameters passed to minimax_h3_rmsnorm_convrot.
typedef struct {
  // Active row count and row width in floats.
  uint32_t m, k;
  // Normalization epsilon.
  float eps;
} RmsNormConvRotParam;

// Parameters passed to minimax_h3_layernorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} LayerNormParam;

// Parameters passed to minimax_h3_convrot_quant_i8.
typedef struct {
  // Active row count, inner dimension and input row stride.
  uint32_t m, k, lda;
} ConvRotParam;

// Parameters passed to minimax_h3_mm_int8.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} Int8MMParam;

// Parameters passed to minimax_h3_x_f16.
typedef struct {
  // Active rows, inner dimension and input row stride.
  uint32_t m, k, lda;
} F16XParam;

// Parameters passed to minimax_h3_mm_f16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} F16MMParam;

// Parameters passed to minimax_h3_add_scaled.
typedef struct {
  // Row count and row width.
  uint32_t rows, dim;
} AddScaledParam;

// Parameters passed to minimax_h3_swiglu_convrot.
typedef struct {
  // Active row count and half-width inner dimension.
  uint32_t m, k;
} SwigluConvRotParam;

// Parameters passed to minimax_h3_vvae_embed.
typedef struct {
  // Latent tile dimensions (t, h, w).
  uint32_t t_lat, h_lat, w_lat;
} VvaeEmbedParam;

// Parameters passed to minimax_h3_vvae_heads.
typedef struct {
  // Sequence length, padded head stride and latent tile dimensions.
  uint32_t rows, padded, t_lat, h_lat, w_lat;
  // Per-head non-affine QK RMSNorm epsilon.
  float eps;
} VvaeHeadsParam;

// Parameters passed to minimax_h3_vvae_attend.
typedef struct {
  // Sequence length and padded head stride.
  uint32_t rows, padded;
} VvaeAttendParam;

// Parameters passed to minimax_h3_vvae_gn_silu_3d.
typedef struct {
  // Pixels per frame, channels and group count.
  uint32_t hw, channels, groups;
  // GroupNorm epsilon.
  float eps;
} VvaeGn3dParam;

// Parameters passed to minimax_h3_vvae_im2col_3d_f16.
typedef struct {
  // Input shape, channels, output width, frame index and strip geometry.
  uint32_t in_t, in_h, in_w, cin, out_w, ot, out_y0, strip_h;
  // Kernel sizes, strides, causal front pad, spatial pads and padded K width.
  uint32_t kt, ks, time_stride, space_stride, front_pad, pad_h, pad_w, k_padded;
} VvaeIm2Col3dParam;

// Parameter block for repacking a 5D FP16 Conv3d weight into 2D FP16 im2col
// matrix [cout_padded, k_padded] in (kt, ky, kx, ci) tap order.
typedef struct {
  // Source 5D F16 weight tensor in the checkpoint.
  Tensor tensor;
  // Output/input channel counts, kernel sizes and 64-aligned dimensions.
  int cout, cin, kt, ks, cout_padded, k_padded;
} EncConv3dParam;

// One 3D causal CNN encoder convolution weight and bias.
typedef struct {
  // Repacked FP16 matrix [cout_padded, k_padded] and FP32 bias [cout].
  Buf w, bias;
  // Channel counts, kernel sizes, strides, causal front pad, spatial pads and
  // padded K width.
  int cin, cout, kt, ks, time_stride, space_stride, front_pad, pad_h, pad_w,
      k_padded;
} EncConv3d;

// One 3D ResnetBlock in the causal CNN encoder.
typedef struct {
  // Pre-conv1 and pre-conv2 temporal-isolated GroupNorm weights and biases.
  Buf norm1_w, norm1_b, norm2_w, norm2_b;
  // Two 3x3x3 causal convolutions and optional 1x1x1 channel shortcut.
  EncConv3d conv1, conv2, shortcut;
  // Nonzero when cin != cout and shortcut is present.
  int has_shortcut;
} EncResBlock3d;

// One downsampling level of the 3D causal CNN encoder.
typedef struct {
  // Two 3D residual blocks.
  EncResBlock3d blocks[ENC_BLOCKS];
  // Optional spatial/temporal downsampling convolution.
  EncConv3d downsample;
  // Nonzero when this level includes a downsampling convolution.
  int has_downsample;
} EncLevel3d;

// Transient GPU buffers allocated for encoding one 3D clip tile.
typedef struct {
  // Three ping-pong NDHWC FP32 activation buffers and one FP16 im2col buffer.
  Buf a, b, c, im2col;
} VvaeEncodeWorkspace;

// INT8-ConvRot linear layer (rotated INT8 weight, per-row F32 scale, F32 bias).
typedef struct {
  // Rotated INT8 matrix [n, k], per-row scale [n] and bias [n].
  Buf w, ws, bias;
  // Output and input widths.
  int n, k;
} VvaeInt8Lin;

// One 3D ViT decoder transformer block.
typedef struct {
  // Pre-attention and pre-FFN RMSNorm weights and residual scales [2048].
  Buf norm1, norm2, scale1, scale2;
  // INT8-ConvRot QKV, output and gated FFN linear layers.
  VvaeInt8Lin to_qkv, to_out, ff_w1, ff_w2;
} VvaeBlock;

// Parameter block for converting an FP32 matrix tensor to FP16 on load.
typedef struct {
  // Source FP32 tensor in the checkpoint.
  Tensor tensor;
  // Element count of the matrix.
  size_t elems;
} F32ToF16Param;

// Transient GPU buffers allocated for decoding one 7-latent 256x256 tile.
typedef struct {
  // Input latent tile, residual stream, normalized stream and INT8 rotated x.
  Buf z_tile, h, norm, xq, x_scale;
  // Shared wide buffer (qkv, ff_up, proj_out) and medium buffer (attn_out,
  // ff_mid).
  Buf wide, mid;
  // Head-major FP16 Q, K, V and FP16 activations for proj_out.
  Buf qh, kh, vh, xh;
} VvaeDecodeWorkspace;

// Loaded MiniMax-H3 Video VAE handle.
struct MinimaxH3VideoVae {
  // GPU handle.
  Gpu *gpu;
  // Memory account owning weight and workspace slots.
  MemoryAccount *account;
  // Automatic workspace memory slot for transient GPU buffers.
  MemorySlot workspace_slot;
  // Open safetensors index.
  TensorIndex *tensors;
  // Decoder and encoder weight groups.
  WeightGroup *dec_weights, *enc_weights;

  // Per-channel latent mean and std read from checkpoint tensors [24].
  float latents_mean[MINIMAX_H3_VAE_LATENT_CHANNELS];
  // Per-channel latent standard deviation [24].
  float latents_std[MINIMAX_H3_VAE_LATENT_CHANNELS];

  // Decoder stem, registers, RoPE inverse frequencies, 36 blocks and head.
  Buf pqc_w, pqc_b, emb_w, emb_b, reg_tokens, inv_freq;
  // Thirty-six INT8-ConvRot 3D ViT decoder blocks.
  VvaeBlock blocks[VVAE_LAYERS];
  // Final LayerNorm weights and FP16-converted patch output head.
  Buf norm_out_w, norm_out_b, proj_out_w, proj_out_b;

  // 3D causal CNN encoder stem, 6 levels, output norm/conv and quant_conv.
  EncConv3d enc_conv_in, enc_conv_out, enc_quant_conv;
  // Six encoder resolution levels.
  EncLevel3d enc_levels[ENC_LEVELS];
  // Encoder output temporal-isolated GroupNorm weights and biases.
  Buf enc_norm_out_w, enc_norm_out_b;

  // Cached Metal kernel indices.
  int k_rmsnorm_convrot, k_layernorm, k_convrot, k_mm_i8, k_x_f16, k_mm_f16,
      k_add_scaled, k_swiglu_convrot, k_embed, k_heads, k_attend, k_gn3d,
      k_im2col3d, k_add;
};

// Returns ceil(a / b) for positive integers.
static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Generates the 8 FP32 per-axis inverse frequencies 100^(-k / 8) for 3D RoPE.
static void fill_vvae_inv_freq(Buf *buffer, const void *context) {
  (void)context;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < VVAE_ROPE_FREQS; i++)
    out[i] = (float)(1.0 /
                     pow(VVAE_ROPE_THETA, (double)i / (double)VVAE_ROPE_FREQS));
}

// Converts an FP32 checkpoint matrix tensor to FP16 in buffer on load.
static void fill_f32_to_f16(Buf *buffer, const void *context) {
  const F32ToF16Param *p = context;
  float *raw = xmalloc(p->elems * sizeof(float));
  minimax_h3_read_tensor_bytes(p->tensor, raw);
  _Float16 *dst = (_Float16 *)buffer->ptr;
  for (size_t i = 0; i < p->elems; i++) dst[i] = (_Float16)raw[i];
  free(raw);
}

// Registers one INT8-ConvRot linear layer (weight, weight_scale, bias).
static void add_i8_lin(WeightGroup *group, const TensorIndex *tensors,
                       VvaeInt8Lin *lin, int n, int k, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor ws = tensor_index_get(tensors, "%s.weight_scale", prefix);
  Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
  if (w.dtype != 'i' || w.ndim != 2 || w.shape[0] != n || w.shape[1] != k)
    die("unexpected video VAE INT8 weight shape for %s", prefix);
  if (ws.dtype != 'f' || ws.size != (uint64_t)n * sizeof(float))
    die("unexpected video VAE INT8 scale shape for %s", prefix);
  if (b.dtype != 'f' || b.ndim != 1 || b.shape[0] != n)
    die("unexpected video VAE bias shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  weight_group_add_tensor(group, &lin->w, w, 1);
  weight_group_add_tensor(group, &lin->ws, ws, 0);
  weight_group_add_tensor(group, &lin->bias, b, 0);
}

// Repacks a 5D F16 Conv3d weight [cout, cin, kt, ks, ks] into padded 2D FP16
// [cout_padded, k_padded] in (kt, ky, kx, ci) tap order.
static void fill_enc_conv3d_weight(Buf *buffer, const void *context) {
  const EncConv3dParam *p = context;
  uint16_t *raw = xmalloc(p->tensor.size);
  minimax_h3_read_tensor_bytes(p->tensor, raw);
  size_t out_bytes =
      (size_t)p->cout_padded * (size_t)p->k_padded * sizeof(uint16_t);
  memset(buffer->ptr, 0, out_bytes);
  uint16_t *dst = (uint16_t *)buffer->ptr;
  for (int co = 0; co < p->cout; co++) {
    uint16_t *drow = dst + (size_t)co * (size_t)p->k_padded;
    for (int ci = 0; ci < p->cin; ci++) {
      for (int tk = 0; tk < p->kt; tk++) {
        for (int ky = 0; ky < p->ks; ky++) {
          for (int kx = 0; kx < p->ks; kx++) {
            size_t sidx =
                (((((size_t)co * (size_t)p->cin + (size_t)ci) * (size_t)p->kt +
                   (size_t)tk) *
                      (size_t)p->ks +
                  (size_t)ky) *
                     (size_t)p->ks +
                 (size_t)kx);
            size_t tap =
                ((size_t)tk * (size_t)p->ks + (size_t)ky) * (size_t)p->ks +
                (size_t)kx;
            drow[tap * (size_t)p->cin + (size_t)ci] = raw[sidx];
          }
        }
      }
    }
  }
  free(raw);
}

// Registers one 5D F16 causal Conv3d weight and FP32 bias into group.
static void add_enc_conv3d(WeightGroup *group, const TensorIndex *tensors,
                           EncConv3d *conv, int cin, int cout, int kt, int ks,
                           int time_stride, int space_stride, int front_pad,
                           int pad_h, int pad_w, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
  if (w.dtype != 'h' || w.ndim != ENC_CONV_RANK || w.shape[0] != cout ||
      w.shape[1] != cin || w.shape[2] != kt || w.shape[3] != ks ||
      w.shape[4] != ks)
    die("unexpected video VAE encoder conv3d weight shape for %s", prefix);
  if (b.ndim != 1 || b.shape[0] != cout)
    die("unexpected video VAE encoder conv3d bias shape for %s", prefix);
  int cout_padded = (int)ceil_div((uint32_t)cout, TILE_N) * (int)TILE_N;
  int k_raw = kt * ks * ks * cin;
  int k_padded = (int)ceil_div((uint32_t)k_raw, TILE_N) * (int)TILE_N;
  conv->cin = cin;
  conv->cout = cout;
  conv->kt = kt;
  conv->ks = ks;
  conv->time_stride = time_stride;
  conv->space_stride = space_stride;
  conv->front_pad = front_pad;
  conv->pad_h = pad_h;
  conv->pad_w = pad_w;
  conv->k_padded = k_padded;
  EncConv3dParam param = {
      .tensor = w,
      .cout = cout,
      .cin = cin,
      .kt = kt,
      .ks = ks,
      .cout_padded = cout_padded,
      .k_padded = k_padded,
  };
  weight_group_add_generated(group, &conv->w,
                             (size_t)cout_padded * (size_t)k_padded *
                                 sizeof(uint16_t),
                             fill_enc_conv3d_weight, &param, sizeof param);
  weight_group_add_vector(group, &conv->bias, b, 0.0f, 0);
}

// Registers all 3D causal CNN encoder weights into vae->enc_weights.
static void register_encoder_weights(MinimaxH3VideoVae *vae) {
  add_enc_conv3d(vae->enc_weights, vae->tensors, &vae->enc_conv_in,
                 MINIMAX_H3_RGB_CHANNELS, ENC_BASE_CH, ENC_CONV_K3, ENC_CONV_K3,
                 1, 1, ENC_FRONT_PAD_K3, 1, 1, "encoder.conv_in");

  char name[MINIMAX_H3_NAME_CAP];
  for (int lv = 0; lv < ENC_LEVELS; lv++) {
    int in_ch = lv == 0 ? ENC_BASE_CH : ENC_CH[lv - 1];
    int out_ch = ENC_CH[lv];
    EncLevel3d *level = &vae->enc_levels[lv];
    for (int bi = 0; bi < ENC_BLOCKS; bi++) {
      int bc_in = bi == 0 ? in_ch : out_ch;
      EncResBlock3d *rb = &level->blocks[bi];
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
      snprintf(name, sizeof name, "encoder.down.%d.block.%d.conv1", lv, bi);
      add_enc_conv3d(vae->enc_weights, vae->tensors, &rb->conv1, bc_in, out_ch,
                     ENC_CONV_K3, ENC_CONV_K3, 1, 1, ENC_FRONT_PAD_K3, 1, 1,
                     name);
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
      snprintf(name, sizeof name, "encoder.down.%d.block.%d.conv2", lv, bi);
      add_enc_conv3d(vae->enc_weights, vae->tensors, &rb->conv2, out_ch, out_ch,
                     ENC_CONV_K3, ENC_CONV_K3, 1, 1, ENC_FRONT_PAD_K3, 1, 1,
                     name);
      if (bc_in != out_ch) {
        rb->has_shortcut = 1;
        snprintf(name, sizeof name, "encoder.down.%d.block.%d.nin_shortcut", lv,
                 bi);
        add_enc_conv3d(vae->enc_weights, vae->tensors, &rb->shortcut, bc_in,
                       out_ch, ENC_CONV_K1, ENC_CONV_K1, 1, 1, 0, 0, 0, name);
      }
    }
    if (ENC_SPACE_DOWN[lv] * ENC_TIME_DOWN[lv] > 1) {
      level->has_downsample = 1;
      int pad_s = ENC_SPACE_DOWN[lv] == 2 ? 0 : 1;
      snprintf(name, sizeof name, "encoder.down.%d.downsample.conv", lv);
      add_enc_conv3d(vae->enc_weights, vae->tensors, &level->downsample, out_ch,
                     out_ch, ENC_CONV_K3, ENC_CONV_K3, ENC_TIME_DOWN[lv],
                     ENC_SPACE_DOWN[lv], ENC_FRONT_PAD_K3, pad_s, pad_s, name);
    }
  }
  weight_group_add_vector(
      vae->enc_weights, &vae->enc_norm_out_w,
      tensor_index_get(vae->tensors, "encoder.norm_out.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->enc_weights, &vae->enc_norm_out_b,
      tensor_index_get(vae->tensors, "encoder.norm_out.bias"), 0.0f, 0);
  add_enc_conv3d(vae->enc_weights, vae->tensors, &vae->enc_conv_out,
                 ENC_CH[ENC_LEVELS - 1], ENC_MOMENT_CH, ENC_CONV_K3,
                 ENC_CONV_K3, 1, 1, ENC_FRONT_PAD_K3, 1, 1, "encoder.conv_out");
  add_enc_conv3d(vae->enc_weights, vae->tensors, &vae->enc_quant_conv,
                 ENC_MOMENT_CH, ENC_MOMENT_CH, ENC_CONV_K1, ENC_CONV_K1, 1, 1,
                 0, 0, 0, "quant_conv");
}

MinimaxH3VvaeTemporalPlan minimax_h3_video_vae_decode_plan(int z_len) {
  if (z_len <= 0) die("video VAE latent length %d must be positive", z_len);
  if (z_len == 1)
    return (MinimaxH3VvaeTemporalPlan){
        .pad_tokens = 0, .num_chunks = 1, .out_frames = 1};

  // Exact match to comfy/ldm/minimax/vae.py:876-924.
  int pseudo = z_len + MINIMAX_H3_VVAE_TOKEN_DROP;
  int rem = pseudo % MINIMAX_H3_VVAE_CHUNK_TOKENS;
  int pad_tokens = rem == 0 ? 0 : (MINIMAX_H3_VVAE_CHUNK_TOKENS - rem);
  pseudo += pad_tokens;
  int num_chunks = pseudo / MINIMAX_H3_VVAE_CHUNK_TOKENS - 1;
  if (num_chunks < 1) {
    pad_tokens += MINIMAX_H3_VVAE_CHUNK_TOKENS;
    num_chunks += 1;
  }

  int z_padded = z_len + pad_tokens;
  int total_frames = 0, final_overlap = 0;
  for (int i = 0; i < num_chunks; i++) {
    int t_start = i * MINIMAX_H3_VVAE_CHUNK_TOKENS;
    int t_end = t_start + VVAE_MAX_CHUNK_LATENTS;
    int clip_end = t_end < z_padded ? t_end : z_padded;
    int clip_start = t_start < z_padded ? t_start : z_padded;
    int clip_tokens = clip_end > clip_start ? (clip_end - clip_start) : 0;
    int clip_frames = clip_tokens * MINIMAX_H3_VVAE_TEMPORAL_RATIO;
    for (int j = 0; j < VVAE_SPLIT_COUNT; j++) {
      int f_start = j * VVAE_CHUNK_DEC_FRAMES;
      int f_end = f_start + VVAE_CHUNK_DEC_FRAMES;
      if (f_end > clip_frames) f_end = clip_frames;
      int chunk_f = f_end - f_start - MINIMAX_H3_VVAE_FRAME_PRE_PAD;
      if (chunk_f < 0) chunk_f = 0;
      if (j == 0)
        total_frames += chunk_f;
      else
        final_overlap = chunk_f;
    }
  }
  total_frames += final_overlap;

  int intra_tail = MINIMAX_H3_VVAE_CLIP_LENGTH % MINIMAX_H3_VVAE_TEMPORAL_RATIO;
  int pad_frames = 0;
  for (int k = 0; k < pad_tokens; k++) {
    int is_boundary = ((z_len + k) % MINIMAX_H3_VVAE_CHUNK_TOKENS) == 0;
    pad_frames += is_boundary ? intra_tail : MINIMAX_H3_VVAE_TEMPORAL_RATIO;
  }
  return (MinimaxH3VvaeTemporalPlan){
      .pad_tokens = pad_tokens,
      .num_chunks = num_chunks,
      .out_frames = total_frames - pad_frames,
  };
}

int minimax_h3_video_vae_encode_latent_t(int frame_count) {
  if (frame_count <= 0)
    die("video VAE frame count %d must be positive", frame_count);
  if (frame_count == 1) return 1;
  int clips = (frame_count + MINIMAX_H3_VVAE_CLIP_LENGTH - 1) /
              MINIMAX_H3_VVAE_CLIP_LENGTH;
  return clips * MINIMAX_H3_VVAE_CHUNK_TOKENS - MINIMAX_H3_VVAE_TOKEN_DROP;
}

MinimaxH3VideoVae *minimax_h3_video_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path) {
  MinimaxH3VideoVae *vae = xmalloc(sizeof *vae);
  memset(vae, 0, sizeof *vae);
  vae->gpu = gpu;
  vae->account = account;
  memory_slot_init(&vae->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-video-vae-workspace", NULL, NULL);
  vae->workspace_slot.automatic = 1;
  vae->tensors = tensor_index_open_file(weights_path);
  vae->dec_weights =
      weight_group_new(gpu, account, "minimax-h3-video-vae-decoder");
  vae->enc_weights =
      weight_group_new(gpu, account, "minimax-h3-video-vae-encoder");

  Tensor lm = tensor_index_get(vae->tensors, "latents_mean");
  Tensor ls = tensor_index_get(vae->tensors, "latents_std");
  if (lm.dtype != 'f' || lm.size != sizeof vae->latents_mean ||
      ls.dtype != 'f' || ls.size != sizeof vae->latents_std)
    die("unexpected video VAE latents_mean/latents_std tensors");
  minimax_h3_read_tensor_bytes(lm, vae->latents_mean);
  minimax_h3_read_tensor_bytes(ls, vae->latents_std);

  vae->k_rmsnorm_convrot = gpu_find_kernel(gpu, "minimax_h3_rmsnorm_convrot");
  vae->k_layernorm = gpu_find_kernel(gpu, "minimax_h3_layernorm");
  vae->k_convrot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  vae->k_mm_i8 = gpu_find_kernel(gpu, "minimax_h3_mm_int8");
  vae->k_x_f16 = gpu_find_kernel(gpu, "minimax_h3_x_f16");
  vae->k_mm_f16 = gpu_find_kernel(gpu, "minimax_h3_mm_f16");
  vae->k_add_scaled = gpu_find_kernel(gpu, "minimax_h3_add_scaled");
  vae->k_swiglu_convrot = gpu_find_kernel(gpu, "minimax_h3_swiglu_convrot");
  vae->k_embed = gpu_find_kernel(gpu, "minimax_h3_vvae_embed");
  vae->k_heads = gpu_find_kernel(gpu, "minimax_h3_vvae_heads");
  vae->k_attend = gpu_find_kernel(gpu, "minimax_h3_vvae_attend");
  vae->k_gn3d = gpu_find_kernel(gpu, "minimax_h3_vvae_gn_silu_3d");
  vae->k_im2col3d = gpu_find_kernel(gpu, "minimax_h3_vvae_im2col_3d_f16");
  vae->k_add = gpu_find_kernel(gpu, "minimax_h3_add");

  weight_group_add_vector(
      vae->dec_weights, &vae->pqc_w,
      tensor_index_get(vae->tensors, "post_quant_conv.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->dec_weights, &vae->pqc_b,
      tensor_index_get(vae->tensors, "post_quant_conv.bias"), 0.0f, 0);
  weight_group_add_vector(
      vae->dec_weights, &vae->emb_w,
      tensor_index_get(vae->tensors, "decoder.x_embedder.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->dec_weights, &vae->emb_b,
      tensor_index_get(vae->tensors, "decoder.x_embedder.bias"), 0.0f, 0);
  weight_group_add_vector(
      vae->dec_weights, &vae->reg_tokens,
      tensor_index_get(vae->tensors, "decoder.register_tokens"), 0.0f, 0);
  weight_group_add_generated(vae->dec_weights, &vae->inv_freq,
                             (size_t)VVAE_ROPE_FREQS * sizeof(float),
                             fill_vvae_inv_freq, NULL, 0);

  char name[MINIMAX_H3_NAME_CAP];
  for (int i = 0; i < VVAE_LAYERS; i++) {
    VvaeBlock *b = &vae->blocks[i];
    weight_group_add_vector(
        vae->dec_weights, &b->norm1,
        tensor_index_get(vae->tensors,
                         "decoder.transformer_blocks.%d.norm1.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vae->dec_weights, &b->norm2,
        tensor_index_get(vae->tensors,
                         "decoder.transformer_blocks.%d.norm2.weight", i),
        0.0f, 0);
    weight_group_add_vector(
        vae->dec_weights, &b->scale1,
        tensor_index_get(vae->tensors, "decoder.transformer_blocks.%d.scale1",
                         i),
        0.0f, 0);
    weight_group_add_vector(
        vae->dec_weights, &b->scale2,
        tensor_index_get(vae->tensors, "decoder.transformer_blocks.%d.scale2",
                         i),
        0.0f, 0);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.attn.to_qkv", i);
    add_i8_lin(vae->dec_weights, vae->tensors, &b->to_qkv, VVAE_QKV_WIDTH,
               VVAE_DIM, name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.attn.to_out", i);
    add_i8_lin(vae->dec_weights, vae->tensors, &b->to_out, VVAE_DIM, VVAE_DIM,
               name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.ff.w1", i);
    add_i8_lin(vae->dec_weights, vae->tensors, &b->ff_w1, VVAE_W1_WIDTH,
               VVAE_DIM, name);
    snprintf(name, sizeof name, "decoder.transformer_blocks.%d.ff.w2", i);
    add_i8_lin(vae->dec_weights, vae->tensors, &b->ff_w2, VVAE_DIM,
               VVAE_FFN_INNER, name);
  }

  weight_group_add_vector(
      vae->dec_weights, &vae->norm_out_w,
      tensor_index_get(vae->tensors, "decoder.norm_out.weight"), 0.0f, 0);
  weight_group_add_vector(
      vae->dec_weights, &vae->norm_out_b,
      tensor_index_get(vae->tensors, "decoder.norm_out.bias"), 0.0f, 0);
  Tensor po_w = tensor_index_get(vae->tensors, "decoder.proj_out.weight");
  if (po_w.dtype != 'f' || po_w.ndim != 2 || po_w.shape[0] != VVAE_PATCH_OUT ||
      po_w.shape[1] != VVAE_DIM)
    die("unexpected decoder.proj_out.weight shape");
  F32ToF16Param po_param = {.tensor = po_w,
                            .elems = (size_t)VVAE_PATCH_OUT * (size_t)VVAE_DIM};
  weight_group_add_generated(vae->dec_weights, &vae->proj_out_w,
                             po_param.elems * sizeof(uint16_t), fill_f32_to_f16,
                             &po_param, sizeof po_param);
  weight_group_add_vector(
      vae->dec_weights, &vae->proj_out_b,
      tensor_index_get(vae->tensors, "decoder.proj_out.bias"), 0.0f, 0);
  register_encoder_weights(vae);
  return vae;
}

void minimax_h3_video_vae_close(MinimaxH3VideoVae *vae) {
  if (!vae) return;
  weight_group_free(vae->enc_weights);
  weight_group_free(vae->dec_weights);
  memory_slot_remove(&vae->workspace_slot);
  tensor_index_close(vae->tensors);
  free(vae);
}

WeightGroup *minimax_h3_video_vae_weights(MinimaxH3VideoVae *vae) {
  return vae->dec_weights;
}

uint64_t minimax_h3_video_vae_required_bytes(const MinimaxH3VideoVae *vae) {
  return weight_group_required_bytes(vae->dec_weights);
}

uint64_t minimax_h3_video_vae_resident_bytes(const MinimaxH3VideoVae *vae) {
  return weight_group_resident_bytes(vae->dec_weights);
}

int minimax_h3_video_vae_load(MinimaxH3VideoVae *vae) {
  return weight_group_load(vae->dec_weights);
}

WeightGroup *minimax_h3_video_vae_encoder_weights(MinimaxH3VideoVae *vae) {
  return vae->enc_weights;
}

uint64_t
minimax_h3_video_vae_encoder_required_bytes(const MinimaxH3VideoVae *vae) {
  return weight_group_required_bytes(vae->enc_weights);
}

uint64_t
minimax_h3_video_vae_encoder_resident_bytes(const MinimaxH3VideoVae *vae) {
  return weight_group_resident_bytes(vae->enc_weights);
}

int minimax_h3_video_vae_load_encoder(MinimaxH3VideoVae *vae) {
  return weight_group_load(vae->enc_weights);
}

uint64_t minimax_h3_video_vae_encode_workspace_bytes(int frame_count, int width,
                                                     int height) {
  if (frame_count <= 0 || width <= 0 || height <= 0) return 0;
  MinimaxH3VaeTilePlan yp = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan xp = minimax_h3_image_vae_split_tiles(width);
  int clip_t = frame_count == 1 ? 1 : MINIMAX_H3_VVAE_CLIP_LENGTH;
  uint64_t act_bytes = gpu_round_up_to_pages(
      (uint64_t)clip_t * (uint64_t)yp.len[0] * (uint64_t)xp.len[0] *
      (uint64_t)ENC_BASE_CH * sizeof(float));
  uint64_t im2col_bytes =
      gpu_round_up_to_pages((uint64_t)ENC_IM2COL_MAX_HALFS * sizeof(uint16_t));
  return act_bytes * (uint64_t)ENC_ACT_BUFS + im2col_bytes;
}

uint64_t minimax_h3_video_vae_decode_workspace_bytes(void) {
  uint64_t max_patches = (uint64_t)VVAE_MAX_CHUNK_LATENTS *
                         (uint64_t)VVAE_TILE_CELLS * (uint64_t)VVAE_TILE_CELLS;
  uint64_t max_seq = max_patches + (uint64_t)VVAE_SUFFIX_TOKENS;
  uint64_t max_padded = ceil_div((uint32_t)max_seq, TILE_M) * TILE_M;
  uint64_t max_padded_i8 = ceil_div((uint32_t)max_seq, TILE_M_I8) * TILE_M_I8;
  uint64_t bytes = 0;
  bytes += gpu_round_up_to_pages((uint64_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                                 max_patches * sizeof(float));
  bytes +=
      2u * gpu_round_up_to_pages(max_seq * (uint64_t)VVAE_DIM * sizeof(float));
  bytes += gpu_round_up_to_pages(max_padded_i8 * (uint64_t)VVAE_FFN_INNER);
  bytes += gpu_round_up_to_pages(max_padded_i8 * sizeof(float));
  bytes +=
      gpu_round_up_to_pages(max_seq * (uint64_t)VVAE_W1_WIDTH * sizeof(float));
  bytes +=
      gpu_round_up_to_pages(max_seq * (uint64_t)VVAE_FFN_INNER * sizeof(float));
  bytes +=
      3u * gpu_round_up_to_pages((uint64_t)VVAE_HEADS * max_padded *
                                 (uint64_t)VVAE_HEAD_DIM * sizeof(uint16_t));
  bytes +=
      gpu_round_up_to_pages(max_padded * (uint64_t)VVAE_DIM * sizeof(uint16_t));
  return bytes;
}

// Frees all GPU buffers in a decode workspace.
static void free_decode_ws(VvaeDecodeWorkspace *ws) {
  gpu_free(&ws->xh);
  gpu_free(&ws->vh);
  gpu_free(&ws->kh);
  gpu_free(&ws->qh);
  gpu_free(&ws->mid);
  gpu_free(&ws->wide);
  gpu_free(&ws->x_scale);
  gpu_free(&ws->xq);
  gpu_free(&ws->norm);
  gpu_free(&ws->h);
  gpu_free(&ws->z_tile);
}

// Allocates a decode workspace sized for up to max_t x max_h x max_w latents.
static int alloc_decode_ws(MinimaxH3VideoVae *vae, VvaeDecodeWorkspace *ws,
                           int max_t, int max_h, int max_w) {
  memset(ws, 0, sizeof *ws);
  size_t max_patches = (size_t)max_t * (size_t)max_h * (size_t)max_w;
  size_t max_seq = max_patches + (size_t)VVAE_SUFFIX_TOKENS;
  size_t max_padded = (size_t)ceil_div((uint32_t)max_seq, TILE_M) * TILE_M;
  size_t max_padded_i8 =
      (size_t)ceil_div((uint32_t)max_seq, TILE_M_I8) * TILE_M_I8;
  size_t head_bytes = (size_t)VVAE_HEADS * max_padded * (size_t)VVAE_HEAD_DIM *
                      sizeof(uint16_t);
  MemorySlot *s = &vae->workspace_slot;
  ws->z_tile = gpu_try_alloc_required(vae->gpu, s,
                                      (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                                          max_patches * sizeof(float));
  ws->h = gpu_try_alloc_required(vae->gpu, s,
                                 max_seq * (size_t)VVAE_DIM * sizeof(float));
  ws->norm = gpu_try_alloc_required(vae->gpu, s,
                                    max_seq * (size_t)VVAE_DIM * sizeof(float));
  ws->xq = gpu_try_alloc_required(vae->gpu, s,
                                  max_padded_i8 * (size_t)VVAE_FFN_INNER);
  ws->x_scale =
      gpu_try_alloc_required(vae->gpu, s, max_padded_i8 * sizeof(float));
  ws->wide = gpu_try_alloc_required(
      vae->gpu, s, max_seq * (size_t)VVAE_W1_WIDTH * sizeof(float));
  ws->mid = gpu_try_alloc_required(
      vae->gpu, s, max_seq * (size_t)VVAE_FFN_INNER * sizeof(float));
  ws->qh = gpu_try_alloc_required(vae->gpu, s, head_bytes);
  ws->kh = gpu_try_alloc_required(vae->gpu, s, head_bytes);
  ws->vh = gpu_try_alloc_required(vae->gpu, s, head_bytes);
  ws->xh = gpu_try_alloc_required(
      vae->gpu, s, max_padded * (size_t)VVAE_DIM * sizeof(uint16_t));
  if (!ws->z_tile.ptr || !ws->h.ptr || !ws->norm.ptr || !ws->xq.ptr ||
      !ws->x_scale.ptr || !ws->wide.ptr || !ws->mid.ptr || !ws->qh.ptr ||
      !ws->kh.ptr || !ws->vh.ptr || !ws->xh.ptr) {
    free_decode_ws(ws);
    return 0;
  }
  return 1;
}

// Encodes native INT8 matmul with bias on already-rotated xq and xs into cmd.
static void enc_mm_i8(const MinimaxH3VideoVae *vae, GpuCommandBuffer *cmd,
                      const VvaeInt8Lin *lin, const Buf *xq, const Buf *xs,
                      const Buf *y, uint32_t rows, uint32_t padded_i8) {
  Int8MMParam mp = {rows, (uint32_t)lin->n, (uint32_t)lin->k, (uint32_t)lin->n,
                    1u};
  gpu_select_kernel(cmd, vae->k_mm_i8);
  gpu_bind_buffer(cmd, 0, &lin->w, 0);
  gpu_bind_buffer(cmd, 1, &lin->ws, 0);
  gpu_bind_buffer(cmd, 2, xq, 0);
  gpu_bind_buffer(cmd, 3, xs, 0);
  gpu_bind_buffer(cmd, 4, &lin->bias, 0);
  gpu_bind_buffer(cmd, 5, y, 0);
  gpu_copy_bytes(cmd, 6, &mp, sizeof mp);
  gpu_run_kernel(cmd,
                 ceil_div((uint32_t)lin->n, TILE_N) * (padded_i8 / TILE_M_I8),
                 1, 1, MM_THREADS, 1, 1);
}

// Encodes ConvRot INT8 quantization + INT8 matmul with bias into cmd.
static void enc_i8_lin(const MinimaxH3VideoVae *vae, GpuCommandBuffer *cmd,
                       const VvaeInt8Lin *lin, const Buf *x, const Buf *xq,
                       const Buf *xs, const Buf *y, uint32_t rows,
                       uint32_t padded_i8) {
  ConvRotParam cp = {rows, (uint32_t)lin->k, (uint32_t)lin->k};
  gpu_select_kernel(cmd, vae->k_convrot);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, xq, 0);
  gpu_bind_buffer(cmd, 2, xs, 0);
  gpu_copy_bytes(cmd, 3, &cp, sizeof cp);
  gpu_run_kernel(cmd, padded_i8, 1, 1, CONVROT_THREADS, 1, 1);
  enc_mm_i8(vae, cmd, lin, xq, xs, y, rows, padded_i8);
}

// Decodes one denormalized latent tile [24, t_lat, h_lat, w_lat] in ws->z_tile
// into ImageNet-normalized FP32 frames [t_lat * 4, 3, h_lat * 16, w_lat * 16]
// in out_tile (frame-major, channel-first per frame).
static int decode_tile_3d(MinimaxH3VideoVae *vae, VvaeDecodeWorkspace *ws,
                          int t_lat, int h_lat, int w_lat, float *out_tile,
                          const MinimaxH3RunHooks *hooks) {
  uint32_t patches = (uint32_t)(t_lat * h_lat * w_lat);
  uint32_t rows = patches + (uint32_t)VVAE_SUFFIX_TOKENS;
  uint32_t padded = ceil_div(rows, TILE_M) * TILE_M;
  uint32_t padded_i8 = ceil_div(rows, TILE_M_I8) * TILE_M_I8;

  VvaeEmbedParam ep = {(uint32_t)t_lat, (uint32_t)h_lat, (uint32_t)w_lat};
  GpuCommandBuffer cmd = gpu_begin(vae->gpu);
  gpu_select_kernel(&cmd, vae->k_embed);
  gpu_bind_buffer(&cmd, 0, &ws->z_tile, 0);
  gpu_bind_buffer(&cmd, 1, &vae->pqc_w, 0);
  gpu_bind_buffer(&cmd, 2, &vae->pqc_b, 0);
  gpu_bind_buffer(&cmd, 3, &vae->emb_w, 0);
  gpu_bind_buffer(&cmd, 4, &vae->emb_b, 0);
  gpu_bind_buffer(&cmd, 5, &vae->reg_tokens, 0);
  gpu_bind_buffer(&cmd, 6, &ws->h, 0);
  gpu_copy_bytes(&cmd, 7, &ep, sizeof ep);
  gpu_run_kernel(&cmd, rows, 1, 1, EMBED_THREADS, 1, 1);
  uint64_t cb = cmd.number;
  gpu_commit(&cmd);
  minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

  RmsNormConvRotParam rnp = {rows, VVAE_DIM, VVAE_NORM_EPS};
  VvaeHeadsParam hp = {
      rows,         padded, (uint32_t)t_lat, (uint32_t)h_lat, (uint32_t)w_lat,
      VVAE_NORM_EPS};
  VvaeAttendParam ap = {rows, padded};
  AddScaledParam asp = {rows, VVAE_DIM};
  SwigluConvRotParam swp = {rows, VVAE_FFN_INNER};

  for (int li = 0; li < VVAE_LAYERS; li++) {
    if (minimax_h3_should_stop(hooks)) return 0;
    const VvaeBlock *b = &vae->blocks[li];
    cmd = gpu_begin(vae->gpu);

    gpu_select_kernel(&cmd, vae->k_rmsnorm_convrot);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm1, 0);
    gpu_bind_buffer(&cmd, 2, &ws->xq, 0);
    gpu_bind_buffer(&cmd, 3, &ws->x_scale, 0);
    gpu_copy_bytes(&cmd, 4, &rnp, sizeof rnp);
    gpu_run_kernel(&cmd, padded_i8, 1, 1, CONVROT_THREADS, 1, 1);

    enc_mm_i8(vae, &cmd, &b->to_qkv, &ws->xq, &ws->x_scale, &ws->wide, rows,
              padded_i8);

    gpu_select_kernel(&cmd, vae->k_heads);
    gpu_bind_buffer(&cmd, 0, &ws->wide, 0);
    gpu_bind_buffer(&cmd, 1, &vae->inv_freq, 0);
    gpu_bind_buffer(&cmd, 2, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 4, &ws->vh, 0);
    gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
    gpu_run_kernel(&cmd, MINIMAX_H3_QKV_PARTS * VVAE_HEADS, padded, 1,
                   HEADS_THREADS, 1, 1);

    gpu_select_kernel(&cmd, vae->k_attend);
    gpu_bind_buffer(&cmd, 0, &ws->qh, 0);
    gpu_bind_buffer(&cmd, 1, &ws->kh, 0);
    gpu_bind_buffer(&cmd, 2, &ws->vh, 0);
    gpu_bind_buffer(&cmd, 3, &ws->mid, 0);
    gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
    gpu_run_kernel(&cmd, ceil_div(rows, ATTN_GROUP_ROWS) * VVAE_HEADS, 1, 1,
                   ATTN_THREADS, 1, 1);

    enc_i8_lin(vae, &cmd, &b->to_out, &ws->mid, &ws->xq, &ws->x_scale,
               &ws->norm, rows, padded_i8);

    gpu_select_kernel(&cmd, vae->k_add_scaled);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &b->scale1, 0);
    gpu_copy_bytes(&cmd, 3, &asp, sizeof asp);
    gpu_run_kernel(&cmd, ceil_div(VVAE_DIM, ELEM_THREADS), rows, 1,
                   ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, vae->k_rmsnorm_convrot);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &b->norm2, 0);
    gpu_bind_buffer(&cmd, 2, &ws->xq, 0);
    gpu_bind_buffer(&cmd, 3, &ws->x_scale, 0);
    gpu_copy_bytes(&cmd, 4, &rnp, sizeof rnp);
    gpu_run_kernel(&cmd, padded_i8, 1, 1, CONVROT_THREADS, 1, 1);

    enc_mm_i8(vae, &cmd, &b->ff_w1, &ws->xq, &ws->x_scale, &ws->wide, rows,
              padded_i8);

    gpu_select_kernel(&cmd, vae->k_swiglu_convrot);
    gpu_bind_buffer(&cmd, 0, &ws->wide, 0);
    gpu_bind_buffer(&cmd, 1, &ws->xq, 0);
    gpu_bind_buffer(&cmd, 2, &ws->x_scale, 0);
    gpu_copy_bytes(&cmd, 3, &swp, sizeof swp);
    gpu_run_kernel(&cmd, padded_i8, 1, 1, CONVROT_THREADS, 1, 1);

    enc_mm_i8(vae, &cmd, &b->ff_w2, &ws->xq, &ws->x_scale, &ws->norm, rows,
              padded_i8);

    gpu_select_kernel(&cmd, vae->k_add_scaled);
    gpu_bind_buffer(&cmd, 0, &ws->h, 0);
    gpu_bind_buffer(&cmd, 1, &ws->norm, 0);
    gpu_bind_buffer(&cmd, 2, &b->scale2, 0);
    gpu_copy_bytes(&cmd, 3, &asp, sizeof asp);
    gpu_run_kernel(&cmd, ceil_div(VVAE_DIM, ELEM_THREADS), rows, 1,
                   ELEM_THREADS, 1, 1);

    cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
  }

  // Final LayerNorm + FP16 proj_out on the patch tokens only.
  uint32_t patch_padded = ceil_div(patches, TILE_M) * TILE_M;
  LayerNormParam lnp = {VVAE_DIM, VVAE_NORM_EPS};
  F16XParam xp = {patches, VVAE_DIM, VVAE_DIM};
  F16MMParam mp = {patches, VVAE_PATCH_OUT, VVAE_DIM, VVAE_PATCH_OUT, 1u};

  cmd = gpu_begin(vae->gpu);
  gpu_select_kernel(&cmd, vae->k_layernorm);
  gpu_bind_buffer(&cmd, 0, &ws->h, 0);
  gpu_bind_buffer(&cmd, 1, &vae->norm_out_w, 0);
  gpu_bind_buffer(&cmd, 2, &vae->norm_out_b, 0);
  gpu_bind_buffer(&cmd, 3, &ws->norm, 0);
  gpu_copy_bytes(&cmd, 4, &lnp, sizeof lnp);
  gpu_run_kernel(&cmd, patches, 1, 1, NORM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, vae->k_x_f16);
  gpu_bind_buffer(&cmd, 0, &ws->norm, 0);
  gpu_bind_buffer(&cmd, 1, &ws->xh, 0);
  gpu_copy_bytes(&cmd, 2, &xp, sizeof xp);
  gpu_run_kernel(&cmd, ceil_div(VVAE_DIM / MINIMAX_H3_VEC4_WIDTH, ELEM_THREADS),
                 patch_padded, 1, ELEM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, vae->k_mm_f16);
  gpu_bind_buffer(&cmd, 0, &vae->proj_out_w, 0);
  gpu_bind_buffer(&cmd, 1, &ws->xh, 0);
  gpu_bind_buffer(&cmd, 2, &vae->proj_out_b, 0);
  gpu_bind_buffer(&cmd, 3, &ws->wide, 0);
  gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
  gpu_run_kernel(&cmd, ceil_div(VVAE_PATCH_OUT, TILE_N), patch_padded / TILE_M,
                 1, MM_THREADS, 1, 1);
  cb = cmd.number;
  gpu_commit(&cmd);
  minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

  // Unpatchify [t_lat*h_lat*w_lat, 3*4*16*16] -> [t_lat*4, 3, h_lat*16,
  // w_lat*16].
  const float *po = (const float *)ws->wide.ptr;
  int tile_h = h_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  int tile_w = w_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  size_t plane = (size_t)tile_h * (size_t)tile_w;
  size_t frame_elems = (size_t)MINIMAX_H3_RGB_CHANNELS * plane;
  for (int pt = 0; pt < t_lat; pt++) {
    for (int py = 0; py < h_lat; py++) {
      for (int px = 0; px < w_lat; px++) {
        size_t tok = ((size_t)pt * (size_t)h_lat + (size_t)py) * (size_t)w_lat +
                     (size_t)px;
        const float *src = po + tok * (size_t)VVAE_PATCH_OUT;
        for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
          for (int dt = 0; dt < MINIMAX_H3_VVAE_TEMPORAL_RATIO; dt++) {
            float *dst_plane =
                out_tile +
                (size_t)(pt * MINIMAX_H3_VVAE_TEMPORAL_RATIO + dt) *
                    frame_elems +
                (size_t)c * plane;
            const float *sp =
                src + (size_t)(c * MINIMAX_H3_VVAE_TEMPORAL_RATIO + dt) *
                          (size_t)(MINIMAX_H3_VAE_SPATIAL_RATIO *
                                   MINIMAX_H3_VAE_SPATIAL_RATIO);
            for (int dy = 0; dy < MINIMAX_H3_VAE_SPATIAL_RATIO; dy++) {
              memcpy(dst_plane +
                         (size_t)(py * MINIMAX_H3_VAE_SPATIAL_RATIO + dy) *
                             (size_t)tile_w +
                         (size_t)(px * MINIMAX_H3_VAE_SPATIAL_RATIO),
                     sp + (size_t)dy * (size_t)MINIMAX_H3_VAE_SPATIAL_RATIO,
                     (size_t)MINIMAX_H3_VAE_SPATIAL_RATIO * sizeof(float));
            }
          }
        }
      }
    }
  }
  return 1;
}

// Decodes one temporal latent chunk [24, clip_t, h_lat, w_lat] using spatial
// tiling and linear overlap blending into [clip_t * 4, 3, height, width].
static int decode_chunk_tiled(MinimaxH3VideoVae *vae, VvaeDecodeWorkspace *ws,
                              const float *z_chunk, int clip_t, int h_lat,
                              int w_lat, float *canvas,
                              const MinimaxH3RunHooks *hooks) {
  int height = h_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  int width = w_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  int frames = clip_t * MINIMAX_H3_VVAE_TEMPORAL_RATIO;
  MinimaxH3VaeTilePlan y_plan = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan x_plan = minimax_h3_image_vae_split_tiles(width);

  size_t canvas_plane = (size_t)height * (size_t)width;
  size_t canvas_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * canvas_plane;
  size_t max_tile_elems = (size_t)frames * (size_t)MINIMAX_H3_RGB_CHANNELS *
                          (size_t)VVAE_TILE_SIZE * (size_t)VVAE_TILE_SIZE;
  float *tile = xmalloc(max_tile_elems * sizeof(float));
  float *strip = NULL;
  int ok = 1, out_y = 0;

  for (int i = 0; i < y_plan.count && ok; i++) {
    int ty = y_plan.start[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int th_lat = y_plan.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int tile_h = y_plan.len[i];
    int ov_top = i > 0 ? y_plan.overlap[i - 1] : 0;
    int ov_bot = i < y_plan.count - 1 ? y_plan.overlap[i] : 0;
    int keep_h = tile_h - ov_bot;

    float *new_strip =
        ov_bot > 0 ? xmalloc((size_t)frames * (size_t)MINIMAX_H3_RGB_CHANNELS *
                             (size_t)ov_bot * (size_t)width * sizeof(float))
                   : NULL;
    float *left_tail = NULL;
    int out_x = 0;

    for (int j = 0; j < x_plan.count; j++) {
      int tx = x_plan.start[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int tw_lat = x_plan.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int tile_w = x_plan.len[j];
      int ov_left = j > 0 ? x_plan.overlap[j - 1] : 0;
      int ov_right = j < x_plan.count - 1 ? x_plan.overlap[j] : 0;

      // Copy denormalized tile [24, clip_t, th_lat, tw_lat] into ws->z_tile.
      float *zd = (float *)ws->z_tile.ptr;
      size_t src_ch_stride = (size_t)clip_t * (size_t)h_lat * (size_t)w_lat;
      size_t dst_ch_stride = (size_t)clip_t * (size_t)th_lat * (size_t)tw_lat;
      for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
        for (int pt = 0; pt < clip_t; pt++) {
          for (int py = 0; py < th_lat; py++) {
            const float *sr = z_chunk + (size_t)c * src_ch_stride +
                              ((size_t)pt * (size_t)h_lat + (size_t)(ty + py)) *
                                  (size_t)w_lat +
                              (size_t)tx;
            float *dr =
                zd + (size_t)c * dst_ch_stride +
                ((size_t)pt * (size_t)th_lat + (size_t)py) * (size_t)tw_lat;
            memcpy(dr, sr, (size_t)tw_lat * sizeof(float));
          }
        }
      }

      if (!decode_tile_3d(vae, ws, clip_t, th_lat, tw_lat, tile, hooks)) {
        ok = 0;
        break;
      }

      size_t tile_plane = (size_t)tile_h * (size_t)tile_w;
      size_t tile_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * tile_plane;
      if (ov_top > 0) {
        size_t s_plane = (size_t)ov_top * (size_t)width;
        size_t s_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * s_plane;
        for (int f = 0; f < frames; f++) {
          for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
            float *tp = tile + (size_t)f * tile_frame + (size_t)c * tile_plane;
            const float *sp = strip + (size_t)f * s_frame + (size_t)c * s_plane;
            for (int y = 0; y < ov_top; y++) {
              float wb = (float)y / (float)ov_top, wa = 1.0f - wb;
              float *tr = tp + (size_t)y * (size_t)tile_w;
              const float *sr =
                  sp + (size_t)y * (size_t)width + (size_t)x_plan.start[j];
              for (int x = 0; x < tile_w; x++) tr[x] = sr[x] * wa + tr[x] * wb;
            }
          }
        }
      }

      if (ov_left > 0) {
        size_t l_plane = (size_t)tile_h * (size_t)ov_left;
        size_t l_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * l_plane;
        for (int f = 0; f < frames; f++) {
          for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
            float *tp = tile + (size_t)f * tile_frame + (size_t)c * tile_plane;
            const float *lp =
                left_tail + (size_t)f * l_frame + (size_t)c * l_plane;
            for (int y = 0; y < tile_h; y++) {
              float *tr = tp + (size_t)y * (size_t)tile_w;
              const float *lr = lp + (size_t)y * (size_t)ov_left;
              for (int x = 0; x < ov_left; x++) {
                float wb = (float)x / (float)ov_left, wa = 1.0f - wb;
                tr[x] = lr[x] * wa + tr[x] * wb;
              }
            }
          }
        }
      }

      free(left_tail);
      left_tail = NULL;
      if (ov_right > 0) {
        size_t r_plane = (size_t)tile_h * (size_t)ov_right;
        size_t r_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * r_plane;
        left_tail = xmalloc((size_t)frames * r_frame * sizeof(float));
        for (int f = 0; f < frames; f++) {
          for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
            const float *tp =
                tile + (size_t)f * tile_frame + (size_t)c * tile_plane;
            float *rp = left_tail + (size_t)f * r_frame + (size_t)c * r_plane;
            for (int y = 0; y < tile_h; y++)
              memcpy(rp + (size_t)y * (size_t)ov_right,
                     tp + (size_t)y * (size_t)tile_w +
                         (size_t)(tile_w - ov_right),
                     (size_t)ov_right * sizeof(float));
          }
        }
      }

      int keep_w = tile_w - ov_right;
      if (ov_bot > 0) {
        size_t ns_plane = (size_t)ov_bot * (size_t)width;
        size_t ns_frame = (size_t)MINIMAX_H3_RGB_CHANNELS * ns_plane;
        for (int f = 0; f < frames; f++) {
          for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
            const float *tp =
                tile + (size_t)f * tile_frame + (size_t)c * tile_plane;
            float *np = new_strip + (size_t)f * ns_frame + (size_t)c * ns_plane;
            for (int y = 0; y < ov_bot; y++)
              memcpy(np + (size_t)y * (size_t)width + (size_t)out_x,
                     tp + (size_t)(keep_h + y) * (size_t)tile_w,
                     (size_t)keep_w * sizeof(float));
          }
        }
      }

      for (int f = 0; f < frames; f++) {
        for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
          const float *tp =
              tile + (size_t)f * tile_frame + (size_t)c * tile_plane;
          float *cp =
              canvas + (size_t)f * canvas_frame + (size_t)c * canvas_plane;
          for (int y = 0; y < keep_h; y++)
            memcpy(cp + (size_t)(out_y + y) * (size_t)width + (size_t)out_x,
                   tp + (size_t)y * (size_t)tile_w,
                   (size_t)keep_w * sizeof(float));
        }
      }
      out_x += keep_w;
    }

    free(left_tail);
    free(strip);
    strip = new_strip;
    out_y += keep_h;
  }

  free(strip);
  free(tile);
  return ok;
}

// Converts ImageNet-normalized FP32 frames [count, 3, height, width] into
// interleaved 8-bit sRGB frames appended to out_rgb at *write_pos.
static void finalize_frames(const float *part, int count, int width, int height,
                            int max_frames, uint8_t *out_rgb, int *write_pos) {
  int copy = count;
  if (*write_pos + copy > max_frames) copy = max_frames - *write_pos;
  if (copy <= 0) return;
  size_t plane = (size_t)width * (size_t)height;
  size_t f_floats = (size_t)MINIMAX_H3_RGB_CHANNELS * plane;
  size_t f_bytes = plane * (size_t)MINIMAX_H3_RGB_CHANNELS;
  for (int f = 0; f < copy; f++) {
    const float *fp = part + (size_t)f * f_floats;
    uint8_t *dst = out_rgb + (size_t)(*write_pos + f) * f_bytes;
    for (size_t p = 0; p < plane; p++) {
      for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
        float v =
            fp[(size_t)c * plane + p] * IMAGENET_STD[c] + IMAGENET_MEAN[c];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        dst[p * (size_t)MINIMAX_H3_RGB_CHANNELS + (size_t)c] =
            (uint8_t)lroundf(v * MINIMAX_H3_PIXEL_MAX);
      }
    }
  }
  *write_pos += copy;
}

int minimax_h3_video_vae_decode(MinimaxH3VideoVae *vae, const float *latents,
                                int t_lat, int h_lat, int w_lat,
                                uint8_t *out_rgb,
                                const MinimaxH3RunHooks *hooks) {
  if (!vae || !latents || !out_rgb || t_lat <= 0 || h_lat <= 0 || w_lat <= 0)
    return 0;
  if (!minimax_h3_video_vae_load(vae)) return 0;
  if (minimax_h3_should_stop(hooks)) return 0;

  int height = h_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  int width = w_lat * MINIMAX_H3_VAE_SPATIAL_RATIO;
  MinimaxH3VaeTilePlan yp = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan xp = minimax_h3_image_vae_split_tiles(width);
  int max_th = yp.len[0] / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int max_tw = xp.len[0] / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int max_ct =
      t_lat == 1 ? MINIMAX_H3_VVAE_CHUNK_TOKENS : VVAE_MAX_CHUNK_LATENTS;

  VvaeDecodeWorkspace ws = {0};
  if (!alloc_decode_ws(vae, &ws, max_ct, max_th, max_tw)) return 0;

  size_t hw_lat = (size_t)h_lat * (size_t)w_lat;
  size_t frame_floats =
      (size_t)MINIMAX_H3_RGB_CHANNELS * (size_t)height * (size_t)width;
  int ok = 1;

  if (t_lat == 1) {
    // Still latent: repeat to 5 latents and keep frame_pre_padding
    // (vae.py:1048).
    int clip_t = MINIMAX_H3_VVAE_CHUNK_TOKENS;
    float *z_rep = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                           (size_t)clip_t * hw_lat * sizeof(float));
    for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
      float m = vae->latents_mean[c], s = vae->latents_std[c];
      for (int pt = 0; pt < clip_t; pt++) {
        float *dst = z_rep + ((size_t)c * (size_t)clip_t + (size_t)pt) * hw_lat;
        const float *src = latents + (size_t)c * hw_lat;
        for (size_t p = 0; p < hw_lat; p++) dst[p] = src[p] * s + m;
      }
    }
    float *canvas = xmalloc((size_t)(clip_t * MINIMAX_H3_VVAE_TEMPORAL_RATIO) *
                            frame_floats * sizeof(float));
    ok = decode_chunk_tiled(vae, &ws, z_rep, clip_t, h_lat, w_lat, canvas,
                            hooks);
    if (ok) {
      int pos = 0;
      finalize_frames(canvas +
                          (size_t)MINIMAX_H3_VVAE_FRAME_PRE_PAD * frame_floats,
                      1, width, height, 1, out_rgb, &pos);
    }
    free(canvas);
    free(z_rep);
    free_decode_ws(&ws);
    return ok;
  }

  MinimaxH3VvaeTemporalPlan plan = minimax_h3_video_vae_decode_plan(t_lat);
  int z_padded_t = t_lat + plan.pad_tokens;
  float *z_chunk =
      xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
              (size_t)VVAE_MAX_CHUNK_LATENTS * hw_lat * sizeof(float));
  float *canvas = xmalloc(
      (size_t)(VVAE_MAX_CHUNK_LATENTS * MINIMAX_H3_VVAE_TEMPORAL_RATIO) *
      frame_floats * sizeof(float));
  float *overlap = NULL;
  int overlap_frames = 0, write_pos = 0;

  for (int i = 0; i < plan.num_chunks && ok; i++) {
    int t_start = i * MINIMAX_H3_VVAE_CHUNK_TOKENS;
    int t_end = t_start + VVAE_MAX_CHUNK_LATENTS;
    if (t_end > z_padded_t) t_end = z_padded_t;
    int clip_t = t_end - t_start;

    for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
      float m = vae->latents_mean[c], s = vae->latents_std[c];
      for (int dt = 0; dt < clip_t; dt++) {
        int src_t = t_start + dt;
        if (src_t >= t_lat) src_t = t_lat - 1;
        const float *src =
            latents + ((size_t)c * (size_t)t_lat + (size_t)src_t) * hw_lat;
        float *dst =
            z_chunk + ((size_t)c * (size_t)clip_t + (size_t)dt) * hw_lat;
        for (size_t p = 0; p < hw_lat; p++) dst[p] = src[p] * s + m;
      }
    }

    if (!decode_chunk_tiled(vae, &ws, z_chunk, clip_t, h_lat, w_lat, canvas,
                            hooks)) {
      ok = 0;
      break;
    }

    int clip_frames = clip_t * MINIMAX_H3_VVAE_TEMPORAL_RATIO;
    for (int j = 0; j < VVAE_SPLIT_COUNT; j++) {
      int f_start = j * VVAE_CHUNK_DEC_FRAMES;
      int f_end = f_start + VVAE_CHUNK_DEC_FRAMES;
      if (f_end > clip_frames) f_end = clip_frames;
      int keep_start = f_start + MINIMAX_H3_VVAE_FRAME_PRE_PAD;
      int keep_n = f_end - keep_start;
      if (keep_n <= 0) continue;
      float *part = canvas + (size_t)keep_start * frame_floats;

      if (j == 0) {
        if (overlap != NULL) {
          int blend_n = MINIMAX_H3_VVAE_FRAME_OVERLAP;
          if (overlap_frames < blend_n) blend_n = overlap_frames;
          if (keep_n < blend_n) blend_n = keep_n;
          for (int f = 0; f < blend_n; f++) {
            float wb = (float)f / (float)blend_n, wa = 1.0f - wb;
            const float *ap =
                overlap + (size_t)(overlap_frames - blend_n + f) * frame_floats;
            float *bp = part + (size_t)f * frame_floats;
            for (size_t p = 0; p < frame_floats; p++)
              bp[p] = ap[p] * wa + bp[p] * wb;
          }
          free(overlap);
          overlap = NULL;
          overlap_frames = 0;
        }
        finalize_frames(part, keep_n, width, height, plan.out_frames, out_rgb,
                        &write_pos);
      } else {
        free(overlap);
        overlap = xmalloc((size_t)keep_n * frame_floats * sizeof(float));
        memcpy(overlap, part, (size_t)keep_n * frame_floats * sizeof(float));
        overlap_frames = keep_n;
      }
    }

    if (i == plan.num_chunks - 1 && overlap != NULL) {
      finalize_frames(overlap, overlap_frames, width, height, plan.out_frames,
                      out_rgb, &write_pos);
      free(overlap);
      overlap = NULL;
      overlap_frames = 0;
    }
  }

  free(overlap);
  free(canvas);
  free(z_chunk);
  free_decode_ws(&ws);
  return ok;
}

// Frees all GPU buffers in an encode workspace.
static void free_encode_ws(VvaeEncodeWorkspace *ws) {
  gpu_free(&ws->im2col);
  gpu_free(&ws->c);
  gpu_free(&ws->b);
  gpu_free(&ws->a);
}

// Allocates a 3D encoder workspace sized for clip_t frames of max_h x max_w.
static int alloc_encode_ws(MinimaxH3VideoVae *vae, VvaeEncodeWorkspace *ws,
                           int clip_t, int max_h, int max_w) {
  memset(ws, 0, sizeof *ws);
  size_t act_bytes = (size_t)clip_t * (size_t)max_h * (size_t)max_w *
                     (size_t)ENC_BASE_CH * sizeof(float);
  size_t im_bytes = ENC_IM2COL_MAX_HALFS * sizeof(uint16_t);
  MemorySlot *s = &vae->workspace_slot;
  ws->a = gpu_try_alloc_required(vae->gpu, s, act_bytes);
  ws->b = gpu_try_alloc_required(vae->gpu, s, act_bytes);
  ws->c = gpu_try_alloc_required(vae->gpu, s, act_bytes);
  ws->im2col = gpu_try_alloc_required(vae->gpu, s, im_bytes);
  if (!ws->a.ptr || !ws->b.ptr || !ws->c.ptr || !ws->im2col.ptr) {
    free_encode_ws(ws);
    return 0;
  }
  return 1;
}

// Runs one 3D causal convolution on NDHWC src [in_t, in_h, in_w, cin] into
// NDHWC dst [out_t, out_h, out_w, cout], tiled into spatial strips per frame.
static int run_enc_conv3d(MinimaxH3VideoVae *vae, VvaeEncodeWorkspace *ws,
                          const EncConv3d *conv, const Buf *src, int in_t,
                          int in_h, int in_w, const Buf *dst, int *out_t_ptr,
                          int *out_h_ptr, int *out_w_ptr,
                          const MinimaxH3RunHooks *hooks) {
  int out_t = (in_t + conv->front_pad - conv->kt) / conv->time_stride + 1;
  int out_h = in_h / conv->space_stride;
  int out_w = in_w / conv->space_stride;
  if (out_t_ptr) *out_t_ptr = out_t;
  if (out_h_ptr) *out_h_ptr = out_h;
  if (out_w_ptr) *out_w_ptr = out_w;

  uint32_t max_rows = (uint32_t)(ENC_IM2COL_MAX_HALFS / (size_t)conv->k_padded);
  max_rows = (max_rows / TILE_M) * TILE_M;
  if (max_rows < TILE_M) max_rows = TILE_M;
  int max_strip_h = (int)(max_rows / (uint32_t)out_w);
  if (max_strip_h < 1) max_strip_h = 1;

  for (int ot = 0; ot < out_t; ot++) {
    if (minimax_h3_should_stop(hooks)) return 0;
    GpuCommandBuffer cmd = gpu_begin(vae->gpu);
    for (int y0 = 0; y0 < out_h; y0 += max_strip_h) {
      int strip_h = out_h - y0;
      if (strip_h > max_strip_h) strip_h = max_strip_h;
      uint32_t m = (uint32_t)(strip_h * out_w);
      uint32_t m_pad = ceil_div(m, TILE_M) * TILE_M;
      size_t dst_off = ((size_t)ot * (size_t)out_h + (size_t)y0) *
                       (size_t)out_w * (size_t)conv->cout * sizeof(float);

      VvaeIm2Col3dParam ip = {
          .in_t = (uint32_t)in_t,
          .in_h = (uint32_t)in_h,
          .in_w = (uint32_t)in_w,
          .cin = (uint32_t)conv->cin,
          .out_w = (uint32_t)out_w,
          .ot = (uint32_t)ot,
          .out_y0 = (uint32_t)y0,
          .strip_h = (uint32_t)strip_h,
          .kt = (uint32_t)conv->kt,
          .ks = (uint32_t)conv->ks,
          .time_stride = (uint32_t)conv->time_stride,
          .space_stride = (uint32_t)conv->space_stride,
          .front_pad = (uint32_t)conv->front_pad,
          .pad_h = (uint32_t)conv->pad_h,
          .pad_w = (uint32_t)conv->pad_w,
          .k_padded = (uint32_t)conv->k_padded,
      };
      F16MMParam mp = {m, (uint32_t)conv->cout, (uint32_t)conv->k_padded,
                       (uint32_t)conv->cout, 1u};

      gpu_select_kernel(&cmd, vae->k_im2col3d);
      gpu_bind_buffer(&cmd, 0, src, 0);
      gpu_bind_buffer(&cmd, 1, &ws->im2col, 0);
      gpu_copy_bytes(&cmd, 2, &ip, sizeof ip);
      gpu_run_kernel(&cmd, ceil_div((uint32_t)conv->k_padded, IM2COL_TX),
                     ceil_div(m_pad, IM2COL_TY), 1, IM2COL_TX, IM2COL_TY, 1);

      gpu_select_kernel(&cmd, vae->k_mm_f16);
      gpu_bind_buffer(&cmd, 0, &conv->w, 0);
      gpu_bind_buffer(&cmd, 1, &ws->im2col, 0);
      gpu_bind_buffer(&cmd, 2, &conv->bias, 0);
      gpu_bind_buffer(&cmd, 3, dst, dst_off);
      gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
      gpu_run_kernel(&cmd, ceil_div((uint32_t)conv->cout, TILE_N),
                     m_pad / TILE_M, 1, MM_THREADS, 1, 1);
    }
    uint64_t cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
  }
  return 1;
}

// Runs temporal-isolated GroupNorm + SiLU over NDHWC [t, h, w, ch].
static void run_enc_gn_silu(MinimaxH3VideoVae *vae, const Buf *x, const Buf *w,
                            const Buf *b, const Buf *y, int t, int h, int w_px,
                            int ch, const MinimaxH3RunHooks *hooks) {
  VvaeGn3dParam gp = {(uint32_t)(h * w_px), (uint32_t)ch, ENC_GN_GROUPS,
                      VVAE_ENC_GN_EPS};
  GpuCommandBuffer cmd = gpu_begin(vae->gpu);
  gpu_select_kernel(&cmd, vae->k_gn3d);
  gpu_bind_buffer(&cmd, 0, x, 0);
  gpu_bind_buffer(&cmd, 1, w, 0);
  gpu_bind_buffer(&cmd, 2, b, 0);
  gpu_bind_buffer(&cmd, 3, y, 0);
  gpu_copy_bytes(&cmd, 4, &gp, sizeof gp);
  gpu_run_kernel(&cmd, ENC_GN_GROUPS, (uint32_t)t, 1, NORM_THREADS, 1, 1);
  uint64_t cb = cmd.number;
  gpu_commit(&cmd);
  minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
}

// Encodes one ImageNet-normalized NDHWC tile [clip_t, tile_h, tile_w, 3] in
// ws->a into channel-first posterior mean [24, t_lat, h_lat, w_lat] in out_z.
static int encode_tile_3d(MinimaxH3VideoVae *vae, VvaeEncodeWorkspace *ws,
                          int clip_t, int tile_h, int tile_w, float *out_z,
                          const MinimaxH3RunHooks *hooks) {
  int t = clip_t, h = tile_h, w = tile_w;
  if (!run_enc_conv3d(vae, ws, &vae->enc_conv_in, &ws->a, t, h, w, &ws->b, &t,
                      &h, &w, hooks))
    return 0;
  Buf cur = ws->b, tmp = ws->a, alt = ws->c;

  for (int lv = 0; lv < ENC_LEVELS; lv++) {
    const EncLevel3d *level = &vae->enc_levels[lv];
    int in_ch = lv == 0 ? ENC_BASE_CH : ENC_CH[lv - 1];
    int out_ch = ENC_CH[lv];
    for (int bi = 0; bi < ENC_BLOCKS; bi++) {
      if (minimax_h3_should_stop(hooks)) return 0;
      const EncResBlock3d *rb = &level->blocks[bi];
      int bc_in = bi == 0 ? in_ch : out_ch;

      run_enc_gn_silu(vae, &cur, &rb->norm1_w, &rb->norm1_b, &tmp, t, h, w,
                      bc_in, hooks);
      if (!run_enc_conv3d(vae, ws, &rb->conv1, &tmp, t, h, w, &alt, NULL, NULL,
                          NULL, hooks))
        return 0;
      run_enc_gn_silu(vae, &alt, &rb->norm2_w, &rb->norm2_b, &tmp, t, h, w,
                      out_ch, hooks);
      if (rb->has_shortcut) {
        if (!run_enc_conv3d(vae, ws, &rb->shortcut, &cur, t, h, w, &alt, NULL,
                            NULL, NULL, hooks))
          return 0;
        Buf swap = cur;
        cur = alt;
        alt = swap;
      }
      if (!run_enc_conv3d(vae, ws, &rb->conv2, &tmp, t, h, w, &alt, NULL, NULL,
                          NULL, hooks))
        return 0;

      uint32_t count =
          (uint32_t)((size_t)t * (size_t)h * (size_t)w * (size_t)out_ch);
      GpuCommandBuffer cmd = gpu_begin(vae->gpu);
      gpu_select_kernel(&cmd, vae->k_add);
      gpu_bind_buffer(&cmd, 0, &cur, 0);
      gpu_bind_buffer(&cmd, 1, &alt, 0);
      gpu_copy_bytes(&cmd, 2, &count, sizeof count);
      gpu_run_kernel(&cmd, ceil_div(count, ADD_THREADS), 1, 1, ADD_THREADS, 1,
                     1);
      uint64_t cb = cmd.number;
      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
    }
    if (level->has_downsample) {
      int nt = t, nh = h, nw = w;
      if (!run_enc_conv3d(vae, ws, &level->downsample, &cur, t, h, w, &tmp, &nt,
                          &nh, &nw, hooks))
        return 0;
      Buf swap = cur;
      cur = tmp;
      tmp = swap;
      t = nt;
      h = nh;
      w = nw;
    }
  }

  run_enc_gn_silu(vae, &cur, &vae->enc_norm_out_w, &vae->enc_norm_out_b, &tmp,
                  t, h, w, ENC_CH[ENC_LEVELS - 1], hooks);
  if (!run_enc_conv3d(vae, ws, &vae->enc_conv_out, &tmp, t, h, w, &alt, NULL,
                      NULL, NULL, hooks))
    return 0;
  if (!run_enc_conv3d(vae, ws, &vae->enc_quant_conv, &alt, t, h, w, &cur, NULL,
                      NULL, NULL, hooks))
    return 0;

  // Keep posterior mean (first 24 of 48 channels), converting NDHWC -> NCDHW.
  const float *mom = (const float *)cur.ptr;
  size_t plane = (size_t)t * (size_t)h * (size_t)w;
  for (size_t p = 0; p < plane; p++) {
    const float *mp = mom + p * (size_t)ENC_MOMENT_CH;
    for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++)
      out_z[(size_t)c * plane + p] = mp[c];
  }
  return 1;
}

// Encodes one clip of clip_t frames (padded by repeating the last frame when
// avail_frames < clip_t) with spatial tiling into unnormalized posterior means
// [24, t_lat, h_lat, w_lat] in clip_out.
static int encode_clip_tiled(MinimaxH3VideoVae *vae, VvaeEncodeWorkspace *ws,
                             const uint8_t *rgb_start, int avail_frames,
                             int clip_t, int width, int height, float *clip_out,
                             const MinimaxH3RunHooks *hooks) {
  MinimaxH3VaeTilePlan yp = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan xp = minimax_h3_image_vae_split_tiles(width);
  int t_lat = clip_t == 1 ? 1 : MINIMAX_H3_VVAE_CHUNK_TOKENS;
  int h_lat = height / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = width / MINIMAX_H3_VAE_SPATIAL_RATIO;

  int n_tiles = yp.count * xp.count;
  float **raw_tiles = xmalloc((size_t)n_tiles * sizeof(float *));
  memset(raw_tiles, 0, (size_t)n_tiles * sizeof(float *));
  size_t frame_bytes =
      (size_t)height * (size_t)width * (size_t)MINIMAX_H3_RGB_CHANNELS;

  int ok = 1;
  for (int i = 0; i < yp.count && ok; i++) {
    int th = yp.len[i], th_lat = th / MINIMAX_H3_VAE_SPATIAL_RATIO;
    for (int j = 0; j < xp.count; j++) {
      int tw = xp.len[j], tw_lat = tw / MINIMAX_H3_VAE_SPATIAL_RATIO;
      // Load tile into ws->a in ImageNet-normalized NDHWC FP32.
      float *dst = (float *)ws->a.ptr;
      for (int f = 0; f < clip_t; f++) {
        int sf = f < avail_frames ? f : avail_frames - 1;
        const uint8_t *fr = rgb_start + (size_t)sf * frame_bytes;
        for (int y = 0; y < th; y++) {
          const uint8_t *sr = fr + ((size_t)(yp.start[i] + y) * (size_t)width +
                                    (size_t)xp.start[j]) *
                                       (size_t)MINIMAX_H3_RGB_CHANNELS;
          float *dr = dst + ((size_t)f * (size_t)th + (size_t)y) * (size_t)tw *
                                (size_t)MINIMAX_H3_RGB_CHANNELS;
          for (int x = 0; x < tw; x++) {
            for (int c = 0; c < MINIMAX_H3_RGB_CHANNELS; c++) {
              float v = (float)sr[x * MINIMAX_H3_RGB_CHANNELS + c] /
                        MINIMAX_H3_PIXEL_MAX;
              dr[x * MINIMAX_H3_RGB_CHANNELS + c] =
                  (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
          }
        }
      }
      size_t tile_elems = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                          (size_t)t_lat * (size_t)th_lat * (size_t)tw_lat;
      raw_tiles[i * xp.count + j] = xmalloc(tile_elems * sizeof(float));
      if (!encode_tile_3d(vae, ws, clip_t, th, tw, raw_tiles[i * xp.count + j],
                          hooks)) {
        ok = 0;
        break;
      }
    }
  }

  if (ok) {
    // Spatial blend in latent space matching comfy/ldm/minimax/vae.py:790-807.
    size_t out_ct = (size_t)t_lat * (size_t)h_lat * (size_t)w_lat;
    int out_y = 0;
    for (int i = 0; i < yp.count; i++) {
      int th_lat = yp.len[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int ov_top = i > 0 ? yp.overlap[i - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
      int ov_bot =
          i < yp.count - 1 ? yp.overlap[i] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
      int keep_h = th_lat - ov_bot;
      int out_x = 0;

      for (int j = 0; j < xp.count; j++) {
        int tw_lat = xp.len[j] / MINIMAX_H3_VAE_SPATIAL_RATIO;
        int ov_left =
            j > 0 ? xp.overlap[j - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
        int ov_right =
            j < xp.count - 1 ? xp.overlap[j] / MINIMAX_H3_VAE_SPATIAL_RATIO : 0;
        int keep_w = tw_lat - ov_right;

        size_t tile_ct = (size_t)t_lat * (size_t)th_lat * (size_t)tw_lat;
        float *cur = xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * tile_ct *
                             sizeof(float));
        memcpy(cur, raw_tiles[i * xp.count + j],
               (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * tile_ct *
                   sizeof(float));

        if (ov_top > 0) {
          int prev_h = yp.len[i - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO;
          const float *above = raw_tiles[(i - 1) * xp.count + j];
          size_t prev_ct = (size_t)t_lat * (size_t)prev_h * (size_t)tw_lat;
          for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
            for (int pt = 0; pt < t_lat; pt++) {
              for (int y = 0; y < ov_top; y++) {
                float wb = (float)y / (float)ov_top, wa = 1.0f - wb;
                const float *ap = above + (size_t)c * prev_ct +
                                  ((size_t)pt * (size_t)prev_h +
                                   (size_t)(prev_h - ov_top + y)) *
                                      (size_t)tw_lat;
                float *bp =
                    cur + (size_t)c * tile_ct +
                    ((size_t)pt * (size_t)th_lat + (size_t)y) * (size_t)tw_lat;
                for (int x = 0; x < tw_lat; x++)
                  bp[x] = ap[x] * wa + bp[x] * wb;
              }
            }
          }
        }
        if (ov_left > 0) {
          int prev_w = xp.len[j - 1] / MINIMAX_H3_VAE_SPATIAL_RATIO;
          const float *left = raw_tiles[i * xp.count + (j - 1)];
          size_t prev_ct = (size_t)t_lat * (size_t)th_lat * (size_t)prev_w;
          for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
            for (int pt = 0; pt < t_lat; pt++) {
              for (int y = 0; y < th_lat; y++) {
                const float *ap =
                    left + (size_t)c * prev_ct +
                    ((size_t)pt * (size_t)th_lat + (size_t)y) * (size_t)prev_w +
                    (size_t)(prev_w - ov_left);
                float *bp =
                    cur + (size_t)c * tile_ct +
                    ((size_t)pt * (size_t)th_lat + (size_t)y) * (size_t)tw_lat;
                for (int x = 0; x < ov_left; x++) {
                  float wb = (float)x / (float)ov_left, wa = 1.0f - wb;
                  bp[x] = ap[x] * wa + bp[x] * wb;
                }
              }
            }
          }
        }

        for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
          for (int pt = 0; pt < t_lat; pt++) {
            for (int y = 0; y < keep_h; y++) {
              const float *sp =
                  cur + (size_t)c * tile_ct +
                  ((size_t)pt * (size_t)th_lat + (size_t)y) * (size_t)tw_lat;
              float *dp = clip_out + (size_t)c * out_ct +
                          ((size_t)pt * (size_t)h_lat + (size_t)(out_y + y)) *
                              (size_t)w_lat +
                          (size_t)out_x;
              memcpy(dp, sp, (size_t)keep_w * sizeof(float));
            }
          }
        }
        free(cur);
        out_x += keep_w;
      }
      out_y += keep_h;
    }
  }

  for (int k = 0; k < n_tiles; k++) free(raw_tiles[k]);
  free(raw_tiles);
  return ok;
}

int minimax_h3_video_vae_encode(MinimaxH3VideoVae *vae, const uint8_t *rgb,
                                int frame_count, int width, int height,
                                float *out_latents,
                                const MinimaxH3RunHooks *hooks) {
  if (!vae || !rgb || !out_latents || frame_count <= 0 || width <= 0 ||
      height <= 0 || width % MINIMAX_H3_VAE_SPATIAL_RATIO != 0 ||
      height % MINIMAX_H3_VAE_SPATIAL_RATIO != 0)
    return 0;
  if (!minimax_h3_video_vae_load_encoder(vae)) return 0;
  if (minimax_h3_should_stop(hooks)) return 0;

  MinimaxH3VaeTilePlan yp = minimax_h3_image_vae_split_tiles(height);
  MinimaxH3VaeTilePlan xp = minimax_h3_image_vae_split_tiles(width);
  int clip_t = frame_count == 1 ? 1 : MINIMAX_H3_VVAE_CLIP_LENGTH;
  int h_lat = height / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = width / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int total_t = minimax_h3_video_vae_encode_latent_t(frame_count);

  VvaeEncodeWorkspace ws = {0};
  if (!alloc_encode_ws(vae, &ws, clip_t, yp.len[0], xp.len[0])) return 0;

  size_t hw_lat = (size_t)h_lat * (size_t)w_lat;
  size_t frame_bytes =
      (size_t)height * (size_t)width * (size_t)MINIMAX_H3_RGB_CHANNELS;
  int ok = 1;

  if (frame_count == 1) {
    ok = encode_clip_tiled(vae, &ws, rgb, 1, 1, width, height, out_latents,
                           hooks);
    if (ok) {
      for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
        float m = vae->latents_mean[c], inv_s = 1.0f / vae->latents_std[c];
        float *zp = out_latents + (size_t)c * hw_lat;
        for (size_t p = 0; p < hw_lat; p++) zp[p] = (zp[p] - m) * inv_s;
      }
    }
    free_encode_ws(&ws);
    return ok;
  }

  int clips = (frame_count + MINIMAX_H3_VVAE_CLIP_LENGTH - 1) /
              MINIMAX_H3_VVAE_CLIP_LENGTH;
  float *clip_z =
      xmalloc((size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
              (size_t)MINIMAX_H3_VVAE_CHUNK_TOKENS * hw_lat * sizeof(float));
  for (int ci = 0; ci < clips && ok; ci++) {
    int f0 = ci * MINIMAX_H3_VVAE_CLIP_LENGTH;
    int avail = frame_count - f0;
    if (avail > MINIMAX_H3_VVAE_CLIP_LENGTH)
      avail = MINIMAX_H3_VVAE_CLIP_LENGTH;
    if (!encode_clip_tiled(vae, &ws, rgb + (size_t)f0 * frame_bytes, avail,
                           MINIMAX_H3_VVAE_CLIP_LENGTH, width, height, clip_z,
                           hooks)) {
      ok = 0;
      break;
    }
    // Append clip latents and normalize, dropping the trailing token_drop=3.
    int dst_t0 = ci * MINIMAX_H3_VVAE_CHUNK_TOKENS;
    int keep_t = MINIMAX_H3_VVAE_CHUNK_TOKENS;
    if (dst_t0 + keep_t > total_t) keep_t = total_t - dst_t0;
    for (int c = 0; c < MINIMAX_H3_VAE_LATENT_CHANNELS; c++) {
      float m = vae->latents_mean[c], inv_s = 1.0f / vae->latents_std[c];
      for (int dt = 0; dt < keep_t; dt++) {
        const float *src =
            clip_z +
            ((size_t)c * (size_t)MINIMAX_H3_VVAE_CHUNK_TOKENS + (size_t)dt) *
                hw_lat;
        float *dst =
            out_latents +
            ((size_t)c * (size_t)total_t + (size_t)(dst_t0 + dt)) * hw_lat;
        for (size_t p = 0; p < hw_lat; p++) dst[p] = (src[p] - m) * inv_s;
      }
    }
  }

  free(clip_z);
  free_encode_ws(&ws);
  return ok;
}
