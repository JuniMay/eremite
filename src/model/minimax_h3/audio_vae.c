#include "model/minimax_h3/audio_vae.h"
#include "base/error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BigVGAN decoder and DAC encoder constants from Comfy-Org/ComfyUI
// comfy/ldm/minimax/audio_vae.py:1256-1515.
static constexpr int AUDIO_LATENT_DIM = 2048, AUDIO_DEC_INIT_CH = 1024,
                     AUDIO_ENC_INIT_CH = 64, AUDIO_UP_STAGES = 7,
                     AUDIO_RES_KERNELS = 3, AUDIO_RES_DILS = 3,
                     AUDIO_RES_ACTS = 2 * AUDIO_RES_DILS,
                     AUDIO_RES_BLOCKS = AUDIO_UP_STAGES * AUDIO_RES_KERNELS,
                     AUDIO_ENC_STAGES = 5, AUDIO_ENC_RES_UNITS = 3,
                     AUDIO_ACT_FILTER_K = 12, AUDIO_ATTN_MLP_HIDDEN = 64,
                     AUDIO_CONV_PRE_K = 7, AUDIO_CONV_PRE_PAD = 3,
                     AUDIO_CONV_POST_K = 7, AUDIO_CONV_POST_PAD = 3,
                     AUDIO_ENC_IN_K = 7, AUDIO_ENC_IN_PAD = 3,
                     AUDIO_ENC_RES_K = 7, AUDIO_ENC_OUT_K = 3,
                     AUDIO_ENC_OUT_PAD = 1, AUDIO_CONV_K1 = 1,
                     AUDIO_CONV_RANK = 3, AUDIO_LIN_RANK = 2,
                     AUDIO_QKV_WIDTH = MINIMAX_H3_QKV_PARTS * AUDIO_LATENT_DIM;

// LayerNorm epsilon in encoder AttnProjection and GeGluMlp (PyTorch default).
static constexpr float AUDIO_NORM_EPS = 1e-5f;

// Minimum tensor byte size for sector-aligned uncached weight reads.
static constexpr uint64_t ALIGN_READ_MIN_BYTES = 65536;

// BigVGAN upsampling factors, transposed-conv kernels and paddings
// (audio_vae.py:1419-1440).
static const int UP_RATES[AUDIO_UP_STAGES] = {5, 5, 2, 2, 2, 2, 2};
static const int UP_KERNELS[AUDIO_UP_STAGES] = {9, 9, 4, 4, 4, 4, 4};
static const int UP_PADS[AUDIO_UP_STAGES] = {2, 2, 1, 1, 1, 1, 1};

// BigVGAN AMPBlock1 kernel sizes and dilations (audio_vae.py:1421-1422).
static const int RES_KERNELS[AUDIO_RES_KERNELS] = {3, 7, 11};
static const int RES_DILATIONS[AUDIO_RES_DILS] = {1, 3, 5};

// DAC encoder downsampling strides and ResidualUnit dilations
// (audio_vae.py:1279-1281, 1297).
static const int ENC_RATES[AUDIO_ENC_STAGES] = {2, 4, 4, 5, 5};
static const int ENC_DILATIONS[AUDIO_ENC_RES_UNITS] = {1, 3, 9};

// Peak channel*length element count per latent frame in decode (256*25 = 6400)
// and encode (64*800 = 51200) workspaces, plus buffer counts.
static constexpr uint64_t DEC_MAX_ELEMS_PER_FRAME = 6400,
                          ENC_MAX_ELEMS_PER_FRAME = 51200;
static constexpr int DEC_WORK_BUFS = 6, ENC_WORK_BUFS = 3, ENC_SMALL_BUFS = 4;

// Metal kernel dispatch parameters matching
// src/kernels/minimax_h3_audio_vae.metal.
static constexpr uint32_t SIMD_W = 32, CONV_TILE_CO = 32, CONV_THREADS = 128,
                          ACT_TILE_T = 256, ELEM_THREADS = 64, TRANS_TILE = 16,
                          NORM_THREADS_WIDE = 256, NORM_THREADS_NARROW = 32,
                          ATTN_THREADS = 256;

// Parameters passed to minimax_h3_audio_conv1d.
typedef struct {
  // Channel counts, lengths, kernel width, stride, padding and dilation.
  uint32_t c_in, c_out, l_in, l_out, k, stride, pad, dilation;
  // Bias flag, residual flag, residual crop and output clamp flag.
  uint32_t has_bias, add_res, res_crop, clamp_out;
} AudioConv1dParam;

// Parameters passed to minimax_h3_audio_conv_transpose1d.
typedef struct {
  // Channel counts, lengths, kernel width, stride and padding.
  uint32_t c_in, c_out, l_in, l_out, k, stride, pad;
} AudioConvT1dParam;

// Parameters passed to minimax_h3_audio_snake1d and act1d_snakebeta.
typedef struct {
  // Channel count and sequence length.
  uint32_t channels, length;
} AudioChLenParam;

// Parameters passed to minimax_h3_audio_affine_ch.
typedef struct {
  // Channel count, sequence length and invert flag (0 denorm, 1 norm).
  uint32_t channels, length, invert;
} AudioAffineParam;

// Parameters passed to minimax_h3_audio_transpose.
typedef struct {
  // Input row and column counts.
  uint32_t rows, cols;
} AudioTransposeParam;

// Parameters passed to minimax_h3_audio_linear_f32.
typedef struct {
  // Row count, output width, input width, bias flag and residual flag.
  uint32_t m, n, k, has_bias, add_res;
} AudioLinearParam;

// Parameters passed to minimax_h3_audio_geglu_tanh.
typedef struct {
  // Element count.
  uint32_t n;
} AudioGeGluParam;

// Parameters passed to minimax_h3_audio_causal_attn_pool.
typedef struct {
  // Sequence length.
  uint32_t seq_len;
} AudioCausalAttnParam;

// Parameters passed to minimax_h3_layernorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} AudioLayerNormParam;

// FP32 1D convolution weights and optional bias.
typedef struct {
  // Convolution weight [c_out, c_in, k] (or [c_in, c_out, k] for transpose).
  Buf w;
  // Optional bias [c_out].
  Buf b;
  // Channel counts, kernel width, stride, padding, dilation and bias flag.
  int c_in, c_out, k, stride, pad, dilation, has_bias;
} AudioConv;

// BigVGAN anti-aliased Activation1d(SnakeBeta) parameters.
typedef struct {
  // Log-scale SnakeBeta frequency and magnitude parameters [channels].
  Buf alpha, beta;
  // 12-tap Kaiser-sinc upsample and downsample filters [1, 1, 12].
  Buf f_up, f_down;
} AudioAct1d;

// BigVGAN AMPBlock1 residual block weights.
typedef struct {
  // Three dilated convolutions and three unit-dilation convolutions.
  AudioConv convs1[AUDIO_RES_DILS], convs2[AUDIO_RES_DILS];
  // Six anti-aliased SnakeBeta activations paired with convs1 and convs2.
  AudioAct1d acts[AUDIO_RES_ACTS];
} AudioAmpBlock;

// DAC encoder dilated ResidualUnit weights.
typedef struct {
  // Per-channel Snake1d frequency parameters before each convolution.
  Buf alpha1, alpha2;
  // Dilated k=7 convolution and pointwise k=1 projection.
  AudioConv conv1, conv2;
} AudioEncResUnit;

// DAC encoder downsampling stage weights.
typedef struct {
  // Three residual units with dilations 1, 3 and 9.
  AudioEncResUnit units[AUDIO_ENC_RES_UNITS];
  // Snake1d parameter before strided downsampling.
  Buf alpha;
  // Strided downsampling convolution.
  AudioConv down;
} AudioEncStage;

// FP32 affine LayerNorm weight and bias pair.
typedef struct {
  // Scale weight and bias vectors.
  Buf w, b;
} AudioNorm;

// FP32 linear projection weight and bias pair.
typedef struct {
  // Row-major weight [out_dim, in_dim] and bias [out_dim].
  Buf w, b;
  // Output and input widths.
  int n, k;
} AudioLinear;

// Three checkpoint bias tensors concatenated into AttnProjection's QKV bias.
typedef struct {
  // Checkpoint tensors for q_bias, zero_k_bias and v_bias.
  Tensor q_bias, k_bias, v_bias;
} QkvBiasParam;

// Loaded MiniMax-H3 Audio VAE state.
struct MinimaxH3AudioVae {
  // GPU device and memory account.
  Gpu *gpu;
  // Memory account owning weight and workspace slots.
  MemoryAccount *account;
  // Automatic workspace memory slot for transient decode/encode buffers.
  MemorySlot workspace_slot;
  // Open safetensors index for the audio VAE file.
  TensorIndex *tensors;
  // Decoder and encoder weight groups.
  WeightGroup *dec_weights, *enc_weights;

  // Decoder latent normalization statistics [32].
  Buf dec_mean, dec_std;
  // Decoder input projection (32 -> 2048, k=1) and BigVGAN conv_pre.
  AudioConv dec_in_proj, conv_pre;
  // Seven BigVGAN transposed-convolution upsampling stages.
  AudioConv ups[AUDIO_UP_STAGES];
  // Twenty-one BigVGAN AMPBlock1 residual blocks (7 stages x 3 kernels).
  AudioAmpBlock resblocks[AUDIO_RES_BLOCKS];
  // Final anti-aliased activation and bias-free waveform projection.
  AudioAct1d act_post;
  // Final bias-free convolution producing mono waveform [1, L].
  AudioConv conv_post;

  // Encoder latent normalization statistics [32].
  Buf enc_mean, enc_std;
  // DAC initial conv, 5 downsampling stages, final Snake1d and final conv.
  AudioConv enc_in;
  // Five DAC encoder downsampling stages.
  AudioEncStage enc_stages[AUDIO_ENC_STAGES];
  // Final DAC encoder Snake1d parameter [1, 2048, 1].
  Buf enc_out_alpha;
  // Final DAC encoder convolution (2048 -> 2048, k=3).
  AudioConv enc_out;
  // Encoder AttnProjection norms, linear layers and GeGLU MLP.
  AudioNorm norm1, norm2, norm3, mlp_norm;
  // Encoder AttnProjection QKV, attention output, skip and MLP linears.
  AudioLinear attn_qkv, attn_proj, skip_proj, mlp_w0, mlp_w1, mlp_w2;
  // Posterior mean pointwise projection (32 -> 32, k=1).
  AudioConv mean_proj;

  // Cached Metal kernel indices.
  int k_conv1d, k_conv_t1d, k_snake1d, k_act1d, k_avg3, k_affine, k_transpose,
      k_linear, k_geglu, k_causal_attn, k_layernorm;
};

// Returns ceil(a / b) for positive integers.
static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Adds an FP32 tensor verified to match expected element count to group.
static void add_f32_tensor(WeightGroup *group, Buf *dst, Tensor t,
                           uint64_t expected_elems, const char *label) {
  if (t.dtype != 'f' || t.size != expected_elems * sizeof(float))
    die("unexpected FP32 tensor %s (dtype %c, size %llu, expected %llu)", label,
        t.dtype, (unsigned long long)t.size,
        (unsigned long long)(expected_elems * sizeof(float)));
  weight_group_add_tensor(group, dst, t, t.size >= ALIGN_READ_MIN_BYTES);
}

// Registers one FP32 1D convolution (and optional bias) into group.
static void add_conv1d(WeightGroup *group, const TensorIndex *tensors,
                       AudioConv *conv, int c_in, int c_out, int k, int stride,
                       int pad, int dilation, int has_bias, int is_transpose,
                       const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  int d0 = is_transpose ? c_in : c_out;
  int d1 = is_transpose ? c_out : c_in;
  if (w.dtype != 'f' || w.ndim != AUDIO_CONV_RANK || w.shape[0] != d0 ||
      w.shape[1] != d1 || w.shape[2] != k)
    die("unexpected audio conv weight shape for %s", prefix);
  conv->c_in = c_in;
  conv->c_out = c_out;
  conv->k = k;
  conv->stride = stride;
  conv->pad = pad;
  conv->dilation = dilation;
  conv->has_bias = has_bias;
  weight_group_add_tensor(group, &conv->w, w, w.size >= ALIGN_READ_MIN_BYTES);
  if (has_bias) {
    Tensor b = tensor_index_get(tensors, "%s.bias", prefix);
    if (b.dtype != 'f' || b.ndim != 1 || b.shape[0] != c_out)
      die("unexpected audio conv bias shape for %s", prefix);
    weight_group_add_tensor(group, &conv->b, b, 0);
  }
}

// Registers one BigVGAN Activation1d(SnakeBeta) into group.
static void add_act1d(WeightGroup *group, const TensorIndex *tensors,
                      AudioAct1d *act, int channels, const char *prefix) {
  add_f32_tensor(group, &act->alpha,
                 tensor_index_get(tensors, "%s.act.alpha", prefix),
                 (uint64_t)channels, prefix);
  add_f32_tensor(group, &act->beta,
                 tensor_index_get(tensors, "%s.act.beta", prefix),
                 (uint64_t)channels, prefix);
  add_f32_tensor(group, &act->f_up,
                 tensor_index_get(tensors, "%s.upsample.filter", prefix),
                 (uint64_t)AUDIO_ACT_FILTER_K, prefix);
  add_f32_tensor(
      group, &act->f_down,
      tensor_index_get(tensors, "%s.downsample.lowpass.filter", prefix),
      (uint64_t)AUDIO_ACT_FILTER_K, prefix);
}

// Registers one FP32 affine LayerNorm into group.
static void add_norm(WeightGroup *group, const TensorIndex *tensors,
                     AudioNorm *norm, int dim, const char *prefix) {
  add_f32_tensor(group, &norm->w,
                 tensor_index_get(tensors, "%s.weight", prefix), (uint64_t)dim,
                 prefix);
  add_f32_tensor(group, &norm->b, tensor_index_get(tensors, "%s.bias", prefix),
                 (uint64_t)dim, prefix);
}

// Registers one FP32 linear projection (weight and bias) into group.
static void add_linear(WeightGroup *group, const TensorIndex *tensors,
                       AudioLinear *lin, int n, int k, const char *prefix) {
  Tensor w = tensor_index_get(tensors, "%s.weight", prefix);
  if (w.dtype != 'f' || w.ndim != AUDIO_LIN_RANK || w.shape[0] != n ||
      w.shape[1] != k)
    die("unexpected audio linear weight shape for %s", prefix);
  lin->n = n;
  lin->k = k;
  weight_group_add_tensor(group, &lin->w, w, w.size >= ALIGN_READ_MIN_BYTES);
  add_f32_tensor(group, &lin->b, tensor_index_get(tensors, "%s.bias", prefix),
                 (uint64_t)n, prefix);
}

// Concatenates q_bias, zero_k_bias and v_bias into one contiguous FP32 buffer.
static void fill_qkv_bias(Buf *buffer, const void *context) {
  const QkvBiasParam *p = context;
  uint8_t *dst = buffer->ptr;
  size_t part = (size_t)AUDIO_LATENT_DIM * sizeof(float);
  minimax_h3_read_tensor_bytes(p->q_bias, dst);
  minimax_h3_read_tensor_bytes(p->k_bias, dst + part);
  minimax_h3_read_tensor_bytes(p->v_bias, dst + 2u * part);
}

// Encodes one FP32 1D convolution dispatch into cmd.
static void enc_conv1d(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                       const AudioConv *conv, const Buf *x, uint32_t l_in,
                       const Buf *res, const Buf *y, uint32_t l_out,
                       int clamp_out) {
  AudioConv1dParam p = {
      .c_in = (uint32_t)conv->c_in,
      .c_out = (uint32_t)conv->c_out,
      .l_in = l_in,
      .l_out = l_out,
      .k = (uint32_t)conv->k,
      .stride = (uint32_t)conv->stride,
      .pad = (uint32_t)conv->pad,
      .dilation = (uint32_t)conv->dilation,
      .has_bias = (uint32_t)conv->has_bias,
      .add_res = res != NULL ? 1u : 0u,
      .res_crop = 0u,
      .clamp_out = (uint32_t)clamp_out,
  };
  const Buf *bias_buf = conv->has_bias ? &conv->b : x;
  const Buf *res_buf = res != NULL ? res : x;
  gpu_select_kernel(cmd, vae->k_conv1d);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &conv->w, 0);
  gpu_bind_buffer(cmd, 2, bias_buf, 0);
  gpu_bind_buffer(cmd, 3, res_buf, 0);
  gpu_bind_buffer(cmd, 4, y, 0);
  gpu_copy_bytes(cmd, 5, &p, sizeof p);
  gpu_run_kernel(cmd, ceil_div(l_out, SIMD_W),
                 ceil_div((uint32_t)conv->c_out, CONV_TILE_CO), 1, CONV_THREADS,
                 1, 1);
}

// Encodes one BigVGAN transposed 1D convolution dispatch into cmd.
static void enc_conv_t1d(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                         const AudioConv *conv, const Buf *x, uint32_t l_in,
                         const Buf *y, uint32_t l_out) {
  AudioConvT1dParam p = {
      .c_in = (uint32_t)conv->c_in,
      .c_out = (uint32_t)conv->c_out,
      .l_in = l_in,
      .l_out = l_out,
      .k = (uint32_t)conv->k,
      .stride = (uint32_t)conv->stride,
      .pad = (uint32_t)conv->pad,
  };
  uint32_t t_slots = ceil_div(l_out, (uint32_t)conv->stride);
  gpu_select_kernel(cmd, vae->k_conv_t1d);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &conv->w, 0);
  gpu_bind_buffer(cmd, 2, &conv->b, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &p, sizeof p);
  gpu_run_kernel(cmd, ceil_div(t_slots, SIMD_W),
                 ceil_div((uint32_t)conv->c_out, CONV_TILE_CO),
                 (uint32_t)conv->stride, CONV_THREADS, 1, 1);
}

// Encodes one BigVGAN anti-aliased Activation1d(SnakeBeta) dispatch into cmd.
static void enc_act1d(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                      const AudioAct1d *act, const Buf *x, const Buf *y,
                      uint32_t channels, uint32_t length) {
  AudioChLenParam p = {channels, length};
  gpu_select_kernel(cmd, vae->k_act1d);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &act->alpha, 0);
  gpu_bind_buffer(cmd, 2, &act->beta, 0);
  gpu_bind_buffer(cmd, 3, &act->f_up, 0);
  gpu_bind_buffer(cmd, 4, &act->f_down, 0);
  gpu_bind_buffer(cmd, 5, y, 0);
  gpu_copy_bytes(cmd, 6, &p, sizeof p);
  gpu_run_kernel(cmd, ceil_div(length, ACT_TILE_T), channels, 1, ACT_TILE_T, 1,
                 1);
}

// Encodes one DAC encoder Snake1d dispatch into cmd.
static void enc_snake1d(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                        const Buf *alpha, const Buf *x, const Buf *y,
                        uint32_t channels, uint32_t length) {
  AudioChLenParam p = {channels, length};
  gpu_select_kernel(cmd, vae->k_snake1d);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, alpha, 0);
  gpu_bind_buffer(cmd, 2, y, 0);
  gpu_copy_bytes(cmd, 3, &p, sizeof p);
  gpu_run_kernel(cmd, ceil_div(length, ELEM_THREADS), channels, 1, ELEM_THREADS,
                 1, 1);
}

// Encodes one per-channel affine norm/denorm dispatch into cmd.
static void enc_affine_ch(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                          const Buf *x, const Buf *mean, const Buf *std,
                          const Buf *y, uint32_t channels, uint32_t length,
                          uint32_t invert) {
  AudioAffineParam p = {channels, length, invert};
  gpu_select_kernel(cmd, vae->k_affine);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, mean, 0);
  gpu_bind_buffer(cmd, 2, std, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &p, sizeof p);
  gpu_run_kernel(cmd, ceil_div(length, ELEM_THREADS), channels, 1, ELEM_THREADS,
                 1, 1);
}

// Encodes one FP32 LayerNorm dispatch into cmd.
static void enc_layernorm(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                          const AudioNorm *norm, const Buf *x, const Buf *y,
                          uint32_t rows, uint32_t dim) {
  AudioLayerNormParam p = {dim, AUDIO_NORM_EPS};
  uint32_t nt =
      dim >= NORM_THREADS_WIDE ? NORM_THREADS_WIDE : NORM_THREADS_NARROW;
  gpu_select_kernel(cmd, vae->k_layernorm);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &norm->w, 0);
  gpu_bind_buffer(cmd, 2, &norm->b, 0);
  gpu_bind_buffer(cmd, 3, y, 0);
  gpu_copy_bytes(cmd, 4, &p, sizeof p);
  gpu_run_kernel(cmd, rows, 1, 1, nt, 1, 1);
}

// Encodes one FP32 row-major linear projection dispatch into cmd.
static void enc_linear(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                       const AudioLinear *lin, const Buf *x, const Buf *res,
                       const Buf *y, uint32_t rows) {
  AudioLinearParam p = {
      .m = rows,
      .n = (uint32_t)lin->n,
      .k = (uint32_t)lin->k,
      .has_bias = 1u,
      .add_res = res != NULL ? 1u : 0u,
  };
  const Buf *res_buf = res != NULL ? res : x;
  gpu_select_kernel(cmd, vae->k_linear);
  gpu_bind_buffer(cmd, 0, x, 0);
  gpu_bind_buffer(cmd, 1, &lin->w, 0);
  gpu_bind_buffer(cmd, 2, &lin->b, 0);
  gpu_bind_buffer(cmd, 3, res_buf, 0);
  gpu_bind_buffer(cmd, 4, y, 0);
  gpu_copy_bytes(cmd, 5, &p, sizeof p);
  gpu_run_kernel(cmd, (uint32_t)lin->n, rows, 1, SIMD_W, 1, 1);
}

// Registers all BigVGAN decoder weights into vae->dec_weights.
static void register_decoder_weights(MinimaxH3AudioVae *vae) {
  add_f32_tensor(vae->dec_weights, &vae->dec_mean,
                 tensor_index_get(vae->tensors, "latents_mean"),
                 (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS, "latents_mean");
  add_f32_tensor(vae->dec_weights, &vae->dec_std,
                 tensor_index_get(vae->tensors, "latents_std"),
                 (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS, "latents_std");
  add_conv1d(vae->dec_weights, vae->tensors, &vae->dec_in_proj,
             MINIMAX_H3_AUDIO_LATENT_CHANNELS, AUDIO_LATENT_DIM, AUDIO_CONV_K1,
             1, 0, 1, 1, 0, "dec_in_proj");
  add_conv1d(vae->dec_weights, vae->tensors, &vae->conv_pre, AUDIO_LATENT_DIM,
             AUDIO_DEC_INIT_CH, AUDIO_CONV_PRE_K, 1, AUDIO_CONV_PRE_PAD, 1, 1,
             0, "decoder.conv_pre");

  char name[MINIMAX_H3_NAME_CAP];
  for (int i = 0; i < AUDIO_UP_STAGES; i++) {
    int c_in = AUDIO_DEC_INIT_CH >> i;
    int c_out = AUDIO_DEC_INIT_CH >> (i + 1);
    snprintf(name, sizeof name, "decoder.ups.%d.0", i);
    add_conv1d(vae->dec_weights, vae->tensors, &vae->ups[i], c_in, c_out,
               UP_KERNELS[i], UP_RATES[i], UP_PADS[i], 1, 1, 1, name);
    for (int j = 0; j < AUDIO_RES_KERNELS; j++) {
      int r = i * AUDIO_RES_KERNELS + j;
      int rk = RES_KERNELS[j];
      AudioAmpBlock *blk = &vae->resblocks[r];
      for (int d = 0; d < AUDIO_RES_DILS; d++) {
        int dil = RES_DILATIONS[d];
        int pad1 = (rk * dil - dil) / 2;
        int pad2 = (rk - 1) / 2;
        snprintf(name, sizeof name, "decoder.resblocks.%d.convs1.%d", r, d);
        add_conv1d(vae->dec_weights, vae->tensors, &blk->convs1[d], c_out,
                   c_out, rk, 1, pad1, dil, 1, 0, name);
        snprintf(name, sizeof name, "decoder.resblocks.%d.convs2.%d", r, d);
        add_conv1d(vae->dec_weights, vae->tensors, &blk->convs2[d], c_out,
                   c_out, rk, 1, pad2, 1, 1, 0, name);
      }
      for (int a = 0; a < AUDIO_RES_ACTS; a++) {
        snprintf(name, sizeof name, "decoder.resblocks.%d.activations.%d", r,
                 a);
        add_act1d(vae->dec_weights, vae->tensors, &blk->acts[a], c_out, name);
      }
    }
  }
  int final_ch = AUDIO_DEC_INIT_CH >> AUDIO_UP_STAGES;
  add_act1d(vae->dec_weights, vae->tensors, &vae->act_post, final_ch,
            "decoder.activation_post");
  add_conv1d(vae->dec_weights, vae->tensors, &vae->conv_post, final_ch, 1,
             AUDIO_CONV_POST_K, 1, AUDIO_CONV_POST_PAD, 1, 0, 0,
             "decoder.conv_post");
}

// Registers all DAC encoder and AttnProjection weights into vae->enc_weights.
static void register_encoder_weights(MinimaxH3AudioVae *vae) {
  add_f32_tensor(vae->enc_weights, &vae->enc_mean,
                 tensor_index_get(vae->tensors, "latents_mean"),
                 (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS, "latents_mean");
  add_f32_tensor(vae->enc_weights, &vae->enc_std,
                 tensor_index_get(vae->tensors, "latents_std"),
                 (uint64_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS, "latents_std");
  add_conv1d(vae->enc_weights, vae->tensors, &vae->enc_in, 1, AUDIO_ENC_INIT_CH,
             AUDIO_ENC_IN_K, 1, AUDIO_ENC_IN_PAD, 1, 1, 0, "encoder.block.0");

  char name[MINIMAX_H3_NAME_CAP];
  int ch = AUDIO_ENC_INIT_CH;
  for (int s = 0; s < AUDIO_ENC_STAGES; s++) {
    int next_ch = ch * 2;
    int stride = ENC_RATES[s];
    int blk_idx = s + 1;
    AudioEncStage *st = &vae->enc_stages[s];
    for (int u = 0; u < AUDIO_ENC_RES_UNITS; u++) {
      int dil = ENC_DILATIONS[u];
      int pad = ((AUDIO_ENC_RES_K - 1) * dil) / 2;
      AudioEncResUnit *ru = &st->units[u];
      snprintf(name, sizeof name, "encoder.block.%d.block.%d.block.0.alpha",
               blk_idx, u);
      add_f32_tensor(vae->enc_weights, &ru->alpha1,
                     tensor_index_get(vae->tensors, "%s", name), (uint64_t)ch,
                     name);
      snprintf(name, sizeof name, "encoder.block.%d.block.%d.block.1", blk_idx,
               u);
      add_conv1d(vae->enc_weights, vae->tensors, &ru->conv1, ch, ch,
                 AUDIO_ENC_RES_K, 1, pad, dil, 1, 0, name);
      snprintf(name, sizeof name, "encoder.block.%d.block.%d.block.2.alpha",
               blk_idx, u);
      add_f32_tensor(vae->enc_weights, &ru->alpha2,
                     tensor_index_get(vae->tensors, "%s", name), (uint64_t)ch,
                     name);
      snprintf(name, sizeof name, "encoder.block.%d.block.%d.block.3", blk_idx,
               u);
      add_conv1d(vae->enc_weights, vae->tensors, &ru->conv2, ch, ch,
                 AUDIO_CONV_K1, 1, 0, 1, 1, 0, name);
    }
    snprintf(name, sizeof name, "encoder.block.%d.block.3.alpha", blk_idx);
    add_f32_tensor(vae->enc_weights, &st->alpha,
                   tensor_index_get(vae->tensors, "%s", name), (uint64_t)ch,
                   name);
    snprintf(name, sizeof name, "encoder.block.%d.block.4", blk_idx);
    add_conv1d(vae->enc_weights, vae->tensors, &st->down, ch, next_ch,
               2 * stride, stride, (stride + 1) / 2, 1, 1, 0, name);
    ch = next_ch;
  }
  add_f32_tensor(vae->enc_weights, &vae->enc_out_alpha,
                 tensor_index_get(vae->tensors, "encoder.block.6.alpha"),
                 (uint64_t)ch, "encoder.block.6.alpha");
  add_conv1d(vae->enc_weights, vae->tensors, &vae->enc_out, ch,
             AUDIO_LATENT_DIM, AUDIO_ENC_OUT_K, 1, AUDIO_ENC_OUT_PAD, 1, 1, 0,
             "encoder.block.7");

  add_norm(vae->enc_weights, vae->tensors, &vae->norm1, AUDIO_LATENT_DIM,
           "pre_block.norm1");
  add_norm(vae->enc_weights, vae->tensors, &vae->norm2,
           MINIMAX_H3_AUDIO_LATENT_CHANNELS, "pre_block.norm2");
  add_norm(vae->enc_weights, vae->tensors, &vae->norm3, AUDIO_LATENT_DIM,
           "pre_block.norm3");
  add_norm(vae->enc_weights, vae->tensors, &vae->mlp_norm,
           MINIMAX_H3_AUDIO_LATENT_CHANNELS, "pre_block.mlp.norm");

  Tensor qkv_w = tensor_index_get(vae->tensors, "pre_block.attn.qkv.weight");
  if (qkv_w.dtype != 'f' || qkv_w.ndim != AUDIO_LIN_RANK ||
      qkv_w.shape[0] != AUDIO_QKV_WIDTH || qkv_w.shape[1] != AUDIO_LATENT_DIM)
    die("unexpected pre_block.attn.qkv.weight shape");
  vae->attn_qkv.n = AUDIO_QKV_WIDTH;
  vae->attn_qkv.k = AUDIO_LATENT_DIM;
  weight_group_add_tensor(vae->enc_weights, &vae->attn_qkv.w, qkv_w, 1);
  QkvBiasParam qbp = {
      .q_bias = tensor_index_get(vae->tensors, "pre_block.attn.q_bias"),
      .k_bias = tensor_index_get(vae->tensors, "pre_block.attn.zero_k_bias"),
      .v_bias = tensor_index_get(vae->tensors, "pre_block.attn.v_bias"),
  };
  if (qbp.q_bias.size != (uint64_t)AUDIO_LATENT_DIM * sizeof(float) ||
      qbp.k_bias.size != (uint64_t)AUDIO_LATENT_DIM * sizeof(float) ||
      qbp.v_bias.size != (uint64_t)AUDIO_LATENT_DIM * sizeof(float))
    die("unexpected pre_block.attn QKV bias size");
  weight_group_add_generated(vae->enc_weights, &vae->attn_qkv.b,
                             (size_t)AUDIO_QKV_WIDTH * sizeof(float),
                             fill_qkv_bias, &qbp, sizeof qbp);

  add_linear(vae->enc_weights, vae->tensors, &vae->attn_proj,
             MINIMAX_H3_AUDIO_LATENT_CHANNELS, MINIMAX_H3_AUDIO_LATENT_CHANNELS,
             "pre_block.attn.proj");
  add_linear(vae->enc_weights, vae->tensors, &vae->skip_proj,
             MINIMAX_H3_AUDIO_LATENT_CHANNELS, AUDIO_LATENT_DIM,
             "pre_block.proj");
  add_linear(vae->enc_weights, vae->tensors, &vae->mlp_w0,
             AUDIO_ATTN_MLP_HIDDEN, MINIMAX_H3_AUDIO_LATENT_CHANNELS,
             "pre_block.mlp.w0");
  add_linear(vae->enc_weights, vae->tensors, &vae->mlp_w1,
             AUDIO_ATTN_MLP_HIDDEN, MINIMAX_H3_AUDIO_LATENT_CHANNELS,
             "pre_block.mlp.w1");
  add_linear(vae->enc_weights, vae->tensors, &vae->mlp_w2,
             MINIMAX_H3_AUDIO_LATENT_CHANNELS, AUDIO_ATTN_MLP_HIDDEN,
             "pre_block.mlp.w2");
  add_conv1d(vae->enc_weights, vae->tensors, &vae->mean_proj,
             MINIMAX_H3_AUDIO_LATENT_CHANNELS, MINIMAX_H3_AUDIO_LATENT_CHANNELS,
             AUDIO_CONV_K1, 1, 0, 1, 1, 0, "mean_proj");
}

MinimaxH3AudioVae *minimax_h3_audio_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path) {
  MinimaxH3AudioVae *vae = xmalloc(sizeof *vae);
  memset(vae, 0, sizeof *vae);
  vae->gpu = gpu;
  vae->account = account;
  memory_slot_init(&vae->workspace_slot, account, MEMORY_MODEL, NULL,
                   "minimax-h3-audio-vae-workspace", NULL, NULL);
  vae->workspace_slot.automatic = 1;
  vae->tensors = tensor_index_open_file(weights_path);
  vae->dec_weights =
      weight_group_new(gpu, account, "minimax-h3-audio-vae-decoder");
  vae->enc_weights =
      weight_group_new(gpu, account, "minimax-h3-audio-vae-encoder");

  vae->k_conv1d = gpu_find_kernel(gpu, "minimax_h3_audio_conv1d");
  vae->k_conv_t1d = gpu_find_kernel(gpu, "minimax_h3_audio_conv_transpose1d");
  vae->k_snake1d = gpu_find_kernel(gpu, "minimax_h3_audio_snake1d");
  vae->k_act1d = gpu_find_kernel(gpu, "minimax_h3_audio_act1d_snakebeta");
  vae->k_avg3 = gpu_find_kernel(gpu, "minimax_h3_audio_avg3");
  vae->k_affine = gpu_find_kernel(gpu, "minimax_h3_audio_affine_ch");
  vae->k_transpose = gpu_find_kernel(gpu, "minimax_h3_audio_transpose");
  vae->k_linear = gpu_find_kernel(gpu, "minimax_h3_audio_linear_f32");
  vae->k_geglu = gpu_find_kernel(gpu, "minimax_h3_audio_geglu_tanh");
  vae->k_causal_attn =
      gpu_find_kernel(gpu, "minimax_h3_audio_causal_attn_pool");
  vae->k_layernorm = gpu_find_kernel(gpu, "minimax_h3_layernorm");

  register_decoder_weights(vae);
  register_encoder_weights(vae);
  return vae;
}

void minimax_h3_audio_vae_close(MinimaxH3AudioVae *vae) {
  if (!vae) return;
  weight_group_free(vae->enc_weights);
  weight_group_free(vae->dec_weights);
  memory_slot_remove(&vae->workspace_slot);
  tensor_index_close(vae->tensors);
  free(vae);
}

WeightGroup *minimax_h3_audio_vae_weights(MinimaxH3AudioVae *vae) {
  return vae->dec_weights;
}

uint64_t minimax_h3_audio_vae_required_bytes(const MinimaxH3AudioVae *vae) {
  return weight_group_required_bytes(vae->dec_weights);
}

uint64_t minimax_h3_audio_vae_resident_bytes(const MinimaxH3AudioVae *vae) {
  return weight_group_resident_bytes(vae->dec_weights);
}

int minimax_h3_audio_vae_load(MinimaxH3AudioVae *vae) {
  return weight_group_load(vae->dec_weights);
}

WeightGroup *minimax_h3_audio_vae_encoder_weights(MinimaxH3AudioVae *vae) {
  return vae->enc_weights;
}

uint64_t
minimax_h3_audio_vae_encoder_required_bytes(const MinimaxH3AudioVae *vae) {
  return weight_group_required_bytes(vae->enc_weights);
}

uint64_t
minimax_h3_audio_vae_encoder_resident_bytes(const MinimaxH3AudioVae *vae) {
  return weight_group_resident_bytes(vae->enc_weights);
}

int minimax_h3_audio_vae_load_encoder(MinimaxH3AudioVae *vae) {
  return weight_group_load(vae->enc_weights);
}

int minimax_h3_audio_vae_encoded_frames(int samples) {
  if (samples <= 0) return 0;
  return (samples + MINIMAX_H3_AUDIO_HOP_LENGTH - 1) /
         MINIMAX_H3_AUDIO_HOP_LENGTH;
}

uint64_t minimax_h3_audio_vae_decode_workspace_bytes(int audio_t) {
  if (audio_t <= 0) return 0;
  uint64_t one = gpu_round_up_to_pages((uint64_t)audio_t *
                                       DEC_MAX_ELEMS_PER_FRAME * sizeof(float));
  return one * (uint64_t)DEC_WORK_BUFS;
}

uint64_t minimax_h3_audio_vae_encode_workspace_bytes(int samples) {
  int audio_t = minimax_h3_audio_vae_encoded_frames(samples);
  if (audio_t <= 0) return 0;
  uint64_t one = gpu_round_up_to_pages((uint64_t)audio_t *
                                       ENC_MAX_ELEMS_PER_FRAME * sizeof(float));
  uint64_t small_one = gpu_round_up_to_pages(
      (uint64_t)audio_t * (uint64_t)AUDIO_ATTN_MLP_HIDDEN * sizeof(float));
  return one * (uint64_t)ENC_WORK_BUFS + small_one * (uint64_t)ENC_SMALL_BUFS;
}

// Frees n GPU buffers in array bufs.
static void free_bufs(Buf *bufs, int n) {
  for (int i = n - 1; i >= 0; i--) gpu_free(&bufs[i]);
}

// Allocates n GPU workspace buffers of bytes each. Returns 1 on success or 0.
static int alloc_bufs(MinimaxH3AudioVae *vae, Buf *bufs, int n, size_t bytes) {
  for (int i = 0; i < n; i++) {
    bufs[i] = gpu_try_alloc_required(vae->gpu, &vae->workspace_slot, bytes);
    if (!bufs[i].ptr) {
      free_bufs(bufs, i);
      return 0;
    }
  }
  return 1;
}

// Runs one BigVGAN AMPBlock1 on input x into out, using t0 and t1 as scratch.
static void enc_amp_block(const MinimaxH3AudioVae *vae, GpuCommandBuffer *cmd,
                          const AudioAmpBlock *blk, const Buf *x, const Buf *t0,
                          const Buf *t1, const Buf *out, uint32_t channels,
                          uint32_t length) {
  // d = 0: x -> out (leaves x intact for sibling branches).
  enc_act1d(vae, cmd, &blk->acts[0], x, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs1[0], t0, length, NULL, t1, length, 0);
  enc_act1d(vae, cmd, &blk->acts[1], t1, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs2[0], t0, length, x, out, length, 0);

  // d = 1: out -> t1.
  enc_act1d(vae, cmd, &blk->acts[2], out, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs1[1], t0, length, NULL, t1, length, 0);
  enc_act1d(vae, cmd, &blk->acts[3], t1, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs2[1], t0, length, out, t1, length, 0);

  // d = 2: t1 -> out.
  enc_act1d(vae, cmd, &blk->acts[4], t1, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs1[2], t0, length, NULL, out, length, 0);
  enc_act1d(vae, cmd, &blk->acts[5], out, t0, channels, length);
  enc_conv1d(vae, cmd, &blk->convs2[2], t0, length, t1, out, length, 0);
}

int minimax_h3_audio_vae_decode(MinimaxH3AudioVae *vae, const float *latents,
                                int audio_t, float *out_pcm,
                                const MinimaxH3RunHooks *hooks) {
  if (!vae || !latents || !out_pcm || audio_t <= 0) return 0;
  if (!minimax_h3_audio_vae_load(vae)) return 0;
  if (minimax_h3_should_stop(hooks)) return 0;

  size_t buf_bytes =
      (size_t)audio_t * (size_t)DEC_MAX_ELEMS_PER_FRAME * sizeof(float);
  Buf ws[DEC_WORK_BUFS] = {0};
  if (!alloc_bufs(vae, ws, DEC_WORK_BUFS, buf_bytes)) return 0;

  // Workspace roles: ws[0]=x, ws[1]=t0, ws[2]=t1, ws[3..5]=b0..b2.
  int ok = 1;
  uint32_t out_samples =
      (uint32_t)audio_t * (uint32_t)MINIMAX_H3_AUDIO_HOP_LENGTH;

  for (int ch = 0; ch < MINIMAX_H3_AUDIO_STEREO_CHANNELS; ch++) {
    if (minimax_h3_should_stop(hooks)) {
      ok = 0;
      break;
    }
    // Extract mono latent [32, audio_t] from stereo [32, 2, audio_t].
    float *z_dst = (float *)ws[1].ptr;
    for (int c = 0; c < MINIMAX_H3_AUDIO_LATENT_CHANNELS; c++) {
      const float *src =
          latents +
          ((size_t)c * (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS + (size_t)ch) *
              (size_t)audio_t;
      memcpy(z_dst + (size_t)c * (size_t)audio_t, src,
             (size_t)audio_t * sizeof(float));
    }

    uint32_t len = (uint32_t)audio_t;
    GpuCommandBuffer cmd = gpu_begin(vae->gpu);
    enc_affine_ch(vae, &cmd, &ws[1], &vae->dec_mean, &vae->dec_std, &ws[0],
                  MINIMAX_H3_AUDIO_LATENT_CHANNELS, len, 0);
    enc_conv1d(vae, &cmd, &vae->dec_in_proj, &ws[0], len, NULL, &ws[1], len, 0);
    enc_conv1d(vae, &cmd, &vae->conv_pre, &ws[1], len, NULL, &ws[0], len, 0);
    uint64_t cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

    for (int stage = 0; stage < AUDIO_UP_STAGES; stage++) {
      if (minimax_h3_should_stop(hooks)) {
        ok = 0;
        break;
      }
      uint32_t next_len = len * (uint32_t)UP_RATES[stage];
      uint32_t stage_ch = (uint32_t)(AUDIO_DEC_INIT_CH >> (stage + 1));
      uint32_t total_elems = stage_ch * next_len;

      cmd = gpu_begin(vae->gpu);
      enc_conv_t1d(vae, &cmd, &vae->ups[stage], &ws[0], len, &ws[1], next_len);
      cb = cmd.number;
      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
      // Swap ws[0] and ws[1] so ws[0] holds the upsampled input.
      Buf swap = ws[0];
      ws[0] = ws[1];
      ws[1] = swap;
      len = next_len;

      for (int j = 0; j < AUDIO_RES_KERNELS; j++) {
        if (minimax_h3_should_stop(hooks)) {
          ok = 0;
          break;
        }
        int r = stage * AUDIO_RES_KERNELS + j;
        cmd = gpu_begin(vae->gpu);
        enc_amp_block(vae, &cmd, &vae->resblocks[r], &ws[0], &ws[1], &ws[2],
                      &ws[3 + j], stage_ch, len);
        cb = cmd.number;
        gpu_commit(&cmd);
        minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
      }
      if (!ok) break;

      cmd = gpu_begin(vae->gpu);
      gpu_select_kernel(&cmd, vae->k_avg3);
      gpu_bind_buffer(&cmd, 0, &ws[3], 0);
      gpu_bind_buffer(&cmd, 1, &ws[4], 0);
      gpu_bind_buffer(&cmd, 2, &ws[5], 0);
      gpu_bind_buffer(&cmd, 3, &ws[0], 0);
      gpu_copy_bytes(&cmd, 4, &total_elems, sizeof total_elems);
      gpu_run_kernel(&cmd, ceil_div(total_elems, ELEM_THREADS), 1, 1,
                     ELEM_THREADS, 1, 1);
      cb = cmd.number;
      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
    }
    if (!ok) break;

    uint32_t final_ch = (uint32_t)(AUDIO_DEC_INIT_CH >> AUDIO_UP_STAGES);
    cmd = gpu_begin(vae->gpu);
    enc_act1d(vae, &cmd, &vae->act_post, &ws[0], &ws[1], final_ch, len);
    enc_conv1d(vae, &cmd, &vae->conv_post, &ws[1], len, NULL, &ws[0], len, 1);
    cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

    memcpy(out_pcm + (size_t)ch * (size_t)out_samples, ws[0].ptr,
           (size_t)out_samples * sizeof(float));
  }

  free_bufs(ws, DEC_WORK_BUFS);
  return ok;
}

int minimax_h3_audio_vae_encode(MinimaxH3AudioVae *vae, const float *pcm,
                                int samples, float *out_latents,
                                const MinimaxH3RunHooks *hooks) {
  if (!vae || !pcm || !out_latents || samples <= 0) return 0;
  if (!minimax_h3_audio_vae_load_encoder(vae)) return 0;
  if (minimax_h3_should_stop(hooks)) return 0;

  int audio_t = minimax_h3_audio_vae_encoded_frames(samples);
  uint32_t padded_len =
      (uint32_t)audio_t * (uint32_t)MINIMAX_H3_AUDIO_HOP_LENGTH;
  size_t buf_bytes =
      (size_t)audio_t * (size_t)ENC_MAX_ELEMS_PER_FRAME * sizeof(float);
  Buf ws[ENC_WORK_BUFS] = {0};
  if (!alloc_bufs(vae, ws, ENC_WORK_BUFS, buf_bytes)) return 0;

  // Small AttnProjection scratch buffers allocated alongside ws[0..2].
  Buf small[ENC_SMALL_BUFS] = {0};
  size_t small_bytes =
      (size_t)audio_t * (size_t)AUDIO_ATTN_MLP_HIDDEN * sizeof(float);
  if (!alloc_bufs(vae, small, ENC_SMALL_BUFS, small_bytes)) {
    free_bufs(ws, ENC_WORK_BUFS);
    return 0;
  }

  int ok = 1;
  for (int ch = 0; ch < MINIMAX_H3_AUDIO_STEREO_CHANNELS; ch++) {
    if (minimax_h3_should_stop(hooks)) {
      ok = 0;
      break;
    }
    float *in_dst = (float *)ws[1].ptr;
    memcpy(in_dst, pcm + (size_t)ch * (size_t)samples,
           (size_t)samples * sizeof(float));
    if (padded_len > (uint32_t)samples)
      memset(in_dst + samples, 0,
             (size_t)(padded_len - (uint32_t)samples) * sizeof(float));

    uint32_t len = padded_len;
    GpuCommandBuffer cmd = gpu_begin(vae->gpu);
    enc_conv1d(vae, &cmd, &vae->enc_in, &ws[1], len, NULL, &ws[0], len, 0);
    uint64_t cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

    uint32_t c_cur = AUDIO_ENC_INIT_CH;
    for (int s = 0; s < AUDIO_ENC_STAGES; s++) {
      if (minimax_h3_should_stop(hooks)) {
        ok = 0;
        break;
      }
      const AudioEncStage *st = &vae->enc_stages[s];
      for (int u = 0; u < AUDIO_ENC_RES_UNITS; u++) {
        const AudioEncResUnit *ru = &st->units[u];
        cmd = gpu_begin(vae->gpu);
        enc_snake1d(vae, &cmd, &ru->alpha1, &ws[0], &ws[1], c_cur, len);
        enc_conv1d(vae, &cmd, &ru->conv1, &ws[1], len, NULL, &ws[2], len, 0);
        enc_snake1d(vae, &cmd, &ru->alpha2, &ws[2], &ws[1], c_cur, len);
        enc_conv1d(vae, &cmd, &ru->conv2, &ws[1], len, &ws[0], &ws[2], len, 0);
        cb = cmd.number;
        gpu_commit(&cmd);
        minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
        Buf swap = ws[0];
        ws[0] = ws[2];
        ws[2] = swap;
      }
      uint32_t next_len = len / (uint32_t)ENC_RATES[s];
      cmd = gpu_begin(vae->gpu);
      enc_snake1d(vae, &cmd, &st->alpha, &ws[0], &ws[1], c_cur, len);
      enc_conv1d(vae, &cmd, &st->down, &ws[1], len, NULL, &ws[0], next_len, 0);
      cb = cmd.number;
      gpu_commit(&cmd);
      minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);
      c_cur *= 2u;
      len = next_len;
    }
    if (!ok) break;

    // Final DAC Snake1d + conv, then transpose [2048, T] -> [T, 2048].
    AudioTransposeParam tr_in = {AUDIO_LATENT_DIM, len};
    AudioCausalAttnParam cap = {len};
    AudioGeGluParam gp = {len * (uint32_t)AUDIO_ATTN_MLP_HIDDEN};
    AudioTransposeParam tr_out = {len, MINIMAX_H3_AUDIO_LATENT_CHANNELS};

    cmd = gpu_begin(vae->gpu);
    enc_snake1d(vae, &cmd, &vae->enc_out_alpha, &ws[0], &ws[1], c_cur, len);
    enc_conv1d(vae, &cmd, &vae->enc_out, &ws[1], len, NULL, &ws[2], len, 0);

    gpu_select_kernel(&cmd, vae->k_transpose);
    gpu_bind_buffer(&cmd, 0, &ws[2], 0);
    gpu_bind_buffer(&cmd, 1, &ws[0], 0);
    gpu_copy_bytes(&cmd, 2, &tr_in, sizeof tr_in);
    gpu_run_kernel(&cmd, ceil_div(len, TRANS_TILE),
                   ceil_div(AUDIO_LATENT_DIM, TRANS_TILE), 1, TRANS_TILE,
                   TRANS_TILE, 1);

    // AttnProjection: attn(norm1(x)) -> small[0], proj(norm3(x)) + small[0] ->
    // small[1].
    enc_layernorm(vae, &cmd, &vae->norm1, &ws[0], &ws[1], len,
                  AUDIO_LATENT_DIM);
    enc_linear(vae, &cmd, &vae->attn_qkv, &ws[1], NULL, &ws[2], len);
    gpu_select_kernel(&cmd, vae->k_causal_attn);
    gpu_bind_buffer(&cmd, 0, &ws[2], 0);
    gpu_bind_buffer(&cmd, 1, &small[2], 0);
    gpu_copy_bytes(&cmd, 2, &cap, sizeof cap);
    gpu_run_kernel(&cmd, len, 1, 1, ATTN_THREADS, 1, 1);
    enc_linear(vae, &cmd, &vae->attn_proj, &small[2], NULL, &small[0], len);

    enc_layernorm(vae, &cmd, &vae->norm3, &ws[0], &ws[1], len,
                  AUDIO_LATENT_DIM);
    enc_linear(vae, &cmd, &vae->skip_proj, &ws[1], &small[0], &small[1], len);

    // GeGluMlp: mlp(norm2(h)) + h -> small[0].
    enc_layernorm(vae, &cmd, &vae->norm2, &small[1], &small[2], len,
                  MINIMAX_H3_AUDIO_LATENT_CHANNELS);
    enc_layernorm(vae, &cmd, &vae->mlp_norm, &small[2], &small[3], len,
                  MINIMAX_H3_AUDIO_LATENT_CHANNELS);
    enc_linear(vae, &cmd, &vae->mlp_w0, &small[3], NULL, &small[0], len);
    enc_linear(vae, &cmd, &vae->mlp_w1, &small[3], NULL, &small[2], len);
    gpu_select_kernel(&cmd, vae->k_geglu);
    gpu_bind_buffer(&cmd, 0, &small[0], 0);
    gpu_bind_buffer(&cmd, 1, &small[2], 0);
    gpu_bind_buffer(&cmd, 2, &small[3], 0);
    gpu_copy_bytes(&cmd, 3, &gp, sizeof gp);
    gpu_run_kernel(&cmd, ceil_div(gp.n, ELEM_THREADS), 1, 1, ELEM_THREADS, 1,
                   1);
    enc_linear(vae, &cmd, &vae->mlp_w2, &small[3], &small[1], &small[0], len);

    // Transpose [T, 32] -> [32, T], run mean_proj and normalize.
    gpu_select_kernel(&cmd, vae->k_transpose);
    gpu_bind_buffer(&cmd, 0, &small[0], 0);
    gpu_bind_buffer(&cmd, 1, &ws[1], 0);
    gpu_copy_bytes(&cmd, 2, &tr_out, sizeof tr_out);
    gpu_run_kernel(&cmd, ceil_div(MINIMAX_H3_AUDIO_LATENT_CHANNELS, TRANS_TILE),
                   ceil_div(len, TRANS_TILE), 1, TRANS_TILE, TRANS_TILE, 1);
    enc_conv1d(vae, &cmd, &vae->mean_proj, &ws[1], len, NULL, &ws[2], len, 0);
    enc_affine_ch(vae, &cmd, &ws[2], &vae->enc_mean, &vae->enc_std, &ws[0],
                  MINIMAX_H3_AUDIO_LATENT_CHANNELS, len, 1);
    cb = cmd.number;
    gpu_commit(&cmd);
    minimax_h3_gpu_wait_hooks(vae->gpu, cb, hooks);

    const float *z_src = (const float *)ws[0].ptr;
    for (int c = 0; c < MINIMAX_H3_AUDIO_LATENT_CHANNELS; c++) {
      float *dst =
          out_latents +
          ((size_t)c * (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS + (size_t)ch) *
              (size_t)audio_t;
      memcpy(dst, z_src + (size_t)c * (size_t)audio_t,
             (size_t)audio_t * sizeof(float));
    }
  }

  free_bufs(small, ENC_SMALL_BUFS);
  free_bufs(ws, ENC_WORK_BUFS);
  return ok;
}
