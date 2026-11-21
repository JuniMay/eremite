// ---- MiniMax-H3 Audio VAE kernels (BigVGAN decoder + DAC encoder) ----
//
// Weights and activations are FP32. Channels-first 1D audio tensors use layout
// [channels, length]; encoder AttnProjection tensors use row-major [T, dim].
// Model constants cite comfy/ldm/minimax/audio_vae.py (MiniMax-H3 Audio VAE).

// Epsilon preventing division by zero in Snake1d and SnakeBeta (audio_vae.py:1128).
constant constexpr float MINIMAX_H3_AUDIO_SNAKE_EPS = 1e-9f;

// Kaiser-sinc anti-aliased activation resampling parameters (audio_vae.py:1188, 1209, 1241):
// 2x up/down ratio, 12-tap filters, 6 odd/even upsample taps, downsample left pad 5.
constant constexpr uint MINIMAX_H3_AUDIO_ACT_RATIO = 2,
                        MINIMAX_H3_AUDIO_ACT_K = 12,
                        MINIMAX_H3_AUDIO_ACT_HALF_K = 6,
                        MINIMAX_H3_AUDIO_ACT_DOWN_PAD_L = 5,
                        MINIMAX_H3_AUDIO_ACT_TILE_T = 256,
                        MINIMAX_H3_AUDIO_ACT_SHMEM =
                            MINIMAX_H3_AUDIO_ACT_TILE_T * MINIMAX_H3_AUDIO_ACT_RATIO +
                            MINIMAX_H3_AUDIO_ACT_K - 1;

// Parallel BigVGAN residual branches averaged per upsample stage (audio_vae.py:1421, 1467).
constant constexpr float MINIMAX_H3_AUDIO_RESBLOCK_INV = 1.0f / 3.0f;

// Audio waveform output clamp bounds (audio_vae.py:1470).
constant constexpr float MINIMAX_H3_AUDIO_CLAMP_LO = -1.0f,
                         MINIMAX_H3_AUDIO_CLAMP_HI = 1.0f;

// Tanh-approximate GELU constants in GeGluMlp (audio_vae.py:1319): sqrt(2/pi) and 0.044715.
constant constexpr float MINIMAX_H3_AUDIO_GELU_SQRT_2_OVER_PI = 0.7978845608028654f,
                         MINIMAX_H3_AUDIO_GELU_COEF = 0.044715f,
                         MINIMAX_H3_AUDIO_HALF = 0.5f;

// Encoder AttnProjection geometry (audio_vae.py:1329-1355, 1504):
// 8 causal attention heads of 256 dims pooled by 8 down to 32 latent channels.
constant constexpr uint MINIMAX_H3_AUDIO_ATTN_HEADS = 8,
                        MINIMAX_H3_AUDIO_ATTN_HEAD_DIM = 256,
                        MINIMAX_H3_AUDIO_ATTN_OUT_DIM = 32,
                        MINIMAX_H3_AUDIO_ATTN_POOL =
                            MINIMAX_H3_AUDIO_ATTN_HEAD_DIM / MINIMAX_H3_AUDIO_ATTN_OUT_DIM,
                        MINIMAX_H3_AUDIO_ATTN_QKV_PARTS = 3;

// Attention dot-product scale 1 / sqrt(256) = 1 / 16 (audio_vae.py:1349).
constant constexpr float MINIMAX_H3_AUDIO_ATTN_SCALE = 0.0625f;

// Channels computed per thread and per 4-simdgroup threadgroup in 1D conv tiles.
constant constexpr uint MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD = 8,
                        MINIMAX_H3_AUDIO_CONV_WARPS = 4,
                        MINIMAX_H3_AUDIO_CONV_TILE_CO =
                            MINIMAX_H3_AUDIO_CONV_WARPS * MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD;

// Parameters for FP32 1D convolution with optional bias, center-cropped residual
// addition, and [-1, 1] waveform clamping.
struct MinimaxH3AudioConv1d {
  uint c_in, c_out, l_in, l_out, k, stride, pad, dilation;
  uint has_bias, add_res, res_crop, clamp_out;
};

// Parameters for FP32 1D transposed convolution (BigVGAN upsampling stages).
struct MinimaxH3AudioConvT1d {
  uint c_in, c_out, l_in, l_out, k, stride, pad;
};

// Parameters for DAC encoder Snake1d activation.
struct MinimaxH3AudioSnake1d {
  uint channels, length;
};

// Parameters for BigVGAN fused anti-aliased Activation1d(SnakeBeta).
struct MinimaxH3AudioAct1d {
  uint channels, length;
};

// Parameters for per-channel affine scaling (latent normalize or denormalize).
struct MinimaxH3AudioAffine {
  uint channels, length, invert;
};

// Parameters for 2D matrix transpose [rows, cols] -> [cols, rows].
struct MinimaxH3AudioTranspose {
  uint rows, cols;
};

// Parameters for FP32 row-major linear y[m, n] = x[m, k] @ w[n, k]^T + bias.
struct MinimaxH3AudioLinear {
  uint m, n, k, has_bias, add_res;
};

// Parameters for tanh-approximate GeGLU activation on [rows, inner].
struct MinimaxH3AudioGeGlu {
  uint n;
};

// Parameters for encoder AttnProjection causal attention + head mean + avg pool.
struct MinimaxH3AudioCausalAttn {
  uint seq_len;
};

// FP32 1D convolution on channels-first [c_in, l_in] -> [c_out, l_out].
// Each threadgroup of 128 threads (4 simdgroups x 32 lanes) computes a 32x32
// tile of (c_out, t_out); lanes in a simdgroup share weights and read coalesced t.
kernel void minimax_h3_audio_conv1d(device const float *x [[buffer(0)]],
                                    device const float *w [[buffer(1)]],
                                    device const float *bias [[buffer(2)]],
                                    device const float *res [[buffer(3)]],
                                    device float *y [[buffer(4)]],
                                    constant MinimaxH3AudioConv1d &p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint lid [[thread_index_in_threadgroup]]) {
  uint lane = lid & MINIMAX_H3_SIMD_MASK;
  uint warp = lid >> MINIMAX_H3_SIMD_SHIFT;
  uint t = tg.x * MINIMAX_H3_SIMD_WIDTH + lane;
  uint co_base = tg.y * MINIMAX_H3_AUDIO_CONV_TILE_CO +
                 warp * MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD;
  if (t >= p.l_out || co_base >= p.c_out) return;

  float acc[MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD];
  for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
    uint co = co_base + r;
    acc[r] = (p.has_bias && co < p.c_out) ? bias[co] : 0.0f;
  }

  uint stride_w = p.c_in * p.k;
  int t_origin = int(t * p.stride) - int(p.pad);
  for (uint ci = 0; ci < p.c_in; ci++) {
    device const float *x_ch = x + ci * p.l_in;
    for (uint ki = 0; ki < p.k; ki++) {
      int ti = t_origin + int(ki * p.dilation);
      float xv = (ti >= 0 && ti < int(p.l_in)) ? x_ch[uint(ti)] : 0.0f;
      uint w_off = ci * p.k + ki;
      for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
        uint co = co_base + r;
        if (co < p.c_out) acc[r] = fma(w[co * stride_w + w_off], xv, acc[r]);
      }
    }
  }

  uint l_res = p.l_out + 2u * p.res_crop;
  for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
    uint co = co_base + r;
    if (co < p.c_out) {
      float v = acc[r];
      if (p.add_res) v += res[co * l_res + t + p.res_crop];
      if (p.clamp_out) v = clamp(v, MINIMAX_H3_AUDIO_CLAMP_LO, MINIMAX_H3_AUDIO_CLAMP_HI);
      y[co * p.l_out + t] = v;
    }
  }
}

// FP32 1D transposed convolution on [c_in, l_in] with weight [c_in, c_out, k].
// Grid z indexes output phase (0..stride-1) so all 128 threads in a threadgroup
// share the same kernel taps and read coalesced input timesteps.
kernel void minimax_h3_audio_conv_transpose1d(device const float *x [[buffer(0)]],
                                              device const float *w [[buffer(1)]],
                                              device const float *bias [[buffer(2)]],
                                              device float *y [[buffer(3)]],
                                              constant MinimaxH3AudioConvT1d &p [[buffer(4)]],
                                              uint3 tg [[threadgroup_position_in_grid]],
                                              uint lid [[thread_index_in_threadgroup]]) {
  uint lane = lid & MINIMAX_H3_SIMD_MASK;
  uint warp = lid >> MINIMAX_H3_SIMD_SHIFT;
  uint phase = tg.z;
  uint t_out = (tg.x * MINIMAX_H3_SIMD_WIDTH + lane) * p.stride + phase;
  uint co_base = tg.y * MINIMAX_H3_AUDIO_CONV_TILE_CO +
                 warp * MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD;
  if (t_out >= p.l_out || co_base >= p.c_out) return;

  float acc[MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD];
  for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
    uint co = co_base + r;
    acc[r] = (co < p.c_out) ? bias[co] : 0.0f;
  }

  uint u = t_out + p.pad;
  uint k0 = u % p.stride;
  uint w_ci_stride = p.c_out * p.k;
  for (uint ci = 0; ci < p.c_in; ci++) {
    device const float *x_ch = x + ci * p.l_in;
    device const float *w_ci = w + ci * w_ci_stride;
    for (uint ki = k0; ki < p.k; ki += p.stride) {
      if (u < ki) break;
      uint ti = (u - ki) / p.stride;
      if (ti >= p.l_in) continue;
      float xv = x_ch[ti];
      for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
        uint co = co_base + r;
        if (co < p.c_out) acc[r] = fma(w_ci[co * p.k + ki], xv, acc[r]);
      }
    }
  }

  for (uint r = 0; r < MINIMAX_H3_AUDIO_CONV_CO_PER_THREAD; r++) {
    uint co = co_base + r;
    if (co < p.c_out) y[co * p.l_out + t_out] = acc[r];
  }
}

// DAC encoder Snake1d activation: y = x + sin^2(alpha * x) / (alpha + 1e-9).
kernel void minimax_h3_audio_snake1d(device const float *x [[buffer(0)]],
                                     device const float *alpha [[buffer(1)]],
                                     device float *y [[buffer(2)]],
                                     constant MinimaxH3AudioSnake1d &p [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  uint t = gid.x, c = gid.y;
  if (t >= p.length || c >= p.channels) return;
  float a = alpha[c];
  float v = x[c * p.length + t];
  float s = sin(a * v);
  y[c * p.length + t] = fma(s * s, 1.0f / (a + MINIMAX_H3_AUDIO_SNAKE_EPS), v);
}

// BigVGAN fused anti-aliased Activation1d(SnakeBeta):
// replicate-padded 2x Kaiser-sinc upsample (12 taps) -> SnakeBeta (log alpha/beta)
// -> replicate-padded 2x Kaiser-sinc downsample (12 taps).
// Threadgroup: 256 threads processing one channel x 256 output timesteps.
kernel void minimax_h3_audio_act1d_snakebeta(device const float *x [[buffer(0)]],
                                             device const float *alpha [[buffer(1)]],
                                             device const float *beta [[buffer(2)]],
                                             device const float *f_up [[buffer(3)]],
                                             device const float *f_down [[buffer(4)]],
                                             device float *y [[buffer(5)]],
                                             constant MinimaxH3AudioAct1d &p [[buffer(6)]],
                                             uint2 tg [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]]) {
  uint c = tg.y;
  if (c >= p.channels) return;
  uint t0 = tg.x * MINIMAX_H3_AUDIO_ACT_TILE_T;
  int max_in = int(p.length) - 1;
  int max_up = int(p.length * MINIMAX_H3_AUDIO_ACT_RATIO) - 1;
  device const float *x_ch = x + c * p.length;

  float a = exp(alpha[c]);
  float inv_b = 1.0f / (exp(beta[c]) + MINIMAX_H3_AUDIO_SNAKE_EPS);
  float fu[MINIMAX_H3_AUDIO_ACT_K];
  float fd[MINIMAX_H3_AUDIO_ACT_K];
  for (uint k = 0; k < MINIMAX_H3_AUDIO_ACT_K; k++) {
    fu[k] = f_up[k];
    fd[k] = f_down[k];
  }

  threadgroup float act_buf[MINIMAX_H3_AUDIO_ACT_SHMEM];
  int j_base = int(t0 * MINIMAX_H3_AUDIO_ACT_RATIO) - int(MINIMAX_H3_AUDIO_ACT_DOWN_PAD_L);
  for (uint s = tid; s < MINIMAX_H3_AUDIO_ACT_SHMEM; s += MINIMAX_H3_AUDIO_ACT_TILE_T) {
    int j = clamp(j_base + int(s), 0, max_up);
    int m = j >> 1;
    float up_v = 0.0f;
    if ((j & 1) == 0) {
      // Even upsampled positions use odd taps k = 1, 3, 5, 7, 9, 11 at offsets +2..-3.
      for (uint h = 0; h < MINIMAX_H3_AUDIO_ACT_HALF_K; h++) {
        int idx = clamp(m + 2 - int(h), 0, max_in);
        up_v = fma(fu[2u * h + 1u], x_ch[uint(idx)], up_v);
      }
    } else {
      // Odd upsampled positions use even taps k = 0, 2, 4, 6, 8, 10 at offsets +3..-2.
      for (uint h = 0; h < MINIMAX_H3_AUDIO_ACT_HALF_K; h++) {
        int idx = clamp(m + 3 - int(h), 0, max_in);
        up_v = fma(fu[2u * h], x_ch[uint(idx)], up_v);
      }
    }
    up_v *= float(MINIMAX_H3_AUDIO_ACT_RATIO);
    float sv = sin(a * up_v);
    act_buf[s] = fma(sv * sv, inv_b, up_v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint t = t0 + tid;
  if (t < p.length) {
    float acc = 0.0f;
    uint base = tid * MINIMAX_H3_AUDIO_ACT_RATIO;
    for (uint k = 0; k < MINIMAX_H3_AUDIO_ACT_K; k++) acc = fma(fd[k], act_buf[base + k], acc);
    y[c * p.length + t] = acc;
  }
}

// Averages three equal-length FP32 buffers: out[i] = (a[i] + b[i] + c[i]) / 3.
kernel void minimax_h3_audio_avg3(device const float *a [[buffer(0)]],
                                  device const float *b [[buffer(1)]],
                                  device const float *c [[buffer(2)]],
                                  device float *out [[buffer(3)]],
                                  constant uint &n [[buffer(4)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = (a[i] + b[i] + c[i]) * MINIMAX_H3_AUDIO_RESBLOCK_INV;
}

// Applies per-channel affine normalization or denormalization on [channels, length].
kernel void minimax_h3_audio_affine_ch(device const float *x [[buffer(0)]],
                                       device const float *mean [[buffer(1)]],
                                       device const float *std [[buffer(2)]],
                                       device float *y [[buffer(3)]],
                                       constant MinimaxH3AudioAffine &p [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  uint t = gid.x, c = gid.y;
  if (t >= p.length || c >= p.channels) return;
  float v = x[c * p.length + t];
  y[c * p.length + t] = p.invert ? ((v - mean[c]) / std[c]) : fma(v, std[c], mean[c]);
}

// Transposes a 2D FP32 matrix [rows, cols] -> [cols, rows].
kernel void minimax_h3_audio_transpose(device const float *x [[buffer(0)]],
                                       device float *y [[buffer(1)]],
                                       constant MinimaxH3AudioTranspose &p [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x < p.cols && gid.y < p.rows) y[gid.x * p.rows + gid.y] = x[gid.y * p.cols + gid.x];
}

// FP32 row-major linear projection y[m, n] = x[m, k] @ w[n, k]^T + bias (+ res).
// Dispatched with 1 simdgroup (32 threads) per output element (col = tg.x, row = tg.y)
// so both x[row, :] and w[col, :] are read with unit stride across lanes.
kernel void minimax_h3_audio_linear_f32(device const float *x [[buffer(0)]],
                                        device const float *w [[buffer(1)]],
                                        device const float *bias [[buffer(2)]],
                                        device const float *res [[buffer(3)]],
                                        device float *y [[buffer(4)]],
                                        constant MinimaxH3AudioLinear &p [[buffer(5)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
  uint col = tg.x, row = tg.y;
  if (col >= p.n || row >= p.m) return;
  device const float *xr = x + row * p.k;
  device const float *wr = w + col * p.k;
  float acc = 0.0f;
  for (uint i = lane; i < p.k; i += MINIMAX_H3_SIMD_WIDTH) acc = fma(xr[i], wr[i], acc);
  float sum = simd_sum(acc);
  if (lane == 0) {
    if (p.has_bias) sum += bias[col];
    if (p.add_res) sum += res[row * p.n + col];
    y[row * p.n + col] = sum;
  }
}

// Tanh-approximate GeGLU: out[i] = gelu_tanh(gate[i]) * up[i].
kernel void minimax_h3_audio_geglu_tanh(device const float *gate [[buffer(0)]],
                                        device const float *up [[buffer(1)]],
                                        device float *out [[buffer(2)]],
                                        constant MinimaxH3AudioGeGlu &p [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float g = gate[i];
  float inner = MINIMAX_H3_AUDIO_GELU_SQRT_2_OVER_PI *
                (g + MINIMAX_H3_AUDIO_GELU_COEF * g * g * g);
  out[i] = MINIMAX_H3_AUDIO_HALF * g * (1.0f + tanh(inner)) * up[i];
}

// Causal 8-head attention (head_dim=256) + mean over heads + 8-way average pool
// to 32 dims for encoder AttnProjection (audio_vae.py:1346-1350).
// qkv layout: [seq_len, 3, 8, 256]; out layout: [seq_len, 32].
// Dispatched with 1 threadgroup of 256 threads (8 simdgroups x 32 lanes) per row t.
kernel void minimax_h3_audio_causal_attn_pool(device const float *qkv [[buffer(0)]],
                                              device float *out [[buffer(1)]],
                                              constant MinimaxH3AudioCausalAttn &p [[buffer(2)]],
                                              uint t [[threadgroup_position_in_grid]],
                                              uint lid [[thread_index_in_threadgroup]]) {
  threadgroup float head_pool[MINIMAX_H3_AUDIO_ATTN_HEADS * MINIMAX_H3_AUDIO_ATTN_OUT_DIM];
  if (t >= p.seq_len) return;
  uint head = lid >> MINIMAX_H3_SIMD_SHIFT;
  uint lane = lid & MINIMAX_H3_SIMD_MASK;
  uint row_stride = MINIMAX_H3_AUDIO_ATTN_QKV_PARTS * MINIMAX_H3_AUDIO_ATTN_HEADS *
                    MINIMAX_H3_AUDIO_ATTN_HEAD_DIM;
  uint k_off = MINIMAX_H3_AUDIO_ATTN_HEADS * MINIMAX_H3_AUDIO_ATTN_HEAD_DIM;
  uint v_off = 2u * k_off;
  uint h_base = head * MINIMAX_H3_AUDIO_ATTN_HEAD_DIM + lane * MINIMAX_H3_AUDIO_ATTN_POOL;

  float qv[MINIMAX_H3_AUDIO_ATTN_POOL];
  float acc[MINIMAX_H3_AUDIO_ATTN_POOL];
  device const float *q_ptr = qkv + t * row_stride + h_base;
  for (uint r = 0; r < MINIMAX_H3_AUDIO_ATTN_POOL; r++) {
    qv[r] = q_ptr[r];
    acc[r] = 0.0f;
  }

  float m_val = -INFINITY;
  float l_val = 0.0f;
  for (uint s = 0; s <= t; s++) {
    device const float *k_ptr = qkv + s * row_stride + k_off + h_base;
    device const float *v_ptr = qkv + s * row_stride + v_off + h_base;
    float partial = 0.0f;
    for (uint r = 0; r < MINIMAX_H3_AUDIO_ATTN_POOL; r++) partial = fma(qv[r], k_ptr[r], partial);
    float score = simd_sum(partial) * MINIMAX_H3_AUDIO_ATTN_SCALE;
    float m_new = max(m_val, score);
    float alpha = exp(m_val - m_new);
    float w = exp(score - m_new);
    l_val = fma(l_val, alpha, w);
    for (uint r = 0; r < MINIMAX_H3_AUDIO_ATTN_POOL; r++) acc[r] = fma(acc[r], alpha, w * v_ptr[r]);
    m_val = m_new;
  }

  // Average the 8 consecutive head dimensions owned by this lane (adaptive_avg_pool1d).
  float pool_sum = 0.0f;
  for (uint r = 0; r < MINIMAX_H3_AUDIO_ATTN_POOL; r++) pool_sum += acc[r];
  head_pool[head * MINIMAX_H3_AUDIO_ATTN_OUT_DIM + lane] =
      pool_sum / (l_val * float(MINIMAX_H3_AUDIO_ATTN_POOL));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (head == 0) {
    float mean_h = 0.0f;
    for (uint h = 0; h < MINIMAX_H3_AUDIO_ATTN_HEADS; h++)
      mean_h += head_pool[h * MINIMAX_H3_AUDIO_ATTN_OUT_DIM + lane];
    out[t * MINIMAX_H3_AUDIO_ATTN_OUT_DIM + lane] = mean_h / float(MINIMAX_H3_AUDIO_ATTN_HEADS);
  }
}
