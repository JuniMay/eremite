// ---- NVIDIA Nemotron-3.5-Transcribe: Parallel Expert Perception Encoder ----
//
// Kernels for the audio perception module (perception.safetensors):
// - 32-layer FP8 ASR TransformerEncoder (d_model=1280, 10 heads, head_dim=128,
//   qk_norm=true, RoPE rotary_fraction=0.5).
// - 31-layer BF16 Sortformer TransformerEncoder (d_model=512, 8 heads,
//   head_dim=64, qk_norm=false, RoPE rotary_fraction=1.0).
// - Sortformer high-resolution subpixel Conv1d upsampling, 8x probability
//   pooling, thresholded sinusoidal speaker fusion and modality projection.

// Matrix tile and threadgroup dimensions for perception kernels.
constant constexpr uint NEMO_AUD_TILE = 64, NEMO_AUD_MM_M_TWO = 128,
                        NEMO_AUD_SGS = 4, NEMO_AUD_SIMD = 32,
                        NEMO_AUD_SWIZZLE_N = 8, NEMO_AUD_MAX_HEAD_DIM = 128,
                        NEMO_AUD_MAX_SPK = 16, NEMO_AUD_CONV_K = 3,
                        NEMO_AUD_CONV_IN_CH = 192;

// Activation identifiers for fused linear epilogues.
constant constexpr uint NEMO_AUD_ACT_NONE = 0, NEMO_AUD_ACT_GELU = 1,
                        NEMO_AUD_ACT_RELU = 2;

// Exact-GELU constants (1/sqrt(2) and Abramowitz-Stegun 7.1.26 erf coefficients).
constant constexpr float NEMO_AUD_INV_SQRT2 = 0.7071067811865475244f,
                         NEMO_AUD_ERF_P = 0.3275911f,
                         NEMO_AUD_ERF_A1 = 0.254829592f,
                         NEMO_AUD_ERF_A2 = -0.284496736f,
                         NEMO_AUD_ERF_A3 = 1.421413741f,
                         NEMO_AUD_ERF_A4 = -1.453152027f,
                         NEMO_AUD_ERF_A5 = 1.061405429f;

// Parameters for LayerNorm over rows of width dim.
struct NemoAudLnParam {
  // Row width in floats and nonzero to round outputs to BF16.
  uint dim, round;
  // LayerNorm variance epsilon.
  float eps;
};

// Parameters for FP32-to-BF16 row packing with zero padding.
struct NemoAudPackParam {
  // Valid input rows, padded output rows and row width in floats (multiple of 4).
  uint rows, padded_rows, cols;
};

// Parameters for per-tensor FP8 and BF16 tiled matrix products with optional
// bias, activation and residual addition.
struct NemoAudMmParam {
  // Active rows, output width, inner width, padded weight rows (multiple of 64).
  uint m, n, k, w_rows;
  // Activation mode (NEMO_AUD_ACT_*), bias flag, residual flag and BF16 round flag.
  uint act, has_bias, has_res, round;
  // Per-tensor weight scale multiplied by 256 to cancel fp8x4's 2^-8 factor.
  float scale;
};

// Parameters for fused QKV split, optional per-head LayerNorm, RoPE and
// head-major BF16 transpose.
struct NemoAudQkvParam {
  // Active sequence rows, 64-aligned output row stride, head count, head dim.
  uint rows, seq_stride, heads, head_dim;
  // Half of rotated dimensions (32 for both ASR and Sortformer) and qk_norm flag.
  uint half_rot, qk_norm;
  // Per-head LayerNorm epsilon.
  float eps;
};

// Parameters for bidirectional cooperative 64x64 FlashAttention.
struct NemoAudAttnParam {
  // Active sequence rows, 64-aligned head-major stride and head count.
  uint rows, seq_stride, heads;
  // Attention scale multiplied by log2(e).
  float log2_scale;
};

// Parameters for Sortformer subpixel Conv1d upsampling (k=3, pad=1) + ReLU.
struct NemoAudUpsampleParam {
  // Encoder frame count, input channels (192) and upsampling factor (8).
  uint frames, in_ch, up_factor;
};

// Parameters for Sortformer sigmoid and 8x temporal average pooling.
struct NemoAudPoolParam {
  // Downsampled encoder frame count, upsampling factor (8) and speaker count (8).
  uint frames, factor, n_spk;
};

// Parameters for thresholded sinusoidal speaker fusion into ASR encoder states.
struct NemoAudFuseParam {
  // Frame count, ASR feature width (1280) and speaker count (8).
  uint rows, dim, n_spk;
  // LayerNorm epsilon, speaker activity threshold (0.5) and kernel scale (1.0).
  float eps, threshold, scale;
};

// Parameters for interleaving even and odd upsampled frame matrices.
struct NemoCtcInterleaveParam {
  // Input encoder frame count and feature width (multiple of 4).
  uint frames, dim;
};

// Parameters for scaled residual blending of upsampled and bridged CTC states.
struct NemoCtcResidParam {
  // Total float4 elements across all rows and BF16 rounding flag.
  uint count4, round;
  // Scalar multiplier on the bridged Transformer output.
  float scale;
};

// Parameters for row-wise CTC log-softmax gather and argmax decoding.
struct NemoCtcLogSoftmaxParam {
  // Row count, full CTC vocabulary size (including blank) and gathered count.
  uint rows, vocab, n_gather;
};

// Exact GELU activation matching PyTorch nn.GELU(approximate='none').
inline float nemo_aud_gelu(float x) {
  float z = x * NEMO_AUD_INV_SQRT2;
  float a = fabs(z);
  float t = 1.0f / (1.0f + NEMO_AUD_ERF_P * a);
  float poly = t * (NEMO_AUD_ERF_A1 +
                    t * (NEMO_AUD_ERF_A2 +
                         t * (NEMO_AUD_ERF_A3 +
                              t * (NEMO_AUD_ERF_A4 + t * NEMO_AUD_ERF_A5))));
  float erf_v = copysign(1.0f - poly * exp(-a * a), z);
  return 0.5f * x * (1.0f + erf_v);
}

// Applies the configured fused epilogue activation.
inline float nemo_aud_act(float v, uint act) {
  if (act == NEMO_AUD_ACT_GELU) return nemo_aud_gelu(v);
  if (act == NEMO_AUD_ACT_RELU) return max(v, 0.0f);
  return v;
}

// LayerNorm over rows of `dim` floats in two-pass FP32 arithmetic.
// Grid: rows threadgroups of 256 threads.
kernel void nemo_aud_layernorm(device const float *x [[buffer(0)]],
                               device const float *w [[buffer(1)]],
                               device const float *b [[buffer(2)]],
                               device float *y [[buffer(3)]],
                               constant NemoAudLnParam &p [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]],
                               uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_AUD_SIMD];
  x += (ulong)row * p.dim;
  y += (ulong)row * p.dim;
  float s = 0.0f;
  for (uint i = t; i < p.dim; i += nt) s += x[i];
  float mean = tg_sum(s, tmp, t, nt) / float(p.dim);
  float var = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    float d = x[i] - mean;
    var += d * d;
  }
  float inv = rsqrt(tg_sum(var, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt) {
    float v = ((x[i] - mean) * inv) * w[i] + b[i];
    y[i] = p.round ? bf(v) : v;
  }
}

// Packs FP32 rows [rows, cols] into BF16 [padded_rows, cols], zero-filling
// padding rows so tiled matrix products never read uninitialized memory.
// Grid: x = cols / 4, y = padded_rows.
kernel void nemo_aud_f32_to_bf16(device const float4 *x [[buffer(0)]],
                                 device bfloat4 *y [[buffer(1)]],
                                 constant NemoAudPackParam &p [[buffer(2)]],
                                 uint2 i [[thread_position_in_grid]]) {
  uint cols4 = p.cols / 4u;
  if (i.x >= cols4 || i.y >= p.padded_rows) return;
  ulong idx = (ulong)i.y * cols4 + i.x;
  y[idx] = i.y < p.rows ? bfloat4(x[idx]) : bfloat4(bfloat(0.0f));
}

// Loads two 16-byte FP8 chunks (32 weights) per thread of a 64x64 weight tile.
struct NemoFp8Chunk {
  uint4 q[2];
};

inline NemoFp8Chunk nemo_fp8_load(device const uchar *w, uint n0, uint n, uint k,
                                  uint k0, uint t) {
  NemoFp8Chunk c;
  for (uint i = 0; i < 2; i++) {
    uint r = n0 + t / 4u + 32u * i;
    uint j = k0 + (t % 4u) * 16u;
    c.q[i] = r < n ? *(device const uint4 *)(w + (ulong)r * k + j) : uint4(0);
  }
  return c;
}

// Unpacks raw FP8 E4M3 codes (scaled by 2^-8, bit-exact in BF16) into a 64x64
// threadgroup BF16 tile.
inline void nemo_fp8_store(threadgroup bfloat *ws, NemoFp8Chunk c, uint t) {
  for (uint i = 0; i < 2; i++) {
    threadgroup bfloat4 *d =
        (threadgroup bfloat4 *)(ws + (t / 4u + 32u * i) * NEMO_AUD_TILE +
                                (t % 4u) * 16u);
    d[0] = bfloat4(fp8x4(c.q[i].x));
    d[1] = bfloat4(fp8x4(c.q[i].y));
    d[2] = bfloat4(fp8x4(c.q[i].z));
    d[3] = bfloat4(fp8x4(c.q[i].w));
  }
}

// Per-tensor FP8 tiled MPP GEMM: y = act(x @ (W * scale)^T + bias) + res.
// Computes 128 output rows x 64 output cols per threadgroup (128 threads) to
// amortize FP8 weight staging over two 64x64 matrix tiles.
// Grid: 1D L2-swizzled ceil(n / 64) * ceil(m / 128) threadgroups of 128 threads.
kernel void nemo_aud_mm_fp8(device const uchar *w [[buffer(0)]],
                            device const bfloat *x_bf16 [[buffer(1)]],
                            device const float *bias [[buffer(2)]],
                            device const float *res [[buffer(3)]],
                            device float *y [[buffer(4)]],
                            constant NemoAudMmParam &p [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat ws[NEMO_AUD_TILE * NEMO_AUD_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_AUD_TILE, NEMO_AUD_TILE, NEMO_AUD_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_AUD_SGS>> op;

  uint tiles_m = (p.m + NEMO_AUD_MM_M_TWO - 1u) / NEMO_AUD_MM_M_TWO;
  uint tiles_n = (p.n + NEMO_AUD_TILE - 1u) / NEMO_AUD_TILE;
  uint group_tiles = tiles_m * NEMO_AUD_SWIZZLE_N;
  uint g = tg / group_tiles;
  uint rem = tg - g * group_tiles;
  uint gn0 = g * NEMO_AUD_SWIZZLE_N;
  uint gwidth = min(NEMO_AUD_SWIZZLE_N, tiles_n - gn0);
  uint tn = gn0 + (rem % gwidth);
  uint tm = rem / gwidth;
  uint m0 = tm * NEMO_AUD_MM_M_TWO, n0 = tn * NEMO_AUD_TILE;

  uint m_pad = tiles_m * NEMO_AUD_MM_M_TWO;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, m_pad));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>,
         tensor_inline>
      tw(ws, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>());

  auto c0 =
      op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                            decltype(tw), float>();
  auto c1 =
      op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                            decltype(tw), float>();
  for (uint16_t i = 0; i < c0.get_capacity(); i++)
    if (c0.is_valid_element(i)) c0[i] = 0.0f, c1[i] = 0.0f;

  NemoFp8Chunk wc = nemo_fp8_load(w, n0, p.n, p.k, 0, t);
  for (uint k0 = 0; k0 < p.k; k0 += NEMO_AUD_TILE) {
    nemo_fp8_store(ws, wc, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + NEMO_AUD_TILE < p.k)
      wc = nemo_fp8_load(w, n0, p.n, p.k, k0 + NEMO_AUD_TILE, t);
    auto a = tx.slice(k0, m0);
    op.run(a, tw, c0);
    if (m0 + NEMO_AUD_TILE < p.m) {
      auto b = tx.slice(k0, m0 + NEMO_AUD_TILE);
      op.run(b, tw, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c0.get_capacity(); i++) {
    if (c0.is_valid_element(i)) {
      auto at = c0.get_multidimensional_index(i);
      uint col = n0 + at[0];
      if (col >= p.n) continue;
      float bv = p.has_bias ? bias[col] : 0.0f;
      for (uint h = 0; h < 2; h++) {
        uint row = m0 + NEMO_AUD_TILE * h + at[1];
        if (row >= p.m) continue;
        float v = (h ? c1[i] : c0[i]) * p.scale + bv;
        if (p.act != NEMO_AUD_ACT_NONE) v = nemo_aud_act(p.round ? bf(v) : v, p.act);
        if (p.has_res) v = (p.round ? bf(v) : v) + res[(ulong)row * p.n + col];
        y[(ulong)row * p.n + col] = p.round ? bf(v) : v;
      }
    }
  }
}

// BF16 tiled MPP GEMM: y = act(x @ W^T + bias) + res.
// Stages 64x64 FP32->BF16 input tiles in threadgroup memory and reads BF16
// weights directly from device memory (padded to p.w_rows >= ceil(n/64)*64).
// Grid: 1D L2-swizzled ceil(n / 64) * ceil(m / 64) threadgroups of 128 threads.
kernel void nemo_aud_mm_bf16(device const bfloat *w [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device const float *bias [[buffer(2)]],
                             device const float *res [[buffer(3)]],
                             device float *y [[buffer(4)]],
                             constant NemoAudMmParam &p [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat xs[NEMO_AUD_TILE * NEMO_AUD_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_AUD_TILE, NEMO_AUD_TILE, NEMO_AUD_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_AUD_SGS>> op;

  uint tiles_m = (p.m + NEMO_AUD_TILE - 1u) / NEMO_AUD_TILE;
  uint tiles_n = (p.n + NEMO_AUD_TILE - 1u) / NEMO_AUD_TILE;
  uint group_tiles = tiles_m * NEMO_AUD_SWIZZLE_N;
  uint g = tg / group_tiles;
  uint rem = tg - g * group_tiles;
  uint gn0 = g * NEMO_AUD_SWIZZLE_N;
  uint gwidth = min(NEMO_AUD_SWIZZLE_N, tiles_n - gn0);
  uint tn = gn0 + (rem % gwidth);
  uint tm = rem / gwidth;
  uint m0 = tm * NEMO_AUD_TILE, n0 = tn * NEMO_AUD_TILE;

  tensor<threadgroup bfloat, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>,
         tensor_inline>
      tx(xs, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>());
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tw(
      (device bfloat *)w, dextents<int32_t, 2>(p.k, p.w_rows));
  auto c = op.get_destination_cooperative_tensor<decltype(tx),
                                                 decltype(tw.slice(0, 0)),
                                                 float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  X8 xc = x_load(x, m0, p.m, p.k, 0, t);
  for (uint k0 = 0; k0 < p.k; k0 += NEMO_AUD_TILE) {
    x_store(xs, xc, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + NEMO_AUD_TILE < p.k)
      xc = x_load(x, m0, p.m, p.k, k0 + NEMO_AUD_TILE, t);
    auto wb = tw.slice(k0, n0);
    op.run(tx, wb, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c.get_capacity(); i++) {
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint col = n0 + at[0], row = m0 + at[1];
      if (row < p.m && col < p.n) {
        float v = c[i] + (p.has_bias ? bias[col] : 0.0f);
        if (p.act != NEMO_AUD_ACT_NONE) v = nemo_aud_act(p.round ? bf(v) : v, p.act);
        if (p.has_res) v = (p.round ? bf(v) : v) + res[(ulong)row * p.n + col];
        y[(ulong)row * p.n + col] = p.round ? bf(v) : v;
      }
    }
  }
}

// Splits fused QKV [rows, 3 * heads * head_dim], applies optional per-head
// LayerNorm to Q and K, applies partial or full RoPE, and writes head-major
// BF16 Q, K, V [heads, seq_stride, head_dim] with zero-padded tail rows.
// Grid: x = heads, y = seq_stride; threadgroup = head_dim threads (64 or 128).
kernel void nemo_aud_qkv_prep(device const float *qkv [[buffer(0)]],
                              device const float *qn_w [[buffer(1)]],
                              device const float *qn_b [[buffer(2)]],
                              device const float *kn_w [[buffer(3)]],
                              device const float *kn_b [[buffer(4)]],
                              device const float *cos_tab [[buffer(5)]],
                              device const float *sin_tab [[buffer(6)]],
                              device bfloat *q_hm [[buffer(7)]],
                              device bfloat *k_hm [[buffer(8)]],
                              device bfloat *v_hm [[buffer(9)]],
                              constant NemoAudQkvParam &p [[buffer(10)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]],
                              uint2 tpg [[threads_per_threadgroup]]) {
  uint nt = tpg.x;
  threadgroup float tmp[NEMO_AUD_SIMD];
  threadgroup float qs[NEMO_AUD_MAX_HEAD_DIM];
  threadgroup float ks[NEMO_AUD_MAX_HEAD_DIM];

  uint h = tg.x, row = tg.y;
  ulong dst_idx = ((ulong)h * p.seq_stride + row) * p.head_dim + d;
  if (row >= p.rows) {
    q_hm[dst_idx] = bfloat(0.0f);
    k_hm[dst_idx] = bfloat(0.0f);
    v_hm[dst_idx] = bfloat(0.0f);
    return;
  }

  uint d_model = p.heads * p.head_dim;
  ulong base = (ulong)row * (3u * d_model) + (ulong)h * p.head_dim + d;
  float qv = qkv[base];
  float kv = qkv[base + d_model];
  float vv = qkv[base + 2u * d_model];

  if (p.qk_norm) {
    float qm = tg_sum(qv, tmp, d, nt) / float(p.head_dim);
    float qd = qv - qm;
    float qinv = rsqrt(tg_sum(qd * qd, tmp, d, nt) / float(p.head_dim) + p.eps);
    qv = bf(qd * qinv * qn_w[d] + qn_b[d]);

    float km = tg_sum(kv, tmp, d, nt) / float(p.head_dim);
    float kd = kv - km;
    float kinv = rsqrt(tg_sum(kd * kd, tmp, d, nt) / float(p.head_dim) + p.eps);
    kv = bf(kd * kinv * kn_w[d] + kn_b[d]);
  }

  qs[d] = qv;
  ks[d] = kv;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint rot_dim = 2u * p.half_rot;
  float q_out = qv, k_out = kv;
  if (d < p.half_rot) {
    float c = cos_tab[(ulong)row * p.half_rot + d];
    float s = sin_tab[(ulong)row * p.half_rot + d];
    q_out = qs[d] * c - qs[d + p.half_rot] * s;
    k_out = ks[d] * c - ks[d + p.half_rot] * s;
  } else if (d < rot_dim) {
    uint j = d - p.half_rot;
    float c = cos_tab[(ulong)row * p.half_rot + j];
    float s = sin_tab[(ulong)row * p.half_rot + j];
    q_out = qs[d] * c + qs[j] * s;
    k_out = ks[d] * c + ks[j] * s;
  }

  q_hm[dst_idx] = bfloat(q_out);
  k_hm[dst_idx] = bfloat(k_out);
  v_hm[dst_idx] = bfloat(vv);
}

// Bidirectional cooperative 64x64 FlashAttention over head-major BF16 Q, K, V
// [heads, seq_stride, D] writing token-major FP32 (BF16-rounded) [rows, H * D].
// Grid: x = ceil(rows / 64), y = heads; threadgroup = 128 threads (4 sgs).
template <uint D>
kernel void nemo_aud_flash_attn(device const bfloat *q_hm [[buffer(0)]],
                                device const bfloat *k_hm [[buffer(1)]],
                                device const bfloat *v_hm [[buffer(2)]],
                                device float *out [[buffer(3)]],
                                constant NemoAudAttnParam &p [[buffer(4)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint q0 = tg.x * NEMO_AUD_TILE;
  uint h = tg.y;
  if (q0 >= p.rows) return;

  threadgroup FlashAttn64Smem<bfloat> smem;
  constexpr auto qk_desc = matmul2d_descriptor(
      NEMO_AUD_TILE, NEMO_AUD_TILE, D, false, true, false,
      matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(
      NEMO_AUD_TILE, D, NEMO_AUD_TILE, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<NEMO_AUD_SGS>> qk_op;
  matmul2d<pv_desc, execution_simdgroups<NEMO_AUD_SGS>> pv_op;

  ulong head_off = (ulong)h * p.seq_stride * D;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)(q_hm + head_off), dextents<int32_t, 2>(D, p.seq_stride));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)(k_hm + head_off), dextents<int32_t, 2>(D, p.seq_stride));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)(v_hm + head_off), dextents<int32_t, 2>(D, p.seq_stride));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>,
         tensor_inline>
      tp(smem.st.probs, extents<int32_t, NEMO_AUD_TILE, NEMO_AUD_TILE>());

  auto q_tile = tq.slice(0, q0);
  auto acc = pv_op.template get_destination_cooperative_tensor<
      decltype(tp), decltype(tv.slice(0, 0)), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  float rmax = -INFINITY, rsum = 0.0f;
  for (uint kv0 = 0; kv0 < p.rows; kv0 += NEMO_AUD_TILE) {
    bool has_mask = (kv0 + NEMO_AUD_TILE) > p.rows;
    auto k_tile = tk.slice(0, kv0);
    auto s = qk_op.template get_destination_cooperative_tensor<
        decltype(q_tile), decltype(k_tile), float>();
    qk_op.run(q_tile, k_tile, s);
    for (uint16_t i = 0; i < s.get_capacity(); i++) {
      if (s.is_valid_element(i)) {
        auto at = s.get_multidimensional_index(i);
        uint col = at[0], row = at[1];
        float v = s[i] * p.log2_scale;
        if (has_mask && kv0 + col >= p.rows) v = -INFINITY;
        smem.scores[row * NEMO_AUD_TILE + col] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint changed = flash_attn_64x64_softmax<bfloat>(smem, sg, lane, has_mask,
                                                    rmax, rsum);
    if (changed) {
      for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
          auto at = acc.get_multidimensional_index(i);
          acc[i] *= smem.st.row_scale[at[1]];
        }
      }
    }
    auto v_tile = tv.slice(0, kv0);
    pv_op.run(tp, v_tile, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint my_row = sg * FLASH_ATTN_64_ROWS_PER_SG + (lane >> 1u);
  if ((lane & 1u) == 0u) smem.st.row_scale[my_row] = rsum > 0.0f ? 1.0f / rsum : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint d_model = p.heads * D;
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint col = at[0], row = q0 + at[1];
      if (row < p.rows) {
        out[(ulong)row * d_model + (ulong)h * D + col] =
            bf(acc[i] * smem.st.row_scale[at[1]]);
      }
    }
  }
}

template [[host_name("nemo_aud_flash_attn_64")]] kernel void
nemo_aud_flash_attn<64>(device const bfloat *, device const bfloat *,
                        device const bfloat *, device float *,
                        constant NemoAudAttnParam &, uint2, uint, uint);
template [[host_name("nemo_aud_flash_attn_80")]] kernel void
nemo_aud_flash_attn<80>(device const bfloat *, device const bfloat *,
                        device const bfloat *, device float *,
                        constant NemoAudAttnParam &, uint2, uint, uint);
template [[host_name("nemo_aud_flash_attn_128")]] kernel void
nemo_aud_flash_attn<128>(device const bfloat *, device const bfloat *,
                         device const bfloat *, device float *,
                         constant NemoAudAttnParam &, uint2, uint, uint);

// Sortformer high-resolution subpixel Conv1d(in=192, out=1536, k=3, pad=1),
// reshaped from [frames, up_factor * in_ch] to [frames * up_factor, in_ch]
// with ReLU for the speaker classifier input.
// Grid: x = ceil(out_ch / 256), y = frames; threadgroup = 256 threads.
kernel void nemo_aud_subpixel_upsample(device const float *x [[buffer(0)]],
                                       device const ushort *w [[buffer(1)]],
                                       device const float *b [[buffer(2)]],
                                       device float *y [[buffer(3)]],
                                       constant NemoAudUpsampleParam &p [[buffer(4)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint t [[thread_index_in_threadgroup]],
                                       uint2 tpg [[threads_per_threadgroup]]) {
  uint nt = tpg.x;
  threadgroup float xs[NEMO_AUD_CONV_K * NEMO_AUD_CONV_IN_CH];
  uint frame = tg.y;
  uint out_ch = p.up_factor * p.in_ch;
  uint total_in = NEMO_AUD_CONV_K * p.in_ch;

  for (uint idx = t; idx < total_in; idx += nt) {
    uint k = idx / p.in_ch;
    uint ic = idx % p.in_ch;
    int src_f = int(frame) + int(k) - 1;
    xs[idx] = (src_f >= 0 && uint(src_f) < p.frames)
                  ? x[(ulong)uint(src_f) * p.in_ch + ic]
                  : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint oc = tg.x * nt + t;
  if (oc >= out_ch) return;
  device const ushort *wr = w + (ulong)oc * p.in_ch * NEMO_AUD_CONV_K;
  float acc = b[oc];
  for (uint ic = 0; ic < p.in_ch; ic++) {
    ulong base = (ulong)ic * NEMO_AUD_CONV_K;
    acc += bf16f(wr[base + 0u]) * xs[0u * p.in_ch + ic];
    acc += bf16f(wr[base + 1u]) * xs[1u * p.in_ch + ic];
    acc += bf16f(wr[base + 2u]) * xs[2u * p.in_ch + ic];
  }
  uint u = oc / p.in_ch;
  uint c = oc % p.in_ch;
  y[(ulong)(frame * p.up_factor + u) * p.in_ch + c] = max(bf(acc), 0.0f);
}

// Applies sigmoid to high-resolution Sortformer speaker logits [frames * factor,
// n_spk] and average-pools factor consecutive frames onto the encoder grid
// [frames, n_spk]. Grid: x = n_spk, y = frames.
kernel void nemo_aud_diar_pool(device const float *logits [[buffer(0)]],
                               device float *probs [[buffer(1)]],
                               constant NemoAudPoolParam &p [[buffer(2)]],
                               uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.n_spk || i.y >= p.frames) return;
  float sum = 0.0f;
  uint base = i.y * p.factor;
  for (uint u = 0; u < p.factor; u++) {
    sum += sigm(logits[(ulong)(base + u) * p.n_spk + i.x]);
  }
  probs[(ulong)i.y * p.n_spk + i.x] = sum / float(p.factor);
}

// Fuses thresholded Sortformer speaker probabilities into ASR encoder states:
// LayerNorm(asr) + scale * (LayerNorm(probs > threshold) @ diar_kernel).
// Grid: rows threadgroups of 256 threads.
kernel void nemo_aud_fuse(device const float *asr [[buffer(0)]],
                          device const float *probs [[buffer(1)]],
                          device const float *asr_nw [[buffer(2)]],
                          device const float *asr_nb [[buffer(3)]],
                          device const float *diar_nw [[buffer(4)]],
                          device const float *diar_nb [[buffer(5)]],
                          device const float *diar_kernel [[buffer(6)]],
                          device float *out [[buffer(7)]],
                          constant NemoAudFuseParam &p [[buffer(8)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_AUD_SIMD];
  threadgroup float dn[NEMO_AUD_MAX_SPK];

  if (t == 0u) {
    device const float *pr = probs + (ulong)row * p.n_spk;
    float s[NEMO_AUD_MAX_SPK];
    float mean = 0.0f;
    for (uint k = 0; k < p.n_spk; k++) {
      s[k] = pr[k] > p.threshold ? 1.0f : 0.0f;
      mean += s[k];
    }
    mean /= float(p.n_spk);
    float var = 0.0f;
    for (uint k = 0; k < p.n_spk; k++) {
      float d = s[k] - mean;
      var += d * d;
    }
    float inv = rsqrt(var / float(p.n_spk) + p.eps);
    for (uint k = 0; k < p.n_spk; k++) {
      dn[k] = bf(((s[k] - mean) * inv) * diar_nw[k] + diar_nb[k]);
    }
  }

  device const float *ar = asr + (ulong)row * p.dim;
  device float *dst = out + (ulong)row * p.dim;
  float sum = 0.0f;
  for (uint i = t; i < p.dim; i += nt) sum += ar[i];
  float mean = tg_sum(sum, tmp, t, nt) / float(p.dim);
  float var = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    float d = ar[i] - mean;
    var += d * d;
  }
  float inv = rsqrt(tg_sum(var, tmp, t, nt) / float(p.dim) + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = t; i < p.dim; i += nt) {
    float an = bf(((ar[i] - mean) * inv) * asr_nw[i] + asr_nb[i]);
    float inf = 0.0f;
    for (uint k = 0; k < p.n_spk; k++)
      inf += dn[k] * diar_kernel[(ulong)k * p.dim + i];
    dst[i] = bf(an + p.scale * bf(inf));
  }
}

// Interleaves even [frames, dim] and odd [frames, dim] ConvTranspose1d outputs
// into [2 * frames, dim]. Grid: x = dim / 4, y = frames.
kernel void nemo_ctc_interleave(device const float4 *even [[buffer(0)]],
                                device const float4 *odd [[buffer(1)]],
                                device float4 *out [[buffer(2)]],
                                constant NemoCtcInterleaveParam &p [[buffer(3)]],
                                uint2 i [[thread_position_in_grid]]) {
  uint cols4 = p.dim / 4u;
  if (i.x >= cols4 || i.y >= p.frames) return;
  ulong src = (ulong)i.y * cols4 + i.x;
  ulong dst_even = (ulong)(2u * i.y) * cols4 + i.x;
  ulong dst_odd = dst_even + cols4;
  out[dst_even] = even[src];
  out[dst_odd] = odd[src];
}

// Computes out = up + scale * bridged elementwise over float4 vectors, with
// optional BF16 rounding. Grid: 1D count4 threads.
kernel void nemo_ctc_residual_add(device const float4 *up [[buffer(0)]],
                                  device const float4 *bridged [[buffer(1)]],
                                  device float4 *out [[buffer(2)]],
                                  constant NemoCtcResidParam &p [[buffer(3)]],
                                  uint idx [[thread_position_in_grid]]) {
  if (idx >= p.count4) return;
  float4 v = up[idx] + p.scale * bridged[idx];
  if (p.round) {
    v = float4(bf(v.x), bf(v.y), bf(v.z), bf(v.w));
  }
  out[idx] = v;
}

// Computes exact FP32 row-wise log_softmax over [rows, vocab], gathers
// log-probabilities for gather_ids[0..n_gather - 1] into out_logp, and writes
// each row's argmax token index into out_argmax.
// Grid: rows threadgroups of 256 threads.
kernel void nemo_ctc_log_softmax_gather(
    device const float *logits [[buffer(0)]],
    device const uint *gather_ids [[buffer(1)]],
    device float *out_logp [[buffer(2)]],
    device uint *out_argmax [[buffer(3)]],
    constant NemoCtcLogSoftmaxParam &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_AUD_SIMD];
  threadgroup uint tmp_id[NEMO_AUD_SIMD];
  if (row >= p.rows) return;
  device const float *lr = logits + (ulong)row * p.vocab;

  float best_v = -INFINITY;
  uint best_id = 0u;
  for (uint v = t; v < p.vocab; v += nt) {
    float x = lr[v];
    if (x > best_v) {
      best_v = x;
      best_id = v;
    }
  }
  float sg_max = simd_max(best_v);
  uint sg_win = simd_min(best_v == sg_max ? best_id : UINT_MAX);
  uint lane = t % NEMO_AUD_SIMD;
  uint sg = t / NEMO_AUD_SIMD;
  uint nsg = (nt + NEMO_AUD_SIMD - 1u) / NEMO_AUD_SIMD;
  if (lane == 0u) {
    tmp[sg] = sg_max;
    tmp_id[sg] = sg_win;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float row_max = -INFINITY;
  uint argmax = 0u;
  for (uint s = 0u; s < nsg; s++) {
    float mv = tmp[s];
    uint mi = tmp_id[s];
    if (mv > row_max || (mv == row_max && mi < argmax)) {
      row_max = mv;
      argmax = mi;
    }
  }
  if (t == 0u) out_argmax[row] = argmax;

  float sum = 0.0f;
  for (uint v = t; v < p.vocab; v += nt) sum += exp(lr[v] - row_max);
  float lse = row_max + log(tg_sum(sum, tmp, t, nt));

  device float *dst = out_logp + (ulong)row * p.n_gather;
  for (uint j = t; j < p.n_gather; j += nt) {
    uint id = gather_ids[j];
    dst[j] = id < p.vocab ? (lr[id] - lse) : -INFINITY;
  }
}
