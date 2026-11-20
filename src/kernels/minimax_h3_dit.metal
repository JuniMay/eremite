// ---- MiniMax-H3-Image DiT and TokenRefiner kernels ----
//
// TokenRefiner runs in BF16 without RoPE or AdaLN. The 50 DiT blocks use
// INT8-ConvRot weights (minimax_h3_quant.metal), pruned curve-form AdaLN
// projections (F16 weight [96768, 8] + BF16 bias), partial 3D split-half RoPE
// (rot_dim = 96 of 128 head dims), and bidirectional 56-head FlashAttention.

// MiniMax-H3 DiT geometry: width 5376, 56 heads of 128 dims (inner 7168).
constant constexpr uint MINIMAX_H3_DIT_DIM = 5376,
                        MINIMAX_H3_DIT_HEADS = 56,
                        MINIMAX_H3_DIT_HEAD_DIM = 128,
                        MINIMAX_H3_DIT_HALF_HEAD = MINIMAX_H3_DIT_HEAD_DIM / 2,
                        MINIMAX_H3_DIT_INNER =
                            MINIMAX_H3_DIT_HEADS * MINIMAX_H3_DIT_HEAD_DIM,
                        MINIMAX_H3_DIT_ROT_PAIRS = 48,
                        MINIMAX_H3_DIT_ROPE_FREQS = 16,
                        MINIMAX_H3_DIT_UNROT_DIMS = 32,
                        MINIMAX_H3_DIT_ADALN_RANK = 8,
                        MINIMAX_H3_DIT_TILE_M = 64,
                        MINIMAX_H3_DIT_ATTN_ROWS = 16,
                        MINIMAX_H3_DIT_ATTN_KEYS = 64,
                        MINIMAX_H3_DIT_ATTN_SGROUPS = 4;

// Parameters for BF16 matrix multiplication with optional FP32 bias.
struct MinimaxH3Bf16MM {
  uint m, n, k, ldy, has_bias;
};

// Parameters for FP32 matrix multiplication with bias (patch embed and PDD out).
struct MinimaxH3F32MM {
  uint m, n, k, ldy;
};

// Parameters for curve-form AdaLN projection: out[t, j] = dot(w[j], t_emb[t]) + b[j].
struct MinimaxH3DitAdaln {
  uint num_t, out_dim;
};

// Parameters for fused RMSNorm + AdaLN modulation.
struct MinimaxH3DitNormMod {
  uint dim, mod_stride, shift_off, scale_off;
  float eps;
};

// Parameters for fused AdaLN gated residual addition.
struct MinimaxH3DitGateAdd {
  uint rows, dim, mod_stride, gate_off;
};

// Parameters for DiT/Refiner contiguous QKV split, affine QK-Norm and optional
// partial 3D split-half RoPE (rot_dim = 96).
struct MinimaxH3DitHeads {
  uint rows, padded, use_rope;
  float eps;
};

// Parameters for DiT/Refiner bidirectional 56-head FlashAttention.
struct MinimaxH3DitAttend {
  uint rows, padded;
};

// Native BF16 matrix product accumulating in FP32, with optional FP32 bias.
// xb must have rows padded to a multiple of 64; k and n must be multiples of 64.
kernel void minimax_h3_mm_bf16(device const bfloat *w [[buffer(0)]],
                               device const bfloat *xb [[buffer(1)]],
                               device const float *bias [[buffer(2)]],
                               device float *y [[buffer(3)]],
                               constant MinimaxH3Bf16MM &p [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc =
      matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true,
                          false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  uint padded_m = (p.m + MINIMAX_H3_DIT_TILE_M - 1u) / MINIMAX_H3_DIT_TILE_M *
                  MINIMAX_H3_DIT_TILE_M;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)xb, dextents<int32_t, 2>(p.k, padded_m));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tw(
      (device bfloat *)w, dextents<int32_t, 2>(p.k, p.n));
  uint m0 = tg.y * 64u, n0 = tg.x * 64u;
  auto a = tx.slice(0, m0);
  auto b = tw.slice(0, n0);
  auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b),
                                                 float>();
  op.run(a, b, c);
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint col = n0 + at[0], row = m0 + at[1];
      if (col < p.n && row < p.m)
        y[row * p.ldy + col] = c[i] + (p.has_bias ? bias[col] : 0.0f);
    }
}

// FP32 matrix product with bias: y[r, c] = dot(x[r], w[c]) + bias[c].
// Used for the checkpoint's FP32 islands (video_patch_proj and final video_out).
// One simdgroup (32 threads) per (row, col) output element: grid.x = n, grid.y = m.
kernel void minimax_h3_mm_f32(device const float *w [[buffer(0)]],
                              device const float *x [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *y [[buffer(3)]],
                              constant MinimaxH3F32MM &p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  uint col = tg.x, row = tg.y;
  if (col >= p.n || row >= p.m) return;
  device const float *xr = x + row * p.k;
  device const float *wr = w + col * p.k;
  float sum = 0.0f;
  for (uint j = lane; j < p.k; j += 32u) sum += xr[j] * wr[j];
  sum = simd_sum(sum);
  if (lane == 0u) y[row * p.ldy + col] = sum + bias[col];
}

// Evaluates a pruned curve-form AdaLN linear layer on num_t rank-8 time
// embeddings without SiLU: out[t, j] = sum_{k=0..7} float(w[j, k]) * t_emb[t, k] + bias[j].
// Thread x: output index j < out_dim; thread y: timestep index t < num_t.
kernel void minimax_h3_dit_adaln(device const half *w [[buffer(0)]],
                                 device const ushort *bias [[buffer(1)]],
                                 device const float *t_emb [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 constant MinimaxH3DitAdaln &p [[buffer(4)]],
                                 uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.out_dim || i.y >= p.num_t) return;
  device const half *wr = w + i.x * MINIMAX_H3_DIT_ADALN_RANK;
  device const float *te = t_emb + i.y * MINIMAX_H3_DIT_ADALN_RANK;
  float acc = bf16f(bias[i.x]);
  for (uint k = 0; k < MINIMAX_H3_DIT_ADALN_RANK; k++)
    acc += float(wr[k]) * te[k];
  out[i.y * p.out_dim + i.x] = acc;
}

// Fused RMSNorm + AdaLN modulation:
// y[r, c] = (w[c] * x[r, c] / rms(x[r])) * (1 + mod[mr, scale_off + c]) + mod[mr, shift_off + c].
// One threadgroup of 256 threads per token row r.
kernel void minimax_h3_dit_norm_mod(device const float *x [[buffer(0)]],
                                    device const float *w [[buffer(1)]],
                                    device const float *mod [[buffer(2)]],
                                    device const uint *mod_rows [[buffer(3)]],
                                    device float *y [[buffer(4)]],
                                    constant MinimaxH3DitNormMod &p [[buffer(5)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint t [[thread_index_in_threadgroup]],
                                    uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  device const float *xr = x + row * p.dim;
  device float *yr = y + row * p.dim;
  float ss = 0.0f;
  for (uint c = t; c < p.dim; c += nt) ss += xr[c] * xr[c];
  float inv = rsqrt(tg_sum(ss, tmp, t, nt) / float(p.dim) + p.eps);
  uint mr = mod_rows[row];
  device const float *shift = mod + mr * p.mod_stride + p.shift_off;
  device const float *scale = mod + mr * p.mod_stride + p.scale_off;
  for (uint c = t; c < p.dim; c += nt) {
    float normed = w[c] * (xr[c] * inv);
    yr[c] = normed * (1.0f + scale[c]) + shift[c];
  }
}

// Fused AdaLN gated residual addition: x[r, c] += mod[mr, gate_off + c] * y[r, c].
kernel void minimax_h3_dit_gate_add(device float *x [[buffer(0)]],
                                    device const float *y [[buffer(1)]],
                                    device const float *mod [[buffer(2)]],
                                    device const uint *mod_rows [[buffer(3)]],
                                    constant MinimaxH3DitGateAdd &p [[buffer(4)]],
                                    uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.dim || i.y >= p.rows) return;
  uint mr = mod_rows[i.y];
  uint idx = i.y * p.dim + i.x;
  x[idx] += mod[mr * p.mod_stride + p.gate_off + i.x] * y[idx];
}

// Splits contiguous QKV [rows, 3 * 7168] into head-major BF16 qh, kh, vh
// [56, padded, 128]. Applies affine per-head QK RMSNorm, and when use_rope != 0,
// applies partial 3D split-half RoPE on the first 96 dims (pairs (i, i + 48)
// for i = 0..47, dims 96..127 unrotated). Zeroes rows >= p.rows.
// Threadgroup: 64 threads; grid.x: 168 (56 Q + 56 K + 56 V); grid.y: padded.
kernel void minimax_h3_dit_heads(device const float *qkv [[buffer(0)]],
                                 device const float *q_norm [[buffer(1)]],
                                 device const float *k_norm [[buffer(2)]],
                                 device const float *pos [[buffer(3)]],
                                 device const float *inv_freq [[buffer(4)]],
                                 device bfloat *qh [[buffer(5)]],
                                 device bfloat *kh [[buffer(6)]],
                                 device bfloat *vh [[buffer(7)]],
                                 constant MinimaxH3DitHeads &p [[buffer(8)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  uint kind = tg.x / MINIMAX_H3_DIT_HEADS;
  uint h = tg.x % MINIMAX_H3_DIT_HEADS;
  uint row = tg.y;
  device bfloat *dst = (kind == 0u ? qh : kind == 1u ? kh : vh) +
                       (h * p.padded + row) * MINIMAX_H3_DIT_HEAD_DIM;
  if (row >= p.rows) {
    dst[t] = bfloat(0.0f);
    dst[t + MINIMAX_H3_DIT_HALF_HEAD] = bfloat(0.0f);
    return;
  }

  device const float *src =
      qkv + row * (3u * MINIMAX_H3_DIT_INNER) + kind * MINIMAX_H3_DIT_INNER +
      h * MINIMAX_H3_DIT_HEAD_DIM;
  float a = src[t];
  float b = src[t + MINIMAX_H3_DIT_HALF_HEAD];
  if (kind == 2u) {
    dst[t] = bfloat(a);
    dst[t + MINIMAX_H3_DIT_HALF_HEAD] = bfloat(b);
    return;
  }

  threadgroup float tmp[32];
  threadgroup float hnorm[MINIMAX_H3_DIT_HEAD_DIM];
  device const float *w = kind == 0u ? q_norm : k_norm;
  float r = rsqrt(tg_sum(a * a + b * b, tmp, t, MINIMAX_H3_DIT_HALF_HEAD) /
                      float(MINIMAX_H3_DIT_HEAD_DIM) +
                  p.eps);
  hnorm[t] = w[t] * (a * r);
  hnorm[t + MINIMAX_H3_DIT_HALF_HEAD] =
      w[t + MINIMAX_H3_DIT_HALF_HEAD] * (b * r);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!p.use_rope) {
    dst[t] = bfloat(hnorm[t]);
    dst[t + MINIMAX_H3_DIT_HALF_HEAD] =
        bfloat(hnorm[t + MINIMAX_H3_DIT_HALF_HEAD]);
    return;
  }

  if (t < MINIMAX_H3_DIT_ROT_PAIRS) {
    uint axis = t / MINIMAX_H3_DIT_ROPE_FREQS;
    uint fi = t % MINIMAX_H3_DIT_ROPE_FREQS;
    float angle = pos[row * 3u + axis] * inv_freq[fi];
    float c = cos(angle), s = sin(angle);
    float u0 = hnorm[t], u1 = hnorm[t + MINIMAX_H3_DIT_ROT_PAIRS];
    dst[t] = bfloat(u0 * c - u1 * s);
    dst[t + MINIMAX_H3_DIT_ROT_PAIRS] = bfloat(u1 * c + u0 * s);
  }
  if (t < MINIMAX_H3_DIT_UNROT_DIMS) {
    uint idx = 2u * MINIMAX_H3_DIT_ROT_PAIRS + t;
    dst[idx] = bfloat(hnorm[idx]);
  }
}

// Bidirectional 56-head FlashAttention (head_dim = 128) for DiT and Refiner.
// Uses BF16 Q/K/V/P with FP32 scores, online softmax and output accumulation.
kernel void minimax_h3_dit_attend(device const bfloat *qh [[buffer(0)]],
                                  device const bfloat *kh [[buffer(1)]],
                                  device const bfloat *vh [[buffer(2)]],
                                  device float *o [[buffer(3)]],
                                  constant MinimaxH3DitAttend &p [[buffer(4)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint rows = MINIMAX_H3_DIT_ATTN_ROWS;
  constexpr uint keys = MINIMAX_H3_DIT_ATTN_KEYS;
  constexpr uint half_keys = keys / 2u;
  threadgroup float scores[MINIMAX_H3_DIT_ATTN_SGROUPS][rows * keys];
  threadgroup float rescales[MINIMAX_H3_DIT_ATTN_SGROUPS][rows];
  threadgroup float sums[MINIMAX_H3_DIT_ATTN_SGROUPS][rows];
  threadgroup bfloat probs[MINIMAX_H3_DIT_ATTN_SGROUPS][rows * keys];

  uint h = tg.y;
  uint first = (tg.x * MINIMAX_H3_DIT_ATTN_SGROUPS + sg) * rows;
  if (first >= p.rows) return;

  threadgroup float *ss = scores[sg], *rescale = rescales[sg], *sum = sums[sg];
  threadgroup bfloat *ps = probs[sg];
  const float scale = rsqrt(float(MINIMAX_H3_DIT_HEAD_DIM));

  constexpr auto sdesc =
      matmul2d_descriptor(rows, keys, MINIMAX_H3_DIT_HEAD_DIM, false, true,
                          false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(rows, MINIMAX_H3_DIT_HEAD_DIM, keys, false, false,
                          false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qh + h * p.padded * MINIMAX_H3_DIT_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_DIT_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)kh + h * p.padded * MINIMAX_H3_DIT_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_DIT_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)vh + h * p.padded * MINIMAX_H3_DIT_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_DIT_HEAD_DIM, p.padded));
  tensor<threadgroup bfloat, extents<int32_t, keys, rows>, tensor_inline> tp(
      ps, extents<int32_t, keys, rows>());

  auto qt = tq.slice(0, first), vt0 = tv.slice(0, 0);
  auto acc =
      oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt0),
                                             float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  uint row = lane / 2u, column = (lane % 2u) * half_keys;
  float running_max = -INFINITY, running_sum = 0.0f;

  for (uint k0 = 0; k0 < p.rows; k0 += keys) {
    auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt),
                                                     float>();
    sop.run(qt, kt, sc);
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto at = sc.get_multidimensional_index(i);
        uint key = k0 + at[0], query = first + at[1];
        bool seen = query < p.rows && key < p.rows;
        ss[at[1] * keys + at[0]] = seen ? sc[i] * scale : -INFINITY;
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float *own = ss + row * keys + column;
    float block_max = -INFINITY;
    for (uint c = 0; c < half_keys; c++) block_max = max(block_max, own[c]);
    block_max = max(block_max, simd_shuffle_xor(block_max, 1u));
    float new_max = max(running_max, block_max);
    float old_scale =
        running_max == -INFINITY ? 0.0f : exp(running_max - new_max);
    float block_sum = 0.0f;
    for (uint c = 0; c < half_keys; c++) {
      float e = own[c] == -INFINITY ? 0.0f : exp(own[c] - new_max);
      ps[row * keys + column + c] = bfloat(e);
      block_sum += e;
    }
    block_sum += simd_shuffle_xor(block_sum, 1u);
    running_sum = running_sum * old_scale + block_sum;
    running_max = new_max;
    if (lane % 2u == 0u) rescale[row] = old_scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint16_t i = 0; i < acc.get_capacity(); i++)
      if (acc.is_valid_element(i))
        acc[i] *= rescale[acc.get_multidimensional_index(i)[1]];
    oop.run(tp, vt, acc);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane % 2u == 0u) sum[row] = running_sum;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint out_row = first + at[1];
      if (out_row < p.rows)
        o[out_row * MINIMAX_H3_DIT_INNER + h * MINIMAX_H3_DIT_HEAD_DIM +
          at[0]] = acc[i] / sum[at[1]];
    }
}
