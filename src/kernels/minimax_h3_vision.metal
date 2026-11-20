// ---- MiniMax-H3 Qwen3-VL BF16 vision tower kernels ----
//
// 27-layer BF16 ViT (width 1152, 16 heads of 72 dims padded to 128 for
// FlashAttention, MLP 4304 padded to 4352 with tanh-approx GELU), bilinear
// 48x48 position embedding interpolation, 2D split-half RoPE, and pre/post
// LayerNorm patch mergers with exact GELU.

// Vision tower geometry matching Qwen3-VL-32B text_encoder/config.json.
constant constexpr uint MINIMAX_H3_VIS_DIM = 1152,
                        MINIMAX_H3_VIS_HEADS = 16,
                        MINIMAX_H3_VIS_HEAD_DIM = 72,
                        MINIMAX_H3_VIS_HALF_HEAD = 36,
                        MINIMAX_H3_VIS_FREQS = 18,
                        MINIMAX_H3_VIS_PAD_HEAD = 128,
                        MINIMAX_H3_VIS_POS_SIDE = 48,
                        MINIMAX_H3_VIS_ATTN_ROWS = 16,
                        MINIMAX_H3_VIS_ATTN_KEYS = 64,
                        MINIMAX_H3_VIS_ATTN_SGROUPS = 4;

// Parameters for bilinearly interpolated 2D position embedding addition.
struct MinimaxH3VisPosAdd {
  uint grid_h, grid_w;
};

// Parameters for vision QKV split, 2D split-half RoPE and head_dim 72->128 pad.
struct MinimaxH3VisHeads {
  uint rows, padded, grid_h, grid_w;
};

// Parameters for vision bidirectional 16-head FlashAttention (head_dim = 72,
// padded to 128 in Q/K/V).
struct MinimaxH3VisAttend {
  uint rows, padded;
};

// Adds bilinearly interpolated 48x48 BF16 position embeddings onto x
// [grid_h * grid_w, 1152] in 2x2 spatial-merge token order (by, bx, my, mx).
// Grid: ((1152 + 31) / 32, grid_h * grid_w, 1), threadgroup: (32, 1, 1).
kernel void minimax_h3_vis_pos_add(device float *x [[buffer(0)]],
                                   device const ushort *pos_embed [[buffer(1)]],
                                   constant MinimaxH3VisPosAdd &p [[buffer(2)]],
                                   uint2 i [[thread_position_in_grid]]) {
  uint col = i.x, tok = i.y;
  uint total = p.grid_h * p.grid_w;
  if (col >= MINIMAX_H3_VIS_DIM || tok >= total) return;

  uint merged_w = p.grid_w / 2u;
  uint blk = tok >> 2;
  uint sub = tok & 3u;
  uint r = (blk / merged_w) * 2u + (sub >> 1);
  uint c = (blk % merged_w) * 2u + (sub & 1u);

  float max_side = float(MINIMAX_H3_VIS_POS_SIDE - 1u);
  float fh = p.grid_h > 1u ? float(r) * max_side / float(p.grid_h - 1u) : 0.0f;
  float fw = p.grid_w > 1u ? float(c) * max_side / float(p.grid_w - 1u) : 0.0f;
  uint h0 = uint(fh), w0 = uint(fw);
  uint h1 = min(h0 + 1u, MINIMAX_H3_VIS_POS_SIDE - 1u);
  uint w1 = min(w0 + 1u, MINIMAX_H3_VIS_POS_SIDE - 1u);
  float dh = fh - float(h0), dw = fw - float(w0);

  uint i00 = (h0 * MINIMAX_H3_VIS_POS_SIDE + w0) * MINIMAX_H3_VIS_DIM + col;
  uint i01 = (h0 * MINIMAX_H3_VIS_POS_SIDE + w1) * MINIMAX_H3_VIS_DIM + col;
  uint i10 = (h1 * MINIMAX_H3_VIS_POS_SIDE + w0) * MINIMAX_H3_VIS_DIM + col;
  uint i11 = (h1 * MINIMAX_H3_VIS_POS_SIDE + w1) * MINIMAX_H3_VIS_DIM + col;

  float pe = (1.0f - dh) * (1.0f - dw) * bf16f(pos_embed[i00]) +
             (1.0f - dh) * dw * bf16f(pos_embed[i01]) +
             dh * (1.0f - dw) * bf16f(pos_embed[i10]) +
             dh * dw * bf16f(pos_embed[i11]);
  x[tok * MINIMAX_H3_VIS_DIM + col] += pe;
}

// In-place tanh-approximated GELU matching torch.nn.functional.gelu(...,
// approximate="tanh"). Clamps the tanh argument to [-15, 15] where tanh
// saturates to +-1 in FP32 so fast-math tanh cannot overflow to NaN.
kernel void minimax_h3_gelu_tanh(device float *x [[buffer(0)]],
                                 constant uint &count [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
  constexpr float CUBIC_COEFF = 0.044715f;
  constexpr float TANH_CLAMP = 15.0f;
  float v = x[i];
  float inner = clamp(SQRT_2_OVER_PI * (v + CUBIC_COEFF * v * v * v),
                      -TANH_CLAMP, TANH_CLAMP);
  x[i] = 0.5f * v * (1.0f + tanh(inner));
}

// High-precision erf approximation (Abramowitz & Stegun 7.1.26, max error
// 1.5e-7) for exact GELU in Qwen3-VL patch mergers.
inline float minimax_h3_erf(float x) {
  float s = x < 0.0f ? -1.0f : 1.0f;
  float a = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float poly = t * (0.254829592f +
                    t * (-0.284496736f +
                         t * (1.421413741f +
                              t * (-1.453152027f + t * 1.061405429f))));
  return s * (1.0f - poly * exp(-a * a));
}

// In-place exact GELU matching torch.nn.functional.gelu (approximate="none").
kernel void minimax_h3_gelu_exact(device float *x [[buffer(0)]],
                                  constant uint &count [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  constexpr float INV_SQRT_2 = 0.7071067811865475f;
  float v = x[i];
  x[i] = 0.5f * v * (1.0f + minimax_h3_erf(v * INV_SQRT_2));
}

// Splits contiguous vision QKV [rows, 3 * 1152] into head-major BF16 qh, kh, vh
// [16, padded, 128] (zero-padding dims 72..127 and rows >= p.rows). Applies 2D
// split-half RoPE to Q and K in 2x2 merge token order (pairs (d, d + 36)).
// Threadgroup: 64 threads; grid.x: 48 (16 Q + 16 K + 16 V); grid.y: padded.
kernel void minimax_h3_vis_heads(device const float *qkv [[buffer(0)]],
                                 device const float *inv_freq [[buffer(1)]],
                                 device bfloat *qh [[buffer(2)]],
                                 device bfloat *kh [[buffer(3)]],
                                 device bfloat *vh [[buffer(4)]],
                                 constant MinimaxH3VisHeads &p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  uint kind = tg.x / MINIMAX_H3_VIS_HEADS;
  uint h = tg.x % MINIMAX_H3_VIS_HEADS;
  uint row = tg.y;
  device bfloat *dst = (kind == 0u ? qh : kind == 1u ? kh : vh) +
                       (h * p.padded + row) * MINIMAX_H3_VIS_PAD_HEAD;
  if (row >= p.rows) {
    dst[t] = bfloat(0.0f);
    dst[t + 64u] = bfloat(0.0f);
    return;
  }

  device const float *src =
      qkv + row * (3u * MINIMAX_H3_VIS_DIM) + kind * MINIMAX_H3_VIS_DIM +
      h * MINIMAX_H3_VIS_HEAD_DIM;
  if (kind == 2u) {
    dst[t] = bfloat(src[t]);
    uint hi = t + 64u;
    dst[hi] = hi < MINIMAX_H3_VIS_HEAD_DIM ? bfloat(src[hi]) : bfloat(0.0f);
    return;
  }

  if (t < MINIMAX_H3_VIS_HALF_HEAD) {
    uint merged_w = p.grid_w / 2u;
    uint blk = row >> 2;
    uint sub = row & 3u;
    uint r_idx = (blk / merged_w) * 2u + (sub >> 1);
    uint c_idx = (blk % merged_w) * 2u + (sub & 1u);
    float coord = t < MINIMAX_H3_VIS_FREQS ? float(r_idx) : float(c_idx);
    uint fi = t < MINIMAX_H3_VIS_FREQS ? t : t - MINIMAX_H3_VIS_FREQS;
    float angle = coord * inv_freq[fi];
    float cs = cos(angle), sn = sin(angle);
    float u0 = src[t], u1 = src[t + MINIMAX_H3_VIS_HALF_HEAD];
    dst[t] = bfloat(u0 * cs - u1 * sn);
    dst[t + MINIMAX_H3_VIS_HALF_HEAD] = bfloat(u1 * cs + u0 * sn);
  }
  uint hi = t + 64u;
  if (hi >= MINIMAX_H3_VIS_HEAD_DIM) dst[hi] = bfloat(0.0f);
}

// Bidirectional 16-head FlashAttention for the Qwen3-VL vision tower
// (head_dim = 72 padded to 128 in Q/K/V, scaled by rsqrt(72)).
kernel void minimax_h3_vis_attend(device const bfloat *qh [[buffer(0)]],
                                  device const bfloat *kh [[buffer(1)]],
                                  device const bfloat *vh [[buffer(2)]],
                                  device float *o [[buffer(3)]],
                                  constant MinimaxH3VisAttend &p [[buffer(4)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint rows = MINIMAX_H3_VIS_ATTN_ROWS;
  constexpr uint keys = MINIMAX_H3_VIS_ATTN_KEYS;
  constexpr uint half_keys = keys / 2u;
  threadgroup float scores[MINIMAX_H3_VIS_ATTN_SGROUPS][rows * keys];
  threadgroup float rescales[MINIMAX_H3_VIS_ATTN_SGROUPS][rows];
  threadgroup float sums[MINIMAX_H3_VIS_ATTN_SGROUPS][rows];
  threadgroup bfloat probs[MINIMAX_H3_VIS_ATTN_SGROUPS][rows * keys];

  uint h = tg.y;
  uint first = (tg.x * MINIMAX_H3_VIS_ATTN_SGROUPS + sg) * rows;
  if (first >= p.rows) return;

  threadgroup float *ss = scores[sg], *rescale = rescales[sg], *sum = sums[sg];
  threadgroup bfloat *ps = probs[sg];
  const float scale = rsqrt(float(MINIMAX_H3_VIS_HEAD_DIM));

  constexpr auto sdesc =
      matmul2d_descriptor(rows, keys, MINIMAX_H3_VIS_PAD_HEAD, false, true,
                          false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(rows, MINIMAX_H3_VIS_PAD_HEAD, keys, false, false,
                          false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qh + h * p.padded * MINIMAX_H3_VIS_PAD_HEAD,
      dextents<int32_t, 2>(MINIMAX_H3_VIS_PAD_HEAD, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)kh + h * p.padded * MINIMAX_H3_VIS_PAD_HEAD,
      dextents<int32_t, 2>(MINIMAX_H3_VIS_PAD_HEAD, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)vh + h * p.padded * MINIMAX_H3_VIS_PAD_HEAD,
      dextents<int32_t, 2>(MINIMAX_H3_VIS_PAD_HEAD, p.padded));
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
      uint out_row = first + at[1], col = at[0];
      if (out_row < p.rows && col < MINIMAX_H3_VIS_HEAD_DIM)
        o[out_row * MINIMAX_H3_VIS_DIM + h * MINIMAX_H3_VIS_HEAD_DIM + col] =
            acc[i] / sum[at[1]];
    }
}
