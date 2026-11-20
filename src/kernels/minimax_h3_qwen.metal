// ---- MiniMax-H3 Qwen3-VL-32B text encoder kernels ----
//
// Activations are FP32 between projections. Attention computes QK-Norm and
// interleaved MRoPE in FP32, writes head-major BF16 Q/K/V, runs causal
// FlashAttention with FP32 scores and online softmax, and writes FP32 outputs.

// Qwen3-VL-32B attention geometry: 64 query heads, 8 KV heads, 128 dim/head.
constant constexpr uint MINIMAX_H3_QWEN_HEAD_DIM = 128,
                        MINIMAX_H3_QWEN_PAIRS = MINIMAX_H3_QWEN_HEAD_DIM / 2,
                        MINIMAX_H3_QWEN_Q_HEADS = 64,
                        MINIMAX_H3_QWEN_KV_HEADS = 8,
                        MINIMAX_H3_QWEN_SHARING =
                            MINIMAX_H3_QWEN_Q_HEADS / MINIMAX_H3_QWEN_KV_HEADS,
                        MINIMAX_H3_QWEN_Q_WIDTH =
                            MINIMAX_H3_QWEN_Q_HEADS * MINIMAX_H3_QWEN_HEAD_DIM,
                        MINIMAX_H3_QWEN_KV_WIDTH =
                            MINIMAX_H3_QWEN_KV_HEADS * MINIMAX_H3_QWEN_HEAD_DIM;

// Interleaved MRoPE H/W replacement boundary (20 H + 20 W in pairs 0..59).
constant constexpr uint MINIMAX_H3_QWEN_MROPE_INTERLEAVE_END = 60;

// Causal FlashAttention tile sizes: 16 query rows over 64-key blocks, 4
// simdgroups per threadgroup.
constant constexpr uint MINIMAX_H3_QWEN_ATTN_ROWS = 16,
                        MINIMAX_H3_QWEN_ATTN_KEYS = 64,
                        MINIMAX_H3_QWEN_ATTN_SGROUPS = 4;

// Parameters for row-wise RMSNorm.
struct MinimaxH3RmsNorm {
  uint n;
  float eps;
};

// Parameters for adding indexed rows (DeepStack visual injection).
struct MinimaxH3AddIndexed {
  uint count, dim;
};

// Parameters for Qwen3-VL QK-Norm and interleaved MRoPE head preparation.
struct MinimaxH3QwenHeads {
  uint rows, padded;
  float eps;
};

// Parameters for Qwen3-VL causal GQA FlashAttention.
struct MinimaxH3QwenAttend {
  uint rows, padded;
};

// Row-wise RMSNorm: y = w * (x / sqrt(mean(x^2) + eps)). One threadgroup per
// row.
kernel void minimax_h3_rmsnorm(device const float *x [[buffer(0)]],
                               device const float *w [[buffer(1)]],
                               device float *y [[buffer(2)]],
                               constant MinimaxH3RmsNorm &p [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]],
                               uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n;
  y += row * p.n;
  float ss = 0.0f;
  for (uint i = t; i < p.n; i += nt) ss += x[i] * x[i];
  float r = rsqrt(tg_sum(ss, tmp, t, nt) / float(p.n) + p.eps);
  for (uint i = t; i < p.n; i += nt) y[i] = w[i] * (x[i] * r);
}

// Elementwise SwiGLU activation: y[i] = silu(g[i]) * u[i].
kernel void minimax_h3_swiglu(device const float *g [[buffer(0)]],
                              device const float *u [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  float a = g[i];
  y[i] = a / (1.0f + exp(-a)) * u[i];
}

// In-place residual add: x[i] += y[i].
kernel void minimax_h3_add(device float *x [[buffer(0)]],
                           device const float *y [[buffer(1)]],
                           constant uint &count [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
  if (i < count) x[i] += y[i];
}

// Adds src[r, :] into x[indices[r], :] for r < count (DeepStack injection).
// Thread x: column; thread y: source row r.
kernel void minimax_h3_add_indexed(device float *x [[buffer(0)]],
                                   device const float *src [[buffer(1)]],
                                   device const int *indices [[buffer(2)]],
                                   constant MinimaxH3AddIndexed &p [[buffer(3)]],
                                   uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.dim || i.y >= p.count) return;
  int dst_row = indices[i.y];
  if (dst_row >= 0) x[uint(dst_row) * p.dim + i.x] += src[i.y * p.dim + i.x];
}

// Prepares head-major BF16 Q, K and V for Qwen3-VL attention.
// Applies per-head RMSNorm and interleaved MRoPE ([24, 20, 20]) to Q and K.
// Rows >= p.rows (up to p.padded) are zeroed so FlashAttention tiles stay clean.
// Threadgroup: 64 threads (one RoPE pair each); grid.x: 64 Q + 8 K + 8 V heads;
// grid.y: row in 0..p.padded-1.
kernel void minimax_h3_qwen_heads(device const float *q [[buffer(0)]],
                                  device const float *k [[buffer(1)]],
                                  device const float *v [[buffer(2)]],
                                  device const float *q_norm [[buffer(3)]],
                                  device const float *k_norm [[buffer(4)]],
                                  device const float *inv_freq [[buffer(5)]],
                                  device const int *pos_ids [[buffer(6)]],
                                  device bfloat *qh [[buffer(7)]],
                                  device bfloat *kh [[buffer(8)]],
                                  device bfloat *vh [[buffer(9)]],
                                  constant MinimaxH3QwenHeads &p [[buffer(10)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]]) {
  uint head = tg.x, row = tg.y;
  uint kv_start = MINIMAX_H3_QWEN_Q_HEADS;
  uint v_start = MINIMAX_H3_QWEN_Q_HEADS + MINIMAX_H3_QWEN_KV_HEADS;
  if (head >= v_start) {
    uint h = head - v_start;
    device bfloat *dst = vh + (h * p.padded + row) * MINIMAX_H3_QWEN_HEAD_DIM;
    if (row >= p.rows) {
      dst[t] = bfloat(0.0f);
      dst[t + MINIMAX_H3_QWEN_PAIRS] = bfloat(0.0f);
      return;
    }
    device const float *src =
        v + row * MINIMAX_H3_QWEN_KV_WIDTH + h * MINIMAX_H3_QWEN_HEAD_DIM;
    dst[t] = bfloat(src[t]);
    dst[t + MINIMAX_H3_QWEN_PAIRS] = bfloat(src[t + MINIMAX_H3_QWEN_PAIRS]);
    return;
  }

  bool is_q = head < kv_start;
  uint h = is_q ? head : head - kv_start;
  device bfloat *dst =
      (is_q ? qh : kh) + (h * p.padded + row) * MINIMAX_H3_QWEN_HEAD_DIM;
  if (row >= p.rows) {
    dst[t] = bfloat(0.0f);
    dst[t + MINIMAX_H3_QWEN_PAIRS] = bfloat(0.0f);
    return;
  }

  threadgroup float tmp[32];
  device const float *src =
      is_q ? q + row * MINIMAX_H3_QWEN_Q_WIDTH + h * MINIMAX_H3_QWEN_HEAD_DIM
           : k + row * MINIMAX_H3_QWEN_KV_WIDTH + h * MINIMAX_H3_QWEN_HEAD_DIM;
  device const float *w = is_q ? q_norm : k_norm;
  float a = src[t], b = src[t + MINIMAX_H3_QWEN_PAIRS];
  float r = rsqrt(tg_sum(a * a + b * b, tmp, t, MINIMAX_H3_QWEN_PAIRS) /
                      float(MINIMAX_H3_QWEN_HEAD_DIM) +
                  p.eps);
  a = w[t] * (a * r);
  b = w[t + MINIMAX_H3_QWEN_PAIRS] * (b * r);

  // Interleaved MRoPE: axis 1 (H) at 1,4,...,58; axis 2 (W) at 2,5,...,59;
  // axis 0 (T) everywhere else.
  uint axis = 0;
  if (t < MINIMAX_H3_QWEN_MROPE_INTERLEAVE_END) {
    uint mod3 = t % 3u;
    if (mod3 == 1u) axis = 1u;
    else if (mod3 == 2u) axis = 2u;
  }
  float angle = float(pos_ids[row * 3u + axis]) * inv_freq[t];
  float c = cos(angle), s = sin(angle);
  float rot_a = a * c - b * s;
  float rot_b = b * c + a * s;
  dst[t] = bfloat(rot_a);
  dst[t + MINIMAX_H3_QWEN_PAIRS] = bfloat(rot_b);
}

// Causal GQA FlashAttention for Qwen3-VL (64 query heads, 8 KV heads, dim 128).
// Each threadgroup has 4 simdgroups (128 threads), handling 64 query rows for
// one query head (grid.y = head, grid.x = ceil(rows / 64)).
kernel void minimax_h3_qwen_attend(device const bfloat *qh [[buffer(0)]],
                                   device const bfloat *kh [[buffer(1)]],
                                   device const bfloat *vh [[buffer(2)]],
                                   device float *o [[buffer(3)]],
                                   constant MinimaxH3QwenAttend &p [[buffer(4)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint rows = MINIMAX_H3_QWEN_ATTN_ROWS;
  constexpr uint keys = MINIMAX_H3_QWEN_ATTN_KEYS;
  constexpr uint half_keys = keys / 2u;
  threadgroup float scores[MINIMAX_H3_QWEN_ATTN_SGROUPS][rows * keys];
  threadgroup float rescales[MINIMAX_H3_QWEN_ATTN_SGROUPS][rows];
  threadgroup float sums[MINIMAX_H3_QWEN_ATTN_SGROUPS][rows];
  threadgroup bfloat probs[MINIMAX_H3_QWEN_ATTN_SGROUPS][rows * keys];

  uint h = tg.y, kv = h / MINIMAX_H3_QWEN_SHARING;
  uint first = (tg.x * MINIMAX_H3_QWEN_ATTN_SGROUPS + sg) * rows;
  if (first >= p.rows) return;
  uint last = min(first + rows, p.rows) - 1u;

  threadgroup float *ss = scores[sg], *rescale = rescales[sg], *sum = sums[sg];
  threadgroup bfloat *ps = probs[sg];
  const float scale = rsqrt(float(MINIMAX_H3_QWEN_HEAD_DIM));

  constexpr auto sdesc =
      matmul2d_descriptor(rows, keys, MINIMAX_H3_QWEN_HEAD_DIM, false, true,
                          false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(rows, MINIMAX_H3_QWEN_HEAD_DIM, keys, false, false,
                          false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qh + h * p.padded * MINIMAX_H3_QWEN_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_QWEN_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)kh + kv * p.padded * MINIMAX_H3_QWEN_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_QWEN_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)vh + kv * p.padded * MINIMAX_H3_QWEN_HEAD_DIM,
      dextents<int32_t, 2>(MINIMAX_H3_QWEN_HEAD_DIM, p.padded));
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
  uint key_end = last + 1u;

  for (uint k0 = 0; k0 < key_end; k0 += keys) {
    auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt),
                                                     float>();
    sop.run(qt, kt, sc);
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto at = sc.get_multidimensional_index(i);
        uint key = k0 + at[0], query = first + at[1];
        bool seen = query <= last && key <= query;
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
        o[out_row * MINIMAX_H3_QWEN_Q_WIDTH + h * MINIMAX_H3_QWEN_HEAD_DIM +
          at[0]] = acc[i] / sum[at[1]];
    }
}
