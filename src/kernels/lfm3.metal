// ---- LFM3: the language model of LiquidAI's d1 decision models ----
//
// A pass's rows are FP32 in memory. The products (mm_bf16) read them as BF16
// and add up in FP32. The norms, the rotations, the short convolution, the
// softmax and the gates run in FP32. Attention multiplies BF16 queries, keys,
// probabilities and values, and adds up in FP32.

// The attention's shape (src/model/lfm3/model.h): 16 query heads of 128 values
// that share 4 key and value heads, 4 query heads each.
constant constexpr uint LFM3_HEAD_DIM = 128, LFM3_PAIRS = LFM3_HEAD_DIM / 2,
                        LFM3_QUERY_HEADS = 16, LFM3_KEY_HEADS = 4,
                        LFM3_SHARING = LFM3_QUERY_HEADS / LFM3_KEY_HEADS,
                        LFM3_QUERY_WIDTH = LFM3_QUERY_HEADS * LFM3_HEAD_DIM,
                        LFM3_KEY_WIDTH = LFM3_KEY_HEADS * LFM3_HEAD_DIM;

// An RMSNorm's row length and epsilon.
struct Lfm3Norm {
  uint n;
  float eps;
};

// The rows of a short convolution: dim channels, the filter's width, and the
// rows of in_proj's outputs before the block's first row.
struct Lfm3Conv {
  uint dim, width, before;
};

// What lfm3_heads needs: the block's rows and its first row (its position),
// the rows that the BF16 arrays hold per head, the value pairs that RoPE
// turns, and the norms' epsilon.
struct Lfm3Heads {
  uint rows, first, padded, rotated;
  float eps;
};

// What lfm3_attend needs: the sequence's rows, the block's first row and rows,
// the rows that the BF16 arrays hold per head, the state's rows and the
// sliding window (0 on a global layer).
struct Lfm3Attend {
  uint rows, first, block_rows, padded, state, window;
};

// Embedding: row r of x gets token ids[r]'s BF16 row of the table as FP32,
// for each row whose id is 0 or more. Thread x: one of the row's dim values.
kernel void lfm3_embed(device const ushort *table [[buffer(0)]], device const int *ids [[buffer(1)]],
                       device float *x [[buffer(2)]], constant uint &dim [[buffer(3)]],
                       uint2 i [[thread_position_in_grid]]) {
  int id = ids[i.y];
  if (id >= 0) x[i.y * dim + i.x] = bf16f(table[uint(id) * dim + i.x]);
}

// RMSNorm of each row of n values: y = w (x / sqrt(mean(x^2) + eps)). One
// threadgroup per row.
kernel void lfm3_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant Lfm3Norm &p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n, y += row * p.n;
  float squares = 0;
  for (uint i = t; i < p.n; i += nt) squares += x[i] * x[i];
  float r = rsqrt(tg_sum(squares, tmp, t, nt) / p.n + p.eps);
  for (uint i = t; i < p.n; i += nt) y[i] = w[i] * (x[i] * r);
}

// A sandwich norm's residual: x += w (y / sqrt(mean(y^2) + eps)) for each
// row of n values. One threadgroup per row.
kernel void lfm3_norm_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                          device const float *w [[buffer(2)]], constant Lfm3Norm &p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n, y += row * p.n;
  float squares = 0;
  for (uint i = t; i < p.n; i += nt) squares += y[i] * y[i];
  float r = rsqrt(tg_sum(squares, tmp, t, nt) / p.n + p.eps);
  for (uint i = t; i < p.n; i += nt) x[i] += w[i] * (y[i] * r);
}

// Copies count floats from x to y.
kernel void lfm3_copy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) y[i] = x[i];
}

// The gated short convolution of a block of rows. Each row of bcx holds
// in_proj's outputs B, C and x, dim values each, for the block's rows and the
// `before` rows ahead of them. With u = B x, channel c of output row t is
// C[t][c] times the sum over j < width of w[c][j] u[t - width + 1 + j][c].
// Rows before the sequence's start count as zero. Thread x: the channel;
// thread y: the block's row.
kernel void lfm3_conv(device const float *bcx [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant Lfm3Conv &p [[buffer(3)]],
                      uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, t = i.y + p.before;
  float sum = 0;
  for (uint j = 0; j < p.width; j++) {
    if (t + j + 1 < p.width) continue;
    device const float *row = bcx + (t + j + 1 - p.width) * 3 * p.dim;
    sum += w[c * p.width + j] * (row[c] * row[2 * p.dim + c]);
  }
  y[i.y * p.dim + c] = bcx[t * 3 * p.dim + p.dim + c] * sum;
}

// The feed-forward's activation: y = silu(g) u.
kernel void lfm3_swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                        device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float a = g[i];
  y[i] = a / (1.0f + exp(-a)) * u[i];
}

// Queries, keys and values for attention, from the block's rows of q, k and v.
// Each query and key head is normalized with its weights (RMSNorm) and turned
// (RoPE: value i < 64 pairs with value i + 64, and the first `rotated` pairs
// turn by the angle position * inv_freq[i]). All three are stored as BF16, a
// row of 128 values per head: head h of the block's row r at row
// h * padded + first + r of qh, kh and vh. The block holds the sequence's
// rows from first on, and their positions. A threadgroup of 64 threads, one
// value pair each, takes one head of one row: grid x is the query heads, then
// the key heads, then the value heads; grid y is the row.
kernel void lfm3_heads(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                       device const float *v [[buffer(2)]], device const float *q_norm [[buffer(3)]],
                       device const float *k_norm [[buffer(4)]], device const float *inv_freq [[buffer(5)]],
                       device bfloat *qh [[buffer(6)]], device bfloat *kh [[buffer(7)]],
                       device bfloat *vh [[buffer(8)]], constant Lfm3Heads &p [[buffer(9)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint head = tg.x, row = tg.y, position = p.first + row;
  if (head >= LFM3_QUERY_HEADS + LFM3_KEY_HEADS) {
    uint h = head - LFM3_QUERY_HEADS - LFM3_KEY_HEADS;
    device const float *src = v + row * LFM3_KEY_WIDTH + h * LFM3_HEAD_DIM;
    device bfloat *dst = vh + (h * p.padded + position) * LFM3_HEAD_DIM;
    dst[t] = bfloat(src[t]), dst[t + LFM3_PAIRS] = bfloat(src[t + LFM3_PAIRS]);
    return;
  }
  bool query = head < LFM3_QUERY_HEADS;
  uint h = query ? head : head - LFM3_QUERY_HEADS;
  device const float *src = query ? q + row * LFM3_QUERY_WIDTH + h * LFM3_HEAD_DIM
                                  : k + row * LFM3_KEY_WIDTH + h * LFM3_HEAD_DIM;
  device const float *w = query ? q_norm : k_norm;
  device bfloat *dst = (query ? qh : kh) + (h * p.padded + position) * LFM3_HEAD_DIM;
  float a = src[t], b = src[t + LFM3_PAIRS];
  float r = rsqrt(tg_sum(a * a + b * b, tmp, t, LFM3_PAIRS) / LFM3_HEAD_DIM + p.eps);
  a = w[t] * (a * r), b = w[t + LFM3_PAIRS] * (b * r);
  if (t < p.rotated) {
    float angle = float(position) * inv_freq[t], c = cos(angle), s = sin(angle);
    float turned = a * c - b * s;
    b = b * c + a * s, a = turned;
  }
  dst[t] = bfloat(a), dst[t + LFM3_PAIRS] = bfloat(b);
}

// Whether query row q sees key row k. On a global layer, the state's rows see
// every row, and a question's rows see the state's rows and their own
// question's rows; group holds each row's question, 0 for the state's rows.
// On a sliding layer, a row sees itself and the window - 1 rows before it.
inline bool lfm3_sees(uint q, uint k, constant Lfm3Attend &p, device const uint *group) {
  if (p.window) return k <= q && k + p.window > q;
  return group[q] == 0 || group[k] == 0 || group[q] == group[k];
}

// Attention of the block's queries, each query head's output times the
// sigmoid of its gate; query head h reads key head h / LFM3_SHARING, and the
// scores are scaled by 1 / sqrt(128). Dispatched as a 1D head-coherent grid
// (ceil(block_rows / 64) * 16) so concurrent threadgroups share each KV head
// in cache. All 4 simdgroups cooperate on one 64x64 tile using exact 16 KiB
// shared threadgroup memory. A threadgroup reads the keys of at most two
// ranges, which hold every key that its rows see: all rows when its first row
// is in the state; else the state's rows, and the rows from its first row's
// question to its last row's (spans[g]: question g's rows, spans[0] the
// state's). qh, kh and vh hold padded rows per head, with zeros in at least
// 64 rows after the sequence's. gate: the block's rows, one value per query
// head. o: the block's rows, one row of 128 per query head.
kernel void lfm3_attend(device const bfloat *qh [[buffer(0)]], device const bfloat *kh [[buffer(1)]],
                        device const bfloat *vh [[buffer(2)]], device const float *gate [[buffer(3)]],
                        device const uint *group [[buffer(4)]], device const uint2 *spans [[buffer(5)]],
                        device float *o [[buffer(6)]], constant Lfm3Attend &p [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BQ = FLASH_ATTN_64_TILE, BK = FLASH_ATTN_64_TILE;
  threadgroup FlashAttn64Smem<bfloat> smem;

  uint q_tiles = (p.block_rows + BQ - 1u) / BQ;
  uint h = tg / q_tiles, kv = h / LFM3_SHARING;
  uint local = (tg % q_tiles) * BQ;
  if (local >= p.block_rows) return;
  uint first = p.first + local, last = min(local + BQ, p.block_rows) - 1u + p.first;
  bool full_q = local + BQ <= p.block_rows;
  const float scale_log2 = rsqrt(float(LFM3_HEAD_DIM)) * M_LOG2E_F;

  constexpr auto sdesc = matmul2d_descriptor(BQ, BK, LFM3_HEAD_DIM, false, true, false,
                                             matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(BQ, LFM3_HEAD_DIM, BK, false, false, false,
                                             matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> sop;
  matmul2d<odesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qh + (ulong)h * p.padded * LFM3_HEAD_DIM,
      dextents<int32_t, 2>(LFM3_HEAD_DIM, p.rows));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)kh + (ulong)kv * p.padded * LFM3_HEAD_DIM,
      dextents<int32_t, 2>(LFM3_HEAD_DIM, p.rows));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)vh + (ulong)kv * p.padded * LFM3_HEAD_DIM,
      dextents<int32_t, 2>(LFM3_HEAD_DIM, p.rows));
  tensor<threadgroup bfloat, extents<int32_t, BK, BQ>, tensor_inline> tp(
      smem.st.probs, extents<int32_t, BK, BQ>());

  auto qt = tq.slice(0, first), vt0 = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  float rmax = -INFINITY, rsum = 0.0f;
  uint2 ranges[2];
  uint count = 1u;
  uint g_first = p.window ? 0u : group[first];
  uint g_last = p.window ? 0u : group[last];
  if (p.window)
    ranges[0] = uint2(first + 1u > p.window ? first + 1u - p.window : 0u, last + 1u);
  else if (g_first == 0u)
    ranges[0] = uint2(0u, p.rows);
  else
    ranges[0] = uint2(0u, p.state), ranges[1] = uint2(spans[g_first].x, spans[g_last].y), count = 2u;

  for (uint range = 0; range < count; range++) {
    uint r_end = ranges[range].y;
    bool global_unmasked = !p.window && (range == 0u ? (g_first > 0u || g_last == 0u) : (g_first == g_last));
    for (uint k0 = ranges[range].x; k0 < r_end; k0 += BK) {
      auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
      auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
      sop.run(qt, kt, sc);
      bool full_tile = full_q && (k0 + BK <= r_end) &&
                       (p.window ? (k0 + BK - 1u <= first && last < k0 + p.window) : global_unmasked);
      if (full_tile) {
        for (uint16_t i = 0; i < sc.get_capacity(); i++)
          if (sc.is_valid_element(i)) {
            auto at = sc.get_multidimensional_index(i);
            smem.scores[at[1] * BK + at[0]] = sc[i] * scale_log2;
          }
      } else {
        for (uint16_t i = 0; i < sc.get_capacity(); i++)
          if (sc.is_valid_element(i)) {
            auto at = sc.get_multidimensional_index(i);
            uint key = k0 + at[0], query = first + at[1];
            bool seen = key < r_end && query <= last && lfm3_sees(query, key, p, group);
            smem.scores[at[1] * BK + at[0]] = seen ? sc[i] * scale_log2 : -INFINITY;
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (flash_attn_64x64_softmax(smem, sg, lane, !full_tile, rmax, rsum)) {
        for (uint16_t i = 0; i < acc.get_capacity(); i++)
          if (acc.is_valid_element(i))
            acc[i] *= smem.st.row_scale[acc.get_multidimensional_index(i)[1]];
      }
      oop.run(tp, vt, acc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  uint row = sg * FLASH_ATTN_64_ROWS_PER_SG + (lane >> 1u);
  if ((lane & 1u) == 0u) {
    uint out_row = local + row;
    float g = out_row < p.block_rows ? gate[(ulong)out_row * LFM3_QUERY_HEADS + h] : 0.0f;
    smem.st.row_scale[row] = (1.0f / rsum) / (1.0f + exp(-g));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint out_row = local + at[1];
      if (out_row < p.block_rows)
        o[(ulong)out_row * LFM3_QUERY_WIDTH + h * LFM3_HEAD_DIM + at[0]] =
            acc[i] * smem.st.row_scale[at[1]];
    }
}
