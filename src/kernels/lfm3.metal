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
// lfm3_attend's tiles: a simdgroup takes 16 query rows, over blocks of 64
// keys; a threadgroup's 4 simdgroups take 64 query rows.
constant constexpr uint LFM3_TILE_ROWS = 16, LFM3_KEY_BLOCK = 64,
                        LFM3_TILE_SIMDGROUPS = 4;

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
// scores are scaled by 1 / sqrt(128). Each simdgroup takes 16 query rows on
// its own, over blocks of 64 keys: the scores and an online softmax are FP32,
// the probabilities are rounded to BF16 for their product with the values,
// and the outputs add up in FP32. Scores and probabilities pass through
// threadgroup memory, a row of 64 per query row; two lanes take each row, 32
// keys each. A simdgroup reads the keys of at most two ranges, which hold
// every key that its rows see: all rows when one of its rows is in the state;
// else the state's rows, and the rows from its first row's question to its
// last row's (spans[g]: question g's rows, spans[0] the state's). qh, kh and
// vh hold padded rows per head, with zeros in at least 64 rows after the
// sequence's. gate: the block's rows, one value per query head. o: the
// block's rows, one row of 128 per query head.
kernel void lfm3_attend(device const bfloat *qh [[buffer(0)]], device const bfloat *kh [[buffer(1)]],
                        device const bfloat *vh [[buffer(2)]], device const float *gate [[buffer(3)]],
                        device const uint *group [[buffer(4)]], device const uint2 *spans [[buffer(5)]],
                        device float *o [[buffer(6)]], constant Lfm3Attend &p [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint rows = LFM3_TILE_ROWS, keys = LFM3_KEY_BLOCK, half_keys = LFM3_KEY_BLOCK / 2;
  threadgroup float scores[LFM3_TILE_SIMDGROUPS][rows * keys], rescales[LFM3_TILE_SIMDGROUPS][rows],
      sums[LFM3_TILE_SIMDGROUPS][rows];
  threadgroup bfloat probabilities[LFM3_TILE_SIMDGROUPS][rows * keys];
  threadgroup float *ss = scores[sg], *rescale = rescales[sg], *sum = sums[sg];
  threadgroup bfloat *ps = probabilities[sg];
  uint h = tg.y, kv = h / LFM3_SHARING, local = (tg.x * LFM3_TILE_SIMDGROUPS + sg) * rows;
  if (local >= p.block_rows) return;
  uint first = p.first + local, last = min(local + rows, p.block_rows) - 1 + p.first;
  const float scale = rsqrt(float(LFM3_HEAD_DIM));
  constexpr auto sdesc = matmul2d_descriptor(rows, keys, LFM3_HEAD_DIM, false, true, false,
                                             matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(rows, LFM3_HEAD_DIM, keys, false, false, false,
                                             matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)qh + h * p.padded * LFM3_HEAD_DIM,
                                                                dextents<int32_t, 2>(LFM3_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk((device bfloat *)kh + kv * p.padded * LFM3_HEAD_DIM,
                                                                dextents<int32_t, 2>(LFM3_HEAD_DIM, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv((device bfloat *)vh + kv * p.padded * LFM3_HEAD_DIM,
                                                                dextents<int32_t, 2>(LFM3_HEAD_DIM, p.padded));
  tensor<threadgroup bfloat, extents<int32_t, keys, rows>, tensor_inline> tp(ps, extents<int32_t, keys, rows>());
  auto qt = tq.slice(0, first), kt = tk.slice(0, 0), vt = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0;
  // Lane pairs: query row lane / 2 of the 16, keys from (lane % 2) * 32 of 64.
  uint row = lane / 2, column = (lane % 2) * half_keys;
  float running_max = -INFINITY, running_sum = 0;
  uint2 ranges[2];
  uint count = 1;
  if (p.window)
    ranges[0] = uint2(first + 1 > p.window ? first + 1 - p.window : 0, last + 1);
  else if (group[first] == 0)
    ranges[0] = uint2(0, p.rows);
  else
    ranges[0] = uint2(0, p.state), ranges[1] = uint2(spans[group[first]].x, spans[group[last]].y), count = 2;
  for (uint range = 0; range < count; range++)
    for (uint k0 = ranges[range].x; k0 < ranges[range].y; k0 += keys) {
      kt = tk.slice(0, k0), vt = tv.slice(0, k0);
      auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
      sop.run(qt, kt, sc);
      for (uint16_t i = 0; i < sc.get_capacity(); i++)
        if (sc.is_valid_element(i)) {
          auto at = sc.get_multidimensional_index(i);
          uint key = k0 + at[0], query = first + at[1];
          bool seen = key < ranges[range].y && query <= last && lfm3_sees(query, key, p, group);
          ss[at[1] * keys + at[0]] = seen ? sc[i] * scale : -INFINITY;
        }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup float *own = ss + row * keys + column;
      float block_max = -INFINITY;
      for (uint c = 0; c < half_keys; c++) block_max = max(block_max, own[c]);
      block_max = max(block_max, simd_shuffle_xor(block_max, 1));
      float new_max = max(running_max, block_max);
      float old_scale = running_max == -INFINITY ? 0.0f : exp(running_max - new_max), block_sum = 0;
      for (uint c = 0; c < half_keys; c++) {
        float e = own[c] == -INFINITY ? 0.0f : exp(own[c] - new_max);
        ps[row * keys + column + c] = bfloat(e);
        block_sum += e;
      }
      block_sum += simd_shuffle_xor(block_sum, 1);
      running_sum = running_sum * old_scale + block_sum, running_max = new_max;
      if (lane % 2 == 0) rescale[row] = old_scale;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint16_t i = 0; i < acc.get_capacity(); i++)
        if (acc.is_valid_element(i)) acc[i] *= rescale[acc.get_multidimensional_index(i)[1]];
      oop.run(tp, vt, acc);
      simdgroup_barrier(mem_flags::mem_threadgroup);  // the next block rewrites the rows' values
    }
  if (lane % 2 == 0) sum[row] = running_sum;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint out_row = local + at[1];
      if (out_row < p.block_rows) {
        float g = gate[out_row * LFM3_QUERY_HEADS + h];
        o[out_row * LFM3_QUERY_WIDTH + h * LFM3_HEAD_DIM + at[0]] = acc[i] / sum[at[1]] / (1.0f + exp(-g));
      }
    }
}
