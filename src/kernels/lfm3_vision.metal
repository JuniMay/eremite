// ---- LFM3-VL's vision tower, for LiquidAI's d1 decision models ----
//
// An image's patches are rows of FP32 in memory, in rows of squares of 2 x 2
// patches, each square's patches in rows. The products (mm_bf16) read them as
// BF16 and add up in FP32. The norms, the rotations, the convolution and the
// softmax run in FP32. Attention multiplies BF16 queries, keys, probabilities
// and values, and adds up in FP32.

// The vision attention's shape (vision_config): 16 query heads of 64 values
// that share 8 key and value heads, 2 query heads each.
constant constexpr uint LFM3_VISION_HEAD_DIM = 64, LFM3_VISION_PAIRS = LFM3_VISION_HEAD_DIM / 2,
                        LFM3_VISION_QUERY_HEADS = 16, LFM3_VISION_KEY_HEADS = 8,
                        LFM3_VISION_SHARING = LFM3_VISION_QUERY_HEADS / LFM3_VISION_KEY_HEADS,
                        LFM3_VISION_QUERY_WIDTH = LFM3_VISION_QUERY_HEADS * LFM3_VISION_HEAD_DIM,
                        LFM3_VISION_KEY_WIDTH = LFM3_VISION_KEY_HEADS * LFM3_VISION_HEAD_DIM;
// The side of a token's square of patches (spatial_merge_size), the pairs of a
// head that RoPE turns by a patch's row (the rest turn by its column), and the
// side of the convolution's filter (conv_kernel_size).
constant constexpr uint LFM3_VISION_MERGE = 2, LFM3_VISION_ROW_PAIRS = LFM3_VISION_PAIRS / 2,
                        LFM3_VISION_FILTER = 3;
// lfm3_vision_attend's tiles: a simdgroup takes 16 query rows, over blocks of
// 64 keys; a threadgroup's 4 simdgroups take 64 query rows.
constant constexpr uint LFM3_VISION_TILE_ROWS = 16, LFM3_VISION_KEY_BLOCK = 64,
                        LFM3_VISION_TILE_SIMDGROUPS = 4;

// What lfm3_vision_conv needs: the channels, and the grid's columns and rows
// of patches.
struct Lfm3VisionConv {
  uint dim, grid_w, grid_h;
};

// What lfm3_vision_heads needs: the image's patches, the rows that the BF16
// arrays hold per head, the grid's columns, and the norms' epsilon.
struct Lfm3VisionHeads {
  uint rows, padded, grid_w;
  float eps;
};

// What lfm3_vision_attend needs: the image's patches, and the rows that the
// BF16 arrays hold per head.
struct Lfm3VisionAttend {
  uint rows, padded;
};

// Returns the column and the row in the grid of patch p, in a grid of grid_w
// columns of patches.
inline uint2 lfm3_vision_place(uint p, uint grid_w) {
  constexpr uint m = LFM3_VISION_MERGE;
  uint squares_w = grid_w / m, square = p / (m * m), at = p % (m * m);
  return uint2(square % squares_w * m + at % m, square / squares_w * m + at / m);
}

// Returns the patch at column x and row y of a grid of grid_w columns.
inline uint lfm3_vision_patch(uint x, uint y, uint grid_w) {
  constexpr uint m = LFM3_VISION_MERGE;
  return (y / m * (grid_w / m) + x / m) * m * m + y % m * m + x % m;
}

// The gated depthwise convolution over the grid of patches. Each patch's row
// of bcx holds in_proj's outputs B, C and x, dim values each. With u = B x,
// channel c of patch p's output is C[p][c] times the sum, over the 3 x 3
// patches around p, of the filter's weight w[c][dy][dx] times u of that patch.
// Patches beyond the grid count as zero. Thread x: the channel; thread y: the
// patch.
kernel void lfm3_vision_conv(device const float *bcx [[buffer(0)]], device const float *w [[buffer(1)]],
                             device float *y [[buffer(2)]], constant Lfm3VisionConv &p [[buffer(3)]],
                             uint2 i [[thread_position_in_grid]]) {
  constexpr int reach = LFM3_VISION_FILTER / 2;
  uint c = i.x;
  int2 at = int2(lfm3_vision_place(i.y, p.grid_w));
  device const float *filter = w + c * LFM3_VISION_FILTER * LFM3_VISION_FILTER;
  float sum = 0;
  for (int dy = -reach; dy <= reach; dy++)
    for (int dx = -reach; dx <= reach; dx++) {
      int x = at.x + dx, row = at.y + dy;
      if (x < 0 || row < 0 || x >= int(p.grid_w) || row >= int(p.grid_h)) continue;
      device const float *near = bcx + lfm3_vision_patch(uint(x), uint(row), p.grid_w) * 3 * p.dim;
      sum += filter[(dy + reach) * LFM3_VISION_FILTER + dx + reach] * (near[c] * near[2 * p.dim + c]);
    }
  y[i.y * p.dim + c] = bcx[i.y * 3 * p.dim + p.dim + c] * sum;
}

// Queries, keys and values for the vision attention, from the patches' rows of
// q, k and v. Each query and key head is normalized with its weights
// (RMSNorm) and turned (RoPE: value i < 32 pairs with value i + 32; pair i < 16
// turns by the patch's row times inv_freq[i], pair i >= 16 by its column times
// inv_freq[i - 16]). All three are stored as BF16, a row of 64 values per
// head: head h of patch r at row h * padded + r of qh, kh and vh. A
// threadgroup of 32 threads, one value pair each, takes one head of one patch:
// grid x is the query heads, then the key heads, then the value heads; grid y
// is the patch.
kernel void lfm3_vision_heads(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                              device const float *v [[buffer(2)]], device const float *q_norm [[buffer(3)]],
                              device const float *k_norm [[buffer(4)]], device const float *inv_freq [[buffer(5)]],
                              device bfloat *qh [[buffer(6)]], device bfloat *kh [[buffer(7)]],
                              device bfloat *vh [[buffer(8)]], constant Lfm3VisionHeads &p [[buffer(9)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  constexpr uint dim = LFM3_VISION_HEAD_DIM, pairs = LFM3_VISION_PAIRS;
  uint head = tg.x, row = tg.y;
  if (head >= LFM3_VISION_QUERY_HEADS + LFM3_VISION_KEY_HEADS) {
    uint h = head - LFM3_VISION_QUERY_HEADS - LFM3_VISION_KEY_HEADS;
    device const float *src = v + row * LFM3_VISION_KEY_WIDTH + h * dim;
    device bfloat *dst = vh + (h * p.padded + row) * dim;
    dst[t] = bfloat(src[t]), dst[t + pairs] = bfloat(src[t + pairs]);
    return;
  }
  bool query = head < LFM3_VISION_QUERY_HEADS;
  uint h = query ? head : head - LFM3_VISION_QUERY_HEADS;
  device const float *src = query ? q + row * LFM3_VISION_QUERY_WIDTH + h * dim
                                  : k + row * LFM3_VISION_KEY_WIDTH + h * dim;
  device const float *w = query ? q_norm : k_norm;
  device bfloat *dst = (query ? qh : kh) + (h * p.padded + row) * dim;
  float a = src[t], b = src[t + pairs];
  float r = rsqrt(tg_sum(a * a + b * b, tmp, t, pairs) / dim + p.eps);
  a = w[t] * (a * r), b = w[t + pairs] * (b * r);
  uint2 at = lfm3_vision_place(row, p.grid_w);
  float angle = t < LFM3_VISION_ROW_PAIRS ? float(at.y) * inv_freq[t]
                                          : float(at.x) * inv_freq[t - LFM3_VISION_ROW_PAIRS];
  float c = cos(angle), s = sin(angle), turned = a * c - b * s;
  b = b * c + a * s, a = turned;
  dst[t] = bfloat(a), dst[t + pairs] = bfloat(b);
}

// Attention of every patch over every patch of its image, both ways; query
// head h reads key head h / LFM3_VISION_SHARING, and the scores are scaled by
// 1 / sqrt(64). Each simdgroup takes 16 query rows on its own, over blocks of
// 64 keys: the scores and an online softmax are FP32, the probabilities are
// rounded to BF16 for their product with the values, and the outputs add up in
// FP32. Scores and probabilities pass through threadgroup memory, a row of 64
// per query row; two lanes take each row, 32 keys each. qh, kh and vh hold
// padded rows per head, with zeros in at least 64 rows after the image's. o:
// the image's patches, one row of 64 per query head.
kernel void lfm3_vision_attend(device const bfloat *qh [[buffer(0)]], device const bfloat *kh [[buffer(1)]],
                               device const bfloat *vh [[buffer(2)]], device float *o [[buffer(3)]],
                               constant Lfm3VisionAttend &p [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint rows = LFM3_VISION_TILE_ROWS, keys = LFM3_VISION_KEY_BLOCK, half_keys = keys / 2,
                 dim = LFM3_VISION_HEAD_DIM;
  threadgroup float scores[LFM3_VISION_TILE_SIMDGROUPS][rows * keys], rescales[LFM3_VISION_TILE_SIMDGROUPS][rows],
      sums[LFM3_VISION_TILE_SIMDGROUPS][rows];
  threadgroup bfloat probabilities[LFM3_VISION_TILE_SIMDGROUPS][rows * keys];
  threadgroup float *ss = scores[sg], *rescale = rescales[sg], *sum = sums[sg];
  threadgroup bfloat *ps = probabilities[sg];
  uint h = tg.y, kv = h / LFM3_VISION_SHARING, first = (tg.x * LFM3_VISION_TILE_SIMDGROUPS + sg) * rows;
  if (first >= p.rows) return;
  const float scale = rsqrt(float(dim));
  constexpr auto sdesc = matmul2d_descriptor(rows, keys, dim, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(rows, dim, keys, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)qh + h * p.padded * dim,
                                                                dextents<int32_t, 2>(dim, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk((device bfloat *)kh + kv * p.padded * dim,
                                                                dextents<int32_t, 2>(dim, p.padded));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv((device bfloat *)vh + kv * p.padded * dim,
                                                                dextents<int32_t, 2>(dim, p.padded));
  tensor<threadgroup bfloat, extents<int32_t, keys, rows>, tensor_inline> tp(ps, extents<int32_t, keys, rows>());
  auto qt = tq.slice(0, first), kt = tk.slice(0, 0), vt = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0;
  // Lane pairs: query row lane / 2 of the 16, keys from (lane % 2) * 32 of 64.
  uint row = lane / 2, column = (lane % 2) * half_keys;
  float running_max = -INFINITY, running_sum = 0;
  for (uint k0 = 0; k0 < p.rows; k0 += keys) {
    kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    sop.run(qt, kt, sc);
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto at = sc.get_multidimensional_index(i);
        ss[at[1] * keys + at[0]] = k0 + at[0] < p.rows ? sc[i] * scale : -INFINITY;
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
      uint out_row = first + at[1];
      if (out_row < p.rows) o[out_row * LFM3_VISION_QUERY_WIDTH + h * dim + at[0]] = acc[i] / sum[at[1]];
    }
}

// A residual: x += y.
kernel void lfm3_vision_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
  x[i] += y[i];
}
