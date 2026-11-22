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
// 1 / sqrt(64). Dispatched as a 1D head-coherent grid (ceil(rows / 64) * 16)
// so concurrent threadgroups share each KV head in cache. All 4 simdgroups
// cooperate on one 64x64 tile using exact 16 KiB shared threadgroup memory.
// qh, kh and vh hold padded rows per head, with zeros in at least 64 rows
// after the image's. o: the image's patches, one row of 64 per query head.
kernel void lfm3_vision_attend(device const bfloat *qh [[buffer(0)]], device const bfloat *kh [[buffer(1)]],
                               device const bfloat *vh [[buffer(2)]], device float *o [[buffer(3)]],
                               constant Lfm3VisionAttend &p [[buffer(4)]],
                               uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BQ = FLASH_ATTN_64_TILE, BK = FLASH_ATTN_64_TILE, dim = LFM3_VISION_HEAD_DIM;
  threadgroup FlashAttn64Smem<bfloat> smem;

  uint q_tiles = (p.rows + BQ - 1u) / BQ;
  uint h = tg / q_tiles, kv = h / LFM3_VISION_SHARING;
  uint q0 = (tg % q_tiles) * BQ;
  if (q0 >= p.rows) return;
  bool full_q = q0 + BQ <= p.rows;
  const float scale_log2 = rsqrt(float(dim)) * M_LOG2E_F;

  constexpr auto sdesc = matmul2d_descriptor(BQ, BK, dim, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(BQ, dim, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> sop;
  matmul2d<odesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qh + (ulong)h * p.padded * dim, dextents<int32_t, 2>(dim, p.rows));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)kh + (ulong)kv * p.padded * dim, dextents<int32_t, 2>(dim, p.rows));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)vh + (ulong)kv * p.padded * dim, dextents<int32_t, 2>(dim, p.rows));
  tensor<threadgroup bfloat, extents<int32_t, BK, BQ>, tensor_inline> tp(
      smem.st.probs, extents<int32_t, BK, BQ>());

  auto qt = tq.slice(0, q0), vt0 = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  float rmax = -INFINITY, rsum = 0.0f;
  for (uint k0 = 0; k0 < p.rows; k0 += BK) {
    auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    sop.run(qt, kt, sc);
    bool full_tile = full_q && (k0 + BK <= p.rows);
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
          bool seen = (q0 + at[1] < p.rows) && (k0 + at[0] < p.rows);
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

  uint row = sg * FLASH_ATTN_64_ROWS_PER_SG + (lane >> 1u);
  if ((lane & 1u) == 0u) smem.st.row_scale[row] = 1.0f / rsum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint out_row = q0 + at[1];
      if (out_row < p.rows)
        o[(ulong)out_row * LFM3_VISION_QUERY_WIDTH + h * dim + at[0]] =
            acc[i] * smem.st.row_scale[at[1]];
    }
}

// A residual: x += y.
kernel void lfm3_vision_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
  x[i] += y[i];
}
