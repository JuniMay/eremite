// ---- MiniMax-H3 quantized linear and embedding kernels ----
//
// NVFP4 weights (Qwen3-VL text encoder): packed hi-nibble-first FP4 E2M1 codes
// with unswizzled row-major FP8 E4M3 block-16 scales, a scalar weight_scale_2,
// and an optional BF16 per-channel pre_quant_scale applied to activations.
//
// INT8-ConvRot weights (MiniMax-H3 DiT): block-256 regular Hadamard rotation
// (H_4^{\otimes 4} / 16) followed by per-row INT8 quantization and native
// int8 x int8 -> int32 matrix multiplication.

// Parameters for converting FP32 activations to padded BF16 rows, with an
// optional BF16 per-channel pre_quant_scale.
struct MinimaxH3Bf16X {
  uint m, k, lda, has_scale;
};

// Parameters for NVFP4 matrix multiplication: y[m, n] = (xb @ W^T) * scale2.
struct MinimaxH3Nvfp4MM {
  uint m, n, k, ldy;
  float scale2;
};

// Parameters for fused ConvRot block-256 Hadamard rotation and per-row INT8
// activation quantization.
struct MinimaxH3ConvRot {
  uint m, k, lda;
};

// Parameters for INT8 matrix multiplication with per-row and per-column FP32
// scales and optional FP32 bias.
struct MinimaxH3Int8MM {
  uint m, n, k, ldy, has_bias;
};

// Minimum per-row activation scale matching comfy-kitchen's quantize_int8_rowwise.
constant constexpr float MINIMAX_H3_INT8_SCALE_MIN = 1e-30f;

// Signed 8-bit integer clamp bounds for per-row INT8 activation quantization.
constant constexpr float MINIMAX_H3_INT8_MIN = -128.0f,
                         MINIMAX_H3_INT8_MAX = 127.0f;

// Normalization factor 1 / sqrt(4) applied at each 4-point regular Hadamard step.
constant constexpr float MINIMAX_H3_H4_NORM = 0.5f;

// Apple GPU simdgroup width, lane mask, shift, and max simdgroups per 1024-thread group.
constant constexpr uint MINIMAX_H3_SIMD_WIDTH = 32,
                        MINIMAX_H3_SIMD_MASK = MINIMAX_H3_SIMD_WIDTH - 1,
                        MINIMAX_H3_SIMD_SHIFT = 5,
                        MINIMAX_H3_TG_MAX_SGROUPS = 32;

// Simdgroups participating in one cooperative matrix multiply tile.
constant constexpr uint MINIMAX_H3_MM_SGROUPS = 4;

// Matrix tile dimensions: 128-row NVFP4 tile made of two 64-row subtiles, 64 columns wide.
constant constexpr uint MINIMAX_H3_TILE_M = 128,
                        MINIMAX_H3_SUBTILE_M = 64,
                        MINIMAX_H3_TILE_N = 64,
                        MINIMAX_H3_TILE_K = 64;

// NVFP4 weight packing: two 4-bit codes per byte, 16 weights per FP8 E4M3 scale,
// and 32 weights (two blocks) loaded per thread in a 128-thread tile loader.
constant constexpr uint MINIMAX_H3_NVFP4_PER_BYTE = 2,
                        MINIMAX_H3_NVFP4_BLOCK = 16,
                        MINIMAX_H3_NVFP4_CHUNK = 32;

// ConvRot regular Hadamard group of 256 values: 8 values per lane across 32 lanes.
constant constexpr uint MINIMAX_H3_CONVROT_GROUP = 256,
                        MINIMAX_H3_CONVROT_GROUP_SHIFT = 8,
                        MINIMAX_H3_CONVROT_PER_LANE =
                            MINIMAX_H3_CONVROT_GROUP / MINIMAX_H3_SIMD_WIDTH;

// Number of spatial-temporal position axes (time, height, width) in 3D RoPE and MRoPE.
constant constexpr uint MINIMAX_H3_POS_AXES = 3;

// Number of projections (Q, K, V) packed into one fused QKV tensor.
constant constexpr uint MINIMAX_H3_QKV_PARTS = 3;

// Elements processed per vectorized float4/bfloat4/half4 load or store.
constant constexpr uint MINIMAX_H3_VEC4_WIDTH = 4;

// Converts m rows of k FP32 activations into BF16 rows (zeroing padded rows
// past m), optionally multiplying each channel by a BF16 pre_quant_scale.
// Thread x: group of 4 columns; thread y: output row up to padded_m.
kernel void minimax_h3_x_bf16(device const float4 *x [[buffer(0)]],
                              device const ushort *scale [[buffer(1)]],
                              device bfloat4 *y [[buffer(2)]],
                              constant MinimaxH3Bf16X &p [[buffer(3)]],
                              uint2 i [[thread_position_in_grid]]) {
  uint col4 = i.x, row = i.y;
  uint k4 = p.k / MINIMAX_H3_VEC4_WIDTH;
  if (col4 >= k4) return;
  if (row >= p.m) {
    y[row * k4 + col4] = bfloat4(0);
    return;
  }
  float4 v = x[row * (p.lda / MINIMAX_H3_VEC4_WIDTH) + col4];
  if (p.has_scale) {
    uint2 q = *(device const uint2 *)(scale + col4 * MINIMAX_H3_VEC4_WIDTH);
    v *= bf16x4(q);
  }
  y[row * k4 + col4] = bfloat4(v);
}

// Unpacks 8 hi-nibble-first FP4 E2M1 codes from a 32-bit word into two float4s.
// Byte b holds even column in its high nibble and odd column in its low nibble.
inline void minimax_h3_fp4x8_hi(uint u, thread float4 &a, thread float4 &b) {
  a = float4(fp4((u >> 4) & 0xFu), fp4(u & 0xFu),
             fp4((u >> 12) & 0xFu), fp4((u >> 8) & 0xFu));
  b = float4(fp4((u >> 20) & 0xFu), fp4((u >> 16) & 0xFu),
             fp4((u >> 28) & 0xFu), fp4((u >> 24) & 0xFu));
}

// Loaded 32-column chunk (two 16-weight blocks) of NVFP4 codes and E4M3 scales
// for one thread of a 128-thread tile loader.
struct MinimaxH3Nvfp4Chunk {
  uint4 q;
  float s0, s1;
};

// Loads 32 NVFP4 weights and their 2 E4M3 scales for thread t in a 64x64 tile.
inline MinimaxH3Nvfp4Chunk minimax_h3_nvfp4_load(device const uchar *w,
                                                 device const uchar *s,
                                                 uint n0, uint n, uint k,
                                                 uint k0, uint t) {
  MinimaxH3Nvfp4Chunk c;
  uint r = n0 + (t >> 1);
  uint j = k0 + (t & 1u) * MINIMAX_H3_NVFP4_CHUNK;
  if (r < n && j < k) {
    c.q = *(device const uint4 *)(w + r * (k / MINIMAX_H3_NVFP4_PER_BYTE) +
                                  j / MINIMAX_H3_NVFP4_PER_BYTE);
    device const uchar *sp = s + r * (k / MINIMAX_H3_NVFP4_BLOCK) +
                             j / MINIMAX_H3_NVFP4_BLOCK;
    c.s0 = fp8(sp[0]);
    c.s1 = fp8(sp[1]);
  } else {
    c.q = uint4(0);
    c.s0 = 0.0f;
    c.s1 = 0.0f;
  }
  return c;
}

// Dequantizes a thread's 32 NVFP4 weights into threadgroup BF16 tile ws[64*64].
// Every FP4 E2M1 times FP8 E4M3 product fits in BF16 without rounding loss.
inline void minimax_h3_nvfp4_store(threadgroup bfloat *ws,
                                   MinimaxH3Nvfp4Chunk c, uint t) {
  uint row_in_tile = t >> 1;
  uint col_in_tile = (t & 1u) * MINIMAX_H3_NVFP4_CHUNK;
  threadgroup bfloat4 *d =
      (threadgroup bfloat4 *)(ws + row_in_tile * MINIMAX_H3_TILE_K +
                              col_in_tile);
  float4 a, b;
  minimax_h3_fp4x8_hi(c.q.x, a, b);
  d[0] = bfloat4(a * c.s0);
  d[1] = bfloat4(b * c.s0);
  minimax_h3_fp4x8_hi(c.q.y, a, b);
  d[2] = bfloat4(a * c.s0);
  d[3] = bfloat4(b * c.s0);
  minimax_h3_fp4x8_hi(c.q.z, a, b);
  d[4] = bfloat4(a * c.s1);
  d[5] = bfloat4(b * c.s1);
  minimax_h3_fp4x8_hi(c.q.w, a, b);
  d[6] = bfloat4(a * c.s1);
  d[7] = bfloat4(b * c.s1);
}

// Matrix product of BF16 activations (rows padded to 128) with NVFP4 weights
// (hi-nibble-first E2M1 codes, unswizzled row-major E4M3 block-16 scales) and
// scalar weight_scale_2. Each threadgroup of 128 threads computes a 128x64 tile.
kernel void minimax_h3_mm_nvfp4(device const uchar *w [[buffer(0)]],
                                device const uchar *s [[buffer(1)]],
                                device const bfloat *x [[buffer(2)]],
                                device float *y [[buffer(3)]],
                                constant MinimaxH3Nvfp4MM &p [[buffer(4)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat ws[MINIMAX_H3_TILE_N * MINIMAX_H3_TILE_K];
  constexpr auto desc =
      matmul2d_descriptor(MINIMAX_H3_SUBTILE_M, MINIMAX_H3_TILE_N,
                          MINIMAX_H3_TILE_K, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<MINIMAX_H3_MM_SGROUPS>> op;
  uint padded_m = (p.m + MINIMAX_H3_TILE_M - 1) / MINIMAX_H3_TILE_M *
                  MINIMAX_H3_TILE_M;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x, dextents<int32_t, 2>(p.k, padded_m));
  tensor<threadgroup bfloat,
         extents<int32_t, MINIMAX_H3_TILE_K, MINIMAX_H3_TILE_N>, tensor_inline>
      tw(ws, extents<int32_t, MINIMAX_H3_TILE_K, MINIMAX_H3_TILE_N>());
  auto c0 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  for (uint16_t i = 0; i < c0.get_capacity(); i++)
    if (c0.is_valid_element(i)) c0[i] = 0, c1[i] = 0;
  uint m0 = tg.y * MINIMAX_H3_TILE_M, n0 = tg.x * MINIMAX_H3_TILE_N;
  MinimaxH3Nvfp4Chunk wc = minimax_h3_nvfp4_load(w, s, n0, p.n, p.k, 0, t);
  for (uint k0 = 0; k0 < p.k; k0 += MINIMAX_H3_TILE_K) {
    minimax_h3_nvfp4_store(ws, wc, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + MINIMAX_H3_TILE_K < p.k)
      wc = minimax_h3_nvfp4_load(w, s, n0, p.n, p.k, k0 + MINIMAX_H3_TILE_K, t);
    auto a = tx.slice(k0, m0);
    op.run(a, tw, c0);
    if (m0 + MINIMAX_H3_SUBTILE_M < p.m) {
      auto b = tx.slice(k0, m0 + MINIMAX_H3_SUBTILE_M);
      op.run(b, tw, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < c0.get_capacity(); i++)
    if (c0.is_valid_element(i)) {
      auto at = c0.get_multidimensional_index(i);
      uint col = n0 + at[0];
      if (col >= p.n) continue;
      uint row0 = m0 + at[1];
      if (row0 < p.m) y[row0 * p.ldy + col] = c0[i] * p.scale2;
      uint row1 = row0 + MINIMAX_H3_SUBTILE_M;
      if (row1 < p.m) y[row1 * p.ldy + col] = c1[i] * p.scale2;
    }
}

// Applies one normalized 4-point regular Hadamard H_4 / sqrt(4) in place:
// [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] * 0.5.
inline void minimax_h3_h4(thread float &a, thread float &b,
                          thread float &c, thread float &d) {
  float s = MINIMAX_H3_H4_NORM * (a + b + c + d);
  float r0 = s - d, r1 = s - c, r2 = s - b, r3 = s - a;
  a = r0;
  b = r1;
  c = r2;
  d = r3;
}

// Applies the 256-point ConvRot regular Hadamard (H_4^{\otimes 4} / 16) across
// a 32-lane simdgroup where lane holds indices lane + 32 * j (j = 0..7).
inline void minimax_h3_hadamard256(thread float v[MINIMAX_H3_CONVROT_PER_LANE],
                                   uint lane) {
  // Stage 1: stride 64 (in-register across the two groups of four values).
  minimax_h3_h4(v[0], v[2], v[4], v[6]);
  minimax_h3_h4(v[1], v[3], v[5], v[7]);

  // Stage 2: stride 16 (pairs lane with lane ^ 16 across even/odd register pairs).
  constexpr uint half_simd = MINIMAX_H3_SIMD_WIDTH / 2u;
  bool lo = lane < half_simd;
  for (uint q = 0; q < MINIMAX_H3_CONVROT_PER_LANE / 2u; q++) {
    float a = v[2 * q], b = v[2 * q + 1];
    float pa = simd_shuffle_xor(a, half_simd);
    float pb = simd_shuffle_xor(b, half_simd);
    float p0 = lo ? a : pa, p1 = lo ? pa : a;
    float p2 = lo ? b : pb, p3 = lo ? pb : b;
    float s = MINIMAX_H3_H4_NORM * (p0 + p1 + p2 + p3);
    v[2 * q] = s - (lo ? p3 : p2);
    v[2 * q + 1] = s - (lo ? p1 : p0);
  }

  // Stage 3: stride 4 (within lanes sharing bits outside 0b1100).
  uint base4 = lane & ~12u;
  uint sub4 = (lane >> 2) & 3u;
  for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++) {
    float p0 = simd_shuffle(v[j], base4);
    float p1 = simd_shuffle(v[j], base4 + 4u);
    float p2 = simd_shuffle(v[j], base4 + 8u);
    float p3 = simd_shuffle(v[j], base4 + 12u);
    float s = MINIMAX_H3_H4_NORM * (p0 + p1 + p2 + p3);
    v[j] = s - (sub4 == 0u ? p3 : sub4 == 1u ? p2 : sub4 == 2u ? p1 : p0);
  }

  // Stage 4: stride 1 (within contiguous quads of 4 lanes).
  uint base1 = lane & ~3u;
  uint sub1 = lane & 3u;
  for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++) {
    float p0 = simd_shuffle(v[j], base1);
    float p1 = simd_shuffle(v[j], base1 + 1u);
    float p2 = simd_shuffle(v[j], base1 + 2u);
    float p3 = simd_shuffle(v[j], base1 + 3u);
    float s = MINIMAX_H3_H4_NORM * (p0 + p1 + p2 + p3);
    v[j] = s - (sub1 == 0u ? p3 : sub1 == 1u ? p2 : sub1 == 2u ? p1 : p0);
  }
}

// Rotates each activation row in blocks of 256 by H_4^{\otimes 4} / 16 and
// quantizes per row to INT8 (zeroing padded rows past m). Dispatched with one
// threadgroup of 256 threads per output row up to padded_m. k must be a
// multiple of 256.
kernel void minimax_h3_convrot_quant_i8(device const float *x [[buffer(0)]],
                                        device int8_t *xq [[buffer(1)]],
                                        device float *xscale [[buffer(2)]],
                                        constant MinimaxH3ConvRot &p [[buffer(3)]],
                                        uint row [[threadgroup_position_in_grid]],
                                        uint t [[thread_index_in_threadgroup]],
                                        uint nt [[threads_per_threadgroup]]) {
  device int8_t *dst = xq + row * p.k;
  if (row >= p.m) {
    for (uint c = t; c < p.k; c += nt) dst[c] = 0;
    return;
  }
  threadgroup float tmp[MINIMAX_H3_TG_MAX_SGROUPS];
  uint lane = t & MINIMAX_H3_SIMD_MASK;
  uint sg = t >> MINIMAX_H3_SIMD_SHIFT;
  uint nsg = nt >> MINIMAX_H3_SIMD_SHIFT;
  uint ngroups = p.k >> MINIMAX_H3_CONVROT_GROUP_SHIFT;
  device const float *src = x + row * p.lda;

  // Pass 1: rotate blocks in registers to find the row's peak magnitude.
  float amax = 0.0f;
  for (uint g = sg; g < ngroups; g += nsg) {
    device const float *gp = src + g * MINIMAX_H3_CONVROT_GROUP;
    float v[MINIMAX_H3_CONVROT_PER_LANE];
    for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++)
      v[j] = gp[lane + j * MINIMAX_H3_SIMD_WIDTH];
    minimax_h3_hadamard256(v, lane);
    for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++)
      amax = max(amax, fabs(v[j]));
  }
  float row_max = tg_max(amax, tmp, t, nt);
  float scale = max(row_max / MINIMAX_H3_INT8_MAX, MINIMAX_H3_INT8_SCALE_MIN);
  if (t == 0) xscale[row] = scale;
  float inv_scale = 1.0f / scale;

  // Pass 2: re-read the L1-cached row, rotate and write INT8 codes.
  for (uint g = sg; g < ngroups; g += nsg) {
    device const float *gp = src + g * MINIMAX_H3_CONVROT_GROUP;
    device int8_t *dp = dst + g * MINIMAX_H3_CONVROT_GROUP;
    float v[MINIMAX_H3_CONVROT_PER_LANE];
    for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++)
      v[j] = gp[lane + j * MINIMAX_H3_SIMD_WIDTH];
    minimax_h3_hadamard256(v, lane);
    for (uint j = 0; j < MINIMAX_H3_CONVROT_PER_LANE; j++) {
      float q = clamp(rint(v[j] * inv_scale), MINIMAX_H3_INT8_MIN,
                      MINIMAX_H3_INT8_MAX);
      dp[lane + j * MINIMAX_H3_SIMD_WIDTH] = int8_t(int(q));
    }
  }
}

// Native INT8 matrix product with per-row activation scale, per-column weight
// scale, and optional FP32 bias: y[r, c] = acc[r, c] * xs[r] * ws[c] + bias[c].
// Activations xq must have rows padded to a multiple of 64; k and n must be
// multiples of 64. Each threadgroup of 128 threads computes a 64x64 tile.
kernel void minimax_h3_mm_int8(device const int8_t *w [[buffer(0)]],
                               device const float *ws [[buffer(1)]],
                               device const int8_t *xq [[buffer(2)]],
                               device const float *xs [[buffer(3)]],
                               device const float *bias [[buffer(4)]],
                               device float *y [[buffer(5)]],
                               constant MinimaxH3Int8MM &p [[buffer(6)]],
                               uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc =
      matmul2d_descriptor(MINIMAX_H3_SUBTILE_M, MINIMAX_H3_TILE_N,
                          static_cast<int>(dynamic_extent), false, true, false,
                          matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<MINIMAX_H3_MM_SGROUPS>> op;
  uint padded_m = (p.m + MINIMAX_H3_SUBTILE_M - 1u) / MINIMAX_H3_SUBTILE_M *
                  MINIMAX_H3_SUBTILE_M;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tx(
      (device int8_t *)xq, dextents<int32_t, 2>(p.k, padded_m));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tw(
      (device int8_t *)w, dextents<int32_t, 2>(p.k, p.n));
  uint m0 = tg.y * MINIMAX_H3_SUBTILE_M, n0 = tg.x * MINIMAX_H3_TILE_N;
  auto a = tx.slice(0, m0);
  auto b = tw.slice(0, n0);
  auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b),
                                                 int>();
  op.run(a, b, c);
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint col = n0 + at[0], row = m0 + at[1];
      if (col < p.n && row < p.m) {
        float bval = p.has_bias ? bias[col] : 0.0f;
        y[row * p.ldy + col] = float(c[i]) * (xs[row] * ws[col]) + bval;
      }
    }
}

// INT8 row-scaled embedding lookup: row r of x gets token ids[r]'s row of table
// multiplied by scale[ids[r]], skipping rows whose id is negative.
kernel void minimax_h3_embed_int8(device const int8_t *table [[buffer(0)]],
                                  device const float *scale [[buffer(1)]],
                                  device const int *ids [[buffer(2)]],
                                  device float *x [[buffer(3)]],
                                  constant uint &dim [[buffer(4)]],
                                  uint2 i [[thread_position_in_grid]]) {
  int id = ids[i.y];
  if (id >= 0)
    x[i.y * dim + i.x] = float(table[uint(id) * dim + i.x]) * scale[uint(id)];
}
