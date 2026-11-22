// ---- MiniMax-H3-Image 2D ViT VAE decoder kernels ----
//
// Weights are stored in FP16 (half). Linear projections run native half x half
// -> float matmul2d with FP32 bias addition. Norms, SwiGLU, residual scaling,
// non-affine QK-Norm and 2D partial split-half RoPE run in FP32. Bidirectional
// attention uses half Q/K/V/P with FP32 scores, softmax and accumulation.

// 2D ViT decoder geometry: 16 heads of 64 values (width 1024), 24 latent
// channels, 4 register tokens + 1 zero suffix token, 32 rotated RoPE dims.
constant constexpr uint MINIMAX_H3_VAE_LATENT_CH = 24,
                        MINIMAX_H3_VAE_DIM = 1024,
                        MINIMAX_H3_VAE_HEADS = 16,
                        MINIMAX_H3_VAE_HEAD_DIM = 64,
                        MINIMAX_H3_VAE_HALF_HEAD = MINIMAX_H3_VAE_HEAD_DIM / 2,
                        MINIMAX_H3_VAE_ROT_PAIRS = 16,
                        MINIMAX_H3_VAE_FREQ_PER_AXIS = 8,
                        MINIMAX_H3_VAE_REGISTERS = 4,
                        MINIMAX_H3_VAE_TILE_M = 64;

// 2 * pi factor applied to normalized [-1, 1] patch coordinates in VAE RoPE.
constant constexpr float MINIMAX_H3_VAE_TWO_PI = 6.28318530717958647692f;

// Center offset of a discrete spatial cell when mapping grid indices to [-1, 1].
constant constexpr float MINIMAX_H3_VAE_CELL_CENTER = 0.5f;

// Parameters for converting FP32 activations to padded FP16 rows.
struct MinimaxH3F16X {
  uint m, k, lda;
};

// Parameters for FP16 matrix multiplication: y = xh @ W_f16^T + bias.
struct MinimaxH3F16MM {
  uint m, n, k, ldy, has_bias;
};

// Parameters for residual scaled add: x[r, c] += scale[c] * y[r, c].
struct MinimaxH3AddScaled {
  uint rows, dim;
};

// Parameters for chunked SwiGLU: out[r, c] = silu(in[r, c]) * in[r, inner + c].
struct MinimaxH3SwigluChunked {
  uint rows, inner;
};

// Parameters for affine LayerNorm over rows of n floats.
struct MinimaxH3LayerNorm {
  uint n;
  float eps;
};

// Parameters for VAE tile embedding (post_quant_conv + x_embedder + suffixes).
struct MinimaxH3VaeEmbed {
  uint h_lat, w_lat;
};

// Parameters for VAE head-interleaved QKV split, non-affine QK-Norm and 2D RoPE.
struct MinimaxH3VaeHeads {
  uint rows, padded, h_lat, w_lat;
  float eps;
};

// Parameters for VAE bidirectional 16-head FlashAttention.
struct MinimaxH3VaeAttend {
  uint rows, padded;
};

// Converts m rows of k FP32 activations into FP16 rows padded to a multiple of
// 64 rows (zeroing padded rows). k must be a multiple of 4.
kernel void minimax_h3_x_f16(device const float4 *x [[buffer(0)]],
                             device half4 *y [[buffer(1)]],
                             constant MinimaxH3F16X &p [[buffer(2)]],
                             uint2 i [[thread_position_in_grid]]) {
  uint col4 = i.x, row = i.y;
  uint k4 = p.k / MINIMAX_H3_VEC4_WIDTH;
  if (col4 >= k4) return;
  if (row >= p.m) {
    y[row * k4 + col4] = half4(0);
    return;
  }
  y[row * k4 + col4] = half4(x[row * (p.lda / MINIMAX_H3_VEC4_WIDTH) + col4]);
}

// Native FP16 matrix product accumulating in FP32, with optional FP32 bias.
// xh must have rows padded to a multiple of 64; w must have rows padded to a
// multiple of 64; k must be a multiple of 64. Each threadgroup of 128 threads
// computes a 64x64 tile.
kernel void minimax_h3_mm_f16(device const half *w [[buffer(0)]],
                              device const half *xh [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *y [[buffer(3)]],
                              constant MinimaxH3F16MM &p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc =
      matmul2d_descriptor(MINIMAX_H3_VAE_TILE_M, MINIMAX_H3_TILE_N,
                          static_cast<int>(dynamic_extent), false, true, false,
                          matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<MINIMAX_H3_MM_SGROUPS>> op;
  uint padded_m = (p.m + MINIMAX_H3_VAE_TILE_M - 1u) / MINIMAX_H3_VAE_TILE_M *
                  MINIMAX_H3_VAE_TILE_M;
  uint padded_n = (p.n + MINIMAX_H3_TILE_N - 1u) / MINIMAX_H3_TILE_N *
                  MINIMAX_H3_TILE_N;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tx(
      (device half *)xh, dextents<int32_t, 2>(p.k, padded_m));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tw(
      (device half *)w, dextents<int32_t, 2>(p.k, padded_n));
  uint m0 = tg.y * MINIMAX_H3_VAE_TILE_M, n0 = tg.x * MINIMAX_H3_TILE_N;
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

// Adds a channel-scaled residual in place: x[r, c] += scale[c] * y[r, c].
kernel void minimax_h3_add_scaled(device float *x [[buffer(0)]],
                                  device const float *y [[buffer(1)]],
                                  device const float *scale [[buffer(2)]],
                                  constant MinimaxH3AddScaled &p [[buffer(3)]],
                                  uint2 i [[thread_position_in_grid]]) {
  if (i.x < p.dim && i.y < p.rows) {
    uint idx = i.y * p.dim + i.x;
    x[idx] += scale[i.x] * y[idx];
  }
}

// Splits each row of 2 * inner floats into (gate, up) and writes silu(gate)*up.
kernel void minimax_h3_swiglu_chunked(device const float *in [[buffer(0)]],
                                      device float *out [[buffer(1)]],
                                      constant MinimaxH3SwigluChunked &p [[buffer(2)]],
                                      uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.inner || i.y >= p.rows) return;
  uint row_in = i.y * (2u * p.inner);
  float g = in[row_in + i.x];
  float u = in[row_in + p.inner + i.x];
  out[i.y * p.inner + i.x] = g / (1.0f + exp(-g)) * u;
}

// Affine LayerNorm over each row of n floats: y = w * (x - mean) / std + b.
kernel void minimax_h3_layernorm(device const float *x [[buffer(0)]],
                                 device const float *w [[buffer(1)]],
                                 device const float *b [[buffer(2)]],
                                 device float *y [[buffer(3)]],
                                 constant MinimaxH3LayerNorm &p [[buffer(4)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[MINIMAX_H3_TG_MAX_SGROUPS];
  x += row * p.n;
  y += row * p.n;
  float sum = 0.0f;
  for (uint i = t; i < p.n; i += nt) sum += x[i];
  float mean = tg_sum(sum, tmp, t, nt) / float(p.n);
  float var = 0.0f;
  for (uint i = t; i < p.n; i += nt) {
    float d = x[i] - mean;
    var += d * d;
  }
  float inv = rsqrt(tg_sum(var, tmp, t, nt) / float(p.n) + p.eps);
  for (uint i = t; i < p.n; i += nt) y[i] = w[i] * ((x[i] - mean) * inv) + b[i];
}

// Runs post_quant_conv (24->24) and x_embedder (24->1024) on a channel-first
// latent tile [24, h_lat, w_lat], and appends 4 register tokens + 1 zero token.
// Dispatched with one threadgroup of 256 threads per output row (h_lat*w_lat+5).
kernel void minimax_h3_vae_embed(device const float *z [[buffer(0)]],
                                 device const float *pqc_w [[buffer(1)]],
                                 device const float *pqc_b [[buffer(2)]],
                                 device const float *emb_w [[buffer(3)]],
                                 device const float *emb_b [[buffer(4)]],
                                 device const float *reg [[buffer(5)]],
                                 device float *out [[buffer(6)]],
                                 constant MinimaxH3VaeEmbed &p [[buffer(7)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint nt [[threads_per_threadgroup]]) {
  uint patches = p.h_lat * p.w_lat;
  device float *dst = out + row * MINIMAX_H3_VAE_DIM;
  if (row >= patches) {
    uint suf = row - patches;
    if (suf < MINIMAX_H3_VAE_REGISTERS) {
      device const float *rp = reg + suf * MINIMAX_H3_VAE_DIM;
      for (uint c = t; c < MINIMAX_H3_VAE_DIM; c += nt) dst[c] = rp[c];
    } else {
      for (uint c = t; c < MINIMAX_H3_VAE_DIM; c += nt) dst[c] = 0.0f;
    }
    return;
  }

  threadgroup float zp[MINIMAX_H3_VAE_LATENT_CH];
  if (t < MINIMAX_H3_VAE_LATENT_CH) {
    float acc = pqc_b[t];
    device const float *wr = pqc_w + t * MINIMAX_H3_VAE_LATENT_CH;
    for (uint k = 0; k < MINIMAX_H3_VAE_LATENT_CH; k++)
      acc += wr[k] * z[k * patches + row];
    zp[t] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint c = t; c < MINIMAX_H3_VAE_DIM; c += nt) {
    float acc = emb_b[c];
    device const float *wr = emb_w + c * MINIMAX_H3_VAE_LATENT_CH;
    for (uint k = 0; k < MINIMAX_H3_VAE_LATENT_CH; k++) acc += wr[k] * zp[k];
    dst[c] = acc;
  }
}

// Splits head-interleaved qkv [rows, 16, 3, 64] into head-major half qh, kh, vh
// [16, padded, 64]. Applies non-affine per-head RMSNorm and 2D partial
// split-half RoPE (rot_dim = 32: pairs (i, i + 16) for i = 0..15; dims 32..63
// unrotated; suffix rows past h_lat * w_lat unrotated). Zeroes rows >= rows.
// Threadgroup: 32 threads (1 simdgroup); grid.x: 48 (16 Q + 16 K + 16 V);
// grid.y: row in 0..padded-1.
kernel void minimax_h3_vae_heads(device const float *qkv [[buffer(0)]],
                                 device const float *inv_freq [[buffer(1)]],
                                 device half *qh [[buffer(2)]],
                                 device half *kh [[buffer(3)]],
                                 device half *vh [[buffer(4)]],
                                 constant MinimaxH3VaeHeads &p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  uint kind = tg.x / MINIMAX_H3_VAE_HEADS;
  uint h = tg.x % MINIMAX_H3_VAE_HEADS;
  uint row = tg.y;
  device half *dst = (kind == 0u ? qh : kind == 1u ? kh : vh) +
                     (h * p.padded + row) * MINIMAX_H3_VAE_HEAD_DIM;
  if (row >= p.rows) {
    dst[lane] = half(0.0f);
    dst[lane + MINIMAX_H3_VAE_HALF_HEAD] = half(0.0f);
    return;
  }

  device const float *src =
      qkv + row * (MINIMAX_H3_QKV_PARTS * MINIMAX_H3_VAE_DIM) +
      (h * MINIMAX_H3_QKV_PARTS + kind) * MINIMAX_H3_VAE_HEAD_DIM;
  float a = src[lane];
  float b = src[lane + MINIMAX_H3_VAE_HALF_HEAD];
  if (kind == 2u) {
    dst[lane] = half(a);
    dst[lane + MINIMAX_H3_VAE_HALF_HEAD] = half(b);
    return;
  }

  // Non-affine per-head RMSNorm across all 64 dimensions.
  float ss = simd_sum(a * a + b * b);
  float r = rsqrt(ss / float(MINIMAX_H3_VAE_HEAD_DIM) + p.eps);
  a *= r;
  b *= r;

  // Partial 2D split-half RoPE on first 32 dimensions (lane holds dim 0..31).
  float out_a = a;
  uint patches = p.h_lat * p.w_lat;
  if (row < patches) {
    uint py = row / p.w_lat, px = row % p.w_lat;
    float cy =
        ((float(py) + MINIMAX_H3_VAE_CELL_CENTER) / float(p.h_lat)) * 2.0f -
        1.0f;
    float cx =
        ((float(px) + MINIMAX_H3_VAE_CELL_CENTER) / float(p.w_lat)) * 2.0f -
        1.0f;
    uint pair = lane & (MINIMAX_H3_VAE_ROT_PAIRS - 1u);
    float coord = pair < MINIMAX_H3_VAE_FREQ_PER_AXIS ? cy : cx;
    uint fi = pair < MINIMAX_H3_VAE_FREQ_PER_AXIS
                  ? pair
                  : pair - MINIMAX_H3_VAE_FREQ_PER_AXIS;
    float angle = MINIMAX_H3_VAE_TWO_PI * coord * inv_freq[fi];
    float c = cos(angle), s = sin(angle);
    float partner = simd_shuffle_xor(a, MINIMAX_H3_VAE_ROT_PAIRS);
    bool lo = lane < MINIMAX_H3_VAE_ROT_PAIRS;
    float u0 = lo ? a : partner;
    float u1 = lo ? partner : a;
    out_a = lo ? (u0 * c - u1 * s) : (u1 * c + u0 * s);
  }
  dst[lane] = half(out_a);
  dst[lane + MINIMAX_H3_VAE_HALF_HEAD] = half(b);
}

// Bidirectional 16-head FlashAttention (head_dim = 64) for the VAE decoder.
// Dispatched as a 1D head-coherent grid (ceil(rows / 64) * 16) using 4-simdgroup
// cooperative 64x64 tiles and exact 16 KiB shared threadgroup memory.
kernel void minimax_h3_vae_attend(device const half *qh [[buffer(0)]],
                                  device const half *kh [[buffer(1)]],
                                  device const half *vh [[buffer(2)]],
                                  device float *o [[buffer(3)]],
                                  constant MinimaxH3VaeAttend &p [[buffer(4)]],
                                  uint tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BQ = FLASH_ATTN_64_TILE, BK = FLASH_ATTN_64_TILE;
  threadgroup FlashAttn64Smem<half> smem;

  uint nq = (p.rows + BQ - 1u) / BQ;
  uint h = tg / nq;
  uint q0 = (tg % nq) * BQ;
  if (q0 >= p.rows) return;
  bool full_q = q0 + BQ <= p.rows;
  const float scale_log2 = rsqrt(float(MINIMAX_H3_VAE_HEAD_DIM)) * M_LOG2E_F;

  constexpr auto sdesc =
      matmul2d_descriptor(BQ, BK, MINIMAX_H3_VAE_HEAD_DIM, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(BQ, MINIMAX_H3_VAE_HEAD_DIM, BK, false, false, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> sop;
  matmul2d<odesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> oop;

  ulong head_off = (ulong)h * p.padded * MINIMAX_H3_VAE_HEAD_DIM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(
      (device half *)qh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VAE_HEAD_DIM, p.padded));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(
      (device half *)kh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VAE_HEAD_DIM, p.padded));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(
      (device half *)vh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VAE_HEAD_DIM, p.padded));
  tensor<threadgroup half, extents<int32_t, BK, BQ>, tensor_inline> tp(
      smem.st.probs, extents<int32_t, BK, BQ>());

  auto qt = tq.slice(0, q0), vt0 = tv.slice(0, 0);
  auto acc =
      oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt0),
                                             float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  float rmax = -INFINITY, rsum = 0.0f;
  for (uint k0 = 0; k0 < p.rows; k0 += BK) {
    auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt),
                                                     float>();
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
          smem.scores[at[1] * BK + at[0]] =
              seen ? sc[i] * scale_log2 : -INFINITY;
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
        o[(ulong)out_row * MINIMAX_H3_VAE_DIM +
          h * MINIMAX_H3_VAE_HEAD_DIM + at[0]] =
            acc[i] * smem.st.row_scale[at[1]];
    }
}

// Parameters for fused NHWC GroupNorm + SiLU over a single-frame feature map.
struct MinimaxH3GroupNorm {
  uint hw, channels, groups;
  float eps;
};

// Parameters for reflect-padded NHWC 2D im2col (kernel 1x1 or 3x3) into FP16.
struct MinimaxH3Im2Col {
  uint in_h, in_w, cin, out_w, out_y0, strip_h, ksize, stride, pad_t, pad_l,
      k_padded;
};

// Fused NHWC GroupNorm + SiLU: normalizes each channel group across all spatial
// pixels hw with two-pass centered variance, applies affine weight/bias, and
// applies SiLU. One threadgroup of 256 threads per group.
kernel void minimax_h3_groupnorm_silu(device const float *x [[buffer(0)]],
                                      device const float *w [[buffer(1)]],
                                      device const float *b [[buffer(2)]],
                                      device float *y [[buffer(3)]],
                                      constant MinimaxH3GroupNorm &p
                                      [[buffer(4)]],
                                      uint g [[threadgroup_position_in_grid]],
                                      uint t [[thread_index_in_threadgroup]],
                                      uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[MINIMAX_H3_TG_MAX_SGROUPS];
  uint gc = p.channels / p.groups;
  uint c0 = g * gc;
  uint count = p.hw * gc;

  float sum = 0.0f;
  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    sum += x[pix * p.channels + ch];
  }
  float mean = tg_sum(sum, tmp, t, nt) / float(count);

  float sq = 0.0f;
  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    float d = x[pix * p.channels + ch] - mean;
    sq += d * d;
  }
  float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(count) + p.eps);

  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    uint idx = pix * p.channels + ch;
    float v = w[ch] * ((x[idx] - mean) * inv) + b[ch];
    y[idx] = v / (1.0f + exp(-v));
  }
}

// Reflects coordinate c into [0, limit - 1] matching PyTorch reflect padding.
inline uint minimax_h3_reflect(int c, uint limit) {
  if (limit <= 1u) return 0u;
  int lim = int(limit);
  if (c < 0) c = -c;
  if (c >= lim) c = 2 * lim - 2 - c;
  return uint(clamp(c, 0, lim - 1));
}

// Extracts an NHWC spatial strip of strip_h output rows starting at out_y0 into
// a row-major FP16 matrix [padded_m, k_padded] with reflect spatial padding and
// zero-padding for columns >= ksize*ksize*cin and rows >= strip_h*out_w.
kernel void minimax_h3_im2col_f16(device const float *x [[buffer(0)]],
                                  device half *dst [[buffer(1)]],
                                  constant MinimaxH3Im2Col &p [[buffer(2)]],
                                  uint2 i [[thread_position_in_grid]]) {
  uint col = i.x, row = i.y;
  if (col >= p.k_padded) return;
  uint m = p.strip_h * p.out_w;
  uint k_raw = p.ksize * p.ksize * p.cin;
  if (row >= m || col >= k_raw) {
    dst[row * p.k_padded + col] = half(0.0f);
    return;
  }
  uint oy = p.out_y0 + row / p.out_w;
  uint ox = row % p.out_w;
  if (p.ksize == 1u) {
    dst[row * p.k_padded + col] =
        half(x[(oy * p.in_w + ox) * p.cin + col]);
    return;
  }
  uint tap = col / p.cin;
  uint c = col % p.cin;
  uint ky = tap / p.ksize;
  uint kx = tap % p.ksize;
  uint iy = minimax_h3_reflect(int(oy * p.stride + ky) - int(p.pad_t), p.in_h);
  uint ix = minimax_h3_reflect(int(ox * p.stride + kx) - int(p.pad_l), p.in_w);
  dst[row * p.k_padded + col] = half(x[(iy * p.in_w + ix) * p.cin + c]);
}
