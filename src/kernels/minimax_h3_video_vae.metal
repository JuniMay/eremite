// ---- MiniMax-H3 Video VAE kernels (3D ViT decoder + 3D causal CNN encoder) ----
//
// Model constants cite Comfy-Org/ComfyUI comfy/ldm/minimax/vae.py:
// ViT3D decoder: 32 heads x 64 dim = 2048 width, 3D partial split-half RoPE
// over 48 of 64 head dims (24 pairs = 8 time + 8 height + 8 width, base 100.0,
// angle scale 2*pi), 4 register tokens + 1 zero suffix token.
// 3D causal CNN encoder: temporal-isolated 32-group GroupNorm + SiLU, reflect
// spatial padding and causal zero front temporal padding.

// 3D ViT decoder geometry (comfy/ldm/minimax/vae.py:421-456).
constant constexpr uint MINIMAX_H3_VVAE_DIM = 2048,
                        MINIMAX_H3_VVAE_HEADS = 32,
                        MINIMAX_H3_VVAE_HEAD_DIM = 64,
                        MINIMAX_H3_VVAE_HALF_HEAD = MINIMAX_H3_VVAE_HEAD_DIM / 2,
                        MINIMAX_H3_VVAE_ROT_PAIRS = 24,
                        MINIMAX_H3_VVAE_FREQ_PER_AXIS = 8,
                        MINIMAX_H3_VVAE_AXIS_H_START = 8,
                        MINIMAX_H3_VVAE_AXIS_W_START = 16,
                        MINIMAX_H3_VVAE_LOW_PAIR_LIMIT = 8,
                        MINIMAX_H3_VVAE_HI_HALF_ROT = 16;

// Parameters for 3D video VAE tile embedding (post_quant_conv + x_embedder + suffixes).
struct MinimaxH3VvaeEmbed {
  uint t_lat, h_lat, w_lat;
};

// Parameters for 32-head QKV split, non-affine QK-Norm and 3D partial RoPE.
struct MinimaxH3VvaeHeads {
  uint rows, padded, t_lat, h_lat, w_lat;
  float eps;
};

// Parameters for 32-head bidirectional FlashAttention.
struct MinimaxH3VvaeAttend {
  uint rows, padded;
};

// Parameters for temporal-isolated NDHWC GroupNorm + SiLU across t frames.
struct MinimaxH3VvaeGroupNorm3d {
  uint hw, channels, groups;
  float eps;
};

// Parameters for causal-temporal, reflect-spatial 3D im2col into FP16.
struct MinimaxH3VvaeIm2Col3d {
  uint in_t, in_h, in_w, cin, out_w, ot, out_y0, strip_h;
  uint kt, ks, time_stride, space_stride, front_pad, pad_h, pad_w, k_padded;
};

// Runs post_quant_conv (24->24) and x_embedder (24->2048) on channel-first
// latent [24, t_lat, h_lat, w_lat] and appends 4 register + 1 zero tokens.
// Dispatched with 1 threadgroup of 256 threads per output row (t*h*w + 5).
kernel void minimax_h3_vvae_embed(device const float *z [[buffer(0)]],
                                  device const float *pqc_w [[buffer(1)]],
                                  device const float *pqc_b [[buffer(2)]],
                                  device const float *emb_w [[buffer(3)]],
                                  device const float *emb_b [[buffer(4)]],
                                  device const float *reg [[buffer(5)]],
                                  device float *out [[buffer(6)]],
                                  constant MinimaxH3VvaeEmbed &p [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint nt [[threads_per_threadgroup]]) {
  uint patches = p.t_lat * p.h_lat * p.w_lat;
  device float *dst = out + row * MINIMAX_H3_VVAE_DIM;
  if (row >= patches) {
    uint suf = row - patches;
    if (suf < MINIMAX_H3_VAE_REGISTERS) {
      device const float *rp = reg + suf * MINIMAX_H3_VVAE_DIM;
      for (uint c = t; c < MINIMAX_H3_VVAE_DIM; c += nt) dst[c] = rp[c];
    } else {
      for (uint c = t; c < MINIMAX_H3_VVAE_DIM; c += nt) dst[c] = 0.0f;
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

  for (uint c = t; c < MINIMAX_H3_VVAE_DIM; c += nt) {
    float acc = emb_b[c];
    device const float *wr = emb_w + c * MINIMAX_H3_VAE_LATENT_CH;
    for (uint k = 0; k < MINIMAX_H3_VAE_LATENT_CH; k++) acc += wr[k] * zp[k];
    dst[c] = acc;
  }
}

// Splits head-interleaved qkv [rows, 32, 3, 64] into head-major half qh, kh, vh
// [32, padded, 64]. Applies non-affine per-head RMSNorm and 3D partial
// split-half RoPE (rot_dim = 48: pairs (i, i + 24) for i = 0..23; dims 48..63
// and suffix rows past t_lat*h_lat*w_lat stay unrotated).
// Threadgroup: 32 threads (1 simdgroup); grid.x: 96 (32 Q + 32 K + 32 V);
// grid.y: row in 0..padded-1.
kernel void minimax_h3_vvae_heads(device const float *qkv [[buffer(0)]],
                                  device const float *inv_freq [[buffer(1)]],
                                  device half *qh [[buffer(2)]],
                                  device half *kh [[buffer(3)]],
                                  device half *vh [[buffer(4)]],
                                  constant MinimaxH3VvaeHeads &p [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  uint kind = tg.x / MINIMAX_H3_VVAE_HEADS;
  uint h = tg.x % MINIMAX_H3_VVAE_HEADS;
  uint row = tg.y;
  device half *dst = (kind == 0u ? qh : kind == 1u ? kh : vh) +
                     (h * p.padded + row) * MINIMAX_H3_VVAE_HEAD_DIM;
  if (row >= p.rows) {
    dst[lane] = half(0.0f);
    dst[lane + MINIMAX_H3_VVAE_HALF_HEAD] = half(0.0f);
    return;
  }

  device const float *src =
      qkv + row * (MINIMAX_H3_QKV_PARTS * MINIMAX_H3_VVAE_DIM) +
      (h * MINIMAX_H3_QKV_PARTS + kind) * MINIMAX_H3_VVAE_HEAD_DIM;
  float a = src[lane];
  float b = src[lane + MINIMAX_H3_VVAE_HALF_HEAD];
  if (kind == 2u) {
    dst[lane] = half(a);
    dst[lane + MINIMAX_H3_VVAE_HALF_HEAD] = half(b);
    return;
  }

  // Non-affine per-head RMSNorm across all 64 dimensions.
  float ss = simd_sum(a * a + b * b);
  float r = rsqrt(ss / float(MINIMAX_H3_VVAE_HEAD_DIM) + p.eps);
  a *= r;
  b *= r;

  float out_a = a;
  float out_b = b;
  uint hw = p.h_lat * p.w_lat;
  uint patches = p.t_lat * hw;
  if (row < patches) {
    uint pt = row / hw;
    uint rem = row % hw;
    uint py = rem / p.w_lat;
    uint px = rem % p.w_lat;
    float ct = ((float(pt) + MINIMAX_H3_VAE_CELL_CENTER) / float(p.t_lat)) * 2.0f - 1.0f;
    float cy = ((float(py) + MINIMAX_H3_VAE_CELL_CENTER) / float(p.h_lat)) * 2.0f - 1.0f;
    float cx = ((float(px) + MINIMAX_H3_VAE_CELL_CENTER) / float(p.w_lat)) * 2.0f - 1.0f;

    // Lanes 0..23 hold u0 = dim[0..23] and gather u1 = dim[24..47] via simd_shuffle.
    float u1_lo = simd_shuffle(a, (lane + MINIMAX_H3_VVAE_ROT_PAIRS) & MINIMAX_H3_SIMD_MASK);
    float u1_hi = simd_shuffle(b, (lane - MINIMAX_H3_VVAE_LOW_PAIR_LIMIT) & MINIMAX_H3_SIMD_MASK);
    float u1 = lane < MINIMAX_H3_VVAE_LOW_PAIR_LIMIT ? u1_lo : u1_hi;

    float coord = lane < MINIMAX_H3_VVAE_AXIS_H_START ? ct
                : (lane < MINIMAX_H3_VVAE_AXIS_W_START ? cy : cx);
    uint fi = lane & (MINIMAX_H3_VVAE_FREQ_PER_AXIS - 1u);
    float angle = MINIMAX_H3_VAE_TWO_PI * coord * inv_freq[fi];
    float cs = cos(angle), sn = sin(angle);
    float r0 = a * cs - u1 * sn;
    float r1 = u1 * cs + a * sn;

    // Scatter rotated upper halves back to dims 24..31 (a[24..31]) and 32..47 (b[0..15]).
    float r1_for_a = simd_shuffle(r1, (lane - MINIMAX_H3_VVAE_ROT_PAIRS) & MINIMAX_H3_SIMD_MASK);
    float r1_for_b = simd_shuffle(r1, (lane + MINIMAX_H3_VVAE_LOW_PAIR_LIMIT) & MINIMAX_H3_SIMD_MASK);
    out_a = lane < MINIMAX_H3_VVAE_ROT_PAIRS ? r0 : r1_for_a;
    out_b = lane < MINIMAX_H3_VVAE_HI_HALF_ROT ? r1_for_b : b;
  }
  dst[lane] = half(out_a);
  dst[lane + MINIMAX_H3_VVAE_HALF_HEAD] = half(out_b);
}

// Bidirectional 32-head FlashAttention (head_dim = 64) for the 3D video VAE decoder.
// Dispatched as a 1D head-coherent grid (ceil(rows / 64) * 32) using 4-simdgroup
// cooperative 64x64 tiles and exact 16 KiB shared threadgroup memory.
kernel void minimax_h3_vvae_attend(device const half *qh [[buffer(0)]],
                                   device const half *kh [[buffer(1)]],
                                   device const half *vh [[buffer(2)]],
                                   device float *o [[buffer(3)]],
                                   constant MinimaxH3VvaeAttend &p [[buffer(4)]],
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
  const float scale_log2 = rsqrt(float(MINIMAX_H3_VVAE_HEAD_DIM)) * M_LOG2E_F;

  constexpr auto sdesc =
      matmul2d_descriptor(BQ, BK, MINIMAX_H3_VVAE_HEAD_DIM, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(BQ, MINIMAX_H3_VVAE_HEAD_DIM, BK, false, false, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> sop;
  matmul2d<odesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> oop;

  ulong head_off = (ulong)h * p.padded * MINIMAX_H3_VVAE_HEAD_DIM;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(
      (device half *)qh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VVAE_HEAD_DIM, p.padded));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(
      (device half *)kh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VVAE_HEAD_DIM, p.padded));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(
      (device half *)vh + head_off,
      dextents<int32_t, 2>(MINIMAX_H3_VVAE_HEAD_DIM, p.padded));
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
        o[(ulong)out_row * MINIMAX_H3_VVAE_DIM +
          h * MINIMAX_H3_VVAE_HEAD_DIM + at[0]] =
            acc[i] * smem.st.row_scale[at[1]];
    }
}

// Temporal-isolated GroupNorm + SiLU over NDHWC [t, h, w, c].
// Grid.x = group (0..groups-1), grid.y = frame (0..t-1); 256 threads/group.
kernel void minimax_h3_vvae_gn_silu_3d(device const float *x [[buffer(0)]],
                                       device const float *w [[buffer(1)]],
                                       device const float *b [[buffer(2)]],
                                       device float *y [[buffer(3)]],
                                       constant MinimaxH3VvaeGroupNorm3d &p [[buffer(4)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint t [[thread_index_in_threadgroup]],
                                       uint2 nt2 [[threads_per_threadgroup]]) {
  threadgroup float tmp[MINIMAX_H3_TG_MAX_SGROUPS];
  uint nt = nt2.x;
  uint frame_off = tg.y * p.hw * p.channels;
  device const float *xf = x + frame_off;
  device float *yf = y + frame_off;
  uint gc = p.channels / p.groups;
  uint c0 = tg.x * gc;
  uint count = p.hw * gc;

  float sum = 0.0f;
  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    sum += xf[pix * p.channels + ch];
  }
  float mean = tg_sum(sum, tmp, t, nt) / float(count);

  float sq = 0.0f;
  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    float d = xf[pix * p.channels + ch] - mean;
    sq += d * d;
  }
  float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(count) + p.eps);

  for (uint i = t; i < count; i += nt) {
    uint pix = i / gc;
    uint ch = c0 + (i % gc);
    uint idx = pix * p.channels + ch;
    float v = w[ch] * ((xf[idx] - mean) * inv) + b[ch];
    yf[idx] = v / (1.0f + exp(-v));
  }
}

// Causal-temporal, reflect-spatial 3D im2col for one output frame ot and
// spatial strip [out_y0, out_y0 + strip_h) from NDHWC [in_t, in_h, in_w, cin]
// into FP16 [padded_m, k_padded] in (kt, ky, kx, ci) tap order.
kernel void minimax_h3_vvae_im2col_3d_f16(device const float *x [[buffer(0)]],
                                          device half *dst [[buffer(1)]],
                                          constant MinimaxH3VvaeIm2Col3d &p [[buffer(2)]],
                                          uint2 i [[thread_position_in_grid]]) {
  uint col = i.x, row = i.y;
  if (col >= p.k_padded) return;
  uint m = p.strip_h * p.out_w;
  uint spatial_taps = p.ks * p.ks;
  uint k_raw = p.kt * spatial_taps * p.cin;
  if (row >= m || col >= k_raw) {
    dst[row * p.k_padded + col] = half(0.0f);
    return;
  }
  uint oy = p.out_y0 + row / p.out_w;
  uint ox = row % p.out_w;
  if (p.kt == 1u && p.ks == 1u) {
    dst[row * p.k_padded + col] =
        half(x[((p.ot * p.in_h + oy) * p.in_w + ox) * p.cin + col]);
    return;
  }
  uint tap = col / p.cin;
  uint c = col % p.cin;
  uint tk = tap / spatial_taps;
  uint st = tap % spatial_taps;
  uint ky = st / p.ks;
  uint kx = st % p.ks;
  int it = int(p.ot * p.time_stride + tk) - int(p.front_pad);
  if (it < 0 || it >= int(p.in_t)) {
    dst[row * p.k_padded + col] = half(0.0f);
    return;
  }
  uint iy = minimax_h3_reflect(int(oy * p.space_stride + ky) - int(p.pad_h), p.in_h);
  uint ix = minimax_h3_reflect(int(ox * p.space_stride + kx) - int(p.pad_w), p.in_w);
  dst[row * p.k_padded + col] =
      half(x[((uint(it) * p.in_h + iy) * p.in_w + ix) * p.cin + c]);
}
