// ---- Inkling-2-Mini: vision encoder (inkling2_vision) ----
//
// Kernels for the 23-layer Vision Transformer and 2x2 spatial-merge projector
// of thinkingmachines/Inkling-2-Mini (config.json vision_config,
// modeling_inkling2.py Inkling2VisionModel).

// Vision Transformer geometry (config.json vision_config).
constant constexpr uint INK2_VIS_DIM = 1024, INK2_VIS_HEADS = 16,
                        INK2_VIS_HEAD_DIM = INK2_VIS_DIM / INK2_VIS_HEADS,
                        INK2_VIS_MLP_DIM = 4096, INK2_VIS_PROJ_IN = 4096,
                        INK2_VIS_TEXT_DIM = 2304, INK2_VIS_PATCH_K_PAD = 1216,
                        INK2_VIS_BILINEAR_TAPS = 4;
// Hardware SIMD width and vectorization constants.
constant constexpr uint INK2_VIS_SIMD = 32, INK2_VIS_VEC4 = 4, INK2_VIS_PAIR = 2;
// Abramowitz and Stegun 7.1.26 constants for exact GELU via erf(x / sqrt(2)).
constant constexpr float INK2_VIS_INV_SQRT2 = 0.70710678118654752f,
                         INK2_VIS_ERF_P = 0.3275911f,
                         INK2_VIS_ERF_A1 = 0.254829592f,
                         INK2_VIS_ERF_A2 = -0.284496736f,
                         INK2_VIS_ERF_A3 = 1.421413741f,
                         INK2_VIS_ERF_A4 = -1.453152027f,
                         INK2_VIS_ERF_A5 = 1.061405429f,
                         INK2_VIS_HALF = 0.5f;

// Parameters for patch/output BF16<->FP32 conversion.
struct Ink2VisCastParam {
  // Total number of elements to convert.
  uint count;
};

// Parameters for vision sequence assembly (CLS token + patches + bilinear pos).
struct Ink2VisEmbedParam {
  // Number of image patches (excluding CLS) and hidden width (1024).
  uint npatches, dim;
};

// Parameters for vision LayerNorm.
struct Ink2VisLnParam {
  // Row width (1024) and LayerNorm epsilon (1e-5).
  uint dim;
  float eps;
};

// Parameters for bias addition with optional exact GELU activation.
struct Ink2VisBiasActParam {
  // Active row count, row width, nonzero to add bias, nonzero to apply GELU.
  uint rows, dim, has_bias, apply_gelu;
};

// Parameters for Q/K/V bias add, pairwise 2D axial RoPE and head transpose.
struct Ink2VisHeadsParam {
  // Active sequence length (1 + npatches) and padded head row stride (mult 64).
  uint seq, padded;
};

// Parameters for bidirectional tiled MPP vision attention.
struct Ink2VisAttendParam {
  // Active sequence length and padded head row stride (multiple of 64).
  uint seq, padded;
};

// Parameters for LayerScale residual addition: x = bf(x + gamma * bf(y + bias)).
struct Ink2VisScaleAddParam {
  // Active row count and row width (1024).
  uint rows, dim;
};

// Exact GELU via Abramowitz and Stegun 7.1.26 approximation to erf(x / sqrt(2)).
inline float ink2_vis_gelu(float x) {
  float z = fabs(x) * INK2_VIS_INV_SQRT2;
  float t = 1.0f / (1.0f + INK2_VIS_ERF_P * z);
  float poly =
      ((((INK2_VIS_ERF_A5 * t + INK2_VIS_ERF_A4) * t + INK2_VIS_ERF_A3) * t +
        INK2_VIS_ERF_A2) *
           t +
       INK2_VIS_ERF_A1) *
      t;
  float erf_abs = 1.0f - poly * exp(-z * z);
  float erf_v = x < 0.0f ? -erf_abs : erf_abs;
  return INK2_VIS_HALF * x * (1.0f + erf_v);
}

// Converts `count` BF16 values to FP32. Grid: count.
kernel void ink2_vis_bf16_to_f32(device const ushort *src [[buffer(0)]],
                                 device float *dst [[buffer(1)]],
                                 constant Ink2VisCastParam &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  dst[i] = bf16f(src[i]);
}

// Converts `count` FP32 values to BF16 codes (ties-to-even). Grid: count.
kernel void ink2_vis_f32_to_bf16(device const float *src [[buffer(0)]],
                                 device bfloat *dst [[buffer(1)]],
                                 constant Ink2VisCastParam &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  dst[i] = bfloat(src[i]);
}

// Assembles ViT input sequence [1 + npatches, 1024]: row 0 receives
// bf(class_emb + cls_pos_emb); rows 1..npatches receive patch_emb plus the
// 4-tap bilinear interpolation of the 32x32 position embedding table.
// Grid: x = dim (1024), y = 1 + npatches.
kernel void ink2_vis_embed_assemble(device const float *patch_emb [[buffer(0)]],
                                    device const float *pos_table [[buffer(1)]],
                                    device const float *cls_emb [[buffer(2)]],
                                    device const float *cls_pos [[buffer(3)]],
                                    device const uint *pos_idx [[buffer(4)]],
                                    device const float *pos_w [[buffer(5)]],
                                    device float *out [[buffer(6)]],
                                    constant Ink2VisEmbedParam &p [[buffer(7)]],
                                    uint2 i [[thread_position_in_grid]]) {
  uint d = i.x, r = i.y;
  if (d >= p.dim || r > p.npatches) return;
  if (r == 0u) {
    out[d] = bf(cls_emb[d] + cls_pos[d]);
    return;
  }
  uint patch = r - 1u;
  float pos = 0.0f;
  for (uint tap = 0; tap < INK2_VIS_BILINEAR_TAPS; tap++) {
    uint idx = pos_idx[(ulong)patch * INK2_VIS_BILINEAR_TAPS + tap];
    float w = pos_w[(ulong)patch * INK2_VIS_BILINEAR_TAPS + tap];
    pos += pos_table[(ulong)idx * p.dim + d] * w;
  }
  out[(ulong)r * p.dim + d] = bf(patch_emb[(ulong)patch * p.dim + d] + bf(pos));
}

// Affine LayerNorm over rows of `dim` floats, rounded to BF16.
// One threadgroup per row.
kernel void ink2_vis_layernorm(device const float *x [[buffer(0)]],
                               device const float *weight [[buffer(1)]],
                               device const float *bias [[buffer(2)]],
                               device float *y [[buffer(3)]],
                               constant Ink2VisLnParam &p [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]],
                               uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[INK2_VIS_SIMD];
  x += (ulong)row * p.dim;
  y += (ulong)row * p.dim;
  float sum = 0.0f;
  for (uint i = t; i < p.dim; i += nt) sum += x[i];
  float mean = tg_sum(sum, tmp, t, nt) / float(p.dim);
  float var_sum = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    float d = x[i] - mean;
    var_sum += d * d;
  }
  float inv = rsqrt(tg_sum(var_sum, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt)
    y[i] = bf((x[i] - mean) * inv * weight[i] + bias[i]);
}

// Adds optional bias and applies optional exact GELU, rounding to BF16.
// Grid: x = dim, y = rows.
kernel void ink2_vis_bias_act(device float *y [[buffer(0)]],
                              device const float *bias [[buffer(1)]],
                              constant Ink2VisBiasActParam &p [[buffer(2)]],
                              uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.dim || i.y >= p.rows) return;
  ulong idx = (ulong)i.y * p.dim + i.x;
  float v = p.has_bias ? bf(y[idx] + bias[i.x]) : y[idx];
  if (p.apply_gelu) v = bf(ink2_vis_gelu(v));
  y[idx] = v;
}

// Adds Q/K/V biases, applies pairwise 2D axial RoPE (width freqs then height
// freqs, modeling_inkling2.py apply_rotary_pos_emb_vision), and writes
// head-major BF16 Q, K, V [16, padded, 64] with zero-padded tail rows.
// Grid: x = 16 heads, y = padded rows. Threadgroup: 32 threads (1 pair each).
kernel void ink2_vis_heads_rope(device const float *q_in [[buffer(0)]],
                                device const float *k_in [[buffer(1)]],
                                device const float *v_in [[buffer(2)]],
                                device const float *q_bias [[buffer(3)]],
                                device const float *k_bias [[buffer(4)]],
                                device const float *v_bias [[buffer(5)]],
                                device const float *rope_cs [[buffer(6)]],
                                device bfloat *q_out [[buffer(7)]],
                                device bfloat *k_out [[buffer(8)]],
                                device bfloat *v_out [[buffer(9)]],
                                constant Ink2VisHeadsParam &p [[buffer(10)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint pair [[thread_index_in_threadgroup]]) {
  uint h = tg.x, r = tg.y;
  if (h >= INK2_VIS_HEADS || r >= p.padded || pair >= INK2_VIS_HEAD_DIM / INK2_VIS_PAIR) return;
  ulong dst = ((ulong)h * p.padded + r) * INK2_VIS_HEAD_DIM + pair * INK2_VIS_PAIR;
  if (r >= p.seq) {
    q_out[dst] = bfloat(0.0f);
    q_out[dst + 1] = bfloat(0.0f);
    k_out[dst] = bfloat(0.0f);
    k_out[dst + 1] = bfloat(0.0f);
    v_out[dst] = bfloat(0.0f);
    v_out[dst + 1] = bfloat(0.0f);
    return;
  }
  uint d0 = h * INK2_VIS_HEAD_DIM + pair * INK2_VIS_PAIR;
  ulong src = (ulong)r * INK2_VIS_DIM + d0;
  float q0 = bf(q_in[src] + q_bias[d0]);
  float q1 = bf(q_in[src + 1] + q_bias[d0 + 1]);
  float k0 = bf(k_in[src] + k_bias[d0]);
  float k1 = bf(k_in[src + 1] + k_bias[d0 + 1]);
  float v0 = bf(v_in[src] + v_bias[d0]);
  float v1 = bf(v_in[src + 1] + v_bias[d0 + 1]);

  // rope_cs stores (cos, sin) per interleaved pair: [seq, 32, 2].
  uint cs_idx = ((uint)r * (INK2_VIS_HEAD_DIM / INK2_VIS_PAIR) + pair) * INK2_VIS_PAIR;
  float c = rope_cs[cs_idx], s = rope_cs[cs_idx + 1];
  q_out[dst] = bfloat(bf(q0 * c - q1 * s));
  q_out[dst + 1] = bfloat(bf(q1 * c + q0 * s));
  k_out[dst] = bfloat(bf(k0 * c - k1 * s));
  k_out[dst + 1] = bfloat(bf(k1 * c + k0 * s));
  v_out[dst] = bfloat(v0);
  v_out[dst + 1] = bfloat(v1);
}

// Tiled MPP bidirectional FlashAttention for Inkling2VisionAttention
// (16 heads, head_dim 64). Dispatched as a 1D head-coherent grid
// (ceil(seq / 64) * 16) with 128 threads (4 simdgroups) cooperating on one
// 64x64 tile in exact 16 KiB threadgroup memory, reading Q/K/V directly from
// device memory (rows padded to a multiple of 64).
kernel void ink2_vis_attend(device const bfloat *q [[buffer(0)]],
                            device const bfloat *k [[buffer(1)]],
                            device const bfloat *v [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            constant Ink2VisAttendParam &p [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BQ = FLASH_ATTN_64_TILE, BK = FLASH_ATTN_64_TILE;
  threadgroup FlashAttn64Smem<bfloat> smem;

  uint q_tiles = (p.seq + BQ - 1u) / BQ;
  uint h = tg / q_tiles;
  uint q0 = (tg % q_tiles) * BQ;
  if (q0 >= p.seq) return;
  bool full_q = q0 + BQ <= p.seq;
  const float scale_log2 = rsqrt(float(INK2_VIS_HEAD_DIM)) * M_LOG2E_F;

  constexpr auto sdesc = matmul2d_descriptor(
      BQ, BK, INK2_VIS_HEAD_DIM, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(
      BQ, INK2_VIS_HEAD_DIM, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> sop;
  matmul2d<odesc, execution_simdgroups<FLASH_ATTN_64_SGROUPS>> oop;

  ulong head_off = (ulong)h * p.padded * INK2_VIS_HEAD_DIM;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)q + head_off, dextents<int32_t, 2>(INK2_VIS_HEAD_DIM, p.seq));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)k + head_off, dextents<int32_t, 2>(INK2_VIS_HEAD_DIM, p.seq));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)v + head_off, dextents<int32_t, 2>(INK2_VIS_HEAD_DIM, p.seq));
  tensor<threadgroup bfloat, extents<int32_t, BK, BQ>, tensor_inline> tp(
      smem.st.probs, extents<int32_t, BK, BQ>());

  auto qt = tq.slice(0, q0), vt0 = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  float rmax = -INFINITY, rsum = 0.0f;
  for (uint k0 = 0; k0 < p.seq; k0 += BK) {
    auto kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    sop.run(qt, kt, sc);
    bool full_tile = full_q && (k0 + BK <= p.seq);
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
          bool seen = (q0 + at[1] < p.seq) && (k0 + at[0] < p.seq);
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
      uint q_local = q0 + at[1];
      if (q_local < p.seq) {
        ulong off = (ulong)q_local * INK2_VIS_DIM + h * INK2_VIS_HEAD_DIM + at[0];
        out[off] = bf(acc[i] * smem.st.row_scale[at[1]]);
      }
    }
}

// Fused bias + LayerScale residual add (modeling_inkling2.py
// Inkling2VisionEncoderLayer): x[r, d] = bf(x[r, d] + gamma[d] * bf(y[r, d] + bias[d])).
// Grid: x = dim, y = rows.
kernel void ink2_vis_layerscale_add(device float *x [[buffer(0)]],
                                    device const float *y [[buffer(1)]],
                                    device const float *bias [[buffer(2)]],
                                    device const float *gamma [[buffer(3)]],
                                    constant Ink2VisScaleAddParam &p [[buffer(4)]],
                                    uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.dim || i.y >= p.rows) return;
  ulong idx = (ulong)i.y * p.dim + i.x;
  float v = bf(y[idx] + bias[i.x]);
  x[idx] = bf(x[idx] + bf(gamma[i.x] * v));
}
