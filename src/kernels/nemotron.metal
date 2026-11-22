// ---- Nemotron-3.5-Transcribe / Nemotron-H 52-layer hybrid backbone ----
//
// Kernels for NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 (model_type
// nemo_speechlm / nemotron_h). Equations match Hugging Face transformers
// modeling_nemotron_h.py and NVIDIA ModelOpt MIXED_PRECISION checkpoints:
// - 23 Mamba-2 SSM layers: per-tensor FP8 in_proj/out_proj, k=4 causal conv1d
//   with SiLU, FP32 selective SSM recurrence, group-512 gated RMSNorm.
// - 6 NoPE GQA 32/2 Attention layers: NVFP4 q/k/v/o_proj, head_dim=128, no RoPE.
// - 23 MoE layers: FP32 sigmoid + bias top-6 router (scale 2.5), NVFP4
//   non-gated relu2 shared expert + 128 NVFP4 routed experts (ExpertStore).

// Architecture geometry constants (llm_backbone/config.json).
constant constexpr uint NEMO_DIM = 2688, NEMO_EH_DIM = 2u * NEMO_DIM,
                        NEMO_HEADS = 32, NEMO_KV_HEADS = 2,
                        NEMO_GQA = NEMO_HEADS / NEMO_KV_HEADS,
                        NEMO_HEAD_DIM = 128,
                        NEMO_Q_DIM = NEMO_HEADS * NEMO_HEAD_DIM,
                        NEMO_KV_DIM = NEMO_KV_HEADS * NEMO_HEAD_DIM,
                        NEMO_MAMBA_HEADS = 64, NEMO_MAMBA_HEAD_DIM = 64,
                        NEMO_MAMBA_INNER = NEMO_MAMBA_HEADS * NEMO_MAMBA_HEAD_DIM,
                        NEMO_MAMBA_GROUPS = 8, NEMO_MAMBA_STATE = 128,
                        NEMO_MAMBA_BC = NEMO_MAMBA_GROUPS * NEMO_MAMBA_STATE,
                        NEMO_MAMBA_CONV_DIM = NEMO_MAMBA_INNER + 2u * NEMO_MAMBA_BC,
                        NEMO_MAMBA_IN_DIM = NEMO_MAMBA_INNER + NEMO_MAMBA_CONV_DIM + NEMO_MAMBA_HEADS,
                        NEMO_MAMBA_CONV_K = 4, NEMO_MAMBA_CONV_HIST = NEMO_MAMBA_CONV_K - 1u,
                        NEMO_MAMBA_NORM_GROUP = NEMO_MAMBA_INNER / NEMO_MAMBA_GROUPS,
                        NEMO_EXPERTS = 128, NEMO_TOPK = 6,
                        NEMO_MOE_INTER = 1856, NEMO_SHARED_INTER = 3712,
                        NEMO_NVFP4_BLOCK = 16;

// Hardware SIMD width and tile sizes for Metal matrix and attention kernels.
constant constexpr uint NEMO_SIMD = 32, NEMO_TILE = 64, NEMO_M_TILE2 = 128,
                        NEMO_MM_SGS = 4, NEMO_MV_ROWS = 4, NEMO_MV_MAX_M = 8,
                        NEMO_FP8_MV_SGS = 8, NEMO_NV_MV_SGS = 4,
                        NEMO_MOE_MV_SGS = 8, NEMO_SWIZZLE_N = 8,
                        NEMO_STAGE_HALF_K = 32, NEMO_STAGE_VEC4S = 8,
                        NEMO_SSM_SGS = 4, NEMO_DECODE_PART_FLOATS = NEMO_HEAD_DIM + 2u,
                        NEMO_ROUTE_SGS = NEMO_EXPERTS / NEMO_SIMD,
                        NEMO_UNEMBED_SGS = 8, NEMO_UNEMBED_ROWS_PER_SG = 4,
                        NEMO_UNEMBED_BF16_PER_U4 = 8,
                        NEMO_ARGMAX_THREADS = 256,
                        NEMO_ARGMAX_SGS = NEMO_ARGMAX_THREADS / NEMO_SIMD,
                        NEMO_KV_SKIP_POS = 0xFFFFFFFFu;

// Activation modes for dense linear kernels.
constant constexpr uint NEMO_ACT_NONE = 0, NEMO_ACT_RELU2 = 1;

// Numerical constants for routing, softplus stability and attention scaling.
constant constexpr float NEMO_ROUTE_EPS = 1e-20f,
                         NEMO_SOFTPLUS_THRESH = 20.0f,
                         NEMO_ATTN_SCALE = 0.08838834764831845f,          // 1 / sqrt(128)
                         NEMO_ATTN_LOG2_SCALE = 0.12751742702963406f;     // log2(e) / sqrt(128)

// Parameters for embedding lookup and NemotronHRMSNorm.
struct NemoNormParam {
  // Row width in floats and BF16 rounding flag.
  uint dim, round;
  // RMSNorm epsilon (1e-5).
  float eps;
};

// Parameters for per-tensor FP8 dense linear (mv and mm).
struct NemoFp8LinParam {
  // Active rows, output width and inner width.
  uint m, n, k;
  // Activation mode (NEMO_ACT_*), residual-add flag and BF16 rounding flag.
  uint act, has_res, round;
  // Per-tensor weight_scale multiplied by 256 to cancel fp8x4's 2^-8.
  float scale;
};

// Parameters for dense NVFP4 linear (mv and mm).
struct NemoNvfp4LinParam {
  // Active rows, output width and inner width.
  uint m, n, k;
  // Activation mode (NEMO_ACT_*), residual-add flag and BF16 rounding flag.
  uint act, has_res, round;
  // Per-tensor weight_scale_2 scalar.
  float scale2;
};

// Parameters for Mamba-2 causal conv1d and SSM recurrence.
struct NemoMambaParam {
  // Active token count (prefill rows or decode batch size).
  uint rows;
  // RMSNorm epsilon for NemotronHRMSNormGated (1e-5).
  float eps;
};

// Per-state device pointers for batched Mamba-2 decode.
struct NemoMambaTarget {
  // Device pointer to this state's conv_state ([3, 6144] FP32) for this layer.
  device float *conv;
  // Device pointer to this state's ssm_state ([64, 64, 128] FP32) for this layer.
  device float *ssm;
};

// Per-state device pointers for batched NoPE GQA 32/2 attention decode.
struct NemoKvTarget {
  // Device pointers to BF16 K and V buffers for KV heads 0 and 1 ([cap, 128]).
  device bfloat *k0;
  device bfloat *k1;
  device bfloat *v0;
  device bfloat *v1;
  // Current token position of this state.
  uint pos;
  // Padding for 16-byte struct alignment.
  uint pad;
};

// Parameters for NoPE QKV head prep and KV store.
struct NemoQkvParam {
  // Active token count, head-major Q row stride (multiple of 64 for prefill),
  // row-0 position (single-state mode), and batched-decode flag.
  uint rows, q_stride, pos0, batched;
};

// Parameters for causal NoPE GQA 32/2 FlashAttention prefill.
struct NemoAttnPrefillParam {
  // Active query rows, head-major Q stride, row-0 position and KV buffer cap.
  uint rows, q_stride, pos0, kv_cap;
};

// Parameters for causal NoPE GQA 32/2 FlashDecoding decode.
struct NemoAttnDecodeParam {
  // Active rows, Q stride, row-0 position, partition count, max partitions
  // stride, and batched-decode flag.
  uint rows, q_stride, pos0, nparts, part_stride, batched;
};

// Parameters for NemotronHTopkRouter (sigmoid + bias, top-6, scale 2.5).
struct NemoRouteParam {
  // Active token count, total experts (128), selected experts per token (6).
  uint rows, experts, topk;
  // Routed weight multiplier (2.5).
  float scale;
};

// One grouped-expert work tile: up to 64 consecutive (token, slot) pairs
// assigned to one routed expert.
struct NemoMoeTile {
  // Expert index (0..127), first pair index, pair count in tile, alignment pad.
  uint expert, start, count, pad;
};

// Device pointers for one NVFP4 routed expert's weights and block scales.
struct NemoExpertRef {
  // Packed up_proj NVFP4 weights [1856, 1344] and E4M3 scales [1856, 168].
  device const uchar *w_up;
  device const uchar *s_up;
  // Packed down_proj NVFP4 weights [2688, 928] and E4M3 scales [2688, 116].
  device const uchar *w_down;
  device const uchar *s_down;
};

// Parameters for NVFP4 routed MoE matvec and matmul kernels.
struct NemoMoeParam {
  // Output rows (N), input columns (K), total active pairs, and total tiles.
  uint n, k, npairs, ntiles;
};

// Parameters for routed MoE combine plus shared expert and residual add.
struct NemoMoeCombineParam {
  // Active token count, hidden width (2688) and selected experts per token (6).
  uint rows, dim, topk;
};

// Parameters for MTP embedding + hidden-state RMSNorm concatenation.
struct NemoMtpEhParam {
  // Active rows, hidden width (2688) and flag selecting h0 for row 0.
  uint rows, dim, use_h0;
  // RMSNorm epsilon (1e-5).
  float eps;
};

// Device pointers and per-tensor scales for one FP8 routed expert in MTP.
struct NemoFp8ExpertRef {
  // Raw F8_E4M3 up_proj [1856, 2688] and down_proj [2688, 1856] weights.
  device const uchar *w_up;
  device const uchar *w_down;
  // Per-tensor up_proj and down_proj weight scales multiplied by 256.
  float s_up;
  float s_down;
};

// Winning logit and token index for one stage-1 unembed argmax threadgroup.
struct NemoArgmaxPair {
  // Maximum logit value found by this threadgroup.
  float val;
  // First vocabulary index achieving val.
  int idx;
};

// Parameters for fused BF16 unembed + first-maximum argmax.
struct NemoUnembedParam {
  // Vocabulary size (131072) and hidden width (2688).
  uint vocab, k;
};

// Numerically stable SiLU activation.
inline float nemo_silu(float x) { return x * sigm(x); }

// Numerically stable softplus(x) = log(1 + exp(x)).
inline float nemo_softplus(float x) {
  return x > NEMO_SOFTPLUS_THRESH ? x : log(1.0f + exp(x));
}

// ReLU-squared activation matching PyTorch ReLUSquaredActivation.
inline float nemo_relu2(float x) {
  float r = max(x, 0.0f);
  return r * r;
}

// ---- Embedding and RMSNorm ----

// BF16 token embedding lookup into FP32 residual rows. Negative token IDs are
// skipped so pre-injected audio feature rows stay intact.
// Grid: rows threadgroups of 256 threads.
kernel void nemo_embed(device const ushort *table [[buffer(0)]],
                       device const int *ids [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant NemoNormParam &p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint t [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]]) {
  int id = ids[row];
  if (id < 0) return;
  device const ushort *src = table + (ulong)uint(id) * p.dim;
  device float *dst = out + (ulong)row * p.dim;
  for (uint i = t; i < p.dim; i += nt) dst[i] = bf16f(src[i]);
}

// Weighted RMSNorm matching NemotronHRMSNorm: FP32 variance and FP32 weight
// multiply, then optional BF16 rounding.
// Grid: rows threadgroups of 256 threads.
kernel void nemo_rmsnorm(device const float *x [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant NemoNormParam &p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_SIMD];
  x += (ulong)row * p.dim;
  y += (ulong)row * p.dim;
  float sq = 0.0f;
  for (uint i = t; i < p.dim; i += nt) sq += x[i] * x[i];
  float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt) {
    float v = weight[i] * (x[i] * inv);
    y[i] = p.round ? bf(v) : v;
  }
}

// ---- Per-tensor FP8 dense linear (Mamba-2 in_proj and out_proj) ----

// Multi-vec per-tensor FP8 linear y = x @ W^T * scale (+ res) for 1 <= m <= 8.
// Uses 8-byte-aligned uint2 weight loads (safe for any 8-byte aligned buffer).
// Grid: ceil(n / 32) threadgroups of 256 threads (8 simdgroups, 4 rows/sg).
kernel void nemo_fp8_mv(device const uchar *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device const float *res [[buffer(2)]],
                        device float *y [[buffer(3)]],
                        constant NemoFp8LinParam &p [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg * NEMO_FP8_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;
  uint chunks = p.k / 8u;
  device const uchar *we = w + (ulong)row * p.k;

#define NEMO_FP8_MV_STORE(m_idx, av)                                           \
  {                                                                            \
    float4 s = float4(simd_sum(av.x), simd_sum(av.y), simd_sum(av.z),          \
                      simd_sum(av.w)) *                                        \
               p.scale;                                                        \
    if (lane < NEMO_MV_ROWS && row + lane < p.n) {                             \
      float v = p.round ? bf(s[lane]) : s[lane];                               \
      if (p.act == NEMO_ACT_RELU2) {                                           \
        v = nemo_relu2(v);                                                     \
        if (p.round) v = bf(v);                                                \
      }                                                                        \
      ulong out_i = (ulong)(m_idx) * p.n + row + lane;                         \
      if (p.has_res) v = p.round ? bf(v + res[out_i]) : (v + res[out_i]);      \
      y[out_i] = v;                                                            \
    }                                                                          \
  }

  if (p.m == 1u) {
    float4 acc = float4(0.0f);
    device const float4 *xr = (device const float4 *)x;
    for (uint c = lane; c < chunks; c += NEMO_SIMD) {
      float4 x0 = xr[2u * c], x1 = xr[2u * c + 1u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
        float4 prod = fp8x4(q.x) * x0 + fp8x4(q.y) * x1;
        acc[r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
    NEMO_FP8_MV_STORE(0u, acc);
    return;
  }

  if (p.m == 2u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    device const float4 *x0r = (device const float4 *)x;
    device const float4 *x1r = (device const float4 *)(x + p.k);
    for (uint c = lane; c < chunks; c += NEMO_SIMD) {
      float4 a0 = x0r[2u * c], a1 = x0r[2u * c + 1u];
      float4 b0 = x1r[2u * c], b1 = x1r[2u * c + 1u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
        float4 w0 = fp8x4(q.x), w1 = fp8x4(q.y);
        float4 p0 = w0 * a0 + w1 * a1;
        float4 p1 = w0 * b0 + w1 * b1;
        acc0[r] += p0.x + p0.y + p0.z + p0.w;
        acc1[r] += p1.x + p1.y + p1.z + p1.w;
      }
    }
    NEMO_FP8_MV_STORE(0u, acc0);
    NEMO_FP8_MV_STORE(1u, acc1);
    return;
  }

  if (p.m <= 4u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
           acc3 = float4(0.0f);
    for (uint c = lane; c < chunks; c += NEMO_SIMD) {
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
        float4 w0 = fp8x4(q.x), w1 = fp8x4(q.y);
#define NEMO_FP8_MV_DOT(mi, av)                                                \
  {                                                                            \
    device const float4 *xr = (device const float4 *)(x + (ulong)(mi) * p.k);  \
    float4 prod = w0 * xr[2u * c] + w1 * xr[2u * c + 1u];                      \
    av[r] += prod.x + prod.y + prod.z + prod.w;                                \
  }
        NEMO_FP8_MV_DOT(0u, acc0);
        NEMO_FP8_MV_DOT(1u, acc1);
        NEMO_FP8_MV_DOT(2u, acc2);
        if (p.m > 3u) NEMO_FP8_MV_DOT(3u, acc3);
      }
    }
    NEMO_FP8_MV_STORE(0u, acc0);
    NEMO_FP8_MV_STORE(1u, acc1);
    NEMO_FP8_MV_STORE(2u, acc2);
    if (p.m > 3u) NEMO_FP8_MV_STORE(3u, acc3);
    return;
  }

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
         acc3 = float4(0.0f), acc4 = float4(0.0f), acc5 = float4(0.0f),
         acc6 = float4(0.0f), acc7 = float4(0.0f);
  for (uint c = lane; c < chunks; c += NEMO_SIMD) {
#pragma unroll
    for (uint r = 0; r < NEMO_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
      float4 w0 = fp8x4(q.x), w1 = fp8x4(q.y);
      NEMO_FP8_MV_DOT(0u, acc0);
      NEMO_FP8_MV_DOT(1u, acc1);
      NEMO_FP8_MV_DOT(2u, acc2);
      NEMO_FP8_MV_DOT(3u, acc3);
      NEMO_FP8_MV_DOT(4u, acc4);
      if (p.m > 5u) NEMO_FP8_MV_DOT(5u, acc5);
      if (p.m > 6u) NEMO_FP8_MV_DOT(6u, acc6);
      if (p.m > 7u) NEMO_FP8_MV_DOT(7u, acc7);
    }
  }
  NEMO_FP8_MV_STORE(0u, acc0);
  NEMO_FP8_MV_STORE(1u, acc1);
  NEMO_FP8_MV_STORE(2u, acc2);
  NEMO_FP8_MV_STORE(3u, acc3);
  NEMO_FP8_MV_STORE(4u, acc4);
  if (p.m > 5u) NEMO_FP8_MV_STORE(5u, acc5);
  if (p.m > 6u) NEMO_FP8_MV_STORE(6u, acc6);
  if (p.m > 7u) NEMO_FP8_MV_STORE(7u, acc7);
#undef NEMO_FP8_MV_DOT
#undef NEMO_FP8_MV_STORE
}

// Packs FP32 activations [rows, cols] into BF16 [padded_rows, cols] with
// zero-padded tail rows for 128x64 MPP GEMM kernels.
// Grid: x = cols / 4, y = padded_rows.
kernel void nemo_pack_bf16(device const float4 *x [[buffer(0)]],
                           device bfloat4 *out [[buffer(1)]],
                           constant uint3 &rc [[buffer(2)]],
                           uint2 i [[thread_position_in_grid]]) {
  uint rows = rc.x, cols4 = rc.z;
  if (i.x >= cols4) return;
  out[(ulong)i.y * cols4 + i.x] =
      i.y < rows ? bfloat4(x[(ulong)i.y * cols4 + i.x])
                 : bfloat4(bfloat(0.0f));
}

// Stages one 64x64 FP8 weight tile (with 2^-8 implicit scale) into threadgroup
// BF16 memory across 128 threads using 8-byte-aligned uint2 loads.
inline void nemo_stage_fp8_tile(device const uchar *w, uint n0, uint k0, uint n,
                                uint k, threadgroup bfloat *ws, uint t) {
  uint r_local = t / 2u, half_k = t & 1u;
  uint r = n0 + r_local;
  threadgroup bfloat4 *dst =
      (threadgroup bfloat4 *)(ws + r_local * NEMO_TILE + half_k * NEMO_STAGE_HALF_K);
  if (r < n) {
    device const uint2 *qp =
        (device const uint2 *)(w + (ulong)r * k + k0 + half_k * NEMO_STAGE_HALF_K);
    uint2 q0 = qp[0], q1 = qp[1], q2 = qp[2], q3 = qp[3];
    dst[0] = bfloat4(fp8x4(q0.x));
    dst[1] = bfloat4(fp8x4(q0.y));
    dst[2] = bfloat4(fp8x4(q1.x));
    dst[3] = bfloat4(fp8x4(q1.y));
    dst[4] = bfloat4(fp8x4(q2.x));
    dst[5] = bfloat4(fp8x4(q2.y));
    dst[6] = bfloat4(fp8x4(q3.x));
    dst[7] = bfloat4(fp8x4(q3.y));
  } else {
    for (uint i = 0; i < NEMO_STAGE_VEC4S; i++) dst[i] = bfloat4(bfloat(0.0f));
  }
}

// 128x64 tiled MPP GEMM for per-tensor FP8 weights against pre-packed BF16
// inputs x_bf16 [padded_m, k]: y = x @ W^T * scale (+ res).
// Grid: 1D L2-swizzled ceil(n / 64) * ceil(m / 128) threadgroups of 128 threads.
kernel void nemo_fp8_mm(device const uchar *w [[buffer(0)]],
                        device const bfloat *x_bf16 [[buffer(1)]],
                        device const float *res [[buffer(2)]],
                        device float *y [[buffer(3)]],
                        constant NemoFp8LinParam &p [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat ws[NEMO_TILE * NEMO_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_TILE, NEMO_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_MM_SGS>> op;

  uint tiles_m = (p.m + NEMO_M_TILE2 - 1u) / NEMO_M_TILE2;
  uint tiles_n = (p.n + NEMO_TILE - 1u) / NEMO_TILE;
  uint group_tiles = tiles_m * NEMO_SWIZZLE_N;
  uint g = tg / group_tiles;
  uint rem = tg - g * group_tiles;
  uint gn0 = g * NEMO_SWIZZLE_N;
  uint gwidth = min(NEMO_SWIZZLE_N, tiles_n - gn0);
  uint tn = gn0 + (rem % gwidth);
  uint tm = rem / gwidth;
  uint m0 = tm * NEMO_M_TILE2, n0 = tn * NEMO_TILE;
  bool has_m1 = (m0 + NEMO_TILE) < p.m;

  uint padded_m = tiles_m * NEMO_M_TILE2;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, padded_m));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_TILE, NEMO_TILE>,
         tensor_inline>
      tw(ws, extents<int32_t, NEMO_TILE, NEMO_TILE>());
  auto c0 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  for (uint16_t i = 0; i < c0.get_capacity(); i++) {
    if (c0.is_valid_element(i)) c0[i] = 0.0f;
    if (c1.is_valid_element(i)) c1[i] = 0.0f;
  }

  for (uint k0 = 0; k0 < p.k; k0 += NEMO_TILE) {
    nemo_stage_fp8_tile(w, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a0 = tx.slice(k0, m0);
    op.run(a0, tw, c0);
    if (has_m1) {
      auto a1 = tx.slice(k0, m0 + NEMO_TILE);
      op.run(a1, tw, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c0.get_capacity(); i++) {
    if (c0.is_valid_element(i)) {
      auto at = c0.get_multidimensional_index(i);
      uint col = n0 + at[0];
      if (col >= p.n) continue;
      uint r0 = m0 + at[1];
      if (r0 < p.m) {
        float v = c0[i] * p.scale;
        if (p.round) v = bf(v);
        if (p.act == NEMO_ACT_RELU2) {
          v = nemo_relu2(v);
          if (p.round) v = bf(v);
        }
        ulong idx = (ulong)r0 * p.n + col;
        if (p.has_res) v = p.round ? bf(v + res[idx]) : (v + res[idx]);
        y[idx] = v;
      }
      if (has_m1) {
        uint r1 = m0 + NEMO_TILE + at[1];
        if (r1 < p.m) {
          float v = c1[i] * p.scale;
          if (p.round) v = bf(v);
          if (p.act == NEMO_ACT_RELU2) {
            v = nemo_relu2(v);
            if (p.round) v = bf(v);
          }
          ulong idx = (ulong)r1 * p.n + col;
          if (p.has_res) v = p.round ? bf(v + res[idx]) : (v + res[idx]);
          y[idx] = v;
        }
      }
    }
  }
}

// ---- Dense NVFP4 linear (Attention q/k/v/o_proj and shared_experts) ----

// Multi-vec dense NVFP4 linear y = act(x @ W^T * scale2) (+ res) for 1 <= m <= 8.
// Uses 8-byte-aligned uint2 weight loads and 16-value E4M3 block scales.
// Grid: ceil(n / 16) threadgroups of 128 threads (4 simdgroups, 4 rows/sg).
kernel void nemo_nvfp4_mv(device const uchar *w [[buffer(0)]],
                          device const uchar *sc [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device const float *res [[buffer(3)]],
                          device float *y [[buffer(4)]],
                          constant NemoNvfp4LinParam &p [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg * NEMO_NV_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;
  uint blocks = p.k / NEMO_NVFP4_BLOCK;
  uint w_row_bytes = p.k / 2u;
  device const uchar *we = w + (ulong)row * w_row_bytes;
  device const uchar *se = sc + (ulong)row * blocks;

#define NEMO_NV_MV_STORE(m_idx, av)                                            \
  {                                                                            \
    float4 s = float4(simd_sum(av.x), simd_sum(av.y), simd_sum(av.z),          \
                      simd_sum(av.w)) *                                        \
               p.scale2;                                                       \
    if (lane < NEMO_MV_ROWS && row + lane < p.n) {                             \
      float v = p.round ? bf(s[lane]) : s[lane];                               \
      if (p.act == NEMO_ACT_RELU2) {                                           \
        v = nemo_relu2(v);                                                     \
        if (p.round) v = bf(v);                                                \
      }                                                                        \
      ulong out_i = (ulong)(m_idx) * p.n + row + lane;                         \
      if (p.has_res) v = p.round ? bf(v + res[out_i]) : (v + res[out_i]);      \
      y[out_i] = v;                                                            \
    }                                                                          \
  }

  if (p.m == 1u) {
    float4 acc = float4(0.0f);
    device const float4 *xr = (device const float4 *)x;
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 x0 = xr[4u * c], x1 = xr[4u * c + 1u], x2 = xr[4u * c + 2u],
             x3 = xr[4u * c + 3u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float s = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
        float4 prod = a * x0 + b * x1 + d0 * x2 + d1 * x3;
        acc[r] += (prod.x + prod.y + prod.z + prod.w) * s;
      }
    }
    NEMO_NV_MV_STORE(0u, acc);
    return;
  }

  if (p.m == 2u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    device const float4 *x0r = (device const float4 *)x;
    device const float4 *x1r = (device const float4 *)(x + p.k);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 a0 = x0r[4u * c], a1 = x0r[4u * c + 1u], a2 = x0r[4u * c + 2u],
             a3 = x0r[4u * c + 3u];
      float4 b0 = x1r[4u * c], b1 = x1r[4u * c + 1u], b2 = x1r[4u * c + 2u],
             b3 = x1r[4u * c + 3u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float s = fp8(se[(ulong)r * blocks + c]);
        float4 wa, wb, wc, wd;
        fp4x8(q.x, wa, wb);
        fp4x8(q.y, wc, wd);
        float4 p0 = wa * a0 + wb * a1 + wc * a2 + wd * a3;
        float4 p1 = wa * b0 + wb * b1 + wc * b2 + wd * b3;
        acc0[r] += (p0.x + p0.y + p0.z + p0.w) * s;
        acc1[r] += (p1.x + p1.y + p1.z + p1.w) * s;
      }
    }
    NEMO_NV_MV_STORE(0u, acc0);
    NEMO_NV_MV_STORE(1u, acc1);
    return;
  }

  if (p.m <= 4u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
           acc3 = float4(0.0f);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float s = fp8(se[(ulong)r * blocks + c]);
        float4 wa, wb, wc, wd;
        fp4x8(q.x, wa, wb);
        fp4x8(q.y, wc, wd);
#define NEMO_NV_MV_DOT(mi, av)                                                 \
  {                                                                            \
    device const float4 *xr =                                                  \
        (device const float4 *)(x + (ulong)(mi) * p.k) + 4u * c;               \
    float4 prod = wa * xr[0] + wb * xr[1] + wc * xr[2] + wd * xr[3];           \
    av[r] += (prod.x + prod.y + prod.z + prod.w) * s;                          \
  }
        NEMO_NV_MV_DOT(0u, acc0);
        NEMO_NV_MV_DOT(1u, acc1);
        NEMO_NV_MV_DOT(2u, acc2);
        if (p.m > 3u) NEMO_NV_MV_DOT(3u, acc3);
      }
    }
    NEMO_NV_MV_STORE(0u, acc0);
    NEMO_NV_MV_STORE(1u, acc1);
    NEMO_NV_MV_STORE(2u, acc2);
    if (p.m > 3u) NEMO_NV_MV_STORE(3u, acc3);
    return;
  }

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
         acc3 = float4(0.0f), acc4 = float4(0.0f), acc5 = float4(0.0f),
         acc6 = float4(0.0f), acc7 = float4(0.0f);
  for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
    for (uint r = 0; r < NEMO_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
      float s = fp8(se[(ulong)r * blocks + c]);
      float4 wa, wb, wc, wd;
      fp4x8(q.x, wa, wb);
      fp4x8(q.y, wc, wd);
      NEMO_NV_MV_DOT(0u, acc0);
      NEMO_NV_MV_DOT(1u, acc1);
      NEMO_NV_MV_DOT(2u, acc2);
      NEMO_NV_MV_DOT(3u, acc3);
      NEMO_NV_MV_DOT(4u, acc4);
      if (p.m > 5u) NEMO_NV_MV_DOT(5u, acc5);
      if (p.m > 6u) NEMO_NV_MV_DOT(6u, acc6);
      if (p.m > 7u) NEMO_NV_MV_DOT(7u, acc7);
    }
  }
  NEMO_NV_MV_STORE(0u, acc0);
  NEMO_NV_MV_STORE(1u, acc1);
  NEMO_NV_MV_STORE(2u, acc2);
  NEMO_NV_MV_STORE(3u, acc3);
  NEMO_NV_MV_STORE(4u, acc4);
  if (p.m > 5u) NEMO_NV_MV_STORE(5u, acc5);
  if (p.m > 6u) NEMO_NV_MV_STORE(6u, acc6);
  if (p.m > 7u) NEMO_NV_MV_STORE(7u, acc7);
#undef NEMO_NV_MV_DOT
#undef NEMO_NV_MV_STORE
}

// Stages one 64x64 NVFP4 weight tile (scaled by E4M3 block scale, bit-exact in
// BF16) into threadgroup BF16 memory across 128 threads using 8-byte-aligned
// uint2 loads.
inline void nemo_stage_nvfp4_tile(device const uchar *we, device const uchar *se,
                                  uint n0, uint k0, uint n, uint k,
                                  threadgroup bfloat *ws, uint t) {
  uint r_local = t / 2u, half_k = t & 1u;
  uint r = n0 + r_local;
  uint w_row_bytes = k / 2u, blocks = k / NEMO_NVFP4_BLOCK;
  threadgroup bfloat4 *dst =
      (threadgroup bfloat4 *)(ws + r_local * NEMO_TILE + half_k * NEMO_STAGE_HALF_K);
  if (r < n) {
    uint b0 = (k0 + half_k * NEMO_STAGE_HALF_K) / NEMO_NVFP4_BLOCK;
    device const uint2 *qp = (device const uint2 *)(
        we + (ulong)r * w_row_bytes + (k0 + half_k * NEMO_STAGE_HALF_K) / 2u);
    uint2 q0 = qp[0], q1 = qp[1];
    float sc0 = fp8(se[(ulong)r * blocks + b0]);
    float sc1 = fp8(se[(ulong)r * blocks + b0 + 1u]);
    float4 a, b, c, d;
    fp4x8(q0.x, a, b);
    dst[0] = bfloat4(a * sc0);
    dst[1] = bfloat4(b * sc0);
    fp4x8(q0.y, c, d);
    dst[2] = bfloat4(c * sc0);
    dst[3] = bfloat4(d * sc0);
    fp4x8(q1.x, a, b);
    dst[4] = bfloat4(a * sc1);
    dst[5] = bfloat4(b * sc1);
    fp4x8(q1.y, c, d);
    dst[6] = bfloat4(c * sc1);
    dst[7] = bfloat4(d * sc1);
  } else {
    for (uint i = 0; i < NEMO_STAGE_VEC4S; i++) dst[i] = bfloat4(bfloat(0.0f));
  }
}

// 128x64 tiled MPP GEMM for dense NVFP4 weights against pre-packed BF16 inputs
// x_bf16 [padded_m, k]: y = act(x @ W^T * scale2) (+ res).
// Grid: 1D L2-swizzled ceil(n / 64) * ceil(m / 128) threadgroups of 128 threads.
kernel void nemo_nvfp4_mm(device const uchar *w [[buffer(0)]],
                          device const uchar *sc [[buffer(1)]],
                          device const bfloat *x_bf16 [[buffer(2)]],
                          device const float *res [[buffer(3)]],
                          device float *y [[buffer(4)]],
                          constant NemoNvfp4LinParam &p [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat ws[NEMO_TILE * NEMO_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_TILE, NEMO_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_MM_SGS>> op;

  uint tiles_m = (p.m + NEMO_M_TILE2 - 1u) / NEMO_M_TILE2;
  uint tiles_n = (p.n + NEMO_TILE - 1u) / NEMO_TILE;
  uint group_tiles = tiles_m * NEMO_SWIZZLE_N;
  uint g = tg / group_tiles;
  uint rem = tg - g * group_tiles;
  uint gn0 = g * NEMO_SWIZZLE_N;
  uint gwidth = min(NEMO_SWIZZLE_N, tiles_n - gn0);
  uint tn = gn0 + (rem % gwidth);
  uint tm = rem / gwidth;
  uint m0 = tm * NEMO_M_TILE2, n0 = tn * NEMO_TILE;
  bool has_m1 = (m0 + NEMO_TILE) < p.m;

  uint padded_m = tiles_m * NEMO_M_TILE2;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, padded_m));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_TILE, NEMO_TILE>,
         tensor_inline>
      tw(ws, extents<int32_t, NEMO_TILE, NEMO_TILE>());
  auto c0 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                  decltype(tw), float>();
  for (uint16_t i = 0; i < c0.get_capacity(); i++) {
    if (c0.is_valid_element(i)) c0[i] = 0.0f;
    if (c1.is_valid_element(i)) c1[i] = 0.0f;
  }

  for (uint k0 = 0; k0 < p.k; k0 += NEMO_TILE) {
    nemo_stage_nvfp4_tile(w, sc, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a0 = tx.slice(k0, m0);
    op.run(a0, tw, c0);
    if (has_m1) {
      auto a1 = tx.slice(k0, m0 + NEMO_TILE);
      op.run(a1, tw, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c0.get_capacity(); i++) {
    if (c0.is_valid_element(i)) {
      auto at = c0.get_multidimensional_index(i);
      uint col = n0 + at[0];
      if (col >= p.n) continue;
      for (uint h = 0; h < 2u; h++) {
        if (h == 1u && !has_m1) break;
        uint row = m0 + h * NEMO_TILE + at[1];
        if (row >= p.m) continue;
        float v = (h ? c1[i] : c0[i]) * p.scale2;
        if (p.round) v = bf(v);
        if (p.act == NEMO_ACT_RELU2) {
          v = nemo_relu2(v);
          if (p.round) v = bf(v);
        }
        ulong idx = (ulong)row * p.n + col;
        if (p.has_res) v = p.round ? bf(v + res[idx]) : (v + res[idx]);
        y[idx] = v;
      }
    }
  }
}

// ---- Mamba-2 SSM (conv1d + SiLU, FP32 selective scan, gated group RMSNorm) ----

// Depthwise causal conv1d (k=4) + SiLU over a contiguous chunk of `rows` tokens
// from zxbcdt [rows, 10304] using conv_state [3, 6144], writing xBC_out
// [rows, 6144]. Grid: x = 6144, y = rows.
kernel void nemo_mamba_conv_prefill(device const float *zxbcdt [[buffer(0)]],
                                    device const float *conv_state [[buffer(1)]],
                                    device const float *conv_w [[buffer(2)]],
                                    device const float *conv_b [[buffer(3)]],
                                    device float *xbc_out [[buffer(4)]],
                                    constant NemoMambaParam &p [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x, r = gid.y;
  if (c >= NEMO_MAMBA_CONV_DIM || r >= p.rows) return;
  float4 w = ((device const float4 *)conv_w)[c];
  float acc = conv_b[c];
  float t0 = r >= 3u ? zxbcdt[(ulong)(r - 3u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c]
                     : conv_state[(ulong)r * NEMO_MAMBA_CONV_DIM + c];
  float t1 = r >= 2u ? zxbcdt[(ulong)(r - 2u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c]
                     : conv_state[(ulong)(r + 1u) * NEMO_MAMBA_CONV_DIM + c];
  float t2 = r >= 1u ? zxbcdt[(ulong)(r - 1u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c]
                     : conv_state[(ulong)2u * NEMO_MAMBA_CONV_DIM + c];
  float t3 = zxbcdt[(ulong)r * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
  acc += w.x * t0 + w.y * t1 + w.z * t2 + w.w * t3;
  xbc_out[(ulong)r * NEMO_MAMBA_CONV_DIM + c] = bf(nemo_silu(bf(acc)));
}

// Updates conv_state [3, 6144] with the last 3 raw pre-conv xBC rows after
// nemo_mamba_conv_prefill. Grid: x = 6144.
kernel void nemo_mamba_conv_save(device const float *zxbcdt [[buffer(0)]],
                                 device float *conv_state [[buffer(1)]],
                                 constant NemoMambaParam &p [[buffer(2)]],
                                 uint c [[thread_position_in_grid]]) {
  if (c >= NEMO_MAMBA_CONV_DIM) return;
  float s0 = conv_state[c];
  float s1 = conv_state[NEMO_MAMBA_CONV_DIM + c];
  float s2 = conv_state[2u * NEMO_MAMBA_CONV_DIM + c];
  float n0, n1, n2;
  if (p.rows >= 3u) {
    n0 = zxbcdt[(ulong)(p.rows - 3u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
    n1 = zxbcdt[(ulong)(p.rows - 2u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
    n2 = zxbcdt[(ulong)(p.rows - 1u) * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
  } else if (p.rows == 2u) {
    n0 = s2;
    n1 = zxbcdt[NEMO_MAMBA_INNER + c];
    n2 = zxbcdt[NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
  } else {
    n0 = s1;
    n1 = s2;
    n2 = zxbcdt[NEMO_MAMBA_INNER + c];
  }
  conv_state[c] = n0;
  conv_state[NEMO_MAMBA_CONV_DIM + c] = n1;
  conv_state[2u * NEMO_MAMBA_CONV_DIM + c] = n2;
}

// Single-token depthwise causal conv1d (k=4) + SiLU + in-place conv_state
// shift for a decode batch of `rows` distinct states.
// Grid: x = 6144, y = rows.
kernel void nemo_mamba_conv_decode(device const float *zxbcdt [[buffer(0)]],
                                   device const NemoMambaTarget *targets [[buffer(1)]],
                                   device const float *conv_w [[buffer(2)]],
                                   device const float *conv_b [[buffer(3)]],
                                   device float *xbc_out [[buffer(4)]],
                                   constant NemoMambaParam &p [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x, r = gid.y;
  if (c >= NEMO_MAMBA_CONV_DIM || r >= p.rows) return;
  device float *cs = targets[r].conv;
  float s0 = cs[c];
  float s1 = cs[NEMO_MAMBA_CONV_DIM + c];
  float s2 = cs[2u * NEMO_MAMBA_CONV_DIM + c];
  float s3 = zxbcdt[(ulong)r * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c];
  cs[c] = s1;
  cs[NEMO_MAMBA_CONV_DIM + c] = s2;
  cs[2u * NEMO_MAMBA_CONV_DIM + c] = s3;
  float4 w = ((device const float4 *)conv_w)[c];
  float acc = conv_b[c] + w.x * s0 + w.y * s1 + w.z * s2 + w.w * s3;
  xbc_out[(ulong)r * NEMO_MAMBA_CONV_DIM + c] = bf(nemo_silu(bf(acc)));
}

// Register-resident FP32 selective SSM scan over a contiguous chunk of `rows`
// tokens for one state, updating ssm_state [64, 64, 128] in place and writing
// FP32 y_out [rows, 4096].
// Grid: x = 16 (head_dim / 4), y = 64 (heads); threadgroup = 128 threads (4 sgs).
kernel void nemo_mamba_ssm_prefill(device const float *zxbcdt [[buffer(0)]],
                                   device const float *xbc [[buffer(1)]],
                                   device float *ssm_state [[buffer(2)]],
                                   device const float *a_exp [[buffer(3)]],
                                   device const float *d_skip [[buffer(4)]],
                                   device const float *dt_bias [[buffer(5)]],
                                   device float *y_out [[buffer(6)]],
                                   constant NemoMambaParam &p [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  uint p_idx = tg.x * NEMO_SSM_SGS + sg;
  uint head = tg.y;
  uint group = head / (NEMO_MAMBA_HEADS / NEMO_MAMBA_GROUPS);

  device float4 *st_ptr =
      (device float4 *)(ssm_state +
                        ((ulong)head * NEMO_MAMBA_HEAD_DIM + p_idx) * NEMO_MAMBA_STATE);
  float4 h = st_ptr[lane];
  float a_neg = -a_exp[head];
  float d_val = d_skip[head];
  float dt_b = dt_bias[head];
  uint x_col = head * NEMO_MAMBA_HEAD_DIM + p_idx;
  uint b_off = NEMO_MAMBA_INNER + group * NEMO_MAMBA_STATE;
  uint c_off = NEMO_MAMBA_INNER + NEMO_MAMBA_BC + group * NEMO_MAMBA_STATE;
  uint dt_off = NEMO_MAMBA_INNER + NEMO_MAMBA_CONV_DIM + head;

  for (uint t = 0; t < p.rows; t++) {
    float dt = nemo_softplus(zxbcdt[(ulong)t * NEMO_MAMBA_IN_DIM + dt_off] + dt_b);
    float dA = exp(dt * a_neg);
    device const float *row_xbc = xbc + (ulong)t * NEMO_MAMBA_CONV_DIM;
    float xv = row_xbc[x_col];
    float4 bv = ((device const float4 *)(row_xbc + b_off))[lane];
    float4 cv = ((device const float4 *)(row_xbc + c_off))[lane];
    h = h * dA + (dt * xv) * bv;
    float4 prod = h * cv;
    float yv = simd_sum(prod.x + prod.y + prod.z + prod.w) + d_val * xv;
    if (lane == 0u) y_out[(ulong)t * NEMO_MAMBA_INNER + x_col] = yv;
  }
  st_ptr[lane] = h;
}

// Single-token FP32 selective SSM update for a decode batch of `rows` distinct
// states. Grid: x = 16, y = 64, z = rows; threadgroup = 128 threads (4 sgs).
kernel void nemo_mamba_ssm_decode(device const float *zxbcdt [[buffer(0)]],
                                  device const float *xbc [[buffer(1)]],
                                  device const NemoMambaTarget *targets [[buffer(2)]],
                                  device const float *a_exp [[buffer(3)]],
                                  device const float *d_skip [[buffer(4)]],
                                  device const float *dt_bias [[buffer(5)]],
                                  device float *y_out [[buffer(6)]],
                                  constant NemoMambaParam &p [[buffer(7)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  uint p_idx = tg.x * NEMO_SSM_SGS + sg;
  uint head = tg.y;
  uint r = tg.z;
  if (r >= p.rows) return;
  uint group = head / (NEMO_MAMBA_HEADS / NEMO_MAMBA_GROUPS);

  device float4 *st_ptr =
      (device float4 *)(targets[r].ssm +
                        ((ulong)head * NEMO_MAMBA_HEAD_DIM + p_idx) * NEMO_MAMBA_STATE);
  float4 h = st_ptr[lane];
  uint x_col = head * NEMO_MAMBA_HEAD_DIM + p_idx;
  uint dt_off = NEMO_MAMBA_INNER + NEMO_MAMBA_CONV_DIM + head;
  float dt = nemo_softplus(zxbcdt[(ulong)r * NEMO_MAMBA_IN_DIM + dt_off] + dt_bias[head]);
  float dA = exp(dt * (-a_exp[head]));
  device const float *row_xbc = xbc + (ulong)r * NEMO_MAMBA_CONV_DIM;
  float xv = row_xbc[x_col];
  float4 bv = ((device const float4 *)(row_xbc + NEMO_MAMBA_INNER + group * NEMO_MAMBA_STATE))[lane];
  float4 cv = ((device const float4 *)(row_xbc + NEMO_MAMBA_INNER + NEMO_MAMBA_BC +
                                       group * NEMO_MAMBA_STATE))[lane];
  h = h * dA + (dt * xv) * bv;
  st_ptr[lane] = h;
  float4 prod = h * cv;
  float yv = simd_sum(prod.x + prod.y + prod.z + prod.w) + d_skip[head] * xv;
  if (lane == 0u) y_out[(ulong)r * NEMO_MAMBA_INNER + x_col] = yv;
}

// Group-wise gated RMSNorm matching NemotronHRMSNormGated (8 groups of 512,
// norm_before_gate = False): y = y * silu(gate), group RMSNorm, * weight.
// Grid: x = 8 (groups), y = rows; threadgroup = 256 threads.
kernel void nemo_mamba_norm_gated(device const float *y_in [[buffer(0)]],
                                  device const float *zxbcdt [[buffer(1)]],
                                  device const float *norm_w [[buffer(2)]],
                                  device float *out [[buffer(3)]],
                                  constant NemoMambaParam &p [[buffer(4)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_SIMD];
  uint nt = tpg.x;
  uint g = tg.x, r = tg.y;
  uint base = g * NEMO_MAMBA_NORM_GROUP;
  device const float *yr = y_in + (ulong)r * NEMO_MAMBA_INNER + base;
  device const float *zr = zxbcdt + (ulong)r * NEMO_MAMBA_IN_DIM + base;
  device const float *wr = norm_w + base;
  device float *dst = out + (ulong)r * NEMO_MAMBA_INNER + base;

  float sq = 0.0f;
  for (uint i = t; i < NEMO_MAMBA_NORM_GROUP; i += nt) {
    float v = yr[i] * nemo_silu(zr[i]);
    sq += v * v;
  }
  float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(NEMO_MAMBA_NORM_GROUP) + p.eps);
  for (uint i = t; i < NEMO_MAMBA_NORM_GROUP; i += nt) {
    float v = yr[i] * nemo_silu(zr[i]);
    dst[i] = bf(wr[i] * bf(v * inv));
  }
}

// ---- NoPE GQA 32/2 Attention (head_dim = 128, no RoPE) ----

// Repacks Q [rows, 4096] into head-major BF16 q_hm [32, q_stride, 128] (zero-
// padded tail) and stores BF16 K/V [rows, 256] into separate per-KV-head cache
// buffers (k0, k1, v0, v1) at pos0 + row or per-row NemoKvTarget positions.
// Grid: x = 36 (32 Q + 2 K + 2 V), y = q_stride; threadgroup = 128 threads.
kernel void nemo_qkv_store(device const float *q_raw [[buffer(0)]],
                           device const float *k_raw [[buffer(1)]],
                           device const float *v_raw [[buffer(2)]],
                           device bfloat *q_hm [[buffer(3)]],
                           device bfloat *k0 [[buffer(4)]],
                           device bfloat *k1 [[buffer(5)]],
                           device bfloat *v0 [[buffer(6)]],
                           device bfloat *v1 [[buffer(7)]],
                           device const NemoKvTarget *targets [[buffer(8)]],
                           constant NemoQkvParam &p [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint d [[thread_index_in_threadgroup]]) {
  uint slot = tg.x, row = tg.y;
  if (slot < NEMO_HEADS) {
    ulong dst = ((ulong)slot * p.q_stride + row) * NEMO_HEAD_DIM + d;
    q_hm[dst] = row < p.rows
                    ? bfloat(q_raw[(ulong)row * NEMO_Q_DIM + slot * NEMO_HEAD_DIM + d])
                    : bfloat(0.0f);
    return;
  }
  if (row >= p.rows) return;
  uint idx = slot - NEMO_HEADS;
  uint is_v = idx >= NEMO_KV_HEADS ? 1u : 0u;
  uint kvh = idx & 1u;
  float val = is_v ? v_raw[(ulong)row * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM + d]
                   : k_raw[(ulong)row * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM + d];
  device bfloat *dst_buf;
  uint pos;
  if (p.batched) {
    NemoKvTarget kt = targets[row];
    pos = kt.pos;
    dst_buf = is_v ? (kvh ? kt.v1 : kt.v0) : (kvh ? kt.k1 : kt.k0);
  } else {
    pos = p.pos0 + row;
    dst_buf = is_v ? (kvh ? v1 : v0) : (kvh ? k1 : k0);
  }
  dst_buf[(ulong)pos * NEMO_HEAD_DIM + d] = bfloat(val);
}

// Causal cooperative 64x64 MPP FlashAttention for NoPE GQA 32/2 prefill.
// Grid: x = ceil(rows / 64), y = 32 (Q heads); threadgroup = 128 threads (4 sgs).
kernel void nemo_attend_prefill(device const bfloat *q_hm [[buffer(0)]],
                                device const bfloat *k0 [[buffer(1)]],
                                device const bfloat *k1 [[buffer(2)]],
                                device const bfloat *v0 [[buffer(3)]],
                                device const bfloat *v1 [[buffer(4)]],
                                device float *out [[buffer(5)]],
                                constant NemoAttnPrefillParam &p [[buffer(6)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint q0 = tg.x * NEMO_TILE;
  uint h = tg.y;
  if (q0 >= p.rows) return;
  uint kvh = h / NEMO_GQA;
  device const bfloat *k_buf = kvh ? k1 : k0;
  device const bfloat *v_buf = kvh ? v1 : v0;

  threadgroup FlashAttn64Smem<bfloat> smem;
  constexpr auto qk_desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_TILE, NEMO_HEAD_DIM, false, true, false,
      matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_HEAD_DIM, NEMO_TILE, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<NEMO_MM_SGS>> qk_op;
  matmul2d<pv_desc, execution_simdgroups<NEMO_MM_SGS>> pv_op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)(q_hm + (ulong)h * p.q_stride * NEMO_HEAD_DIM),
      dextents<int32_t, 2>(NEMO_HEAD_DIM, p.q_stride));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk(
      (device bfloat *)k_buf, dextents<int32_t, 2>(NEMO_HEAD_DIM, p.kv_cap));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv(
      (device bfloat *)v_buf, dextents<int32_t, 2>(NEMO_HEAD_DIM, p.kv_cap));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_TILE, NEMO_TILE>,
         tensor_inline>
      tp(smem.st.probs, extents<int32_t, NEMO_TILE, NEMO_TILE>());

  auto q_tile = tq.slice(0, q0);
  auto acc = pv_op.get_destination_cooperative_tensor<decltype(tp),
                                                      decltype(tv.slice(0, 0)),
                                                      float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  uint kv_end = min(p.pos0 + q0 + NEMO_TILE, p.pos0 + p.rows);
  float rmax = -INFINITY, rsum = 0.0f;
  for (uint kv0 = 0; kv0 < kv_end; kv0 += NEMO_TILE) {
    bool has_mask = (kv0 + NEMO_TILE) > (p.pos0 + q0);
    auto k_tile = tk.slice(0, kv0);
    auto s = qk_op.get_destination_cooperative_tensor<decltype(q_tile),
                                                      decltype(k_tile), float>();
    qk_op.run(q_tile, k_tile, s);
    for (uint16_t i = 0; i < s.get_capacity(); i++) {
      if (s.is_valid_element(i)) {
        auto at = s.get_multidimensional_index(i);
        uint col = at[0], row = at[1];
        float v = s[i] * NEMO_ATTN_LOG2_SCALE;
        if (has_mask && (kv0 + col > p.pos0 + q0 + row || kv0 + col >= p.pos0 + p.rows))
          v = -INFINITY;
        smem.scores[row * NEMO_TILE + col] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint changed = flash_attn_64x64_softmax<bfloat>(smem, sg, lane, has_mask,
                                                    rmax, rsum);
    if (changed) {
      for (uint16_t i = 0; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
          auto at = acc.get_multidimensional_index(i);
          acc[i] *= smem.st.row_scale[at[1]];
        }
      }
    }
    auto v_tile = tv.slice(0, kv0);
    pv_op.run(tp, v_tile, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint my_row = sg * FLASH_ATTN_64_ROWS_PER_SG + (lane >> 1u);
  if ((lane & 1u) == 0u)
    smem.st.row_scale[my_row] = rsum > 0.0f ? 1.0f / rsum : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint col = at[0], row = q0 + at[1];
      if (row < p.rows) {
        out[(ulong)row * NEMO_Q_DIM + (ulong)h * NEMO_HEAD_DIM + col] =
            bf(acc[i] * smem.st.row_scale[at[1]]);
      }
    }
  }
}

// FlashDecoding partition stage for NoPE GQA 32/2 decode: 16 simdgroups (512
// threads) share one KV head read across all 16 Q heads in its group.
// Grid: x = 2 (KV heads), y = nparts, z = rows; threadgroup = 512 threads.
kernel void nemo_attend_part(device const bfloat *q_hm [[buffer(0)]],
                             device const bfloat *k0 [[buffer(1)]],
                             device const bfloat *k1 [[buffer(2)]],
                             device const bfloat *v0 [[buffer(3)]],
                             device const bfloat *v1 [[buffer(4)]],
                             device const NemoKvTarget *targets [[buffer(5)]],
                             device float *part_out [[buffer(6)]],
                             constant NemoAttnDecodeParam &p [[buffer(7)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  uint kvh = tg.x, part = tg.y, row = tg.z;
  uint h = kvh * NEMO_GQA + sg;
  uint pos;
  device const bfloat *k_buf;
  device const bfloat *v_buf;
  if (p.batched) {
    NemoKvTarget kt = targets[row];
    pos = kt.pos;
    k_buf = kvh ? kt.k1 : kt.k0;
    v_buf = kvh ? kt.v1 : kt.v0;
  } else {
    pos = p.pos0 + row;
    k_buf = kvh ? k1 : k0;
    v_buf = kvh ? v1 : v0;
  }

  uint total_k = pos + 1u;
  uint chunk = (total_k + p.nparts - 1u) / p.nparts;
  uint t0 = part * chunk;
  uint t1 = min(total_k, t0 + chunk);

  device float *dst =
      part_out + (((ulong)row * NEMO_HEADS + h) * p.part_stride + part) *
                     NEMO_DECODE_PART_FLOATS;
  if (t0 >= t1) {
    if (lane == 0u) {
      dst[NEMO_HEAD_DIM] = -INFINITY;
      dst[NEMO_HEAD_DIM + 1u] = 0.0f;
    }
    return;
  }

  device const bfloat4 *q4 =
      (device const bfloat4 *)(q_hm + ((ulong)h * p.q_stride + row) * NEMO_HEAD_DIM);
  float4 qv = float4(q4[lane]) * NEMO_ATTN_SCALE;
  float4 acc = float4(0.0f);
  float rmax = -INFINITY, rsum = 0.0f;

  device const bfloat4 *k4 = (device const bfloat4 *)k_buf;
  device const bfloat4 *v4 = (device const bfloat4 *)v_buf;
  for (uint t = t0; t < t1; t++) {
    float4 kv = float4(k4[(ulong)t * NEMO_SIMD + lane]);
    float score = simd_sum(dot(qv, kv));
    float m_new = max(rmax, score);
    float a = exp(rmax - m_new);
    float b = exp(score - m_new);
    acc = acc * a + float4(v4[(ulong)t * NEMO_SIMD + lane]) * b;
    rsum = rsum * a + b;
    rmax = m_new;
  }

  ((device float4 *)dst)[lane] = acc;
  if (lane == 0u) {
    dst[NEMO_HEAD_DIM] = rmax;
    dst[NEMO_HEAD_DIM + 1u] = rsum;
  }
}

// Combines FlashDecoding partitions into BF16-rounded token-major attn_out
// [rows, 4096]. Grid: x = 32 (Q heads), y = rows; threadgroup = 32 threads.
kernel void nemo_attend_combine(device const float *part_in [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant NemoAttnDecodeParam &p [[buffer(2)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint h = tg.x, row = tg.y;
  device const float *base =
      part_in + ((ulong)row * NEMO_HEADS + h) * p.part_stride * NEMO_DECODE_PART_FLOATS;
  float rmax = -INFINITY;
  for (uint pt = 0; pt < p.nparts; pt++) {
    rmax = max(rmax, base[(ulong)pt * NEMO_DECODE_PART_FLOATS + NEMO_HEAD_DIM]);
  }
  float4 acc = float4(0.0f);
  float rsum = 0.0f;
  for (uint pt = 0; pt < p.nparts; pt++) {
    device const float *pr = base + (ulong)pt * NEMO_DECODE_PART_FLOATS;
    float pm = pr[NEMO_HEAD_DIM];
    float pl = pr[NEMO_HEAD_DIM + 1u];
    if (pl > 0.0f) {
      float s = exp(pm - rmax);
      acc += ((device const float4 *)pr)[lane] * s;
      rsum += pl * s;
    }
  }
  float inv = rsum > 0.0f ? 1.0f / rsum : 0.0f;
  float4 v = acc * inv;
  device float *dst = out + (ulong)row * NEMO_Q_DIM + (ulong)h * NEMO_HEAD_DIM + 4u * lane;
  dst[0] = bf(v.x);
  dst[1] = bf(v.y);
  dst[2] = bf(v.z);
  dst[3] = bf(v.w);
}

// ---- Routed MoE (top-6 sigmoid+bias, NVFP4 non-gated relu2 experts) ----

// NemotronHTopkRouter: sigmoid scores, + e_score_correction_bias for top-6
// selection, normalized un-biased scores multiplied by routed_scaling_factor
// (2.5). Grid: rows threadgroups of 128 threads.
kernel void nemo_route_topk(device const float *logits [[buffer(0)]],
                            device const float *bias [[buffer(1)]],
                            device uint *out_ids [[buffer(2)]],
                            device float *out_weights [[buffer(3)]],
                            constant NemoRouteParam &p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  threadgroup float vals[NEMO_ROUTE_SGS * NEMO_TOPK];
  threadgroup uint idxs[NEMO_ROUTE_SGS * NEMO_TOPK];
  threadgroup float sel_dummy[NEMO_TOPK];
  threadgroup float scores[NEMO_EXPERTS];
  if (row >= p.rows || t >= p.experts) return;
  float sc = sigm(logits[(ulong)row * p.experts + t]);
  scores[t] = sc;
  float key = sc + bias[t];
  device uint *row_ids = out_ids + (ulong)row * p.topk;
  group_topk(key, t, p.topk, vals, idxs, row_ids, sel_dummy, t, p.experts);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float sum = 0.0f;
    for (uint k = 0; k < p.topk; k++) sum += scores[row_ids[k]];
    float inv = p.scale / (sum + NEMO_ROUTE_EPS);
    device float *row_w = out_weights + (ulong)row * p.topk;
    for (uint k = 0; k < p.topk; k++) row_w[k] = scores[row_ids[k]] * inv;
  }
}

// Groups decode pairs [rows * topk] by expert ID on GPU when all 128 experts
// are resident, emitting up to rows * topk tiles (unused slots have count = 0).
// Grid: 1 threadgroup of 128 threads.
kernel void nemo_moe_group(device const uint *route_ids [[buffer(0)]],
                           device const float *route_w [[buffer(1)]],
                           device int *ptok [[buffer(2)]],
                           device float *pw [[buffer(3)]],
                           device int *tp [[buffer(4)]],
                           device NemoMoeTile *tiles [[buffer(5)]],
                           constant uint2 &mt [[buffer(6)]],
                           uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint counts[NEMO_EXPERTS];
  threadgroup uint offsets[NEMO_EXPERTS];
  uint rows = mt.x, topk = mt.y, npairs = rows * topk;
  counts[t] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (t == 0u) {
    for (uint p = 0; p < npairs; p++) counts[route_ids[p]]++;
    uint off = 0u, nt = 0u;
    for (uint e = 0; e < NEMO_EXPERTS; e++) {
      offsets[e] = off;
      uint cnt = counts[e];
      if (cnt > 0u) {
        tiles[nt] = {e, off, cnt, 0u};
        nt++;
      }
      off += cnt;
    }
    for (uint ti = nt; ti < npairs; ti++) tiles[ti] = {0u, 0u, 0u, 0u};
    for (uint e = 0; e < NEMO_EXPERTS; e++) counts[e] = offsets[e];
    for (uint r = 0; r < rows; r++) {
      for (uint k = 0; k < topk; k++) {
        uint src = r * topk + k;
        uint e = route_ids[src];
        uint dst = counts[e]++;
        ptok[dst] = int(r);
        pw[dst] = route_w[src];
        tp[src] = int(dst);
      }
    }
  }
}

// Gathers FP32 token rows into BF16 pair order for MPP MoE matmul tiles.
// Grid: x = dim / 4, y = npairs.
kernel void nemo_moe_gather_bf16(device const float4 *x [[buffer(0)]],
                                 device const int *ptok [[buffer(1)]],
                                 device bfloat4 *out [[buffer(2)]],
                                 constant uint &dim4 [[buffer(3)]],
                                 uint2 i [[thread_position_in_grid]]) {
  int tok = ptok[i.y];
  out[(ulong)i.y * dim4 + i.x] = bfloat4(x[(ulong)uint(tok) * dim4 + i.x]);
}

// Multi-vec NVFP4 routed expert up_proj + relu2 for tiles with count <= 8.
// Uses 8-byte-aligned uint2 loads (safe for ExpertStore slots with off&15==8).
// Grid: x = ceil(n / 32), y = ntiles; threadgroup = 256 threads (8 sgs).
kernel void nemo_moe_nvfp4_up_mv(device const NemoExpertRef *experts [[buffer(0)]],
                                 device const float *factor_up [[buffer(1)]],
                                 device const float *x [[buffer(2)]],
                                 device const int *ptok [[buffer(3)]],
                                 device const NemoMoeTile *tiles [[buffer(4)]],
                                 device bfloat *h_out [[buffer(5)]],
                                 constant NemoMoeParam &p [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count == 0u || tl.count > NEMO_MV_MAX_M) return;
  uint row = (tg.x * NEMO_MOE_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / NEMO_NVFP4_BLOCK;
  uint w_row_bytes = p.k / 2u;
  NemoExpertRef er = experts[tl.expert];
  device const uchar *we = er.w_up + (ulong)row * w_row_bytes;
  device const uchar *se = er.s_up + (ulong)row * blocks;
  float ef = factor_up[tl.expert];

#define NEMO_MOE_UP_EMIT(j, av)                                                \
  {                                                                            \
    float4 s = float4(simd_sum(av.x), simd_sum(av.y), simd_sum(av.z),          \
                      simd_sum(av.w)) *                                        \
               ef;                                                             \
    if (lane < NEMO_MV_ROWS && row + lane < p.n) {                             \
      float v = bf(s[lane]);                                                   \
      h_out[(ulong)(tl.start + (j)) * p.n + row + lane] = bfloat(nemo_relu2(v)); \
    }                                                                          \
  }

  if (tl.count == 1u) {
    float4 acc = float4(0.0f);
    device const float4 *xbase = (device const float4 *)(x + (ulong)uint(ptok[tl.start]) * p.k);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 x0 = xbase[4u * c], x1 = xbase[4u * c + 1u], x2 = xbase[4u * c + 2u],
             x3 = xbase[4u * c + 3u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
        float4 prod = a * x0 + b * x1 + d0 * x2 + d1 * x3;
        acc[r] += (prod.x + prod.y + prod.z + prod.w) * sc;
      }
    }
    NEMO_MOE_UP_EMIT(0u, acc);
    return;
  }

  if (tl.count == 2u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    device const float4 *xb0 = (device const float4 *)(x + (ulong)uint(ptok[tl.start]) * p.k);
    device const float4 *xb1 = (device const float4 *)(x + (ulong)uint(ptok[tl.start + 1u]) * p.k);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 x00 = xb0[4u * c], x01 = xb0[4u * c + 1u], x02 = xb0[4u * c + 2u],
             x03 = xb0[4u * c + 3u];
      float4 x10 = xb1[4u * c], x11 = xb1[4u * c + 1u], x12 = xb1[4u * c + 2u],
             x13 = xb1[4u * c + 3u];
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
        float4 p0 = a * x00 + b * x01 + d0 * x02 + d1 * x03;
        float4 p1 = a * x10 + b * x11 + d0 * x12 + d1 * x13;
        acc0[r] += (p0.x + p0.y + p0.z + p0.w) * sc;
        acc1[r] += (p1.x + p1.y + p1.z + p1.w) * sc;
      }
    }
    NEMO_MOE_UP_EMIT(0u, acc0);
    NEMO_MOE_UP_EMIT(1u, acc1);
    return;
  }

  if (tl.count <= 4u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
           acc3 = float4(0.0f);
    uint t0 = uint(ptok[tl.start]), t1 = uint(ptok[tl.start + 1u]),
         t2 = uint(ptok[tl.start + 2u]),
         t3 = tl.count > 3u ? uint(ptok[tl.start + 3u]) : 0u;
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
#define NEMO_MOE_UP_DOT(tk, av)                                                \
  {                                                                            \
    device const float4 *xr = (device const float4 *)(x + (ulong)(tk) * p.k) + 4u * c; \
    float4 prod = a * xr[0] + b * xr[1] + d0 * xr[2] + d1 * xr[3];             \
    av[r] += (prod.x + prod.y + prod.z + prod.w) * sc;                         \
  }
        NEMO_MOE_UP_DOT(t0, acc0);
        NEMO_MOE_UP_DOT(t1, acc1);
        NEMO_MOE_UP_DOT(t2, acc2);
        if (tl.count > 3u) NEMO_MOE_UP_DOT(t3, acc3);
      }
    }
    NEMO_MOE_UP_EMIT(0u, acc0);
    NEMO_MOE_UP_EMIT(1u, acc1);
    NEMO_MOE_UP_EMIT(2u, acc2);
    if (tl.count > 3u) NEMO_MOE_UP_EMIT(3u, acc3);
    return;
  }

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
         acc3 = float4(0.0f), acc4 = float4(0.0f), acc5 = float4(0.0f),
         acc6 = float4(0.0f), acc7 = float4(0.0f);
  uint t0 = uint(ptok[tl.start]), t1 = uint(ptok[tl.start + 1u]),
       t2 = uint(ptok[tl.start + 2u]), t3 = uint(ptok[tl.start + 3u]),
       t4 = uint(ptok[tl.start + 4u]),
       t5 = tl.count > 5u ? uint(ptok[tl.start + 5u]) : 0u,
       t6 = tl.count > 6u ? uint(ptok[tl.start + 6u]) : 0u,
       t7 = tl.count > 7u ? uint(ptok[tl.start + 7u]) : 0u;
  for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
    for (uint r = 0; r < NEMO_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
      float sc = fp8(se[(ulong)r * blocks + c]);
      float4 a, b, d0, d1;
      fp4x8(q.x, a, b);
      fp4x8(q.y, d0, d1);
      NEMO_MOE_UP_DOT(t0, acc0);
      NEMO_MOE_UP_DOT(t1, acc1);
      NEMO_MOE_UP_DOT(t2, acc2);
      NEMO_MOE_UP_DOT(t3, acc3);
      NEMO_MOE_UP_DOT(t4, acc4);
      if (tl.count > 5u) NEMO_MOE_UP_DOT(t5, acc5);
      if (tl.count > 6u) NEMO_MOE_UP_DOT(t6, acc6);
      if (tl.count > 7u) NEMO_MOE_UP_DOT(t7, acc7);
    }
  }
  NEMO_MOE_UP_EMIT(0u, acc0);
  NEMO_MOE_UP_EMIT(1u, acc1);
  NEMO_MOE_UP_EMIT(2u, acc2);
  NEMO_MOE_UP_EMIT(3u, acc3);
  NEMO_MOE_UP_EMIT(4u, acc4);
  if (tl.count > 5u) NEMO_MOE_UP_EMIT(5u, acc5);
  if (tl.count > 6u) NEMO_MOE_UP_EMIT(6u, acc6);
  if (tl.count > 7u) NEMO_MOE_UP_EMIT(7u, acc7);
#undef NEMO_MOE_UP_DOT
#undef NEMO_MOE_UP_EMIT
}

// Multi-vec NVFP4 routed expert down_proj for tiles with count <= 8, reading
// BF16 intermediate activations and scaling outputs by routing weight pw.
// Grid: x = ceil(n / 32), y = ntiles; threadgroup = 256 threads (8 sgs).
kernel void nemo_moe_nvfp4_down_mv(device const NemoExpertRef *experts [[buffer(0)]],
                                   device const float *factor_down [[buffer(1)]],
                                   device const bfloat *h [[buffer(2)]],
                                   device const float *pw [[buffer(3)]],
                                   device const NemoMoeTile *tiles [[buffer(4)]],
                                   device float *y_out [[buffer(5)]],
                                   constant NemoMoeParam &p [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count == 0u || tl.count > NEMO_MV_MAX_M) return;
  uint row = (tg.x * NEMO_MOE_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / NEMO_NVFP4_BLOCK;
  uint w_row_bytes = p.k / 2u;
  NemoExpertRef er = experts[tl.expert];
  device const uchar *we = er.w_down + (ulong)row * w_row_bytes;
  device const uchar *se = er.s_down + (ulong)row * blocks;
  float ef = factor_down[tl.expert];

#define NEMO_MOE_DN_EMIT(j, av)                                                \
  {                                                                            \
    float4 s = float4(simd_sum(av.x), simd_sum(av.y), simd_sum(av.z),          \
                      simd_sum(av.w)) *                                        \
               ef;                                                             \
    float rw = pw[tl.start + (j)];                                             \
    if (lane < NEMO_MV_ROWS && row + lane < p.n)                               \
      y_out[(ulong)(tl.start + (j)) * p.n + row + lane] = bf(s[lane]) * rw;    \
  }

  if (tl.count == 1u) {
    float4 acc = float4(0.0f);
    device const bfloat4 *hbase = (device const bfloat4 *)(h + (ulong)tl.start * p.k);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 x0 = float4(hbase[4u * c]), x1 = float4(hbase[4u * c + 1u]),
             x2 = float4(hbase[4u * c + 2u]), x3 = float4(hbase[4u * c + 3u]);
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
        float4 prod = a * x0 + b * x1 + d0 * x2 + d1 * x3;
        acc[r] += (prod.x + prod.y + prod.z + prod.w) * sc;
      }
    }
    NEMO_MOE_DN_EMIT(0u, acc);
    return;
  }

  if (tl.count == 2u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
    device const bfloat4 *hb0 = (device const bfloat4 *)(h + (ulong)tl.start * p.k);
    device const bfloat4 *hb1 = (device const bfloat4 *)(h + (ulong)(tl.start + 1u) * p.k);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
      float4 x00 = float4(hb0[4u * c]), x01 = float4(hb0[4u * c + 1u]),
             x02 = float4(hb0[4u * c + 2u]), x03 = float4(hb0[4u * c + 3u]);
      float4 x10 = float4(hb1[4u * c]), x11 = float4(hb1[4u * c + 1u]),
             x12 = float4(hb1[4u * c + 2u]), x13 = float4(hb1[4u * c + 3u]);
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
        float4 p0 = a * x00 + b * x01 + d0 * x02 + d1 * x03;
        float4 p1 = a * x10 + b * x11 + d0 * x12 + d1 * x13;
        acc0[r] += (p0.x + p0.y + p0.z + p0.w) * sc;
        acc1[r] += (p1.x + p1.y + p1.z + p1.w) * sc;
      }
    }
    NEMO_MOE_DN_EMIT(0u, acc0);
    NEMO_MOE_DN_EMIT(1u, acc1);
    return;
  }

  if (tl.count <= 4u) {
    float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
           acc3 = float4(0.0f);
    for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
      for (uint r = 0; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
        float sc = fp8(se[(ulong)r * blocks + c]);
        float4 a, b, d0, d1;
        fp4x8(q.x, a, b);
        fp4x8(q.y, d0, d1);
#define NEMO_MOE_DN_DOT(j, av)                                                 \
  {                                                                            \
    device const bfloat4 *hr =                                                 \
        (device const bfloat4 *)(h + (ulong)(tl.start + (j)) * p.k) + 4u * c;  \
    float4 prod = a * float4(hr[0]) + b * float4(hr[1]) + d0 * float4(hr[2]) + \
                  d1 * float4(hr[3]);                                          \
    av[r] += (prod.x + prod.y + prod.z + prod.w) * sc;                         \
  }
        NEMO_MOE_DN_DOT(0u, acc0);
        NEMO_MOE_DN_DOT(1u, acc1);
        NEMO_MOE_DN_DOT(2u, acc2);
        if (tl.count > 3u) NEMO_MOE_DN_DOT(3u, acc3);
      }
    }
    NEMO_MOE_DN_EMIT(0u, acc0);
    NEMO_MOE_DN_EMIT(1u, acc1);
    NEMO_MOE_DN_EMIT(2u, acc2);
    if (tl.count > 3u) NEMO_MOE_DN_EMIT(3u, acc3);
    return;
  }

  float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f),
         acc3 = float4(0.0f), acc4 = float4(0.0f), acc5 = float4(0.0f),
         acc6 = float4(0.0f), acc7 = float4(0.0f);
  for (uint c = lane; c < blocks; c += NEMO_SIMD) {
#pragma unroll
    for (uint r = 0; r < NEMO_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
      float sc = fp8(se[(ulong)r * blocks + c]);
      float4 a, b, d0, d1;
      fp4x8(q.x, a, b);
      fp4x8(q.y, d0, d1);
      NEMO_MOE_DN_DOT(0u, acc0);
      NEMO_MOE_DN_DOT(1u, acc1);
      NEMO_MOE_DN_DOT(2u, acc2);
      NEMO_MOE_DN_DOT(3u, acc3);
      NEMO_MOE_DN_DOT(4u, acc4);
      if (tl.count > 5u) NEMO_MOE_DN_DOT(5u, acc5);
      if (tl.count > 6u) NEMO_MOE_DN_DOT(6u, acc6);
      if (tl.count > 7u) NEMO_MOE_DN_DOT(7u, acc7);
    }
  }
  NEMO_MOE_DN_EMIT(0u, acc0);
  NEMO_MOE_DN_EMIT(1u, acc1);
  NEMO_MOE_DN_EMIT(2u, acc2);
  NEMO_MOE_DN_EMIT(3u, acc3);
  NEMO_MOE_DN_EMIT(4u, acc4);
  if (tl.count > 5u) NEMO_MOE_DN_EMIT(5u, acc5);
  if (tl.count > 6u) NEMO_MOE_DN_EMIT(6u, acc6);
  if (tl.count > 7u) NEMO_MOE_DN_EMIT(7u, acc7);
#undef NEMO_MOE_DN_DOT
#undef NEMO_MOE_DN_EMIT
}

// Tiled MPP NVFP4 routed expert up_proj + relu2 for tiles with count > 8.
// Grid: x = n / 64, y = ntiles; threadgroup = 128 threads (4 simdgroups).
kernel void nemo_moe_nvfp4_up_mm(device const NemoExpertRef *experts [[buffer(0)]],
                                 device const float *factor_up [[buffer(1)]],
                                 device const bfloat *x_bf16 [[buffer(2)]],
                                 device const NemoMoeTile *tiles [[buffer(3)]],
                                 device bfloat *h_out [[buffer(4)]],
                                 constant NemoMoeParam &p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count <= NEMO_MV_MAX_M) return;

  threadgroup bfloat ws[NEMO_TILE * NEMO_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_TILE, NEMO_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_MM_SGS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, p.npairs + NEMO_TILE));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_TILE, NEMO_TILE>,
         tensor_inline>
      tw(ws, extents<int32_t, NEMO_TILE, NEMO_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)),
                                                 decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * NEMO_TILE;
  NemoExpertRef er = experts[tl.expert];
  float ef = factor_up[tl.expert];

  for (uint k0 = 0; k0 < p.k; k0 += NEMO_TILE) {
    nemo_stage_nvfp4_tile(er.w_up, er.s_up, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = tx.slice(k0, tl.start);
    op.run(a, tw, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c.get_capacity(); i++) {
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint col = n0 + at[0], r = at[1];
      if (r < tl.count && col < p.n) {
        float v = bf(c[i] * ef);
        h_out[(ulong)(tl.start + r) * p.n + col] = bfloat(nemo_relu2(v));
      }
    }
  }
}

// Tiled MPP NVFP4 routed expert down_proj for tiles with count > 8, reading
// BF16 intermediate activations and scaling outputs by routing weight pw.
// Grid: x = n / 64, y = ntiles; threadgroup = 128 threads (4 simdgroups).
kernel void nemo_moe_nvfp4_down_mm(device const NemoExpertRef *experts [[buffer(0)]],
                                   device const float *factor_down [[buffer(1)]],
                                   device const bfloat *h [[buffer(2)]],
                                   device const float *pw [[buffer(3)]],
                                   device const NemoMoeTile *tiles [[buffer(4)]],
                                   device float *y_out [[buffer(5)]],
                                   constant NemoMoeParam &p [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count <= NEMO_MV_MAX_M) return;

  threadgroup bfloat ws[NEMO_TILE * NEMO_TILE];
  constexpr auto desc = matmul2d_descriptor(
      NEMO_TILE, NEMO_TILE, NEMO_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NEMO_MM_SGS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> th(
      (device bfloat *)h, dextents<int32_t, 2>(p.k, p.npairs + NEMO_TILE));
  tensor<threadgroup bfloat, extents<int32_t, NEMO_TILE, NEMO_TILE>,
         tensor_inline>
      tw(ws, extents<int32_t, NEMO_TILE, NEMO_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(th.slice(0, 0)),
                                                 decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * NEMO_TILE;
  NemoExpertRef er = experts[tl.expert];
  float ef = factor_down[tl.expert];

  for (uint k0 = 0; k0 < p.k; k0 += NEMO_TILE) {
    nemo_stage_nvfp4_tile(er.w_down, er.s_down, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = th.slice(k0, tl.start);
    op.run(a, tw, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < c.get_capacity(); i++) {
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint col = n0 + at[0], r = at[1];
      if (r < tl.count && col < p.n) {
        y_out[(ulong)(tl.start + r) * p.n + col] =
            bf(c[i] * ef) * pw[tl.start + r];
      }
    }
  }
}

// Sums top-6 weighted routed expert outputs in FP32, rounds to BF16, adds the
// shared expert output, and adds the residual into cur (BF16-rounded).
// Grid: x = dim (2688), y = rows.
kernel void nemo_moe_combine_add(device const float *y_pair [[buffer(0)]],
                                 device const int *tp [[buffer(1)]],
                                 device const float *shared_out [[buffer(2)]],
                                 device float *cur [[buffer(3)]],
                                 constant NemoMoeCombineParam &p [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  uint col = gid.x, row = gid.y;
  if (col >= p.dim || row >= p.rows) return;
  float sum = 0.0f;
  ulong tp_base = (ulong)row * p.topk;
  for (uint k = 0; k < p.topk; k++) {
    int pair_idx = tp[tp_base + k];
    sum += y_pair[(ulong)uint(pair_idx) * p.dim + col];
  }
  ulong idx = (ulong)row * p.dim + col;
  float moe = bf(bf(sum) + shared_out[idx]);
  cur[idx] = bf(cur[idx] + moe);
}

// ---- Multi-Token Prediction (MTP) draft kernels ----

// Stores BF16 K and V rows [rows, 256] into separate per-KV-head cache buffers
// without touching Q. In batched mode, skips rows whose target position equals
// NEMO_KV_SKIP_POS. Grid: x = 4 (2 K + 2 V), y = rows; TG = 128 threads.
kernel void nemo_kv_store(device const float *k_raw [[buffer(0)]],
                          device const float *v_raw [[buffer(1)]],
                          device bfloat *k0 [[buffer(2)]],
                          device bfloat *k1 [[buffer(3)]],
                          device bfloat *v0 [[buffer(4)]],
                          device bfloat *v1 [[buffer(5)]],
                          device const NemoKvTarget *targets [[buffer(6)]],
                          constant NemoQkvParam &p [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint d [[thread_index_in_threadgroup]]) {
  uint slot = tg.x, row = tg.y;
  if (slot >= 2u * NEMO_KV_HEADS || row >= p.rows) return;
  uint is_v = slot >= NEMO_KV_HEADS ? 1u : 0u;
  uint kvh = slot & 1u;
  float val = is_v ? v_raw[(ulong)row * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM + d]
                   : k_raw[(ulong)row * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM + d];
  device bfloat *dst_buf;
  uint pos;
  if (p.batched) {
    NemoKvTarget kt = targets[row];
    if (kt.pos == NEMO_KV_SKIP_POS) return;
    pos = kt.pos;
    dst_buf = is_v ? (kvh ? kt.v1 : kt.v0) : (kvh ? kt.k1 : kt.k0);
  } else {
    pos = p.pos0 + row;
    dst_buf = is_v ? (kvh ? v1 : v0) : (kvh ? k1 : k0);
  }
  dst_buf[(ulong)pos * NEMO_HEAD_DIM + d] = bfloat(val);
}

// Applies enorm to token embeddings and hnorm to preceding hidden states, then
// concatenates [enorm(emb), hnorm(h)] into eh_out [rows, 2 * dim] (BF16).
// When use_h0 is nonzero, row 0 reads h0 and rows r >= 1 read h_seq[r - 1].
// Grid: rows threadgroups of 256 threads.
kernel void nemo_mtp_eh_cat(device const float *emb [[buffer(0)]],
                            device const float *h0 [[buffer(1)]],
                            device const float *h_seq [[buffer(2)]],
                            device const float *enorm_w [[buffer(3)]],
                            device const float *hnorm_w [[buffer(4)]],
                            device float *eh_out [[buffer(5)]],
                            constant NemoMtpEhParam &p [[buffer(6)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[NEMO_SIMD];
  if (row >= p.rows) return;
  device const float *er = emb + (ulong)row * p.dim;
  device const float *hr =
      (p.use_h0 && row == 0u)
          ? h0
          : h_seq + (ulong)(p.use_h0 ? row - 1u : row) * p.dim;
  device float *dst = eh_out + (ulong)row * (2u * p.dim);

  float sq_e = 0.0f, sq_h = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    sq_e += er[i] * er[i];
    sq_h += hr[i] * hr[i];
  }
  float inv_e = rsqrt(tg_sum(sq_e, tmp, t, nt) / float(p.dim) + p.eps);
  float inv_h = rsqrt(tg_sum(sq_h, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt) {
    dst[i] = bf(enorm_w[i] * (er[i] * inv_e));
    dst[p.dim + i] = bf(hnorm_w[i] * (hr[i] * inv_h));
  }
}

// Multi-vec per-tensor FP8 routed expert up_proj + relu2 for MTP tiles
// (count <= 8). Uses 8-byte-aligned uint2 weight loads.
// Grid: x = ceil(n / 32), y = ntiles; threadgroup = 256 threads (8 sgs).
kernel void nemo_moe_fp8_up_mv(device const NemoFp8ExpertRef *experts [[buffer(0)]],
                               device const float *x [[buffer(1)]],
                               device const int *ptok [[buffer(2)]],
                               device const NemoMoeTile *tiles [[buffer(3)]],
                               device bfloat *h_out [[buffer(4)]],
                               constant NemoMoeParam &p [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count == 0u || tl.count > NEMO_MV_MAX_M) return;
  uint row = (tg.x * NEMO_MOE_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;

  uint chunks = p.k / 8u;
  NemoFp8ExpertRef er = experts[tl.expert];
  device const uchar *we = er.w_up + (ulong)row * p.k;

  for (uint j = 0u; j < tl.count; j++) {
    float4 acc = float4(0.0f);
    device const float4 *xr =
        (device const float4 *)(x + (ulong)uint(ptok[tl.start + j]) * p.k);
    for (uint c = lane; c < chunks; c += NEMO_SIMD) {
      float4 x0 = xr[2u * c], x1 = xr[2u * c + 1u];
#pragma unroll
      for (uint r = 0u; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
        float4 prod = fp8x4(q.x) * x0 + fp8x4(q.y) * x1;
        acc[r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
    float4 s = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z),
                      simd_sum(acc.w)) *
               er.s_up;
    if (lane < NEMO_MV_ROWS && row + lane < p.n) {
      float v = bf(s[lane]);
      h_out[(ulong)(tl.start + j) * p.n + row + lane] = bfloat(nemo_relu2(v));
    }
  }
}

// Multi-vec per-tensor FP8 routed expert down_proj for MTP tiles (count <= 8),
// reading BF16 intermediate activations and scaling outputs by routing weight pw.
// Grid: x = ceil(n / 32), y = ntiles; threadgroup = 256 threads (8 sgs).
kernel void nemo_moe_fp8_down_mv(device const NemoFp8ExpertRef *experts [[buffer(0)]],
                                 device const bfloat *h [[buffer(1)]],
                                 device const float *pw [[buffer(2)]],
                                 device const NemoMoeTile *tiles [[buffer(3)]],
                                 device float *y_out [[buffer(4)]],
                                 constant NemoMoeParam &p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  NemoMoeTile tl = tiles[tg.y];
  if (tl.count == 0u || tl.count > NEMO_MV_MAX_M) return;
  uint row = (tg.x * NEMO_MOE_MV_SGS + sg) * NEMO_MV_ROWS;
  if (row >= p.n) return;

  uint chunks = p.k / 8u;
  NemoFp8ExpertRef er = experts[tl.expert];
  device const uchar *we = er.w_down + (ulong)row * p.k;

  for (uint j = 0u; j < tl.count; j++) {
    float4 acc = float4(0.0f);
    device const bfloat4 *hr =
        (device const bfloat4 *)(h + (ulong)(tl.start + j) * p.k);
    for (uint c = lane; c < chunks; c += NEMO_SIMD) {
      float4 x0 = float4(hr[2u * c]), x1 = float4(hr[2u * c + 1u]);
#pragma unroll
      for (uint r = 0u; r < NEMO_MV_ROWS; r++) {
        uint2 q = ((device const uint2 *)(we + (ulong)r * p.k))[c];
        float4 prod = fp8x4(q.x) * x0 + fp8x4(q.y) * x1;
        acc[r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
    float4 s = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z),
                      simd_sum(acc.w)) *
               er.s_down;
    float rw = pw[tl.start + j];
    if (lane < NEMO_MV_ROWS && row + lane < p.n)
      y_out[(ulong)(tl.start + j) * p.n + row + lane] = bf(s[lane]) * rw;
  }
}

// Fused BF16 unembed + threadgroup first-maximum argmax (stage 1) for MTP
// draft sampling without writing 131072 FP32 logits to DRAM.
// Grid: ceil(vocab / 32) threadgroups of 256 threads (8 sgs, 4 rows/sg).
kernel void nemo_unembed_argmax_part(device const ushort *w [[buffer(0)]],
                                     device const float *x [[buffer(1)]],
                                     device NemoArgmaxPair *part_max [[buffer(2)]],
                                     constant NemoUnembedParam &p [[buffer(3)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg * NEMO_UNEMBED_SGS + sg) * NEMO_UNEMBED_ROWS_PER_SG;
  uint chunks = p.k / NEMO_UNEMBED_BF16_PER_U4;
  float4 acc = float4(0.0f);
  if (row < p.vocab) {
    uint valid_rows = min(NEMO_UNEMBED_ROWS_PER_SG, p.vocab - row);
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.k);
    device const float4 *xr = (device const float4 *)x;
    if (valid_rows == NEMO_UNEMBED_ROWS_PER_SG) {
      for (uint c = lane; c < chunks; c += NEMO_SIMD) {
        float4 x0 = xr[2u * c], x1 = xr[2u * c + 1u];
#pragma unroll
        for (uint r = 0u; r < NEMO_UNEMBED_ROWS_PER_SG; r++) {
          uint4 q = wr[r * chunks + c];
          float4 d = bf16x4(q.xy) * x0 + bf16x4(q.zw) * x1;
          acc[r] += d.x + d.y + d.z + d.w;
        }
      }
    } else {
      for (uint c = lane; c < chunks; c += NEMO_SIMD) {
        float4 x0 = xr[2u * c], x1 = xr[2u * c + 1u];
        for (uint r = 0u; r < valid_rows; r++) {
          uint4 q = wr[r * chunks + c];
          float4 d = bf16x4(q.xy) * x0 + bf16x4(q.zw) * x1;
          acc[r] += d.x + d.y + d.z + d.w;
        }
      }
    }
  }
  acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z),
               simd_sum(acc.w));

  threadgroup float sm_v[NEMO_UNEMBED_SGS];
  threadgroup int sm_i[NEMO_UNEMBED_SGS];
  if (lane == 0u) {
    float bv = -INFINITY;
    int bi = int(p.vocab);
    for (uint r = 0u; r < NEMO_UNEMBED_ROWS_PER_SG; r++) {
      uint id = row + r;
      if (id < p.vocab) {
        float v = acc[r];
        if (v > bv || (v == bv && int(id) < bi)) {
          bv = v;
          bi = int(id);
        }
      }
    }
    sm_v[sg] = bv;
    sm_i[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float bv = sm_v[0];
    int bi = sm_i[0];
    for (uint s = 1u; s < NEMO_UNEMBED_SGS; s++) {
      if (sm_v[s] > bv || (sm_v[s] == bv && sm_i[s] < bi)) {
        bv = sm_v[s];
        bi = sm_i[s];
      }
    }
    part_max[tg] = NemoArgmaxPair{bv, bi};
  }
}

// Reduces stage-1 threadgroup argmax pairs to one winning token index (stage 2).
// Grid: 1 threadgroup of 256 threads.
kernel void nemo_unembed_argmax_final(device const NemoArgmaxPair *part_max [[buffer(0)]],
                                      device int *out_tok [[buffer(1)]],
                                      constant uint &nparts [[buffer(2)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (uint i = tid; i < nparts; i += NEMO_ARGMAX_THREADS) {
    NemoArgmaxPair cur = part_max[i];
    if (cur.val > bv || (cur.val == bv && cur.idx < bi)) {
      bv = cur.val;
      bi = cur.idx;
    }
  }
  for (uint off = NEMO_SIMD / 2u; off > 0u; off >>= 1u) {
    float ov = simd_shuffle_xor(bv, off);
    int oi = simd_shuffle_xor(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  threadgroup float sm_v[NEMO_ARGMAX_SGS];
  threadgroup int sm_i[NEMO_ARGMAX_SGS];
  if (lane == 0u) {
    sm_v[sg] = bv;
    sm_i[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fbv = sm_v[0];
    int fbi = sm_i[0];
    for (uint s = 1u; s < NEMO_ARGMAX_SGS; s++) {
      if (sm_v[s] > fbv || (sm_v[s] == fbv && sm_i[s] < fbi)) {
        fbv = sm_v[s];
        fbi = sm_i[s];
      }
    }
    out_tok[0] = fbi;
  }
}
