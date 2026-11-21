// ---- Inkling-2-Mini: looped MoE language backbone ----
//
// Kernels for thinkingmachines/Inkling-2-Mini (model_type inkling2_loop).
// Constants match config.json and Hugging Face transformers modeling_inkling2.py
// / modular_inkling2_loop.py (PR 51050) / core_model_loading.py.

// Attention and model geometry (config.json text_config).
constant constexpr uint INK2_HEADS = 24, INK2_KV_HEADS = 8,
                        INK2_GQA = INK2_HEADS / INK2_KV_HEADS,
                        INK2_HEAD_DIM = 128, INK2_Q_DIM = INK2_HEADS * INK2_HEAD_DIM,
                        INK2_KV_DIM = INK2_KV_HEADS * INK2_HEAD_DIM,
                        INK2_D_REL = 16, INK2_REL_EXTENT = 1024,
                        INK2_WINDOW = 1024, INK2_SCONV_K = 4,
                        INK2_EXPERTS = 128, INK2_TOPK = 8,
                        INK2_INTER = 1152, INK2_DIM = 2304;
// Block sizes of quantized expert weights (hf_quant_config.json).
constant constexpr uint INK2_NVFP4_BLOCK = 16, INK2_MXFP8_BLOCK = 32;
// Maximum number of rows (completed blocks + partial) in an AttnRes stack.
constant constexpr uint INK2_MAX_RES_BLOCKS = 5;
// Hardware SIMD width and tile sizes for Metal matrix kernels.
constant constexpr uint INK2_SIMD = 32, INK2_MM_TILE = 64, INK2_MM_SIMDGROUPS = 4,
                        INK2_PREFILL_Q_TILE = 16, INK2_PREFILL_K_TILE = 64,
                        INK2_PREFILL_SGS = 2, INK2_DECODE_SGS = 4,
                        INK2_DECODE_VEC4 = 4, INK2_MV_ROWS = 4, INK2_MV_MAX_M = 8,
                        INK2_MV_SGS = 8, INK2_MAX_GATE_RANK = 128,
                        INK2_STAGE_HALF_K = 32, INK2_STAGE_VEC4S = 8,
                        INK2_ROUTE_SGS = INK2_EXPERTS / INK2_SIMD;
// Normalization, clipping, softcapping and scaling constants (config.json).
constant constexpr float INK2_RMS_EPS = 1e-6f, INK2_CLIP_EPS = 1e-6f,
                         INK2_QK_CLIP = 22.627416997969522f,
                         INK2_REL_CLIP = 8.0f, INK2_LOG_FLOOR = 8192.0f,
                         INK2_LOG_ALPHA = 0.1f, INK2_GATE_CAP = 4.0f,
                         INK2_UP_CAP = 25.0f, INK2_ROUTE_SCALE = 8.0f,
                         INK2_ROUTE_EPS = 1e-20f, INK2_LOGIT_CAP = 30.0f,
                         INK2_SCONV_STATIC_SCALE = 0.44721359549995793f,   // 1 / sqrt(4 + 1)
                         INK2_SCONV_DYNAMIC_SCALE = 0.33333333333333331f,  // 1 / sqrt(2 * 4 + 1)
                         INK2_FP8_SCALE = 256.0f;                          // cancels fp8x4's 2^-8

// Parameters for weighted RMSNorm and embedding lookup + norm.
struct Ink2NormParam {
  // Row width in floats and whether to round outputs to BF16.
  uint dim, round;
  // RMSNorm epsilon.
  float eps;
};

// Parameters for Block Attention Residuals (1, 3 or 4 queries).
struct Ink2AttnResParam {
  // Active token count, row stride between stacked blocks, hidden width.
  uint rows, block_stride, dim;
  // Number of completed blocks in `blocks` (1..4) and number of queries (1..4).
  uint nblocks, nqueries;
  // Row stride between query outputs (in rows, not floats).
  uint out_row_stride;
  // Unweighted RMSNorm epsilon.
  float eps;
};

// Parameters for loop message gate input concatenation and fused application.
struct Ink2MsgGateParam {
  // Active token count, hidden width, and bottleneck gate rank (64).
  uint rows, dim, rank;
  // Unweighted RMSNorm epsilon.
  float eps;
};

// Parameters for static and dynamic short causal convolutions (K = 4).
struct Ink2SconvParam {
  // Active token count, channel width, ring capacity in rows.
  uint rows, dim, ring_rows;
  // Position of row 0 when contiguous; ignored when per-row metadata is used.
  uint pos0;
  // Nonzero to round output to BF16.
  uint round;
};

// Per-row ring destination for batched multi-state short convolutions.
struct Ink2SconvTarget {
  // Device address of this state's FP32 ring buffer ([ring_rows, dim]).
  device float *ring;
  // Token position of this row.
  uint pos;
  // Padding for 16-byte alignment.
  uint pad;
};

// Per-row KV cache destination for batched multi-state attention and prep.
struct Ink2KvTarget {
  // Device addresses of this state's BF16 K and V buffers.
  device bfloat *k;
  device bfloat *v;
  // Token position of this row.
  uint pos;
  // Ring capacity in rows (0 for linear global KV).
  uint ring_rows;
};

// Parameters for Q/K/V/R head normalization, L2 clipping and log scaling.
struct Ink2QkvrParam {
  // Active token count and head-major Q row stride (padded to 16 for prefill).
  uint rows, q_stride;
  // Position of row 0 in single-state mode.
  uint pos0;
  // Ring capacity in rows in single-state mode (0 for linear global KV).
  uint ring_rows;
  // Nonzero on global layers (5, 11, 17, 23) to apply log scaling tau.
  uint is_global;
  // Nonzero to read per-row KV targets instead of single-state K/V buffers.
  uint batched;
};

// Parameters for relative position bias projection (r @ proj).
struct Ink2RelBiasParam {
  // Active token count, head count (24), relative dim (16), extent (1024).
  uint rows, heads, d_rel, extent;
};

// Parameters for FlashDecoding attention over sliding or global KV.
struct Ink2AttendDecodeParam {
  // Active token count, head-major Q row stride, row-0 position, ring capacity.
  uint rows, q_stride, pos0, ring_rows;
  // Sliding window size (1024 on sliding layers, 0 on global layers).
  uint window;
  // Number of KV partitions per (row, kv_head) and max partitions stride.
  uint nparts, part_stride;
  // Nonzero to read per-row Ink2KvTarget records instead of single-state KV.
  uint batched;
};

// Parameters for tiled MPP FlashAttention prefill over a contiguous chunk.
struct Ink2AttendPrefillParam {
  // Active query rows, head-major Q row stride (multiple of 16), row-0 pos.
  uint rows, q_stride, pos0;
  // Ring capacity in rows (0 for linear global KV).
  uint ring_rows;
  // Sliding window size (1024 on sliding layers, 0 on global layers).
  uint window;
};

// Parameters for MoE top-8 routing.
struct Ink2RouteParam {
  // Active token count, total experts (128), selected experts per token (8).
  uint rows, experts, topk;
  // Routing weight multiplier (8.0).
  float scale;
};

// One grouped-expert work tile: up to 64 consecutive (token, slot) pairs
// assigned to one expert.
struct Ink2MoeTile {
  // Expert index (0..127), first pair index, and pair count in this tile.
  uint expert, start, count, pad;
};

// Parameters for NVFP4 and MXFP8 MoE matvec/matmul kernels.
struct Ink2MoeParam {
  // Number of output rows (N) and input columns (K) per expert matrix.
  uint n, k;
  // Total active (token, slot) pairs across the batch, and total tiles.
  uint npairs, ntiles;
  // Softcap values for gate (4.0) and up (25.0) in w13.
  float gate_cap, up_cap;
};

// Parameters for MoE token combine across top-8 routed pairs.
struct Ink2MoeCombineParam {
  // Active token count, hidden width (2304), selected experts per token (8).
  uint rows, dim, topk;
};

// Parameters for logit softcap (30 * tanh(x / 30)).
struct Ink2SoftcapParam {
  // Active token count, unpadded vocabulary size, row stride in floats.
  uint rows, vocab, stride;
  // Softcap bound (30.0).
  float cap;
};

// Softcapped SwiGLU (modeling_inkling2.py Inkling2SoftcappedSwiGLU): sigmoid
// sees the raw uncapped gate while gate and up are softcapped separately.
inline float ink2_softcapped_swiglu(float gate, float up, float gate_cap, float up_cap) {
  float s = sigm(gate);
  float gc = gate_cap * tanh(gate / gate_cap);
  float uc = up_cap * tanh(up / up_cap);
  return bf(gc * s * uc);
}

// Embedding lookup plus weighted RMSNorm (modeling_inkling2.py
// Inkling2NormedEmbedding). Rows with id < 0 are skipped (image placeholders).
// One threadgroup per row.
kernel void ink2_embed_norm(device const ushort *table [[buffer(0)]],
                            device const int *ids [[buffer(1)]],
                            device const float *weight [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            constant Ink2NormParam &p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[INK2_SIMD];
  int id = ids[row];
  if (id < 0) return;
  device const ushort *src = table + (ulong)uint(id) * p.dim;
  device float *dst = out + (ulong)row * p.dim;
  float sq = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    float v = bf16f(src[i]);
    sq += v * v;
  }
  float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt) {
    float v = weight[i] * (bf16f(src[i]) * inv);
    dst[i] = p.round ? bf(v) : v;
  }
}

// Weighted RMSNorm over rows of `dim` floats. One threadgroup per row.
kernel void ink2_rmsnorm(device const float *x [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant Ink2NormParam &p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[INK2_SIMD];
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

// Block Attention Residuals (modeling_inkling2.py Inkling2BlockAttentionResidual
// and modular_inkling2_loop.py Inkling2LoopMessageAttentionResidual).
// Stacks `nblocks` completed blocks and `partial`, applies unweighted RMSNorm,
// and attends with `nqueries` FP32 queries scaled by 1 / sqrt(dim).
// One threadgroup per row.
kernel void ink2_attn_res(device const float *blocks [[buffer(0)]],
                          device const float *partial [[buffer(1)]],
                          device const float *queries [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          constant Ink2AttnResParam &p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[INK2_SIMD];
  threadgroup float inv_rms[INK2_MAX_RES_BLOCKS];
  threadgroup float weights[INK2_MAX_RES_BLOCKS];
  uint total = p.nblocks + 1;
  float scale = rsqrt(float(p.dim));
  // Compute unweighted RMSNorm inverse root-mean-square for each stacked block.
  for (uint b = 0; b < total; b++) {
    device const float *src = b < p.nblocks
                                  ? blocks + ((ulong)b * p.block_stride + row) * p.dim
                                  : partial + (ulong)row * p.dim;
    float sq = 0.0f;
    for (uint i = t; i < p.dim; i += nt) sq += src[i] * src[i];
    float inv = rsqrt(tg_sum(sq, tmp, t, nt) / float(p.dim) + p.eps);
    if (t == 0) inv_rms[b] = inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = 0; q < p.nqueries; q++) {
    device const float *qv = queries + (ulong)q * p.dim;
    float sc[INK2_MAX_RES_BLOCKS];
    for (uint b = 0; b < total; b++) {
      device const float *src = b < p.nblocks
                                    ? blocks + ((ulong)b * p.block_stride + row) * p.dim
                                    : partial + (ulong)row * p.dim;
      float dotv = 0.0f;
      for (uint i = t; i < p.dim; i += nt) dotv += qv[i] * (src[i] * inv_rms[b]);
      sc[b] = tg_sum(dotv, tmp, t, nt) * scale;
    }
    if (t == 0) {
      float mx = sc[0];
      for (uint b = 1; b < total; b++) mx = max(mx, sc[b]);
      float sum = 0.0f;
      for (uint b = 0; b < total; b++) {
        weights[b] = exp(sc[b] - mx);
        sum += weights[b];
      }
      for (uint b = 0; b < total; b++) weights[b] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *dst = out + ((ulong)q * p.out_row_stride + row) * p.dim;
    for (uint i = t; i < p.dim; i += nt) {
      float acc = 0.0f;
      for (uint b = 0; b < total; b++) {
        device const float *src = b < p.nblocks
                                      ? blocks + ((ulong)b * p.block_stride + row) * p.dim
                                      : partial + (ulong)row * p.dim;
        acc += weights[b] * (src[i] * inv_rms[b]);
      }
      dst[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Prepares loop controller input: concatenates unweighted-RMSNorm(h) and
// unweighted-RMSNorm(m), rounded to BF16, into [rows, 2 * dim] FP32 for down_proj.
// One threadgroup per row.
kernel void ink2_msg_cat_norm(device const float *h [[buffer(0)]],
                              device const float *m [[buffer(1)]],
                              device float *cat [[buffer(2)]],
                              constant Ink2MsgGateParam &p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[INK2_SIMD];
  device const float *hr = h + (ulong)row * p.dim;
  device const float *mr = m + (ulong)row * p.dim;
  device float *dst = cat + (ulong)row * (2 * p.dim);
  float sq_h = 0.0f, sq_m = 0.0f;
  for (uint i = t; i < p.dim; i += nt) {
    sq_h += hr[i] * hr[i];
    sq_m += mr[i] * mr[i];
  }
  float inv_h = rsqrt(tg_sum(sq_h, tmp, t, nt) / float(p.dim) + p.eps);
  float inv_m = rsqrt(tg_sum(sq_m, tmp, t, nt) / float(p.dim) + p.eps);
  for (uint i = t; i < p.dim; i += nt) {
    dst[i] = bf(hr[i] * inv_h);
    dst[p.dim + i] = bf(mr[i] * inv_m);
  }
}

// Applies loop message gate (modular_inkling2_loop.py Inkling2LoopMessageGate):
// out = (1 - tanh(forget_w @ g)) * h + tanh(gain_w @ g) * m.
// `forget_w` and `gain_w` are BF16 [dim, rank], `g` is FP32 [rows, rank].
// One threadgroup per row.
kernel void ink2_msg_gate_apply(device const float *h [[buffer(0)]],
                                device const float *m [[buffer(1)]],
                                device const float *g [[buffer(2)]],
                                device const ushort *forget_w [[buffer(3)]],
                                device const ushort *gain_w [[buffer(4)]],
                                device float *out [[buffer(5)]],
                                constant Ink2MsgGateParam &p [[buffer(6)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_threadgroup]],
                                uint nt [[threads_per_threadgroup]]) {
  threadgroup float gs[INK2_MAX_GATE_RANK];
  device const float *gr = g + (ulong)row * p.rank;
  for (uint i = t; i < p.rank; i += nt) gs[i] = gr[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const float *hr = h + (ulong)row * p.dim;
  device const float *mr = m + (ulong)row * p.dim;
  device float *dst = out + (ulong)row * p.dim;
  for (uint d = t; d < p.dim; d += nt) {
    device const ushort *fw = forget_w + (ulong)d * p.rank;
    device const ushort *gw = gain_w + (ulong)d * p.rank;
    float df = 0.0f, dg = 0.0f;
    for (uint j = 0; j < p.rank; j++) {
      float gv = gs[j];
      df += bf16f(fw[j]) * gv;
      dg += bf16f(gw[j]) * gv;
    }
    dst[d] = (1.0f - tanh(df)) * hr[d] + tanh(dg) * mr[d];
  }
}

// Static causal short convolution (modeling_inkling2.py Inkling2ShortConvolution):
// w[c, 0..3] applies to [t-3, t-2, t-1, t]; y = (conv + x) / sqrt(5).
// Reads past rows (< pos0) from `ring` and in-chunk rows from `x`, without
// mutating `ring`. Thread x: channel; thread y: row.
kernel void ink2_sconv_static(device const float *x [[buffer(0)]],
                              device const float *ring [[buffer(1)]],
                              device const float *w [[buffer(2)]],
                              device float *y [[buffer(3)]],
                              constant Ink2SconvParam &p [[buffer(4)]],
                              uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, r = i.y;
  if (c >= p.dim || r >= p.rows) return;
  uint pos = p.pos0 + r;
  device const float *wc = w + (ulong)c * INK2_SCONV_K;
  float xt = x[(ulong)r * p.dim + c];
  float sum = wc[INK2_SCONV_K - 1] * xt;
  for (uint lag = 1; lag < INK2_SCONV_K; lag++) {
    if (pos < lag) continue;
    float v = r >= lag ? x[(ulong)(r - lag) * p.dim + c]
                       : ring[(ulong)((pos - lag) % p.ring_rows) * p.dim + c];
    sum += wc[INK2_SCONV_K - 1 - lag] * v;
  }
  float out = (sum + xt) * INK2_SCONV_STATIC_SCALE;
  y[(ulong)r * p.dim + c] = p.round ? bf(out) : out;
}

// Batched multi-state static causal short convolution (one row per state),
// updating each state's ring slot at `pos % ring_rows` in place (safe because
// ring_rows >= 4). Thread x: channel; thread y: row.
kernel void ink2_sconv_static_batched(device const float *x [[buffer(0)]],
                                      device const Ink2SconvTarget *targets [[buffer(1)]],
                                      device const float *w [[buffer(2)]],
                                      device float *y [[buffer(3)]],
                                      constant Ink2SconvParam &p [[buffer(4)]],
                                      uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, r = i.y;
  if (c >= p.dim || r >= p.rows) return;
  device float *ring = targets[r].ring;
  uint pos = targets[r].pos;
  device const float *wc = w + (ulong)c * INK2_SCONV_K;
  float xt = x[(ulong)r * p.dim + c];
  float sum = wc[INK2_SCONV_K - 1] * xt;
  for (uint lag = 1; lag < INK2_SCONV_K; lag++) {
    if (pos < lag) continue;
    sum += wc[INK2_SCONV_K - 1 - lag] *
           ring[(ulong)((pos - lag) % p.ring_rows) * p.dim + c];
  }
  ring[(ulong)(pos % p.ring_rows) * p.dim + c] = xt;
  float out = (sum + xt) * INK2_SCONV_STATIC_SCALE;
  y[(ulong)r * p.dim + c] = p.round ? bf(out) : out;
}

// Dynamic causal short convolution for global layers 5, 11, 17, 23
// (modeling_inkling2.py Inkling2DynamicShortConvolution):
// static_w[lag, c] + dyn_w[row, kv_head, lag] weights x[t - lag, c];
// y = (conv + x) / sqrt(2 * 4 + 1) = (conv + x) / 3.
// Thread x: channel (0..1023); thread y: row.
kernel void ink2_sconv_dynamic(device const float *x [[buffer(0)]],
                               device const float *ring [[buffer(1)]],
                               device const float *static_w [[buffer(2)]],
                               device const float *dyn_w [[buffer(3)]],
                               device float *y [[buffer(4)]],
                               constant Ink2SconvParam &p [[buffer(5)]],
                               uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, r = i.y;
  if (c >= p.dim || r >= p.rows) return;
  uint pos = p.pos0 + r;
  uint kv_h = c / INK2_HEAD_DIM;
  device const float *dw = dyn_w + ((ulong)r * INK2_KV_HEADS + kv_h) * INK2_SCONV_K;
  float xt = x[(ulong)r * p.dim + c];
  float sum = (static_w[c] + dw[0]) * xt;
  for (uint lag = 1; lag < INK2_SCONV_K; lag++) {
    if (pos < lag) continue;
    float v = r >= lag ? x[(ulong)(r - lag) * p.dim + c]
                       : ring[(ulong)((pos - lag) % p.ring_rows) * p.dim + c];
    sum += (static_w[(ulong)lag * p.dim + c] + dw[lag]) * v;
  }
  float out = (sum + xt) * INK2_SCONV_DYNAMIC_SCALE;
  y[(ulong)r * p.dim + c] = p.round ? bf(out) : out;
}

// Batched multi-state dynamic causal short convolution (one row per state),
// updating each state's ring slot at `pos % ring_rows` in place.
kernel void ink2_sconv_dynamic_batched(device const float *x [[buffer(0)]],
                                       device const Ink2SconvTarget *targets [[buffer(1)]],
                                       device const float *static_w [[buffer(2)]],
                                       device const float *dyn_w [[buffer(3)]],
                                       device float *y [[buffer(4)]],
                                       constant Ink2SconvParam &p [[buffer(5)]],
                                       uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, r = i.y;
  if (c >= p.dim || r >= p.rows) return;
  device float *ring = targets[r].ring;
  uint pos = targets[r].pos;
  uint kv_h = c / INK2_HEAD_DIM;
  device const float *dw = dyn_w + ((ulong)r * INK2_KV_HEADS + kv_h) * INK2_SCONV_K;
  float xt = x[(ulong)r * p.dim + c];
  float sum = (static_w[c] + dw[0]) * xt;
  for (uint lag = 1; lag < INK2_SCONV_K; lag++) {
    if (pos < lag) continue;
    sum += (static_w[(ulong)lag * p.dim + c] + dw[lag]) *
           ring[(ulong)((pos - lag) % p.ring_rows) * p.dim + c];
  }
  ring[(ulong)(pos % p.ring_rows) * p.dim + c] = xt;
  float out = (sum + xt) * INK2_SCONV_DYNAMIC_SCALE;
  y[(ulong)r * p.dim + c] = p.round ? bf(out) : out;
}

// Writes the last `min(rows, ring_rows)` rows of a contiguous chunk into the
// short-convolution ring buffer after the convolution kernel completes.
kernel void ink2_sconv_ring_save(device const float *x [[buffer(0)]],
                                 device float *ring [[buffer(1)]],
                                 constant Ink2SconvParam &p [[buffer(2)]],
                                 uint2 i [[thread_position_in_grid]]) {
  uint c = i.x;
  uint keep = min(p.rows, p.ring_rows);
  if (c >= p.dim || i.y >= keep) return;
  uint r = p.rows - keep + i.y;
  uint slot = (p.pos0 + r) % p.ring_rows;
  ring[(ulong)slot * p.dim + c] = x[(ulong)r * p.dim + c];
}

// Per-head Q/K/V/R normalization, L2 clipping and optional global log scaling
// (modeling_inkling2.py Inkling2ScaledL2Clip, Inkling2L2Clip, Inkling2LogScale,
// Inkling2UnweightedRMSNorm). Writes BF16 Q into `q_out`, BF16 K and V directly
// into the KV cache, and FP32 (BF16-rounded) R into `r_out`.
// Grid: x = 24 (Q) + 8 (K) + 8 (V) + 24 (R) = 64 heads; y = rows.
// Threadgroup: 128 threads.
kernel void ink2_qkvr_prep(device const float *q_in [[buffer(0)]],
                           device const float *k_in [[buffer(1)]],
                           device const float *v_in [[buffer(2)]],
                           device const float *r_in [[buffer(3)]],
                           device const float *q_scale [[buffer(4)]],
                           device const float *r_scale [[buffer(5)]],
                           device bfloat *q_out [[buffer(6)]],
                           device bfloat *k_cache [[buffer(7)]],
                           device bfloat *v_cache [[buffer(8)]],
                           device float *r_out [[buffer(9)]],
                           device const Ink2KvTarget *targets [[buffer(10)]],
                           constant Ink2QkvrParam &p [[buffer(11)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[INK2_SIMD];
  uint head = tg.x, row = tg.y;
  if (row >= p.rows) return;
  uint pos = p.batched ? targets[row].pos : p.pos0 + row;
  uint ring_rows = p.batched ? targets[row].ring_rows : p.ring_rows;
  uint slot = ring_rows ? (pos % ring_rows) : pos;
  float tau = p.is_global
                  ? 1.0f + INK2_LOG_ALPHA * log(max(float(pos + 1u) / INK2_LOG_FLOOR, 1.0f))
                  : 1.0f;

  // Query heads 0..23: scaled L2 clip + optional log scale -> head-major BF16.
  if (head < INK2_HEADS) {
    uint h = head;
    float v = q_in[(ulong)row * INK2_Q_DIM + h * INK2_HEAD_DIM + t] *
              q_scale[h * INK2_HEAD_DIM + t];
    float norm = sqrt(tg_sum(v * v, tmp, t, INK2_HEAD_DIM));
    float clipped = bf(v * min(INK2_QK_CLIP / (norm + INK2_CLIP_EPS), 1.0f));
    float final_v = p.is_global ? bf(clipped * tau) : clipped;
    q_out[((ulong)h * p.q_stride + row) * INK2_HEAD_DIM + t] = bfloat(final_v);
    return;
  }

  // Key heads 24..31: unweighted L2 clip -> BF16 KV cache.
  if (head < INK2_HEADS + INK2_KV_HEADS) {
    uint h = head - INK2_HEADS;
    float v = k_in[(ulong)row * INK2_KV_DIM + h * INK2_HEAD_DIM + t];
    float norm = sqrt(tg_sum(v * v, tmp, t, INK2_HEAD_DIM));
    float clipped = v * min(INK2_QK_CLIP / (norm + INK2_CLIP_EPS), 1.0f);
    device bfloat *dst = p.batched ? targets[row].k : k_cache;
    dst[(ulong)slot * INK2_KV_DIM + h * INK2_HEAD_DIM + t] = bfloat(bf(clipped));
    return;
  }

  // Value heads 32..39: unweighted RMSNorm -> BF16 KV cache.
  if (head < INK2_HEADS + 2 * INK2_KV_HEADS) {
    uint h = head - INK2_HEADS - INK2_KV_HEADS;
    float v = v_in[(ulong)row * INK2_KV_DIM + h * INK2_HEAD_DIM + t];
    float inv = rsqrt(tg_sum(v * v, tmp, t, INK2_HEAD_DIM) / float(INK2_HEAD_DIM) + INK2_RMS_EPS);
    device bfloat *dst = p.batched ? targets[row].v : v_cache;
    dst[(ulong)slot * INK2_KV_DIM + h * INK2_HEAD_DIM + t] = bfloat(bf(v * inv));
    return;
  }

  // Relative heads 40..63 (16 dims each): scaled L2 clip + optional log scale.
  uint h = head - INK2_HEADS - 2 * INK2_KV_HEADS;
  if (t >= INK2_D_REL) return;
  float v = r_in[((ulong)row * INK2_HEADS + h) * INK2_D_REL + t] *
            r_scale[h * INK2_D_REL + t];
  float sq = v * v;
  // Reduce across the first 16 lanes of simdgroup 0.
  for (uint off = INK2_D_REL / 2; off > 0; off >>= 1) sq += simd_shuffle_xor(sq, off);
  float clipped = bf(v * min(INK2_REL_CLIP / (sqrt(sq) + INK2_CLIP_EPS), 1.0f));
  float final_v = p.is_global ? bf(clipped * tau) : clipped;
  r_out[((ulong)row * INK2_HEADS + h) * INK2_D_REL + t] = final_v;
}

// Relative position bias projection (modeling_inkling2.py Inkling2RelativeLogits):
// rel_bias[row, h, dist] = sum_j r[row, h, j] * proj[j, dist].
// Grid: x = heads (24), y = rows. Threadgroup: 256 threads (4 distances each).
kernel void ink2_rel_bias(device const float *r [[buffer(0)]],
                          device const ushort *proj [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          constant Ink2RelBiasParam &p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint2 nt [[threads_per_threadgroup]]) {
  uint h = tg.x, row = tg.y;
  if (h >= p.heads || row >= p.rows) return;
  device const float *rv = r + ((ulong)row * p.heads + h) * p.d_rel;
  float reg[INK2_D_REL];
  for (uint j = 0; j < p.d_rel; j++) reg[j] = rv[j];
  device float *dst = out + ((ulong)row * p.heads + h) * p.extent;
  for (uint d = t; d < p.extent; d += nt.x) {
    float acc = 0.0f;
    for (uint j = 0; j < p.d_rel; j++) acc += reg[j] * bf16f(proj[(ulong)j * p.extent + d]);
    dst[d] = acc;
  }
}

// Phase 1 of FlashDecoding attention: one threadgroup (4 simdgroups, 128 threads)
// handles one (row, kv_head, partition), computing online softmax over its key
// slice for all 3 GQA query heads simultaneously. Each lane holds 4 head dims
// (32 * 4 = 128).
// Output layout in `part`: [row][head][part] of (m, l, acc[128]) = 130 floats.
constant constexpr uint INK2_PART_FLOATS = INK2_HEAD_DIM + 2;

kernel void ink2_attend_part(device const bfloat *q [[buffer(0)]],
                             device const bfloat *k_cache [[buffer(1)]],
                             device const bfloat *v_cache [[buffer(2)]],
                             device const float *rel_bias [[buffer(3)]],
                             device const Ink2KvTarget *targets [[buffer(4)]],
                             device float *part [[buffer(5)]],
                             constant Ink2AttendDecodeParam &p [[buffer(6)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_m[INK2_GQA][INK2_DECODE_SGS];
  threadgroup float sg_l[INK2_GQA][INK2_DECODE_SGS];
  threadgroup float4 sg_o[INK2_GQA][INK2_DECODE_SGS][INK2_SIMD];

  uint kv_h = tg.x, part_idx = tg.y, row = tg.z;
  if (row >= p.rows || kv_h >= INK2_KV_HEADS || part_idx >= p.nparts) return;

  uint q_pos = p.batched ? targets[row].pos : p.pos0 + row;
  uint ring_rows = p.batched ? targets[row].ring_rows : p.ring_rows;
  device const bfloat *k_base = p.batched ? targets[row].k : k_cache;
  device const bfloat *v_base = p.batched ? targets[row].v : v_cache;

  uint kv_lo = (p.window && q_pos + 1u > p.window) ? (q_pos + 1u - p.window) : 0u;
  uint kv_hi = q_pos + 1u;
  uint total_k = kv_hi - kv_lo;
  uint per_part = (total_k + p.nparts - 1u) / p.nparts;
  uint k_start = kv_lo + part_idx * per_part;
  uint k_end = min(kv_hi, k_start + per_part);

  float scale = rsqrt(float(INK2_HEAD_DIM));
  float4 qv[INK2_GQA];
  for (uint g = 0; g < INK2_GQA; g++) {
    uint h = kv_h * INK2_GQA + g;
    device const bfloat4 *qp =
        (device const bfloat4 *)(q + ((ulong)h * p.q_stride + row) * INK2_HEAD_DIM);
    qv[g] = float4(qp[lane]);
  }

  float m_run[INK2_GQA] = {-INFINITY, -INFINITY, -INFINITY};
  float l_run[INK2_GQA] = {0.0f, 0.0f, 0.0f};
  float4 o_run[INK2_GQA] = {float4(0.0f), float4(0.0f), float4(0.0f)};

  for (uint j = k_start + sg; j < k_end; j += INK2_DECODE_SGS) {
    uint slot = ring_rows ? (j % ring_rows) : j;
    ulong kv_off = (ulong)slot * INK2_KV_DIM + kv_h * INK2_HEAD_DIM;
    float4 kv = float4(((device const bfloat4 *)(k_base + kv_off))[lane]);
    float4 vv = float4(((device const bfloat4 *)(v_base + kv_off))[lane]);
    uint dist = q_pos - j;
    for (uint g = 0; g < INK2_GQA; g++) {
      uint h = kv_h * INK2_GQA + g;
      float s = simd_sum(dot(qv[g], kv)) * scale;
      if (dist < INK2_REL_EXTENT)
        s += rel_bias[((ulong)row * INK2_HEADS + h) * INK2_REL_EXTENT + dist];
      float mn = max(m_run[g], s);
      float a = m_run[g] == -INFINITY ? 0.0f : exp(m_run[g] - mn);
      float b = exp(s - mn);
      o_run[g] = o_run[g] * a + vv * b;
      l_run[g] = l_run[g] * a + b;
      m_run[g] = mn;
    }
  }

  for (uint g = 0; g < INK2_GQA; g++) {
    if (lane == 0) sg_m[g][sg] = m_run[g], sg_l[g][sg] = l_run[g];
    sg_o[g][sg][lane] = o_run[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Merge the 4 simdgroups for each of the 3 query heads; simdgroup g writes head g.
  if (sg < INK2_GQA) {
    uint g = sg, h = kv_h * INK2_GQA + g;
    float m_all = max(max(sg_m[g][0], sg_m[g][1]), max(sg_m[g][2], sg_m[g][3]));
    float l_all = 0.0f;
    float4 o_all = float4(0.0f);
    if (m_all > -INFINITY) {
      for (uint s = 0; s < INK2_DECODE_SGS; s++) {
        if (sg_m[g][s] == -INFINITY) continue;
        float w = exp(sg_m[g][s] - m_all);
        l_all += sg_l[g][s] * w;
        o_all += sg_o[g][s][lane] * w;
      }
    }
    device float *dst =
        part + (((ulong)row * INK2_HEADS + h) * p.part_stride + part_idx) * INK2_PART_FLOATS;
    if (lane == 0) dst[0] = m_all, dst[1] = l_all;
    ((device float4 *)(dst + 2))[lane] = o_all;
  }
}

// Phase 2 of FlashDecoding attention: merges partitions, adds per-token sink
// logit to the softmax denominator, multiplies by sigmoid(gate), and rounds to
// BF16. Grid: x = heads (24), y = rows. Threadgroup: 32 threads (4 dims each).
kernel void ink2_attend_combine(device const float *part [[buffer(0)]],
                                device const float *sink [[buffer(1)]],
                                device const float *gate [[buffer(2)]],
                                device float *out [[buffer(3)]],
                                constant Ink2AttendDecodeParam &p [[buffer(4)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint h = tg.x, row = tg.y;
  if (h >= INK2_HEADS || row >= p.rows) return;
  device const float *base =
      part + ((ulong)row * INK2_HEADS + h) * p.part_stride * INK2_PART_FLOATS;
  float sink_v = sink[(ulong)row * INK2_HEADS + h];
  float m_all = sink_v;
  for (uint k = 0; k < p.nparts; k++) {
    float mk = base[(ulong)k * INK2_PART_FLOATS];
    m_all = max(m_all, mk);
  }
  float l_all = exp(sink_v - m_all);
  float4 o_all = float4(0.0f);
  for (uint k = 0; k < p.nparts; k++) {
    device const float *pk = base + (ulong)k * INK2_PART_FLOATS;
    float mk = pk[0];
    if (mk == -INFINITY) continue;
    float w = exp(mk - m_all);
    l_all += pk[1] * w;
    o_all += ((device const float4 *)(pk + 2))[lane] * w;
  }
  float inv_l = 1.0f / l_all;
  ulong off = (ulong)row * INK2_Q_DIM + h * INK2_HEAD_DIM + lane * INK2_DECODE_VEC4;
  float4 gv = *(device const float4 *)(gate + off);
  float4 res;
  for (uint c = 0; c < INK2_DECODE_VEC4; c++)
    res[c] = bf(bf(o_all[c] * inv_l) * sigm(gv[c]));
  *(device float4 *)(out + off) = res;
}

// Tiled MPP FlashAttention prefill over a contiguous chunk of queries.
// One threadgroup (2 simdgroups, 64 threads) computes 32 query rows for one
// head, streaming 64-key tiles from the token-major KV cache into shared
// threadgroup memory so both linear global KV and wrapping sliding-window rings
// feed the matrix units directly.
// Grid: x = ceil(rows / 32), y = heads (24). Threadgroup: 64 threads.
kernel void ink2_attend_prefill(device const bfloat *q [[buffer(0)]],
                                device const bfloat *k_cache [[buffer(1)]],
                                device const bfloat *v_cache [[buffer(2)]],
                                device const float *rel_bias [[buffer(3)]],
                                device const float *sink [[buffer(4)]],
                                device const float *gate [[buffer(5)]],
                                device float *out [[buffer(6)]],
                                constant Ink2AttendPrefillParam &p [[buffer(7)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint t [[thread_index_in_threadgroup]]) {
  constexpr uint QR = INK2_PREFILL_Q_TILE, KR = INK2_PREFILL_K_TILE,
                 HALF_K = KR / 2;
  threadgroup bfloat kv_tile[KR * INK2_HEAD_DIM];
  threadgroup float scores[INK2_PREFILL_SGS][QR * KR];
  threadgroup bfloat probs[INK2_PREFILL_SGS][QR * KR];
  threadgroup float rescales[INK2_PREFILL_SGS][QR];
  threadgroup float sums[INK2_PREFILL_SGS][QR];
  threadgroup float maxes[INK2_PREFILL_SGS][QR];

  uint h = tg.y, kv_h = h / INK2_GQA;
  uint block_q0 = tg.x * (INK2_PREFILL_SGS * QR);
  uint sg_q0 = block_q0 + sg * QR;
  uint sg_q_end = min(sg_q0 + QR, p.rows);

  constexpr auto sdesc = matmul2d_descriptor(
      QR, KR, INK2_HEAD_DIM, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(
      QR, INK2_HEAD_DIM, KR, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq_all(
      (device bfloat *)q + (ulong)h * p.q_stride * INK2_HEAD_DIM,
      dextents<int32_t, 2>(INK2_HEAD_DIM, p.q_stride));
  tensor<threadgroup bfloat, extents<int32_t, INK2_HEAD_DIM, KR>, tensor_inline> tkv(
      kv_tile, extents<int32_t, INK2_HEAD_DIM, KR>());
  tensor<threadgroup bfloat, extents<int32_t, KR, QR>, tensor_inline> tp(
      probs[sg], extents<int32_t, KR, QR>());

  auto qt = tq_all.slice(0, min(sg_q0, p.q_stride - QR));
  auto acc = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tkv), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;

  uint first_pos = p.pos0 + block_q0;
  uint last_row = min(block_q0 + INK2_PREFILL_SGS * QR, p.rows) - 1u;
  uint last_pos = p.pos0 + last_row;
  uint k_lo = (p.window && first_pos + 1u > p.window) ? (first_pos + 1u - p.window) : 0u;
  uint k_hi = last_pos + 1u;
  float scale = rsqrt(float(INK2_HEAD_DIM));

  uint row = lane / 2u, col0 = (lane % 2u) * HALF_K;
  float running_max = -INFINITY, running_sum = 0.0f;

  for (uint k0 = k_lo; k0 < k_hi; k0 += KR) {
    // Collaboratively load 64 K rows into kv_tile across all 64 threads.
    for (uint idx = t; idx < KR * (INK2_HEAD_DIM / INK2_DECODE_VEC4);
         idx += INK2_PREFILL_SGS * INK2_SIMD) {
      uint kr = idx / (INK2_HEAD_DIM / INK2_DECODE_VEC4);
      uint v4 = idx % (INK2_HEAD_DIM / INK2_DECODE_VEC4);
      bfloat4 kv = bfloat4(bfloat(0.0f));
      uint kpos = k0 + kr;
      if (kpos < k_hi) {
        uint slot = p.ring_rows ? (kpos % p.ring_rows) : kpos;
        device const bfloat4 *src =
            (device const bfloat4 *)(k_cache + (ulong)slot * INK2_KV_DIM + kv_h * INK2_HEAD_DIM);
        kv = src[v4];
      }
      ((threadgroup bfloat4 *)kv_tile)[idx] = kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_q0 < p.rows) {
      auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(tkv), float>();
      sop.run(qt, tkv, sc);
      for (uint16_t i = 0; i < sc.get_capacity(); i++) {
        if (sc.is_valid_element(i)) {
          auto at = sc.get_multidimensional_index(i);
          uint kr = at[0], qr = at[1];
          uint q_local = sg_q0 + qr;
          uint kpos = k0 + kr;
          float val = -INFINITY;
          if (q_local < sg_q_end && kpos < k_hi) {
            uint qpos = p.pos0 + q_local;
            if (kpos <= qpos && (!p.window || kpos + p.window > qpos)) {
              val = sc[i] * scale;
              uint dist = qpos - kpos;
              if (dist < INK2_REL_EXTENT)
                val += rel_bias[((ulong)q_local * INK2_HEADS + h) * INK2_REL_EXTENT + dist];
            }
          }
          scores[sg][qr * KR + kr] = val;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup float *own = scores[sg] + row * KR + col0;
      float bmax = -INFINITY;
      for (uint c = 0; c < HALF_K; c++) bmax = max(bmax, own[c]);
      bmax = max(bmax, simd_shuffle_xor(bmax, 1));
      float new_max = max(running_max, bmax);
      float old_scale = (running_max == -INFINITY || new_max == -INFINITY)
                            ? 0.0f
                            : exp(running_max - new_max);
      float bsum = 0.0f;
      for (uint c = 0; c < HALF_K; c++) {
        float e = own[c] == -INFINITY ? 0.0f : exp(own[c] - new_max);
        probs[sg][row * KR + col0 + c] = bfloat(e);
        bsum += e;
      }
      bsum += simd_shuffle_xor(bsum, 1);
      running_sum = running_sum * old_scale + bsum;
      running_max = new_max;
      if ((lane & 1u) == 0u) rescales[sg][row] = old_scale;
      simdgroup_barrier(mem_flags::mem_threadgroup);

      for (uint16_t i = 0; i < acc.get_capacity(); i++)
        if (acc.is_valid_element(i))
          acc[i] *= rescales[sg][acc.get_multidimensional_index(i)[1]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Collaboratively load 64 V rows into kv_tile across all 64 threads.
    for (uint idx = t; idx < KR * (INK2_HEAD_DIM / INK2_DECODE_VEC4);
         idx += INK2_PREFILL_SGS * INK2_SIMD) {
      uint kr = idx / (INK2_HEAD_DIM / INK2_DECODE_VEC4);
      uint v4 = idx % (INK2_HEAD_DIM / INK2_DECODE_VEC4);
      bfloat4 vv = bfloat4(bfloat(0.0f));
      uint kpos = k0 + kr;
      if (kpos < k_hi) {
        uint slot = p.ring_rows ? (kpos % p.ring_rows) : kpos;
        device const bfloat4 *src =
            (device const bfloat4 *)(v_cache + (ulong)slot * INK2_KV_DIM + kv_h * INK2_HEAD_DIM);
        vv = src[v4];
      }
      ((threadgroup bfloat4 *)kv_tile)[idx] = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_q0 < p.rows) oop.run(tp, tkv, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (sg_q0 >= p.rows) return;

  // Fold per-token sink logit into softmax denominator.
  if ((lane & 1u) == 0u) {
    uint q_local = sg_q0 + row;
    if (q_local < sg_q_end) {
      float sv = sink[(ulong)q_local * INK2_HEADS + h];
      float fm = max(running_max, sv);
      float oscale = running_max == -INFINITY ? 0.0f : exp(running_max - fm);
      float fl = running_sum * oscale + exp(sv - fm);
      maxes[sg][row] = oscale;
      sums[sg][row] = fl;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      auto at = acc.get_multidimensional_index(i);
      uint q_local = sg_q0 + at[1];
      if (q_local < sg_q_end) {
        float val = bf(acc[i] * maxes[sg][at[1]] / sums[sg][at[1]]);
        ulong off = (ulong)q_local * INK2_Q_DIM + h * INK2_HEAD_DIM + at[0];
        out[off] = bf(val * sigm(gate[off]));
      }
    }
  }
}

// MoE router (modeling_inkling2.py Inkling2TopKRouter): sigmoid scores, top-8
// selected by (score + bias), unbiased normalized weights times route_scale.
// One threadgroup of 128 threads per token row.
kernel void ink2_route_topk(device const float *logits [[buffer(0)]],
                            device const float *bias [[buffer(1)]],
                            device uint *out_ids [[buffer(2)]],
                            device float *out_weights [[buffer(3)]],
                            constant Ink2RouteParam &p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  threadgroup float vals[INK2_ROUTE_SGS * INK2_TOPK];
  threadgroup uint idxs[INK2_ROUTE_SGS * INK2_TOPK];
  threadgroup float sel_dummy[INK2_TOPK];
  threadgroup float scores[INK2_EXPERTS];
  if (row >= p.rows || t >= p.experts) return;
  float sc = sigm(logits[(ulong)row * p.experts + t]);
  scores[t] = sc;
  float key = sc + bias[t];
  device uint *row_ids = out_ids + (ulong)row * p.topk;
  group_topk(key, t, p.topk, vals, idxs, row_ids, sel_dummy, t, p.experts);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < p.topk; k++) sum += scores[row_ids[k]];
    float inv = p.scale / (sum + INK2_ROUTE_EPS);
    device float *row_w = out_weights + (ulong)row * p.topk;
    for (uint k = 0; k < p.topk; k++) row_w[k] = scores[row_ids[k]] * inv;
  }
}

// Gathers FP32 token rows into BF16 pair order for MPP MoE matmul tiles.
// Grid: x = dim / 4, y = npairs.
kernel void ink2_moe_gather_bf16(device const float4 *x [[buffer(0)]],
                                 device const int *ptok [[buffer(1)]],
                                 device bfloat4 *out [[buffer(2)]],
                                 constant uint &dim4 [[buffer(3)]],
                                 uint2 i [[thread_position_in_grid]]) {
  int tok = ptok[i.y];
  out[(ulong)i.y * dim4 + i.x] = bfloat4(x[(ulong)tok * dim4 + i.x]);
}

// Multi-vec NVFP4 w13 + softcapped SwiGLU for tiles with count <= 8.
// Each simdgroup computes 4 consecutive rows (2 interleaved gate/up pairs) for
// up to 8 tokens sharing one expert, reading weights once per tile and writing
// BF16 intermediate activations.
// Grid: x = ceil(n / 32) = 72, y = ntiles. Threadgroup: 256 threads (8 sgs).
kernel void ink2_moe_nvfp4_up_mv(device const uchar *w13 [[buffer(0)]],
                                 device const uchar *s13 [[buffer(1)]],
                                 device const float *factor13 [[buffer(2)]],
                                 device const float *x [[buffer(3)]],
                                 device const int *ptok [[buffer(4)]],
                                 device const Ink2MoeTile *tiles [[buffer(5)]],
                                 device bfloat *h_out [[buffer(6)]],
                                 constant Ink2MoeParam &p [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count == 0 || tl.count > INK2_MV_MAX_M) return;
  uint row = (tg.x * INK2_MV_SGS + sg) * INK2_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / INK2_NVFP4_BLOCK;
  uint w_row_bytes = p.k / 2u;
  device const uchar *we = w13 + (ulong)tl.expert * p.n * w_row_bytes + (ulong)row * w_row_bytes;
  device const uchar *se = s13 + (ulong)tl.expert * p.n * blocks + (ulong)row * blocks;
  float ef = factor13[tl.expert];

  float4 acc[INK2_MV_MAX_M];
  uint tok[INK2_MV_MAX_M];
  for (uint j = 0; j < tl.count; j++) {
    acc[j] = float4(0.0f);
    tok[j] = uint(ptok[tl.start + j]);
  }

  for (uint c = lane; c < blocks; c += INK2_SIMD) {
    float4 wv[INK2_MV_ROWS][4];
    for (uint r = 0; r < INK2_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
      float sc = fp8(se[(ulong)r * blocks + c]);
      float4 a, b, d0, d1;
      fp4x8(q.x, a, b);
      fp4x8(q.y, d0, d1);
      wv[r][0] = a * sc;
      wv[r][1] = b * sc;
      wv[r][2] = d0 * sc;
      wv[r][3] = d1 * sc;
    }
    for (uint j = 0; j < tl.count; j++) {
      device const float4 *xr = (device const float4 *)(x + (ulong)tok[j] * p.k) + 4u * c;
      float4 x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
      for (uint r = 0; r < INK2_MV_ROWS; r++) {
        float4 prod = wv[r][0] * x0 + wv[r][1] * x1 + wv[r][2] * x2 + wv[r][3] * x3;
        acc[j][r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
  }

  uint inter = p.n / 2u;
  uint pair_col = row / 2u;
  for (uint j = 0; j < tl.count; j++) {
    float g0 = bf(simd_sum(acc[j].x) * ef);
    float u0 = bf(simd_sum(acc[j].y) * ef);
    float g1 = bf(simd_sum(acc[j].z) * ef);
    float u1 = bf(simd_sum(acc[j].w) * ef);
    if (lane == 0) {
      device bfloat *dst = h_out + (ulong)(tl.start + j) * inter + pair_col;
      dst[0] = bfloat(ink2_softcapped_swiglu(g0, u0, p.gate_cap, p.up_cap));
      dst[1] = bfloat(ink2_softcapped_swiglu(g1, u1, p.gate_cap, p.up_cap));
    }
  }
}

// Multi-vec NVFP4 w2 for tiles with count <= 8, reading BF16 intermediate
// activations and scaling outputs by each pair's routing weight.
// Grid: x = ceil(n / 32) = 72, y = ntiles. Threadgroup: 256 threads (8 sgs).
kernel void ink2_moe_nvfp4_down_mv(device const uchar *w2 [[buffer(0)]],
                                   device const uchar *s2 [[buffer(1)]],
                                   device const float *factor2 [[buffer(2)]],
                                   device const bfloat *h [[buffer(3)]],
                                   device const float *pw [[buffer(4)]],
                                   device const Ink2MoeTile *tiles [[buffer(5)]],
                                   device float *y_out [[buffer(6)]],
                                   constant Ink2MoeParam &p [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count == 0 || tl.count > INK2_MV_MAX_M) return;
  uint row = (tg.x * INK2_MV_SGS + sg) * INK2_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / INK2_NVFP4_BLOCK;
  uint w_row_bytes = p.k / 2u;
  device const uchar *we = w2 + (ulong)tl.expert * p.n * w_row_bytes + (ulong)row * w_row_bytes;
  device const uchar *se = s2 + (ulong)tl.expert * p.n * blocks + (ulong)row * blocks;
  float ef = factor2[tl.expert];

  float4 acc[INK2_MV_MAX_M];
  for (uint j = 0; j < tl.count; j++) acc[j] = float4(0.0f);

  for (uint c = lane; c < blocks; c += INK2_SIMD) {
    float4 wv[INK2_MV_ROWS][4];
    for (uint r = 0; r < INK2_MV_ROWS; r++) {
      uint2 q = ((device const uint2 *)(we + (ulong)r * w_row_bytes))[c];
      float sc = fp8(se[(ulong)r * blocks + c]);
      float4 a, b, d0, d1;
      fp4x8(q.x, a, b);
      fp4x8(q.y, d0, d1);
      wv[r][0] = a * sc;
      wv[r][1] = b * sc;
      wv[r][2] = d0 * sc;
      wv[r][3] = d1 * sc;
    }
    for (uint j = 0; j < tl.count; j++) {
      device const bfloat4 *hr =
          (device const bfloat4 *)(h + (ulong)(tl.start + j) * p.k) + 4u * c;
      float4 x0 = float4(hr[0]), x1 = float4(hr[1]), x2 = float4(hr[2]), x3 = float4(hr[3]);
      for (uint r = 0; r < INK2_MV_ROWS; r++) {
        float4 prod = wv[r][0] * x0 + wv[r][1] * x1 + wv[r][2] * x2 + wv[r][3] * x3;
        acc[j][r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
  }

  for (uint j = 0; j < tl.count; j++) {
    float4 s = float4(simd_sum(acc[j].x), simd_sum(acc[j].y), simd_sum(acc[j].z),
                      simd_sum(acc[j].w)) *
               ef;
    float rw = pw[tl.start + j];
    if (lane < INK2_MV_ROWS && row + lane < p.n)
      y_out[(ulong)(tl.start + j) * p.n + row + lane] = bf(s[lane]) * rw;
  }
}

// Multi-vec MXFP8 w13 + softcapped SwiGLU for layers 20..23 (tile count <= 8).
// Grid: x = ceil(n / 32) = 72, y = ntiles. Threadgroup: 256 threads (8 sgs).
kernel void ink2_moe_mxfp8_up_mv(device const uchar *w13 [[buffer(0)]],
                                 device const uchar *s13 [[buffer(1)]],
                                 device const float *x [[buffer(2)]],
                                 device const int *ptok [[buffer(3)]],
                                 device const Ink2MoeTile *tiles [[buffer(4)]],
                                 device bfloat *h_out [[buffer(5)]],
                                 constant Ink2MoeParam &p [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count == 0 || tl.count > INK2_MV_MAX_M) return;
  uint row = (tg.x * INK2_MV_SGS + sg) * INK2_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / INK2_MXFP8_BLOCK;
  device const uchar *we = w13 + (ulong)tl.expert * p.n * p.k + (ulong)row * p.k;
  device const uchar *se = s13 + (ulong)tl.expert * p.n * blocks + (ulong)row * blocks;

  float4 acc[INK2_MV_MAX_M];
  uint tok[INK2_MV_MAX_M];
  for (uint j = 0; j < tl.count; j++) {
    acc[j] = float4(0.0f);
    tok[j] = uint(ptok[tl.start + j]);
  }

  for (uint c = lane; c < blocks; c += INK2_SIMD) {
    float4 wv[INK2_MV_ROWS][INK2_STAGE_VEC4S];
    for (uint r = 0; r < INK2_MV_ROWS; r++) {
      device const uint4 *wr = (device const uint4 *)(we + (ulong)r * p.k);
      uint4 q0 = wr[2u * c], q1 = wr[2u * c + 1u];
      float sc = e8m0(se[(ulong)r * blocks + c]) * INK2_FP8_SCALE;
      wv[r][0] = fp8x4(q0.x) * sc;
      wv[r][1] = fp8x4(q0.y) * sc;
      wv[r][2] = fp8x4(q0.z) * sc;
      wv[r][3] = fp8x4(q0.w) * sc;
      wv[r][4] = fp8x4(q1.x) * sc;
      wv[r][5] = fp8x4(q1.y) * sc;
      wv[r][6] = fp8x4(q1.z) * sc;
      wv[r][7] = fp8x4(q1.w) * sc;
    }
    for (uint j = 0; j < tl.count; j++) {
      device const float4 *xr =
          (device const float4 *)(x + (ulong)tok[j] * p.k) + INK2_STAGE_VEC4S * c;
      float4 xv[INK2_STAGE_VEC4S];
      for (uint b = 0; b < INK2_STAGE_VEC4S; b++) xv[b] = xr[b];
      for (uint r = 0; r < INK2_MV_ROWS; r++) {
        float4 prod = float4(0.0f);
        for (uint b = 0; b < INK2_STAGE_VEC4S; b++) prod += wv[r][b] * xv[b];
        acc[j][r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
  }

  uint inter = p.n / 2u;
  uint pair_col = row / 2u;
  for (uint j = 0; j < tl.count; j++) {
    float g0 = bf(simd_sum(acc[j].x));
    float u0 = bf(simd_sum(acc[j].y));
    float g1 = bf(simd_sum(acc[j].z));
    float u1 = bf(simd_sum(acc[j].w));
    if (lane == 0) {
      device bfloat *dst = h_out + (ulong)(tl.start + j) * inter + pair_col;
      dst[0] = bfloat(ink2_softcapped_swiglu(g0, u0, p.gate_cap, p.up_cap));
      dst[1] = bfloat(ink2_softcapped_swiglu(g1, u1, p.gate_cap, p.up_cap));
    }
  }
}

// Multi-vec MXFP8 w2 for layers 20..23 (tile count <= 8), reading BF16
// intermediate activations and scaling outputs by routing weight.
// Grid: x = ceil(n / 32) = 72, y = ntiles. Threadgroup: 256 threads (8 sgs).
kernel void ink2_moe_mxfp8_down_mv(device const uchar *w2 [[buffer(0)]],
                                   device const uchar *s2 [[buffer(1)]],
                                   device const bfloat *h [[buffer(2)]],
                                   device const float *pw [[buffer(3)]],
                                   device const Ink2MoeTile *tiles [[buffer(4)]],
                                   device float *y_out [[buffer(5)]],
                                   constant Ink2MoeParam &p [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count == 0 || tl.count > INK2_MV_MAX_M) return;
  uint row = (tg.x * INK2_MV_SGS + sg) * INK2_MV_ROWS;
  if (row >= p.n) return;

  uint blocks = p.k / INK2_MXFP8_BLOCK;
  device const uchar *we = w2 + (ulong)tl.expert * p.n * p.k + (ulong)row * p.k;
  device const uchar *se = s2 + (ulong)tl.expert * p.n * blocks + (ulong)row * blocks;

  float4 acc[INK2_MV_MAX_M];
  for (uint j = 0; j < tl.count; j++) acc[j] = float4(0.0f);

  for (uint c = lane; c < blocks; c += INK2_SIMD) {
    float4 wv[INK2_MV_ROWS][INK2_STAGE_VEC4S];
    for (uint r = 0; r < INK2_MV_ROWS; r++) {
      device const uint4 *wr = (device const uint4 *)(we + (ulong)r * p.k);
      uint4 q0 = wr[2u * c], q1 = wr[2u * c + 1u];
      float sc = e8m0(se[(ulong)r * blocks + c]) * INK2_FP8_SCALE;
      wv[r][0] = fp8x4(q0.x) * sc;
      wv[r][1] = fp8x4(q0.y) * sc;
      wv[r][2] = fp8x4(q0.z) * sc;
      wv[r][3] = fp8x4(q0.w) * sc;
      wv[r][4] = fp8x4(q1.x) * sc;
      wv[r][5] = fp8x4(q1.y) * sc;
      wv[r][6] = fp8x4(q1.z) * sc;
      wv[r][7] = fp8x4(q1.w) * sc;
    }
    for (uint j = 0; j < tl.count; j++) {
      device const bfloat4 *hr =
          (device const bfloat4 *)(h + (ulong)(tl.start + j) * p.k) + INK2_STAGE_VEC4S * c;
      float4 xv[INK2_STAGE_VEC4S];
      for (uint b = 0; b < INK2_STAGE_VEC4S; b++) xv[b] = float4(hr[b]);
      for (uint r = 0; r < INK2_MV_ROWS; r++) {
        float4 prod = float4(0.0f);
        for (uint b = 0; b < INK2_STAGE_VEC4S; b++) prod += wv[r][b] * xv[b];
        acc[j][r] += prod.x + prod.y + prod.z + prod.w;
      }
    }
  }

  for (uint j = 0; j < tl.count; j++) {
    float4 s = float4(simd_sum(acc[j].x), simd_sum(acc[j].y), simd_sum(acc[j].z),
                      simd_sum(acc[j].w));
    float rw = pw[tl.start + j];
    if (lane < INK2_MV_ROWS && row + lane < p.n)
      y_out[(ulong)(tl.start + j) * p.n + row + lane] = bf(s[lane]) * rw;
  }
}

// Stages one 64x64 NVFP4 weight tile (scaled by E4M3 block scale only, which is
// bit-exact in BF16; the per-expert FP32 factor is applied after matmul2d) into
// threadgroup BF16 memory across 128 threads (32 values/thread).
inline void ink2_stage_nvfp4_tile(device const uchar *we, device const uchar *se,
                                  uint n0, uint k0, uint n, uint k, threadgroup bfloat *ws,
                                  uint t) {
  uint r_local = t / 2u, half_k = t & 1u;
  uint r = n0 + r_local;
  uint w_row_bytes = k / 2u, blocks = k / INK2_NVFP4_BLOCK;
  threadgroup bfloat4 *dst =
      (threadgroup bfloat4 *)(ws + r_local * INK2_MM_TILE + half_k * INK2_STAGE_HALF_K);
  if (r < n) {
    uint b0 = (k0 + half_k * INK2_STAGE_HALF_K) / INK2_NVFP4_BLOCK;
    uint4 q =
        *(device const uint4 *)(we + (ulong)r * w_row_bytes + (k0 + half_k * INK2_STAGE_HALF_K) / 2u);
    float sc0 = fp8(se[(ulong)r * blocks + b0]);
    float sc1 = fp8(se[(ulong)r * blocks + b0 + 1u]);
    float4 a, b, c, d;
    fp4x8(q.x, a, b);
    dst[0] = bfloat4(a * sc0);
    dst[1] = bfloat4(b * sc0);
    fp4x8(q.y, c, d);
    dst[2] = bfloat4(c * sc0);
    dst[3] = bfloat4(d * sc0);
    fp4x8(q.z, a, b);
    dst[4] = bfloat4(a * sc1);
    dst[5] = bfloat4(b * sc1);
    fp4x8(q.w, c, d);
    dst[6] = bfloat4(c * sc1);
    dst[7] = bfloat4(d * sc1);
  } else {
    for (uint i = 0; i < INK2_STAGE_VEC4S; i++) dst[i] = bfloat4(bfloat(0.0f));
  }
}

// Stages one 64x64 MXFP8 weight tile (per-row UE8M0 scale per 32) into
// threadgroup BF16 memory across 128 threads (32 values/thread).
inline void ink2_stage_mxfp8_tile(device const uchar *we, device const uchar *se,
                                  uint n0, uint k0, uint n, uint k, threadgroup bfloat *ws,
                                  uint t) {
  uint r_local = t / 2u, half_k = t & 1u;
  uint r = n0 + r_local;
  uint blocks = k / INK2_MXFP8_BLOCK;
  threadgroup bfloat4 *dst =
      (threadgroup bfloat4 *)(ws + r_local * INK2_MM_TILE + half_k * INK2_STAGE_HALF_K);
  if (r < n) {
    uint b0 = (k0 + half_k * INK2_STAGE_HALF_K) / INK2_MXFP8_BLOCK;
    device const uint4 *wr =
        (device const uint4 *)(we + (ulong)r * k + k0 + half_k * INK2_STAGE_HALF_K);
    uint4 q0 = wr[0], q1 = wr[1];
    float sc = e8m0(se[(ulong)r * blocks + b0]) * INK2_FP8_SCALE;
    dst[0] = bfloat4(fp8x4(q0.x) * sc);
    dst[1] = bfloat4(fp8x4(q0.y) * sc);
    dst[2] = bfloat4(fp8x4(q0.z) * sc);
    dst[3] = bfloat4(fp8x4(q0.w) * sc);
    dst[4] = bfloat4(fp8x4(q1.x) * sc);
    dst[5] = bfloat4(fp8x4(q1.y) * sc);
    dst[6] = bfloat4(fp8x4(q1.z) * sc);
    dst[7] = bfloat4(fp8x4(q1.w) * sc);
  } else {
    for (uint i = 0; i < INK2_STAGE_VEC4S; i++) dst[i] = bfloat4(bfloat(0.0f));
  }
}

// Tiled MPP NVFP4 w13 + softcapped SwiGLU for tiles with count > 8 (up to 64).
// Grid: x = n / 64, y = ntiles. Threadgroup: 128 threads (4 simdgroups).
kernel void ink2_moe_nvfp4_up_mm(device const uchar *w13 [[buffer(0)]],
                                 device const uchar *s13 [[buffer(1)]],
                                 device const float *factor13 [[buffer(2)]],
                                 device const bfloat *x_bf16 [[buffer(3)]],
                                 device const Ink2MoeTile *tiles [[buffer(4)]],
                                 device bfloat *h_out [[buffer(5)]],
                                 constant Ink2MoeParam &p [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count <= INK2_MV_MAX_M) return;

  threadgroup bfloat ws[INK2_MM_TILE * INK2_MM_TILE];
  threadgroup float out_tile[INK2_MM_TILE * INK2_MM_TILE];
  constexpr auto desc = matmul2d_descriptor(
      INK2_MM_TILE, INK2_MM_TILE, INK2_MM_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<INK2_MM_SIMDGROUPS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, p.npairs + INK2_MM_TILE));
  tensor<threadgroup bfloat, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> tw(
      ws, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * INK2_MM_TILE;
  uint w_row_bytes = p.k / 2u, blocks = p.k / INK2_NVFP4_BLOCK;
  device const uchar *we = w13 + (ulong)tl.expert * p.n * w_row_bytes;
  device const uchar *se = s13 + (ulong)tl.expert * p.n * blocks;
  float ef = factor13[tl.expert];

  for (uint k0 = 0; k0 < p.k; k0 += INK2_MM_TILE) {
    ink2_stage_nvfp4_tile(we, se, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = tx.slice(k0, tl.start);
    op.run(a, tw, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  tensor<threadgroup float, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> to(
      out_tile, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  c.store(to);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint inter = p.n / 2u, pair_col0 = n0 / 2u;
  constexpr uint HALF_TILE = INK2_MM_TILE / 2u;
  for (uint idx = t; idx < tl.count * HALF_TILE; idx += INK2_MM_SIMDGROUPS * INK2_SIMD) {
    uint r = idx / HALF_TILE, j = idx % HALF_TILE;
    float g = bf(out_tile[r * INK2_MM_TILE + 2u * j] * ef);
    float u = bf(out_tile[r * INK2_MM_TILE + 2u * j + 1u] * ef);
    h_out[(ulong)(tl.start + r) * inter + pair_col0 + j] =
        bfloat(ink2_softcapped_swiglu(g, u, p.gate_cap, p.up_cap));
  }
}

// Tiled MPP NVFP4 w2 for tiles with count > 8 (up to 64).
// Grid: x = n / 64, y = ntiles. Threadgroup: 128 threads (4 simdgroups).
kernel void ink2_moe_nvfp4_down_mm(device const uchar *w2 [[buffer(0)]],
                                   device const uchar *s2 [[buffer(1)]],
                                   device const float *factor2 [[buffer(2)]],
                                   device const bfloat *h_bf16 [[buffer(3)]],
                                   device const float *pw [[buffer(4)]],
                                   device const Ink2MoeTile *tiles [[buffer(5)]],
                                   device float *y_out [[buffer(6)]],
                                   constant Ink2MoeParam &p [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count <= INK2_MV_MAX_M) return;

  threadgroup bfloat ws[INK2_MM_TILE * INK2_MM_TILE];
  constexpr auto desc = matmul2d_descriptor(
      INK2_MM_TILE, INK2_MM_TILE, INK2_MM_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<INK2_MM_SIMDGROUPS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)h_bf16, dextents<int32_t, 2>(p.k, p.npairs + INK2_MM_TILE));
  tensor<threadgroup bfloat, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> tw(
      ws, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * INK2_MM_TILE;
  uint w_row_bytes = p.k / 2u, blocks = p.k / INK2_NVFP4_BLOCK;
  device const uchar *we = w2 + (ulong)tl.expert * p.n * w_row_bytes;
  device const uchar *se = s2 + (ulong)tl.expert * p.n * blocks;
  float ef = factor2[tl.expert];

  for (uint k0 = 0; k0 < p.k; k0 += INK2_MM_TILE) {
    ink2_stage_nvfp4_tile(we, se, n0, k0, p.n, p.k, ws, t);
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
        uint pair = tl.start + r;
        y_out[(ulong)pair * p.n + col] = bf(c[i] * ef) * pw[pair];
      }
    }
  }
}

// Tiled MPP MXFP8 w13 + softcapped SwiGLU for layers 20..23 (tile count > 8).
// Grid: x = n / 64, y = ntiles. Threadgroup: 128 threads (4 simdgroups).
kernel void ink2_moe_mxfp8_up_mm(device const uchar *w13 [[buffer(0)]],
                                 device const uchar *s13 [[buffer(1)]],
                                 device const bfloat *x_bf16 [[buffer(2)]],
                                 device const Ink2MoeTile *tiles [[buffer(3)]],
                                 device bfloat *h_out [[buffer(4)]],
                                 constant Ink2MoeParam &p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count <= INK2_MV_MAX_M) return;

  threadgroup bfloat ws[INK2_MM_TILE * INK2_MM_TILE];
  threadgroup float out_tile[INK2_MM_TILE * INK2_MM_TILE];
  constexpr auto desc = matmul2d_descriptor(
      INK2_MM_TILE, INK2_MM_TILE, INK2_MM_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<INK2_MM_SIMDGROUPS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)x_bf16, dextents<int32_t, 2>(p.k, p.npairs + INK2_MM_TILE));
  tensor<threadgroup bfloat, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> tw(
      ws, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * INK2_MM_TILE;
  uint blocks = p.k / INK2_MXFP8_BLOCK;
  device const uchar *we = w13 + (ulong)tl.expert * p.n * p.k;
  device const uchar *se = s13 + (ulong)tl.expert * p.n * blocks;

  for (uint k0 = 0; k0 < p.k; k0 += INK2_MM_TILE) {
    ink2_stage_mxfp8_tile(we, se, n0, k0, p.n, p.k, ws, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = tx.slice(k0, tl.start);
    op.run(a, tw, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  tensor<threadgroup float, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> to(
      out_tile, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  c.store(to);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint inter = p.n / 2u, pair_col0 = n0 / 2u;
  constexpr uint HALF_TILE = INK2_MM_TILE / 2u;
  for (uint idx = t; idx < tl.count * HALF_TILE; idx += INK2_MM_SIMDGROUPS * INK2_SIMD) {
    uint r = idx / HALF_TILE, j = idx % HALF_TILE;
    float g = bf(out_tile[r * INK2_MM_TILE + 2u * j]);
    float u = bf(out_tile[r * INK2_MM_TILE + 2u * j + 1u]);
    h_out[(ulong)(tl.start + r) * inter + pair_col0 + j] =
        bfloat(ink2_softcapped_swiglu(g, u, p.gate_cap, p.up_cap));
  }
}

// Tiled MPP MXFP8 w2 for layers 20..23 (tile count > 8, up to 64).
// Grid: x = n / 64, y = ntiles. Threadgroup: 128 threads (4 simdgroups).
kernel void ink2_moe_mxfp8_down_mm(device const uchar *w2 [[buffer(0)]],
                                   device const uchar *s2 [[buffer(1)]],
                                   device const bfloat *h_bf16 [[buffer(2)]],
                                   device const float *pw [[buffer(3)]],
                                   device const Ink2MoeTile *tiles [[buffer(4)]],
                                   device float *y_out [[buffer(5)]],
                                   constant Ink2MoeParam &p [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
  Ink2MoeTile tl = tiles[tg.y];
  if (tl.count <= INK2_MV_MAX_M) return;

  threadgroup bfloat ws[INK2_MM_TILE * INK2_MM_TILE];
  constexpr auto desc = matmul2d_descriptor(
      INK2_MM_TILE, INK2_MM_TILE, INK2_MM_TILE, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<INK2_MM_SIMDGROUPS>> op;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx(
      (device bfloat *)h_bf16, dextents<int32_t, 2>(p.k, p.npairs + INK2_MM_TILE));
  tensor<threadgroup bfloat, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>, tensor_inline> tw(
      ws, extents<int32_t, INK2_MM_TILE, INK2_MM_TILE>());
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0.0f;

  uint n0 = tg.x * INK2_MM_TILE;
  uint blocks = p.k / INK2_MXFP8_BLOCK;
  device const uchar *we = w2 + (ulong)tl.expert * p.n * p.k;
  device const uchar *se = s2 + (ulong)tl.expert * p.n * blocks;

  for (uint k0 = 0; k0 < p.k; k0 += INK2_MM_TILE) {
    ink2_stage_mxfp8_tile(we, se, n0, k0, p.n, p.k, ws, t);
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
        uint pair = tl.start + r;
        y_out[(ulong)pair * p.n + col] = bf(c[i]) * pw[pair];
      }
    }
  }
}

// Sums each token's top-k routed expert outputs in FP32 and rounds to BF16
// (modeling_inkling2.py Inkling2Experts). Grid: x = dim, y = rows.
kernel void ink2_moe_combine(device const float *y_pair [[buffer(0)]],
                             device const int *tp [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant Ink2MoeCombineParam &p [[buffer(3)]],
                             uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.dim || i.y >= p.rows) return;
  float sum = 0.0f;
  for (uint k = 0; k < p.topk; k++) {
    int pair = tp[(ulong)i.y * p.topk + k];
    sum += y_pair[(ulong)pair * p.dim + i.x];
  }
  out[(ulong)i.y * p.dim + i.x] = bf(sum);
}

// Residual addition: out = a + b, optionally rounded to BF16.
// Grid: total elements.
kernel void ink2_add(device const float *a [[buffer(0)]],
                     device const float *b [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant uint2 &count_round [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= count_round.x) return;
  float v = a[i] + b[i];
  out[i] = count_round.y ? bf(v) : v;
}

// Final logit softcap (modeling_inkling2.py Inkling2Softcap):
// logits = cap * tanh(logits / cap) on rows of `vocab` unpadded logits.
// Grid: x = vocab, y = rows.
kernel void ink2_logit_softcap(device float *logits [[buffer(0)]],
                               constant Ink2SoftcapParam &p [[buffer(1)]],
                               uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.vocab || i.y >= p.rows) return;
  ulong idx = (ulong)i.y * p.stride + i.x;
  logits[idx] = p.cap * tanh(logits[idx] / p.cap);
}
