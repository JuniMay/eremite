// Inkling-2-Mini MTP (DSpark/DFlare + Markov RNN) Metal kernels.
//
// Shapes match config.json mtp_config:
//   3 draft layers (local SWA 1024 on layers 0, 2; global on layer 1),
//   hidden = 2304, heads = 24, kv_heads = 8, head_dim = 128, inter = 9216,
//   block_size = 8 (1 anchor + 7 mask slots), markov_rank = 256,
//   rope_theta = 1e6 (NeoX split-half), 4 target layer taps [5, 23, 41, 48].

// Model geometry constants for Inkling-2-Mini MTP.
constant uint INK2_MTP_DIM = 2304u;
constant uint INK2_MTP_HEADS = 24u;
constant uint INK2_MTP_KV_HEADS = 8u;
constant uint INK2_MTP_GQA = INK2_MTP_HEADS / INK2_MTP_KV_HEADS;
constant uint INK2_MTP_HEAD_DIM = 128u;
constant uint INK2_MTP_HALF_HEAD = INK2_MTP_HEAD_DIM / 2u;
constant uint INK2_MTP_KV_DIM = INK2_MTP_KV_HEADS * INK2_MTP_HEAD_DIM;
constant uint INK2_MTP_Q_DIM = INK2_MTP_HEADS * INK2_MTP_HEAD_DIM;
constant uint INK2_MTP_MARKOV_RANK = 256u;
constant uint INK2_MTP_RNN_IN = 2u * INK2_MTP_MARKOV_RANK + 2u * INK2_MTP_DIM;
constant uint INK2_MTP_VEC4_LANES = 4u;
constant uint INK2_MTP_SIMD_WIDTH = 32u;
constant uint INK2_MTP_ROW_THREADS = 256u;
constant uint INK2_MTP_ROW_SGS = INK2_MTP_ROW_THREADS / INK2_MTP_SIMD_WIDTH;
constant uint INK2_MTP_HEAD_SGS = INK2_MTP_HALF_HEAD / INK2_MTP_SIMD_WIDTH;
constant uint INK2_MTP_ATTEND_SGS = 4u;
constant uint INK2_MTP_PART_FLOATS = INK2_MTP_HEAD_DIM + 2u;
constant uint INK2_MTP_BF16_SHIFT = 16u;
constant float INK2_MTP_ATTN_SCALE = 0.08838834764831845f; // 1 / sqrt(128)
constant uint INK2_MTP_FUSE_MAX_ELEMS = 12u; // Covers dim up to 3072.

// Converts raw BF16 bit pattern to FP32.
static inline float ink2_mtp_bf16_to_f32(ushort bits) {
  return as_type<float>(uint(bits) << INK2_MTP_BF16_SHIFT);
}

// Per-row target address for batched f_rnn ring writes.
struct Ink2MtpFrnnTarget {
  ulong ring;
  uint pos;
  uint pad;
};

// Per-row target addresses for batched MTP context KV writes.
struct Ink2MtpKvTarget {
  ulong k;
  ulong v;
  uint pos;
  uint ring_rows;
};

// Parameters for ink2_mtp_fuse_norm.
struct Ink2MtpFuseParam {
  float w0;
  float w1;
  float w2;
  float w3;
  uint rows;
  uint tap_stride;
  uint dim;
  uint pos0;
  uint frnn_ring;
  uint batched;
  float eps;
};

// Parameters for ink2_mtp_ctx_kv_store.
struct Ink2MtpCtxKvParam {
  uint rows;
  uint pos0;
  uint ring_rows;
  uint batched;
  float rope_theta;
  float eps;
  uint page_shift;
  uint k_slice_off;
  uint v_slice_off;
};

// Parameters for ink2_mtp_block_init.
struct Ink2MtpBlockInitParam {
  uint anchor;
  uint dim;
  float eps;
};

// Parameters for ink2_mtp_block_qk_prep.
struct Ink2MtpBlockQkParam {
  uint pos0;
  float rope_theta;
  float eps;
};

// Parameters for ink2_mtp_attend_part and ink2_mtp_attend_combine.
struct Ink2MtpAttendParam {
  uint pos0;
  uint block;
  uint ring_rows;
  uint window;
  uint nparts;
  uint part_stride;
  uint page_shift;
  uint k_slice_off;
  uint v_slice_off;
};

// Parameters for ink2_mtp_markov_concat, gru, and add.
struct Ink2MtpMarkovParam {
  uint step_idx;
  uint pos0;
  uint frnn_ring;
  uint dim;
  uint rank;
};

// Mixes 4 target taps with softmax weights p.w0..p.w3 and applies weighted
// RMSNorm. Writes contiguous rows when frnn_ring == 0, or into the rolling
// f_rnn ring when frnn_ring > 0.
// Grid: (rows, 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_fuse_norm(
    device const float *taps [[buffer(0)]],
    device const float *norm_w [[buffer(1)]],
    device float *out [[buffer(2)]],
    device const Ink2MtpFrnnTarget *tgt [[buffer(3)]],
    constant Ink2MtpFuseParam &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint r = tg;
  if (r >= p.rows) return;
  uint dim = p.dim;
  ulong tap_floats = ulong(p.tap_stride) * ulong(dim);
  device const float *t0 = taps + ulong(r) * dim;
  device const float *t1 = t0 + tap_floats;
  device const float *t2 = t1 + tap_floats;
  device const float *t3 = t2 + tap_floats;

  float vals[INK2_MTP_FUSE_MAX_ELEMS];
  float s2 = 0.0f;
  uint k = 0;
  for (uint d = tid; d < dim; d += INK2_MTP_ROW_THREADS, k++) {
    float v = p.w0 * t0[d] + p.w1 * t1[d] + p.w2 * t2[d] + p.w3 * t3[d];
    vals[k] = v;
    s2 += v * v;
  }
  s2 = simd_sum(s2);
  threadgroup float sm[INK2_MTP_ROW_SGS];
  if (lane == 0) sm[sg] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float tot = lane < INK2_MTP_ROW_SGS ? sm[lane] : 0.0f;
    tot = simd_sum(tot);
    if (lane == 0) sm[0] = rsqrt(tot / float(dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = sm[0];

  device float *dst;
  if (p.frnn_ring == 0u) {
    dst = out + ulong(r) * dim;
  } else if (p.batched) {
    Ink2MtpFrnnTarget tr = tgt[r];
    uint slot = tr.pos % p.frnn_ring;
    dst = (device float *)tr.ring + ulong(slot) * dim;
  } else {
    uint slot = (p.pos0 + r) % p.frnn_ring;
    dst = out + ulong(slot) * dim;
  }

  k = 0;
  for (uint d = tid; d < dim; d += INK2_MTP_ROW_THREADS, k++) {
    dst[d] = vals[k] * inv * norm_w[d];
  }
}

// Applies per-head RMSNorm and NeoX split-half RoPE to context K rows and
// stores BF16 K and V rows into one draft layer's context KV cache.
// Grid: (8, rows, 1), TG: (64, 1, 1).
kernel void ink2_mtp_ctx_kv_store(
    device const float *k_raw [[buffer(0)]],
    device const float *v_raw [[buffer(1)]],
    device const float *k_norm [[buffer(2)]],
    device bfloat *k_dst [[buffer(3)]],
    device bfloat *v_dst [[buffer(4)]],
    device const Ink2MtpKvTarget *tgt [[buffer(5)]],
    constant Ink2MtpCtxKvParam &p [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint kh = tg.x;
  uint r = tg.y;
  if (kh >= INK2_MTP_KV_HEADS || r >= p.rows) return;
  uint d0 = tid;
  uint d1 = tid + INK2_MTP_HALF_HEAD;
  ulong in_off = ulong(r) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
  float k0 = k_raw[in_off + d0];
  float k1 = k_raw[in_off + d1];
  float v0 = v_raw[in_off + d0];
  float v1 = v_raw[in_off + d1];

  float s2 = simd_sum(k0 * k0 + k1 * k1);
  threadgroup float sm[INK2_MTP_HEAD_SGS];
  if (lane == 0) sm[sg] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = rsqrt((sm[0] + sm[1]) / float(INK2_MTP_HEAD_DIM) + p.eps);

  float x0 = k0 * inv * k_norm[d0];
  float x1 = k1 * inv * k_norm[d1];

  uint pos;
  device bfloat *kd;
  device bfloat *vd;
  if (p.page_shift) {
    pos = p.batched ? tgt[r].pos : (p.pos0 + r);
    device const ulong *addrs =
        p.batched ? (device const ulong *)tgt[r].k : (device const ulong *)k_dst;
    ulong page_base = addrs[pos >> p.page_shift];
    uint slot = pos & ((1u << p.page_shift) - 1u);
    ulong kv_off = ulong(slot) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
    kd = (device bfloat *)(page_base + p.k_slice_off) + kv_off;
    vd = (device bfloat *)(page_base + p.v_slice_off) + kv_off;
  } else if (p.batched) {
    Ink2MtpKvTarget tr = tgt[r];
    pos = tr.pos;
    uint slot = tr.ring_rows ? (pos % tr.ring_rows) : pos;
    ulong kv_off = ulong(slot) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
    kd = (device bfloat *)tr.k + kv_off;
    vd = (device bfloat *)tr.v + kv_off;
  } else {
    pos = p.pos0 + r;
    uint slot = p.ring_rows ? (pos % p.ring_rows) : pos;
    ulong kv_off = ulong(slot) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
    kd = k_dst + kv_off;
    vd = v_dst + kv_off;
  }

  float freq = pow(p.rope_theta, -float(2u * d0) / float(INK2_MTP_HEAD_DIM));
  float ang = float(pos) * freq;
  float cs = cos(ang);
  float sn = sin(ang);
  kd[d0] = bfloat(x0 * cs - x1 * sn);
  kd[d1] = bfloat(x0 * sn + x1 * cs);
  vd[d0] = bfloat(v0);
  vd[d1] = bfloat(v1);
}

// Applies per-head RMSNorm and NeoX split-half RoPE to context K rows and
// stores packed FP8 E4M3 K and V rows with per-head FP16 scales.
// Grid: (8, rows, 1), TG: (64, 1, 1).
kernel void ink2_mtp_ctx_kv_store_fp8(
    device const float *k_raw [[buffer(0)]],
    device const float *v_raw [[buffer(1)]],
    device const float *k_norm [[buffer(2)]],
    device uchar *k_dst [[buffer(3)]],
    device uchar *v_dst [[buffer(4)]],
    device const Ink2MtpKvTarget *tgt [[buffer(5)]],
    constant Ink2MtpCtxKvParam &p [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint kh = tg.x;
  uint r = tg.y;
  if (kh >= INK2_MTP_KV_HEADS || r >= p.rows) return;
  uint d0 = tid;
  uint d1 = tid + INK2_MTP_HALF_HEAD;
  ulong in_off = ulong(r) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
  float k0 = k_raw[in_off + d0];
  float k1 = k_raw[in_off + d1];
  float v0 = v_raw[in_off + d0];
  float v1 = v_raw[in_off + d1];

  float s2 = simd_sum(k0 * k0 + k1 * k1);
  threadgroup float sm[INK2_MTP_HEAD_SGS];
  threadgroup float sm_k[INK2_MTP_HEAD_SGS];
  threadgroup float sm_v[INK2_MTP_HEAD_SGS];
  if (lane == 0) sm[sg] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = rsqrt((sm[0] + sm[1]) / float(INK2_MTP_HEAD_DIM) + p.eps);

  float x0 = k0 * inv * k_norm[d0];
  float x1 = k1 * inv * k_norm[d1];

  uint pos;
  device uchar *krow;
  device uchar *vrow;
  if (p.page_shift) {
    pos = p.batched ? tgt[r].pos : (p.pos0 + r);
    device const ulong *addrs =
        p.batched ? (device const ulong *)tgt[r].k : (device const ulong *)k_dst;
    ulong page_base = addrs[pos >> p.page_shift];
    uint slot = pos & ((1u << p.page_shift) - 1u);
    ulong row_off = ulong(slot) * INK2_FP8_KV_ROW_BYTES;
    krow = (device uchar *)(page_base + p.k_slice_off) + row_off;
    vrow = (device uchar *)(page_base + p.v_slice_off) + row_off;
  } else if (p.batched) {
    Ink2MtpKvTarget tr = tgt[r];
    pos = tr.pos;
    uint slot = tr.ring_rows ? (pos % tr.ring_rows) : pos;
    ulong row_off = ulong(slot) * INK2_FP8_KV_ROW_BYTES;
    krow = (device uchar *)tr.k + row_off;
    vrow = (device uchar *)tr.v + row_off;
  } else {
    pos = p.pos0 + r;
    uint slot = p.ring_rows ? (pos % p.ring_rows) : pos;
    ulong row_off = ulong(slot) * INK2_FP8_KV_ROW_BYTES;
    krow = k_dst + row_off;
    vrow = v_dst + row_off;
  }

  float freq = pow(p.rope_theta, -float(2u * d0) / float(INK2_MTP_HEAD_DIM));
  float ang = float(pos) * freq;
  float cs = cos(ang);
  float sn = sin(ang);
  float rk0 = x0 * cs - x1 * sn;
  float rk1 = x0 * sn + x1 * cs;

  float kmax = simd_max(max(fabs(rk0), fabs(rk1)));
  float vmax = simd_max(max(fabs(v0), fabs(v1)));
  if (lane == 0) {
    sm_k[sg] = kmax;
    sm_v[sg] = vmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float k_amax = max(sm_k[0], sm_k[1]);
  float v_amax = max(sm_v[0], sm_v[1]);

  half ks256 = ink2_fp8_head_scale256(k_amax);
  half vs256 = ink2_fp8_head_scale256(v_amax);
  float k_inv = k_amax > 0.0f ? (INK2_FP8_SCALE / float(ks256)) : 0.0f;
  float v_inv = v_amax > 0.0f ? (INK2_FP8_SCALE / float(vs256)) : 0.0f;

  device uchar *kd = krow + ulong(kh) * INK2_MTP_HEAD_DIM;
  device uchar *vd = vrow + ulong(kh) * INK2_MTP_HEAD_DIM;
  kd[d0] = k_amax > 0.0f ? ink2_fp8_encode(rk0 * k_inv) : 0u;
  kd[d1] = k_amax > 0.0f ? ink2_fp8_encode(rk1 * k_inv) : 0u;
  vd[d0] = v_amax > 0.0f ? ink2_fp8_encode(v0 * v_inv) : 0u;
  vd[d1] = v_amax > 0.0f ? ink2_fp8_encode(v1 * v_inv) : 0u;
  if (tid == 0u) {
    ((device half *)(krow + INK2_FP8_KV_SCALES_OFF))[kh] = ks256;
    ((device half *)(vrow + INK2_FP8_KV_SCALES_OFF))[kh] = vs256;
  }
}

// Initializes the 8-row draft block input: row 0 is RMSNorm(embed[anchor],
// embed_norm) and rows 1..7 are mask_embed.
// Grid: (block_size, 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_block_init(
    device const ushort *embed_w [[buffer(0)]],
    device const float *embed_norm [[buffer(1)]],
    device const float *mask_embed [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Ink2MtpBlockInitParam &p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint r = tg;
  uint dim = p.dim;
  device float *dst = out + ulong(r) * dim;
  if (r > 0u) {
    for (uint d = tid; d < dim; d += INK2_MTP_ROW_THREADS) {
      dst[d] = mask_embed[d];
    }
    return;
  }
  device const ushort *src = embed_w + ulong(p.anchor) * dim;
  float vals[INK2_MTP_FUSE_MAX_ELEMS];
  float s2 = 0.0f;
  uint k = 0;
  for (uint d = tid; d < dim; d += INK2_MTP_ROW_THREADS, k++) {
    float v = ink2_mtp_bf16_to_f32(src[d]);
    vals[k] = v;
    s2 += v * v;
  }
  s2 = simd_sum(s2);
  threadgroup float sm[INK2_MTP_ROW_SGS];
  if (lane == 0) sm[sg] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float tot = lane < INK2_MTP_ROW_SGS ? sm[lane] : 0.0f;
    tot = simd_sum(tot);
    if (lane == 0) sm[0] = rsqrt(tot / float(dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = sm[0];
  k = 0;
  for (uint d = tid; d < dim; d += INK2_MTP_ROW_THREADS, k++) {
    dst[d] = vals[k] * inv * embed_norm[d];
  }
}

// Applies per-head RMSNorm and NeoX split-half RoPE to draft-block Q and K and
// casts Q, K, V to BF16 for attention.
// Grid: (24 + 8, block_size, 1), TG: (64, 1, 1).
kernel void ink2_mtp_block_qk_prep(
    device const float *q_raw [[buffer(0)]],
    device const float *k_raw [[buffer(1)]],
    device const float *v_raw [[buffer(2)]],
    device const float *q_norm [[buffer(3)]],
    device const float *k_norm [[buffer(4)]],
    device bfloat *q_blk [[buffer(5)]],
    device bfloat *k_blk [[buffer(6)]],
    device bfloat *v_blk [[buffer(7)]],
    constant Ink2MtpBlockQkParam &p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint hx = tg.x;
  uint r = tg.y;
  uint d0 = tid;
  uint d1 = tid + INK2_MTP_HALF_HEAD;
  uint pos = p.pos0 + r;
  float freq = pow(p.rope_theta, -float(2u * d0) / float(INK2_MTP_HEAD_DIM));
  float ang = float(pos) * freq;
  float cs = cos(ang);
  float sn = sin(ang);

  threadgroup float sm[INK2_MTP_HEAD_SGS];
  if (hx < INK2_MTP_HEADS) {
    ulong off = ulong(r) * INK2_MTP_Q_DIM + ulong(hx) * INK2_MTP_HEAD_DIM;
    float q0 = q_raw[off + d0];
    float q1 = q_raw[off + d1];
    float s2 = simd_sum(q0 * q0 + q1 * q1);
    if (lane == 0) sm[sg] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((sm[0] + sm[1]) / float(INK2_MTP_HEAD_DIM) + p.eps);
    float x0 = q0 * inv * q_norm[d0];
    float x1 = q1 * inv * q_norm[d1];
    q_blk[off + d0] = bfloat(x0 * cs - x1 * sn);
    q_blk[off + d1] = bfloat(x0 * sn + x1 * cs);
    return;
  }

  uint kh = hx - INK2_MTP_HEADS;
  ulong off = ulong(r) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM;
  float k0 = k_raw[off + d0];
  float k1 = k_raw[off + d1];
  float v0 = v_raw[off + d0];
  float v1 = v_raw[off + d1];
  float s2 = simd_sum(k0 * k0 + k1 * k1);
  if (lane == 0) sm[sg] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = rsqrt((sm[0] + sm[1]) / float(INK2_MTP_HEAD_DIM) + p.eps);
  float x0 = k0 * inv * k_norm[d0];
  float x1 = k1 * inv * k_norm[d1];
  k_blk[off + d0] = bfloat(x0 * cs - x1 * sn);
  k_blk[off + d1] = bfloat(x0 * sn + x1 * cs);
  v_blk[off + d0] = bfloat(v0);
  v_blk[off + d1] = bfloat(v1);
}

// Noncausal FlashDecoding attention partition over committed context KV
// [ctx_lo, pos0) followed by all draft block rows [0, block). Processes all 3
// GQA query heads sharing one KV head in a single pass.
// Grid: (8, nparts, block), TG: (128, 1, 1).
kernel void ink2_mtp_attend_part(
    device const bfloat *q_blk [[buffer(0)]],
    device const bfloat *k_ctx [[buffer(1)]],
    device const bfloat *v_ctx [[buffer(2)]],
    device const bfloat *k_blk [[buffer(3)]],
    device const bfloat *v_blk [[buffer(4)]],
    device float *part_out [[buffer(5)]],
    constant Ink2MtpAttendParam &p [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint kh = tg.x;
  uint part = tg.y;
  uint r = tg.z;
  uint ctx_lo = (p.window > 0u && p.pos0 > p.window) ? (p.pos0 - p.window) : 0u;
  uint ctx_len = p.pos0 - ctx_lo;
  uint total_k = ctx_len + p.block;
  uint per = (total_k + p.nparts - 1u) / p.nparts;
  uint k_beg = part * per;
  uint k_end = min(total_k, k_beg + per);

  uint d_base = lane * INK2_MTP_VEC4_LANES;
  float4 qv[INK2_MTP_GQA];
  float m[INK2_MTP_GQA];
  float l[INK2_MTP_GQA];
  float4 ov[INK2_MTP_GQA];
  for (uint g = 0u; g < INK2_MTP_GQA; g++) {
    uint h = kh * INK2_MTP_GQA + g;
    device const bfloat *qr =
        q_blk + ulong(r) * INK2_MTP_Q_DIM + ulong(h) * INK2_MTP_HEAD_DIM + d_base;
    qv[g] = float4(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])) *
            INK2_MTP_ATTN_SCALE;
    m[g] = -INFINITY;
    l[g] = 0.0f;
    ov[g] = float4(0.0f);
  }

  device const ulong *page_addrs = p.page_shift ? (device const ulong *)k_ctx : nullptr;
  uint page_mask = p.page_shift ? ((1u << p.page_shift) - 1u) : 0u;
  uint cached_page = 0xFFFFFFFFu;
  ulong cached_page_base = 0u;

  for (uint ki = k_beg + sg; ki < k_end; ki += INK2_MTP_ATTEND_SGS) {
    device const bfloat *kr;
    device const bfloat *vr;
    if (ki < ctx_len) {
      uint pos = ctx_lo + ki;
      if (p.page_shift) {
        uint p_idx = pos >> p.page_shift;
        if (p_idx != cached_page) {
          cached_page = p_idx;
          cached_page_base = page_addrs[p_idx];
        }
        ulong kv_off = ulong(pos & page_mask) * INK2_MTP_KV_DIM +
                       ulong(kh) * INK2_MTP_HEAD_DIM + d_base;
        kr = (device const bfloat *)(cached_page_base + p.k_slice_off) + kv_off;
        vr = (device const bfloat *)(cached_page_base + p.v_slice_off) + kv_off;
      } else {
        uint slot = p.ring_rows ? (pos % p.ring_rows) : pos;
        ulong kv_off =
            ulong(slot) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM + d_base;
        kr = k_ctx + kv_off;
        vr = v_ctx + kv_off;
      }
    } else {
      uint bi = ki - ctx_len;
      ulong kv_off =
          ulong(bi) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM + d_base;
      kr = k_blk + kv_off;
      vr = v_blk + kv_off;
    }
    float4 kv = float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]));
    float4 vv = float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
    for (uint g = 0u; g < INK2_MTP_GQA; g++) {
      float score = simd_sum(dot(qv[g], kv));
      float mn = max(m[g], score);
      float alpha = (l[g] > 0.0f) ? exp(m[g] - mn) : 0.0f;
      float beta = exp(score - mn);
      ov[g] = ov[g] * alpha + vv * beta;
      l[g] = l[g] * alpha + beta;
      m[g] = mn;
    }
  }

  threadgroup float sm_m[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS];
  threadgroup float sm_l[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS];
  threadgroup float4 sm_o[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS][INK2_MTP_SIMD_WIDTH];
  for (uint g = 0u; g < INK2_MTP_GQA; g++) {
    if (lane == 0) {
      sm_m[g][sg] = m[g];
      sm_l[g][sg] = l[g];
    }
    sm_o[g][sg][lane] = ov[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg < INK2_MTP_GQA) {
    uint g = sg;
    uint h = kh * INK2_MTP_GQA + g;
    float mmax = -INFINITY;
    for (uint s = 0u; s < INK2_MTP_ATTEND_SGS; s++) {
      if (sm_l[g][s] > 0.0f) mmax = max(mmax, sm_m[g][s]);
    }
    float lsum = 0.0f;
    float4 osum = float4(0.0f);
    for (uint s = 0u; s < INK2_MTP_ATTEND_SGS; s++) {
      float ls = sm_l[g][s];
      if (ls > 0.0f) {
        float w = exp(sm_m[g][s] - mmax);
        lsum += ls * w;
        osum += sm_o[g][s][lane] * w;
      }
    }
    device float *dst =
        part_out + ((ulong(r) * INK2_MTP_HEADS + h) * p.part_stride + part) *
                       INK2_MTP_PART_FLOATS;
    dst[d_base + 0u] = osum.x;
    dst[d_base + 1u] = osum.y;
    dst[d_base + 2u] = osum.z;
    dst[d_base + 3u] = osum.w;
    if (lane == 0) {
      dst[INK2_MTP_HEAD_DIM] = mmax;
      dst[INK2_MTP_HEAD_DIM + 1u] = lsum;
    }
  }
}

// Noncausal FlashDecoding attention partition over packed FP8 E4M3 context KV
// [ctx_lo, pos0) followed by BF16 draft block rows [0, block).
// Grid: (8, nparts, block), TG: (128, 1, 1).
kernel void ink2_mtp_attend_part_fp8(
    device const bfloat *q_blk [[buffer(0)]],
    device const uchar *k_ctx [[buffer(1)]],
    device const uchar *v_ctx [[buffer(2)]],
    device const bfloat *k_blk [[buffer(3)]],
    device const bfloat *v_blk [[buffer(4)]],
    device float *part_out [[buffer(5)]],
    constant Ink2MtpAttendParam &p [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint kh = tg.x;
  uint part = tg.y;
  uint r = tg.z;
  uint ctx_lo = (p.window > 0u && p.pos0 > p.window) ? (p.pos0 - p.window) : 0u;
  uint ctx_len = p.pos0 - ctx_lo;
  uint total_k = ctx_len + p.block;
  uint per = (total_k + p.nparts - 1u) / p.nparts;
  uint k_beg = part * per;
  uint k_end = min(total_k, k_beg + per);

  uint d_base = lane * INK2_MTP_VEC4_LANES;
  float4 qv[INK2_MTP_GQA];
  float m[INK2_MTP_GQA];
  float l[INK2_MTP_GQA];
  float4 ov[INK2_MTP_GQA];
  for (uint g = 0u; g < INK2_MTP_GQA; g++) {
    uint h = kh * INK2_MTP_GQA + g;
    device const bfloat *qr =
        q_blk + ulong(r) * INK2_MTP_Q_DIM + ulong(h) * INK2_MTP_HEAD_DIM + d_base;
    qv[g] = float4(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])) *
            INK2_MTP_ATTN_SCALE;
    m[g] = -INFINITY;
    l[g] = 0.0f;
    ov[g] = float4(0.0f);
  }

  device const ulong *page_addrs = p.page_shift ? (device const ulong *)k_ctx : nullptr;
  uint page_mask = p.page_shift ? ((1u << p.page_shift) - 1u) : 0u;
  uint cached_page = 0xFFFFFFFFu;
  ulong cached_page_base = 0u;

  for (uint ki = k_beg + sg; ki < k_end; ki += INK2_MTP_ATTEND_SGS) {
    float4 kv;
    float4 vv;
    if (ki < ctx_len) {
      uint pos = ctx_lo + ki;
      device const uchar *krow;
      device const uchar *vrow;
      if (p.page_shift) {
        uint p_idx = pos >> p.page_shift;
        if (p_idx != cached_page) {
          cached_page = p_idx;
          cached_page_base = page_addrs[p_idx];
        }
        ulong row_off = ulong(pos & page_mask) * INK2_FP8_KV_ROW_BYTES;
        krow = (device const uchar *)(cached_page_base + p.k_slice_off) + row_off;
        vrow = (device const uchar *)(cached_page_base + p.v_slice_off) + row_off;
      } else {
        uint slot = p.ring_rows ? (pos % p.ring_rows) : pos;
        ulong row_off = ulong(slot) * INK2_FP8_KV_ROW_BYTES;
        krow = k_ctx + row_off;
        vrow = v_ctx + row_off;
      }
      float ksc = float(((device const half *)(krow + INK2_FP8_KV_SCALES_OFF))[kh]);
      float vsc = float(((device const half *)(vrow + INK2_FP8_KV_SCALES_OFF))[kh]);
      kv = fp8x4(((device const uint *)(krow + ulong(kh) * INK2_MTP_HEAD_DIM))[lane]) * ksc;
      vv = fp8x4(((device const uint *)(vrow + ulong(kh) * INK2_MTP_HEAD_DIM))[lane]) * vsc;
    } else {
      uint bi = ki - ctx_len;
      ulong kv_off =
          ulong(bi) * INK2_MTP_KV_DIM + ulong(kh) * INK2_MTP_HEAD_DIM + d_base;
      device const bfloat *kr = k_blk + kv_off;
      device const bfloat *vr = v_blk + kv_off;
      kv = float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]));
      vv = float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
    }
    for (uint g = 0u; g < INK2_MTP_GQA; g++) {
      float score = simd_sum(dot(qv[g], kv));
      float mn = max(m[g], score);
      float alpha = (l[g] > 0.0f) ? exp(m[g] - mn) : 0.0f;
      float beta = exp(score - mn);
      ov[g] = ov[g] * alpha + vv * beta;
      l[g] = l[g] * alpha + beta;
      m[g] = mn;
    }
  }

  threadgroup float sm_m[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS];
  threadgroup float sm_l[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS];
  threadgroup float4 sm_o[INK2_MTP_GQA][INK2_MTP_ATTEND_SGS][INK2_MTP_SIMD_WIDTH];
  for (uint g = 0u; g < INK2_MTP_GQA; g++) {
    if (lane == 0) {
      sm_m[g][sg] = m[g];
      sm_l[g][sg] = l[g];
    }
    sm_o[g][sg][lane] = ov[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg < INK2_MTP_GQA) {
    uint g = sg;
    uint h = kh * INK2_MTP_GQA + g;
    float mmax = -INFINITY;
    for (uint s = 0u; s < INK2_MTP_ATTEND_SGS; s++) {
      if (sm_l[g][s] > 0.0f) mmax = max(mmax, sm_m[g][s]);
    }
    float lsum = 0.0f;
    float4 osum = float4(0.0f);
    for (uint s = 0u; s < INK2_MTP_ATTEND_SGS; s++) {
      float ls = sm_l[g][s];
      if (ls > 0.0f) {
        float w = exp(sm_m[g][s] - mmax);
        lsum += ls * w;
        osum += sm_o[g][s][lane] * w;
      }
    }
    device float *dst =
        part_out + ((ulong(r) * INK2_MTP_HEADS + h) * p.part_stride + part) *
                       INK2_MTP_PART_FLOATS;
    dst[d_base + 0u] = osum.x;
    dst[d_base + 1u] = osum.y;
    dst[d_base + 2u] = osum.z;
    dst[d_base + 3u] = osum.w;
    if (lane == 0) {
      dst[INK2_MTP_HEAD_DIM] = mmax;
      dst[INK2_MTP_HEAD_DIM + 1u] = lsum;
    }
  }
}

// Combines FlashDecoding attention partitions into attn_out [block, 3072].
// Grid: (24, block, 1), TG: (32, 1, 1).
kernel void ink2_mtp_attend_combine(
    device const float *part_in [[buffer(0)]],
    device float *attn_out [[buffer(1)]],
    constant Ink2MtpAttendParam &p [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint h = tg.x;
  uint r = tg.y;
  uint d_base = lane * INK2_MTP_VEC4_LANES;
  device const float *base =
      part_in + (ulong(r) * INK2_MTP_HEADS + h) * p.part_stride *
                    INK2_MTP_PART_FLOATS;
  float mmax = -INFINITY;
  for (uint i = 0u; i < p.nparts; i++) {
    device const float *pi = base + ulong(i) * INK2_MTP_PART_FLOATS;
    if (pi[INK2_MTP_HEAD_DIM + 1u] > 0.0f) {
      mmax = max(mmax, pi[INK2_MTP_HEAD_DIM]);
    }
  }
  float lsum = 0.0f;
  float4 osum = float4(0.0f);
  for (uint i = 0u; i < p.nparts; i++) {
    device const float *pi = base + ulong(i) * INK2_MTP_PART_FLOATS;
    float lp = pi[INK2_MTP_HEAD_DIM + 1u];
    if (lp > 0.0f) {
      float w = exp(pi[INK2_MTP_HEAD_DIM] - mmax);
      lsum += lp * w;
      osum += float4(pi[d_base + 0u], pi[d_base + 1u], pi[d_base + 2u],
                     pi[d_base + 3u]) *
              w;
    }
  }
  float inv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
  device float *dst =
      attn_out + ulong(r) * INK2_MTP_Q_DIM + ulong(h) * INK2_MTP_HEAD_DIM +
      d_base;
  dst[0] = osum.x * inv;
  dst[1] = osum.y * inv;
  dst[2] = osum.z * inv;
  dst[3] = osum.w * inv;
}

// Standard SwiGLU activation: out[i] = silu(gate[i]) * up[i].
kernel void ink2_mtp_swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  float g = gate[gid];
  out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// Assembles the 5120-element Markov RNN input [state, markov_w1[prev], h[i],
// f_rnn[(pos0 - 1) % frnn_ring]] for draft slot step_idx (1..7).
// Grid: (ceil(5120 / 256), 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_markov_concat(
    device const float *state [[buffer(0)]],
    device const ushort *markov_w1 [[buffer(1)]],
    device const int *tok_buf [[buffer(2)]],
    device const float *h_block [[buffer(3)]],
    device const float *frnn_ring [[buffer(4)]],
    device float *joint [[buffer(5)]],
    constant Ink2MtpMarkovParam &p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  uint r2 = 2u * p.rank;
  uint total = r2 + 2u * p.dim;
  if (gid >= total) return;
  if (gid < p.rank) {
    joint[gid] = (p.step_idx == 1u) ? 0.0f : state[gid];
  } else if (gid < r2) {
    int prev = tok_buf[p.step_idx - 1u];
    uint off = gid - p.rank;
    joint[gid] = ink2_mtp_bf16_to_f32(markov_w1[ulong(prev) * p.rank + off]);
  } else if (gid < r2 + p.dim) {
    uint off = gid - r2;
    joint[gid] = h_block[ulong(p.step_idx) * p.dim + off];
  } else {
    uint off = gid - (r2 + p.dim);
    uint slot = (p.pos0 - 1u) % p.frnn_ring;
    joint[gid] = frnn_ring[ulong(slot) * p.dim + off];
  }
}

// Applies the Markov GRU recurrence: splits proj + bias into (c0, c1, c2),
// updates state = cand + gate * (state - cand), and writes latent = tanh(c2).
// Grid: (1, 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_markov_gru(
    device const float *proj [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *latent [[buffer(3)]],
    constant Ink2MtpMarkovParam &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (tid >= p.rank) return;
  float c0 = proj[tid] + bias[tid];
  float c1 = proj[p.rank + tid] + bias[p.rank + tid];
  float c2 = proj[2u * p.rank + tid] + bias[2u * p.rank + tid];
  float gate = 1.0f / (1.0f + exp(-c0));
  float cand = tanh(c1);
  float prev_s = (p.step_idx == 1u) ? 0.0f : state[tid];
  state[tid] = cand + gate * (prev_s - cand);
  latent[tid] = tanh(c2);
}

// Adds the Markov correction and hidden_out bias to draft slot h[step_idx] so
// a single unembed matvec computes base + bias logits identically.
// Grid: (ceil(dim / 256), 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_markov_add(
    device const float *h_block [[buffer(0)]],
    device const float *corr [[buffer(1)]],
    device const float *hidden_bias [[buffer(2)]],
    device float *h_eff [[buffer(3)]],
    constant Ink2MtpMarkovParam &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.dim) return;
  h_eff[gid] =
      h_block[ulong(p.step_idx) * p.dim + gid] + corr[gid] + hidden_bias[gid];
}

// One threadgroup's winning (softcapped logit, token id) pair.
struct Ink2MtpArgmaxPair {
  float val;
  int idx;
};

// Parameters for fused unembed + softcap + first-maximum argmax.
struct Ink2MtpUnembedParam {
  uint vocab;
  uint k;
  float cap;
};

// Constants for fused unembed argmax (8 simdgroups x 4 rows = 32 rows per TG).
constant uint INK2_MTP_UNEMBED_ROWS_PER_SG = 4u;
constant uint INK2_MTP_UNEMBED_SGS = 8u;
constant uint INK2_MTP_UNEMBED_BF16_PER_U4 = 8u;

// Fused BF16 unembed + softcap + threadgroup first-maximum argmax (stage 1).
// Eliminates writing 200k FP32 logits to DRAM and the separate softcap pass.
// Grid: (ceil(vocab / 32), 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_unembed_argmax_part(
    device const ushort *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device Ink2MtpArgmaxPair *part_max [[buffer(2)]],
    constant Ink2MtpUnembedParam &p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg * INK2_MTP_UNEMBED_SGS + sg) * INK2_MTP_UNEMBED_ROWS_PER_SG;
  uint chunks = p.k / INK2_MTP_UNEMBED_BF16_PER_U4;
  float4 acc = float4(0.0f);
  if (row < p.vocab) {
    uint valid_rows = min(INK2_MTP_UNEMBED_ROWS_PER_SG, p.vocab - row);
    device const uint4 *wr = (device const uint4 *)(w + ulong(row) * p.k);
    device const float4 *xr = (device const float4 *)x;
    if (valid_rows == INK2_MTP_UNEMBED_ROWS_PER_SG) {
      for (uint c = lane; c < chunks; c += INK2_MTP_SIMD_WIDTH) {
        float4 x0 = xr[2u * c];
        float4 x1 = xr[2u * c + 1u];
#pragma unroll
        for (uint r = 0u; r < INK2_MTP_UNEMBED_ROWS_PER_SG; r++) {
          uint4 q = wr[r * chunks + c];
          float4 d = bf16x4(q.xy) * x0 + bf16x4(q.zw) * x1;
          acc[r] += d.x + d.y + d.z + d.w;
        }
      }
    } else {
      for (uint c = lane; c < chunks; c += INK2_MTP_SIMD_WIDTH) {
        float4 x0 = xr[2u * c];
        float4 x1 = xr[2u * c + 1u];
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

  threadgroup float sm_v[INK2_MTP_UNEMBED_SGS];
  threadgroup int sm_i[INK2_MTP_UNEMBED_SGS];
  if (lane == 0u) {
    float bv = -INFINITY;
    int bi = int(p.vocab);
    float inv_cap = 1.0f / p.cap;
    for (uint r = 0u; r < INK2_MTP_UNEMBED_ROWS_PER_SG; r++) {
      uint id = row + r;
      if (id < p.vocab) {
        float v = p.cap * tanh(acc[r] * inv_cap);
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
    for (uint s = 1u; s < INK2_MTP_UNEMBED_SGS; s++) {
      if (sm_v[s] > bv || (sm_v[s] == bv && sm_i[s] < bi)) {
        bv = sm_v[s];
        bi = sm_i[s];
      }
    }
    part_max[tg] = Ink2MtpArgmaxPair{bv, bi};
  }
}

// Reduces stage-1 threadgroup argmax pairs to one winning token id (stage 2).
// Grid: (1, 1, 1), TG: (256, 1, 1).
kernel void ink2_mtp_unembed_argmax_final(
    device const Ink2MtpArgmaxPair *part_max [[buffer(0)]],
    device int *out_tok [[buffer(1)]],
    constant uint &nparts [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (uint i = tid; i < nparts; i += INK2_MTP_ROW_THREADS) {
    Ink2MtpArgmaxPair cur = part_max[i];
    if (cur.val > bv || (cur.val == bv && cur.idx < bi)) {
      bv = cur.val;
      bi = cur.idx;
    }
  }
  for (uint off = INK2_MTP_SIMD_WIDTH / 2u; off > 0u; off >>= 1u) {
    float ov = simd_shuffle_xor(bv, off);
    int oi = simd_shuffle_xor(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  threadgroup float sm_v[INK2_MTP_ROW_SGS];
  threadgroup int sm_i[INK2_MTP_ROW_SGS];
  if (lane == 0u) {
    sm_v[sg] = bv;
    sm_i[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fbv = sm_v[0];
    int fbi = sm_i[0];
    for (uint s = 1u; s < INK2_MTP_ROW_SGS; s++) {
      if (sm_v[s] > fbv || (sm_v[s] == fbv && sm_i[s] < fbi)) {
        fbv = sm_v[s];
        fbi = sm_i[s];
      }
    }
    out_tok[0] = fbi;
  }
}
