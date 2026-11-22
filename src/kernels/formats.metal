// Number formats and small functions shared by every kernel source; the
// first source of the library (src/kernels/library.c).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// An FP8 E4M3 code as a float (no NaN codes in the weights).
inline float fp8(uchar b) {
  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f;
}
// An FP4 E2M1 code (the low 4 bits) as a float.
inline float fp4(uint nib) { return float(as_type<half>(ushort(((nib & 7u) << 9) | ((nib & 8u) << 12)))) * 16384.0f; }
// A BF16 value as a float.
inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
// Four BF16 values, the low half of each word first, as floats.
inline float4 bf16x4(uint2 q) {
  return float4(as_type<float>(q.x << 16), as_type<float>(q.x & 0xFFFF0000u),
                as_type<float>(q.y << 16), as_type<float>(q.y & 0xFFFF0000u));
}
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// The global top k is contained in the union of each SIMD group's top k.
// Equal scores keep the smaller original index. Scratch holds k entries per
// SIMD group; k is at most 16 for the routed-expert callers.
inline void group_topk(float key, uint index, uint k, threadgroup float *values,
                       threadgroup uint *indexes, device uint *out, threadgroup float *selected,
                       uint t, uint threads) {
  uint lane = t % 32, group = t / 32;
  for (uint rank = 0; rank < k; rank++) {
    float score = simd_max(key);
    uint winner = simd_min(key == score ? index : UINT_MAX);
    if (!lane) {
      values[group * k + rank] = score;
      indexes[group * k + rank] = winner;
    }
    if (index == winner) { key = -INFINITY; index = UINT_MAX; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!group) {
    uint at = 0;
    float score = lane < threads / 32 ? values[lane * k] : -INFINITY;
    uint candidate = lane < threads / 32 ? indexes[lane * k] : UINT_MAX;
    for (uint rank = 0; rank < k; rank++) {
      float maximum = simd_max(score);
      uint winner = simd_min(score == maximum ? candidate : UINT_MAX);
      if (!lane) { out[rank] = winner; selected[rank] = maximum; }
      if (candidate == winner) {
        at++;
        score = at < k ? values[lane * k + at] : -INFINITY;
        candidate = at < k ? indexes[lane * k + at] : UINT_MAX;
      }
    }
  }
}
// log(1 + e) without losing a small e (Metal has no log1p): Goldberg's
// correction of log(u) by e / (u - 1), u = 1 + e rounded.
inline float log1p_(float e) {
  float u = 1.0f + e;
  return u == 1.0f ? e : log(u) * e / (u - 1.0f);
}

// Tile geometry for 4-simdgroup cooperative 64x64 FlashAttention kernels.
constant constexpr uint FLASH_ATTN_64_TILE = 64,
                        FLASH_ATTN_64_HALF = FLASH_ATTN_64_TILE / 2,
                        FLASH_ATTN_64_VEC4 = 4,
                        FLASH_ATTN_64_VECS = FLASH_ATTN_64_HALF / FLASH_ATTN_64_VEC4,
                        FLASH_ATTN_64_SGROUPS = 4,
                        FLASH_ATTN_64_ROWS_PER_SG =
                            FLASH_ATTN_64_TILE / FLASH_ATTN_64_SGROUPS;

// Exact 16 KiB shared threadgroup workspace for 64x64 cooperative FlashAttention:
// FP32 scores occupy all 16 KiB during scoring, then probabilities (8 KiB) and
// per-row softmax state share the same 16 KiB during accumulation.
template <typename T>
union FlashAttn64Smem {
  // FP32 log2-scaled attention scores for one 64x64 tile.
  float scores[FLASH_ATTN_64_TILE * FLASH_ATTN_64_TILE];
  // Probability weights and per-row online softmax scalars after scoring ends.
  struct {
    // Attention probabilities rounded to the value matrix element type.
    T probs[FLASH_ATTN_64_TILE * FLASH_ATTN_64_TILE];
    // Per-query-row output rescale factor or final normalization denominator.
    float row_scale[FLASH_ATTN_64_TILE];
    // Per-query-row secondary scalar for optional sink-logit normalization.
    float row_aux[FLASH_ATTN_64_TILE];
    // Per-simdgroup flag set when any row's running maximum changes.
    uint sg_changed[FLASH_ATTN_64_SGROUPS];
  } st;
};

// Runs base-2 online softmax over one 64x64 score tile in smem.scores, writes
// rounded probabilities and per-row rescale factors into smem.st, and returns
// nonzero if any query row's running maximum changed.
template <typename T>
inline uint flash_attn_64x64_softmax(threadgroup FlashAttn64Smem<T> &smem,
                                     uint sg, uint lane, bool has_mask,
                                     thread float &rmax, thread float &rsum) {
  uint row = sg * FLASH_ATTN_64_ROWS_PER_SG + (lane >> 1u);
  uint col0 = (lane & 1u) * FLASH_ATTN_64_HALF;
  threadgroup float4 *own4 =
      (threadgroup float4 *)(smem.scores + row * FLASH_ATTN_64_TILE + col0);
  float4 v[FLASH_ATTN_64_VECS];
  float4 m4 = float4(-INFINITY);
  for (uint c = 0; c < FLASH_ATTN_64_VECS; c++) {
    v[c] = own4[c];
    m4 = max(m4, v[c]);
  }
  float bmax = max(max(m4.x, m4.y), max(m4.z, m4.w));
  bmax = max(bmax, simd_shuffle_xor(bmax, 1u));
  float new_max = max(rmax, bmax);
  float old_scale = (rmax == -INFINITY || new_max == -INFINITY)
                        ? 0.0f
                        : exp2(rmax - new_max);
  vec<T, FLASH_ATTN_64_VEC4> pout[FLASH_ATTN_64_VECS];
  float bsum = 0.0f;
  if (!has_mask) {
    float4 s4 = float4(0.0f);
    for (uint c = 0; c < FLASH_ATTN_64_VECS; c++) {
      float4 e = exp2(v[c] - new_max);
      pout[c] = vec<T, FLASH_ATTN_64_VEC4>(e);
      s4 += e;
    }
    bsum = (s4.x + s4.y) + (s4.z + s4.w);
    bsum += simd_shuffle_xor(bsum, 1u);
  } else if (bmax == -INFINITY) {
    for (uint c = 0; c < FLASH_ATTN_64_VECS; c++)
      pout[c] = vec<T, FLASH_ATTN_64_VEC4>(T(0.0f));
  } else {
    float4 s4 = float4(0.0f);
    for (uint c = 0; c < FLASH_ATTN_64_VECS; c++) {
      float4 e =
          select(exp2(v[c] - new_max), float4(0.0f), v[c] == float4(-INFINITY));
      pout[c] = vec<T, FLASH_ATTN_64_VEC4>(e);
      s4 += e;
    }
    bsum = (s4.x + s4.y) + (s4.z + s4.w);
    bsum += simd_shuffle_xor(bsum, 1u);
  }
  rsum = rsum * old_scale + bsum;
  rmax = new_max;
  uint any_sg = simd_any(old_scale != 1.0f) ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup vec<T, FLASH_ATTN_64_VEC4> *dst4 =
      (threadgroup vec<T, FLASH_ATTN_64_VEC4> *)(smem.st.probs +
                                                 row * FLASH_ATTN_64_TILE +
                                                 col0);
  for (uint c = 0; c < FLASH_ATTN_64_VECS; c++) dst4[c] = pout[c];
  if ((lane & 1u) == 0u) smem.st.row_scale[row] = old_scale;
  if (lane == 0u) smem.st.sg_changed[sg] = any_sg;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return smem.st.sg_changed[0] | smem.st.sg_changed[1] |
         smem.st.sg_changed[2] | smem.st.sg_changed[3];
}
