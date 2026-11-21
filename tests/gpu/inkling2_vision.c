#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/inkling2/vision.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GPU kernel tests for every Inkling-2-Mini vision kernel in
// src/kernels/inkling2_vision.metal against independent CPU references.

// Test geometry, threadgroup sizes and tolerances.
static constexpr uint32_t TEST_SEQ = 37;
static constexpr uint32_t TEST_PADDED = 48;
static constexpr uint32_t ELEM_THREADS = 256;
static constexpr uint32_t LN_THREADS = 256;
static constexpr uint32_t GRID2D_THREADS = 64;
static constexpr uint32_t ATTN_THREADS = 64;
static constexpr uint32_t ATTN_Q_GROUP = 32;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_MASK = 0xFFFF0000u;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr float LN_EPS = 1e-5f;
static constexpr float TOL_F32 = 1e-4f;
static constexpr float TOL_BF16 = 3e-2f;
static constexpr float INV_SQRT2 = 0.70710678118654752f;

// Parameter structs matching src/kernels/inkling2_vision.metal.
typedef struct {
  // Element count for conversion.
  uint32_t count;
} CastParam;

typedef struct {
  // Patch count and vision width.
  uint32_t npatches, dim;
} EmbedParam;

typedef struct {
  // Row width and LayerNorm epsilon.
  uint32_t dim;
  float eps;
} LnParam;

typedef struct {
  // Row count, row width, add-bias flag and GELU flag.
  uint32_t rows, dim, has_bias, apply_gelu;
} BiasActParam;

typedef struct {
  // Sequence length and padded head row stride.
  uint32_t seq, padded;
} HeadsParam;

typedef struct {
  // Sequence length and padded head row stride.
  uint32_t seq, padded;
} AttendParam;

typedef struct {
  // Row count and row width.
  uint32_t rows, dim;
} ScaleAddParam;

// Deterministic xorshift32 PRNG.
static uint32_t rng_next(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

// Uniform float in [-1, 1].
static float rng_f32(uint32_t *s) {
  return (float)(rng_next(s) & 0xFFFFu) / 32767.5f - 1.0f;
}

// Rounds FP32 to BF16 ties-to-even in FP32.
static float bf_round(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Decodes one BF16 code to FP32.
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << BF16_SHIFT;
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Checks error threshold and prints the measured max error.
static void check_err(const char *label, float err, float tol) {
  char msg[160];
  snprintf(msg, sizeof msg, "%s (max err %.6f < %.6f)", label, err, tol);
  check(err < tol, msg);
}

// Tests BF16<->FP32 conversion, embed assembly, LayerNorm, bias+GELU and
// LayerScale residual addition.
static void test_pointwise_and_norm(Gpu *gpu, MemorySlot *slot) {
  int k_to_f32 = gpu_find_kernel(gpu, "ink2_vis_bf16_to_f32");
  int k_to_bf16 = gpu_find_kernel(gpu, "ink2_vis_f32_to_bf16");
  int k_embed = gpu_find_kernel(gpu, "ink2_vis_embed_assemble");
  int k_ln = gpu_find_kernel(gpu, "ink2_vis_layernorm");
  int k_act = gpu_find_kernel(gpu, "ink2_vis_bias_act");
  int k_ls = gpu_find_kernel(gpu, "ink2_vis_layerscale_add");

  uint32_t rng = 0x2468ace1u;
  uint32_t np = 8, seq = 1 + np, dim = INK2_VIS_DIM;

  Buf patch_emb = gpu_alloc(gpu, slot, (size_t)np * dim * sizeof(float));
  Buf pos_table =
      gpu_alloc(gpu, slot, (size_t)INK2_VIS_POS_ENTRIES * dim * sizeof(float));
  Buf cls_emb = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf cls_pos = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf pos_idx =
      gpu_alloc(gpu, slot, (size_t)np * INK2_VIS_TAPS * sizeof(uint32_t));
  Buf pos_w = gpu_alloc(gpu, slot, (size_t)np * INK2_VIS_TAPS * sizeof(float));
  Buf seq_out = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf ln_w = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf ln_b = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf ln_out = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf act_buf = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf bias = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf gamma = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf ls_x = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf bf_tmp = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(uint16_t));
  Buf f32_rt = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));

  float *pe = (float *)patch_emb.ptr, *pt = (float *)pos_table.ptr,
        *ce = (float *)cls_emb.ptr, *cp = (float *)cls_pos.ptr,
        *lw = (float *)ln_w.ptr, *lb = (float *)ln_b.ptr,
        *bp = (float *)bias.ptr, *gp = (float *)gamma.ptr,
        *lxp = (float *)ls_x.ptr, *ap = (float *)act_buf.ptr;
  for (size_t i = 0; i < (size_t)np * dim; i++) pe[i] = bf_round(rng_f32(&rng));
  for (size_t i = 0; i < (size_t)INK2_VIS_POS_ENTRIES * dim; i++)
    pt[i] = bf_round(rng_f32(&rng) * 0.25f);
  for (uint32_t d = 0; d < dim; d++) {
    ce[d] = bf_round(rng_f32(&rng));
    cp[d] = bf_round(rng_f32(&rng));
    lw[d] = bf_round(0.8f + 0.4f * rng_f32(&rng));
    lb[d] = bf_round(rng_f32(&rng) * 0.2f);
    bp[d] = bf_round(rng_f32(&rng) * 0.3f);
    gp[d] = bf_round(0.1f + 0.2f * fabsf(rng_f32(&rng)));
  }
  inkling2_vision_bilinear_taps(2, 4, (uint32_t *)pos_idx.ptr,
                                (float *)pos_w.ptr);
  for (size_t i = 0; i < (size_t)seq * dim; i++) {
    lxp[i] = bf_round(rng_f32(&rng));
    ap[i] = bf_round(rng_f32(&rng) * 2.0f);
  }
  float *lxp_orig = malloc((size_t)seq * dim * sizeof(float));
  float *ap_orig = malloc((size_t)seq * dim * sizeof(float));
  memcpy(lxp_orig, lxp, (size_t)seq * dim * sizeof(float));
  memcpy(ap_orig, ap, (size_t)seq * dim * sizeof(float));

  EmbedParam ep = {np, dim};
  LnParam lp = {dim, LN_EPS};
  BiasActParam bap = {seq, dim, 1u, 1u};
  ScaleAddParam sap = {seq, dim};
  CastParam cap = {seq * dim};

  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_embed);
  gpu_bind_buffer(&cmd, 0, &patch_emb, 0);
  gpu_bind_buffer(&cmd, 1, &pos_table, 0);
  gpu_bind_buffer(&cmd, 2, &cls_emb, 0);
  gpu_bind_buffer(&cmd, 3, &cls_pos, 0);
  gpu_bind_buffer(&cmd, 4, &pos_idx, 0);
  gpu_bind_buffer(&cmd, 5, &pos_w, 0);
  gpu_bind_buffer(&cmd, 6, &seq_out, 0);
  gpu_copy_bytes(&cmd, 7, &ep, sizeof ep);
  gpu_run_kernel(&cmd, dim / GRID2D_THREADS, seq, 1, GRID2D_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_ln);
  gpu_bind_buffer(&cmd, 0, &seq_out, 0);
  gpu_bind_buffer(&cmd, 1, &ln_w, 0);
  gpu_bind_buffer(&cmd, 2, &ln_b, 0);
  gpu_bind_buffer(&cmd, 3, &ln_out, 0);
  gpu_copy_bytes(&cmd, 4, &lp, sizeof lp);
  gpu_run_kernel(&cmd, seq, 1, 1, LN_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_act);
  gpu_bind_buffer(&cmd, 0, &act_buf, 0);
  gpu_bind_buffer(&cmd, 1, &bias, 0);
  gpu_copy_bytes(&cmd, 2, &bap, sizeof bap);
  gpu_run_kernel(&cmd, dim / GRID2D_THREADS, seq, 1, GRID2D_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_ls);
  gpu_bind_buffer(&cmd, 0, &ls_x, 0);
  gpu_bind_buffer(&cmd, 1, &ln_out, 0);
  gpu_bind_buffer(&cmd, 2, &bias, 0);
  gpu_bind_buffer(&cmd, 3, &gamma, 0);
  gpu_copy_bytes(&cmd, 4, &sap, sizeof sap);
  gpu_run_kernel(&cmd, dim / GRID2D_THREADS, seq, 1, GRID2D_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_to_bf16);
  gpu_bind_buffer(&cmd, 0, &ls_x, 0);
  gpu_bind_buffer(&cmd, 1, &bf_tmp, 0);
  gpu_copy_bytes(&cmd, 2, &cap, sizeof cap);
  gpu_run_kernel(&cmd, (cap.count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_to_f32);
  gpu_bind_buffer(&cmd, 0, &bf_tmp, 0);
  gpu_bind_buffer(&cmd, 1, &f32_rt, 0);
  gpu_copy_bytes(&cmd, 2, &cap, sizeof cap);
  gpu_run_kernel(&cmd, (cap.count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Verify embed_assemble.
  const float *so = (const float *)seq_out.ptr;
  const uint32_t *pip = (const uint32_t *)pos_idx.ptr;
  const float *pwp = (const float *)pos_w.ptr;
  float emb_err = 0.0f;
  for (uint32_t d = 0; d < dim; d++)
    emb_err = fmaxf(emb_err, fabsf(so[d] - bf_round(ce[d] + cp[d])));
  for (uint32_t p = 0; p < np; p++) {
    for (uint32_t d = 0; d < dim; d++) {
      float pos = 0.0f;
      for (uint32_t tap = 0; tap < INK2_VIS_TAPS; tap++)
        pos += pt[pip[p * INK2_VIS_TAPS + tap] * dim + d] *
               pwp[p * INK2_VIS_TAPS + tap];
      float ref = bf_round(pe[p * dim + d] + bf_round(pos));
      emb_err = fmaxf(emb_err, fabsf(so[(p + 1) * dim + d] - ref));
    }
  }
  check_err("ink2_vis_embed_assemble", emb_err, TOL_F32);

  // Verify LayerNorm, bias+GELU, LayerScale add and BF16 round-trip.
  const float *lo = (const float *)ln_out.ptr;
  const float *rt = (const float *)f32_rt.ptr;
  float ln_err = 0.0f, act_err = 0.0f, ls_err = 0.0f, rt_err = 0.0f;
  for (uint32_t r = 0; r < seq; r++) {
    double sum = 0.0, vsum = 0.0;
    for (uint32_t d = 0; d < dim; d++) sum += so[r * dim + d];
    float mean = (float)(sum / dim);
    for (uint32_t d = 0; d < dim; d++) {
      double diff = so[r * dim + d] - mean;
      vsum += diff * diff;
    }
    float inv = 1.0f / sqrtf((float)(vsum / dim) + LN_EPS);
    for (uint32_t d = 0; d < dim; d++) {
      float ln_ref = bf_round((so[r * dim + d] - mean) * inv * lw[d] + lb[d]);
      ln_err = fmaxf(ln_err, fabsf(lo[r * dim + d] - ln_ref));

      float v_act = bf_round(ap_orig[r * dim + d] + bp[d]);
      float g_ref = bf_round(0.5f * v_act * (1.0f + erff(v_act * INV_SQRT2)));
      act_err = fmaxf(act_err, fabsf(ap[r * dim + d] - g_ref));

      float ls_ref =
          bf_round(lxp_orig[r * dim + d] +
                   bf_round(gp[d] * bf_round(lo[r * dim + d] + bp[d])));
      ls_err = fmaxf(ls_err, fabsf(lxp[r * dim + d] - ls_ref));
      rt_err = fmaxf(rt_err, fabsf(rt[r * dim + d] - lxp[r * dim + d]));
    }
  }
  check_err("ink2_vis_layernorm", ln_err, TOL_BF16);
  check_err("ink2_vis_bias_act (exact GELU)", act_err, TOL_BF16);
  check_err("ink2_vis_layerscale_add", ls_err, TOL_F32);
  check_err("ink2_vis_f32_to_bf16 + bf16_to_f32", rt_err, TOL_F32);

  free(lxp_orig);
  free(ap_orig);
  gpu_free(&patch_emb);
  gpu_free(&pos_table);
  gpu_free(&cls_emb);
  gpu_free(&cls_pos);
  gpu_free(&pos_idx);
  gpu_free(&pos_w);
  gpu_free(&seq_out);
  gpu_free(&ln_w);
  gpu_free(&ln_b);
  gpu_free(&ln_out);
  gpu_free(&act_buf);
  gpu_free(&bias);
  gpu_free(&gamma);
  gpu_free(&ls_x);
  gpu_free(&bf_tmp);
  gpu_free(&f32_rt);
}

// Tests ink2_vis_heads_rope and ink2_vis_attend bidirectional FlashAttention.
static void test_rope_and_attend(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "ink2_vis_heads_rope");
  int k_attend = gpu_find_kernel(gpu, "ink2_vis_attend");

  uint32_t rng = 0x13572468u;
  uint32_t seq = TEST_SEQ, padded = TEST_PADDED, dim = INK2_VIS_DIM;
  Buf q = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf k = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf v = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));
  Buf qb = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf kb = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf vb = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf cs = gpu_alloc(gpu, slot,
                     (size_t)seq * INK2_VIS_ROPE_PAIRS * 2 * sizeof(float));
  size_t hb =
      (size_t)INK2_VIS_HEADS * padded * INK2_VIS_HEAD_DIM * sizeof(uint16_t);
  Buf qh = gpu_alloc(gpu, slot, hb);
  Buf kh = gpu_alloc(gpu, slot, hb);
  Buf vh = gpu_alloc(gpu, slot, hb);
  Buf out = gpu_alloc(gpu, slot, (size_t)seq * dim * sizeof(float));

  float *qp = (float *)q.ptr, *kp = (float *)k.ptr, *vp = (float *)v.ptr;
  float *qbp = (float *)qb.ptr, *kbp = (float *)kb.ptr, *vbp = (float *)vb.ptr;
  for (size_t i = 0; i < (size_t)seq * dim; i++) {
    qp[i] = bf_round(rng_f32(&rng) * 0.5f);
    kp[i] = bf_round(rng_f32(&rng) * 0.5f);
    vp[i] = bf_round(rng_f32(&rng) * 0.5f);
  }
  for (uint32_t d = 0; d < dim; d++) {
    qbp[d] = bf_round(rng_f32(&rng) * 0.1f);
    kbp[d] = bf_round(rng_f32(&rng) * 0.1f);
    vbp[d] = bf_round(rng_f32(&rng) * 0.1f);
  }
  // 37 = 1 + 6 * 6 patches (vh=6, vw=6).
  inkling2_vision_rope_table(6, 6, (float *)cs.ptr);

  HeadsParam hp = {seq, padded};
  AttendParam ap = {seq, padded};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_heads);
  gpu_bind_buffer(&cmd, 0, &q, 0);
  gpu_bind_buffer(&cmd, 1, &k, 0);
  gpu_bind_buffer(&cmd, 2, &v, 0);
  gpu_bind_buffer(&cmd, 3, &qb, 0);
  gpu_bind_buffer(&cmd, 4, &kb, 0);
  gpu_bind_buffer(&cmd, 5, &vb, 0);
  gpu_bind_buffer(&cmd, 6, &cs, 0);
  gpu_bind_buffer(&cmd, 7, &qh, 0);
  gpu_bind_buffer(&cmd, 8, &kh, 0);
  gpu_bind_buffer(&cmd, 9, &vh, 0);
  gpu_copy_bytes(&cmd, 10, &hp, sizeof hp);
  gpu_run_kernel(&cmd, (uint32_t)INK2_VIS_HEADS, padded, 1,
                 (uint32_t)INK2_VIS_ROPE_PAIRS, 1, 1);

  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &qh, 0);
  gpu_bind_buffer(&cmd, 1, &kh, 0);
  gpu_bind_buffer(&cmd, 2, &vh, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (seq + ATTN_Q_GROUP - 1) / ATTN_Q_GROUP,
                 (uint32_t)INK2_VIS_HEADS, 1, ATTN_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Verify RoPE head outputs and bidirectional attention against CPU reference.
  const uint16_t *qhp = (const uint16_t *)qh.ptr;
  const uint16_t *khp = (const uint16_t *)kh.ptr;
  const uint16_t *vhp = (const uint16_t *)vh.ptr;
  const float *csp = (const float *)cs.ptr;
  const float *op = (const float *)out.ptr;
  float rope_err = 0.0f, attn_err = 0.0f;
  float scale = 1.0f / sqrtf((float)INK2_VIS_HEAD_DIM);

  for (uint32_t h = 0; h < (uint32_t)INK2_VIS_HEADS; h++) {
    for (uint32_t r = 0; r < seq; r++) {
      for (uint32_t pair = 0; pair < (uint32_t)INK2_VIS_ROPE_PAIRS; pair++) {
        uint32_t d0 = h * INK2_VIS_HEAD_DIM + pair * 2;
        float q0 = bf_round(qp[r * dim + d0] + qbp[d0]);
        float q1 = bf_round(qp[r * dim + d0 + 1] + qbp[d0 + 1]);
        float c = csp[(r * INK2_VIS_ROPE_PAIRS + pair) * 2];
        float s = csp[(r * INK2_VIS_ROPE_PAIRS + pair) * 2 + 1];
        size_t off = ((size_t)h * padded + r) * INK2_VIS_HEAD_DIM + pair * 2;
        rope_err = fmaxf(
            rope_err, fabsf(bf16_to_f32(qhp[off]) - bf_round(q0 * c - q1 * s)));
        rope_err = fmaxf(rope_err, fabsf(bf16_to_f32(qhp[off + 1]) -
                                         bf_round(q1 * c + q0 * s)));
      }
      float scores[TEST_SEQ];
      float mx = -INFINITY;
      size_t qb_off = ((size_t)h * padded + r) * INK2_VIS_HEAD_DIM;
      for (uint32_t j = 0; j < seq; j++) {
        size_t kb_off = ((size_t)h * padded + j) * INK2_VIS_HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < INK2_VIS_HEAD_DIM; d++)
          dot += bf16_to_f32(qhp[qb_off + d]) * bf16_to_f32(khp[kb_off + d]);
        scores[j] = dot * scale;
        if (scores[j] > mx) mx = scores[j];
      }
      float sum = 0.0f;
      for (uint32_t j = 0; j < seq; j++) {
        scores[j] = expf(scores[j] - mx);
        sum += scores[j];
      }
      for (int d = 0; d < INK2_VIS_HEAD_DIM; d++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < seq; j++) {
          size_t vb_off = ((size_t)h * padded + j) * INK2_VIS_HEAD_DIM;
          acc += scores[j] * bf16_to_f32(vhp[vb_off + d]);
        }
        float ref = bf_round(acc / sum);
        float got = op[r * dim + h * INK2_VIS_HEAD_DIM + d];
        attn_err = fmaxf(attn_err, fabsf(got - ref));
      }
    }
  }
  check_err("ink2_vis_heads_rope", rope_err, TOL_F32);
  check_err("ink2_vis_attend bidirectional FlashAttention", attn_err, TOL_BF16);

  gpu_free(&q);
  gpu_free(&k);
  gpu_free(&v);
  gpu_free(&qb);
  gpu_free(&kb);
  gpu_free(&vb);
  gpu_free(&cs);
  gpu_free(&qh);
  gpu_free(&kh);
  gpu_free(&vh);
  gpu_free(&out);
}

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *gpu = gpu_new(src, len);

  test_pointwise_and_norm(gpu, slot);
  test_rope_and_attend(gpu, slot);
  gpu_return_freed_memory(gpu);
  return failures != 0;
}
