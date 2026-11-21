#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests every Inkling-2-Mini text backbone Metal kernel in
// src/kernels/inkling2.metal against an independent CPU reference.

// Attention and model constants matching src/kernels/inkling2.metal.
static constexpr uint32_t HEADS = 24, KV_HEADS = 8, GQA = HEADS / KV_HEADS,
                          HEAD_DIM = 128, Q_DIM = HEADS * HEAD_DIM,
                          KV_DIM = KV_HEADS * HEAD_DIM, D_REL = 16,
                          REL_EXTENT = 1024, WINDOW = 1024, SCONV_K = 4,
                          EXPERTS = 128, TOPK = 8, NVFP4_BLOCK = 16,
                          MXFP8_BLOCK = 32;

// Threadgroup sizes and tile alignments matching src/kernels/inkling2.metal.
static constexpr uint32_t ROW_THREADS = 256, QKVR_THREADS = 128,
                          QKVR_HEADS = HEADS + 2 * KV_HEADS + HEADS,
                          DECODE_THREADS = 128, COMBINE_THREADS = 32,
                          PREFILL_THREADS = 64, PREFILL_Q_GROUP = 32,
                          Q_PAD = 16, MV_THREADS = 256, MV_ROWS_PER_TG = 32,
                          MM_THREADS = 128, MM_TILE = 64, PART_FLOATS = 130,
                          ELEM_THREADS = 256, ROUTE_THREADS = 128,
                          VEC4_WIDTH = 4, E2M1_MASK = 0xFu, E2M1_SHIFT = 4,
                          FP8_EXP_BIAS = 127, BF16_ROUND_MASK = 0xFFFF0000u,
                          BF16_HALF = 0x7FFFu, BF16_SHIFT = 16;

// Numerical constants from config.json and modeling_inkling2.py.
static constexpr float RMS_EPS = 1e-6f, CLIP_EPS = 1e-6f,
                       QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f,
                       LOG_FLOOR = 8192.0f, LOG_ALPHA = 0.1f, GATE_CAP = 4.0f,
                       UP_CAP = 25.0f, ROUTE_SCALE = 8.0f, ROUTE_EPS = 1e-20f,
                       LOGIT_CAP = 30.0f,
                       STATIC_SCONV_SCALE = 0.44721359549995793f,
                       DYNAMIC_SCONV_SCALE = 0.33333333333333331f;

// Verification tolerances for FP32 and BF16-rounded kernel outputs.
static constexpr float TOL_F32 = 1e-4f, TOL_BF16 = 3e-2f, TOL_ATTN = 2e-2f;

// NVFP4 E2M1 lookup table (core_model_loading.py
// Inkling2LoopDequantizeExperts).
static const float E2M1_TABLE[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Kernel parameter structs matching src/kernels/inkling2.metal.
typedef struct {
  // Row width in floats and whether to round outputs to BF16.
  uint32_t dim, round;
  // RMSNorm epsilon.
  float eps;
} NormParam;

typedef struct {
  // Active rows, block row stride, hidden width, completed blocks, queries.
  uint32_t rows, block_stride, dim, nblocks, nqueries;
  // Output row stride between queries.
  uint32_t out_row_stride;
  // Unweighted RMSNorm epsilon.
  float eps;
} AttnResParam;

typedef struct {
  // Active rows, hidden width and bottleneck gate rank.
  uint32_t rows, dim, rank;
  // Unweighted RMSNorm epsilon.
  float eps;
} MsgGateParam;

typedef struct {
  // Active rows, channel width, ring capacity, row-0 position, BF16 round flag.
  uint32_t rows, dim, ring_rows, pos0, round;
} SconvParam;

typedef struct {
  // Device address of the state's FP32 ring buffer.
  uint64_t ring;
  // Token position and alignment padding.
  uint32_t pos, pad;
} SconvTarget;

typedef struct {
  // Device addresses of the state's BF16 K and V buffers.
  uint64_t k, v;
  // Token position and ring capacity (0 for global linear KV).
  uint32_t pos, ring_rows;
} KvTarget;

typedef struct {
  // Active rows, head-major Q stride, row-0 pos, ring capacity, flags.
  uint32_t rows, q_stride, pos0, ring_rows, is_global, batched;
} QkvrParam;

typedef struct {
  // Active rows, heads, relative dim, relative extent.
  uint32_t rows, heads, d_rel, extent;
} RelBiasParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, ring capacity, window, partitions, flag.
  uint32_t rows, q_stride, pos0, ring_rows, window, nparts, part_stride,
      batched;
} AttendDecodeParam;

typedef struct {
  // Active query rows, Q stride, row-0 pos, ring capacity, sliding window.
  uint32_t rows, q_stride, pos0, ring_rows, window;
} AttendPrefillParam;

typedef struct {
  // Active rows, total experts, top-k count.
  uint32_t rows, experts, topk;
  // Routing weight multiplier.
  float scale;
} RouteParam;

typedef struct {
  // Expert index, first pair index, pair count, padding.
  uint32_t expert, start, count, pad;
} MoeTile;

typedef struct {
  // Matrix N, K, total pairs, total tiles.
  uint32_t n, k, npairs, ntiles;
  // Softcap bounds for gate and up.
  float gate_cap, up_cap;
} MoeParam;

typedef struct {
  // Active rows, hidden width, top-k count.
  uint32_t rows, dim, topk;
} MoeCombineParam;

typedef struct {
  // Active rows, unpadded vocabulary size, row stride in floats.
  uint32_t rows, vocab, stride;
  // Logit softcap bound.
  float cap;
} SoftcapParam;

// Deterministic xorshift32 PRNG for reproducible test vectors.
static uint32_t rng_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// Uniform random float in [-1, 1].
static float rng_f32(uint32_t *state) {
  return (float)(rng_next(state) & 0xffffffu) / (float)0x800000u - 1.0f;
}

// Rounds an FP32 value to BF16 ties-to-even as a 16-bit code.
static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return (uint16_t)((u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) >> BF16_SHIFT);
}

// Decodes a 16-bit BF16 code to FP32.
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << BF16_SHIFT;
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Rounds an FP32 value to BF16 precision in FP32.
static float bf_round(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_ROUND_MASK;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Decodes an FP8 E4M3FN byte to FP32 (no NaN codes in test weights).
static float e4m3_to_f32(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int mant = b & 0x7;
  float v = exp == 0 ? ldexpf((float)mant / 8.0f, -6)
                     : ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
  return sign ? -v : v;
}

// Decodes a UE8M0 byte to its exact power-of-two FP32 scale.
static float e8m0_to_f32(uint8_t b) {
  return ldexpf(1.0f, (int)b - (int)FP8_EXP_BIAS);
}

// Logistic sigmoid in FP32.
static float cpu_sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// Softcapped SwiGLU matching Inkling2SoftcappedSwiGLU.
static float cpu_softcapped_swiglu(float gate, float up) {
  float s = cpu_sigm(gate);
  float gc = GATE_CAP * tanhf(gate / GATE_CAP);
  float uc = UP_CAP * tanhf(up / UP_CAP);
  return bf_round(gc * s * uc);
}

// Encodes one kernel dispatch binding `nb` buffers at slots 0..nb-1 and a
// parameter struct at slot `nb`.
static void run_op(GpuCommandBuffer *c, int kernel, const Buf *bufs, int nb,
                   const void *param, size_t pbytes, uint32_t gx, uint32_t gy,
                   uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  gpu_select_kernel(c, kernel);
  for (int i = 0; i < nb; i++) gpu_bind_buffer(c, i, &bufs[i], 0);
  if (param && pbytes) gpu_copy_bytes(c, nb, param, pbytes);
  gpu_run_kernel(c, gx, gy, gz, tx, ty, tz);
}

// Checks a floating-point max-error threshold and reports the measured error.
static void check_err(const char *label, float err, float tol) {
  char msg[160];
  snprintf(msg, sizeof msg, "%s (max err %.6f < %.6f)", label, err, tol);
  check(err < tol, msg);
}

// Tests ink2_embed_norm (including negative placeholder skip), ink2_rmsnorm,
// ink2_add, and ink2_logit_softcap.
static void test_embed_norm_and_softcap(Gpu *gpu, MemorySlot *slot) {
  int k_embed = gpu_find_kernel(gpu, "ink2_embed_norm");
  int k_norm = gpu_find_kernel(gpu, "ink2_rmsnorm");
  int k_add = gpu_find_kernel(gpu, "ink2_add");
  int k_cap = gpu_find_kernel(gpu, "ink2_logit_softcap");

  uint32_t rng = 0x13579bdfu;
  uint32_t vocab = 32, dim = 256, rows = 4;
  Buf table = gpu_alloc(gpu, slot, (size_t)vocab * dim * sizeof(uint16_t));
  Buf ids = gpu_alloc(gpu, slot, (size_t)rows * sizeof(int));
  Buf w = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf out = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
  Buf y = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));

  uint16_t *tp = (uint16_t *)table.ptr;
  for (uint32_t i = 0; i < vocab * dim; i++)
    tp[i] = f32_to_bf16(rng_f32(&rng) * 1.5f);
  float *wp = (float *)w.ptr;
  for (uint32_t i = 0; i < dim; i++) wp[i] = 0.5f + rng_f32(&rng) * 0.5f;
  int host_ids[4] = {3, -1, 17, 0};
  memcpy(ids.ptr, host_ids, sizeof host_ids);
  // Pre-fill row 1 (placeholder) with a sentinel value that must stay intact.
  float *op = (float *)out.ptr;
  for (uint32_t i = 0; i < rows * dim; i++) op[i] = 7.25f;

  NormParam np = {.dim = dim, .round = 1, .eps = RMS_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  run_op(&cmd, k_embed, (Buf[]){table, ids, w, out}, 4, &np, sizeof np, rows, 1,
         1, ROW_THREADS, 1, 1);
  run_op(&cmd, k_norm, (Buf[]){out, w, y}, 3, &np, sizeof np, rows, 1, 1,
         ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float max_err = 0.0f;
  for (uint32_t r = 0; r < rows; r++) {
    if (host_ids[r] < 0) {
      for (uint32_t d = 0; d < dim; d++)
        max_err = fmaxf(max_err, fabsf(op[r * dim + d] - 7.25f));
      continue;
    }
    double sq = 0.0;
    for (uint32_t d = 0; d < dim; d++) {
      float v = bf16_to_f32(tp[(uint32_t)host_ids[r] * dim + d]);
      sq += (double)v * v;
    }
    float inv = 1.0f / sqrtf((float)(sq / dim) + RMS_EPS);
    for (uint32_t d = 0; d < dim; d++) {
      float v = bf16_to_f32(tp[(uint32_t)host_ids[r] * dim + d]);
      float ref = bf_round(wp[d] * v * inv);
      max_err = fmaxf(max_err, fabsf(op[r * dim + d] - ref));
    }
  }
  check_err("ink2_embed_norm", max_err, TOL_F32);

  // Check ink2_add and ink2_logit_softcap (leaving padded columns untouched).
  uint32_t stride = 24, unpadded = 19, lrows = 3;
  Buf logits = gpu_alloc(gpu, slot, (size_t)lrows * stride * sizeof(float));
  Buf ref_add = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
  float *lp = (float *)logits.ptr;
  float orig_logits[3 * 24];
  for (uint32_t i = 0; i < lrows * stride; i++) {
    orig_logits[i] = rng_f32(&rng) * 45.0f;
    lp[i] = orig_logits[i];
  }
  uint32_t count_round[2] = {rows * dim, 1};
  SoftcapParam sp = {
      .rows = lrows, .vocab = unpadded, .stride = stride, .cap = LOGIT_CAP};
  cmd = gpu_begin(gpu);
  run_op(&cmd, k_add, (Buf[]){out, y, ref_add}, 3, count_round,
         sizeof count_round, (rows * dim + ELEM_THREADS - 1) / ELEM_THREADS, 1,
         1, ELEM_THREADS, 1, 1);
  run_op(&cmd, k_cap, (Buf[]){logits}, 1, &sp, sizeof sp, (unpadded + 15) / 16,
         lrows, 1, 16, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float add_err = 0.0f;
  const float *yp = (const float *)y.ptr, *ap = (const float *)ref_add.ptr;
  for (uint32_t i = 0; i < rows * dim; i++)
    add_err = fmaxf(add_err, fabsf(ap[i] - bf_round(op[i] + yp[i])));
  check_err("ink2_add", add_err, TOL_F32);

  float cap_err = 0.0f;
  for (uint32_t r = 0; r < lrows; r++) {
    for (uint32_t c = 0; c < stride; c++) {
      float x0 = orig_logits[r * stride + c];
      float ref = c < unpadded ? LOGIT_CAP * tanhf(x0 / LOGIT_CAP) : x0;
      cap_err = fmaxf(cap_err, fabsf(lp[r * stride + c] - ref));
    }
  }
  check_err("ink2_logit_softcap", cap_err, TOL_F32);

  gpu_free(&table);
  gpu_free(&ids);
  gpu_free(&w);
  gpu_free(&out);
  gpu_free(&y);
  gpu_free(&logits);
  gpu_free(&ref_add);
}

// Tests ink2_attn_res for 4 queries and ink2_msg_cat_norm +
// ink2_msg_gate_apply.
static void test_attn_res_and_msg_gate(Gpu *gpu, MemorySlot *slot) {
  int k_res = gpu_find_kernel(gpu, "ink2_attn_res");
  int k_cat = gpu_find_kernel(gpu, "ink2_msg_cat_norm");
  int k_apply = gpu_find_kernel(gpu, "ink2_msg_gate_apply");

  uint32_t rng = 0x2468ace1u;
  uint32_t rows = 5, dim = 512, nblocks = 3, nqueries = 4, rank = 64;
  Buf blocks =
      gpu_alloc(gpu, slot, (size_t)nblocks * rows * dim * sizeof(float));
  Buf partial = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
  Buf queries = gpu_alloc(gpu, slot, (size_t)nqueries * dim * sizeof(float));
  Buf out = gpu_alloc(gpu, slot, (size_t)nqueries * rows * dim * sizeof(float));

  float *bp = (float *)blocks.ptr, *pp = (float *)partial.ptr,
        *qp = (float *)queries.ptr, *op = (float *)out.ptr;
  for (uint32_t i = 0; i < nblocks * rows * dim; i++)
    bp[i] = rng_f32(&rng) * 2.0f;
  for (uint32_t i = 0; i < rows * dim; i++) pp[i] = rng_f32(&rng) * 2.0f;
  for (uint32_t i = 0; i < nqueries * dim; i++) qp[i] = rng_f32(&rng) * 1.5f;

  AttnResParam rp = {.rows = rows,
                     .block_stride = rows,
                     .dim = dim,
                     .nblocks = nblocks,
                     .nqueries = nqueries,
                     .out_row_stride = rows,
                     .eps = RMS_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  run_op(&cmd, k_res, (Buf[]){blocks, partial, queries, out}, 4, &rp, sizeof rp,
         rows, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float max_err = 0.0f;
  float inv_sqrt_d = 1.0f / sqrtf((float)dim);
  uint32_t total = nblocks + 1;
  float *normed = xmalloc((size_t)total * dim * sizeof(float));
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t b = 0; b < total; b++) {
      const float *src = b < nblocks ? bp + (b * rows + r) * dim : pp + r * dim;
      double sq = 0.0;
      for (uint32_t d = 0; d < dim; d++) sq += (double)src[d] * src[d];
      float inv = 1.0f / sqrtf((float)(sq / dim) + RMS_EPS);
      for (uint32_t d = 0; d < dim; d++) normed[b * dim + d] = src[d] * inv;
    }
    for (uint32_t q = 0; q < nqueries; q++) {
      float sc[5], mx = -INFINITY, sum = 0.0f;
      for (uint32_t b = 0; b < total; b++) {
        double dotv = 0.0;
        for (uint32_t d = 0; d < dim; d++)
          dotv += (double)qp[q * dim + d] * normed[b * dim + d];
        sc[b] = (float)dotv * inv_sqrt_d;
        mx = fmaxf(mx, sc[b]);
      }
      for (uint32_t b = 0; b < total; b++) {
        sc[b] = expf(sc[b] - mx);
        sum += sc[b];
      }
      for (uint32_t d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (uint32_t b = 0; b < total; b++)
          acc += (sc[b] / sum) * normed[b * dim + d];
        max_err = fmaxf(max_err, fabsf(op[(q * rows + r) * dim + d] - acc));
      }
    }
  }
  free(normed);
  check_err("ink2_attn_res", max_err, TOL_F32);

  // Test message gate: cat_norm + ink2_msg_gate_apply.
  Buf h = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
  Buf m = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
  Buf cat = gpu_alloc(gpu, slot, (size_t)rows * 2 * dim * sizeof(float));
  Buf g = gpu_alloc(gpu, slot, (size_t)rows * rank * sizeof(float));
  Buf fw = gpu_alloc(gpu, slot, (size_t)dim * rank * sizeof(uint16_t));
  Buf gw = gpu_alloc(gpu, slot, (size_t)dim * rank * sizeof(uint16_t));
  Buf mixed = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));

  float *hp = (float *)h.ptr, *mp = (float *)m.ptr, *gp = (float *)g.ptr;
  uint16_t *fwp = (uint16_t *)fw.ptr, *gwp = (uint16_t *)gw.ptr;
  for (uint32_t i = 0; i < rows * dim; i++) {
    hp[i] = rng_f32(&rng) * 2.0f;
    mp[i] = rng_f32(&rng) * 2.0f;
  }
  for (uint32_t i = 0; i < rows * rank; i++) gp[i] = rng_f32(&rng) * 0.5f;
  for (uint32_t i = 0; i < dim * rank; i++) {
    fwp[i] = f32_to_bf16(rng_f32(&rng) * 0.25f);
    gwp[i] = f32_to_bf16(rng_f32(&rng) * 0.25f);
  }

  MsgGateParam mgp = {.rows = rows, .dim = dim, .rank = rank, .eps = RMS_EPS};
  cmd = gpu_begin(gpu);
  run_op(&cmd, k_cat, (Buf[]){h, m, cat}, 3, &mgp, sizeof mgp, rows, 1, 1,
         ROW_THREADS, 1, 1);
  run_op(&cmd, k_apply, (Buf[]){h, m, g, fw, gw, mixed}, 6, &mgp, sizeof mgp,
         rows, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float cat_err = 0.0f, gate_err = 0.0f;
  const float *cp = (const float *)cat.ptr, *xp = (const float *)mixed.ptr;
  for (uint32_t r = 0; r < rows; r++) {
    double sqh = 0.0, sqm = 0.0;
    for (uint32_t d = 0; d < dim; d++) {
      sqh += (double)hp[r * dim + d] * hp[r * dim + d];
      sqm += (double)mp[r * dim + d] * mp[r * dim + d];
    }
    float ih = 1.0f / sqrtf((float)(sqh / dim) + RMS_EPS);
    float im = 1.0f / sqrtf((float)(sqm / dim) + RMS_EPS);
    for (uint32_t d = 0; d < dim; d++) {
      cat_err = fmaxf(
          cat_err, fabsf(cp[r * 2 * dim + d] - bf_round(hp[r * dim + d] * ih)));
      cat_err = fmaxf(cat_err, fabsf(cp[r * 2 * dim + dim + d] -
                                     bf_round(mp[r * dim + d] * im)));
      float df = 0.0f, dg = 0.0f;
      for (uint32_t j = 0; j < rank; j++) {
        df += bf16_to_f32(fwp[d * rank + j]) * gp[r * rank + j];
        dg += bf16_to_f32(gwp[d * rank + j]) * gp[r * rank + j];
      }
      float ref =
          (1.0f - tanhf(df)) * hp[r * dim + d] + tanhf(dg) * mp[r * dim + d];
      gate_err = fmaxf(gate_err, fabsf(xp[r * dim + d] - ref));
    }
  }
  check_err("ink2_msg_cat_norm", cat_err, TOL_F32);
  check_err("ink2_msg_gate_apply", gate_err, TOL_F32);

  gpu_free(&blocks);
  gpu_free(&partial);
  gpu_free(&queries);
  gpu_free(&out);
  gpu_free(&h);
  gpu_free(&m);
  gpu_free(&cat);
  gpu_free(&g);
  gpu_free(&fw);
  gpu_free(&gw);
  gpu_free(&mixed);
}

// Tests static and dynamic short convolutions across chunk boundaries, ring
// wrap-around, and batched multi-state execution.
static void test_short_convs(Gpu *gpu, MemorySlot *slot) {
  int k_st = gpu_find_kernel(gpu, "ink2_sconv_static");
  int k_stb = gpu_find_kernel(gpu, "ink2_sconv_static_batched");
  int k_dy = gpu_find_kernel(gpu, "ink2_sconv_dynamic");
  int k_dyb = gpu_find_kernel(gpu, "ink2_sconv_dynamic_batched");
  int k_save = gpu_find_kernel(gpu, "ink2_sconv_ring_save");

  uint32_t rng = 0x31415926u;
  uint32_t total = 25, dim = KV_DIM, ring_rows = 7;
  uint32_t chunk1 = 10, chunk2 = 14, chunk3 = 1;

  Buf x_all = gpu_alloc(gpu, slot, (size_t)total * dim * sizeof(float));
  Buf ring = gpu_alloc(gpu, slot, (size_t)ring_rows * dim * sizeof(float));
  Buf w_st = gpu_alloc(gpu, slot, (size_t)dim * SCONV_K * sizeof(float));
  Buf w_dy_st = gpu_alloc(gpu, slot, (size_t)SCONV_K * dim * sizeof(float));
  Buf dyn_all =
      gpu_alloc(gpu, slot, (size_t)total * KV_HEADS * SCONV_K * sizeof(float));
  Buf y_st = gpu_alloc(gpu, slot, (size_t)total * dim * sizeof(float));
  Buf y_dy = gpu_alloc(gpu, slot, (size_t)total * dim * sizeof(float));

  float *xp = (float *)x_all.ptr, *wsp = (float *)w_st.ptr,
        *wdsp = (float *)w_dy_st.ptr, *dwp = (float *)dyn_all.ptr;
  for (uint32_t i = 0; i < total * dim; i++) xp[i] = rng_f32(&rng);
  for (uint32_t i = 0; i < dim * SCONV_K; i++) wsp[i] = rng_f32(&rng) * 0.5f;
  for (uint32_t i = 0; i < SCONV_K * dim; i++) wdsp[i] = rng_f32(&rng) * 0.5f;
  for (uint32_t i = 0; i < total * KV_HEADS * SCONV_K; i++)
    dwp[i] = rng_f32(&rng) * 0.5f;
  memset(ring.ptr, 0, ring.size);

  // Run static sconv in three chunks (10, 14, 1) exercising ring wrap-around.
  uint32_t starts[3] = {0, chunk1, chunk1 + chunk2};
  uint32_t lens[3] = {chunk1, chunk2, chunk3};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  for (int ci = 0; ci < 3; ci++) {
    size_t xoff = (size_t)starts[ci] * dim * sizeof(float);
    SconvParam sp = {.rows = lens[ci],
                     .dim = dim,
                     .ring_rows = ring_rows,
                     .pos0 = starts[ci],
                     .round = 0};
    gpu_select_kernel(&cmd, k_st);
    gpu_bind_buffer(&cmd, 0, &x_all, xoff);
    gpu_bind_buffer(&cmd, 1, &ring, 0);
    gpu_bind_buffer(&cmd, 2, &w_st, 0);
    gpu_bind_buffer(&cmd, 3, &y_st, xoff);
    gpu_copy_bytes(&cmd, 4, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (dim + 63) / 64, lens[ci], 1, 64, 1, 1);

    uint32_t keep = lens[ci] < ring_rows ? lens[ci] : ring_rows;
    gpu_select_kernel(&cmd, k_save);
    gpu_bind_buffer(&cmd, 0, &x_all, xoff);
    gpu_bind_buffer(&cmd, 1, &ring, 0);
    gpu_copy_bytes(&cmd, 2, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (dim + 63) / 64, keep, 1, 64, 1, 1);
  }
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float st_err = 0.0f;
  const float *ysp = (const float *)y_st.ptr;
  for (uint32_t t = 0; t < total; t++) {
    for (uint32_t c = 0; c < dim; c++) {
      float sum = 0.0f;
      for (uint32_t lag = 0; lag < SCONV_K; lag++) {
        if (t < lag) continue;
        sum += wsp[c * SCONV_K + (SCONV_K - 1 - lag)] * xp[(t - lag) * dim + c];
      }
      float ref = (sum + xp[t * dim + c]) * STATIC_SCONV_SCALE;
      st_err = fmaxf(st_err, fabsf(ysp[t * dim + c] - ref));
    }
  }
  check_err("ink2_sconv_static", st_err, TOL_F32);

  // Run dynamic sconv in three chunks exercising ring wrap-around.
  memset(ring.ptr, 0, ring.size);
  cmd = gpu_begin(gpu);
  for (int ci = 0; ci < 3; ci++) {
    size_t xoff = (size_t)starts[ci] * dim * sizeof(float);
    size_t doff = (size_t)starts[ci] * KV_HEADS * SCONV_K * sizeof(float);
    SconvParam sp = {.rows = lens[ci],
                     .dim = dim,
                     .ring_rows = ring_rows,
                     .pos0 = starts[ci],
                     .round = 1};
    gpu_select_kernel(&cmd, k_dy);
    gpu_bind_buffer(&cmd, 0, &x_all, xoff);
    gpu_bind_buffer(&cmd, 1, &ring, 0);
    gpu_bind_buffer(&cmd, 2, &w_dy_st, 0);
    gpu_bind_buffer(&cmd, 3, &dyn_all, doff);
    gpu_bind_buffer(&cmd, 4, &y_dy, xoff);
    gpu_copy_bytes(&cmd, 5, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (dim + 63) / 64, lens[ci], 1, 64, 1, 1);

    uint32_t keep = lens[ci] < ring_rows ? lens[ci] : ring_rows;
    gpu_select_kernel(&cmd, k_save);
    gpu_bind_buffer(&cmd, 0, &x_all, xoff);
    gpu_bind_buffer(&cmd, 1, &ring, 0);
    gpu_copy_bytes(&cmd, 2, &sp, sizeof sp);
    gpu_run_kernel(&cmd, (dim + 63) / 64, keep, 1, 64, 1, 1);
  }
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float dy_err = 0.0f;
  const float *ydp = (const float *)y_dy.ptr;
  for (uint32_t t = 0; t < total; t++) {
    for (uint32_t c = 0; c < dim; c++) {
      uint32_t h = c / HEAD_DIM;
      float sum = 0.0f;
      for (uint32_t lag = 0; lag < SCONV_K; lag++) {
        if (t < lag) continue;
        float w = wdsp[lag * dim + c] + dwp[(t * KV_HEADS + h) * SCONV_K + lag];
        sum += w * xp[(t - lag) * dim + c];
      }
      float ref = bf_round((sum + xp[t * dim + c]) * DYNAMIC_SCONV_SCALE);
      dy_err = fmaxf(dy_err, fabsf(ydp[t * dim + c] - ref));
    }
  }
  check_err("ink2_sconv_dynamic", dy_err, TOL_F32);

  // Test batched static and dynamic sconv on 2 resident states at pos 0..4.
  Buf r0 = gpu_alloc(gpu, slot, (size_t)ring_rows * dim * sizeof(float));
  Buf r1 = gpu_alloc(gpu, slot, (size_t)ring_rows * dim * sizeof(float));
  gpu_set_resident(&r0, 1);
  gpu_set_resident(&r1, 1);
  gpu_commit_residency(gpu);

  Buf tgt = gpu_alloc(gpu, slot, 2 * sizeof(SconvTarget));
  Buf xb = gpu_alloc(gpu, slot, 2 * (size_t)dim * sizeof(float));
  Buf db = gpu_alloc(gpu, slot, 2 * (size_t)KV_HEADS * SCONV_K * sizeof(float));
  Buf yb = gpu_alloc(gpu, slot, 2 * (size_t)dim * sizeof(float));
  memset(r0.ptr, 0, r0.size);
  memset(r1.ptr, 0, r1.size);

  SconvTarget *tp = (SconvTarget *)tgt.ptr;
  tp[0] = (SconvTarget){.ring = r0.address, .pos = 0, .pad = 0};
  tp[1] = (SconvTarget){.ring = r1.address, .pos = 0, .pad = 0};
  float bat_err = 0.0f;
  for (uint32_t step = 0; step < 5; step++) {
    tp[0].pos = step;
    tp[1].pos = step;
    memcpy(xb.ptr, xp + step * dim, dim * sizeof(float));
    memcpy((float *)xb.ptr + dim, xp + (step + 5) * dim, dim * sizeof(float));
    SconvParam sp = {
        .rows = 2, .dim = dim, .ring_rows = ring_rows, .pos0 = 0, .round = 0};
    cmd = gpu_begin(gpu);
    run_op(&cmd, k_stb, (Buf[]){xb, tgt, w_st, yb}, 4, &sp, sizeof sp,
           (dim + 63) / 64, 2, 1, 64, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);
    const float *ybp = (const float *)yb.ptr;
    for (uint32_t c = 0; c < dim; c++)
      bat_err = fmaxf(bat_err, fabsf(ybp[c] - ysp[step * dim + c]));
  }
  check_err("ink2_sconv_static_batched", bat_err, TOL_F32);

  memset(r0.ptr, 0, r0.size);
  memset(r1.ptr, 0, r1.size);
  float bat_dy_err = 0.0f;
  for (uint32_t step = 0; step < 5; step++) {
    tp[0].pos = step;
    tp[1].pos = step;
    memcpy(xb.ptr, xp + step * dim, dim * sizeof(float));
    memcpy((float *)xb.ptr + dim, xp + (step + 5) * dim, dim * sizeof(float));
    memcpy(db.ptr, dwp + step * KV_HEADS * SCONV_K,
           KV_HEADS * SCONV_K * sizeof(float));
    memcpy((float *)db.ptr + KV_HEADS * SCONV_K,
           dwp + (step + 5) * KV_HEADS * SCONV_K,
           KV_HEADS * SCONV_K * sizeof(float));
    SconvParam sp = {
        .rows = 2, .dim = dim, .ring_rows = ring_rows, .pos0 = 0, .round = 1};
    cmd = gpu_begin(gpu);
    run_op(&cmd, k_dyb, (Buf[]){xb, tgt, w_dy_st, db, yb}, 5, &sp, sizeof sp,
           (dim + 63) / 64, 2, 1, 64, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);
    const float *ybp = (const float *)yb.ptr;
    for (uint32_t c = 0; c < dim; c++)
      bat_dy_err = fmaxf(bat_dy_err, fabsf(ybp[c] - ydp[step * dim + c]));
  }
  check_err("ink2_sconv_dynamic_batched", bat_dy_err, TOL_F32);

  gpu_free(&x_all);
  gpu_free(&ring);
  gpu_free(&w_st);
  gpu_free(&w_dy_st);
  gpu_free(&dyn_all);
  gpu_free(&y_st);
  gpu_free(&y_dy);
  gpu_free(&r0);
  gpu_free(&r1);
  gpu_free(&tgt);
  gpu_free(&xb);
  gpu_free(&db);
  gpu_free(&yb);
}

// Tests ink2_qkvr_prep, ink2_rel_bias, ink2_attend_part + ink2_attend_combine,
// and ink2_attend_prefill on both global (with log scaling tau) and sliding
// window (with ring wrap-around) attention.
static void test_attention(Gpu *gpu, MemorySlot *slot) {
  int k_prep = gpu_find_kernel(gpu, "ink2_qkvr_prep");
  int k_rel = gpu_find_kernel(gpu, "ink2_rel_bias");
  int k_part = gpu_find_kernel(gpu, "ink2_attend_part");
  int k_comb = gpu_find_kernel(gpu, "ink2_attend_combine");
  int k_pref = gpu_find_kernel(gpu, "ink2_attend_prefill");

  uint32_t rng = 0x55aa55aau;
  uint32_t past = 70, qrows = 37, total = past + qrows;
  uint32_t q_stride = ((qrows + Q_PAD - 1) / Q_PAD) * Q_PAD;
  uint32_t nparts = 4;

  Buf q_in = gpu_alloc(gpu, slot, (size_t)qrows * Q_DIM * sizeof(float));
  Buf k_in = gpu_alloc(gpu, slot, (size_t)qrows * KV_DIM * sizeof(float));
  Buf v_in = gpu_alloc(gpu, slot, (size_t)qrows * KV_DIM * sizeof(float));
  Buf r_in =
      gpu_alloc(gpu, slot, (size_t)qrows * HEADS * D_REL * sizeof(float));
  Buf q_sc = gpu_alloc(gpu, slot, (size_t)Q_DIM * sizeof(float));
  Buf r_sc = gpu_alloc(gpu, slot, (size_t)HEADS * D_REL * sizeof(float));
  Buf rel_proj =
      gpu_alloc(gpu, slot, (size_t)D_REL * REL_EXTENT * sizeof(uint16_t));
  Buf q_hm = gpu_alloc(gpu, slot,
                       (size_t)HEADS * q_stride * HEAD_DIM * sizeof(uint16_t));
  Buf k_cache = gpu_alloc(gpu, slot, (size_t)total * KV_DIM * sizeof(uint16_t));
  Buf v_cache = gpu_alloc(gpu, slot, (size_t)total * KV_DIM * sizeof(uint16_t));
  Buf r_out =
      gpu_alloc(gpu, slot, (size_t)qrows * HEADS * D_REL * sizeof(float));
  Buf rel_bias =
      gpu_alloc(gpu, slot, (size_t)qrows * HEADS * REL_EXTENT * sizeof(float));
  Buf sink = gpu_alloc(gpu, slot, (size_t)qrows * HEADS * sizeof(float));
  Buf gate = gpu_alloc(gpu, slot, (size_t)qrows * Q_DIM * sizeof(float));
  Buf part = gpu_alloc(
      gpu, slot, (size_t)qrows * HEADS * nparts * PART_FLOATS * sizeof(float));
  Buf out_dec = gpu_alloc(gpu, slot, (size_t)qrows * Q_DIM * sizeof(float));
  Buf out_pref = gpu_alloc(gpu, slot, (size_t)qrows * Q_DIM * sizeof(float));
  Buf dummy_tgt = gpu_alloc(gpu, slot, sizeof(KvTarget));

  float *qip = (float *)q_in.ptr, *kip = (float *)k_in.ptr,
        *vip = (float *)v_in.ptr, *rip = (float *)r_in.ptr;
  float *qsp = (float *)q_sc.ptr, *rsp = (float *)r_sc.ptr,
        *skp = (float *)sink.ptr, *gtp = (float *)gate.ptr;
  uint16_t *rpp = (uint16_t *)rel_proj.ptr, *kcp = (uint16_t *)k_cache.ptr,
           *vcp = (uint16_t *)v_cache.ptr;

  // Fill large values to exercise both L2 clip saturation and non-saturation.
  for (uint32_t i = 0; i < qrows * Q_DIM; i++) qip[i] = rng_f32(&rng) * 4.0f;
  for (uint32_t i = 0; i < qrows * KV_DIM; i++) {
    kip[i] = rng_f32(&rng) * 4.0f;
    vip[i] = rng_f32(&rng) * 2.0f;
  }
  for (uint32_t i = 0; i < qrows * HEADS * D_REL; i++)
    rip[i] = rng_f32(&rng) * 3.0f;
  for (uint32_t i = 0; i < Q_DIM; i++) qsp[i] = 0.5f + rng_f32(&rng) * 0.5f;
  for (uint32_t i = 0; i < HEADS * D_REL; i++)
    rsp[i] = 0.5f + rng_f32(&rng) * 0.5f;
  for (uint32_t i = 0; i < D_REL * REL_EXTENT; i++)
    rpp[i] = f32_to_bf16(rng_f32(&rng) * 0.2f);
  for (uint32_t i = 0; i < past * KV_DIM; i++) {
    kcp[i] = f32_to_bf16(rng_f32(&rng) * 1.2f);
    vcp[i] = f32_to_bf16(rng_f32(&rng) * 1.2f);
  }
  for (uint32_t i = 0; i < qrows * HEADS; i++) skp[i] = rng_f32(&rng) * 2.0f;
  for (uint32_t i = 0; i < qrows * Q_DIM; i++) gtp[i] = rng_f32(&rng) * 2.0f;
  memset(q_hm.ptr, 0, q_hm.size);

  QkvrParam qp = {.rows = qrows,
                  .q_stride = q_stride,
                  .pos0 = past,
                  .ring_rows = 0,
                  .is_global = 1,
                  .batched = 0};
  RelBiasParam rbp = {
      .rows = qrows, .heads = HEADS, .d_rel = D_REL, .extent = REL_EXTENT};
  AttendDecodeParam adp = {.rows = qrows,
                           .q_stride = q_stride,
                           .pos0 = past,
                           .ring_rows = 0,
                           .window = 0,
                           .nparts = nparts,
                           .part_stride = nparts,
                           .batched = 0};
  AttendPrefillParam app = {.rows = qrows,
                            .q_stride = q_stride,
                            .pos0 = past,
                            .ring_rows = 0,
                            .window = 0};

  GpuCommandBuffer cmd = gpu_begin(gpu);
  run_op(&cmd, k_prep,
         (Buf[]){q_in, k_in, v_in, r_in, q_sc, r_sc, q_hm, k_cache, v_cache,
                 r_out, dummy_tgt},
         11, &qp, sizeof qp, QKVR_HEADS, qrows, 1, QKVR_THREADS, 1, 1);
  run_op(&cmd, k_rel, (Buf[]){r_out, rel_proj, rel_bias}, 3, &rbp, sizeof rbp,
         HEADS, qrows, 1, ROW_THREADS, 1, 1);
  run_op(&cmd, k_part,
         (Buf[]){q_hm, k_cache, v_cache, rel_bias, dummy_tgt, part}, 6, &adp,
         sizeof adp, KV_HEADS, nparts, qrows, DECODE_THREADS, 1, 1);
  run_op(&cmd, k_comb, (Buf[]){part, sink, gate, out_dec}, 4, &adp, sizeof adp,
         HEADS, qrows, 1, COMBINE_THREADS, 1, 1);
  run_op(&cmd, k_pref,
         (Buf[]){q_hm, k_cache, v_cache, rel_bias, sink, gate, out_pref}, 7,
         &app, sizeof app, (qrows + PREFILL_Q_GROUP - 1) / PREFILL_Q_GROUP,
         HEADS, 1, PREFILL_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  float dec_err = 0.0f, pref_err = 0.0f, prep_err = 0.0f;
  const uint16_t *qhmp = (const uint16_t *)q_hm.ptr;
  const float *rop = (const float *)r_out.ptr,
              *odp = (const float *)out_dec.ptr,
              *opp = (const float *)out_pref.ptr;

  for (uint32_t r = 0; r < qrows; r++) {
    uint32_t pos = past + r;
    float tau =
        1.0f + LOG_ALPHA * logf(fmaxf((float)(pos + 1) / LOG_FLOOR, 1.0f));
    for (uint32_t h = 0; h < HEADS; h++) {
      double sq = 0.0;
      float tmp[HEAD_DIM];
      for (uint32_t d = 0; d < HEAD_DIM; d++) {
        tmp[d] = qip[r * Q_DIM + h * HEAD_DIM + d] * qsp[h * HEAD_DIM + d];
        sq += (double)tmp[d] * tmp[d];
      }
      float s = fminf(QK_CLIP / (sqrtf((float)sq) + CLIP_EPS), 1.0f);
      for (uint32_t d = 0; d < HEAD_DIM; d++) {
        float ref = bf_round(bf_round(tmp[d] * s) * tau);
        float got = bf16_to_f32(qhmp[(h * q_stride + r) * HEAD_DIM + d]);
        prep_err = fmaxf(prep_err, fabsf(got - ref));
      }
      double rsq = 0.0;
      float rtmp[D_REL];
      for (uint32_t d = 0; d < D_REL; d++) {
        rtmp[d] = rip[(r * HEADS + h) * D_REL + d] * rsp[h * D_REL + d];
        rsq += (double)rtmp[d] * rtmp[d];
      }
      float rs = fminf(REL_CLIP / (sqrtf((float)rsq) + CLIP_EPS), 1.0f);
      for (uint32_t d = 0; d < D_REL; d++) {
        float ref = bf_round(bf_round(rtmp[d] * rs) * tau);
        prep_err =
            fmaxf(prep_err, fabsf(rop[(r * HEADS + h) * D_REL + d] - ref));
      }
    }
  }
  check_err("ink2_qkvr_prep", prep_err, TOL_F32);

  for (uint32_t r = 0; r < qrows; r++) {
    uint32_t qpos = past + r;
    for (uint32_t h = 0; h < HEADS; h++) {
      uint32_t kvh = h / GQA;
      float sc[128], mx = skp[r * HEADS + h];
      for (uint32_t j = 0; j <= qpos; j++) {
        double dotv = 0.0;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
          float qv = bf16_to_f32(qhmp[(h * q_stride + r) * HEAD_DIM + d]);
          float kv = bf16_to_f32(kcp[j * KV_DIM + kvh * HEAD_DIM + d]);
          dotv += (double)qv * kv;
        }
        float val = (float)dotv * scale;
        uint32_t dist = qpos - j;
        if (dist < REL_EXTENT) {
          float rb = 0.0f;
          for (uint32_t k = 0; k < D_REL; k++)
            rb += rop[(r * HEADS + h) * D_REL + k] *
                  bf16_to_f32(rpp[k * REL_EXTENT + dist]);
          val += rb;
        }
        sc[j] = val;
        mx = fmaxf(mx, val);
      }
      float denom = expf(skp[r * HEADS + h] - mx);
      for (uint32_t j = 0; j <= qpos; j++) {
        sc[j] = expf(sc[j] - mx);
        denom += sc[j];
      }
      for (uint32_t d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j <= qpos; j++)
          acc += sc[j] * bf16_to_f32(vcp[j * KV_DIM + kvh * HEAD_DIM + d]);
        float v = bf_round(acc / denom);
        float g = gtp[r * Q_DIM + h * HEAD_DIM + d];
        float ref = bf_round(v * cpu_sigm(g));
        uint32_t idx = r * Q_DIM + h * HEAD_DIM + d;
        dec_err = fmaxf(dec_err, fabsf(odp[idx] - ref));
        pref_err = fmaxf(pref_err, fabsf(opp[idx] - ref));
      }
    }
  }
  check_err("ink2_attend_decode global", dec_err, TOL_ATTN);
  check_err("ink2_attend_prefill global", pref_err, TOL_ATTN);

  // Test sliding window with ring wrap-around (pos0 = 1050, ring = 1040).
  uint32_t ring_rows = WINDOW + 16, sw_pos0 = 1050, sw_rows = 20;
  uint32_t sw_qstride = ((sw_rows + Q_PAD - 1) / Q_PAD) * Q_PAD;
  Buf sw_k =
      gpu_alloc(gpu, slot, (size_t)ring_rows * KV_DIM * sizeof(uint16_t));
  Buf sw_v =
      gpu_alloc(gpu, slot, (size_t)ring_rows * KV_DIM * sizeof(uint16_t));
  uint16_t *swkp = (uint16_t *)sw_k.ptr, *swvp = (uint16_t *)sw_v.ptr;
  for (uint32_t i = 0; i < ring_rows * KV_DIM; i++) {
    swkp[i] = f32_to_bf16(rng_f32(&rng) * 0.8f);
    swvp[i] = f32_to_bf16(rng_f32(&rng) * 0.8f);
  }
  QkvrParam sw_qp = {.rows = sw_rows,
                     .q_stride = sw_qstride,
                     .pos0 = sw_pos0,
                     .ring_rows = ring_rows,
                     .is_global = 0,
                     .batched = 0};
  RelBiasParam sw_rbp = {
      .rows = sw_rows, .heads = HEADS, .d_rel = D_REL, .extent = REL_EXTENT};
  AttendDecodeParam sw_adp = {.rows = sw_rows,
                              .q_stride = sw_qstride,
                              .pos0 = sw_pos0,
                              .ring_rows = ring_rows,
                              .window = WINDOW,
                              .nparts = nparts,
                              .part_stride = nparts,
                              .batched = 0};
  AttendPrefillParam sw_app = {.rows = sw_rows,
                               .q_stride = sw_qstride,
                               .pos0 = sw_pos0,
                               .ring_rows = ring_rows,
                               .window = WINDOW};

  cmd = gpu_begin(gpu);
  run_op(&cmd, k_prep,
         (Buf[]){q_in, k_in, v_in, r_in, q_sc, r_sc, q_hm, sw_k, sw_v, r_out,
                 dummy_tgt},
         11, &sw_qp, sizeof sw_qp, QKVR_HEADS, sw_rows, 1, QKVR_THREADS, 1, 1);
  run_op(&cmd, k_rel, (Buf[]){r_out, rel_proj, rel_bias}, 3, &sw_rbp,
         sizeof sw_rbp, HEADS, sw_rows, 1, ROW_THREADS, 1, 1);
  run_op(&cmd, k_part, (Buf[]){q_hm, sw_k, sw_v, rel_bias, dummy_tgt, part}, 6,
         &sw_adp, sizeof sw_adp, KV_HEADS, nparts, sw_rows, DECODE_THREADS, 1,
         1);
  run_op(&cmd, k_comb, (Buf[]){part, sink, gate, out_dec}, 4, &sw_adp,
         sizeof sw_adp, HEADS, sw_rows, 1, COMBINE_THREADS, 1, 1);
  run_op(&cmd, k_pref,
         (Buf[]){q_hm, sw_k, sw_v, rel_bias, sink, gate, out_pref}, 7, &sw_app,
         sizeof sw_app, (sw_rows + PREFILL_Q_GROUP - 1) / PREFILL_Q_GROUP,
         HEADS, 1, PREFILL_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float sw_diff = 0.0f;
  for (uint32_t i = 0; i < sw_rows * Q_DIM; i++)
    sw_diff = fmaxf(sw_diff, fabsf(odp[i] - opp[i]));
  check_err("sliding decode vs prefill (ring wrap)", sw_diff, TOL_ATTN);

  float sw_cpu_err = 0.0f;
  uint32_t cr = 19, ch = 7, ckvh = ch / GQA, cqpos = sw_pos0 + cr;
  uint32_t klo = cqpos + 1 - WINDOW;
  float sw_sc[WINDOW], sw_mx = skp[cr * HEADS + ch];
  for (uint32_t j = klo; j <= cqpos; j++) {
    uint32_t sl = j % ring_rows;
    double dotv = 0.0;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
      float qv = bf16_to_f32(qhmp[(ch * sw_qstride + cr) * HEAD_DIM + d]);
      float kv = bf16_to_f32(swkp[sl * KV_DIM + ckvh * HEAD_DIM + d]);
      dotv += (double)qv * kv;
    }
    float val = (float)dotv * scale;
    uint32_t dist = cqpos - j;
    for (uint32_t k = 0; k < D_REL; k++)
      val += rop[(cr * HEADS + ch) * D_REL + k] *
             bf16_to_f32(rpp[k * REL_EXTENT + dist]);
    sw_sc[j - klo] = val;
    sw_mx = fmaxf(sw_mx, val);
  }
  float sw_den = expf(skp[cr * HEADS + ch] - sw_mx);
  for (uint32_t j = 0; j < WINDOW; j++) {
    sw_sc[j] = expf(sw_sc[j] - sw_mx);
    sw_den += sw_sc[j];
  }
  for (uint32_t d = 0; d < HEAD_DIM; d++) {
    float acc = 0.0f;
    for (uint32_t j = klo; j <= cqpos; j++) {
      uint32_t sl = j % ring_rows;
      acc +=
          sw_sc[j - klo] * bf16_to_f32(swvp[sl * KV_DIM + ckvh * HEAD_DIM + d]);
    }
    float ref = bf_round(bf_round(acc / sw_den) *
                         cpu_sigm(gtp[cr * Q_DIM + ch * HEAD_DIM + d]));
    sw_cpu_err =
        fmaxf(sw_cpu_err, fabsf(odp[cr * Q_DIM + ch * HEAD_DIM + d] - ref));
  }
  check_err("sliding wrapped CPU ref", sw_cpu_err, TOL_ATTN);

  // Verify global log scaling tau > 1 at pos = 16384.
  QkvrParam hi_qp = {.rows = 1,
                     .q_stride = Q_PAD,
                     .pos0 = 16384,
                     .ring_rows = ring_rows,
                     .is_global = 1,
                     .batched = 0};
  cmd = gpu_begin(gpu);
  run_op(&cmd, k_prep,
         (Buf[]){q_in, k_in, v_in, r_in, q_sc, r_sc, q_hm, sw_k, sw_v, r_out,
                 dummy_tgt},
         11, &hi_qp, sizeof hi_qp, QKVR_HEADS, 1, 1, QKVR_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  float tau_hi = 1.0f + LOG_ALPHA * logf(16385.0f / LOG_FLOOR);
  float hi_err = 0.0f;
  for (uint32_t h = 0; h < HEADS; h++) {
    double sq = 0.0;
    float tmp[HEAD_DIM];
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
      tmp[d] = qip[h * HEAD_DIM + d] * qsp[h * HEAD_DIM + d];
      sq += (double)tmp[d] * tmp[d];
    }
    float s = fminf(QK_CLIP / (sqrtf((float)sq) + CLIP_EPS), 1.0f);
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
      float ref = bf_round(bf_round(tmp[d] * s) * tau_hi);
      float got = bf16_to_f32(qhmp[h * Q_PAD * HEAD_DIM + d]);
      hi_err = fmaxf(hi_err, fabsf(got - ref));
    }
  }
  check_err("global log-scaling tau at pos 16384", hi_err, TOL_F32);

  gpu_free(&q_in);
  gpu_free(&k_in);
  gpu_free(&v_in);
  gpu_free(&r_in);
  gpu_free(&q_sc);
  gpu_free(&r_sc);
  gpu_free(&rel_proj);
  gpu_free(&q_hm);
  gpu_free(&k_cache);
  gpu_free(&v_cache);
  gpu_free(&r_out);
  gpu_free(&rel_bias);
  gpu_free(&sink);
  gpu_free(&gate);
  gpu_free(&part);
  gpu_free(&out_dec);
  gpu_free(&out_pref);
  gpu_free(&dummy_tgt);
  gpu_free(&sw_k);
  gpu_free(&sw_v);
}

// Tests ink2_route_topk, NVFP4 MoE (mv and mm paths), MXFP8 MoE (mv and mm
// paths), and ink2_moe_combine.
static void test_router_and_moe(Gpu *gpu, MemorySlot *slot) {
  int k_route = gpu_find_kernel(gpu, "ink2_route_topk");
  int k_gather = gpu_find_kernel(gpu, "ink2_moe_gather_bf16");
  int k_nv_up_mv = gpu_find_kernel(gpu, "ink2_moe_nvfp4_up_mv");
  int k_nv_dn_mv = gpu_find_kernel(gpu, "ink2_moe_nvfp4_down_mv");
  int k_nv_up_mm = gpu_find_kernel(gpu, "ink2_moe_nvfp4_up_mm");
  int k_nv_dn_mm = gpu_find_kernel(gpu, "ink2_moe_nvfp4_down_mm");
  int k_mx_up_mv = gpu_find_kernel(gpu, "ink2_moe_mxfp8_up_mv");
  int k_mx_dn_mv = gpu_find_kernel(gpu, "ink2_moe_mxfp8_down_mv");
  int k_mx_up_mm = gpu_find_kernel(gpu, "ink2_moe_mxfp8_up_mm");
  int k_mx_dn_mm = gpu_find_kernel(gpu, "ink2_moe_mxfp8_down_mm");
  int k_comb = gpu_find_kernel(gpu, "ink2_moe_combine");

  uint32_t rng = 0xdeadbeefu;
  uint32_t rrows = 4;
  Buf rlog = gpu_alloc(gpu, slot, (size_t)rrows * EXPERTS * sizeof(float));
  Buf rbias = gpu_alloc(gpu, slot, (size_t)EXPERTS * sizeof(float));
  Buf rids = gpu_alloc(gpu, slot, (size_t)rrows * TOPK * sizeof(uint32_t));
  Buf rw = gpu_alloc(gpu, slot, (size_t)rrows * TOPK * sizeof(float));

  float *rlp = (float *)rlog.ptr, *rbp = (float *)rbias.ptr;
  for (uint32_t i = 0; i < rrows * EXPERTS; i++) rlp[i] = rng_f32(&rng) * 3.0f;
  for (uint32_t i = 0; i < EXPERTS; i++) rbp[i] = rng_f32(&rng) * 0.5f;
  // Force an exact tie on row 0 between experts 12 and 77.
  rlp[12] = 5.0f;
  rlp[77] = 5.0f;
  rbp[12] = 0.25f;
  rbp[77] = 0.25f;

  RouteParam rp = {
      .rows = rrows, .experts = EXPERTS, .topk = TOPK, .scale = ROUTE_SCALE};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  run_op(&cmd, k_route, (Buf[]){rlog, rbias, rids, rw}, 4, &rp, sizeof rp,
         rrows, 1, 1, ROUTE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  const uint32_t *idp = (const uint32_t *)rids.ptr;
  const float *rwp = (const float *)rw.ptr;
  int ids_ok = 1;
  float rw_err = 0.0f;
  for (uint32_t r = 0; r < rrows; r++) {
    uint32_t ref_id[TOPK];
    float sc[EXPERTS], key[EXPERTS];
    for (uint32_t e = 0; e < EXPERTS; e++) {
      sc[e] = cpu_sigm(rlp[r * EXPERTS + e]);
      key[e] = sc[e] + rbp[e];
    }
    int used[EXPERTS] = {0};
    float sum = 0.0f;
    for (uint32_t k = 0; k < TOPK; k++) {
      uint32_t best = 0;
      float best_k = -INFINITY;
      for (uint32_t e = 0; e < EXPERTS; e++) {
        if (!used[e] && key[e] > best_k) best_k = key[e], best = e;
      }
      used[best] = 1;
      ref_id[k] = best;
      sum += sc[best];
    }
    for (uint32_t k = 0; k < TOPK; k++) {
      if (idp[r * TOPK + k] != ref_id[k]) ids_ok = 0;
      float ref_w = sc[ref_id[k]] * ROUTE_SCALE / (sum + ROUTE_EPS);
      rw_err = fmaxf(rw_err, fabsf(rwp[r * TOPK + k] - ref_w));
    }
  }
  check(ids_ok, "ink2_route_topk expert ids and tie break");
  check_err("ink2_route_topk weights", rw_err, TOL_F32);

  // Test NVFP4 and MXFP8 experts on 2 experts with small tile (3 pairs -> _mv)
  // and large tile (11 pairs -> _mm) in the SAME batch (14 tokens).
  uint32_t dim = 128, inter = 64, n13 = 2 * inter, nexp = 2;
  uint32_t tokens = 14, topk_small = 1, npairs = tokens;
  Buf x = gpu_alloc(gpu, slot, (size_t)tokens * dim * sizeof(float));
  Buf x_bf =
      gpu_alloc(gpu, slot, (size_t)(npairs + MM_TILE) * dim * sizeof(uint16_t));
  Buf ptok = gpu_alloc(gpu, slot, (size_t)npairs * sizeof(int));
  Buf pw = gpu_alloc(gpu, slot, (size_t)npairs * sizeof(float));
  Buf tp = gpu_alloc(gpu, slot, (size_t)tokens * sizeof(int));
  Buf tiles = gpu_alloc(gpu, slot, 2 * sizeof(MoeTile));
  Buf h_bf = gpu_alloc(gpu, slot,
                       (size_t)(npairs + MM_TILE) * inter * sizeof(uint16_t));
  Buf y_pair = gpu_alloc(gpu, slot, (size_t)npairs * dim * sizeof(float));
  Buf y_out = gpu_alloc(gpu, slot, (size_t)tokens * dim * sizeof(float));

  float *xp = (float *)x.ptr, *pwp = (float *)pw.ptr;
  int *ptp = (int *)ptok.ptr, *tpp = (int *)tp.ptr;
  for (uint32_t i = 0; i < tokens * dim; i++)
    xp[i] = bf_round(rng_f32(&rng) * 0.5f);
  for (uint32_t i = 0; i < npairs; i++) {
    ptp[i] = (int)i;
    tpp[i] = (int)i;
    pwp[i] = 0.5f + rng_f32(&rng) * 0.3f;
  }
  MoeTile *tlp = (MoeTile *)tiles.ptr;
  tlp[0] = (MoeTile){.expert = 0, .start = 0, .count = 3, .pad = 0};
  tlp[1] = (MoeTile){.expert = 1, .start = 3, .count = 11, .pad = 0};
  memset(x_bf.ptr, 0, x_bf.size);
  memset(h_bf.ptr, 0, h_bf.size);

  Buf nv_w13 = gpu_alloc(gpu, slot, (size_t)nexp * n13 * (dim / 2));
  Buf nv_s13 = gpu_alloc(gpu, slot, (size_t)nexp * n13 * (dim / NVFP4_BLOCK));
  Buf nv_f13 = gpu_alloc(gpu, slot, (size_t)nexp * sizeof(float));
  Buf nv_w2 = gpu_alloc(gpu, slot, (size_t)nexp * dim * (inter / 2));
  Buf nv_s2 = gpu_alloc(gpu, slot, (size_t)nexp * dim * (inter / NVFP4_BLOCK));
  Buf nv_f2 = gpu_alloc(gpu, slot, (size_t)nexp * sizeof(float));

  uint8_t *nw13 = nv_w13.ptr, *ns13 = nv_s13.ptr, *nw2 = nv_w2.ptr,
          *ns2 = nv_s2.ptr;
  float *nf13 = (float *)nv_f13.ptr, *nf2 = (float *)nv_f2.ptr;
  for (size_t i = 0; i < nv_w13.size; i++) nw13[i] = (uint8_t)rng_next(&rng);
  for (size_t i = 0; i < nv_s13.size; i++)
    ns13[i] = (uint8_t)(0x28u + (rng_next(&rng) % 0x18u));
  for (size_t i = 0; i < nv_w2.size; i++) nw2[i] = (uint8_t)rng_next(&rng);
  for (size_t i = 0; i < nv_s2.size; i++)
    ns2[i] = (uint8_t)(0x28u + (rng_next(&rng) % 0x18u));
  nf13[0] = 0.13f;
  nf13[1] = 0.17f;
  nf2[0] = 0.11f;
  nf2[1] = 0.19f;

  uint32_t dim4 = dim / VEC4_WIDTH;
  MoeParam mp13 = {.n = n13,
                   .k = dim,
                   .npairs = npairs,
                   .ntiles = 2,
                   .gate_cap = GATE_CAP,
                   .up_cap = UP_CAP};
  MoeParam mp2 = {.n = dim,
                  .k = inter,
                  .npairs = npairs,
                  .ntiles = 2,
                  .gate_cap = GATE_CAP,
                  .up_cap = UP_CAP};
  MoeCombineParam mcp = {.rows = tokens, .dim = dim, .topk = topk_small};

  cmd = gpu_begin(gpu);
  run_op(&cmd, k_gather, (Buf[]){x, ptok, x_bf}, 3, &dim4, sizeof dim4,
         (dim4 + 31) / 32, npairs, 1, 32, 1, 1);
  run_op(&cmd, k_nv_up_mv,
         (Buf[]){nv_w13, nv_s13, nv_f13, x, ptok, tiles, h_bf}, 7, &mp13,
         sizeof mp13, (n13 + MV_ROWS_PER_TG - 1) / MV_ROWS_PER_TG, 2, 1,
         MV_THREADS, 1, 1);
  run_op(&cmd, k_nv_up_mm, (Buf[]){nv_w13, nv_s13, nv_f13, x_bf, tiles, h_bf},
         6, &mp13, sizeof mp13, (n13 + MM_TILE - 1) / MM_TILE, 2, 1, MM_THREADS,
         1, 1);
  run_op(&cmd, k_nv_dn_mv,
         (Buf[]){nv_w2, nv_s2, nv_f2, h_bf, pw, tiles, y_pair}, 7, &mp2,
         sizeof mp2, (dim + MV_ROWS_PER_TG - 1) / MV_ROWS_PER_TG, 2, 1,
         MV_THREADS, 1, 1);
  run_op(&cmd, k_nv_dn_mm,
         (Buf[]){nv_w2, nv_s2, nv_f2, h_bf, pw, tiles, y_pair}, 7, &mp2,
         sizeof mp2, (dim + MM_TILE - 1) / MM_TILE, 2, 1, MM_THREADS, 1, 1);
  run_op(&cmd, k_comb, (Buf[]){y_pair, tp, y_out}, 3, &mcp, sizeof mcp,
         (dim + 63) / 64, tokens, 1, 64, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float nv_err = 0.0f;
  const float *yop = (const float *)y_out.ptr;
  for (uint32_t t = 0; t < tokens; t++) {
    uint32_t e = t < 3 ? 0 : 1;
    float h_ref[64];
    for (uint32_t j = 0; j < inter; j++) {
      double acc_g = 0.0, acc_u = 0.0;
      for (uint32_t c = 0; c < dim; c++) {
        uint8_t bg = nw13[(e * n13 + 2 * j) * (dim / 2) + c / 2];
        uint8_t bu = nw13[(e * n13 + 2 * j + 1) * (dim / 2) + c / 2];
        float vg = E2M1_TABLE[(c & 1) ? (bg >> E2M1_SHIFT) : (bg & E2M1_MASK)];
        float vu = E2M1_TABLE[(c & 1) ? (bu >> E2M1_SHIFT) : (bu & E2M1_MASK)];
        float sg = e4m3_to_f32(
            ns13[(e * n13 + 2 * j) * (dim / NVFP4_BLOCK) + c / NVFP4_BLOCK]);
        float su =
            e4m3_to_f32(ns13[(e * n13 + 2 * j + 1) * (dim / NVFP4_BLOCK) +
                             c / NVFP4_BLOCK]);
        acc_g += (double)(vg * sg * nf13[e]) * xp[t * dim + c];
        acc_u += (double)(vu * su * nf13[e]) * xp[t * dim + c];
      }
      h_ref[j] =
          cpu_softcapped_swiglu(bf_round((float)acc_g), bf_round((float)acc_u));
    }
    for (uint32_t d = 0; d < dim; d++) {
      double acc = 0.0;
      for (uint32_t j = 0; j < inter; j++) {
        uint8_t b = nw2[(e * dim + d) * (inter / 2) + j / 2];
        float v = E2M1_TABLE[(j & 1) ? (b >> E2M1_SHIFT) : (b & E2M1_MASK)];
        float s = e4m3_to_f32(
            ns2[(e * dim + d) * (inter / NVFP4_BLOCK) + j / NVFP4_BLOCK]);
        acc += (double)(v * s * nf2[e]) * h_ref[j];
      }
      float ref = bf_round(bf_round((float)acc) * pwp[t]);
      nv_err = fmaxf(nv_err, fabsf(yop[t * dim + d] - ref));
    }
  }
  check_err("NVFP4 MoE (mv + mm)", nv_err, TOL_BF16);

  // MXFP8 expert buffers (same 2-tile split: mv for 3 pairs, mm for 11 pairs).
  Buf mx_w13 = gpu_alloc(gpu, slot, (size_t)nexp * n13 * dim);
  Buf mx_s13 = gpu_alloc(gpu, slot, (size_t)nexp * n13 * (dim / MXFP8_BLOCK));
  Buf mx_w2 = gpu_alloc(gpu, slot, (size_t)nexp * dim * inter);
  Buf mx_s2 = gpu_alloc(gpu, slot, (size_t)nexp * dim * (inter / MXFP8_BLOCK));

  uint8_t *mw13 = mx_w13.ptr, *ms13 = mx_s13.ptr, *mw2 = mx_w2.ptr,
          *ms2 = mx_s2.ptr;
  for (size_t i = 0; i < mx_w13.size; i++) {
    uint8_t b = (uint8_t)(rng_next(&rng) & 0xFFu);
    mw13[i] = (b & 0x7Fu) == 0x7Fu ? 0x7Eu : b;
  }
  for (size_t i = 0; i < mx_s13.size; i++)
    ms13[i] = (uint8_t)(123u + (rng_next(&rng) % 5u));
  for (size_t i = 0; i < mx_w2.size; i++) {
    uint8_t b = (uint8_t)(rng_next(&rng) & 0xFFu);
    mw2[i] = (b & 0x7Fu) == 0x7Fu ? 0x7Eu : b;
  }
  for (size_t i = 0; i < mx_s2.size; i++)
    ms2[i] = (uint8_t)(123u + (rng_next(&rng) % 5u));
  memset(h_bf.ptr, 0, h_bf.size);

  cmd = gpu_begin(gpu);
  run_op(&cmd, k_mx_up_mv, (Buf[]){mx_w13, mx_s13, x, ptok, tiles, h_bf}, 6,
         &mp13, sizeof mp13, (n13 + MV_ROWS_PER_TG - 1) / MV_ROWS_PER_TG, 2, 1,
         MV_THREADS, 1, 1);
  run_op(&cmd, k_mx_up_mm, (Buf[]){mx_w13, mx_s13, x_bf, tiles, h_bf}, 5, &mp13,
         sizeof mp13, (n13 + MM_TILE - 1) / MM_TILE, 2, 1, MM_THREADS, 1, 1);
  run_op(&cmd, k_mx_dn_mv, (Buf[]){mx_w2, mx_s2, h_bf, pw, tiles, y_pair}, 6,
         &mp2, sizeof mp2, (dim + MV_ROWS_PER_TG - 1) / MV_ROWS_PER_TG, 2, 1,
         MV_THREADS, 1, 1);
  run_op(&cmd, k_mx_dn_mm, (Buf[]){mx_w2, mx_s2, h_bf, pw, tiles, y_pair}, 6,
         &mp2, sizeof mp2, (dim + MM_TILE - 1) / MM_TILE, 2, 1, MM_THREADS, 1,
         1);
  run_op(&cmd, k_comb, (Buf[]){y_pair, tp, y_out}, 3, &mcp, sizeof mcp,
         (dim + 63) / 64, tokens, 1, 64, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float mx_err = 0.0f;
  for (uint32_t t = 0; t < tokens; t++) {
    uint32_t e = t < 3 ? 0 : 1;
    float h_ref[64];
    for (uint32_t j = 0; j < inter; j++) {
      double acc_g = 0.0, acc_u = 0.0;
      for (uint32_t c = 0; c < dim; c++) {
        float vg = e4m3_to_f32(mw13[(e * n13 + 2 * j) * dim + c]);
        float vu = e4m3_to_f32(mw13[(e * n13 + 2 * j + 1) * dim + c]);
        float sg = e8m0_to_f32(
            ms13[(e * n13 + 2 * j) * (dim / MXFP8_BLOCK) + c / MXFP8_BLOCK]);
        float su =
            e8m0_to_f32(ms13[(e * n13 + 2 * j + 1) * (dim / MXFP8_BLOCK) +
                             c / MXFP8_BLOCK]);
        acc_g += (double)(vg * sg) * xp[t * dim + c];
        acc_u += (double)(vu * su) * xp[t * dim + c];
      }
      h_ref[j] =
          cpu_softcapped_swiglu(bf_round((float)acc_g), bf_round((float)acc_u));
    }
    for (uint32_t d = 0; d < dim; d++) {
      double acc = 0.0;
      for (uint32_t j = 0; j < inter; j++) {
        float v = e4m3_to_f32(mw2[(e * dim + d) * inter + j]);
        float s = e8m0_to_f32(
            ms2[(e * dim + d) * (inter / MXFP8_BLOCK) + j / MXFP8_BLOCK]);
        acc += (double)(v * s) * h_ref[j];
      }
      float ref = bf_round(bf_round((float)acc) * pwp[t]);
      mx_err = fmaxf(mx_err, fabsf(yop[t * dim + d] - ref));
    }
  }
  check_err("MXFP8 MoE (mv + mm)", mx_err, TOL_BF16);

  gpu_free(&rlog);
  gpu_free(&rbias);
  gpu_free(&rids);
  gpu_free(&rw);
  gpu_free(&x);
  gpu_free(&x_bf);
  gpu_free(&ptok);
  gpu_free(&pw);
  gpu_free(&tp);
  gpu_free(&tiles);
  gpu_free(&h_bf);
  gpu_free(&y_pair);
  gpu_free(&y_out);
  gpu_free(&nv_w13);
  gpu_free(&nv_s13);
  gpu_free(&nv_f13);
  gpu_free(&nv_w2);
  gpu_free(&nv_s2);
  gpu_free(&nv_f2);
  gpu_free(&mx_w13);
  gpu_free(&mx_s13);
  gpu_free(&mx_w2);
  gpu_free(&mx_s2);
}

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);

  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *gpu = gpu_new(src, len);

  test_embed_norm_and_softcap(gpu, slot);
  test_attn_res_and_msg_gate(gpu, slot);
  test_short_convs(gpu, slot);
  test_attention(gpu, slot);
  test_router_and_moe(gpu, slot);

  gpu_return_freed_memory(gpu);
  return failures != 0;
}
