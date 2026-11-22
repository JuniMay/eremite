#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tests all MiniMax-H3-Image DiT and TokenRefiner Metal kernels in
// src/kernels/minimax_h3_dit.metal against CPU references without a checkpoint.

// DiT attention geometry matching src/kernels/minimax_h3_dit.metal.
static constexpr int DIT_HEADS = 56, DIT_HEAD_DIM = 128,
                     DIT_INNER = DIT_HEADS * DIT_HEAD_DIM,
                     DIT_QKV_WIDTH = 3 * DIT_INNER, DIT_ROT_PAIRS = 48,
                     DIT_ROPE_FREQS = 16, DIT_ADALN_RANK = 8;

// Tile padding, threadgroup sizes and tolerances.
static constexpr uint32_t TILE_M = 64, TILE_M_I8 = 128, TILE_N = 64,
                          MM_THREADS = 128, F32MM_TILE_M = 64,
                          F32MM_TILE_N = 32, F32MM_THREADS = 128,
                          CONVROT_THREADS = 256, NORM_THREADS = 256,
                          HEADS_THREADS = 64, ELEM_THREADS = 256,
                          GRID2D_THREADS = 32;
static constexpr float NORM_EPS = 1e-5f, FP32_TOL = 1e-4f, BF16_TOL = 5e-2f;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Active rows, inner width, input row stride and scale flag.
  uint32_t m, k, lda, has_scale;
} Bf16XParam;

// Parameters passed to minimax_h3_mm_bf16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} Bf16MMParam;

// Parameters passed to minimax_h3_mm_f32.
typedef struct {
  // Active rows, output width, inner width and output row stride.
  uint32_t m, n, k, ldy;
} F32MMParam;

// Parameters passed to minimax_h3_dit_adaln.
typedef struct {
  // Number of timesteps, output width per timestep and F16 bias flag.
  uint32_t num_t, out_dim, bias_f16;
} DitAdalnParam;

// Parameters passed to minimax_h3_dit_norm_mod.
typedef struct {
  // Row width, modulation stride, shift offset and scale offset.
  uint32_t dim, mod_stride, shift_off, scale_off;
  // RMSNorm epsilon.
  float eps;
} DitNormModParam;

// Parameters passed to minimax_h3_dit_norm_mod_convrot.
typedef struct {
  // Active rows, row width, modulation stride, shift offset and scale offset.
  uint32_t m, dim, mod_stride, shift_off, scale_off;
  // RMSNorm epsilon.
  float eps;
} DitNormModConvRotParam;

// Parameters passed to minimax_h3_convrot_quant_i8.
typedef struct {
  // Active rows, inner width and input row stride.
  uint32_t m, k, lda;
} ConvRotParam;

// Parameters passed to minimax_h3_dit_gate_add.
typedef struct {
  // Rows, row width, modulation stride and gate offset.
  uint32_t rows, dim, mod_stride, gate_off;
} DitGateAddParam;

// Parameters passed to minimax_h3_dit_heads.
typedef struct {
  // Active sequence length, padded head stride and RoPE flag.
  uint32_t rows, padded, use_rope;
  // Per-head affine QK RMSNorm epsilon.
  float eps;
} DitHeadsParam;

// Parameters passed to minimax_h3_dit_attend.
typedef struct {
  // Active sequence length, padded head stride, starting tile and slice tiles.
  uint32_t rows, padded, tile_offset, slice_tiles;
} DitAttendParam;

// Deterministic PRNG for synthetic test inputs.
static uint32_t rng_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// Uniform float in [-1, 1].
static float rng_float(uint32_t *state) {
  return (float)(rng_next(state) & 0xffffffu) / (float)0x800000u - 1.0f;
}

// Rounds FP32 to BF16 (round-to-nearest-even) and returns raw 16-bit bits.
static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return (uint16_t)(u >> 16);
}

// Converts raw BF16 bits to FP32.
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// Tests minimax_h3_mm_bf16 with and without bias across tile boundaries.
static void test_mm_bf16(Gpu *gpu, MemorySlot *slot) {
  int k_x = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_bf16");
  static const int ms[] = {17, 65};
  int k = 128, n = 128;
  uint32_t rng = 0x10293847u;

  float worst = 0.0f;
  for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
    int m = ms[mi];
    uint32_t padded = ((uint32_t)m + TILE_M - 1) / TILE_M * TILE_M;
    for (int has_bias = 0; has_bias <= 1; has_bias++) {
      Buf x = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * sizeof(float));
      Buf dummy = gpu_alloc(gpu, slot, 16);
      Buf xb =
          gpu_alloc(gpu, slot, (size_t)padded * (size_t)k * sizeof(uint16_t));
      Buf w = gpu_alloc(gpu, slot, (size_t)n * (size_t)k * sizeof(uint16_t));
      Buf b = gpu_alloc(gpu, slot, (size_t)n * sizeof(float));
      Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * sizeof(float));

      float *xp = (float *)x.ptr, *bp = (float *)b.ptr;
      uint16_t *wp = (uint16_t *)w.ptr;
      for (int i = 0; i < m * k; i++) xp[i] = 0.5f * rng_float(&rng);
      for (int i = 0; i < n * k; i++)
        wp[i] = f32_to_bf16(0.5f * rng_float(&rng));
      for (int i = 0; i < n; i++) bp[i] = rng_float(&rng);

      Bf16XParam xp_arg = {(uint32_t)m, (uint32_t)k, (uint32_t)k, 0};
      Bf16MMParam mp_arg = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                            (uint32_t)has_bias};
      GpuCommandBuffer cmd = gpu_begin(gpu);
      gpu_select_kernel(&cmd, k_x);
      gpu_bind_buffer(&cmd, 0, &x, 0);
      gpu_bind_buffer(&cmd, 1, &dummy, 0);
      gpu_bind_buffer(&cmd, 2, &xb, 0);
      gpu_copy_bytes(&cmd, 3, &xp_arg, sizeof xp_arg);
      gpu_run_kernel(&cmd,
                     ((uint32_t)k / 4u + GRID2D_THREADS - 1) / GRID2D_THREADS,
                     padded, 1, GRID2D_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w, 0);
      gpu_bind_buffer(&cmd, 1, &xb, 0);
      gpu_bind_buffer(&cmd, 2, &b, 0);
      gpu_bind_buffer(&cmd, 3, &y, 0);
      gpu_copy_bytes(&cmd, 4, &mp_arg, sizeof mp_arg);
      gpu_run_kernel(&cmd, ((uint32_t)n + TILE_N - 1) / TILE_N, padded / TILE_M,
                     1, MM_THREADS, 1, 1);
      gpu_commit(&cmd);
      gpu_sync(gpu);

      const float *yp = (const float *)y.ptr;
      for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
          float ref = has_bias ? bp[c] : 0.0f;
          for (int j = 0; j < k; j++)
            ref += bf16_to_f32(f32_to_bf16(xp[r * k + j])) *
                   bf16_to_f32(wp[c * k + j]);
          float d = fabsf(yp[r * n + c] - ref);
          if (d > worst) worst = d;
        }
      }

      gpu_free(&y);
      gpu_free(&b);
      gpu_free(&w);
      gpu_free(&xb);
      gpu_free(&dummy);
      gpu_free(&x);
    }
  }
  check(worst < BF16_TOL,
        "minimax_h3_mm_bf16 matches CPU reference with and without bias");
}

// Tests minimax_h3_mm_f32 across patch-embed, PDD output, multi-tile and odd
// shapes with NaN guard tails past the input buffers.
static void test_mm_f32(Gpu *gpu, MemorySlot *slot) {
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_f32");
  static const int shapes[5][3] = {
      {17, 96, 128}, {11, 128, 96}, {1, 32, 32}, {65, 32, 96}, {73, 19, 37}};
  uint32_t rng = 0x55aa1234u;
  float worst = 0.0f;
  int all_finite = 1;

  for (int si = 0; si < 5; si++) {
    int m = shapes[si][0], k = shapes[si][1], n = shapes[si][2];
    size_t guard = 64u * (size_t)k;
    Buf w =
        gpu_alloc(gpu, slot, ((size_t)n * (size_t)k + guard) * sizeof(float));
    Buf x =
        gpu_alloc(gpu, slot, ((size_t)m * (size_t)k + guard) * sizeof(float));
    Buf b = gpu_alloc(gpu, slot, (size_t)n * sizeof(float));
    Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * sizeof(float));
    float *wp = (float *)w.ptr, *xp = (float *)x.ptr, *bp = (float *)b.ptr;
    for (int i = 0; i < n * k; i++) wp[i] = 0.5f * rng_float(&rng);
    for (int i = 0; i < m * k; i++) xp[i] = 0.5f * rng_float(&rng);
    for (size_t i = 0; i < guard; i++) {
      wp[(size_t)n * (size_t)k + i] = NAN;
      xp[(size_t)m * (size_t)k + i] = NAN;
    }
    for (int i = 0; i < n; i++) bp[i] = rng_float(&rng);

    F32MMParam p = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_mm);
    gpu_bind_buffer(&cmd, 0, &w, 0);
    gpu_bind_buffer(&cmd, 1, &x, 0);
    gpu_bind_buffer(&cmd, 2, &b, 0);
    gpu_bind_buffer(&cmd, 3, &y, 0);
    gpu_copy_bytes(&cmd, 4, &p, sizeof p);
    gpu_run_kernel(&cmd, ((uint32_t)n + F32MM_TILE_N - 1) / F32MM_TILE_N,
                   ((uint32_t)m + F32MM_TILE_M - 1) / F32MM_TILE_M, 1,
                   F32MM_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const float *yp = (const float *)y.ptr;
    for (int r = 0; r < m; r++) {
      for (int c = 0; c < n; c++) {
        float ref = bp[c];
        for (int j = 0; j < k; j++) ref += xp[r * k + j] * wp[c * k + j];
        float got = yp[r * n + c];
        if (!isfinite(got)) all_finite = 0;
        float d = fabsf(got - ref);
        if (d > worst) worst = d;
      }
    }

    gpu_free(&y);
    gpu_free(&b);
    gpu_free(&x);
    gpu_free(&w);
  }
  check(all_finite && worst < FP32_TOL,
        "minimax_h3_mm_f32 matches CPU reference across tiled and odd shapes");
}

// Tests minimax_h3_dit_adaln (F16 weight + BF16 and F16 bias, 2 timesteps).
static void test_adaln(Gpu *gpu, MemorySlot *slot) {
  int k_adaln = gpu_find_kernel(gpu, "minimax_h3_dit_adaln");
  int num_t = 2, out_dim = 384;
  uint32_t rng = 0xabcdef01u;

  Buf w =
      gpu_alloc(gpu, slot, (size_t)out_dim * DIT_ADALN_RANK * sizeof(_Float16));
  Buf b = gpu_alloc(gpu, slot, (size_t)out_dim * sizeof(uint16_t));
  Buf t_emb =
      gpu_alloc(gpu, slot, (size_t)num_t * DIT_ADALN_RANK * sizeof(float));
  Buf out =
      gpu_alloc(gpu, slot, (size_t)num_t * (size_t)out_dim * sizeof(float));

  _Float16 *wp = (_Float16 *)w.ptr;
  float *tp = (float *)t_emb.ptr;
  for (int i = 0; i < out_dim * DIT_ADALN_RANK; i++)
    wp[i] = (_Float16)rng_float(&rng);
  for (int i = 0; i < num_t * DIT_ADALN_RANK; i++) tp[i] = rng_float(&rng);

  for (uint32_t bias_f16 = 0; bias_f16 <= 1; bias_f16++) {
    uint16_t *bp = (uint16_t *)b.ptr;
    _Float16 *b16 = (_Float16 *)b.ptr;
    for (int i = 0; i < out_dim; i++) {
      if (bias_f16)
        b16[i] = (_Float16)rng_float(&rng);
      else
        bp[i] = f32_to_bf16(rng_float(&rng));
    }

    DitAdalnParam p = {(uint32_t)num_t, (uint32_t)out_dim, bias_f16};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_adaln);
    gpu_bind_buffer(&cmd, 0, &w, 0);
    gpu_bind_buffer(&cmd, 1, &b, 0);
    gpu_bind_buffer(&cmd, 2, &t_emb, 0);
    gpu_bind_buffer(&cmd, 3, &out, 0);
    gpu_copy_bytes(&cmd, 4, &p, sizeof p);
    gpu_run_kernel(&cmd, ((uint32_t)out_dim + ELEM_THREADS - 1) / ELEM_THREADS,
                   (uint32_t)num_t, 1, ELEM_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const float *op = (const float *)out.ptr;
    float worst = 0.0f;
    for (int t = 0; t < num_t; t++) {
      for (int j = 0; j < out_dim; j++) {
        float ref = bias_f16 ? (float)b16[j] : bf16_to_f32(bp[j]);
        for (int k = 0; k < DIT_ADALN_RANK; k++)
          ref += (float)wp[j * DIT_ADALN_RANK + k] * tp[t * DIT_ADALN_RANK + k];
        float d = fabsf(op[t * out_dim + j] - ref);
        if (d > worst) worst = d;
      }
    }
    check(worst < FP32_TOL,
          bias_f16 ? "minimax_h3_dit_adaln (F16 bias) matches CPU reference"
                   : "minimax_h3_dit_adaln (BF16 bias) matches CPU reference");
  }

  gpu_free(&out);
  gpu_free(&t_emb);
  gpu_free(&b);
  gpu_free(&w);
}

// Tests minimax_h3_dit_norm_mod and minimax_h3_dit_gate_add with per-token
// modality rows.
static void test_norm_mod_and_gate_add(Gpu *gpu, MemorySlot *slot) {
  int k_nm = gpu_find_kernel(gpu, "minimax_h3_dit_norm_mod");
  int k_ga = gpu_find_kernel(gpu, "minimax_h3_dit_gate_add");
  int rows = 13, dim = 128, num_mod_rows = 4, mod_stride = 6 * dim;
  uint32_t shift_off = 3u * (uint32_t)dim, scale_off = 4u * (uint32_t)dim,
           gate_off = 5u * (uint32_t)dim;
  uint32_t rng = 0x77112233u;

  Buf x = gpu_alloc(gpu, slot, (size_t)rows * (size_t)dim * sizeof(float));
  Buf w = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
  Buf mod = gpu_alloc(
      gpu, slot, (size_t)num_mod_rows * (size_t)mod_stride * sizeof(float));
  Buf mod_rows = gpu_alloc(gpu, slot, (size_t)rows * sizeof(uint32_t));
  Buf y = gpu_alloc(gpu, slot, (size_t)rows * (size_t)dim * sizeof(float));

  float *xp = (float *)x.ptr, *wp = (float *)w.ptr, *mp = (float *)mod.ptr;
  uint32_t *mrp = (uint32_t *)mod_rows.ptr;
  float *x_orig = xmalloc((size_t)rows * (size_t)dim * sizeof(float));
  for (int i = 0; i < rows * dim; i++) x_orig[i] = xp[i] = rng_float(&rng);
  for (int i = 0; i < dim; i++) wp[i] = 0.5f + rng_float(&rng);
  for (int i = 0; i < num_mod_rows * mod_stride; i++)
    mp[i] = 0.4f * rng_float(&rng);
  for (int r = 0; r < rows; r++) mrp[r] = (uint32_t)(r % num_mod_rows);

  DitNormModParam nmp = {(uint32_t)dim, (uint32_t)mod_stride, shift_off,
                         scale_off, NORM_EPS};
  DitGateAddParam gap = {(uint32_t)rows, (uint32_t)dim, (uint32_t)mod_stride,
                         gate_off};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_nm);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &w, 0);
  gpu_bind_buffer(&cmd, 2, &mod, 0);
  gpu_bind_buffer(&cmd, 3, &mod_rows, 0);
  gpu_bind_buffer(&cmd, 4, &y, 0);
  gpu_copy_bytes(&cmd, 5, &nmp, sizeof nmp);
  gpu_run_kernel(&cmd, (uint32_t)rows, 1, 1, NORM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_ga);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &y, 0);
  gpu_bind_buffer(&cmd, 2, &mod, 0);
  gpu_bind_buffer(&cmd, 3, &mod_rows, 0);
  gpu_copy_bytes(&cmd, 4, &gap, sizeof gap);
  gpu_run_kernel(&cmd, ((uint32_t)dim + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 (uint32_t)rows, 1, GRID2D_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  const float *yp = (const float *)y.ptr;
  float worst_nm = 0.0f, worst_ga = 0.0f;
  for (int r = 0; r < rows; r++) {
    float ss = 0.0f;
    for (int c = 0; c < dim; c++)
      ss += x_orig[r * dim + c] * x_orig[r * dim + c];
    float inv = 1.0f / sqrtf(ss / (float)dim + NORM_EPS);
    uint32_t mr = mrp[r];
    for (int c = 0; c < dim; c++) {
      float shift = mp[mr * (uint32_t)mod_stride + shift_off + (uint32_t)c];
      float scale = mp[mr * (uint32_t)mod_stride + scale_off + (uint32_t)c];
      float gate = mp[mr * (uint32_t)mod_stride + gate_off + (uint32_t)c];
      float ref_y =
          (wp[c] * x_orig[r * dim + c] * inv) * (1.0f + scale) + shift;
      float ref_x = x_orig[r * dim + c] + gate * ref_y;
      float dy = fabsf(yp[r * dim + c] - ref_y);
      float dx = fabsf(xp[r * dim + c] - ref_x);
      if (dy > worst_nm) worst_nm = dy;
      if (dx > worst_ga) worst_ga = dx;
    }
  }
  check(worst_nm < FP32_TOL, "minimax_h3_dit_norm_mod matches CPU reference");
  check(worst_ga < FP32_TOL, "minimax_h3_dit_gate_add matches CPU reference");

  free(x_orig);
  gpu_free(&y);
  gpu_free(&mod_rows);
  gpu_free(&mod);
  gpu_free(&w);
  gpu_free(&x);
}

// Tests fused minimax_h3_dit_norm_mod_convrot against unfused
// minimax_h3_dit_norm_mod + minimax_h3_convrot_quant_i8 across 1-pass and
// 2-pass widths.
static void test_norm_mod_convrot(Gpu *gpu, MemorySlot *slot) {
  int k_nm = gpu_find_kernel(gpu, "minimax_h3_dit_norm_mod");
  int k_rot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  int k_fused = gpu_find_kernel(gpu, "minimax_h3_dit_norm_mod_convrot");
  static const uint32_t dims[] = {512u, 16640u};
  uint32_t rows = 7u, num_mod_rows = 3u;
  uint32_t padded = (rows + TILE_M_I8 - 1u) / TILE_M_I8 * TILE_M_I8;
  uint32_t rng = 0x42427788u;
  int ok = 1;

  for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
    uint32_t dim = dims[di], mod_stride = 6u * dim;
    uint32_t shift_off = 3u * dim, scale_off = 4u * dim;
    Buf x = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
    Buf w = gpu_alloc(gpu, slot, (size_t)dim * sizeof(float));
    Buf mod =
        gpu_alloc(gpu, slot, (size_t)num_mod_rows * mod_stride * sizeof(float));
    Buf mod_rows = gpu_alloc(gpu, slot, (size_t)rows * sizeof(uint32_t));
    Buf y = gpu_alloc(gpu, slot, (size_t)rows * dim * sizeof(float));
    Buf xq_ref = gpu_alloc(gpu, slot, (size_t)padded * dim);
    Buf xs_ref = gpu_alloc(gpu, slot, (size_t)rows * sizeof(float));
    Buf xq_got = gpu_alloc(gpu, slot, (size_t)padded * dim);
    Buf xs_got = gpu_alloc(gpu, slot, (size_t)rows * sizeof(float));

    float *xp = (float *)x.ptr, *wp = (float *)w.ptr, *mp = (float *)mod.ptr;
    uint32_t *mrp = (uint32_t *)mod_rows.ptr;
    for (size_t i = 0; i < (size_t)rows * dim; i++) xp[i] = rng_float(&rng);
    for (uint32_t i = 0; i < dim; i++) wp[i] = 0.5f + rng_float(&rng);
    for (size_t i = 0; i < (size_t)num_mod_rows * mod_stride; i++)
      mp[i] = 0.4f * rng_float(&rng);
    for (uint32_t r = 0; r < rows; r++) mrp[r] = r % num_mod_rows;
    memset(xq_ref.ptr, 0x5a, xq_ref.size);
    memset(xq_got.ptr, 0x5a, xq_got.size);

    DitNormModParam nmp = {dim, mod_stride, shift_off, scale_off, NORM_EPS};
    ConvRotParam rp = {rows, dim, dim};
    DitNormModConvRotParam fp = {rows,      dim,       mod_stride,
                                 shift_off, scale_off, NORM_EPS};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_nm);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &w, 0);
    gpu_bind_buffer(&cmd, 2, &mod, 0);
    gpu_bind_buffer(&cmd, 3, &mod_rows, 0);
    gpu_bind_buffer(&cmd, 4, &y, 0);
    gpu_copy_bytes(&cmd, 5, &nmp, sizeof nmp);
    gpu_run_kernel(&cmd, rows, 1, 1, NORM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_rot);
    gpu_bind_buffer(&cmd, 0, &y, 0);
    gpu_bind_buffer(&cmd, 1, &xq_ref, 0);
    gpu_bind_buffer(&cmd, 2, &xs_ref, 0);
    gpu_copy_bytes(&cmd, 3, &rp, sizeof rp);
    gpu_run_kernel(&cmd, padded, 1, 1, CONVROT_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_fused);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &w, 0);
    gpu_bind_buffer(&cmd, 2, &mod, 0);
    gpu_bind_buffer(&cmd, 3, &mod_rows, 0);
    gpu_bind_buffer(&cmd, 4, &xq_got, 0);
    gpu_bind_buffer(&cmd, 5, &xs_got, 0);
    gpu_copy_bytes(&cmd, 6, &fp, sizeof fp);
    gpu_run_kernel(&cmd, padded, 1, 1, CONVROT_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const int8_t *qr = (const int8_t *)xq_ref.ptr;
    const int8_t *qg = (const int8_t *)xq_got.ptr;
    const float *sr = (const float *)xs_ref.ptr;
    const float *sg = (const float *)xs_got.ptr;
    for (uint32_t r = 0; r < rows; r++) {
      if (fabsf(sr[r] - sg[r]) > fmaxf(1e-7f, sr[r] * 1e-5f)) ok = 0;
      for (uint32_t c = 0; c < dim; c++)
        if (abs((int)qr[(size_t)r * dim + c] - (int)qg[(size_t)r * dim + c]) >
            1)
          ok = 0;
    }
    for (uint32_t r = rows; r < padded; r++)
      for (uint32_t c = 0; c < dim; c++)
        if (qg[(size_t)r * dim + c] != 0) ok = 0;

    gpu_free(&xs_got);
    gpu_free(&xq_got);
    gpu_free(&xs_ref);
    gpu_free(&xq_ref);
    gpu_free(&y);
    gpu_free(&mod_rows);
    gpu_free(&mod);
    gpu_free(&w);
    gpu_free(&x);
  }
  check(ok,
        "minimax_h3_dit_norm_mod_convrot matches unfused norm_mod + ConvRot "
        "across 1-pass and 2-pass widths");
}

// Tests minimax_h3_dit_heads for both TokenRefiner (no RoPE) and DiT (partial
// 3D split-half RoPE on first 96 of 128 head dims).
static void test_heads(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "minimax_h3_dit_heads");
  int rows = 9;
  uint32_t padded = TILE_M;
  uint32_t rng = 0x33445566u;

  Buf qkv = gpu_alloc(gpu, slot,
                      (size_t)rows * (size_t)DIT_QKV_WIDTH * sizeof(float));
  Buf qn = gpu_alloc(gpu, slot, (size_t)DIT_HEAD_DIM * sizeof(float));
  Buf kn = gpu_alloc(gpu, slot, (size_t)DIT_HEAD_DIM * sizeof(float));
  Buf pos = gpu_alloc(gpu, slot, (size_t)rows * 3 * sizeof(float));
  Buf inv = gpu_alloc(gpu, slot, (size_t)DIT_ROPE_FREQS * sizeof(float));
  Buf qh = gpu_alloc(
      gpu, slot, (size_t)DIT_HEADS * padded * DIT_HEAD_DIM * sizeof(uint16_t));
  Buf kh = gpu_alloc(
      gpu, slot, (size_t)DIT_HEADS * padded * DIT_HEAD_DIM * sizeof(uint16_t));
  Buf vh = gpu_alloc(
      gpu, slot, (size_t)DIT_HEADS * padded * DIT_HEAD_DIM * sizeof(uint16_t));

  float *qkvp = (float *)qkv.ptr, *qnp = (float *)qn.ptr,
        *knp = (float *)kn.ptr;
  float *posp = (float *)pos.ptr, *invp = (float *)inv.ptr;
  for (int i = 0; i < rows * DIT_QKV_WIDTH; i++) qkvp[i] = rng_float(&rng);
  for (int i = 0; i < DIT_HEAD_DIM; i++) {
    qnp[i] = 0.5f + 0.5f * rng_float(&rng);
    knp[i] = 0.5f + 0.5f * rng_float(&rng);
  }
  for (int i = 0; i < rows * 3; i++) posp[i] = 4.0f * rng_float(&rng);
  for (int i = 0; i < DIT_ROPE_FREQS; i++)
    invp[i] = powf(10000.0f, -(float)i / (float)DIT_ROPE_FREQS);

  for (uint32_t use_rope = 0; use_rope <= 1; use_rope++) {
    DitHeadsParam p = {(uint32_t)rows, padded, use_rope, NORM_EPS};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_heads);
    gpu_bind_buffer(&cmd, 0, &qkv, 0);
    gpu_bind_buffer(&cmd, 1, &qn, 0);
    gpu_bind_buffer(&cmd, 2, &kn, 0);
    gpu_bind_buffer(&cmd, 3, &pos, 0);
    gpu_bind_buffer(&cmd, 4, &inv, 0);
    gpu_bind_buffer(&cmd, 5, &qh, 0);
    gpu_bind_buffer(&cmd, 6, &kh, 0);
    gpu_bind_buffer(&cmd, 7, &vh, 0);
    gpu_copy_bytes(&cmd, 8, &p, sizeof p);
    gpu_run_kernel(&cmd, (uint32_t)(3 * DIT_HEADS), padded, 1, HEADS_THREADS, 1,
                   1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const uint16_t *qhp = (const uint16_t *)qh.ptr;
    const uint16_t *khp = (const uint16_t *)kh.ptr;
    const uint16_t *vhp = (const uint16_t *)vh.ptr;
    float worst = 0.0f;
    int pad_zero_ok = 1;
    for (int h = 0; h < DIT_HEADS; h++) {
      for (int r = 0; r < (int)padded; r++) {
        size_t base = ((size_t)h * padded + (size_t)r) * DIT_HEAD_DIM;
        if (r >= rows) {
          for (int d = 0; d < DIT_HEAD_DIM; d++)
            pad_zero_ok &= (qhp[base + d] == 0 && khp[base + d] == 0 &&
                            vhp[base + d] == 0);
          continue;
        }
        // Check V pass-through.
        const float *vs = qkvp + (size_t)r * DIT_QKV_WIDTH + 2u * DIT_INNER +
                          h * DIT_HEAD_DIM;
        for (int d = 0; d < DIT_HEAD_DIM; d++) {
          float diff = fabsf(bf16_to_f32(vhp[base + d]) - vs[d]);
          if (diff > worst) worst = diff;
        }
        // Check Q and K: RMSNorm + optional partial 3D split-half RoPE.
        for (int kind = 0; kind < 2; kind++) {
          const float *src = qkvp + (size_t)r * DIT_QKV_WIDTH +
                             (size_t)kind * DIT_INNER + h * DIT_HEAD_DIM;
          const float *nw = kind == 0 ? qnp : knp;
          const uint16_t *got = (kind == 0 ? qhp : khp) + base;
          float ss = 0.0f;
          for (int d = 0; d < DIT_HEAD_DIM; d++) ss += src[d] * src[d];
          float inv_rms = 1.0f / sqrtf(ss / (float)DIT_HEAD_DIM + NORM_EPS);
          float hnorm[DIT_HEAD_DIM];
          for (int d = 0; d < DIT_HEAD_DIM; d++)
            hnorm[d] = nw[d] * src[d] * inv_rms;
          float ref[DIT_HEAD_DIM];
          if (!use_rope) {
            memcpy(ref, hnorm, sizeof ref);
          } else {
            for (int i = 0; i < DIT_ROT_PAIRS; i++) {
              int axis = i / DIT_ROPE_FREQS, fi = i % DIT_ROPE_FREQS;
              float ang = posp[r * 3 + axis] * invp[fi];
              float c = cosf(ang), s = sinf(ang);
              ref[i] = hnorm[i] * c - hnorm[i + DIT_ROT_PAIRS] * s;
              ref[i + DIT_ROT_PAIRS] =
                  hnorm[i + DIT_ROT_PAIRS] * c + hnorm[i] * s;
            }
            for (int d = 2 * DIT_ROT_PAIRS; d < DIT_HEAD_DIM; d++)
              ref[d] = hnorm[d];
          }
          for (int d = 0; d < DIT_HEAD_DIM; d++) {
            float diff = fabsf(bf16_to_f32(got[d]) - ref[d]);
            if (diff > worst) worst = diff;
          }
        }
      }
    }
    check(pad_zero_ok && worst < BF16_TOL,
          use_rope
              ? "minimax_h3_dit_heads partial 3D split-half RoPE matches CPU"
              : "minimax_h3_dit_heads without RoPE matches CPU");
  }

  gpu_free(&vh);
  gpu_free(&kh);
  gpu_free(&qh);
  gpu_free(&inv);
  gpu_free(&pos);
  gpu_free(&kn);
  gpu_free(&qn);
  gpu_free(&qkv);
}

// Tests minimax_h3_dit_attend bidirectional 56-head FlashAttention across
// interior, boundary and tail 64-key tiles plus multi-tile slicing.
static void test_attend(Gpu *gpu, MemorySlot *slot) {
  int k_attend = gpu_find_kernel(gpu, "minimax_h3_dit_attend");
  int rows = 137;
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t total_tiles = padded / TILE_M;
  uint32_t rng = 0x99887766u;

  size_t head_elems = (size_t)DIT_HEADS * padded * DIT_HEAD_DIM;
  Buf qh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf kh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf vh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf out =
      gpu_alloc(gpu, slot, (size_t)rows * (size_t)DIT_INNER * sizeof(float));

  uint16_t *qp = (uint16_t *)qh.ptr;
  uint16_t *kp = (uint16_t *)kh.ptr;
  uint16_t *vp = (uint16_t *)vh.ptr;
  memset(qp, 0, head_elems * sizeof(uint16_t));
  memset(kp, 0, head_elems * sizeof(uint16_t));
  memset(vp, 0, head_elems * sizeof(uint16_t));
  for (int h = 0; h < DIT_HEADS; h++) {
    for (int r = 0; r < rows; r++) {
      size_t base = ((size_t)h * padded + (size_t)r) * DIT_HEAD_DIM;
      for (int d = 0; d < DIT_HEAD_DIM; d++) {
        qp[base + d] = f32_to_bf16(0.5f * rng_float(&rng));
        kp[base + d] = f32_to_bf16(0.5f * rng_float(&rng));
        vp[base + d] = f32_to_bf16(0.5f * rng_float(&rng));
      }
    }
  }

  DitAttendParam p = {(uint32_t)rows, padded, 0u, total_tiles};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &qh, 0);
  gpu_bind_buffer(&cmd, 1, &kh, 0);
  gpu_bind_buffer(&cmd, 2, &vh, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &p, sizeof p);
  gpu_run_kernel(&cmd, total_tiles * (uint32_t)DIT_HEADS, 1, 1, MM_THREADS, 1,
                 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Verify 1-tile and 2-tile sliced dispatches match the full dispatch
  // bit-for-bit across tile boundaries.
  Buf out_sliced =
      gpu_alloc(gpu, slot, (size_t)rows * (size_t)DIT_INNER * sizeof(float));
  for (uint32_t step = 1; step <= 2; step++) {
    memset(out_sliced.ptr, 0, (size_t)rows * (size_t)DIT_INNER * sizeof(float));
    for (uint32_t t = 0; t < total_tiles; t += step) {
      uint32_t st = t + step <= total_tiles ? step : total_tiles - t;
      DitAttendParam sp = {(uint32_t)rows, padded, t, st};
      GpuCommandBuffer scmd = gpu_begin(gpu);
      gpu_select_kernel(&scmd, k_attend);
      gpu_bind_buffer(&scmd, 0, &qh, 0);
      gpu_bind_buffer(&scmd, 1, &kh, 0);
      gpu_bind_buffer(&scmd, 2, &vh, 0);
      gpu_bind_buffer(&scmd, 3, &out_sliced, 0);
      gpu_copy_bytes(&scmd, 4, &sp, sizeof sp);
      gpu_run_kernel(&scmd, st * (uint32_t)DIT_HEADS, 1, 1, MM_THREADS, 1, 1);
      gpu_commit(&scmd);
      gpu_sync(gpu);
    }
    check(memcmp(out.ptr, out_sliced.ptr,
                 (size_t)rows * (size_t)DIT_INNER * sizeof(float)) == 0,
          step == 1
              ? "minimax_h3_dit_attend 1-tile slicing matches full dispatch "
                "bit-for-bit"
              : "minimax_h3_dit_attend 2-tile slicing matches full dispatch "
                "bit-for-bit");
  }
  gpu_free(&out_sliced);

  const float *op = (const float *)out.ptr;
  float scale = 1.0f / sqrtf((float)DIT_HEAD_DIM);
  float worst = 0.0f;
  // Spot-check first, middle and last heads across all query rows.
  static const int check_heads[] = {0, 17, 55};
  for (size_t hi = 0; hi < sizeof check_heads / sizeof check_heads[0]; hi++) {
    int h = check_heads[hi];
    for (int q = 0; q < rows; q++) {
      float scores[137];
      float mx = -INFINITY;
      size_t qbase = ((size_t)h * padded + (size_t)q) * DIT_HEAD_DIM;
      for (int k = 0; k < rows; k++) {
        size_t kbase = ((size_t)h * padded + (size_t)k) * DIT_HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < DIT_HEAD_DIM; d++)
          dot += bf16_to_f32(qp[qbase + d]) * bf16_to_f32(kp[kbase + d]);
        scores[k] = dot * scale;
        if (scores[k] > mx) mx = scores[k];
      }
      float sum = 0.0f;
      for (int k = 0; k < rows; k++) {
        scores[k] = expf(scores[k] - mx);
        sum += scores[k];
      }
      for (int d = 0; d < DIT_HEAD_DIM; d++) {
        float acc = 0.0f;
        for (int k = 0; k < rows; k++) {
          size_t vbase = ((size_t)h * padded + (size_t)k) * DIT_HEAD_DIM;
          acc += scores[k] * bf16_to_f32(vp[vbase + d]);
        }
        float ref = acc / sum;
        float got = op[(size_t)q * DIT_INNER + h * DIT_HEAD_DIM + d];
        float diff = fabsf(got - ref);
        if (diff > worst) worst = diff;
      }
    }
  }
  check(
      worst < 2e-2f,
      "minimax_h3_dit_attend bidirectional 56-head FlashAttention matches CPU");

  gpu_free(&out);
  gpu_free(&vh);
  gpu_free(&kh);
  gpu_free(&qh);
}

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_mm_bf16(gpu, slot);
  test_mm_f32(gpu, slot);
  test_adaln(gpu, slot);
  test_norm_mod_and_gate_add(gpu, slot);
  test_norm_mod_convrot(gpu, slot);
  test_heads(gpu, slot);
  test_attend(gpu, slot);
  return failures != 0;
}
