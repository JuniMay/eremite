#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/nemotron/model.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Formats a check description and records the result.
static void checkf(int ok, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void checkf(int ok, const char *fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  check(ok, buf);
}

// Verifies every Metal kernel in src/kernels/nemotron.metal against independent
// CPU references, including 8-byte-aligned (non-16-byte-aligned) weight views.

static constexpr float RMS_EPS = 1e-5f;
static constexpr float FP8_UNDO = 256.0f;
static constexpr float ROUTE_SCALE = 2.5f;
static constexpr float ROUTE_EPS = 1e-20f;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_MASK = 0xFFFF0000u;
static constexpr uint32_t ACT_NONE = 0, ACT_RELU2 = 1;
static constexpr uint32_t ROW_THREADS = 256, FP8_MV_THREADS = 256,
                          NV_MV_THREADS = 128, MM_THREADS = 128,
                          CONV_THREADS = 256, SSM_THREADS = 128,
                          QKV_THREADS = 128, PREFILL_THREADS = 128,
                          DECODE_PART_THREADS = 512,
                          DECODE_COMBINE_THREADS = 32, ROUTE_THREADS = 128,
                          MOE_MV_THREADS = 256, MOE_COMBINE_THREADS = 64,
                          PACK_THREADS = 64;
static constexpr uint32_t TILE_M2 = 128, TILE_N = 64, MOE_MV_MAX_M = 8,
                          MOE_MM_TILE = 64;
static constexpr size_t OFF8 = 8;
static constexpr float TOL_F32 = 1e-4f, TOL_BF16 = 3e-2f, TOL_REL_BF16 = 2e-2f,
                       TOL_ATTN = 2e-2f, TOL_SSM = 1e-2f;

// Kernel parameter structs matching src/kernels/nemotron.metal.
typedef struct {
  // Row width and BF16 rounding flag.
  uint32_t dim, round;
  // RMSNorm epsilon.
  float eps;
} NormParam;

typedef struct {
  // Active rows, output width, inner width, residual and BF16 round flags.
  uint32_t m, n, k, has_res, round;
  // Per-tensor weight scale multiplied by 256.
  float scale;
} Fp8LinParam;

typedef struct {
  // Active rows, output width, inner width, activation, residual and round.
  uint32_t m, n, k, act, has_res, round;
  // Per-tensor weight_scale_2 scalar.
  float scale2;
} Nvfp4LinParam;

typedef struct {
  // Active rows.
  uint32_t rows;
  // RMSNorm epsilon.
  float eps;
} MambaParam;

typedef struct {
  // Device addresses of one state's conv and ssm buffers.
  uint64_t conv, ssm;
} MambaTarget;

typedef struct {
  // Device addresses of one state's per-KV-head BF16 K and V buffers.
  uint64_t k0, k1, v0, v1;
  // Token position and alignment padding.
  uint32_t pos, pad;
} KvTarget;

typedef struct {
  // Active rows, Q stride, row-0 pos and batched flag.
  uint32_t rows, q_stride, pos0, batched;
} QkvParam;

typedef struct {
  // Active rows, Q stride, row-0 pos and KV capacity.
  uint32_t rows, q_stride, pos0, kv_cap;
} AttnPrefillParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, partitions, stride and batched flag.
  uint32_t rows, q_stride, pos0, nparts, part_stride, batched;
} AttnDecodeParam;

typedef struct {
  // Active rows, total experts and top-k count.
  uint32_t rows, experts, topk;
  // Routed scaling factor.
  float scale;
} RouteParam;

typedef struct {
  // Device addresses for one expert's packed weights and scales.
  uint64_t w_up, s_up, w_down, s_down;
} ExpertRef;

typedef struct {
  // Output rows, inner width, active pairs and tile count.
  uint32_t n, k, npairs, ntiles;
} MoeParam;

typedef struct {
  // Active rows, hidden width and top-k count.
  uint32_t rows, dim, topk;
} MoeCombineParam;

// Deterministic xorshift64* PRNG state and helpers.
static uint64_t rng_state = 0x243F6A8885A308D3ull;

static uint32_t randu(void) {
  uint64_t x = rng_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  rng_state = x;
  return (uint32_t)((x * 2685821657736338717ull) >> 32);
}

static float randf(void) {
  return ((float)(randu() & 0xFFFFu) / 32768.0f) - 1.0f;
}

static uint16_t f32_to_bf16(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  return (uint16_t)(u >> BF16_SHIFT);
}

static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << BF16_SHIFT;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

static float round_bf(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// Decodes one E4M3FN byte to FP32.
static float e4m3_to_f32(uint8_t code) {
  uint32_t sign = (code >> 7) & 1u;
  uint32_t exp = (code >> 3) & 15u;
  uint32_t mant = code & 7u;
  float val;
  if (exp == 0) {
    val = ldexpf((float)mant, -9);
  } else if (exp == 15 && mant == 7) {
    return NAN;
  } else {
    val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
  }
  return sign ? -val : val;
}

// Decodes one NVFP4 E2M1 nibble (0..15) to FP32.
static float e2m1_to_f32(uint8_t nib) {
  static constexpr float LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  float v = LUT[nib & 7u];
  return (nib & 8u) ? -v : v;
}

// Returns a random valid non-NaN E4M3 code (positive when pos_only is nonzero).
static uint8_t rand_e4m3(int pos_only) {
  uint8_t c;
  do {
    c = (uint8_t)(randu() & (pos_only ? 0x7Fu : 0xFFu));
  } while ((c & 0x7Fu) == 0x7Fu || (pos_only && (c & 0x7Fu) < 0x10u));
  return c;
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
  float mx = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
    float d = fabsf(a[i] - b[i]);
    if (d > mx) mx = d;
  }
  return mx;
}

static float max_rel_diff(const float *a, const float *b, size_t n) {
  float mx = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
    float d = fabsf(a[i] - b[i]) / fmaxf(1.0f, fabsf(b[i]));
    if (d > mx) mx = d;
  }
  return mx;
}

static float siluf(float x) { return x / (1.0f + expf(-x)); }

static float softplusf(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

static float relu2f(float x) { return x > 0.0f ? x * x : 0.0f; }

// Allocates a resident GPU buffer in work.
static Buf make_buf(Gpu *g, MemorySlot *work, size_t bytes) {
  Buf b = gpu_alloc(g, work, bytes);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(g);
  return b;
}

// Tests nemo_embed (with negative media skip) and nemo_rmsnorm.
static void test_embed_and_norm(Gpu *g, MemorySlot *work) {
  static constexpr int VOCAB = 16, DIM = 256, ROWS = 5;
  Buf table = make_buf(g, work, (size_t)VOCAB * DIM * sizeof(uint16_t));
  Buf ids = make_buf(g, work, (size_t)ROWS * sizeof(int32_t));
  Buf cur = make_buf(g, work, (size_t)ROWS * DIM * sizeof(float));
  Buf nw = make_buf(g, work, (size_t)DIM * sizeof(float));
  Buf out = make_buf(g, work, (size_t)ROWS * DIM * sizeof(float));

  uint16_t *tp = (uint16_t *)table.ptr;
  for (int i = 0; i < VOCAB * DIM; i++) tp[i] = f32_to_bf16(randf());
  int32_t id_vals[ROWS] = {3, -7, 12, -1, 0};
  memcpy(ids.ptr, id_vals, sizeof id_vals);
  float *cp = (float *)cur.ptr;
  for (int i = 0; i < ROWS * DIM; i++) cp[i] = 42.5f;
  float *wp = (float *)nw.ptr;
  for (int i = 0; i < DIM; i++) wp[i] = 0.8f + 0.4f * randf();

  float ref_emb[ROWS * DIM];
  for (int r = 0; r < ROWS; r++) {
    for (int d = 0; d < DIM; d++) {
      ref_emb[r * DIM + d] =
          id_vals[r] < 0 ? 42.5f : bf16_to_f32(tp[id_vals[r] * DIM + d]);
    }
  }

  NormParam p = {DIM, 1u, RMS_EPS};
  GpuCommandBuffer cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_embed"));
  gpu_bind_buffer(&cmd, 0, &table, 0);
  gpu_bind_buffer(&cmd, 1, &ids, 0);
  gpu_bind_buffer(&cmd, 2, &cur, 0);
  gpu_copy_bytes(&cmd, 3, &p, sizeof p);
  gpu_run_kernel(&cmd, ROWS, 1, 1, ROW_THREADS, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_rmsnorm"));
  gpu_bind_buffer(&cmd, 0, &cur, 0);
  gpu_bind_buffer(&cmd, 1, &nw, 0);
  gpu_bind_buffer(&cmd, 2, &out, 0);
  gpu_copy_bytes(&cmd, 3, &p, sizeof p);
  gpu_run_kernel(&cmd, ROWS, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  checkf(max_abs_diff((const float *)cur.ptr, ref_emb, ROWS * DIM) == 0.0f,
         "nemo_embed skips negative media rows and matches CPU");

  float ref_norm[ROWS * DIM];
  for (int r = 0; r < ROWS; r++) {
    double sq = 0.0;
    for (int d = 0; d < DIM; d++) {
      double v = ref_emb[r * DIM + d];
      sq += v * v;
    }
    float inv = 1.0f / sqrtf((float)(sq / DIM) + RMS_EPS);
    for (int d = 0; d < DIM; d++)
      ref_norm[r * DIM + d] = round_bf(wp[d] * (ref_emb[r * DIM + d] * inv));
  }
  float diff = max_abs_diff((const float *)out.ptr, ref_norm, ROWS * DIM);
  checkf(diff <= TOL_F32, "nemo_rmsnorm matches CPU (max=%.2e)", diff);

  gpu_free(&table), gpu_free(&ids), gpu_free(&cur), gpu_free(&nw),
      gpu_free(&out);
}

// Tests nemo_fp8_mv (m=1,2,4,7) and nemo_fp8_mm (m=19,137) with 8-byte offset
// weights and optional residual addition.
static void test_fp8_linear(Gpu *g, MemorySlot *work) {
  static constexpr int N = 192, K = 256, MAX_M = 137;
  static constexpr int MS[] = {1, 2, 4, 7, 19, 137};
  static constexpr float W_SCALE = 0.0025f;

  Buf w = make_buf(g, work, OFF8 + (size_t)N * K);
  Buf x = make_buf(g, work, (size_t)MAX_M * K * sizeof(float));
  Buf res = make_buf(g, work, (size_t)MAX_M * N * sizeof(float));
  Buf y = make_buf(g, work, (size_t)MAX_M * N * sizeof(float));
  Buf x_pack = make_buf(g, work, (size_t)256 * K * sizeof(uint16_t));

  uint8_t *wp = w.ptr + OFF8;
  for (int i = 0; i < N * K; i++) wp[i] = rand_e4m3(0);
  float *xp = (float *)x.ptr;
  for (int i = 0; i < MAX_M * K; i++) xp[i] = round_bf(randf());
  float *rp = (float *)res.ptr;
  for (int i = 0; i < MAX_M * N; i++) rp[i] = round_bf(randf() * 0.5f);

  float *ref = xmalloc((size_t)MAX_M * N * sizeof(float));
  int k_mv = gpu_find_kernel(g, "nemo_fp8_mv");
  int k_mm = gpu_find_kernel(g, "nemo_fp8_mm");
  int k_pack = gpu_find_kernel(g, "nemo_pack_bf16");

  for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
    int m = MS[mi];
    int has_res = (mi & 1u) ? 1 : 0;
    for (int r = 0; r < m; r++) {
      for (int o = 0; o < N; o++) {
        double acc = 0.0;
        for (int k = 0; k < K; k++)
          acc += (double)xp[r * K + k] * (double)e4m3_to_f32(wp[o * K + k]);
        float v = round_bf((float)(acc * W_SCALE));
        if (has_res) v = round_bf(v + rp[r * N + o]);
        ref[r * N + o] = v;
      }
    }

    Fp8LinParam p = {(uint32_t)m,       N,  K,
                     (uint32_t)has_res, 1u, W_SCALE * FP8_UNDO};
    GpuCommandBuffer cmd = gpu_begin(g);
    if (m <= (int)MOE_MV_MAX_M) {
      gpu_select_kernel(&cmd, k_mv);
      gpu_bind_buffer(&cmd, 0, &w, OFF8);
      gpu_bind_buffer(&cmd, 1, &x, 0);
      gpu_bind_buffer(&cmd, 2, &res, 0);
      gpu_bind_buffer(&cmd, 3, &y, 0);
      gpu_copy_bytes(&cmd, 4, &p, sizeof p);
      gpu_run_kernel(&cmd, (N + 31u) / 32u, 1, 1, FP8_MV_THREADS, 1, 1);
    } else {
      uint32_t padded = ((uint32_t)m + TILE_M2 - 1u) / TILE_M2 * TILE_M2;
      uint32_t rc[3] = {(uint32_t)m, padded, K / 4u};
      gpu_select_kernel(&cmd, k_pack);
      gpu_bind_buffer(&cmd, 0, &x, 0);
      gpu_bind_buffer(&cmd, 1, &x_pack, 0);
      gpu_copy_bytes(&cmd, 2, rc, sizeof rc);
      gpu_run_kernel(&cmd, (K / 4u + PACK_THREADS - 1u) / PACK_THREADS, padded,
                     1, PACK_THREADS, 1, 1);
      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w, OFF8);
      gpu_bind_buffer(&cmd, 1, &x_pack, 0);
      gpu_bind_buffer(&cmd, 2, &res, 0);
      gpu_bind_buffer(&cmd, 3, &y, 0);
      gpu_copy_bytes(&cmd, 4, &p, sizeof p);
      uint32_t tm = ((uint32_t)m + TILE_M2 - 1u) / TILE_M2;
      uint32_t tn = (N + TILE_N - 1u) / TILE_N;
      gpu_run_kernel(&cmd, tm * tn, 1, 1, MM_THREADS, 1, 1);
    }
    gpu_commit(&cmd);
    gpu_sync(g);

    float diff = max_abs_diff((const float *)y.ptr, ref, (size_t)m * N);
    checkf(diff <= TOL_BF16,
           "nemo_fp8_%s m=%d res=%d (off=8) matches CPU (max=%.2e)",
           m <= (int)MOE_MV_MAX_M ? "mv" : "mm", m, has_res, diff);
  }
  free(ref);
  gpu_free(&w), gpu_free(&x), gpu_free(&res), gpu_free(&y), gpu_free(&x_pack);
}

// Tests nemo_nvfp4_mv and nemo_nvfp4_mm with 8-byte offset weights, relu2 and
// residual add.
static void test_nvfp4_linear(Gpu *g, MemorySlot *work) {
  static constexpr int N = 192, K = 256, MAX_M = 137;
  static constexpr int MS[] = {1, 2, 4, 7, 19, 137};
  static constexpr float SCALE2 = 0.015f;

  int blocks = K / NEMO_NVFP4_BLOCK;
  Buf w = make_buf(g, work, OFF8 + (size_t)N * (K / 2));
  Buf sc = make_buf(g, work, OFF8 + (size_t)N * blocks);
  Buf x = make_buf(g, work, (size_t)MAX_M * K * sizeof(float));
  Buf res = make_buf(g, work, (size_t)MAX_M * N * sizeof(float));
  Buf y = make_buf(g, work, (size_t)MAX_M * N * sizeof(float));
  Buf x_pack = make_buf(g, work, (size_t)256 * K * sizeof(uint16_t));

  uint8_t *wp = w.ptr + OFF8, *sp = sc.ptr + OFF8;
  for (int i = 0; i < N * (K / 2); i++) wp[i] = (uint8_t)(randu() & 0xFFu);
  for (int i = 0; i < N * blocks; i++) sp[i] = rand_e4m3(1);
  float *xp = (float *)x.ptr;
  for (int i = 0; i < MAX_M * K; i++) xp[i] = round_bf(randf() * 0.2f);
  float *rp = (float *)res.ptr;
  for (int i = 0; i < MAX_M * N; i++) rp[i] = round_bf(randf() * 0.5f);

  float *ref = xmalloc((size_t)MAX_M * N * sizeof(float));
  int k_mv = gpu_find_kernel(g, "nemo_nvfp4_mv");
  int k_mm = gpu_find_kernel(g, "nemo_nvfp4_mm");
  int k_pack = gpu_find_kernel(g, "nemo_pack_bf16");

  for (size_t mi = 0; mi < sizeof MS / sizeof *MS; mi++) {
    int m = MS[mi];
    uint32_t act = (mi & 1u) ? ACT_RELU2 : ACT_NONE;
    uint32_t has_res = (mi % 3u == 0u) ? 1u : 0u;
    for (int r = 0; r < m; r++) {
      for (int o = 0; o < N; o++) {
        double acc = 0.0;
        for (int b = 0; b < blocks; b++) {
          float s = e4m3_to_f32(sp[o * blocks + b]);
          double bsum = 0.0;
          for (int j = 0; j < NEMO_NVFP4_BLOCK; j++) {
            int col = b * NEMO_NVFP4_BLOCK + j;
            uint8_t byte = wp[o * (K / 2) + col / 2];
            uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 15u);
            bsum += (double)e2m1_to_f32(nib) * (double)xp[r * K + col];
          }
          acc += bsum * (double)s;
        }
        float v = round_bf((float)(acc * SCALE2));
        if (act == ACT_RELU2) v = round_bf(relu2f(v));
        if (has_res) v = round_bf(v + rp[r * N + o]);
        ref[r * N + o] = v;
      }
    }

    Nvfp4LinParam p = {(uint32_t)m, N, K, act, has_res, 1u, SCALE2};
    GpuCommandBuffer cmd = gpu_begin(g);
    if (m <= (int)MOE_MV_MAX_M) {
      gpu_select_kernel(&cmd, k_mv);
      gpu_bind_buffer(&cmd, 0, &w, OFF8);
      gpu_bind_buffer(&cmd, 1, &sc, OFF8);
      gpu_bind_buffer(&cmd, 2, &x, 0);
      gpu_bind_buffer(&cmd, 3, &res, 0);
      gpu_bind_buffer(&cmd, 4, &y, 0);
      gpu_copy_bytes(&cmd, 5, &p, sizeof p);
      gpu_run_kernel(&cmd, (N + 15u) / 16u, 1, 1, NV_MV_THREADS, 1, 1);
    } else {
      uint32_t padded = ((uint32_t)m + TILE_M2 - 1u) / TILE_M2 * TILE_M2;
      uint32_t rc[3] = {(uint32_t)m, padded, K / 4u};
      gpu_select_kernel(&cmd, k_pack);
      gpu_bind_buffer(&cmd, 0, &x, 0);
      gpu_bind_buffer(&cmd, 1, &x_pack, 0);
      gpu_copy_bytes(&cmd, 2, rc, sizeof rc);
      gpu_run_kernel(&cmd, (K / 4u + PACK_THREADS - 1u) / PACK_THREADS, padded,
                     1, PACK_THREADS, 1, 1);
      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w, OFF8);
      gpu_bind_buffer(&cmd, 1, &sc, OFF8);
      gpu_bind_buffer(&cmd, 2, &x_pack, 0);
      gpu_bind_buffer(&cmd, 3, &res, 0);
      gpu_bind_buffer(&cmd, 4, &y, 0);
      gpu_copy_bytes(&cmd, 5, &p, sizeof p);
      uint32_t tm = ((uint32_t)m + TILE_M2 - 1u) / TILE_M2;
      uint32_t tn = (N + TILE_N - 1u) / TILE_N;
      gpu_run_kernel(&cmd, tm * tn, 1, 1, MM_THREADS, 1, 1);
    }
    gpu_commit(&cmd);
    gpu_sync(g);

    float diff = max_rel_diff((const float *)y.ptr, ref, (size_t)m * N);
    checkf(diff <= TOL_REL_BF16,
           "nemo_nvfp4_%s m=%d act=%u res=%u (off=8) matches CPU (rel=%.2e)",
           m <= (int)MOE_MV_MAX_M ? "mv" : "mm", m, act, has_res, diff);
  }
  free(ref);
  gpu_free(&w), gpu_free(&sc), gpu_free(&x), gpu_free(&res), gpu_free(&y),
      gpu_free(&x_pack);
}

// Tests Mamba-2 conv1d (prefill + save + batched decode), SSM recurrence
// (prefill + batched decode), and group-512 gated RMSNorm across 2 prefill
// chunks (7 + 5) followed by 3 decode steps on 2 distinct states.
static void test_mamba2(Gpu *g, MemorySlot *work) {
  static constexpr int CH1 = 7, CH2 = 5, DEC = 3, TOT = CH1 + CH2 + DEC;
  static constexpr int NB = 2;

  size_t conv_bytes =
      (size_t)NEMO_MAMBA_CONV_HIST * NEMO_MAMBA_CONV_DIM * sizeof(float);
  size_t ssm_bytes = (size_t)NEMO_MAMBA_HEADS * NEMO_MAMBA_HEAD_DIM *
                     NEMO_MAMBA_STATE * sizeof(float);

  Buf zxbcdt_all[NB];
  Buf conv_st[NB], ssm_st[NB];
  for (int b = 0; b < NB; b++) {
    zxbcdt_all[b] =
        make_buf(g, work, (size_t)TOT * NEMO_MAMBA_IN_DIM * sizeof(float));
    float *zp = (float *)zxbcdt_all[b].ptr;
    for (int i = 0; i < TOT * NEMO_MAMBA_IN_DIM; i++)
      zp[i] = round_bf(randf() * 0.25f);
    conv_st[b] = make_buf(g, work, conv_bytes);
    ssm_st[b] = make_buf(g, work, ssm_bytes);
    memset(conv_st[b].ptr, 0, conv_bytes);
    memset(ssm_st[b].ptr, 0, ssm_bytes);
  }

  Buf conv_w =
      make_buf(g, work, (size_t)NEMO_MAMBA_CONV_DIM * 4 * sizeof(float));
  Buf conv_b = make_buf(g, work, (size_t)NEMO_MAMBA_CONV_DIM * sizeof(float));
  Buf a_exp = make_buf(g, work, (size_t)NEMO_MAMBA_HEADS * sizeof(float));
  Buf d_skip = make_buf(g, work, (size_t)NEMO_MAMBA_HEADS * sizeof(float));
  Buf dt_bias = make_buf(g, work, (size_t)NEMO_MAMBA_HEADS * sizeof(float));
  Buf norm_w = make_buf(g, work, (size_t)NEMO_MAMBA_INNER * sizeof(float));
  for (int i = 0; i < NEMO_MAMBA_CONV_DIM * 4; i++)
    ((float *)conv_w.ptr)[i] = randf() * 0.2f;
  for (int i = 0; i < NEMO_MAMBA_CONV_DIM; i++)
    ((float *)conv_b.ptr)[i] = randf() * 0.1f;
  for (int h = 0; h < NEMO_MAMBA_HEADS; h++) {
    ((float *)a_exp.ptr)[h] = 0.5f + 0.5f * fabsf(randf());
    ((float *)d_skip.ptr)[h] = 0.5f + randf() * 0.3f;
    ((float *)dt_bias.ptr)[h] = -1.5f + randf() * 0.5f;
  }
  for (int i = 0; i < NEMO_MAMBA_INNER; i++)
    ((float *)norm_w.ptr)[i] = 0.8f + 0.3f * randf();

  Buf xbc =
      make_buf(g, work, (size_t)TOT * NEMO_MAMBA_CONV_DIM * sizeof(float));
  Buf y_ssm = make_buf(g, work, (size_t)TOT * NEMO_MAMBA_INNER * sizeof(float));
  Buf y_norm =
      make_buf(g, work, (size_t)TOT * NEMO_MAMBA_INNER * sizeof(float));
  Buf dec_in =
      make_buf(g, work, (size_t)NB * NEMO_MAMBA_IN_DIM * sizeof(float));
  Buf tgt = make_buf(g, work, (size_t)NB * sizeof(MambaTarget));

  int k_cp = gpu_find_kernel(g, "nemo_mamba_conv_prefill");
  int k_cs = gpu_find_kernel(g, "nemo_mamba_conv_save");
  int k_cd = gpu_find_kernel(g, "nemo_mamba_conv_decode");
  int k_sp = gpu_find_kernel(g, "nemo_mamba_ssm_prefill");
  int k_sd = gpu_find_kernel(g, "nemo_mamba_ssm_decode");
  int k_ng = gpu_find_kernel(g, "nemo_mamba_norm_gated");

  uint32_t conv_gx = (NEMO_MAMBA_CONV_DIM + CONV_THREADS - 1u) / CONV_THREADS;
  uint32_t ssm_gx = NEMO_MAMBA_HEAD_DIM / 4u;

  float *gpu_out[NB];
  for (int b = 0; b < NB; b++) {
    gpu_out[b] = xmalloc((size_t)TOT * NEMO_MAMBA_INNER * sizeof(float));
    int chunks[2] = {CH1, CH2};
    int off = 0;
    for (int ci = 0; ci < 2; ci++) {
      int rows = chunks[ci];
      MambaParam mp = {(uint32_t)rows, RMS_EPS};
      size_t zoff = (size_t)off * NEMO_MAMBA_IN_DIM * sizeof(float);
      GpuCommandBuffer cmd = gpu_begin(g);
      gpu_select_kernel(&cmd, k_cp);
      gpu_bind_buffer(&cmd, 0, &zxbcdt_all[b], zoff);
      gpu_bind_buffer(&cmd, 1, &conv_st[b], 0);
      gpu_bind_buffer(&cmd, 2, &conv_w, 0);
      gpu_bind_buffer(&cmd, 3, &conv_b, 0);
      gpu_bind_buffer(&cmd, 4, &xbc, 0);
      gpu_copy_bytes(&cmd, 5, &mp, sizeof mp);
      gpu_run_kernel(&cmd, conv_gx, (uint32_t)rows, 1, CONV_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_cs);
      gpu_bind_buffer(&cmd, 0, &zxbcdt_all[b], zoff);
      gpu_bind_buffer(&cmd, 1, &conv_st[b], 0);
      gpu_copy_bytes(&cmd, 2, &mp, sizeof mp);
      gpu_run_kernel(&cmd, conv_gx, 1, 1, CONV_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_sp);
      gpu_bind_buffer(&cmd, 0, &zxbcdt_all[b], zoff);
      gpu_bind_buffer(&cmd, 1, &xbc, 0);
      gpu_bind_buffer(&cmd, 2, &ssm_st[b], 0);
      gpu_bind_buffer(&cmd, 3, &a_exp, 0);
      gpu_bind_buffer(&cmd, 4, &d_skip, 0);
      gpu_bind_buffer(&cmd, 5, &dt_bias, 0);
      gpu_bind_buffer(&cmd, 6, &y_ssm, 0);
      gpu_copy_bytes(&cmd, 7, &mp, sizeof mp);
      gpu_run_kernel(&cmd, ssm_gx, NEMO_MAMBA_HEADS, 1, SSM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_ng);
      gpu_bind_buffer(&cmd, 0, &y_ssm, 0);
      gpu_bind_buffer(&cmd, 1, &zxbcdt_all[b], zoff);
      gpu_bind_buffer(&cmd, 2, &norm_w, 0);
      gpu_bind_buffer(&cmd, 3, &y_norm, 0);
      gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
      gpu_run_kernel(&cmd, NEMO_MAMBA_GROUPS, (uint32_t)rows, 1, ROW_THREADS, 1,
                     1);
      gpu_commit(&cmd);
      gpu_sync(g);
      memcpy(gpu_out[b] + (size_t)off * NEMO_MAMBA_INNER, y_norm.ptr,
             (size_t)rows * NEMO_MAMBA_INNER * sizeof(float));
      off += rows;
    }
  }

  // Batched 2-state decode for final DEC tokens.
  MambaTarget *tp = (MambaTarget *)tgt.ptr;
  for (int b = 0; b < NB; b++) {
    tp[b] = (MambaTarget){conv_st[b].address + conv_st[b].off,
                          ssm_st[b].address + ssm_st[b].off};
  }
  for (int step = 0; step < DEC; step++) {
    int tok = CH1 + CH2 + step;
    for (int b = 0; b < NB; b++) {
      memcpy(dec_in.ptr + (size_t)b * NEMO_MAMBA_IN_DIM * sizeof(float),
             zxbcdt_all[b].ptr +
                 (size_t)tok * NEMO_MAMBA_IN_DIM * sizeof(float),
             (size_t)NEMO_MAMBA_IN_DIM * sizeof(float));
    }
    MambaParam mp = {NB, RMS_EPS};
    GpuCommandBuffer cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_cd);
    gpu_bind_buffer(&cmd, 0, &dec_in, 0);
    gpu_bind_buffer(&cmd, 1, &tgt, 0);
    gpu_bind_buffer(&cmd, 2, &conv_w, 0);
    gpu_bind_buffer(&cmd, 3, &conv_b, 0);
    gpu_bind_buffer(&cmd, 4, &xbc, 0);
    gpu_copy_bytes(&cmd, 5, &mp, sizeof mp);
    gpu_run_kernel(&cmd, conv_gx, NB, 1, CONV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_sd);
    gpu_bind_buffer(&cmd, 0, &dec_in, 0);
    gpu_bind_buffer(&cmd, 1, &xbc, 0);
    gpu_bind_buffer(&cmd, 2, &tgt, 0);
    gpu_bind_buffer(&cmd, 3, &a_exp, 0);
    gpu_bind_buffer(&cmd, 4, &d_skip, 0);
    gpu_bind_buffer(&cmd, 5, &dt_bias, 0);
    gpu_bind_buffer(&cmd, 6, &y_ssm, 0);
    gpu_copy_bytes(&cmd, 7, &mp, sizeof mp);
    gpu_run_kernel(&cmd, ssm_gx, NEMO_MAMBA_HEADS, NB, SSM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_ng);
    gpu_bind_buffer(&cmd, 0, &y_ssm, 0);
    gpu_bind_buffer(&cmd, 1, &dec_in, 0);
    gpu_bind_buffer(&cmd, 2, &norm_w, 0);
    gpu_bind_buffer(&cmd, 3, &y_norm, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(&cmd, NEMO_MAMBA_GROUPS, NB, 1, ROW_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(g);
    for (int b = 0; b < NB; b++) {
      memcpy(gpu_out[b] + (size_t)tok * NEMO_MAMBA_INNER,
             (const float *)y_norm.ptr + (size_t)b * NEMO_MAMBA_INNER,
             (size_t)NEMO_MAMBA_INNER * sizeof(float));
    }
  }

  // Independent CPU reference over full sequence of TOT tokens per state.
  float *cpu_xbc = xmalloc((size_t)TOT * NEMO_MAMBA_CONV_DIM * sizeof(float));
  float *cpu_ssm = xmalloc(ssm_bytes);
  float *cpu_out = xmalloc((size_t)TOT * NEMO_MAMBA_INNER * sizeof(float));
  const float *cw = (const float *)conv_w.ptr, *cb = (const float *)conv_b.ptr;
  const float *ae = (const float *)a_exp.ptr, *ds = (const float *)d_skip.ptr,
              *db = (const float *)dt_bias.ptr, *nw = (const float *)norm_w.ptr;

  float worst = 0.0f;
  for (int b = 0; b < NB; b++) {
    const float *zp = (const float *)zxbcdt_all[b].ptr;
    for (int t = 0; t < TOT; t++) {
      for (int c = 0; c < NEMO_MAMBA_CONV_DIM; c++) {
        float acc = cb[c];
        for (int k = 0; k < 4; k++) {
          int sr = t + k - 3;
          float xv =
              sr >= 0
                  ? zp[(size_t)sr * NEMO_MAMBA_IN_DIM + NEMO_MAMBA_INNER + c]
                  : 0.0f;
          acc += cw[c * 4 + k] * xv;
        }
        cpu_xbc[(size_t)t * NEMO_MAMBA_CONV_DIM + c] =
            round_bf(siluf(round_bf(acc)));
      }
    }
    memset(cpu_ssm, 0, ssm_bytes);
    for (int t = 0; t < TOT; t++) {
      float yrow[NEMO_MAMBA_INNER];
      for (int h = 0; h < NEMO_MAMBA_HEADS; h++) {
        int grp = h / (NEMO_MAMBA_HEADS / NEMO_MAMBA_GROUPS);
        float dt = softplusf(zp[(size_t)t * NEMO_MAMBA_IN_DIM +
                                NEMO_MAMBA_INNER + NEMO_MAMBA_CONV_DIM + h] +
                             db[h]);
        float dA = expf(dt * (-ae[h]));
        const float *bp = cpu_xbc + (size_t)t * NEMO_MAMBA_CONV_DIM +
                          NEMO_MAMBA_INNER + grp * NEMO_MAMBA_STATE;
        const float *cp = cpu_xbc + (size_t)t * NEMO_MAMBA_CONV_DIM +
                          NEMO_MAMBA_INNER + NEMO_MAMBA_BC +
                          grp * NEMO_MAMBA_STATE;
        for (int p = 0; p < NEMO_MAMBA_HEAD_DIM; p++) {
          float xv = cpu_xbc[(size_t)t * NEMO_MAMBA_CONV_DIM +
                             h * NEMO_MAMBA_HEAD_DIM + p];
          float *hs = cpu_ssm +
                      ((size_t)h * NEMO_MAMBA_HEAD_DIM + p) * NEMO_MAMBA_STATE;
          double dot = 0.0;
          for (int n = 0; n < NEMO_MAMBA_STATE; n++) {
            hs[n] = hs[n] * dA + (dt * xv) * bp[n];
            dot += (double)hs[n] * (double)cp[n];
          }
          yrow[h * NEMO_MAMBA_HEAD_DIM + p] = (float)dot + ds[h] * xv;
        }
      }
      int gsize = NEMO_MAMBA_INNER / NEMO_MAMBA_GROUPS;
      for (int gr = 0; gr < NEMO_MAMBA_GROUPS; gr++) {
        int base = gr * gsize;
        double sq = 0.0;
        for (int i = 0; i < gsize; i++) {
          float gval = zp[(size_t)t * NEMO_MAMBA_IN_DIM + base + i];
          float v = yrow[base + i] * siluf(gval);
          sq += (double)v * v;
        }
        float inv = 1.0f / sqrtf((float)(sq / gsize) + RMS_EPS);
        for (int i = 0; i < gsize; i++) {
          float gval = zp[(size_t)t * NEMO_MAMBA_IN_DIM + base + i];
          float v = yrow[base + i] * siluf(gval);
          cpu_out[(size_t)t * NEMO_MAMBA_INNER + base + i] =
              round_bf(nw[base + i] * round_bf(v * inv));
        }
      }
    }
    float d = max_abs_diff(gpu_out[b], cpu_out, (size_t)TOT * NEMO_MAMBA_INNER);
    if (d > worst) worst = d;
    free(gpu_out[b]);
  }
  checkf(worst <= TOL_SSM,
         "Mamba-2 chunked prefill + batched decode + gated norm matches CPU "
         "(max=%.2e)",
         worst);

  free(cpu_xbc), free(cpu_ssm), free(cpu_out);
  for (int b = 0; b < NB; b++) {
    gpu_free(&zxbcdt_all[b]);
    gpu_free(&conv_st[b]);
    gpu_free(&ssm_st[b]);
  }
  gpu_free(&conv_w), gpu_free(&conv_b), gpu_free(&a_exp), gpu_free(&d_skip),
      gpu_free(&dt_bias), gpu_free(&norm_w), gpu_free(&xbc), gpu_free(&y_ssm),
      gpu_free(&y_norm), gpu_free(&dec_in), gpu_free(&tgt);
}

// Tests NoPE GQA 32/2 causal prefill (2 non-aligned chunks 50 + 35) and
// 2-state batched FlashDecoding decode.
static void test_nope_attention(Gpu *g, MemorySlot *work) {
  static constexpr int C1 = 50, C2 = 35, TOT = C1 + C2 + 1, CAP = 128;
  static constexpr int NB = 2, NPARTS = 2, PART_STRIDE = 4;

  Buf k0[NB], k1[NB], v0[NB], v1[NB];
  float *q_all[NB], *k_all[NB], *v_all[NB];
  for (int b = 0; b < NB; b++) {
    k0[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    k1[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    v0[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    v1[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    memset(k0[b].ptr, 0, k0[b].size), memset(k1[b].ptr, 0, k1[b].size);
    memset(v0[b].ptr, 0, v0[b].size), memset(v1[b].ptr, 0, v1[b].size);
    q_all[b] = xmalloc((size_t)TOT * NEMO_Q_DIM * sizeof(float));
    k_all[b] = xmalloc((size_t)TOT * NEMO_KV_DIM * sizeof(float));
    v_all[b] = xmalloc((size_t)TOT * NEMO_KV_DIM * sizeof(float));
    for (int i = 0; i < TOT * NEMO_Q_DIM; i++)
      q_all[b][i] = round_bf(randf() * 0.4f);
    for (int i = 0; i < TOT * NEMO_KV_DIM; i++) {
      k_all[b][i] = round_bf(randf() * 0.4f);
      v_all[b][i] = round_bf(randf() * 0.4f);
    }
  }

  Buf q_raw = make_buf(g, work, (size_t)CAP * NEMO_Q_DIM * sizeof(float));
  Buf k_raw = make_buf(g, work, (size_t)CAP * NEMO_KV_DIM * sizeof(float));
  Buf v_raw = make_buf(g, work, (size_t)CAP * NEMO_KV_DIM * sizeof(float));
  Buf q_hm = make_buf(
      g, work, (size_t)NEMO_HEADS * CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf out = make_buf(g, work, (size_t)CAP * NEMO_Q_DIM * sizeof(float));
  Buf part = make_buf(g, work,
                      (size_t)NB * NEMO_HEADS * PART_STRIDE *
                          (NEMO_HEAD_DIM + 2) * sizeof(float));
  Buf kt = make_buf(g, work, (size_t)NB * sizeof(KvTarget));

  int k_store = gpu_find_kernel(g, "nemo_qkv_store");
  int k_pre = gpu_find_kernel(g, "nemo_attend_prefill");
  int k_part = gpu_find_kernel(g, "nemo_attend_part");
  int k_comb = gpu_find_kernel(g, "nemo_attend_combine");

  float *gpu_out[NB];
  for (int b = 0; b < NB; b++) {
    gpu_out[b] = xmalloc((size_t)TOT * NEMO_Q_DIM * sizeof(float));
    int chunks[2] = {C1, C2}, pos0 = 0;
    for (int ci = 0; ci < 2; ci++) {
      int rows = chunks[ci];
      int qs = ((rows + 63) / 64) * 64;
      memcpy(q_raw.ptr, q_all[b] + (size_t)pos0 * NEMO_Q_DIM,
             (size_t)rows * NEMO_Q_DIM * sizeof(float));
      memcpy(k_raw.ptr, k_all[b] + (size_t)pos0 * NEMO_KV_DIM,
             (size_t)rows * NEMO_KV_DIM * sizeof(float));
      memcpy(v_raw.ptr, v_all[b] + (size_t)pos0 * NEMO_KV_DIM,
             (size_t)rows * NEMO_KV_DIM * sizeof(float));

      QkvParam qp = {(uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, 0u};
      AttnPrefillParam ap = {(uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, CAP};
      GpuCommandBuffer cmd = gpu_begin(g);
      gpu_select_kernel(&cmd, k_store);
      gpu_bind_buffer(&cmd, 0, &q_raw, 0);
      gpu_bind_buffer(&cmd, 1, &k_raw, 0);
      gpu_bind_buffer(&cmd, 2, &v_raw, 0);
      gpu_bind_buffer(&cmd, 3, &q_hm, 0);
      gpu_bind_buffer(&cmd, 4, &k0[b], 0);
      gpu_bind_buffer(&cmd, 5, &k1[b], 0);
      gpu_bind_buffer(&cmd, 6, &v0[b], 0);
      gpu_bind_buffer(&cmd, 7, &v1[b], 0);
      gpu_bind_buffer(&cmd, 8, &kt, 0);
      gpu_copy_bytes(&cmd, 9, &qp, sizeof qp);
      gpu_run_kernel(&cmd, 36, (uint32_t)qs, 1, QKV_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_pre);
      gpu_bind_buffer(&cmd, 0, &q_hm, 0);
      gpu_bind_buffer(&cmd, 1, &k0[b], 0);
      gpu_bind_buffer(&cmd, 2, &k1[b], 0);
      gpu_bind_buffer(&cmd, 3, &v0[b], 0);
      gpu_bind_buffer(&cmd, 4, &v1[b], 0);
      gpu_bind_buffer(&cmd, 5, &out, 0);
      gpu_copy_bytes(&cmd, 6, &ap, sizeof ap);
      gpu_run_kernel(&cmd, (uint32_t)(qs / 64), NEMO_HEADS, 1, PREFILL_THREADS,
                     1, 1);
      gpu_commit(&cmd);
      gpu_sync(g);
      memcpy(gpu_out[b] + (size_t)pos0 * NEMO_Q_DIM, out.ptr,
             (size_t)rows * NEMO_Q_DIM * sizeof(float));
      pos0 += rows;
    }
  }

  // Batched 2-state decode at position C1 + C2.
  int dec_pos = C1 + C2;
  KvTarget *ktp = (KvTarget *)kt.ptr;
  for (int b = 0; b < NB; b++) {
    ktp[b] = (KvTarget){k0[b].address + k0[b].off, k1[b].address + k1[b].off,
                        v0[b].address + v0[b].off, v1[b].address + v1[b].off,
                        (uint32_t)dec_pos,         0u};
    memcpy((float *)q_raw.ptr + (size_t)b * NEMO_Q_DIM,
           q_all[b] + (size_t)dec_pos * NEMO_Q_DIM,
           (size_t)NEMO_Q_DIM * sizeof(float));
    memcpy((float *)k_raw.ptr + (size_t)b * NEMO_KV_DIM,
           k_all[b] + (size_t)dec_pos * NEMO_KV_DIM,
           (size_t)NEMO_KV_DIM * sizeof(float));
    memcpy((float *)v_raw.ptr + (size_t)b * NEMO_KV_DIM,
           v_all[b] + (size_t)dec_pos * NEMO_KV_DIM,
           (size_t)NEMO_KV_DIM * sizeof(float));
  }
  QkvParam qp = {NB, NB, 0u, 1u};
  AttnDecodeParam dp = {NB, NB, 0u, NPARTS, PART_STRIDE, 1u};
  GpuCommandBuffer cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &q_raw, 0);
  gpu_bind_buffer(&cmd, 1, &k_raw, 0);
  gpu_bind_buffer(&cmd, 2, &v_raw, 0);
  gpu_bind_buffer(&cmd, 3, &q_hm, 0);
  gpu_bind_buffer(&cmd, 4, &k0[0], 0);
  gpu_bind_buffer(&cmd, 5, &k1[0], 0);
  gpu_bind_buffer(&cmd, 6, &v0[0], 0);
  gpu_bind_buffer(&cmd, 7, &v1[0], 0);
  gpu_bind_buffer(&cmd, 8, &kt, 0);
  gpu_copy_bytes(&cmd, 9, &qp, sizeof qp);
  gpu_run_kernel(&cmd, 36, NB, 1, QKV_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &q_hm, 0);
  gpu_bind_buffer(&cmd, 1, &k0[0], 0);
  gpu_bind_buffer(&cmd, 2, &k1[0], 0);
  gpu_bind_buffer(&cmd, 3, &v0[0], 0);
  gpu_bind_buffer(&cmd, 4, &v1[0], 0);
  gpu_bind_buffer(&cmd, 5, &kt, 0);
  gpu_bind_buffer(&cmd, 6, &part, 0);
  gpu_copy_bytes(&cmd, 7, &dp, sizeof dp);
  gpu_run_kernel(&cmd, NEMO_KV_HEADS, NPARTS, NB, DECODE_PART_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out, 0);
  gpu_copy_bytes(&cmd, 2, &dp, sizeof dp);
  gpu_run_kernel(&cmd, NEMO_HEADS, NB, 1, DECODE_COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);
  for (int b = 0; b < NB; b++) {
    memcpy(gpu_out[b] + (size_t)dec_pos * NEMO_Q_DIM,
           (const float *)out.ptr + (size_t)b * NEMO_Q_DIM,
           (size_t)NEMO_Q_DIM * sizeof(float));
  }

  // Independent CPU causal GQA 32/2 attention check.
  float worst_pre = 0.0f, worst_dec = 0.0f;
  double scale = 1.0 / sqrt((double)NEMO_HEAD_DIM);
  for (int b = 0; b < NB; b++) {
    for (int t = 0; t < TOT; t++) {
      for (int h = 0; h < NEMO_HEADS; h++) {
        int kvh = h / (NEMO_HEADS / NEMO_KV_HEADS);
        const float *qv = q_all[b] + (size_t)t * NEMO_Q_DIM + h * NEMO_HEAD_DIM;
        double scs[TOT], mx = -1e30;
        for (int s = 0; s <= t; s++) {
          const float *kv =
              k_all[b] + (size_t)s * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM;
          double dot = 0.0;
          for (int d = 0; d < NEMO_HEAD_DIM; d++) dot += (double)qv[d] * kv[d];
          scs[s] = dot * scale;
          if (scs[s] > mx) mx = scs[s];
        }
        double sum = 0.0;
        for (int s = 0; s <= t; s++) {
          scs[s] = exp(scs[s] - mx);
          sum += scs[s];
        }
        for (int d = 0; d < NEMO_HEAD_DIM; d++) {
          double acc = 0.0;
          for (int s = 0; s <= t; s++) {
            const float *vv =
                v_all[b] + (size_t)s * NEMO_KV_DIM + kvh * NEMO_HEAD_DIM;
            acc += (scs[s] / sum) * (double)vv[d];
          }
          float refv = round_bf((float)acc);
          float gv = gpu_out[b][(size_t)t * NEMO_Q_DIM + h * NEMO_HEAD_DIM + d];
          float diff = fabsf(gv - refv);
          if (t < dec_pos) {
            if (diff > worst_pre) worst_pre = diff;
          } else {
            if (diff > worst_dec) worst_dec = diff;
          }
        }
      }
    }
    free(gpu_out[b]), free(q_all[b]), free(k_all[b]), free(v_all[b]);
    gpu_free(&k0[b]), gpu_free(&k1[b]), gpu_free(&v0[b]), gpu_free(&v1[b]);
  }
  checkf(worst_pre <= TOL_ATTN,
         "NoPE GQA 32/2 chunked causal prefill matches CPU (max=%.2e)",
         worst_pre);
  checkf(worst_dec <= TOL_ATTN,
         "NoPE GQA 32/2 batched FlashDecoding matches CPU (max=%.2e)",
         worst_dec);

  gpu_free(&q_raw), gpu_free(&k_raw), gpu_free(&v_raw), gpu_free(&q_hm),
      gpu_free(&out), gpu_free(&part), gpu_free(&kt);
}

// Tests MoE top-6 routing, GPU grouping, CPU tile grouping, NVFP4 up/down
// MV and MM kernels (relu2, 8-byte-aligned addresses), and combine+shared+res.
static void test_moe_pipeline(Gpu *g, MemorySlot *work) {
  static constexpr int DIM = 128, INTER = 128, EXPS = 128, TOPK = 6;
  static constexpr int ROWS = 24;
  static constexpr int NPAIRS = ROWS * TOPK;
  int blocks_up = DIM / NEMO_NVFP4_BLOCK;
  int blocks_dn = INTER / NEMO_NVFP4_BLOCK;
  size_t w_up_bytes = (size_t)INTER * (DIM / 2);
  size_t s_up_bytes = (size_t)INTER * blocks_up;
  size_t w_dn_bytes = (size_t)DIM * (INTER / 2);
  size_t s_dn_bytes = (size_t)DIM * blocks_dn;
  size_t one_exp = OFF8 + w_up_bytes + s_up_bytes + w_dn_bytes + s_dn_bytes;

  Buf pool = make_buf(g, work, (size_t)EXPS * one_exp);
  Buf etable = make_buf(g, work, (size_t)EXPS * sizeof(ExpertRef));
  Buf f_up = make_buf(g, work, (size_t)EXPS * sizeof(float));
  Buf f_dn = make_buf(g, work, (size_t)EXPS * sizeof(float));
  Buf rlogits = make_buf(g, work, (size_t)ROWS * EXPS * sizeof(float));
  Buf rbias = make_buf(g, work, (size_t)EXPS * sizeof(float));
  Buf rids = make_buf(g, work, (size_t)NPAIRS * sizeof(uint32_t));
  Buf rw = make_buf(g, work, (size_t)NPAIRS * sizeof(float));
  Buf ptok = make_buf(g, work, (size_t)NPAIRS * sizeof(int32_t));
  Buf pw = make_buf(g, work, (size_t)NPAIRS * sizeof(float));
  Buf tp = make_buf(g, work, (size_t)NPAIRS * sizeof(int32_t));
  Buf tiles = make_buf(g, work, (size_t)(NPAIRS + EXPS) * sizeof(NemoMoeTile));
  Buf x = make_buf(g, work, (size_t)ROWS * DIM * sizeof(float));
  Buf shared = make_buf(g, work, (size_t)ROWS * DIM * sizeof(float));
  Buf cur = make_buf(g, work, (size_t)ROWS * DIM * sizeof(float));
  Buf x_bf16 =
      make_buf(g, work, (size_t)(NPAIRS + 64) * DIM * sizeof(uint16_t));
  Buf h_bf16 =
      make_buf(g, work, (size_t)(NPAIRS + 64) * INTER * sizeof(uint16_t));
  Buf y_pair = make_buf(g, work, (size_t)NPAIRS * DIM * sizeof(float));

  // Populate 128 experts at 8-byte (non-16-byte) aligned offsets.
  ExpertRef *er = (ExpertRef *)etable.ptr;
  for (int e = 0; e < EXPS; e++) {
    size_t base_off = (size_t)e * one_exp + OFF8;
    uint8_t *bp = pool.ptr + base_off;
    uint64_t ga = pool.address + pool.off + base_off;
    for (size_t i = 0; i < w_up_bytes; i++) bp[i] = (uint8_t)(randu() & 0xFFu);
    for (size_t i = 0; i < s_up_bytes; i++) bp[w_up_bytes + i] = rand_e4m3(1);
    size_t dn_off = w_up_bytes + s_up_bytes;
    for (size_t i = 0; i < w_dn_bytes; i++)
      bp[dn_off + i] = (uint8_t)(randu() & 0xFFu);
    for (size_t i = 0; i < s_dn_bytes; i++)
      bp[dn_off + w_dn_bytes + i] = rand_e4m3(1);
    er[e] =
        (ExpertRef){ga, ga + w_up_bytes, ga + dn_off, ga + dn_off + w_dn_bytes};
    ((float *)f_up.ptr)[e] = 0.0005f + 0.0003f * fabsf(randf());
    ((float *)f_dn.ptr)[e] = 0.0005f + 0.0003f * fabsf(randf());
  }

  // Bias experts 0..5 heavily so some tiles have count > 8 (MM) and others <= 8
  // (MV), exercising both paths in one pass.
  float *rlp = (float *)rlogits.ptr, *rbp = (float *)rbias.ptr;
  for (int e = 0; e < EXPS; e++)
    rbp[e] = (e < 4 ? 1.5f : 0.0f) + 0.05f * randf();
  for (int r = 0; r < ROWS; r++) {
    for (int e = 0; e < EXPS; e++) rlp[r * EXPS + e] = randf() * 1.2f;
  }
  float *xp = (float *)x.ptr, *shp = (float *)shared.ptr,
        *cp = (float *)cur.ptr;
  float orig_cur[ROWS * DIM];
  for (int i = 0; i < ROWS * DIM; i++) {
    xp[i] = round_bf(randf() * 0.25f);
    shp[i] = round_bf(randf() * 0.2f);
    orig_cur[i] = cp[i] = round_bf(randf() * 0.3f);
  }

  // Run GPU routing first.
  RouteParam rp = {ROWS, EXPS, TOPK, ROUTE_SCALE};
  GpuCommandBuffer cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_route_topk"));
  gpu_bind_buffer(&cmd, 0, &rlogits, 0);
  gpu_bind_buffer(&cmd, 1, &rbias, 0);
  gpu_bind_buffer(&cmd, 2, &rids, 0);
  gpu_bind_buffer(&cmd, 3, &rw, 0);
  gpu_copy_bytes(&cmd, 4, &rp, sizeof rp);
  gpu_run_kernel(&cmd, ROWS, 1, 1, ROUTE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  // Verify routing against CPU.
  int route_ok = 1;
  float route_w_err = 0.0f;
  const uint32_t *got_ids = (const uint32_t *)rids.ptr;
  const float *got_w = (const float *)rw.ptr;
  for (int r = 0; r < ROWS; r++) {
    float sc[EXPS], key[EXPS];
    for (int e = 0; e < EXPS; e++) {
      sc[e] = 1.0f / (1.0f + expf(-rlp[r * EXPS + e]));
      key[e] = sc[e] + rbp[e];
    }
    int best[TOPK];
    uint8_t picked[EXPS] = {0};
    for (int k = 0; k < TOPK; k++) {
      int bi = -1;
      for (int e = 0; e < EXPS; e++)
        if (!picked[e] && (bi < 0 || key[e] > key[bi])) bi = e;
      picked[bi] = 1;
      best[k] = bi;
    }
    double sum = 0.0;
    for (int k = 0; k < TOPK; k++) sum += sc[best[k]];
    for (int k = 0; k < TOPK; k++) {
      uint32_t ge = got_ids[r * TOPK + k];
      if (ge >= (uint32_t)EXPS || !picked[ge]) route_ok = 0;
      float ew = (float)((double)sc[ge] / (sum + ROUTE_EPS) * ROUTE_SCALE);
      float d = fabsf(got_w[r * TOPK + k] - ew);
      if (d > route_w_err) route_w_err = d;
    }
  }
  checkf(route_ok && route_w_err < TOL_F32,
         "nemo_route_topk matches CPU top-6 sigmoid+bias*2.5 (w_err=%.2e)",
         route_w_err);

  // Group into tiles on CPU and verify both MV and MM tiles exist.
  int ntiles = nemotron_group_expert_tiles(
      got_ids, got_w, ROWS, MOE_MM_TILE, (int *)ptok.ptr, (float *)pw.ptr,
      (int *)tp.ptr, (NemoMoeTile *)tiles.ptr);
  int has_mv = 0, has_mm = 0;
  for (int t = 0; t < ntiles; t++) {
    if (((const NemoMoeTile *)tiles.ptr)[t].count <= MOE_MV_MAX_M)
      has_mv = 1;
    else
      has_mm = 1;
  }
  checkf(has_mv && has_mm,
         "grouped %d tiles covering both MV (<=8) and MM (>8) paths", ntiles);

  // Run MoE gather + up (MV+MM) + down (MV+MM) + combine_add.
  uint32_t dim4 = DIM / 4u;
  MoeParam mp_up = {INTER, DIM, NPAIRS, (uint32_t)ntiles};
  MoeParam mp_dn = {DIM, INTER, NPAIRS, (uint32_t)ntiles};
  MoeCombineParam mcp = {ROWS, DIM, TOPK};
  cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_gather_bf16"));
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &ptok, 0);
  gpu_bind_buffer(&cmd, 2, &x_bf16, 0);
  gpu_copy_bytes(&cmd, 3, &dim4, sizeof dim4);
  gpu_run_kernel(&cmd, (dim4 + 31u) / 32u, NPAIRS, 1, 32, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_nvfp4_up_mv"));
  gpu_bind_buffer(&cmd, 0, &etable, 0);
  gpu_bind_buffer(&cmd, 1, &f_up, 0);
  gpu_bind_buffer(&cmd, 2, &x, 0);
  gpu_bind_buffer(&cmd, 3, &ptok, 0);
  gpu_bind_buffer(&cmd, 4, &tiles, 0);
  gpu_bind_buffer(&cmd, 5, &h_bf16, 0);
  gpu_copy_bytes(&cmd, 6, &mp_up, sizeof mp_up);
  gpu_run_kernel(&cmd, (INTER + 31u) / 32u, (uint32_t)ntiles, 1, MOE_MV_THREADS,
                 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_nvfp4_up_mm"));
  gpu_bind_buffer(&cmd, 0, &etable, 0);
  gpu_bind_buffer(&cmd, 1, &f_up, 0);
  gpu_bind_buffer(&cmd, 2, &x_bf16, 0);
  gpu_bind_buffer(&cmd, 3, &tiles, 0);
  gpu_bind_buffer(&cmd, 4, &h_bf16, 0);
  gpu_copy_bytes(&cmd, 5, &mp_up, sizeof mp_up);
  gpu_run_kernel(&cmd, INTER / 64u, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_nvfp4_down_mv"));
  gpu_bind_buffer(&cmd, 0, &etable, 0);
  gpu_bind_buffer(&cmd, 1, &f_dn, 0);
  gpu_bind_buffer(&cmd, 2, &h_bf16, 0);
  gpu_bind_buffer(&cmd, 3, &pw, 0);
  gpu_bind_buffer(&cmd, 4, &tiles, 0);
  gpu_bind_buffer(&cmd, 5, &y_pair, 0);
  gpu_copy_bytes(&cmd, 6, &mp_dn, sizeof mp_dn);
  gpu_run_kernel(&cmd, (DIM + 31u) / 32u, (uint32_t)ntiles, 1, MOE_MV_THREADS,
                 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_nvfp4_down_mm"));
  gpu_bind_buffer(&cmd, 0, &etable, 0);
  gpu_bind_buffer(&cmd, 1, &f_dn, 0);
  gpu_bind_buffer(&cmd, 2, &h_bf16, 0);
  gpu_bind_buffer(&cmd, 3, &pw, 0);
  gpu_bind_buffer(&cmd, 4, &tiles, 0);
  gpu_bind_buffer(&cmd, 5, &y_pair, 0);
  gpu_copy_bytes(&cmd, 6, &mp_dn, sizeof mp_dn);
  gpu_run_kernel(&cmd, DIM / 64u, (uint32_t)ntiles, 1, MM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_combine_add"));
  gpu_bind_buffer(&cmd, 0, &y_pair, 0);
  gpu_bind_buffer(&cmd, 1, &tp, 0);
  gpu_bind_buffer(&cmd, 2, &shared, 0);
  gpu_bind_buffer(&cmd, 3, &cur, 0);
  gpu_copy_bytes(&cmd, 4, &mcp, sizeof mcp);
  gpu_run_kernel(&cmd, (DIM + MOE_COMBINE_THREADS - 1u) / MOE_COMBINE_THREADS,
                 ROWS, 1, MOE_COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  // CPU reference for full MoE + shared + residual.
  float ref_cur[ROWS * DIM];
  for (int r = 0; r < ROWS; r++) {
    float routed[DIM] = {0};
    for (int k = 0; k < TOPK; k++) {
      uint32_t e = got_ids[r * TOPK + k];
      float rw_val = got_w[r * TOPK + k];
      const uint8_t *bp = pool.ptr + (size_t)e * one_exp + OFF8;
      const uint8_t *wu = bp, *su = bp + w_up_bytes;
      const uint8_t *wd = bp + w_up_bytes + s_up_bytes, *sd = wd + w_dn_bytes;
      float efu = ((const float *)f_up.ptr)[e],
            efd = ((const float *)f_dn.ptr)[e];
      float h[INTER];
      for (int o = 0; o < INTER; o++) {
        double acc = 0.0;
        for (int b = 0; b < blocks_up; b++) {
          float s = e4m3_to_f32(su[o * blocks_up + b]);
          double bs = 0.0;
          for (int j = 0; j < NEMO_NVFP4_BLOCK; j++) {
            int col = b * NEMO_NVFP4_BLOCK + j;
            uint8_t byte = wu[o * (DIM / 2) + col / 2];
            uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 15u);
            bs += (double)e2m1_to_f32(nib) * (double)xp[r * DIM + col];
          }
          acc += bs * (double)s;
        }
        h[o] = round_bf(relu2f(round_bf((float)(acc * efu))));
      }
      for (int o = 0; o < DIM; o++) {
        double acc = 0.0;
        for (int b = 0; b < blocks_dn; b++) {
          float s = e4m3_to_f32(sd[o * blocks_dn + b]);
          double bs = 0.0;
          for (int j = 0; j < NEMO_NVFP4_BLOCK; j++) {
            int col = b * NEMO_NVFP4_BLOCK + j;
            uint8_t byte = wd[o * (INTER / 2) + col / 2];
            uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 15u);
            bs += (double)e2m1_to_f32(nib) * (double)h[col];
          }
          acc += bs * (double)s;
        }
        routed[o] += round_bf((float)(acc * efd)) * rw_val;
      }
    }
    for (int d = 0; d < DIM; d++) {
      float moe = round_bf(round_bf(routed[d]) + shp[r * DIM + d]);
      ref_cur[r * DIM + d] = round_bf(orig_cur[r * DIM + d] + moe);
    }
  }
  float diff = max_rel_diff((const float *)cur.ptr, ref_cur, ROWS * DIM);
  checkf(diff <= TOL_REL_BF16,
         "MoE NVFP4 relu2 MV+MM (off=8) + combine_add matches CPU (rel=%.2e)",
         diff);

  // Also test GPU nemo_moe_group on 3 decode rows.
  uint32_t mt[2] = {3u, TOPK};
  cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_group"));
  gpu_bind_buffer(&cmd, 0, &rids, 0);
  gpu_bind_buffer(&cmd, 1, &rw, 0);
  gpu_bind_buffer(&cmd, 2, &ptok, 0);
  gpu_bind_buffer(&cmd, 3, &pw, 0);
  gpu_bind_buffer(&cmd, 4, &tp, 0);
  gpu_bind_buffer(&cmd, 5, &tiles, 0);
  gpu_copy_bytes(&cmd, 6, mt, sizeof mt);
  gpu_run_kernel(&cmd, 1, 1, 1, ROUTE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  int grp_ok = 1;
  uint32_t total_cnt = 0;
  for (uint32_t t = 0; t < 3u * TOPK; t++)
    total_cnt += ((const NemoMoeTile *)tiles.ptr)[t].count;
  if (total_cnt != 3u * TOPK) grp_ok = 0;
  for (int r = 0; r < 3; r++) {
    for (int k = 0; k < TOPK; k++) {
      int dst = ((const int *)tp.ptr)[r * TOPK + k];
      if (dst < 0 || dst >= 3 * TOPK || ((const int *)ptok.ptr)[dst] != r ||
          ((const float *)pw.ptr)[dst] != got_w[r * TOPK + k])
        grp_ok = 0;
    }
  }
  checkf(grp_ok, "nemo_moe_group matches routing inverse map on 3 decode rows");

  gpu_free(&pool), gpu_free(&etable), gpu_free(&f_up), gpu_free(&f_dn),
      gpu_free(&rlogits), gpu_free(&rbias), gpu_free(&rids), gpu_free(&rw),
      gpu_free(&ptok), gpu_free(&pw), gpu_free(&tp), gpu_free(&tiles),
      gpu_free(&x), gpu_free(&shared), gpu_free(&cur), gpu_free(&x_bf16),
      gpu_free(&h_bf16), gpu_free(&y_pair);
}

int main(void) {
  Memory *mem = memory_new();
  MemoryAccount account;
  MemorySlot work;
  memory_account_init(mem, &account, "nemotron-gpu-test", UINT64_MAX);
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "test work", NULL,
                   NULL);
  work.automatic = 1;

  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *g = gpu_new(src, len);

  test_embed_and_norm(g, &work);
  test_fp8_linear(g, &work);
  test_nvfp4_linear(g, &work);
  test_mamba2(g, &work);
  test_nope_attention(g, &work);
  test_moe_pipeline(g, &work);
  return failures != 0;
}
