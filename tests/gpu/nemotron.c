#include "check.h"
#include "base/error.h"
#include "engine/kv_pages.h"
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
static constexpr uint32_t KV_SKIP_POS = 0xFFFFFFFFu;
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
  // Active rows, output width, inner width, activation, residual and round.
  uint32_t m, n, k, act, has_res, round;
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
  // Base byte offset of this layer's 4 KV slices inside a page.
  uint64_t kv_base_off;
  // Byte size of one head slice inside a page, and log2(page_tokens) (0: flat).
  uint32_t slice_bytes, page_shift;
} QkvParam;

typedef struct {
  // Active rows, Q stride, row-0 pos and KV capacity.
  uint32_t rows, q_stride, pos0, kv_cap;
  // Base byte offset of this layer's 4 KV slices inside a page.
  uint64_t kv_base_off;
  // Byte size of one head slice inside a page, and log2(page_tokens) (0: flat).
  uint32_t slice_bytes, page_shift;
} AttnPrefillParam;

typedef struct {
  // Active rows, Q stride, row-0 pos, partitions, stride and batched flag.
  uint32_t rows, q_stride, pos0, nparts, part_stride, batched;
  // Base byte offset of this layer's 4 KV slices inside a page.
  uint64_t kv_base_off;
  // Byte size of one head slice inside a page, and log2(page_tokens) (0: flat).
  uint32_t slice_bytes, page_shift;
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

typedef struct {
  // Active rows, hidden width and flag selecting h0 for row 0.
  uint32_t rows, dim, use_h0;
  // RMSNorm epsilon.
  float eps;
} MtpEhParam;

typedef struct {
  // Device addresses of one FP8 routed expert's up and down projections.
  uint64_t w_up, w_down;
  // Per-tensor up and down weight scales multiplied by 256.
  float s_up, s_down;
} Fp8ExpertRef;

typedef struct {
  // Winning logit value for one stage-1 unembed threadgroup.
  float val;
  // First vocabulary index achieving val.
  int32_t idx;
} ArgmaxPair;

typedef struct {
  // Vocabulary size and hidden width.
  uint32_t vocab, k;
} UnembedParam;

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
    uint32_t act = (mi % 3u == 1u) ? ACT_RELU2 : ACT_NONE;
    uint32_t has_res = (mi & 1u) ? 1u : 0u;
    for (int r = 0; r < m; r++) {
      for (int o = 0; o < N; o++) {
        double acc = 0.0;
        for (int k = 0; k < K; k++)
          acc += (double)xp[r * K + k] * (double)e4m3_to_f32(wp[o * K + k]);
        float v = round_bf((float)(acc * W_SCALE));
        if (act == ACT_RELU2) v = round_bf(relu2f(v));
        if (has_res) v = round_bf(v + rp[r * N + o]);
        ref[r * N + o] = v;
      }
    }

    Fp8LinParam p = {(uint32_t)m, N, K, act, has_res, 1u, W_SCALE * FP8_UNDO};
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

    float diff = max_rel_diff((const float *)y.ptr, ref, (size_t)m * N);
    checkf(diff <= TOL_REL_BF16,
           "nemo_fp8_%s m=%d act=%u res=%u (off=8) matches CPU (rel=%.2e)",
           m <= (int)MOE_MV_MAX_M ? "mv" : "mm", m, act, has_res, diff);
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

      QkvParam qp = {(uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, 0u, 0, 0, 0};
      AttnPrefillParam ap = {
          (uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, CAP, 0, 0, 0};
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
  QkvParam qp = {NB, NB, 0u, 1u, 0, 0, 0};
  AttnDecodeParam dp = {NB, NB, 0u, NPARTS, PART_STRIDE, 1u, 0, 0, 0};
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

// Tests MTP-specific kernels: nemo_kv_store (single-state and batched with
// skip), nemo_mtp_eh_cat (both use_h0 modes), nemo_moe_fp8_up_mv +
// nemo_moe_fp8_down_mv + combine_add (8-byte offset weights), and fused
// nemo_unembed_argmax_part + nemo_unembed_argmax_final (with tie-breaking).
static void test_mtp_kernels(Gpu *g, MemorySlot *work) {
  // 1. nemo_kv_store: single-state write at pos0=3 (rows=2) + batched write
  // with row 1 skipped via KV_SKIP_POS.
  static constexpr int KV_CAP = 16, KV_ROWS = 2, KV_POS0 = 3;
  Buf k_raw = make_buf(g, work, (size_t)KV_ROWS * NEMO_KV_DIM * sizeof(float));
  Buf v_raw = make_buf(g, work, (size_t)KV_ROWS * NEMO_KV_DIM * sizeof(float));
  Buf k0 = make_buf(g, work, (size_t)KV_CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf k1 = make_buf(g, work, (size_t)KV_CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf v0 = make_buf(g, work, (size_t)KV_CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf v1 = make_buf(g, work, (size_t)KV_CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf kt = make_buf(g, work, (size_t)KV_ROWS * sizeof(KvTarget));
  memset(k0.ptr, 0, k0.size), memset(k1.ptr, 0, k1.size);
  memset(v0.ptr, 0, v0.size), memset(v1.ptr, 0, v1.size);
  for (int i = 0; i < KV_ROWS * NEMO_KV_DIM; i++) {
    ((float *)k_raw.ptr)[i] = round_bf(randf());
    ((float *)v_raw.ptr)[i] = round_bf(randf());
  }
  QkvParam qp_single = {KV_ROWS, KV_ROWS, KV_POS0, 0u, 0, 0, 0};
  GpuCommandBuffer cmd = gpu_begin(g);
  int k_kv_store = gpu_find_kernel(g, "nemo_kv_store");
  gpu_select_kernel(&cmd, k_kv_store);
  gpu_bind_buffer(&cmd, 0, &k_raw, 0);
  gpu_bind_buffer(&cmd, 1, &v_raw, 0);
  gpu_bind_buffer(&cmd, 2, &k0, 0);
  gpu_bind_buffer(&cmd, 3, &k1, 0);
  gpu_bind_buffer(&cmd, 4, &v0, 0);
  gpu_bind_buffer(&cmd, 5, &v1, 0);
  gpu_bind_buffer(&cmd, 6, &kt, 0);
  gpu_copy_bytes(&cmd, 7, &qp_single, sizeof qp_single);
  gpu_run_kernel(&cmd, 2u * NEMO_KV_HEADS, KV_ROWS, 1, QKV_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  int kv_ok = 1;
  for (int r = 0; r < KV_ROWS; r++) {
    int p = KV_POS0 + r;
    for (int d = 0; d < NEMO_HEAD_DIM; d++) {
      if (bf16_to_f32(((const uint16_t *)k0.ptr)[p * NEMO_HEAD_DIM + d]) !=
              ((const float *)k_raw.ptr)[r * NEMO_KV_DIM + d] ||
          bf16_to_f32(((const uint16_t *)k1.ptr)[p * NEMO_HEAD_DIM + d]) !=
              ((const float *)k_raw.ptr)[r * NEMO_KV_DIM + NEMO_HEAD_DIM + d] ||
          bf16_to_f32(((const uint16_t *)v0.ptr)[p * NEMO_HEAD_DIM + d]) !=
              ((const float *)v_raw.ptr)[r * NEMO_KV_DIM + d] ||
          bf16_to_f32(((const uint16_t *)v1.ptr)[p * NEMO_HEAD_DIM + d]) !=
              ((const float *)v_raw.ptr)[r * NEMO_KV_DIM + NEMO_HEAD_DIM + d])
        kv_ok = 0;
    }
  }
  ((KvTarget *)kt.ptr)[0] = (KvTarget){k0.address + k0.off,
                                       k1.address + k1.off,
                                       v0.address + v0.off,
                                       v1.address + v1.off,
                                       9u,
                                       0u};
  ((KvTarget *)kt.ptr)[1] =
      (KvTarget){k0.address + k0.off, k1.address + k1.off, v0.address + v0.off,
                 v1.address + v1.off, KV_SKIP_POS,         0u};
  QkvParam qp_batch = {KV_ROWS, KV_ROWS, 0u, 1u, 0, 0, 0};
  cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, k_kv_store);
  gpu_bind_buffer(&cmd, 0, &k_raw, 0);
  gpu_bind_buffer(&cmd, 1, &v_raw, 0);
  gpu_bind_buffer(&cmd, 2, &k0, 0);
  gpu_bind_buffer(&cmd, 3, &k1, 0);
  gpu_bind_buffer(&cmd, 4, &v0, 0);
  gpu_bind_buffer(&cmd, 5, &v1, 0);
  gpu_bind_buffer(&cmd, 6, &kt, 0);
  gpu_copy_bytes(&cmd, 7, &qp_batch, sizeof qp_batch);
  gpu_run_kernel(&cmd, 2u * NEMO_KV_HEADS, KV_ROWS, 1, QKV_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);
  for (int d = 0; d < NEMO_HEAD_DIM; d++) {
    if (bf16_to_f32(((const uint16_t *)k0.ptr)[9 * NEMO_HEAD_DIM + d]) !=
            ((const float *)k_raw.ptr)[d] ||
        bf16_to_f32(((const uint16_t *)v1.ptr)[9 * NEMO_HEAD_DIM + d]) !=
            ((const float *)v_raw.ptr)[NEMO_HEAD_DIM + d])
      kv_ok = 0;
  }
  checkf(kv_ok, "nemo_kv_store single-state and batched+skip match CPU");
  gpu_free(&k_raw), gpu_free(&v_raw), gpu_free(&k0), gpu_free(&k1),
      gpu_free(&v0), gpu_free(&v1), gpu_free(&kt);

  // 2. nemo_mtp_eh_cat for use_h0 = 0 and use_h0 = 1.
  static constexpr int EH_ROWS = 4, EH_DIM = 256;
  Buf emb = make_buf(g, work, (size_t)EH_ROWS * EH_DIM * sizeof(float));
  Buf h0 = make_buf(g, work, (size_t)EH_DIM * sizeof(float));
  Buf hseq = make_buf(g, work, (size_t)EH_ROWS * EH_DIM * sizeof(float));
  Buf ew = make_buf(g, work, (size_t)EH_DIM * sizeof(float));
  Buf hw = make_buf(g, work, (size_t)EH_DIM * sizeof(float));
  Buf eh_out = make_buf(g, work, (size_t)EH_ROWS * 2 * EH_DIM * sizeof(float));
  for (int i = 0; i < EH_ROWS * EH_DIM; i++) {
    ((float *)emb.ptr)[i] = randf();
    ((float *)hseq.ptr)[i] = randf();
  }
  for (int i = 0; i < EH_DIM; i++) {
    ((float *)h0.ptr)[i] = randf();
    ((float *)ew.ptr)[i] = 0.25f + 0.1f * randf();
    ((float *)hw.ptr)[i] = 1.8f + 0.3f * randf();
  }
  float eh_ref[EH_ROWS * 2 * EH_DIM];
  int k_eh = gpu_find_kernel(g, "nemo_mtp_eh_cat");
  for (uint32_t use_h0 = 0; use_h0 <= 1u; use_h0++) {
    MtpEhParam ep = {EH_ROWS, EH_DIM, use_h0, RMS_EPS};
    cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_eh);
    gpu_bind_buffer(&cmd, 0, &emb, 0);
    gpu_bind_buffer(&cmd, 1, &h0, 0);
    gpu_bind_buffer(&cmd, 2, &hseq, 0);
    gpu_bind_buffer(&cmd, 3, &ew, 0);
    gpu_bind_buffer(&cmd, 4, &hw, 0);
    gpu_bind_buffer(&cmd, 5, &eh_out, 0);
    gpu_copy_bytes(&cmd, 6, &ep, sizeof ep);
    gpu_run_kernel(&cmd, EH_ROWS, 1, 1, ROW_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(g);
    for (int r = 0; r < EH_ROWS; r++) {
      const float *er = (const float *)emb.ptr + r * EH_DIM;
      const float *hr = (use_h0 && r == 0) ? (const float *)h0.ptr
                                           : (const float *)hseq.ptr +
                                                 (use_h0 ? r - 1 : r) * EH_DIM;
      double sqe = 0.0, sqh = 0.0;
      for (int d = 0; d < EH_DIM; d++) {
        sqe += (double)er[d] * er[d];
        sqh += (double)hr[d] * hr[d];
      }
      float inve = 1.0f / sqrtf((float)(sqe / EH_DIM) + RMS_EPS);
      float invh = 1.0f / sqrtf((float)(sqh / EH_DIM) + RMS_EPS);
      for (int d = 0; d < EH_DIM; d++) {
        eh_ref[r * 2 * EH_DIM + d] =
            round_bf(((const float *)ew.ptr)[d] * (er[d] * inve));
        eh_ref[r * 2 * EH_DIM + EH_DIM + d] =
            round_bf(((const float *)hw.ptr)[d] * (hr[d] * invh));
      }
    }
    float diff =
        max_abs_diff((const float *)eh_out.ptr, eh_ref, EH_ROWS * 2 * EH_DIM);
    checkf(diff <= TOL_BF16, "nemo_mtp_eh_cat use_h0=%u matches CPU (max=%.2e)",
           use_h0, diff);
  }
  gpu_free(&emb), gpu_free(&h0), gpu_free(&hseq), gpu_free(&ew), gpu_free(&hw),
      gpu_free(&eh_out);

  // 3. nemo_moe_fp8_up_mv + nemo_moe_fp8_down_mv + nemo_moe_combine_add on
  // 2 rows (creating both count==1 and count==2 tiles) with 8-byte offsets.
  static constexpr int FDIM = 128, FINTER = 128, FEXPS = 16, FTOPK = 6,
                       FROWS = 2, FNPAIRS = FROWS * FTOPK;
  size_t fu_bytes = (size_t)FINTER * FDIM;
  size_t fd_bytes = (size_t)FDIM * FINTER;
  size_t fone = OFF8 + fu_bytes + fd_bytes;
  Buf fpool = make_buf(g, work, (size_t)FEXPS * fone);
  Buf ftab = make_buf(g, work, (size_t)FEXPS * sizeof(Fp8ExpertRef));
  Buf fx = make_buf(g, work, (size_t)FROWS * FDIM * sizeof(float));
  Buf fsh = make_buf(g, work, (size_t)FROWS * FDIM * sizeof(float));
  Buf fcur = make_buf(g, work, (size_t)FROWS * FDIM * sizeof(float));
  Buf fptok = make_buf(g, work, (size_t)FNPAIRS * sizeof(int32_t));
  Buf fpw = make_buf(g, work, (size_t)FNPAIRS * sizeof(float));
  Buf ftp = make_buf(g, work, (size_t)FNPAIRS * sizeof(int32_t));
  Buf ftiles =
      make_buf(g, work, (size_t)(FNPAIRS + FEXPS) * sizeof(NemoMoeTile));
  Buf fh = make_buf(g, work, (size_t)FNPAIRS * FINTER * sizeof(uint16_t));
  Buf fyp = make_buf(g, work, (size_t)FNPAIRS * FDIM * sizeof(float));

  Fp8ExpertRef *fer = (Fp8ExpertRef *)ftab.ptr;
  float fsu_raw[FEXPS], fsd_raw[FEXPS];
  for (int e = 0; e < FEXPS; e++) {
    size_t base_off = (size_t)e * fone + OFF8;
    uint8_t *bp = fpool.ptr + base_off;
    uint64_t ga = fpool.address + fpool.off + base_off;
    for (size_t i = 0; i < fu_bytes + fd_bytes; i++) bp[i] = rand_e4m3(0);
    fsu_raw[e] = 0.002f + 0.001f * fabsf(randf());
    fsd_raw[e] = 0.002f + 0.001f * fabsf(randf());
    fer[e] = (Fp8ExpertRef){ga, ga + fu_bytes, fsu_raw[e] * FP8_UNDO,
                            fsd_raw[e] * FP8_UNDO};
  }
  float fcur0[FROWS * FDIM];
  for (int i = 0; i < FROWS * FDIM; i++) {
    ((float *)fx.ptr)[i] = round_bf(randf() * 0.25f);
    ((float *)fsh.ptr)[i] = round_bf(randf() * 0.2f);
    fcur0[i] = ((float *)fcur.ptr)[i] = round_bf(randf() * 0.3f);
  }
  uint32_t frids[FNPAIRS] = {0, 1, 2, 3, 4, 5, 0, 1, 6, 7, 8, 9};
  float frw[FNPAIRS];
  for (int i = 0; i < FNPAIRS; i++) frw[i] = 0.3f + 0.2f * fabsf(randf());
  int fntiles = nemotron_group_expert_tiles(
      frids, frw, FROWS, MOE_MM_TILE, (int *)fptok.ptr, (float *)fpw.ptr,
      (int *)ftp.ptr, (NemoMoeTile *)ftiles.ptr);

  MoeParam fmp_up = {FINTER, FDIM, FNPAIRS, (uint32_t)fntiles};
  MoeParam fmp_dn = {FDIM, FINTER, FNPAIRS, (uint32_t)fntiles};
  MoeCombineParam fmcp = {FROWS, FDIM, FTOPK};
  cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_fp8_up_mv"));
  gpu_bind_buffer(&cmd, 0, &ftab, 0);
  gpu_bind_buffer(&cmd, 1, &fx, 0);
  gpu_bind_buffer(&cmd, 2, &fptok, 0);
  gpu_bind_buffer(&cmd, 3, &ftiles, 0);
  gpu_bind_buffer(&cmd, 4, &fh, 0);
  gpu_copy_bytes(&cmd, 5, &fmp_up, sizeof fmp_up);
  gpu_run_kernel(&cmd, (FINTER + 31u) / 32u, (uint32_t)fntiles, 1,
                 MOE_MV_THREADS, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_fp8_down_mv"));
  gpu_bind_buffer(&cmd, 0, &ftab, 0);
  gpu_bind_buffer(&cmd, 1, &fh, 0);
  gpu_bind_buffer(&cmd, 2, &fpw, 0);
  gpu_bind_buffer(&cmd, 3, &ftiles, 0);
  gpu_bind_buffer(&cmd, 4, &fyp, 0);
  gpu_copy_bytes(&cmd, 5, &fmp_dn, sizeof fmp_dn);
  gpu_run_kernel(&cmd, (FDIM + 31u) / 32u, (uint32_t)fntiles, 1, MOE_MV_THREADS,
                 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_moe_combine_add"));
  gpu_bind_buffer(&cmd, 0, &fyp, 0);
  gpu_bind_buffer(&cmd, 1, &ftp, 0);
  gpu_bind_buffer(&cmd, 2, &fsh, 0);
  gpu_bind_buffer(&cmd, 3, &fcur, 0);
  gpu_copy_bytes(&cmd, 4, &fmcp, sizeof fmcp);
  gpu_run_kernel(&cmd, (FDIM + MOE_COMBINE_THREADS - 1u) / MOE_COMBINE_THREADS,
                 FROWS, 1, MOE_COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  float fref[FROWS * FDIM];
  for (int r = 0; r < FROWS; r++) {
    float routed[FDIM] = {0};
    for (int k = 0; k < FTOPK; k++) {
      uint32_t e = frids[r * FTOPK + k];
      float rw_val = frw[r * FTOPK + k];
      const uint8_t *bp = fpool.ptr + (size_t)e * fone + OFF8;
      const uint8_t *wu = bp, *wd = bp + fu_bytes;
      float hv[FINTER];
      for (int o = 0; o < FINTER; o++) {
        double acc = 0.0;
        for (int c = 0; c < FDIM; c++)
          acc += (double)((const float *)fx.ptr)[r * FDIM + c] *
                 (double)e4m3_to_f32(wu[o * FDIM + c]);
        hv[o] = round_bf(relu2f(round_bf((float)(acc * fsu_raw[e]))));
      }
      for (int o = 0; o < FDIM; o++) {
        double acc = 0.0;
        for (int c = 0; c < FINTER; c++)
          acc += (double)hv[c] * (double)e4m3_to_f32(wd[o * FINTER + c]);
        routed[o] += round_bf((float)(acc * fsd_raw[e])) * rw_val;
      }
    }
    for (int d = 0; d < FDIM; d++) {
      float moe = round_bf(round_bf(routed[d]) +
                           ((const float *)fsh.ptr)[r * FDIM + d]);
      fref[r * FDIM + d] = round_bf(fcur0[r * FDIM + d] + moe);
    }
  }
  float fdiff = max_rel_diff((const float *)fcur.ptr, fref, FROWS * FDIM);
  checkf(fdiff <= TOL_REL_BF16,
         "nemo_moe_fp8_up_mv + down_mv (off=8) + combine_add matches CPU "
         "(rel=%.2e)",
         fdiff);
  gpu_free(&fpool), gpu_free(&ftab), gpu_free(&fx), gpu_free(&fsh),
      gpu_free(&fcur), gpu_free(&fptok), gpu_free(&fpw), gpu_free(&ftp),
      gpu_free(&ftiles), gpu_free(&fh), gpu_free(&fyp);

  // 4. nemo_unembed_argmax_part + nemo_unembed_argmax_final (non-multiple-of-32
  // vocab and exact tie-breaking to lowest index).
  static constexpr int UVOCAB = 1005, UK = 256;
  static constexpr int TIE_LOW = 317, TIE_HIGH = 829;
  uint32_t unparts = (UVOCAB + 31u) / 32u;
  Buf uw = make_buf(g, work, (size_t)UVOCAB * UK * sizeof(uint16_t));
  Buf ux = make_buf(g, work, (size_t)UK * sizeof(float));
  Buf uparts = make_buf(g, work, (size_t)unparts * sizeof(ArgmaxPair));
  Buf utok = make_buf(g, work, sizeof(int32_t));
  uint16_t *uwp = (uint16_t *)uw.ptr;
  for (int i = 0; i < UVOCAB * UK; i++) uwp[i] = f32_to_bf16(randf() * 0.25f);
  float *uxp = (float *)ux.ptr;
  for (int i = 0; i < UK; i++) uxp[i] = round_bf(0.2f + 0.1f * fabsf(randf()));
  // Make rows TIE_LOW and TIE_HIGH identical large positive rows so both tie
  // for maximum and lowest index TIE_LOW must win.
  for (int i = 0; i < UK; i++) {
    uint16_t big = f32_to_bf16(2.0f);
    uwp[TIE_LOW * UK + i] = big;
    uwp[TIE_HIGH * UK + i] = big;
  }
  UnembedParam up = {UVOCAB, UK};
  cmd = gpu_begin(g);
  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_unembed_argmax_part"));
  gpu_bind_buffer(&cmd, 0, &uw, 0);
  gpu_bind_buffer(&cmd, 1, &ux, 0);
  gpu_bind_buffer(&cmd, 2, &uparts, 0);
  gpu_copy_bytes(&cmd, 3, &up, sizeof up);
  gpu_run_kernel(&cmd, unparts, 1, 1, FP8_MV_THREADS, 1, 1);

  gpu_select_kernel(&cmd, gpu_find_kernel(g, "nemo_unembed_argmax_final"));
  gpu_bind_buffer(&cmd, 0, &uparts, 0);
  gpu_bind_buffer(&cmd, 1, &utok, 0);
  gpu_copy_bytes(&cmd, 2, &unparts, sizeof unparts);
  gpu_run_kernel(&cmd, 1, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(g);

  int32_t got_tok = ((const int32_t *)utok.ptr)[0];
  checkf(got_tok == TIE_LOW,
         "nemo_unembed_argmax selects first-maximum winner (got=%d expect=%d)",
         got_tok, TIE_LOW);
  gpu_free(&uw), gpu_free(&ux), gpu_free(&uparts), gpu_free(&utok);
}

// Tests paged NoPE GQA 32/2 attention across the 256-token page boundary:
// chunked prefill (190 + 115 = 305), copy-on-write fork at 305, divergent
// single-state and 2-state batched decode, and paged nemo_kv_store with skip.
// Compares every output bit-for-bit against the flat reference kernels.
static void test_paged_attention(Gpu *g, MemorySlot *work) {
  static constexpr int C1 = 190, C2 = 115, PREFIX = C1 + C2;
  static constexpr int EXTRA = 2, TOT = PREFIX + EXTRA, CAP = 512;
  static constexpr int NB = 2, NPARTS = 4, PART_STRIDE = 4;
  static constexpr int LAYERS = 2, LAYER_IDX = 1;

  size_t slice_bytes =
      (size_t)NEMO_KV_PAGE_TOKENS * NEMO_HEAD_DIM * sizeof(uint16_t);
  StateGrowthPolicy growth = {4, 2, 16, 4, 64};
  KvPageLayout layout =
      kv_page_layout_init(NEMO_KV_PAGE_TOKENS, slice_bytes,
                          LAYERS * NEMO_KV_SLICES_PER_LAYER, growth);
  KvPageBackend backend = kv_page_backend_gpu(g, work);
  uint64_t kv_base_off =
      (uint64_t)(LAYER_IDX * NEMO_KV_SLICES_PER_LAYER) * slice_bytes;

  KvPageTable pt[NB];
  for (int b = 0; b < NB; b++) kv_page_table_init(&pt[b], backend, &layout);
  check(kv_page_table_prepare_write(&pt[0], 0, PREFIX),
        "allocate paged KV for Nemotron prefix across page boundary");

  Buf fk0[NB], fk1[NB], fv0[NB], fv1[NB];
  float *q_all[NB], *k_all[NB], *v_all[NB];
  for (int b = 0; b < NB; b++) {
    fk0[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    fk1[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    fv0[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    fv1[b] = make_buf(g, work, (size_t)CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
    memset(fk0[b].ptr, 0, fk0[b].size), memset(fk1[b].ptr, 0, fk1[b].size);
    memset(fv0[b].ptr, 0, fv0[b].size), memset(fv1[b].ptr, 0, fv1[b].size);
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
  // Branch 1 shares prefix [0, PREFIX) with branch 0 and diverges afterward.
  memcpy(q_all[1], q_all[0], (size_t)PREFIX * NEMO_Q_DIM * sizeof(float));
  memcpy(k_all[1], k_all[0], (size_t)PREFIX * NEMO_KV_DIM * sizeof(float));
  memcpy(v_all[1], v_all[0], (size_t)PREFIX * NEMO_KV_DIM * sizeof(float));

  Buf q_raw = make_buf(g, work, (size_t)CAP * NEMO_Q_DIM * sizeof(float));
  Buf k_raw = make_buf(g, work, (size_t)CAP * NEMO_KV_DIM * sizeof(float));
  Buf v_raw = make_buf(g, work, (size_t)CAP * NEMO_KV_DIM * sizeof(float));
  Buf q_hm = make_buf(
      g, work, (size_t)NEMO_HEADS * CAP * NEMO_HEAD_DIM * sizeof(uint16_t));
  Buf out_flat = make_buf(g, work, (size_t)CAP * NEMO_Q_DIM * sizeof(float));
  Buf out_paged = make_buf(g, work, (size_t)CAP * NEMO_Q_DIM * sizeof(float));
  Buf part = make_buf(g, work,
                      (size_t)NB * NEMO_HEADS * PART_STRIDE *
                          (NEMO_HEAD_DIM + 2) * sizeof(float));
  Buf kt_flat = make_buf(g, work, (size_t)NB * sizeof(KvTarget));
  Buf kt_paged = make_buf(g, work, (size_t)NB * sizeof(KvTarget));

  int k_store = gpu_find_kernel(g, "nemo_qkv_store");
  int k_kv = gpu_find_kernel(g, "nemo_kv_store");
  int k_pre = gpu_find_kernel(g, "nemo_attend_prefill");
  int k_part = gpu_find_kernel(g, "nemo_attend_part");
  int k_comb = gpu_find_kernel(g, "nemo_attend_combine");

  // 1. Chunked causal prefill across page boundary 256 (190 + 115 = 305).
  float pre_diff = 0.0f;
  int chunks[2] = {C1, C2}, pos0 = 0;
  for (int ci = 0; ci < 2; ci++) {
    int rows = chunks[ci];
    int qs = ((rows + 63) / 64) * 64;
    memcpy(q_raw.ptr, q_all[0] + (size_t)pos0 * NEMO_Q_DIM,
           (size_t)rows * NEMO_Q_DIM * sizeof(float));
    memcpy(k_raw.ptr, k_all[0] + (size_t)pos0 * NEMO_KV_DIM,
           (size_t)rows * NEMO_KV_DIM * sizeof(float));
    memcpy(v_raw.ptr, v_all[0] + (size_t)pos0 * NEMO_KV_DIM,
           (size_t)rows * NEMO_KV_DIM * sizeof(float));

    QkvParam qf = {(uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, 0u, 0, 0, 0};
    AttnPrefillParam af = {
        (uint32_t)rows, (uint32_t)qs, (uint32_t)pos0, CAP, 0, 0, 0};
    QkvParam qp = {(uint32_t)rows, (uint32_t)qs,          (uint32_t)pos0,   0u,
                   kv_base_off,    (uint32_t)slice_bytes, layout.page_shift};
    AttnPrefillParam ap = {
        (uint32_t)rows,   (uint32_t)qs,
        (uint32_t)pos0,   (uint32_t)(pt[0].npages * NEMO_KV_PAGE_TOKENS),
        kv_base_off,      (uint32_t)slice_bytes,
        layout.page_shift};

    GpuCommandBuffer cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 5, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 6, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 7, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 8, &kt_flat, 0);
    gpu_copy_bytes(&cmd, 9, &qf, sizeof qf);
    gpu_run_kernel(&cmd, 36, (uint32_t)qs, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_pre);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 2, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 3, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 4, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 5, &out_flat, 0);
    gpu_copy_bytes(&cmd, 6, &af, sizeof af);
    gpu_run_kernel(&cmd, (uint32_t)(qs / 64), NEMO_HEADS, 1, PREFILL_THREADS, 1,
                   1);

    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 6, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 7, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 8, &kt_paged, 0);
    gpu_copy_bytes(&cmd, 9, &qp, sizeof qp);
    gpu_run_kernel(&cmd, 36, (uint32_t)qs, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_pre);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 2, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 3, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &out_paged, 0);
    gpu_copy_bytes(&cmd, 6, &ap, sizeof ap);
    gpu_run_kernel(&cmd, (uint32_t)(qs / 64), NEMO_HEADS, 1, PREFILL_THREADS, 1,
                   1);
    gpu_commit(&cmd);
    gpu_sync(g);

    float d =
        max_abs_diff((const float *)out_paged.ptr, (const float *)out_flat.ptr,
                     (size_t)rows * NEMO_Q_DIM);
    if (d > pre_diff) pre_diff = d;
    pos0 += rows;
  }
  checkf(pre_diff == 0.0f,
         "paged nemo_attend_prefill matches flat across 256-token boundary "
         "(max=%.2e)",
         pre_diff);

  // 2. Fork branch 1 from branch 0 at PREFIX (page 0 shared, page 1 copied).
  check(kv_page_table_fork(&pt[1], &pt[0], PREFIX),
        "fork Nemotron paged KV at 305 tokens");
  check(pt[0].pages[0] == pt[1].pages[0] && pt[0].pages[0]->refs == 2 &&
            pt[0].pages[1] != pt[1].pages[1] && pt[1].pages[1]->refs == 1,
        "Nemotron fork shares full prefix page 0 and copies partial page 1");
  memcpy(fk0[1].ptr, fk0[0].ptr,
         (size_t)PREFIX * NEMO_HEAD_DIM * sizeof(uint16_t));
  memcpy(fk1[1].ptr, fk1[0].ptr,
         (size_t)PREFIX * NEMO_HEAD_DIM * sizeof(uint16_t));
  memcpy(fv0[1].ptr, fv0[0].ptr,
         (size_t)PREFIX * NEMO_HEAD_DIM * sizeof(uint16_t));
  memcpy(fv1[1].ptr, fv1[0].ptr,
         (size_t)PREFIX * NEMO_HEAD_DIM * sizeof(uint16_t));

  // 3. Single-state paged FlashDecoding on branch 0 at pos = PREFIX.
  check(kv_page_table_prepare_write(&pt[0], PREFIX, 1),
        "prepare write for single-state decode");
  memcpy(q_raw.ptr, q_all[0] + (size_t)PREFIX * NEMO_Q_DIM,
         (size_t)NEMO_Q_DIM * sizeof(float));
  memcpy(k_raw.ptr, k_all[0] + (size_t)PREFIX * NEMO_KV_DIM,
         (size_t)NEMO_KV_DIM * sizeof(float));
  memcpy(v_raw.ptr, v_all[0] + (size_t)PREFIX * NEMO_KV_DIM,
         (size_t)NEMO_KV_DIM * sizeof(float));
  {
    QkvParam qf = {1u, 1u, (uint32_t)PREFIX, 0u, 0, 0, 0};
    AttnDecodeParam df = {1u, 1u, (uint32_t)PREFIX, NPARTS, PART_STRIDE, 0u, 0,
                          0,  0};
    QkvParam qp = {1u,
                   1u,
                   (uint32_t)PREFIX,
                   0u,
                   kv_base_off,
                   (uint32_t)slice_bytes,
                   layout.page_shift};
    AttnDecodeParam dp = {
        1u, 1u,          (uint32_t)PREFIX,      NPARTS,           PART_STRIDE,
        0u, kv_base_off, (uint32_t)slice_bytes, layout.page_shift};
    GpuCommandBuffer cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 5, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 6, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 7, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 8, &kt_flat, 0);
    gpu_copy_bytes(&cmd, 9, &qf, sizeof qf);
    gpu_run_kernel(&cmd, 36, 1, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_part);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 2, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 3, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 4, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 5, &kt_flat, 0);
    gpu_bind_buffer(&cmd, 6, &part, 0);
    gpu_copy_bytes(&cmd, 7, &df, sizeof df);
    gpu_run_kernel(&cmd, NEMO_KV_HEADS, NPARTS, 1, DECODE_PART_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_comb);
    gpu_bind_buffer(&cmd, 0, &part, 0);
    gpu_bind_buffer(&cmd, 1, &out_flat, 0);
    gpu_copy_bytes(&cmd, 2, &df, sizeof df);
    gpu_run_kernel(&cmd, NEMO_HEADS, 1, 1, DECODE_COMBINE_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 6, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 7, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 8, &kt_paged, 0);
    gpu_copy_bytes(&cmd, 9, &qp, sizeof qp);
    gpu_run_kernel(&cmd, 36, 1, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_part);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 2, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 3, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &kt_paged, 0);
    gpu_bind_buffer(&cmd, 6, &part, 0);
    gpu_copy_bytes(&cmd, 7, &dp, sizeof dp);
    gpu_run_kernel(&cmd, NEMO_KV_HEADS, NPARTS, 1, DECODE_PART_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_comb);
    gpu_bind_buffer(&cmd, 0, &part, 0);
    gpu_bind_buffer(&cmd, 1, &out_paged, 0);
    gpu_copy_bytes(&cmd, 2, &dp, sizeof dp);
    gpu_run_kernel(&cmd, NEMO_HEADS, 1, 1, DECODE_COMBINE_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(g);
  }
  float dec1_diff = max_abs_diff((const float *)out_paged.ptr,
                                 (const float *)out_flat.ptr, NEMO_Q_DIM);
  checkf(dec1_diff == 0.0f,
         "paged single-state FlashDecoding matches flat (max=%.2e)", dec1_diff);

  // 4. Batched 2-state decode at differing positions (PREFIX + 1 vs PREFIX).
  int bpos[NB] = {PREFIX + 1, PREFIX};
  for (int b = 0; b < NB; b++) {
    check(kv_page_table_prepare_write(&pt[b], bpos[b], 1),
          "prepare write for batched decode");
    ((KvTarget *)kt_flat.ptr)[b] =
        (KvTarget){fk0[b].address + fk0[b].off, fk1[b].address + fk1[b].off,
                   fv0[b].address + fv0[b].off, fv1[b].address + fv1[b].off,
                   (uint32_t)bpos[b],           0u};
    ((KvTarget *)kt_paged.ptr)[b] =
        (KvTarget){pt[b].gpu_addrs.address + pt[b].gpu_addrs.off,
                   0,
                   0,
                   0,
                   (uint32_t)bpos[b],
                   0u};
    memcpy((float *)q_raw.ptr + (size_t)b * NEMO_Q_DIM,
           q_all[b] + (size_t)bpos[b] * NEMO_Q_DIM,
           (size_t)NEMO_Q_DIM * sizeof(float));
    memcpy((float *)k_raw.ptr + (size_t)b * NEMO_KV_DIM,
           k_all[b] + (size_t)bpos[b] * NEMO_KV_DIM,
           (size_t)NEMO_KV_DIM * sizeof(float));
    memcpy((float *)v_raw.ptr + (size_t)b * NEMO_KV_DIM,
           v_all[b] + (size_t)bpos[b] * NEMO_KV_DIM,
           (size_t)NEMO_KV_DIM * sizeof(float));
  }
  {
    QkvParam qf = {NB, NB, 0u, 1u, 0, 0, 0};
    AttnDecodeParam df = {NB, NB, 0u, NPARTS, PART_STRIDE, 1u, 0, 0, 0};
    QkvParam qp = {
        NB, NB, 0u, 1u, kv_base_off, (uint32_t)slice_bytes, layout.page_shift};
    AttnDecodeParam dp = {NB,
                          NB,
                          0u,
                          NPARTS,
                          PART_STRIDE,
                          1u,
                          kv_base_off,
                          (uint32_t)slice_bytes,
                          layout.page_shift};
    GpuCommandBuffer cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 5, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 6, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 7, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 8, &kt_flat, 0);
    gpu_copy_bytes(&cmd, 9, &qf, sizeof qf);
    gpu_run_kernel(&cmd, 36, NB, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_part);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &fk0[0], 0);
    gpu_bind_buffer(&cmd, 2, &fk1[0], 0);
    gpu_bind_buffer(&cmd, 3, &fv0[0], 0);
    gpu_bind_buffer(&cmd, 4, &fv1[0], 0);
    gpu_bind_buffer(&cmd, 5, &kt_flat, 0);
    gpu_bind_buffer(&cmd, 6, &part, 0);
    gpu_copy_bytes(&cmd, 7, &df, sizeof df);
    gpu_run_kernel(&cmd, NEMO_KV_HEADS, NPARTS, NB, DECODE_PART_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_comb);
    gpu_bind_buffer(&cmd, 0, &part, 0);
    gpu_bind_buffer(&cmd, 1, &out_flat, 0);
    gpu_copy_bytes(&cmd, 2, &df, sizeof df);
    gpu_run_kernel(&cmd, NEMO_HEADS, NB, 1, DECODE_COMBINE_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_store);
    gpu_bind_buffer(&cmd, 0, &q_raw, 0);
    gpu_bind_buffer(&cmd, 1, &k_raw, 0);
    gpu_bind_buffer(&cmd, 2, &v_raw, 0);
    gpu_bind_buffer(&cmd, 3, &q_hm, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 6, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 7, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 8, &kt_paged, 0);
    gpu_copy_bytes(&cmd, 9, &qp, sizeof qp);
    gpu_run_kernel(&cmd, 36, NB, 1, QKV_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_part);
    gpu_bind_buffer(&cmd, 0, &q_hm, 0);
    gpu_bind_buffer(&cmd, 1, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 2, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 3, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &kt_paged, 0);
    gpu_bind_buffer(&cmd, 6, &part, 0);
    gpu_copy_bytes(&cmd, 7, &dp, sizeof dp);
    gpu_run_kernel(&cmd, NEMO_KV_HEADS, NPARTS, NB, DECODE_PART_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_comb);
    gpu_bind_buffer(&cmd, 0, &part, 0);
    gpu_bind_buffer(&cmd, 1, &out_paged, 0);
    gpu_copy_bytes(&cmd, 2, &dp, sizeof dp);
    gpu_run_kernel(&cmd, NEMO_HEADS, NB, 1, DECODE_COMBINE_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(g);
  }
  float decb_diff =
      max_abs_diff((const float *)out_paged.ptr, (const float *)out_flat.ptr,
                   (size_t)NB * NEMO_Q_DIM);
  checkf(decb_diff == 0.0f,
         "paged batched FlashDecoding matches flat on CoW branches (max=%.2e)",
         decb_diff);

  // 5. Paged nemo_kv_store (layer 0, batched with KV_SKIP_POS on row 1) writes
  // into page 1 without corrupting shared page 0 or branch 1.
  uint32_t target_pos = NEMO_KV_PAGE_TOKENS + 7u;
  ((KvTarget *)kt_paged.ptr)[0] = (KvTarget){
      pt[0].gpu_addrs.address + pt[0].gpu_addrs.off, 0, 0, 0, target_pos, 0u};
  ((KvTarget *)kt_paged.ptr)[1] = (KvTarget){
      pt[1].gpu_addrs.address + pt[1].gpu_addrs.off, 0, 0, 0, KV_SKIP_POS, 0u};
  memset((uint8_t *)pt[1].pages[1]->buf.ptr, 0, slice_bytes * 4);
  {
    QkvParam qp = {NB, NB, 0u, 1u, 0, (uint32_t)slice_bytes, layout.page_shift};
    GpuCommandBuffer cmd = gpu_begin(g);
    gpu_select_kernel(&cmd, k_kv);
    gpu_bind_buffer(&cmd, 0, &k_raw, 0);
    gpu_bind_buffer(&cmd, 1, &v_raw, 0);
    gpu_bind_buffer(&cmd, 2, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 3, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 4, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 5, &pt[0].gpu_addrs, 0);
    gpu_bind_buffer(&cmd, 6, &kt_paged, 0);
    gpu_copy_bytes(&cmd, 7, &qp, sizeof qp);
    gpu_run_kernel(&cmd, 2u * NEMO_KV_HEADS, NB, 1, QKV_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(g);
  }
  int pkv_ok = 1;
  uint32_t slot = target_pos & (NEMO_KV_PAGE_TOKENS - 1u);
  const uint8_t *p0 = (const uint8_t *)pt[0].pages[1]->buf.ptr;
  const uint8_t *p1 = (const uint8_t *)pt[1].pages[1]->buf.ptr;
  for (int d = 0; d < NEMO_HEAD_DIM; d++) {
    uint16_t k0v = ((const uint16_t *)p0)[slot * NEMO_HEAD_DIM + d];
    uint16_t k1v =
        ((const uint16_t *)(p0 + slice_bytes))[slot * NEMO_HEAD_DIM + d];
    uint16_t v0v =
        ((const uint16_t *)(p0 + 2 * slice_bytes))[slot * NEMO_HEAD_DIM + d];
    uint16_t v1v =
        ((const uint16_t *)(p0 + 3 * slice_bytes))[slot * NEMO_HEAD_DIM + d];
    if (bf16_to_f32(k0v) != ((const float *)k_raw.ptr)[d] ||
        bf16_to_f32(k1v) != ((const float *)k_raw.ptr)[NEMO_HEAD_DIM + d] ||
        bf16_to_f32(v0v) != ((const float *)v_raw.ptr)[d] ||
        bf16_to_f32(v1v) != ((const float *)v_raw.ptr)[NEMO_HEAD_DIM + d])
      pkv_ok = 0;
    if (((const uint16_t *)p1)[slot * NEMO_HEAD_DIM + d] != 0) pkv_ok = 0;
  }
  checkf(pkv_ok, "paged nemo_kv_store batched+skip matches CPU");

  for (int b = 0; b < NB; b++) {
    kv_page_table_free(&pt[b]);
    free(q_all[b]), free(k_all[b]), free(v_all[b]);
    gpu_free(&fk0[b]), gpu_free(&fk1[b]), gpu_free(&fv0[b]), gpu_free(&fv1[b]);
  }
  gpu_free(&q_raw), gpu_free(&k_raw), gpu_free(&v_raw), gpu_free(&q_hm),
      gpu_free(&out_flat), gpu_free(&out_paged), gpu_free(&part),
      gpu_free(&kt_flat), gpu_free(&kt_paged);
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
  test_paged_attention(g, &work);
  test_moe_pipeline(g, &work);
  test_mtp_kernels(g, &work);
  return failures != 0;
}
