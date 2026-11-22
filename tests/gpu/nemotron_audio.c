#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
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

// Tests every Metal kernel in src/kernels/nemotron_audio.metal against an
// independent CPU reference implementation.

// Shared tile and tolerance constants matching
// src/kernels/nemotron_audio.metal.
static constexpr uint32_t AUD_TILE = 64, AUD_FP8_M_TILE = 128,
                          AUD_MM_THREADS = 128, AUD_ROW_THREADS = 256,
                          AUD_PACK_THREADS = 64, AUD_UPSAMPLE_THREADS = 256;
static constexpr uint32_t ACT_GELU = 1, ACT_RELU = 2;
static constexpr float LN_EPS = 1e-5f;
static constexpr float FP8_UNDO = 256.0f;
static constexpr double LOG2_E = 1.4426950408889634074;
static constexpr double ROPE_BASE = 10000.0;
static constexpr uint32_t BF16_SHIFT = 16, BF16_HALF = 0x7FFFu,
                          BF16_MASK = 0xFFFF0000u;
static constexpr float TOL_F32 = 1e-4f, TOL_BF16 = 3e-2f, TOL_ATTN = 2e-2f;

// Kernel parameter structs matching src/kernels/nemotron_audio.metal.
typedef struct {
  // Row width and BF16 rounding flag.
  uint32_t dim, round;
  // LayerNorm epsilon.
  float eps;
} AudLnParam;

typedef struct {
  // Active rows, padded rows and column count.
  uint32_t rows, padded_rows, cols;
} AudPackParam;

typedef struct {
  // Active rows, output width, inner width and padded weight rows.
  uint32_t m, n, k, w_rows;
  // Activation mode, bias flag, residual flag and BF16 rounding flag.
  uint32_t act, has_bias, has_res, round;
  // FP8 per-tensor scale multiplied by 256.
  float scale;
} AudMmParam;

typedef struct {
  // Rows, 64-aligned stride, heads, head dim, half rotary dim and qk_norm flag.
  uint32_t rows, seq_stride, heads, head_dim, half_rot, qk_norm;
  // Per-head LayerNorm epsilon.
  float eps;
} AudQkvParam;

typedef struct {
  // Rows, 64-aligned stride and head count.
  uint32_t rows, seq_stride, heads;
  // Attention scale multiplied by log2(e).
  float log2_scale;
} AudAttnParam;

typedef struct {
  // Encoder frames, input channels and upsampling factor.
  uint32_t frames, in_ch, up_factor;
} AudUpsampleParam;

typedef struct {
  // Encoder frames, temporal pooling factor and speaker count.
  uint32_t frames, factor, n_spk;
} AudPoolParam;

typedef struct {
  // Rows, ASR width and speaker count.
  uint32_t rows, dim, n_spk;
  // LayerNorm epsilon, threshold and infusion scale.
  float eps, threshold, scale;
} AudFuseParam;

typedef struct {
  // Input frame count and feature width (multiple of 4).
  uint32_t frames, dim;
} CtcInterleaveParam;

typedef struct {
  // Number of float4 elements and BF16 rounding flag.
  uint32_t count4, round;
  // Residual blend scale factor.
  float scale;
} CtcResidParam;

typedef struct {
  // Row count, full vocabulary size and gathered token count.
  uint32_t rows, vocab, n_gather;
} CtcLogSoftmaxParam;

// Deterministic PRNG state for reproducible test tensors.
static uint64_t rng_state = 0x5EED123456789ABCULL;

// Returns a uniform float in [-1, 1].
static float randf(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (float)((int32_t)(rng_state >> 33) - (1 << 30)) / (float)(1 << 30);
}

// Rounds v to BF16 (ties to even).
static float rbf(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Encodes v as a raw BF16 bit pattern.
static uint16_t f32_to_bf16(float v) {
  float r = rbf(v);
  uint32_t u;
  memcpy(&u, &r, sizeof u);
  return (uint16_t)(u >> BF16_SHIFT);
}

// Decodes a raw BF16 bit pattern to FP32.
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << BF16_SHIFT;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Decodes an FP8 E4M3FN byte (non-NaN) to FP32.
static float decode_e4m3(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 0x7;
  float v;
  if (exp == 0)
    v = ldexpf((float)man / 8.0f, -6);
  else
    v = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  return sign ? -v : v;
}

// Exact GELU reference using libm erff.
static float ref_gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

// Returns maximum absolute difference across n floats.
static float max_diff(const float *a, const float *b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
    if (d > m) m = d;
  }
  return m;
}

// Verifies nemo_aud_layernorm against FP32 CPU reference.
static void test_layernorm(Gpu *gpu, MemorySlot *slot) {
  static constexpr int ROWS = 13, DIM = 1280;
  Buf bx = gpu_alloc(gpu, slot, (size_t)ROWS * DIM * sizeof(float));
  Buf bw = gpu_alloc(gpu, slot, (size_t)DIM * sizeof(float));
  Buf bb = gpu_alloc(gpu, slot, (size_t)DIM * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)ROWS * DIM * sizeof(float));

  float *x = (float *)bx.ptr, *w = (float *)bw.ptr, *b = (float *)bb.ptr;
  for (int i = 0; i < ROWS * DIM; i++) x[i] = randf() * 3.0f + 1.5f;
  for (int i = 0; i < DIM; i++) {
    w[i] = randf() + 1.0f;
    b[i] = randf() * 0.5f;
  }

  AudLnParam p = {.dim = DIM, .round = 0, .eps = LN_EPS};
  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_layernorm"));
  gpu_bind_buffer(&cb, 0, &bx, 0);
  gpu_bind_buffer(&cb, 1, &bw, 0);
  gpu_bind_buffer(&cb, 2, &bb, 0);
  gpu_bind_buffer(&cb, 3, &by, 0);
  gpu_copy_bytes(&cb, 4, &p, sizeof p);
  gpu_run_kernel(&cb, ROWS, 1, 1, AUD_ROW_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref = xmalloc((size_t)ROWS * DIM * sizeof(float));
  for (int r = 0; r < ROWS; r++) {
    double mean = 0.0;
    for (int d = 0; d < DIM; d++) mean += x[r * DIM + d];
    mean /= DIM;
    double var = 0.0;
    for (int d = 0; d < DIM; d++) {
      double df = x[r * DIM + d] - mean;
      var += df * df;
    }
    double inv = 1.0 / sqrt(var / DIM + LN_EPS);
    for (int d = 0; d < DIM; d++)
      ref[r * DIM + d] = (float)(((x[r * DIM + d] - mean) * inv) * w[d] + b[d]);
  }
  float err = max_diff((const float *)by.ptr, ref, (size_t)ROWS * DIM);
  checkf(err < TOL_F32, "nemo_aud_layernorm matches CPU (max=%.2e)", err);

  free(ref);
  gpu_free(&by);
  gpu_free(&bb);
  gpu_free(&bw);
  gpu_free(&bx);
}

// Verifies nemo_aud_f32_to_bf16 + nemo_aud_mm_fp8 (with bias, GELU and
// residual add) for M spanning multiple 128-row tiles (M = 135).
static void test_mm_fp8(Gpu *gpu, MemorySlot *slot) {
  static constexpr int M = 135, N = 192, K = 256;
  int m_pad = ((M + (int)AUD_FP8_M_TILE - 1) / (int)AUD_FP8_M_TILE) *
              (int)AUD_FP8_M_TILE;
  Buf bw = gpu_alloc(gpu, slot, (size_t)N * K);
  Buf bx = gpu_alloc(gpu, slot, (size_t)M * K * sizeof(float));
  Buf bx_bf = gpu_alloc(gpu, slot, (size_t)m_pad * K * sizeof(uint16_t));
  Buf bb = gpu_alloc(gpu, slot, (size_t)N * sizeof(float));
  Buf br = gpu_alloc(gpu, slot, (size_t)M * N * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)M * N * sizeof(float));

  uint8_t *w = bw.ptr;
  float *x = (float *)bx.ptr, *b = (float *)bb.ptr, *res = (float *)br.ptr;
  for (int i = 0; i < N * K; i++) {
    uint8_t code = (uint8_t)((i * 37 + 11) & 0x77);  // avoids NaN 0x7F/0xFF
    if ((i & 3) == 1) code |= 0x80;
    w[i] = code;
  }
  for (int i = 0; i < M * K; i++) x[i] = rbf(randf() * 0.5f);
  for (int i = 0; i < N; i++) b[i] = rbf(randf() * 0.25f);
  for (int i = 0; i < M * N; i++) res[i] = rbf(randf() * 0.5f);

  float w_scale = 0.00125f;
  AudPackParam pp = {.rows = M, .padded_rows = (uint32_t)m_pad, .cols = K};
  AudMmParam mp = {.m = M,
                   .n = N,
                   .k = K,
                   .w_rows = N,
                   .act = ACT_GELU,
                   .has_bias = 1,
                   .has_res = 1,
                   .round = 1,
                   .scale = w_scale * FP8_UNDO};

  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_f32_to_bf16"));
  gpu_bind_buffer(&cb, 0, &bx, 0);
  gpu_bind_buffer(&cb, 1, &bx_bf, 0);
  gpu_copy_bytes(&cb, 2, &pp, sizeof pp);
  uint32_t gx = (K / 4 + AUD_PACK_THREADS - 1) / AUD_PACK_THREADS;
  gpu_run_kernel(&cb, gx, (uint32_t)m_pad, 1, AUD_PACK_THREADS, 1, 1);

  uint32_t tm = (uint32_t)(m_pad / (int)AUD_FP8_M_TILE);
  uint32_t tn = (uint32_t)((N + (int)AUD_TILE - 1) / (int)AUD_TILE);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_mm_fp8"));
  gpu_bind_buffer(&cb, 0, &bw, 0);
  gpu_bind_buffer(&cb, 1, &bx_bf, 0);
  gpu_bind_buffer(&cb, 2, &bb, 0);
  gpu_bind_buffer(&cb, 3, &br, 0);
  gpu_bind_buffer(&cb, 4, &by, 0);
  gpu_copy_bytes(&cb, 5, &mp, sizeof mp);
  gpu_run_kernel(&cb, tm * tn, 1, 1, AUD_MM_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref = xmalloc((size_t)M * N * sizeof(float));
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++)
        acc += (double)x[m * K + k] * (double)decode_e4m3(w[n * K + k]);
      float v = rbf((float)acc * w_scale + b[n]);
      v = rbf(ref_gelu(v));
      ref[m * N + n] = rbf(v + res[m * N + n]);
    }
  }
  float err = max_diff((const float *)by.ptr, ref, (size_t)M * N);
  checkf(err < TOL_BF16, "nemo_aud_mm_fp8 + bias + GELU + res (max=%.2e)", err);

  free(ref);
  gpu_free(&by);
  gpu_free(&br);
  gpu_free(&bb);
  gpu_free(&bx_bf);
  gpu_free(&bx);
  gpu_free(&bw);
}

// Verifies nemo_aud_mm_bf16 for narrow padded output N = 8 (w_rows = 64) with
// bias and ReLU, matching single_hidden_to_spks and first_hidden_to_hidden.
static void test_mm_bf16(Gpu *gpu, MemorySlot *slot) {
  static constexpr int M = 75, N = 8, W_ROWS = 64, K = 192;
  Buf bw = gpu_alloc(gpu, slot, (size_t)W_ROWS * K * sizeof(uint16_t));
  Buf bx = gpu_alloc(gpu, slot, (size_t)M * K * sizeof(float));
  Buf bb = gpu_alloc(gpu, slot, (size_t)N * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)M * N * sizeof(float));

  uint16_t *w = (uint16_t *)bw.ptr;
  memset(w, 0, bw.size);
  float *x = (float *)bx.ptr, *b = (float *)bb.ptr;
  for (int i = 0; i < N * K; i++) w[i] = f32_to_bf16(randf() * 0.5f);
  for (int i = 0; i < M * K; i++) x[i] = rbf(randf() * 0.5f);
  for (int i = 0; i < N; i++) b[i] = rbf(randf() * 0.2f);

  AudMmParam mp = {.m = M,
                   .n = N,
                   .k = K,
                   .w_rows = W_ROWS,
                   .act = ACT_RELU,
                   .has_bias = 1,
                   .has_res = 0,
                   .round = 1,
                   .scale = 0.0f};
  uint32_t tm = (M + AUD_TILE - 1) / AUD_TILE;
  uint32_t tn = (N + AUD_TILE - 1) / AUD_TILE;
  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_mm_bf16"));
  gpu_bind_buffer(&cb, 0, &bw, 0);
  gpu_bind_buffer(&cb, 1, &bx, 0);
  gpu_bind_buffer(&cb, 2, &bb, 0);
  gpu_bind_buffer(&cb, 3, &by, 0);
  gpu_bind_buffer(&cb, 4, &by, 0);
  gpu_copy_bytes(&cb, 5, &mp, sizeof mp);
  gpu_run_kernel(&cb, tm * tn, 1, 1, AUD_MM_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref = xmalloc((size_t)M * N * sizeof(float));
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++)
        acc += (double)x[m * K + k] * (double)bf16_to_f32(w[n * K + k]);
      float v = rbf((float)acc + b[n]);
      ref[m * N + n] = v > 0.0f ? v : 0.0f;
    }
  }
  float err = max_diff((const float *)by.ptr, ref, (size_t)M * N);
  checkf(err < TOL_BF16, "nemo_aud_mm_bf16 narrow N=8 + bias + ReLU (max=%.2e)",
         err);

  free(ref);
  gpu_free(&by);
  gpu_free(&bb);
  gpu_free(&bx);
  gpu_free(&bw);
}

// Verifies nemo_aud_qkv_prep + bidirectional nemo_aud_flash_attn_{64,128}.
static void test_qkv_and_attn(Gpu *gpu, MemorySlot *slot, int heads,
                              int head_dim, int half_rot, int qk_norm,
                              const char *label) {
  static constexpr int ROWS = 77;
  int seq_stride = ((ROWS + (int)AUD_TILE - 1) / (int)AUD_TILE) * (int)AUD_TILE;
  int d_model = heads * head_dim;

  Buf bqkv = gpu_alloc(gpu, slot, (size_t)ROWS * 3u * d_model * sizeof(float));
  Buf bqnw = gpu_alloc(gpu, slot, (size_t)head_dim * sizeof(float));
  Buf bqnb = gpu_alloc(gpu, slot, (size_t)head_dim * sizeof(float));
  Buf bknw = gpu_alloc(gpu, slot, (size_t)head_dim * sizeof(float));
  Buf bknb = gpu_alloc(gpu, slot, (size_t)head_dim * sizeof(float));
  Buf bcos = gpu_alloc(gpu, slot, (size_t)ROWS * half_rot * sizeof(float));
  Buf bsin = gpu_alloc(gpu, slot, (size_t)ROWS * half_rot * sizeof(float));
  size_t hm_bytes = (size_t)heads * seq_stride * head_dim * sizeof(uint16_t);
  Buf bq = gpu_alloc(gpu, slot, hm_bytes);
  Buf bk = gpu_alloc(gpu, slot, hm_bytes);
  Buf bv = gpu_alloc(gpu, slot, hm_bytes);
  Buf bout = gpu_alloc(gpu, slot, (size_t)ROWS * d_model * sizeof(float));

  float *qkv = (float *)bqkv.ptr;
  float *qnw = (float *)bqnw.ptr, *qnb = (float *)bqnb.ptr;
  float *knw = (float *)bknw.ptr, *knb = (float *)bknb.ptr;
  float *ctab = (float *)bcos.ptr, *stab = (float *)bsin.ptr;

  for (int i = 0; i < ROWS * 3 * d_model; i++) qkv[i] = rbf(randf() * 0.6f);
  for (int d = 0; d < head_dim; d++) {
    qnw[d] = 0.8f + 0.4f * randf();
    qnb[d] = 0.1f * randf();
    knw[d] = 0.8f + 0.4f * randf();
    knb[d] = 0.1f * randf();
  }
  int d_rot = 2 * half_rot;
  for (int r = 0; r < ROWS; r++) {
    for (int j = 0; j < half_rot; j++) {
      double inv = 1.0 / pow(ROPE_BASE, (double)(2 * j) / (double)d_rot);
      ctab[r * half_rot + j] = rbf((float)cos((double)r * inv));
      stab[r * half_rot + j] = rbf((float)sin((double)r * inv));
    }
  }

  AudQkvParam qp = {.rows = ROWS,
                    .seq_stride = (uint32_t)seq_stride,
                    .heads = (uint32_t)heads,
                    .head_dim = (uint32_t)head_dim,
                    .half_rot = (uint32_t)half_rot,
                    .qk_norm = (uint32_t)qk_norm,
                    .eps = LN_EPS};
  AudAttnParam ap = {.rows = ROWS,
                     .seq_stride = (uint32_t)seq_stride,
                     .heads = (uint32_t)heads,
                     .log2_scale = (float)(LOG2_E / sqrt((double)head_dim))};
  const char *attn_name = head_dim == 128  ? "nemo_aud_flash_attn_128"
                          : head_dim == 80 ? "nemo_aud_flash_attn_80"
                                           : "nemo_aud_flash_attn_64";
  int k_attn = gpu_find_kernel(gpu, attn_name);

  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_qkv_prep"));
  gpu_bind_buffer(&cb, 0, &bqkv, 0);
  gpu_bind_buffer(&cb, 1, &bqnw, 0);
  gpu_bind_buffer(&cb, 2, &bqnb, 0);
  gpu_bind_buffer(&cb, 3, &bknw, 0);
  gpu_bind_buffer(&cb, 4, &bknb, 0);
  gpu_bind_buffer(&cb, 5, &bcos, 0);
  gpu_bind_buffer(&cb, 6, &bsin, 0);
  gpu_bind_buffer(&cb, 7, &bq, 0);
  gpu_bind_buffer(&cb, 8, &bk, 0);
  gpu_bind_buffer(&cb, 9, &bv, 0);
  gpu_copy_bytes(&cb, 10, &qp, sizeof qp);
  gpu_run_kernel(&cb, (uint32_t)heads, (uint32_t)seq_stride, 1,
                 (uint32_t)head_dim, 1, 1);

  gpu_select_kernel(&cb, k_attn);
  gpu_bind_buffer(&cb, 0, &bq, 0);
  gpu_bind_buffer(&cb, 1, &bk, 0);
  gpu_bind_buffer(&cb, 2, &bv, 0);
  gpu_bind_buffer(&cb, 3, &bout, 0);
  gpu_copy_bytes(&cb, 4, &ap, sizeof ap);
  gpu_run_kernel(&cb, (uint32_t)(seq_stride / (int)AUD_TILE), (uint32_t)heads,
                 1, AUD_MM_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  // CPU reference for QKV prep + bidirectional scaled dot-product attention.
  float *qr = xmalloc((size_t)heads * ROWS * head_dim * sizeof(float));
  float *kr = xmalloc((size_t)heads * ROWS * head_dim * sizeof(float));
  float *vr = xmalloc((size_t)heads * ROWS * head_dim * sizeof(float));
  for (int r = 0; r < ROWS; r++) {
    for (int h = 0; h < heads; h++) {
      float qv[128], kv[128];
      for (int d = 0; d < head_dim; d++) {
        size_t base = (size_t)r * 3u * d_model + (size_t)h * head_dim + d;
        qv[d] = qkv[base];
        kv[d] = qkv[base + d_model];
        vr[(h * ROWS + r) * head_dim + d] = qkv[base + 2 * d_model];
      }
      if (qk_norm) {
        double qm = 0.0, km = 0.0;
        for (int d = 0; d < head_dim; d++) qm += qv[d], km += kv[d];
        qm /= head_dim, km /= head_dim;
        double qvar = 0.0, kvar = 0.0;
        for (int d = 0; d < head_dim; d++) {
          qvar += (qv[d] - qm) * (qv[d] - qm);
          kvar += (kv[d] - km) * (kv[d] - km);
        }
        double qinv = 1.0 / sqrt(qvar / head_dim + LN_EPS);
        double kinv = 1.0 / sqrt(kvar / head_dim + LN_EPS);
        for (int d = 0; d < head_dim; d++) {
          qv[d] = rbf((float)((qv[d] - qm) * qinv * qnw[d] + qnb[d]));
          kv[d] = rbf((float)((kv[d] - km) * kinv * knw[d] + knb[d]));
        }
      }
      for (int d = 0; d < head_dim; d++) {
        float qo = qv[d], ko = kv[d];
        if (d < half_rot) {
          float c = ctab[r * half_rot + d], s = stab[r * half_rot + d];
          qo = qv[d] * c - qv[d + half_rot] * s;
          ko = kv[d] * c - kv[d + half_rot] * s;
        } else if (d < d_rot) {
          int j = d - half_rot;
          float c = ctab[r * half_rot + j], s = stab[r * half_rot + j];
          qo = qv[d] * c + qv[j] * s;
          ko = kv[d] * c + kv[j] * s;
        }
        qr[(h * ROWS + r) * head_dim + d] = rbf(qo);
        kr[(h * ROWS + r) * head_dim + d] = rbf(ko);
      }
    }
  }

  float *ref = xmalloc((size_t)ROWS * d_model * sizeof(float));
  double scale = 1.0 / sqrt((double)head_dim);
  for (int h = 0; h < heads; h++) {
    for (int r = 0; r < ROWS; r++) {
      double scores[ROWS], mx = -INFINITY;
      for (int c = 0; c < ROWS; c++) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; d++)
          dot += (double)qr[(h * ROWS + r) * head_dim + d] *
                 (double)kr[(h * ROWS + c) * head_dim + d];
        scores[c] = dot * scale;
        if (scores[c] > mx) mx = scores[c];
      }
      double sum = 0.0;
      for (int c = 0; c < ROWS; c++) {
        scores[c] = exp(scores[c] - mx);
        sum += scores[c];
      }
      for (int d = 0; d < head_dim; d++) {
        double acc = 0.0;
        for (int c = 0; c < ROWS; c++)
          acc += (scores[c] / sum) * (double)vr[(h * ROWS + c) * head_dim + d];
        ref[r * d_model + h * head_dim + d] = rbf((float)acc);
      }
    }
  }
  float err = max_diff((const float *)bout.ptr, ref, (size_t)ROWS * d_model);
  checkf(err < TOL_ATTN, "qkv_prep + flash_attn (%s, max=%.2e)", label, err);

  free(ref);
  free(vr);
  free(kr);
  free(qr);
  gpu_free(&bout);
  gpu_free(&bv);
  gpu_free(&bk);
  gpu_free(&bq);
  gpu_free(&bsin);
  gpu_free(&bcos);
  gpu_free(&bknb);
  gpu_free(&bknw);
  gpu_free(&bqnb);
  gpu_free(&bqnw);
  gpu_free(&bqkv);
}

// Verifies nemo_aud_subpixel_upsample, nemo_aud_diar_pool and nemo_aud_fuse.
static void test_diar_head_and_fuse(Gpu *gpu, MemorySlot *slot) {
  static constexpr int FRAMES = 11, IN_CH = 192, UP = 8, SPK = 8, DIM = 1280;
  static constexpr int OUT_CH = UP * IN_CH;

  Buf bx = gpu_alloc(gpu, slot, (size_t)FRAMES * IN_CH * sizeof(float));
  Buf bw = gpu_alloc(gpu, slot, (size_t)OUT_CH * IN_CH * 3u * sizeof(uint16_t));
  Buf bb = gpu_alloc(gpu, slot, (size_t)OUT_CH * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)FRAMES * UP * IN_CH * sizeof(float));

  float *x = (float *)bx.ptr, *b = (float *)bb.ptr;
  uint16_t *w = (uint16_t *)bw.ptr;
  for (int i = 0; i < FRAMES * IN_CH; i++) x[i] = rbf(randf() * 0.5f);
  for (int i = 0; i < OUT_CH * IN_CH * 3; i++)
    w[i] = f32_to_bf16(randf() * 0.1f);
  for (int i = 0; i < OUT_CH; i++) b[i] = rbf(randf() * 0.2f);

  AudUpsampleParam up = {.frames = FRAMES, .in_ch = IN_CH, .up_factor = UP};
  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_subpixel_upsample"));
  gpu_bind_buffer(&cb, 0, &bx, 0);
  gpu_bind_buffer(&cb, 1, &bw, 0);
  gpu_bind_buffer(&cb, 2, &bb, 0);
  gpu_bind_buffer(&cb, 3, &by, 0);
  gpu_copy_bytes(&cb, 4, &up, sizeof up);
  uint32_t gx = (OUT_CH + AUD_UPSAMPLE_THREADS - 1) / AUD_UPSAMPLE_THREADS;
  gpu_run_kernel(&cb, gx, FRAMES, 1, AUD_UPSAMPLE_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref_up = xmalloc((size_t)FRAMES * UP * IN_CH * sizeof(float));
  for (int f = 0; f < FRAMES; f++) {
    for (int oc = 0; oc < OUT_CH; oc++) {
      double acc = b[oc];
      for (int ic = 0; ic < IN_CH; ic++) {
        for (int k = 0; k < 3; k++) {
          int sf = f + k - 1;
          if (sf >= 0 && sf < FRAMES)
            acc += (double)bf16_to_f32(w[(oc * IN_CH + ic) * 3 + k]) *
                   (double)x[sf * IN_CH + ic];
        }
      }
      int u = oc / IN_CH, c = oc % IN_CH;
      float v = rbf((float)acc);
      ref_up[(f * UP + u) * IN_CH + c] = v > 0.0f ? v : 0.0f;
    }
  }
  float up_err =
      max_diff((const float *)by.ptr, ref_up, (size_t)FRAMES * UP * IN_CH);
  checkf(up_err < TOL_BF16, "nemo_aud_subpixel_upsample + ReLU (max=%.2e)",
         up_err);

  // Test nemo_aud_diar_pool + nemo_aud_fuse.
  Buf blogits = gpu_alloc(gpu, slot, (size_t)FRAMES * UP * SPK * sizeof(float));
  Buf bprobs = gpu_alloc(gpu, slot, (size_t)FRAMES * SPK * sizeof(float));
  Buf basr = gpu_alloc(gpu, slot, (size_t)FRAMES * DIM * sizeof(float));
  Buf banw = gpu_alloc(gpu, slot, (size_t)DIM * sizeof(float));
  Buf banb = gpu_alloc(gpu, slot, (size_t)DIM * sizeof(float));
  Buf bdnw = gpu_alloc(gpu, slot, (size_t)SPK * sizeof(float));
  Buf bdnb = gpu_alloc(gpu, slot, (size_t)SPK * sizeof(float));
  Buf bkern = gpu_alloc(gpu, slot, (size_t)SPK * DIM * sizeof(float));
  Buf bfused = gpu_alloc(gpu, slot, (size_t)FRAMES * DIM * sizeof(float));

  float *logits = (float *)blogits.ptr;
  float *asr = (float *)basr.ptr, *anw = (float *)banw.ptr,
        *anb = (float *)banb.ptr;
  float *dnw = (float *)bdnw.ptr, *dnb = (float *)bdnb.ptr,
        *kern = (float *)bkern.ptr;
  for (int i = 0; i < FRAMES * UP * SPK; i++) logits[i] = randf() * 3.0f;
  for (int i = 0; i < FRAMES * DIM; i++) asr[i] = rbf(randf() * 2.0f);
  for (int i = 0; i < DIM; i++)
    anw[i] = 1.0f + 0.2f * randf(), anb[i] = 0.1f * randf();
  for (int i = 0; i < SPK; i++)
    dnw[i] = 1.0f + 0.2f * randf(), dnb[i] = 0.1f * randf();
  for (int i = 0; i < SPK * DIM; i++) kern[i] = rbf(randf());

  AudPoolParam pp = {.frames = FRAMES, .factor = UP, .n_spk = SPK};
  AudFuseParam fp = {.rows = FRAMES,
                     .dim = DIM,
                     .n_spk = SPK,
                     .eps = LN_EPS,
                     .threshold = 0.5f,
                     .scale = 1.0f};
  cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_diar_pool"));
  gpu_bind_buffer(&cb, 0, &blogits, 0);
  gpu_bind_buffer(&cb, 1, &bprobs, 0);
  gpu_copy_bytes(&cb, 2, &pp, sizeof pp);
  gpu_run_kernel(&cb, 1, FRAMES, 1, SPK, 1, 1);

  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_aud_fuse"));
  gpu_bind_buffer(&cb, 0, &basr, 0);
  gpu_bind_buffer(&cb, 1, &bprobs, 0);
  gpu_bind_buffer(&cb, 2, &banw, 0);
  gpu_bind_buffer(&cb, 3, &banb, 0);
  gpu_bind_buffer(&cb, 4, &bdnw, 0);
  gpu_bind_buffer(&cb, 5, &bdnb, 0);
  gpu_bind_buffer(&cb, 6, &bkern, 0);
  gpu_bind_buffer(&cb, 7, &bfused, 0);
  gpu_copy_bytes(&cb, 8, &fp, sizeof fp);
  gpu_run_kernel(&cb, FRAMES, 1, 1, AUD_ROW_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float ref_p[FRAMES * SPK];
  for (int f = 0; f < FRAMES; f++) {
    for (int s = 0; s < SPK; s++) {
      double acc = 0.0;
      for (int u = 0; u < UP; u++) {
        double v = logits[(f * UP + u) * SPK + s];
        acc += 1.0 / (1.0 + exp(-v));
      }
      ref_p[f * SPK + s] = (float)(acc / UP);
    }
  }
  float p_err = max_diff((const float *)bprobs.ptr, ref_p, FRAMES * SPK);
  checkf(p_err < TOL_F32, "nemo_aud_diar_pool matches CPU (max=%.2e)", p_err);

  float *ref_f = xmalloc((size_t)FRAMES * DIM * sizeof(float));
  for (int f = 0; f < FRAMES; f++) {
    double sf[SPK], dm = 0.0, dv = 0.0;
    for (int s = 0; s < SPK; s++) {
      sf[s] = ref_p[f * SPK + s] > 0.5f ? 1.0 : 0.0;
      dm += sf[s];
    }
    dm /= SPK;
    for (int s = 0; s < SPK; s++) dv += (sf[s] - dm) * (sf[s] - dm);
    double dinv = 1.0 / sqrt(dv / SPK + LN_EPS);
    float dn[SPK];
    for (int s = 0; s < SPK; s++)
      dn[s] = rbf((float)((sf[s] - dm) * dinv * dnw[s] + dnb[s]));

    double am = 0.0, av = 0.0;
    for (int d = 0; d < DIM; d++) am += asr[f * DIM + d];
    am /= DIM;
    for (int d = 0; d < DIM; d++)
      av += (asr[f * DIM + d] - am) * (asr[f * DIM + d] - am);
    double ainv = 1.0 / sqrt(av / DIM + LN_EPS);
    for (int d = 0; d < DIM; d++) {
      float an = rbf((float)((asr[f * DIM + d] - am) * ainv * anw[d] + anb[d]));
      float inf = 0.0f;
      for (int s = 0; s < SPK; s++) inf += dn[s] * kern[s * DIM + d];
      ref_f[f * DIM + d] = rbf(an + rbf(inf));
    }
  }
  float f_err =
      max_diff((const float *)bfused.ptr, ref_f, (size_t)FRAMES * DIM);
  checkf(f_err < TOL_BF16, "nemo_aud_fuse matches CPU (max=%.2e)", f_err);

  free(ref_f);
  free(ref_up);
  gpu_free(&bfused);
  gpu_free(&bkern);
  gpu_free(&bdnb);
  gpu_free(&bdnw);
  gpu_free(&banb);
  gpu_free(&banw);
  gpu_free(&basr);
  gpu_free(&bprobs);
  gpu_free(&blogits);
  gpu_free(&by);
  gpu_free(&bb);
  gpu_free(&bw);
  gpu_free(&bx);
}

// Verifies nemo_ctc_interleave, nemo_ctc_residual_add and
// nemo_ctc_log_softmax_gather against FP32 CPU references.
static void test_ctc_kernels(Gpu *gpu, MemorySlot *slot) {
  static constexpr int FRAMES = 11, DIM = 1280, UP = 2 * FRAMES;
  static constexpr int VOCAB = 23863, GATHER = 37;
  static constexpr float RESID_SCALE = 0.1f;
  static constexpr uint32_t RESID_THREADS = 256;

  Buf beven = gpu_alloc(gpu, slot, (size_t)FRAMES * DIM * sizeof(float));
  Buf bodd = gpu_alloc(gpu, slot, (size_t)FRAMES * DIM * sizeof(float));
  Buf bup = gpu_alloc(gpu, slot, (size_t)UP * DIM * sizeof(float));
  Buf bbridged = gpu_alloc(gpu, slot, (size_t)UP * DIM * sizeof(float));
  Buf bblend = gpu_alloc(gpu, slot, (size_t)UP * DIM * sizeof(float));

  float *even = (float *)beven.ptr, *odd = (float *)bodd.ptr,
        *bridged = (float *)bbridged.ptr;
  for (int i = 0; i < FRAMES * DIM; i++) {
    even[i] = randf() * 2.0f;
    odd[i] = randf() * 2.0f;
  }
  for (int i = 0; i < UP * DIM; i++) bridged[i] = randf() * 3.0f;

  CtcInterleaveParam ip = {.frames = FRAMES, .dim = DIM};
  uint32_t count4 = (uint32_t)((UP * DIM) / 4);
  CtcResidParam rp = {.count4 = count4, .round = 1, .scale = RESID_SCALE};

  GpuCommandBuffer cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_ctc_interleave"));
  gpu_bind_buffer(&cb, 0, &beven, 0);
  gpu_bind_buffer(&cb, 1, &bodd, 0);
  gpu_bind_buffer(&cb, 2, &bup, 0);
  gpu_copy_bytes(&cb, 3, &ip, sizeof ip);
  gpu_run_kernel(&cb,
                 (uint32_t)(DIM / 4 + AUD_PACK_THREADS - 1) / AUD_PACK_THREADS,
                 FRAMES, 1, AUD_PACK_THREADS, 1, 1);

  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_ctc_residual_add"));
  gpu_bind_buffer(&cb, 0, &bup, 0);
  gpu_bind_buffer(&cb, 1, &bbridged, 0);
  gpu_bind_buffer(&cb, 2, &bblend, 0);
  gpu_copy_bytes(&cb, 3, &rp, sizeof rp);
  gpu_run_kernel(&cb, (count4 + RESID_THREADS - 1) / RESID_THREADS, 1, 1,
                 RESID_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref_up = xmalloc((size_t)UP * DIM * sizeof(float));
  float *ref_blend = xmalloc((size_t)UP * DIM * sizeof(float));
  for (int f = 0; f < FRAMES; f++) {
    memcpy(ref_up + (size_t)(2 * f) * DIM, even + (size_t)f * DIM,
           (size_t)DIM * sizeof(float));
    memcpy(ref_up + (size_t)(2 * f + 1) * DIM, odd + (size_t)f * DIM,
           (size_t)DIM * sizeof(float));
  }
  for (int i = 0; i < UP * DIM; i++)
    ref_blend[i] = rbf(ref_up[i] + RESID_SCALE * bridged[i]);
  float up_err = max_diff((const float *)bup.ptr, ref_up, (size_t)UP * DIM);
  float bl_err =
      max_diff((const float *)bblend.ptr, ref_blend, (size_t)UP * DIM);
  checkf(up_err == 0.0f, "nemo_ctc_interleave exact match (max=%.2e)", up_err);
  checkf(bl_err == 0.0f, "nemo_ctc_residual_add exact match (max=%.2e)",
         bl_err);

  // Test nemo_ctc_log_softmax_gather with full 23863-class CTC vocabulary,
  // including ties in maximum logits to verify lowest-index argmax.
  static constexpr int LROWS = 7;
  Buf blogits = gpu_alloc(gpu, slot, (size_t)LROWS * VOCAB * sizeof(float));
  Buf bgids = gpu_alloc(gpu, slot, (size_t)GATHER * sizeof(uint32_t));
  Buf blogp = gpu_alloc(gpu, slot, (size_t)LROWS * GATHER * sizeof(float));
  Buf bargmax = gpu_alloc(gpu, slot, (size_t)LROWS * sizeof(uint32_t));

  float *logits = (float *)blogits.ptr;
  uint32_t *gids = (uint32_t *)bgids.ptr;
  for (int r = 0; r < LROWS; r++) {
    for (int v = 0; v < VOCAB; v++) logits[r * VOCAB + v] = randf() * 5.0f;
    // Plant a known tie on row 3 so lowest-index tie-breaking is exercised.
    if (r == 3) {
      logits[r * VOCAB + 1024] = 9.5f;
      logits[r * VOCAB + 20000] = 9.5f;
    }
  }
  for (int j = 0; j < GATHER - 1; j++) gids[j] = (uint32_t)((j * 617) % VOCAB);
  gids[GATHER - 1] = VOCAB - 1;

  CtcLogSoftmaxParam lp = {.rows = LROWS, .vocab = VOCAB, .n_gather = GATHER};
  cb = gpu_begin(gpu);
  gpu_select_kernel(&cb, gpu_find_kernel(gpu, "nemo_ctc_log_softmax_gather"));
  gpu_bind_buffer(&cb, 0, &blogits, 0);
  gpu_bind_buffer(&cb, 1, &bgids, 0);
  gpu_bind_buffer(&cb, 2, &blogp, 0);
  gpu_bind_buffer(&cb, 3, &bargmax, 0);
  gpu_copy_bytes(&cb, 4, &lp, sizeof lp);
  gpu_run_kernel(&cb, LROWS, 1, 1, AUD_ROW_THREADS, 1, 1);
  gpu_commit(&cb);
  gpu_sync(gpu);

  float *ref_lp = xmalloc((size_t)LROWS * GATHER * sizeof(float));
  int argmax_ok = 1;
  for (int r = 0; r < LROWS; r++) {
    const float *row = logits + (size_t)r * VOCAB;
    float mx = -INFINITY;
    uint32_t best = 0;
    for (int v = 0; v < VOCAB; v++) {
      if (row[v] > mx) {
        mx = row[v];
        best = (uint32_t)v;
      }
    }
    if (((const uint32_t *)bargmax.ptr)[r] != best) argmax_ok = 0;
    double sum = 0.0;
    for (int v = 0; v < VOCAB; v++) sum += exp((double)row[v] - (double)mx);
    double lse = (double)mx + log(sum);
    for (int j = 0; j < GATHER; j++)
      ref_lp[r * GATHER + j] = (float)((double)row[gids[j]] - lse);
  }
  float lp_err =
      max_diff((const float *)blogp.ptr, ref_lp, (size_t)LROWS * GATHER);
  checkf(argmax_ok && lp_err < TOL_F32,
         "nemo_ctc_log_softmax_gather matches CPU (argmax=%d, max=%.2e)",
         argmax_ok, lp_err);

  free(ref_lp);
  free(ref_blend);
  free(ref_up);
  gpu_free(&bargmax);
  gpu_free(&blogp);
  gpu_free(&bgids);
  gpu_free(&blogits);
  gpu_free(&bblend);
  gpu_free(&bbridged);
  gpu_free(&bup);
  gpu_free(&bodd);
  gpu_free(&beven);
}

int main(void) {
  Memory *mem = memory_new();
  MemoryAccount account;
  MemorySlot slot;
  memory_account_init(mem, &account, "nemotron_audio_test", UINT64_MAX);
  memory_slot_init(&slot, &account, MEMORY_MODEL, NULL, "test", NULL, NULL);
  slot.automatic = 1;

  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *gpu = gpu_new(src, len);

  test_layernorm(gpu, &slot);
  test_mm_fp8(gpu, &slot);
  test_mm_bf16(gpu, &slot);
  test_qkv_and_attn(gpu, &slot, 10, 128, 32, 1, "ASR 10x128 qk_norm half-RoPE");
  test_qkv_and_attn(gpu, &slot, 16, 80, 20, 1, "CTC 16x80 qk_norm half-RoPE");
  test_qkv_and_attn(gpu, &slot, 8, 64, 32, 0, "Diar 8x64 full-RoPE");
  test_diar_head_and_fuse(gpu, &slot);
  test_ctc_kernels(gpu, &slot);

  return failures != 0;
}
