#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tests MiniMax-H3's Qwen3-VL GPU kernels against CPU references: RMSNorm,
// SwiGLU, residual add, DeepStack indexed add, QK-Norm with interleaved MRoPE,
// and causal 64-head / 8-KV-head FlashAttention.

// Qwen3-VL attention dimensions matching src/kernels/minimax_h3_qwen.metal.
static constexpr int Q_HEADS = 64, KV_HEADS = 8, HEAD_DIM = 128,
                     PAIRS = HEAD_DIM / 2, SHARING = Q_HEADS / KV_HEADS,
                     Q_WIDTH = Q_HEADS * HEAD_DIM,
                     KV_WIDTH = KV_HEADS * HEAD_DIM;

// Interleaved MRoPE H/W replacement boundary (pairs 0..59).
static constexpr int MROPE_END = 60;

// Attention head stride padding and threadgroup sizes.
static constexpr uint32_t ATTN_PAD = 64, NORM_THREADS = 256, ELEM_THREADS = 256,
                          ATTN_THREADS = 128;

// RMSNorm epsilon and numerical tolerances for BF16 attention comparison.
static constexpr float RMS_EPS = 1e-6f, ATTN_TOL = 2e-2f, NORM_TOL = 1e-5f;

// Parameters passed to minimax_h3_rmsnorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} RmsNormParam;

// Parameters passed to minimax_h3_add_indexed.
typedef struct {
  // Number of indexed rows and width of each row in floats.
  uint32_t count, dim;
} AddIndexedParam;

// Parameters passed to minimax_h3_qwen_heads.
typedef struct {
  // Active token count and padded head stride.
  uint32_t rows, padded;
  // Per-head QK RMSNorm epsilon.
  float eps;
} QwenHeadsParam;

// Parameters passed to minimax_h3_qwen_attend.
typedef struct {
  // Active token count and padded head stride.
  uint32_t rows, padded;
} QwenAttendParam;

// Deterministic xorshift PRNG for test inputs.
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

// Decodes a 16-bit BF16 value to FP32.
static float bf16_to_f32(uint16_t bits) {
  uint32_t u = (uint32_t)bits << 16;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Tests RMSNorm, SwiGLU, residual add and DeepStack indexed add.
static void test_elementwise_and_norm(Gpu *gpu, MemorySlot *slot) {
  int k_norm = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  int k_swiglu = gpu_find_kernel(gpu, "minimax_h3_swiglu");
  int k_add = gpu_find_kernel(gpu, "minimax_h3_add");
  int k_add_idx = gpu_find_kernel(gpu, "minimax_h3_add_indexed");
  uint32_t rows = 5, dim = 512;
  uint32_t count = rows * dim;
  uint32_t rng = 0x2468ace1u;

  Buf x = gpu_alloc(gpu, slot, (size_t)count * 4);
  Buf w = gpu_alloc(gpu, slot, (size_t)dim * 4);
  Buf y = gpu_alloc(gpu, slot, (size_t)count * 4);
  Buf u = gpu_alloc(gpu, slot, (size_t)count * 4);
  Buf out = gpu_alloc(gpu, slot, (size_t)count * 4);
  for (uint32_t i = 0; i < count; i++) {
    ((float *)x.ptr)[i] = rng_float(&rng) * 2.0f;
    ((float *)u.ptr)[i] = rng_float(&rng) * 2.0f;
  }
  for (uint32_t i = 0; i < dim; i++)
    ((float *)w.ptr)[i] = 0.5f + 0.5f * rng_float(&rng);

  RmsNormParam np = {dim, RMS_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_norm);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &w, 0);
  gpu_bind_buffer(&cmd, 2, &y, 0);
  gpu_copy_bytes(&cmd, 3, &np, sizeof np);
  gpu_run_kernel(&cmd, rows, 1, 1, NORM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_swiglu);
  gpu_bind_buffer(&cmd, 0, &y, 0);
  gpu_bind_buffer(&cmd, 1, &u, 0);
  gpu_bind_buffer(&cmd, 2, &out, 0);
  gpu_copy_bytes(&cmd, 3, &count, sizeof count);
  gpu_run_kernel(&cmd, (count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int norm_ok = 1, swiglu_ok = 1;
  const float *xf = (const float *)x.ptr, *wf = (const float *)w.ptr;
  const float *yf = (const float *)y.ptr, *uf = (const float *)u.ptr;
  const float *of = (const float *)out.ptr;
  for (uint32_t r = 0; r < rows; r++) {
    float ss = 0.0f;
    for (uint32_t d = 0; d < dim; d++) ss += xf[r * dim + d] * xf[r * dim + d];
    float inv = 1.0f / sqrtf(ss / (float)dim + RMS_EPS);
    for (uint32_t d = 0; d < dim; d++) {
      float ref_n = wf[d] * (xf[r * dim + d] * inv);
      if (fabsf(yf[r * dim + d] - ref_n) > NORM_TOL) norm_ok = 0;
      float ref_s = ref_n / (1.0f + expf(-ref_n)) * uf[r * dim + d];
      if (fabsf(of[r * dim + d] - ref_s) > NORM_TOL) swiglu_ok = 0;
    }
  }
  check(norm_ok, "Qwen3-VL RMSNorm matches CPU reference");
  check(swiglu_ok, "Qwen3-VL SwiGLU matches CPU reference");

  // Test residual add and DeepStack indexed add.
  Buf src = gpu_alloc(gpu, slot, 2 * (size_t)dim * 4);
  Buf idx = gpu_alloc(gpu, slot, 2 * sizeof(int));
  static const int targets[2] = {1, 4};
  memcpy(idx.ptr, targets, sizeof targets);
  for (uint32_t i = 0; i < 2 * dim; i++) ((float *)src.ptr)[i] = 3.25f;
  float *x_before = xmalloc((size_t)count * 4);
  memcpy(x_before, x.ptr, (size_t)count * 4);

  AddIndexedParam dp = {2, dim};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_add);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &u, 0);
  gpu_copy_bytes(&cmd, 2, &count, sizeof count);
  gpu_run_kernel(&cmd, (count + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_add_idx);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &src, 0);
  gpu_bind_buffer(&cmd, 2, &idx, 0);
  gpu_copy_bytes(&cmd, 3, &dp, sizeof dp);
  gpu_run_kernel(&cmd, dim / 64, 2, 1, 64, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int add_ok = 1;
  for (uint32_t r = 0; r < rows; r++) {
    float bonus = (r == 1 || r == 4) ? 3.25f : 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
      float ref = x_before[r * dim + d] + uf[r * dim + d] + bonus;
      if (fabsf(((const float *)x.ptr)[r * dim + d] - ref) > NORM_TOL)
        add_ok = 0;
    }
  }
  check(add_ok, "residual add and DeepStack indexed add match CPU reference");
  free(x_before);
  gpu_free(&idx);
  gpu_free(&src);
  gpu_free(&out);
  gpu_free(&u);
  gpu_free(&y);
  gpu_free(&w);
  gpu_free(&x);
}

// Tests QK-Norm with interleaved 3D MRoPE and causal 64/8 GQA FlashAttention.
static void test_heads_and_attention(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "minimax_h3_qwen_heads");
  int k_attend = gpu_find_kernel(gpu, "minimax_h3_qwen_attend");
  // Test sequence length crossing a 64-key tile boundary.
  int seq_len = 71;
  uint32_t padded = ((uint32_t)seq_len + ATTN_PAD - 1) / ATTN_PAD * ATTN_PAD;
  uint32_t rng = 0x13579bdfu;

  Buf q = gpu_alloc(gpu, slot, (size_t)seq_len * Q_WIDTH * 4);
  Buf k = gpu_alloc(gpu, slot, (size_t)seq_len * KV_WIDTH * 4);
  Buf v = gpu_alloc(gpu, slot, (size_t)seq_len * KV_WIDTH * 4);
  Buf qn = gpu_alloc(gpu, slot, (size_t)HEAD_DIM * 4);
  Buf kn = gpu_alloc(gpu, slot, (size_t)HEAD_DIM * 4);
  Buf inv = gpu_alloc(gpu, slot, (size_t)PAIRS * 4);
  Buf pos = gpu_alloc(gpu, slot, (size_t)seq_len * 3 * sizeof(int));
  Buf qh = gpu_alloc(gpu, slot, (size_t)Q_HEADS * padded * HEAD_DIM * 2);
  Buf kh = gpu_alloc(gpu, slot, (size_t)KV_HEADS * padded * HEAD_DIM * 2);
  Buf vh = gpu_alloc(gpu, slot, (size_t)KV_HEADS * padded * HEAD_DIM * 2);
  Buf out = gpu_alloc(gpu, slot, (size_t)seq_len * Q_WIDTH * 4);

  for (int i = 0; i < seq_len * Q_WIDTH; i++)
    ((float *)q.ptr)[i] = rng_float(&rng);
  for (int i = 0; i < seq_len * KV_WIDTH; i++) {
    ((float *)k.ptr)[i] = rng_float(&rng);
    ((float *)v.ptr)[i] = rng_float(&rng);
  }
  for (int i = 0; i < HEAD_DIM; i++) {
    ((float *)qn.ptr)[i] = 0.7f + 0.3f * rng_float(&rng);
    ((float *)kn.ptr)[i] = 0.7f + 0.3f * rng_float(&rng);
  }
  for (int i = 0; i < PAIRS; i++)
    ((float *)inv.ptr)[i] =
        (float)(1.0 / pow(5000000.0, (double)(2 * i) / (double)HEAD_DIM));
  // Distinct T, H, W coordinates to verify interleaved MRoPE axis routing.
  int *pos_i = (int *)pos.ptr;
  for (int r = 0; r < seq_len; r++) {
    pos_i[3 * r] = r;
    pos_i[3 * r + 1] = r * 2 + 5;
    pos_i[3 * r + 2] = r * 3 + 11;
  }

  QwenHeadsParam hp = {(uint32_t)seq_len, padded, RMS_EPS};
  QwenAttendParam ap = {(uint32_t)seq_len, padded};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_heads);
  gpu_bind_buffer(&cmd, 0, &q, 0);
  gpu_bind_buffer(&cmd, 1, &k, 0);
  gpu_bind_buffer(&cmd, 2, &v, 0);
  gpu_bind_buffer(&cmd, 3, &qn, 0);
  gpu_bind_buffer(&cmd, 4, &kn, 0);
  gpu_bind_buffer(&cmd, 5, &inv, 0);
  gpu_bind_buffer(&cmd, 6, &pos, 0);
  gpu_bind_buffer(&cmd, 7, &qh, 0);
  gpu_bind_buffer(&cmd, 8, &kh, 0);
  gpu_bind_buffer(&cmd, 9, &vh, 0);
  gpu_copy_bytes(&cmd, 10, &hp, sizeof hp);
  gpu_run_kernel(&cmd, (uint32_t)(Q_HEADS + 2 * KV_HEADS), padded, 1,
                 (uint32_t)PAIRS, 1, 1);

  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &qh, 0);
  gpu_bind_buffer(&cmd, 1, &kh, 0);
  gpu_bind_buffer(&cmd, 2, &vh, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
  gpu_run_kernel(&cmd, ((uint32_t)seq_len + ATTN_PAD - 1) / ATTN_PAD,
                 (uint32_t)Q_HEADS, 1, ATTN_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Verify QK-Norm + interleaved MRoPE on selected heads and rows.
  int mrope_ok = 1;
  const uint16_t *qh16 = (const uint16_t *)qh.ptr;
  const uint16_t *kh16 = (const uint16_t *)kh.ptr;
  const uint16_t *vh16 = (const uint16_t *)vh.ptr;
  const float *qf = (const float *)q.ptr, *kf = (const float *)k.ptr;
  const float *qnf = (const float *)qn.ptr, *knf = (const float *)kn.ptr;
  const float *invf = (const float *)inv.ptr;
  for (int r = 0; r < seq_len; r += 17) {
    for (int h = 0; h < Q_HEADS; h += 13) {
      const float *src = qf + (size_t)r * Q_WIDTH + (size_t)h * HEAD_DIM;
      float ss = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) ss += src[d] * src[d];
      float inv_r = 1.0f / sqrtf(ss / (float)HEAD_DIM + RMS_EPS);
      const uint16_t *dst = qh16 + ((size_t)h * padded + (size_t)r) * HEAD_DIM;
      for (int t = 0; t < PAIRS; t++) {
        float a = qnf[t] * (src[t] * inv_r);
        float b = qnf[t + PAIRS] * (src[t + PAIRS] * inv_r);
        int axis = (t < MROPE_END && t % 3 == 1)   ? 1
                   : (t < MROPE_END && t % 3 == 2) ? 2
                                                   : 0;
        float angle = (float)pos_i[3 * r + axis] * invf[t];
        float ra = a * cosf(angle) - b * sinf(angle);
        float rb = b * cosf(angle) + a * sinf(angle);
        if (fabsf(bf16_to_f32(dst[t]) - ra) > 2e-2f ||
            fabsf(bf16_to_f32(dst[t + PAIRS]) - rb) > 2e-2f)
          mrope_ok = 0;
      }
    }
    for (int h = 0; h < KV_HEADS; h += 3) {
      const float *src = kf + (size_t)r * KV_WIDTH + (size_t)h * HEAD_DIM;
      float ss = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) ss += src[d] * src[d];
      float inv_r = 1.0f / sqrtf(ss / (float)HEAD_DIM + RMS_EPS);
      const uint16_t *dst = kh16 + ((size_t)h * padded + (size_t)r) * HEAD_DIM;
      for (int t = 0; t < PAIRS; t++) {
        float a = knf[t] * (src[t] * inv_r);
        float b = knf[t + PAIRS] * (src[t + PAIRS] * inv_r);
        int axis = (t < MROPE_END && t % 3 == 1)   ? 1
                   : (t < MROPE_END && t % 3 == 2) ? 2
                                                   : 0;
        float angle = (float)pos_i[3 * r + axis] * invf[t];
        float ra = a * cosf(angle) - b * sinf(angle);
        float rb = b * cosf(angle) + a * sinf(angle);
        if (fabsf(bf16_to_f32(dst[t]) - ra) > 2e-2f ||
            fabsf(bf16_to_f32(dst[t + PAIRS]) - rb) > 2e-2f)
          mrope_ok = 0;
      }
    }
  }
  // Check padded tail rows are zero.
  for (uint32_t r = (uint32_t)seq_len; r < padded; r++)
    if (qh16[r * HEAD_DIM] != 0 || kh16[r * HEAD_DIM] != 0 ||
        vh16[r * HEAD_DIM] != 0)
      mrope_ok = 0;
  check(mrope_ok,
        "Qwen3-VL QK-Norm and interleaved 3D MRoPE match CPU reference");

  // Verify causal FlashAttention against CPU reference over BF16 Q/K/V on
  // multiple heads and rows spanning both 64-key blocks.
  int attn_ok = 1;
  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  float scores[128];
  const float *outf = (const float *)out.ptr;
  static const int check_heads[4] = {0, 7, 31, 63};
  static const int check_rows[5] = {0, 1, 15, 63, 70};
  for (int hi = 0; hi < 4; hi++) {
    int h = check_heads[hi], kv = h / SHARING;
    for (int ri = 0; ri < 5; ri++) {
      int r = check_rows[ri];
      const uint16_t *qr = qh16 + ((size_t)h * padded + (size_t)r) * HEAD_DIM;
      float max_s = -INFINITY;
      for (int j = 0; j <= r; j++) {
        const uint16_t *kr =
            kh16 + ((size_t)kv * padded + (size_t)j) * HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
          dot += bf16_to_f32(qr[d]) * bf16_to_f32(kr[d]);
        scores[j] = dot * scale;
        if (scores[j] > max_s) max_s = scores[j];
      }
      float denom = 0.0f;
      for (int j = 0; j <= r; j++) {
        scores[j] = expf(scores[j] - max_s);
        denom += scores[j];
      }
      const float *got_row = outf + (size_t)r * Q_WIDTH + (size_t)h * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0f;
        for (int j = 0; j <= r; j++) {
          const uint16_t *vr =
              vh16 + ((size_t)kv * padded + (size_t)j) * HEAD_DIM;
          acc += scores[j] * bf16_to_f32(vr[d]);
        }
        float ref = acc / denom;
        if (!isfinite(got_row[d]) || fabsf(got_row[d] - ref) > ATTN_TOL)
          attn_ok = 0;
      }
    }
  }
  check(attn_ok,
        "Qwen3-VL causal 64/8 GQA FlashAttention matches CPU reference");

  gpu_free(&out);
  gpu_free(&vh);
  gpu_free(&kh);
  gpu_free(&qh);
  gpu_free(&pos);
  gpu_free(&inv);
  gpu_free(&kn);
  gpu_free(&qn);
  gpu_free(&v);
  gpu_free(&k);
  gpu_free(&q);
}

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_elementwise_and_norm(gpu, slot);
  test_heads_and_attention(gpu, slot);
  return failures != 0;
}
