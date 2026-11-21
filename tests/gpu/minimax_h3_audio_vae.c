// Tests MiniMax-H3 Audio VAE Metal kernels against independent C CPU references
// without requiring checkpoint files.
#include "base/error.h"
#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// GPU dispatch tile widths matching src/kernels/minimax_h3_audio_vae.metal.
static constexpr uint32_t SIMD_W = 32, CONV_TILE_CO = 32, CONV_THREADS = 128,
                          ACT_TILE_T = 256, ACT_K = 12, ACT_RATIO = 2,
                          ACT_UP_PAD = 5, ACT_UP_CROP = 15, ACT_DOWN_PAD_L = 5,
                          ACT_DOWN_PAD_R = 6, ATTN_HEADS = 8,
                          ATTN_HEAD_DIM = 256, ATTN_OUT_DIM = 32, ATTN_POOL = 8,
                          ATTN_QKV_PARTS = 3, ATTN_THREADS = 256,
                          ELEM_THREADS = 64, TRANS_TILE = 16;

// Numerical constants and test tolerances matching the audio VAE kernels.
static constexpr float SNAKE_EPS = 1e-9f, ATTN_SCALE = 0.0625f,
                       GELU_SQRT_2_OVER_PI = 0.7978845608028654f,
                       GELU_COEF = 0.044715f, HALF = 0.5f, FP32_TOL = 1e-4f,
                       CLAMP_LO = -1.0f, CLAMP_HI = 1.0f;

// Linear congruential generator constants for deterministic synthetic inputs.
static constexpr uint32_t LCG_MUL = 1664525u, LCG_ADD = 1013904223u,
                          LCG_SHIFT = 8, LCG_MASK = 0xffffffu;
static constexpr float LCG_SCALE = 1.0f / (float)(1u << 24);

// Parameters passed to minimax_h3_audio_conv1d.
typedef struct {
  // Channel counts, lengths, kernel width, stride, padding and dilation.
  uint32_t c_in, c_out, l_in, l_out, k, stride, pad, dilation;
  // Optional bias flag, residual flag, residual center crop and clamp flag.
  uint32_t has_bias, add_res, res_crop, clamp_out;
} AudioConv1dParam;

// Parameters passed to minimax_h3_audio_conv_transpose1d.
typedef struct {
  // Channel counts, lengths, kernel width, stride and padding.
  uint32_t c_in, c_out, l_in, l_out, k, stride, pad;
} AudioConvT1dParam;

// Parameters passed to minimax_h3_audio_snake1d and act1d_snakebeta.
typedef struct {
  // Channel count and sequence length.
  uint32_t channels, length;
} AudioChLenParam;

// Parameters passed to minimax_h3_audio_affine_ch.
typedef struct {
  // Channel count, sequence length and invert flag (0 denorm, 1 norm).
  uint32_t channels, length, invert;
} AudioAffineParam;

// Parameters passed to minimax_h3_audio_transpose.
typedef struct {
  // Input row and column counts.
  uint32_t rows, cols;
} AudioTransposeParam;

// Parameters passed to minimax_h3_audio_linear_f32.
typedef struct {
  // Row count, output width, input width, bias flag and residual flag.
  uint32_t m, n, k, has_bias, add_res;
} AudioLinearParam;

// Parameters passed to minimax_h3_audio_geglu_tanh.
typedef struct {
  // Element count.
  uint32_t n;
} AudioGeGluParam;

// Parameters passed to minimax_h3_audio_causal_attn_pool.
typedef struct {
  // Sequence length.
  uint32_t seq_len;
} AudioCausalAttnParam;

// Returns a deterministic pseudo-random float in [-1, 1].
static float rng_float(uint32_t *state) {
  *state = *state * LCG_MUL + LCG_ADD;
  uint32_t v = (*state >> LCG_SHIFT) & LCG_MASK;
  return 2.0f * (float)v * LCG_SCALE - 1.0f;
}

// Clamps integer i to [lo, hi].
static int clamp_int(int i, int lo, int hi) {
  if (i < lo) return lo;
  if (i > hi) return hi;
  return i;
}

// Returns ceil(a / b) for positive integers.
static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Tests dilated, strided, residual-cropped and clamped 1D convolutions.
static void test_conv1d(Gpu *gpu, MemorySlot *slot) {
  int k_conv = gpu_find_kernel(gpu, "minimax_h3_audio_conv1d");
  uint32_t rng = 0x10203040u;

  // Case 1: dilated conv1d with bias and center-cropped residual add.
  const uint32_t ci = 19, co = 37, li = 45, k = 7, stride = 1, dil = 3, pad = 9;
  const uint32_t lo = (li + 2u * pad - dil * (k - 1u) - 1u) / stride + 1u;
  const uint32_t crop = 2, l_res = lo + 2u * crop;

  Buf bx = gpu_alloc(gpu, slot, (size_t)ci * li * sizeof(float));
  Buf bw = gpu_alloc(gpu, slot, (size_t)co * ci * k * sizeof(float));
  Buf bb = gpu_alloc(gpu, slot, (size_t)co * sizeof(float));
  Buf br = gpu_alloc(gpu, slot, (size_t)co * l_res * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)co * lo * sizeof(float));

  float *x = (float *)bx.ptr, *w = (float *)bw.ptr, *b = (float *)bb.ptr,
        *r = (float *)br.ptr, *y = (float *)by.ptr;
  for (size_t i = 0; i < (size_t)ci * li; i++) x[i] = rng_float(&rng);
  for (size_t i = 0; i < (size_t)co * ci * k; i++)
    w[i] = rng_float(&rng) * 0.25f;
  for (uint32_t i = 0; i < co; i++) b[i] = rng_float(&rng);
  for (size_t i = 0; i < (size_t)co * l_res; i++) r[i] = rng_float(&rng);

  AudioConv1dParam p = {ci, co, li, lo, k, stride, pad, dil, 1, 1, crop, 0};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_conv);
  gpu_bind_buffer(&cmd, 0, &bx, 0);
  gpu_bind_buffer(&cmd, 1, &bw, 0);
  gpu_bind_buffer(&cmd, 2, &bb, 0);
  gpu_bind_buffer(&cmd, 3, &br, 0);
  gpu_bind_buffer(&cmd, 4, &by, 0);
  gpu_copy_bytes(&cmd, 5, &p, sizeof p);
  gpu_run_kernel(&cmd, ceil_div(lo, SIMD_W), ceil_div(co, CONV_TILE_CO), 1,
                 CONV_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float max_err = 0.0f;
  for (uint32_t o = 0; o < co; o++) {
    for (uint32_t t = 0; t < lo; t++) {
      float acc = b[o];
      for (uint32_t c = 0; c < ci; c++) {
        for (uint32_t ki = 0; ki < k; ki++) {
          int ti = (int)(t * stride + ki * dil) - (int)pad;
          if (ti >= 0 && ti < (int)li)
            acc += w[(o * ci + c) * k + ki] * x[c * li + (uint32_t)ti];
        }
      }
      acc += r[o * l_res + t + crop];
      float err = fabsf(y[o * lo + t] - acc);
      if (err > max_err) max_err = err;
    }
  }
  check(max_err < FP32_TOL, "conv1d dilated + bias + cropped residual matches");

  // Case 2: strided conv1d without bias and with [-1, 1] output clamp.
  const uint32_t ci2 = 8, co2 = 1, li2 = 50, k2 = 10, s2 = 5, pad2 = 3;
  const uint32_t lo2 = (li2 + 2u * pad2 - (k2 - 1u) - 1u) / s2 + 1u;
  AudioConv1dParam p2 = {ci2, co2, li2, lo2, k2, s2, pad2, 1, 0, 0, 0, 1};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_conv);
  gpu_bind_buffer(&cmd, 0, &bx, 0);
  gpu_bind_buffer(&cmd, 1, &bw, 0);
  gpu_bind_buffer(&cmd, 2, &bb, 0);
  gpu_bind_buffer(&cmd, 3, &br, 0);
  gpu_bind_buffer(&cmd, 4, &by, 0);
  gpu_copy_bytes(&cmd, 5, &p2, sizeof p2);
  gpu_run_kernel(&cmd, ceil_div(lo2, SIMD_W), ceil_div(co2, CONV_TILE_CO), 1,
                 CONV_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float max_err2 = 0.0f;
  for (uint32_t t = 0; t < lo2; t++) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < ci2; c++) {
      for (uint32_t ki = 0; ki < k2; ki++) {
        int ti = (int)(t * s2 + ki) - (int)pad2;
        if (ti >= 0 && ti < (int)li2)
          acc += w[c * k2 + ki] * x[c * li2 + (uint32_t)ti];
      }
    }
    if (acc < CLAMP_LO) acc = CLAMP_LO;
    if (acc > CLAMP_HI) acc = CLAMP_HI;
    float err = fabsf(y[t] - acc);
    if (err > max_err2) max_err2 = err;
  }
  check(max_err2 < FP32_TOL, "conv1d strided + no-bias + clamp matches");

  gpu_free(&bx);
  gpu_free(&bw);
  gpu_free(&bb);
  gpu_free(&br);
  gpu_free(&by);
}

// Tests 1D transposed convolution for both BigVGAN geometries (s=5/k=9,
// s=2/k=4).
static void test_conv_transpose1d(Gpu *gpu, MemorySlot *slot) {
  int k_ct = gpu_find_kernel(gpu, "minimax_h3_audio_conv_transpose1d");
  uint32_t rng = 0x20304050u;
  static const uint32_t strides[] = {5, 2};
  static const uint32_t ks[] = {9, 4};
  static const uint32_t pads[] = {2, 1};

  int all_ok = 1;
  for (size_t c_idx = 0; c_idx < 2; c_idx++) {
    uint32_t ci = 23, co = 37, li = 39;
    uint32_t s = strides[c_idx], k = ks[c_idx], pad = pads[c_idx];
    uint32_t lo = (li - 1u) * s - 2u * pad + k;

    Buf bx = gpu_alloc(gpu, slot, (size_t)ci * li * sizeof(float));
    Buf bw = gpu_alloc(gpu, slot, (size_t)ci * co * k * sizeof(float));
    Buf bb = gpu_alloc(gpu, slot, (size_t)co * sizeof(float));
    Buf by = gpu_alloc(gpu, slot, (size_t)co * lo * sizeof(float));

    float *x = (float *)bx.ptr, *w = (float *)bw.ptr, *b = (float *)bb.ptr,
          *y = (float *)by.ptr;
    for (size_t i = 0; i < (size_t)ci * li; i++) x[i] = rng_float(&rng);
    for (size_t i = 0; i < (size_t)ci * co * k; i++)
      w[i] = rng_float(&rng) * 0.25f;
    for (uint32_t i = 0; i < co; i++) b[i] = rng_float(&rng);

    AudioConvT1dParam p = {ci, co, li, lo, k, s, pad};
    uint32_t t_slots = ceil_div(lo, s);
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_ct);
    gpu_bind_buffer(&cmd, 0, &bx, 0);
    gpu_bind_buffer(&cmd, 1, &bw, 0);
    gpu_bind_buffer(&cmd, 2, &bb, 0);
    gpu_bind_buffer(&cmd, 3, &by, 0);
    gpu_copy_bytes(&cmd, 4, &p, sizeof p);
    gpu_run_kernel(&cmd, ceil_div(t_slots, SIMD_W), ceil_div(co, CONV_TILE_CO),
                   s, CONV_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    // Independent scatter-form CPU reference matching PyTorch conv_transpose1d.
    float *ref = calloc((size_t)co * lo, sizeof(float));
    for (uint32_t o = 0; o < co; o++)
      for (uint32_t t = 0; t < lo; t++) ref[o * lo + t] = b[o];
    for (uint32_t c = 0; c < ci; c++) {
      for (uint32_t ti = 0; ti < li; ti++) {
        float xv = x[c * li + ti];
        for (uint32_t o = 0; o < co; o++) {
          for (uint32_t ki = 0; ki < k; ki++) {
            int to = (int)(ti * s + ki) - (int)pad;
            if (to >= 0 && to < (int)lo)
              ref[o * lo + (uint32_t)to] += xv * w[(c * co + o) * k + ki];
          }
        }
      }
    }
    for (size_t i = 0; i < (size_t)co * lo; i++) {
      if (fabsf(y[i] - ref[i]) > FP32_TOL) all_ok = 0;
    }
    free(ref);
    gpu_free(&bx);
    gpu_free(&bw);
    gpu_free(&bb);
    gpu_free(&by);
  }
  check(all_ok, "conv_transpose1d s=5/k=9 and s=2/k=4 match scatter CPU ref");
}

// Tests Snake1d and fused anti-aliased Activation1d(SnakeBeta).
static void test_activations(Gpu *gpu, MemorySlot *slot) {
  int k_snake = gpu_find_kernel(gpu, "minimax_h3_audio_snake1d");
  int k_act = gpu_find_kernel(gpu, "minimax_h3_audio_act1d_snakebeta");
  uint32_t rng = 0x30405060u;

  // Test across tile boundary (301 > 256) and length-1 edge case.
  static const uint32_t lengths[] = {301, 1};
  const uint32_t ch = 9;
  int snake_ok = 1, act_ok = 1;

  for (size_t li = 0; li < 2; li++) {
    uint32_t len = lengths[li];
    Buf bx = gpu_alloc(gpu, slot, (size_t)ch * len * sizeof(float));
    Buf ba = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
    Buf bb = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
    Buf bfu = gpu_alloc(gpu, slot, (size_t)ACT_K * sizeof(float));
    Buf bfd = gpu_alloc(gpu, slot, (size_t)ACT_K * sizeof(float));
    Buf by = gpu_alloc(gpu, slot, (size_t)ch * len * sizeof(float));

    float *x = (float *)bx.ptr, *a = (float *)ba.ptr, *b = (float *)bb.ptr,
          *fu = (float *)bfu.ptr, *fd = (float *)bfd.ptr, *y = (float *)by.ptr;
    for (size_t i = 0; i < (size_t)ch * len; i++) x[i] = rng_float(&rng);
    for (uint32_t c = 0; c < ch; c++) {
      a[c] = rng_float(&rng) * 0.7f;
      b[c] = rng_float(&rng) * 0.7f;
    }
    // Deliberately non-symmetric filters to catch any tap-reversal bugs.
    for (uint32_t k = 0; k < ACT_K; k++) {
      fu[k] = 0.05f + 0.02f * (float)k + 0.03f * rng_float(&rng);
      fd[k] = 0.04f + 0.015f * (float)k + 0.03f * rng_float(&rng);
    }

    AudioChLenParam sp = {ch, len};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_snake);
    gpu_bind_buffer(&cmd, 0, &bx, 0);
    gpu_bind_buffer(&cmd, 1, &ba, 0);
    gpu_bind_buffer(&cmd, 2, &by, 0);
    gpu_copy_bytes(&cmd, 3, &sp, sizeof sp);
    gpu_run_kernel(&cmd, ceil_div(len, ELEM_THREADS), ch, 1, ELEM_THREADS, 1,
                   1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    for (uint32_t c = 0; c < ch; c++) {
      for (uint32_t t = 0; t < len; t++) {
        float v = x[c * len + t];
        float s = sinf(a[c] * v);
        float ref = v + (s * s) / (a[c] + SNAKE_EPS);
        if (fabsf(y[c * len + t] - ref) > FP32_TOL) snake_ok = 0;
      }
    }

    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_act);
    gpu_bind_buffer(&cmd, 0, &bx, 0);
    gpu_bind_buffer(&cmd, 1, &ba, 0);
    gpu_bind_buffer(&cmd, 2, &bb, 0);
    gpu_bind_buffer(&cmd, 3, &bfu, 0);
    gpu_bind_buffer(&cmd, 4, &bfd, 0);
    gpu_bind_buffer(&cmd, 5, &by, 0);
    gpu_copy_bytes(&cmd, 6, &sp, sizeof sp);
    gpu_run_kernel(&cmd, ceil_div(len, ACT_TILE_T), ch, 1, ACT_TILE_T, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    // Literal 3-step CPU reference matching audio_vae.py UpSample1d ->
    // SnakeBeta
    // -> DownSample1d.
    uint32_t pad_len = len + 2u * ACT_UP_PAD;
    uint32_t ct_len = (pad_len - 1u) * ACT_RATIO + ACT_K;
    uint32_t up_len = len * ACT_RATIO;
    uint32_t dp_len = up_len + ACT_DOWN_PAD_L + ACT_DOWN_PAD_R;
    float *xp = calloc(pad_len, sizeof(float));
    float *ct = calloc(ct_len, sizeof(float));
    float *up = calloc(up_len, sizeof(float));
    float *dp = calloc(dp_len, sizeof(float));
    for (uint32_t c = 0; c < ch; c++) {
      for (uint32_t i = 0; i < pad_len; i++)
        xp[i] = x[c * len + (uint32_t)clamp_int((int)i - (int)ACT_UP_PAD, 0,
                                                (int)len - 1)];
      memset(ct, 0, ct_len * sizeof(float));
      for (uint32_t p = 0; p < pad_len; p++)
        for (uint32_t k = 0; k < ACT_K; k++)
          ct[p * ACT_RATIO + k] += xp[p] * fu[k] * (float)ACT_RATIO;
      float ea = expf(a[c]), eb = expf(b[c]);
      for (uint32_t j = 0; j < up_len; j++) {
        float v = ct[j + ACT_UP_CROP];
        float sv = sinf(ea * v);
        up[j] = v + (sv * sv) / (eb + SNAKE_EPS);
      }
      for (uint32_t i = 0; i < dp_len; i++)
        dp[i] = up[clamp_int((int)i - (int)ACT_DOWN_PAD_L, 0, (int)up_len - 1)];
      for (uint32_t t = 0; t < len; t++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < ACT_K; k++)
          acc += dp[t * ACT_RATIO + k] * fd[k];
        if (fabsf(y[c * len + t] - acc) > FP32_TOL) act_ok = 0;
      }
    }
    free(xp);
    free(ct);
    free(up);
    free(dp);
    gpu_free(&bx);
    gpu_free(&ba);
    gpu_free(&bb);
    gpu_free(&bfu);
    gpu_free(&bfd);
    gpu_free(&by);
  }
  check(snake_ok, "snake1d matches CPU reference");
  check(act_ok, "fused act1d_snakebeta matches literal 3-step CPU reference");
}

// Tests helper kernels (avg3, affine_ch, transpose, linear_f32, geglu_tanh) and
// AttnProjection causal attention + head mean + average pool.
static void test_helpers_and_causal_attn(Gpu *gpu, MemorySlot *slot) {
  int k_avg3 = gpu_find_kernel(gpu, "minimax_h3_audio_avg3");
  int k_aff = gpu_find_kernel(gpu, "minimax_h3_audio_affine_ch");
  int k_tr = gpu_find_kernel(gpu, "minimax_h3_audio_transpose");
  int k_lin = gpu_find_kernel(gpu, "minimax_h3_audio_linear_f32");
  int k_geglu = gpu_find_kernel(gpu, "minimax_h3_audio_geglu_tanh");
  int k_attn = gpu_find_kernel(gpu, "minimax_h3_audio_causal_attn_pool");
  uint32_t rng = 0x40506070u;

  // Elementwise avg3, affine_ch, transpose, linear_f32, geglu_tanh.
  const uint32_t rows = 13, cols = 32, out_n = 17, total = rows * cols;
  Buf ba = gpu_alloc(gpu, slot, (size_t)total * sizeof(float));
  Buf bb = gpu_alloc(gpu, slot, (size_t)total * sizeof(float));
  Buf bc = gpu_alloc(gpu, slot, (size_t)total * sizeof(float));
  Buf bo = gpu_alloc(gpu, slot, (size_t)total * sizeof(float));
  Buf bm = gpu_alloc(gpu, slot, (size_t)rows * sizeof(float));
  Buf bs = gpu_alloc(gpu, slot, (size_t)rows * sizeof(float));
  Buf bw = gpu_alloc(gpu, slot, (size_t)out_n * cols * sizeof(float));
  Buf bbias = gpu_alloc(gpu, slot, (size_t)out_n * sizeof(float));
  Buf bres = gpu_alloc(gpu, slot, (size_t)rows * out_n * sizeof(float));
  Buf blin = gpu_alloc(gpu, slot, (size_t)rows * out_n * sizeof(float));

  float *a = (float *)ba.ptr, *b = (float *)bb.ptr, *c = (float *)bc.ptr,
        *o = (float *)bo.ptr;
  float *m = (float *)bm.ptr, *s = (float *)bs.ptr, *w = (float *)bw.ptr,
        *bias = (float *)bbias.ptr, *res = (float *)bres.ptr,
        *lin = (float *)blin.ptr;
  for (uint32_t i = 0; i < total; i++) {
    a[i] = rng_float(&rng);
    b[i] = rng_float(&rng);
    c[i] = rng_float(&rng);
  }
  for (uint32_t r = 0; r < rows; r++) {
    m[r] = rng_float(&rng);
    s[r] = 0.5f + fabsf(rng_float(&rng));
  }
  for (uint32_t i = 0; i < out_n * cols; i++) w[i] = rng_float(&rng) * 0.5f;
  for (uint32_t i = 0; i < out_n; i++) bias[i] = rng_float(&rng);
  for (uint32_t i = 0; i < rows * out_n; i++) res[i] = rng_float(&rng);

  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_avg3);
  gpu_bind_buffer(&cmd, 0, &ba, 0);
  gpu_bind_buffer(&cmd, 1, &bb, 0);
  gpu_bind_buffer(&cmd, 2, &bc, 0);
  gpu_bind_buffer(&cmd, 3, &bo, 0);
  gpu_copy_bytes(&cmd, 4, &total, sizeof total);
  gpu_run_kernel(&cmd, ceil_div(total, ELEM_THREADS), 1, 1, ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  int avg_ok = 1;
  for (uint32_t i = 0; i < total; i++)
    if (fabsf(o[i] - (a[i] + b[i] + c[i]) / 3.0f) > FP32_TOL) avg_ok = 0;
  check(avg_ok, "avg3 matches CPU reference");

  AudioAffineParam ap_f = {rows, cols, 0}, ap_i = {rows, cols, 1};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_aff);
  gpu_bind_buffer(&cmd, 0, &ba, 0);
  gpu_bind_buffer(&cmd, 1, &bm, 0);
  gpu_bind_buffer(&cmd, 2, &bs, 0);
  gpu_bind_buffer(&cmd, 3, &bo, 0);
  gpu_copy_bytes(&cmd, 4, &ap_f, sizeof ap_f);
  gpu_run_kernel(&cmd, ceil_div(cols, ELEM_THREADS), rows, 1, ELEM_THREADS, 1,
                 1);
  gpu_select_kernel(&cmd, k_aff);
  gpu_bind_buffer(&cmd, 0, &bo, 0);
  gpu_bind_buffer(&cmd, 1, &bm, 0);
  gpu_bind_buffer(&cmd, 2, &bs, 0);
  gpu_bind_buffer(&cmd, 3, &bc, 0);
  gpu_copy_bytes(&cmd, 4, &ap_i, sizeof ap_i);
  gpu_run_kernel(&cmd, ceil_div(cols, ELEM_THREADS), rows, 1, ELEM_THREADS, 1,
                 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  int aff_ok = 1;
  for (uint32_t i = 0; i < total; i++)
    if (fabsf(c[i] - a[i]) > FP32_TOL) aff_ok = 0;
  check(aff_ok, "affine_ch denormalize/normalize round-trip matches");

  AudioTransposeParam tp = {rows, cols};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_tr);
  gpu_bind_buffer(&cmd, 0, &ba, 0);
  gpu_bind_buffer(&cmd, 1, &bo, 0);
  gpu_copy_bytes(&cmd, 2, &tp, sizeof tp);
  gpu_run_kernel(&cmd, ceil_div(cols, TRANS_TILE), ceil_div(rows, TRANS_TILE),
                 1, TRANS_TILE, TRANS_TILE, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  int tr_ok = 1;
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t cl = 0; cl < cols; cl++)
      if (o[cl * rows + r] != a[r * cols + cl]) tr_ok = 0;
  check(tr_ok, "transpose matches CPU reference");

  AudioLinearParam lp = {rows, out_n, cols, 1, 1};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_lin);
  gpu_bind_buffer(&cmd, 0, &ba, 0);
  gpu_bind_buffer(&cmd, 1, &bw, 0);
  gpu_bind_buffer(&cmd, 2, &bbias, 0);
  gpu_bind_buffer(&cmd, 3, &bres, 0);
  gpu_bind_buffer(&cmd, 4, &blin, 0);
  gpu_copy_bytes(&cmd, 5, &lp, sizeof lp);
  gpu_run_kernel(&cmd, out_n, rows, 1, SIMD_W, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  int lin_ok = 1;
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t cl = 0; cl < out_n; cl++) {
      float acc = bias[cl] + res[r * out_n + cl];
      for (uint32_t ki = 0; ki < cols; ki++)
        acc += a[r * cols + ki] * w[cl * cols + ki];
      if (fabsf(lin[r * out_n + cl] - acc) > FP32_TOL) lin_ok = 0;
    }
  }
  check(lin_ok, "linear_f32 with bias and residual matches");

  AudioGeGluParam gp = {total};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_geglu);
  gpu_bind_buffer(&cmd, 0, &ba, 0);
  gpu_bind_buffer(&cmd, 1, &bb, 0);
  gpu_bind_buffer(&cmd, 2, &bo, 0);
  gpu_copy_bytes(&cmd, 3, &gp, sizeof gp);
  gpu_run_kernel(&cmd, ceil_div(total, ELEM_THREADS), 1, 1, ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  int geglu_ok = 1;
  for (uint32_t i = 0; i < total; i++) {
    float g = a[i];
    float inner = GELU_SQRT_2_OVER_PI * (g + GELU_COEF * g * g * g);
    float ref = HALF * g * (1.0f + tanhf(inner)) * b[i];
    if (fabsf(o[i] - ref) > FP32_TOL) geglu_ok = 0;
  }
  check(geglu_ok, "geglu_tanh matches CPU reference");

  // Causal 8-head attention + head mean + 8-way average pool.
  const uint32_t seq = 17;
  const uint32_t row_w = ATTN_QKV_PARTS * ATTN_HEADS * ATTN_HEAD_DIM;
  Buf bqkv = gpu_alloc(gpu, slot, (size_t)seq * row_w * sizeof(float));
  Buf bout = gpu_alloc(gpu, slot, (size_t)seq * ATTN_OUT_DIM * sizeof(float));
  float *qkv = (float *)bqkv.ptr, *aout = (float *)bout.ptr;
  for (size_t i = 0; i < (size_t)seq * row_w; i++)
    qkv[i] = rng_float(&rng) * 0.5f;

  AudioCausalAttnParam cap = {seq};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_attn);
  gpu_bind_buffer(&cmd, 0, &bqkv, 0);
  gpu_bind_buffer(&cmd, 1, &bout, 0);
  gpu_copy_bytes(&cmd, 2, &cap, sizeof cap);
  gpu_run_kernel(&cmd, seq, 1, 1, ATTN_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int attn_ok = 1;
  const uint32_t kv_step = ATTN_HEADS * ATTN_HEAD_DIM;
  for (uint32_t t = 0; t < seq; t++) {
    float mean_d[ATTN_HEAD_DIM] = {0};
    for (uint32_t h = 0; h < ATTN_HEADS; h++) {
      float scores[seq];
      float mx = -INFINITY;
      for (uint32_t s_i = 0; s_i <= t; s_i++) {
        float dot = 0.0f;
        const float *qp = qkv + t * row_w + h * ATTN_HEAD_DIM;
        const float *kp = qkv + s_i * row_w + kv_step + h * ATTN_HEAD_DIM;
        for (uint32_t d = 0; d < ATTN_HEAD_DIM; d++) dot += qp[d] * kp[d];
        scores[s_i] = dot * ATTN_SCALE;
        if (scores[s_i] > mx) mx = scores[s_i];
      }
      float lsum = 0.0f;
      for (uint32_t s_i = 0; s_i <= t; s_i++) {
        scores[s_i] = expf(scores[s_i] - mx);
        lsum += scores[s_i];
      }
      for (uint32_t d = 0; d < ATTN_HEAD_DIM; d++) {
        float val = 0.0f;
        for (uint32_t s_i = 0; s_i <= t; s_i++)
          val += scores[s_i] *
                 qkv[s_i * row_w + 2u * kv_step + h * ATTN_HEAD_DIM + d];
        mean_d[d] += (val / lsum) / (float)ATTN_HEADS;
      }
    }
    for (uint32_t o_c = 0; o_c < ATTN_OUT_DIM; o_c++) {
      float pool = 0.0f;
      for (uint32_t r = 0; r < ATTN_POOL; r++)
        pool += mean_d[o_c * ATTN_POOL + r];
      pool /= (float)ATTN_POOL;
      if (fabsf(aout[t * ATTN_OUT_DIM + o_c] - pool) > FP32_TOL) attn_ok = 0;
    }
  }
  check(attn_ok, "causal_attn_pool matches CPU reference");

  gpu_free(&bqkv);
  gpu_free(&bout);
  gpu_free(&ba);
  gpu_free(&bb);
  gpu_free(&bc);
  gpu_free(&bo);
  gpu_free(&bm);
  gpu_free(&bs);
  gpu_free(&bw);
  gpu_free(&bbias);
  gpu_free(&bres);
  gpu_free(&blin);
}

// Compiles the Metal library and runs all MiniMax-H3 Audio VAE kernel tests.
int main(void) {
  Memory *mem = memory_new();
  MemorySlot *slot = memory_shared(mem);
  size_t src_len = 0;
  const char *src = kernel_library_source(&src_len);
  Gpu *gpu = gpu_new(src, src_len);
  check(gpu != NULL, "gpu opened");

  test_conv1d(gpu, slot);
  test_conv_transpose1d(gpu, slot);
  test_activations(gpu, slot);
  test_helpers_and_causal_attn(gpu, slot);

  return failures != 0;
}
