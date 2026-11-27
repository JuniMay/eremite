#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/inkling2/mtp.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GPU kernel tests for src/kernels/inkling2_mtp.metal against CPU references:
// 1. ink2_mtp_fuse_norm (contiguous, single-state f_rnn ring, batched ring)
// 2. ink2_mtp_ctx_kv_store (linear global, SWA rolling ring, batched targets)
// 3. ink2_mtp_block_init (anchor RMSNorm + 7 mask rows)
// 4. ink2_mtp_block_qk_prep (per-head RMSNorm + NeoX split-half RoPE)
// 5. ink2_mtp_attend_part + ink2_mtp_attend_combine (global + SWA multi-part)
// 6. ink2_mtp_swiglu
// 7. ink2_mtp_markov_concat + ink2_mtp_markov_gru + ink2_mtp_markov_add

static constexpr float TEST_EPS = 1e-6f;
static constexpr float TEST_ROPE_THETA = 1000000.0f;
static constexpr float TOL_F32 = 2e-5f;
static constexpr float TOL_BF16 = 2e-2f;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr uint32_t ROW_THREADS = 256;
static constexpr uint32_t HEAD_THREADS = 64;
static constexpr uint32_t ATTEND_THREADS = 128;
static constexpr uint32_t COMBINE_THREADS = 32;
static constexpr uint32_t ELEM_THREADS = 256;

// Matching parameter structs for src/kernels/inkling2_mtp.metal.
typedef struct {
  uint64_t ring;
  uint32_t pos, pad;
} FrnnTarget;

typedef struct {
  uint64_t k, v;
  uint32_t pos, ring_rows;
} KvTarget;

typedef struct {
  float w0, w1, w2, w3;
  uint32_t rows, tap_stride, dim, pos0, frnn_ring, batched;
  float eps;
} FuseParam;

typedef struct {
  uint32_t rows, pos0, ring_rows, batched;
  float rope_theta, eps;
  uint32_t page_shift, k_slice_off, v_slice_off;
} CtxKvParam;

typedef struct {
  uint32_t anchor, dim;
  float eps;
} BlockInitParam;

typedef struct {
  uint32_t pos0;
  float rope_theta, eps;
} BlockQkParam;

typedef struct {
  uint32_t pos0, block, ring_rows, window, nparts, part_stride, page_shift,
      k_slice_off, v_slice_off;
} AttendParam;

typedef struct {
  uint32_t step_idx, pos0, frnn_ring, dim, rank;
} MarkovParam;

// Converts FP32 to BF16 (round-to-nearest-even).
static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  uint32_t lsb = (u >> BF16_SHIFT) & 1u;
  u += 0x7fffu + lsb;
  return (uint16_t)(u >> BF16_SHIFT);
}

// Converts BF16 bit pattern to FP32.
static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << BF16_SHIFT;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// Decodes an FP8 E4M3FN byte to FP32.
static float e4m3_to_f32(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int mant = b & 0x7;
  float v = exp == 0 ? ldexpf((float)mant / 8.0f, -6)
                     : ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
  return sign ? -v : v;
}

// Decodes an IEEE 754 binary16 bit pattern to FP32.
static float fp16_to_f32(uint16_t h) {
  int sign = (h >> 15) & 1;
  int exp = (h >> 10) & 0x1F;
  int mant = h & 0x3FF;
  float v;
  if (exp == 0)
    v = ldexpf((float)mant, -24);
  else if (exp == 31)
    v = mant ? NAN : INFINITY;
  else
    v = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
  return sign ? -v : v;
}

// Encodes an FP32 value to a finite IEEE 754 binary16 bit pattern.
static uint16_t f32_to_fp16(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
  int exp = (int)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) return sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
  uint32_t m = mant + 0x1000u + ((mant >> 13) & 1u);
  if (m & 0x800000u) m = 0, exp++;
  if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
  return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)(m >> 13));
}

// Encodes an FP32 value to finite FP8 E4M3 round-to-nearest-even.
static uint8_t f32_to_e4m3(float v) {
  uint8_t sign = signbit(v) ? 0x80u : 0u;
  float a = fabsf(v);
  if (a > 448.0f) a = 448.0f;
  if (a < 0.015625f) return (uint8_t)(sign | (uint8_t)lrintf(a * 512.0f));
  int e = 0;
  (void)frexpf(a, &e);
  int exp = e - 1;
  uint32_t q = (uint32_t)lrintf(ldexpf(a, 3 - exp));
  uint32_t code = ((uint32_t)(exp + 7) << 3) + (q - 8u);
  if (code > 0x7Eu) code = 0x7Eu;
  return (uint8_t)(sign | code);
}

// Packs one KV head of 128 floats into a packed FP8 E4M3 row.
static void fp8_kv_pack_head(uint8_t *base, int slot, int kh,
                             const float *src) {
  uint8_t *row = base + (size_t)slot * INK2_FP8_KV_ROW_BYTES;
  float amax = 0.0f;
  for (int d = 0; d < INK2_HEAD_DIM; d++) amax = fmaxf(amax, fabsf(src[d]));
  uint16_t hscale = 0;
  float inv = 0.0f;
  if (amax > 0.0f) {
    float s256 = amax * (256.0f / 448.0f);
    if (s256 < 6.103515625e-5f) s256 = 6.103515625e-5f;
    if (s256 > 65504.0f) s256 = 65504.0f;
    hscale = f32_to_fp16(s256);
    inv = 256.0f / fp16_to_f32(hscale);
  }
  memcpy(row + INK2_FP8_KV_SCALES_OFF + (size_t)kh * 2, &hscale, sizeof hscale);
  for (int d = 0; d < INK2_HEAD_DIM; d++)
    row[(size_t)kh * INK2_HEAD_DIM + d] =
        amax > 0.0f ? f32_to_e4m3(src[d] * inv) : 0u;
}

// Decodes one element from a packed FP8 E4M3 row.
static float fp8_kv_get(const uint8_t *base, int slot, int kh, int d) {
  const uint8_t *row = base + (size_t)slot * INK2_FP8_KV_ROW_BYTES;
  uint16_t hscale;
  memcpy(&hscale, row + INK2_FP8_KV_SCALES_OFF + (size_t)kh * 2, sizeof hscale);
  float scale = fp16_to_f32(hscale) / 256.0f;
  return e4m3_to_f32(row[(size_t)kh * INK2_HEAD_DIM + d]) * scale;
}

// Deterministic pseudo-random float in [-1, 1] from seed state.
static float rand_f32(uint64_t *st) {
  *st = *st * 6364136223846793005ull + 1442695040888963407ull;
  int32_t v = (int32_t)((*st >> 32) & 0xffffu) - 32768;
  return (float)v / 32768.0f;
}

// Allocates a resident GPU buffer in slot.
static Buf make_buf(Gpu *gpu, MemorySlot *slot, size_t bytes) {
  Buf b = gpu_alloc(gpu, slot, bytes);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  return b;
}

// Applies CPU per-head RMSNorm + NeoX split-half RoPE to a 128-wide head.
static void cpu_head_norm_rope(const float *in, const float *w, int pos,
                               float *out) {
  double s2 = 0.0;
  for (int d = 0; d < INK2_HEAD_DIM; d++) s2 += (double)in[d] * in[d];
  float inv = 1.0f / sqrtf((float)(s2 / INK2_HEAD_DIM) + TEST_EPS);
  int half = INK2_HEAD_DIM / 2;
  for (int d = 0; d < half; d++) {
    float x0 = in[d] * inv * w[d];
    float x1 = in[d + half] * inv * w[d + half];
    float freq = powf(TEST_ROPE_THETA, -(float)(2 * d) / (float)INK2_HEAD_DIM);
    float ang = (float)pos * freq;
    float cs = cosf(ang), sn = sinf(ang);
    out[d] = x0 * cs - x1 * sn;
    out[d + half] = x0 * sn + x1 * cs;
  }
}

// Verifies ink2_mtp_fuse_norm in contiguous, ring and batched modes.
static void test_fuse_norm(Gpu *gpu, MemorySlot *slot) {
  int k_fuse = gpu_find_kernel(gpu, "ink2_mtp_fuse_norm");
  int rows = 3, stride = 5, dim = INK2_DIM, ring = 8, pos0 = 6;
  Buf taps = make_buf(gpu, slot, (size_t)4 * stride * dim * sizeof(float));
  Buf nw = make_buf(gpu, slot, (size_t)dim * sizeof(float));
  Buf out = make_buf(gpu, slot, (size_t)ring * dim * sizeof(float));
  Buf out_b0 = make_buf(gpu, slot, (size_t)ring * dim * sizeof(float));
  Buf out_b1 = make_buf(gpu, slot, (size_t)ring * dim * sizeof(float));
  Buf tgt = make_buf(gpu, slot, (size_t)rows * sizeof(FrnnTarget));

  uint64_t st = 101;
  float *tp = (float *)taps.ptr, *wp = (float *)nw.ptr;
  for (int i = 0; i < 4 * stride * dim; i++) tp[i] = rand_f32(&st);
  for (int i = 0; i < dim; i++) wp[i] = 0.5f + 0.5f * rand_f32(&st);
  float w[4] = {0.15f, 0.25f, 0.35f, 0.25f};

  // Mode 1: contiguous output.
  FuseParam fp = {w[0],
                  w[1],
                  w[2],
                  w[3],
                  (uint32_t)rows,
                  (uint32_t)stride,
                  (uint32_t)dim,
                  (uint32_t)pos0,
                  0u,
                  0u,
                  TEST_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_fuse);
  gpu_bind_buffer(&cmd, 0, &taps, 0);
  gpu_bind_buffer(&cmd, 1, &nw, 0);
  gpu_bind_buffer(&cmd, 2, &out, 0);
  gpu_bind_buffer(&cmd, 3, &tgt, 0);
  gpu_copy_bytes(&cmd, 4, &fp, sizeof fp);
  gpu_run_kernel(&cmd, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst = 0.0f;
  float *ref = xmalloc((size_t)rows * dim * sizeof(float));
  for (int r = 0; r < rows; r++) {
    double s2 = 0.0;
    for (int d = 0; d < dim; d++) {
      float v = 0.0f;
      for (int t = 0; t < 4; t++)
        v += w[t] * tp[((size_t)t * stride + r) * dim + d];
      ref[(size_t)r * dim + d] = v;
      s2 += (double)v * v;
    }
    float inv = 1.0f / sqrtf((float)(s2 / dim) + TEST_EPS);
    for (int d = 0; d < dim; d++) {
      ref[(size_t)r * dim + d] = ref[(size_t)r * dim + d] * inv * wp[d];
      float err = fabsf(((float *)out.ptr)[(size_t)r * dim + d] -
                        ref[(size_t)r * dim + d]);
      if (err > worst) worst = err;
    }
  }
  printf("ink2_mtp_fuse_norm contiguous max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_fuse_norm contiguous matches CPU");

  // Mode 2: single-state ring write wrapping around ring capacity.
  memset(out.ptr, 0, (size_t)ring * dim * sizeof(float));
  fp.frnn_ring = (uint32_t)ring;
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_fuse);
  gpu_bind_buffer(&cmd, 0, &taps, 0);
  gpu_bind_buffer(&cmd, 1, &nw, 0);
  gpu_bind_buffer(&cmd, 2, &out, 0);
  gpu_bind_buffer(&cmd, 3, &tgt, 0);
  gpu_copy_bytes(&cmd, 4, &fp, sizeof fp);
  gpu_run_kernel(&cmd, (uint32_t)rows, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  worst = 0.0f;
  for (int r = 0; r < rows; r++) {
    int slot_idx = (pos0 + r) % ring;
    for (int d = 0; d < dim; d++) {
      float err = fabsf(((float *)out.ptr)[(size_t)slot_idx * dim + d] -
                        ref[(size_t)r * dim + d]);
      if (err > worst) worst = err;
    }
  }
  printf("ink2_mtp_fuse_norm ring max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_fuse_norm ring wrap matches CPU");

  // Mode 3: batched multi-state ring write.
  FrnnTarget *ft = (FrnnTarget *)tgt.ptr;
  ft[0] = (FrnnTarget){out_b0.address + out_b0.off, 3u, 0u};
  ft[1] = (FrnnTarget){out_b1.address + out_b1.off, 11u, 0u};
  fp.rows = 2u;
  fp.batched = 1u;
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_fuse);
  gpu_bind_buffer(&cmd, 0, &taps, 0);
  gpu_bind_buffer(&cmd, 1, &nw, 0);
  gpu_bind_buffer(&cmd, 2, &out, 0);
  gpu_bind_buffer(&cmd, 3, &tgt, 0);
  gpu_copy_bytes(&cmd, 4, &fp, sizeof fp);
  gpu_run_kernel(&cmd, 2u, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  worst = 0.0f;
  for (int d = 0; d < dim; d++) {
    float e0 = fabsf(((float *)out_b0.ptr)[3 * dim + d] - ref[0 * dim + d]);
    float e1 =
        fabsf(((float *)out_b1.ptr)[(11 % ring) * dim + d] - ref[1 * dim + d]);
    if (e0 > worst) worst = e0;
    if (e1 > worst) worst = e1;
  }
  printf("ink2_mtp_fuse_norm batched max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_fuse_norm batched matches CPU");

  free(ref);
  gpu_free(&taps);
  gpu_free(&nw);
  gpu_free(&out);
  gpu_free(&out_b0);
  gpu_free(&out_b1);
  gpu_free(&tgt);
}

// Verifies ink2_mtp_ctx_kv_store in ring and batched modes.
static void test_ctx_kv_store(Gpu *gpu, MemorySlot *slot) {
  int k_store = gpu_find_kernel(gpu, "ink2_mtp_ctx_kv_store");
  int rows = 3, ring = 8, pos0 = 6;
  Buf kr = make_buf(gpu, slot, (size_t)rows * INK2_KV_DIM * sizeof(float));
  Buf vr = make_buf(gpu, slot, (size_t)rows * INK2_KV_DIM * sizeof(float));
  Buf kn = make_buf(gpu, slot, (size_t)INK2_HEAD_DIM * sizeof(float));
  Buf kd = make_buf(gpu, slot, (size_t)ring * INK2_KV_DIM * sizeof(uint16_t));
  Buf vd = make_buf(gpu, slot, (size_t)ring * INK2_KV_DIM * sizeof(uint16_t));
  Buf tgt = make_buf(gpu, slot, (size_t)rows * sizeof(KvTarget));

  uint64_t st = 202;
  float *kp = (float *)kr.ptr, *vp = (float *)vr.ptr, *knp = (float *)kn.ptr;
  for (int i = 0; i < rows * INK2_KV_DIM; i++) {
    kp[i] = rand_f32(&st);
    vp[i] = rand_f32(&st);
  }
  for (int i = 0; i < INK2_HEAD_DIM; i++) knp[i] = 0.7f + 0.3f * rand_f32(&st);

  CtxKvParam p = {(uint32_t)rows,
                  (uint32_t)pos0,
                  (uint32_t)ring,
                  0u,
                  TEST_ROPE_THETA,
                  TEST_EPS,
                  0u,
                  0u,
                  0u};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &kd, 0);
  gpu_bind_buffer(&cmd, 4, &vd, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &p, sizeof p);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)rows, 1, HEAD_THREADS,
                 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst = 0.0f;
  float ref[INK2_HEAD_DIM];
  const uint16_t *kout = (const uint16_t *)kd.ptr;
  const uint16_t *vout = (const uint16_t *)vd.ptr;
  for (int r = 0; r < rows; r++) {
    int pos = pos0 + r;
    int sidx = pos % ring;
    for (int kh = 0; kh < INK2_KV_HEADS; kh++) {
      cpu_head_norm_rope(kp + (size_t)r * INK2_KV_DIM + kh * INK2_HEAD_DIM, knp,
                         pos, ref);
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        size_t oidx = (size_t)sidx * INK2_KV_DIM + kh * INK2_HEAD_DIM + d;
        float ek = fabsf(bf16_to_f32(kout[oidx]) - ref[d]);
        float ev = fabsf(bf16_to_f32(vout[oidx]) -
                         vp[(size_t)r * INK2_KV_DIM + kh * INK2_HEAD_DIM + d]);
        if (ek > worst) worst = ek;
        if (ev > worst) worst = ev;
      }
    }
  }
  printf("ink2_mtp_ctx_kv_store ring max err: %.6f\n", worst);
  check(worst < TOL_BF16, "ink2_mtp_ctx_kv_store matches CPU");

  gpu_free(&kr);
  gpu_free(&vr);
  gpu_free(&kn);
  gpu_free(&kd);
  gpu_free(&vd);
  gpu_free(&tgt);
}

// Verifies ink2_mtp_ctx_kv_store_fp8 in ring and batched modes, including
// wide-range unnormalized V magnitudes (0, 1e-3, 100) and NaN-free encoding.
static void test_ctx_kv_store_fp8(Gpu *gpu, MemorySlot *slot) {
  int k_store = gpu_find_kernel(gpu, "ink2_mtp_ctx_kv_store_fp8");
  int rows = 3, ring = 8, pos0 = 6;
  Buf kr = make_buf(gpu, slot, (size_t)rows * INK2_KV_DIM * sizeof(float));
  Buf vr = make_buf(gpu, slot, (size_t)rows * INK2_KV_DIM * sizeof(float));
  Buf kn = make_buf(gpu, slot, (size_t)INK2_HEAD_DIM * sizeof(float));
  Buf kd = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf vd = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf kd_b0 = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf vd_b0 = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf kd_b1 = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf vd_b1 = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf tgt = make_buf(gpu, slot, (size_t)rows * sizeof(KvTarget));

  uint64_t st = 209;
  float *kp = (float *)kr.ptr, *vp = (float *)vr.ptr, *knp = (float *)kn.ptr;
  for (int i = 0; i < rows * INK2_KV_DIM; i++) {
    kp[i] = rand_f32(&st);
    vp[i] = rand_f32(&st);
  }
  // Exercise unnormalized V scales: head 0 = 0, head 1 ~ 1e-3, head 2 ~ 100.
  for (int d = 0; d < INK2_HEAD_DIM; d++) {
    vp[0 * INK2_HEAD_DIM + d] = 0.0f;
    vp[1 * INK2_HEAD_DIM + d] *= 1e-3f;
    vp[2 * INK2_HEAD_DIM + d] *= 100.0f;
  }
  for (int i = 0; i < INK2_HEAD_DIM; i++) knp[i] = 0.7f + 0.3f * rand_f32(&st);

  CtxKvParam p = {(uint32_t)rows,
                  (uint32_t)pos0,
                  (uint32_t)ring,
                  0u,
                  TEST_ROPE_THETA,
                  TEST_EPS,
                  0u,
                  0u,
                  0u};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &kd, 0);
  gpu_bind_buffer(&cmd, 4, &vd, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &p, sizeof p);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)rows, 1, HEAD_THREADS,
                 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int no_nan = 1, zero_ok = 1;
  float worst_rel = 0.0f;
  float ref[INK2_HEAD_DIM];
  const uint8_t *kout = (const uint8_t *)kd.ptr;
  const uint8_t *vout = (const uint8_t *)vd.ptr;
  for (int r = 0; r < rows; r++) {
    int pos = pos0 + r;
    int sidx = pos % ring;
    const uint8_t *krow = kout + (size_t)sidx * INK2_FP8_KV_ROW_BYTES;
    const uint8_t *vrow = vout + (size_t)sidx * INK2_FP8_KV_ROW_BYTES;
    for (int i = 0; i < INK2_KV_DIM; i++)
      if ((krow[i] & 0x7Fu) == 0x7Fu || (vrow[i] & 0x7Fu) == 0x7Fu) no_nan = 0;
    for (int kh = 0; kh < INK2_KV_HEADS; kh++) {
      cpu_head_norm_rope(kp + (size_t)r * INK2_KV_DIM + kh * INK2_HEAD_DIM, knp,
                         pos, ref);
      const float *vrh = vp + (size_t)r * INK2_KV_DIM + kh * INK2_HEAD_DIM;
      float kamax = 0.0f, vamax = 0.0f;
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        kamax = fmaxf(kamax, fabsf(ref[d]));
        vamax = fmaxf(vamax, fabsf(vrh[d]));
      }
      if (r == 0 && kh == 0) {
        uint16_t vs16;
        memcpy(&vs16, vrow + INK2_FP8_KV_SCALES_OFF, sizeof vs16);
        if (vs16 != 0) zero_ok = 0;
        for (int d = 0; d < INK2_HEAD_DIM; d++)
          if (vrow[d] != 0) zero_ok = 0;
      }
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        float ek =
            fabsf(fp8_kv_get(kout, sidx, kh, d) - ref[d]) / fmaxf(kamax, 1e-6f);
        if (ek > worst_rel) worst_rel = ek;
        if (vamax > 0.0f) {
          float ev = fabsf(fp8_kv_get(vout, sidx, kh, d) - vrh[d]) / vamax;
          if (ev > worst_rel) worst_rel = ev;
        }
      }
    }
  }
  printf("ink2_mtp_ctx_kv_store_fp8 ring max rel err: %.6f\n", worst_rel);
  check(no_nan, "ink2_mtp_ctx_kv_store_fp8 emits no FP8 NaN codes");
  check(zero_ok, "ink2_mtp_ctx_kv_store_fp8 encodes all-zero V head cleanly");
  check(worst_rel < 0.07f,
        "ink2_mtp_ctx_kv_store_fp8 matches CPU across scale range");

  // Batched mode writing row 0 to b0 at pos=pos0 (ring) and row 1 to b1 at
  // pos=pos0+1 (ring): must match single-state bytes exactly.
  KvTarget *kt = (KvTarget *)tgt.ptr;
  kt[0] = (KvTarget){kd_b0.address + kd_b0.off, vd_b0.address + vd_b0.off,
                     (uint32_t)pos0, (uint32_t)ring};
  kt[1] = (KvTarget){kd_b1.address + kd_b1.off, vd_b1.address + vd_b1.off,
                     (uint32_t)(pos0 + 1), (uint32_t)ring};
  p.rows = 2u;
  p.batched = 1u;
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &kd, 0);
  gpu_bind_buffer(&cmd, 4, &vd, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &p, sizeof p);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, 2u, 1, HEAD_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int s0 = pos0 % ring, s1 = (pos0 + 1) % ring;
  check(!memcmp((const uint8_t *)kd_b0.ptr + (size_t)s0 * INK2_FP8_KV_ROW_BYTES,
                kout + (size_t)s0 * INK2_FP8_KV_ROW_BYTES,
                INK2_FP8_KV_ROW_BYTES) &&
            !memcmp((const uint8_t *)vd_b0.ptr +
                        (size_t)s0 * INK2_FP8_KV_ROW_BYTES,
                    vout + (size_t)s0 * INK2_FP8_KV_ROW_BYTES,
                    INK2_FP8_KV_ROW_BYTES) &&
            !memcmp((const uint8_t *)kd_b1.ptr +
                        (size_t)s1 * INK2_FP8_KV_ROW_BYTES,
                    kout + (size_t)s1 * INK2_FP8_KV_ROW_BYTES,
                    INK2_FP8_KV_ROW_BYTES) &&
            !memcmp((const uint8_t *)vd_b1.ptr +
                        (size_t)s1 * INK2_FP8_KV_ROW_BYTES,
                    vout + (size_t)s1 * INK2_FP8_KV_ROW_BYTES,
                    INK2_FP8_KV_ROW_BYTES),
        "ink2_mtp_ctx_kv_store_fp8 batched matches single-state bytes");

  gpu_free(&kr);
  gpu_free(&vr);
  gpu_free(&kn);
  gpu_free(&kd);
  gpu_free(&vd);
  gpu_free(&kd_b0);
  gpu_free(&vd_b0);
  gpu_free(&kd_b1);
  gpu_free(&vd_b1);
  gpu_free(&tgt);
}

// Verifies ink2_mtp_block_init and ink2_mtp_block_qk_prep.
static void test_block_init_and_qk(Gpu *gpu, MemorySlot *slot) {
  int k_init = gpu_find_kernel(gpu, "ink2_mtp_block_init");
  int k_qk = gpu_find_kernel(gpu, "ink2_mtp_block_qk_prep");
  int vocab = 16, anchor = 7, pos0 = 42;
  Buf emb = make_buf(gpu, slot, (size_t)vocab * INK2_DIM * sizeof(uint16_t));
  Buf en = make_buf(gpu, slot, (size_t)INK2_DIM * sizeof(float));
  Buf mask = make_buf(gpu, slot, (size_t)INK2_DIM * sizeof(float));
  Buf cur = make_buf(gpu, slot,
                     (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));

  uint64_t st = 303;
  uint16_t *ep = (uint16_t *)emb.ptr;
  float *enp = (float *)en.ptr, *mp = (float *)mask.ptr;
  for (int i = 0; i < vocab * INK2_DIM; i++) ep[i] = f32_to_bf16(rand_f32(&st));
  for (int i = 0; i < INK2_DIM; i++) {
    enp[i] = 0.5f + 0.5f * rand_f32(&st);
    mp[i] = rand_f32(&st);
  }

  BlockInitParam bip = {(uint32_t)anchor, (uint32_t)INK2_DIM, TEST_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_init);
  gpu_bind_buffer(&cmd, 0, &emb, 0);
  gpu_bind_buffer(&cmd, 1, &en, 0);
  gpu_bind_buffer(&cmd, 2, &mask, 0);
  gpu_bind_buffer(&cmd, 3, &cur, 0);
  gpu_copy_bytes(&cmd, 4, &bip, sizeof bip);
  gpu_run_kernel(&cmd, (uint32_t)INK2_MTP_BLOCK_SIZE, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  double s2 = 0.0;
  for (int d = 0; d < INK2_DIM; d++) {
    float v = bf16_to_f32(ep[(size_t)anchor * INK2_DIM + d]);
    s2 += (double)v * v;
  }
  float inv = 1.0f / sqrtf((float)(s2 / INK2_DIM) + TEST_EPS);
  float worst = 0.0f;
  const float *cp = (const float *)cur.ptr;
  for (int d = 0; d < INK2_DIM; d++) {
    float ref = bf16_to_f32(ep[(size_t)anchor * INK2_DIM + d]) * inv * enp[d];
    float e = fabsf(cp[d] - ref);
    if (e > worst) worst = e;
    for (int r = 1; r < INK2_MTP_BLOCK_SIZE; r++) {
      float em = fabsf(cp[(size_t)r * INK2_DIM + d] - mp[d]);
      if (em > worst) worst = em;
    }
  }
  printf("ink2_mtp_block_init max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_block_init matches CPU");

  // Block Q/K prep check.
  Buf qr = make_buf(gpu, slot,
                    (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(float));
  Buf kr = make_buf(gpu, slot,
                    (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(float));
  Buf vr = make_buf(gpu, slot,
                    (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(float));
  Buf qn = make_buf(gpu, slot, (size_t)INK2_HEAD_DIM * sizeof(float));
  Buf kn = make_buf(gpu, slot, (size_t)INK2_HEAD_DIM * sizeof(float));
  Buf qb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(uint16_t));
  Buf kb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf vb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    ((float *)qr.ptr)[i] = rand_f32(&st);
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_KV_DIM; i++) {
    ((float *)kr.ptr)[i] = rand_f32(&st);
    ((float *)vr.ptr)[i] = rand_f32(&st);
  }
  for (int i = 0; i < INK2_HEAD_DIM; i++) {
    ((float *)qn.ptr)[i] = 0.6f + 0.4f * rand_f32(&st);
    ((float *)kn.ptr)[i] = 0.6f + 0.4f * rand_f32(&st);
  }

  BlockQkParam qkp = {(uint32_t)pos0, TEST_ROPE_THETA, TEST_EPS};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_qk);
  gpu_bind_buffer(&cmd, 0, &qr, 0);
  gpu_bind_buffer(&cmd, 1, &kr, 0);
  gpu_bind_buffer(&cmd, 2, &vr, 0);
  gpu_bind_buffer(&cmd, 3, &qn, 0);
  gpu_bind_buffer(&cmd, 4, &kn, 0);
  gpu_bind_buffer(&cmd, 5, &qb, 0);
  gpu_bind_buffer(&cmd, 6, &kb, 0);
  gpu_bind_buffer(&cmd, 7, &vb, 0);
  gpu_copy_bytes(&cmd, 8, &qkp, sizeof qkp);
  gpu_run_kernel(&cmd, (uint32_t)(INK2_HEADS + INK2_KV_HEADS),
                 (uint32_t)INK2_MTP_BLOCK_SIZE, 1, HEAD_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  worst = 0.0f;
  float ref[INK2_HEAD_DIM];
  for (int r = 0; r < INK2_MTP_BLOCK_SIZE; r++) {
    for (int h = 0; h < INK2_HEADS; h++) {
      cpu_head_norm_rope((float *)qr.ptr + (size_t)r * INK2_Q_DIM +
                             h * INK2_HEAD_DIM,
                         (float *)qn.ptr, pos0 + r, ref);
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        float got = bf16_to_f32(
            ((uint16_t *)
                 qb.ptr)[(size_t)r * INK2_Q_DIM + h * INK2_HEAD_DIM + d]);
        float e = fabsf(got - ref[d]);
        if (e > worst) worst = e;
      }
    }
    for (int kh = 0; kh < INK2_KV_HEADS; kh++) {
      cpu_head_norm_rope((float *)kr.ptr + (size_t)r * INK2_KV_DIM +
                             kh * INK2_HEAD_DIM,
                         (float *)kn.ptr, pos0 + r, ref);
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        size_t idx = (size_t)r * INK2_KV_DIM + kh * INK2_HEAD_DIM + d;
        float ek = fabsf(bf16_to_f32(((uint16_t *)kb.ptr)[idx]) - ref[d]);
        float ev = fabsf(bf16_to_f32(((uint16_t *)vb.ptr)[idx]) -
                         ((float *)vr.ptr)[idx]);
        if (ek > worst) worst = ek;
        if (ev > worst) worst = ev;
      }
    }
  }
  printf("ink2_mtp_block_qk_prep max err: %.6f\n", worst);
  check(worst < TOL_BF16, "ink2_mtp_block_qk_prep matches CPU");

  gpu_free(&emb);
  gpu_free(&en);
  gpu_free(&mask);
  gpu_free(&cur);
  gpu_free(&qr);
  gpu_free(&kr);
  gpu_free(&vr);
  gpu_free(&qn);
  gpu_free(&kn);
  gpu_free(&qb);
  gpu_free(&kb);
  gpu_free(&vb);
}

// Verifies ink2_mtp_attend_part + ink2_mtp_attend_combine for SWA wrap +
// multi-part FlashDecoding noncausal attention.
static void test_attend(Gpu *gpu, MemorySlot *slot) {
  int k_part = gpu_find_kernel(gpu, "ink2_mtp_attend_part");
  int k_comb = gpu_find_kernel(gpu, "ink2_mtp_attend_combine");
  int ring = 40, window = 32, pos0 = 50, nparts = 3, part_stride = 4;
  int ctx_lo = pos0 - window;
  int ctx_len = window;
  int total_k = ctx_len + INK2_MTP_BLOCK_SIZE;

  Buf qb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(uint16_t));
  Buf kc = make_buf(gpu, slot, (size_t)ring * INK2_KV_DIM * sizeof(uint16_t));
  Buf vc = make_buf(gpu, slot, (size_t)ring * INK2_KV_DIM * sizeof(uint16_t));
  Buf kb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf vb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf part = make_buf(gpu, slot,
                      (size_t)INK2_MTP_BLOCK_SIZE * INK2_HEADS * part_stride *
                          (INK2_HEAD_DIM + 2) * sizeof(float));
  Buf out = make_buf(gpu, slot,
                     (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(float));

  uint64_t st = 404;
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    ((uint16_t *)qb.ptr)[i] = f32_to_bf16(rand_f32(&st));
  for (int i = 0; i < ring * INK2_KV_DIM; i++) {
    ((uint16_t *)kc.ptr)[i] = f32_to_bf16(rand_f32(&st));
    ((uint16_t *)vc.ptr)[i] = f32_to_bf16(rand_f32(&st));
  }
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_KV_DIM; i++) {
    ((uint16_t *)kb.ptr)[i] = f32_to_bf16(rand_f32(&st));
    ((uint16_t *)vb.ptr)[i] = f32_to_bf16(rand_f32(&st));
  }

  AttendParam ap = {(uint32_t)pos0,
                    (uint32_t)INK2_MTP_BLOCK_SIZE,
                    (uint32_t)ring,
                    (uint32_t)window,
                    (uint32_t)nparts,
                    (uint32_t)part_stride,
                    0u,
                    0u,
                    0u};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &kc, 0);
  gpu_bind_buffer(&cmd, 2, &vc, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out, 0);
  gpu_copy_bytes(&cmd, 2, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float scale = 1.0f / sqrtf((float)INK2_HEAD_DIM);
  float worst = 0.0f;
  float *scores = xmalloc((size_t)total_k * sizeof(float));
  for (int r = 0; r < INK2_MTP_BLOCK_SIZE; r++) {
    for (int h = 0; h < INK2_HEADS; h++) {
      int kh = h / INK2_GQA;
      const uint16_t *qr =
          (const uint16_t *)qb.ptr + (size_t)r * INK2_Q_DIM + h * INK2_HEAD_DIM;
      float mx = -INFINITY;
      for (int ki = 0; ki < total_k; ki++) {
        const uint16_t *kr =
            ki < ctx_len
                ? (const uint16_t *)kc.ptr +
                      (size_t)((ctx_lo + ki) % ring) * INK2_KV_DIM +
                      kh * INK2_HEAD_DIM
                : (const uint16_t *)kb.ptr +
                      (size_t)(ki - ctx_len) * INK2_KV_DIM + kh * INK2_HEAD_DIM;
        double dot = 0.0;
        for (int d = 0; d < INK2_HEAD_DIM; d++)
          dot += (double)bf16_to_f32(qr[d]) * bf16_to_f32(kr[d]);
        scores[ki] = (float)dot * scale;
        if (scores[ki] > mx) mx = scores[ki];
      }
      double lsum = 0.0;
      for (int ki = 0; ki < total_k; ki++) {
        scores[ki] = expf(scores[ki] - mx);
        lsum += scores[ki];
      }
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        double acc = 0.0;
        for (int ki = 0; ki < total_k; ki++) {
          const uint16_t *vr =
              ki < ctx_len ? (const uint16_t *)vc.ptr +
                                 (size_t)((ctx_lo + ki) % ring) * INK2_KV_DIM +
                                 kh * INK2_HEAD_DIM
                           : (const uint16_t *)vb.ptr +
                                 (size_t)(ki - ctx_len) * INK2_KV_DIM +
                                 kh * INK2_HEAD_DIM;
          acc += scores[ki] * bf16_to_f32(vr[d]);
        }
        float ref = (float)(acc / lsum);
        float got =
            ((float *)out.ptr)[(size_t)r * INK2_Q_DIM + h * INK2_HEAD_DIM + d];
        float e = fabsf(got - ref);
        if (e > worst) worst = e;
      }
    }
  }
  printf("ink2_mtp_attend SWA multi-part max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_attend matches CPU noncausal reference");

  free(scores);
  gpu_free(&qb);
  gpu_free(&kc);
  gpu_free(&vc);
  gpu_free(&kb);
  gpu_free(&vb);
  gpu_free(&part);
  gpu_free(&out);
}

// Verifies ink2_mtp_attend_part_fp8 + ink2_mtp_attend_combine for SWA wrap +
// multi-part FlashDecoding over FP8 context KV + BF16 draft block KV.
static void test_attend_fp8(Gpu *gpu, MemorySlot *slot) {
  int k_part = gpu_find_kernel(gpu, "ink2_mtp_attend_part_fp8");
  int k_comb = gpu_find_kernel(gpu, "ink2_mtp_attend_combine");
  int ring = 40, window = 32, pos0 = 50, nparts = 3, part_stride = 4;
  int ctx_lo = pos0 - window;
  int ctx_len = window;
  int total_k = ctx_len + INK2_MTP_BLOCK_SIZE;

  Buf qb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(uint16_t));
  Buf kc = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf vc = make_buf(gpu, slot, (size_t)ring * INK2_FP8_KV_ROW_BYTES);
  Buf kb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf vb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf part = make_buf(gpu, slot,
                      (size_t)INK2_MTP_BLOCK_SIZE * INK2_HEADS * part_stride *
                          (INK2_HEAD_DIM + 2) * sizeof(float));
  Buf out = make_buf(gpu, slot,
                     (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(float));

  uint64_t st = 409;
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    ((uint16_t *)qb.ptr)[i] = f32_to_bf16(rand_f32(&st));
  for (int sl = 0; sl < ring; sl++) {
    for (int kh = 0; kh < INK2_KV_HEADS; kh++) {
      float khv[INK2_HEAD_DIM], vhv[INK2_HEAD_DIM];
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        khv[d] = rand_f32(&st);
        vhv[d] = rand_f32(&st) * 3.0f;
      }
      fp8_kv_pack_head((uint8_t *)kc.ptr, sl, kh, khv);
      fp8_kv_pack_head((uint8_t *)vc.ptr, sl, kh, vhv);
    }
  }
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_KV_DIM; i++) {
    ((uint16_t *)kb.ptr)[i] = f32_to_bf16(rand_f32(&st));
    ((uint16_t *)vb.ptr)[i] = f32_to_bf16(rand_f32(&st) * 3.0f);
  }

  AttendParam ap = {(uint32_t)pos0,
                    (uint32_t)INK2_MTP_BLOCK_SIZE,
                    (uint32_t)ring,
                    (uint32_t)window,
                    (uint32_t)nparts,
                    (uint32_t)part_stride,
                    0u,
                    0u,
                    0u};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &kc, 0);
  gpu_bind_buffer(&cmd, 2, &vc, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out, 0);
  gpu_copy_bytes(&cmd, 2, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float scale = 1.0f / sqrtf((float)INK2_HEAD_DIM);
  float worst = 0.0f;
  float *scores = xmalloc((size_t)total_k * sizeof(float));
  const uint8_t *kcp = (const uint8_t *)kc.ptr, *vcp = (const uint8_t *)vc.ptr;
  for (int r = 0; r < INK2_MTP_BLOCK_SIZE; r++) {
    for (int h = 0; h < INK2_HEADS; h++) {
      int kh = h / INK2_GQA;
      const uint16_t *qr =
          (const uint16_t *)qb.ptr + (size_t)r * INK2_Q_DIM + h * INK2_HEAD_DIM;
      float mx = -INFINITY;
      for (int ki = 0; ki < total_k; ki++) {
        double dot = 0.0;
        if (ki < ctx_len) {
          int sl = (ctx_lo + ki) % ring;
          for (int d = 0; d < INK2_HEAD_DIM; d++)
            dot += (double)bf16_to_f32(qr[d]) * fp8_kv_get(kcp, sl, kh, d);
        } else {
          const uint16_t *kr = (const uint16_t *)kb.ptr +
                               (size_t)(ki - ctx_len) * INK2_KV_DIM +
                               kh * INK2_HEAD_DIM;
          for (int d = 0; d < INK2_HEAD_DIM; d++)
            dot += (double)bf16_to_f32(qr[d]) * bf16_to_f32(kr[d]);
        }
        scores[ki] = (float)dot * scale;
        if (scores[ki] > mx) mx = scores[ki];
      }
      double lsum = 0.0;
      for (int ki = 0; ki < total_k; ki++) {
        scores[ki] = expf(scores[ki] - mx);
        lsum += scores[ki];
      }
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        double acc = 0.0;
        for (int ki = 0; ki < total_k; ki++) {
          float vv =
              ki < ctx_len
                  ? fp8_kv_get(vcp, (ctx_lo + ki) % ring, kh, d)
                  : bf16_to_f32(
                        ((const uint16_t *)
                             vb.ptr)[(size_t)(ki - ctx_len) * INK2_KV_DIM +
                                     kh * INK2_HEAD_DIM + d]);
          acc += scores[ki] * vv;
        }
        float ref = (float)(acc / lsum);
        float got =
            ((float *)out.ptr)[(size_t)r * INK2_Q_DIM + h * INK2_HEAD_DIM + d];
        float e = fabsf(got - ref);
        if (e > worst) worst = e;
      }
    }
  }
  printf("ink2_mtp_attend_fp8 SWA multi-part max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_attend_fp8 matches CPU noncausal reference");

  free(scores);
  gpu_free(&qb);
  gpu_free(&kc);
  gpu_free(&vc);
  gpu_free(&kb);
  gpu_free(&vb);
  gpu_free(&part);
  gpu_free(&out);
}

// Verifies ink2_mtp_swiglu and the 3 Markov RNN head kernels across 2 steps.
static void test_swiglu_and_markov(Gpu *gpu, MemorySlot *slot) {
  int k_swiglu = gpu_find_kernel(gpu, "ink2_mtp_swiglu");
  int k_cat = gpu_find_kernel(gpu, "ink2_mtp_markov_concat");
  int k_gru = gpu_find_kernel(gpu, "ink2_mtp_markov_gru");
  int k_add = gpu_find_kernel(gpu, "ink2_mtp_markov_add");

  // SwiGLU check.
  uint32_t n = 4096;
  Buf g = make_buf(gpu, slot, n * sizeof(float));
  Buf u = make_buf(gpu, slot, n * sizeof(float));
  Buf o = make_buf(gpu, slot, n * sizeof(float));
  uint64_t st = 505;
  for (uint32_t i = 0; i < n; i++) {
    ((float *)g.ptr)[i] = 3.0f * rand_f32(&st);
    ((float *)u.ptr)[i] = 2.0f * rand_f32(&st);
  }
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_swiglu);
  gpu_bind_buffer(&cmd, 0, &g, 0);
  gpu_bind_buffer(&cmd, 1, &u, 0);
  gpu_bind_buffer(&cmd, 2, &o, 0);
  gpu_copy_bytes(&cmd, 3, &n, sizeof n);
  gpu_run_kernel(&cmd, (n + ELEM_THREADS - 1u) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst = 0.0f;
  for (uint32_t i = 0; i < n; i++) {
    float gv = ((float *)g.ptr)[i], uv = ((float *)u.ptr)[i];
    float ref = (gv / (1.0f + expf(-gv))) * uv;
    float e = fabsf(((float *)o.ptr)[i] - ref);
    if (e > worst) worst = e;
  }
  printf("ink2_mtp_swiglu max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_swiglu matches CPU");
  gpu_free(&g);
  gpu_free(&u);
  gpu_free(&o);

  // Markov concat + GRU + add check over steps 1 and 2.
  int vocab = 16, fr_ring = 8, pos0 = 10;
  Buf rstate = make_buf(gpu, slot, INK2_MTP_MARKOV_RANK * sizeof(float));
  Buf mw1 = make_buf(gpu, slot,
                     (size_t)vocab * INK2_MTP_MARKOV_RANK * sizeof(uint16_t));
  Buf tok = make_buf(gpu, slot, INK2_MTP_BLOCK_SIZE * sizeof(int32_t));
  Buf hblk = make_buf(gpu, slot,
                      (size_t)INK2_MTP_BLOCK_SIZE * INK2_DIM * sizeof(float));
  Buf frnn = make_buf(gpu, slot, (size_t)fr_ring * INK2_DIM * sizeof(float));
  Buf joint = make_buf(gpu, slot, INK2_MTP_RNN_IN * sizeof(float));
  Buf proj = make_buf(gpu, slot, INK2_MTP_RNN_OUT * sizeof(float));
  Buf rbias = make_buf(gpu, slot, INK2_MTP_RNN_OUT * sizeof(float));
  Buf lat = make_buf(gpu, slot, INK2_MTP_MARKOV_RANK * sizeof(float));
  Buf corr = make_buf(gpu, slot, INK2_DIM * sizeof(float));
  Buf hbias = make_buf(gpu, slot, INK2_DIM * sizeof(float));
  Buf heff = make_buf(gpu, slot, INK2_DIM * sizeof(float));

  for (int i = 0; i < INK2_MTP_MARKOV_RANK; i++)
    ((float *)rstate.ptr)[i] = 99.0f;  // Must be ignored on step 1.
  for (int i = 0; i < vocab * INK2_MTP_MARKOV_RANK; i++)
    ((uint16_t *)mw1.ptr)[i] = f32_to_bf16(rand_f32(&st));
  ((int32_t *)tok.ptr)[0] = 5;
  ((int32_t *)tok.ptr)[1] = 11;
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_DIM; i++)
    ((float *)hblk.ptr)[i] = rand_f32(&st);
  for (int i = 0; i < fr_ring * INK2_DIM; i++)
    ((float *)frnn.ptr)[i] = rand_f32(&st);
  for (int i = 0; i < INK2_MTP_RNN_OUT; i++) {
    ((float *)proj.ptr)[i] = rand_f32(&st);
    ((float *)rbias.ptr)[i] = rand_f32(&st);
  }
  for (int i = 0; i < INK2_DIM; i++) {
    ((float *)corr.ptr)[i] = rand_f32(&st);
    ((float *)hbias.ptr)[i] = rand_f32(&st);
  }

  float ref_state[INK2_MTP_MARKOV_RANK] = {0};
  worst = 0.0f;
  for (int step = 1; step <= 2; step++) {
    MarkovParam mp = {(uint32_t)step, (uint32_t)pos0, (uint32_t)fr_ring,
                      (uint32_t)INK2_DIM, (uint32_t)INK2_MTP_MARKOV_RANK};
    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_cat);
    gpu_bind_buffer(&cmd, 0, &rstate, 0);
    gpu_bind_buffer(&cmd, 1, &mw1, 0);
    gpu_bind_buffer(&cmd, 2, &tok, 0);
    gpu_bind_buffer(&cmd, 3, &hblk, 0);
    gpu_bind_buffer(&cmd, 4, &frnn, 0);
    gpu_bind_buffer(&cmd, 5, &joint, 0);
    gpu_copy_bytes(&cmd, 6, &mp, sizeof mp);
    gpu_run_kernel(
        &cmd, ((uint32_t)INK2_MTP_RNN_IN + ELEM_THREADS - 1u) / ELEM_THREADS, 1,
        1, ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_gru);
    gpu_bind_buffer(&cmd, 0, &proj, 0);
    gpu_bind_buffer(&cmd, 1, &rbias, 0);
    gpu_bind_buffer(&cmd, 2, &rstate, 0);
    gpu_bind_buffer(&cmd, 3, &lat, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(&cmd, 1, 1, 1, (uint32_t)INK2_MTP_MARKOV_RANK, 1, 1);

    gpu_select_kernel(&cmd, k_add);
    gpu_bind_buffer(&cmd, 0, &hblk, 0);
    gpu_bind_buffer(&cmd, 1, &corr, 0);
    gpu_bind_buffer(&cmd, 2, &hbias, 0);
    gpu_bind_buffer(&cmd, 3, &heff, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(&cmd,
                   ((uint32_t)INK2_DIM + ELEM_THREADS - 1u) / ELEM_THREADS, 1,
                   1, ELEM_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    int prev = ((int32_t *)tok.ptr)[step - 1];
    int fr_slot = (pos0 - 1) % fr_ring;
    const float *jp = (const float *)joint.ptr;
    for (int i = 0; i < INK2_MTP_MARKOV_RANK; i++) {
      float e0 = fabsf(jp[i] - ref_state[i]);
      float e1 = fabsf(
          jp[INK2_MTP_MARKOV_RANK + i] -
          bf16_to_f32(
              ((uint16_t *)mw1.ptr)[(size_t)prev * INK2_MTP_MARKOV_RANK + i]));
      if (e0 > worst) worst = e0;
      if (e1 > worst) worst = e1;
    }
    for (int d = 0; d < INK2_DIM; d++) {
      float e2 = fabsf(jp[2 * INK2_MTP_MARKOV_RANK + d] -
                       ((float *)hblk.ptr)[(size_t)step * INK2_DIM + d]);
      float e3 = fabsf(jp[2 * INK2_MTP_MARKOV_RANK + INK2_DIM + d] -
                       ((float *)frnn.ptr)[(size_t)fr_slot * INK2_DIM + d]);
      float ea = fabsf(((float *)heff.ptr)[d] -
                       (((float *)hblk.ptr)[(size_t)step * INK2_DIM + d] +
                        ((float *)corr.ptr)[d] + ((float *)hbias.ptr)[d]));
      if (e2 > worst) worst = e2;
      if (e3 > worst) worst = e3;
      if (ea > worst) worst = ea;
    }
    for (int i = 0; i < INK2_MTP_MARKOV_RANK; i++) {
      float c0 = ((float *)proj.ptr)[i] + ((float *)rbias.ptr)[i];
      float c1 = ((float *)proj.ptr)[INK2_MTP_MARKOV_RANK + i] +
                 ((float *)rbias.ptr)[INK2_MTP_MARKOV_RANK + i];
      float c2 = ((float *)proj.ptr)[2 * INK2_MTP_MARKOV_RANK + i] +
                 ((float *)rbias.ptr)[2 * INK2_MTP_MARKOV_RANK + i];
      float gate = 1.0f / (1.0f + expf(-c0));
      float cand = tanhf(c1);
      ref_state[i] = cand + gate * (ref_state[i] - cand);
      float ref_lat = tanhf(c2);
      float es = fabsf(((float *)rstate.ptr)[i] - ref_state[i]);
      float el = fabsf(((float *)lat.ptr)[i] - ref_lat);
      if (es > worst) worst = es;
      if (el > worst) worst = el;
    }
  }
  printf("ink2_mtp_markov (2 steps) max err: %.6f\n", worst);
  check(worst < TOL_F32, "ink2_mtp_markov concat/gru/add match CPU");

  gpu_free(&rstate);
  gpu_free(&mw1);
  gpu_free(&tok);
  gpu_free(&hblk);
  gpu_free(&frnn);
  gpu_free(&joint);
  gpu_free(&proj);
  gpu_free(&rbias);
  gpu_free(&lat);
  gpu_free(&corr);
  gpu_free(&hbias);
  gpu_free(&heff);
}

// Verifies fused unembed + softcap + first-maximum argmax (including tail rows
// not divisible by 32 and tie-breaking to lowest token index).
static void test_unembed_argmax(Gpu *gpu, MemorySlot *slot) {
  int k_part = gpu_find_kernel(gpu, "ink2_mtp_unembed_argmax_part");
  int k_final = gpu_find_kernel(gpu, "ink2_mtp_unembed_argmax_final");

  typedef struct {
    uint32_t vocab, k;
    float cap;
  } UnembedParam;

  typedef struct {
    float val;
    int32_t idx;
  } ArgmaxPair;

  uint32_t vocab = 357, k = 256, nparts = (vocab + 31u) / 32u;
  Buf w = make_buf(gpu, slot, (size_t)vocab * k * sizeof(uint16_t));
  Buf x = make_buf(gpu, slot, (size_t)k * sizeof(float));
  Buf parts = make_buf(gpu, slot, (size_t)nparts * sizeof(ArgmaxPair));
  Buf out = make_buf(gpu, slot, sizeof(int32_t));

  uint64_t st = 606;
  uint16_t *wp = (uint16_t *)w.ptr;
  float *xp = (float *)x.ptr;
  for (uint32_t i = 0; i < vocab * k; i++) wp[i] = f32_to_bf16(rand_f32(&st));
  for (uint32_t i = 0; i < k; i++) xp[i] = rand_f32(&st);
  // Make rows 123 and 355 tied maxima (unsatured under softcap 30) to test
  // tail handling and first-max rule.
  for (uint32_t i = 0; i < k; i++) {
    uint16_t v = f32_to_bf16(xp[i] >= 0.0f ? 0.1f : -0.1f);
    wp[123u * k + i] = v;
    wp[355u * k + i] = v;
  }

  UnembedParam up = {vocab, k, 30.0f};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &w, 0);
  gpu_bind_buffer(&cmd, 1, &x, 0);
  gpu_bind_buffer(&cmd, 2, &parts, 0);
  gpu_copy_bytes(&cmd, 3, &up, sizeof up);
  gpu_run_kernel(&cmd, nparts, 1, 1, ROW_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_final);
  gpu_bind_buffer(&cmd, 0, &parts, 0);
  gpu_bind_buffer(&cmd, 1, &out, 0);
  gpu_copy_bytes(&cmd, 2, &nparts, sizeof nparts);
  gpu_run_kernel(&cmd, 1, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int32_t got = *(const int32_t *)out.ptr;
  printf("ink2_mtp_unembed_argmax tie-break winner: %d (expected 123)\n", got);
  check(got == 123, "ink2_mtp_unembed_argmax picks first maximum on tie");

  // Break tie in favor of tail row 355.
  wp[355u * k] = f32_to_bf16(xp[0] >= 0.0f ? 2.0f : -2.0f);
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &w, 0);
  gpu_bind_buffer(&cmd, 1, &x, 0);
  gpu_bind_buffer(&cmd, 2, &parts, 0);
  gpu_copy_bytes(&cmd, 3, &up, sizeof up);
  gpu_run_kernel(&cmd, nparts, 1, 1, ROW_THREADS, 1, 1);
  gpu_select_kernel(&cmd, k_final);
  gpu_bind_buffer(&cmd, 0, &parts, 0);
  gpu_bind_buffer(&cmd, 1, &out, 0);
  gpu_copy_bytes(&cmd, 2, &nparts, sizeof nparts);
  gpu_run_kernel(&cmd, 1, 1, 1, ROW_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  got = *(const int32_t *)out.ptr;
  printf("ink2_mtp_unembed_argmax tail winner: %d (expected 355)\n", got);
  check(got == 355, "ink2_mtp_unembed_argmax handles non-multiple-of-32 tail");

  gpu_free(&w);
  gpu_free(&x);
  gpu_free(&parts);
  gpu_free(&out);
}

// Verifies paged global MTP context store and noncausal attention in both BF16
// and FP8 across a 256-token page boundary (pos0 = 250, 12 stored context rows
// spanning pages 0 and 1).
static void test_paged_global_mtp(Gpu *gpu, MemorySlot *slot) {
  int k_store = gpu_find_kernel(gpu, "ink2_mtp_ctx_kv_store");
  int k_store_fp8 = gpu_find_kernel(gpu, "ink2_mtp_ctx_kv_store_fp8");
  int k_part = gpu_find_kernel(gpu, "ink2_mtp_attend_part");
  int k_part_fp8 = gpu_find_kernel(gpu, "ink2_mtp_attend_part_fp8");
  int k_comb = gpu_find_kernel(gpu, "ink2_mtp_attend_combine");

  const int page_tokens = 256, pos0 = 250, add_rows = 12,
            ctx_end = pos0 + add_rows;
  const int nparts = 3, part_stride = 4;
  const int slice_count = 4, k_slice = 2, v_slice = 3;
  StateGrowthPolicy tg = {
      .start_cap = 4, .factor = 2, .max_step = 8, .align = 4, .max_cap = 32};

  Buf kr = make_buf(gpu, slot, (size_t)add_rows * INK2_KV_DIM * sizeof(float));
  Buf vr = make_buf(gpu, slot, (size_t)add_rows * INK2_KV_DIM * sizeof(float));
  Buf kn = make_buf(gpu, slot, (size_t)INK2_HEAD_DIM * sizeof(float));
  Buf qb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(uint16_t));
  Buf kb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf vb = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_KV_DIM * sizeof(uint16_t));
  Buf part = make_buf(gpu, slot,
                      (size_t)INK2_MTP_BLOCK_SIZE * INK2_HEADS * part_stride *
                          (INK2_HEAD_DIM + 2) * sizeof(float));
  Buf out_flat = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(float));
  Buf out_paged = make_buf(
      gpu, slot, (size_t)INK2_MTP_BLOCK_SIZE * INK2_Q_DIM * sizeof(float));
  Buf tgt = make_buf(gpu, slot, sizeof(KvTarget));

  uint64_t st = 777;
  for (int i = 0; i < add_rows * INK2_KV_DIM; i++) {
    ((float *)kr.ptr)[i] = rand_f32(&st);
    ((float *)vr.ptr)[i] = rand_f32(&st);
  }
  for (int i = 0; i < INK2_HEAD_DIM; i++)
    ((float *)kn.ptr)[i] = 0.7f + 0.3f * rand_f32(&st);
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    ((uint16_t *)qb.ptr)[i] = f32_to_bf16(rand_f32(&st));
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_KV_DIM; i++) {
    ((uint16_t *)kb.ptr)[i] = f32_to_bf16(rand_f32(&st));
    ((uint16_t *)vb.ptr)[i] = f32_to_bf16(rand_f32(&st));
  }

  // BF16 flat vs paged across pages 0 and 1.
  size_t bf_row = (size_t)INK2_KV_DIM * sizeof(uint16_t);
  size_t bf_slice = (size_t)page_tokens * bf_row;
  KvPageLayout bf_layout =
      kv_page_layout_init(page_tokens, bf_slice, slice_count, tg);
  KvPageTable pt_bf;
  kv_page_table_init(&pt_bf, kv_page_backend_gpu(gpu, slot), &bf_layout);
  check(kv_page_table_prepare_write(&pt_bf, 0, ctx_end) == 1,
        "MTP paged BF16 table allocates 2 pages");
  Buf kc_bf = make_buf(gpu, slot, (size_t)ctx_end * bf_row);
  Buf vc_bf = make_buf(gpu, slot, (size_t)ctx_end * bf_row);
  for (int pos = 0; pos < pos0; pos++) {
    for (int i = 0; i < INK2_KV_DIM; i++) {
      ((uint16_t *)kc_bf.ptr)[(size_t)pos * INK2_KV_DIM + i] =
          f32_to_bf16(rand_f32(&st));
      ((uint16_t *)vc_bf.ptr)[(size_t)pos * INK2_KV_DIM + i] =
          f32_to_bf16(rand_f32(&st));
    }
    int p = pos >> 8, in_p = pos & 255;
    memcpy((uint8_t *)pt_bf.host_ptrs[p] + (size_t)k_slice * bf_slice +
               (size_t)in_p * bf_row,
           (const uint8_t *)kc_bf.ptr + (size_t)pos * bf_row, bf_row);
    memcpy((uint8_t *)pt_bf.host_ptrs[p] + (size_t)v_slice * bf_slice +
               (size_t)in_p * bf_row,
           (const uint8_t *)vc_bf.ptr + (size_t)pos * bf_row, bf_row);
  }

  uint32_t k_off_bf = (uint32_t)((size_t)k_slice * bf_slice);
  uint32_t v_off_bf = (uint32_t)((size_t)v_slice * bf_slice);
  CtxKvParam sp_flat = {(uint32_t)add_rows,
                        (uint32_t)pos0,
                        0u,
                        0u,
                        TEST_ROPE_THETA,
                        TEST_EPS,
                        0u,
                        0u,
                        0u};
  CtxKvParam sp_paged = {
      (uint32_t)add_rows, (uint32_t)pos0, 0u, 0u, TEST_ROPE_THETA, TEST_EPS, 8u,
      k_off_bf,           v_off_bf};
  AttendParam ap_flat = {(uint32_t)ctx_end,
                         (uint32_t)INK2_MTP_BLOCK_SIZE,
                         0u,
                         0u,
                         (uint32_t)nparts,
                         (uint32_t)part_stride,
                         0u,
                         0u,
                         0u};
  AttendParam ap_paged = {
      (uint32_t)ctx_end, (uint32_t)INK2_MTP_BLOCK_SIZE, 0u, 0u,
      (uint32_t)nparts,  (uint32_t)part_stride,         8u, k_off_bf,
      v_off_bf};

  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &kc_bf, 0);
  gpu_bind_buffer(&cmd, 4, &vc_bf, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &sp_flat, sizeof sp_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)add_rows, 1,
                 HEAD_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &kc_bf, 0);
  gpu_bind_buffer(&cmd, 2, &vc_bf, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap_flat, sizeof ap_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);
  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out_flat, 0);
  gpu_copy_bytes(&cmd, 2, &ap_flat, sizeof ap_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_store);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &pt_bf.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 4, &pt_bf.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &sp_paged, sizeof sp_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)add_rows, 1,
                 HEAD_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_part);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &pt_bf.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 2, &pt_bf.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap_paged, sizeof ap_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);
  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out_paged, 0);
  gpu_copy_bytes(&cmd, 2, &ap_paged, sizeof ap_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst_bf = 0.0f;
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    worst_bf = fmaxf(worst_bf, fabsf(((float *)out_flat.ptr)[i] -
                                     ((float *)out_paged.ptr)[i]));
  printf("ink2_mtp paged global BF16 max diff: %.6f\n", worst_bf);
  check(worst_bf < 1e-6f, "MTP paged global BF16 store+attend matches flat");

  // FP8 flat vs paged across pages 0 and 1.
  size_t fp8_slice = (size_t)page_tokens * INK2_FP8_KV_ROW_BYTES;
  KvPageLayout fp8_layout =
      kv_page_layout_init(page_tokens, fp8_slice, slice_count, tg);
  KvPageTable pt_fp8;
  kv_page_table_init(&pt_fp8, kv_page_backend_gpu(gpu, slot), &fp8_layout);
  check(kv_page_table_prepare_write(&pt_fp8, 0, ctx_end) == 1,
        "MTP paged FP8 table allocates 2 pages");
  Buf kc_fp8 = make_buf(gpu, slot, (size_t)ctx_end * INK2_FP8_KV_ROW_BYTES);
  Buf vc_fp8 = make_buf(gpu, slot, (size_t)ctx_end * INK2_FP8_KV_ROW_BYTES);
  for (int pos = 0; pos < pos0; pos++) {
    for (int kh = 0; kh < INK2_KV_HEADS; kh++) {
      float khv[INK2_HEAD_DIM], vhv[INK2_HEAD_DIM];
      for (int d = 0; d < INK2_HEAD_DIM; d++) {
        khv[d] = bf16_to_f32(((uint16_t *)kc_bf.ptr)[(size_t)pos * INK2_KV_DIM +
                                                     kh * INK2_HEAD_DIM + d]);
        vhv[d] = bf16_to_f32(((uint16_t *)vc_bf.ptr)[(size_t)pos * INK2_KV_DIM +
                                                     kh * INK2_HEAD_DIM + d]);
      }
      fp8_kv_pack_head((uint8_t *)kc_fp8.ptr, pos, kh, khv);
      fp8_kv_pack_head((uint8_t *)vc_fp8.ptr, pos, kh, vhv);
    }
    int p = pos >> 8, in_p = pos & 255;
    memcpy((uint8_t *)pt_fp8.host_ptrs[p] + (size_t)k_slice * fp8_slice +
               (size_t)in_p * INK2_FP8_KV_ROW_BYTES,
           (const uint8_t *)kc_fp8.ptr + (size_t)pos * INK2_FP8_KV_ROW_BYTES,
           INK2_FP8_KV_ROW_BYTES);
    memcpy((uint8_t *)pt_fp8.host_ptrs[p] + (size_t)v_slice * fp8_slice +
               (size_t)in_p * INK2_FP8_KV_ROW_BYTES,
           (const uint8_t *)vc_fp8.ptr + (size_t)pos * INK2_FP8_KV_ROW_BYTES,
           INK2_FP8_KV_ROW_BYTES);
  }
  uint32_t k_off_fp8 = (uint32_t)((size_t)k_slice * fp8_slice);
  uint32_t v_off_fp8 = (uint32_t)((size_t)v_slice * fp8_slice);
  sp_paged.k_slice_off = k_off_fp8;
  sp_paged.v_slice_off = v_off_fp8;
  ap_paged.k_slice_off = k_off_fp8;
  ap_paged.v_slice_off = v_off_fp8;

  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_store_fp8);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &kc_fp8, 0);
  gpu_bind_buffer(&cmd, 4, &vc_fp8, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &sp_flat, sizeof sp_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)add_rows, 1,
                 HEAD_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_part_fp8);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &kc_fp8, 0);
  gpu_bind_buffer(&cmd, 2, &vc_fp8, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap_flat, sizeof ap_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);
  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out_flat, 0);
  gpu_copy_bytes(&cmd, 2, &ap_flat, sizeof ap_flat);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_store_fp8);
  gpu_bind_buffer(&cmd, 0, &kr, 0);
  gpu_bind_buffer(&cmd, 1, &vr, 0);
  gpu_bind_buffer(&cmd, 2, &kn, 0);
  gpu_bind_buffer(&cmd, 3, &pt_fp8.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 4, &pt_fp8.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 5, &tgt, 0);
  gpu_copy_bytes(&cmd, 6, &sp_paged, sizeof sp_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)add_rows, 1,
                 HEAD_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_part_fp8);
  gpu_bind_buffer(&cmd, 0, &qb, 0);
  gpu_bind_buffer(&cmd, 1, &pt_fp8.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 2, &pt_fp8.gpu_addrs, 0);
  gpu_bind_buffer(&cmd, 3, &kb, 0);
  gpu_bind_buffer(&cmd, 4, &vb, 0);
  gpu_bind_buffer(&cmd, 5, &part, 0);
  gpu_copy_bytes(&cmd, 6, &ap_paged, sizeof ap_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_KV_HEADS, (uint32_t)nparts,
                 (uint32_t)INK2_MTP_BLOCK_SIZE, ATTEND_THREADS, 1, 1);
  gpu_select_kernel(&cmd, k_comb);
  gpu_bind_buffer(&cmd, 0, &part, 0);
  gpu_bind_buffer(&cmd, 1, &out_paged, 0);
  gpu_copy_bytes(&cmd, 2, &ap_paged, sizeof ap_paged);
  gpu_run_kernel(&cmd, (uint32_t)INK2_HEADS, (uint32_t)INK2_MTP_BLOCK_SIZE, 1,
                 COMBINE_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst_fp8 = 0.0f;
  for (int i = 0; i < INK2_MTP_BLOCK_SIZE * INK2_Q_DIM; i++)
    worst_fp8 = fmaxf(worst_fp8, fabsf(((float *)out_flat.ptr)[i] -
                                       ((float *)out_paged.ptr)[i]));
  printf("ink2_mtp paged global FP8 max diff: %.6f\n", worst_fp8);
  check(worst_fp8 < 1e-6f, "MTP paged global FP8 store+attend matches flat");

  kv_page_table_free(&pt_bf);
  kv_page_table_free(&pt_fp8);
  gpu_free(&kc_bf);
  gpu_free(&vc_bf);
  gpu_free(&kc_fp8);
  gpu_free(&vc_fp8);
  gpu_free(&kr);
  gpu_free(&vr);
  gpu_free(&kn);
  gpu_free(&qb);
  gpu_free(&kb);
  gpu_free(&vb);
  gpu_free(&part);
  gpu_free(&out_flat);
  gpu_free(&out_paged);
  gpu_free(&tgt);
}

int main(void) {
  Memory *mem = memory_new();
  MemorySlot *slot = memory_shared(mem);
  size_t len = 0;
  const char *src = kernel_library_source(&len);
  Gpu *gpu = gpu_new(src, len);

  test_fuse_norm(gpu, slot);
  test_ctx_kv_store(gpu, slot);
  test_ctx_kv_store_fp8(gpu, slot);
  test_block_init_and_qk(gpu, slot);
  test_attend(gpu, slot);
  test_attend_fp8(gpu, slot);
  test_paged_global_mtp(gpu, slot);
  test_swiglu_and_markov(gpu, slot);
  test_unembed_argmax(gpu, slot);
  gpu_return_freed_memory(gpu);

  return failures != 0;
}
