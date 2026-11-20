#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/vision_encoder.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tests MiniMax-H3 Qwen3-VL vision sizing rules and all Metal kernels in
// src/kernels/minimax_h3_vision.metal against CPU references.

// Vision geometry matching src/kernels/minimax_h3_vision.metal.
static constexpr int VIS_DIM = 1152, VIS_HEADS = 16, VIS_HEAD_DIM = 72,
                     VIS_HALF_HEAD = 36, VIS_FREQS = 18, VIS_PAD_HEAD = 128,
                     VIS_POS_SIDE = 48, VIS_POS_ENTRIES = 2304,
                     VIS_QKV_WIDTH = 3 * VIS_DIM;

// Threadgroup sizes, tile padding and tolerances.
static constexpr uint32_t TILE_M = 64, MM_THREADS = 128, HEADS_THREADS = 64,
                          ELEM_THREADS = 256, GRID2D_THREADS = 32;
static constexpr float FP32_TOL = 1e-4f, BF16_TOL = 5e-2f;

// Parameters passed to minimax_h3_vis_pos_add.
typedef struct {
  // Patch grid height and width.
  uint32_t grid_h, grid_w;
} VisPosAddParam;

// Parameters passed to minimax_h3_vis_heads.
typedef struct {
  // Patch count, padded head stride and patch grid height/width.
  uint32_t rows, padded, grid_h, grid_w;
} VisHeadsParam;

// Parameters passed to minimax_h3_vis_attend.
typedef struct {
  // Patch count and padded head stride.
  uint32_t rows, padded;
} VisAttendParam;

// Deterministic PRNG for synthetic inputs.
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

// Tests Qwen3-VL vision and VAE reference sizing helpers.
static void test_ref_sizes(void) {
  int w = 0, h = 0;
  minimax_h3_vision_ref_size(1024, 1024, &w, &h);
  check(w == 1024 && h == 1024, "1024x1024 keeps exact 1024x1024 vision size");

  minimax_h3_vision_ref_size(4096, 2048, &w, &h);
  check(w == 2048 && h == 1024,
        "4096x2048 scales short edge to 1024 for vision (2048x1024)");

  minimax_h3_vae_ref_size(4096, 2048, &w, &h);
  check(w == 4096 && h == 2048,
        "4096x2048 keeps short edge at 2048 for VAE (4096x2048)");

  minimax_h3_vae_ref_size(6000, 3000, &w, &h);
  check(w == 4096 && h == 2048,
        "6000x3000 clamps VAE short edge down to 2048 (4096x2048)");

  minimax_h3_vae_ref_size(512, 768, &w, &h);
  check(w == 1024 && h == 1536,
        "512x768 scales VAE short edge up to 1024 (1024x1536)");

  // Extreme wide aspect ratio triggers smart_resize max_pixels cap.
  minimax_h3_vision_ref_size(20000, 1000, &w, &h);
  check((w % 32 == 0) && (h % 32 == 0) && (int64_t)w * h <= 12845056,
        "extreme aspect ratio stays 32-aligned within smart_resize max_pixels");
}

// Tests minimax_h3_vis_pos_add on a non-square patch grid (6x8).
static void test_pos_add(Gpu *gpu, MemorySlot *slot) {
  int k_pos = gpu_find_kernel(gpu, "minimax_h3_vis_pos_add");
  int gh = 6, gw = 8, tokens = gh * gw;
  uint32_t rng = 0x13572468u;

  Buf x = gpu_alloc(gpu, slot, (size_t)tokens * VIS_DIM * sizeof(float));
  Buf pe = gpu_alloc(gpu, slot,
                     (size_t)VIS_POS_ENTRIES * VIS_DIM * sizeof(uint16_t));
  float *xp = (float *)x.ptr;
  uint16_t *pep = (uint16_t *)pe.ptr;
  float *x0 = xmalloc((size_t)tokens * VIS_DIM * sizeof(float));
  for (int i = 0; i < tokens * VIS_DIM; i++) x0[i] = xp[i] = rng_float(&rng);
  for (int i = 0; i < VIS_POS_ENTRIES * VIS_DIM; i++)
    pep[i] = f32_to_bf16(0.5f * rng_float(&rng));

  VisPosAddParam p = {(uint32_t)gh, (uint32_t)gw};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_pos);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &pe, 0);
  gpu_copy_bytes(&cmd, 2, &p, sizeof p);
  gpu_run_kernel(&cmd,
                 ((uint32_t)VIS_DIM + GRID2D_THREADS - 1) / GRID2D_THREADS,
                 (uint32_t)tokens, 1, GRID2D_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst = 0.0f;
  int mw = gw / 2;
  for (int tok = 0; tok < tokens; tok++) {
    int blk = tok / 4, sub = tok % 4;
    int r = (blk / mw) * 2 + (sub / 2);
    int c = (blk % mw) * 2 + (sub % 2);
    float fh = (float)r * (float)(VIS_POS_SIDE - 1) / (float)(gh - 1);
    float fw = (float)c * (float)(VIS_POS_SIDE - 1) / (float)(gw - 1);
    int h0 = (int)fh, w0 = (int)fw;
    int h1 = h0 + 1 < VIS_POS_SIDE ? h0 + 1 : VIS_POS_SIDE - 1;
    int w1 = w0 + 1 < VIS_POS_SIDE ? w0 + 1 : VIS_POS_SIDE - 1;
    float dh = fh - (float)h0, dw = fw - (float)w0;
    for (int d = 0; d < VIS_DIM; d++) {
      float v00 = bf16_to_f32(pep[(h0 * VIS_POS_SIDE + w0) * VIS_DIM + d]);
      float v01 = bf16_to_f32(pep[(h0 * VIS_POS_SIDE + w1) * VIS_DIM + d]);
      float v10 = bf16_to_f32(pep[(h1 * VIS_POS_SIDE + w0) * VIS_DIM + d]);
      float v11 = bf16_to_f32(pep[(h1 * VIS_POS_SIDE + w1) * VIS_DIM + d]);
      float ref = x0[tok * VIS_DIM + d] + (1.0f - dh) * (1.0f - dw) * v00 +
                  (1.0f - dh) * dw * v01 + dh * (1.0f - dw) * v10 +
                  dh * dw * v11;
      float diff = fabsf(xp[tok * VIS_DIM + d] - ref);
      if (diff > worst) worst = diff;
    }
  }
  check(worst < FP32_TOL,
        "minimax_h3_vis_pos_add matches bilinear 48x48 CPU reference");

  free(x0);
  gpu_free(&pe), gpu_free(&x);
}

// Tests minimax_h3_gelu_tanh and minimax_h3_gelu_exact against libm references.
static void test_gelu(Gpu *gpu, MemorySlot *slot) {
  int k_tanh = gpu_find_kernel(gpu, "minimax_h3_gelu_tanh");
  int k_exact = gpu_find_kernel(gpu, "minimax_h3_gelu_exact");
  uint32_t n = 1025;
  Buf a = gpu_alloc(gpu, slot, (size_t)n * sizeof(float));
  Buf b = gpu_alloc(gpu, slot, (size_t)n * sizeof(float));
  float *ap = (float *)a.ptr, *bp = (float *)b.ptr;
  float *orig = xmalloc((size_t)n * sizeof(float));
  for (uint32_t i = 0; i < n; i++) {
    float v = -100.0f + 200.0f * (float)i / (float)(n - 1);
    orig[i] = ap[i] = bp[i] = v;
  }

  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_tanh);
  gpu_bind_buffer(&cmd, 0, &a, 0);
  gpu_copy_bytes(&cmd, 1, &n, sizeof n);
  gpu_run_kernel(&cmd, (n + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_exact);
  gpu_bind_buffer(&cmd, 0, &b, 0);
  gpu_copy_bytes(&cmd, 1, &n, sizeof n);
  gpu_run_kernel(&cmd, (n + ELEM_THREADS - 1) / ELEM_THREADS, 1, 1,
                 ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  float worst_t = 0.0f, worst_e = 0.0f;
  int finite_t = 1, finite_e = 1;
  for (uint32_t i = 0; i < n; i++) {
    float x = orig[i];
    float rt =
        0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    float re = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    if (!isfinite(ap[i])) finite_t = 0;
    if (!isfinite(bp[i])) finite_e = 0;
    float dt = fabsf(ap[i] - rt), de = fabsf(bp[i] - re);
    if (dt > worst_t) worst_t = dt;
    if (de > worst_e) worst_e = de;
  }
  check(finite_t && worst_t < FP32_TOL,
        "minimax_h3_gelu_tanh matches CPU reference across [-100, 100]");
  check(finite_e && worst_e < FP32_TOL,
        "minimax_h3_gelu_exact matches libm erff GELU across [-100, 100]");

  free(orig);
  gpu_free(&b), gpu_free(&a);
}

// Tests minimax_h3_vis_heads and minimax_h3_vis_attend across a 64-key tile
// boundary (grid 8x10 -> 80 tokens).
static void test_heads_and_attend(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "minimax_h3_vis_heads");
  int k_attend = gpu_find_kernel(gpu, "minimax_h3_vis_attend");
  int gh = 8, gw = 10, rows = gh * gw;
  uint32_t padded = ((uint32_t)rows + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t rng = 0x24681357u;

  Buf qkv = gpu_alloc(gpu, slot, (size_t)rows * VIS_QKV_WIDTH * sizeof(float));
  Buf inv = gpu_alloc(gpu, slot, (size_t)VIS_FREQS * sizeof(float));
  size_t head_elems = (size_t)VIS_HEADS * padded * VIS_PAD_HEAD;
  Buf qh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf kh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf vh = gpu_alloc(gpu, slot, head_elems * sizeof(uint16_t));
  Buf out = gpu_alloc(gpu, slot, (size_t)rows * VIS_DIM * sizeof(float));

  float *qkvp = (float *)qkv.ptr, *invp = (float *)inv.ptr;
  for (int i = 0; i < rows * VIS_QKV_WIDTH; i++)
    qkvp[i] = 0.5f * rng_float(&rng);
  for (int i = 0; i < VIS_FREQS; i++)
    invp[i] = (float)(1.0 / pow(10000.0, (double)i / (double)VIS_FREQS));

  VisHeadsParam hp = {(uint32_t)rows, padded, (uint32_t)gh, (uint32_t)gw};
  VisAttendParam ap = {(uint32_t)rows, padded};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_heads);
  gpu_bind_buffer(&cmd, 0, &qkv, 0);
  gpu_bind_buffer(&cmd, 1, &inv, 0);
  gpu_bind_buffer(&cmd, 2, &qh, 0);
  gpu_bind_buffer(&cmd, 3, &kh, 0);
  gpu_bind_buffer(&cmd, 4, &vh, 0);
  gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
  gpu_run_kernel(&cmd, (uint32_t)(3 * VIS_HEADS), padded, 1, HEADS_THREADS, 1,
                 1);

  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &qh, 0);
  gpu_bind_buffer(&cmd, 1, &kh, 0);
  gpu_bind_buffer(&cmd, 2, &vh, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
  gpu_run_kernel(&cmd, padded / TILE_M, (uint32_t)VIS_HEADS, 1, MM_THREADS, 1,
                 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  const uint16_t *qhp = (const uint16_t *)qh.ptr;
  const uint16_t *khp = (const uint16_t *)kh.ptr;
  const uint16_t *vhp = (const uint16_t *)vh.ptr;
  float worst_h = 0.0f;
  int pad_ok = 1;
  int mw = gw / 2;
  for (int h = 0; h < VIS_HEADS; h++) {
    for (int r = 0; r < (int)padded; r++) {
      size_t base = ((size_t)h * padded + (size_t)r) * VIS_PAD_HEAD;
      if (r >= rows) {
        for (int d = 0; d < VIS_PAD_HEAD; d++)
          pad_ok &=
              (qhp[base + d] == 0 && khp[base + d] == 0 && vhp[base + d] == 0);
        continue;
      }
      for (int d = VIS_HEAD_DIM; d < VIS_PAD_HEAD; d++)
        pad_ok &=
            (qhp[base + d] == 0 && khp[base + d] == 0 && vhp[base + d] == 0);
      int blk = r / 4, sub = r % 4;
      int ri = (blk / mw) * 2 + (sub / 2), ci = (blk % mw) * 2 + (sub % 2);
      const float *vs =
          qkvp + (size_t)r * VIS_QKV_WIDTH + 2u * VIS_DIM + h * VIS_HEAD_DIM;
      for (int d = 0; d < VIS_HEAD_DIM; d++) {
        float diff = fabsf(bf16_to_f32(vhp[base + d]) - vs[d]);
        if (diff > worst_h) worst_h = diff;
      }
      for (int kind = 0; kind < 2; kind++) {
        const float *src = qkvp + (size_t)r * VIS_QKV_WIDTH +
                           (size_t)kind * VIS_DIM + h * VIS_HEAD_DIM;
        const uint16_t *got = (kind == 0 ? qhp : khp) + base;
        for (int i = 0; i < VIS_HALF_HEAD; i++) {
          float coord = i < VIS_FREQS ? (float)ri : (float)ci;
          int fi = i < VIS_FREQS ? i : i - VIS_FREQS;
          float ang = coord * invp[fi];
          float cs = cosf(ang), sn = sinf(ang);
          float r0 = src[i] * cs - src[i + VIS_HALF_HEAD] * sn;
          float r1 = src[i + VIS_HALF_HEAD] * cs + src[i] * sn;
          float d0 = fabsf(bf16_to_f32(got[i]) - r0);
          float d1 = fabsf(bf16_to_f32(got[i + VIS_HALF_HEAD]) - r1);
          if (d0 > worst_h) worst_h = d0;
          if (d1 > worst_h) worst_h = d1;
        }
      }
    }
  }
  check(pad_ok && worst_h < BF16_TOL,
        "minimax_h3_vis_heads 2D RoPE and 72->128 padding match CPU reference");

  const float *op = (const float *)out.ptr;
  float scale = 1.0f / sqrtf((float)VIS_HEAD_DIM);
  float worst_a = 0.0f;
  static const int check_heads[] = {0, 7, 15};
  for (size_t hi = 0; hi < sizeof check_heads / sizeof check_heads[0]; hi++) {
    int h = check_heads[hi];
    for (int q = 0; q < rows; q++) {
      float scores[80], mx = -INFINITY;
      size_t qbase = ((size_t)h * padded + (size_t)q) * VIS_PAD_HEAD;
      for (int k = 0; k < rows; k++) {
        size_t kbase = ((size_t)h * padded + (size_t)k) * VIS_PAD_HEAD;
        float dot = 0.0f;
        for (int d = 0; d < VIS_HEAD_DIM; d++)
          dot += bf16_to_f32(qhp[qbase + d]) * bf16_to_f32(khp[kbase + d]);
        scores[k] = dot * scale;
        if (scores[k] > mx) mx = scores[k];
      }
      float sum = 0.0f;
      for (int k = 0; k < rows; k++) {
        scores[k] = expf(scores[k] - mx);
        sum += scores[k];
      }
      for (int d = 0; d < VIS_HEAD_DIM; d++) {
        float acc = 0.0f;
        for (int k = 0; k < rows; k++) {
          size_t vbase = ((size_t)h * padded + (size_t)k) * VIS_PAD_HEAD;
          acc += scores[k] * bf16_to_f32(vhp[vbase + d]);
        }
        float ref = acc / sum;
        float got = op[(size_t)q * VIS_DIM + h * VIS_HEAD_DIM + d];
        float diff = fabsf(got - ref);
        if (diff > worst_a) worst_a = diff;
      }
    }
  }
  check(worst_a < 2e-2f,
        "minimax_h3_vis_attend bidirectional 16-head FlashAttention matches "
        "CPU");

  gpu_free(&out), gpu_free(&vh), gpu_free(&kh), gpu_free(&qh);
  gpu_free(&inv), gpu_free(&qkv);
}

int main(void) {
  test_ref_sizes();
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_pos_add(gpu, slot);
  test_gelu(gpu, slot);
  test_heads_and_attend(gpu, slot);
  return failures != 0;
}
