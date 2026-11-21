// Tests MiniMax-H3 Video VAE Metal kernels (3D ViT decoder + 3D causal CNN
// encoder) against independent C CPU references without checkpoint files.
#include "base/error.h"
#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// 3D ViT decoder constants matching src/kernels/minimax_h3_video_vae.metal.
static constexpr uint32_t LATENT_CH = 24, VVAE_DIM = 2048, HEADS = 32,
                          HEAD_DIM = 64, QKV_PARTS = 3, ROT_PAIRS = 24,
                          FREQ_PER_AXIS = 8, REGISTERS = 4,
                          SUFFIX = REGISTERS + 1, TILE_M = 64,
                          ATTN_GROUP_ROWS = 64, ATTN_THREADS = 128,
                          EMBED_THREADS = 256, HEADS_THREADS = 32,
                          GN_THREADS = 256, IM2COL_TX = 32, IM2COL_TY = 8;

// RoPE, normalization and comparison tolerances.
static constexpr float NORM_EPS = 1e-5f, GN_EPS = 1e-6f,
                       TWO_PI = 6.28318530717958647692f, CELL_CENTER = 0.5f,
                       FP32_TOL = 1e-4f, FP16_TOL = 2e-2f;
static constexpr double ROPE_THETA = 100.0;

// Linear congruential generator constants for deterministic test tensors.
static constexpr uint32_t LCG_MUL = 1664525u, LCG_ADD = 1013904223u,
                          LCG_SHIFT = 8, LCG_MASK = 0xffffffu;
static constexpr float LCG_SCALE = 1.0f / (float)(1u << 24);

// Parameters passed to minimax_h3_vvae_embed.
typedef struct {
  // Latent tile dimensions (t, h, w).
  uint32_t t_lat, h_lat, w_lat;
} VvaeEmbedParam;

// Parameters passed to minimax_h3_vvae_heads.
typedef struct {
  // Sequence length, padded head stride and latent grid dimensions.
  uint32_t rows, padded, t_lat, h_lat, w_lat;
  // Per-head non-affine QK RMSNorm epsilon.
  float eps;
} VvaeHeadsParam;

// Parameters passed to minimax_h3_vvae_attend.
typedef struct {
  // Sequence length and padded head stride.
  uint32_t rows, padded;
} VvaeAttendParam;

// Parameters passed to minimax_h3_vvae_gn_silu_3d.
typedef struct {
  // Pixels per frame, channels and group count.
  uint32_t hw, channels, groups;
  // GroupNorm epsilon.
  float eps;
} VvaeGn3dParam;

// Parameters passed to minimax_h3_vvae_im2col_3d_f16.
typedef struct {
  // Input shape, channels, output width, frame index and strip geometry.
  uint32_t in_t, in_h, in_w, cin, out_w, ot, out_y0, strip_h;
  // Kernel sizes, strides, causal front pad, spatial pads and padded K width.
  uint32_t kt, ks, time_stride, space_stride, front_pad, pad_h, pad_w, k_padded;
} VvaeIm2Col3dParam;

// Returns a deterministic pseudo-random float in [-1, 1].
static float rng_float(uint32_t *state) {
  *state = *state * LCG_MUL + LCG_ADD;
  uint32_t v = (*state >> LCG_SHIFT) & LCG_MASK;
  return 2.0f * (float)v * LCG_SCALE - 1.0f;
}

// Returns ceil(a / b) for positive integers.
static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Reflects coordinate c into [0, limit - 1] matching PyTorch reflect padding.
static uint32_t reflect_coord(int c, uint32_t limit) {
  if (limit <= 1u) return 0u;
  int lim = (int)limit;
  if (c < 0) c = -c;
  if (c >= lim) c = 2 * lim - 2 - c;
  if (c < 0) c = 0;
  if (c >= lim) c = lim - 1;
  return (uint32_t)c;
}

// Tests 3D video VAE tile embedding (post_quant_conv + x_embedder + suffixes).
static void test_vvae_embed(Gpu *gpu, MemorySlot *slot) {
  int k_emb = gpu_find_kernel(gpu, "minimax_h3_vvae_embed");
  const uint32_t tl = 3, hl = 4, wl = 5;
  const uint32_t patches = tl * hl * wl, rows = patches + SUFFIX;
  uint32_t rng = 0x11112222u;

  Buf bz = gpu_alloc(gpu, slot, (size_t)LATENT_CH * patches * sizeof(float));
  Buf bpw = gpu_alloc(gpu, slot, (size_t)LATENT_CH * LATENT_CH * sizeof(float));
  Buf bpb = gpu_alloc(gpu, slot, (size_t)LATENT_CH * sizeof(float));
  Buf bew = gpu_alloc(gpu, slot, (size_t)VVAE_DIM * LATENT_CH * sizeof(float));
  Buf beb = gpu_alloc(gpu, slot, (size_t)VVAE_DIM * sizeof(float));
  Buf breg = gpu_alloc(gpu, slot, (size_t)REGISTERS * VVAE_DIM * sizeof(float));
  Buf bout = gpu_alloc(gpu, slot, (size_t)rows * VVAE_DIM * sizeof(float));

  float *z = (float *)bz.ptr, *pw = (float *)bpw.ptr, *pb = (float *)bpb.ptr;
  float *ew = (float *)bew.ptr, *eb = (float *)beb.ptr,
        *reg = (float *)breg.ptr;
  float *out = (float *)bout.ptr;
  for (uint32_t i = 0; i < LATENT_CH * patches; i++) z[i] = rng_float(&rng);
  for (uint32_t i = 0; i < LATENT_CH * LATENT_CH; i++)
    pw[i] = rng_float(&rng) * 0.25f;
  for (uint32_t i = 0; i < LATENT_CH; i++) pb[i] = rng_float(&rng);
  for (uint32_t i = 0; i < VVAE_DIM * LATENT_CH; i++)
    ew[i] = rng_float(&rng) * 0.25f;
  for (uint32_t i = 0; i < VVAE_DIM; i++) eb[i] = rng_float(&rng);
  for (uint32_t i = 0; i < REGISTERS * VVAE_DIM; i++) reg[i] = rng_float(&rng);

  VvaeEmbedParam p = {tl, hl, wl};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_emb);
  gpu_bind_buffer(&cmd, 0, &bz, 0);
  gpu_bind_buffer(&cmd, 1, &bpw, 0);
  gpu_bind_buffer(&cmd, 2, &bpb, 0);
  gpu_bind_buffer(&cmd, 3, &bew, 0);
  gpu_bind_buffer(&cmd, 4, &beb, 0);
  gpu_bind_buffer(&cmd, 5, &breg, 0);
  gpu_bind_buffer(&cmd, 6, &bout, 0);
  gpu_copy_bytes(&cmd, 7, &p, sizeof p);
  gpu_run_kernel(&cmd, rows, 1, 1, EMBED_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int ok = 1;
  for (uint32_t r = 0; r < rows; r++) {
    if (r < patches) {
      float zp[LATENT_CH];
      for (uint32_t c = 0; c < LATENT_CH; c++) {
        float acc = pb[c];
        for (uint32_t k = 0; k < LATENT_CH; k++)
          acc += pw[c * LATENT_CH + k] * z[k * patches + r];
        zp[c] = acc;
      }
      for (uint32_t c = 0; c < VVAE_DIM; c++) {
        float acc = eb[c];
        for (uint32_t k = 0; k < LATENT_CH; k++)
          acc += ew[c * LATENT_CH + k] * zp[k];
        if (fabsf(out[r * VVAE_DIM + c] - acc) > FP32_TOL) ok = 0;
      }
    } else if (r - patches < REGISTERS) {
      for (uint32_t c = 0; c < VVAE_DIM; c++)
        if (out[r * VVAE_DIM + c] != reg[(r - patches) * VVAE_DIM + c]) ok = 0;
    } else {
      for (uint32_t c = 0; c < VVAE_DIM; c++)
        if (out[r * VVAE_DIM + c] != 0.0f) ok = 0;
    }
  }
  check(ok, "vvae_embed matches CPU reference");

  gpu_free(&bz);
  gpu_free(&bpw);
  gpu_free(&bpb);
  gpu_free(&bew);
  gpu_free(&beb);
  gpu_free(&breg);
  gpu_free(&bout);
}

// Tests 32-head QK-Norm + 3D partial RoPE and 32-head FlashAttention.
static void test_vvae_heads_and_attend(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "minimax_h3_vvae_heads");
  int k_attend = gpu_find_kernel(gpu, "minimax_h3_vvae_attend");
  const uint32_t tl = 3, hl = 4, wl = 5;
  const uint32_t patches = tl * hl * wl, rows = patches + SUFFIX;
  const uint32_t padded = ceil_div(rows, TILE_M) * TILE_M;
  uint32_t rng = 0x22223333u;

  Buf bqkv =
      gpu_alloc(gpu, slot, (size_t)rows * QKV_PARTS * VVAE_DIM * sizeof(float));
  Buf binv = gpu_alloc(gpu, slot, (size_t)FREQ_PER_AXIS * sizeof(float));
  Buf bq = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2u);
  Buf bk = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2u);
  Buf bv = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2u);
  Buf bo = gpu_alloc(gpu, slot, (size_t)rows * VVAE_DIM * sizeof(float));

  float *qkv = (float *)bqkv.ptr, *inv = (float *)binv.ptr;
  for (size_t i = 0; i < (size_t)rows * QKV_PARTS * VVAE_DIM; i++)
    qkv[i] = rng_float(&rng) * 0.5f;
  for (uint32_t i = 0; i < FREQ_PER_AXIS; i++)
    inv[i] = (float)(1.0 / pow(ROPE_THETA, (double)i / (double)FREQ_PER_AXIS));

  VvaeHeadsParam hp = {rows, padded, tl, hl, wl, NORM_EPS};
  VvaeAttendParam ap = {rows, padded};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_heads);
  gpu_bind_buffer(&cmd, 0, &bqkv, 0);
  gpu_bind_buffer(&cmd, 1, &binv, 0);
  gpu_bind_buffer(&cmd, 2, &bq, 0);
  gpu_bind_buffer(&cmd, 3, &bk, 0);
  gpu_bind_buffer(&cmd, 4, &bv, 0);
  gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
  gpu_run_kernel(&cmd, QKV_PARTS * HEADS, padded, 1, HEADS_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &bq, 0);
  gpu_bind_buffer(&cmd, 1, &bk, 0);
  gpu_bind_buffer(&cmd, 2, &bv, 0);
  gpu_bind_buffer(&cmd, 3, &bo, 0);
  gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
  gpu_run_kernel(&cmd, ceil_div(rows, ATTN_GROUP_ROWS), HEADS, 1, ATTN_THREADS,
                 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // CPU reference for 3D split-half RoPE (24 pairs = 8t + 8h + 8w) + attention.
  float *ref_q = calloc((size_t)HEADS * rows * HEAD_DIM, sizeof(float));
  float *ref_k = calloc((size_t)HEADS * rows * HEAD_DIM, sizeof(float));
  float *ref_v = calloc((size_t)HEADS * rows * HEAD_DIM, sizeof(float));
  const _Float16 *gq = (const _Float16 *)bq.ptr;
  const _Float16 *gk = (const _Float16 *)bk.ptr;
  const _Float16 *gv = (const _Float16 *)bv.ptr;

  int heads_ok = 1;
  uint32_t hw = hl * wl;
  for (uint32_t r = 0; r < rows; r++) {
    float coords[3] = {0.0f, 0.0f, 0.0f};
    if (r < patches) {
      uint32_t pt = r / hw, rem = r % hw, py = rem / wl, px = rem % wl;
      coords[0] = ((float)pt + CELL_CENTER) / (float)tl * 2.0f - 1.0f;
      coords[1] = ((float)py + CELL_CENTER) / (float)hl * 2.0f - 1.0f;
      coords[2] = ((float)px + CELL_CENTER) / (float)wl * 2.0f - 1.0f;
    }
    for (uint32_t h = 0; h < HEADS; h++) {
      for (uint32_t kind = 0; kind < QKV_PARTS; kind++) {
        const float *src =
            qkv + r * QKV_PARTS * VVAE_DIM + (h * QKV_PARTS + kind) * HEAD_DIM;
        float vec[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) vec[d] = src[d];
        if (kind < 2u) {
          float ss = 0.0f;
          for (uint32_t d = 0; d < HEAD_DIM; d++) ss += vec[d] * vec[d];
          float inv_r = 1.0f / sqrtf(ss / (float)HEAD_DIM + NORM_EPS);
          for (uint32_t d = 0; d < HEAD_DIM; d++) vec[d] *= inv_r;
          if (r < patches) {
            for (uint32_t i = 0; i < ROT_PAIRS; i++) {
              uint32_t axis = i / FREQ_PER_AXIS, fi = i % FREQ_PER_AXIS;
              float ang = TWO_PI * coords[axis] * inv[fi];
              float cs = cosf(ang), sn = sinf(ang);
              float u0 = vec[i], u1 = vec[i + ROT_PAIRS];
              vec[i] = u0 * cs - u1 * sn;
              vec[i + ROT_PAIRS] = u1 * cs + u0 * sn;
            }
          }
        }
        float *rdst = (kind == 0u   ? ref_q
                       : kind == 1u ? ref_k
                                    : ref_v) +
                      (h * rows + r) * HEAD_DIM;
        const _Float16 *gdst = (kind == 0u   ? gq
                                : kind == 1u ? gk
                                             : gv) +
                               (h * padded + r) * HEAD_DIM;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
          rdst[d] = vec[d];
          if (fabsf((float)gdst[d] - vec[d]) > FP16_TOL) heads_ok = 0;
        }
      }
    }
  }
  check(heads_ok, "vvae_heads 32-head QK-Norm + 3D partial RoPE matches");

  const float *got_o = (const float *)bo.ptr;
  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  int attn_ok = 1;
  for (uint32_t h = 0; h < HEADS; h++) {
    for (uint32_t qr = 0; qr < rows; qr++) {
      float scores[rows];
      float mx = -INFINITY;
      for (uint32_t kr = 0; kr < rows; kr++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM; d++)
          dot += ref_q[(h * rows + qr) * HEAD_DIM + d] *
                 ref_k[(h * rows + kr) * HEAD_DIM + d];
        scores[kr] = dot * scale;
        if (scores[kr] > mx) mx = scores[kr];
      }
      float lsum = 0.0f;
      for (uint32_t kr = 0; kr < rows; kr++) {
        scores[kr] = expf(scores[kr] - mx);
        lsum += scores[kr];
      }
      for (uint32_t d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0f;
        for (uint32_t kr = 0; kr < rows; kr++)
          acc += scores[kr] * ref_v[(h * rows + kr) * HEAD_DIM + d];
        float got = got_o[qr * VVAE_DIM + h * HEAD_DIM + d];
        if (fabsf(got - acc / lsum) > FP16_TOL) attn_ok = 0;
      }
    }
  }
  check(attn_ok, "vvae_attend 32-head FlashAttention matches CPU reference");

  free(ref_q);
  free(ref_k);
  free(ref_v);
  gpu_free(&bqkv);
  gpu_free(&binv);
  gpu_free(&bq);
  gpu_free(&bk);
  gpu_free(&bv);
  gpu_free(&bo);
}

// Tests temporal-isolated 3D GroupNorm + SiLU and causal 3D im2col.
static void test_vvae_encoder_3d(Gpu *gpu, MemorySlot *slot) {
  int k_gn = gpu_find_kernel(gpu, "minimax_h3_vvae_gn_silu_3d");
  int k_im = gpu_find_kernel(gpu, "minimax_h3_vvae_im2col_3d_f16");
  uint32_t rng = 0x33334444u;

  // Temporal-isolated GroupNorm + SiLU across 5 distinct frames.
  const uint32_t frames = 5, h = 7, w = 9, hw = h * w, ch = 64, groups = 32;
  const uint32_t gc = ch / groups;
  Buf bx = gpu_alloc(gpu, slot, (size_t)frames * hw * ch * sizeof(float));
  Buf bw = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
  Buf bb = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
  Buf by = gpu_alloc(gpu, slot, (size_t)frames * hw * ch * sizeof(float));
  float *x = (float *)bx.ptr, *gw = (float *)bw.ptr, *gb = (float *)bb.ptr,
        *y = (float *)by.ptr;
  for (uint32_t f = 0; f < frames; f++)
    for (uint32_t i = 0; i < hw * ch; i++)
      x[f * hw * ch + i] = rng_float(&rng) + (float)f * 2.0f;
  for (uint32_t c = 0; c < ch; c++) {
    gw[c] = 0.5f + rng_float(&rng);
    gb[c] = rng_float(&rng);
  }

  VvaeGn3dParam gp = {hw, ch, groups, GN_EPS};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_gn);
  gpu_bind_buffer(&cmd, 0, &bx, 0);
  gpu_bind_buffer(&cmd, 1, &bw, 0);
  gpu_bind_buffer(&cmd, 2, &bb, 0);
  gpu_bind_buffer(&cmd, 3, &by, 0);
  gpu_copy_bytes(&cmd, 4, &gp, sizeof gp);
  gpu_run_kernel(&cmd, groups, frames, 1, GN_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int gn_ok = 1;
  for (uint32_t f = 0; f < frames; f++) {
    const float *xf = x + f * hw * ch;
    const float *yf = y + f * hw * ch;
    for (uint32_t g = 0; g < groups; g++) {
      float sum = 0.0f;
      uint32_t count = hw * gc;
      for (uint32_t pix = 0; pix < hw; pix++)
        for (uint32_t c = g * gc; c < (g + 1u) * gc; c++)
          sum += xf[pix * ch + c];
      float mean = sum / (float)count;
      float sq = 0.0f;
      for (uint32_t pix = 0; pix < hw; pix++)
        for (uint32_t c = g * gc; c < (g + 1u) * gc; c++) {
          float d = xf[pix * ch + c] - mean;
          sq += d * d;
        }
      float inv = 1.0f / sqrtf(sq / (float)count + GN_EPS);
      for (uint32_t pix = 0; pix < hw; pix++)
        for (uint32_t c = g * gc; c < (g + 1u) * gc; c++) {
          float v = gw[c] * ((xf[pix * ch + c] - mean) * inv) + gb[c];
          float ref = v / (1.0f + expf(-v));
          if (fabsf(yf[pix * ch + c] - ref) > FP32_TOL) gn_ok = 0;
        }
    }
  }
  check(gn_ok, "vvae_gn_silu_3d temporal-isolated GroupNorm + SiLU matches");

  // Causal-zero temporal + reflect spatial 3D im2col (Downsample3D s_t=2,
  // s_s=2, pad_h=0, pad_w=0, front_pad=2) and single-frame T=1 edge case.
  const uint32_t in_t = 5, in_h = 6, in_w = 8, cin = 7;
  const uint32_t out_h = in_h / 2u, out_w = in_w / 2u;
  const uint32_t k_raw = 3u * 3u * 3u * cin;
  const uint32_t k_pad = ceil_div(k_raw, TILE_M) * TILE_M;
  const uint32_t m_pad = ceil_div(out_h * out_w, TILE_M) * TILE_M;
  Buf bim = gpu_alloc(gpu, slot, (size_t)m_pad * k_pad * sizeof(_Float16));
  const _Float16 *im = (const _Float16 *)bim.ptr;

  int im_ok = 1;
  for (uint32_t ot = 0; ot < 3u; ot++) {
    VvaeIm2Col3dParam ip = {in_t, in_h, in_w, cin, out_w, ot, 0, out_h,
                            3,    3,    2,    2,   2,     0,  0, k_pad};
    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_im);
    gpu_bind_buffer(&cmd, 0, &bx, 0);
    gpu_bind_buffer(&cmd, 1, &bim, 0);
    gpu_copy_bytes(&cmd, 2, &ip, sizeof ip);
    gpu_run_kernel(&cmd, ceil_div(k_pad, IM2COL_TX), ceil_div(m_pad, IM2COL_TY),
                   1, IM2COL_TX, IM2COL_TY, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    for (uint32_t row = 0; row < m_pad; row++) {
      for (uint32_t col = 0; col < k_pad; col++) {
        _Float16 want = (_Float16)0.0f;
        if (row < out_h * out_w && col < k_raw) {
          uint32_t oy = row / out_w, ox = row % out_w;
          uint32_t tap = col / cin, c = col % cin;
          uint32_t tk = tap / 9u, st = tap % 9u, ky = st / 3u, kx = st % 3u;
          int it = (int)(ot * 2u + tk) - 2;
          if (it >= 0 && it < (int)in_t) {
            uint32_t iy = reflect_coord((int)(oy * 2u + ky), in_h);
            uint32_t ix = reflect_coord((int)(ox * 2u + kx), in_w);
            want =
                (_Float16)x[(((uint32_t)it * in_h + iy) * in_w + ix) * cin + c];
          }
        }
        if (im[row * k_pad + col] != want) im_ok = 0;
      }
    }
  }
  check(im_ok, "vvae_im2col_3d_f16 causal temporal + reflect spatial matches");

  gpu_free(&bim);
  gpu_free(&bx);
  gpu_free(&bw);
  gpu_free(&bb);
  gpu_free(&by);
}

// Compiles the Metal library and runs all MiniMax-H3 Video VAE kernel tests.
int main(void) {
  Memory *mem = memory_new();
  MemorySlot *slot = memory_shared(mem);
  size_t src_len = 0;
  const char *src = kernel_library_source(&src_len);
  Gpu *gpu = gpu_new(src, src_len);
  check(gpu != NULL, "gpu opened");

  test_vvae_embed(gpu, slot);
  test_vvae_heads_and_attend(gpu, slot);
  test_vvae_encoder_3d(gpu, slot);
  return failures != 0;
}
