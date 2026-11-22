#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/image_vae.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tests MiniMax-H3-Image VAE spatial tiling arithmetic and all 2D ViT decoder
// GPU kernels against CPU references without requiring a checkpoint.

// 2D ViT decoder geometry matching src/kernels/minimax_h3_vae.metal.
static constexpr int LATENT_CH = 24, VAE_DIM = 1024, HEADS = 16, HEAD_DIM = 64,
                     ROT_PAIRS = 16, FREQ_PER_AXIS = 8, REGISTERS = 4,
                     SUFFIX = REGISTERS + 1;

// Tile padding, threadgroup sizes and tolerances.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          NORM_THREADS = 256, EMBED_THREADS = 256,
                          HEADS_THREADS = 32, ELEM_THREADS = 32;
static constexpr float NORM_EPS = 1e-5f, TWO_PI = 6.28318530717958647692f,
                       FP32_TOL = 1e-4f, FP16_TOL = 2e-2f;

// Parameters passed to minimax_h3_x_f16.
typedef struct {
  // Active rows, inner width and input row stride.
  uint32_t m, k, lda;
} F16XParam;

// Parameters passed to minimax_h3_mm_f16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} F16MMParam;

// Parameters passed to minimax_h3_add_scaled.
typedef struct {
  // Row count and row width.
  uint32_t rows, dim;
} AddScaledParam;

// Parameters passed to minimax_h3_swiglu_chunked.
typedef struct {
  // Row count and half-width inner dimension.
  uint32_t rows, inner;
} SwigluChunkedParam;

// Parameters passed to minimax_h3_layernorm.
typedef struct {
  // Row width and epsilon.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} LayerNormParam;

// Parameters passed to minimax_h3_vae_embed.
typedef struct {
  // Latent tile height and width.
  uint32_t h_lat, w_lat;
} VaeEmbedParam;

// Parameters passed to minimax_h3_vae_heads.
typedef struct {
  // Sequence length, padded head stride, latent height and latent width.
  uint32_t rows, padded, h_lat, w_lat;
  // Per-head non-affine QK RMSNorm epsilon.
  float eps;
} VaeHeadsParam;

// Parameters passed to minimax_h3_vae_attend.
typedef struct {
  // Sequence length and padded head stride.
  uint32_t rows, padded;
} VaeAttendParam;

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

// Tests split_tiles invariants across single-tile and multi-tile sizes.
static void test_split_tiles(void) {
  MinimaxH3VaeTilePlan p256 = minimax_h3_image_vae_split_tiles(256);
  check(p256.count == 1 && p256.start[0] == 0 && p256.len[0] == 256,
        "256 px fits in one tile with no overlap");

  MinimaxH3VaeTilePlan p1024 = minimax_h3_image_vae_split_tiles(1024);
  int ok1024 = p1024.count == 5 && p1024.start[0] == 0 &&
               p1024.start[4] + p1024.len[4] == 1024;
  for (int i = 0; i < 4; i++) ok1024 &= p1024.overlap[i] == 64;
  check(ok1024, "1024 px splits into five 256 px tiles with 64 px overlaps");

  static const int sizes[] = {128, 320, 512, 768, 1088, 1440, 2048};
  int all_ok = 1;
  for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
    int len = sizes[s];
    MinimaxH3VaeTilePlan p = minimax_h3_image_vae_split_tiles(len);
    all_ok &= p.count >= 1 && p.start[0] == 0 &&
              p.start[p.count - 1] + p.len[p.count - 1] == len;
    for (int i = 0; i < p.count; i++)
      all_ok &= (p.start[i] % 16 == 0) && (p.len[i] % 16 == 0);
    for (int i = 0; i < p.count - 1; i++) {
      all_ok &= p.overlap[i] >= 64 && (p.overlap[i] % 16 == 0);
      all_ok &= p.start[i] + p.len[i] - p.overlap[i] == p.start[i + 1];
    }
  }
  check(all_ok,
        "spatial tile plans cover every axis cleanly with 16 px alignment");
}

// Tests native FP16 matmul with and without bias.
static void test_f16_matmul(Gpu *gpu, MemorySlot *slot) {
  int k_xf16 = gpu_find_kernel(gpu, "minimax_h3_x_f16");
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_f16");
  static const int ms[] = {1, 17, 65, 261};
  int k = 128, n = 192;
  uint32_t rng = 0x11223344u;

  Buf w = gpu_alloc(gpu, slot, (size_t)n * (size_t)k * 2);
  Buf bias = gpu_alloc(gpu, slot, (size_t)n * 4);
  _Float16 *wh = (_Float16 *)w.ptr;
  float *bf = (float *)bias.ptr;
  for (int i = 0; i < n * k; i++) wh[i] = (_Float16)(rng_float(&rng) * 0.5f);
  for (int c = 0; c < n; c++) bf[c] = rng_float(&rng);

  int ok_plain = 1, ok_bias = 1;
  for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
    int m = ms[mi];
    uint32_t padded = ((uint32_t)m + TILE_M - 1) / TILE_M * TILE_M;
    Buf x = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * 4);
    Buf xh = gpu_alloc(gpu, slot, (size_t)padded * (size_t)k * 2);
    Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * 4);
    for (int i = 0; i < m * k; i++) ((float *)x.ptr)[i] = rng_float(&rng);

    for (int has_bias = 0; has_bias < 2; has_bias++) {
      F16XParam xp = {(uint32_t)m, (uint32_t)k, (uint32_t)k};
      F16MMParam mp = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                       (uint32_t)has_bias};
      GpuCommandBuffer cmd = gpu_begin(gpu);
      gpu_select_kernel(&cmd, k_xf16);
      gpu_bind_buffer(&cmd, 0, &x, 0);
      gpu_bind_buffer(&cmd, 1, &xh, 0);
      gpu_copy_bytes(&cmd, 2, &xp, sizeof xp);
      gpu_run_kernel(&cmd, ((uint32_t)k / 4u + ELEM_THREADS - 1) / ELEM_THREADS,
                     padded, 1, ELEM_THREADS, 1, 1);

      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w, 0);
      gpu_bind_buffer(&cmd, 1, &xh, 0);
      gpu_bind_buffer(&cmd, 2, &bias, 0);
      gpu_bind_buffer(&cmd, 3, &y, 0);
      gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
      gpu_run_kernel(&cmd, ((uint32_t)n + TILE_N - 1) / TILE_N, padded / TILE_M,
                     1, MM_THREADS, 1, 1);
      gpu_commit(&cmd);
      gpu_sync(gpu);

      const float *xf = (const float *)x.ptr, *yf = (const float *)y.ptr;
      for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
          float sum = 0.0f;
          for (int j = 0; j < k; j++)
            sum += (float)(_Float16)xf[r * k + j] * (float)wh[c * k + j];
          float ref = sum + (has_bias ? bf[c] : 0.0f);
          if (!isfinite(yf[r * n + c]) ||
              fabsf(yf[r * n + c] - ref) > FP16_TOL) {
            if (has_bias)
              ok_bias = 0;
            else
              ok_plain = 0;
          }
        }
      }
    }
    gpu_free(&y);
    gpu_free(&xh);
    gpu_free(&x);
  }
  gpu_free(&bias);
  gpu_free(&w);
  check(ok_plain, "native FP16 matmul without bias matches CPU reference");
  check(ok_bias, "native FP16 matmul with bias matches CPU reference");
}

// Tests LayerNorm, chunked SwiGLU, scaled residual add and VAE tile embedding.
static void test_norms_and_embed(Gpu *gpu, MemorySlot *slot) {
  int k_ln = gpu_find_kernel(gpu, "minimax_h3_layernorm");
  int k_sw = gpu_find_kernel(gpu, "minimax_h3_swiglu_chunked");
  int k_sc = gpu_find_kernel(gpu, "minimax_h3_add_scaled");
  int k_emb = gpu_find_kernel(gpu, "minimax_h3_vae_embed");
  uint32_t rng = 0x55667788u;

  // LayerNorm + chunked SwiGLU + add_scaled.
  uint32_t rows = 5, inner = 128, dim = 2 * inner;
  Buf x = gpu_alloc(gpu, slot, (size_t)rows * dim * 4);
  Buf w = gpu_alloc(gpu, slot, (size_t)dim * 4);
  Buf b = gpu_alloc(gpu, slot, (size_t)dim * 4);
  Buf ln = gpu_alloc(gpu, slot, (size_t)rows * dim * 4);
  Buf sw = gpu_alloc(gpu, slot, (size_t)rows * inner * 4);
  Buf scale = gpu_alloc(gpu, slot, (size_t)dim * 4);
  for (uint32_t i = 0; i < rows * dim; i++)
    ((float *)x.ptr)[i] = rng_float(&rng) * 2.0f;
  for (uint32_t i = 0; i < dim; i++) {
    ((float *)w.ptr)[i] = 0.5f + rng_float(&rng);
    ((float *)b.ptr)[i] = rng_float(&rng) * 0.25f;
    ((float *)scale.ptr)[i] = rng_float(&rng);
  }
  float *x_copy = xmalloc((size_t)rows * dim * 4);
  memcpy(x_copy, x.ptr, (size_t)rows * dim * 4);

  LayerNormParam lnp = {dim, NORM_EPS};
  SwigluChunkedParam swp = {rows, inner};
  AddScaledParam scp = {rows, dim};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_ln);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &w, 0);
  gpu_bind_buffer(&cmd, 2, &b, 0);
  gpu_bind_buffer(&cmd, 3, &ln, 0);
  gpu_copy_bytes(&cmd, 4, &lnp, sizeof lnp);
  gpu_run_kernel(&cmd, rows, 1, 1, NORM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_sw);
  gpu_bind_buffer(&cmd, 0, &ln, 0);
  gpu_bind_buffer(&cmd, 1, &sw, 0);
  gpu_copy_bytes(&cmd, 2, &swp, sizeof swp);
  gpu_run_kernel(&cmd, (inner + ELEM_THREADS - 1) / ELEM_THREADS, rows, 1,
                 ELEM_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_sc);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &ln, 0);
  gpu_bind_buffer(&cmd, 2, &scale, 0);
  gpu_copy_bytes(&cmd, 3, &scp, sizeof scp);
  gpu_run_kernel(&cmd, (dim + ELEM_THREADS - 1) / ELEM_THREADS, rows, 1,
                 ELEM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int ln_ok = 1, sw_ok = 1, sc_ok = 1;
  const float *wf = (const float *)w.ptr, *bf = (const float *)b.ptr;
  const float *lnf = (const float *)ln.ptr, *swf = (const float *)sw.ptr;
  const float *scf = (const float *)scale.ptr, *xf = (const float *)x.ptr;
  for (uint32_t r = 0; r < rows; r++) {
    float mean = 0.0f;
    for (uint32_t c = 0; c < dim; c++) mean += x_copy[r * dim + c];
    mean /= (float)dim;
    float var = 0.0f;
    for (uint32_t c = 0; c < dim; c++) {
      float d = x_copy[r * dim + c] - mean;
      var += d * d;
    }
    float inv = 1.0f / sqrtf(var / (float)dim + NORM_EPS);
    for (uint32_t c = 0; c < dim; c++) {
      float ref_ln = wf[c] * ((x_copy[r * dim + c] - mean) * inv) + bf[c];
      if (fabsf(lnf[r * dim + c] - ref_ln) > FP32_TOL) ln_ok = 0;
      float ref_sc = x_copy[r * dim + c] + scf[c] * ref_ln;
      if (fabsf(xf[r * dim + c] - ref_sc) > FP32_TOL) sc_ok = 0;
    }
    for (uint32_t c = 0; c < inner; c++) {
      float g = lnf[r * dim + c], u = lnf[r * dim + inner + c];
      float ref_sw = g / (1.0f + expf(-g)) * u;
      if (fabsf(swf[r * inner + c] - ref_sw) > FP32_TOL) sw_ok = 0;
    }
  }
  free(x_copy);
  gpu_free(&scale);
  gpu_free(&sw);
  gpu_free(&ln);
  gpu_free(&b);
  gpu_free(&w);
  gpu_free(&x);
  check(ln_ok, "VAE LayerNorm matches CPU reference");
  check(sw_ok, "VAE chunked SwiGLU matches CPU reference");
  check(sc_ok, "VAE channel-scaled residual add matches CPU reference");

  // VAE tile embedding: post_quant_conv + x_embedder + 4 registers + 1 zero.
  uint32_t h_lat = 3, w_lat = 4, patches = h_lat * w_lat,
           seq = patches + SUFFIX;
  Buf z = gpu_alloc(gpu, slot, (size_t)LATENT_CH * patches * 4);
  Buf pqc_w = gpu_alloc(gpu, slot, (size_t)LATENT_CH * LATENT_CH * 4);
  Buf pqc_b = gpu_alloc(gpu, slot, (size_t)LATENT_CH * 4);
  Buf emb_w = gpu_alloc(gpu, slot, (size_t)VAE_DIM * LATENT_CH * 4);
  Buf emb_b = gpu_alloc(gpu, slot, (size_t)VAE_DIM * 4);
  Buf reg = gpu_alloc(gpu, slot, (size_t)REGISTERS * VAE_DIM * 4);
  Buf out = gpu_alloc(gpu, slot, (size_t)seq * VAE_DIM * 4);
  for (size_t i = 0; i < LATENT_CH * patches; i++)
    ((float *)z.ptr)[i] = rng_float(&rng);
  for (size_t i = 0; i < LATENT_CH * LATENT_CH; i++)
    ((float *)pqc_w.ptr)[i] = rng_float(&rng) * 0.2f;
  for (size_t i = 0; i < LATENT_CH; i++)
    ((float *)pqc_b.ptr)[i] = rng_float(&rng) * 0.1f;
  for (size_t i = 0; i < (size_t)VAE_DIM * LATENT_CH; i++)
    ((float *)emb_w.ptr)[i] = rng_float(&rng) * 0.2f;
  for (size_t i = 0; i < VAE_DIM; i++)
    ((float *)emb_b.ptr)[i] = rng_float(&rng) * 0.1f;
  for (size_t i = 0; i < (size_t)REGISTERS * VAE_DIM; i++)
    ((float *)reg.ptr)[i] = rng_float(&rng);

  VaeEmbedParam ep = {h_lat, w_lat};
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_emb);
  gpu_bind_buffer(&cmd, 0, &z, 0);
  gpu_bind_buffer(&cmd, 1, &pqc_w, 0);
  gpu_bind_buffer(&cmd, 2, &pqc_b, 0);
  gpu_bind_buffer(&cmd, 3, &emb_w, 0);
  gpu_bind_buffer(&cmd, 4, &emb_b, 0);
  gpu_bind_buffer(&cmd, 5, &reg, 0);
  gpu_bind_buffer(&cmd, 6, &out, 0);
  gpu_copy_bytes(&cmd, 7, &ep, sizeof ep);
  gpu_run_kernel(&cmd, seq, 1, 1, EMBED_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int emb_ok = 1;
  const float *zf = (const float *)z.ptr, *pw = (const float *)pqc_w.ptr,
              *pb = (const float *)pqc_b.ptr, *ew = (const float *)emb_w.ptr,
              *eb = (const float *)emb_b.ptr, *rf = (const float *)reg.ptr,
              *of = (const float *)out.ptr;
  for (uint32_t r = 0; r < seq; r++) {
    if (r < patches) {
      float zp[LATENT_CH];
      for (int c = 0; c < LATENT_CH; c++) {
        float acc = pb[c];
        for (int k = 0; k < LATENT_CH; k++)
          acc += pw[c * LATENT_CH + k] * zf[k * patches + r];
        zp[c] = acc;
      }
      for (int c = 0; c < VAE_DIM; c++) {
        float acc = eb[c];
        for (int k = 0; k < LATENT_CH; k++)
          acc += ew[c * LATENT_CH + k] * zp[k];
        if (fabsf(of[r * VAE_DIM + c] - acc) > FP32_TOL) emb_ok = 0;
      }
    } else if (r < patches + REGISTERS) {
      for (int c = 0; c < VAE_DIM; c++)
        if (of[r * VAE_DIM + c] != rf[(r - patches) * VAE_DIM + c]) emb_ok = 0;
    } else {
      for (int c = 0; c < VAE_DIM; c++)
        if (of[r * VAE_DIM + c] != 0.0f) emb_ok = 0;
    }
  }
  gpu_free(&out);
  gpu_free(&reg);
  gpu_free(&emb_b);
  gpu_free(&emb_w);
  gpu_free(&pqc_b);
  gpu_free(&pqc_w);
  gpu_free(&z);
  check(
      emb_ok,
      "VAE tile embedding runs post_quant_conv, x_embedder and suffix tokens");
}

// Tests head-interleaved QKV split, non-affine QK-Norm, 2D partial split-half
// RoPE and bidirectional 16-head FlashAttention.
static void test_vae_heads_and_attention(Gpu *gpu, MemorySlot *slot) {
  int k_heads = gpu_find_kernel(gpu, "minimax_h3_vae_heads");
  int k_attend = gpu_find_kernel(gpu, "minimax_h3_vae_attend");
  // 10x13 = 130 patches + 5 suffix tokens = 135 rows (interior, boundary and
  // tail 64-key tiles).
  int h_lat = 10, w_lat = 13, patches = h_lat * w_lat, seq = patches + SUFFIX;
  uint32_t padded = ((uint32_t)seq + TILE_M - 1) / TILE_M * TILE_M;
  uint32_t rng = 0x99aabbccu;

  Buf qkv = gpu_alloc(gpu, slot, (size_t)seq * 3 * VAE_DIM * 4);
  Buf inv = gpu_alloc(gpu, slot, (size_t)FREQ_PER_AXIS * 4);
  Buf qh = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2);
  Buf kh = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2);
  Buf vh = gpu_alloc(gpu, slot, (size_t)HEADS * padded * HEAD_DIM * 2);
  Buf out = gpu_alloc(gpu, slot, (size_t)seq * VAE_DIM * 4);
  for (int i = 0; i < seq * 3 * VAE_DIM; i++)
    ((float *)qkv.ptr)[i] = rng_float(&rng);
  for (int i = 0; i < FREQ_PER_AXIS; i++)
    ((float *)inv.ptr)[i] =
        (float)(1.0 / pow(100.0, (double)i / (double)FREQ_PER_AXIS));

  VaeHeadsParam hp = {(uint32_t)seq, padded, (uint32_t)h_lat, (uint32_t)w_lat,
                      NORM_EPS};
  VaeAttendParam ap = {(uint32_t)seq, padded};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_heads);
  gpu_bind_buffer(&cmd, 0, &qkv, 0);
  gpu_bind_buffer(&cmd, 1, &inv, 0);
  gpu_bind_buffer(&cmd, 2, &qh, 0);
  gpu_bind_buffer(&cmd, 3, &kh, 0);
  gpu_bind_buffer(&cmd, 4, &vh, 0);
  gpu_copy_bytes(&cmd, 5, &hp, sizeof hp);
  gpu_run_kernel(&cmd, (uint32_t)(3 * HEADS), padded, 1, HEADS_THREADS, 1, 1);

  gpu_select_kernel(&cmd, k_attend);
  gpu_bind_buffer(&cmd, 0, &qh, 0);
  gpu_bind_buffer(&cmd, 1, &kh, 0);
  gpu_bind_buffer(&cmd, 2, &vh, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &ap, sizeof ap);
  gpu_run_kernel(&cmd, (padded / TILE_M) * (uint32_t)HEADS, 1, 1, MM_THREADS, 1,
                 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  const float *qkvf = (const float *)qkv.ptr, *invf = (const float *)inv.ptr;
  const _Float16 *qh16 = (const _Float16 *)qh.ptr;
  const _Float16 *kh16 = (const _Float16 *)kh.ptr;
  const _Float16 *vh16 = (const _Float16 *)vh.ptr;
  int rope_ok = 1;
  static const int check_rows[] = {0, 13, 63, 71, 127, 130, 134};
  for (size_t ri = 0; ri < sizeof check_rows / sizeof check_rows[0]; ri++) {
    int r = check_rows[ri];
    for (int h = 0; h < HEADS; h += 5) {
      for (int kind = 0; kind < 2; kind++) {
        const float *src = qkvf + (size_t)r * (3 * VAE_DIM) +
                           (size_t)(h * 3 + kind) * HEAD_DIM;
        float ss = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) ss += src[d] * src[d];
        float inv_r = 1.0f / sqrtf(ss / (float)HEAD_DIM + NORM_EPS);
        float normed[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) normed[d] = src[d] * inv_r;
        float expected[HEAD_DIM];
        memcpy(expected, normed, sizeof expected);
        if (r < patches) {
          int py = r / w_lat, px = r % w_lat;
          float cy = (((float)py + 0.5f) / (float)h_lat) * 2.0f - 1.0f;
          float cx = (((float)px + 0.5f) / (float)w_lat) * 2.0f - 1.0f;
          for (int p = 0; p < ROT_PAIRS; p++) {
            float coord = p < FREQ_PER_AXIS ? cy : cx;
            int fi = p < FREQ_PER_AXIS ? p : p - FREQ_PER_AXIS;
            float angle = TWO_PI * coord * invf[fi];
            float c = cosf(angle), s = sinf(angle);
            expected[p] = normed[p] * c - normed[p + ROT_PAIRS] * s;
            expected[p + ROT_PAIRS] = normed[p + ROT_PAIRS] * c + normed[p] * s;
          }
        }
        const _Float16 *dst = (kind == 0 ? qh16 : kh16) +
                              ((size_t)h * padded + (size_t)r) * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++)
          if (fabsf((float)dst[d] - expected[d]) > 1e-2f) rope_ok = 0;
      }
    }
  }
  for (uint32_t r = (uint32_t)seq; r < padded; r++)
    if ((float)qh16[r * HEAD_DIM] != 0.0f ||
        (float)kh16[r * HEAD_DIM] != 0.0f || (float)vh16[r * HEAD_DIM] != 0.0f)
      rope_ok = 0;
  check(rope_ok,
        "VAE QKV split, non-affine QK-Norm and 2D partial RoPE match CPU");

  int attn_ok = 1;
  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  float scores[256];
  const float *outf = (const float *)out.ptr;
  for (int h = 0; h < HEADS; h += 5) {
    for (size_t ri = 0; ri < sizeof check_rows / sizeof check_rows[0]; ri++) {
      int r = check_rows[ri];
      const _Float16 *qr = qh16 + ((size_t)h * padded + (size_t)r) * HEAD_DIM;
      float max_s = -INFINITY;
      for (int j = 0; j < seq; j++) {
        const _Float16 *kr = kh16 + ((size_t)h * padded + (size_t)j) * HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) dot += (float)qr[d] * (float)kr[d];
        scores[j] = dot * scale;
        if (scores[j] > max_s) max_s = scores[j];
      }
      float denom = 0.0f;
      for (int j = 0; j < seq; j++) {
        scores[j] = expf(scores[j] - max_s);
        denom += scores[j];
      }
      const float *got_row = outf + (size_t)r * VAE_DIM + (size_t)h * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq; j++) {
          const _Float16 *vr =
              vh16 + ((size_t)h * padded + (size_t)j) * HEAD_DIM;
          acc += scores[j] * (float)vr[d];
        }
        float ref = acc / denom;
        if (!isfinite(got_row[d]) || fabsf(got_row[d] - ref) > FP16_TOL)
          attn_ok = 0;
      }
    }
  }
  check(attn_ok,
        "VAE bidirectional 16-head FlashAttention matches CPU reference");

  gpu_free(&out);
  gpu_free(&vh);
  gpu_free(&kh);
  gpu_free(&qh);
  gpu_free(&inv);
  gpu_free(&qkv);
}

// Parameters passed to minimax_h3_groupnorm_silu.
typedef struct {
  // Spatial pixel count, channels and group count.
  uint32_t hw, channels, groups;
  // Normalization epsilon.
  float eps;
} GroupNormParam;

// Parameters passed to minimax_h3_im2col_f16.
typedef struct {
  // Input spatial height, width and channels.
  uint32_t in_h, in_w, cin;
  // Output width, starting output row and strip height.
  uint32_t out_w, out_y0, strip_h;
  // Kernel side, stride, top/left pad and padded K width.
  uint32_t ksize, stride, pad_t, pad_l, k_padded;
} Im2ColParam;

// Reflects coordinate c into [0, limit - 1] matching PyTorch reflect padding.
static int reflect_coord(int c, int limit) {
  if (limit <= 1) return 0;
  if (c < 0) c = -c;
  if (c >= limit) c = 2 * limit - 2 - c;
  if (c < 0) c = 0;
  if (c >= limit) c = limit - 1;
  return c;
}

// Tests VAE encoder kernels: minimax_h3_groupnorm_silu and
// minimax_h3_im2col_f16 paired with minimax_h3_mm_f16 (3x3 reflect conv,
// stride-2 downsample, and 48-channel padded-N 1x1 conv).
static void test_encoder_kernels(Gpu *gpu, MemorySlot *slot) {
  int k_gn = gpu_find_kernel(gpu, "minimax_h3_groupnorm_silu");
  int k_i2c = gpu_find_kernel(gpu, "minimax_h3_im2col_f16");
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_f16");
  uint32_t rng = 0x51a2b3c4u;

  // 1. GroupNorm(32, 128, eps=1e-6) + SiLU.
  int hw = 17 * 13, ch = 128, groups = 32;
  float gn_eps = 1e-6f;
  Buf gx = gpu_alloc(gpu, slot, (size_t)hw * (size_t)ch * sizeof(float));
  Buf gw = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
  Buf gb = gpu_alloc(gpu, slot, (size_t)ch * sizeof(float));
  Buf gy = gpu_alloc(gpu, slot, (size_t)hw * (size_t)ch * sizeof(float));
  float *gxp = (float *)gx.ptr, *gwp = (float *)gw.ptr, *gbp = (float *)gb.ptr;
  for (int i = 0; i < hw * ch; i++) gxp[i] = 2.0f * rng_float(&rng) + 0.3f;
  for (int c = 0; c < ch; c++) {
    gwp[c] = 0.5f + rng_float(&rng);
    gbp[c] = 0.25f * rng_float(&rng);
  }
  GroupNormParam gnp = {(uint32_t)hw, (uint32_t)ch, (uint32_t)groups, gn_eps};
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_gn);
  gpu_bind_buffer(&cmd, 0, &gx, 0);
  gpu_bind_buffer(&cmd, 1, &gw, 0);
  gpu_bind_buffer(&cmd, 2, &gb, 0);
  gpu_bind_buffer(&cmd, 3, &gy, 0);
  gpu_copy_bytes(&cmd, 4, &gnp, sizeof gnp);
  gpu_run_kernel(&cmd, (uint32_t)groups, 1, 1, NORM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  const float *gyp = (const float *)gy.ptr;
  float worst_gn = 0.0f;
  int gc = ch / groups;
  for (int g = 0; g < groups; g++) {
    double sum = 0.0, sq = 0.0;
    for (int p = 0; p < hw; p++)
      for (int k = 0; k < gc; k++) sum += (double)gxp[p * ch + g * gc + k];
    double mean = sum / (double)(hw * gc);
    for (int p = 0; p < hw; p++)
      for (int k = 0; k < gc; k++) {
        double d = (double)gxp[p * ch + g * gc + k] - mean;
        sq += d * d;
      }
    double inv = 1.0 / sqrt(sq / (double)(hw * gc) + (double)gn_eps);
    for (int p = 0; p < hw; p++)
      for (int k = 0; k < gc; k++) {
        int c = g * gc + k;
        float v =
            gwp[c] * (float)(((double)gxp[p * ch + c] - mean) * inv) + gbp[c];
        float ref = v / (1.0f + expf(-v));
        float d = fabsf(gyp[p * ch + c] - ref);
        if (d > worst_gn) worst_gn = d;
      }
  }
  check(worst_gn < FP32_TOL,
        "minimax_h3_groupnorm_silu matches CPU GroupNorm + SiLU reference");
  gpu_free(&gy), gpu_free(&gb), gpu_free(&gw), gpu_free(&gx);

  // 2. Reflect-padded 3x3 conv (stride=1, pad=1), stride-2 Downsample3D
  //    (stride=2, pad=0), and padded-N 48-channel 1x1 conv (quant_conv).
  struct {
    int in_h, in_w, cin, cout, ksize, stride, pad;
    const char *label;
  } cases[] = {
      {8, 10, 3, 64, 3, 1, 1,
       "3x3 reflect-padded conv (cin=3, k padded to 64) matches CPU"},
      {8, 10, 16, 64, 3, 2, 0,
       "stride-2 Downsample3D conv (right/bottom reflect pad) matches CPU"},
      {6, 8, 48, 48, 1, 1, 0,
       "1x1 quant_conv (48->48 padded to 64) matches CPU"},
  };

  for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
    int ih = cases[ci].in_h, iw = cases[ci].in_w;
    int cin = cases[ci].cin, cout = cases[ci].cout;
    int ks = cases[ci].ksize, st = cases[ci].stride, pd = cases[ci].pad;
    int oh = ih / st, ow = iw / st;
    int k_raw = ks * ks * cin;
    int k_pad = (k_raw + 63) / 64 * 64;
    int n_pad = (cout + 63) / 64 * 64;
    uint32_t m = (uint32_t)(oh * ow);
    uint32_t m_pad = (m + TILE_M - 1) / TILE_M * TILE_M;

    Buf x = gpu_alloc(gpu, slot, (size_t)ih * (size_t)iw * (size_t)cin * 4);
    Buf w = gpu_alloc(gpu, slot, (size_t)n_pad * (size_t)k_pad * 2);
    Buf b = gpu_alloc(gpu, slot, (size_t)cout * 4);
    Buf im2col = gpu_alloc(gpu, slot, (size_t)m_pad * (size_t)k_pad * 2);
    Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)cout * 4);

    float *xp = (float *)x.ptr, *bp = (float *)b.ptr;
    _Float16 *wp = (_Float16 *)w.ptr;
    memset(wp, 0, (size_t)n_pad * (size_t)k_pad * 2);
    for (int i = 0; i < ih * iw * cin; i++) xp[i] = 0.5f * rng_float(&rng);
    for (int co = 0; co < cout; co++) {
      bp[co] = 0.25f * rng_float(&rng);
      for (int j = 0; j < k_raw; j++)
        wp[co * k_pad + j] = (_Float16)(0.4f * rng_float(&rng));
    }

    Im2ColParam ip = {
        (uint32_t)ih,   (uint32_t)iw, (uint32_t)cin, (uint32_t)ow, 0,
        (uint32_t)oh,   (uint32_t)ks, (uint32_t)st,  (uint32_t)pd, (uint32_t)pd,
        (uint32_t)k_pad};
    F16MMParam mp = {m, (uint32_t)cout, (uint32_t)k_pad, (uint32_t)cout, 1};
    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_i2c);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &im2col, 0);
    gpu_copy_bytes(&cmd, 2, &ip, sizeof ip);
    gpu_run_kernel(&cmd, ((uint32_t)k_pad + ELEM_THREADS - 1) / ELEM_THREADS,
                   m_pad, 1, ELEM_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_mm);
    gpu_bind_buffer(&cmd, 0, &w, 0);
    gpu_bind_buffer(&cmd, 1, &im2col, 0);
    gpu_bind_buffer(&cmd, 2, &b, 0);
    gpu_bind_buffer(&cmd, 3, &y, 0);
    gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
    gpu_run_kernel(&cmd, ((uint32_t)cout + TILE_N - 1) / TILE_N, m_pad / TILE_M,
                   1, MM_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const float *yp = (const float *)y.ptr;
    float worst = 0.0f;
    for (int oy = 0; oy < oh; oy++) {
      for (int ox = 0; ox < ow; ox++) {
        for (int co = 0; co < cout; co++) {
          float acc = bp[co];
          for (int ky = 0; ky < ks; ky++) {
            int iy = reflect_coord(oy * st + ky - pd, ih);
            for (int kx = 0; kx < ks; kx++) {
              int ix = reflect_coord(ox * st + kx - pd, iw);
              int tap = ky * ks + kx;
              for (int c = 0; c < cin; c++) {
                float xv = (float)(_Float16)xp[(iy * iw + ix) * cin + c];
                float wv = (float)wp[co * k_pad + tap * cin + c];
                acc += xv * wv;
              }
            }
          }
          float d = fabsf(yp[(oy * ow + ox) * cout + co] - acc);
          if (d > worst) worst = d;
        }
      }
    }
    check(worst < FP16_TOL, cases[ci].label);
    gpu_free(&y), gpu_free(&im2col), gpu_free(&b), gpu_free(&w), gpu_free(&x);
  }
}

int main(void) {
  test_split_tiles();
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_f16_matmul(gpu, slot);
  test_norms_and_embed(gpu, slot);
  test_vae_heads_and_attention(gpu, slot);
  test_encoder_kernels(gpu, slot);
  return failures != 0;
}
