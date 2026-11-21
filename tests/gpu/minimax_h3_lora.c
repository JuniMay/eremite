#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/lora.h"
#include "model/minimax_h3/quant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// GPU tests for MiniMax-H3 runtime BF16 LoRA kernels and safetensors loader.

// Tile size, threadgroup counts and BF16 conversion constants matching
// minimax_h3_dit.metal and minimax_h3_quant.metal.
static constexpr uint32_t TILE_M = 64, TILE_N = 64, MM_THREADS = 128,
                          GRID2D_THREADS = 32;

// Memory budget for the GPU test account (512 MiB).
static constexpr uint64_t TEST_BUDGET_BYTES = 512ull << 20;

// Out-proj dimensions matching MiniMax-H3 DiT and TokenRefiner.
static constexpr int OUT_N = 5376, OUT_K = 7168;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Active rows, inner width, input row stride and scale flag.
  uint32_t m, k, lda, has_scale;
} TestBf16XParam;

// Parameters passed to minimax_h3_mm_bf16_out_bf16.
typedef struct {
  // Active rows, output width, inner width, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} TestBf16MMParam;

// Parameters passed to minimax_h3_mm_bf16_add.
typedef struct {
  // Active rows, output width, inner width and output row stride.
  uint32_t m, n, k, ldy;
  // Runtime multiplier applied before accumulating into y.
  float scale;
} TestBf16MMAddParam;

// Failed check counter.
static int failures;

// Records one assertion outcome and prints label on failure.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

// Rounds an IEEE-754 float32 value to BF16 (round-to-nearest-even).
static uint16_t f32_to_bf16(float x) {
  uint32_t u = 0;
  memcpy(&u, &x, sizeof u);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return (uint16_t)(u >> 16);
}

// Converts a BF16 bit pattern back to float32.
static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float x = 0.0f;
  memcpy(&x, &u, sizeof x);
  return x;
}

// Deterministic signed pseudo-random float in [-amp, amp].
static float det_val(uint64_t i, float amp) {
  uint64_t x = (i + 1u) * 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x ^= x >> 31;
  double u = (double)(x >> 11) * (1.0 / (double)(1ull << 53));
  return (float)((2.0 * u - 1.0) * (double)amp);
}

// Verifies minimax_h3_mm_bf16_out_bf16 followed by minimax_h3_mm_bf16_add
// against a CPU BF16 reference for shape (m, k) -> (m, r) -> (m, n).
static void test_kernels(Gpu *gpu, MemorySlot *slot, int m, int n, int k, int r,
                         float scale, const char *label) {
  int k_x = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  int k_out = gpu_find_kernel(gpu, "minimax_h3_mm_bf16_out_bf16");
  int k_add = gpu_find_kernel(gpu, "minimax_h3_mm_bf16_add");

  uint32_t padded = ((uint32_t)m + TILE_M - 1) / TILE_M * TILE_M;
  Buf x = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * sizeof(float));
  Buf xb = gpu_alloc(gpu, slot, (size_t)padded * (size_t)k * sizeof(uint16_t));
  Buf a = gpu_alloc(gpu, slot, (size_t)r * (size_t)k * sizeof(uint16_t));
  Buf b = gpu_alloc(gpu, slot, (size_t)n * (size_t)r * sizeof(uint16_t));
  Buf mid = gpu_alloc(gpu, slot, (size_t)padded * (size_t)r * sizeof(uint16_t));
  Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * sizeof(float));

  float *xp = (float *)x.ptr;
  uint16_t *ap = (uint16_t *)a.ptr;
  uint16_t *bp = (uint16_t *)b.ptr;
  float *yp = (float *)y.ptr;
  float *y0 = xmalloc((size_t)m * (size_t)n * sizeof(float));

  for (size_t i = 0; i < (size_t)m * (size_t)k; i++) xp[i] = det_val(i, 0.5f);
  for (size_t i = 0; i < (size_t)r * (size_t)k; i++)
    ap[i] = f32_to_bf16(det_val(i + 100000, 0.1f));
  for (size_t i = 0; i < (size_t)n * (size_t)r; i++)
    bp[i] = f32_to_bf16(det_val(i + 200000, 0.1f));
  for (size_t i = 0; i < (size_t)m * (size_t)n; i++) {
    y0[i] = det_val(i + 300000, 1.0f);
    yp[i] = y0[i];
  }
  memset(mid.ptr, 0x5A, (size_t)padded * (size_t)r * sizeof(uint16_t));

  GpuCommandBuffer cmd = gpu_begin(gpu);
  TestBf16XParam px = {(uint32_t)m, (uint32_t)k, (uint32_t)k, 0};
  gpu_select_kernel(&cmd, k_x);
  gpu_bind_buffer(&cmd, 0, &x, 0);
  gpu_bind_buffer(&cmd, 1, &a, 0);
  gpu_bind_buffer(&cmd, 2, &xb, 0);
  gpu_copy_bytes(&cmd, 3, &px, sizeof px);
  gpu_run_kernel(&cmd,
                 ((uint32_t)k / MINIMAX_H3_VEC4_WIDTH + GRID2D_THREADS - 1) /
                     GRID2D_THREADS,
                 padded, 1, GRID2D_THREADS, 1, 1);

  TestBf16MMParam pa = {(uint32_t)m, (uint32_t)r, (uint32_t)k, (uint32_t)r, 0};
  gpu_select_kernel(&cmd, k_out);
  gpu_bind_buffer(&cmd, 0, &a, 0);
  gpu_bind_buffer(&cmd, 1, &xb, 0);
  gpu_bind_buffer(&cmd, 2, &mid, 0);
  gpu_copy_bytes(&cmd, 3, &pa, sizeof pa);
  gpu_run_kernel(&cmd, ((uint32_t)r + TILE_N - 1) / TILE_N, padded / TILE_M, 1,
                 MM_THREADS, 1, 1);

  TestBf16MMAddParam pb = {(uint32_t)m, (uint32_t)n, (uint32_t)r, (uint32_t)n,
                           scale};
  gpu_select_kernel(&cmd, k_add);
  gpu_bind_buffer(&cmd, 0, &b, 0);
  gpu_bind_buffer(&cmd, 1, &mid, 0);
  gpu_bind_buffer(&cmd, 2, &y, 0);
  gpu_copy_bytes(&cmd, 3, &pb, sizeof pb);
  gpu_run_kernel(&cmd, ((uint32_t)n + TILE_N - 1) / TILE_N, padded / TILE_M, 1,
                 MM_THREADS, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Check padded rows in mid are exact zeros.
  const uint16_t *mp = (const uint16_t *)mid.ptr;
  int pad_zero = 1;
  for (uint32_t row = (uint32_t)m; row < padded; row++) {
    for (int col = 0; col < r; col++) {
      if (mp[(size_t)row * (size_t)r + (size_t)col] != 0) pad_zero = 0;
    }
  }
  check(pad_zero, "minimax_h3_mm_bf16_out_bf16 zeros padded rows past m");

  // Compare active output against CPU BF16 reference.
  float max_err = 0.0f;
  uint16_t *mid_ref = xmalloc((size_t)r * sizeof(uint16_t));
  for (int row = 0; row < m; row++) {
    for (int j = 0; j < r; j++) {
      float acc = 0.0f;
      for (int c = 0; c < k; c++)
        acc += bf16_to_f32(f32_to_bf16(xp[(size_t)row * (size_t)k + c])) *
               bf16_to_f32(ap[(size_t)j * (size_t)k + c]);
      mid_ref[j] = f32_to_bf16(acc);
    }
    for (int col = 0; col < n; col++) {
      float acc = 0.0f;
      for (int j = 0; j < r; j++)
        acc += bf16_to_f32(mid_ref[j]) *
               bf16_to_f32(bp[(size_t)col * (size_t)r + j]);
      float ref = y0[(size_t)row * (size_t)n + col] + scale * acc;
      float err = fabsf(yp[(size_t)row * (size_t)n + col] - ref);
      if (err > max_err) max_err = err;
    }
  }
  free(mid_ref);
  free(y0);

  if (max_err >= 5e-3f) fprintf(stderr, "  %s max_err=%.6f\n", label, max_err);
  check(max_err < 5e-3f, label);

  gpu_free(&y);
  gpu_free(&mid);
  gpu_free(&b);
  gpu_free(&a);
  gpu_free(&xb);
  gpu_free(&x);
}

// Writes a valid two-target ComfyUI safetensors LoRA file to path:
// - blocks.0.attn.out_proj: rank 128, alpha 8.0 (scale 0.0625, aligned)
// - token_refiner.blocks.0.attn.out_proj: rank 32, alpha 16.0 (scale 0.5,
//   padded up to 64)
static void write_test_lora(const char *path, const uint16_t *a0,
                            const uint16_t *b0, float alpha0,
                            const uint16_t *a1, const uint16_t *b1,
                            float alpha1) {
  size_t s_a0 = (size_t)128 * (size_t)OUT_K * sizeof(uint16_t);
  size_t s_b0 = (size_t)OUT_N * (size_t)128 * sizeof(uint16_t);
  size_t s_al = sizeof(float);
  size_t s_a1 = (size_t)32 * (size_t)OUT_K * sizeof(uint16_t);
  size_t s_b1 = (size_t)OUT_N * (size_t)32 * sizeof(uint16_t);

  size_t off0 = 0;
  size_t off1 = off0 + s_al;
  size_t off2 = off1 + s_a0;
  size_t off3 = off2 + s_b0;
  size_t off4 = off3 + s_al;
  size_t off5 = off4 + s_a1;
  size_t off6 = off5 + s_b1;

  char hdr[2048];
  int hlen = snprintf(
      hdr, sizeof hdr,
      "{"
      "\"diffusion_model.blocks.0.attn.out_proj.alpha\":"
      "{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[%zu,%zu]},"
      "\"diffusion_model.blocks.0.attn.out_proj.lora_A.weight\":"
      "{\"dtype\":\"BF16\",\"shape\":[128,%d],\"data_offsets\":[%zu,%zu]},"
      "\"diffusion_model.blocks.0.attn.out_proj.lora_B.weight\":"
      "{\"dtype\":\"BF16\",\"shape\":[%d,128],\"data_offsets\":[%zu,%zu]},"
      "\"diffusion_model.token_refiner.blocks.0.attn.out_proj.alpha\":"
      "{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[%zu,%zu]},"
      "\"diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_A.weight\":"
      "{\"dtype\":\"BF16\",\"shape\":[32,%d],\"data_offsets\":[%zu,%zu]},"
      "\"diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight\":"
      "{\"dtype\":\"BF16\",\"shape\":[%d,32],\"data_offsets\":[%zu,%zu]}"
      "}",
      off0, off1, OUT_K, off1, off2, OUT_N, off2, off3, off3, off4, OUT_K, off4,
      off5, OUT_N, off5, off6);
  if (hlen <= 0 || (size_t)hlen >= sizeof hdr) die("LoRA test header overflow");

  FILE *f = fopen(path, "wb");
  if (!f) die("cannot create %s", path);
  uint64_t hl = (uint64_t)hlen;
  if (fwrite(&hl, sizeof hl, 1, f) != 1 ||
      fwrite(hdr, 1, (size_t)hlen, f) != (size_t)hlen ||
      fwrite(&alpha0, 1, s_al, f) != s_al || fwrite(a0, 1, s_a0, f) != s_a0 ||
      fwrite(b0, 1, s_b0, f) != s_b0 || fwrite(&alpha1, 1, s_al, f) != s_al ||
      fwrite(a1, 1, s_a1, f) != s_a1 || fwrite(b1, 1, s_b1, f) != s_b1)
    die("failed writing %s", path);
  fclose(f);
}

// Tests opening, loading and applying a safetensors LoRA through
// minimax_h3_lora_open/load/apply on both aligned and padded ranks.
static void test_lora_module(Gpu *gpu, MemoryAccount *account,
                             MemorySlot *slot) {
  uint16_t *a0 = xmalloc((size_t)128 * (size_t)OUT_K * sizeof(uint16_t));
  uint16_t *b0 = xmalloc((size_t)OUT_N * (size_t)128 * sizeof(uint16_t));
  uint16_t *a1 = xmalloc((size_t)32 * (size_t)OUT_K * sizeof(uint16_t));
  uint16_t *b1 = xmalloc((size_t)OUT_N * (size_t)32 * sizeof(uint16_t));
  for (size_t i = 0; i < (size_t)128 * (size_t)OUT_K; i++)
    a0[i] = f32_to_bf16(det_val(i + 1, 0.05f));
  for (size_t i = 0; i < (size_t)OUT_N * (size_t)128; i++)
    b0[i] = f32_to_bf16(det_val(i + 2, 0.05f));
  for (size_t i = 0; i < (size_t)32 * (size_t)OUT_K; i++)
    a1[i] = f32_to_bf16(det_val(i + 3, 0.05f));
  for (size_t i = 0; i < (size_t)OUT_N * (size_t)32; i++)
    b1[i] = f32_to_bf16(det_val(i + 4, 0.05f));

  char path[] = "/tmp/eremite-test-lora-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) die("mkstemp failed");
  close(fd);
  write_test_lora(path, a0, b0, 8.0f, a1, b1, 16.0f);

  float strength = 0.75f;
  MinimaxH3Lora *lora = minimax_h3_lora_open(gpu, account, path, strength);
  check(minimax_h3_lora_max_rank(lora) == 128,
        "LoRA reports max padded rank 128");
  check(minimax_h3_lora_required_bytes(lora) > 0,
        "LoRA reports positive required GPU bytes");
  check(minimax_h3_lora_load(lora) == 1, "LoRA loads into GPU memory");

  const MinimaxH3LoraBlock *db0 = minimax_h3_lora_dit_block(lora, 0);
  const MinimaxH3LoraBlock *rb0 = minimax_h3_lora_refiner_block(lora, 0);
  check(db0->out_proj.present && !db0->qkv_proj.present &&
            fabsf(db0->out_proj.scale - 0.0625f) < 1e-6f,
        "DiT block 0 out_proj scale is 8/128 = 0.0625");
  check(rb0->out_proj.present && rb0->out_proj.rank == 64 &&
            fabsf(rb0->out_proj.scale - 0.5f) < 1e-6f,
        "TokenRefiner block 0 out_proj pads rank 32 to 64 with scale 16/32");

  int rows = 5;
  uint32_t padded = TILE_M;
  Buf x = gpu_alloc(gpu, slot, (size_t)rows * (size_t)OUT_K * sizeof(float));
  Buf xb =
      gpu_alloc(gpu, slot, (size_t)padded * (size_t)OUT_K * sizeof(uint16_t));
  Buf mid = gpu_alloc(gpu, slot, minimax_h3_lora_workspace_bytes(lora, rows));
  Buf y_dit =
      gpu_alloc(gpu, slot, (size_t)rows * (size_t)OUT_N * sizeof(float));
  Buf y_ref =
      gpu_alloc(gpu, slot, (size_t)rows * (size_t)OUT_N * sizeof(float));

  float *xp = (float *)x.ptr;
  float *yd = (float *)y_dit.ptr;
  float *yr = (float *)y_ref.ptr;
  for (size_t i = 0; i < (size_t)rows * (size_t)OUT_K; i++)
    xp[i] = det_val(i + 50, 0.25f);
  for (size_t i = 0; i < (size_t)rows * (size_t)OUT_N; i++) {
    yd[i] = 1.25f;
    yr[i] = -0.5f;
  }

  // Apply DiT block 0 out_proj with xb_ready=0, then reuse xb for TokenRefiner
  // block 0 out_proj with xb_ready=1.
  GpuCommandBuffer cmd = gpu_begin(gpu);
  minimax_h3_lora_apply(lora, &cmd, &db0->out_proj, &x, &xb, &mid, &y_dit, rows,
                        0);
  minimax_h3_lora_apply(lora, &cmd, &rb0->out_proj, &x, &xb, &mid, &y_ref, rows,
                        1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  // Spot-check all 5 rows across 64 strided output columns against CPU ref.
  float err_dit = 0.0f, err_ref = 0.0f;
  for (int r = 0; r < rows; r++) {
    uint16_t m0[128], m1[32];
    for (int j = 0; j < 128; j++) {
      float acc = 0.0f;
      for (int c = 0; c < OUT_K; c++)
        acc += bf16_to_f32(f32_to_bf16(xp[(size_t)r * (size_t)OUT_K + c])) *
               bf16_to_f32(a0[(size_t)j * (size_t)OUT_K + c]);
      m0[j] = f32_to_bf16(acc);
    }
    for (int j = 0; j < 32; j++) {
      float acc = 0.0f;
      for (int c = 0; c < OUT_K; c++)
        acc += bf16_to_f32(f32_to_bf16(xp[(size_t)r * (size_t)OUT_K + c])) *
               bf16_to_f32(a1[(size_t)j * (size_t)OUT_K + c]);
      m1[j] = f32_to_bf16(acc);
    }
    for (int col = 0; col < OUT_N; col += 83) {
      float acc0 = 0.0f, acc1 = 0.0f;
      for (int j = 0; j < 128; j++)
        acc0 +=
            bf16_to_f32(m0[j]) * bf16_to_f32(b0[(size_t)col * (size_t)128 + j]);
      for (int j = 0; j < 32; j++)
        acc1 +=
            bf16_to_f32(m1[j]) * bf16_to_f32(b1[(size_t)col * (size_t)32 + j]);
      float ref0 = 1.25f + (0.0625f * strength) * acc0;
      float ref1 = -0.5f + (0.5f * strength) * acc1;
      float e0 = fabsf(yd[(size_t)r * (size_t)OUT_N + col] - ref0);
      float e1 = fabsf(yr[(size_t)r * (size_t)OUT_N + col] - ref1);
      if (e0 > err_dit) err_dit = e0;
      if (e1 > err_ref) err_ref = e1;
    }
  }
  check(err_dit < 5e-3f,
        "minimax_h3_lora_apply matches CPU ref on aligned rank 128 target");
  check(err_ref < 5e-3f,
        "minimax_h3_lora_apply matches CPU ref on padded rank 32 target");

  gpu_free(&y_ref);
  gpu_free(&y_dit);
  gpu_free(&mid);
  gpu_free(&xb);
  gpu_free(&x);
  minimax_h3_lora_close(lora);
  unlink(path);
  free(b1);
  free(a1);
  free(b0);
  free(a0);
}

int main(void) {
  Memory *mem = memory_new();
  MemoryAccount account = {0};
  memory_account_init(mem, &account, "minimax-h3-lora-test", TEST_BUDGET_BYTES);
  MemorySlot *slot = memory_shared(mem);
  size_t source_bytes = 0;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_kernels(gpu, slot, 13, 192, 256, 128, 1.0f,
               "LoRA rank 128 scale 1.0 (m=13) matches CPU");
  test_kernels(gpu, slot, 77, 256, 512, 384, 0.0625f,
               "LoRA fused QKV rank 384 scale 0.0625 (m=77) matches CPU");
  test_lora_module(gpu, &account, slot);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("minimax_h3_lora: all checks passed\n");
  return 0;
}
