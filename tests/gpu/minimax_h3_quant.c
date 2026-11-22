#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/quant.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Tests MiniMax-H3's quantized linear and embedding kernels against CPU
// references: cuBLASLt NVFP4 scale unswizzling, hi-nibble-first NVFP4 matmul
// with optional pre_quant_scale, ConvRot block-256 Hadamard rotation with
// per-row INT8 quantization and native INT8 matmul, and INT8 embeddings.

// Tile height used to pad activation matrices for 128-row matmul tiles.
static constexpr int TILE_M = 128;

// Tile width of one matmul threadgroup along output columns.
static constexpr int TILE_N = 64;

// Threads per threadgroup for matmul and ConvRot kernels.
static constexpr int MM_THREADS = 128, CONVROT_THREADS = 256;

// Minimum per-row activation scale matching comfy-kitchen's
// quantize_int8_rowwise.
static constexpr float INT8_SCALE_MIN = 1e-30f;

// Largest positive magnitude used to scale INT8 activations.
static constexpr float INT8_MAX_VAL = 127.0f;

// Relative tolerance for NVFP4 BF16 matmul comparison against FP32 CPU sum.
static constexpr float NVFP4_REL_TOL = 1e-2f, NVFP4_ABS_TOL = 1e-3f;

// Absolute tolerance for INT8-ConvRot matmul comparison against CPU reference.
static constexpr float INT8_ABS_TOL = 1e-3f;

// Parameters passed to minimax_h3_x_bf16.
typedef struct {
  // Rows, columns, input row stride and whether pre_quant_scale is active.
  uint32_t m, k, lda, has_scale;
} Bf16XParam;

// Parameters passed to minimax_h3_mm_nvfp4.
typedef struct {
  // Rows, output columns, inner columns and output row stride.
  uint32_t m, n, k, ldy;
  // Per-tensor NVFP4 global scale.
  float scale2;
} Nvfp4MMParam;

// Parameters passed to minimax_h3_convrot_quant_i8.
typedef struct {
  // Rows, inner columns and input row stride.
  uint32_t m, k, lda;
} ConvRotParam;

// Parameters passed to minimax_h3_swiglu_convrot.
typedef struct {
  // Active rows and half-width inner dimension.
  uint32_t m, k;
} SwigluConvRotParam;

// Parameters passed to minimax_h3_swiglu_chunked.
typedef struct {
  // Row count and half-width inner dimension.
  uint32_t rows, inner;
} SwigluChunkedParam;

// Parameters passed to minimax_h3_rmsnorm.
typedef struct {
  // Row width in floats.
  uint32_t n;
  // Normalization epsilon.
  float eps;
} RmsNormParam;

// Parameters passed to minimax_h3_rmsnorm_convrot.
typedef struct {
  // Active rows and row width in floats.
  uint32_t m, k;
  // Normalization epsilon.
  float eps;
} RmsNormConvRotParam;

// Parameters passed to minimax_h3_mm_int8.
typedef struct {
  // Rows, output columns, inner columns, output row stride and bias flag.
  uint32_t m, n, k, ldy, has_bias;
} Int8MMParam;

// Deterministic PRNG for test inputs.
static uint32_t rng_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// Uniform float in [-1, 1] from state.
static float rng_float(uint32_t *state) {
  return (float)(rng_next(state) & 0xffffffu) / (float)0x800000u - 1.0f;
}

// Rounds a float to BF16 and converts back to float.
static float to_bf16_f32(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
  u = (u + rounding) & 0xffff0000u;
  float out;
  memcpy(&out, &u, sizeof out);
  return out;
}

// Encodes a float as a BF16 16-bit code.
static uint16_t f32_to_bf16(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)((u + rounding) >> 16);
}

// Decodes an FP4 E2M1 nibble to float.
static float decode_fp4(uint8_t nib) {
  static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
  float v = table[nib & 7u];
  return (nib & 8u) ? -v : v;
}

// Decodes an FP8 E4M3 byte to float (non-NaN codes).
static float decode_fp8_e4m3(uint8_t b) {
  unsigned exp = (b >> 3) & 15u, mant = b & 7u;
  float v = exp ? ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7)
                : ldexpf((float)mant, -9);
  return (b & 128u) ? -v : v;
}

// Swizzles a row-major rows by cols matrix into cuBLASLt's 128 by 4 blocked
// layout (comfy-kitchen's to_blocked).
static uint8_t *to_blocked_ref(const uint8_t *plain, int rows, int cols,
                               size_t *out_bytes) {
  int rb_count =
      (rows + MINIMAX_H3_SCALE_TILE_ROWS - 1) / MINIMAX_H3_SCALE_TILE_ROWS;
  int cb_count =
      (cols + MINIMAX_H3_SCALE_TILE_COLS - 1) / MINIMAX_H3_SCALE_TILE_COLS;
  size_t total = (size_t)rb_count * (size_t)cb_count *
                 (size_t)MINIMAX_H3_SCALE_TILE_ROWS *
                 (size_t)MINIMAX_H3_SCALE_TILE_COLS;
  uint8_t *blocked = xmalloc(total);
  memset(blocked, 0, total);
  for (int r = 0; r < rows; r++) {
    int rb = r / MINIMAX_H3_SCALE_TILE_ROWS;
    int r_in = r % MINIMAX_H3_SCALE_TILE_ROWS;
    int sub = r_in / 32;
    int lane = r_in % 32;
    for (int c = 0; c < cols; c++) {
      int cb = c / MINIMAX_H3_SCALE_TILE_COLS;
      int c_in = c % MINIMAX_H3_SCALE_TILE_COLS;
      size_t idx = ((size_t)rb * (size_t)cb_count + (size_t)cb) * 512u +
                   (size_t)lane * 16u + (size_t)sub * 4u + (size_t)c_in;
      blocked[idx] = plain[(size_t)r * (size_t)cols + (size_t)c];
    }
  }
  *out_bytes = total;
  return blocked;
}

// Builds the explicit 256 by 256 ConvRot regular Hadamard matrix
// H_4^{\otimes 4} / 16 into h[256 * 256].
static void build_hadamard256(float *h) {
  static const float h4[4][4] = {
      {1, 1, 1, -1},
      {1, 1, -1, 1},
      {1, -1, 1, 1},
      {-1, 1, 1, 1},
  };
  for (int r = 0; r < MINIMAX_H3_CONVROT_GROUP; r++) {
    int r0 = (r >> 6) & 3, r1 = (r >> 4) & 3, r2 = (r >> 2) & 3, r3 = r & 3;
    for (int c = 0; c < MINIMAX_H3_CONVROT_GROUP; c++) {
      int c0 = (c >> 6) & 3, c1 = (c >> 4) & 3, c2 = (c >> 2) & 3, c3 = c & 3;
      h[(size_t)r * MINIMAX_H3_CONVROT_GROUP + (size_t)c] =
          h4[r0][c0] * h4[r1][c1] * h4[r2][c2] * h4[r3][c3] * (1.0f / 16.0f);
    }
  }
}

// Tests cuBLASLt NVFP4 scale unswizzling and WeightGroup loading from disk.
static void test_scale_unswizzle(Gpu *gpu, MemoryAccount *account) {
  static const int shapes[2][2] = {{131, 6}, {5120, 320}};
  int exact = 1;
  for (int s = 0; s < 2; s++) {
    int rows = shapes[s][0], cols = shapes[s][1];
    size_t plain_bytes = (size_t)rows * (size_t)cols;
    uint8_t *plain = xmalloc(plain_bytes), *roundtrip = xmalloc(plain_bytes);
    for (size_t i = 0; i < plain_bytes; i++)
      plain[i] = (uint8_t)((i * 37u + 11u) & 0x7eu);
    size_t blocked_bytes = 0;
    uint8_t *blocked = to_blocked_ref(plain, rows, cols, &blocked_bytes);
    minimax_h3_unswizzle_nvfp4_scales(roundtrip, blocked, rows, cols);
    exact &= memcmp(plain, roundtrip, plain_bytes) == 0;
    free(blocked);
    free(roundtrip);
    free(plain);
  }
  check(exact, "cuBLASLt blocked NVFP4 scales unswizzle byte for byte");

  // Also test WeightGroup loading and scalar reading from a safetensors file.
  char dir[] = "/tmp/eremite-minimax-h3-quant-XXXXXX";
  if (!mkdtemp(dir)) die("cannot make temp dir");
  char path[256];
  snprintf(path, sizeof path, "%s/test.safetensors", dir);
  int rows = 128, cols = 8;
  size_t plain_bytes = (size_t)rows * (size_t)cols;
  uint8_t *plain = xmalloc(plain_bytes);
  for (size_t i = 0; i < plain_bytes; i++)
    plain[i] = (uint8_t)((i * 19u + 5u) & 0x7eu);
  size_t blocked_bytes = 0;
  uint8_t *blocked = to_blocked_ref(plain, rows, cols, &blocked_bytes);
  float scalar_val = 0.03125f;
  char header[256];
  snprintf(header, sizeof header,
           "{\"scales\":{\"dtype\":\"F8_E4M3\",\"shape\":[%d,%d],"
           "\"data_offsets\":[0,%zu]},"
           "\"scale2\":{\"dtype\":\"F32\",\"shape\":[],"
           "\"data_offsets\":[%zu,%zu]}}",
           rows, cols, blocked_bytes, blocked_bytes, blocked_bytes + 4);
  uint64_t hlen = strlen(header);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0 || write(fd, &hlen, 8) != 8 ||
      write(fd, header, hlen) != (ssize_t)hlen ||
      write(fd, blocked, blocked_bytes) != (ssize_t)blocked_bytes ||
      write(fd, &scalar_val, 4) != 4)
    die("cannot write temp safetensors");
  close(fd);

  TensorIndex *idx = tensor_index_open_file(path);
  float read_s2 = minimax_h3_read_scalar_f32(tensor_index_get(idx, "scale2"));
  WeightGroup *group = weight_group_new(gpu, account, "nvfp4-scales-test");
  Buf scale_buf = {0};
  minimax_h3_nvfp4_scales_add(group, &scale_buf,
                              tensor_index_get(idx, "scales"), rows, cols);
  int loaded = weight_group_load(group);
  check(loaded && read_s2 == scalar_val &&
            memcmp(scale_buf.ptr, plain, plain_bytes) == 0,
        "WeightGroup loads and unswizzles NVFP4 scales from a file");
  weight_group_free(group);
  tensor_index_close(idx);
  unlink(path);
  rmdir(dir);
  free(blocked);
  free(plain);
}

// Tests NVFP4 linear kernel with and without pre_quant_scale across multiple M.
static void test_nvfp4_linear(Gpu *gpu, MemorySlot *slot) {
  int k_x_bf16 = gpu_find_kernel(gpu, "minimax_h3_x_bf16");
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_nvfp4");
  static const int ms[] = {1, 17, 65, 131};
  int k = 256, n = 192;
  uint32_t rng = 0x12345678u;

  Buf w_buf = gpu_alloc(gpu, slot, (size_t)n * (size_t)(k / 2));
  Buf s_buf = gpu_alloc(gpu, slot, (size_t)n * (size_t)(k / 16));
  Buf pre_buf = gpu_alloc(gpu, slot, (size_t)k * 2);
  for (size_t i = 0; i < w_buf.size; i++)
    w_buf.ptr[i] = (uint8_t)(rng_next(&rng) & 0xffu);
  for (size_t i = 0; i < s_buf.size; i++) {
    // Non-NaN E4M3 codes spanning subnormals and normal exponents.
    uint8_t mag = (uint8_t)(rng_next(&rng) % 0x68u);
    uint8_t sign = (uint8_t)((rng_next(&rng) & 1u) << 7);
    s_buf.ptr[i] = (uint8_t)(sign | mag);
  }
  uint16_t *pre_bf16 = (uint16_t *)pre_buf.ptr;
  for (int j = 0; j < k; j++)
    pre_bf16[j] = f32_to_bf16(0.5f + 0.5f * rng_float(&rng));
  float scale2 = 0.015625f;

  int ok_plain = 1, ok_pre = 1;
  for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
    int m = ms[mi];
    int padded_m = (m + TILE_M - 1) / TILE_M * TILE_M;
    Buf x_buf = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * 4);
    Buf xb_buf = gpu_alloc(gpu, slot, (size_t)padded_m * (size_t)k * 2);
    Buf y_buf = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * 4);
    float *x_f32 = (float *)x_buf.ptr;
    for (int i = 0; i < m * k; i++) x_f32[i] = rng_float(&rng);

    for (int use_pre = 0; use_pre < 2; use_pre++) {
      Bf16XParam xp = {(uint32_t)m, (uint32_t)k, (uint32_t)k,
                       (uint32_t)use_pre};
      Nvfp4MMParam mp = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                         scale2};
      GpuCommandBuffer cmd = gpu_begin(gpu);
      gpu_select_kernel(&cmd, k_x_bf16);
      gpu_bind_buffer(&cmd, 0, &x_buf, 0);
      gpu_bind_buffer(&cmd, 1, &pre_buf, 0);
      gpu_bind_buffer(&cmd, 2, &xb_buf, 0);
      gpu_copy_bytes(&cmd, 3, &xp, sizeof xp);
      gpu_run_kernel(&cmd, (uint32_t)((k / 4 + 31) / 32), (uint32_t)padded_m, 1,
                     32, 1, 1);

      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w_buf, 0);
      gpu_bind_buffer(&cmd, 1, &s_buf, 0);
      gpu_bind_buffer(&cmd, 2, &xb_buf, 0);
      gpu_bind_buffer(&cmd, 3, &y_buf, 0);
      gpu_copy_bytes(&cmd, 4, &mp, sizeof mp);
      gpu_run_kernel(&cmd, (uint32_t)((n + TILE_N - 1) / TILE_N),
                     (uint32_t)((m + TILE_M - 1) / TILE_M), 1, MM_THREADS, 1,
                     1);
      gpu_commit(&cmd);
      gpu_sync(gpu);

      const float *y_got = (const float *)y_buf.ptr;
      for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
          float sum = 0.0f;
          const uint8_t *wr = w_buf.ptr + (size_t)c * (size_t)(k / 2);
          const uint8_t *sr = s_buf.ptr + (size_t)c * (size_t)(k / 16);
          for (int j = 0; j < k; j++) {
            float xv = x_f32[(size_t)r * (size_t)k + (size_t)j];
            if (use_pre) {
              uint32_t u = (uint32_t)pre_bf16[j] << 16;
              float ps;
              memcpy(&ps, &u, sizeof ps);
              xv *= ps;
            }
            xv = to_bf16_f32(xv);
            uint8_t byte = wr[j / 2];
            uint8_t nib = (j & 1) ? (byte & 0xfu) : (byte >> 4);
            float wv =
                to_bf16_f32(decode_fp4(nib) * decode_fp8_e4m3(sr[j / 16]));
            sum += xv * wv;
          }
          float ref = sum * scale2;
          float got = y_got[(size_t)r * (size_t)n + (size_t)c];
          float diff = fabsf(got - ref);
          float tol = fmaxf(NVFP4_ABS_TOL, fabsf(ref) * NVFP4_REL_TOL);
          if (!isfinite(got) || diff > tol) {
            if (use_pre)
              ok_pre = 0;
            else
              ok_plain = 0;
          }
        }
      }
    }
    gpu_free(&y_buf);
    gpu_free(&xb_buf);
    gpu_free(&x_buf);
  }
  gpu_free(&pre_buf);
  gpu_free(&s_buf);
  gpu_free(&w_buf);
  check(ok_plain, "NVFP4 linear matches the CPU hi-nibble-first reference");
  check(ok_pre, "NVFP4 linear with pre_quant_scale matches the CPU reference");
}

// Tests ConvRot block-256 Hadamard rotation, per-row INT8 quantization and
// native INT8 matmul against explicit CPU Kronecker H_4^{\otimes 4} reference.
static void test_convrot_int8(Gpu *gpu, MemorySlot *slot) {
  int k_rot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  int k_mm = gpu_find_kernel(gpu, "minimax_h3_mm_int8");
  float *h256 =
      xmalloc((size_t)MINIMAX_H3_CONVROT_GROUP * MINIMAX_H3_CONVROT_GROUP * 4);
  build_hadamard256(h256);

  static const int ms[] = {1, 33, 131};
  int k = 512, n = 192;
  uint32_t rng = 0x87654321u;

  Buf w_buf = gpu_alloc(gpu, slot, (size_t)n * (size_t)k);
  Buf ws_buf = gpu_alloc(gpu, slot, (size_t)n * 4);
  Buf bias_buf = gpu_alloc(gpu, slot, (size_t)n * 4);
  int8_t *w_i8 = (int8_t *)w_buf.ptr;
  float *ws_f32 = (float *)ws_buf.ptr;
  float *bias_f32 = (float *)bias_buf.ptr;
  for (int i = 0; i < n * k; i++)
    w_i8[i] = (int8_t)((int)(rng_next(&rng) % 255u) - 127);
  for (int c = 0; c < n; c++) {
    ws_f32[c] = 0.002f + 0.004f * fabsf(rng_float(&rng));
    bias_f32[c] = rng_float(&rng) * 0.5f;
  }

  int quant_ok = 1, mm_no_bias_ok = 1, mm_bias_ok = 1;
  float *rot_ref = xmalloc((size_t)k * 4);
  int8_t *q_ref = xmalloc((size_t)k);

  for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
    int m = ms[mi];
    int padded_m = (m + TILE_M - 1) / TILE_M * TILE_M;
    Buf x_buf = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * 4);
    Buf xq_buf = gpu_alloc(gpu, slot, (size_t)padded_m * (size_t)k);
    Buf xs_buf = gpu_alloc(gpu, slot, (size_t)m * 4);
    Buf y_buf = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * 4);
    float *x_f32 = (float *)x_buf.ptr;
    for (int r = 0; r < m; r++) {
      // Make the last row of multi-row batches all zeros to test the scale
      // floor.
      int all_zero = (m > 1 && r == m - 1);
      for (int j = 0; j < k; j++)
        x_f32[(size_t)r * (size_t)k + (size_t)j] =
            all_zero ? 0.0f : rng_float(&rng) * 3.0f;
    }

    ConvRotParam rp = {(uint32_t)m, (uint32_t)k, (uint32_t)k};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_rot);
    gpu_bind_buffer(&cmd, 0, &x_buf, 0);
    gpu_bind_buffer(&cmd, 1, &xq_buf, 0);
    gpu_bind_buffer(&cmd, 2, &xs_buf, 0);
    gpu_copy_bytes(&cmd, 3, &rp, sizeof rp);
    gpu_run_kernel(&cmd, (uint32_t)padded_m, 1, 1, CONVROT_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const int8_t *xq_got = (const int8_t *)xq_buf.ptr;
    const float *xs_got = (const float *)xs_buf.ptr;
    // Check padded tail rows are zeroed.
    for (int r = m; r < padded_m; r++)
      for (int j = 0; j < k; j++)
        if (xq_got[(size_t)r * (size_t)k + (size_t)j] != 0) quant_ok = 0;

    // Check rotated quantized rows and scales against explicit H_256 matmul.
    for (int r = 0; r < m; r++) {
      const float *xr = x_f32 + (size_t)r * (size_t)k;
      float amax = 0.0f;
      for (int g = 0; g < k / MINIMAX_H3_CONVROT_GROUP; g++) {
        const float *xg = xr + (size_t)g * MINIMAX_H3_CONVROT_GROUP;
        float *rg = rot_ref + (size_t)g * MINIMAX_H3_CONVROT_GROUP;
        for (int c = 0; c < MINIMAX_H3_CONVROT_GROUP; c++) {
          float sum = 0.0f;
          for (int i = 0; i < MINIMAX_H3_CONVROT_GROUP; i++)
            sum +=
                xg[i] * h256[(size_t)i * MINIMAX_H3_CONVROT_GROUP + (size_t)c];
          rg[c] = sum;
          amax = fmaxf(amax, fabsf(sum));
        }
      }
      float s_ref = fmaxf(amax / INT8_MAX_VAL, INT8_SCALE_MIN);
      if (fabsf(xs_got[r] - s_ref) > fmaxf(1e-7f, s_ref * 1e-5f)) quant_ok = 0;
      for (int j = 0; j < k; j++) {
        float scaled = rot_ref[j] / s_ref;
        float rounded = rintf(scaled);
        if (rounded < -128.0f) rounded = -128.0f;
        if (rounded > 127.0f) rounded = 127.0f;
        q_ref[j] = (int8_t)(int)rounded;
        // Allow 1-ULP boundary tie difference only when scaled is near a
        // half-integer.
        int diff =
            abs((int)xq_got[(size_t)r * (size_t)k + (size_t)j] - (int)q_ref[j]);
        if (diff > 1 ||
            (diff == 1 && fabsf(fabsf(scaled - floorf(scaled)) - 0.5f) > 1e-3f))
          quant_ok = 0;
      }
    }

    // Check native INT8 matmul with and without bias against exact int32 sums
    // over the quantized activations xq_got.
    for (int has_bias = 0; has_bias < 2; has_bias++) {
      Int8MMParam mp = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n,
                        (uint32_t)has_bias};
      cmd = gpu_begin(gpu);
      gpu_select_kernel(&cmd, k_mm);
      gpu_bind_buffer(&cmd, 0, &w_buf, 0);
      gpu_bind_buffer(&cmd, 1, &ws_buf, 0);
      gpu_bind_buffer(&cmd, 2, &xq_buf, 0);
      gpu_bind_buffer(&cmd, 3, &xs_buf, 0);
      gpu_bind_buffer(&cmd, 4, &bias_buf, 0);
      gpu_bind_buffer(&cmd, 5, &y_buf, 0);
      gpu_copy_bytes(&cmd, 6, &mp, sizeof mp);
      gpu_run_kernel(&cmd,
                     (uint32_t)((n + TILE_N - 1) / TILE_N) *
                         (uint32_t)((m + TILE_M - 1) / TILE_M),
                     1, 1, MM_THREADS, 1, 1);
      gpu_commit(&cmd);
      gpu_sync(gpu);

      const float *y_got = (const float *)y_buf.ptr;
      for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
          int32_t acc = 0;
          const int8_t *xr = xq_got + (size_t)r * (size_t)k;
          const int8_t *wr = w_i8 + (size_t)c * (size_t)k;
          for (int j = 0; j < k; j++) acc += (int32_t)xr[j] * (int32_t)wr[j];
          float ref = (float)acc * (xs_got[r] * ws_f32[c]) +
                      (has_bias ? bias_f32[c] : 0.0f);
          float got = y_got[(size_t)r * (size_t)n + (size_t)c];
          if (!isfinite(got) || fabsf(got - ref) > INT8_ABS_TOL) {
            if (has_bias)
              mm_bias_ok = 0;
            else
              mm_no_bias_ok = 0;
          }
        }
      }
    }

    gpu_free(&y_buf);
    gpu_free(&xs_buf);
    gpu_free(&xq_buf);
    gpu_free(&x_buf);
  }
  free(q_ref);
  free(rot_ref);
  free(h256);
  gpu_free(&bias_buf);
  gpu_free(&ws_buf);
  gpu_free(&w_buf);
  check(quant_ok,
        "ConvRot block-256 Hadamard and per-row INT8 quantization match CPU");
  check(mm_no_bias_ok, "native INT8 matmul without bias matches CPU int32 sum");
  check(mm_bias_ok, "native INT8 matmul with bias matches CPU int32 sum");
}

// Tests INT8 row-scaled embedding lookup, including negative IDs left
// untouched.
static void test_embed_int8(Gpu *gpu, MemorySlot *slot) {
  int k_embed = gpu_find_kernel(gpu, "minimax_h3_embed_int8");
  uint32_t vocab = 16, dim = 64, rows = 5;
  Buf table = gpu_alloc(gpu, slot, (size_t)vocab * dim);
  Buf scale = gpu_alloc(gpu, slot, (size_t)vocab * 4);
  Buf ids = gpu_alloc(gpu, slot, (size_t)rows * 4);
  Buf out = gpu_alloc(gpu, slot, (size_t)rows * dim * 4);
  int8_t *t = (int8_t *)table.ptr;
  float *s = (float *)scale.ptr;
  for (uint32_t v = 0; v < vocab; v++) {
    s[v] = 0.01f * (float)(v + 1);
    for (uint32_t d = 0; d < dim; d++)
      t[v * dim + d] = (int8_t)((int)((v * 7u + d * 3u) % 255u) - 127);
  }
  static const int token_ids[5] = {3, -1, 0, 15, -2};
  memcpy(ids.ptr, token_ids, sizeof token_ids);
  float sentinel = 42.5f;
  float *out_f = (float *)out.ptr;
  for (uint32_t i = 0; i < rows * dim; i++) out_f[i] = sentinel;

  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, k_embed);
  gpu_bind_buffer(&cmd, 0, &table, 0);
  gpu_bind_buffer(&cmd, 1, &scale, 0);
  gpu_bind_buffer(&cmd, 2, &ids, 0);
  gpu_bind_buffer(&cmd, 3, &out, 0);
  gpu_copy_bytes(&cmd, 4, &dim, sizeof dim);
  gpu_run_kernel(&cmd, 1, rows, 1, dim, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);

  int ok = 1;
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t d = 0; d < dim; d++) {
      float got = out_f[r * dim + d];
      float ref = token_ids[r] < 0
                      ? sentinel
                      : (float)t[(uint32_t)token_ids[r] * dim + d] *
                            s[(uint32_t)token_ids[r]];
      if (fabsf(got - ref) > 1e-6f) ok = 0;
    }
  }
  gpu_free(&out);
  gpu_free(&ids);
  gpu_free(&scale);
  gpu_free(&table);
  check(ok,
        "INT8 row-scaled embedding lookup matches CPU and keeps negative IDs");
}

// Tests single-pass and 2-pass-fallback fused ConvRot kernels
// (minimax_h3_swiglu_convrot and minimax_h3_rmsnorm_convrot) against unfused
// reference kernel chains.
static void test_fused_convrot(Gpu *gpu, MemorySlot *slot) {
  int k_rot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  int k_sw = gpu_find_kernel(gpu, "minimax_h3_swiglu_chunked");
  int k_sw_rot = gpu_find_kernel(gpu, "minimax_h3_swiglu_convrot");
  int k_rms = gpu_find_kernel(gpu, "minimax_h3_rmsnorm");
  int k_rms_rot = gpu_find_kernel(gpu, "minimax_h3_rmsnorm_convrot");

  // Cover both register-cached single-pass (512 <= 16384) and 2-pass fallback
  // (16640 > 16384).
  static const uint32_t ks[] = {512u, 16640u};
  uint32_t m = 5u;
  uint32_t padded_m =
      (m + (uint32_t)TILE_M - 1u) / (uint32_t)TILE_M * (uint32_t)TILE_M;
  uint32_t rng = 0x51525354u;
  int sw_ok = 1, rms_ok = 1;

  for (size_t ki = 0; ki < sizeof ks / sizeof ks[0]; ki++) {
    uint32_t k = ks[ki];
    Buf in2 = gpu_alloc(gpu, slot, (size_t)m * (size_t)(2u * k) * 4u);
    Buf mid = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * 4u);
    Buf w = gpu_alloc(gpu, slot, (size_t)k * 4u);
    Buf xq_ref = gpu_alloc(gpu, slot, (size_t)padded_m * (size_t)k);
    Buf xs_ref = gpu_alloc(gpu, slot, (size_t)m * 4u);
    Buf xq_got = gpu_alloc(gpu, slot, (size_t)padded_m * (size_t)k);
    Buf xs_got = gpu_alloc(gpu, slot, (size_t)m * 4u);

    float *in2p = (float *)in2.ptr, *wp = (float *)w.ptr;
    for (size_t i = 0; i < (size_t)m * (size_t)(2u * k); i++)
      in2p[i] = rng_float(&rng) * 2.0f;
    for (uint32_t c = 0; c < k; c++) wp[c] = 0.5f + rng_float(&rng);
    memset(xq_ref.ptr, 0x5a, xq_ref.size);
    memset(xq_got.ptr, 0x5a, xq_got.size);

    SwigluChunkedParam swp = {m, k};
    ConvRotParam rp = {m, k, k};
    SwigluConvRotParam swrp = {m, k};
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_sw);
    gpu_bind_buffer(&cmd, 0, &in2, 0);
    gpu_bind_buffer(&cmd, 1, &mid, 0);
    gpu_copy_bytes(&cmd, 2, &swp, sizeof swp);
    gpu_run_kernel(&cmd, (k + 31u) / 32u, m, 1, 32u, 1, 1);

    gpu_select_kernel(&cmd, k_rot);
    gpu_bind_buffer(&cmd, 0, &mid, 0);
    gpu_bind_buffer(&cmd, 1, &xq_ref, 0);
    gpu_bind_buffer(&cmd, 2, &xs_ref, 0);
    gpu_copy_bytes(&cmd, 3, &rp, sizeof rp);
    gpu_run_kernel(&cmd, padded_m, 1, 1, (uint32_t)CONVROT_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_sw_rot);
    gpu_bind_buffer(&cmd, 0, &in2, 0);
    gpu_bind_buffer(&cmd, 1, &xq_got, 0);
    gpu_bind_buffer(&cmd, 2, &xs_got, 0);
    gpu_copy_bytes(&cmd, 3, &swrp, sizeof swrp);
    gpu_run_kernel(&cmd, padded_m, 1, 1, (uint32_t)CONVROT_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    if (memcmp(xq_ref.ptr, xq_got.ptr, (size_t)padded_m * (size_t)k) != 0 ||
        memcmp(xs_ref.ptr, xs_got.ptr, (size_t)m * 4u) != 0)
      sw_ok = 0;

    // Now test minimax_h3_rmsnorm_convrot against minimax_h3_rmsnorm + convrot.
    memset(xq_ref.ptr, 0x5a, xq_ref.size);
    memset(xq_got.ptr, 0x5a, xq_got.size);
    float eps = 1e-5f;
    RmsNormParam rnp = {k, eps};
    RmsNormConvRotParam rnrp = {m, k, eps};
    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_rms);
    gpu_bind_buffer(&cmd, 0, &in2, 0);
    gpu_bind_buffer(&cmd, 1, &w, 0);
    gpu_bind_buffer(&cmd, 2, &mid, 0);
    gpu_copy_bytes(&cmd, 3, &rnp, sizeof rnp);
    gpu_run_kernel(&cmd, m, 1, 1, (uint32_t)CONVROT_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_rot);
    gpu_bind_buffer(&cmd, 0, &mid, 0);
    gpu_bind_buffer(&cmd, 1, &xq_ref, 0);
    gpu_bind_buffer(&cmd, 2, &xs_ref, 0);
    gpu_copy_bytes(&cmd, 3, &rp, sizeof rp);
    gpu_run_kernel(&cmd, padded_m, 1, 1, (uint32_t)CONVROT_THREADS, 1, 1);

    gpu_select_kernel(&cmd, k_rms_rot);
    gpu_bind_buffer(&cmd, 0, &in2, 0);
    gpu_bind_buffer(&cmd, 1, &w, 0);
    gpu_bind_buffer(&cmd, 2, &xq_got, 0);
    gpu_bind_buffer(&cmd, 3, &xs_got, 0);
    gpu_copy_bytes(&cmd, 4, &rnrp, sizeof rnrp);
    gpu_run_kernel(&cmd, padded_m, 1, 1, (uint32_t)CONVROT_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);

    const int8_t *qr = (const int8_t *)xq_ref.ptr;
    const int8_t *qg = (const int8_t *)xq_got.ptr;
    const float *sr = (const float *)xs_ref.ptr;
    const float *sg = (const float *)xs_got.ptr;
    for (uint32_t r = 0; r < m; r++) {
      if (fabsf(sr[r] - sg[r]) > fmaxf(1e-7f, sr[r] * 1e-5f)) rms_ok = 0;
      for (uint32_t c = 0; c < k; c++)
        if (abs((int)qr[(size_t)r * k + c] - (int)qg[(size_t)r * k + c]) > 1)
          rms_ok = 0;
    }
    for (uint32_t r = m; r < padded_m; r++)
      for (uint32_t c = 0; c < k; c++)
        if (qg[(size_t)r * k + c] != 0) rms_ok = 0;

    gpu_free(&xs_got);
    gpu_free(&xq_got);
    gpu_free(&xs_ref);
    gpu_free(&xq_ref);
    gpu_free(&w);
    gpu_free(&mid);
    gpu_free(&in2);
  }
  check(sw_ok, "minimax_h3_swiglu_convrot matches unfused SwiGLU + ConvRot "
               "bit-for-bit across 1-pass and 2-pass widths");
  check(rms_ok,
        "minimax_h3_rmsnorm_convrot matches unfused RMSNorm + ConvRot across "
        "1-pass and 2-pass widths");
}

// Benchmarks one DiT-sized INT8-ConvRot linear (1024 x 5376 x 5376) and one
// Qwen3-VL-sized NVFP4 linear (256 x 5120 x 5120).
static void bench_realistic_shapes(Gpu *gpu, MemorySlot *slot) {
  int k_rot = gpu_find_kernel(gpu, "minimax_h3_convrot_quant_i8");
  int k_mm_i8 = gpu_find_kernel(gpu, "minimax_h3_mm_int8");
  int m = 1024, k = 5376, n = 5376;
  Buf w = gpu_alloc(gpu, slot, (size_t)n * (size_t)k);
  Buf ws = gpu_alloc(gpu, slot, (size_t)n * 4);
  Buf bias = gpu_alloc(gpu, slot, (size_t)n * 4);
  Buf x = gpu_alloc(gpu, slot, (size_t)m * (size_t)k * 4);
  Buf xq = gpu_alloc(gpu, slot, (size_t)m * (size_t)k);
  Buf xs = gpu_alloc(gpu, slot, (size_t)m * 4);
  Buf y = gpu_alloc(gpu, slot, (size_t)m * (size_t)n * 4);
  memset(w.ptr, 1, w.size);
  memset(ws.ptr, 0, ws.size);
  memset(bias.ptr, 0, bias.size);
  memset(x.ptr, 0, x.size);
  ConvRotParam rp = {(uint32_t)m, (uint32_t)k, (uint32_t)k};
  Int8MMParam mp = {(uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)n, 0};

  // Warmup, then timed passes for ConvRot and INT8 matmul separately.
  for (int iter = 0; iter < 5; iter++) {
    double t0 = now();
    GpuCommandBuffer cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_rot);
    gpu_bind_buffer(&cmd, 0, &x, 0);
    gpu_bind_buffer(&cmd, 1, &xq, 0);
    gpu_bind_buffer(&cmd, 2, &xs, 0);
    gpu_copy_bytes(&cmd, 3, &rp, sizeof rp);
    gpu_run_kernel(&cmd, (uint32_t)m, 1, 1, CONVROT_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);
    double t1 = now();

    cmd = gpu_begin(gpu);
    gpu_select_kernel(&cmd, k_mm_i8);
    gpu_bind_buffer(&cmd, 0, &w, 0);
    gpu_bind_buffer(&cmd, 1, &ws, 0);
    gpu_bind_buffer(&cmd, 2, &xq, 0);
    gpu_bind_buffer(&cmd, 3, &xs, 0);
    gpu_bind_buffer(&cmd, 4, &bias, 0);
    gpu_bind_buffer(&cmd, 5, &y, 0);
    gpu_copy_bytes(&cmd, 6, &mp, sizeof mp);
    gpu_run_kernel(&cmd, (uint32_t)(n / TILE_N) * (uint32_t)(m / TILE_M), 1, 1,
                   MM_THREADS, 1, 1);
    gpu_commit(&cmd);
    gpu_sync(gpu);
    double t2 = now();
    if (iter == 4)
      printf("INT8-ConvRot %dx%dx%d: rot %.2f ms, mm %.2f ms\n", m, n, k,
             (t1 - t0) * 1000.0, (t2 - t1) * 1000.0);
  }
  gpu_free(&y);
  gpu_free(&xs);
  gpu_free(&xq);
  gpu_free(&x);
  gpu_free(&bias);
  gpu_free(&ws);
  gpu_free(&w);
}

int main(void) {
  Memory *memory = memory_new();
  MemoryAccount account = {0};
  memory_account_init(memory, &account, "minimax-h3-quant-test", 0);
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  test_scale_unswizzle(gpu, &account);
  test_nvfp4_linear(gpu, slot);
  test_convrot_int8(gpu, slot);
  test_fused_convrot(gpu, slot);
  test_embed_int8(gpu, slot);
  bench_realistic_shapes(gpu, slot);

  return failures != 0;
}
