#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "reference/activations.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests how prompt passes turn activations into MXFP4 for the experts. It
// compares them with the scalar expert kernel's, on one made-up expert of
// DeepSeek V4.1 Flash's shape whose weights are all 1. It checks the quantized
// activations, the down products, the smallest half values in the gather, and
// activations over a range of exponents.

// Rows, the model's width, and an expert's hidden width.
static constexpr int M = 32, K = 5120, H = 2304;

// Bytes of an MXFP4 expert's codes, two to a byte, and of their scales, one
// for every 32 codes. An expert has three matrices of H by K.
static constexpr uint32_t EXPERT_W = 3 * H * K / 2, EXPERT_S = 3 * H * K / 32;

typedef struct {
  uint64_t p[3];
  float weight, pad;
} ExpertEntry;

typedef struct {
  uint64_t p[3];
  uint32_t start, count;
} Tile;

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);
  Buf weights = gpu_alloc(gpu, slot, EXPERT_W + EXPERT_S),
      entry = gpu_alloc(gpu, slot, sizeof(ExpertEntry));
  Buf tiles = gpu_alloc(gpu, slot, sizeof(Tile)),
      x = gpu_alloc(gpu, slot, M * K * 4), ids = gpu_alloc(gpu, slot, M * 4),
      pw = gpu_alloc(gpu, slot, M * 4);
  Buf xh = gpu_alloc(gpu, slot, (M + 64) * K * 2),
      h = gpu_alloc(gpu, slot, M * H * 4),
      quantized = gpu_alloc(gpu, slot, M * H * 4);
  Buf reference = gpu_alloc(gpu, slot, M * H * 4);
  Buf hh = gpu_alloc(gpu, slot, (M + 64) * H * 2),
      y = gpu_alloc(gpu, slot, M * K * 4);
  // The kernels read the weights through their GPU addresses, so the weights
  // must be resident.
  gpu_set_resident(&weights, 1);
  gpu_commit_residency(gpu);
  memset(weights.ptr, 0x22, EXPERT_W);  // E2M1 value 1 in every weight
  memset(weights.ptr + EXPERT_W, 127, EXPERT_S);  // E8M0 scale 1
  *(ExpertEntry *)entry.ptr =
      (ExpertEntry){{weights.address, weights.address + EXPERT_W, 0}, 1, 0};
  *(Tile *)tiles.ptr =
      (Tile){{weights.address, weights.address + EXPERT_W, 0}, 0, M};
  memset(xh.ptr, 0, xh.size);
  memset(hh.ptr, 0, hh.size);
  for (int r = 0; r < M; r++) {
    ((int *)ids.ptr)[r] = r;
    ((float *)pw.ptr)[r] = 1;
    float value = r == M - 1   ? 0x1p-31f
                  : r % 3 == 0 ? 0x1p-26f
                  : r % 3 == 1 ? 0x1p-12f
                               : -0x1p-12f;
    for (int d = 0; d < K; d++) ((float *)x.ptr)[r * K + d] = value;
  }
  int scalar = gpu_find_kernel(gpu, "moe_up"),
      gather = gpu_find_kernel(gpu, "gather_mxfp4");
  int batch = gpu_find_kernel(gpu, "moe_up_m"),
      quant = gpu_find_kernel(gpu, "act_quant");
  int quant_scaled = gpu_find_kernel(gpu, "act_quant_mxfp4"),
      down = gpu_find_kernel(gpu, "moe_down_m");
  uint32_t count = M;
  GpuCommandBuffer cmd = gpu_begin(gpu);
  for (int r = 0; r < M; r++) {
    gpu_select_kernel(&cmd, scalar);
    gpu_bind_buffer(&cmd, 0, &entry, 0),
        gpu_bind_buffer(&cmd, 1, &x, (size_t)r * K * 4),
        gpu_bind_buffer(&cmd, 2, &reference, (size_t)r * H * 4);
    gpu_run_kernel(&cmd, H / 32, 1, 1, 256, 1, 1);
  }
  gpu_select_kernel(&cmd, gather);
  gpu_bind_buffer(&cmd, 0, &x, 0), gpu_bind_buffer(&cmd, 1, &ids, 0),
      gpu_bind_buffer(&cmd, 2, &xh, 0);
  gpu_run_kernel(&cmd, K / 4 / 256, M, 1, 256, 1, 1);
  gpu_select_kernel(&cmd, batch);
  gpu_bind_buffer(&cmd, 0, &tiles, 0), gpu_bind_buffer(&cmd, 1, &xh, 0),
      gpu_bind_buffer(&cmd, 2, &pw, 0), gpu_bind_buffer(&cmd, 3, &h, 0),
      gpu_copy_bytes(&cmd, 4, &count, 4);
  gpu_run_kernel(&cmd, 1, H / 64, 1, 128, 1, 1);
  gpu_select_kernel(&cmd, quant);
  gpu_bind_buffer(&cmd, 0, &h, 0), gpu_bind_buffer(&cmd, 1, &quantized, 0);
  gpu_run_kernel(&cmd, M * H / 256, 1, 1, 256, 1, 1);
  gpu_select_kernel(&cmd, quant_scaled);
  gpu_bind_buffer(&cmd, 0, &h, 0), gpu_bind_buffer(&cmd, 1, &hh, 0);
  gpu_run_kernel(&cmd, M * H / 256, 1, 1, 256, 1, 1);
  gpu_select_kernel(&cmd, down);
  gpu_bind_buffer(&cmd, 0, &tiles, 0), gpu_bind_buffer(&cmd, 1, &hh, 0),
      gpu_bind_buffer(&cmd, 2, &y, 0), gpu_copy_bytes(&cmd, 3, &count, 4);
  gpu_run_kernel(&cmd, 1, K / 128, 1, 128, 1, 1);
  gpu_commit(&cmd), gpu_sync(gpu);
  int small = 0, ordinary = 0, down_errors = 0;
  for (int r = 0; r < M; r++)
    for (int d = 0; d < H; d++) {
      float a = ((float *)reference.ptr)[r * H + d],
            b = ((float *)quantized.ptr)[r * H + d];
      if (!isfinite(a) || !isfinite(b) || memcmp(&a, &b, sizeof a)) {
        if (r % 3 == 0)
          small++;
        else
          ordinary++;
      }
    }
  for (int r = 0; r < M; r++) {
    float expected =
        ref_from_bf16(ref_bf16(((float *)reference.ptr)[r * H] * H));
    for (int d = 0; d < K; d++) {
      float actual = ((float *)y.ptr)[r * K + d];
      down_errors +=
          !isfinite(actual) || memcmp(&expected, &actual, sizeof actual) != 0;
    }
  }
  int gather_edge = ((uint16_t *)xh.ptr)[(M - 1) * K] !=
                    1;  // 2^-31 * 128 is half's smallest subnormal.
  const int exponents[] = {-31, -30, -26, -17, -12, -8, 0, 8};
  for (int r = 0; r < M; r++) {
    float value = ldexpf(r & 8 ? -1 : 1, exponents[r % 8]);
    for (int d = 0; d < H; d++) ((float *)h.ptr)[r * H + d] = value;
  }
  cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, quant_scaled);
  gpu_bind_buffer(&cmd, 0, &h, 0), gpu_bind_buffer(&cmd, 1, &hh, 0);
  gpu_run_kernel(&cmd, M * H / 256, 1, 1, 256, 1, 1);
  gpu_select_kernel(&cmd, down);
  gpu_bind_buffer(&cmd, 0, &tiles, 0), gpu_bind_buffer(&cmd, 1, &hh, 0),
      gpu_bind_buffer(&cmd, 2, &y, 0), gpu_copy_bytes(&cmd, 3, &count, 4);
  gpu_run_kernel(&cmd, 1, K / 128, 1, 128, 1, 1);
  gpu_commit(&cmd), gpu_sync(gpu);
  int edge_down = 0;
  for (int r = 0; r < M; r++) {
    float expected = ref_from_bf16(ref_bf16(((float *)h.ptr)[r * H] * H));
    for (int d = 0; d < K; d++) {
      float actual = ((float *)y.ptr)[r * K + d];
      edge_down +=
          !isfinite(actual) || memcmp(&expected, &actual, sizeof actual) != 0;
    }
  }
  printf("small reference %.9g, prompt staging %.9g\n",
         ((float *)reference.ptr)[0], ((float *)quantized.ptr)[0]);
  check(!small, "tiny activations quantize as the scalar kernel's do");
  check(!ordinary, "ordinary activations quantize as the scalar kernel's do");
  check(!down_errors, "the down products of the staged activations are exact");
  check(!gather_edge, "the gather keeps half's smallest subnormal");
  check(!edge_down, "activations from 2^-31 to 2^8 give exact down products");
  return failures != 0;
}
