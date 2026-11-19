#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <string.h>

// Tests the kernels of LFM3-VL's projector against the CPU: lfm3_layer_norm
// on rows of the vision tower's width, lfm3_bias_gelu (GELU with erf, of each
// value plus its column's bias) and lfm3_add_bias.

// The vision tower's width, the rows of a test, and the threads of the norm's
// threadgroup.
static constexpr int WIDTH = 1024, ROWS = 5, NORM_THREADS = 256;

typedef struct {
  uint32_t n;
  float eps;
} NormParams;

static uint64_t seed = 88172645463325252ull;

// A pseudo-random number in [-1, 1).
static float random_value(void) {
  seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
  return (float)((seed >> 11) * 0x1p-53 * 2 - 1);
}

static Gpu *gpu;
static MemorySlot *slot;

static void test_layer_norm(void) {
  Buf x = gpu_alloc(gpu, slot, ROWS * WIDTH * 4),
      w = gpu_alloc(gpu, slot, WIDTH * 4), b = gpu_alloc(gpu, slot, WIDTH * 4),
      y = gpu_alloc(gpu, slot, ROWS * WIDTH * 4);
  float *xs = (float *)x.ptr, *ws = (float *)w.ptr, *bs = (float *)b.ptr;
  for (int i = 0; i < ROWS * WIDTH; i++) xs[i] = 4 * random_value() + 1;
  for (int i = 0; i < WIDTH; i++)
    ws[i] = 1 + random_value() / 2, bs[i] = random_value();
  NormParams p = {WIDTH, 1e-6f};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_layer_norm"));
  gpu_bind_buffer(&c, 0, &x, 0), gpu_bind_buffer(&c, 1, &w, 0);
  gpu_bind_buffer(&c, 2, &b, 0), gpu_bind_buffer(&c, 3, &y, 0);
  gpu_copy_bytes(&c, 4, &p, sizeof p);
  gpu_run_kernel(&c, ROWS, 1, 1, NORM_THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  double worst = 0;
  const float *ys = (const float *)y.ptr;
  for (int r = 0; r < ROWS; r++) {
    const float *row = xs + r * WIDTH;
    double mean = 0, var = 0;
    for (int i = 0; i < WIDTH; i++) mean += row[i];
    mean /= WIDTH;
    for (int i = 0; i < WIDTH; i++) var += (row[i] - mean) * (row[i] - mean);
    var /= WIDTH;
    for (int i = 0; i < WIDTH; i++) {
      double want = ws[i] * (row[i] - mean) / sqrt(var + 1e-6) + bs[i];
      worst = fmax(worst, fabs(ys[r * WIDTH + i] - want));
    }
  }
  check(worst < 1e-5, "LayerNorm centers each row, scales it to unit "
                      "variance, then applies its weights and bias");
  Buf *all[] = {&x, &w, &b, &y};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

// Runs the kernel `name` on rows of WIDTH values plus their columns' biases,
// and returns the largest difference from want(value + bias).
static double check_bias(const char *name, double (*want)(double)) {
  Buf x = gpu_alloc(gpu, slot, ROWS * WIDTH * 4),
      b = gpu_alloc(gpu, slot, WIDTH * 4);
  float *xs = (float *)x.ptr, *bs = (float *)b.ptr, in[ROWS * WIDTH];
  for (int i = 0; i < ROWS * WIDTH; i++) in[i] = xs[i] = 5 * random_value();
  for (int i = 0; i < WIDTH; i++) bs[i] = random_value();
  uint32_t n = WIDTH;
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, name));
  gpu_bind_buffer(&c, 0, &x, 0), gpu_bind_buffer(&c, 1, &b, 0);
  gpu_copy_bytes(&c, 2, &n, sizeof n);
  gpu_run_kernel(&c, ROWS * WIDTH / NORM_THREADS, 1, 1, NORM_THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  double worst = 0;
  for (int i = 0; i < ROWS * WIDTH; i++)
    worst = fmax(worst, fabs(xs[i] - want((double)in[i] + bs[i % WIDTH])));
  gpu_free(&x), gpu_free(&b);
  return worst;
}

static double gelu(double z) { return z * (1 + erf(z / sqrt(2))) / 2; }

static double same(double z) { return z; }

int main(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  slot = memory_shared(memory);
  size_t length;
  const char *source = kernel_library_source(&length);
  gpu = gpu_new(source, length);
  test_layer_norm();
  check(check_bias("lfm3_bias_gelu", gelu) < 2e-6,
        "GELU with erf, of each value plus its column's bias");
  check(check_bias("lfm3_add_bias", same) < 1e-6,
        "each value plus its column's bias");
  return failures != 0;
}
