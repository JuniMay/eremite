#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <string.h>

// Tests LFM3's kernels over rows against the CPU: the embedding, the RMSNorm
// and the sandwich norm's residual, the copy, the gated short convolution of a
// block (at the sequence's start and after it), and the feed-forward's
// activation.

// The values per row of the norm tests, the channels of the convolution test,
// and the threads of their threadgroups, as the model runs them.
static constexpr int ROW = 2048, CHANNELS = 512, THREADS = 256;
// The convolution's filter width, and its rows before a row.
static constexpr int WIDTH = 7, BEFORE = WIDTH - 1;

typedef struct {
  uint32_t n;
  float eps;
} NormParams;

typedef struct {
  uint32_t dim, width, before;
} ConvParams;

static uint64_t seed = 88172645463325252ull;

// A pseudo-random number in [-1, 1).
static float random_value(void) {
  seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
  return (float)((seed >> 11) * 0x1p-53 * 2 - 1);
}

static float bf16_value(uint16_t u) {
  uint32_t bits = (uint32_t)u << 16;
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

static Gpu *gpu;
static MemorySlot *slot;

static Buf alloc_floats(size_t count) {
  return gpu_alloc(gpu, slot, count * sizeof(float));
}

static void fill_random(Buf *b, size_t count) {
  for (size_t i = 0; i < count; i++) ((float *)b->ptr)[i] = random_value();
}

// The largest difference between count floats of got and want, over the
// largest magnitude in want.
static double relative_difference(const float *got, const double *want,
                                  size_t count) {
  double worst = 0, scale = 0;
  for (size_t i = 0; i < count; i++)
    worst = fmax(worst, fabs(got[i] - want[i])),
    scale = fmax(scale, fabs(want[i]));
  return worst / scale;
}

static void test_embed(void) {
  enum { VOCAB = 10, DIM = 256, ROWS = 4 };

  Buf table = gpu_alloc(gpu, slot, VOCAB * DIM * 2),
      ids = gpu_alloc(gpu, slot, ROWS * 4), x = alloc_floats(ROWS * DIM);
  for (int i = 0; i < VOCAB * DIM; i++) {  // the upper half of a float's bits
    float value = random_value();
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    ((uint16_t *)table.ptr)[i] = (uint16_t)(bits >> 16);
  }
  int32_t chosen[ROWS] = {3, -1, 7, 0};
  memcpy(ids.ptr, chosen, sizeof chosen);
  for (int i = 0; i < ROWS * DIM; i++) ((float *)x.ptr)[i] = 42;
  uint32_t dim = DIM;
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_embed"));
  gpu_bind_buffer(&c, 0, &table, 0), gpu_bind_buffer(&c, 1, &ids, 0);
  gpu_bind_buffer(&c, 2, &x, 0), gpu_copy_bytes(&c, 3, &dim, sizeof dim);
  gpu_run_kernel(&c, DIM / THREADS, ROWS, 1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  int same = 1;
  for (int r = 0; r < ROWS; r++)
    for (int d = 0; d < DIM; d++) {
      float got = ((float *)x.ptr)[r * DIM + d];
      float want =
          chosen[r] < 0
              ? 42
              : bf16_value(((uint16_t *)table.ptr)[chosen[r] * DIM + d]);
      same &= got == want;
    }
  check(same, "each row gets its token's BF16 row as FP32, and a row with a "
              "negative id keeps its values");
  gpu_free(&table), gpu_free(&ids), gpu_free(&x);
}

static void test_norms(void) {
  enum { ROWS = 3 };

  Buf x = alloc_floats(ROWS * ROW), w = alloc_floats(ROW),
      y = alloc_floats(ROWS * ROW), sum = alloc_floats(ROWS * ROW);
  fill_random(&x, ROWS * ROW), fill_random(&w, ROW),
      fill_random(&sum, ROWS * ROW);
  double want[ROWS * ROW], added[ROWS * ROW];
  float eps = 1e-5f;
  for (int r = 0; r < ROWS; r++) {
    const float *row = (float *)x.ptr + r * ROW;
    double squares = 0;
    for (int i = 0; i < ROW; i++) squares += (double)row[i] * row[i];
    double scale = 1 / sqrt(squares / ROW + eps);
    for (int i = 0; i < ROW; i++) {
      want[r * ROW + i] = ((float *)w.ptr)[i] * (row[i] * scale);
      added[r * ROW + i] = ((float *)sum.ptr)[r * ROW + i] + want[r * ROW + i];
    }
  }
  NormParams p = {ROW, eps};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_norm"));
  gpu_bind_buffer(&c, 0, &x, 0), gpu_bind_buffer(&c, 1, &w, 0);
  gpu_bind_buffer(&c, 2, &y, 0), gpu_copy_bytes(&c, 3, &p, sizeof p);
  gpu_run_kernel(&c, ROWS, 1, 1, THREADS, 1, 1);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_norm_add"));
  gpu_bind_buffer(&c, 0, &sum, 0), gpu_bind_buffer(&c, 1, &x, 0);
  gpu_bind_buffer(&c, 2, &w, 0), gpu_copy_bytes(&c, 3, &p, sizeof p);
  gpu_run_kernel(&c, ROWS, 1, 1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(relative_difference((const float *)y.ptr, want, ROWS * ROW) < 1e-6,
        "RMSNorm with weights matches the CPU");
  check(relative_difference((const float *)sum.ptr, added, ROWS * ROW) < 1e-6,
        "the sandwich norm's residual adds the normed rows");
  gpu_free(&x), gpu_free(&w), gpu_free(&y), gpu_free(&sum);
}

static void test_copy(void) {
  enum { COUNT = 1000 };

  Buf x = alloc_floats(COUNT), y = alloc_floats(COUNT + 1);
  fill_random(&x, COUNT);
  ((float *)y.ptr)[COUNT] = 42;
  uint32_t count = COUNT;
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_copy"));
  gpu_bind_buffer(&c, 0, &x, 0), gpu_bind_buffer(&c, 1, &y, 0);
  gpu_copy_bytes(&c, 2, &count, sizeof count);
  gpu_run_kernel(&c, (COUNT + THREADS - 1) / THREADS, 1, 1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(!memcmp(x.ptr, y.ptr, COUNT * 4) && ((float *)y.ptr)[COUNT] == 42,
        "the copy writes its count of floats and nothing after them");
  gpu_free(&x), gpu_free(&y);
}

// Runs the convolution of `rows` rows after `before` rows of bcx, and checks
// it against the CPU.
static void check_conv(int rows, int before, const char *what) {
  int stored = rows + before;
  Buf bcx = alloc_floats((size_t)stored * 3 * CHANNELS),
      w = alloc_floats(CHANNELS * WIDTH),
      y = alloc_floats((size_t)rows * CHANNELS);
  fill_random(&bcx, (size_t)stored * 3 * CHANNELS),
      fill_random(&w, CHANNELS * WIDTH);
  const float *in = (const float *)bcx.ptr, *filter = (const float *)w.ptr;
  double want[64 * CHANNELS];
  for (int t = 0; t < rows; t++)
    for (int c = 0; c < CHANNELS; c++) {
      double sum = 0;
      for (int j = 0; j < WIDTH; j++) {
        int r = t + before - BEFORE +
                j;  // a row of bcx; below 0, before the sequence
        if (r < 0) continue;
        sum += filter[c * WIDTH + j] * ((double)in[(r * 3) * CHANNELS + c] *
                                        in[(r * 3 + 2) * CHANNELS + c]);
      }
      want[t * CHANNELS + c] = in[((t + before) * 3 + 1) * CHANNELS + c] * sum;
    }
  ConvParams p = {CHANNELS, WIDTH, (uint32_t)before};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_conv"));
  gpu_bind_buffer(&c, 0, &bcx, 0), gpu_bind_buffer(&c, 1, &w, 0);
  gpu_bind_buffer(&c, 2, &y, 0), gpu_copy_bytes(&c, 3, &p, sizeof p);
  gpu_run_kernel(&c, CHANNELS / THREADS, (uint32_t)rows, 1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(relative_difference((const float *)y.ptr, want,
                            (size_t)rows * CHANNELS) < 1e-6,
        what);
  gpu_free(&bcx), gpu_free(&w), gpu_free(&y);
}

static void test_swiglu(void) {
  enum { COUNT = 4096 };

  Buf g = alloc_floats(COUNT), u = alloc_floats(COUNT), y = alloc_floats(COUNT);
  fill_random(&g, COUNT), fill_random(&u, COUNT);
  double want[COUNT];
  for (int i = 0; i < COUNT; i++) {
    double a = ((float *)g.ptr)[i] * 8;
    ((float *)g.ptr)[i] = (float)a;
    want[i] = a / (1 + exp(-a)) * ((float *)u.ptr)[i];
  }
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_swiglu"));
  gpu_bind_buffer(&c, 0, &g, 0), gpu_bind_buffer(&c, 1, &u, 0);
  gpu_bind_buffer(&c, 2, &y, 0);
  gpu_run_kernel(&c, COUNT / THREADS, 1, 1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(relative_difference((const float *)y.ptr, want, COUNT) < 1e-6,
        "the feed-forward's activation is silu(g) u");
  gpu_free(&g), gpu_free(&u), gpu_free(&y);
}

int main(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  slot = memory_shared(memory);
  size_t length;
  const char *source = kernel_library_source(&length);
  gpu = gpu_new(source, length);
  test_embed();
  test_norms();
  test_copy();
  check_conv(10, BEFORE,
             "the convolution of a block after the sequence's start "
             "reads the 6 rows before it");
  check_conv(10, 2,
             "the convolution of a block 2 rows after the start counts "
             "the rows before the start as zero");
  check_conv(10, 0,
             "the convolution of the first block counts the rows "
             "before it as zero");
  test_swiglu();
  return failures != 0;
}
