#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <string.h>

// Tests LFM3-VL's vision kernels against the CPU on a grid of patches in rows
// of 2 x 2 squares: lfm3_vision_conv, the gated 3 x 3 convolution over the
// grid; lfm3_vision_heads, the per-head norms and the RoPE that turns by a
// patch's row and column; and lfm3_vision_attend, attention of every patch
// over every patch, for an image whose patches fill no whole tile and for
// one smaller than a tile; and lfm3_vision_add, a residual. The CPU finds
// each patch's place by walking the squares in order, and its attention reads
// the same BF16 queries, keys and values, and keeps the probabilities in
// double precision.

// The attention's shape: query heads of 64 values sharing key heads, and the
// rows that the kernel takes in a threadgroup.
static constexpr int HEAD_DIM = 64, PAIRS = HEAD_DIM / 2, QUERY_HEADS = 16,
                     KEY_HEADS = 8, SHARING = QUERY_HEADS / KEY_HEADS,
                     TILE_ROWS = 64, THREADS = 128;
// The side of a token's square of patches, the side of the convolution's
// filter, and the pairs of a head that turn by a patch's row.
static constexpr int MERGE = 2, FILTER = 3, ROW_PAIRS = PAIRS / 2;
// The most patches of a test's image.
static constexpr int MOST_PATCHES = 160;

typedef struct {
  uint32_t dim, grid_w, grid_h;
} ConvParams;

typedef struct {
  uint32_t rows, padded, grid_w;
  float eps;
} HeadsParams;

typedef struct {
  uint32_t rows, padded;
} AttendParams;

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

// The BF16 value nearest to f, ties to even.
static uint16_t bf16_bits(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof bits);
  return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

static int round_up(int n, int unit) { return (n + unit - 1) / unit * unit; }

static Gpu *gpu;
static MemorySlot *slot;

// Each patch's column and row, and the patch at each place, for a grid of
// grid_w x grid_h patches: the squares in rows, each square's patches in rows.
static void walk_squares(int grid_w, int grid_h, int *x, int *y, int *patch) {
  int p = 0;
  for (int sy = 0; sy < grid_h / MERGE; sy++)
    for (int sx = 0; sx < grid_w / MERGE; sx++)
      for (int my = 0; my < MERGE; my++)
        for (int mx = 0; mx < MERGE; mx++, p++) {
          x[p] = sx * MERGE + mx, y[p] = sy * MERGE + my;
          patch[y[p] * grid_w + x[p]] = p;
        }
}

static void test_conv(void) {
  enum { GRID_W = 8, GRID_H = 6, PATCHES = GRID_W * GRID_H, DIM = 32 };

  Buf bcx = gpu_alloc(gpu, slot, PATCHES * 3 * DIM * 4),
      filter = gpu_alloc(gpu, slot, DIM * FILTER * FILTER * 4),
      out = gpu_alloc(gpu, slot, PATCHES * DIM * 4);
  float *in = (float *)bcx.ptr, *w = (float *)filter.ptr;
  for (int i = 0; i < PATCHES * 3 * DIM; i++) in[i] = random_value();
  for (int i = 0; i < DIM * FILTER * FILTER; i++) w[i] = random_value();
  ConvParams p = {DIM, GRID_W, GRID_H};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_vision_conv"));
  gpu_bind_buffer(&c, 0, &bcx, 0), gpu_bind_buffer(&c, 1, &filter, 0);
  gpu_bind_buffer(&c, 2, &out, 0), gpu_copy_bytes(&c, 3, &p, sizeof p);
  gpu_run_kernel(&c, 1, PATCHES, 1, DIM, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  int x[PATCHES], y[PATCHES], patch[PATCHES];
  walk_squares(GRID_W, GRID_H, x, y, patch);
  double worst = 0;
  const float *o = (const float *)out.ptr;
  for (int q = 0; q < PATCHES; q++)
    for (int ch = 0; ch < DIM; ch++) {
      double sum = 0;
      for (int dy = 0; dy < FILTER; dy++)
        for (int dx = 0; dx < FILTER; dx++) {
          int nx = x[q] + dx - FILTER / 2, ny = y[q] + dy - FILTER / 2;
          if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H) continue;
          const float *near = in + patch[ny * GRID_W + nx] * 3 * DIM;
          sum += (double)w[(ch * FILTER + dy) * FILTER + dx] * near[ch] *
                 near[2 * DIM + ch];
        }
      double want = in[q * 3 * DIM + DIM + ch] * sum;
      worst = fmax(worst, fabs(o[q * DIM + ch] - want));
    }
  check(worst < 1e-5, "the gated convolution sums the 3 x 3 patches around "
                      "each patch of the grid, beyond it zero");
  Buf *all[] = {&bcx, &filter, &out};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

static void test_heads(void) {
  enum { GRID_W = 8, GRID_H = 6, ROWS = GRID_W * GRID_H, PADDED = 128 };

  Buf q = gpu_alloc(gpu, slot, ROWS * QUERY_HEADS * HEAD_DIM * 4),
      k = gpu_alloc(gpu, slot, ROWS * KEY_HEADS * HEAD_DIM * 4),
      v = gpu_alloc(gpu, slot, ROWS * KEY_HEADS * HEAD_DIM * 4),
      norms = gpu_alloc(gpu, slot, 2 * HEAD_DIM * 4),
      inv_freq = gpu_alloc(gpu, slot, ROW_PAIRS * 4),
      qh = gpu_alloc(gpu, slot, QUERY_HEADS * PADDED * HEAD_DIM * 2),
      kh = gpu_alloc(gpu, slot, KEY_HEADS * PADDED * HEAD_DIM * 2),
      vh = gpu_alloc(gpu, slot, KEY_HEADS * PADDED * HEAD_DIM * 2);
  float *qs = (float *)q.ptr, *ks = (float *)k.ptr, *vs = (float *)v.ptr,
        *w = (float *)norms.ptr, *freq = (float *)inv_freq.ptr;
  for (int i = 0; i < ROWS * QUERY_HEADS * HEAD_DIM; i++)
    qs[i] = 3 * random_value();
  for (int i = 0; i < ROWS * KEY_HEADS * HEAD_DIM; i++)
    ks[i] = 3 * random_value(), vs[i] = random_value();
  for (int i = 0; i < 2 * HEAD_DIM; i++) w[i] = 1 + random_value() / 2;
  for (int i = 0; i < ROW_PAIRS; i++)
    freq[i] = 1.0f / powf(1e6f, (float)(2 * i) / PAIRS);
  HeadsParams p = {ROWS, PADDED, GRID_W, 1e-5f};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_vision_heads"));
  gpu_bind_buffer(&c, 0, &q, 0), gpu_bind_buffer(&c, 1, &k, 0);
  gpu_bind_buffer(&c, 2, &v, 0), gpu_bind_buffer(&c, 3, &norms, 0);
  gpu_bind_buffer(&c, 4, &norms, HEAD_DIM * 4);
  gpu_bind_buffer(&c, 5, &inv_freq, 0), gpu_bind_buffer(&c, 6, &qh, 0);
  gpu_bind_buffer(&c, 7, &kh, 0), gpu_bind_buffer(&c, 8, &vh, 0);
  gpu_copy_bytes(&c, 9, &p, sizeof p);
  gpu_run_kernel(&c, QUERY_HEADS + 2 * KEY_HEADS, ROWS, 1, PAIRS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  int x[ROWS], y[ROWS], patch[ROWS];
  walk_squares(GRID_W, GRID_H, x, y, patch);
  // Each head's values as the CPU computes them: a difference of one BF16
  // step is allowed, since cos and sin may differ in their last bits.
  double worst = 0;
  int values_copied = 1;
  for (int r = 0; r < ROWS; r++)
    for (int h = 0; h < QUERY_HEADS + KEY_HEADS; h++) {
      int query = h < QUERY_HEADS, head = query ? h : h - QUERY_HEADS;
      const float *src = query ? qs + (r * QUERY_HEADS + head) * HEAD_DIM
                               : ks + (r * KEY_HEADS + head) * HEAD_DIM;
      const float *weight = w + (query ? 0 : HEAD_DIM);
      const uint16_t *got = (const uint16_t *)(query ? qh.ptr : kh.ptr) +
                            ((size_t)head * PADDED + r) * HEAD_DIM;
      double squares = 0;
      for (int i = 0; i < HEAD_DIM; i++) squares += (double)src[i] * src[i];
      float scale = (float)(1 / sqrt(squares / HEAD_DIM + 1e-5));
      for (int i = 0; i < PAIRS; i++) {
        float a = weight[i] * (src[i] * scale),
              b = weight[i + PAIRS] * (src[i + PAIRS] * scale),
              angle = i < ROW_PAIRS ? (float)y[r] * freq[i]
                                    : (float)x[r] * freq[i - ROW_PAIRS];
        float want[2] = {a * cosf(angle) - b * sinf(angle),
                         b * cosf(angle) + a * sinf(angle)},
              have[2] = {bf16_value(got[i]), bf16_value(got[i + PAIRS])};
        for (int j = 0; j < 2; j++)
          worst = fmax(worst, fabs(have[j] - want[j]) / fabs(want[j]));
      }
    }
  for (int r = 0; r < ROWS; r++)
    for (int h = 0; h < KEY_HEADS; h++)
      for (int i = 0; i < HEAD_DIM; i++)
        values_copied &=
            ((const uint16_t *)
                 vh.ptr)[((size_t)h * PADDED + r) * HEAD_DIM + i] ==
            bf16_bits(vs[(r * KEY_HEADS + h) * HEAD_DIM + i]);
  check(worst <= 0x1p-8 && values_copied,
        "the heads are normed and turned by each patch's row and column; "
        "values are copied as BF16");
  Buf *all[] = {&q, &k, &v, &norms, &inv_freq, &qh, &kh, &vh};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

// Runs lfm3_vision_attend on an image of n patches, and checks every output
// against the CPU's attention.
static void check_attention(int n, const char *what) {
  int padded = round_up(n, TILE_ROWS) + TILE_ROWS;
  Buf qh = gpu_alloc(gpu, slot, (size_t)QUERY_HEADS * padded * HEAD_DIM * 2),
      kh = gpu_alloc(gpu, slot, (size_t)KEY_HEADS * padded * HEAD_DIM * 2),
      vh = gpu_alloc(gpu, slot, (size_t)KEY_HEADS * padded * HEAD_DIM * 2),
      out = gpu_alloc(gpu, slot, (size_t)n * QUERY_HEADS * HEAD_DIM * 4);
  uint16_t *q = (uint16_t *)qh.ptr, *k = (uint16_t *)kh.ptr,
           *v = (uint16_t *)vh.ptr;
  for (int h = 0; h < QUERY_HEADS; h++)
    for (int r = 0; r < padded; r++)
      for (int i = 0; i < HEAD_DIM; i++)
        q[((size_t)h * padded + r) * HEAD_DIM + i] =
            r < n ? bf16_bits(3 * random_value()) : 0;
  for (int h = 0; h < KEY_HEADS; h++)
    for (int r = 0; r < padded; r++)
      for (int i = 0; i < HEAD_DIM; i++) {
        size_t at = ((size_t)h * padded + r) * HEAD_DIM + i;
        k[at] = r < n ? bf16_bits(3 * random_value()) : 0;
        v[at] = r < n ? bf16_bits(random_value()) : 0;
      }
  AttendParams p = {(uint32_t)n, (uint32_t)padded};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_vision_attend"));
  gpu_bind_buffer(&c, 0, &qh, 0), gpu_bind_buffer(&c, 1, &kh, 0);
  gpu_bind_buffer(&c, 2, &vh, 0), gpu_bind_buffer(&c, 3, &out, 0);
  gpu_copy_bytes(&c, 4, &p, sizeof p);
  gpu_run_kernel(&c, (uint32_t)(n + TILE_ROWS - 1) / TILE_ROWS, QUERY_HEADS, 1,
                 THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  // The probabilities' rounding to BF16 changes an output by about 2^-9 of
  // the values' magnitude, at most 1.
  double worst = 0, weights[MOST_PATCHES];
  const float *o = (const float *)out.ptr;
  for (int r = 0; r < n; r++)
    for (int h = 0; h < QUERY_HEADS; h++) {
      int kv = h / SHARING;
      double most = -INFINITY, sum = 0;
      for (int key = 0; key < n; key++) {
        double score = 0;
        for (int i = 0; i < HEAD_DIM; i++)
          score +=
              (double)bf16_value(q[((size_t)h * padded + r) * HEAD_DIM + i]) *
              bf16_value(k[((size_t)kv * padded + key) * HEAD_DIM + i]);
        weights[key] = score / sqrt(HEAD_DIM);
        most = fmax(most, weights[key]);
      }
      for (int key = 0; key < n; key++)
        sum += weights[key] = exp(weights[key] - most);
      for (int i = 0; i < HEAD_DIM; i++) {
        double value = 0;
        for (int key = 0; key < n; key++)
          value += weights[key] *
                   bf16_value(v[((size_t)kv * padded + key) * HEAD_DIM + i]);
        worst =
            fmax(worst, fabs(o[((size_t)r * QUERY_HEADS + h) * HEAD_DIM + i] -
                             value / sum));
      }
    }
  check(worst < 4e-3, what);
  Buf *all[] = {&qh, &kh, &vh, &out};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

static void test_add(void) {
  enum { COUNT = 1000 };

  Buf x = gpu_alloc(gpu, slot, COUNT * 4), y = gpu_alloc(gpu, slot, COUNT * 4);
  float *xs = (float *)x.ptr, *ys = (float *)y.ptr, want[COUNT];
  for (int i = 0; i < COUNT; i++)
    xs[i] = random_value(), ys[i] = random_value(), want[i] = xs[i] + ys[i];
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_vision_add"));
  gpu_bind_buffer(&c, 0, &x, 0), gpu_bind_buffer(&c, 1, &y, 0);
  gpu_run_kernel(&c, COUNT / 8, 1, 1, 8, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(!memcmp(xs, want, sizeof want), "a residual adds its values");
  gpu_free(&x), gpu_free(&y);
}

int main(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  slot = memory_shared(memory);
  size_t length;
  const char *source = kernel_library_source(&length);
  gpu = gpu_new(source, length);
  test_conv();
  test_heads();
  check_attention(150, "every patch attends to every patch, 150 patches in "
                       "tiles of 64");
  check_attention(40, "40 patches, fewer than a tile");
  test_add();
  return failures != 0;
}
