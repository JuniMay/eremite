#include "check.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <string.h>

// Tests LFM3's attention kernels against the CPU: lfm3_heads, the per-head
// norms and RoPE of a block that starts after the sequence's start, and
// lfm3_attend under d1's question layout (a state and three questions, and a
// lone question) and under a sliding window shorter than the sequence, each
// for a whole sequence and for a block after its start. The CPU's attention
// reads the same BF16 queries, keys and values, and keeps the probabilities in
// double precision.

// The attention's shape: query heads of 128 values sharing key heads, and the
// rows that the kernel takes in a threadgroup.
static constexpr int HEAD_DIM = 128, PAIRS = HEAD_DIM / 2, QUERY_HEADS = 16,
                     KEY_HEADS = 4, SHARING = QUERY_HEADS / KEY_HEADS,
                     TILE_ROWS = 64, THREADS = 128;
// The most rows of a test's sequence, and of its questions.
static constexpr int MOST_ROWS = 704, MOST_QUESTIONS = 4;

typedef struct {
  uint32_t rows, first, padded, rotated;
  float eps;
} HeadsParams;

typedef struct {
  uint32_t rows, first, block_rows, padded, state, window;
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

static void test_heads(int rotated, const char *what) {
  enum { ROWS = 5, FIRST = 3, PADDED = 128 };

  Buf q = gpu_alloc(gpu, slot, ROWS * QUERY_HEADS * HEAD_DIM * 4),
      k = gpu_alloc(gpu, slot, ROWS * KEY_HEADS * HEAD_DIM * 4),
      v = gpu_alloc(gpu, slot, ROWS * KEY_HEADS * HEAD_DIM * 4),
      norms = gpu_alloc(gpu, slot, 2 * HEAD_DIM * 4),
      inv_freq = gpu_alloc(gpu, slot, PAIRS * 4),
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
  for (int i = 0; i < PAIRS; i++)
    freq[i] = 1.0f / powf(1e6f, (float)(2 * i) / HEAD_DIM);
  memset(qh.ptr, 0xFF, qh.size), memset(kh.ptr, 0xFF, kh.size);
  HeadsParams p = {ROWS, FIRST, PADDED, (uint32_t)rotated, 1e-5f};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_heads"));
  gpu_bind_buffer(&c, 0, &q, 0), gpu_bind_buffer(&c, 1, &k, 0);
  gpu_bind_buffer(&c, 2, &v, 0), gpu_bind_buffer(&c, 3, &norms, 0);
  gpu_bind_buffer(&c, 4, &norms, HEAD_DIM * 4);
  gpu_bind_buffer(&c, 5, &inv_freq, 0), gpu_bind_buffer(&c, 6, &qh, 0);
  gpu_bind_buffer(&c, 7, &kh, 0), gpu_bind_buffer(&c, 8, &vh, 0);
  gpu_copy_bytes(&c, 9, &p, sizeof p);
  gpu_run_kernel(&c, QUERY_HEADS + 2 * KEY_HEADS, ROWS, 1, PAIRS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  // Each head's values as the CPU computes them, then rounded to BF16: a
  // difference of one BF16 step is allowed, since cos and sin may differ in
  // their last bits.
  double worst = 0;
  int values_copied = 1, others_kept = 1;
  for (int r = 0; r < ROWS; r++)
    for (int h = 0; h < QUERY_HEADS + KEY_HEADS; h++) {
      int query = h < QUERY_HEADS, head = query ? h : h - QUERY_HEADS;
      const float *src = query ? qs + (r * QUERY_HEADS + head) * HEAD_DIM
                               : ks + (r * KEY_HEADS + head) * HEAD_DIM;
      const float *weight = w + (query ? 0 : HEAD_DIM);
      const uint16_t *got = (const uint16_t *)(query ? qh.ptr : kh.ptr) +
                            ((size_t)head * PADDED + FIRST + r) * HEAD_DIM;
      double squares = 0;
      for (int i = 0; i < HEAD_DIM; i++) squares += (double)src[i] * src[i];
      float scale = (float)(1 / sqrt(squares / HEAD_DIM + 1e-5));
      for (int i = 0; i < PAIRS; i++) {
        float a = weight[i] * (src[i] * scale),
              b = weight[i + PAIRS] * (src[i + PAIRS] * scale);
        if (i < rotated) {
          float angle = (float)(FIRST + r) * freq[i];
          float turned = a * cosf(angle) - b * sinf(angle);
          b = b * cosf(angle) + a * sinf(angle), a = turned;
        }
        float want[2] = {a, b},
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
                 vh.ptr)[((size_t)h * PADDED + FIRST + r) * HEAD_DIM + i] ==
            bf16_bits(vs[(r * KEY_HEADS + h) * HEAD_DIM + i]);
  for (int i = 0; i < FIRST * HEAD_DIM; i++)
    others_kept &= ((const uint16_t *)qh.ptr)[i] == 0xFFFF;
  check(worst <= 0x1p-8 && values_copied && others_kept, what);
  Buf *all[] = {&q, &k, &v, &norms, &inv_freq, &qh, &kh, &vh};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

// A sequence's layout: its rows, its state's rows and its questions' ends;
// window is the sliding window, or 0 for the question layout.
typedef struct {
  int rows, state, questions, window;
  int ends[MOST_QUESTIONS];
} Layout;

// Whether query row q sees key row k, as d1's masks say.
static int sees(const Layout *l, const int *group, int q, int k) {
  if (l->window) return k <= q && k + l->window > q;
  return group[q] == 0 || group[k] == 0 || group[q] == group[k];
}

// Runs lfm3_attend on the block of `block_rows` rows from row first, and
// checks every output against the CPU's attention.
static void check_attention(const Layout *l, int first, int block_rows,
                            const char *what) {
  int n = l->rows, padded = round_up(n, TILE_ROWS) + TILE_ROWS;
  Buf qh = gpu_alloc(gpu, slot, (size_t)QUERY_HEADS * padded * HEAD_DIM * 2),
      kh = gpu_alloc(gpu, slot, (size_t)KEY_HEADS * padded * HEAD_DIM * 2),
      vh = gpu_alloc(gpu, slot, (size_t)KEY_HEADS * padded * HEAD_DIM * 2),
      gates = gpu_alloc(gpu, slot, (size_t)block_rows * QUERY_HEADS * 4),
      groups = gpu_alloc(gpu, slot, (size_t)n * 4),
      spans = gpu_alloc(gpu, slot, (MOST_QUESTIONS + 1) * 8),
      out =
          gpu_alloc(gpu, slot, (size_t)block_rows * QUERY_HEADS * HEAD_DIM * 4);
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
  float *gate = (float *)gates.ptr;
  for (int i = 0; i < block_rows * QUERY_HEADS; i++)
    gate[i] = 3 * random_value();
  uint32_t *group = (uint32_t *)groups.ptr, *span = (uint32_t *)spans.ptr;
  int group_of[MOST_ROWS] = {0};
  span[0] = 0, span[1] = (uint32_t)l->state;
  for (int g = 0, start = l->state; g < l->questions; g++) {
    span[2 * g + 2] = (uint32_t)start, span[2 * g + 3] = (uint32_t)l->ends[g];
    for (int r = start; r < l->ends[g]; r++) group_of[r] = g + 1;
    start = l->ends[g];
  }
  for (int r = 0; r < n; r++) group[r] = (uint32_t)group_of[r];
  AttendParams p = {(uint32_t)n,      (uint32_t)first,    (uint32_t)block_rows,
                    (uint32_t)padded, (uint32_t)l->state, (uint32_t)l->window};
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, gpu_find_kernel(gpu, "lfm3_attend"));
  gpu_bind_buffer(&c, 0, &qh, 0), gpu_bind_buffer(&c, 1, &kh, 0);
  gpu_bind_buffer(&c, 2, &vh, 0), gpu_bind_buffer(&c, 3, &gates, 0);
  gpu_bind_buffer(&c, 4, &groups, 0), gpu_bind_buffer(&c, 5, &spans, 0);
  gpu_bind_buffer(&c, 6, &out, 0), gpu_copy_bytes(&c, 7, &p, sizeof p);
  gpu_run_kernel(
      &c, ((uint32_t)(block_rows + TILE_ROWS - 1) / TILE_ROWS) * QUERY_HEADS, 1,
      1, THREADS, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  // The probabilities' rounding to BF16 changes an output by about 2^-9 of
  // the values' magnitude, at most 1.
  double worst = 0, weights[MOST_ROWS];
  const float *o = (const float *)out.ptr;
  for (int r = 0; r < block_rows; r++)
    for (int h = 0; h < QUERY_HEADS; h++) {
      int row = first + r, kv = h / SHARING;
      double most = -INFINITY, sum = 0;
      for (int key = 0; key < n; key++) {
        weights[key] = -INFINITY;
        if (!sees(l, group_of, row, key)) continue;
        double score = 0;
        for (int i = 0; i < HEAD_DIM; i++)
          score +=
              (double)bf16_value(q[((size_t)h * padded + row) * HEAD_DIM + i]) *
              bf16_value(k[((size_t)kv * padded + key) * HEAD_DIM + i]);
        weights[key] = score / sqrt(HEAD_DIM);
        most = fmax(most, weights[key]);
      }
      for (int key = 0; key < n; key++)
        sum += weights[key] =
            weights[key] == -INFINITY ? 0 : exp(weights[key] - most);
      double opened = 1 / (1 + exp(-gate[r * QUERY_HEADS + h]));
      for (int i = 0; i < HEAD_DIM; i++) {
        double value = 0;
        for (int key = 0; key < n; key++)
          if (weights[key])
            value += weights[key] *
                     bf16_value(v[((size_t)kv * padded + key) * HEAD_DIM + i]);
        double want = value / sum * opened;
        worst =
            fmax(worst,
                 fabs(o[((size_t)r * QUERY_HEADS + h) * HEAD_DIM + i] - want));
      }
    }
  check(worst < 4e-3, what);
  Buf *all[] = {&qh, &kh, &vh, &gates, &groups, &spans, &out};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
}

int main(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  slot = memory_shared(memory);
  size_t length;
  const char *source = kernel_library_source(&length);
  gpu = gpu_new(source, length);
  test_heads(PAIRS / 2, "the heads of a block after the sequence's start are "
                        "normed and half of their pairs turned, as on a "
                        "global layer; values are copied as BF16");
  test_heads(PAIRS, "all pairs turn, as on a sliding layer");
  Layout questions = {200, 77, 3, 0, {90, 150, 200}};
  check_attention(&questions, 0, 200,
                  "a state and three questions: the state sees every row, a "
                  "question sees the state and itself");
  check_attention(&questions, 100, 100,
                  "the same layout, for a block that starts inside a question");
  Layout lone = {80, 50, 1, 0, {80}};
  check_attention(&lone, 0, 80, "a lone question: every row sees every row");
  Layout sliding = {700, 0, 1, 256, {700}};
  check_attention(&sliding, 0, 700,
                  "a sliding window: a row sees itself and the 255 rows "
                  "before it");
  check_attention(&sliding, 300, 400,
                  "a sliding window, for a block after the sequence's start");
  return failures != 0;
}
