#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests the prompt attention kernel against a reference kernel that stages a
// slice of a chunk's keys at a time, for its scores and again for its values:
// every output must be the same bits. The kernel reads compressed rows as BF16
// or packed, and the reference the same values as BF16. Also times them.

#define METAL(...) #__VA_ARGS__
static const char reference[] = METAL(kernel void attend_m_reference(
    device const bfloat *qb [[buffer(0)]],
    device const bfloat *ring [[buffer(1)]],
    device const uchar *comp [[buffer(2)]], device const int *sel [[buffer(3)]],
    device const float *sink [[buffer(4)]], device float *o [[buffer(5)]],
    constant Att &at [[buffer(6)]], uint2 gr [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat kv[32 * 128], ps[32 * 32];
  threadgroup float ss[32 * 32], mx[32], sum[32], rescale[32];
  threadgroup uint rows[640];
  uint h0 = gr.x * 32, pos = at.win.pos + gr.y;
  o += gr.y * 32768, sel += gr.y * 513 + 1;
  uint lo = max(pos >= 127 ? pos - 127 : 0u, at.first);
  uint nwin = pos + 1 - lo, ncomp = at.comp_rows ? uint(sel[-1]) : 0,
       n = nwin + ncomp;
  for (uint r = t; r < n; r += 256)
    rows[r] =
        r < nwin ? win_row(lo + r, at.win) : 0x80000000u | uint(sel[r - nwin]);
  if (t < 32) mx[t] = -1e30f, sum[t] = 0;
  constexpr auto sdesc =
      matmul2d_descriptor(32, 32, 128, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto odesc =
      matmul2d_descriptor(32, 128, 32, false, false, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<8>> sop;
  matmul2d<odesc, execution_simdgroups<8>> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq(
      (device bfloat *)qb + gr.y * 32768 + h0 * 512,
      dextents<int32_t, 2>(512, 32));
  tensor<threadgroup bfloat, extents<int32_t, 128, 32>, tensor_inline> tk(
      kv, extents<int32_t, 128, 32>());
  tensor<threadgroup bfloat, extents<int32_t, 32, 32>, tensor_inline> tp(
      ps, extents<int32_t, 32, 32>());
  auto acc0 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk),
                                                     float>();
  auto acc1 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk),
                                                     float>();
  auto acc2 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk),
                                                     float>();
  auto acc3 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk),
                                                     float>();
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) acc0[i] = acc1[i] = acc2[i] = acc3[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Keys kc = keys_load(ring, comp, 0, rows, 0, n, 0, t);
  for (uint r0 = 0; r0 < n; r0 += 32) {
    auto sc = sop.get_destination_cooperative_tensor<decltype(tq.slice(0, 0)),
                                                     decltype(tk), float>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = 0;
    for (uint ds = 0; ds < 4; ds++) {
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      kc = keys_load(ring, comp, 0, rows, r0, n, (ds + 1) % 4, t);
      auto qd = tq.slice(ds * 128, 0);
      sop.run(qd, tk, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto a = sc.get_multidimensional_index(i);
        ss[a[1] * 32 + a[0]] =
            r0 + a[0] < n ? sc[i] * 0.044194173824159216f : -INFINITY;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {
      float m = mx[t];
      for (uint j = 0; j < 32; j++) m = max(m, ss[t * 32 + j]);
      float f = exp(mx[t] - m), s2 = sum[t] * f;
      for (uint j = 0; j < 32; j++) {
        float e = exp(ss[t * 32 + j] - m);
        s2 += e;
        ps[t * 32 + j] = bfloat(e);
      }
      mx[t] = m, sum[t] = s2, rescale[t] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); i++)
      if (acc0.is_valid_element(i)) {
        float f = rescale[acc0.get_multidimensional_index(i)[1]];
        acc0[i] *= f, acc1[i] *= f, acc2[i] *= f, acc3[i] *= f;
      }
    for (uint ds = 0; ds < 4; ds++) {
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ds < 3)
        kc = keys_load(ring, comp, 0, rows, r0, n, ds + 1, t);
      else if (r0 + 32 < n)
        kc = keys_load(ring, comp, 0, rows, r0 + 32, n, 0, t);
      if (ds == 0)
        oop.run(tp, tk, acc0);
      else if (ds == 1)
        oop.run(tp, tk, acc1);
      else if (ds == 2)
        oop.run(tp, tk, acc2);
      else
        oop.run(tp, tk, acc3);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) {
      auto a = acc0.get_multidimensional_index(i);
      uint h = a[1], d = a[0];
      float den = sum[h] + exp(sink[h0 + h] - mx[h]);
      device float *out = o + (h0 + h) * 512 + d;
      out[0] = bf(acc0[i] / den), out[128] = bf(acc1[i] / den),
      out[256] = bf(acc2[i] / den), out[384] = bf(acc3[i] / den);
    }
});
#undef METAL

typedef struct {
  uint32_t pos, base, mod, first, comp_rows, packed;
  uint64_t slice_off;
  uint32_t page_shift, pad;
} Att;

// Query rows, and compressed key rows.
static constexpr int ROWS = 512, COMP = 8192;
// A packed compressed row: 512 E2M1 codes, two to a byte, then an E4M3 scale
// for every 16 of them.
static constexpr int ROW_BYTES = 288, SCALES = 256;
// The forms of the compressed rows that the kernel reads.
static const char *const FORMS[2] = {"BF16", "packed"};

static Buf QB, WIN, KEYS, PACKED, SEL, SINK, OUT[2];
static uint32_t seed = 2463534242u;

static uint32_t next(void) { return seed = seed * 1664525u + 1013904223u; }

static uint16_t bf16(float v) {  // values of few bits are exact
  uint32_t u;
  memcpy(&u, &v, 4);
  return (uint16_t)(u >> 16);
}

// The value of an E2M1 code: codes 8 and up are the negatives of 0 to 7.
static float e2m1(unsigned code) {
  static const float values[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  return code & 8 ? -values[code & 7] : values[code & 7];
}

// The value of an E4M3 code that is not negative.
static float e4m3(unsigned code) {
  unsigned e = code >> 3 & 15, m = code & 7;
  return e ? ldexpf(1 + m / 8.0f, (int)e - 7) : ldexpf((float)m, -9);
}

// Runs kernel over `rows` query rows with the compressed rows `comp`;
// returns the GPU's microseconds.
static double run(Gpu *gpu, int kernel, const Buf *comp, Att at, int rows,
                  const Buf *out, int repeat) {
  double before = gpu_times(gpu).busy_seconds;
  GpuCommandBuffer cmd = gpu_begin(gpu);
  gpu_select_kernel(&cmd, kernel);
  gpu_bind_buffer(&cmd, 0, &QB, 0), gpu_bind_buffer(&cmd, 1, &WIN, 0),
      gpu_bind_buffer(&cmd, 2, comp, 0), gpu_bind_buffer(&cmd, 3, &SEL, 0),
      gpu_bind_buffer(&cmd, 4, &SINK, 0);
  gpu_bind_buffer(&cmd, 5, out, 0), gpu_copy_bytes(&cmd, 6, &at, sizeof at);
  for (int i = 0; i < repeat; i++) gpu_run_kernel(&cmd, 2, rows, 1, 256, 1, 1);
  gpu_commit(&cmd);
  gpu_sync(gpu);
  return (gpu_times(gpu).busy_seconds - before) * 1e6 / repeat;
}

// Each row's selected compressed rows: `count(r)` of them, ascending.
static void select_rows(int rows, int (*count)(int)) {
  int *sel = (int *)SEL.ptr;
  for (int r = 0; r < rows; r++) {
    int n = count(r), at = (int)(next() % 7);
    sel[r * 513] = n;
    for (int i = 0; i < n; i++)
      sel[r * 513 + 1 + i] = at, at += 1 + (int)(next() % 15);
  }
}

static int all(int r) { return (void)r, 512; }

static int none(int r) { return (void)r, 0; }

static int varied(int r) { return (r * 37) % 513; }

static int few(int r) { return 100 + r % 60; }

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  size_t source_bytes;
  const char *source = kernel_library_source(&source_bytes);
  char *code = xmalloc(source_bytes + sizeof reference + 1);
  memcpy(code, source, source_bytes), code[source_bytes] = '\n';
  memcpy(code + source_bytes + 1, reference, sizeof reference);
  Gpu *gpu = gpu_new(code, source_bytes + sizeof reference);
  free(code);
  int before = gpu_find_kernel(gpu, "attend_m_reference"),
      now_kernel = gpu_find_kernel(gpu, "attend_m");
  QB = gpu_alloc(gpu, slot, (size_t)ROWS * 32768 * 2),
  WIN = gpu_alloc(gpu, slot, (size_t)(ROWS + 128) * 512 * 2);
  KEYS = gpu_alloc(gpu, slot, (size_t)COMP * 512 * 2),
  PACKED = gpu_alloc(gpu, slot, (size_t)COMP * ROW_BYTES),
  SEL = gpu_alloc(gpu, slot, (size_t)ROWS * 513 * 4),
  SINK = gpu_alloc(gpu, slot, 64 * 4);
  for (int i = 0; i < 2; i++)
    OUT[i] = gpu_alloc(gpu, slot, (size_t)ROWS * 32768 * 4);
  for (size_t i = 0; i < (size_t)ROWS * 32768; i++)
    ((uint16_t *)QB.ptr)[i] = bf16((float)((int)(next() >> 27) - 16) / 16);
  for (size_t i = 0; i < (size_t)(ROWS + 128) * 512; i++)
    ((uint16_t *)WIN.ptr)[i] = bf16((float)((int)(next() >> 27) - 16) / 16);
  // Packed rows of random codes and E4M3 scales between 1/8 and 8, and their
  // values as BF16, which holds each exactly.
  for (size_t r = 0; r < COMP; r++) {
    uint8_t *row = (uint8_t *)PACKED.ptr + r * ROW_BYTES;
    for (int b = 0; b < SCALES; b++) row[b] = (uint8_t)(next() >> 24);
    for (int s = 0; s < 512 / 16; s++)
      row[SCALES + s] = (uint8_t)((4 + next() % 6) << 3 | next() % 8);
    for (int i = 0; i < 512; i++)
      ((uint16_t *)KEYS.ptr)[r * 512 + i] = bf16(
          e2m1(row[i / 2] >> (i % 2 * 4) & 15) * e4m3(row[SCALES + i / 16]));
  }
  for (int i = 0; i < 64; i++)
    ((float *)SINK.ptr)[i] = (float)(next() >> 8) / (1 << 22) - 2;

  struct {
    const char *name;
    uint32_t pos, first, comp;
    int (*count)(int);
    int rows;
  } cases[] = {
      {"a full window and 512 selected rows", 5000, 0, 1, all, 96},
      {"the first positions, no compressed rows", 0, 0, 0, none, 200},
      {"0 to 512 selected rows", 300, 0, 1, varied, 96},
      {"a window cut short, 100 to 159 selected rows", 5000, 4990, 1, few, 96},
  };

  for (uint32_t form = 0; form < 2; form++)
    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
      uint32_t pos = cases[c].pos, base = pos >= 127 ? pos - 127 : 0;
      Att at = {pos, base, 1u << 30, cases[c].first, cases[c].comp, form,
                0,   0,    0};
      select_rows(cases[c].rows, cases[c].count);
      memset(OUT[0].ptr, 0xa5, OUT[0].size),
          memset(OUT[1].ptr, 0xa5, OUT[1].size);
      run(gpu, before, &KEYS, at, cases[c].rows, &OUT[0], 1),
          run(gpu, now_kernel, form ? &PACKED : &KEYS, at, cases[c].rows,
              &OUT[1], 1);
      const float *a = (const float *)OUT[0].ptr,
                  *b = (const float *)OUT[1].ptr;
      size_t count = (size_t)cases[c].rows * 32768, nonzero = 0, finite = 1,
             same = 1;
      for (size_t i = 0; i < count && same && finite; i++) {
        if (memcmp(a + i, b + i, 4))
          same = 0, printf("row %zu, head %zu, column %zu: %g against %g\n",
                           i / 32768, i % 32768 / 512, i % 512, b[i], a[i]);
        finite = isfinite(a[i]);
        nonzero += a[i] != 0;
      }
      char what[160];
      snprintf(what, sizeof what,
               "%s rows, %s: %zu outputs are equal, finite, and mostly nonzero",
               FORMS[form], cases[c].name, count);
      check(same && finite && nonzero >= count / 2, what);
    }
  // Paged attend_m and paged decode (attend_scores/values/combine) parity
  // across 32 pages of 256 rows, plus paged attend_m timing.
  static constexpr int PAGE_ROWS = 256, PAGE_SHIFT = 8,
                       NPAGES = COMP / PAGE_ROWS;
  static constexpr size_t BF16_OFF = 512,
                          PACKED_OFF = BF16_OFF + PAGE_ROWS * 512 * 2,
                          PAGE_BYTES = PACKED_OFF + PAGE_ROWS * ROW_BYTES;
  Buf pages[NPAGES],
      page_addrs = gpu_alloc(gpu, slot, (size_t)NPAGES * sizeof(uint64_t));
  for (int p = 0; p < NPAGES; p++) {
    pages[p] = gpu_alloc(gpu, slot, PAGE_BYTES);
    gpu_set_resident(&pages[p], 1);
    memcpy(pages[p].ptr + BF16_OFF,
           (const uint8_t *)KEYS.ptr + (size_t)p * PAGE_ROWS * 512 * 2,
           (size_t)PAGE_ROWS * 512 * 2);
    memcpy(pages[p].ptr + PACKED_OFF,
           (const uint8_t *)PACKED.ptr + (size_t)p * PAGE_ROWS * ROW_BYTES,
           (size_t)PAGE_ROWS * ROW_BYTES);
    ((uint64_t *)page_addrs.ptr)[p] = pages[p].address + pages[p].off;
  }
  gpu_commit_residency(gpu);
  int k_scores = gpu_find_kernel(gpu, "attend_scores"),
      k_values = gpu_find_kernel(gpu, "attend_values"),
      k_combine = gpu_find_kernel(gpu, "attend_combine");
  Buf q_f32 = gpu_alloc(gpu, slot, 32768 * 4),
      cs_buf = gpu_alloc(gpu, slot, 6000 * 32 * 8),
      asc = gpu_alloc(gpu, slot, 64 * 640 * 4),
      amax = gpu_alloc(gpu, slot, 10 * 64 * 4),
      apart = gpu_alloc(gpu, slot, 10 * 64 * 514 * 4);
  for (size_t i = 0; i < 32768; i++)
    ((float *)q_f32.ptr)[i] = (float)((int)(next() >> 27) - 16) / 16.0f;
  for (size_t p = 0; p < 6000; p++)
    for (int i = 0; i < 32; i++) {
      float a = (float)p * powf(10000.0f, -(float)i / 32.0f);
      ((float *)cs_buf.ptr)[(p * 32 + i) * 2] = cosf(a);
      ((float *)cs_buf.ptr)[(p * 32 + i) * 2 + 1] = sinf(a);
    }
  select_rows(96, all);
  for (uint32_t form = 0; form < 2; form++) {
    uint32_t pos = 5000, base = pos - 127;
    Att flat = {pos, base, 1u << 30, 0, 1, form, 0, 0, 0};
    Att paged = {
        pos,        base, 1u << 30, 0, 1, form, form ? PACKED_OFF : BF16_OFF,
        PAGE_SHIFT, 0};
    memset(OUT[0].ptr, 0xa5, OUT[0].size),
        memset(OUT[1].ptr, 0xa5, OUT[1].size);
    run(gpu, now_kernel, form ? &PACKED : &KEYS, flat, 96, &OUT[0], 1);
    run(gpu, now_kernel, &page_addrs, paged, 96, &OUT[1], 1);
    char what[160];
    snprintf(what, sizeof what,
             "paged %s attend_m matches flat bit-for-bit across %d pages",
             FORMS[form], NPAGES);
    check(!memcmp(OUT[0].ptr, OUT[1].ptr, (size_t)96 * 32768 * 4), what);

    // Paged decode attention (attend_scores + attend_values + attend_combine).
    for (int mode = 0; mode < 2; mode++) {
      Att da = mode == 0 ? flat : paged;
      const Buf *cbuf = mode == 0 ? (form ? &PACKED : &KEYS) : &page_addrs;
      GpuCommandBuffer cmd = gpu_begin(gpu);
      gpu_select_kernel(&cmd, k_scores);
      gpu_bind_buffer(&cmd, 0, &q_f32, 0), gpu_bind_buffer(&cmd, 1, &WIN, 0),
          gpu_bind_buffer(&cmd, 2, cbuf, 0), gpu_bind_buffer(&cmd, 3, &SEL, 0),
          gpu_bind_buffer(&cmd, 4, &cs_buf, 0),
          gpu_bind_buffer(&cmd, 5, &asc, 0), gpu_bind_buffer(&cmd, 6, &amax, 0),
          gpu_copy_bytes(&cmd, 7, &da, sizeof da);
      gpu_run_kernel(&cmd, 16, 5, 1, 512, 1, 1);
      gpu_select_kernel(&cmd, k_values);
      gpu_bind_buffer(&cmd, 0, &WIN, 0), gpu_bind_buffer(&cmd, 1, cbuf, 0),
          gpu_bind_buffer(&cmd, 2, &SEL, 0), gpu_bind_buffer(&cmd, 3, &asc, 0),
          gpu_bind_buffer(&cmd, 4, &amax, 0),
          gpu_bind_buffer(&cmd, 5, &apart, 0),
          gpu_copy_bytes(&cmd, 6, &da, sizeof da);
      gpu_run_kernel(&cmd, 8, 5, 1, 512, 1, 1);
      gpu_select_kernel(&cmd, k_combine);
      gpu_bind_buffer(&cmd, 0, &apart, 0), gpu_bind_buffer(&cmd, 1, &amax, 0),
          gpu_bind_buffer(&cmd, 2, &SINK, 0),
          gpu_bind_buffer(&cmd, 3, &OUT[mode], 0),
          gpu_bind_buffer(&cmd, 4, &SEL, 0),
          gpu_copy_bytes(&cmd, 5, &da, sizeof da);
      gpu_run_kernel(&cmd, 64, 2, 1, 128, 1, 1);
      gpu_commit(&cmd);
      gpu_sync(gpu);
    }
    snprintf(what, sizeof what,
             "paged %s decode attention matches flat bit-for-bit across %d "
             "pages",
             FORMS[form], NPAGES);
    check(!memcmp(OUT[0].ptr, OUT[1].ptr, (size_t)32768 * 4), what);
  }

  Att at = {20000, 20000 - 127, 1u << 30, 0, 1, 0, 0, 0, 0}, packed = at;
  packed.packed = 1;
  Att paged_packed = packed;
  paged_packed.slice_off = PACKED_OFF, paged_packed.page_shift = PAGE_SHIFT;
  select_rows(ROWS, all);
  run(gpu, now_kernel, &KEYS, at, ROWS, &OUT[1], 1);  // raise the GPU's clock
  double first = run(gpu, before, &KEYS, at, ROWS, &OUT[0], 4),
         bf16_us = run(gpu, now_kernel, &KEYS, at, ROWS, &OUT[1], 4),
         packed_us = run(gpu, now_kernel, &PACKED, packed, ROWS, &OUT[1], 4),
         paged_us =
             run(gpu, now_kernel, &page_addrs, paged_packed, ROWS, &OUT[1], 4);
  double last = run(gpu, before, &KEYS, at, ROWS, &OUT[0], 4);
  printf("prompt attention of %d rows over 640 keys: a slice at a time %.0f / "
         "%.0f us, BF16 %.0f us, packed flat %.0f us, packed paged %.0f us\n",
         ROWS, first, last, bf16_us, packed_us, paged_us);
  return failures != 0;
}
