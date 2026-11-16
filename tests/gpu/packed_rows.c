#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests the index keys, packed and as BF16, against a reference kernel that
// rounds the same values and keeps them as BF16: every value must be its
// reference value, bit for bit. The norm weights take three sizes.

#define METAL(...) #__VA_ARGS__
static const char reference[] = METAL(kernel void index_key_reference(
    device const float *k [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float2 *cs [[buffer(2)]], device bfloat *keys [[buffer(3)]],
    constant Groups &g [[buffer(4)]], uint grp [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint ratio = g.ratio, pos = g.start + grp * ratio;
  float v = k[grp * 128 + t];
  float r = rsqrt(tg_sum(v * v, tmp, t, 128) / 128 + 1e-20f);
  v = rope1(bf(w[t] * (v * r)), t, 128, cs + pos * 32, false);
  keys[(pos / ratio) * 128 + t] = bfloat(fp4_act(v));
});
#undef METAL

typedef struct {
  uint32_t ratio, start, row0, packed;
} Groups;

// The groups of a run, and a packed index key: 128 E2M1 codes, two to a byte,
// then an E8M0 scale for every 32 of them.
static constexpr int GROUPS = 64, KEY_BYTES = 68, KEY_SCALES = 64;

static uint32_t seed = 88172645u;

static uint32_t next(void) { return seed = seed * 1664525u + 1013904223u; }

// A random value from -1 to 1.
static float uniform(void) { return (float)(next() >> 8) / (1 << 23) - 1; }

// The value of an E2M1 code: codes 8 and up are the negatives of 0 to 7.
static float e2m1(unsigned code) {
  static const float values[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  return code & 8 ? -values[code & 7] : values[code & 7];
}

static uint16_t bf16_bits(float v) {  // the value must be exact in BF16
  uint32_t u;
  memcpy(&u, &v, 4);
  return (uint16_t)(u >> 16);
}

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
  int index_key = gpu_find_kernel(gpu, "index_key"),
      index_key_reference = gpu_find_kernel(gpu, "index_key_reference");
  // A row of positions per group, for ratio 2.
  int positions = 2 * GROUPS + 2;
  Buf cs = gpu_alloc(gpu, slot, (size_t)positions * 32 * 8),
      k = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 4),
      kw = gpu_alloc(gpu, slot, 128 * 4),
      keys = gpu_alloc(gpu, slot, (size_t)GROUPS * KEY_BYTES),
      keys_bf16 = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 2),
      kernel_keys = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 2);
  for (int p = 0; p < positions; p++)
    for (int i = 0; i < 32; i++) {
      float a = 6.2831853f * (float)(next() >> 8) / (1 << 24);
      ((float *)cs.ptr)[(p * 32 + i) * 2] = cosf(a),
                                       ((float *)cs.ptr)[(p * 32 + i) * 2 + 1] =
                                           sinf(a);
    }
  // The first group's values are zero; the others' magnitudes vary.
  for (int g = 0; g < GROUPS; g++)
    for (int i = 0; i < 128; i++)
      ((float *)k.ptr)[g * 128 + i] =
          g == 0 ? 0 : ldexpf(uniform(), (int)(next() % 12) - 6);
  const int weight_exponents[] = {0, -12, 3};
  for (size_t c = 0; c < sizeof weight_exponents / sizeof *weight_exponents;
       c++)
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
      for (int i = 0; i < 128; i++)
        ((float *)kw.ptr)[i] = ldexpf(0.5f + (float)(next() >> 8) / (1 << 24),
                                      weight_exponents[c]);
      memset(keys.ptr, 0, keys.size),
          memset(kernel_keys.ptr, 0, kernel_keys.size);
      GpuCommandBuffer cmd = gpu_begin(gpu);
      // Pass 0: the kernel, packed. Pass 1: the reference. Pass 2: the
      // kernel, as BF16.
      for (int pass = 0; pass < 3; pass++) {
        Groups g = {ratio, 0, 0, pass == 0};
        const Buf *keys_out = pass == 0   ? &keys
                              : pass == 1 ? &keys_bf16
                                          : &kernel_keys;
        gpu_select_kernel(&cmd, pass == 1 ? index_key_reference : index_key);
        gpu_bind_buffer(&cmd, 0, &k, 0), gpu_bind_buffer(&cmd, 1, &kw, 0),
            gpu_bind_buffer(&cmd, 2, &cs, 0),
            gpu_bind_buffer(&cmd, 3, keys_out, 0),
            gpu_copy_bytes(&cmd, 4, &g, sizeof g);
        gpu_run_kernel(&cmd, GROUPS, 1, 1, 128, 1, 1);
      }
      gpu_commit(&cmd);
      gpu_sync(gpu);
      int keys_same = 1;
      for (int r = 0; r < GROUPS; r++) {
        const uint8_t *key = (const uint8_t *)keys.ptr + r * KEY_BYTES;
        const uint16_t *want_key = (const uint16_t *)keys_bf16.ptr + r * 128;
        for (int i = 0; i < 128; i++)
          keys_same &=
              bf16_bits(ldexpf(e2m1(key[i / 2] >> (i % 2 * 4) & 15),
                               key[KEY_SCALES + i / 32] - 127)) == want_key[i];
      }
      char what[160];
      snprintf(what, sizeof what,
               "ratio %u, norm weights near 2^%d: the packed index keys hold "
               "the reference keys' values",
               ratio, weight_exponents[c]);
      check(keys_same, what);
      snprintf(what, sizeof what,
               "ratio %u, norm weights near 2^%d: the BF16 index keys are the "
               "reference's, bit for bit",
               ratio, weight_exponents[c]);
      check(!memcmp(kernel_keys.ptr, keys_bf16.ptr, kernel_keys.size), what);
    }
  return failures != 0;
}
