#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests the compressor's rows and the index keys, packed and as BF16, against
// reference kernels that round the same values and keep them as BF16: every
// value must be its reference value, bit for bit. The norm weights take three
// sizes, so that the scales also take E4M3's smallest values.

#define METAL(...) #__VA_ARGS__
static const char reference[] = METAL(
    kernel void compress_reference(device const float *kv [[buffer(0)]],
                                   device const float *score [[buffer(1)]],
                                   device const float *w [[buffer(2)]],
                                   device const float2 *cs [[buffer(3)]],
                                   device float *latent [[buffer(4)]],
                                   device bfloat *rows [[buffer(5)]],
                                   constant Groups &g [[buffer(6)]],
                                   uint grp [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
      threadgroup float tmp[32];
      uint2 p = uint2(g.ratio, g.start + grp * g.ratio);
      kv += (g.row0 + grp * g.ratio) * 512,
          score += (g.row0 + grp * g.ratio) * 512, latent += grp * 512;
      float v = kv[t];
      if (p.x == 2) {
        float a = score[t], b = score[512 + t], m = max(a, b);
        float ea = exp(a - m), eb = exp(b - m);
        v = bf((kv[t] * ea + kv[512 + t] * eb) / (ea + eb));
      }
      float r = rsqrt(tg_sum(v * v, tmp, t, 512) / 512 + 1e-20f);
      v = bf(w[t] * (v * r));
      latent[t] = v;
      v = rope1(v, t, 512, cs + p.y * 32, false);
      float amax = fabs(v);
      for (uint o = 1; o < 16; o <<= 1)
        amax = max(amax, simd_shuffle_xor(amax, o));
      float s = e4m3(max(amax, 6.0f * 0x1p-9f) / 6.0f);
      rows[(p.y / p.x) * 512 + t] = bfloat(e2m1(clamp(v / s, -6.0f, 6.0f)) * s);
    }

    kernel void index_key_reference(device const float *k [[buffer(0)]],
                                    device const float *w [[buffer(1)]],
                                    device const float2 *cs [[buffer(2)]],
                                    device bfloat *keys [[buffer(3)]],
                                    constant Groups &g [[buffer(4)]],
                                    uint grp [[threadgroup_position_in_grid]],
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

// The groups of a run, and a packed compressed row and index key: E2M1 codes,
// two to a byte, then an E4M3 scale for every 16 codes of a row and an E8M0
// scale for every 32 codes of a key.
static constexpr int GROUPS = 64, ROW_BYTES = 288, ROW_SCALES = 256,
                     KEY_BYTES = 68, KEY_SCALES = 64;

static uint32_t seed = 88172645u;

static uint32_t next(void) { return seed = seed * 1664525u + 1013904223u; }

// A random value from -1 to 1.
static float uniform(void) { return (float)(next() >> 8) / (1 << 23) - 1; }

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
  int compress = gpu_find_kernel(gpu, "compress"),
      compress_reference = gpu_find_kernel(gpu, "compress_reference"),
      index_key = gpu_find_kernel(gpu, "index_key"),
      index_key_reference = gpu_find_kernel(gpu, "index_key_reference");
  // Two rows per group for ratio 2, and a row of positions per group.
  int rows = 2 * GROUPS + 2, positions = 2 * GROUPS + 2;
  Buf kv = gpu_alloc(gpu, slot, (size_t)rows * 512 * 4),
      score = gpu_alloc(gpu, slot, (size_t)rows * 512 * 4),
      w = gpu_alloc(gpu, slot, 512 * 4),
      cs = gpu_alloc(gpu, slot, (size_t)positions * 32 * 8),
      latent = gpu_alloc(gpu, slot, (size_t)GROUPS * 512 * 4),
      latent_reference = gpu_alloc(gpu, slot, (size_t)GROUPS * 512 * 4),
      packed = gpu_alloc(gpu, slot, (size_t)GROUPS * ROW_BYTES),
      bf16 = gpu_alloc(gpu, slot, (size_t)GROUPS * 512 * 2),
      k = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 4),
      kw = gpu_alloc(gpu, slot, 128 * 4),
      keys = gpu_alloc(gpu, slot, (size_t)GROUPS * KEY_BYTES),
      keys_bf16 = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 2),
      kernel_rows = gpu_alloc(gpu, slot, (size_t)GROUPS * 512 * 2),
      kernel_keys = gpu_alloc(gpu, slot, (size_t)GROUPS * 128 * 2);
  for (int p = 0; p < positions; p++)
    for (int i = 0; i < 32; i++) {
      float a = 6.2831853f * (float)(next() >> 8) / (1 << 24);
      ((float *)cs.ptr)[(p * 32 + i) * 2] = cosf(a),
                                       ((float *)cs.ptr)[(p * 32 + i) * 2 + 1] =
                                           sinf(a);
    }
  // The first group's values are zero; the others' magnitudes vary.
  for (int r = 0; r < rows; r++)
    for (int i = 0; i < 512; i++) {
      float scale = r < 2 ? 0 : ldexpf(1, (int)(next() % 12) - 6);
      ((float *)kv.ptr)[r * 512 + i] = scale * uniform(),
                                  ((float *)score.ptr)[r * 512 + i] = uniform();
    }
  for (int g = 0; g < GROUPS; g++)
    for (int i = 0; i < 128; i++)
      ((float *)k.ptr)[g * 128 + i] =
          g == 0 ? 0 : ldexpf(uniform(), (int)(next() % 12) - 6);
  const int weight_exponents[] = {0, -12, 3};
  for (size_t c = 0; c < sizeof weight_exponents / sizeof *weight_exponents;
       c++)
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
      for (int i = 0; i < 512; i++)
        ((float *)w.ptr)[i] = ldexpf(0.5f + (float)(next() >> 8) / (1 << 24),
                                     weight_exponents[c]);
      for (int i = 0; i < 128; i++)
        ((float *)kw.ptr)[i] = ldexpf(0.5f + (float)(next() >> 8) / (1 << 24),
                                      weight_exponents[c]);
      memset(packed.ptr, 0, packed.size), memset(keys.ptr, 0, keys.size);
      memset(kernel_rows.ptr, 0, kernel_rows.size),
          memset(kernel_keys.ptr, 0, kernel_keys.size);
      GpuCommandBuffer cmd = gpu_begin(gpu);
      // Pass 0: the kernels, packed. Pass 1: the references. Pass 2: the
      // kernels, as BF16.
      for (int pass = 0; pass < 3; pass++) {
        Groups g = {ratio, 0, 0, pass == 0};
        const Buf *rows_out = pass == 0   ? &packed
                              : pass == 1 ? &bf16
                                          : &kernel_rows,
                  *keys_out = pass == 0   ? &keys
                              : pass == 1 ? &keys_bf16
                                          : &kernel_keys;
        gpu_select_kernel(&cmd, pass == 1 ? compress_reference : compress);
        gpu_bind_buffer(&cmd, 0, &kv, 0), gpu_bind_buffer(&cmd, 1, &score, 0),
            gpu_bind_buffer(&cmd, 2, &w, 0), gpu_bind_buffer(&cmd, 3, &cs, 0);
        gpu_bind_buffer(&cmd, 4, pass == 1 ? &latent_reference : &latent, 0),
            gpu_bind_buffer(&cmd, 5, rows_out, 0),
            gpu_copy_bytes(&cmd, 6, &g, sizeof g);
        gpu_run_kernel(&cmd, GROUPS, 1, 1, 512, 1, 1);
        gpu_select_kernel(&cmd, pass == 1 ? index_key_reference : index_key);
        gpu_bind_buffer(&cmd, 0, &k, 0), gpu_bind_buffer(&cmd, 1, &kw, 0),
            gpu_bind_buffer(&cmd, 2, &cs, 0),
            gpu_bind_buffer(&cmd, 3, keys_out, 0),
            gpu_copy_bytes(&cmd, 4, &g, sizeof g);
        gpu_run_kernel(&cmd, GROUPS, 1, 1, 128, 1, 1);
      }
      gpu_commit(&cmd);
      gpu_sync(gpu);
      int rows_same = 1, keys_same = 1, small_scales = 0, zeros = 1;
      for (int r = 0; r < GROUPS; r++) {
        const uint8_t *row = (const uint8_t *)packed.ptr + r * ROW_BYTES;
        const uint16_t *want = (const uint16_t *)bf16.ptr + r * 512;
        for (int i = 0; i < 512; i++) {
          float v = e2m1(row[i / 2] >> (i % 2 * 4) & 15) *
                    e4m3(row[ROW_SCALES + i / 16]);
          rows_same &= bf16_bits(v) == want[i];
          if (r == 0) zeros &= v == 0;
        }
        for (int s = 0; s < 512 / 16; s++)
          small_scales += (row[ROW_SCALES + s] >> 3 & 15) == 0;
        const uint8_t *key = (const uint8_t *)keys.ptr + r * KEY_BYTES;
        const uint16_t *want_key = (const uint16_t *)keys_bf16.ptr + r * 128;
        for (int i = 0; i < 128; i++)
          keys_same &=
              bf16_bits(ldexpf(e2m1(key[i / 2] >> (i % 2 * 4) & 15),
                               key[KEY_SCALES + i / 32] - 127)) == want_key[i];
      }
      int latents_same =
          !memcmp(latent.ptr, latent_reference.ptr, (size_t)GROUPS * 512 * 4);
      char what[160];
      snprintf(what, sizeof what,
               "ratio %u, norm weights near 2^%d: the packed rows hold the "
               "reference rows' values (%d of their scales subnormal)",
               ratio, weight_exponents[c], small_scales);
      check(rows_same && latents_same && zeros, what);
      snprintf(what, sizeof what,
               "ratio %u, norm weights near 2^%d: the packed index keys hold "
               "the reference keys' values",
               ratio, weight_exponents[c]);
      check(keys_same, what);
      snprintf(what, sizeof what,
               "ratio %u, norm weights near 2^%d: the BF16 rows and index keys "
               "are the reference's, bit for bit",
               ratio, weight_exponents[c]);
      check(!memcmp(kernel_rows.ptr, bf16.ptr, kernel_rows.size) &&
                !memcmp(kernel_keys.ptr, keys_bf16.ptr, kernel_keys.size),
            what);
    }
  return failures != 0;
}
