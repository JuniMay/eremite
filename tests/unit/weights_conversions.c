#include "check.h"
#include "base/error.h"
#include "engine/io.h"
#include "engine/weights.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the weight conversions and a group's life. The conversions: BF16 and
// FP32 vectors with an addition and an exponential, products, padded BF16
// rows, matrices of E4M3 alone, E4M3 with E8M0 scales and NVFP4 expanded to
// BF16, compressed scales, generated weights and aliases. The group's life:
// the bytes it needs and holds, its residency, a dependent group that is
// evicted first, removing a weight, and freeing. Buffers are plain memory, and
// the checkpoint is an array that the test's reads copy from. No Metal device
// is used.

static constexpr size_t FILE_BYTES = 1 << 16;

static uint8_t file[FILE_BYTES];

// The slots of the freed buffers, in the order they were freed, and how many
// residency changes the groups asked for.
static MemorySlot *freed[64];
static int freed_count, resident_calls;

static ssize_t fixture_pread(int fd, void *buffer, size_t bytes, off_t offset) {
  (void)fd;
  if ((size_t)offset >= FILE_BYTES) return 0;
  size_t n =
      (size_t)offset + bytes > FILE_BYTES ? FILE_BYTES - (size_t)offset : bytes;
  memcpy(buffer, file + offset, n);
  return (ssize_t)n;
}

static Buf fixture_new(Gpu *gpu, MemorySlot *slot, size_t size) {
  (void)gpu;
  Buf b = {.allocated = size, .size = size, .memory = slot};
  if (!memory_claim(b.memory, b.allocated, MEMORY_REQUIRED))
    die("the fixture's memory refused %zu bytes", size);
  b.ptr = calloc(1, size ? size : 1);
  b.obj = b.ptr;
  return b;
}

static void fixture_free(Buf *b) {
  if (b->obj) {
    if (freed_count < (int)(sizeof freed / sizeof *freed))
      freed[freed_count++] = b->memory;
    memory_release(b->memory, b->allocated);
    free(b->ptr);
  }
  *b = (Buf){0};
}

static size_t fixture_allocation_size(size_t size) { return size; }

static void fixture_resident(const Buf *b, int on) {
  (void)b, (void)on;
  resident_calls++;
}

static void fixture_commit(Gpu *gpu) { (void)gpu; }

static void fixture_read(int fd, uint64_t off, uint64_t len, Buf *dst) {
  (void)fd;
  memcpy(dst->ptr, file + off, len);
  dst->off = 0;
}

#define pread fixture_pread
#define gpu_alloc fixture_new
#define gpu_free fixture_free
#define gpu_round_up_to_pages fixture_allocation_size
#define gpu_set_resident fixture_resident
#define gpu_commit_residency fixture_commit
#define io_read fixture_read
#include "engine/weights.c"
#undef pread
#undef gpu_alloc
#undef gpu_free
#undef gpu_round_up_to_pages
#undef gpu_set_resident
#undef gpu_commit_residency
#undef io_read

// Returns a tensor of size bytes at off in the test's file, with rows rows.
static Tensor tensor_at(uint64_t off, uint64_t size, int dtype, int64_t rows) {
  return (Tensor){.fd = 3,
                  .off = off,
                  .size = size,
                  .dtype = dtype,
                  .shape = {rows},
                  .ndim = 1};
}

// Returns the BF16 bits of a value that BF16 holds exactly.
static uint16_t bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  return (uint16_t)(bits >> 16);
}

// Writes the BF16 bits of count values to the file at off.
static void put_bf16(uint64_t off, const float *values, int count) {
  for (int i = 0; i < count; i++) {
    uint16_t bits = bf16(values[i]);
    memcpy(file + off + 2 * i, &bits, 2);
  }
}

// Returns whether the buffer holds the BF16 bits of the count values.
static int holds_bf16(const Buf *b, const float *values, int count) {
  for (int i = 0; i < count; i++)
    if (((const uint16_t *)b->ptr)[i] != bf16(values[i])) return 0;
  return 1;
}

// Fills a generated weight with its parameter byte.
static void fill_with_byte(Buf *buffer, const void *parameters) {
  memset(buffer->ptr, *(const uint8_t *)parameters, buffer->size);
}

int main(void) {
  // The checkpoint's tensors.
  constexpr uint64_t BF16_VECTOR = 1000, F32_VECTOR = 2000, BF16_MATRIX = 3000,
                     E4M3 = 4000, E4M3_BLOCK = 5000, E8M0_SCALE = 5100,
                     NVFP4 = 6000, NVFP4_SCALE = 6100, CSF_FIXED = 7000,
                     CSF_EXCEPTIONS = 7100;
  const float bf16_values[] = {1, -2, 0.5f, 3},
              f32_values[] = {2, 4, -1, 0.25f}, matrix[] = {1, 2, 3, 4, 5, 6};
  put_bf16(BF16_VECTOR, bf16_values, 4);
  memcpy(file + F32_VECTOR, f32_values, sizeof f32_values);
  put_bf16(BF16_MATRIX, matrix, 6);
  // E4M3 codes of 1, 2, -1 and 0.75.
  const uint8_t e4m3_codes[] = {0x38, 0x40, 0xb8, 0x34};
  memcpy(file + E4M3, e4m3_codes, sizeof e4m3_codes);
  // 32 E4M3 codes of 1, and an E8M0 scale of 2.
  memset(file + E4M3_BLOCK, 0x38, 32);
  file[E8M0_SCALE] = 128;
  // NVFP4 codes of 1, -1, 6 and 0.5, then zeros, and an E4M3 scale of 2.
  const uint8_t nvfp4_codes[8] = {0xa2, 0x17};
  memcpy(file + NVFP4, nvfp4_codes, sizeof nvfp4_codes);
  file[NVFP4_SCALE] = 0x40;
  // 16 rows of 2 compressed scales: row bases, then a byte of two nibbles
  // per row, and one exception.
  constexpr int CSF_ROWS = 16, CSF_COLS = 2;
  uint8_t expected_scales[CSF_ROWS * CSF_COLS];
  for (int row = 0; row < CSF_ROWS; row++) {
    file[CSF_FIXED + row] = (uint8_t)(10 + row);
    file[CSF_FIXED + 16 + row] = (uint8_t)(row | (15 - row) << 4);
    expected_scales[row * CSF_COLS] = (uint8_t)(10 + row + row);
    expected_scales[row * CSF_COLS + 1] = (uint8_t)(10 + row + 15 - row);
  }
  const uint32_t exception = 200u << 24 | 5;
  memcpy(file + CSF_EXCEPTIONS, &exception, sizeof exception);
  expected_scales[5] = 200;

  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  MemoryAccount account;
  memory_account_init(memory, &account, "fixture model", 0);
  Gpu *gpu = (Gpu *)file;
  WeightGroup *set = weight_group_new(gpu, &account, "fixture weights");

  Tensor bf16_vector = tensor_at(BF16_VECTOR, 8, 'b', 4),
         f32_vector = tensor_at(F32_VECTOR, 16, 'f', 4),
         bf16_matrix = tensor_at(BF16_MATRIX, 12, 'b', 2),
         no_scales = tensor_at(0, 0, 'u', 0);
  Buf plain = {0}, added = {0}, exponential = {0}, product = {0}, padded = {0},
      e4m3 = {0}, scaled = {0}, nvfp4 = {0}, scales = {0}, generated = {0},
      alias = {0};
  weight_group_add_vector(set, &plain, bf16_vector, 0, 0);
  weight_group_add_vector(set, &added, bf16_vector, 1, WEIGHT_ADD);
  weight_group_add_vector(set, &exponential, f32_vector, 0.5f,
                          WEIGHT_ADD | WEIGHT_EXP);
  weight_group_add_product(set, &product, bf16_vector, f32_vector);
  weight_group_add_padded(set, &padded, bf16_matrix, 4);
  weight_group_add_expanded(set, &e4m3, tensor_at(E4M3, 4, '8', 1), no_scales,
                            'e', 4, 6);
  weight_group_add_expanded(set, &scaled, tensor_at(E4M3_BLOCK, 32, '8', 1),
                            tensor_at(E8M0_SCALE, 1, 'e', 1), 'm', 32, 32);
  weight_group_add_expanded(set, &nvfp4, tensor_at(NVFP4, 8, 'u', 1),
                            tensor_at(NVFP4_SCALE, 1, '8', 1), 'n', 16, 16);
  weight_group_add_compressed_scales(
      set, &scales, tensor_at(CSF_FIXED, 32, 'u', 1),
      tensor_at(CSF_EXCEPTIONS, 4, 'U', 1), CSF_ROWS, CSF_COLS);
  uint8_t byte = 0x5a;
  weight_group_add_generated(set, &generated, 100, fill_with_byte, &byte, 1);
  byte = 0;
  weight_group_add_alias(set, &alias, &plain);

  // The buffers' bytes: 5 vectors and rows of 16, 12, 64, 32, 32 and 100
  // bytes. The largest conversion reads 32 + 4 bytes of compressed scales.
  constexpr uint64_t BUFFERS = 5 * 16 + 12 + 64 + 32 + 32 + 100, SCRATCH = 36;
  check(weight_group_required_bytes(set) == BUFFERS + SCRATCH,
        "a group needs its buffers' bytes and room for the largest "
        "conversion's tensors");
  check(weight_group_resident_bytes(set) == 0,
        "a group holds no bytes before it loads");

  check(weight_group_load(set), "the group loads");
  check(resident_calls == 10,
        "the load makes every buffer resident, and no alias");
  check(weight_group_resident_bytes(set) == BUFFERS,
        "a loaded group holds its buffers' bytes");

  const float expect_added[] = {2, -1, 1.5f, 4},
              expect_exponential[] = {expf(2.5f), expf(4.5f), expf(-0.5f),
                                      expf(0.75f)},
              expect_product[] = {2, -8, -0.5f, 0.75f};
  check(!memcmp(plain.ptr, bf16_values, sizeof bf16_values) &&
            !memcmp(added.ptr, expect_added, sizeof expect_added),
        "a BF16 vector becomes FP32 values, with the addition");
  check(!memcmp(exponential.ptr, expect_exponential, sizeof expect_exponential),
        "an FP32 vector is added to and then exponentiated");
  check(!memcmp(product.ptr, expect_product, sizeof expect_product),
        "a product multiplies two vectors elementwise");
  const float expect_padded[] = {1, 2, 3, 0, 4, 5, 6, 0};
  check(holds_bf16(&padded, expect_padded, 8),
        "a BF16 matrix's rows are padded with zeros to the stride");
  const float expect_e4m3[] = {1, 2, -1, 0.75f, 0, 0};
  check(holds_bf16(&e4m3, expect_e4m3, 6),
        "E4M3 codes alone expand to their BF16 values");
  float expect_scaled[32];
  for (int i = 0; i < 32; i++) expect_scaled[i] = 2;
  check(holds_bf16(&scaled, expect_scaled, 32),
        "E4M3 codes with an E8M0 scale expand to their scaled values");
  const float expect_nvfp4[16] = {2, -2, 12, 1};
  check(holds_bf16(&nvfp4, expect_nvfp4, 16),
        "NVFP4 codes with an E4M3 scale expand to their scaled values");
  check(!memcmp(scales.ptr, expected_scales, sizeof expected_scales),
        "compressed scales decode to row bases plus nibbles, with the "
        "exception");
  int filled = 1;
  for (int i = 0; i < 100; i++) filled &= generated.ptr[i] == 0x5a;
  check(filled, "a generated weight is filled from a copy of its parameters");
  check(alias.ptr == plain.ptr && alias.size == plain.size,
        "an alias is a copy of its source buffer");

  resident_calls = 0;
  weight_group_set_resident(set, 0);
  check(resident_calls == 10,
        "a residency change goes to every buffer of the group, and no alias");

  fflush(stdout);
  pid_t child = fork();
  if (!child) weight_group_remove(set, &plain), _exit(0);
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "removing a weight from a loaded group ends the program");

  WeightGroup *dependent = weight_group_new_dependent(set, "fixture dependent");
  Buf extra = {0};
  weight_group_add_vector(dependent, &extra, bf16_vector, 0, 0);
  check(weight_group_load(dependent), "the dependent group loads");
  check(weight_group_slot(dependent)->dependency == weight_group_slot(set) &&
            weight_group_slot(dependent)->account == &account &&
            dependent->gpu == gpu,
        "a dependent group shares its backbone's account and GPU");
  freed_count = 0;
  memory_unload(&account);
  check(freed_count == 11 && freed[0] == weight_group_slot(dependent),
        "unloading frees every buffer, the dependent group's first");
  check(weight_group_resident_bytes(set) == 0,
        "an evicted group holds no bytes");

  weight_group_remove(set, &generated);
  check(weight_group_required_bytes(set) == BUFFERS - 100 + SCRATCH &&
            !generated.obj && !generated.size,
        "removing a weight lowers the bytes the group needs, and empties "
        "the weight's buffer");
  check(weight_group_load(set) &&
            !memcmp(plain.ptr, bf16_values, sizeof bf16_values),
        "the group loads again after an eviction");

  weight_group_free(dependent);
  weight_group_free(set);
  check(memory_usage(memory).allocated == 0,
        "freeing the groups gives back their memory");
  return failures != 0;
}
