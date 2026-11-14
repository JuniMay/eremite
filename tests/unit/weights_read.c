#include "check.h"
#include "base/error.h"
#include "engine/io.h"
#include "engine/weights.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

// Tests that weights load without filling the file cache: every read of a
// tensor at its own offset covers whole 16 KiB sectors into aligned memory,
// and the tensor's bytes arrive intact. Buffers are plain memory and the file
// is an array that the test's read copies from; no Metal device is used.

enum { FILE_BYTES = 1 << 20, ALIGN = 16384 };

static uint8_t file[FILE_BYTES];
static int reads, unaligned;

static ssize_t fixture_pread(int fd, void *buffer, size_t bytes, off_t offset) {
  (void)fd;
  reads++;
  if (offset % ALIGN || bytes % ALIGN || (uintptr_t)buffer % ALIGN) unaligned++;
  if (offset >= FILE_BYTES) return 0;
  size_t n =
      (size_t)offset + bytes > FILE_BYTES ? FILE_BYTES - (size_t)offset : bytes;
  memcpy(buffer, file + offset, n);
  return (ssize_t)n;
}

static Buf fixture_new(Gpu *gpu, MemorySlot *slot, size_t size) {
  (void)gpu;
  Buf b = {.allocated = size, .size = size};
  b.memory = slot;
  if (!memory_claim(b.memory, b.allocated, MEMORY_REQUIRED))
    die("the fixture's pool refused %zu bytes", size);
  b.ptr = calloc(1, size ? size : 1);
  b.obj = b.ptr;
  return b;
}

static void fixture_free(Buf *b) {
  if (b->obj) {
    memory_release(b->memory, b->allocated);
    free(b->ptr);
  }
  *b = (Buf){0};
}

static size_t fixture_allocation_size(size_t size) { return size; }

static void fixture_resident(const Buf *b, int on) { (void)b, (void)on; }

static void fixture_commit(Gpu *gpu) { (void)gpu; }

static void fixture_read(int fd, uint64_t off, uint64_t len, Buf *dst) {
  (void)fd, (void)off, (void)len, (void)dst;
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

int main(void) {
  for (size_t i = 0; i < FILE_BYTES; i++)
    file[i] = (uint8_t)(i * 2654435761u >> 24);
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  MemoryAccount account;
  memory_account_init(memory, &account, "fixture model", 0);
  WeightGroup *set = weight_group_new(NULL, &account, "fixture weights");

  // Tensors at offsets that a safetensors file gives: unaligned, of any size,
  // one reaching the end of the file.
  const struct {
    uint64_t off;
    size_t size;
  } at[] = {
      {12345, 100000}, {200001, 3}, {300000, 70000}, {FILE_BYTES - 5000, 5000}};

  enum { TENSORS = sizeof at / sizeof *at };

  Buf targets[TENSORS] = {0};
  for (int i = 0; i < TENSORS; i++)
    weight_group_add_tensor(
        set, &targets[i],
        (Tensor){.fd = 3, .off = at[i].off, .size = at[i].size}, 0);
  int loaded = weight_group_load(set), intact = 1;
  for (int i = 0; i < TENSORS; i++)
    intact &= !memcmp(targets[i].ptr, file + at[i].off, at[i].size);
  printf("%d reads, %d not whole sectors into aligned memory\n", reads,
         unaligned);
  check(loaded, "the weights load");
  check(!unaligned,
        "every read covers whole 16 KiB sectors into aligned memory");
  check(intact, "every tensor's bytes match the file's");
  return failures != 0;
}
