#include "check.h"
#include "gpu/gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the residency sets: how resident buffers spread over sets, views,
// removing and adding again, freeing, Metal's limit of 32 sets, the table of
// resident buffers, and that the GPU reads what the CPU wrote.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void read_value(device const uint *x [[buffer(0)]],\n"
    "                       device uint *y [[buffer(1)]]) {\n"
    "  y[0] = x[0];\n"
    "}\n";

// A set holds 1 MiB here, so it has room for one buffer of 768 KiB, not two.
static constexpr size_t SET_BYTES = 1 << 20;
static constexpr size_t BUFFER_BYTES = 768 << 10;

// Sets an empty buffer resident in a child process, before the GPU starts.
// Returns the child's exit status, or -1 if it did not exit.
static int empty_buffer_status(void) {
  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    Buf empty = {0};
    gpu_set_resident(&empty, 1);
    _exit(0);
  }
  int status;
  if (child < 0 || waitpid(child, &status, 0) != child) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main(void) {
  check(empty_buffer_status() == 1,
        "setting an empty buffer resident ends the program");

  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  gpu_set_residency_set_bytes(gpu, SET_BYTES);
  Buf a = gpu_alloc(gpu, slot, BUFFER_BYTES),
      b = gpu_alloc(gpu, slot, BUFFER_BYTES);
  gpu_set_resident(&a, 1);
  gpu_set_resident(&b, 1);
  gpu_commit_residency(gpu);
  GpuResidency r = gpu_residency(gpu);
  check(r.sets == 2 && r.buffers == 2 && r.bytes == a.allocated + b.allocated &&
            r.largest_set == a.allocated,
        "two buffers that do not fit in one set go to two sets");

  Buf view = a;
  view.off = 16, view.size -= 16, view.allocated = 0;
  gpu_set_resident(&view, 1);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 2 && r.bytes == a.allocated + b.allocated,
        "a view adds nothing to the sets");

  gpu_set_resident(&a, 0);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 1 && r.bytes == b.allocated,
        "a removed buffer leaves its set");
  gpu_set_resident(&a, 1);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.sets == 2 && r.buffers == 2,
        "a buffer added again goes to an empty set, not a new one");

  ((uint32_t *)a.ptr)[0] = 123456789;
  int read = gpu_find_kernel(gpu, "read_value");
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, read);
  gpu_bind_buffer(&c, 0, &a, 0);
  gpu_bind_buffer(&c, 1, &b, 0);
  gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(((uint32_t *)b.ptr)[0] == 123456789,
        "the GPU reads what the CPU wrote to a resident buffer");

  gpu_free(&a);
  r = gpu_residency(gpu);
  check(r.buffers == 2 && r.bytes == b.allocated,
        "a freed resident buffer stays held until the next commit");
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 1, "at the commit, the freed buffer leaves its set");
  gpu_free(&b);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 0 && r.bytes == 0, "freed buffers leave the sets");

  Buf large = gpu_alloc(gpu, slot, 2 * SET_BYTES),
      small = gpu_alloc(gpu, slot, 128 << 10);
  gpu_set_resident(&large, 1);
  gpu_set_resident(&small, 1);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.sets == 2 && r.largest_set == large.allocated,
        "a buffer larger than a set's bytes has a set to itself");
  gpu_free(&large), gpu_free(&small);
  gpu_commit_residency(gpu);

  constexpr int COUNT = 80;
  Buf many[COUNT];
  for (int i = 0; i < COUNT; i++) {
    many[i] = gpu_alloc(gpu, slot, BUFFER_BYTES);
    gpu_set_resident(&many[i], 1);
  }
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  uint64_t each = many[0].allocated;
  check(r.sets == 32 && r.buffers == COUNT && r.bytes == COUNT * each,
        "80 buffers share Metal's 32 sets");
  // The first 32 buffers fill the 32 sets. Each of the other 48 goes to the
  // set that holds the fewest bytes, so no set holds more than 3.
  check(r.largest_set == 3 * each,
        "past 32 sets, a buffer goes to the set that holds the fewest bytes");
  int kept = 1;
  for (int round = 0; round < 4; round++) {
    for (int i = 0; i < COUNT; i += 2) gpu_set_resident(&many[i], 0);
    gpu_commit_residency(gpu);
    r = gpu_residency(gpu);
    kept &= r.buffers == COUNT / 2 && r.bytes == COUNT / 2 * each;
    for (int i = 0; i < COUNT; i += 2) gpu_set_resident(&many[i], 1);
    gpu_commit_residency(gpu);
    r = gpu_residency(gpu);
    kept &= r.buffers == COUNT && r.bytes == COUNT * each;
  }
  check(kept, "removing half of them and adding them again, four times, "
              "keeps the counts");
  for (int i = 0; i < COUNT; i++) gpu_free(&many[i]);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 0 && r.bytes == 0, "freeing them all empties the sets");

  // A window of 100 resident buffers slides over 300 buffers, so that buffers
  // keep leaving and joining the table, as cached buffers do.
  constexpr int WINDOW = 100, SLIDE = 300;
  static Buf slide[SLIDE];
  for (int i = 0; i < SLIDE; i++) slide[i] = gpu_alloc(gpu, slot, 16 << 10);
  for (int i = 0; i < SLIDE; i++) {
    gpu_set_resident(&slide[i], 1);
    if (i >= WINDOW) gpu_set_resident(&slide[i - WINDOW], 0);
  }
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == WINDOW && r.bytes == WINDOW * slide[0].allocated,
        "a window of 100 resident buffers sliding over 300 keeps the counts");
  for (int i = 0; i < SLIDE; i++) gpu_free(&slide[i]);
  gpu_commit_residency(gpu);
  r = gpu_residency(gpu);
  check(r.buffers == 0 && r.bytes == 0,
        "freeing the window's buffers empties the sets");
  return failures != 0;
}
