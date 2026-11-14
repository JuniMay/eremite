#include "check.h"
#include "copy.h"
#include "gpu/gpu.h"
#include <string.h>

// Tests the order of command buffers: one opened in the queue runs before
// those opened after it, even when it is committed later, and a detached one
// runs after those committed before it, even ones opened after it. Also a
// delayed free with a detached command buffer, and that encoding into a
// committed command buffer or committing it again ends the program.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add_one(device uint *y [[buffer(0)]]) { y[0] += 1; }\n"
    "kernel void double_it(device uint *y [[buffer(0)]]) { y[0] *= 2; }\n";

// Encodes kernel k on y.
static void run(GpuCommandBuffer *c, int k, const Buf *y) {
  gpu_select_kernel(c, k);
  gpu_bind_buffer(c, 0, y, 0);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

// Runs one case that must end the program; the copy returns if it does not.
static void end_case(const char *how) {
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_commit(&c);
  if (!strcmp(how, "--encode-committed")) gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  if (!strcmp(how, "--commit-twice")) gpu_commit(&c);
}

int main(int argc, char **argv) {
  if (argc == 2) {
    end_case(argv[1]);
    return 0;
  }
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  int add_one = gpu_find_kernel(gpu, "add_one"),
      double_it = gpu_find_kernel(gpu, "double_it");
  Buf y = gpu_alloc(gpu, slot, 4);
  uint32_t *value = (uint32_t *)y.ptr;

  *value = 1;
  GpuCommandBuffer first = gpu_begin(gpu), second = gpu_begin(gpu);
  run(&second, add_one, &y);
  gpu_commit(&second);
  run(&first, double_it, &y);
  gpu_commit(&first);
  gpu_sync(gpu);
  check(*value == 3, "a command buffer opened in the queue runs before those "
                     "opened after it, though committed later");

  *value = 1;
  GpuCommandBuffer front = gpu_begin_detached(gpu);
  run(&front, double_it, &y);
  GpuCommandBuffer now = gpu_begin(gpu);
  run(&now, add_one, &y);
  gpu_commit(&now);
  gpu_commit(&front);
  gpu_sync(gpu);
  check(*value == 4, "a detached command buffer runs after those committed "
                     "before it, though opened before them");

  *value = 1;
  GpuCommandBuffer empty = gpu_begin(gpu), behind = gpu_begin(gpu);
  run(&behind, add_one, &y);
  gpu_commit(&behind);
  gpu_commit(&empty);
  gpu_sync(gpu);
  check(*value == 2, "committing a command buffer with nothing encoded lets "
                     "the ones behind it run");

  Buf used = gpu_alloc(gpu, slot, 4);
  uint64_t held = memory_usage(memory).allocated;
  GpuCommandBuffer later = gpu_begin_detached(gpu);
  run(&later, add_one, &used);
  gpu_mark_to_free(&used);
  GpuCommandBuffer other = gpu_begin(gpu);
  run(&other, add_one, &y);
  gpu_commit(&other);
  gpu_sync(gpu);
  int stayed = memory_usage(memory).allocated == held;
  gpu_commit(&later);
  gpu_sync(gpu);
  check(stayed && memory_usage(memory).allocated == held - y.allocated,
        "a buffer that a detached command buffer uses stays counted until it "
        "has finished, though later ones finished first");

  check(ends_with(argv[0], "--encode-committed", "after it was committed"),
        "encoding into a committed command buffer ends the program");
  check(ends_with(argv[0], "--commit-twice", "twice"),
        "committing a command buffer twice ends the program");
  return failures != 0;
}
