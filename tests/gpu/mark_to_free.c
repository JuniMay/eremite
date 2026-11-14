#include "check.h"
#include "gpu/gpu.h"

// Tests gpu_mark_to_free. A buffer that unfinished GPU work may use must stay
// allocated and counted until that work has finished. A buffer that no
// unfinished work can use is freed at once.
//
// The GPU runs command buffers in the order they were begun. So while one
// command buffer is begun but not committed, the ones begun after it cannot
// run. The test uses this to keep GPU work unfinished as long as it needs.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add_one(device uint *y [[buffer(0)]]) { y[0] += 1; }\n";

static int add_one;

// Encodes y += 1.
static void add_one_to(GpuCommandBuffer *c, const Buf *y) {
  gpu_select_kernel(c, add_one);
  gpu_bind_buffer(c, 0, y, 0);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  add_one = gpu_find_kernel(gpu, "add_one");
  size_t page = gpu_round_up_to_pages(4);

  Buf idle = gpu_alloc(gpu, slot, 4);
  uint64_t held = memory_usage(memory).allocated;
  gpu_mark_to_free(&idle);
  check(!idle.obj && memory_usage(memory).allocated == held - page,
        "when no GPU work is unfinished, the buffer is freed at once");

  Buf used = gpu_alloc(gpu, slot, 4);
  held = memory_usage(memory).allocated;
  GpuCommandBuffer ahead = gpu_begin(gpu), c = gpu_begin(gpu);
  add_one_to(&c, &used);
  gpu_commit(&c);
  gpu_mark_to_free(&used);
  check(
      !used.obj && memory_usage(memory).allocated == held,
      "while GPU work that may use the buffer is unfinished, it stays counted");
  gpu_check_finished(gpu);
  check(memory_usage(memory).allocated == held,
        "gpu_check_finished does not free it while that work is unfinished");
  gpu_commit(&ahead);
  // Now the work runs. Metal can report a command buffer as unfinished for a
  // moment after its work is done. So the test calls gpu_check_finished until
  // the buffer is freed.
  while (memory_usage(memory).allocated == held) gpu_check_finished(gpu);
  check(memory_usage(memory).allocated == held - page,
        "once that work has finished, gpu_check_finished frees it, with no "
        "gpu_sync");

  Buf encoding = gpu_alloc(gpu, slot, 4);
  held = memory_usage(memory).allocated;
  GpuCommandBuffer d = gpu_begin(gpu);
  add_one_to(&d, &encoding);
  gpu_mark_to_free(&encoding);
  int stayed = memory_usage(memory).allocated == held;
  gpu_commit(&d);
  gpu_sync(gpu);
  check(stayed && memory_usage(memory).allocated == held - page,
        "a command buffer still being encoded also keeps the buffer until it "
        "has finished");

  Buf early = gpu_alloc(gpu, slot, 4), out = gpu_alloc(gpu, slot, 4);
  held = memory_usage(memory).allocated;
  GpuCommandBuffer e = gpu_begin(gpu);
  add_one_to(&e, &early);
  gpu_commit(&e);
  gpu_mark_to_free(&early);
  GpuCommandBuffer later = gpu_begin(gpu);
  gpu_sync(gpu);
  check(memory_usage(memory).allocated == held - page,
        "a command buffer begun after the call does not keep the buffer");
  add_one_to(&later, &out);
  gpu_commit(&later);
  gpu_sync(gpu);
  check(((uint32_t *)out.ptr)[0] == 1,
        "gpu_sync does not break a command buffer that is still being encoded");
  return failures != 0;
}
