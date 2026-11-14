#include "check.h"
#include "copy.h"
#include "gpu/gpu.h"
#include <string.h>

// Tests signals and waits: after a wait for a signal, the CPU reads what the
// GPU wrote before it, and a wait that cannot end ends the program instead of
// returning. Those cases run in a copy of this program, because Metal does not
// work after fork.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add(device const uint *x [[buffer(0)]],\n"
    "                device uint *y [[buffer(1)]],\n"
    "                constant uint &n [[buffer(2)]]) {\n"
    "  y[0] = x[0] + n;\n"
    "}\n";

// Encodes y = x + 1.
static void add_one(GpuCommandBuffer *c, int add, const Buf *x, const Buf *y) {
  uint32_t one = 1;
  gpu_select_kernel(c, add);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, y, 0);
  gpu_copy_bytes(c, 2, &one, sizeof one);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

// Runs one case that must end the program; the copy returns if it does not.
static void end_case(const char *how) {
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuEvent *event = gpu_event_new(gpu);
  if (!strcmp(how, "--never-given")) gpu_wait_signal(event, 1);
  if (!strcmp(how, "--never-committed")) {
    GpuCommandBuffer c = gpu_begin(gpu);
    uint64_t value = gpu_signal(&c, event);
    gpu_set_signal_timeout(gpu, 0.5);
    gpu_wait_signal(event, value);
  }
  if (!strcmp(how, "--another-gpu")) {
    GpuCommandBuffer c = gpu_begin(gpu_new(library, sizeof library - 1));
    gpu_signal(&c, event);
  }
}

int main(int argc, char **argv) {
  if (argc == 2) {
    end_case(argv[1]);
    return 0;
  }
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuEvent *event = gpu_event_new(gpu);
  int add = gpu_find_kernel(gpu, "add");
  Buf x = gpu_alloc(gpu, memory_shared(memory), 4),
      y = gpu_alloc(gpu, memory_shared(memory), 4),
      z = gpu_alloc(gpu, memory_shared(memory), 4);
  ((uint32_t *)x.ptr)[0] = 41;
  GpuCommandBuffer c = gpu_begin(gpu);
  add_one(&c, add, &x, &y);
  uint64_t value = gpu_signal(&c, event);
  add_one(&c, add, &y, &z);
  gpu_commit(&c);
  gpu_wait_signal(event, value);
  check(
      value == 1 && ((uint32_t *)y.ptr)[0] == 42,
      "after a wait for a signal, the CPU reads what the GPU wrote before it");
  gpu_sync(gpu);
  check(((uint32_t *)z.ptr)[0] == 43,
        "the work after a signal runs in the same command buffer");

  check(ends_with(argv[0], "--never-given", "never gave out"),
        "a wait for a value the event never gave out ends the program without "
        "waiting");
  check(ends_with(argv[0], "--never-committed", "not signaled in 0.5 s"),
        "a wait for a signal that does not come ends the program after the "
        "timeout");
  check(ends_with(argv[0], "--another-gpu", "another GPU"),
        "a signal on an event of another GPU ends the program");
  return failures != 0;
}
