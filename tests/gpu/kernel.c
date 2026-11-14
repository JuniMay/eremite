#include "check.h"
#include "gpu/gpu.h"

// Tests running a kernel: the GPU reads what the CPU wrote, the CPU reads what
// the kernel wrote once the work is synced, and buffers are counted in model
// memory while they exist.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add(device const uint *x [[buffer(0)]], device uint *y "
    "[[buffer(1)]],\n"
    "                constant uint &n [[buffer(2)]]) {\n"
    "  y[0] = x[0] + n;\n"
    "}\n";

int main(void) {
  memory_set_limit(1ull << 30);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  int add = gpu_find_kernel(gpu, "add");
  check(gpu_find_kernel(gpu, "add") == add,
        "a kernel asked for twice keeps its number");

  uint64_t before = memory_usage().allocated;
  Buf x = gpu_alloc(gpu, memory_shared(), 4),
      y = gpu_alloc(gpu, memory_shared(), 4);
  check(memory_usage().allocated == before + 2 * gpu_round_up_to_pages(4),
        "each buffer is counted as whole pages");

  ((uint32_t *)x.ptr)[0] = 41;
  uint32_t n = 1;
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, add);
  gpu_bind_buffer(&c, 0, &x, 0);
  gpu_bind_buffer(&c, 1, &y, 0);
  gpu_copy_bytes(&c, 2, &n, sizeof n);
  gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(((uint32_t *)y.ptr)[0] == 42,
        "after a sync, the CPU reads what the kernel wrote");

  gpu_free(&x), gpu_free(&y);
  check(!x.obj && memory_usage().allocated == before,
        "freed buffers are no longer counted");
  return failures != 0;
}
