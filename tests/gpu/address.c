#include "check.h"
#include "gpu/gpu.h"

// Tests reaching a buffer through its GPU address: a kernel reads x through
// an address kept in a table, when the command buffer uses x.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Ref {\n"
    "  device const uint *x;\n"
    "};\n"
    "kernel void through(device const Ref *refs [[buffer(0)]],\n"
    "                    device uint *y [[buffer(1)]]) {\n"
    "  y[0] = refs[0].x[0];\n"
    "}\n";

int main(void) {
  Memory *memory = memory_new();
  MemorySlot *slot = memory_shared(memory);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  int through = gpu_find_kernel(gpu, "through");
  Buf x = gpu_alloc(gpu, slot, 4), table = gpu_alloc(gpu, slot, 8),
      y = gpu_alloc(gpu, slot, 4);
  check(x.address && x.address != table.address,
        "each buffer has its own GPU address");
  ((uint32_t *)x.ptr)[0] = 123456789;
  ((uint64_t *)table.ptr)[0] = x.address;
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, through);
  gpu_bind_buffer(&c, 0, &table, 0);
  gpu_bind_buffer(&c, 1, &y, 0);
  gpu_declare_read(&c, &x);
  gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  gpu_commit(&c);
  gpu_sync(gpu);
  check(((uint32_t *)y.ptr)[0] == 123456789,
        "a kernel reads a used buffer through its address");
  return failures != 0;
}
