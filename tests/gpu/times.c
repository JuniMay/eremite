#include "check.h"
#include "gpu/gpu.h"
#include <unistd.h>

// Tests the GPU's times: busy time, an idle time between two command buffers
// committed 20 ms apart, and no idle time counted across a gpu_sync.

static const char library[] =
    "kernel void nothing(device uint *y [[buffer(0)]]) { y[0] = 1; }\n";

static int nothing;

// Commits one command buffer that runs the kernel on y.
static void commit_one(Gpu *gpu, const Buf *y) {
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, nothing);
  gpu_bind_buffer(&c, 0, y, 0);
  gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  gpu_commit(&c);
}

int main(void) {
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  nothing = gpu_find_kernel(gpu, "nothing");
  Buf y = gpu_alloc(gpu, memory_shared(memory), 4);

  // A first command buffer wakes the GPU, whose first start can come late.
  // Then two more are committed 20 ms apart. The first of them has finished by
  // then, so the GPU sits idle in between.
  constexpr useconds_t GAP_US = 20000;
  commit_one(gpu, &y);
  gpu_sync(gpu);
  commit_one(gpu, &y);
  usleep(GAP_US);
  commit_one(gpu, &y);
  gpu_sync(gpu);
  GpuTimes times = gpu_times(gpu);
  check(times.busy_seconds > 0, "the GPU's busy time is counted");
  check(times.idle_seconds > GAP_US * 0.5e-6,
        "an idle time of about 20 ms between two command buffers is counted");

  commit_one(gpu, &y);
  gpu_sync(gpu);
  GpuTimes after = gpu_times(gpu);
  check(after.idle_seconds == times.idle_seconds &&
            after.busy_seconds > times.busy_seconds,
        "the time across a gpu_sync is not counted as idle");
  return failures != 0;
}
