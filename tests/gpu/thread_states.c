#include "check.h"
#include "copy.h"
#include "gpu/gpu.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Tests how the GPU's idle times are split by the committing thread's states:
// an idle time of about 20 ms while the thread is in one state counts for that
// state, and the thread's seconds in each state are counted. Also that a state
// out of range ends the program.

static const char library[] =
    "kernel void nothing(device uint *y [[buffer(0)]]) { y[0] = 1; }\n";

// Thread states of the test: before the idle time, during it, after it.
enum { BEFORE = 1, IDLE = 2, AFTER = 3 };

// Commits one command buffer that runs the kernel on y, and returns its
// number.
static uint64_t commit_one(Gpu *gpu, int kernel, const Buf *y) {
  GpuCommandBuffer c = gpu_begin(gpu);
  gpu_select_kernel(&c, kernel);
  gpu_bind_buffer(&c, 0, y, 0);
  gpu_run_kernel(&c, 1, 1, 1, 1, 1, 1);
  gpu_commit(&c);
  return c.number;
}

int main(int argc, char **argv) {
  if (argc == 2) {
    gpu_set_thread_state(gpu_new(library, sizeof library - 1),
                         GPU_THREAD_STATES);
    return 0;
  }
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  int nothing = gpu_find_kernel(gpu, "nothing");
  Buf y = gpu_alloc(gpu, memory_shared(memory), 4);

  // A first command buffer wakes the GPU, whose first start can come late.
  commit_one(gpu, nothing, &y);
  gpu_sync(gpu);
  // The waits below check the command buffers without a sync, which would
  // leave the idle time uncounted.
  constexpr useconds_t GAP_US = 20000;
  gpu_set_thread_state(gpu, BEFORE);
  uint64_t first = commit_one(gpu, nothing, &y);
  gpu_wait_until_finished(gpu, first);
  gpu_check_finished(gpu);
  gpu_set_thread_state(gpu, IDLE);
  usleep(GAP_US);
  gpu_set_thread_state(gpu, AFTER);
  uint64_t second = commit_one(gpu, nothing, &y);
  gpu_wait_until_finished(gpu, second);
  gpu_check_finished(gpu);
  GpuTimes times = gpu_times(gpu);
  double split = 0;
  for (int k = 0; k < GPU_THREAD_STATES; k++) split += times.idle_by_state[k];
  printf("idle %.2f ms, %.2f ms of it in the idle state; the thread spent "
         "%.2f ms in that state\n",
         times.idle_seconds * 1e3, times.idle_by_state[IDLE] * 1e3,
         times.thread_seconds[IDLE] * 1e3);
  check(times.idle_by_state[IDLE] > GAP_US * 0.9e-6 &&
            times.idle_by_state[IDLE] <= times.idle_seconds,
        "the GPU's idle time while the thread waited counts for that state");
  check(split <= times.idle_seconds * 1.000001 &&
            split > times.idle_seconds * 0.9,
        "the idle time is split over the thread's states");
  check(times.thread_seconds[IDLE] >= GAP_US * 1e-6 &&
            times.thread_seconds[BEFORE] > 0 && times.thread_seconds[AFTER] > 0,
        "the thread's seconds in each state are counted");
  check(ends_with(argv[0], "--out-of-range", "out of range"),
        "a thread state out of range ends the program");
  return failures != 0;
}
