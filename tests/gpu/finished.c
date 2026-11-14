#include "check.h"
#include "copy.h"
#include "gpu/gpu.h"
#include <stdio.h>
#include <string.h>

// Tests the checks that do not wait: whether a command buffer has finished,
// and whether an event has a value, return at once while the GPU still runs,
// and the waits for them return once it has finished. A detached command
// buffer left open holds back no other. Also the count of open command
// buffers in the queue, and that checking a command buffer that is open or
// was never begun ends the program.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void slow(device uint *y [[buffer(0)]],\n"
    "                 constant uint &n [[buffer(1)]]) {\n"
    "  uint x = y[0];\n"
    "  for (uint i = 0; i < n; i++) x = x * 1664525u + 1013904223u;\n"
    "  y[0] = x;\n"
    "}\n";

// Steps of the slow kernel's loop: about 60 ms on one GPU thread of the M5
// Max.
static constexpr uint32_t SLOW_STEPS = 4'000'000;

// Encodes the slow kernel on y.
static void slow(GpuCommandBuffer *c, int kernel, const Buf *y) {
  gpu_select_kernel(c, kernel);
  gpu_bind_buffer(c, 0, y, 0);
  gpu_copy_bytes(c, 1, &SLOW_STEPS, sizeof SLOW_STEPS);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

// Runs one case that must end the program; the copy returns if it does not.
static void end_case(const char *how) {
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuCommandBuffer c = gpu_begin(gpu);
  if (!strcmp(how, "--open")) gpu_has_finished(gpu, c.number);
  if (!strcmp(how, "--never-begun")) gpu_wait_until_finished(gpu, c.number + 1);
}

int main(int argc, char **argv) {
  if (argc == 2) {
    end_case(argv[1]);
    return 0;
  }
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  int kernel = gpu_find_kernel(gpu, "slow");
  Buf y = gpu_alloc(gpu, memory_shared(memory), 4);
  GpuEvent *event = gpu_event_new(gpu);

  GpuCommandBuffer c = gpu_begin(gpu);
  slow(&c, kernel, &y);
  uint64_t value = gpu_signal(&c, event);
  gpu_commit(&c);
  int running = !gpu_has_finished(gpu, c.number),
      unsignaled = !gpu_has_signaled(event, value);
  gpu_wait_until_finished(gpu, c.number);
  check(running && gpu_has_finished(gpu, c.number),
        "a command buffer shows as unfinished while it runs, and finished "
        "after a wait for it");
  check(unsignaled && gpu_has_signaled(event, value),
        "an event shows its value only once the GPU has signaled it");
  gpu_sync(gpu);
  printf("the slow kernel ran %.1f ms\n", gpu_times(gpu).busy_seconds * 1e3);
  check(gpu_has_finished(gpu, c.number),
        "a command buffer still shows as finished after it is checked");

  check(gpu_count_uncommitted_in_queue(gpu) == 0,
        "no command buffer is open at first");
  GpuCommandBuffer detached = gpu_begin_detached(gpu);
  slow(&detached, kernel, &y);
  check(gpu_count_uncommitted_in_queue(gpu) == 0,
        "a detached command buffer is not counted as open in the queue");
  GpuCommandBuffer queued = gpu_begin(gpu);
  check(gpu_count_uncommitted_in_queue(gpu) == 1,
        "a command buffer begun in the queue is counted until it is "
        "committed");
  slow(&queued, kernel, &y);
  gpu_commit(&queued);
  check(gpu_count_uncommitted_in_queue(gpu) == 0,
        "committing it takes it out of the count");
  gpu_wait_until_finished(gpu, queued.number);
  check(gpu_has_finished(gpu, queued.number),
        "a command buffer finishes while a detached one begun before it is "
        "still open");
  gpu_commit(&detached);
  gpu_sync(gpu);

  check(ends_with(argv[0], "--open", "still open"),
        "checking a command buffer that is still open ends the program");
  check(ends_with(argv[0], "--never-begun", "never begun"),
        "waiting for a command buffer that was never begun ends the program");
  return failures != 0;
}
