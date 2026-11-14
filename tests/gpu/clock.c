#include "check.h"
#include "base/clock.h"
#include "gpu/gpu.h"
#include <stdio.h>
#include <stdlib.h>

// Tests keeping the GPU's clock up: the clock keeper's buffer is counted; once
// the GPU's clock has settled, a spin asked to last 20 ms takes about 20 ms;
// work on the main queue runs beside a spin; and a command buffer open in the
// main queue does not hold a spin back. The CPU times each spin, from asking
// for it to the end of a wait for it.

static const char library[] = "kernel void nothing() {}\n";

static int ascending(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(void) {
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  uint64_t before = memory_usage(memory).allocated;
  GpuClockKeeper *keeper = gpu_clock_keeper_new(gpu, memory_shared(memory));
  check(memory_usage(memory).allocated ==
            before + gpu_round_up_to_pages(sizeof(float)),
        "the clock keeper's buffer is counted in its slot");

  // The first 15 spins let the GPU's clock settle; the last 5 are timed.
  constexpr double SPIN_SECONDS = 0.02;
  constexpr int SETTLE = 15, TIMED = 5;
  double samples[TIMED];
  for (int i = 0; i < SETTLE + TIMED; i++) {
    double start = now();
    gpu_keep_clock_up(keeper, SPIN_SECONDS);
    gpu_wait_for_clock_keeper(keeper);
    if (i >= SETTLE) samples[i - SETTLE] = now() - start;
  }
  qsort(samples, TIMED, sizeof *samples, ascending);
  double median = samples[TIMED / 2];
  printf("a spin of 20 ms took %.1f ms, the median of 5\n", median * 1e3);
  check(median >= SPIN_SECONDS * 0.75 && median <= SPIN_SECONDS * 1.5,
        "a spin of 20 ms takes between 15 and 30 ms");

  // Work committed on the main queue during a spin of 100 ms is done long
  // before the spin.
  constexpr double LONG_SPIN_SECONDS = 0.1;
  int nothing = gpu_find_kernel(gpu, "nothing");
  double start = now();
  gpu_keep_clock_up(keeper, LONG_SPIN_SECONDS);
  GpuCommandBuffer beside = gpu_begin(gpu);
  gpu_select_kernel(&beside, nothing);
  gpu_run_kernel(&beside, 1, 1, 1, 1, 1, 1);
  gpu_commit(&beside);
  gpu_sync(gpu);
  double done = now() - start;
  gpu_wait_for_clock_keeper(keeper);
  printf("work committed during a spin of 100 ms was done after %.1f ms\n",
         done * 1e3);
  check(done < LONG_SPIN_SECONDS / 2,
        "work on the main queue runs beside a spin");

  // A spin in the main queue would wait behind the open command buffer until
  // it is committed; the test's time limit would end the test.
  GpuCommandBuffer open = gpu_begin(gpu);
  start = now();
  gpu_keep_clock_up(keeper, SPIN_SECONDS);
  gpu_wait_for_clock_keeper(keeper);
  check(now() - start < SPIN_SECONDS * 1.5,
        "a command buffer open in the main queue does not hold a spin back");
  gpu_commit(&open);
  gpu_sync(gpu);
  return failures != 0;
}
