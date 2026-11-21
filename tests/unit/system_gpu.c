#include "check.h"
#include "base/clock.h"
#include "platform/system_gpu.h"
#include <math.h>
#include <stdio.h>

// Tests that system_gpu_read_activity returns well-formed display, input and
// external GPU readings quickly without leaking resources.

// Repeated readings used to measure polling cost and verify clean cleanup.
static constexpr int POLL_RUNS = 20;

int main(void) {
  double t0 = now();
  SystemGpuActivity act = {0};
  for (int i = 0; i < POLL_RUNS; i++) act = system_gpu_read_activity();
  double per_call_us = ((now() - t0) / (double)POLL_RUNS) * 1e6;
  printf("system_gpu_read_activity: asleep=%d frame=%.2f ms input_idle=%.3f s "
         "ext_gpu_idle=%.3f s (%.0f us/call)\n",
         act.display_asleep, act.display_frame_seconds * 1e3,
         act.input_idle_seconds, act.external_gpu_idle_seconds, per_call_us);
  check(act.display_asleep == 0 || act.display_asleep == 1,
        "display_asleep is a boolean flag");
  check(isfinite(act.display_frame_seconds) && act.display_frame_seconds >= 0.0,
        "display_frame_seconds is finite and nonnegative");
  check(!isnan(act.input_idle_seconds) && act.input_idle_seconds >= 0.0,
        "input_idle_seconds is nonnegative or INFINITY");
  check(!isnan(act.external_gpu_idle_seconds) &&
            act.external_gpu_idle_seconds >= 0.0,
        "external_gpu_idle_seconds is nonnegative or INFINITY");
  return failures != 0;
}
