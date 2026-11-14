#ifndef EREMITE_TESTS_UNIT_RESIDENCY_STAND_INS_H
#define EREMITE_TESTS_UNIT_RESIDENCY_STAND_INS_H
#include "gpu/gpu.h"

// Stand-ins for the GPU's residency calls and working set, for tests that
// include a source that makes them and run without a GPU. Include it before
// that source. A change of residency is only counted, a commit does nothing,
// and Metal reports no working set.

// Buffers made resident, and buffers no longer resident.
static int added, removed;

static void stand_in_set_resident(const Buf *b, int resident) {
  (void)b;
  if (resident)
    added++;
  else
    removed++;
}

static void stand_in_commit_residency(Gpu *gpu) { (void)gpu; }

static uint64_t stand_in_working_set(Gpu *gpu) {
  (void)gpu;
  return 0;
}

#define gpu_set_resident stand_in_set_resident
#define gpu_commit_residency stand_in_commit_residency
#define gpu_recommended_working_set stand_in_working_set

#endif
