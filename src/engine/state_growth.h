#ifndef EREMITE_ENGINE_STATE_GROWTH_H
#define EREMITE_ENGINE_STATE_GROWTH_H
#include <stddef.h>

// Configurable capacity growth rules for sequence-indexed state buffers.
// Capacities double from start_cap until a step would exceed max_step, then
// grow linearly by max_step up to max_cap.

// Capacity growth parameters for one family or buffer layout.
typedef struct {
  // Initial capacity allocated for a fresh state, in positions.
  int start_cap;
  // Multiplicative factor applied during geometric growth (values below 2 use
  // 2).
  int factor;
  // Maximum positions added in a single growth step; 0 leaves doubling
  // uncapped.
  int max_step;
  // Rounding granularity for capacities above start_cap; values <= 1 disable
  // alignment.
  int align;
  // Maximum permitted capacity, in positions; 0 leaves capacity uncapped.
  int max_cap;
} StateGrowthPolicy;

// Returns the next capacity holding at least `needed` positions, starting from
// `current` (0 for a fresh state), following `policy`. Returns -1 when `needed`
// is negative or exceeds `policy->max_cap`.
int state_growth_next(const StateGrowthPolicy *policy, int current, int needed);

// Builds a growth policy whose linear growth step fits within `max_step_bytes`
// for buffers consuming `bytes_per_pos` bytes per position, bounded below by
// `min_step` and rounded down to `align`.
StateGrowthPolicy state_growth_policy_for_bytes(int start_cap,
                                                size_t bytes_per_pos,
                                                size_t max_step_bytes,
                                                int min_step, int align,
                                                int max_cap);

#endif
