#include "engine/state_growth.h"
#include <limits.h>
#include <stdint.h>

// Default geometric growth multiplier when a policy leaves factor unset.
static constexpr int DEFAULT_GROWTH_FACTOR = 2;

// Rounds `value` up to the next multiple of `align`, clamped to `max_cap` when
// positive.
static int64_t align_up(int64_t value, int align, int max_cap) {
  if (align <= 1) return value;
  int64_t rem = value % align;
  if (rem) value += align - rem;
  if (max_cap > 0 && value > max_cap) value = max_cap;
  return value;
}

int state_growth_next(const StateGrowthPolicy *policy, int current,
                      int needed) {
  if (!policy || needed < 0) return -1;
  int max_cap = policy->max_cap;
  if (max_cap > 0 && needed > max_cap) return -1;
  int start = policy->start_cap > 0 ? policy->start_cap : 1;
  if (max_cap > 0 && start > max_cap) start = max_cap;
  if (current > 0 && needed <= current) return current;

  int64_t cap = current > 0 ? current : start;
  int factor = policy->factor >= DEFAULT_GROWTH_FACTOR ? policy->factor
                                                       : DEFAULT_GROWTH_FACTOR;
  int max_step = policy->max_step;

  while (cap < needed) {
    int64_t inc = cap * (int64_t)(factor - 1);
    if (max_step > 0 && inc > max_step) inc = max_step;
    if (inc < 1) inc = 1;
    int64_t next = cap + inc;
    if (max_cap > 0 && next >= max_cap) {
      cap = max_cap;
      break;
    }
    if (next > INT_MAX) return max_cap > 0 ? max_cap : INT_MAX;
    cap = next;
  }

  if (policy->align > 1 && cap > start) {
    cap = align_up(cap, policy->align, max_cap);
  }
  if (max_cap > 0 && cap > max_cap) cap = max_cap;
  return (int)cap;
}

StateGrowthPolicy state_growth_policy_for_bytes(int start_cap,
                                                size_t bytes_per_pos,
                                                size_t max_step_bytes,
                                                int min_step, int align,
                                                int max_cap) {
  int start = start_cap > 0 ? start_cap : 1;
  int step = 0;
  if (max_step_bytes > 0 && bytes_per_pos > 0) {
    size_t raw = max_step_bytes / bytes_per_pos;
    if (align > 1) raw = raw / (size_t)align * (size_t)align;
    int floor_step = min_step > 0 ? min_step : start;
    if (raw < (size_t)floor_step) raw = (size_t)floor_step;
    if (raw > (size_t)INT_MAX) raw = (size_t)INT_MAX;
    step = (int)raw;
  } else if (min_step > 0) {
    step = min_step;
  }
  return (StateGrowthPolicy){.start_cap = start,
                             .factor = DEFAULT_GROWTH_FACTOR,
                             .max_step = step,
                             .align = align,
                             .max_cap = max_cap};
}
