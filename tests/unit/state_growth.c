#include "check.h"
#include "engine/state_growth.h"
#include <stdio.h>

// Tests configurable geometric-then-linear state capacity growth policies.

// Standard test geometry matching a 128 MiB linear step cap at 32 KiB/token.
static constexpr size_t BYTES_PER_TOKEN_BF16 = 32768;
static constexpr size_t MAX_STEP_BYTES = 128ull << 20;
static constexpr int START_CAP = 256;
static constexpr int ALIGN_TOKENS = 256;
static constexpr int MAX_CTX = 131072;

int main(void) {
  StateGrowthPolicy p = state_growth_policy_for_bytes(
      START_CAP, BYTES_PER_TOKEN_BF16, MAX_STEP_BYTES, START_CAP, ALIGN_TOKENS,
      MAX_CTX);
  check(p.start_cap == START_CAP && p.max_step == 4096 &&
            p.align == ALIGN_TOKENS && p.max_cap == MAX_CTX,
        "byte-budgeted policy derives 4096-token linear step for 32 KiB/token");

  check(state_growth_next(&p, 0, 1) == 256, "fresh state allocates start_cap");
  check(state_growth_next(&p, 256, 256) == 256,
        "existing capacity stays unchanged when sufficient");
  check(state_growth_next(&p, 256, 257) == 512,
        "doubles below max_step (256 -> 512)");
  check(state_growth_next(&p, 2048, 2049) == 4096,
        "doubles up to max_step (2048 -> 4096)");
  check(state_growth_next(&p, 4096, 4097) == 8192,
        "reaches 8192 at the first max_step increment");
  check(state_growth_next(&p, 8192, 8193) == 12288,
        "grows linearly by max_step above doubling cap (8192 -> 12288)");
  check(state_growth_next(&p, 65536, 65537) == 69632,
        "65537 tokens grows by 4096 instead of doubling to 131072");
  check(state_growth_next(&p, 0, 65537) == 69632,
        "fresh state jumping to 65537 follows the same capped sequence");
  check(state_growth_next(&p, 126976, 131000) == 131072, "clamps to max_cap");
  check(state_growth_next(&p, 131072, 131073) == -1,
        "returns -1 beyond max_cap");
  check(state_growth_next(&p, 0, -1) == -1, "returns -1 for negative need");

  StateGrowthPolicy uncapped = {
      .start_cap = 256, .factor = 2, .max_step = 0, .align = 0, .max_cap = 0};
  check(state_growth_next(&uncapped, 65536, 65537) == 131072,
        "max_step 0 preserves pure doubling when configured");

  StateGrowthPolicy small_ctx = {
      .start_cap = 16384, .factor = 2, .max_step = 32768, .max_cap = 4096};
  check(state_growth_next(&small_ctx, 0, 1) == 4096,
        "start_cap clamps to max_cap when context is smaller");

  return failures != 0;
}
