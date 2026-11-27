#include "check.h"
#include "base/clock.h"
#include "engine/memory.h"
#include "platform/memory_policy.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Tests the layered macOS platform memory policy and its integration with
// Memory: strictly non-increasing pre-existing file-cache floor, counting
// growing file cache as available, immediate guard-pool occupation of released
// and idle headroom above reserve + hysteresis, synchronous release on
// memory_claim, and immediate yield under external memory growth, OS pressure
// or process-local compression.

static constexpr uint64_t MIB = 1ull << 20;
static constexpr uint64_t GIB = 1ull << 30;
static constexpr size_t CHUNK = 64ull << 20;
static constexpr uint64_t HYSTERESIS = 256ull << 20;

// Simulated host counters for deterministic policy testing.
static uint64_t sim_total = 128 * GIB;
static uint64_t sim_free = 40 * GIB;
static uint64_t sim_file = 12 * GIB;
static uint64_t sim_compressions = 100;
static uint64_t sim_swapouts = 5;
static uint64_t sim_own_compressed = 0;
static SystemMemoryPressure sim_pressure = SYSTEM_MEMORY_NORMAL;
static uint64_t sim_alloc_calls = 0;
static uint64_t sim_free_calls = 0;

static SystemMemoryCounters sim_read_counters(void) {
  return (SystemMemoryCounters){.free = sim_free,
                                .file = sim_file,
                                .swapouts = sim_swapouts,
                                .compressions = sim_compressions,
                                .total = sim_total};
}

static SystemMemoryPressure sim_read_pressure(void) { return sim_pressure; }

static uint64_t sim_read_compressed(int peak) {
  (void)peak;
  return sim_own_compressed;
}

// Simulates chunk allocation consuming free (or reclaimable file-cache) pages.
static void *sim_alloc_chunk(size_t bytes) {
  sim_alloc_calls++;
  if (sim_free >= bytes) {
    sim_free -= bytes;
  } else {
    uint64_t rem = bytes - sim_free;
    sim_free = 0;
    sim_file = sim_file >= rem ? sim_file - rem : 0;
  }
  return (void *)(uintptr_t)(0x1000 + sim_alloc_calls * 64);
}

// Simulates chunk release returning physical pages to free memory.
static void sim_free_chunk(void *ptr, size_t bytes) {
  (void)ptr;
  sim_free_calls++;
  sim_free += bytes;
}

int main(void) {
  // Run full deterministic lifecycle in one clean sequence with exact numbers.
  sim_total = 128 * GIB;
  sim_free = 50 * GIB;
  sim_file = 12 * GIB;
  sim_compressions = 100;
  sim_swapouts = 5;
  sim_own_compressed = 0;
  sim_pressure = SYSTEM_MEMORY_NORMAL;
  sim_alloc_calls = 0;
  sim_free_calls = 0;

  PlatformMemoryMacosConfig cfg = {.enable_occupy = 1,
                                   .chunk_bytes = CHUNK,
                                   .hysteresis_bytes = HYSTERESIS,
                                   .read_counters = sim_read_counters,
                                   .read_pressure = sim_read_pressure,
                                   .read_compressed = sim_read_compressed,
                                   .alloc_chunk = sim_alloc_chunk,
                                   .free_chunk = sim_free_chunk};
  PlatformMemory *pm = platform_memory_new_macos(&cfg);
  Memory *mem = memory_new();
  memory_set_limit(mem, 32 * GIB);
  memory_set_reserve(mem, 10 * GIB, NULL, NULL);
  memory_set_policy(mem, pm);

  MemoryAccount acc;
  memory_account_init(mem, &acc, "model", 0);
  MemorySlot model_slot, state_slot;
  memory_slot_init(&model_slot, &acc, MEMORY_MODEL, NULL, "weights", NULL,
                   NULL);
  memory_slot_set_role(&model_slot, MEMORY_ROLE_WEIGHTS);
  memory_slot_init(&state_slot, &acc, MEMORY_STATE, NULL, "states", NULL, NULL);

  uint64_t avail = 0;
  memory_read_system(mem, &avail);
  check(memory_kept_file_cache(mem) == 12 * GIB,
        "pre-existing file cache sets initial floor");
  check(avail == 50 * GIB, "available excludes pre-existing file cache");

  // File cache grows by 8 GiB into free pages and unrelated compressions rise.
  sim_free -= 8 * GIB;
  sim_file += 8 * GIB;
  sim_compressions += 1000;
  memory_read_system(mem, &avail);
  check(memory_kept_file_cache(mem) == 12 * GIB,
        "unrelated compressions do not raise file-cache floor");
  check(avail == 50 * GIB, "file cache above floor counts as available");

  // Model takes 16 GiB (sim_free 42 -> 26 GiB, available = 34 GiB).
  check(memory_claim(&model_slot, 16 * GIB, MEMORY_REQUIRED),
        "model claims 16 GiB");
  sim_free -= 16 * GIB;

  // Limit is 32 GiB -> limit_room = 16 GiB; headroom above 10.25 GiB = 23.75
  // GiB -> guard pool occupies exactly 16 GiB.
  uint64_t occ = memory_maintain(mem, 0);
  check(occ == 16 * GIB, "guard pool respects memory limit (16 GiB)");
  check(sim_file >= 12 * GIB, "pre-existing file cache is respected");

  // State claims 4 GiB: guard pool yields 4 GiB synchronously without eviction.
  check(memory_claim(&state_slot, 4 * GIB, MEMORY_REQUIRED),
        "state claims 4 GiB by taking from guard pool");
  check(memory_occupied(mem) == 12 * GIB,
        "guard pool synchronously released 4 GiB on claim");
  sim_free -= 4 * GIB;

  // State frees 4 GiB -> maintain immediately re-occupies those 4 GiB.
  memory_release(&state_slot, 4 * GIB);
  sim_free += 4 * GIB;
  occ = memory_maintain(mem, 0);
  check(occ == 16 * GIB,
        "guard pool immediately re-occupies memory after release");

  // Another app consumes 10 GiB, pushing external available below 10 GiB
  // reserve (external available was 18 GiB -> 8 GiB, shortfall 2 GiB).
  sim_free -= 10 * GIB;
  uint64_t yielded = memory_yield_if_pressured(mem);
  check(yielded == 2 * GIB && memory_occupied(mem) == 14 * GIB,
        "guard pool yields exact reserve shortfall when another app grows");

  // OS warning pressure drops all remaining guard memory immediately.
  sim_pressure = SYSTEM_MEMORY_WARNING;
  yielded = memory_yield_if_pressured(mem);
  check(yielded == 14 * GIB && memory_occupied(mem) == 0,
        "warning pressure sheds all occupied guard memory");

  // Pressure clears and pre-existing file cache drains from 12 GiB to 4 GiB.
  sim_pressure = SYSTEM_MEMORY_NORMAL;
  sim_file = 4 * GIB;
  memory_read_system(mem, &avail);
  check(memory_kept_file_cache(mem) == 4 * GIB,
        "drained file cache lowers the floor permanently");

  // Process-local compression growth sheds guard pool.
  sim_free = 30 * GIB;
  occ = memory_maintain(mem, 0);
  check(occ == 16 * GIB, "guard pool refills when pressure clears");
  sim_own_compressed = 64 * MIB;
  memory_maintain(mem, 0);
  sim_own_compressed = 128 * MIB;
  occ = memory_maintain(mem, 0);
  check(occ == 0, "process-local compression growth sheds guard pool");

  platform_memory_free(pm);
  free(mem);

  // Real macOS anonymous mmap + PRNG chunk check (128 MiB): fast and
  // incompressible (high byte variety, no all-zero blocks).
  PlatformMemoryMacosConfig real_cfg = {
      .enable_occupy = 1, .chunk_bytes = CHUNK, .max_occupy_bytes = 2 * CHUNK};
  PlatformMemory *real_pm = platform_memory_new_macos(&real_cfg);
  double t0 = now();
  uint64_t real_occ = platform_memory_set_occupied(real_pm, 2 * CHUNK, 0);
  double elapsed = now() - t0;
  check(real_occ == 2 * CHUNK,
        "real macOS policy allocates and faults 128 MiB guard pool");
  uint64_t real_freed = platform_memory_release_occupied(real_pm, 2 * CHUNK);
  check(real_freed == 2 * CHUNK && platform_memory_occupied_bytes(real_pm) == 0,
        "real macOS policy releases all guard chunks cleanly");
  printf("real guard pool 128 MiB faulted in %.2f ms\n", elapsed * 1e3);
  platform_memory_free(real_pm);
  return failures != 0;
}
