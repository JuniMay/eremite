#ifndef EREMITE_PLATFORM_MEMORY_POLICY_H
#define EREMITE_PLATFORM_MEMORY_POLICY_H
#include "platform/system_memory.h"
#include <stddef.h>
#include <stdint.h>

// Layered platform memory policy separating OS- and architecture-specific
// memory behavior (such as macOS unified-memory file cache tracking and
// releasable guard-pool occupation) from the engine's hardware-agnostic Memory
// governor. Future discrete GPU or multi-tier backends implement
// PlatformMemoryOps without changing core engine accounting.

// One observation from the platform memory backend, in bytes unless noted.
typedef struct {
  // Total physical memory managed by this tier; 0 when unknown.
  uint64_t total;
  // Immediately free memory reported by the OS or device.
  uint64_t free;
  // Total OS file cache on unified-memory platforms; 0 on discrete tiers.
  uint64_t file_cache;
  // Pre-existing OS file cache respected until drained or purged; 0 on
  // discrete tiers.
  uint64_t file_cache_kept;
  // External bytes available to this process (excluding the guard pool).
  uint64_t available;
  // Bytes currently held by this backend's releasable guard pool.
  uint64_t occupied;
  // This process's own memory currently stored in an OS compressor; 0 when
  // inapplicable.
  uint64_t process_compressed;
  // Cumulative OS swap-out and compression page counts (informational).
  uint64_t swapouts, compressions;
  // Current OS memory pressure level.
  SystemMemoryPressure pressure;
} PlatformMemoryState;

// Virtual table implemented by each platform memory backend.
typedef struct {
  // Reads the current platform memory state.
  PlatformMemoryState (*read)(void *context);
  // Adjusts the releasable guard pool toward `target_bytes`, adding at most
  // `max_grow_bytes` in one call (0: unbounded). Returns occupied bytes after
  // adjustment. May be null on backends without a guard pool.
  uint64_t (*set_occupied)(void *context, uint64_t target_bytes,
                           uint64_t max_grow_bytes);
  // Synchronously releases at least `bytes` from the guard pool and returns
  // the bytes actually released. May be null on backends without a guard pool.
  uint64_t (*release_occupied)(void *context, uint64_t bytes);
  // Returns the bytes currently held by the guard pool. May be null.
  uint64_t (*occupied_bytes)(void *context);
  // Releases all guard memory and destroys backend state.
  void (*destroy)(void *context);
} PlatformMemoryOps;

// Polymorphic handle to a platform memory policy instance.
typedef struct {
  // Backend operations table.
  const PlatformMemoryOps *ops;
  // Backend-owned private context.
  void *context;
} PlatformMemory;

// Configurable knobs for the macOS unified-memory policy backend. Zero fields
// take documented defaults.
typedef struct {
  // Nonzero to enable the releasable incompressible guard pool that occupies
  // released and idle headroom above reserve so macOS file cache cannot grow
  // into it.
  int enable_occupy;
  // Granularity of each guard-pool allocation chunk in bytes (default 64 MiB).
  size_t chunk_bytes;
  // Safety band left free above reserve before growing the guard pool, in
  // bytes (default 256 MiB).
  uint64_t hysteresis_bytes;
  // Upper bound on guard-pool size in bytes; 0 leaves it bounded only by
  // available memory, reserve and engine limit.
  uint64_t max_occupy_bytes;
  // Optional counter reader override for deterministic testing; null uses
  // system_memory_read_counters.
  SystemMemoryCounters (*read_counters)(void);
  // Optional pressure reader override for testing; null uses
  // system_memory_pressure.
  SystemMemoryPressure (*read_pressure)(void);
  // Optional process-compressed reader override for testing; null uses
  // system_memory_compressed.
  uint64_t (*read_compressed)(int peak);
  // Optional chunk allocator override for testing; null uses mmap and fills
  // with incompressible pseudo-random data.
  void *(*alloc_chunk)(size_t bytes);
  // Optional chunk freer override paired with alloc_chunk; null uses munmap.
  void (*free_chunk)(void *ptr, size_t bytes);
} PlatformMemoryMacosConfig;

// Default chunk size (64 MiB) for macOS guard-pool allocations.
static constexpr size_t PLATFORM_MEMORY_DEFAULT_CHUNK_BYTES = 64ull << 20;
// Default free headroom band (256 MiB) left above reserve before occupying.
static constexpr uint64_t PLATFORM_MEMORY_DEFAULT_HYSTERESIS_BYTES = 256ull
                                                                     << 20;

// Creates a macOS unified-memory policy with optional configuration `config`
// (null uses defaults with occupation disabled).
PlatformMemory *
platform_memory_new_macos(const PlatformMemoryMacosConfig *config);

// Destroys `pm` and releases any guard memory it holds. Null is a no-op.
void platform_memory_free(PlatformMemory *pm);

// Reads the current platform memory state from `pm`.
PlatformMemoryState platform_memory_read(PlatformMemory *pm);

// Adjusts `pm`'s guard pool toward `target_bytes`, growing by at most
// `max_grow_bytes` (0: unbounded). Returns the occupied bytes afterward.
uint64_t platform_memory_set_occupied(PlatformMemory *pm, uint64_t target_bytes,
                                      uint64_t max_grow_bytes);

// Synchronously releases at least `bytes` from `pm`'s guard pool and returns
// the bytes released.
uint64_t platform_memory_release_occupied(PlatformMemory *pm, uint64_t bytes);

// Returns the bytes currently held by `pm`'s guard pool.
uint64_t platform_memory_occupied_bytes(const PlatformMemory *pm);

// Returns the configured hysteresis band in bytes for `pm` (0 if none).
uint64_t platform_memory_hysteresis_bytes(const PlatformMemory *pm);

// Returns whether `pm` actively occupies released/idle headroom above reserve.
int platform_memory_occupy_enabled(const PlatformMemory *pm);

#endif
