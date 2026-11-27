#include "platform/memory_policy.h"
#include "base/error.h"
#include <stdlib.h>
#include <sys/mman.h>

// macOS unified-memory policy backend:
// - Respects pre-existing OS file cache via a strictly non-increasing floor
//   (file_cache_floor). Any file cache growing above that floor counts as
//   available memory.
// - Optionally holds released and idle headroom above reserve in anonymous
//   mmap chunks filled with incompressible pseudo-random bytes so macOS's VM
//   compressor cannot collapse them back into file cache, while staying out of
//   Metal residency sets so GPU working-set budgets stay untouched.

// Initial array capacity for guard-pool chunk pointers.
static constexpr size_t INITIAL_CHUNK_CAP = 32;
// SplitMix64 / XorShift64* constants used to generate incompressible filler.
static constexpr uint64_t PRNG_SEED_INIT = 0x9E3779B97F4A7C15ull;
static constexpr uint64_t PRNG_STEP_MULT = 0x2545F4914F6CDD1Dull;
static constexpr unsigned PRNG_SHIFT_A = 12;
static constexpr unsigned PRNG_SHIFT_B = 25;
static constexpr unsigned PRNG_SHIFT_C = 27;

// Private context for the macOS unified-memory policy backend.
typedef struct {
  // Configuration parameters and injectable test hooks.
  PlatformMemoryMacosConfig cfg;
  // Lowest file cache observed so far (UINT64_MAX before first reading).
  uint64_t file_cache_floor;
  // Array of active guard-pool chunk pointers.
  void **chunks;
  // Count and capacity of `chunks`.
  size_t nchunks, cap_chunks;
  // Running PRNG state for generating distinct incompressible pages.
  uint64_t prng;
} MacosPolicy;

// Allocates one anonymous page-aligned chunk and faults every page with
// incompressible pseudo-random bytes so the macOS VM compressor cannot collapse
// it to near-zero bytes.
static void *macos_alloc_chunk(MacosPolicy *p, size_t bytes) {
  if (p->cfg.alloc_chunk) return p->cfg.alloc_chunk(bytes);
  void *mem =
      mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (mem == MAP_FAILED) return NULL;
  uint64_t s0 = p->prng ^ 0xA0761D6478BD642Full;
  uint64_t s1 = (p->prng * PRNG_STEP_MULT) ^ 0xE7037ED1A0B428DBull;
  uint64_t s2 = (s1 * PRNG_STEP_MULT) ^ 0x8EBC6AF09C88C6E3ull;
  uint64_t s3 = (s2 * PRNG_STEP_MULT) ^ 0x589965CC75374CC3ull;
  uint64_t *w = (uint64_t *)mem;
  size_t words = bytes / sizeof(uint64_t);
  size_t i = 0;
  for (; i + 4 <= words; i += 4) {
    s0 ^= s0 >> PRNG_SHIFT_A;
    s0 ^= s0 << PRNG_SHIFT_B;
    s0 ^= s0 >> PRNG_SHIFT_C;
    s1 ^= s1 >> PRNG_SHIFT_A;
    s1 ^= s1 << PRNG_SHIFT_B;
    s1 ^= s1 >> PRNG_SHIFT_C;
    s2 ^= s2 >> PRNG_SHIFT_A;
    s2 ^= s2 << PRNG_SHIFT_B;
    s2 ^= s2 >> PRNG_SHIFT_C;
    s3 ^= s3 >> PRNG_SHIFT_A;
    s3 ^= s3 << PRNG_SHIFT_B;
    s3 ^= s3 >> PRNG_SHIFT_C;
    w[i] = s0 * PRNG_STEP_MULT;
    w[i + 1] = s1 * PRNG_STEP_MULT;
    w[i + 2] = s2 * PRNG_STEP_MULT;
    w[i + 3] = s3 * PRNG_STEP_MULT;
  }
  for (; i < words; i++) {
    s0 ^= s0 >> PRNG_SHIFT_A;
    s0 ^= s0 << PRNG_SHIFT_B;
    s0 ^= s0 >> PRNG_SHIFT_C;
    w[i] = s0 * PRNG_STEP_MULT;
  }
  p->prng = s0 ^ s1 ^ s2 ^ s3;
  if (!p->prng) p->prng = PRNG_SEED_INIT;
  return mem;
}

// Frees one guard-pool chunk allocated by macos_alloc_chunk.
static void macos_free_chunk(MacosPolicy *p, void *ptr, size_t bytes) {
  if (!ptr) return;
  if (p->cfg.free_chunk) {
    p->cfg.free_chunk(ptr, bytes);
    return;
  }
  munmap(ptr, bytes);
}

// Returns bytes currently held in `p`'s guard pool.
static uint64_t macos_occupied_bytes(void *context) {
  const MacosPolicy *p = context;
  return (uint64_t)p->nchunks * (uint64_t)p->cfg.chunk_bytes;
}

// Reads counters, updates the strictly non-increasing file-cache floor, and
// returns the unified-memory availability state.
static PlatformMemoryState macos_read(void *context) {
  MacosPolicy *p = context;
  SystemMemoryCounters c = p->cfg.read_counters ? p->cfg.read_counters()
                                                : system_memory_read_counters();
  if (c.file < p->file_cache_floor) p->file_cache_floor = c.file;
  uint64_t kept = p->file_cache_floor == UINT64_MAX ? 0 : p->file_cache_floor;
  uint64_t reclaimable_file = c.file > kept ? c.file - kept : 0;
  SystemMemoryPressure pressure =
      p->cfg.read_pressure ? p->cfg.read_pressure() : system_memory_pressure();
  uint64_t compressed = p->cfg.read_compressed ? p->cfg.read_compressed(0)
                                               : system_memory_compressed(0);
  return (PlatformMemoryState){.total = c.total,
                               .free = c.free,
                               .file_cache = c.file,
                               .file_cache_kept = kept,
                               .available = c.free + reclaimable_file,
                               .occupied = macos_occupied_bytes(p),
                               .process_compressed = compressed,
                               .swapouts = c.swapouts,
                               .compressions = c.compressions,
                               .pressure = pressure};
}

// Synchronously releases at least `bytes` from the guard pool.
static uint64_t macos_release_occupied(void *context, uint64_t bytes) {
  MacosPolicy *p = context;
  if (!bytes || !p->nchunks) return 0;
  size_t chunk = p->cfg.chunk_bytes;
  size_t drop = (size_t)((bytes + chunk - 1) / chunk);
  if (drop > p->nchunks) drop = p->nchunks;
  uint64_t freed = 0;
  for (size_t i = 0; i < drop; i++) {
    void *ptr = p->chunks[--p->nchunks];
    macos_free_chunk(p, ptr, chunk);
    freed += chunk;
  }
  return freed;
}

// Grows or shrinks the guard pool toward `target_bytes`, growing by at most
// `max_grow_bytes` (0: unbounded).
static uint64_t macos_set_occupied(void *context, uint64_t target_bytes,
                                   uint64_t max_grow_bytes) {
  MacosPolicy *p = context;
  if (!p->cfg.enable_occupy) target_bytes = 0;
  if (p->cfg.max_occupy_bytes && target_bytes > p->cfg.max_occupy_bytes)
    target_bytes = p->cfg.max_occupy_bytes;
  size_t chunk = p->cfg.chunk_bytes;
  size_t target_chunks = (size_t)(target_bytes / chunk);

  if (target_chunks < p->nchunks) {
    while (p->nchunks > target_chunks) {
      void *ptr = p->chunks[--p->nchunks];
      macos_free_chunk(p, ptr, chunk);
    }
    return macos_occupied_bytes(p);
  }

  size_t add = target_chunks - p->nchunks;
  if (max_grow_bytes > 0) {
    size_t max_add = (size_t)(max_grow_bytes / chunk);
    if (max_add == 0 && target_chunks > p->nchunks) max_add = 1;
    if (add > max_add) add = max_add;
  }

  for (size_t i = 0; i < add; i++) {
    if (p->nchunks == p->cap_chunks) {
      size_t next_cap = p->cap_chunks ? p->cap_chunks * 2 : INITIAL_CHUNK_CAP;
      void **next = realloc(p->chunks, next_cap * sizeof *next);
      if (!next) break;
      p->chunks = next;
      p->cap_chunks = next_cap;
    }
    void *mem = macos_alloc_chunk(p, chunk);
    if (!mem) break;
    p->chunks[p->nchunks++] = mem;
  }
  return macos_occupied_bytes(p);
}

// Destroys the macOS backend context and releases all held guard chunks.
static void macos_destroy(void *context) {
  MacosPolicy *p = context;
  if (!p) return;
  macos_set_occupied(p, 0, 0);
  free(p->chunks);
  free(p);
}

// Operations table for the macOS unified-memory policy backend.
static const PlatformMemoryOps MACOS_OPS = {
    .read = macos_read,
    .set_occupied = macos_set_occupied,
    .release_occupied = macos_release_occupied,
    .occupied_bytes = macos_occupied_bytes,
    .destroy = macos_destroy};

PlatformMemory *
platform_memory_new_macos(const PlatformMemoryMacosConfig *config) {
  MacosPolicy *p = calloc(1, sizeof *p);
  if (!p) die("out of memory for macOS memory policy");
  if (config) p->cfg = *config;
  if (!p->cfg.chunk_bytes)
    p->cfg.chunk_bytes = PLATFORM_MEMORY_DEFAULT_CHUNK_BYTES;
  if (!p->cfg.hysteresis_bytes)
    p->cfg.hysteresis_bytes = PLATFORM_MEMORY_DEFAULT_HYSTERESIS_BYTES;
  p->file_cache_floor = UINT64_MAX;
  p->prng = PRNG_SEED_INIT;

  PlatformMemory *pm = calloc(1, sizeof *pm);
  if (!pm) {
    free(p);
    die("out of memory for platform memory handle");
  }
  pm->ops = &MACOS_OPS;
  pm->context = p;
  return pm;
}

void platform_memory_free(PlatformMemory *pm) {
  if (!pm) return;
  if (pm->ops && pm->ops->destroy) pm->ops->destroy(pm->context);
  free(pm);
}

PlatformMemoryState platform_memory_read(PlatformMemory *pm) {
  if (!pm || !pm->ops || !pm->ops->read) return (PlatformMemoryState){0};
  return pm->ops->read(pm->context);
}

uint64_t platform_memory_set_occupied(PlatformMemory *pm, uint64_t target_bytes,
                                      uint64_t max_grow_bytes) {
  if (!pm || !pm->ops || !pm->ops->set_occupied) return 0;
  return pm->ops->set_occupied(pm->context, target_bytes, max_grow_bytes);
}

uint64_t platform_memory_release_occupied(PlatformMemory *pm, uint64_t bytes) {
  if (!pm || !pm->ops || !pm->ops->release_occupied) return 0;
  return pm->ops->release_occupied(pm->context, bytes);
}

uint64_t platform_memory_occupied_bytes(const PlatformMemory *pm) {
  if (!pm || !pm->ops || !pm->ops->occupied_bytes) return 0;
  return pm->ops->occupied_bytes(pm->context);
}

uint64_t platform_memory_hysteresis_bytes(const PlatformMemory *pm) {
  if (!pm || pm->ops != &MACOS_OPS || !pm->context) return 0;
  const MacosPolicy *p = pm->context;
  return p->cfg.hysteresis_bytes;
}

int platform_memory_occupy_enabled(const PlatformMemory *pm) {
  if (!pm || pm->ops != &MACOS_OPS || !pm->context) return 0;
  const MacosPolicy *p = pm->context;
  return p->cfg.enable_occupy != 0;
}
