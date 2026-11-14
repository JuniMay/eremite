#ifndef EREMITE_ENGINE_MEMORY_H
#define EREMITE_ENGINE_MEMORY_H
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

// Tracks how much memory each model uses. Keeps the total under an optional
// limit, and keeps a reserve free for other programs.
//
// Each model has one account. Its memory is split into slots: weights, cached
// experts, workspaces and state. When memory runs short, unused slots are
// freed. This is called evicting.
//
// State slots hold the model's requests' and conversations' state, such as KV
// caches. They count toward the limit, and each account also has a limit for
// its state. When an account's state reaches it, its idle state slots are
// evicted, oldest first. Otherwise state is evicted only once the rest of its
// model is gone, so that loading experts never evicts state.

// Holds the accounts, the limit, the reserve and the counts.
typedef struct Memory Memory;
typedef struct MemoryAccount MemoryAccount;
typedef struct MemorySlot MemorySlot;

// What adding memory may evict when memory runs short. If evicting cannot make
// room, the addition is refused and nothing is evicted.
//
// Only idle slots are evicted. A slot is idle when:
// - it holds memory and has an evict function,
// - it is not pinned, and memory is not being added to it,
// - its account is not raising its minimum,
// - its can_evict function, if any, allows it,
// - every slot that depends on it is idle. Those are evicted first.
// - if it is a state slot, the rest of its account holds no memory.
typedef enum {
  // For needed memory. May evict any idle slot.
  MEMORY_REQUIRED,
  // For growing a cache. May evict idle slots of other accounts without a
  // minimum, except slots that depend on nothing and were used within the
  // last keep_recently_used_weights seconds.
  MEMORY_CACHE,
  // For useful but optional memory. Evicts nothing, and is refused unless its
  // own size stays free above the reserve.
  MEMORY_SPECULATIVE,
  // For a workspace that can shrink. May evict only slots that depend on
  // another slot.
  MEMORY_WORKSPACE
} MemoryAdmission;

// What a slot holds.
typedef enum {
  // Weights, cached experts or a workspace.
  MEMORY_MODEL,
  // A request's or a conversation's state, such as a KV cache. It counts just
  // its bytes in use.
  MEMORY_STATE
} MemoryKind;

struct MemoryAccount {
  // The Memory that holds the account.
  Memory *memory;
  // Next account in the list.
  MemoryAccount *next;
  const char *name;
  // Bytes counted for its slots, and the least the account counts.
  uint64_t bytes, minimum;
  // Set while it raises its minimum. Its slots are then not evicted.
  int admitting;
  // Bytes counted for its state slots, and the most they may hold.
  uint64_t state_bytes, state_limit;
};

struct MemorySlot {
  // Next slot in the list.
  MemorySlot *next;
  MemoryAccount *account;
  MemoryKind kind;
  // This slot is evicted before its dependency.
  MemorySlot *dependency;
  // First slot that depends on this one, and next slot with the same
  // dependency.
  MemorySlot *children, *sibling;
  const char *name;
  // Bytes counted for the slot, and bytes in use.
  uint64_t bytes, payload;
  // Nonzero while pinned.
  unsigned pins;
  // Time of the last use.
  double touched;
  // When set, bytes shrinks with payload.
  int automatic;
  // Whether the contents are ready. Any thread may read it.
  atomic_int ready;
  // Whether the slot may be evicted now. Null means always.
  int (*can_evict)(void *context);
  // Frees the contents. Null means the slot is never evicted.
  void (*evict)(void *context);
  void *context;
};

typedef struct {
  // The limit, bytes in use, bytes counted, and the most ever counted.
  uint64_t limit, allocated, committed, peak;
  // Slots evicted, and additions refused.
  uint64_t evictions, refusals;
} MemoryUsage;

// How the memory follows the system's counts, and how long it keeps weights
// that were used. A field of 0 takes its default.
typedef struct {
  // Seconds that freed memory takes to show in the system's memory counts: no
  // new reading is taken within this time of a release (default 1.5).
  double freed_memory_delay;
  // Seconds between two readings of the system's available memory; a request
  // of 1 GiB or more reads at once (default 1).
  double available_memory_reading_interval;
  // Seconds after which a reading is taken again, even within
  // freed_memory_delay of a release (default 5).
  double available_memory_reading_max_age;
  // Seconds that used weights count as recently used: a growing cache does not
  // evict them, and the eviction order ages weights on the same scale (default
  // 30).
  double keep_recently_used_weights;
} MemoryTimes;

// The times that a field of 0 takes.
static constexpr MemoryTimes MEMORY_DEFAULT_TIMES = {
    .freed_memory_delay = 1.5,
    .available_memory_reading_interval = 1,
    .available_memory_reading_max_age = 5,
    .keep_recently_used_weights = 30};

// Returns a new Memory with no limit and no reserve.
Memory *memory_new(void);

// Sets the times. A field of 0 takes its default.
void memory_set_times(Memory *memory, MemoryTimes times);

// Returns the times in use, with the defaults filled in.
MemoryTimes memory_times(const Memory *memory);

// Sets the limit in bytes. There is none at first.
void memory_set_limit(Memory *memory, uint64_t bytes);

// Sets the reserve to keep free for other programs. available() reads the
// available memory. unwritten(), or null, reads the bytes counted but not yet
// written, which count as used.
void memory_set_reserve(Memory *memory, uint64_t reserve,
                        uint64_t (*available)(void),
                        uint64_t (*unwritten)(void));

// The most memory the models may use now, within the limit and the reserve.
uint64_t memory_current_limit(Memory *memory);

// Starts an empty account and adds it to memory. Its state slots may hold up
// to state_limit bytes.
void memory_account_init(Memory *memory, MemoryAccount *account,
                         const char *name, uint64_t state_limit);

// Starts an empty slot. dependency may be null. evict(context) frees its
// contents; null means the slot is never evicted. Ends the program if
// dependency belongs to another Memory.
void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemoryKind kind,
                      MemorySlot *dependency, const char *name,
                      void (*evict)(void *), void *context);

// Removes the slot and its bytes.
void memory_slot_remove(MemorySlot *slot);

// Touches the slot: marks it as used now. Recently used slots are evicted
// last.
void memory_touch(MemorySlot *slot);

// Pins or unpins the slot. Pinned slots are never evicted.
void memory_pin(MemorySlot *slot, int pinned);

// Whether the slot's contents are ready.
int memory_ready(const MemorySlot *slot);

// Sets whether the slot's contents are ready.
void memory_set_ready(MemorySlot *slot, int ready);

// Admits the slots: sets each slot's bytes to its size, evicting as mode
// allows. Returns whether all fit. If not, nothing changes. Ends the program if
// a slot is a state slot, or if the slots belong to different Memory objects.
int memory_admit(MemorySlot *const *slots, const uint64_t *bytes, size_t count,
                 MemoryAdmission mode);

// Sets the account's minimum, evicting as mode allows. Returns whether it
// fits.
int memory_set_minimum(MemoryAccount *account, uint64_t bytes,
                       MemoryAdmission mode);

// Claims bytes for the slot: adds them to its bytes in use, evicting as mode
// allows. Returns whether they fit. A state slot first stays within its
// account's state limit: the account's idle state slots are evicted, oldest
// first, and the bytes are refused if that is not enough.
int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode);

// Releases bytes: removes them from the slot's bytes in use.
void memory_release(MemorySlot *slot, uint64_t bytes);

// Shrinks the slot's bytes to its bytes in use.
void memory_shrink(MemorySlot *slot);

// Returns the counts.
MemoryUsage memory_usage(const Memory *memory);

// Sets a function that runs after slots are evicted, before the call that
// evicted them returns. Use it to free memory that evict functions had to leave
// for later, such as GPU buffers that unfinished GPU work still reads.
void memory_set_eviction_callback(Memory *memory, void (*callback)(void));

// Returns the shared slot: the slot for memory that the engine itself uses and
// no model owns.
// That memory counts toward the limit. The slot is never evicted, and its
// memory stops being counted as soon as it is freed.
MemorySlot *memory_shared(Memory *memory);

// Returns the account's room: the bytes it may still add within the limit.
uint64_t memory_room(const MemoryAccount *account);

// Reclaims memory: evicts idle slots until bytes are freed or none is left.
// Returns the bytes freed.
uint64_t memory_reclaim(Memory *memory, uint64_t bytes);

// Unloads the account: evicts its idle slots, its state slots last. Returns the
// bytes freed.
uint64_t memory_unload(MemoryAccount *account);

// Returns zeroed host memory of bytes, counted in slot. Null if it does not
// fit. Ends the program if slot is null.
void *memory_host_alloc(MemorySlot *slot, size_t bytes);

// Returns zeroed host memory of bytes, counted in slot. Ends the program if
// slot is null or the memory does not fit.
void *memory_host_alloc_required(MemorySlot *slot, size_t bytes);

// Frees host memory counted in slot, also memory from plain malloc. Ends the
// program if slot is null or the memory was counted in another slot.
void memory_host_free(MemorySlot *slot, void *pointer);

#endif
