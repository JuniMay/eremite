#ifndef EREMITE_ENGINE_MEMORY_H
#define EREMITE_ENGINE_MEMORY_H
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

// Tracks how much memory each model uses. Keeps the total under an optional
// limit, and keeps a reserve free for other programs.
//
// Each model has one account. Its memory is split into slots: weights, cached
// experts and workspaces. When memory runs short, unused slots are freed. This
// is called evicting.
//
// State memory holds a request's or a conversation's state. It is counted
// apart from model memory and comes first.

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
typedef enum {
  // For needed memory. May evict any idle slot.
  MEMORY_REQUIRED,
  // For growing a cache. May evict idle slots of other accounts without a
  // minimum, except slots that depend on nothing and were used in the last 30
  // seconds.
  MEMORY_CACHE,
  // For useful but optional memory. Evicts nothing, and is refused unless its
  // own size stays free above the reserve.
  MEMORY_SPECULATIVE,
  // For a workspace that can shrink. May evict only slots that depend on
  // another slot.
  MEMORY_WORKSPACE
} MemoryAdmission;

struct MemoryAccount {
  // Next account in the list.
  MemoryAccount *next;
  const char *name;
  // Bytes counted for its slots, and the least the account counts.
  uint64_t bytes, minimum;
  // Set while it raises its minimum. Its slots are then not evicted.
  int admitting;
};

struct MemorySlot {
  // Next slot in the list.
  MemorySlot *next;
  MemoryAccount *account;
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
  // The limit, bytes in use, bytes counted, the most ever counted, and state
  // memory.
  uint64_t limit, allocated, committed, peak, state;
  // Slots evicted, and additions refused.
  uint64_t evictions, refusals;
} MemoryUsage;

// Sets the limit in bytes. There is none at first.
void memory_set_limit(uint64_t bytes);

// Sets the reserve to keep free for other programs. available() reads the
// available memory. unwritten(), or null, reads the bytes counted but not yet
// written, which count as used.
void memory_set_reserve(uint64_t reserve, uint64_t (*available)(void),
                        uint64_t (*unwritten)(void));

// The most memory the models may use now, within the limit and the reserve.
uint64_t memory_current_limit(void);

// Starts an empty account.
void memory_account_init(MemoryAccount *account, const char *name);

// Starts an empty slot. dependency may be null. evict(context) frees its
// contents; null means the slot is never evicted.
void memory_slot_init(MemorySlot *slot, MemoryAccount *account,
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
// allows. Returns whether all fit. If not, nothing changes.
int memory_admit(MemorySlot *const *slots, const uint64_t *bytes, size_t count,
                 MemoryAdmission mode);

// Sets the account's minimum, evicting as mode allows. Returns whether it
// fits.
int memory_set_minimum(MemoryAccount *account, uint64_t bytes,
                       MemoryAdmission mode);

// Claims bytes for the slot: adds them to its bytes in use, evicting as mode
// allows. Returns whether they fit.
int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode);

// Releases bytes: removes them from the slot's bytes in use.
void memory_release(MemorySlot *slot, uint64_t bytes);

// Shrinks the slot's bytes to its bytes in use.
void memory_shrink(MemorySlot *slot);

// Adds bytes to state memory when allocated is set, or removes them. Adding
// first evicts idle slots to keep the reserve free.
void memory_state_change(uint64_t bytes, int allocated);

// Returns the counts.
MemoryUsage memory_usage(void);

// Sets a function that runs after evictions.
void memory_collect_with(void (*collect)(void));

// The slot that counts this thread's host memory. Null counts nothing. Each
// thread has its own.
extern thread_local MemorySlot *memory_scope_slot;

// Sets this thread's memory_scope_slot. Returns the old one.
MemorySlot *memory_scope(MemorySlot *slot);

// Returns the shared slot: the one slot for memory that no model owns. Its
// memory counts toward the limit. The slot has no evict function, so it is
// never evicted, and its memory stops being counted as soon as it is released.
MemorySlot *memory_shared(void);

// Returns the account's room: the bytes it may still add within the limit.
uint64_t memory_room(const MemoryAccount *account);

// Reclaims memory: evicts idle slots until bytes are freed or none is left.
// Returns the bytes freed.
uint64_t memory_reclaim(uint64_t bytes);

// Unloads the account: evicts its idle slots. Returns the bytes freed.
uint64_t memory_unload(MemoryAccount *account);

// Returns zeroed host memory, counted by memory_scope_slot, or as state memory
// when state is set. Null if it does not fit.
void *memory_host_alloc(size_t bytes, int state);

// Returns zeroed host memory, counted by memory_scope_slot, or as state memory
// when state is set. Ends the program if it does not fit.
void *memory_host_required(size_t bytes, int state);

// Frees host memory, also memory from plain malloc.
void memory_host_free(void *pointer);

#endif
