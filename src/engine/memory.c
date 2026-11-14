#include "engine/memory.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// A record of one piece of host memory.
typedef struct HostAllocation {
  // Next record in the list.
  struct HostAllocation *next;
  // The memory, the slot that counts it, and its size in bytes.
  void *pointer;
  MemorySlot *slot;
  size_t bytes;
} HostAllocation;

struct Memory {
  // The lists of all accounts and all slots.
  MemoryAccount *accounts;
  MemorySlot *slots;
  // The limit and the counts.
  MemoryUsage usage;
  // The function that runs after slots are evicted, or null.
  void (*eviction_callback)(void);
  // The kernel's count of available memory lags behind this program. So each
  // reading is corrected for what this program used since, and bytes counted
  // but not yet written count as used.
  uint64_t (*read_available)(void), (*read_unwritten)(void), reserve;
  // The last reading, what this program used then, and how much of that was
  // not written yet.
  uint64_t read_bytes, read_used, read_pending;
  // Times of the last reading and the last release.
  double read_at, released_at;
  // How the memory follows the system's counts.
  MemoryTimes times;
  // Records of the host memory in use, and the lock that guards them.
  HostAllocation *host_allocations;
  pthread_mutex_t host_lock;
  // The shared account and slot, for memory that the engine itself uses.
  MemoryAccount shared_account;
  MemorySlot shared;
};

// A time before any reading or release.
#define LONG_AGO -1e9

static uint64_t in_use(const Memory *memory) { return memory->usage.allocated; }
#define GIB_F 1073741824.0
// Evictions that free at least this much at once are logged.
#define LOGGED_EVICTION (1ull << 30)

// The available memory. A new reading is taken after
// available_memory_reading_interval, or at once for 1 GiB or more, but not
// within freed_memory_delay of a release unless the last is
// available_memory_reading_max_age old.
static uint64_t available(Memory *memory, uint64_t adding) {
  double t = now();
  const MemoryTimes *times = &memory->times;
  if ((t - memory->released_at >= times->freed_memory_delay ||
       t - memory->read_at >= times->available_memory_reading_max_age) &&
      (adding >= 1ull << 30 ||
       t - memory->read_at >= times->available_memory_reading_interval)) {
    memory->read_bytes = memory->read_available();
    memory->read_used = in_use(memory);
    memory->read_at = t;
    memory->read_pending =
        memory->read_unwritten ? memory->read_unwritten() : 0;
  }
  const MemoryUsage *usage = &memory->usage;
  uint64_t read_bytes = memory->read_bytes, read_used = memory->read_used;
  uint64_t used = in_use(memory),
           promised = usage->committed > usage->allocated
                          ? usage->committed - usage->allocated
                          : 0;
  uint64_t a = used >= read_used ? (read_bytes > used - read_used
                                        ? read_bytes - (used - read_used)
                                        : 0)
                                 : read_bytes + (read_used - used);
  a = a > memory->read_pending ? a - memory->read_pending : 0;
  return a > promised ? a - promised : 0;
}

// How far adding bytes would go below the reserve.
static uint64_t shortage(Memory *memory, uint64_t bytes) {
  if (!memory->read_available || !bytes) return 0;
  uint64_t a = available(memory, bytes),
           room = a > memory->reserve ? a - memory->reserve : 0;
  return bytes > room ? bytes - room : 0;
}

// The bytes counted for the account: its model memory's bytes, or its minimum
// if larger, and its state memory's bytes.
static uint64_t counted(const MemoryAccount *account) {
  uint64_t model = account->bytes - account->state_bytes;
  return (model > account->minimum ? model : account->minimum) +
         account->state_bytes;
}

static void change(MemorySlot *slot, uint64_t bytes) {
  MemoryUsage *usage = &slot->account->memory->usage;
  if (slot->kind == MEMORY_STATE)
    slot->account->state_bytes =
        slot->account->state_bytes - slot->bytes + bytes;
  uint64_t before = counted(slot->account);
  slot->account->bytes -= slot->bytes;
  slot->account->bytes += bytes;
  slot->bytes = bytes;
  usage->committed = usage->committed - before + counted(slot->account);
  if (usage->committed > usage->peak) usage->peak = usage->committed;
}

Memory *memory_new(void) {
  Memory *memory = calloc(1, sizeof *memory);
  if (!memory) die("out of memory (%zu bytes)", sizeof *memory);
  memory->usage.limit = UINT64_MAX;
  memory->read_at = memory->released_at = LONG_AGO;
  memory_set_times(memory, (MemoryTimes){0});
  pthread_mutex_init(&memory->host_lock, NULL);
  memory_account_init(memory, &memory->shared_account, "shared", 0);
  memory_slot_init(&memory->shared, &memory->shared_account, MEMORY_MODEL, NULL,
                   "shared runtime", NULL, NULL);
  memory->shared.automatic = 1;
  return memory;
}

void memory_set_limit(Memory *memory, uint64_t bytes) {
  memory->usage.limit = bytes;
}

void memory_set_times(Memory *memory, MemoryTimes times) {
  const MemoryTimes d = MEMORY_DEFAULT_TIMES;
  memory->times = (MemoryTimes){
      .freed_memory_delay = times.freed_memory_delay > 0
                                ? times.freed_memory_delay
                                : d.freed_memory_delay,
      .available_memory_reading_interval =
          times.available_memory_reading_interval > 0
              ? times.available_memory_reading_interval
              : d.available_memory_reading_interval,
      .available_memory_reading_max_age =
          times.available_memory_reading_max_age > 0
              ? times.available_memory_reading_max_age
              : d.available_memory_reading_max_age,
      .keep_recently_used_weights = times.keep_recently_used_weights > 0
                                        ? times.keep_recently_used_weights
                                        : d.keep_recently_used_weights};
}

MemoryTimes memory_times(const Memory *memory) { return memory->times; }

void memory_set_reserve(Memory *memory, uint64_t bytes,
                        uint64_t (*reader)(void), uint64_t (*unwritten)(void)) {
  memory->reserve = bytes;
  memory->read_available = reader;
  memory->read_unwritten = unwritten;
  memory->read_at = LONG_AGO;  // the next use takes a new reading
}

uint64_t memory_current_limit(Memory *memory) {
  const MemoryUsage *usage = &memory->usage;
  if (!memory->read_available) return usage->limit;
  uint64_t a = available(memory, 0),
           room = a > memory->reserve ? a - memory->reserve : 0;
  uint64_t limit = UINT64_MAX - usage->committed < room
                       ? UINT64_MAX
                       : usage->committed + room;
  return limit < usage->limit ? limit : usage->limit;
}

void memory_account_init(Memory *memory, MemoryAccount *account,
                         const char *name, uint64_t state_limit) {
  *account = (MemoryAccount){.memory = memory,
                             .next = memory->accounts,
                             .name = name,
                             .state_limit = state_limit};
  memory->accounts = account;
}

void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemoryKind kind,
                      MemorySlot *dependency, const char *name,
                      void (*evict)(void *), void *context) {
  Memory *memory = account->memory;
  if (dependency && dependency->account->memory != memory)
    die("%s: depends on a slot of another Memory", name);
  *slot = (MemorySlot){.next = memory->slots,
                       .account = account,
                       .kind = kind,
                       .dependency = dependency,
                       .name = name,
                       .evict = evict,
                       .context = context,
                       .touched = now()};
  atomic_init(&slot->ready, 0);
  if (dependency) {
    slot->sibling = dependency->children;
    dependency->children = slot;
  }
  memory->slots = slot;
}

void memory_slot_remove(MemorySlot *slot) {
  change(slot, 0);
  MemorySlot **at = &slot->account->memory->slots;
  while (*at && *at != slot) at = &(*at)->next;
  if (*at) *at = slot->next;
  if (slot->dependency) {
    at = &slot->dependency->children;
    while (*at && *at != slot) at = &(*at)->sibling;
    if (*at) *at = slot->sibling;
  }
}

void memory_touch(MemorySlot *slot) { slot->touched = now(); }
void memory_pin(MemorySlot *slot, int pinned) {
  slot->pins = !!pinned;
  if (pinned) memory_touch(slot);
}
int memory_ready(const MemorySlot *slot) {
  return atomic_load_explicit(&slot->ready, memory_order_acquire);
}
void memory_set_ready(MemorySlot *slot, int ready) {
  atomic_store_explicit(&slot->ready, !!ready, memory_order_release);
}

static int requested(MemorySlot *slot, MemorySlot *const *wanted, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (wanted[i] == slot) return 1;
  return 0;
}

// Whether the slot is idle for mode. needed is set when memory is needed and
// the other idle slots cannot make the room.
static int evictable(MemorySlot *slot, MemorySlot *const *wanted, size_t n,
                     MemoryAdmission mode, int needed) {
  if (!slot->bytes || !slot->evict || slot->pins || slot->account->admitting ||
      requested(slot, wanted, n) ||
      (slot->can_evict && !slot->can_evict(slot->context, needed)))
    return 0;
  // State goes only after the rest of its model.
  if (slot->kind == MEMORY_STATE &&
      slot->account->bytes > slot->account->state_bytes)
    return 0;
  if (mode == MEMORY_SPECULATIVE) return 0;
  if (mode == MEMORY_WORKSPACE && !slot->dependency) return 0;
  if (mode == MEMORY_CACHE) {
    if (slot->account->minimum) return 0;
    for (size_t i = 0; i < n; i++)
      if (wanted[i]->account == slot->account) return 0;
  }
  // A growing cache keeps weights used within keep_recently_used_weights.
  if (mode == MEMORY_CACHE && !slot->dependency &&
      now() - slot->touched <
          slot->account->memory->times.keep_recently_used_weights)
    return 0;
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes && !evictable(child, wanted, n, mode, needed)) return 0;
  return 1;
}

static int has_children(MemorySlot *slot) {
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes) return 1;
  return 0;
}

// The bytes of the account's slots of kind, with the wanted slots at their
// new sizes. UINT64_MAX when they are more than that.
static uint64_t proposed_bytes(const MemoryAccount *account, MemoryKind kind,
                               MemorySlot *const *wanted, const uint64_t *sizes,
                               size_t n) {
  uint64_t proposed = kind == MEMORY_STATE
                          ? account->state_bytes
                          : account->bytes - account->state_bytes;
  for (size_t i = 0; i < n; i++)
    if (wanted[i]->account == account && wanted[i]->kind == kind) {
      uint64_t bytes =
          sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload;
      if (bytes > wanted[i]->bytes &&
          UINT64_MAX - proposed < bytes - wanted[i]->bytes)
        return UINT64_MAX;
      proposed = proposed - wanted[i]->bytes + bytes;
    }
  return proposed;
}

static uint64_t growth(Memory *memory, MemorySlot *const *wanted,
                       const uint64_t *sizes, size_t n) {
  uint64_t extra = 0;
  for (MemoryAccount *a = memory->accounts; a; a = a->next) {
    uint64_t model = proposed_bytes(a, MEMORY_MODEL, wanted, sizes, n),
             state = proposed_bytes(a, MEMORY_STATE, wanted, sizes, n);
    if (model == UINT64_MAX || state == UINT64_MAX) return UINT64_MAX;
    uint64_t floor = model > a->minimum ? model : a->minimum;
    if (UINT64_MAX - floor < state) return UINT64_MAX;
    uint64_t after = floor + state;
    if (after > counted(a)) {
      if (UINT64_MAX - extra < after - counted(a)) return UINT64_MAX;
      extra += after - counted(a);
    }
  }
  return extra;
}

// The bytes that evicting the account's slots of kind can take off its count:
// its model memory's down to the minimum, and all of its state memory's.
static uint64_t reclaim_room(const MemoryAccount *account, MemoryKind kind,
                             MemorySlot *const *wanted, const uint64_t *sizes,
                             size_t n) {
  if (kind == MEMORY_STATE) return account->state_bytes;
  uint64_t bytes = proposed_bytes(account, MEMORY_MODEL, wanted, sizes, n),
           model = account->bytes - account->state_bytes;
  if (bytes < model) bytes = model;
  return bytes > account->minimum ? bytes - account->minimum : 0;
}

// The bytes that evicting the account's idle slots of kind can take off its
// count.
static uint64_t reclaimable_of(Memory *memory, const MemoryAccount *account,
                               MemoryKind kind, MemorySlot *const *wanted,
                               const uint64_t *sizes, size_t n,
                               MemoryAdmission mode, int needed) {
  uint64_t possible = 0;
  for (MemorySlot *s = memory->slots; s; s = s->next)
    if (s->account == account && s->kind == kind &&
        evictable(s, wanted, n, mode, needed))
      possible += s->bytes;
  uint64_t room = reclaim_room(account, kind, wanted, sizes, n);
  return possible < room ? possible : room;
}

static uint64_t reclaimable(Memory *memory, MemorySlot *const *wanted,
                            const uint64_t *sizes, size_t n,
                            MemoryAdmission mode, int needed) {
  uint64_t total = 0;
  for (MemoryAccount *a = memory->accounts; a; a = a->next) {
    total +=
        reclaimable_of(memory, a, MEMORY_MODEL, wanted, sizes, n, mode, needed);
    total +=
        reclaimable_of(memory, a, MEMORY_STATE, wanted, sizes, n, mode, needed);
  }
  return total;
}

static int fits(const MemoryUsage *usage, uint64_t extra) {
  return usage->committed <= usage->limit &&
         extra <= usage->limit - usage->committed;
}

static MemorySlot *victim(Memory *memory, MemorySlot *const *wanted,
                          const uint64_t *sizes, size_t n, MemoryAdmission mode,
                          uint64_t need, int needed) {
  MemorySlot *best = NULL;
  double rank = INFINITY;
  for (MemorySlot *s = memory->slots; s; s = s->next) {
    if (!evictable(s, wanted, n, mode, needed) || has_children(s)) continue;
    uint64_t useful = reclaim_room(s->account, s->kind, wanted, sizes, n);
    if (!useful) continue;
    if (useful > s->bytes) useful = s->bytes;
    if (useful > need) useful = need;
    double age = fmax(0, now() - s->touched),
           score = exp(-age / memory->times.keep_recently_used_weights) *
                   (double)s->bytes / useful;
    if (!best || score < rank) best = s, rank = score;
  }
  return best;
}

// Evicts the slot. Returns the counted bytes it freed.
static uint64_t evict_slot(MemorySlot *slot) {
  Memory *memory = slot->account->memory;
  uint64_t before = memory->usage.committed;
  slot->evict(slot->context);
  if (slot->payload) die("memory eviction retained its payload");
  change(slot, 0);
  memory_set_ready(slot, 0);
  memory->usage.evictions++;
  memory->released_at = now();
  return before - memory->usage.committed;
}

uint64_t memory_reclaim(Memory *memory, uint64_t bytes) {
  const MemoryUsage *usage = &memory->usage;
  uint64_t before = usage->allocated;
  while (before - usage->allocated < bytes) {
    MemorySlot *slot = victim(memory, NULL, NULL, 0, MEMORY_REQUIRED,
                              bytes - (before - usage->allocated), 0);
    if (!slot) break;
    evict_slot(slot);
  }
  uint64_t freed = before - usage->allocated;
  if (freed && memory->eviction_callback) memory->eviction_callback();
  return freed;
}

uint64_t memory_unload(MemoryAccount *account) {
  Memory *memory = account->memory;
  uint64_t before = memory->usage.allocated;
  // Each round evicts the idle slots that nothing depends on. A slot whose
  // dependents went in one round can go in the next, and the state slots once
  // the rest of the model is gone.
  for (int again = 1; again;) {
    again = 0;
    for (MemorySlot *s = memory->slots; s; s = s->next)
      if (s->account == account && !has_children(s) &&
          evictable(s, NULL, 0, MEMORY_REQUIRED, 0)) {
        evict_slot(s);
        again = 1;
      }
  }
  uint64_t freed = before - memory->usage.allocated;
  if (freed && memory->eviction_callback) memory->eviction_callback();
  return freed;
}

// The bytes to free so that extra fits within the limit. UINT64_MAX if freeing
// cannot make it fit.
static uint64_t over_limit(const MemoryUsage *usage, uint64_t extra) {
  if (fits(usage, extra)) return 0;
  return extra == UINT64_MAX || usage->committed > usage->limit
             ? UINT64_MAX
             : extra - (usage->limit - usage->committed);
}

static int admit(Memory *memory, MemorySlot *const *wanted,
                 const uint64_t *sizes, size_t n, MemoryAdmission mode) {
  MemoryUsage *usage = &memory->usage;
  int evicted = 0;
  uint64_t extra = growth(memory, wanted, sizes, n);
  // Bytes needed for the reserve are freed like bytes over the limit.
  // Speculative memory also leaves its own size free, so that small buffers
  // fit later without evicting an expert.
  uint64_t probe =
      mode == MEMORY_SPECULATIVE && extra <= UINT64_MAX / 2 ? 2 * extra : extra;
  uint64_t short_by = extra == UINT64_MAX ? 0 : shortage(memory, probe),
           over = over_limit(usage, extra);
  uint64_t need = over > short_by ? over : short_by, freed = 0;
  if (mode == MEMORY_SPECULATIVE && need) {
    usage->refusals++;
    return 0;
  }
  // Required memory also takes the slots that their owners keep only while
  // other slots can go, once those others cannot make the room.
  int needed = mode == MEMORY_REQUIRED && need && need != UINT64_MAX &&
               need > reclaimable(memory, wanted, sizes, n, mode, 0);
  if (need && (need == UINT64_MAX ||
               need > reclaimable(memory, wanted, sizes, n, mode, needed))) {
    if (short_by)
      say("memory: %s: refused %.2f GiB: %.2f GiB short of the %.1f GiB "
          "reserve",
          wanted[0]->name, extra / GIB_F, short_by / GIB_F,
          memory->reserve / GIB_F);
    usage->refusals++;
    return 0;
  }
  int taken = 0;
  // Evictions lower the counted bytes, down to each account's minimum. In a
  // growing account they also lower extra. The reserve is met once the counted
  // bytes and extra together are short_by below where they started.
  uint64_t target = usage->committed + extra - short_by;
  for (;;) {
    over = over_limit(usage, extra);
    uint64_t total = usage->committed + extra,
             left = total > target ? total - target : 0;
    if (!over && !left) break;
    uint64_t want = over > left ? over : left;
    MemorySlot *best = victim(memory, wanted, sizes, n, mode, want, 0);
    if (!best && needed) best = victim(memory, wanted, sizes, n, mode, want, 1);
    if (!best) die("memory reservation changed during eviction");
    freed += evict_slot(best);
    evicted = 1, taken++;
    extra = growth(memory, wanted, sizes, n);
  }
  if (short_by && freed >= LOGGED_EVICTION)
    say("memory: %s: adding %.2f GiB, %.2f GiB short of the %.1f GiB reserve "
        "(reading %.1f s old): evicted %d "
        "slots, %.2f GiB",
        wanted[0]->name, extra / GIB_F, short_by / GIB_F,
        memory->reserve / GIB_F, now() - memory->read_at, taken, freed / GIB_F);
  for (size_t i = 0; i < n; i++)
    change(wanted[i],
           sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload);
  if (evicted && memory->eviction_callback) memory->eviction_callback();
  return 1;
}

int memory_admit(MemorySlot *const *wanted, const uint64_t *sizes, size_t n,
                 MemoryAdmission mode) {
  // No slots means nothing to add, which always fits.
  if (!n) return 1;
  Memory *memory = wanted[0]->account->memory;
  for (size_t i = 0; i < n; i++) {
    if (wanted[i]->kind == MEMORY_STATE)
      die("%s: state memory is added only by claims", wanted[i]->name);
    if (wanted[i]->account->memory != memory)
      die("%s: admitted with slots of another Memory", wanted[i]->name);
  }
  return admit(memory, wanted, sizes, n, mode);
}

int memory_set_minimum(MemoryAccount *account, uint64_t bytes,
                       MemoryAdmission mode) {
  uint64_t old = account->minimum, before = counted(account);
  account->minimum = bytes;
  uint64_t after = counted(account);
  account->minimum = old;
  if (after > before) {
    MemorySlot promise;
    memory_slot_init(&promise, account, MEMORY_MODEL, NULL, "admission", NULL,
                     NULL);
    uint64_t extra = after - account->bytes;
    MemorySlot *wanted[] = {&promise};
    account->admitting = 1;
    int admitted = memory_admit(wanted, &extra, 1, mode);
    account->admitting = 0;
    if (!admitted) {
      memory_slot_remove(&promise);
      return 0;
    }
    memory_slot_remove(&promise);
  }
  MemoryUsage *usage = &account->memory->usage;
  before = counted(account);
  account->minimum = bytes;
  usage->committed = usage->committed - before + counted(account);
  if (usage->committed > usage->peak) usage->peak = usage->committed;
  return 1;
}

// Whether the state slot may be evicted to keep its account within its state
// limit, while the rest of its model stays.
static int idle_state(MemorySlot *slot) {
  return slot->kind == MEMORY_STATE && slot->bytes && slot->evict &&
         !slot->pins && !has_children(slot) &&
         (!slot->can_evict || slot->can_evict(slot->context, 0));
}

// Evicts the account's idle state slots, oldest first, until adding bytes to
// slot stays within its state limit. Returns whether it does.
static int within_state_limit(MemorySlot *slot, uint64_t bytes) {
  MemoryAccount *account = slot->account;
  Memory *memory = account->memory;
  uint64_t idle = 0;
  for (MemorySlot *s = memory->slots; s; s = s->next)
    if (s->account == account && s != slot && idle_state(s)) idle += s->bytes;
  if (UINT64_MAX - account->state_bytes < bytes ||
      account->state_bytes + bytes - idle > account->state_limit)
    return 0;
  int evicted = 0;
  while (account->state_bytes + bytes > account->state_limit) {
    MemorySlot *oldest = NULL;
    for (MemorySlot *s = memory->slots; s; s = s->next)
      if (s->account == account && s != slot && idle_state(s) &&
          (!oldest || s->touched < oldest->touched))
        oldest = s;
    if (!oldest) die("state memory changed during eviction");
    evict_slot(oldest);
    evicted = 1;
  }
  if (evicted && memory->eviction_callback) memory->eviction_callback();
  return 1;
}

int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode) {
  Memory *memory = slot->account->memory;
  if (UINT64_MAX - slot->payload < bytes) return 0;
  if (slot->kind == MEMORY_STATE && !within_state_limit(slot, bytes)) {
    memory->usage.refusals++;
    return 0;
  }
  uint64_t wanted_size = slot->payload + bytes;
  MemorySlot *wanted[] = {slot};
  if (wanted_size > slot->bytes &&
      !admit(memory, wanted, &wanted_size, 1, mode))
    return 0;
  slot->payload += bytes;
  memory->usage.allocated += bytes;
  return 1;
}

void memory_release(MemorySlot *slot, uint64_t bytes) {
  slot->payload -= bytes;
  slot->account->memory->usage.allocated -= bytes;
  // A state slot always counts just its bytes in use.
  if (slot->automatic || slot->kind == MEMORY_STATE) memory_shrink(slot);
}
void memory_shrink(MemorySlot *slot) { change(slot, slot->payload); }
MemoryUsage memory_usage(const Memory *memory) { return memory->usage; }
void memory_set_eviction_callback(Memory *memory, void (*callback)(void)) {
  memory->eviction_callback = callback;
}
uint64_t memory_room(const MemoryAccount *account) {
  const MemoryUsage *usage = &account->memory->usage;
  uint64_t free =
      usage->limit > usage->committed ? usage->limit - usage->committed : 0;
  uint64_t model = account->bytes - account->state_bytes,
           own = account->minimum > model ? account->minimum - model : 0;
  return UINT64_MAX - free < own ? UINT64_MAX : free + own;
}
MemorySlot *memory_shared(Memory *memory) { return &memory->shared; }

void *memory_host_alloc(MemorySlot *slot, size_t bytes) {
  if (!slot) die("host memory of %zu bytes without a memory slot", bytes);
  if (!memory_claim(slot, bytes, MEMORY_REQUIRED)) return NULL;
  void *pointer = calloc(1, bytes ? bytes : 1);
  HostAllocation *record = pointer ? malloc(sizeof *record) : NULL;
  if (!pointer || !record) {
    free(pointer);
    memory_release(slot, bytes);
    return NULL;
  }
  Memory *memory = slot->account->memory;
  pthread_mutex_lock(&memory->host_lock);
  *record = (HostAllocation){memory->host_allocations, pointer, slot, bytes};
  memory->host_allocations = record;
  pthread_mutex_unlock(&memory->host_lock);
  return pointer;
}

void *memory_host_alloc_required(MemorySlot *slot, size_t bytes) {
  void *pointer = memory_host_alloc(slot, bytes);
  if (!pointer)
    die("cannot allocate %zu bytes of %s memory", bytes,
        slot->kind == MEMORY_STATE ? "state" : "model");
  return pointer;
}

void memory_host_free(MemorySlot *slot, void *pointer) {
  if (!slot) die("host memory freed without a memory slot");
  Memory *memory = slot->account->memory;
  pthread_mutex_lock(&memory->host_lock);
  HostAllocation **at = &memory->host_allocations;
  while (*at && (*at)->pointer != pointer) at = &(*at)->next;
  HostAllocation *record = *at;
  if (record) *at = record->next;
  pthread_mutex_unlock(&memory->host_lock);
  if (record) {
    if (record->slot != slot)
      die("host memory counted in %s freed with %s", record->slot->name,
          slot->name);
    memory_release(record->slot, record->bytes);
    free(record);
  }
  free(pointer);
}
