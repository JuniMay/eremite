#include "engine/memory.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include "base/shutdown.h"
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
  // The eviction callback, or null.
  void (*eviction_callback)(void *context);
  // The eviction callback's argument.
  void *eviction_context;
  // The kernel's count of available memory lags behind this program. So each
  // reading is corrected for what this program used since, and bytes counted
  // but not yet written count as used.
  SystemMemoryCounters (*read_counters)(void);
  uint64_t (*read_unwritten)(void), reserve;
  // The file cache that the kernel keeps rather than give up for other
  // memory: the smallest file cache read, or more after
  // memory_keep_file_cache. UINT64_MAX before the first reading.
  uint64_t kept_file_cache;
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
static constexpr double LONG_AGO = -1e9;
static constexpr double GIB_F = 1073741824.0;
// Evictions that free at least this much at once are logged.
static constexpr uint64_t LOGGED_EVICTION = 1ull << 30;

static uint64_t in_use(const Memory *memory) { return memory->usage.allocated; }

// The bytes available to this program in the counters c: the free pages, and
// the file cache above the one that the kernel keeps. A smaller file cache
// lowers the one kept.
static uint64_t available_in(Memory *memory, SystemMemoryCounters c) {
  if (c.file < memory->kept_file_cache) memory->kept_file_cache = c.file;
  return c.free + (c.file - memory->kept_file_cache);
}

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
    memory->read_bytes = available_in(memory, memory->read_counters());
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
  if (!memory->read_counters || !bytes) return 0;
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
  slot->account->role_bytes[slot->role] =
      slot->account->role_bytes[slot->role] - slot->bytes + bytes;
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
  memory->kept_file_cache = UINT64_MAX;
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
                        SystemMemoryCounters (*reader)(void),
                        uint64_t (*unwritten)(void)) {
  memory->reserve = bytes;
  memory->read_counters = reader;
  memory->read_unwritten = unwritten;
  memory->read_at = LONG_AGO;  // the next use takes a new reading
}

SystemMemoryCounters memory_read_system(Memory *memory, uint64_t *available) {
  SystemMemoryCounters c = {0};
  if (memory->read_counters) c = memory->read_counters();
  *available = memory->read_counters ? available_in(memory, c) : 0;
  return c;
}

uint64_t memory_kept_file_cache(const Memory *memory) {
  return memory->kept_file_cache == UINT64_MAX ? 0 : memory->kept_file_cache;
}

void memory_keep_file_cache(Memory *memory, uint64_t bytes) {
  if (memory->kept_file_cache == UINT64_MAX ||
      bytes > memory->kept_file_cache) {
    memory->kept_file_cache = bytes;
    memory->read_at = LONG_AGO;  // the last reading counted more as available
  }
}

uint64_t memory_current_limit(Memory *memory) {
  const MemoryUsage *usage = &memory->usage;
  if (!memory->read_counters) return usage->limit;
  uint64_t a = available(memory, 0),
           room = a > memory->reserve ? a - memory->reserve : 0;
  uint64_t limit = UINT64_MAX - usage->committed < room
                       ? UINT64_MAX
                       : usage->committed + room;
  return limit < usage->limit ? limit : usage->limit;
}

uint64_t memory_ceiling(Memory *memory) {
  uint64_t ceiling = memory->usage.limit;
  if (!memory->read_counters) return ceiling;
  uint64_t machine = memory->read_counters().total;
  // A reading without the machine's RAM sets no bound of its own.
  if (!machine) return ceiling;
  uint64_t room = machine > memory->reserve ? machine - memory->reserve : 0;
  return room < ceiling ? room : ceiling;
}

void memory_account_init(Memory *memory, MemoryAccount *account,
                         const char *name, uint64_t limit) {
  *account = (MemoryAccount){.memory = memory,
                             .next = memory->accounts,
                             .name = name,
                             .limit = limit ? limit : UINT64_MAX};
  memory->accounts = account;
}

void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemoryKind kind,
                      MemorySlot *dependency, const char *name,
                      void (*evict)(void *), void *context) {
  Memory *memory = account->memory;
  if (dependency && dependency->account->memory != memory)
    die("%s: depends on a slot of another Memory", name);
  MemoryRole role =
      kind == MEMORY_STATE ? MEMORY_ROLE_STATE : MEMORY_ROLE_WORKSPACE;
  *slot = (MemorySlot){.next = memory->slots,
                       .account = account,
                       .kind = kind,
                       .role = role,
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

void memory_slot_set_role(MemorySlot *slot, MemoryRole role) {
  if (role < 0 || role >= MEMORY_ROLE_COUNT)
    die("%s: invalid memory role %d", slot->name, (int)role);
  if ((slot->kind == MEMORY_STATE) != (role == MEMORY_ROLE_STATE))
    die("%s: slot kind and reporting role disagree", slot->name);
  if (slot->role == role) return;
  slot->account->role_bytes[slot->role] -= slot->bytes;
  slot->role = role;
  slot->account->role_bytes[slot->role] += slot->bytes;
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

// The bytes counted for the account with the wanted slots at their new sizes,
// as counted() counts them. UINT64_MAX when they are more than that.
static uint64_t proposed_count(const MemoryAccount *account,
                               MemorySlot *const *wanted, const uint64_t *sizes,
                               size_t n) {
  uint64_t model = proposed_bytes(account, MEMORY_MODEL, wanted, sizes, n),
           state = proposed_bytes(account, MEMORY_STATE, wanted, sizes, n);
  if (model == UINT64_MAX || state == UINT64_MAX) return UINT64_MAX;
  uint64_t floor = model > account->minimum ? model : account->minimum;
  return UINT64_MAX - floor < state ? UINT64_MAX : floor + state;
}

static uint64_t growth(Memory *memory, MemorySlot *const *wanted,
                       const uint64_t *sizes, size_t n) {
  uint64_t extra = 0;
  for (MemoryAccount *a = memory->accounts; a; a = a->next) {
    uint64_t after = proposed_count(a, wanted, sizes, n);
    if (after == UINT64_MAX) return UINT64_MAX;
    if (after > counted(a)) {
      if (UINT64_MAX - extra < after - counted(a)) return UINT64_MAX;
      extra += after - counted(a);
    }
  }
  return extra;
}

// The bytes that the account counts over its own limit with the wanted slots
// at their new sizes. UINT64_MAX when they are more than that.
static uint64_t over_own_limit(const MemoryAccount *account,
                               MemorySlot *const *wanted, const uint64_t *sizes,
                               size_t n) {
  uint64_t after = proposed_count(account, wanted, sizes, n);
  if (after == UINT64_MAX) return UINT64_MAX;
  return after > account->limit ? after - account->limit : 0;
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

// The bytes that evicting the account's idle slots can take off its count.
static uint64_t reclaimable_in(Memory *memory, const MemoryAccount *account,
                               MemorySlot *const *wanted, const uint64_t *sizes,
                               size_t n, MemoryAdmission mode, int needed) {
  return reclaimable_of(memory, account, MEMORY_MODEL, wanted, sizes, n, mode,
                        needed) +
         reclaimable_of(memory, account, MEMORY_STATE, wanted, sizes, n, mode,
                        needed);
}

static uint64_t reclaimable(Memory *memory, MemorySlot *const *wanted,
                            const uint64_t *sizes, size_t n,
                            MemoryAdmission mode, int needed) {
  uint64_t total = 0;
  for (MemoryAccount *a = memory->accounts; a; a = a->next)
    total += reclaimable_in(memory, a, wanted, sizes, n, mode, needed);
  return total;
}

static int fits(const MemoryUsage *usage, uint64_t extra) {
  return usage->committed <= usage->limit &&
         extra <= usage->limit - usage->committed;
}

// The idle slot to evict next for need bytes: of the account only, when it is
// not null.
static MemorySlot *victim(Memory *memory, const MemoryAccount *account,
                          MemorySlot *const *wanted, const uint64_t *sizes,
                          size_t n, MemoryAdmission mode, uint64_t need,
                          int needed) {
  MemorySlot *best = NULL;
  double rank = INFINITY;
  for (MemorySlot *s = memory->slots; s; s = s->next) {
    if ((account && s->account != account) ||
        !evictable(s, wanted, n, mode, needed) || has_children(s))
      continue;
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
    MemorySlot *slot = victim(memory, NULL, NULL, NULL, 0, MEMORY_REQUIRED,
                              bytes - (before - usage->allocated), 0);
    if (!slot) break;
    evict_slot(slot);
  }
  uint64_t freed = before - usage->allocated;
  if (freed && memory->eviction_callback)
    memory->eviction_callback(memory->eviction_context);
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
  if (freed && memory->eviction_callback)
    memory->eviction_callback(memory->eviction_context);
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
    if (short_by) {
      // The parts of the available memory (available()).
      double since = ((double)in_use(memory) - (double)memory->read_used);
      uint64_t promised = usage->committed > usage->allocated
                              ? usage->committed - usage->allocated
                              : 0;
      say("memory: %s: refused %.2f GiB: %.2f GiB short of the %.1f GiB "
          "reserve (read %.2f GiB available %.1f s ago; since then %+.2f GiB "
          "used; %.2f GiB not written; %.2f GiB promised)",
          wanted[0]->name, extra / GIB_F, short_by / GIB_F,
          memory->reserve / GIB_F, memory->read_bytes / GIB_F,
          now() - memory->read_at, since / GIB_F, memory->read_pending / GIB_F,
          promised / GIB_F);
    }
    usage->refusals++;
    return 0;
  }
  // Each account stays within its own limit, with room from its own slots.
  for (MemoryAccount *a = memory->accounts; a; a = a->next) {
    uint64_t own = over_own_limit(a, wanted, sizes, n);
    if (!own) continue;
    int own_needed = mode == MEMORY_REQUIRED && own != UINT64_MAX &&
                     own > reclaimable_in(memory, a, wanted, sizes, n, mode, 0);
    if (own == UINT64_MAX ||
        own > reclaimable_in(memory, a, wanted, sizes, n, mode, own_needed)) {
      if (mode == MEMORY_REQUIRED && own != UINT64_MAX)
        say("memory: %s: refused: %.2f GiB over the %.1f GiB limit of %s",
            wanted[0]->name, own / GIB_F, a->limit / GIB_F, a->name);
      usage->refusals++;
      return 0;
    }
  }
  int taken = 0;
  // Evictions lower the counted bytes, down to each account's minimum. In a
  // growing account they also lower extra. The reserve is met once the counted
  // bytes and extra together are short_by below where they started.
  uint64_t target = usage->committed + extra - short_by;
  // First the bytes over each account's own limit, from its own slots.
  for (MemoryAccount *a = memory->accounts; a; a = a->next)
    for (uint64_t own; (own = over_own_limit(a, wanted, sizes, n));) {
      MemorySlot *best = victim(memory, a, wanted, sizes, n, mode, own, 0);
      if (!best && mode == MEMORY_REQUIRED)
        best = victim(memory, a, wanted, sizes, n, mode, own, 1);
      if (!best) die("memory reservation changed during eviction");
      freed += evict_slot(best);
      evicted = 1, taken++;
    }
  extra = growth(memory, wanted, sizes, n);
  for (;;) {
    over = over_limit(usage, extra);
    uint64_t total = usage->committed + extra,
             left = total > target ? total - target : 0;
    if (!over && !left) break;
    uint64_t want = over > left ? over : left;
    MemorySlot *best = victim(memory, NULL, wanted, sizes, n, mode, want, 0);
    if (!best && needed)
      best = victim(memory, NULL, wanted, sizes, n, mode, want, 1);
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
  if (evicted && memory->eviction_callback)
    memory->eviction_callback(memory->eviction_context);
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

// Tries attempt(context) until it succeeds. While it fails but each new
// reading shows more available memory than the one before, it waits
// freed_memory_delay before the next try. Returns whether it succeeded. A stop
// request ends the waiting.
static int retry_while_memory_returns(Memory *memory,
                                      int (*attempt)(void *context),
                                      void *context) {
  uint64_t seen =
      memory->read_counters ? available_in(memory, memory->read_counters()) : 0;
  while (!attempt(context)) {
    if (!memory->read_counters || shutdown_requested()) return 0;
    sleep_seconds(memory->times.freed_memory_delay);
    uint64_t now_available = available_in(memory, memory->read_counters());
    if (now_available <= seen) return 0;
    seen = now_available;
    memory->read_at = LONG_AGO;  // the next try takes a new reading
  }
  return 1;
}

// A minimum or a claim to retry.
typedef struct {
  MemoryAccount *account;
  MemorySlot *slot;
  uint64_t bytes;
} Retried;

static int set_minimum_required(void *context) {
  Retried *retried = context;
  return memory_set_minimum(retried->account, retried->bytes, MEMORY_REQUIRED);
}

static int claim_required(void *context) {
  Retried *retried = context;
  return memory_claim(retried->slot, retried->bytes, MEMORY_REQUIRED);
}

int memory_wait_for_minimum(MemoryAccount *account, uint64_t bytes) {
  Retried retried = {.account = account, .bytes = bytes};
  return retry_while_memory_returns(account->memory, set_minimum_required,
                                    &retried);
}

int memory_wait_for_claim(MemorySlot *slot, uint64_t bytes) {
  Retried retried = {.slot = slot, .bytes = bytes};
  return retry_while_memory_returns(slot->account->memory, claim_required,
                                    &retried);
}

// Whether the state slot may be evicted to make room for another state of its
// account, while the rest of its model stays.
static int idle_state(MemorySlot *slot) {
  return slot->kind == MEMORY_STATE && slot->bytes && slot->evict &&
         !slot->pins && !has_children(slot) &&
         (!slot->can_evict || slot->can_evict(slot->context, 0));
}

// Makes room for bytes more in the state slot within its account's own limit.
// The account's idle memory above its minimum makes room first, when the
// claim's admission evicts it. As far as that is not enough, this evicts the
// account's idle state slots, oldest first. Returns whether the bytes then fit
// within the limit; if not, nothing is evicted.
static int make_state_room(MemorySlot *slot, uint64_t bytes,
                           MemoryAdmission mode) {
  MemoryAccount *account = slot->account;
  Memory *memory = account->memory;
  uint64_t size = slot->payload + bytes;
  MemorySlot *wanted[] = {slot};
  uint64_t over = over_own_limit(account, wanted, &size, 1);
  if (!over) return 1;
  if (over == UINT64_MAX) return 0;
  uint64_t model =
      reclaimable_of(memory, account, MEMORY_MODEL, wanted, &size, 1, mode, 0);
  if (over <= model) return 1;
  uint64_t idle = 0;
  for (MemorySlot *s = memory->slots; s; s = s->next)
    if (s->account == account && s != slot && idle_state(s)) idle += s->bytes;
  if (over - model > idle) return 0;
  int evicted = 0;
  while (over_own_limit(account, wanted, &size, 1) > model) {
    MemorySlot *oldest = NULL;
    for (MemorySlot *s = memory->slots; s; s = s->next)
      if (s->account == account && s != slot && idle_state(s) &&
          (!oldest || s->touched < oldest->touched))
        oldest = s;
    if (!oldest) die("state memory changed during eviction");
    evict_slot(oldest);
    evicted = 1;
  }
  if (evicted && memory->eviction_callback)
    memory->eviction_callback(memory->eviction_context);
  return 1;
}

int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode) {
  Memory *memory = slot->account->memory;
  if (UINT64_MAX - slot->payload < bytes) return 0;
  if (slot->kind == MEMORY_STATE && !make_state_room(slot, bytes, mode)) {
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

void memory_set_eviction_callback(Memory *memory,
                                  void (*callback)(void *context),
                                  void *context) {
  memory->eviction_callback = callback;
  memory->eviction_context = context;
}

uint64_t memory_room(const MemoryAccount *account) {
  const MemoryUsage *usage = &account->memory->usage;
  uint64_t free =
      usage->limit > usage->committed ? usage->limit - usage->committed : 0;
  uint64_t count = counted(account),
           own_free = account->limit > count ? account->limit - count : 0;
  if (own_free < free) free = own_free;
  uint64_t model = account->bytes - account->state_bytes,
           own = account->minimum > model ? account->minimum - model : 0;
  return UINT64_MAX - free < own ? UINT64_MAX : free + own;
}

uint64_t memory_counted(const MemoryAccount *account) {
  return counted(account);
}

MemoryAccountBreakdown memory_account_breakdown(const MemoryAccount *account) {
  return (MemoryAccountBreakdown){
      .weights = account->role_bytes[MEMORY_ROLE_WEIGHTS],
      .experts = account->role_bytes[MEMORY_ROLE_EXPERTS],
      .workspace = account->role_bytes[MEMORY_ROLE_WORKSPACE],
      .states = account->role_bytes[MEMORY_ROLE_STATE],
      .total = account->bytes};
}

MemorySlot *memory_shared(Memory *memory) { return &memory->shared; }

// Returns zeroed host memory of bytes, already claimed in slot, and records it
// there. Null if the system has no such memory: the claim is released then.
static void *host_alloc_claimed(MemorySlot *slot, size_t bytes) {
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

void *memory_host_alloc(MemorySlot *slot, size_t bytes) {
  if (!slot) die("host memory of %zu bytes without a memory slot", bytes);
  if (!memory_claim(slot, bytes, MEMORY_REQUIRED)) return NULL;
  return host_alloc_claimed(slot, bytes);
}

void *memory_host_alloc_required(MemorySlot *slot, size_t bytes) {
  if (!slot) die("host memory of %zu bytes without a memory slot", bytes);
  void *pointer =
      memory_wait_for_claim(slot, bytes) ? host_alloc_claimed(slot, bytes) : 0;
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
