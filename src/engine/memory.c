#include "engine/memory.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static MemoryAccount *accounts;
static MemorySlot *slots;
static MemoryUsage usage = {.limit = UINT64_MAX};
static void (*collect_released)(void);

// The kernel's count of available memory lags behind this program. So each
// reading is corrected for what this program used since, and bytes counted
// but not yet written count as used.
static uint64_t (*read_available)(void), (*read_unwritten)(void), reserve;
// The last reading, what this program used then, and how much of that was
// not written yet.
static uint64_t read_bytes, read_used, read_pending;
// A time before any reading or release.
#define LONG_AGO -1e9
static double read_at = LONG_AGO, released_at = LONG_AGO;

static uint64_t in_use(void) { return usage.allocated; }
#define GIB_F 1073741824.0
// Evictions that free at least this much at once are logged.
#define LOGGED_EVICTION (1ull << 30)

// The available memory. A new reading is taken after 1 s, or at once for 1 GiB
// or more, but not within 1.5 s of a release unless the last is 5 s old.
static uint64_t available(uint64_t adding) {
  double t = now();
  if ((t - released_at >= 1.5 || t - read_at >= 5) &&
      (adding >= 1ull << 30 || t - read_at >= 1)) {
    read_bytes = read_available(), read_used = in_use(), read_at = t;
    read_pending = read_unwritten ? read_unwritten() : 0;
  }
  uint64_t used = in_use(), promised = usage.committed > usage.allocated
                                           ? usage.committed - usage.allocated
                                           : 0;
  uint64_t a = used >= read_used ? (read_bytes > used - read_used
                                        ? read_bytes - (used - read_used)
                                        : 0)
                                 : read_bytes + (read_used - used);
  a = a > read_pending ? a - read_pending : 0;
  return a > promised ? a - promised : 0;
}

// How far adding bytes would go below the reserve.
static uint64_t shortage(uint64_t bytes) {
  if (!read_available || !bytes) return 0;
  uint64_t a = available(bytes), room = a > reserve ? a - reserve : 0;
  return bytes > room ? bytes - room : 0;
}

// The bytes counted for the account: its slots' bytes, or its minimum if
// larger.
static uint64_t counted(const MemoryAccount *account) {
  return account->bytes > account->minimum ? account->bytes : account->minimum;
}

static void change(MemorySlot *slot, uint64_t bytes) {
  if (slot->kind == MEMORY_STATE)
    slot->account->state_bytes =
        slot->account->state_bytes - slot->bytes + bytes;
  uint64_t before = counted(slot->account);
  slot->account->bytes -= slot->bytes;
  slot->account->bytes += bytes;
  slot->bytes = bytes;
  usage.committed = usage.committed - before + counted(slot->account);
  if (usage.committed > usage.peak) usage.peak = usage.committed;
}

void memory_set_limit(uint64_t bytes) { usage.limit = bytes; }

void memory_set_reserve(uint64_t bytes, uint64_t (*reader)(void),
                        uint64_t (*unwritten)(void)) {
  reserve = bytes, read_available = reader, read_unwritten = unwritten;
  read_at = LONG_AGO;  // the next use takes a new reading
}

uint64_t memory_current_limit(void) {
  if (!read_available) return usage.limit;
  uint64_t a = available(0), room = a > reserve ? a - reserve : 0;
  uint64_t limit =
      UINT64_MAX - usage.committed < room ? UINT64_MAX : usage.committed + room;
  return limit < usage.limit ? limit : usage.limit;
}

void memory_account_init(MemoryAccount *account, const char *name,
                         uint64_t state_limit) {
  *account = (MemoryAccount){
      .next = accounts, .name = name, .state_limit = state_limit};
  accounts = account;
}

void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemoryKind kind,
                      MemorySlot *dependency, const char *name,
                      void (*evict)(void *), void *context) {
  *slot = (MemorySlot){.next = slots,
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
  slots = slot;
}

void memory_slot_remove(MemorySlot *slot) {
  change(slot, 0);
  MemorySlot **at = &slots;
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

static int evictable(MemorySlot *slot, MemorySlot *const *wanted, size_t n,
                     MemoryAdmission mode) {
  if (!slot->bytes || !slot->evict || slot->pins || slot->account->admitting ||
      requested(slot, wanted, n) ||
      (slot->can_evict && !slot->can_evict(slot->context)))
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
  // A growing cache keeps weights used in the last 30 seconds.
  if (mode == MEMORY_CACHE && !slot->dependency && now() - slot->touched < 30)
    return 0;
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes && !evictable(child, wanted, n, mode)) return 0;
  return 1;
}

static int has_children(MemorySlot *slot) {
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes) return 1;
  return 0;
}

static uint64_t proposed_bytes(const MemoryAccount *account,
                               MemorySlot *const *wanted, const uint64_t *sizes,
                               size_t n) {
  uint64_t proposed = account->bytes;
  for (size_t i = 0; i < n; i++)
    if (wanted[i]->account == account) {
      uint64_t bytes =
          sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload;
      if (bytes > wanted[i]->bytes &&
          UINT64_MAX - proposed < bytes - wanted[i]->bytes)
        return UINT64_MAX;
      proposed = proposed - wanted[i]->bytes + bytes;
    }
  return proposed;
}

static uint64_t growth(MemorySlot *const *wanted, const uint64_t *sizes,
                       size_t n) {
  uint64_t extra = 0;
  for (MemoryAccount *a = accounts; a; a = a->next) {
    uint64_t proposed = proposed_bytes(a, wanted, sizes, n);
    uint64_t after = proposed > a->minimum ? proposed : a->minimum;
    if (after > counted(a)) {
      if (UINT64_MAX - extra < after - counted(a)) return UINT64_MAX;
      extra += after - counted(a);
    }
  }
  return extra;
}

static uint64_t reclaim_room(const MemoryAccount *account,
                             MemorySlot *const *wanted, const uint64_t *sizes,
                             size_t n) {
  uint64_t bytes = proposed_bytes(account, wanted, sizes, n);
  if (bytes < account->bytes) bytes = account->bytes;
  return bytes > account->minimum ? bytes - account->minimum : 0;
}

static uint64_t reclaimable(MemorySlot *const *wanted, const uint64_t *sizes,
                            size_t n, MemoryAdmission mode) {
  uint64_t total = 0;
  for (MemoryAccount *a = accounts; a; a = a->next) {
    uint64_t possible = 0;
    for (MemorySlot *s = slots; s; s = s->next)
      if (s->account == a && evictable(s, wanted, n, mode))
        possible += s->bytes;
    uint64_t room = reclaim_room(a, wanted, sizes, n);
    total += possible < room ? possible : room;
  }
  return total;
}

static int fits(uint64_t extra) {
  return usage.committed <= usage.limit &&
         extra <= usage.limit - usage.committed;
}

static MemorySlot *victim(MemorySlot *const *wanted, const uint64_t *sizes,
                          size_t n, MemoryAdmission mode, uint64_t need) {
  MemorySlot *best = NULL;
  double rank = INFINITY;
  for (MemorySlot *s = slots; s; s = s->next) {
    if (!evictable(s, wanted, n, mode) || has_children(s)) continue;
    uint64_t useful = reclaim_room(s->account, wanted, sizes, n);
    if (!useful) continue;
    if (useful > s->bytes) useful = s->bytes;
    if (useful > need) useful = need;
    double age = fmax(0, now() - s->touched),
           score = exp(-age / 30) * (double)s->bytes / useful;
    if (!best || score < rank) best = s, rank = score;
  }
  return best;
}

// Evicts the slot. Returns the counted bytes it freed.
static uint64_t evict_slot(MemorySlot *slot) {
  uint64_t before = usage.committed;
  slot->evict(slot->context);
  if (slot->payload) die("memory eviction retained its payload");
  change(slot, 0);
  memory_set_ready(slot, 0);
  usage.evictions++;
  released_at = now();
  return before - usage.committed;
}

uint64_t memory_reclaim(uint64_t bytes) {
  uint64_t before = usage.allocated;
  while (before - usage.allocated < bytes) {
    MemorySlot *slot = victim(NULL, NULL, 0, MEMORY_REQUIRED,
                              bytes - (before - usage.allocated));
    if (!slot) break;
    evict_slot(slot);
  }
  uint64_t freed = before - usage.allocated;
  if (freed && collect_released) collect_released();
  return freed;
}

uint64_t memory_unload(MemoryAccount *account) {
  uint64_t before = usage.allocated;
  // Each round evicts the idle slots that nothing depends on. A slot whose
  // dependents went in one round can go in the next, and the state slots once
  // the rest of the model is gone.
  for (int again = 1; again;) {
    again = 0;
    for (MemorySlot *s = slots; s; s = s->next)
      if (s->account == account && !has_children(s) &&
          evictable(s, NULL, 0, MEMORY_REQUIRED)) {
        evict_slot(s);
        again = 1;
      }
  }
  uint64_t freed = before - usage.allocated;
  if (freed && collect_released) collect_released();
  return freed;
}

// The bytes to free so that extra fits within the limit. UINT64_MAX if freeing
// cannot make it fit.
static uint64_t over_limit(uint64_t extra) {
  if (fits(extra)) return 0;
  return extra == UINT64_MAX || usage.committed > usage.limit
             ? UINT64_MAX
             : extra - (usage.limit - usage.committed);
}

static int admit(MemorySlot *const *wanted, const uint64_t *sizes, size_t n,
                 MemoryAdmission mode) {
  int evicted = 0;
  uint64_t extra = growth(wanted, sizes, n);
  // Bytes needed for the reserve are freed like bytes over the limit.
  // Speculative memory also leaves its own size free, so that small buffers
  // fit later without evicting an expert.
  uint64_t probe =
      mode == MEMORY_SPECULATIVE && extra <= UINT64_MAX / 2 ? 2 * extra : extra;
  uint64_t short_by = extra == UINT64_MAX ? 0 : shortage(probe),
           over = over_limit(extra);
  uint64_t need = over > short_by ? over : short_by, freed = 0;
  if (mode == MEMORY_SPECULATIVE && need) {
    usage.refusals++;
    return 0;
  }
  if (need &&
      (need == UINT64_MAX || need > reclaimable(wanted, sizes, n, mode))) {
    if (short_by)
      say("memory: %s: refused %.2f GiB: %.2f GiB short of the %.1f GiB "
          "reserve",
          wanted[0]->name, extra / GIB_F, short_by / GIB_F, reserve / GIB_F);
    usage.refusals++;
    return 0;
  }
  int taken = 0;
  while ((over = over_limit(extra)) || freed < short_by) {
    uint64_t left = freed < short_by ? short_by - freed : 0,
             want = over > left ? over : left;
    MemorySlot *best = victim(wanted, sizes, n, mode, want);
    if (!best) die("memory reservation changed during eviction");
    freed += evict_slot(best);
    evicted = 1, taken++;
    extra = growth(wanted, sizes, n);
  }
  if (short_by && freed >= LOGGED_EVICTION)
    say("memory: %s: adding %.2f GiB, %.2f GiB short of the %.1f GiB reserve "
        "(reading %.1f s old): evicted %d "
        "slots, %.2f GiB",
        wanted[0]->name, extra / GIB_F, short_by / GIB_F, reserve / GIB_F,
        now() - read_at, taken, freed / GIB_F);
  for (size_t i = 0; i < n; i++)
    change(wanted[i],
           sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload);
  if (evicted && collect_released) collect_released();
  return 1;
}

int memory_admit(MemorySlot *const *wanted, const uint64_t *sizes, size_t n,
                 MemoryAdmission mode) {
  for (size_t i = 0; i < n; i++)
    if (wanted[i]->kind == MEMORY_STATE)
      die("%s: state memory is added only by claims", wanted[i]->name);
  return admit(wanted, sizes, n, mode);
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
  before = counted(account);
  account->minimum = bytes;
  usage.committed = usage.committed - before + counted(account);
  if (usage.committed > usage.peak) usage.peak = usage.committed;
  return 1;
}

// Whether the state slot may be evicted to keep its account within its state
// limit, while the rest of its model stays.
static int idle_state(MemorySlot *slot) {
  return slot->kind == MEMORY_STATE && slot->bytes && slot->evict &&
         !slot->pins && !has_children(slot) &&
         (!slot->can_evict || slot->can_evict(slot->context));
}

// Evicts the account's idle state slots, oldest first, until adding bytes to
// slot stays within its state limit. Returns whether it does.
static int within_state_limit(MemorySlot *slot, uint64_t bytes) {
  MemoryAccount *account = slot->account;
  uint64_t idle = 0;
  for (MemorySlot *s = slots; s; s = s->next)
    if (s->account == account && s != slot && idle_state(s)) idle += s->bytes;
  if (UINT64_MAX - account->state_bytes < bytes ||
      account->state_bytes + bytes - idle > account->state_limit)
    return 0;
  int evicted = 0;
  while (account->state_bytes + bytes > account->state_limit) {
    MemorySlot *oldest = NULL;
    for (MemorySlot *s = slots; s; s = s->next)
      if (s->account == account && s != slot && idle_state(s) &&
          (!oldest || s->touched < oldest->touched))
        oldest = s;
    if (!oldest) die("state memory changed during eviction");
    evict_slot(oldest);
    evicted = 1;
  }
  if (evicted && collect_released) collect_released();
  return 1;
}

int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode) {
  if (UINT64_MAX - slot->payload < bytes) return 0;
  if (slot->kind == MEMORY_STATE && !within_state_limit(slot, bytes)) {
    usage.refusals++;
    return 0;
  }
  uint64_t wanted_size = slot->payload + bytes;
  MemorySlot *wanted[] = {slot};
  if (wanted_size > slot->bytes && !admit(wanted, &wanted_size, 1, mode))
    return 0;
  slot->payload += bytes;
  usage.allocated += bytes;
  return 1;
}

void memory_release(MemorySlot *slot, uint64_t bytes) {
  slot->payload -= bytes;
  usage.allocated -= bytes;
  // A state slot always counts just its bytes in use.
  if (slot->automatic || slot->kind == MEMORY_STATE) memory_shrink(slot);
}
void memory_shrink(MemorySlot *slot) { change(slot, slot->payload); }
MemoryUsage memory_usage(void) { return usage; }
void memory_collect_with(void (*collect)(void)) { collect_released = collect; }
uint64_t memory_room(const MemoryAccount *account) {
  uint64_t free =
      usage.limit > usage.committed ? usage.limit - usage.committed : 0;
  uint64_t own =
      account->minimum > account->bytes ? account->minimum - account->bytes : 0;
  return UINT64_MAX - free < own ? UINT64_MAX : free + own;
}
MemorySlot *memory_shared(void) {
  static MemoryAccount shared;
  static MemorySlot slot;
  static int initialized;
  if (!initialized) {
    memory_account_init(&shared, "shared", 0);
    memory_slot_init(&slot, &shared, MEMORY_MODEL, NULL, "shared runtime", NULL,
                     NULL);
    slot.automatic = 1;
    initialized = 1;
  }
  return &slot;
}

typedef struct HostAllocation {
  struct HostAllocation *next;
  void *pointer;
  MemorySlot *slot;
  size_t bytes;
} HostAllocation;
static HostAllocation *host_allocations;
static pthread_mutex_t host_lock = PTHREAD_MUTEX_INITIALIZER;

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
  pthread_mutex_lock(&host_lock);
  *record = (HostAllocation){host_allocations, pointer, slot, bytes};
  host_allocations = record;
  pthread_mutex_unlock(&host_lock);
  return pointer;
}

void *memory_host_alloc_required(MemorySlot *slot, size_t bytes) {
  void *pointer = memory_host_alloc(slot, bytes);
  if (!pointer)
    die("cannot allocate %zu bytes of %s memory", bytes,
        slot->kind == MEMORY_STATE ? "state" : "model");
  return pointer;
}

void memory_host_free(void *pointer) {
  pthread_mutex_lock(&host_lock);
  HostAllocation **at = &host_allocations;
  while (*at && (*at)->pointer != pointer) at = &(*at)->next;
  HostAllocation *record = *at;
  if (record) *at = record->next;
  pthread_mutex_unlock(&host_lock);
  if (record) {
    memory_release(record->slot, record->bytes);
    free(record);
  }
  free(pointer);
}
