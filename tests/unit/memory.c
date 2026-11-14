#include "base/shutdown.h"
#include "check.h"
#include "engine/memory.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests memory accounting: the eviction order, pins, minimums, the admission
// modes, host memory and state memory. Each case runs in its own process with
// a fixed clock.

// The time that now() returns.
static double clock_now = 100;
// The ids of the evicted test slots, in the order they were evicted, and how
// many there are.
static int order[16], order_count;
// The available memory that the system reports to the reserve.
static uint64_t available_bytes;
// The times the eviction callback ran.
static int callbacks;

// Memory that another program released, and the bytes of it that come back to
// the system's counts at each sleep.
static uint64_t coming_back, coming_back_step;

double now(void) { return clock_now; }

void sleep_seconds(double seconds) {
  clock_now += seconds;
  uint64_t step =
      coming_back < coming_back_step ? coming_back : coming_back_step;
  available_bytes += step, coming_back -= step;
}

[[noreturn]] void die(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  exit(2);
}

void say(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}

#define REQUIRE(test)                                                          \
  do {                                                                         \
    if (!(test)) die("%s:%d: %s", __func__, __LINE__, #test);                  \
  } while (0)

typedef struct {
  MemorySlot slot;
  int id, evicted;
} TestSlot;

static void evict(void *context) {
  TestSlot *test = context;
  order[order_count++] = test->id;
  test->evicted++;
  memory_release(&test->slot, test->slot.payload);
}

static void test_slot(TestSlot *test, MemoryAccount *account, TestSlot *parent,
                      uint64_t bytes, int id) {
  *test = (TestSlot){.id = id};
  memory_slot_init(&test->slot, account, MEMORY_MODEL,
                   parent ? &parent->slot : NULL, "fixture", evict, test);
  REQUIRE(memory_claim(&test->slot, bytes, MEMORY_REQUIRED));
  memory_set_ready(&test->slot, 1);
}

static void test_state_slot(TestSlot *test, MemoryAccount *account,
                            uint64_t bytes, int id) {
  *test = (TestSlot){.id = id};
  memory_slot_init(&test->slot, account, MEMORY_STATE, NULL, "state", evict,
                   test);
  REQUIRE(memory_claim(&test->slot, bytes, MEMORY_REQUIRED));
  memory_set_ready(&test->slot, 1);
}

static int claim(TestSlot *test, uint64_t bytes, MemoryAdmission mode) {
  return memory_claim(&test->slot, bytes, mode);
}

static void dependents_are_evicted_first(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 120);
  MemoryAccount moe, dense, other;
  memory_account_init(memory, &moe, "MoE", 0);
  memory_account_init(memory, &dense, "dense", 0);
  memory_account_init(memory, &other, "other", 0);
  TestSlot core, a, b, fixed, wanted;
  test_slot(&core, &moe, NULL, 20, 1);
  core.slot.touched = -1000;
  test_slot(&a, &moe, &core, 8, 2);
  test_slot(&b, &moe, &core, 12, 3);
  test_slot(&fixed, &dense, NULL, 60, 4);
  memory_pin(&fixed.slot, 1);
  test_slot(&wanted, &other, NULL, 0, 5);
  REQUIRE(claim(&wanted, 40, MEMORY_REQUIRED));
  REQUIRE(a.evicted == 1 && b.evicted == 1 && core.evicted == 0);
  REQUIRE(memory_usage(memory).committed == 120);
  REQUIRE(claim(&wanted, 20, MEMORY_REQUIRED));
  REQUIRE(core.evicted == 1 && fixed.evicted == 0 && order_count == 3 &&
          order[2] == 1);
  REQUIRE(memory_usage(memory).allocated == 120);
}

static void dense_weights_are_one_slot(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 10, 2);
  REQUIRE(claim(&wanted, 25, MEMORY_REQUIRED));
  REQUIRE(dense.evicted == 1 && !memory_ready(&dense.slot));
  REQUIRE(dense.slot.bytes == 0 && wanted.slot.payload == 35);
}

static void pinned_slots_and_state_memory(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 1000);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 0, 2);
  memory_pin(&dense.slot, 1);
  MemorySlot state;
  memory_slot_init(&state, &b, MEMORY_STATE, NULL, "state", NULL, NULL);
  REQUIRE(memory_claim(&state, 10, MEMORY_REQUIRED));
  REQUIRE(claim(&wanted, 10, MEMORY_REQUIRED));
  REQUIRE(!claim(&wanted, 1, MEMORY_REQUIRED));
  REQUIRE(dense.evicted == 0 && wanted.slot.payload == 10);
  REQUIRE(memory_usage(memory).committed == 100 && b.state_bytes == 10);
  memory_release(&state, 10);
  REQUIRE(memory_usage(memory).allocated == 90 && b.state_bytes == 0);
}

static void state_memory_has_its_own_limit(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1000);
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 100);
  TestSlot weights, older, newer, running;
  test_slot(&weights, &model, NULL, 100, 1);
  test_state_slot(&older, &model, 40, 2);
  clock_now += 10;
  test_state_slot(&newer, &model, 40, 3);
  test_state_slot(&running, &model, 0, 4);
  memory_pin(&running.slot, 1);
  REQUIRE(claim(&running, 30, MEMORY_REQUIRED));
  REQUIRE(older.evicted == 1 && newer.evicted == 0 && weights.evicted == 0);
  REQUIRE(model.state_bytes == 70 && memory_usage(memory).committed == 170);
  memory_pin(&newer.slot, 1);
  REQUIRE(!claim(&running, 40, MEMORY_REQUIRED));
  REQUIRE(newer.evicted == 0 && model.state_bytes == 70);
}

static void loading_experts_never_evicts_state(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 100);
  TestSlot weights, saved, expert;
  test_slot(&weights, &model, NULL, 50, 1);
  memory_pin(&weights.slot, 1);
  test_state_slot(&saved, &model, 50, 2);
  test_slot(&expert, &model, &weights, 0, 3);
  REQUIRE(!claim(&expert, 10, MEMORY_REQUIRED));
  REQUIRE(saved.evicted == 0 && model.state_bytes == 50);
}

static void state_goes_after_its_model(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount first, second;
  memory_account_init(memory, &first, "first", 100);
  memory_account_init(memory, &second, "second", 0);
  TestSlot weights, saved, other;
  test_slot(&weights, &first, NULL, 40, 1);
  test_state_slot(&saved, &first, 40, 2);
  test_slot(&other, &second, NULL, 0, 3);
  REQUIRE(claim(&other, 50, MEMORY_REQUIRED));
  REQUIRE(weights.evicted == 1 && saved.evicted == 0);
  REQUIRE(claim(&other, 40, MEMORY_REQUIRED));
  REQUIRE(saved.evicted == 1 && order[0] == 1 && order[1] == 2);
}

static uint64_t fake_available(void) { return available_bytes; }

static void state_memory_keeps_the_reserve(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1000);
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 1000);
  TestSlot cache, state;
  test_slot(&cache, &model, NULL, 100, 1);
  available_bytes = 150;
  memory_set_reserve(memory, 100, fake_available, NULL);
  test_state_slot(&state, &model, 100, 2);
  REQUIRE(cache.evicted == 1 && model.state_bytes == 100);
}

static void current_limit_keeps_the_reserve(void) {
  Memory *memory = memory_new();
  REQUIRE(memory_current_limit(memory) == UINT64_MAX);
  memory_set_limit(memory, 1000);
  REQUIRE(memory_current_limit(memory) == 1000);
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  TestSlot weights;
  test_slot(&weights, &model, NULL, 100, 1);
  available_bytes = 300;
  memory_set_reserve(memory, 100, fake_available, NULL);
  // 100 bytes are counted, and 200 more fit above the reserve.
  REQUIRE(memory_current_limit(memory) == 300);
  // The reading is corrected for the 50 bytes added since.
  REQUIRE(claim(&weights, 50, MEMORY_REQUIRED));
  REQUIRE(memory_current_limit(memory) == 300);
  memory_set_limit(memory, 250);
  REQUIRE(memory_current_limit(memory) == 250);
  // A second later, less memory is available than the reserve. Then the
  // limit is what is already counted.
  available_bytes = 50;
  clock_now += 1;
  REQUIRE(memory_current_limit(memory) == 150);
}

static void unload_evicts_idle_state(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 100);
  TestSlot weights, saved, running;
  test_slot(&weights, &model, NULL, 50, 1);
  test_state_slot(&saved, &model, 20, 2);
  test_state_slot(&running, &model, 10, 3);
  memory_pin(&running.slot, 1);
  REQUIRE(memory_unload(&model) == 70);
  REQUIRE(weights.evicted == 1 && saved.evicted == 1 && running.evicted == 0);
  REQUIRE(order[0] == 1 && order[1] == 2 && model.state_bytes == 10);
  MemoryAccount busy;
  memory_account_init(memory, &busy, "busy", 100);
  TestSlot loaded, kept;
  test_slot(&loaded, &busy, NULL, 10, 4);
  memory_pin(&loaded.slot, 1);
  test_state_slot(&kept, &busy, 10, 5);
  REQUIRE(memory_unload(&busy) == 0 && kept.evicted == 0);
}

static void minimum_must_fit(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  REQUIRE(memory_set_minimum(&a, 60, MEMORY_REQUIRED));
  TestSlot fixed;
  test_slot(&fixed, &b, NULL, 40, 1);
  memory_pin(&fixed.slot, 1);
  REQUIRE(!memory_set_minimum(&a, 80, MEMORY_REQUIRED));
  REQUIRE(a.minimum == 60 && memory_usage(memory).committed == 100 &&
          fixed.evicted == 0);
}

static void room_counts_the_unused_minimum(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  REQUIRE(memory_set_minimum(&a, 40, MEMORY_REQUIRED));
  TestSlot used;
  test_slot(&used, &a, NULL, 10, 1);
  // 60 bytes are free, and 30 bytes of A's minimum are unused.
  REQUIRE(memory_room(&a) == 90 && memory_room(&b) == 60);
  // Over the limit, only A's unused minimum is left.
  memory_set_limit(memory, 20);
  REQUIRE(memory_room(&a) == 30 && memory_room(&b) == 0);
}

// A slot grows in an account at its minimum, and the reserve is short. Evicting
// another slot of the account lowers the counted bytes only to the minimum. It
// also lowers the growth. Together they make room for the reserve.
static void evictions_at_the_minimum_also_lower_the_growth(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  REQUIRE(memory_set_minimum(&model, 80, MEMORY_REQUIRED));
  TestSlot cache, kept, growing;
  test_slot(&cache, &model, NULL, 50, 1);
  test_slot(&kept, &model, NULL, 50, 2);
  memory_pin(&kept.slot, 1);
  test_slot(&growing, &model, NULL, 0, 3);
  // 100 bytes are counted, and 10 fit above the reserve. So adding 40 bytes is
  // 30 bytes short of the reserve.
  available_bytes = 110;
  memory_set_reserve(memory, 100, fake_available, NULL);
  REQUIRE(claim(&growing, 40, MEMORY_REQUIRED));
  // Evicting the cache lowered the counted bytes by 20, to the minimum, and
  // the growth by 30.
  REQUIRE(cache.evicted == 1 && memory_usage(memory).committed == 90);
}

// The minimum is the least that an account's model memory counts, and its
// state memory counts beside it. So state bytes leave the minimum's room to the
// model's own slots.
static void state_leaves_the_minimum_to_the_model(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount model, other;
  memory_account_init(memory, &model, "model", 100);
  memory_account_init(memory, &other, "other", 0);
  TestSlot weights, state, cache, wanted;
  test_slot(&weights, &model, NULL, 40, 1);
  memory_pin(&weights.slot, 1);
  test_state_slot(&state, &model, 30, 2);
  test_slot(&cache, &model, NULL, 0, 3);
  test_slot(&wanted, &other, NULL, 0, 4);
  // The minimum keeps 20 bytes for the cache beside the weights.
  REQUIRE(memory_set_minimum(&model, 60, MEMORY_REQUIRED));
  REQUIRE(memory_usage(memory).committed == 90);
  // Another account finds only the 10 bytes above them.
  REQUIRE(!claim(&wanted, 20, MEMORY_REQUIRED));
  REQUIRE(claim(&wanted, 10, MEMORY_REQUIRED));
  // The cache fills its 20 bytes without new room, also speculatively.
  REQUIRE(claim(&cache, 20, MEMORY_SPECULATIVE));
  REQUIRE(memory_usage(memory).committed == 100 && state.evicted == 0);
}

static void recently_used_weights_stay_for_their_time(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  memory_set_times(memory, (MemoryTimes){.keep_recently_used_weights = 10});
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 0, 2);
  clock_now += 5;
  REQUIRE(!claim(&wanted, 40, MEMORY_CACHE));
  clock_now += 5;
  REQUIRE(claim(&wanted, 40, MEMORY_CACHE));
  REQUIRE(dense.evicted == 1 && memory_times(memory).freed_memory_delay == 1.5);
}

static void readings_wait_for_their_interval(void) {
  Memory *memory = memory_new();
  memory_set_times(memory,
                   (MemoryTimes){.available_memory_reading_interval = 2});
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  available_bytes = 300;
  memory_set_reserve(memory, 100, fake_available, NULL);
  REQUIRE(memory_current_limit(memory) == 200);
  // Within the period the reading stays; after it, a new one is taken.
  available_bytes = 150;
  clock_now += 1;
  REQUIRE(memory_current_limit(memory) == 200);
  clock_now += 1;
  REQUIRE(memory_current_limit(memory) == 50);
}

static void minimum_waits_while_memory_comes_back(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  // 120 bytes are available and 100 stay for other programs: a minimum of 100
  // fits once 80 more come back, 50 at each sleep.
  available_bytes = 120, coming_back = 80, coming_back_step = 50;
  memory_set_reserve(memory, 100, fake_available, NULL);
  double started = clock_now;
  REQUIRE(memory_wait_for_minimum(&model, 100));
  REQUIRE(clock_now - started == 2 * memory_times(memory).freed_memory_delay);
}

static void minimum_fails_once_no_memory_comes_back(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  // The first sleep brings 30 bytes, too few, and the second none.
  available_bytes = 120, coming_back = 30, coming_back_step = 50;
  memory_set_reserve(memory, 100, fake_available, NULL);
  double started = clock_now;
  REQUIRE(!memory_wait_for_minimum(&model, 100));
  REQUIRE(clock_now - started == 2 * memory_times(memory).freed_memory_delay);
  REQUIRE(memory_usage(memory).committed == 0);
}

static void claim_waits_while_memory_comes_back(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  TestSlot slot;
  test_slot(&slot, &model, NULL, 0, 1);
  // 120 bytes are available and 100 stay for other programs: a claim of 100
  // fits once 80 more come back, 50 at each sleep.
  available_bytes = 120, coming_back = 80, coming_back_step = 50;
  memory_set_reserve(memory, 100, fake_available, NULL);
  double started = clock_now;
  REQUIRE(memory_wait_for_claim(&slot.slot, 100));
  REQUIRE(clock_now - started == 2 * memory_times(memory).freed_memory_delay);
  REQUIRE(slot.slot.payload == 100);
}

static void stop_ends_the_wait_for_a_minimum(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", 0);
  available_bytes = 120, coming_back = 80, coming_back_step = 50;
  memory_set_reserve(memory, 100, fake_available, NULL);
  shutdown_request();
  double started = clock_now;
  REQUIRE(!memory_wait_for_minimum(&model, 100) && clock_now == started);
}

static void speculative_memory_never_evicts(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 0, 2);
  clock_now += 20;
  REQUIRE(!claim(&wanted, 40, MEMORY_CACHE));
  clock_now += 40;
  REQUIRE(!claim(&wanted, 40, MEMORY_SPECULATIVE));
  REQUIRE(dense.evicted == 0);
  REQUIRE(claim(&wanted, 40, MEMORY_CACHE));
  REQUIRE(dense.evicted == 1 && memory_usage(memory).committed == 40);
}

static void refused_addition_evicts_nothing(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount a, b;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  TestSlot idle, busy, one, two;
  test_slot(&idle, &a, NULL, 30, 1);
  test_slot(&busy, &a, NULL, 60, 2);
  memory_pin(&busy.slot, 1);
  test_slot(&one, &b, NULL, 0, 3);
  test_slot(&two, &b, NULL, 0, 4);
  MemorySlot *slots[] = {&one.slot, &two.slot};
  uint64_t sizes[] = {25, 25};
  REQUIRE(!memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  REQUIRE(idle.evicted == 0 && memory_usage(memory).committed == 90);
  sizes[1] = 15;
  REQUIRE(memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  REQUIRE(idle.evicted == 1 && memory_usage(memory).committed == 100);
  REQUIRE(claim(&one, 25, MEMORY_REQUIRED) && claim(&two, 15, MEMORY_REQUIRED));
  REQUIRE(memory_usage(memory).allocated == 100);
}

// A can_evict function that keeps the slot unless the memory is needed.
static int keep_unless_needed(void *context, int needed) {
  (void)context;
  return needed;
}

static void required_memory_takes_kept_slots_last(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount cache, other;
  memory_account_init(memory, &cache, "cache", 0);
  memory_account_init(memory, &other, "other", 0);
  TestSlot kept, plain, wanted;
  test_slot(&kept, &cache, NULL, 50, 1);
  kept.slot.can_evict = keep_unless_needed;
  test_slot(&plain, &cache, NULL, 20, 2);
  // Both were used long ago, the kept one first: by age alone it goes first.
  kept.slot.touched = clock_now - 2000, plain.slot.touched = clock_now - 1000;
  test_slot(&wanted, &other, NULL, 0, 3);
  REQUIRE(!claim(&wanted, 60, MEMORY_CACHE) && order_count == 0);
  REQUIRE(claim(&wanted, 60, MEMORY_REQUIRED));
  REQUIRE(order_count == 2 && order[0] == 2 && order[1] == 1);
  REQUIRE(wanted.slot.payload == 60 && memory_usage(memory).committed == 60);
}

// A new state's memory takes the cache slots that their owner keeps, when
// nothing else makes room for it above the reserve.
static void a_new_state_takes_kept_slots_for_the_reserve(void) {
  Memory *memory = memory_new();
  MemoryAccount model;
  memory_account_init(memory, &model, "model", UINT64_MAX);
  TestSlot weights, kept, state;
  test_slot(&weights, &model, NULL, 50, 1);
  memory_pin(&weights.slot, 1);
  test_slot(&kept, &model, &weights, 30, 2);
  kept.slot.can_evict = keep_unless_needed;
  memory_slot_init(&state.slot, &model, MEMORY_STATE, NULL, "state", NULL,
                   NULL);
  // 80 bytes are counted, and nothing more fits above the reserve.
  available_bytes = 100;
  memory_set_reserve(memory, 100, fake_available, NULL);
  REQUIRE(claim(&state, 10, MEMORY_REQUIRED));
  REQUIRE(kept.evicted == 1 && state.slot.payload == 10);
}

static void workspace_evicts_its_models_cache(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount account;
  memory_account_init(memory, &account, "model", 0);
  TestSlot work, cache;
  test_slot(&work, &account, NULL, 40, 1);
  memory_pin(&work.slot, 1);
  test_slot(&cache, &account, NULL, 50, 2);
  REQUIRE(memory_set_minimum(&account, 100, MEMORY_REQUIRED));
  REQUIRE(claim(&work, 20, MEMORY_REQUIRED));
  REQUIRE(cache.evicted == 1 && work.slot.payload == 60);
  REQUIRE(memory_usage(memory).committed == 100 &&
          memory_usage(memory).allocated == 60);
}

static void host_memory_counts_toward_the_limit(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount account;
  memory_account_init(memory, &account, "model", 1000);
  MemorySlot work;
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "workspace", NULL,
                   NULL);
  work.automatic = 1;
  MemorySlot saved;
  memory_slot_init(&saved, &account, MEMORY_STATE, NULL, "saved", NULL, NULL);
  void *host = memory_host_alloc(&work, 60),
       *state = memory_host_alloc(&saved, 20);
  REQUIRE(host && state && memory_usage(memory).allocated == 80 &&
          account.state_bytes == 20);
  REQUIRE(memory_host_alloc(&work, 30) == NULL);
  REQUIRE(memory_usage(memory).allocated == 80);
  void *more = memory_host_alloc(&work, 20);
  REQUIRE(more && memory_usage(memory).committed == 100);
  memory_host_free(&saved, state);
  memory_host_free(&work, host);
  memory_host_free(&work, more);
  REQUIRE(memory_usage(memory).allocated == 0 && account.state_bytes == 0 &&
          memory_usage(memory).committed == 0);
}

static void host_memory_without_a_slot(void) { memory_host_alloc(NULL, 60); }

static void host_memory_freed_with_another_slot(void) {
  Memory *memory = memory_new();
  MemoryAccount account;
  memory_account_init(memory, &account, "model", 0);
  MemorySlot first, second;
  memory_slot_init(&first, &account, MEMORY_MODEL, NULL, "first", NULL, NULL);
  memory_slot_init(&second, &account, MEMORY_MODEL, NULL, "second", NULL, NULL);
  memory_host_free(&second, memory_host_alloc(&first, 60));
}

// Starts a workspace slot in a new Memory with a limit of 100 bytes.
static void workspace_in_100_bytes(MemoryAccount *account, MemorySlot *work) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  memory_account_init(memory, account, "model", 0);
  memory_slot_init(work, account, MEMORY_MODEL, NULL, "workspace", NULL, NULL);
}

static void required_host_memory_is_counted(void) {
  MemoryAccount account;
  MemorySlot work;
  workspace_in_100_bytes(&account, &work);
  unsigned char *bytes = memory_host_alloc_required(&work, 100);
  for (int i = 0; i < 100; i++) REQUIRE(bytes[i] == 0);
  REQUIRE(memory_usage(account.memory).allocated == 100);
  memory_host_free(&work, bytes);
  REQUIRE(memory_usage(account.memory).allocated == 0);
}

static void required_host_memory_that_does_not_fit(void) {
  MemoryAccount account;
  MemorySlot work;
  workspace_in_100_bytes(&account, &work);
  memory_host_alloc_required(&work, 101);
}

static void reclaim_evicts_idle_slots(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 120);
  MemoryAccount moe, dense;
  memory_account_init(memory, &moe, "MoE", 0);
  memory_account_init(memory, &dense, "dense", 0);
  TestSlot core, a, b, fixed;
  test_slot(&core, &moe, NULL, 20, 1);
  core.slot.touched = -1000;
  test_slot(&a, &moe, &core, 8, 2);
  test_slot(&b, &moe, &core, 12, 3);
  test_slot(&fixed, &dense, NULL, 60, 4);
  memory_pin(&fixed.slot, 1);
  uint64_t peak = memory_usage(memory).peak;
  REQUIRE(memory_reclaim(memory, 19) == 20);
  REQUIRE(a.evicted == 1 && b.evicted == 1 && core.evicted == 0);
  REQUIRE(memory_reclaim(memory, 100) == 20);
  REQUIRE(core.evicted == 1 && fixed.evicted == 0 && order[2] == 1);
  REQUIRE(memory_usage(memory).allocated == 60 &&
          memory_usage(memory).peak == peak);
  REQUIRE(memory_reclaim(memory, 1) == 0);
}

static void unload_evicts_one_accounts_idle_slots(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount moe, other;
  memory_account_init(memory, &moe, "MoE", 0);
  memory_account_init(memory, &other, "other", 0);
  TestSlot core, a, b, held, kept;
  test_slot(&core, &moe, NULL, 20, 1);
  test_slot(&a, &moe, &core, 8, 2);
  test_slot(&b, &moe, &core, 12, 3);
  test_slot(&held, &moe, NULL, 5, 4);
  memory_pin(&held.slot, 1);
  test_slot(&kept, &other, NULL, 30, 5);
  REQUIRE(memory_unload(&moe) == 40);
  REQUIRE(a.evicted == 1 && b.evicted == 1 && core.evicted == 1);
  // The weights after the experts on them.
  REQUIRE(order_count == 3 && order[2] == 1);
  REQUIRE(held.evicted == 0 && kept.evicted == 0 &&
          memory_usage(memory).allocated == 35);
  REQUIRE(memory_unload(&moe) == 0);
}

static void limit_holds_for_new_accounts(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  MemoryAccount first, second;
  memory_account_init(memory, &first, "first", 0);
  TestSlot cached;
  test_slot(&cached, &first, NULL, 40, 1);
  REQUIRE(memory_reclaim(memory, 40) == 40);
  memory_set_limit(memory, 0);
  memory_account_init(memory, &second, "second", 0);
  REQUIRE(memory_usage(memory).limit == 0 &&
          memory_usage(memory).allocated == 0);
  REQUIRE(!claim(&cached, 1, MEMORY_CACHE));
  memory_set_limit(memory, 100);
  REQUIRE(claim(&cached, 40, MEMORY_REQUIRED));
  REQUIRE(memory_usage(memory).limit == 100 &&
          memory_usage(memory).allocated == 40);
}

static void memories_count_separately(void) {
  Memory *first = memory_new(), *second = memory_new();
  memory_set_limit(first, 100);
  MemoryAccount a, b;
  memory_account_init(first, &a, "A", 0);
  memory_account_init(second, &b, "B", 0);
  TestSlot held, other;
  test_slot(&held, &a, NULL, 80, 1);
  test_slot(&other, &b, NULL, 80, 2);
  REQUIRE(held.evicted == 0 && memory_usage(first).allocated == 80);
  REQUIRE(memory_usage(second).allocated == 80 &&
          memory_usage(second).limit == UINT64_MAX);
}

static void count_callback(void) { callbacks++; }

static void callback_runs_after_evictions(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 100);
  memory_set_eviction_callback(memory, count_callback);
  MemoryAccount a, b, s;
  memory_account_init(memory, &a, "A", 0);
  memory_account_init(memory, &b, "B", 0);
  memory_account_init(memory, &s, "S", 10);
  TestSlot first, second, wanted, older, newer;
  test_slot(&first, &a, NULL, 40, 1);
  test_slot(&second, &a, NULL, 40, 2);
  test_slot(&wanted, &b, NULL, 20, 3);
  memory_pin(&wanted.slot, 1);
  REQUIRE(callbacks == 0);
  REQUIRE(claim(&wanted, 30, MEMORY_REQUIRED) && callbacks == 1);
  REQUIRE(memory_reclaim(memory, 40) == 40 && callbacks == 2);
  REQUIRE(memory_reclaim(memory, 40) == 0 && callbacks == 2);
  memory_pin(&wanted.slot, 0);
  REQUIRE(memory_unload(&b) == 50 && callbacks == 3);
  test_state_slot(&older, &s, 10, 4);
  test_state_slot(&newer, &s, 0, 5);
  REQUIRE(claim(&newer, 10, MEMORY_REQUIRED) && older.evicted == 1);
  REQUIRE(callbacks == 4);
}

// Starts accounts a and b, each in a Memory of its own.
static void accounts_in_two_memories(MemoryAccount *a, MemoryAccount *b) {
  memory_account_init(memory_new(), a, "A", 0);
  memory_account_init(memory_new(), b, "B", 0);
}

static void dependency_in_another_memory(void) {
  MemoryAccount a, b;
  accounts_in_two_memories(&a, &b);
  MemorySlot weights, expert;
  memory_slot_init(&weights, &a, MEMORY_MODEL, NULL, "weights", NULL, NULL);
  memory_slot_init(&expert, &b, MEMORY_MODEL, &weights, "expert", NULL, NULL);
}

static void slots_of_two_memories_admitted(void) {
  MemoryAccount a, b;
  accounts_in_two_memories(&a, &b);
  TestSlot one, two;
  test_slot(&one, &a, NULL, 0, 1);
  test_slot(&two, &b, NULL, 0, 2);
  MemorySlot *slots[] = {&one.slot, &two.slot};
  uint64_t sizes[] = {10, 10};
  memory_admit(slots, sizes, 2, MEMORY_REQUIRED);
}

// Runs a case in its own process. Returns its exit status, or -1 if it did not
// exit.
static int exit_status(void (*run)(void)) {
  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    run();
    _exit(0);
  }
  int status;
  REQUIRE(child > 0 && waitpid(child, &status, 0) == child);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main(void) {
  struct {
    const char *name;
    void (*run)(void);
  } cases[] = {
      {"slots that depend on others are evicted first",
       dependents_are_evicted_first},
      {"dense weights are one slot", dense_weights_are_one_slot},
      {"pinned slots stay, and state memory counts toward the limit",
       pinned_slots_and_state_memory},
      {"state memory has a limit of its own, and its oldest idle slot goes "
       "first",
       state_memory_has_its_own_limit},
      {"loading experts never evicts state memory",
       loading_experts_never_evicts_state},
      {"state memory goes only after the rest of its model",
       state_goes_after_its_model},
      {"state memory evicts model memory to keep the reserve free",
       state_memory_keeps_the_reserve},
      {"the current limit keeps the reserve free and stays within the limit",
       current_limit_keeps_the_reserve},
      {"unloading a model evicts its idle state slots last",
       unload_evicts_idle_state},
      {"a minimum that does not fit is refused", minimum_must_fit},
      {"an account's room is the free bytes plus its unused minimum",
       room_counts_the_unused_minimum},
      {"evictions at an account's minimum also lower the growth, and so keep "
       "the reserve",
       evictions_at_the_minimum_also_lower_the_growth},
      {"an account's minimum keeps room for its model memory beside its state "
       "memory",
       state_leaves_the_minimum_to_the_model},
      {"speculative memory never evicts, and a cache waits 30 seconds to evict "
       "weights",
       speculative_memory_never_evicts},
      {"a cache waits keep_recently_used_weights to evict weights that were "
       "used, and the other times keep their defaults",
       recently_used_weights_stay_for_their_time},
      {"a reading is taken again only after available_memory_reading_interval",
       readings_wait_for_their_interval},
      {"a minimum that does not fit waits while memory comes back, and fits "
       "once enough is back",
       minimum_waits_while_memory_comes_back},
      {"a minimum fails once a reading shows no more available memory",
       minimum_fails_once_no_memory_comes_back},
      {"a stop request ends the wait for a minimum",
       stop_ends_the_wait_for_a_minimum},
      {"a required claim waits while memory comes back, and fits once enough "
       "is back",
       claim_waits_while_memory_comes_back},
      {"a refused addition to two slots evicts nothing",
       refused_addition_evicts_nothing},
      {"slots that their owners keep go only for required memory, after the "
       "others",
       required_memory_takes_kept_slots_last},
      {"a new state takes slots that their owners keep, to keep the reserve",
       a_new_state_takes_kept_slots_for_the_reserve},
      {"a workspace can evict its own model's cache",
       workspace_evicts_its_models_cache},
      {"host memory and state memory count toward the limit",
       host_memory_counts_toward_the_limit},
      {"required host memory is zeroed and counted in its slot",
       required_host_memory_is_counted},
      {"reclaiming evicts idle slots, dependents first",
       reclaim_evicts_idle_slots},
      {"unloading evicts one account's idle slots",
       unload_evicts_one_accounts_idle_slots},
      {"a limit holds for accounts added later", limit_holds_for_new_accounts},
      {"two Memory objects keep separate counts and limits",
       memories_count_separately},
      {"the eviction callback runs after evictions, and only then",
       callback_runs_after_evictions},
  };

  for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
    check(exit_status(cases[i].run) == 0, cases[i].name);
  // This test's die exits with status 2.
  check(exit_status(host_memory_without_a_slot) == 2,
        "host memory without a slot ends the program");
  check(exit_status(host_memory_freed_with_another_slot) == 2,
        "host memory freed with another slot ends the program");
  check(exit_status(required_host_memory_that_does_not_fit) == 2,
        "required host memory that does not fit ends the program");
  check(exit_status(dependency_in_another_memory) == 2,
        "a slot that depends on a slot of another Memory ends the program");
  check(exit_status(slots_of_two_memories_admitted) == 2,
        "admitting slots of two Memory objects together ends the program");
  return failures != 0;
}
