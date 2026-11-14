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

static double clock_now = 100;
double now(void) { return clock_now; }
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
static int order[16], order_count;
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
  memory_set_limit(120);
  MemoryAccount moe, dense, other;
  memory_account_init(&moe, "MoE", 0), memory_account_init(&dense, "dense", 0),
      memory_account_init(&other, "other", 0);
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
  REQUIRE(memory_usage().committed == 120);
  REQUIRE(claim(&wanted, 20, MEMORY_REQUIRED));
  REQUIRE(core.evicted == 1 && fixed.evicted == 0 && order_count == 3 &&
          order[2] == 1);
  REQUIRE(memory_usage().allocated == 120);
}

static void dense_weights_are_one_slot(void) {
  memory_set_limit(100);
  MemoryAccount a, b;
  memory_account_init(&a, "A", 0), memory_account_init(&b, "B", 0);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 10, 2);
  REQUIRE(claim(&wanted, 25, MEMORY_REQUIRED));
  REQUIRE(dense.evicted == 1 && !memory_ready(&dense.slot));
  REQUIRE(dense.slot.bytes == 0 && wanted.slot.payload == 35);
}

static void pinned_slots_and_state_memory(void) {
  memory_set_limit(100);
  MemoryAccount a, b;
  memory_account_init(&a, "A", 0), memory_account_init(&b, "B", 1000);
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
  REQUIRE(memory_usage().committed == 100 && b.state_bytes == 10);
  memory_release(&state, 10);
  REQUIRE(memory_usage().allocated == 90 && b.state_bytes == 0);
}

static void state_memory_has_its_own_limit(void) {
  memory_set_limit(1000);
  MemoryAccount model;
  memory_account_init(&model, "model", 100);
  TestSlot weights, older, newer, running;
  test_slot(&weights, &model, NULL, 100, 1);
  test_state_slot(&older, &model, 40, 2);
  clock_now += 10;
  test_state_slot(&newer, &model, 40, 3);
  test_state_slot(&running, &model, 0, 4);
  memory_pin(&running.slot, 1);
  REQUIRE(claim(&running, 30, MEMORY_REQUIRED));
  REQUIRE(older.evicted == 1 && newer.evicted == 0 && weights.evicted == 0);
  REQUIRE(model.state_bytes == 70 && memory_usage().committed == 170);
  memory_pin(&newer.slot, 1);
  REQUIRE(!claim(&running, 40, MEMORY_REQUIRED));
  REQUIRE(newer.evicted == 0 && model.state_bytes == 70);
}

static void loading_experts_never_evicts_state(void) {
  memory_set_limit(100);
  MemoryAccount model;
  memory_account_init(&model, "model", 100);
  TestSlot weights, saved, expert;
  test_slot(&weights, &model, NULL, 50, 1);
  memory_pin(&weights.slot, 1);
  test_state_slot(&saved, &model, 50, 2);
  test_slot(&expert, &model, &weights, 0, 3);
  REQUIRE(!claim(&expert, 10, MEMORY_REQUIRED));
  REQUIRE(saved.evicted == 0 && model.state_bytes == 50);
}

static void state_goes_after_its_model(void) {
  memory_set_limit(100);
  MemoryAccount first, second;
  memory_account_init(&first, "first", 100);
  memory_account_init(&second, "second", 0);
  TestSlot weights, saved, other;
  test_slot(&weights, &first, NULL, 40, 1);
  test_state_slot(&saved, &first, 40, 2);
  test_slot(&other, &second, NULL, 0, 3);
  REQUIRE(claim(&other, 50, MEMORY_REQUIRED));
  REQUIRE(weights.evicted == 1 && saved.evicted == 0);
  REQUIRE(claim(&other, 40, MEMORY_REQUIRED));
  REQUIRE(saved.evicted == 1 && order[0] == 1 && order[1] == 2);
}

// The available memory that the system reports to the reserve.
static uint64_t available_bytes;
static uint64_t fake_available(void) { return available_bytes; }

static void state_memory_keeps_the_reserve(void) {
  memory_set_limit(1000);
  MemoryAccount model;
  memory_account_init(&model, "model", 1000);
  TestSlot cache, state;
  test_slot(&cache, &model, NULL, 100, 1);
  available_bytes = 150;
  memory_set_reserve(100, fake_available, NULL);
  test_state_slot(&state, &model, 100, 2);
  REQUIRE(cache.evicted == 1 && model.state_bytes == 100);
}

static void unload_evicts_idle_state(void) {
  MemoryAccount model;
  memory_account_init(&model, "model", 100);
  TestSlot weights, saved, running;
  test_slot(&weights, &model, NULL, 50, 1);
  test_state_slot(&saved, &model, 20, 2);
  test_state_slot(&running, &model, 10, 3);
  memory_pin(&running.slot, 1);
  REQUIRE(memory_unload(&model) == 70);
  REQUIRE(weights.evicted == 1 && saved.evicted == 1 && running.evicted == 0);
  REQUIRE(order[0] == 1 && order[1] == 2 && model.state_bytes == 10);
  MemoryAccount busy;
  memory_account_init(&busy, "busy", 100);
  TestSlot loaded, kept;
  test_slot(&loaded, &busy, NULL, 10, 4);
  memory_pin(&loaded.slot, 1);
  test_state_slot(&kept, &busy, 10, 5);
  REQUIRE(memory_unload(&busy) == 0 && kept.evicted == 0);
}

static void minimum_must_fit(void) {
  memory_set_limit(100);
  MemoryAccount a, b;
  memory_account_init(&a, "A", 0), memory_account_init(&b, "B", 0);
  REQUIRE(memory_set_minimum(&a, 60, MEMORY_REQUIRED));
  TestSlot fixed;
  test_slot(&fixed, &b, NULL, 40, 1);
  memory_pin(&fixed.slot, 1);
  REQUIRE(!memory_set_minimum(&a, 80, MEMORY_REQUIRED));
  REQUIRE(a.minimum == 60 && memory_usage().committed == 100 &&
          fixed.evicted == 0);
}

static void speculative_memory_never_evicts(void) {
  memory_set_limit(100);
  MemoryAccount a, b;
  memory_account_init(&a, "A", 0), memory_account_init(&b, "B", 0);
  TestSlot dense, wanted;
  test_slot(&dense, &a, NULL, 80, 1);
  test_slot(&wanted, &b, NULL, 0, 2);
  clock_now += 20;
  REQUIRE(!claim(&wanted, 40, MEMORY_CACHE));
  clock_now += 40;
  REQUIRE(!claim(&wanted, 40, MEMORY_SPECULATIVE));
  REQUIRE(dense.evicted == 0);
  REQUIRE(claim(&wanted, 40, MEMORY_CACHE));
  REQUIRE(dense.evicted == 1 && memory_usage().committed == 40);
}

static void refused_addition_evicts_nothing(void) {
  memory_set_limit(100);
  MemoryAccount a, b;
  memory_account_init(&a, "A", 0), memory_account_init(&b, "B", 0);
  TestSlot idle, busy, one, two;
  test_slot(&idle, &a, NULL, 30, 1);
  test_slot(&busy, &a, NULL, 60, 2);
  memory_pin(&busy.slot, 1);
  test_slot(&one, &b, NULL, 0, 3);
  test_slot(&two, &b, NULL, 0, 4);
  MemorySlot *slots[] = {&one.slot, &two.slot};
  uint64_t sizes[] = {25, 25};
  REQUIRE(!memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  REQUIRE(idle.evicted == 0 && memory_usage().committed == 90);
  sizes[1] = 15;
  REQUIRE(memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  REQUIRE(idle.evicted == 1 && memory_usage().committed == 100);
  REQUIRE(claim(&one, 25, MEMORY_REQUIRED) && claim(&two, 15, MEMORY_REQUIRED));
  REQUIRE(memory_usage().allocated == 100);
}

static void workspace_evicts_its_models_cache(void) {
  memory_set_limit(100);
  MemoryAccount account;
  memory_account_init(&account, "model", 0);
  TestSlot work, cache;
  test_slot(&work, &account, NULL, 40, 1);
  memory_pin(&work.slot, 1);
  test_slot(&cache, &account, NULL, 50, 2);
  REQUIRE(memory_set_minimum(&account, 100, MEMORY_REQUIRED));
  REQUIRE(claim(&work, 20, MEMORY_REQUIRED));
  REQUIRE(cache.evicted == 1 && work.slot.payload == 60);
  REQUIRE(memory_usage().committed == 100 && memory_usage().allocated == 60);
}

static void host_memory_counts_toward_the_limit(void) {
  memory_set_limit(100);
  MemoryAccount account;
  memory_account_init(&account, "model", 1000);
  MemorySlot work;
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "workspace", NULL,
                   NULL);
  work.automatic = 1;
  MemorySlot saved;
  memory_slot_init(&saved, &account, MEMORY_STATE, NULL, "saved", NULL, NULL);
  void *host = memory_host_alloc(&work, 60),
       *state = memory_host_alloc(&saved, 20);
  REQUIRE(host && state && memory_usage().allocated == 80 &&
          account.state_bytes == 20);
  REQUIRE(memory_host_alloc(&work, 30) == NULL);
  REQUIRE(memory_usage().allocated == 80);
  void *more = memory_host_alloc(&work, 20);
  REQUIRE(more && memory_usage().committed == 100);
  memory_host_free(state);
  memory_host_free(host);
  memory_host_free(more);
  REQUIRE(memory_usage().allocated == 0 && account.state_bytes == 0 &&
          memory_usage().committed == 0);
}

static void host_memory_without_a_slot(void) { memory_host_alloc(NULL, 60); }

static void reclaim_evicts_idle_slots(void) {
  memory_set_limit(120);
  MemoryAccount moe, dense;
  memory_account_init(&moe, "MoE", 0);
  memory_account_init(&dense, "dense", 0);
  TestSlot core, a, b, fixed;
  test_slot(&core, &moe, NULL, 20, 1);
  core.slot.touched = -1000;
  test_slot(&a, &moe, &core, 8, 2);
  test_slot(&b, &moe, &core, 12, 3);
  test_slot(&fixed, &dense, NULL, 60, 4);
  memory_pin(&fixed.slot, 1);
  uint64_t peak = memory_usage().peak;
  REQUIRE(memory_reclaim(19) == 20);
  REQUIRE(a.evicted == 1 && b.evicted == 1 && core.evicted == 0);
  REQUIRE(memory_reclaim(100) == 20);
  REQUIRE(core.evicted == 1 && fixed.evicted == 0 && order[2] == 1);
  REQUIRE(memory_usage().allocated == 60 && memory_usage().peak == peak);
  REQUIRE(memory_reclaim(1) == 0);
}

static void unload_evicts_one_accounts_idle_slots(void) {
  memory_set_limit(100);
  MemoryAccount moe, other;
  memory_account_init(&moe, "MoE", 0), memory_account_init(&other, "other", 0);
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
          memory_usage().allocated == 35);
  REQUIRE(memory_unload(&moe) == 0);
}

static void limit_holds_for_new_accounts(void) {
  memory_set_limit(100);
  MemoryAccount first, second;
  memory_account_init(&first, "first", 0);
  TestSlot cached;
  test_slot(&cached, &first, NULL, 40, 1);
  REQUIRE(memory_reclaim(40) == 40);
  memory_set_limit(0);
  memory_account_init(&second, "second", 0);
  REQUIRE(memory_usage().limit == 0 && memory_usage().allocated == 0);
  REQUIRE(!claim(&cached, 1, MEMORY_CACHE));
  memory_set_limit(100);
  REQUIRE(claim(&cached, 40, MEMORY_REQUIRED));
  REQUIRE(memory_usage().limit == 100 && memory_usage().allocated == 40);
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
      {"unloading a model evicts its idle state slots last",
       unload_evicts_idle_state},
      {"a minimum that does not fit is refused", minimum_must_fit},
      {"speculative memory never evicts, and a cache waits 30 seconds to evict "
       "weights",
       speculative_memory_never_evicts},
      {"a refused addition to two slots evicts nothing",
       refused_addition_evicts_nothing},
      {"a workspace can evict its own model's cache",
       workspace_evicts_its_models_cache},
      {"host memory and state memory count toward the limit",
       host_memory_counts_toward_the_limit},
      {"reclaiming evicts idle slots, dependents first",
       reclaim_evicts_idle_slots},
      {"unloading evicts one account's idle slots",
       unload_evicts_one_accounts_idle_slots},
      {"a limit holds for accounts added later", limit_holds_for_new_accounts},
  };
  for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
    check(exit_status(cases[i].run) == 0, cases[i].name);
  // This test's die exits with status 2.
  check(exit_status(host_memory_without_a_slot) == 2,
        "host memory without a slot ends the program");
  return failures != 0;
}
