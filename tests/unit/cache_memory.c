#include "check.h"
#include "engine/expert_store.h"
#include "platform/threads.h"
#include <mach/mach.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the cache's initialization and resizing against page counters,
// pressure, footprint and working set that the test sets, and the slot bytes
// that reads have not written yet. No model weights, readers or GPU commands
// are made. Each resizing case runs in a child process.

static vm_statistics64_data_t vm_fixture;
static int pressure_fixture = 1;
static uint64_t compressed_fixture, wakeups_fixture,
    working_set_fixture = 128ull << 30;

static kern_return_t fixture_host(host_t host, host_flavor_t flavor,
                                  host_info64_t out,
                                  mach_msg_type_number_t *n) {
  (void)host, (void)flavor, (void)n;
  memcpy(out, &vm_fixture, sizeof vm_fixture);
  return KERN_SUCCESS;
}

static kern_return_t fixture_task(task_name_t task, task_flavor_t flavor,
                                  task_info_t out, mach_msg_type_number_t *n) {
  (void)task, (void)flavor, (void)n;
  task_vm_info_data_t value = {.phys_footprint = 8ull << 30,
                               .compressed = compressed_fixture};
  memcpy(out, &value, sizeof value);
  return KERN_SUCCESS;
}

static int fixture_sysctl(const char *name, void *out, size_t *n, void *in,
                          size_t len) {
  (void)in, (void)len;
  uint64_t value = !strcmp(name, "kern.memorystatus_vm_pressure_level")
                       ? (uint64_t)pressure_fixture
                   : !strcmp(name, "vm.compressor.swapper.wakeups.total")
                       ? wakeups_fixture
                       : 0;
  if (!out || !n) return -1;
  size_t size = *n < sizeof value ? *n : sizeof value;
  memcpy(out, &value, size), *n = size;
  return 0;
}

static uint64_t fixture_working_set(Gpu *gpu) {
  (void)gpu;
  return working_set_fixture;
}

static void fixture_residency(Gpu *gpu) { (void)gpu; }

static void fixture_spawn(ThreadsPriority priority, void *(*fn)(void *),
                          void *context) {
  (void)priority, (void)fn, (void)context;
}

#define host_statistics64 fixture_host
#define task_info fixture_task
#define sysctlbyname fixture_sysctl
#define gpu_recommended_working_set fixture_working_set
#define gpu_commit_residency fixture_residency
#define threads_spawn fixture_spawn
#include "engine/expert_store.c"
#include "platform/system_memory.c"
#include "store_layout.h"
#undef host_statistics64
#undef task_info
#undef sysctlbyname
#undef gpu_recommended_working_set
#undef gpu_commit_residency
#undef threads_spawn

static void child_case(int kind) {
  vm_fixture.free_count = (1ull << 30) / vm_page_size;
  vm_fixture.external_page_count = (20ull << 30) / vm_page_size;
  vm_fixture.speculative_count = 0;
  st->nslots = st->cap = st->max_slots - 256;
  if (kind == 7) st->nslots = 0;
  if (kind == 8) st->nslots = st->cap = 2 * EXPERTS;
  if (kind >= 9 && kind <= 12)
    vm_fixture.external_page_count =
        (8ull << 30) / vm_page_size;  // below the 10 GiB kept
  if (kind >= 10 && kind <= 12)
    vm_fixture.free_count =
        (15ull << 28) /
        vm_page_size;  // 3.75 GiB: 0.25 GiB short of the reserve
  if (kind == 12) {  // a store that starts with less available memory than the
                     // reserve and its minimum
    ExpertStore *other = store_create();
    expert_store_set_pass_rows(other, 1);
    expert_store_start(other, 0, LAYERS);
    _exit(other->cap == other->minimum_slots ? 0 : 1);
  }
  if (kind == 11) {  // a second model's store beside the first, the same
                     // shortfall for both
    ExpertStore *other = store_create();
    expert_store_start(other, 56ull << 30, LAYERS);
    other->nslots = other->cap = other->max_slots - 256;
    uint64_t alone = cache_unwritten();
    other->slots[0].b =
        (Buf){.obj = &other->slots[0],
              .allocated = 1ull << 20};  // counted, never written
    int counted = cache_unwritten() == alone + (1ull << 20);
    other->slots[0].b = (Buf){0};
    int first = st->cap, second = other->cap;
    resize(st, -1);
    resize(other,
           -1);  // at once: the memory the first gave back cannot show yet
    int expected =
        first - (int)(((4ull << 30) - (15ull << 28)) / st->SLOT_SIZE) - 1;
    printf("two stores 0.25 GiB short of the reserve: limits %d -> %d and %d "
           "-> %d (expected %d and %d)\n",
           first, st->cap, second, other->cap, expected, second);
    _exit(counted && st->cap == expected && other->cap == second ? 0 : 1);
  }
  int before = st->cap;
  if (kind == 2) pressure_fixture = 2;
  resize(st, -1);
  if (kind ==
      10) {  // the slots that restore the reserve, not a sixteenth of the cache
    int expected =
        before - (int)(((4ull << 30) - (15ull << 28)) / st->SLOT_SIZE) - 1;
    printf("0.25 GiB short of the reserve, no pressure: limit %d -> %d "
           "(expected %d)\n",
           before, st->cap, expected);
    _exit(st->cap == expected ? 0 : 1);
  }
  if (kind == 0) _exit(st->cap == before ? 0 : 1);
  if (kind == 1) {
    for (int i = 0; i < 32; i++) resize(st, -1);
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind == 3 || kind == 5) {  // no file cache above the one kept counts
    vm_fixture.free_count = (11ull << 30) / vm_page_size;
    vm_fixture.external_page_count = (10ull << 30) / vm_page_size;
  }
  if (kind == 3) vm_fixture.compressions++, compressed_fixture = 0;
  if (kind == 13) {  // unrelated system-wide compression while file cache
                     // above the floor counts: floor stays at 10 GiB and cache
                     // does NOT shrink
    vm_fixture.compressions++;
    resize(st, -1);
    int kept = memory_kept_file_cache(cache_memory) == 10ull << 30;
    _exit(st->cap == before && kept ? 0 : 1);
  }
  if (kind == 4) compressed_fixture = 128ull << 20;
  if (kind == 6) wakeups_fixture++;
  if (kind == 7 || kind == 8) compressed_fixture = 128ull << 20;
  if (kind == 5) {
    for (int i = 0; i < 32; i++) {
      vm_fixture.compressions++;
      resize(st, -1);
    }
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind >= 3) resize(st, -1);
  if (kind == 6 || kind == 7) _exit(st->cap == before ? 0 : 1);
  if (kind == 8) {
    for (int i = 0; i < 32; i++) resize(st, -1);
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind == 3) _exit(st->cap == before ? 0 : 1);
  _exit(st->cap < before ? 0 : 1);
}

// macOS counts a slot's pages as in use only once a read writes them, so the
// cache reports the counted bytes that no finished load has written. Layer
// 1's experts here are smaller than the slot.
static void reads(Slot *s, int from,
                  int to) {  // a reader completes pieces from..to-1
  for (int k = from; k < to; k++)
    atomic_store(&s->taken[k], 1), piece_finished(st, s, 1);
}

static void unwritten_checks(void) {
  for (int e = 0; e < EXPERTS; e++) expert_span[1][e][0].len = EXPERT_W / 2;
  Slot *s = &st->slots[0];
  uint64_t whole = gpu_round_up_to_pages(st->SLOT_SIZE);
  s->b = (Buf){.obj = s,
               .ptr = malloc(whole),
               .allocated = whole};  // a new buffer, never touched
  s->layer = 1, s->expert = 0;
  start(st, 0, 0);
  check(cache_unwritten() == whole,
        "a new slot counts as unwritten while its read is under way");
  reads(s, 0, s->npiece - 1);
  check(cache_unwritten() == whole,
        "a partly read slot still counts as unwritten");
  reads(s, s->npiece - 1, s->npiece);
  uint64_t small = s->extent;
  check(small && small < whole && cache_unwritten() == whole - small,
        "a completed read counts as written before the engine thread reaps it");
  finish(st, 0);
  check(
      s->written == small && cache_unwritten() == whole - small,
      "a reaped read leaves unwritten only the slot beyond its smaller expert");
  s->layer = 0;
  start(st, 0, 0);
  reads(s, 0, 1);
  cancel(st, 0);
  check(s->written == small && cache_unwritten() == whole - small,
        "a load that dropped a piece adds nothing written");
  start(st, 0, 0);
  reads(s, 0, s->npiece);
  finish(st, 0);
  check(s->written == s->extent && cache_unwritten() == whole - s->extent &&
            s->extent > small,
        "a larger expert's completed read writes the rest of the slot");
  free(s->b.ptr);
  memset(s, 0, sizeof *s);
}

int main(void) {
  st = store_create();
  working_set_fixture = 0;
  check(gpu_target(st) == INT64_MAX,
        "a GPU that is not open gives no residency target");
  working_set_fixture = 100ull << 30;
  check(gpu_target(st) == (int64_t)working_set_fixture,
        "the residency target is Metal's recommended working set");
  working_set_fixture = 128ull << 30;
  check(gpu_target(st) == (int64_t)working_set_fixture,
        "an early reading does not freeze the residency target");
  // The memory reads the file cache that the kernel keeps: 10 GiB, the
  // smallest file cache read.
  cache_memory = memory_new();
  memory_set_reserve(cache_memory, 0, system_memory_read_counters, NULL);
  vm_fixture.external_page_count = (10ull << 30) / vm_page_size;
  uint64_t available;
  memory_read_system(cache_memory, &available);
  vm_fixture.free_count = (48ull << 30) / vm_page_size;
  vm_fixture.external_page_count = (30ull << 30) / vm_page_size;
  vm_fixture.speculative_count = (2ull << 30) / vm_page_size;
  for (int l = 0; l < CACHE_LAYERS; l++) {
    expert_spans[l] = 2;
    for (int e = 0; e < EXPERTS; e++) {
      expert_span[l][e][0].len = EXPERT_W;
      expert_span[l][e][1].len = EXPERT_S;
    }
  }
  expert_store_set_reserve(4);
  expert_store_start(st, 56ull << 30, LAYERS);
  check(st->cap == (int)((56ull << 30) / st->SLOT_SIZE),
        "the initial limit uses the file cache above the one kept to fit the "
        "56 GiB asked for");
  check(st->cap <= st->max_slots,
        "the initial limit stays within the most slots");
  const char *names[] = {
      "few free pages with file cache above the one kept do not shrink the "
      "cache",
      "calm growth uses the file cache above the one kept",
      "memory pressure shrinks the cache",
      "system-wide compression alone does not shrink the cache",
      "process-local compression growth shrinks the cache",
      "system-wide compression alone does not prevent growth",
      "a swapper wakeup without local compression does not shrink the cache",
      "local compression cannot take memory back from an empty cache",
      "a cache at its minimum grows again with available memory",
      "a file cache smaller than the one kept lowers it, so few free pages "
      "shrink the cache",
      "a shortfall without pressure gives back only the slots that restore the "
      "reserve",
      "two stores give one shortfall back once, and both stores' unwritten "
      "slots count",
      "a store that starts short of the reserve holds its minimum slots",
      "system-wide compression while file cache above floor counts keeps the "
      "floor intact and does not shrink the cache"};
  for (int kind = 0; kind < 14; kind++) {
    fflush(stdout);
    pid_t pid = fork();
    if (!pid) child_case(kind);
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 1;
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, names[kind]);
  }
  unwritten_checks();
  free(st->slots), free(st->score), free(st->free_list);
  return failures != 0;
}
