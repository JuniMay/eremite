#include "check.h"
#include "base/error.h"
#include "engine/io.h"
#include "engine/weights.h"
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that a stop request ends a model's load between weights: the stopped
// load returns its memory and leaves the weights unloaded, while weights that
// a pass loads complete with the pass. Buffers are plain memory and no file
// is read. Each case runs in its own child process, since the stop request
// stays set.

static Buf fixture_new(Gpu *gpu, MemorySlot *slot, size_t size) {
  (void)gpu;
  Buf b = {.allocated = size, .size = size};
  b.memory = slot;
  if (!memory_claim(b.memory, b.allocated, MEMORY_REQUIRED))
    die("the fixture's pool refused %zu bytes", size);
  b.ptr = calloc(1, size ? size : 1);
  b.obj = b.ptr;
  return b;
}

static void fixture_free(Buf *b) {
  if (b->obj) {
    memory_release(b->memory, b->allocated);
    free(b->ptr);
  }
  *b = (Buf){0};
}

static size_t fixture_allocation_size(size_t size) { return size; }

static void fixture_resident(const Buf *b, int on) { (void)b, (void)on; }

static void fixture_commit(Gpu *gpu) { (void)gpu; }

static void fixture_read(int fd, uint64_t off, uint64_t len, Buf *dst) {
  (void)fd, (void)off, (void)len, (void)dst;
}

#define gpu_alloc fixture_new
#define gpu_free fixture_free
#define gpu_round_up_to_pages fixture_allocation_size
#define gpu_set_resident fixture_resident
#define gpu_commit_residency fixture_commit
#define io_read fixture_read
#include "engine/weights.c"
#undef gpu_alloc
#undef gpu_free
#undef gpu_round_up_to_pages
#undef gpu_set_resident
#undef gpu_commit_residency
#undef io_read

enum { WEIGHTS = 8, BYTES = 4096 };

static int fills;

static void fill(Buf *buffer, const void *parameters) {
  (void)buffer, (void)parameters;
  if (++fills == 1)
    shutdown_request();  // the stop request comes while the first weight is
                         // read
}

// Eight weights load; a stop request comes while the first one is read.
static int load(int model_load) {
  Memory *memory = memory_new();
  memory_set_limit(memory, 1ull << 30);
  MemoryAccount account;
  memory_account_init(memory, &account, "fixture model", 0);
  WeightGroup *set = weight_group_new(NULL, &account, "fixture weights");
  Buf targets[WEIGHTS] = {0};
  for (int i = 0; i < WEIGHTS; i++)
    weight_group_add_generated(set, &targets[i], BYTES, fill, NULL, 0);
  weight_group_set_stop_on_quit(model_load);
  int loaded = weight_group_load(set);
  printf("%s: %d of %d weights read, %s\n",
         model_load ? "a model's load" : "a pass's load", fills, WEIGHTS,
         loaded ? "loaded" : "not loaded");
  if (!model_load)
    return fills == WEIGHTS && loaded && memory_ready(weight_group_slot(set));
  return fills == 1 && !loaded && !memory_ready(weight_group_slot(set)) &&
         memory_usage(memory).allocated == 0 &&
         memory_usage(memory).committed == 0;
}

int main(void) {
  for (int model_load = 1; model_load >= 0; model_load--) {
    fflush(stdout);
    pid_t child = fork();
    if (child == 0) {
      int ok = load(model_load);
      fflush(stdout);
      _exit(ok ? 0 : 1);
    }
    int status = 0;
    waitpid(child, &status, 0);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          model_load ? "a stop request ends a model's load, which returns its "
                       "memory and loads nothing"
                     : "a pass's load completes despite a stop request");
  }
  return failures != 0;
}
