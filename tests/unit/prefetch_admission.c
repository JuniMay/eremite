#include "check.h"
#include "deferred_reads.h"
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

// Tests who owns a read-ahead's admission while the demand reads that took
// it over wait to run. Reads come from /dev/zero; no reader thread or GPU
// buffer is made.

int main(void) {
  st = store_create();

  enum { CURRENT = 6, COUNT = CURRENT + 1 };

  Slot fixture[COUNT] = {0};
  double priorities[COUNT];
  int ids[CURRENT], future = 0;
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  st->slots = fixture, st->score = priorities;
  st->max_slots = st->cap = st->nslots = COUNT, st->nfree = st->npinned = 0;
  st->head = COUNT - 1, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  expert_spans[0] = expert_spans[2] = 1;
  expert_span[2][future][0] = (FileSpan){fd, 0, 16384};
  for (int i = 0; i < COUNT; i++) {
    fixture[i].b.ptr = calloc(1, 65536), fixture[i].b.size = 65536;
    if (!fixture[i].b.ptr) die("fixture allocation failed");
    fixture[i].mapped = 1, fixture[i].layer = 0;
    fixture[i].expert = i < CURRENT ? i : -1;
    fixture[i].prev = i == COUNT - 1 ? -1 : i + 1;
    fixture[i].next = i ? i - 1 : -1;
    priorities[i] = i < CURRENT ? 0 : -INFINITY;
    if (i == CURRENT) continue;
    ids[i] = i, WHERE(0, i) = i;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
    fixture[i].prefetched = 1;
    start(st, i, 0);
  }
  check(atomic_load(&st->prefetching) == CURRENT &&
            atomic_load(&st->demand) == 0,
        "six read-aheads take six admissions");
  check(expert_store_start_reads(st, 0, ids, CURRENT) == CURRENT,
        "requesting them finds all six pending");
  expert_store_unprotect_read_aheads(st, 0);
  check(atomic_load(&st->demand) == CURRENT,
        "the request claims all six as demand reads");
  check(atomic_load(&st->prefetching) == CURRENT,
        "the claimed reads keep their admissions until they finish");
  check(st->stats.prefetch_hits == CURRENT && st->stats.late == CURRENT,
        "the request counts six late read-ahead hits");
  expert_store_read_ahead(st, 2, &future, 1, CURRENT);
  check(WHERE(2, future) == -1 && st->stats.prefetches == 0,
        "a seventh read-ahead finds no admission");

  deferred_run();
  expert_store_wait_for_reads(st, 0, ids, CURRENT);
  check(atomic_load(&st->prefetching) == 0 && atomic_load(&st->demand) == 0,
        "the finished reads release their admissions");
  check(WHERE(2, future) == -1,
        "a refused read-ahead is not started later on its own");
  check(st->stats.bytes == (uint64_t)CURRENT * 16384,
        "each expert is read once");

  expert_store_read_ahead(st, 2, &future, 1, CURRENT);
  check(WHERE(2, future) == CURRENT && atomic_load(&st->prefetching) == 1,
        "the same read-ahead offered again takes the free admission");
  check(st->stats.prefetches == 1, "it counts one read-ahead");
  expert_store_read_ahead(st, 2, &future, 1, CURRENT);
  check(st->stats.prefetches == 1 && atomic_load(&st->prefetching) == 1,
        "a read-ahead of an expert with a slot starts nothing");
  int ready = 1;
  for (int i = 0; i < CURRENT; i++)
    ready &= WHERE(0, i) == i && expert_store_is_ready(st, 0, i);
  check(ready, "the demanded experts stay in their slots, ready");
  cancel(st, CURRENT);
  check(atomic_load(&st->prefetching) == 0 && !fixture[CURRENT].loading,
        "a cancelled read-ahead releases its admission");
  for (int i = 0; i < COUNT; i++) free(fixture[i].b.ptr);
  close(fd);
  st->slots = NULL, st->score = NULL;
  return failures != 0;
}
