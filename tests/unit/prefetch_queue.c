#include "check.h"
#include "deferred_reads.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

// Tests the read-ahead admissions while demand reads wait to run: a promoted,
// cancelled, demanded or demoted read-ahead, and the order in which the
// readers take queued pieces. Reads come from /dev/zero; no reader thread or
// GPU buffer is made.

static Slot fixture;
static double fixture_score;

static void setup(int fd) {
  memset(&fixture, 0, sizeof fixture);
  fixture.layer = 0;
  fixture.expert = 0;
  fixture.b.ptr = calloc(1, 65536);
  fixture.b.size = 65536;
  fixture.mapped = 1;
  fixture.prev = fixture.next = -1;
  st->slots = &fixture, st->score = &fixture_score;
  st->head = st->tail = 0, st->weight = 1;
  memset(st->where, 0xff, WHERE_BYTES);
  WHERE(0, 0) = 0;
  expert_spans[0] = 1;
  expert_span[0][0][0] = (FileSpan){fd, 0, 16384};
  st->qhead = st->qtail = 0;
  atomic_store(&st->demand, 0);
  atomic_store(&st->prefetching, 0);
  memset(&st->stats, 0, sizeof st->stats);
}

static void teardown(void) {
  deferred_run();
  finish(st, 0);
  check(!fixture.loading && !atomic_load(&st->demand) &&
            !atomic_load(&st->prefetching),
        "all the work is reaped");
  free(fixture.b.ptr);
  st->slots = NULL;
}

int main(void) {
  st = store_create();
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;

  setup(fd);
  start(st, 0, 0);
  check(atomic_load(&st->prefetching) == 1 && slot_reading(st, 0),
        "a queued read-ahead occupies one admission");
  atomic_store(&fixture.taken[0], 1);
  atomic_store(&st->demand, 1);
  read_demand(NULL);  // the promoted read finishes its one piece while other
                      // demand reads wait
  check(!slot_reading(st, 0), "the read data is ready");
  check(atomic_load(&st->prefetching) == 0,
        "a finished read-ahead releases its admission at once");
  check(st->stats.bytes == 16384, "a promoted read is read once");
  teardown();

  setup(fd);
  start(st, 0, 0);
  cancel(st, 0);
  check(st->stats.bytes == 0 && !fixture.loading,
        "a cancelled read-ahead reads nothing");
  check(atomic_load(&st->prefetching) == 0,
        "a cancelled read-ahead releases its admission at once");
  teardown();

  setup(fd);
  start(st, 0, 1);
  check(atomic_load(&st->prefetching) == 0,
        "demand reads take no read-ahead admission");
  teardown();

  setup(fd);
  start(st, 0, 0);
  fixture.prefetched = 1;
  expert_store_unprotect_read_aheads(
      st, 0);  // not selected in its predicted pass, and still queued
  int wanted = 0;
  check(!fixture.prefetched && slot_reading(st, 0),
        "a demotion leaves the read-ahead pending");
  int reading = expert_store_start_reads(st, 0, &wanted, 1);
  check(reading == 1, "a demand sees a demoted read-ahead still pending");
  check(atomic_load(&st->demand) == 1 && atomic_load(&fixture.taken[0]) == 1,
        "a demand claims every queued piece of a demoted read-ahead");
  if (!atomic_load(&st->demand)) claim(st, 0);
  teardown();

  // The readers take the lowest layer's pieces first, and each layer's in the
  // order they were queued, also across the ring's end.
  static const uint16_t layers[] = {2, 0, 1, 0, 2};
  st->qhead = st->qtail = QUEUE - 2;
  for (uint32_t k = 0; k < 5; k++) queue_push(st, k, layers[k]);
  uint32_t order[5];
  for (int k = 0; k < 5; k++)
    order[k] = st->queue[st->qhead++ % QUEUE];  // as a reader takes them
  check(order[0] == 1 && order[1] == 3 && order[2] == 2 && order[3] == 0 &&
            order[4] == 4 && st->qhead == st->qtail,
        "the readers take the nearest layer's pieces first, each layer's in "
        "order");
  st->qhead = st->qtail = 0;
  close(fd);
  return failures != 0;
}
