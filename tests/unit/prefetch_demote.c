#include "check.h"
#include "deferred_reads.h"
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

// Tests demotion and refinement: a demotion keeps every read, and a newer
// prediction cancels only the read-aheads it excludes that no reader has
// started. Demand reads wait to run and the test completes pieces itself, so
// that every change of ownership happens in a known order. Reads come from
// /dev/zero; no reader thread or GPU buffer is made.

enum { NSLOT = 7, LAYER = 3, BUFSIZE = 131072, QMAX = 32 };

static Slot fixture[NSLOT];
static double priorities[NSLOT];

static void complete_piece(int slot, int piece) {
  Slot *s = &fixture[slot];
  io_read_piece(&s->piece[piece], 0);
  __atomic_fetch_add(&st->stats.bytes, s->piece[piece].len, __ATOMIC_RELAXED);
  piece_finished(st, s, 1);
}

static int snapshot(uint32_t out[QMAX]) {
  uint32_t n = st->qtail - st->qhead;
  if (n > QMAX) die("the fixture's queue exceeded its bound");
  for (uint32_t k = 0; k < n; k++) out[k] = st->queue[(st->qhead + k) % QUEUE];
  return (int)n;
}

static int same_queue(const uint32_t *expected, int count) {
  uint32_t actual[QMAX];
  return snapshot(actual) == count &&
         !memcmp(actual, expected, (size_t)count * sizeof *actual);
}

static void setup(int fd, int prefill) {
  memset(fixture, 0, sizeof fixture);
  memset(st->where, 0xff, WHERE_BYTES);
  memset(st->escore, 0, SCORE_BYTES);
  memset(&st->stats, 0, sizeof st->stats);
  st->slots = fixture, st->score = priorities;
  st->max_slots = st->cap = st->nslots = NSLOT, st->nfree = st->npinned = 0;
  st->head = NSLOT - 1, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = prefill ? LAYER : -1, st->serving = -1;
  st->qhead = st->qtail = 0;
  atomic_store(&st->demand, 0), atomic_store(&st->prefetching, 0);
  // Slot 0 is selected, 1 and 6 are unused and unstarted, 2 is partly
  // started, and 5 is ready. 3 and 4 belong to other layers and keep their
  // places in the queue.
  const int layers[NSLOT] = {LAYER,     LAYER, LAYER, LAYER + 1,
                             LAYER - 1, LAYER, LAYER};
  const int experts[NSLOT] = {0, 1, 2, 0, 0, 3, 4};
  for (int l = LAYER - 1; l <= LAYER + 2; l++) expert_spans[l] = 2;
  for (int i = 0; i < NSLOT; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(BUFSIZE), s->b.size = BUFSIZE;
    memset(s->b.ptr, 0xa5, BUFSIZE);
    s->mapped = 1, s->layer = layers[i], s->expert = experts[i],
    s->prefetched = 1, s->used = st->token;
    s->prev = i == NSLOT - 1 ? -1 : i + 1, s->next = i ? i - 1 : -1;
    WHERE(s->layer, s->expert) = i;
    priorities[i] = st->escore[s->layer * EXPERTS + s->expert] = 10 + i;
    for (int k = 0; k < 2; k++)
      expert_span[s->layer][s->expert][k] =
          (FileSpan){fd, (uint64_t)k * 32768, 16384};
    start(st, i, 0);
  }
  for (int k = 0; k < 2; k++)
    expert_span[LAYER + 2][0][k] = (FileSpan){fd, (uint64_t)k * 32768, 16384};

  int selected = 0;
  check(expert_store_start_reads(st, LAYER, &selected, 1) == 1,
        "the selected read-ahead is pending");
  atomic_store(&fixture[2].taken[0],
               1);  // a reader owns this piece but has not completed it
  for (int k = 0; k < 2; k++) {
    atomic_store(&fixture[5].taken[k], 1);
    complete_piece(5, k);
  }
  finish(st, 5);
  fixture[5].prefetched = 0, fixture[5].used = st->token - 1;
  priorities[5] = st->escore[LAYER * EXPERTS + 3] =
      0;  // empty slots must go before this low resident

  // The slots' entries interleave, claimed entries stay, and slot 1 has an
  // old duplicate entry. Both the ring's index and its 32-bit counter wrap.
  const uint32_t order[] = {0x30, 0x10, 0x00, 0x20, 0x60, 0x40, 0x50, 0x31,
                            0x11, 0x01, 0x21, 0x61, 0x41, 0x51, 0x10};
  st->qhead = UINT32_MAX - 4;
  st->qtail = st->qhead + sizeof order / sizeof *order;
  for (size_t k = 0; k < sizeof order / sizeof *order; k++)
    st->queue[(st->qhead + (uint32_t)k) % QUEUE] = order[k];
}

// Finishes the read that a reader started, then the demand reads and the
// queued pieces, so that no read outlives a fixture's buffer.
static void teardown(void) {
  if (fixture[2].loading) complete_piece(2, 0);
  deferred_run();
  while (st->qhead != st->qtail) {
    uint32_t q = st->queue[st->qhead++ % QUEUE];
    Slot *s = &fixture[q >> 4];
    if (!atomic_exchange(&s->taken[q & 15], 1)) complete_piece(q >> 4, q & 15);
  }
  for (int i = 0; i < NSLOT; i++) {
    finish(st, i);
    free(fixture[i].b.ptr);
  }
  check(!atomic_load(&st->prefetching) && !atomic_load(&st->demand),
        "every admission and demand read finishes");
  st->slots = NULL, st->score = NULL;
}

int main(void) {
  st = store_create();
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  for (int prefill = 0; prefill <= 1;
       prefill++) {  // a demotion cancels nothing, in either kind of pass
    setup(fd, prefill);
    uint32_t before[QMAX];
    int count = snapshot(before), admissions = atomic_load(&st->prefetching);
    uint64_t bytes = st->stats.bytes, cancelled = st->stats.cancelled;
    expert_store_unprotect_read_aheads(st, LAYER);
    check(WHERE(LAYER, 0) == 0 && fixture[0].loading,
          "a demotion keeps the selected pending expert");
    check(WHERE(LAYER, 1) == 1 && WHERE(LAYER, 4) == 6 && fixture[1].loading &&
              fixture[6].loading && !fixture[1].prefetched &&
              !fixture[6].prefetched,
          "a demotion keeps unused reads queued, no longer protected");
    check(WHERE(LAYER + 1, 0) == 3 && fixture[3].prefetched &&
              WHERE(LAYER - 1, 0) == 4 && fixture[4].prefetched,
          "other layers' read-aheads stay protected");
    check(st->stats.bytes == bytes && same_queue(before, count) &&
              atomic_load(&st->prefetching) == admissions &&
              st->stats.cancelled == cancelled,
          "a demotion reads nothing and changes no queue");
    teardown();
  }

  // A newer prediction cancels the unstarted read-aheads it excludes: here
  // experts 1 and 4, in slots 1 and 6, and not experts 0, 2 or 3.
  const int newer[3] = {0, 2, 3};
  setup(fd, 0);
  {
    uint32_t before[QMAX], expected[QMAX];
    int count = snapshot(before), kept = 0;
    for (int k = 0; k < count; k++)
      if ((before[k] >> 4) != 1 && (before[k] >> 4) != 6)
        expected[kept++] = before[k];
    int admissions = atomic_load(&st->prefetching);
    uint64_t bytes = st->stats.bytes, cancelled = st->stats.cancelled;
    uint8_t *selected_buffer = fixture[0].b.ptr,
            *partial_buffer = fixture[2].b.ptr,
            *reuse_buffer = fixture[1].b.ptr;
    expert_store_cancel_read_aheads(st, LAYER, newer, 3);
    check(WHERE(LAYER, 0) == 0 && fixture[0].b.ptr == selected_buffer &&
              fixture[0].loading && !fixture[0].prefetched &&
              atomic_load(&fixture[0].unfinished) == 2,
          "the selected pending expert stays protected");
    check(WHERE(LAYER, 2) == 2 && fixture[2].b.ptr == partial_buffer &&
              fixture[2].loading && atomic_load(&fixture[2].unfinished) == 2 &&
              atomic_load(&fixture[2].taken[0]) == 1 &&
              atomic_load(&fixture[2].taken[1]) == 0,
          "a partly started read-ahead can still finish");
    check(WHERE(LAYER, 3) == 5 && !slot_reading(st, 5),
          "an expert already read stays cached");
    check(WHERE(LAYER + 1, 0) == 3 && fixture[3].prefetched &&
              WHERE(LAYER - 1, 0) == 4 && fixture[4].prefetched,
          "other layers' read-aheads stay protected");
    check(st->stats.bytes == bytes, "a refinement reads nothing");
    check(same_queue(expected, kept),
          "the remaining queue keeps its order across the wrap");
    check(atomic_load(&st->prefetching) == admissions - 2,
          "only the cancelled admissions are released");
    check(st->stats.cancelled == cancelled + 2,
          "each cancellation counts once");
    check(WHERE(LAYER, 1) == -1 && WHERE(LAYER, 4) == -1 &&
              fixture[1].expert == -1 && fixture[6].expert == -1 &&
              !slot_reading(st, 1) && !slot_reading(st, 6),
          "the excluded unstarted read-aheads leave empty slots");
    expert_store_cancel_read_aheads(st, LAYER, newer, 3);
    check(same_queue(expected, kept) && st->stats.cancelled == cancelled + 2,
          "a repeated refinement changes nothing");
    int incoming = 0;
    expert_store_read_ahead(st, LAYER + 2, &incoming, 1, 100);
    check(WHERE(LAYER + 2, 0) == 1 && fixture[1].b.ptr == reuse_buffer,
          "an empty slot is reused without allocation");
    expected[kept++] = 0x10, expected[kept++] = 0x11;
    check(same_queue(expected, kept),
          "the reused slot has only its new queue entries");
    check(WHERE(LAYER, 0) == 0 && WHERE(LAYER, 2) == 2,
          "the reuse displaces neither selected nor partly read data");
  }
  teardown();

  setup(fd, 0);
  int keep = 1;
  expert_store_cancel_read_aheads(st, LAYER, &keep, 1);
  check(WHERE(LAYER, 1) == 1 && fixture[1].loading && fixture[1].prefetched &&
            !atomic_load(&fixture[1].taken[0]) &&
            !atomic_load(&fixture[1].taken[1]),
        "a refinement keeps an unstarted read-ahead that it names");
  check(WHERE(LAYER, 4) == -1 && fixture[6].expert == -1 &&
            !slot_reading(st, 6),
        "a refinement cancels an excluded unstarted read-ahead");
  check(WHERE(LAYER, 0) == 0 && WHERE(LAYER, 2) == 2 &&
            atomic_load(&fixture[2].taken[0]) &&
            !atomic_load(&fixture[2].taken[1]),
        "a refinement keeps the selected and the partly started reads");
  teardown();

  setup(fd, 0);
  fixture[1].holds = 1;
  fixture[5].prefetched = 1;
  expert_store_cancel_read_aheads(st, LAYER, newer, 3);
  check(WHERE(LAYER, 1) == 1 && fixture[1].holds == 1 && fixture[1].loading &&
            !atomic_load(&fixture[1].taken[0]) &&
            !atomic_load(&fixture[1].taken[1]),
        "a held read-ahead keeps its pending buffer");
  check(WHERE(LAYER, 4) == -1 && fixture[6].expert == -1 &&
            !slot_reading(st, 6),
        "a held read-ahead does not protect an unrelated unused one");
  check(WHERE(LAYER, 3) == 5 && !slot_reading(st, 5),
        "a finished read-ahead stays available");
  teardown();

  // Passes that run together: one pass's newer prediction and demotion
  // concern only its own read-aheads.
  setup(fd, 0);
  for (int i = 0; i < NSLOT; i++)
    if (i != 1 && i != 2) fixture[i].issuer = (const Pass *)&keep;
  {
    uint64_t cancelled = st->stats.cancelled;
    expert_store_cancel_read_aheads(st, LAYER, newer, 3);
    check(WHERE(LAYER, 4) == 6 && fixture[6].loading && fixture[6].prefetched &&
              st->stats.cancelled == cancelled + 1,
          "another pass's unstarted read-ahead is not cancelled");
    check(WHERE(LAYER, 1) == -1 && fixture[1].expert == -1,
          "the pass's own excluded read-ahead is cancelled");
    expert_store_unprotect_read_aheads(st, LAYER);
    check(fixture[6].prefetched,
          "another pass's read-ahead keeps its protection");
    check(!fixture[2].prefetched, "the pass's own read-ahead is demoted");
    fixture[1].issuer = NULL;
  }
  teardown();
  close(fd);
  return failures != 0;
}
