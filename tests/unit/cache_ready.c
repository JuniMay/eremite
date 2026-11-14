#include "check.h"
#include "engine/expert_store.c"
#include "store_layout.h"

// Tests readiness and read-ahead counts without reads, a GPU or a model: a
// read-ahead whose load is finished, still pending, or already reaped.

static void expect(const char *state, const char *field, uint64_t got,
                   uint64_t want) {
  char what[160];
  snprintf(what, sizeof what, "%s: %s", state, field);
  check(got == want, what);
}

static void check_prefetch(const char *state, int has_group, int pending) {
  const int layer = 0, expert = 7;
  Slot slot = {.layer = layer,
               .expert = expert,
               .prev = -1,
               .next = -1,
               .prefetched = 1,
               .mapped = 1,
               .npiece = 1};
  double slot_score = 0;
  st->slots = &slot;
  st->score = &slot_score;
  st->head = st->tail = 0;
  st->token = 1;
  memset(st->where, 0xff, WHERE_BYTES);
  WHERE(layer, expert) = 0;
  memset(&st->stats, 0, sizeof st->stats);
  // The read was claimed already, so a request starts no read.
  atomic_store(&slot.taken[0], 1);
  if (has_group) {
    slot.loading = threads_group_new();
    threads_group_enter(slot.loading);
    if (!pending) threads_group_leave(slot.loading);
    expect(state, "the group has pending work",
           !threads_group_done(slot.loading), pending);
  }

  expect(state, "the selected expert has a slot",
         expert_store_has_slot(st, layer, expert), 1);
  expect(state, "ready before the request",
         expert_store_is_ready(st, layer, expert), !pending);
  expect(state, "where the read-ahead is",
         expert_store_locate(st, layer, expert),
         pending ? EXPERT_BEING_READ_AHEAD : EXPERT_READ_AHEAD);
  expect(state, "the request's pending reads",
         expert_store_start_reads(st, layer, &expert, 1), pending);
  expect(state, "read-ahead hits", st->stats.prefetch_hits, 1);
  expect(state, "late read-ahead hits", st->stats.late, pending);
  expect(state, "demand misses", st->stats.misses, 0);
  expect(state, "bytes read", st->stats.bytes, 0);
  expect(state, "the selected expert keeps its slot",
         expert_store_has_slot(st, layer, expert), 1);

  if (pending) {
    threads_group_leave(slot.loading);
    expect(state, "ready after the read completes",
           expert_store_is_ready(st, layer, expert), 1);
  }
  // Reaping the finished load cannot block.
  expert_store_wait_for_reads(st, layer, &expert, 1);
  expect(state, "ready after the wait",
         expert_store_is_ready(st, layer, expert), 1);
  expect(state, "the wait releases the group", slot.loading == NULL, 1);
  expect(state, "a missing expert has no slot",
         expert_store_has_slot(st, layer, expert + 1), 0);
  expect(state, "a missing expert is not ready",
         expert_store_is_ready(st, layer, expert + 1), 0);
  expect(state, "where a missing expert is",
         expert_store_locate(st, layer, expert + 1), EXPERT_ON_SSD);
  st->slots = NULL;
  st->score = NULL;
}

int main(void) {
  st = store_create();
  check_prefetch("a finished load", 1, 0);
  check_prefetch("a pending load", 1, 1);
  check_prefetch("a reaped load", 0, 0);
  return failures != 0;
}
