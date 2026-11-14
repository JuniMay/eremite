#include "check.h"
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#include <sys/wait.h>
#include <unistd.h>

// Tests the eviction policy with in-memory slots and empty reads: aging,
// victims while decoding and in a prompt pass, near misses, history across
// eviction, the weight's rescaling, a victim still being read, and a refined
// read-ahead's empty slot. No GPU or model is used.

static Slot fixture[3];
static double fixture_score[3];

static int near(double a, double b) {
  return fabs(a - b) <= 1e-12 * fmax(fabs(b), 1e-300);
}

static void setup(void) {
  if (st->slots)
    for (int i = 0; i < st->max_slots; i++) finish(st, i);
  st->slots = NULL;
  memset(fixture, 0, sizeof fixture);
  memset(st->where, 0xff, WHERE_BYTES);
  memset(st->escore, 0, SCORE_BYTES);
  memset(st->dscore, 0, SCORE_BYTES);
  memset(st->nscore, 0, SCORE_BYTES);
  st->near_weight = 1;
  memset(expert_spans, 0, sizeof expert_spans);
  memset(&st->stats, 0, sizeof st->stats);
  st->slots = fixture, st->score = fixture_score;
  st->max_slots = st->cap = st->nslots = 3, st->npinned = st->nfree = 0;
  st->head = 2, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  for (int i = 0; i < 3; i++) {
    st->slots[i].layer = 0, st->slots[i].expert = i, st->slots[i].mapped = 1;
    st->slots[i].prev = i == 2 ? -1 : i + 1;
    st->slots[i].next = i == 0 ? -1 : i - 1;
    WHERE(0, i) = i;
    st->score[i] = st->escore[i] = i + 1;
  }
}

// A reader has started the cheapest victim's last piece, but routing has
// demoted its read-ahead. Optional work leaves it alone rather than wait for
// that read or evict a more valuable resident; a demand waits and reuses it.
static void *complete_read(void *group) {
  usleep(100000);
  threads_group_leave(group);
  return NULL;
}

static void busy_victim(int optional, int prefill) {
  setup();
  st->pass_layer = prefill ? 1 : -1;
  st->slots[0].loading = threads_group_new();
  st->slots[0].npiece = 1;
  atomic_store(&st->slots[0].taken[0], 1);
  threads_group_enter(st->slots[0].loading);
  pthread_t reader;
  if (pthread_create(&reader, NULL, complete_read, st->slots[0].loading))
    die("cannot start the fixture's reader");
  double began = now();
  int got = take(st, 2, 0, optional);
  double elapsed = now() - began;
  pthread_join(reader, NULL);
  printf("%s %s with a victim being read: slot %d, %.3f ms\n",
         prefill ? "a prompt pass's" : "decoding's",
         optional ? "read-ahead" : "demand", got, elapsed * 1000);
  if (optional) {
    check(got == -1, "a read-ahead declines a victim still being read");
    check(WHERE(0, 0) == 0 && WHERE(0, 1) == 1 && WHERE(0, 2) == 2 &&
              WHERE(2, 0) == -1 && st->stats.evictions == 0,
          "a declined read-ahead keeps every resident");
    finish(st, 0);
    check(take(st, 2, 0, 1) == 0,
          "a read-ahead can take the same victim once its read ends");
  } else {
    check(got == 0 && !st->slots[0].loading && WHERE(0, 0) == -1 &&
              WHERE(2, 0) == 0 && st->stats.evictions == 1,
          "a demand waits for the cheapest victim and reuses it");
  }
}

// A prediction for layer 1 is refined while layer 0 is still being served;
// its cancelled slot stays reusable when routing reaches layer 1.
static void refined_empty_setup(void) {
  setup();
  memset(st->where, 0xff, WHERE_BYTES);
  st->slots[0].layer = 1, st->slots[0].expert = 0;
  st->slots[1].layer = 0, st->slots[1].expert = 1;
  st->slots[2].layer = 1, st->slots[2].expert = 2;
  WHERE(1, 0) = 0, WHERE(0, 1) = 1, WHERE(1, 2) = 2;
  st->score[0] = st->escore[EXPERTS] = 1;
  st->score[2] = st->escore[EXPERTS + 2] = 3;
  st->slots[0].prefetched = 1, st->slots[0].used = st->token;
  st->slots[0].npiece = 1, st->slots[0].background_load = 1;
  atomic_store(&st->slots[0].taken[0], 0);
  atomic_store(&st->slots[0].unfinished, 1);
  atomic_store(&st->prefetching, 1);
  st->slots[0].loading = threads_group_new();
  threads_group_enter(st->slots[0].loading);
  st->serving = 0;
  int keep = 2;
  expert_store_cancel_read_aheads(st, 1, &keep, 1);
  check(WHERE(1, 0) == -1 && st->slots[0].expert == -1 &&
            st->score[0] == -INFINITY && !st->slots[0].loading &&
            !atomic_load(&st->prefetching),
        "a refinement leaves an empty finished slot");
  // A read-ahead of a resident expert, as follows a refinement, leaves the
  // slot to the routing.
  expert_store_read_ahead(st, 1, &keep, 1, 6);
  check(st->slots[0].expert == -1 && WHERE(1, 2) == 2,
        "a read-ahead of a resident expert leaves the refined slot free");
}

static void refined_empty_reuse(void) {
  refined_empty_setup();
  int routed[] = {2, 3};
  expert_store_start_reads(st, 1, routed, 2);
  check(WHERE(1, 3) == 0, "routing reuses its layer's refined empty slot");
  check(WHERE(0, 1) == 1, "the reuse keeps another layer's ready expert");
  check(WHERE(1, 2) == 2 && st->slots[2].used == st->token,
        "the reuse keeps the selected resident");
  check(st->stats.bytes == 0, "the refinement and the reuse read nothing");

  refined_empty_setup();
  st->serving = 1;
  check(victim(st) == 0,
        "the layer being served may take a refined empty slot");
  st->slots[0].holds = 1;
  check(victim(st) == 1, "a hold excludes an otherwise empty slot");
  st->slots[0].holds = 0;
  check(victim(st) == 0,
        "releasing the hold makes the empty slot a victim again");
}

int main(void) {
  st = store_create();
  setup();
  for (int i = 0; i < 50; i++) advance_score_clock(st);
  check(near(st->weight, 2), "the weight doubles over 50 forward passes");
  check(st->token == 60, "the clock counts forward passes");
  check(st->escore[0] == 1, "aging keeps the raw history between rescalings");
  check(near(st->score[0], 0.5),
        "aging computes a slot's priority from its history again");

  setup();
  check(victim(st) == 0, "the victim has the lowest score");
  st->slots[0].prefetched = 1;
  check(victim(st) == 1, "a read-ahead is protected");
  st->serving = 0, st->slots[1].used = st->token;
  check(
      victim(st) == 2,
      "the experts that the layer being served used this token are protected");
  st->serving = -1, st->slots[0].prefetched = 0;
  st->score[0] = st->score[1] = 2, st->score[2] = 3;
  check(victim(st) == 0, "equal priorities go to the lower slot");
  st->score[2] = -INFINITY, st->slots[2].expert = -1;
  check(victim(st) == 2, "empty slots go first");
  st->score[2] = INFINITY, st->slots[2].layer = 32767,
  st->slots[2].expert = 32767;
  check(victim(st) == 0, "unallocated slots are no victims");

  // A near miss counts NEAR_CREDIT of a selection at once and fades with a
  // half-life of NEAR_HALF_LIFE steps against the selections' HALF_LIFE: slot
  // 0's expert, selected once, outranks slot 1's, selected twice, while its
  // three near misses are fresh, and no longer NEAR_HALF_LIFE steps later.
  setup();
  st->score[0] = st->escore[0] = 1, st->score[1] = st->escore[1] = 2,
  st->score[2] = st->escore[2] = 4;
  check(victim(st) == 0, "without near misses the least selected expert goes");
  for (int k = 0; k < 3; k++)
    expert_store_credit_near_misses(st, 0, (int[]){0}, 1);
  check(near(st->score[0], 1 + 3 * NEAR_CREDIT),
        "near misses raise a resident expert's priority at once");
  check(victim(st) == 1, "fresh near misses keep an expert");
  for (int k = 0; k < NEAR_HALF_LIFE; k++) advance_score_clock(st);
  double fade = exp2(-(double)NEAR_HALF_LIFE / HALF_LIFE);
  check(near(st->score[0], fade + 3 * NEAR_CREDIT / 2) &&
            near(st->score[1], 2 * fade),
        "aging computes both parts of the priorities again");
  check(victim(st) == 0, "near misses fade faster than selections");
  WHERE(0, 3) = -1;
  expert_store_credit_near_misses(st, 0, (int[]){3}, 1);
  check(st->nscore[3] > 0, "an expert without a slot keeps its near misses");
  expert_store_credit_near_misses(st, st->scored_layers, (int[]){0}, 1);
  check(st->nscore[(size_t)st->scored_layers * st->experts] == 0,
        "layers after the scored ones take no near misses");

  // A prompt pass keeps what decoding uses most: its victim is the slot of a
  // layer the pass is done with whose expert decoding used least, rather than
  // the least recently used one, else of a layer after the next. Slot 0 is
  // the least recently used, and the one decoding uses most.
  setup();
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->score[0] = 9, st->score[1] = 2, st->score[2] = 1;
  st->dscore[0] = 9, st->dscore[EXPERTS + 1] = 2,
  st->dscore[3 * EXPERTS + 2] = 1;
  st->pass_layer = 2;
  check(victim(st) == 1,
        "a prompt pass evicts the done layers' slot that decoding used least");
  st->pass_layer = 0;
  check(victim(st) == 2,
        "a prompt pass otherwise spares its current and next layers");
  st->slots[2].layer = 1;
  check(victim(st) == -1,
        "a prompt pass finds no victim in its current and next layers");
  st->pass_layer = -1;

  // A first prompt, with no decoding history: a prompt pass ranks by
  // decoding's use alone, and among equals takes the least recently used
  // slot, the earlier layer's slot 0, not the later layer's lower-scored slot
  // 1, whose expert the decoding after the prompt needs.
  setup();
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->score[0] = st->escore[0] = 4, st->score[1] = st->escore[EXPERTS + 1] = 1,
  st->score[2] = 1;
  st->pass_layer = 2;
  check(victim(st) == 0,
        "a first prompt's pass evicts the least recently used slot");
  st->pass_layer = -1;

  // Between slots that decoding used alike, recency decides, not the size of
  // the experts' reads.
  setup();
  st->slots[2].layer = 3;
  st->score[0] = st->score[1] = 5, st->score[2] = 1;
  expert_spans[0] = 1;
  expert_span[0][0][0].len = 200, expert_span[0][1][0].len = 100;
  st->pass_layer = 2;
  check(victim(st) == 0,
        "a prompt pass breaks a tie in decoding's use by recency");
  expert_spans[0] = 0, st->pass_layer = -1;

  // Decoding's use outranks recency: an older slot that decoding used stays.
  setup();
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->dscore[0] = 3;
  st->pass_layer = 2;
  check(victim(st) == 1,
        "a prompt pass keeps an older slot that decoding used");
  st->pass_layer = -1;

  setup();
  st->max_slots = st->cap = st->nslots = 2, st->head = 1, st->tail = 0;
  WHERE(0, 2) = -1, st->escore[2] = 0;
  st->slots[1].prev = -1;
  st->score[0] = st->escore[0] = 2, st->score[1] = st->escore[1] = 5;
  int incoming = 2;
  expert_store_read_and_wait(st, 0, &incoming, 1);
  check(WHERE(0, 0) == -1 && st->escore[0] == 2,
        "an evicted expert keeps its history");
  check(st->escore[2] == 1 && st->score[WHERE(0, 2)] == 1,
        "a new request adds the current weight");
  advance_score_clock(st);
  double before = st->weight;
  incoming = 0;
  expert_store_read_and_wait(st, 0, &incoming, 1);
  check(near(st->escore[0], 2 + before),
        "an expert that comes back adds to its earlier history");
  check(near(st->score[WHERE(0, 0)], st->escore[0] / st->weight),
        "its priority comes from that history");
  check(st->stats.bytes == 0, "the empty spans read nothing");

  st->weight = 0x1p500;
  st->escore[3] = 7;
  double scale = st->weight * exp2(1.0 / HALF_LIFE), old = st->escore[0];
  advance_score_clock(st);
  check(st->weight == 1, "a weight past 2^500 is reset to 1");
  check(near(st->escore[0], old / scale),
        "the reset rescales a resident expert's history");
  check(near(st->escore[3], 7 / scale),
        "the reset rescales an evicted expert's history");
  check(st->score[WHERE(0, 0)] == st->escore[0],
        "the resident priorities match the rescaled history");
  busy_victim(1, 0);
  busy_victim(1, 1);
  busy_victim(0, 0);
  refined_empty_reuse();
  for (int i = 0; i < st->max_slots; i++) finish(st, i);
  return failures != 0;
}
