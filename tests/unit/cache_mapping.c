#include "check.h"
#include "engine/expert_store.h"
#include <fcntl.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests mapping: reads into a slot need no GPU residency, and only the GPU's
// use of an expert maps it. Also tests when a layer's experts are reachable
// through a table of addresses, a layout's transform, and the slot limit that
// rises to a pass's fewest slots. Buffers are plain memory that the test
// counts in place of the GPU's.

// Filled with 0xaa, so that a read of zeros shows.
static Buf fixture_new(Gpu *gpu, MemorySlot *slot, size_t bytes) {
  (void)gpu, (void)slot;
  void *p = malloc(bytes);
  memset(p, 0xaa, bytes);
  return (Buf){.obj = p, .ptr = p, .size = bytes};
}

static void fixture_free(Buf *b) {
  free(b->ptr);
  *b = (Buf){0};
}

#define gpu_alloc fixture_new
#define gpu_free fixture_free
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#undef gpu_alloc
#undef gpu_free

// The transform case's layout: each expert's two spans, and what its
// transform saw, since readers may run two at once: its calls, the calls
// before every piece of the expert's slot was read, and the bytes of the spans
// not read.
static const uint64_t read_len[2] = {3 * IO_PIECE + 5, 77};

enum { EXTRA = 40 };

typedef struct {
  atomic_int calls, early, unread;
  uint32_t at[2][2];
  uint64_t size[2];
} Marks;

static void mark(void *context, int layer, int expert, uint8_t *slot,
                 const uint32_t *at, uint64_t slot_bytes) {
  Marks *m = context;
  Slot *s = &st->slots[WHERE(layer, expert)];
  m->early += atomic_load(&s->read) != s->npiece;
  int unread = 0;
  for (int k = 0; k < 2; k++)
    for (uint64_t b = 0; b < read_len[k]; b++) unread += slot[at[k] + b] != 0;
  m->unread += unread;
  m->at[expert][0] = at[0], m->at[expert][1] = at[1],
  m->size[expert] = slot_bytes;
  memset(slot + slot_bytes - EXTRA, expert + 1, EXTRA);
  m->calls++;
}

int main(void) {
  st = store_create();
  Slot fixture[2] = {0};
  double priorities[2] = {INFINITY, INFINITY};
  int unused[2] = {1, 0};
  st->slots = fixture, st->score = priorities, st->free_list = unused;
  st->max_slots = st->cap = st->nfree = 2, st->SLOT_SIZE = 64;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  int a = take(st, 0, 0, 1), b = take(st, 1, 0, 0);
  check(st->nslots == 2 && st->nmapped == 0 && added == 0 &&
            !st->slots[a].mapped && !st->slots[b].mapped,
        "new slots are not mapped");
  int ids[] = {0};
  check(expert_store_start_reads(st, 0, ids, 1) == 0 && st->nmapped == 0 &&
            added == 0,
        "a request maps nothing");
  uint32_t at[EXPERT_STORE_PARTS];
  int fresh;
  check(expert_store_get_buffer(st, 0, 0, at, &fresh) == &st->slots[a].b &&
            st->nmapped == 1 && added == 1 && fresh && st->slots[a].mapped &&
            st->stats.remaps == 0,
        "the GPU's use maps the slot, fresh");
  check(expert_store_get_buffer(st, 0, 0, at, &fresh) == &st->slots[a].b &&
            st->nmapped == 1 && added == 1,
        "a mapped slot is mapped once");
  // A read-ahead that reuses an old mapped buffer unmaps it.
  st->token++, st->serving = -1;
  st->score[a] = -100, st->score[b] = 100;
  int reused = take(st, 2, 0, 1);
  check(reused == a && st->nmapped == 0 && removed == 1 && !st->slots[a].mapped,
        "reusing a mapped buffer for a read-ahead unmaps it");
  check(expert_store_get_buffer(st, 2, 0, at, &fresh) == &st->slots[a].b &&
            st->nmapped == 1 && added == 2 && fresh && st->stats.remaps == 1,
        "the GPU's use maps it again, as a remap");
  check(st->nslots == 2 && WHERE(0, 0) == -1 && WHERE(2, 0) == a,
        "the reused slot holds the new expert");
  for (int i = 0; i < 2; i++) fixture_free(&fixture[i].b);

  // A layer reaches its experts through a table of their addresses only while
  // each is in a slot, read, mapped and committed; until one moves or a slot
  // is unmapped, the layer stays ready. A store of 2 layers of 4 experts.
  static FileSpan small_span[2][4]
                            [1];  // empty spans: their reads finish at once
  static int small_spans[2] = {1, 1};
  st = expert_store_new(NULL, &(ExpertLayout){.layers = 2,
                                              .scored_layers = 2,
                                              .experts = 4,
                                              .chosen = 2,
                                              .parts = 1,
                                              .span = &small_span[0][0][0],
                                              .spans = small_spans});
  Slot slots[8] = {0};
  double scores[8];
  int frees[8];
  for (int i = 0; i < 8; i++) scores[i] = INFINITY, frees[i] = 7 - i;
  st->slots = slots, st->score = scores, st->free_list = frees;
  st->max_slots = st->cap = st->mcap = st->nfree = 8, st->SLOT_SIZE = 64;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  uint64_t g0, g1, g2, g3;
  int layer0[] = {0, 1, 2, 3};
  check(!expert_store_layer_is_reachable(st, 0, &g0),
        "an empty layer is not ready");
  check(expert_store_hold(st, 0, layer0, 4) == 0,
        "a whole layer, more than a row chooses, is held");
  check(!expert_store_layer_is_reachable(st, 0, &g1) && g1 != g0,
        "a layer read but not reaped is not ready");
  expert_store_wait_for_reads(st, 0, layer0, 4);
  check(!expert_store_layer_is_reachable(st, 0, &g1),
        "a layer that is not mapped is not ready");
  int all_fresh = 1;
  for (int e = 0; e < 4; e++)
    all_fresh &= expert_store_get_buffer(st, 0, e, at, &fresh) && fresh;
  check(all_fresh && !expert_store_layer_is_reachable(st, 0, &g1) &&
            st->any_fresh,
        "a layer mapped but not committed is not ready");
  expert_store_release(st, 0, layer0, 4);
  expert_store_age_and_resize(st);
  check(!st->any_fresh && expert_store_layer_is_reachable(st, 0, &g2) &&
            g2 == g1,
        "the committed layer is ready");
  check(expert_store_layer_is_reachable(st, 0, &g2) &&
            !expert_store_layer_is_reachable(st, 1, &g3),
        "a ready layer stays ready");
  st->mcap = 3;  // the residency limit takes one slot back
  expert_store_age_and_resize(st);
  check(st->nmapped == 3 && !expert_store_layer_is_reachable(st, 0, &g3) &&
            g3 == g2,
        "an unmapped slot makes its layer not ready");
  st->mcap = 8;
  for (int e = 0; e < 4; e++) expert_store_get_buffer(st, 0, e, at, &fresh);
  expert_store_age_and_resize(st);
  check(expert_store_layer_is_reachable(st, 0, &g3),
        "mapped again and committed, the layer is ready");
  st->cap = 4;  // full: a layer-1 expert takes a layer-0 expert's slot
  take(st, 1, 0, 0);
  int gone = -1;
  for (int e = 0; e < 4; e++)
    if (WHERE(0, e) < 0) gone = e;
  check(gone >= 0 && !expert_store_layer_is_reachable(st, 0, &g3) && g3 != g2,
        "an expert leaving its slot changes the layer's generation");
  // A table's experts count as used, without a read for one that left its
  // slot.
  uint64_t hits = st->stats.hits, misses = st->stats.misses;
  int kept = gone ? 0 : 1, used[] = {gone, kept};
  expert_store_count_use(st, 0, used, 2);
  check(WHERE(0, gone) < 0 && st->stats.misses == misses &&
            st->stats.hits == hits + 1,
        "a use counts the expert in a slot and reads none");
  check(st->slots[WHERE(0, kept)].used == st->token &&
            st->head == WHERE(0, kept),
        "a use makes the slot recent");
  for (int i = 0; i < 8; i++) fixture_free(&slots[i].b);

  // A layout's transform runs once an expert's pieces are all read, before the
  // expert counts as read: it sees the read bytes at the spans' offsets and
  // writes the layout's extra bytes at the slot's end, which the slot's size
  // includes. One layer of 2 experts, each 2 spans (5 pieces) read from
  // /dev/zero.
  int zero = open("/dev/zero", O_RDONLY);
  static FileSpan read_span[1][2][2];
  static int read_spans[1] = {2};
  for (int e = 0; e < 2; e++)
    read_span[0][e][0] = (FileSpan){zero, IO_SECTOR + 100 * e, read_len[0]},
    read_span[0][e][1] = (FileSpan){zero, 7, read_len[1]};
  static Marks marks;
  st = expert_store_new(NULL, &(ExpertLayout){.layers = 1,
                                              .scored_layers = 1,
                                              .experts = 2,
                                              .chosen = 1,
                                              .parts = 2,
                                              .span = &read_span[0][0][0],
                                              .spans = read_spans,
                                              .extra = EXTRA,
                                              .transform = mark,
                                              .context = &marks});
  expert_store_minimum_bytes(st, 1);
  check(expert_store_slot_bytes(st) ==
            EXTRA + IO_REGION(read_len[0]) + IO_REGION(read_len[1]),
        "a slot has room for the spans' sectors and the extra bytes");
  Slot read_slots[2] = {0};
  double read_scores[2] = {INFINITY, INFINITY};
  int read_frees[2] = {1, 0};
  st->slots = read_slots, st->score = read_scores, st->free_list = read_frees;
  st->max_slots = st->cap = st->mcap = st->nfree = 2;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  int pair[] = {0, 1};
  expert_store_hold(
      st, 0, pair,
      2);  // returns the reads still pending when checked, which varies
  expert_store_wait_for_reads(st, 0, pair, 2);
  check(marks.calls == 2 && marks.early == 0 && marks.unread == 0,
        "the transform runs once per expert, after all its pieces were read");
  for (int e = 0; e < 2; e++) {
    const Buf *slot = expert_store_get_buffer(st, 0, e, at, &fresh);
    int marked = slot != NULL;
    for (int b = 0; marked && b < EXTRA; b++)
      marked = slot->ptr[slot->size - EXTRA + b] == e + 1;
    check(marked && marks.size[e] == slot->size && at[0] == marks.at[e][0] &&
              at[1] == marks.at[e][1],
          "the transform's bytes end at the slot's end, and it saw the spans' "
          "offsets");
    check(at[1] >= IO_REGION(read_len[0]) &&
              at[1] + read_len[1] <= slot->size - EXTRA,
          "the second span follows the first span's sectors, before the extra "
          "bytes");
  }
  expert_store_release(st, 0, pair, 2);
  close(zero);
  for (int i = 0; i < 2; i++) fixture_free(&read_slots[i].b);

  // A pass's fewest slots are room that the model reserved before the pass,
  // so the slot limit rises to them. A store whose limit was set for one row
  // (4 slots) holds a whole layer of 8 experts in a prompt pass of 4 rows (16
  // slots); without the room, no slot of the layer may go and the process
  // ends. The case runs in a child, which such an end fails.
  pid_t child = fork();
  if (!child) {
    static FileSpan wide_span[2][8]
                             [1];  // empty spans: their reads finish at once
    static int wide_spans[2] = {1, 1};
    st = expert_store_new(NULL, &(ExpertLayout){.layers = 2,
                                                .scored_layers = 2,
                                                .experts = 8,
                                                .chosen = 2,
                                                .parts = 1,
                                                .span = &wide_span[0][0][0],
                                                .spans = wide_spans});
    static Slot wide[16];
    static double wide_scores[16];
    static int wide_frees[16];
    for (int i = 0; i < 16; i++)
      wide_scores[i] = INFINITY, wide_frees[i] = 15 - i;
    st->slots = wide, st->score = wide_scores, st->free_list = wide_frees;
    st->max_slots = st->nfree = 16, st->SLOT_SIZE = 64;
    st->head = st->tail = st->pass_layer = st->serving = -1;
    memset(st->where, 0xff, WHERE_BYTES);
    expert_store_set_pass_rows(st, 1);
    st->cap = st->mcap = st->minimum_slots;  // as the limits are set for one
                                             // row, with little memory free
    expert_store_set_pass_rows(st, 4);
    // A pass of one row beside this one leaves the slots that this one needs.
    expert_store_set_other_passes(st, 1);
    expert_store_set_pass_rows(st, 1);
    int kept = st->minimum_slots == 16;
    expert_store_set_other_passes(st, 0);
    expert_store_set_pass_rows(st, 1);
    kept &= st->minimum_slots == 4;
    expert_store_set_pass_rows(st, 4);
    int layer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    expert_store_set_prompt_layer(st, 0);
    expert_store_hold(st, 0, layer, 8);
    expert_store_wait_for_reads(st, 0, layer, 8);
    int held = st->nslots == 8;
    for (int e = 0; e < 8; e++) held &= WHERE(0, e) >= 0;
    _exit(held && kept ? 0 : 1);
  }
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "the slot limit rises to a pass's fewest slots, and passes running "
        "together keep them");

  // Preloading reads and maps all scored experts when capacity holds them all,
  // and leaves the store streaming when capacity is smaller.
  static FileSpan pre_spans[2][3][1];
  static int pre_counts[2] = {1, 1};
  int pre_fd = open("/dev/zero", O_RDONLY);
  for (int l = 0; l < 2; l++)
    for (int e = 0; e < 3; e++)
      pre_spans[l][e][0] = (FileSpan){.fd = pre_fd, .off = 0, .len = 64};
  st = expert_store_new(NULL, &(ExpertLayout){.layers = 2,
                                              .scored_layers = 2,
                                              .experts = 3,
                                              .chosen = 1,
                                              .parts = 1,
                                              .span = &pre_spans[0][0][0],
                                              .spans = pre_counts});
  static Slot pre_slots[6];
  static double pre_scores[6];
  static int pre_frees[6];
  for (int i = 0; i < 6; i++) pre_scores[i] = INFINITY, pre_frees[i] = 5 - i;
  st->slots = pre_slots, st->score = pre_scores, st->free_list = pre_frees;
  st->max_slots = st->nfree = 6, st->SLOT_SIZE = 16384;
  st->cap = 4, st->mcap = 6;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  check(expert_store_preload_if_fits(st) == 0 && st->nslots == 0,
        "preload refuses when cap cannot hold all scored experts");
  st->cap = 6;
  check(expert_store_preload_if_fits(st) == 1 && st->nslots == 6 &&
            st->nmapped == 6 && !st->any_fresh,
        "preload reads, maps and commits every scored expert when all fit");
  int all_present = 1;
  for (int l = 0; l < 2; l++)
    for (int e = 0; e < 3; e++) all_present &= WHERE(l, e) >= 0;
  check(all_present && expert_store_preload_if_fits(st) == 1,
        "all preloaded experts stay resident and a second preload succeeds");
  close(pre_fd);
  for (int i = 0; i < 6; i++) fixture_free(&pre_slots[i].b);
  return failures != 0;
}
