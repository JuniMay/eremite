#include "check.h"
#include "engine/expert_store.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

// Tests slots sized per layer: when a store's layers differ in expert size, a
// slot's buffer has its layer's size and the cache's limits count bytes.
// Buffers are plain memory, optionally counted in a memory pool, and the
// experts are read from /dev/zero; no GPU or model is used.

static int freed;
// The memory that the pool's cases count the slots in.
static Memory *memory;

static Buf fixture_new(size_t bytes) {
  void *p = malloc(bytes);
  memset(p, 0xaa, bytes);
  return (Buf){.obj = p, .ptr = p, .size = bytes, .allocated = bytes};
}

// With a memory pool: counted in the slot as the GPU's buffers are, or
// refused.
static Buf fixture_try_new(Gpu *gpu, MemorySlot *slot, size_t bytes,
                           MemoryAdmission admission) {
  (void)gpu;
  if (!memory_claim(slot, bytes, admission)) return (Buf){0};
  Buf b = fixture_new(bytes);
  b.memory = slot;
  return b;
}

static void fixture_free(Buf *b) {
  if (b->memory) memory_release(b->memory, b->allocated);
  free(b->ptr);
  *b = (Buf){0};
  freed++;
}

static int pooled;  // the slots are entries of a memory pool

// With a pool, memory that the pool must make room for.
static Buf fixture_required(Gpu *gpu, MemorySlot *slot, size_t bytes) {
  if (!pooled) return fixture_new(bytes);
  Buf b = fixture_try_new(gpu, slot, bytes, MEMORY_REQUIRED);
  if (!b.obj)
    fprintf(stderr, "the pool cannot take %zu required bytes\n", bytes),
        abort();
  return b;
}

#define gpu_alloc fixture_required
#define gpu_try_alloc fixture_try_new
#define gpu_free fixture_free
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#undef gpu_alloc
#undef gpu_try_alloc
#undef gpu_free

// Three layers of 8 experts, which take 16, 17 and 24 sectors.
enum { LAYERS3 = 3, EXPERTS8 = 8, SLOTS = 96 };

#define SMALL (16 * IO_SECTOR)
#define MIDDLE (17 * IO_SECTOR)
#define LARGE (24 * IO_SECTOR)
static FileSpan span[LAYERS3][EXPERTS8][1];
static int spans[LAYERS3] = {1, 1, 1};
static ExpertStore *st;
static Slot slots[SLOTS];
static double scores[SLOTS];
static int frees[SLOTS];

static MemoryAccount account;
static MemorySlot backbone;

// A pass that demands one expert, for a demand that has to wait.
static int wanted[2], got = -9;

static void demand(void *argument) {
  (void)argument;
  got = take(st, wanted[0], wanted[1], 0);
}

// A store whose layers' experts have these payload bytes, with room for units
// slots of the largest size.
static void setup(uint64_t len0, uint64_t len1, uint64_t len2, int units) {
  static int zero = -1;
  if (zero < 0) zero = open("/dev/zero", O_RDONLY);
  if (!st)
    st = expert_store_new(NULL, &(ExpertLayout){.layers = LAYERS3,
                                                .scored_layers = LAYERS3,
                                                .experts = EXPERTS8,
                                                .chosen = 2,
                                                .parts = 1,
                                                .span = &span[0][0][0],
                                                .spans = spans});
  for (int i = 0; st->slots && i < SLOTS; i++) {
    finish(st, i);
    if (slots[i].b.obj) fixture_free(&slots[i].b);
  }
  if (pooled) {
    for (int i = 0; i < SLOTS; i++) memory_slot_remove(&slots[i].memory);
    memory_slot_remove(&backbone);
    expert_store_set_backbone(st, NULL);
    pooled = 0;
  }
  uint64_t len[LAYERS3] = {len0, len1, len2};
  for (int l = 0; l < LAYERS3; l++)
    for (int e = 0; e < EXPERTS8; e++)
      span[l][e][0] = (FileSpan){zero, 0, len[l]};
  st->layout_ready = 0, st->SLOT_SIZE = 0,
  st->sized = 0;  // the layout is read again
  memset(st->layer_bytes, 0, LAYERS3 * sizeof *st->layer_bytes);
  expert_store_minimum_bytes(st, LAYERS3);
  memset(slots, 0, sizeof slots);
  for (int i = 0; i < SLOTS; i++)
    scores[i] = INFINITY, frees[i] = SLOTS - 1 - i;
  st->slots = slots, st->score = scores, st->free_list = frees;
  st->max_slots = st->nfree = SLOTS, st->cap = st->mcap = st->max_cap = units;
  st->nslots = st->nmapped = 0,
  st->bytes = st->mapped_bytes = st->conversions = 0;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  st->cache_busy = 0;
  st->token = 10, st->weight = st->near_weight = 1;
  memset(st->where, 0xff, WHERE_BYTES);
  memset(st->escore, 0, SCORE_BYTES), memset(st->dscore, 0, SCORE_BYTES),
      memset(st->nscore, 0, SCORE_BYTES);
  memset(&st->stats, 0, sizeof st->stats);
  added = removed = freed = 0;
}

// Makes the store's slots entries of a memory pool of limit bytes, as the
// cache's initialization does; setup takes them out again.
static void pool(uint64_t limit) {
  static int registered;
  if (!registered)
    memory = memory_new(), memory_account_init(memory, &account, "model", 0),
    registered = 1;
  memory_set_limit(memory, limit);
  memory_slot_init(&backbone, &account, MEMORY_MODEL, NULL, "backbone", NULL,
                   NULL);
  expert_store_set_backbone(st, &backbone);
  for (int i = 0; i < SLOTS; i++) {
    slots[i].store = st;
    memory_slot_init(&slots[i].memory, &account, MEMORY_MODEL, &backbone,
                     "routed expert", pool_evict, &slots[i]);
    slots[i].memory.can_evict = pool_can_evict, slots[i].memory.automatic = 1;
  }
  pooled = 1;
}

static uint64_t
held_bytes(void) {  // what the slots' buffers take, counted from the slots
  uint64_t bytes = 0;
  for (int i = 0; i < st->max_slots; i++)
    if (slots[i].b.obj) bytes += slots[i].b.size;
  return bytes;
}

int main(void) {
  // The layers' sizes are their spans' sectors; the limits count the largest.
  setup(220000, 236000, 350000, 2);
  check(st->sized && st->SLOT_SIZE == LARGE,
        "layers that differ by more than a sixteenth make a sized store");
  check(st->layer_bytes[0] == SMALL && st->layer_bytes[1] == MIDDLE &&
            st->layer_bytes[2] == LARGE,
        "each layer's bytes are its experts' sectors");
  check(slot_need(st, 0) == SMALL && slot_need(st, 2) == LARGE,
        "a slot needs its layer's bytes");
  check(slot_fits(MIDDLE, SMALL) && !slot_fits(SMALL, MIDDLE) &&
            !slot_fits(LARGE, SMALL) && slot_fits(SMALL, SMALL),
        "a buffer fits an expert that it holds with at most a sixteenth to "
        "spare");

  // Bytes, not slots: a limit of two of the largest slots (48 sectors) holds
  // three small experts.
  int a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  check(st->nslots == 3 && st->bytes == 3 * SMALL && held_bytes() == st->bytes,
        "two of the largest slots' bytes hold three small experts");
  check(slots[a].b.size == SMALL && slots[b].b.size == SMALL &&
            slots[c].b.size == SMALL,
        "small experts get small buffers");
  check(expert_store_bytes(st) == 3 * SMALL &&
            expert_store_slot_bytes(st) == SMALL &&
            expert_store_slot_count(st) == 3,
        "the cache reports its bytes and slots");
  check(expert_store_slot_limit(st) == 3 && expert_store_slot_maximum(st) == 3,
        "no further slot of the largest size fits");
  // A fourth takes the lowest-priority slot, whose buffer has the size.
  st->score[a] = 3, st->score[b] = 1, st->score[c] = 2;
  int d = take(st, 0, 3, 0);
  check(d == b && st->nslots == 3 && st->bytes == 3 * SMALL &&
            WHERE(0, 1) < 0 && WHERE(0, 3) == b && st->conversions == 0 &&
            freed == 0,
        "a fourth small expert reuses the lowest slot's buffer");
  // A buffer a sixteenth larger than the expert is reused; a larger expert
  // does not go into a smaller buffer.
  setup(220000, 236000, 350000, 2);
  a = take(st, 1, 0, 0),
  b = take(st, 1, 1, 0);  // two middle buffers, 34 of 48 sectors
  st->score[a] = 1, st->score[b] = 2;
  c = take(st, 0, 0, 0);  // a small expert: 16 sectors more do not fit, the
                          // lowest middle buffer does
  check(c == a && slots[c].b.size == MIDDLE && st->nslots == 2 &&
            st->bytes == 2 * MIDDLE && st->conversions == 0,
        "a small expert reuses a middle buffer, a sixteenth larger");

  // A conversion: no buffer holds the large expert, so the two lowest
  // priorities give theirs back.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 5, st->score[b] = 1, st->score[c] = 2;
  d = take(st, 2, 0, 0);
  check(
      st->conversions == 1 && freed == 2 && slots[d].b.size == LARGE,
      "a large expert converts the two lowest small buffers into a large one");
  check(WHERE(0, 0) == a && WHERE(0, 1) < 0 && WHERE(0, 2) < 0 &&
            WHERE(2, 0) == d,
        "the conversion evicts the two lowest experts");
  check(st->nslots == 2 && st->bytes == SMALL + LARGE &&
            held_bytes() == st->bytes && st->stats.evictions == 2,
        "the conversion's bytes add up");
  // The other way: a large buffer would waste a third of itself on a small
  // expert. With the small buffer held, the large one gives way to a small
  // one, and the bytes left over hold a second small expert without another
  // eviction.
  st->score[a] = 9, st->score[d] = 1;
  slots[a].holds = 1;
  int e = take(st, 0, 3, 0);
  check(st->conversions == 2 && slots[e].b.size == SMALL && WHERE(2, 0) < 0 &&
            st->bytes == 2 * SMALL,
        "a small expert converts a large buffer into a small one");
  int f = take(st, 0, 4, 0);
  check(st->conversions == 2 && st->nslots == 3 && st->bytes == 3 * SMALL &&
            WHERE(0, 0) == a && WHERE(0, 3) == e && WHERE(0, 4) == f,
        "the bytes left over hold another small expert");
  slots[a].holds = 0;

  // Among the lowest priorities a fitting buffer is reused rather than a
  // buffer converted: 70 large buffers and one small one, in a cache of 71 of
  // the largest.
  for (int near = 0; near < 2; near++) {
    setup(220000, 236000, 350000, 71);
    for (int k = 0; k < 70; k++) {
      int i = take(st, 2, k % EXPERTS8, 0);
      place(st, 2, k % EXPERTS8, -1);  // 70 buffers of the large layer; which
                                       // experts they hold does not matter
      slots[i].expert = -1, st->score[i] = k + 1;
    }
    int small = take(st, 0, 0, 0);
    // near: 50 of the large buffers are lower, fewer than FIT_RANK, so the
    // small buffer is taken. Otherwise all 70 are lower, and the lowest large
    // buffer is given back for a small one.
    st->score[small] = near ? 50.5 : 100;
    int before = st->nslots;
    int got = take(st, 0, 1, 0);
    if (near)
      check(got == small && st->conversions == 0 && WHERE(0, 0) < 0 &&
                st->nslots == before,
            "a fitting buffer among the FIT_RANK lowest is reused");
    else
      check(got != small && st->conversions == 1 && WHERE(0, 0) == small &&
                slots[got].b.size == SMALL && freed == 1 &&
                st->nslots == before,
            "a fitting buffer above the FIT_RANK lowest stays, and the lowest "
            "is converted");
  }

  // What a conversion may give back while decoding: no expert used or read
  // ahead this token.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 1, st->score[b] = 2, st->score[c] = 3;
  slots[a].used =
      st->token;  // the lowest was used this token, by a layer already served
  d = take(st, 2, 0, 0);
  check(st->conversions == 1 && WHERE(0, 0) == a && WHERE(0, 1) < 0 &&
            WHERE(0, 2) < 0 && slots[d].b.size == LARGE,
        "a conversion keeps an expert used this token");
  // A read-ahead gives at most two buffers back, and none that is still
  // protected.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 1, st->score[b] = 2, st->score[c] = 3;
  slots[b].used = slots[c].used = st->token;
  check(take(st, 2, 0, 1) == -1 && WHERE(0, 1) == b && WHERE(0, 2) == c,
        "a read-ahead with one buffer to give back gets no slot");
  check(st->bytes == 2 * SMALL && st->nslots == 2,
        "what it gave back stays free for the next request");
  slots[b].used = slots[c].used = st->token - 1;
  d = take(st, 2, 0, 1);
  check(d >= 0 && slots[d].b.size == LARGE && st->bytes == SMALL + LARGE,
        "a read-ahead converts when two buffers can go back");
  // A demand that can give nothing back reuses the lowest buffer that is
  // large enough.
  setup(220000, 236000, 350000, 2);
  a = take(st, 2, 0, 0), b = take(st, 2, 1, 0);
  st->score[a] = 1, st->score[b] = 2;
  slots[a].used = slots[b].used = st->token, st->serving = -1;
  c = take(st, 0, 0, 0);
  check(c == a && slots[c].b.size == LARGE && st->conversions == 0 &&
            WHERE(2, 0) < 0 && WHERE(0, 0) == a,
        "a demand that can give nothing back reuses the lowest large enough "
        "buffer");
  // Among the buffers that are large enough it keeps the victim's
  // preferences: a read-ahead that is still protected stays while another
  // slot will do, although its priority is lower.
  setup(220000, 236000, 350000, 2);
  a = take(st, 2, 0, 0), b = take(st, 2, 1, 0);
  st->score[a] = 1, st->score[b] = 2;
  slots[a].used = slots[b].used = st->token, slots[a].prefetched = 1;
  c = take(st, 0, 0, 0);
  check(c == b && WHERE(2, 0) == a && WHERE(2, 1) < 0 && WHERE(0, 0) == b &&
            st->conversions == 0,
        "the demand keeps a protected read-ahead while another buffer will do");
  // When only the read-ahead's buffer is large enough, the demand takes it.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 2, 0, 0);  // 40 of 48 sectors
  slots[a].used = slots[b].used = st->token, slots[b].prefetched = 1;
  c = take(st, 1, 0, 0);  // a middle expert: the small buffer cannot hold it
                          // and nothing can be given back
  check(
      c == b && WHERE(2, 0) < 0 && WHERE(0, 0) == a && WHERE(1, 0) == b &&
          st->conversions == 0 && st->nslots == 2,
      "the demand takes the read-ahead's buffer when only it is large enough");

  // A prompt pass allocates buffers of the largest size, which serve all its
  // layers; decoding allocates its experts' sizes.
  setup(220000, 236000, 350000, 3);
  expert_store_set_prompt_layer(st, 0);
  a = take(st, 0, 0, 0);
  expert_store_set_prompt_layer(st, -1);
  b = take(st, 0, 1, 0);
  check(slots[a].b.size == LARGE && slots[b].b.size == SMALL &&
            st->bytes == LARGE + SMALL,
        "a prompt pass allocates the largest size, decoding the expert's");
  // A prompt pass at layer 2 over a cache that decoding filled. Of the slots
  // it may take, the layers it is done with, all of one priority before any
  // decoding, the least recently used whose buffer is large enough goes
  // first.
  setup(220000, 236000, 350000, 3);
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0), c = take(st, 0, 1, 0),
  d = take(st, 1, 1, 0);  // 66 of 72 sectors
  expert_store_set_prompt_layer(st, 2);
  int m = take(st, 1, 2, 0);  // a middle expert: the oldest slot (a) is too
                              // small, the oldest large enough is b
  check(m == b && WHERE(1, 0) < 0 && WHERE(0, 0) == a && st->conversions == 0 &&
            st->bytes == 2 * SMALL + 2 * MIDDLE,
        "a prompt pass takes the least recently used buffer that is large "
        "enough");
  // An expert of the pass's own layer finds no buffer large enough. Buffers go
  // back, the least recently used first, but none of the layer before the
  // pass's own.
  int big = take(st, 2, 0, 0);
  check(st->conversions == 1 && slots[big].b.size == LARGE && WHERE(0, 0) < 0 &&
            WHERE(0, 1) < 0,
        "a prompt pass converts the least recently used buffers");
  check(WHERE(1, 1) == d && WHERE(1, 2) == b &&
            st->bytes == 2 * MIDDLE + LARGE && held_bytes() == st->bytes,
        "the conversion keeps the layer before the pass's own");
  // A smaller expert of the pass goes into the large buffer it is done with:
  // no buffer is given back.
  expert_store_set_prompt_layer(st,
                                3);  // the store has three layers; the pass's
                                     // layer only orders the victims here
  int again = take(st, 0, 5, 0);
  check(st->conversions == 1 && st->bytes == 2 * MIDDLE + LARGE &&
            slots[again].b.size >= SMALL,
        "a smaller expert of the pass reuses a large buffer without a "
        "conversion");
  expert_store_set_prompt_layer(st, -1);
  // A prompt pass that can give nothing back reuses a buffer that is large
  // enough from a layer it is done with, not one of its own layer whose
  // expert decoding used less.
  setup(220000, 236000, 350000, 3);
  a = take(st, 0, 0, 0);  // a small buffer, as decoding allocates it
  expert_store_set_prompt_layer(st, 0);
  b = take(st, 0, 1, 0);  // the pass's buffers have the largest size
  expert_store_set_prompt_layer(st, 1);
  c = take(st, 1, 0, 0);  // 64 of 72 sectors
  st->dscore[0 * EXPERTS8 + 1] = 5,
                            st->score[b] =
                                5;  // decoding used b's expert, and not c's
  d = take(st, 1, 1, 0);  // a (the victim) is too small, and no buffer of
                          // layers 0 and 1 may be given back
  check(d == b && WHERE(0, 1) < 0 && WHERE(1, 0) == c && WHERE(0, 0) == a &&
            WHERE(1, 1) == b,
        "a prompt pass reuses a done layer's large buffer rather than its own "
        "layer's");
  check(st->conversions == 0 && st->bytes == SMALL + 2 * LARGE && freed == 0,
        "that reuse gives nothing back");
  expert_store_set_prompt_layer(st, -1);
  // Resizing counts bytes: the limit falls to one largest slot, and the least
  // recently used leave until the rest fit; the mapped bytes follow their own
  // limit.
  setup(220000, 236000, 350000, 3);
  int held[4];
  for (int k = 0; k < 4; k++) held[k] = take(st, 0, k, 0);  // 64 of 72 sectors
  uint32_t at[EXPERT_STORE_PARTS];
  int fresh;
  for (int k = 0; k < 4; k++) expert_store_get_buffer(st, 0, k, at, &fresh);
  check(st->nmapped == 4 && st->mapped_bytes == 4 * SMALL,
        "the mapped bytes count the mapped buffers");
  st->cap = 1;
  for (int i = st->tail;
       over_cap(st) && i >= 0;) {  // as a resize gives slots back
    int prev = slots[i].prev;
    release(st, i);
    i = prev;
  }
  check(st->nslots == 1 && st->bytes == SMALL && st->nmapped == 1 &&
            st->mapped_bytes == SMALL && WHERE(0, 3) == held[3],
        "a lower limit keeps the most recently used slots that fit");
  check(used_units(st) == 1 && mapped_units(st) == 1 && !over_cap(st) &&
            !over_mcap(st),
        "the remaining bytes are within both limits");
  st->mcap = 0;
  check(over_mcap(st), "the mapped bytes follow their own limit");

  // Experts are read into buffers of their own size, and the reads stay
  // inside them.
  setup(220000, 236000, 350000, 4);
  int ids[] = {0, 1};
  expert_store_read_and_wait(st, 0, ids, 2);
  expert_store_read_and_wait(st, 2, ids, 2);
  for (int k = 0; k < 2; k++) {
    Slot *s0 = &slots[WHERE(0, k)], *s2 = &slots[WHERE(2, k)];
    check(s0->b.size == SMALL && s2->b.size == LARGE && s0->extent <= SMALL &&
              s2->extent <= LARGE,
          "experts are read into buffers of their own size");
    check(s0->b.ptr[s0->at[0]] == 0 && s0->b.ptr[s0->at[0] + 219999] == 0 &&
              s2->b.ptr[s2->at[0] + 349999] == 0,
          "the reads write the experts' bytes");
  }
  expert_store_wait_for_all_reads();

  // Layers within a sixteenth of the largest keep slots of one size, the
  // largest.
  setup(220000, 236000, 236000, 2);
  check(
      !st->sized && st->SLOT_SIZE == MIDDLE && st->layer_bytes[0] == SMALL &&
          slot_need(st, 0) == MIDDLE,
      "layers within a sixteenth of the largest make a store of one slot size");
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0);
  check(slots[a].b.size == MIDDLE && slots[b].b.size == MIDDLE &&
            st->nslots == 2,
        "every slot then has the largest size");
  st->score[a] = 2, st->score[b] = 1;
  check(take(st, 0, 1, 0) == b && st->nslots == 2 && st->conversions == 0,
        "any expert then reuses the lowest slot");
  // A store whose layers have one size keeps slots, and its limits count
  // them.
  setup(220000, 220000, 220000, 2);
  check(!st->sized && st->SLOT_SIZE == SMALL && slot_need(st, 1) == SMALL,
        "layers of one size make slots of that size");
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0);
  st->score[a] = 1, st->score[b] = 2;
  c = take(st, 2, 0, 0);
  check(c == a && st->nslots == 2 && st->conversions == 0 &&
            expert_store_slot_limit(st) == 2 &&
            expert_store_bytes(st) == 2 * SMALL,
        "such a store's limits count slots");

  // In a memory pool that admits 72 sectors, under a cache limit that is not
  // reached: the pool's refusal ends the growth, and a conversion gives back
  // only what the new buffer needs.
  setup(220000, 236000, 350000, 8);
  pool(72 * IO_SECTOR);
  int four[5];
  for (int k = 0; k < 4; k++)
    four[k] = take(st, 0, k, 0), st->score[four[k]] = k + 1;
  check(st->nslots == 4 && st->bytes == 4 * SMALL &&
            memory_usage(memory).allocated == 4 * SMALL,
        "the pool counts each slot's buffer");
  four[4] = take(st, 0, 4,
                 0);  // the pool refuses a fifth buffer: the lowest slot's fits
  check(four[4] == four[0] && st->nslots == 4 && st->conversions == 0 &&
            memory_usage(memory).allocated == 4 * SMALL,
        "the pool's refusal ends the growth, and the lowest slot is reused");
  st->score[four[0]] = 9;
  big = take(st, 2, 0, 0);  // one small buffer back makes the room: 3 small and
                            // 1 large are the 72 sectors
  check(st->conversions == 1 && freed == 1 && slots[big].b.size == LARGE &&
            st->nslots == 4 && WHERE(0, 1) < 0,
        "a conversion in a pool gives back only what the new buffer needs");
  check(st->bytes == 3 * SMALL + LARGE &&
            memory_usage(memory).allocated == st->bytes &&
            held_bytes() == st->bytes,
        "the pool's count matches the cache's bytes");
  // A pool over its limit admits no further buffer. A demand gives at most
  // four buffers back. No buffer is large enough for it, so it takes its
  // buffer as required memory, and the pool makes the room by taking two
  // further slots back. The buffer belongs to the slot it is counted in.
  setup(220000, 236000, 350000, 12);
  pool(128 * IO_SECTOR);
  for (int k = 0; k < 8; k++) st->score[take(st, 0, k, 0)] = k + 1;
  check(st->nslots == 8 && memory_usage(memory).allocated == 8 * SMALL,
        "eight small slots fill half the pool");
  memory_set_limit(memory, 64 * IO_SECTOR);
  big = take(st, 2, 0, 0);
  check(freed == CONVERT_DEMAND + 2 && st->stats.given_back == 2 &&
            st->nslots == 3 && st->nfree == SLOTS - 3,
        "a demand gives four buffers back, and the pool takes two more for its "
        "required buffer");
  check(slots[big].b.size == LARGE && WHERE(2, 0) == big &&
            slots[big].b.memory == &slots[big].memory,
        "the required buffer belongs to the slot it is counted in");
  check(st->bytes == 2 * SMALL + LARGE && held_bytes() == st->bytes &&
            memory_usage(memory).allocated == st->bytes,
        "the pool's count matches the cache's bytes");
  int lowest_gone = 1;
  for (int k = 0; k < CONVERT_DEMAND; k++) lowest_gone &= WHERE(0, k) < 0;
  check(lowest_gone, "the lowest priorities went first");
  int payloads = 1;
  for (int i = 0; i < SLOTS; i++)
    payloads &=
        slots[i].memory.payload == (slots[i].b.obj ? slots[i].b.size : 0);
  check(payloads, "every slot's count is its buffer");
  // A read-ahead in a shortage, with the large buffer in use, gives two back
  // and gets no slot.
  memory_set_limit(memory, 16 * IO_SECTOR);
  freed = 0, slots[big].holds = 1;
  check(take(st, 2, 1, 1) == -1 && freed == CONVERT_OPTIONAL &&
            st->nslots == 1 && st->bytes == LARGE,
        "a read-ahead in a shortage gives two buffers back and gets no slot");
  slots[big].holds = 0;

  // A busy cache whose pool has no room even for required memory: two slots
  // are held and two were used this token, so the pool may take none back.
  // The demand gives back what nothing holds, the lowest priority first,
  // until its buffer fits; the program does not stop.
  setup(220000, 236000, 350000, 4);
  pool(64 * IO_SECTOR);
  for (int k = 0; k < 4; k++)
    four[k] = take(st, 0, k, 0), st->score[four[k]] = k + 1;
  slots[four[0]].holds = slots[four[1]].holds = 1;
  slots[four[2]].used = slots[four[3]].used = st->token;
  expert_store_set_request_running(st, 1);
  big = take(st, 2, 0, 0);
  check(slots[big].b.size == LARGE && WHERE(2, 0) == big &&
            slots[big].b.memory == &slots[big].memory,
        "a demand in a full pool gets its buffer");
  check(WHERE(0, 0) == four[0] && WHERE(0, 1) == four[1] && WHERE(0, 2) < 0 &&
            WHERE(0, 3) < 0,
        "it gives back what nothing holds");
  check(st->nslots == 3 && st->bytes == 2 * SMALL + LARGE &&
            memory_usage(memory).allocated == st->bytes &&
            held_bytes() == st->bytes,
        "the pool's count matches the cache's bytes");
  slots[four[0]].holds = slots[four[1]].holds = 0;
  // With every other slot held, the demand has given back what nothing held
  // and still has no room: its pass waits, as a pass does when every slot is
  // held, until another pass releases a slot.
  setup(220000, 236000, 350000, 4);
  pool(40 * IO_SECTOR);
  a = take(st, 2, 0, 0), b = take(st, 0, 0, 0);  // the pool's 40 sectors
  slots[a].holds = 1, slots[b].used = st->token;
  expert_store_set_request_running(st, 1);
  // The test stands in for the other pass that holds slot a.
  expert_store_set_other_passes(st, 1);
  wanted[0] = 2, wanted[1] = 1;
  Pass *waiting = pass_new(demand, NULL, 1 << 20);
  int yielded = !pass_resume(waiting);
  check(yielded && got == -9 && !pass_ready(waiting),
        "a demand with no room waits");
  check(WHERE(0, 0) < 0 && WHERE(2, 0) == a && st->nslots == 1 &&
            st->bytes == LARGE,
        "before waiting it gives back what nothing holds");
  slots[a].holds = 0;  // the other pass's GPU work has finished
  check(pass_ready(waiting), "the demand can continue once a hold goes");
  check(pass_resume(waiting) && got == a && WHERE(2, 1) == a &&
            WHERE(2, 0) < 0 && st->nslots == 1 && st->bytes == LARGE,
        "the released slot holds the demanded expert");
  pass_free(waiting);
  expert_store_set_other_passes(st, 0);
  expert_store_set_request_running(st, 0);
  return failures != 0;
}
