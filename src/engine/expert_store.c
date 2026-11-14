#include "engine/expert_store.h"
#include "base/clock.h"
#include "base/error.h"
#include "engine/io.h"
#include "platform/system_memory.h"
#include "platform/threads.h"
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A slot holds one expert's spans one after the other. Every slot's buffer
// has the size of the largest expert, so that any expert fits any slot; when
// a layer's experts are more than a sixteenth smaller than the largest, a
// buffer has the size of its layer's experts instead, and the cache counts
// bytes (sized).
//
// The stages' experts are preloaded into slots of their own, whose bytes come
// off the same budget and which the backbone's eviction does not reuse.
//
// The GPU reads a slot only while it is mapped, in the queue's residency sets.
// The mapped slots stay below the point where the GPU stalls: a slot whose
// expert a layer needs is mapped before the GPU uses it, and between tokens,
// or in a prompt for the layers done, the mapped slots with the lowest scores
// leave the sets until the rest fit.
//
// The cache counts free pages and reclaimable file pages as available, and
// leaves a reserve for other programs. It gives slots back when the kernel
// reports pressure, when anything was swapped out, or when the available
// memory falls below the reserve, waiting freed_memory_delay between steps for
// the memory given back to show; it grows again only after a calm period. It
// checks after every token, after every layer of a prompt, and when idle.

static constexpr double GiB = 1024.0 * 1024 * 1024;

typedef struct {
  Buf b;
  // Where the spans are in b.
  uint32_t at[EXPERT_STORE_PARTS];
  int16_t layer, expert;
  // Links in the recency list, head = most recent.
  int prev, next;
  // Read ahead on a prediction, not requested yet.
  int prefetched;
  // The pass that predicted it, the only one that cancels or demotes it.
  const Pass *issuer;
  // A stage's slot, which the backbone's eviction and unmapping leave alone.
  int pinned;
  // Mapped since the residency sets were last committed: needs gpu_declare_read
  // until then.
  int fresh;
  // In the residency sets, or joining them while fresh.
  int mapped;
  // Mapped before: the next mapping is a remap.
  int mapped_once;
  // The token of the last use.
  uint64_t used;
  // Passes whose GPU work uses the expert.
  unsigned holds;
  // The load's pieces, until the load is finished.
  ThreadsGroup *loading;
  IoPiece piece[16];
  int npiece;
  // The load counts as a read-ahead in flight.
  int background_load;
  // Pieces not read or dropped yet.
  _Atomic int unfinished;
  // Piece k was started or dropped.
  _Atomic char taken[16];
  // Pieces that a reader completed.
  _Atomic int read;
  // The bytes of b that the load writes.
  uint64_t extent;
  // The bytes of b that finished loads wrote.
  uint64_t written;
  MemorySlot memory;
  // For the memory pool's callbacks.
  ExpertStore *store;
} Slot;

static uint64_t reserve;  // every store keeps it available to other programs
// Every store's settings, with the defaults filled in.
static MemorySettings cache_settings = EXPERT_STORE_DEFAULT_SETTINGS;

// The reader threads of a store, and the entries of its queue of pieces to
// read.
static constexpr int READERS = 4;
static constexpr uint32_t QUEUE = 1u << 14;

struct ExpertStore {
  // In stores.
  int index;
  // The GPU that the slots' buffers are made on.
  Gpu *gpu;
  // Passes with GPU work besides the running one.
  int other_passes;
  ExpertStoreCounts stats;
  // The layout.
  int layers, scored_layers, experts, chosen, stage_experts, parts;
  const FileSpan *span;
  const int *spans;
  uint64_t extra;
  void (*transform)(void *context, int layer, int expert, uint8_t *slot,
                    const uint32_t *at, uint64_t slot_bytes);
  void *context;
  // The largest expert in use.
  uint64_t SLOT_SIZE;
  // When the scored layers' experts differ in size by more than a sixteenth,
  // the store is sized: a slot's buffer has the size of its layer's experts
  // (layer_bytes), or the largest when a prompt pass allocates it, and the
  // cache counts bytes. cap, mcap and max_cap are then bytes in units of
  // SLOT_SIZE, and more slots than cap fit in them. Otherwise every slot has
  // SLOT_SIZE and the limits count slots.
  // [scored_layers]; 0 while the layer's spans are unknown.
  uint64_t *layer_bytes;
  int sized;
  // The scored slots' buffers, and those mapped for the GPU.
  uint64_t bytes, mapped_bytes;
  // The most that cap and mcap may be (max_slots when not sized).
  int max_cap;
  // Buffers given back for one of another size.
  uint64_t conversions;
  uint64_t token;
  Slot *slots;
  int pass_layer;
  MemorySlot *cache_backbone;
  int cache_busy;
  int minimum_slots;
  double last_pool_growth;
  // [layers][experts]: an expert's slot, -1 for none.
  int16_t *where;
  // [layers]: counts the changes of the layer's where.
  uint64_t *generation;
  // The last finding that a layer was ready: its generation then, and the
  // unmappings by then. Until either changes, the layer stays ready.
  uint64_t *ready_generation, *ready_unmaps, unmaps;
  // A slot may be fresh.
  int any_fresh;
  int nslots, cap, max_slots, *free_list, nfree, npinned;
  // Slots mapped for the GPU, and how many may be.
  int nmapped, mcap;
  uint64_t pinned_bytes, pinned_slot_size;
  int layout_ready;
  // The eviction order.
  int head, tail;
  // The current token's.
  double weight;
  // [layers][experts]: per expert.
  double *escore;
  // [layers][experts]: the part of escore that decoding's requests gave.
  double *dscore;
  // The current token's weight of a near miss.
  double near_weight;
  // [layers][experts]: near misses, each of the near weight then.
  double *nscore;
  // Per slot: its expert's priority; INFINITY for a free slot, -INFINITY for an
  // empty one.
  double *score;
  // The readers' queue: entries are slot << 4 | piece.
  uint32_t queue[QUEUE], qhead, qtail;
  // Each entry's layer, as it was queued.
  uint16_t qlayer[QUEUE];
  pthread_mutex_t qlock;
  pthread_cond_t qcond;
  // Demand pieces, and read-ahead slots, in flight.
  _Atomic int demand, prefetching;
  // The layer that requests make resident: its experts used this token stay.
  int serving;
  unsigned held_references;
  // [layers]: passes whose GPU work reaches the layer through its table.
  unsigned *layer_holds;
  // The resizing's readings and clocks; the clock of the last give-back is
  // shared.
  uint64_t swapouts;
  int calm, init;
  // The last swap-out while the cache could release memory.
  double bad;
  int calm_map;
  // When the GPU's excess was last unmapped: the driver shows it within
  // freed_memory_delay.
  double unmapped;
};

// Every store, for what concerns them all: the slot memory that reads have
// not written yet, draining their reads, and the time of the last give-back.
// That clock is shared so that one shortfall goes back once: after any store
// gives slots back, none gives more until the memory readings can show it.
// There are at most STORES stores.
static constexpr int STORES = 8;

static ExpertStore *stores[STORES];
static int nstores;
static double gave;  // when a store last gave slots back
#define WHERE(l, e) st->where[(size_t)(l) * st->experts + (e)]
#define SPAN(l, e, k)                                                          \
  st->span[((size_t)(l) * st->experts + (e)) * st->parts + (k)]
// The arrays' bytes: a slot index and the scores for every expert of every
// layer.
#define WHERE_BYTES ((size_t)st->layers * st->experts * sizeof *st->where)
#define SCORE_BYTES ((size_t)st->layers * st->experts * sizeof *st->escore)

// The model memory follows the available memory, within model_gib when it is
// set, and the slot bytes that reads have not written yet count as in use.
static uint64_t cache_unwritten(void);

void expert_store_set_memory(Memory *memory, MemorySettings settings) {
  const MemorySettings d = EXPERT_STORE_DEFAULT_SETTINGS;
  memory_set_limit(memory, settings.model_gib > 0
                               ? (uint64_t)(settings.model_gib * GiB)
                               : UINT64_MAX);
  expert_store_set_reserve(settings.reserve_gib);
  memory_set_reserve(memory, reserve, system_memory_available, cache_unwritten);
  memory_set_times(memory, settings.times);
  cache_settings = (MemorySettings){
      .model_gib = settings.model_gib,
      .reserve_gib =
          settings.reserve_gib > 0 ? settings.reserve_gib : d.reserve_gib,
      .times = memory_times(memory),
      .free_pages_only_after_swapout =
          settings.free_pages_only_after_swapout > 0
              ? settings.free_pages_only_after_swapout
              : d.free_pages_only_after_swapout,
      .expert_cache_growth_margin_gib =
          settings.expert_cache_growth_margin_gib > 0
              ? settings.expert_cache_growth_margin_gib
              : d.expert_cache_growth_margin_gib,
      .gpu_memory_limit_margin_gib = settings.gpu_memory_limit_margin_gib > 0
                                         ? settings.gpu_memory_limit_margin_gib
                                         : d.gpu_memory_limit_margin_gib};
}

void expert_store_set_reserve(double gib) {
  reserve =
      (uint64_t)((gib > 0 ? gib : EXPERT_STORE_RESERVE_GIB_DEFAULT) * GiB);
}

// Metal's recommended working set, which guides residency; INT64_MAX when
// Metal reports none.
static int64_t gpu_target(const ExpertStore *st) {
  uint64_t target = gpu_recommended_working_set(st->gpu);
  return target ? (int64_t)target : INT64_MAX;
}

// The working set less the GPU driver's resident memory.
static int64_t gpu_room(const ExpertStore *st) {
  int64_t target = gpu_target(st);
  return target == INT64_MAX ? INT64_MAX
                             : target - (int64_t)system_memory_gpu_driver();
}

// Whether the expert in slot s must stay: a pass holds it, or GPU work reaches
// its layer through the layer's table.
static int slot_held(const ExpertStore *st, const Slot *s) {
  return s->holds || (s->expert >= 0 && st->layer_holds[s->layer]);
}

// Puts expert e of the layer in slot i, or in none for -1, which makes a table
// of the layer's addresses out of date.
static void place(ExpertStore *st, int layer, int e, int i) {
  WHERE(layer, e) = i;
  st->generation[layer]++;
}

ExpertStore *expert_store_new(Gpu *gpu, const ExpertLayout *l) {
  if (l->experts < 1 || l->experts > EXPERT_STORE_EXPERTS_MAX ||
      l->scored_layers < 1 || l->scored_layers > l->layers || l->chosen < 1 ||
      l->chosen > l->experts || l->stage_experts > l->experts || l->parts < 1 ||
      l->parts > EXPERT_STORE_PARTS || !l->span || !l->spans)
    die("invalid expert layout");
  if (nstores == STORES) die("more than %d expert stores", STORES);
  ExpertStore *st = xmalloc(sizeof *st);
  memset(st, 0, sizeof *st);
  st->gpu = gpu;
  st->pass_layer = -1, st->cache_busy = 1, st->head = -1, st->tail = -1,
  st->weight = 1;
  st->serving = -1, st->bad = -1e9;
  pthread_mutex_init(&st->qlock, NULL), pthread_cond_init(&st->qcond, NULL);
  st->index = nstores, stores[nstores++] = st;
  st->layers = l->layers, st->scored_layers = l->scored_layers,
  st->experts = l->experts, st->chosen = l->chosen;
  st->stage_experts = l->stage_experts, st->parts = l->parts;
  st->span = l->span, st->spans = l->spans;
  st->extra = l->transform ? l->extra : 0, st->transform = l->transform,
  st->context = l->context;
  st->where = xmalloc(WHERE_BYTES);
  memset(st->where, 0, WHERE_BYTES);
  st->generation = xmalloc((size_t)l->layers * sizeof *st->generation);
  memset(st->generation, 0, (size_t)l->layers * sizeof *st->generation);
  st->ready_generation =
      xmalloc((size_t)l->layers * sizeof *st->ready_generation);
  memset(st->ready_generation, 0xff,
         (size_t)l->layers *
             sizeof *st->ready_generation);  // never found ready
  st->layer_holds = calloc((size_t)l->layers, sizeof *st->layer_holds);
  if (!st->layer_holds) die("out of memory");
  st->ready_unmaps = xmalloc((size_t)l->layers * sizeof *st->ready_unmaps);
  memset(st->ready_unmaps, 0, (size_t)l->layers * sizeof *st->ready_unmaps);
  st->escore = xmalloc(SCORE_BYTES), st->dscore = xmalloc(SCORE_BYTES),
  st->nscore = xmalloc(SCORE_BYTES);
  memset(st->escore, 0, SCORE_BYTES), memset(st->dscore, 0, SCORE_BYTES),
      memset(st->nscore, 0, SCORE_BYTES);
  st->near_weight = 1;
  st->layer_bytes = xmalloc((size_t)l->scored_layers * sizeof *st->layer_bytes);
  memset(st->layer_bytes, 0,
         (size_t)l->scored_layers * sizeof *st->layer_bytes);
  st->minimum_slots = 2 * l->experts;
  return st;
}

// The bytes a slot for an expert of the layer takes.
static uint64_t slot_need(const ExpertStore *st, int layer) {
  return st->sized && layer < st->scored_layers && st->layer_bytes[layer]
             ? st->layer_bytes[layer]
             : st->SLOT_SIZE;
}

// Whether a buffer of size bytes may hold an expert that takes need bytes: it
// is large enough and wastes at most a sixteenth of the expert's bytes.
static int slot_fits(uint64_t size, uint64_t need) {
  return size >= need && size - need <= need / 16;
}

// The cache's bytes and its mapped bytes in the unit of its limits: slots of
// the largest size when the layers differ, else slots.
static int used_units(const ExpertStore *st) {
  return st->sized ? (int)((st->bytes + st->SLOT_SIZE - 1) / st->SLOT_SIZE)
                   : st->nslots;
}

static int mapped_units(const ExpertStore *st) {
  return st->sized
             ? (int)((st->mapped_bytes + st->SLOT_SIZE - 1) / st->SLOT_SIZE)
             : st->nmapped;
}

// Whether the cache holds more than its limit, or maps more than it may.
static int over_cap(const ExpertStore *st) {
  return st->sized ? st->bytes > (uint64_t)st->cap * st->SLOT_SIZE
                   : st->nslots > st->cap;
}

static int over_mcap(const ExpertStore *st) {
  return st->sized ? st->mapped_bytes > (uint64_t)st->mcap * st->SLOT_SIZE
                   : st->nmapped > st->mcap;
}

// The most that the limits may be.
static int cap_ceiling(const ExpertStore *st) {
  return st->sized ? st->max_cap : st->max_slots;
}

ExpertStoreCounts *expert_store_counts(ExpertStore *st) { return &st->stats; }

static int slot_reading(ExpertStore *st, int i);

// The eviction order: every expert has a score, the sum of a weight for each
// request that selected it; the weight doubles every HALF_LIFE forward passes.
// A verification batch advances once, as does the end of a prompt. A decoded
// layer also credits its near misses, the experts its router ranked just
// below the ones it selected: each counts NEAR_CREDIT of a selection, with a
// weight that doubles every NEAR_HALF_LIFE steps, because such an expert is
// often selected within a few tokens and seldom later. A slot's priority is
// its expert's two sums, each over its current weight. The victim has the
// lowest priority, an empty slot first. A recency list, whose head is the most
// recently used, orders the slots for giving slots back.
static constexpr int HALF_LIFE = 50, NEAR_HALF_LIFE = 12;
static constexpr double NEAR_CREDIT = 0.5;

// The priority of expert key = layer * experts + expert, of a scored layer.
static double expert_priority(const ExpertStore *st, size_t key) {
  return st->escore[key] / st->weight +
         NEAR_CREDIT * st->nscore[key] / st->near_weight;
}

// Ages every history by one step. The two sums of a scored layer's expert
// decay at different rates, so its slot's priority is computed again; the
// stages' slots keep their raw score, rescaled with the history.
static void advance_score_clock(ExpertStore *st) {
  st->token++;
  if ((st->weight *= exp2(1.0 / HALF_LIFE)) > 0x1p500) {
    for (int k = 0; k < st->layers * st->experts; k++)
      st->escore[k] /= st->weight, st->dscore[k] /= st->weight;
    for (int i = st->max_slots; i < st->max_slots + st->npinned; i++)
      st->score[i] /= st->weight;
    st->weight = 1;
  }
  if ((st->near_weight *= exp2(1.0 / NEAR_HALF_LIFE)) > 0x1p500) {
    for (int k = 0; k < st->layers * st->experts; k++)
      st->nscore[k] /= st->near_weight;
    st->near_weight = 1;
  }
  for (int i = 0; i < st->max_slots; i++) {
    const Slot *s = &st->slots[i];
    if (isfinite(st->score[i]) && s->expert >= 0 && s->layer >= 0 &&
        s->layer < st->scored_layers)
      st->score[i] =
          expert_priority(st, (size_t)s->layer * st->experts + s->expert);
  }
}

void expert_store_credit_near_misses(ExpertStore *st, int layer,
                                     const int *experts, int n) {
  if (layer < 0 || layer >= st->scored_layers) return;
  for (int k = 0; k < n; k++) {
    size_t key = (size_t)layer * st->experts + experts[k];
    st->nscore[key] += st->near_weight;
    int i = WHERE(layer, experts[k]);
    if (i >= 0 && i < st->max_slots && isfinite(st->score[i]))
      st->score[i] = expert_priority(st, key);
  }
}

static void unlink_slot(ExpertStore *st, int i) {
  Slot *s = &st->slots[i];
  if (s->prev >= 0)
    st->slots[s->prev].next = s->next;
  else
    st->head = s->next;
  if (s->next >= 0)
    st->slots[s->next].prev = s->prev;
  else
    st->tail = s->prev;
}

static void push_back(ExpertStore *st, int i) {
  st->slots[i].next = -1;
  st->slots[i].prev = st->tail;
  if (st->tail >= 0) st->slots[st->tail].next = i;
  st->tail = i;
  if (st->head < 0) st->head = i;
}

static void push_front(ExpertStore *st, int i) {
  st->slots[i].prev = -1;
  st->slots[i].next = st->head;
  if (st->head >= 0) st->slots[st->head].prev = i;
  st->head = i;
  if (st->tail < 0) st->tail = i;
}

static void cancel(ExpertStore *st, int i), start_readers(ExpertStore *st),
    start(ExpertStore *st, int i, int urgent), finish(ExpertStore *st, int i),
    map_slot(ExpertStore *st, int i);

// Gives slot i's buffer back: its expert leaves, and the slot is free again.
static void release(ExpertStore *st, int i) {
  if (slot_held(st, &st->slots[i]))
    die("cannot release an expert that a pass holds");
  cancel(st, i);
  unlink_slot(st, i);
  if (st->slots[i].expert >= 0)
    place(st, st->slots[i].layer, st->slots[i].expert, -1);
  if (st->slots[i].mapped)
    st->nmapped--, st->mapped_bytes -= st->slots[i].b.size,
        st->slots[i].mapped = 0;
  st->bytes -= st->slots[i].b.size;
  gpu_free(&st->slots[i].b);
  st->score[i] = INFINITY;
  st->free_list[st->nfree++] = i;
  st->nslots--;
  st->stats.evictions++;
}

void expert_store_set_backbone(ExpertStore *st, MemorySlot *backbone) {
  st->cache_backbone = backbone;
}

void expert_store_set_request_running(ExpertStore *st, int running) {
  st->cache_busy = !!running;
}

void expert_store_set_pass_rows(ExpertStore *st, int rows) {
  int experts =
      rows * st->chosen < st->experts ? rows * st->chosen : st->experts;
  int slots = 2 * experts;
  // Passes that run together each need their slots: while others run, the
  // minimum only rises, and a pass that needs fewer does not take it away.
  if (__atomic_load_n(&st->other_passes, __ATOMIC_RELAXED) > 0 &&
      slots < st->minimum_slots)
    slots = st->minimum_slots;
  st->minimum_slots = slots;
  // The model reserves these slots' memory before such a pass, so limits that
  // were set for fewer rows rise to them.
  if (st->max_slots) {
    int fewest = st->minimum_slots < cap_ceiling(st) ? st->minimum_slots
                                                     : cap_ceiling(st);
    if (st->cap < fewest) st->cap = fewest;
    if (st->mcap < fewest) st->mcap = fewest;
  }
}

void expert_store_set_other_passes(ExpertStore *st, int passes) {
  __atomic_store_n(&st->other_passes, passes, __ATOMIC_RELAXED);
}

// While a request runs, the memory pool spares the slots that a stage owns,
// that the current token used, or that a prompt pass may still use. Memory
// that is needed also takes the last two kinds, once nothing else makes the
// room: their experts are read again. GPU work uses only slots that a pass
// holds.
static int pool_can_evict(void *context, int needed) {
  Slot *slot = context;
  ExpertStore *st = slot->store;
  if (!slot->b.obj || slot_held(st, slot)) return 0;
  if (!st->cache_busy) return 1;
  if (slot->pinned) return 0;
  if (needed) return 1;
  if (slot->used == st->token) return 0;
  return st->pass_layer < 0 || slot->layer < st->pass_layer - 1;
}

static void pool_evict(void *context) {
  Slot *slot = context;
  ExpertStore *st = slot->store;
  int index = (int)(slot - st->slots);
  if (slot->pinned) {
    cancel(st, index);
    if (slot->expert >= 0) place(st, slot->layer, slot->expert, -1);
    slot->expert = -1;
    gpu_free(&slot->b);
  } else
    release(st, index);
  st->stats.given_back++;
  gpu_commit_residency(st->gpu);
}

// A slot's new buffer, counted in the slot. It only borrows memory that is
// free: a prediction never evicts another model's weights.
static Buf pool_buffer(ExpertStore *st, int index, size_t bytes, int mapped) {
  Buf buffer = st->cache_backbone
                   ? gpu_try_alloc(st->gpu, &st->slots[index].memory, bytes,
                                   MEMORY_SPECULATIVE)
                   : gpu_alloc(st->gpu, NULL, bytes);
  if (buffer.obj && mapped)
    gpu_set_resident(&buffer, 1), gpu_commit_residency(st->gpu);
  return buffer;
}

// A slot's buffer that a demand cannot do without: required memory, for which
// the pool takes idle model memory back. Without an object when the pool
// cannot make the room.
static Buf pool_buffer_required(ExpertStore *st, int index, size_t bytes) {
  if (!st->cache_backbone) return gpu_alloc(st->gpu, NULL, bytes);
  return gpu_try_alloc(st->gpu, &st->slots[index].memory, bytes,
                       MEMORY_REQUIRED);
}

uint64_t expert_store_minimum_bytes(ExpertStore *st, int layers) {
  if (!st->layout_ready) {
    // A slot holds the layer's largest expert; an empty span, of an expert
    // with fewer spans, takes no room.
    uint64_t pin_size = 0;  // the stages' slots
    for (int l = 0; l < layers; l++)
      for (int e = 0; e < st->experts; e++) {
        uint64_t size = st->extra, pieces = 0;
        for (int k = 0; k < st->spans[l]; k++)
          if (SPAN(l, e, k).len)
            size += IO_REGION(SPAN(l, e, k).len),
                pieces +=
                (IO_REGION(SPAN(l, e, k).len) + IO_PIECE - 1) / IO_PIECE;
        if (pieces > 16)
          die("layer %d's experts take %llu reads of %llu MiB, more than a "
              "slot tracks",
              l, pieces, IO_PIECE >> 20);
        uint64_t *at = l < st->scored_layers ? &st->SLOT_SIZE : &pin_size;
        if (size > *at) *at = size;
        if (l < st->scored_layers && st->spans[l] && size > st->layer_bytes[l])
          st->layer_bytes[l] = size;
      }
    // The layers' own sizes are used when one of them is more than a
    // sixteenth below the largest. A store whose layers are closer keeps
    // slots of one size, which any expert fits, and counts slots: buffers of
    // nearly one size would be given back for one another without holding
    // more experts.
    for (int l = 0; l < layers && l < st->scored_layers; l++)
      if (st->layer_bytes[l] &&
          st->layer_bytes[l] < st->SLOT_SIZE - st->SLOT_SIZE / 16)
        st->sized = 1;
    st->npinned = layers > st->scored_layers
                      ? (layers - st->scored_layers) * st->stage_experts
                      : 0;
    st->pinned_bytes = st->npinned * pin_size;
    st->pinned_slot_size = pin_size;
    st->layout_ready = 1;
  }
  return st->minimum_slots * gpu_round_up_to_pages(st->SLOT_SIZE) +
         st->npinned * gpu_round_up_to_pages(st->pinned_slot_size);
}

void expert_store_start(ExpertStore *st, uint64_t budget, int layers) {
  expert_store_minimum_bytes(st, layers);
  memset(st->where, 0xff, WHERE_BYTES);
  uint64_t pin_size = st->pinned_slot_size;
  SystemMemoryCounters counters = system_memory_read_counters();
  uint64_t free = counters.free, avail = counters.available,
           gpu_limit = gpu_recommended_working_set(st->gpu), limit = gpu_limit;
  if (!reserve) expert_store_set_reserve(0);
  int fixed = budget > 0;
  if (!budget) budget = avail > reserve ? avail - reserve : 0;
  if (st->cache_backbone &&
      memory_usage(st->cache_backbone->account->memory).limit < limit)
    limit = memory_usage(st->cache_backbone->account->memory).limit;
  if (budget > limit) budget = limit;
  budget = budget > st->pinned_bytes ? budget - st->pinned_bytes : 0;
  // The cache may grow up to max_slots when memory becomes free; a given
  // budget is a fixed bound.
  st->max_slots = (int)((fixed ? budget : limit) / st->SLOT_SIZE);
  if (st->max_slots > st->scored_layers * st->experts)
    st->max_slots = st->scored_layers * st->experts;
  if (st->max_slots < st->minimum_slots) st->max_slots = st->minimum_slots;
  st->max_cap = st->max_slots;
  if (st->sized) {  // the same bytes hold more slots when they are the smallest
                    // layer's
    uint64_t smallest = st->SLOT_SIZE;
    for (int l = 0; l < layers && l < st->scored_layers; l++)
      if (st->layer_bytes[l] && st->layer_bytes[l] < smallest)
        smallest = st->layer_bytes[l];
    uint64_t most = (uint64_t)st->max_cap * st->SLOT_SIZE / smallest;
    st->max_slots = most < (uint64_t)st->scored_layers * st->experts
                        ? (int)most
                        : st->scored_layers * st->experts;
  }
  // A measured budget below the minimum still holds the minimum: the model
  // admitted its bytes before loading its weights, which took them from the
  // available memory.
  if (fixed && budget < st->minimum_slots * st->SLOT_SIZE)
    die("an expert cache of %.1f GiB is too small: a prompt needs %.1f GiB",
        budget / GiB, (double)st->minimum_slots * st->SLOT_SIZE / GiB);
  uint64_t within = avail > reserve + st->pinned_bytes
                        ? avail - reserve - st->pinned_bytes
                        : 0;
  st->cap = (int)((budget < within ? budget : within) / st->SLOT_SIZE);
  if (st->cap > cap_ceiling(st)) st->cap = cap_ceiling(st);
  if (st->cap < st->minimum_slots) st->cap = st->minimum_slots;
  // The mapped slots stay within Metal's recommendation, beside weights that
  // count as resident only once the GPU uses them. The driver still counts a
  // slot taken out of the residency sets for up to freed_memory_delay, so
  // gpu_memory_limit_margin_gib stays unused for such slots.
  int64_t room = gpu_room(st) - (int64_t)st->pinned_bytes -
                 (int64_t)system_memory_footprint() -
                 (int64_t)(cache_settings.gpu_memory_limit_margin_gib * GiB);
  st->mcap = room > st->minimum_slots * (int64_t)st->SLOT_SIZE
                 ? (int)(room / (int64_t)st->SLOT_SIZE)
                 : st->minimum_slots;
  if (st->mcap > cap_ceiling(st)) st->mcap = cap_ceiling(st);
  st->slots = xmalloc((st->max_slots + st->npinned) * sizeof *st->slots);
  for (int i = 0; i < st->max_slots + st->npinned; i++) st->slots[i].store = st;
  if (st->cache_backbone)
    for (int i = 0; i < st->max_slots + st->npinned; i++) {
      memory_slot_init(&st->slots[i].memory, st->cache_backbone->account,
                       MEMORY_MODEL, st->cache_backbone, "routed expert",
                       pool_evict, &st->slots[i]);
      st->slots[i].memory.can_evict = pool_can_evict;
      st->slots[i].memory.automatic = 1;
    }
  st->score = xmalloc((st->max_slots + st->npinned) * sizeof *st->score);
  for (int i = 0; i < st->max_slots + st->npinned; i++) st->score[i] = INFINITY;
  start_readers(st);
  st->free_list = xmalloc(st->max_slots * sizeof *st->free_list);
  for (int i = st->max_slots - 1; i >= 0; i--) st->free_list[st->nfree++] = i;
  for (int i = st->max_slots; i < st->max_slots + st->npinned; i++) {
    Slot *s = &st->slots[i];
    s->b = pool_buffer(st, i, pin_size, 1), s->written = 0;
    if (!s->b.obj) die("the memory for the stages' experts is unavailable");
    s->layer = st->scored_layers + (i - st->max_slots) / st->stage_experts;
    s->expert = (i - st->max_slots) % st->stage_experts;
    s->pinned = 1;
    place(st, s->layer, s->expert, i);
    start(st, i, 1);
  }
  for (int i = st->max_slots; i < st->max_slots + st->npinned; i++)
    finish(st, i);
  int64_t gpu_left = gpu_room(st);
  fprintf(stderr,
          "expert cache: slot ceiling %d (%.1f GiB) within the shared model "
          "budget, up to %d mapped for the GPU, %.1f "
          "GiB free and %.1f GiB of reclaimable file cache, %.1f GiB kept for "
          "other programs, GPU resident memory "
          "%.1f GiB",
          st->cap, st->cap * (double)st->SLOT_SIZE / GiB, st->mcap, free / GiB,
          (avail > free ? avail - free : 0) / GiB, reserve / GiB,
          system_memory_gpu_driver() / GiB);
  if (gpu_left < INT64_MAX)
    fprintf(stderr, " (%.1f GiB below the residency target)", gpu_left / GiB);
  fprintf(stderr, ", GPU working set %.1f GiB\n", gpu_limit / GiB);
  if (st->sized) {
    uint64_t smallest = st->SLOT_SIZE;
    for (int l = 0; l < layers && l < st->scored_layers; l++)
      if (st->layer_bytes[l] && st->layer_bytes[l] < smallest)
        smallest = st->layer_bytes[l];
    fprintf(stderr,
            "expert slots have their layer's size, %.2f to %.2f MiB: the slot "
            "ceiling counts the largest, "
            "and up to %d slots fit\n",
            smallest / 1048576.0, st->SLOT_SIZE / 1048576.0, st->max_slots);
  }
  if (st->npinned)
    fprintf(stderr, "the stages' %d experts: %.1f GiB, resident\n", st->npinned,
            st->pinned_bytes / GiB);
}

const Buf *expert_store_get_buffer(ExpertStore *st, int layer, int expert,
                                   uint32_t at[EXPERT_STORE_PARTS],
                                   int *fresh) {
  int i = WHERE(layer, expert);
  if (i < 0) return 0;
  if (!st->slots[i].pinned) map_slot(st, i);
  *fresh = st->slots[i].fresh;
  memcpy(at, st->slots[i].at, (size_t)st->parts * sizeof *at);
  return &st->slots[i].b;
}

// Read-aheads: their pieces wait in a queue for READERS threads, which start a
// piece only while no demand piece is queued or running, so a demand read
// shares the SSD with at most READERS pieces. A demand for an expert that is
// still being read ahead claims the pieces that no reader has started. The
// readers and the demand reads run above the utility QoS class, whose disk
// reads macOS throttles.

// Marks a piece of slot s finished: read when a reader wrote it, dropped
// otherwise. After the last piece of a complete load, the layout's transform
// runs before the load's group completes, so the expert counts as read only
// after it; the transform's bytes end where the slot's own buffer ends.
static void piece_finished(ExpertStore *st, Slot *s, int read) {
  if (read) atomic_fetch_add_explicit(&s->read, 1, memory_order_release);
  int last_piece =
      atomic_fetch_sub_explicit(&s->unfinished, 1, memory_order_acq_rel) == 1;
  if (last_piece) {
    if (s->background_load)
      atomic_fetch_sub_explicit(&st->prefetching, 1, memory_order_relaxed);
    if (st->transform &&
        atomic_load_explicit(&s->read, memory_order_acquire) == s->npiece)
      st->transform(st->context, s->layer, s->expert, s->b.ptr, s->at,
                    s->pinned ? st->SLOT_SIZE : s->b.size);
  }
  threads_group_leave(s->loading);
}

// Queues a read-ahead piece of the layer, under qlock and with room in the
// queue, after the queued pieces of its layer and the layers before it. The
// readers take the lowest layer's pieces first, because a pass runs its layers
// in order and the nearest layer needs its experts soonest. Pieces of one
// layer keep the order they were queued in.
static void queue_push(ExpertStore *st, uint32_t entry, uint16_t layer) {
  uint32_t at = st->qtail++;
  for (; at != st->qhead && st->qlayer[(at - 1) % QUEUE] > layer; at--)
    st->queue[at % QUEUE] = st->queue[(at - 1) % QUEUE],
                   st->qlayer[at % QUEUE] = st->qlayer[(at - 1) % QUEUE];
  st->queue[at % QUEUE] = entry, st->qlayer[at % QUEUE] = layer;
}

// The SSD's time for expert reads: the seconds with at least one read in
// flight, and the demand and background reads' summed seconds.
static pthread_mutex_t ssd_lock = PTHREAD_MUTEX_INITIALIZER;
static int ssd_reads;
static double ssd_since, ssd_busy, ssd_read[2];  // ssd_read: demand, background

static double ssd_begin(void) {
  double t = now();
  pthread_mutex_lock(&ssd_lock);
  if (!ssd_reads++) ssd_since = t;
  pthread_mutex_unlock(&ssd_lock);
  return t;
}

static void ssd_end(int background, double began) {
  double t = now();
  pthread_mutex_lock(&ssd_lock);
  ssd_read[background] += t - began;
  if (!--ssd_reads) ssd_busy += t - ssd_since;
  pthread_mutex_unlock(&ssd_lock);
}

void expert_store_read_times(double *busy, double *demand, double *background) {
  pthread_mutex_lock(&ssd_lock);
  *busy = ssd_busy + (ssd_reads ? now() - ssd_since : 0);
  *demand = ssd_read[0], *background = ssd_read[1];
  pthread_mutex_unlock(&ssd_lock);
}

static void *reader(void *arg) {
  ExpertStore *st = arg;
  for (;;) {
    pthread_mutex_lock(&st->qlock);
    // Demand reads go first, as long as they are this pass's own: while other
    // passes run, read-aheads go on beside another pass's demand reads, which
    // would otherwise hold them back almost all the time.
    while (st->qhead == st->qtail ||
           (st->demand &&
            __atomic_load_n(&st->other_passes, __ATOMIC_RELAXED) <= 0))
      pthread_cond_wait(&st->qcond, &st->qlock);
    if (st->qtail - st->qhead == QUEUE)
      pthread_cond_broadcast(&st->qcond);  // room for start
    uint32_t q = st->queue[st->qhead++ % QUEUE];
    // A cancellation examines taken under this lock, so the reader takes the
    // piece before it leaves the queue's view.
    int owned = !atomic_exchange(&st->slots[q >> 4].taken[q & 15], 1);
    pthread_mutex_unlock(&st->qlock);
    Slot *s = &st->slots[q >> 4];
    if (owned) {
      double reading = ssd_begin();
      io_read_piece(&s->piece[q & 15], 0);
      ssd_end(1, reading);
      __atomic_fetch_add(&st->stats.bytes, s->piece[q & 15].len,
                         __ATOMIC_RELAXED);
      piece_finished(st, s, 1);
    }
  }
}

static void start_readers(ExpertStore *st) {
  for (int k = 0; k < READERS; k++)
    threads_spawn(THREADS_INITIATED, reader, st);
}

// A demand read's piece: its store's index in the high 32 bits, then its
// slot << 4 | piece.
static void read_demand(void *ctx) {
  uint64_t entry = (uintptr_t)ctx;
  ExpertStore *st = stores[entry >> 32];
  uint32_t q = (uint32_t)entry;
  Slot *s = &st->slots[q >> 4];
  double reading = ssd_begin();
  io_read_piece(&s->piece[q & 15], 0);
  ssd_end(0, reading);
  __atomic_fetch_add(&st->stats.bytes, s->piece[q & 15].len, __ATOMIC_RELAXED);
  piece_finished(st, s, 1);
  if (--st->demand == 0)
    pthread_mutex_lock(&st->qlock), pthread_cond_broadcast(&st->qcond),
        pthread_mutex_unlock(&st->qlock);
}

static void submit_demand(ExpertStore *st, uint32_t q) {
  uint64_t entry = (uint64_t)st->index << 32 | q;
  threads_async(NULL, THREADS_INTERACTIVE, (void *)(uintptr_t)entry,
                read_demand);
}

// Moves slot i's pieces that no reader has started to demand reads. demand
// counts queued and running pieces, so the readers hold back even before a
// demand read starts.
static void claim(ExpertStore *st, int i) {
  for (int k = 0; k < st->slots[i].npiece; k++)
    if (!atomic_exchange(&st->slots[i].taken[k], 1))
      st->demand++, submit_demand(st, (uint32_t)(i << 4 | k));
}

// The engine thread owns each load's group until the load is finished.
// Helpers may ask while the engine thread waits for them, and readers only
// complete the group's pieces.
static int slot_reading(ExpertStore *st, int i) {
  ThreadsGroup *g = st->slots[i].loading;
  return g && !threads_group_done(g);
}

static void finish(ExpertStore *st, int i) {
  if (!st->slots[i].loading) return;
  threads_group_wait(st->slots[i].loading);
  threads_group_free(st->slots[i].loading);
  st->slots[i].loading = 0;
  // A load that dropped a piece adds nothing: its pages may be partly written.
  if (atomic_load_explicit(&st->slots[i].read, memory_order_acquire) ==
          st->slots[i].npiece &&
      st->slots[i].extent > st->slots[i].written)
    st->slots[i].written = st->slots[i].extent;
}

void expert_store_wait_for_all_reads(void) {
  for (int k = 0; k < nstores; k++) {
    ExpertStore *st = stores[k];
    for (int i = 0; st->slots && i < st->max_slots + st->npinned; i++)
      finish(st, i);
  }
}

// Slot i is reused: its pieces that no reader has started are not read.
static void cancel(ExpertStore *st, int i) {
  if (!st->slots[i].loading) return;
  for (int k = 0; k < st->slots[i].npiece; k++)
    if (!atomic_exchange(&st->slots[i].taken[k], 1))
      piece_finished(st, &st->slots[i], 0);
  finish(st, i);
}

// The counted slot bytes that no finished load has written: a new buffer's
// pages are in use only once a read writes them, and a slot beyond its
// expert's bytes stays unwritten until a larger expert fills it. A load whose
// pieces were all read counts before the engine thread reaps it.
static uint64_t cache_unwritten(void) {
  uint64_t bytes = 0;
  for (int k = 0; k < nstores; k++) {
    ExpertStore *st = stores[k];
    for (int i = 0; st->slots && i < st->max_slots + st->npinned; i++) {
      const Slot *s = &st->slots[i];
      if (!s->b.obj) continue;
      uint64_t written = s->written;
      if (s->loading &&
          atomic_load_explicit(&s->read, memory_order_acquire) == s->npiece &&
          s->extent > written)
        written = s->extent;
      if (s->b.allocated > written) bytes += s->b.allocated - written;
    }
  }
  return bytes;
}

void expert_store_set_prompt_layer(ExpertStore *st, int layer) {
  st->pass_layer = layer;
}

void expert_store_write_pass_variables(
    ExpertStore *st, PassVariable out[EXPERT_STORE_PASS_VARIABLES]) {
  out[0] = (PassVariable){&st->pass_layer, sizeof st->pass_layer};
  out[1] = (PassVariable){&st->serving, sizeof st->serving};
}

// A slot for an expert: a free one, or a victim's. While decoding, the victim
// has the lowest priority, so the experts used most stay; the experts that
// the layer being served selected this token, and read-aheads whose layer has
// not been served yet, are not victims. In a prompt pass the victim is of a
// layer the pass is done with, else of a layer after the next, and is the one
// whose expert decoding used least: the experts the prompt alone needed go
// first, whatever their layer, and decoding after the prompt finds the
// experts it used before. Among equals, as in a cache that has not decoded
// yet, the least recently used goes, so that the experts of the prompt's last
// layers and rows, which decoding needs first, stay longest. A read-ahead
// that finds no slot is skipped rather than evicting experts in use.

// What a prompt pass evicts by: how much decoding used slot i's expert; an
// empty slot first.
static double decoding_priority(ExpertStore *st, int i) {
  if (!isfinite(st->score[i]) || st->slots[i].expert < 0 ||
      st->slots[i].layer < 0 || st->slots[i].layer >= st->layers)
    return st->score[i];
  return st->dscore[st->slots[i].layer * st->experts + st->slots[i].expert];
}

static int victim(ExpertStore *st) {
  int best = -1;
  double lowest = INFINITY;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched &&
          !slot_held(st, &st->slots[j]) &&
          !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving &&
            st->slots[j].used == st->token)) {
        double priority = st->score[j];
        if (priority < lowest) best = j, lowest = priority;
      }
    return best;
  }
  // From the least recently used on, so that the oldest of equals goes.
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
    }
  return best;
}

// The victim when the layers' experts differ in size, for an expert that
// takes need bytes. It is the plain victim if that slot's buffer fits the
// expert. Otherwise a slot whose buffer fits is reused if it is nearly as low:
// while decoding, the lowest-priority fitting slot among the FIT_RANK lowest
// priorities, which are the slots the next misses would take anyway; in a
// prompt pass, the least recently used slot of the same priority whose buffer
// is large enough, since a pass's buffers are of the largest size. When there
// is none, *convert is set and the plain victim is returned: the caller gives
// buffers back until one of the wanted size fits in the cache's bytes.
static constexpr int FIT_RANK = 64;

static int victim_sized(ExpertStore *st, uint64_t need, int *convert) {
  int best = -1, fit = -1;
  double lowest = INFINITY, fit_lowest = INFINITY;
  *convert = 0;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched &&
          !slot_held(st, &st->slots[j]) &&
          !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving &&
            st->slots[j].used == st->token)) {
        double priority = st->score[j];
        if (priority < lowest) best = j, lowest = priority;
        if (priority < fit_lowest && slot_fits(st->slots[j].b.size, need))
          fit = j, fit_lowest = priority;
      }
    if (best < 0 || fit == best) return best;
    if (fit >= 0) {
      int below = 0;
      for (int j = 0; j < st->max_slots && below < FIT_RANK; j++)
        below +=
            st->score[j] < INFINITY && !st->slots[j].prefetched &&
            !slot_held(st, &st->slots[j]) &&
            !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving &&
              st->slots[j].used == st->token) &&
            st->score[j] < fit_lowest;
      if (below < FIT_RANK) return fit;
    }
    *convert = 1;
    return best;
  }
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
      if (st->slots[j].b.size >= need && (fit < 0 || priority < fit_lowest))
        fit = j, fit_lowest = priority;
    }
  if (best < 0 || st->slots[best].b.size >= need) return best;
  if (fit >= 0 && fit_lowest == lowest) return fit;
  *convert = 1;
  return best;
}

// Maps slot i for the GPU; until the residency sets are committed, between
// tokens, the GPU reaches it through gpu_declare_read.
static void map_slot(ExpertStore *st, int i) {
  if (st->slots[i].mapped) return;
  gpu_set_resident(&st->slots[i].b, 1);
  st->slots[i].mapped = 1, st->slots[i].fresh = 1, st->any_fresh = 1;
  st->nmapped++, st->mapped_bytes += st->slots[i].b.size;
  if (st->slots[i].mapped_once) st->stats.remaps++;
  st->slots[i].mapped_once = 1;
}

// take's result when, during its wait for a slot, another pass made the
// expert resident.
static constexpr int TAKE_RESIDENT = -2;

static int slot_unheld(ExpertStore *st) {
  for (int j = 0; j < st->max_slots; j++)
    if (st->score[j] < INFINITY && !slot_held(st, &st->slots[j])) return j;
  return -1;
}

// Whether a slot that nothing holds has come free in the store.
static int slot_released(void *store) { return slot_unheld(store) >= 0; }

static void slot_released_block(void *store) {
  while (!slot_released(store)) sched_yield();
}

// The slot whose buffer a conversion gives back next, -1 if none. Nothing
// holds it, and it is no read-ahead that is still protected. While decoding,
// it is the lowest priority among the slots not used or read ahead in this
// token, which the memory pool does not free either. In a prompt pass it
// follows the victim's order, the layers the pass is done with before the
// layers after the next, but leaves out the layer just before the pass's
// own, whose buffers the pool does not free either.
static int victim_freeable(ExpertStore *st) {
  int best = -1;
  double lowest = INFINITY;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched &&
          !slot_held(st, &st->slots[j]) && st->slots[j].used != st->token) {
        double priority = st->score[j];
        if (priority < lowest) best = j, lowest = priority;
      }
    return best;
  }
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer - 1) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
    }
  return best;
}

// The slot that a demand reuses when no buffer can be given back or allocated
// for it, -1 if none: its buffer is at least need bytes, whatever it wastes,
// and nothing holds it. Among such slots the victim's preferences decide.
// First come the slots that victims come from: while decoding, those that are
// no protected read-ahead and that the layer being served did not select this
// token; in a prompt pass, those of the layers the pass is done with, then
// those of the layers after the next. The other slots come last. Within each
// group the lowest priority goes, and among equals the least recently used.
static int victim_large_enough(ExpertStore *st, uint64_t need) {
  int best[3] = {-1, -1, -1};
  double lowest[3] = {INFINITY, INFINITY, INFINITY};
  for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
    const Slot *s = &st->slots[j];
    if (st->score[j] == INFINITY || slot_held(st, s) || s->b.size < need)
      continue;
    int group;
    double priority;
    if (st->pass_layer < 0) {
      group = s->prefetched || (s->expert >= 0 && s->layer == st->serving &&
                                s->used == st->token)
                  ? 2
                  : 0;
      priority = st->score[j];
    } else {
      group = s->layer < st->pass_layer       ? 0
              : s->layer > st->pass_layer + 1 ? 1
                                              : 2;
      priority = decoding_priority(st, j);
    }
    if (best[group] < 0 || priority < lowest[group])
      best[group] = j, lowest[group] = priority;
  }
  return best[0] >= 0 ? best[0] : best[1] >= 0 ? best[1] : best[2];
}

// A conversion: no slot with a fitting buffer is low enough, so the cache
// gives buffers back, the lowest priorities first, until one of need bytes
// fits in its bytes and the memory pool admits it. Returns the free slot that
// the new buffer *buffer is for, or -1. A read-ahead gives at most two buffers
// back and a demand four: two of the smallest make room for one of the
// largest, and the pool admits a buffer only while a second one would fit
// above the reserve. A pool that still refuses is short of memory, and giving
// more back would empty the cache into it; the demand then reuses a buffer
// that is large enough.
static constexpr int CONVERT_OPTIONAL = 2, CONVERT_DEMAND = 4;

static int convert(ExpertStore *st, uint64_t need, int optional, Buf *buffer) {
  for (int given = 0;; given++) {
    if (st->nfree && st->bytes + need <= (uint64_t)st->cap * st->SLOT_SIZE) {
      int j = st->free_list[st->nfree - 1];
      *buffer = pool_buffer(st, j, need, 0);
      if (buffer->obj) {
        st->conversions += given > 0;
        return j;
      }
    }
    if (given >= (optional ? CONVERT_OPTIONAL : CONVERT_DEMAND)) return -1;
    int i = victim_freeable(st);
    if (i < 0 || (optional && slot_reading(st, i))) return -1;
    release(st, i);
  }
}

// A slot for expert e of the layer; -1 for an optional request when none is
// free; or TAKE_RESIDENT when, during the wait for a slot that passes hold,
// another pass made the expert resident.
static int take(ExpertStore *st, int layer, int e, int optional) {
  int i = -1;
  // What the expert takes, and the size of a new buffer for it: the expert's
  // while decoding. A prompt pass goes through every layer and needs each
  // one's experts only until the next, so its new buffers have the largest
  // size and serve all its layers; decoding gives them back for buffers of
  // its experts' sizes as it replaces their experts.
  uint64_t need = slot_need(st, layer),
           size = st->pass_layer >= 0 ? st->SLOT_SIZE : need;
  int reuse;
  Buf buffer;
  for (;;) {  // once; again after a demand that found no buffer at all gave a
              // slot back
    i = -1, buffer = (Buf){0};
    reuse = (st->sized ? st->bytes + size > (uint64_t)st->cap * st->SLOT_SIZE
                       : st->nslots >= st->cap) ||
            !st->nfree;
    if (!reuse) {
      i = st->free_list[st->nfree - 1];
      buffer = pool_buffer(st, i, size, 0);
      if (!buffer.obj) reuse = 1;
    }
    if (reuse) {
      int converting = 0;
      i = st->sized ? victim_sized(st, need, &converting) : victim(st);
      if (i < 0) {
        if (optional) return -1;
        // The preferences leave only slots that another pass may read soon: a
        // demand may take any slot that nothing holds. With every slot held, a
        // pass waits for the other passes to release one after their GPU work.
        // If they all wait too, the run ends the program.
        i = slot_unheld(st);
        if (i < 0 && pass_current() &&
            __atomic_load_n(&st->other_passes, __ATOMIC_RELAXED) > 0) {
          pass_wait(&(PassWait){slot_released, slot_released_block, st,
                                PASS_WAIT_OTHER, NULL});
          if (WHERE(layer, e) >= 0) return TAKE_RESIDENT;
          i = slot_unheld(st);
        }
        if (i < 0)
          die("the expert cache (%d slots) cannot hold the experts of two "
              "layers",
              st->nslots);
        converting = st->sized && st->slots[i].b.size < need;
      }
      // Routing can demote a read-ahead while its read is still running: an
      // optional request neither waits for it nor evicts a higher score.
      if (optional && slot_reading(st, i)) return -1;
      if (converting) {
        // The buffer cannot hold the expert, or wastes too much of the cache
        // on it: buffers go back to make room for one of the wanted size.
        i = convert(st, size, optional, &buffer);
        if (i >= 0)
          reuse = 0;
        else if (optional)
          return -1;
        else if ((i = victim_large_enough(st, need)) < 0) {
          // No buffer could be converted and none is large enough: the demand
          // allocates one as required memory, for which the pool makes room.
          // The pool may take other slots of this cache back for it, and they
          // join the free list: the slot that the buffer is counted in leaves
          // the list meanwhile and returns to its top.
          if (st->nfree) {
            i = st->free_list[--st->nfree];
            buffer = pool_buffer_required(st, i, size);
            st->free_list[st->nfree++] = i;
          }
          if (!buffer.obj) {
            // The pool cannot make the room: passes hold what it would take
            // back. The lowest priority that nothing holds goes back, and the
            // demand starts over. It then finds room in the cache's own
            // bytes; or, with nothing left that is not held, it waits above
            // for another pass to release a slot, as with slots of one size.
            if ((i = victim_large_enough(st, 0)) >= 0) release(st, i);
            continue;
          }
          reuse = 0;
        }
      }
    }
    break;
  }
  if (reuse) {
    cancel(st, i);
    unlink_slot(st, i);
    if (st->slots[i].expert >= 0)
      place(st, st->slots[i].layer, st->slots[i].expert, -1);
    st->stats.evictions++;
    if (st->slots[i].mapped) {
      gpu_set_resident(&st->slots[i].b, 0);
      st->slots[i].mapped = st->slots[i].fresh = 0;
      st->nmapped--, st->mapped_bytes -= st->slots[i].b.size;
    }
  } else {
    i = st->free_list[--st->nfree];
    st->slots[i].b = buffer, st->slots[i].written = 0;
    // Reads fill the buffer, and the GPU work that uses it maps it, so a
    // prediction that is never selected takes no GPU residency.
    st->slots[i].mapped = st->slots[i].fresh = st->slots[i].mapped_once = 0;
    st->nslots++, st->bytes += buffer.size;
  }
  if (st->cache_backbone) memory_touch(&st->slots[i].memory);
  st->slots[i].prefetched = 0;
  st->slots[i].layer = layer, st->slots[i].expert = e;
  place(st, layer, e, i);
  size_t key = (size_t)layer * st->experts + e;
  st->score[i] =
      layer < st->scored_layers ? expert_priority(st, key) : st->escore[key];
  push_front(st, i);
  return i;
}

// Starts reading slot i's expert: a read-ahead goes to the readers' queue, and
// a demand (urgent) is claimed at once.
static void start(ExpertStore *st, int i, int urgent) {
  Slot *s = &st->slots[i];
  int n = 0;
  uint64_t region = 0;
  for (int k = 0; k < st->spans[s->layer]; k++) {
    const FileSpan *p = &SPAN(s->layer, s->expert, k);
    if (!p->len) continue;  // an expert with fewer spans than the layer's most
    n += io_split(s->piece + n, p->fd, p->off, p->len, s->b.ptr + region,
                  &s->at[k]);
    s->at[k] += region;
    region += IO_REGION(p->len);
  }
  s->npiece = n;
  s->extent = st->extra;  // the transform's bytes at the slot's end
  for (int k = 0; k < n; k++) s->extent += s->piece[k].len;
  atomic_store_explicit(&s->read, 0, memory_order_relaxed);
  s->background_load = !urgent;
  atomic_store_explicit(&s->unfinished, n, memory_order_relaxed);
  s->loading = threads_group_new();
  // A stale queue entry may claim a reused slot as soon as taken is reset, so
  // the group's pieces and the read-ahead's count go first.
  for (int k = 0; k < n; k++) threads_group_enter(s->loading);
  if (!urgent)
    atomic_fetch_add_explicit(&st->prefetching, 1, memory_order_relaxed);
  for (int k = 0; k < n; k++)
    atomic_store_explicit(&s->taken[k], 0, memory_order_release);
  if (urgent) {
    claim(st, i);
    return;
  }
  pthread_mutex_lock(&st->qlock);
  for (int k = 0; k < n; k++) {
    while (st->qtail - st->qhead == QUEUE)
      pthread_cond_wait(&st->qcond, &st->qlock);
    queue_push(st, (uint32_t)(i << 4 | k), (uint16_t)s->layer);
  }
  pthread_cond_broadcast(&st->qcond);
  pthread_mutex_unlock(&st->qlock);
}

// Slot i's expert e of the layer is in use this token: eviction keeps it
// longest, and its score grows.
static void mark_used(ExpertStore *st, int layer, int e, int i) {
  st->slots[i].prefetched = 0;
  st->slots[i].used = st->token;
  if (st->cache_backbone) memory_touch(&st->slots[i].memory);
  size_t key = (size_t)layer * st->experts + e;
  st->escore[key] += st->weight;
  st->score[i] = i < st->max_slots && layer < st->scored_layers
                     ? expert_priority(st, key)
                     : st->escore[key];
  if (st->pass_layer < 0)
    st->dscore[key] += st->weight, st->stats.decode_uses++;
}

int expert_store_start_reads(ExpertStore *st, int layer, const int *experts,
                             int n) {
  int reading = 0;
  st->serving = layer;
  for (int k = 0; k < n; k++)  // the resident ones first, so that the others'
                               // slots are not theirs
    if (WHERE(layer, experts[k]) >= 0)
      st->slots[WHERE(layer, experts[k])].used = st->token;
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = WHERE(layer, e), pending, fresh = 0;
    if (i < 0) {
      i = take(st, layer, e, 0);
      if (i == TAKE_RESIDENT)
        i = WHERE(layer,
                  e);  // another pass read it while this one waited for a slot
      else
        fresh = 1;
    }
    if (!fresh) {
      if (!st->slots[i].pinned) unlink_slot(st, i), push_front(st, i);
      pending = slot_reading(st, i);
      // A read-ahead can outlive its predicting pass and lose its protection,
      // so any later demand claims its queued pieces.
      if (pending) claim(st, i);
      if (st->slots[i].prefetched) {
        st->stats.prefetch_hits++;
        if (pending) st->stats.late++;
      } else {
        st->stats.hits++;
      }
    } else {
      st->stats.misses++;
      start(st, i, 1);
      pending = slot_reading(st, i);
    }
    mark_used(st, layer, e, i);
    reading += pending;
  }
  return reading;
}

void expert_store_count_use(ExpertStore *st, int layer, const int *experts,
                            int n) {
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = WHERE(layer, e);
    if (i < 0) continue;
    if (!st->slots[i].pinned) unlink_slot(st, i), push_front(st, i);
    if (st->slots[i].prefetched)
      st->stats.prefetch_hits++;
    else
      st->stats.hits++;
    mark_used(st, layer, e, i);
  }
}

int expert_store_has_slot(ExpertStore *st, int layer, int e) {
  return WHERE(layer, e) >= 0;
}

int expert_store_hold(ExpertStore *st, int layer, const int *experts, int n) {
  if (layer < 0 || layer >= st->scored_layers || n < 1 || n > st->experts)
    die("invalid expert set to hold");
  unsigned char resident[n];
  // The resident experts are held before the missing ones get slots, and each
  // missing one is held as soon as it has a slot, so that the next one's slot
  // is not taken from it. Passes that hold the same expert share its slot and
  // its read.
  for (int k = 0; k < n; k++) {
    int e = experts[k];
    if (e < 0 || e >= st->experts) die("invalid expert to hold");
    int i = WHERE(layer, e);
    resident[k] = i >= 0;
    if (i >= 0) st->slots[i].holds++;
  }
  int pending = 0;
  for (int k = 0; k < n; k++) {
    pending += expert_store_start_reads(st, layer, &experts[k], 1);
    if (!resident[k]) st->slots[WHERE(layer, experts[k])].holds++;
  }
  st->held_references += n;
  return pending;
}

void expert_store_hold_layer(ExpertStore *st, int layer) {
  st->layer_holds[layer]++;
}

void expert_store_release_layer(ExpertStore *st, int layer) {
  if (!st->layer_holds[layer]) die("a layer's hold was released twice");
  st->layer_holds[layer]--;
}

void expert_store_release(ExpertStore *st, int layer, const int *experts,
                          int n) {
  for (int k = 0; k < n; k++) {
    int i = WHERE(layer, experts[k]);
    if (i < 0 || !st->slots[i].holds) die("a held expert lost its slot");
    st->slots[i].holds--;
  }
  if (st->held_references < (unsigned)n) die("more experts released than held");
  st->held_references -= n;
}

int expert_store_is_ready(ExpertStore *st, int layer, int e) {
  int i = WHERE(layer, e);
  return i >= 0 && !slot_reading(st, i);
}

// A layer found ready stays ready until one of its experts moves or a slot is
// unmapped: a slot is fresh or being read only after either.
int expert_store_layer_is_reachable(ExpertStore *st, int layer,
                                    uint64_t *generation) {
  *generation = st->generation[layer];
  if (st->ready_generation[layer] == *generation &&
      st->ready_unmaps[layer] == st->unmaps)
    return 1;
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i < 0 || st->slots[i].loading || !st->slots[i].mapped ||
        st->slots[i].fresh)
      return 0;
  }
  st->ready_generation[layer] = *generation,
  st->ready_unmaps[layer] = st->unmaps;
  return 1;
}

ExpertPlace expert_store_locate(ExpertStore *st, int layer, int e) {
  int i = WHERE(layer, e);
  return i < 0                      ? EXPERT_ON_SSD
         : !st->slots[i].prefetched ? EXPERT_IN_MEMORY
         : slot_reading(st, i)      ? EXPERT_BEING_READ_AHEAD
                                    : EXPERT_READ_AHEAD;
}

typedef struct {
  ExpertStore *st;
  int layer;
  const int *experts;
  int n;
} ReadsWait;

static int reads_ready(void *context) {
  ReadsWait *w = context;
  ExpertStore *st = w->st;
  for (int k = 0; k < w->n; k++)
    if (slot_reading(st, WHERE(w->layer, w->experts[k]))) return 0;
  return 1;
}

static void reads_block(void *context) {
  ReadsWait *w = context;
  ExpertStore *st = w->st;
  for (int k = 0; k < w->n; k++) {
    ThreadsGroup *g = st->slots[WHERE(w->layer, w->experts[k])].loading;
    if (g) threads_group_wait(g);
  }
}

// Has each expert's read signal the event when it is in.
static void reads_notify(void *context, ThreadsEvent *event) {
  ReadsWait *w = context;
  ExpertStore *st = w->st;
  for (int k = 0; k < w->n; k++) {
    ThreadsGroup *g = st->slots[WHERE(w->layer, w->experts[k])].loading;
    if (g) threads_group_notify(g, event);
  }
}

void expert_store_wait_for_reads(ExpertStore *st, int layer, const int *experts,
                                 int n) {
  double t = now();
  if (pass_current()) {
    ReadsWait wait = {st, layer, experts, n};
    pass_wait(&(PassWait){reads_ready, reads_block, &wait, PASS_WAIT_READ,
                          reads_notify});
  }
  for (int k = 0; k < n; k++) finish(st, WHERE(layer, experts[k]));
  st->stats.read_s += now() - t;
}

void expert_store_read_and_wait(ExpertStore *st, int layer, const int *experts,
                                int n) {
  expert_store_start_reads(st, layer, experts, n);
  expert_store_wait_for_reads(st, layer, experts, n);
}

void expert_store_read_ahead(ExpertStore *st, int layer, const int *experts,
                             int n, int inflight) {
  // The limit is per pass: passes running together each read ahead as many.
  int others = __atomic_load_n(&st->other_passes, __ATOMIC_RELAXED);
  int limit = inflight * (1 + (others > 0 ? others : 0));
  for (int k = 0; k < n && st->prefetching < limit; k++) {
    int e = experts[k];
    if (WHERE(layer, e) >= 0) continue;
    int i = take(st, layer, e, 1);
    if (i < 0) return;
    st->slots[i].prefetched = 1;
    st->slots[i].issuer = pass_current();
    st->slots[i].used = st->token;
    if (st->cache_backbone) memory_touch(&st->slots[i].memory);
    start(st, i, 0);
    st->stats.prefetches++;
  }
}

// The engine thread cancels only pieces that no reader owns, and checks and
// claims them under the readers' lock, so a cancellation never waits for a
// read. Only the engine thread changes a slot's other fields; readers change
// taken, so the first scan only filters, and ownership is checked again under
// the lock.
static int discard_unstarted(ExpertStore *st, int layer, const int *keep,
                             int n) {
  if (!atomic_load(&st->prefetching)) return 0;
  int retired[st->experts], nc = 0;
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i < 0 || !st->slots[i].prefetched || !st->slots[i].loading ||
        !st->slots[i].npiece)
      continue;
    if (st->slots[i].issuer != pass_current())
      continue;  // another pass's prediction, for its own layer
    int k = 0;
    while (k < n && keep[k] != e) k++;
    if (k < n) continue;
    if (slot_held(st, &st->slots[i])) continue;
    int started = 0;
    for (int p = 0; p < st->slots[i].npiece; p++)
      started |= atomic_load(&st->slots[i].taken[p]);
    if (!started) retired[nc++] = i;
  }
  if (!nc) return 0;
  unsigned char discard[st->experts];
  memset(discard, 0, sizeof discard);
  int nr = 0;
  pthread_mutex_lock(&st->qlock);
  for (int k = 0; k < nc; k++) {
    int i = retired[k], started = 0;
    for (int p = 0; p < st->slots[i].npiece; p++)
      started |= atomic_load(&st->slots[i].taken[p]);
    if (started) continue;
    // Readers take pieces under qlock too, and the engine thread is the only
    // one that claims demand reads, so this cancellation now owns every piece.
    for (int p = 0; p < st->slots[i].npiece; p++)
      atomic_store(&st->slots[i].taken[p], 1);
    discard[st->slots[i].expert] = 1, retired[nr++] = i;
  }
  if (nr) {
    uint32_t end = st->qtail, write = st->qhead;
    for (uint32_t read = st->qhead; read != end; read++) {
      uint32_t q = st->queue[read % QUEUE];
      uint16_t ql = st->qlayer[read % QUEUE];
      Slot *s = &st->slots[q >> 4];
      if (s->layer == layer && s->expert >= 0 && discard[s->expert]) continue;
      st->qlayer[write % QUEUE] = ql;
      st->queue[write++ % QUEUE] = q;
    }
    st->qtail = write;
    pthread_cond_broadcast(&st->qcond);
  }
  pthread_mutex_unlock(&st->qlock);
  // The queue no longer names the retired slots and no reader owns their
  // pieces, so their loads finish before the slots are reused.
  for (int k = 0; k < nr; k++) {
    int i = retired[k], expert = st->slots[i].expert;
    for (int p = 0; p < st->slots[i].npiece; p++)
      piece_finished(st, &st->slots[i], 0);
    finish(st, i);
    place(st, layer, expert, -1);
    st->slots[i].expert = -1, st->slots[i].prefetched = 0;
    st->slots[i].used =
        st->token - 1;  // an empty slot is not among this token's selections
    st->score[i] = -INFINITY;
    unlink_slot(st, i), push_back(st, i);
    st->stats.cancelled++;
  }
  return nr;
}

void expert_store_unprotect_read_aheads(ExpertStore *st, int layer) {
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i >= 0 && st->slots[i].issuer == pass_current())
      st->slots[i].prefetched = 0;
  }
}

// The cancelled read-aheads' slots hold no expert, and are taken first.
void expert_store_cancel_read_aheads(ExpertStore *st, int layer,
                                     const int *keep, int n) {
  discard_unstarted(st, layer, keep, n);
}

// Resizes the cache. It gives slots back, the least recently used of the
// layers below below first, on memory pressure or when the available memory
// falls below the reserve: a sixteenth of the cache under pressure or
// swapping, a quarter at critical pressure, and otherwise the slots that
// restore the reserve. Swap-outs shrink a cache above its minimum; a
// swap-out cannot take back memory from an empty cache. Growth uses a
// quarter of the headroom above expert_cache_growth_margin_gib after 32 calm
// calls, and for free_pages_only_after_swapout after a swap-out only free
// pages.
static void resize(ExpertStore *st, int below) {
  SystemMemoryCounters counters = system_memory_read_counters();
  uint64_t free = counters.free, avail = counters.available,
           slot = st->SLOT_SIZE,
           growth_margin =
               (uint64_t)(cache_settings.expert_cache_growth_margin_gib * GiB);
  double freed_memory_delay = cache_settings.times.freed_memory_delay,
         caution = cache_settings.free_pages_only_after_swapout;
  if (!st->init) st->swapouts = counters.swapouts, st->init = 1;
  int64_t room = gpu_room(st);
  // The driver still counts a slot taken out of the residency sets for up to
  // freed_memory_delay, so the mapped slots leave this much of the GPU's room
  // unused.
  int64_t gpu_margin =
      (int64_t)(cache_settings.gpu_memory_limit_margin_gib * GiB);
  uint64_t swap_delta =
      counters.swapouts > st->swapouts ? counters.swapouts - st->swapouts : 0;
  // What the cache holds and maps, in the unit of its limits, and the most
  // they may be.
  int used = used_units(st), mapped = mapped_units(st),
      ceiling = cap_ceiling(st);
  SystemMemoryPressure level = system_memory_pressure();
  int swapped = swap_delta && used > st->minimum_slots;
  int low = level >= SYSTEM_MEMORY_WARNING || swapped || avail < reserve;
  if (swapped) st->bad = now();
  st->swapouts = counters.swapouts;
  int old_cap = st->cap, old_mcap = st->mcap;
  uint64_t headroom = now() - st->bad > caution ? avail : free;
  if (low && (level >= SYSTEM_MEMORY_CRITICAL || swapped ||
              now() - gave > freed_memory_delay)) {
    uint64_t to = avail;
    int64_t give = level >= SYSTEM_MEMORY_CRITICAL             ? used / 4
                   : level >= SYSTEM_MEMORY_WARNING || swapped ? used / 16 + 1
                                                               : 0;
    if (to < reserve && (int64_t)((reserve - to) / slot) + 1 > give)
      give = (reserve - to) / slot + 1;
    st->cap = used - (int)give > st->minimum_slots ? used - (int)give
                                                   : st->minimum_slots;
    st->calm = 0, gave = now();
  } else if (!low && ++st->calm >= 32 && st->cap < ceiling &&
             headroom > reserve + growth_margin) {
    st->cap += (int)((headroom - reserve - growth_margin) / slot / 4) + 1;
    if (st->cap > ceiling) st->cap = ceiling;
    st->calm = 0;
  }
  // Past the GPU's limit, the cache maps exactly the excess fewer.
  if (room < 0 && now() - st->unmapped > freed_memory_delay) {
    st->mcap = mapped - (int)(-room / (int64_t)slot) - 1 > st->minimum_slots
                   ? mapped - (int)(-room / (int64_t)slot) - 1
                   : st->minimum_slots;
    st->calm_map = 0, st->unmapped = now();
  } else if (room >= 0 && ++st->calm_map >= 32 && st->mcap < ceiling &&
             room > gpu_margin) {
    st->mcap += (int)((room - gpu_margin) / (int64_t)slot / 4) + 1;
    if (st->mcap > ceiling) st->mcap = ceiling;
    st->calm_map = 0;
  }
  for (int i = st->tail; over_cap(st) && i >= 0;) {
    int prev = st->slots[i].prev;
    if (st->slots[i].layer < below && !slot_held(st, &st->slots[i]))
      release(st, i), st->stats.given_back++;
    i = prev;
  }
  // The mapped slots with the lowest scores that nothing of this pass still
  // needs leave the residency sets.
  while (over_mcap(st)) {
    int best = -1;
    for (int j = 0; j < st->max_slots; j++)
      if (st->slots[j].b.obj && st->slots[j].mapped &&
          st->slots[j].layer < below && !slot_held(st, &st->slots[j]) &&
          !st->slots[j].loading && !st->slots[j].prefetched &&
          (best < 0 || st->score[j] < st->score[best]))
        best = j;
    if (best < 0) break;
    gpu_set_resident(&st->slots[best].b, 0);
    st->slots[best].mapped = 0, st->slots[best].fresh = 0;
    st->nmapped--, st->mapped_bytes -= st->slots[best].b.size, st->unmaps++;
  }
  gpu_commit_residency(st->gpu);
  static int trace = -1;
  if (trace < 0) trace = getenv("EREMITE_CACHE_TRACE") != NULL;
  if (trace && (st->cap != old_cap || st->mcap != old_mcap))
    fprintf(
        stderr,
        "cache resize: limit %d -> %d, allocated %d, map limit %d -> %d, "
        "mapped %d, "
        "maximum %d, pressure %d, swap-outs +%llu, available %.2f GiB, free "
        "%.2f GiB, "
        "slots not yet written %.2f GiB, GPU room %.2f GiB, growth uses %s\n",
        old_cap, st->cap, st->nslots, old_mcap, st->mcap, st->nmapped,
        st->max_slots, level, (unsigned long long)swap_delta, avail / GiB,
        free / GiB, cache_unwritten() / GiB, room / GiB,
        now() - st->bad > caution ? "available memory" : "free pages");
}

// Between tokens no GPU work of this pass references a slot. In a prompt pass
// at a layer, the work may still use the slots of the layer before, so only
// those of earlier layers are released. Held slots stay, for the work of any
// pass that holds them.
void expert_store_resize_in_prompt(ExpertStore *st, int pass) {
  if (pass < 0) advance_score_clock(st);
  if (st->cache_backbone && pass < 0 &&
      now() - st->last_pool_growth >=
          cache_settings.times.available_memory_reading_interval &&
      memory_room(st->cache_backbone->account) < 4 * st->SLOT_SIZE) {
    MemorySlot growth;
    memory_slot_init(&growth, st->cache_backbone->account, MEMORY_MODEL, NULL,
                     "cache growth", NULL, NULL);
    MemorySlot *wanted[] = {&growth};
    uint64_t bytes = 4 * st->SLOT_SIZE;
    memory_admit(wanted, &bytes, 1, MEMORY_CACHE);
    memory_slot_remove(&growth);
    st->last_pool_growth = now();
  }
  resize(st, pass < 0 ? st->layers : pass - 1);
  if (pass < 0 && st->any_fresh) {  // the resizing committed the residency sets
    for (int i = st->head; i >= 0; i = st->slots[i].next)
      st->slots[i].fresh = 0;
    st->any_fresh = 0;
  }
}

void expert_store_age_and_resize(ExpertStore *st) {
  expert_store_resize_in_prompt(st, -1);
}

// While no request runs, no GPU work references a slot: the cache resizes as
// between tokens, so it gives slots back when other programs need the memory
// and leaves them as they are otherwise.
void expert_store_resize_when_idle(ExpertStore *st) { resize(st, st->layers); }

// With slots of the layers' own sizes, the limits are bytes: the counts below
// say how many slots fit in them, the allocated ones and as many of the
// largest size as the remaining bytes hold; a slot's bytes are the allocated
// slots' average.
static int units_as_slots(const ExpertStore *st, int units) {
  uint64_t limit = (uint64_t)units * st->SLOT_SIZE;
  return st->sized
             ? st->nslots + (int)(limit > st->bytes
                                      ? (limit - st->bytes) / st->SLOT_SIZE
                                      : 0)
             : units;
}

uint64_t expert_store_bytes(ExpertStore *st) {
  return (st->sized ? st->bytes : (uint64_t)st->nslots * st->SLOT_SIZE) +
         st->pinned_bytes;
}

int expert_store_slot_count(ExpertStore *st) { return st->nslots; }

int expert_store_mapped_count(ExpertStore *st) { return st->nmapped; }

int expert_store_slot_limit(ExpertStore *st) {
  return units_as_slots(st, st->cap);
}

int expert_store_slot_maximum(ExpertStore *st) {
  return units_as_slots(st, cap_ceiling(st));
}

uint64_t expert_store_slot_bytes(ExpertStore *st) {
  return st->sized && st->nslots ? st->bytes / st->nslots : st->SLOT_SIZE;
}

uint64_t expert_store_resized_buffer_count(ExpertStore *st) {
  return st->conversions;
}

uint64_t expert_store_bytes_without_slot(ExpertStore *st, int *experts) {
  uint64_t b = 0;
  int n = 0;
  for (int l = 0; l < st->scored_layers; l++)
    for (int e = 0; e < st->experts; e++)
      if (WHERE(l, e) < 0) {
        n++;
        for (int k = 0; k < st->spans[l]; k++) b += SPAN(l, e, k).len;
      }
  *experts = n;
  return b;
}
