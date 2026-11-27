#include "check.h"
#include "base/error.h"
#include "model/deepseek/model.h"
#include <stdlib.h>
#include <string.h>

// Tests forking a state from a snapshot with host buffers, as the prefix
// cache does. No model weights or GPU are used.

static size_t allocated, peak;
static int pending, unsafe_frees;

static Buf fixture_alloc(size_t n) {
  if (allocated + n > (64u << 20)) die("state fixture exceeded 64 MiB");
  void *p = xmalloc(n);
  memset(p, 0xD7, n);
  allocated += n;
  if (allocated > peak) peak = allocated;
  return (Buf){
      .obj = p, .ptr = p, .size = n, .address = (uint64_t)(uintptr_t)p};
}

static void fixture_free(Buf *b) {
  if (!b->obj) return;
  unsafe_frees += pending;
  allocated -= b->size;
  free(b->obj);
  *b = (Buf){0};
}

// As gpu_alloc: the model's buffers, whatever their slot.
static Buf fixture_gpu_alloc(Gpu *gpu, MemorySlot *slot, size_t n) {
  (void)gpu, (void)slot;
  return fixture_alloc(n);
}

static void fixture_sync(Gpu *gpu) {
  (void)gpu;
  pending = 0;
}

static size_t fixture_round(size_t bytes) { return bytes; }

#include "residency_stand_ins.h"
#define gpu_alloc fixture_gpu_alloc
#define gpu_try_alloc_required fixture_gpu_alloc
#define gpu_free fixture_free
#define gpu_mark_to_free fixture_free
#define gpu_round_up_to_pages fixture_round
#define gpu_sync fixture_sync
#include "model/deepseek/model.c"
#undef gpu_alloc
#undef gpu_try_alloc_required
#undef gpu_free
#undef gpu_mark_to_free
#undef gpu_round_up_to_pages
#undef gpu_sync

static int filled(const Buf *b, size_t first, unsigned char value) {
  for (size_t i = first; i < b->size; i++)
    if (b->ptr[i] != value) return 0;
  return 1;
}

static State *make_state(int positions) {
  State *s = xmalloc(sizeof *s);
  *s = (State){0};
  for (int l = 0; l < LAYERS; l++) {
    s->ring[l] = fixture_alloc(16);
    if (!L[l].kv_source) continue;
    s->cstate[l] = fixture_alloc(32);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) s->dring[d] = fixture_alloc(16);
  remember_state_layout();
  KvPageLayout layout = ds_kv_layout();
  kv_page_table_init(&s->kv, ds_kv_backend(), &layout);
  if (!kv_page_table_prepare_write(&s->kv, 0, positions + 1))
    die("fixture failed allocating KV pages");
  return s;
}

// The proportional bytes a state's buffers take, plus its rewind copy if any.
static size_t owned(const State *s) {
  size_t n = s->saved ? saved_bytes() : 0;
  for (int l = 0; l < LAYERS; l++) n += s->ring[l].size + s->cstate[l].size;
  for (int d = 0; opt.draft && d < DRAFTS; d++) n += s->dring[d].size;
  n += s->kv.gpu_addrs.size;
  for (int p = 0; p < s->kv.npages; p++)
    n += s->kv.pages[p]->buf.size / s->kv.pages[p]->refs;
  return n;
}

// The source's rolling rows (and history) set to `value`, as at some position.
static void rolling_fill(State *src, unsigned char value) {
  for (int l = 0; l < LAYERS; l++)
    memset(src->ring[l].ptr, value, src->ring[l].size);
  memset(src->cstate[2].ptr, value, src->cstate[2].size);
  for (int d = 0; opt.draft && d < DRAFTS; d++)
    memset(src->dring[d].ptr, value, src->dring[d].size);
}

// A fork as the server's prefix cache makes it: a snapshot of the source at the
// fork, the source going on past it, then the source's global rows before the
// fork shared copy-on-write into another state and the snapshot restored there.
static void run(int ratio, int positions, int large, int draft) {
  memset(L, 0, sizeof L);
  memset(D, 0, sizeof D);
  opt.ctx = large ? 32768 : 1024;
  opt.draft = draft;
  L[2].kv_source = 1;
  L[2].ratio = ratio;
  int src_pos = large ? 2048 : positions + 32;
  State *src = make_state(src_pos), *dst = make_state(0);
  active = src, history = src->history;
  bind(src);
  remember_state_layout();  // the fixture supplies the model's small
                            // rolling-buffer layout
  // What the server's prefix cache charges for a state.
  check(deepseek_state_bytes(src, src_pos) == owned(src),
        "a state is charged the bytes it owns");
  State *fresh = make_state(positions);
  check(deepseek_state_bytes(NULL, positions) == owned(fresh),
        "a new state is charged what it will own");
  deepseek_state_free(fresh);
  rolling_fill(src, 0x42);
  int32_t fork_history[3] = {11, 12, 13};
  memcpy(
      history, fork_history,
      sizeof fork_history);  // the state in use keeps its history in the state
  Snapshot *x = deepseek_snapshot_take(src, positions, 0);
  rolling_fill(src, 0x39);
  for (int i = 0; i < 3; i++) history[i] = src->history[i] = 70 + i;
  for (int p = 0; p < src->kv.npages; p++) {
    uint8_t *base = (uint8_t *)src->kv.host_ptrs[p];
    memset(base + L[2].comp_off, (int)(0x31 + p),
           (size_t)L[2].kv_page_rows * (size_t)comp_row_bytes());
    memset(base + L[2].key_off, (int)(0x51 + p),
           (size_t)L[2].kv_page_rows * (size_t)key_row_bytes());
  }
  size_t saved = saved_bytes();
  dst->saved = xmalloc(saved);  // a rewind point of the destination's old rows
  memset(dst->saved, 0x67, saved);
  check(deepseek_state_bytes(dst, 0) == owned(dst),
        "a state is charged its rewind copy");
  pending = 1;
  deepseek_state_copy(dst, src, positions);
  deepseek_snapshot_restore(dst, x);
  int full_pages = (positions + 1) / DS_KV_PAGE_TOKENS;
  int need_pages = (positions + 1 + DS_KV_PAGE_TOKENS - 1) / DS_KV_PAGE_TOKENS;
  check(dst->kv.npages == need_pages,
        "page count follows the prefix, not the source allocation");
  int shared_ok = 1;
  for (int p = 0; p < full_pages; p++)
    shared_ok &=
        dst->kv.pages[p] == src->kv.pages[p] && src->kv.pages[p]->refs == 2;
  if (full_pages < need_pages)
    shared_ok &= dst->kv.pages[full_pages] != src->kv.pages[full_pages] &&
                 dst->kv.pages[full_pages]->refs == 1;
  check(shared_ok,
        "full prefix pages are shared CoW and trailing page is private");
  int pages_match = 1;
  for (int p = 0; p < need_pages; p++)
    pages_match &=
        !memcmp(src->kv.host_ptrs[p], dst->kv.host_ptrs[p], kv_page_bytes);
  check(pages_match, "global KV rows and guard row match across all pages");
  if (full_pages > 0) {
    if (positions >= DS_KV_PAGE_TOKENS) {
      check(deepseek_state_copy_bytes(src, positions, positions) <
                deepseek_state_bytes(NULL, positions),
            "fork copy bytes reflect shared prefix pages");
    }
    check(deepseek_state_bytes(dst, positions) == owned(dst),
          "proportional state bytes account for shared prefix pages");
    // CoW write on dst splits only the written shared page without altering
    // src.
    state_prepare_write(dst, 0, 1);
    check(dst->kv.pages[0] != src->kv.pages[0] && src->kv.pages[0]->refs == 1 &&
              dst->kv.pages[0]->refs == 1,
          "CoW prepare_write splits shared page 0 on write");
    ((uint8_t *)dst->kv.host_ptrs[0])[0] ^= 0xFF;
    check(((const uint8_t *)src->kv.host_ptrs[0])[0] == 0x31,
          "CoW write into forked branch leaves source page intact");
  }
  int local = 1;
  for (int l = 0; l < LAYERS; l++) local &= filled(&dst->ring[l], 0, 0x42);
  local &= filled(&dst->cstate[2], 0, 0x42);
  for (int d = 0; opt.draft && d < DRAFTS; d++)
    local &= filled(&dst->dring[d], 0, 0x42);
  check(local, "local and draft state comes from the snapshot, not the source "
               "as it is now");
  check(!memcmp(dst->history, fork_history, sizeof fork_history),
        "the snapshot's history is restored");
  check(dst->saved == NULL, "the destination's old rewind point is dropped");
  check(active == src && history[0] == 70 && filled(&src->ring[0], 0, 0x39),
        "active source and its state are unchanged");
  check(!pending && !unsafe_frees,
        "submitted work completes before replacing buffers");
  if (large)
    printf("prefix copy: %d/%d shared pages; delta %zu vs full %zu bytes\n",
           full_pages, need_pages,
           deepseek_state_copy_bytes(src, positions, positions),
           deepseek_state_bytes(NULL, positions));
  deepseek_snapshot_free(x);
  active = NULL;
  deepseek_state_free(dst), deepseek_state_free(src);
}

static void recreate_after_eviction(void) {
  memset(L, 0, sizeof L), memset(D, 0, sizeof D);
  opt.ctx = 128, opt.draft = 0;
  L[2].kv_source = 1, L[2].ratio = 2;
  State *seed = make_state(0);
  bind(seed);
  base_state();
  kv_page_table_free(&seed->kv);
  free(seed);  // base_state owns these same fixture rolling buffers now
  taken = 1;
  State *first = active;
  size_t expected = owned(first), ring = first->ring[0].size,
         local = first->cstate[2].size;
  ds_state_free((ModelState *)first);
  check(active == &parked, "an evicted active state leaves the model parked");
  check(deepseek_state_bytes(NULL, 128) == expected,
        "new-state accounting survives an empty binding");
  State *next = deepseek_state_new();
  check(next->ring[0].size == ring && ring != 0,
        "new window buffers survive eviction of the active state");
  check(next->cstate[2].size == local && local != 0,
        "new compressor buffers survive eviction of the active state");
  check(owned(next) == expected,
        "recreated state owns its complete allocation");
  ds_state_free((ModelState *)next);
}

int main(void) {
  // The model's slots, as deepseek_load makes them: snapshots and saved rows
  // are counted in the state slot.
  Memory *memory = memory_new();
  MemoryAccount account;
  memory_account_init(memory, &account, "test model", UINT64_MAX);
  static MemorySlot work;
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "workspace", NULL,
                   NULL);
  memory_slot_init(&state_slot, &account, MEMORY_STATE, NULL, "state", NULL,
                   NULL);
  work_slot = &work;
  run(2, 545, 1, 1);
  for (int ratio = 1; ratio <= 2; ratio++)
    for (int positions = 255; positions <= 257; positions++)
      run(ratio, positions, 0, positions & 1);
  recreate_after_eviction();
  check(allocated == 0, "all host buffers are released");
  printf("peak %zu bytes\n", peak);
  return failures != 0;
}
