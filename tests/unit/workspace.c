#include "check.h"
#include "base/error.h"
#include "model/deepseek/model.h"
#include <stdlib.h>
#include <string.h>

// Tests that the request workspace releases prompt-sized buffers before
// decoding. Host allocations stand in for Metal buffers; no device or model
// is opened.

static size_t live_bytes, peak_bytes;
static int allocations, pending, unsafe_frees, damaged;
static int cache_clock_advances, cache_maintenance;

static Buf fixture_alloc(size_t n) {
  if (n > (32u << 20) || live_bytes + n > (128u << 20))
    die("workspace fixture exceeded its memory limit");
  unsigned char *p = xmalloc(n + 32);
  memset(p, 0xA5, n + 32);
  live_bytes += n;
  if (live_bytes > peak_bytes) peak_bytes = live_bytes;
  allocations++;
  return (Buf){.obj = p, .ptr = p + 16, .size = n};
}

static void fixture_free(Buf *b) {
  if (!b->obj) return;
  unsafe_frees += pending;
  unsigned char *p = b->obj;
  for (int i = 0; i < 16; i++)
    damaged |= p[i] != 0xA5 || p[16 + b->size + i] != 0xA5;
  live_bytes -= b->size;
  free(p);
  *b = (Buf){0};
}

// As gpu_mark_to_free: a buffer stays until the submitted work has finished.
static Buf later[256];
static int nlater;

static void fixture_free_later(Buf *b) {
  if (!b->obj) return;
  if (!pending) {
    fixture_free(b);
    return;
  }
  if (nlater == 256) die("too many deferred frees");
  later[nlater++] = *b;
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
  while (nlater) fixture_free(&later[--nlater]);
}

static void fixture_adapt(ExpertStore *store) {
  (void)store, cache_clock_advances++, cache_maintenance++;
}

static void fixture_idle(ExpertStore *store) {
  (void)store, cache_maintenance++;
}

#include "residency_stand_ins.h"
#define gpu_alloc fixture_gpu_alloc
#define gpu_try_alloc_required fixture_gpu_alloc
#define gpu_free fixture_free
#define gpu_mark_to_free fixture_free_later
#define gpu_sync fixture_sync
#define expert_store_age_and_resize fixture_adapt
#define expert_store_resize_when_idle fixture_idle
#include "model/deepseek/model.c"
#undef gpu_alloc
#undef gpu_try_alloc_required
#undef gpu_free
#undef gpu_mark_to_free
#undef gpu_sync
#undef expert_store_age_and_resize
#undef expert_store_resize_when_idle

int main(void) {
  opt.ctx = 4096;
  opt.draft = 1;
  cand_words = 17;
  Q = fixture_alloc(128u * HEADS * HEAD_DIM * sizeof(float));
  O = fixture_alloc(Q.size);
  request_capacity(64);
  size_t prompt_bytes = live_bytes;
  check(request_rows >= 64 && HS.size >= 64u * HC * DIM * STREAM_BYTES,
        "prompt rows fit");
  memset(HS.ptr, 0x19, HS.size);
  pending = 1;
  request_capacity(1);
  check(nlater > 0 && !unsafe_frees,
        "submitted work keeps the prompt buffers until it finishes");
  fixture_sync(NULL);
  size_t decode_bytes = live_bytes;
  // a verify pass's rows (the token and the rewind margin's), or batched rows
  size_t small =
      replay_margin() + 1 > BATCH_ROWS ? replay_margin() + 1 : BATCH_ROWS;
  check(request_rows == (int)small,
        "decoding retains only the small-batch row capacity");
  check(decode_bytes < prompt_bytes, "prompt allocation is released");
  check(HS.size == small * HC * DIM * STREAM_BYTES,
        "residual storage fits every verification row");
  check(CAND.size == small * cand_words * sizeof(uint32_t),
        "candidate masks retain only the rows being served");
  check(Q.size == small * HEADS * HEAD_DIM * sizeof(float),
        "query workspace releases unused prompt rows");
  check(O.size == Q.size,
        "attention output workspace releases unused prompt rows");
  check(XB.size >= 128u * 3 * DIM * 2,
        "matrix input staging also fits the draft history projection");
  check(HS.ptr[0] == 0 && HS.ptr[HS.size - 1] == 0,
        "new residual storage is cleared");
  int before = allocations;
  for (int m = 1; m <= (int)small; m++) request_capacity(m);
  check(allocations == before,
        "switching among decode batch sizes does not reallocate");
  pending = 1;
  request_capacity(32);
  check(request_rows >= 32 && HS.size >= 32u * HC * DIM * STREAM_BYTES,
        "a later prompt grows the workspace");
  check(nlater > 0 && !unsafe_frees,
        "growth also keeps the old buffers until the work finishes");
  fixture_sync(NULL);
  request_capacity(1);
  check(live_bytes == decode_bytes,
        "later decoding releases prompt storage again");
  opt.lookup = VERIFY_DRAFTS;
  request_capacity(1);
  check(request_rows == VERIFY_DRAFTS + 1,
        "context lookup reserves its wider verification capacity");
  before = allocations;
  for (int m = 1; m <= VERIFY_DRAFTS + 1; m++) request_capacity(m);
  check(allocations == before, "wide verification changes do not reallocate");
  check(cache_maintenance > 0,
        "workspace maintenance re-evaluates cache capacity");
  check(cache_clock_advances == 0,
        "workspace maintenance does not advance the cache-use clock");
  fixture_sync(NULL);
  Buf *all[] = {&HS, &MF, &SEL, &CAND, &IMG, &EC[0], &ES[0], &EC[1], &ES[1]};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) fixture_free(all[i]);
  Buf *work[] = {&H1,  &MA,   &PART, &X,   &XQ, &QA, &QR,    &QRQ, &Q,
                 &O,   &OA,   &AO,   &KV,  &CK, &CS, &LAT,   &IK,  &QI,
                 &IW,  &SC,   &GL,   &GP,  &RT, &PR, &SHG,   &SHU, &SHH,
                 &SHO, &EROW, &EX,   &EKV, &MH, &YK, &TILES, &XH,  &MHH,
                 &XB,  &PTOK, &PW,   &TP,  &WIN};
  for (size_t i = 0; i < sizeof work / sizeof *work; i++) fixture_free(work[i]);
  check(!live_bytes, "all host allocations are released");
  check(!damaged, "buffer guards are intact");
  printf("prompt %zu, decode %zu, peak %zu bytes\n", prompt_bytes, decode_bytes,
         peak_bytes);
  return failures != 0;
}
