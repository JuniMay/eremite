#include "check.h"
#include "engine/kv_pages.h"
#include "engine/memory.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tests KvPageTable page growth, zero-copy prefix sharing on fork, partial
// boundary page copying, copy-on-write isolation, OOM rollback, proportional
// memory accounting, and fork delta estimation using an in-memory backend.

static constexpr int TEST_PAGE_TOKENS = 16;
static constexpr size_t TEST_ROW_BYTES = 8;
static constexpr size_t TEST_SLICE_BYTES = TEST_PAGE_TOKENS * TEST_ROW_BYTES;
static constexpr int TEST_SLICE_COUNT = 4;
static constexpr size_t TEST_ALIGN_BYTES = 64;
static constexpr int TEST_MAX_PAGES = 64;

typedef struct {
  int alloc_count;
  int release_count;
  int commit_count;
  int fail_after;
} HostBackendCtx;

static size_t host_round_bytes(size_t bytes) {
  return (bytes + TEST_ALIGN_BYTES - 1) & ~(TEST_ALIGN_BYTES - 1);
}

static Buf host_alloc(void *ctx_ptr, MemorySlot *slot, size_t bytes) {
  HostBackendCtx *ctx = (HostBackendCtx *)ctx_ptr;
  if (ctx->fail_after >= 0 && ctx->alloc_count >= ctx->fail_after)
    return (Buf){0};
  size_t rounded = host_round_bytes(bytes);
  if (!memory_claim(slot, rounded, MEMORY_REQUIRED)) return (Buf){0};
  uint8_t *ptr = calloc(1, rounded);
  ctx->alloc_count++;
  return (Buf){
      .ptr = ptr,
      .size = bytes,
      .address = (uint64_t)(uintptr_t)ptr,
      .memory = slot,
      .allocated = rounded,
  };
}

static void host_release(void *ctx_ptr, Buf *buf) {
  HostBackendCtx *ctx = (HostBackendCtx *)ctx_ptr;
  if (!buf->ptr) return;
  memory_release(buf->memory, buf->allocated);
  free(buf->ptr);
  *buf = (Buf){0};
  ctx->release_count++;
}

static void host_commit(void *ctx_ptr) {
  HostBackendCtx *ctx = (HostBackendCtx *)ctx_ptr;
  ctx->commit_count++;
}

static KvPageBackend make_backend(HostBackendCtx *ctx, MemorySlot *slot) {
  return (KvPageBackend){
      .ctx = ctx,
      .slot = slot,
      .alloc = host_alloc,
      .release = host_release,
      .commit = host_commit,
      .round_bytes = host_round_bytes,
  };
}

// Writes a deterministic pattern into slice `s`, position `pos` of `t`.
static void write_row(KvPageTable *t, int s, int pos, uint64_t val) {
  int page_idx = pos >> t->layout.page_shift;
  int in_page = pos & (t->layout.page_tokens - 1);
  uint8_t *base = (uint8_t *)t->host_ptrs[page_idx] +
                  (size_t)s * t->layout.slice_bytes +
                  (size_t)in_page * TEST_ROW_BYTES;
  memcpy(base, &val, sizeof val);
}

// Reads the pattern stored at slice `s`, position `pos` of `t`.
static uint64_t read_row(const KvPageTable *t, int s, int pos) {
  int page_idx = pos >> t->layout.page_shift;
  int in_page = pos & (t->layout.page_tokens - 1);
  const uint8_t *base = (const uint8_t *)t->host_ptrs[page_idx] +
                        (size_t)s * t->layout.slice_bytes +
                        (size_t)in_page * TEST_ROW_BYTES;
  uint64_t val = 0;
  memcpy(&val, base, sizeof val);
  return val;
}

int main(void) {
  Memory *memory = memory_new();
  MemoryAccount account = {0};
  MemorySlot slot = {0};
  memory_set_limit(memory, 16ull << 20);
  memory_account_init(memory, &account, "kv-pages-test", 0);
  memory_slot_init(&slot, &account, MEMORY_STATE, NULL, "kv pages", NULL, NULL);
  slot.automatic = 1;

  HostBackendCtx ctx = {.fail_after = -1};
  KvPageBackend backend = make_backend(&ctx, &slot);
  StateGrowthPolicy table_growth = {
      .start_cap = 4,
      .factor = 2,
      .max_step = 8,
      .align = 4,
      .max_cap = TEST_MAX_PAGES,
  };
  KvPageLayout layout = kv_page_layout_init(TEST_PAGE_TOKENS, TEST_SLICE_BYTES,
                                            TEST_SLICE_COUNT, table_growth);
  size_t page_alloc_bytes = kv_page_alloc_bytes(&backend, &layout);

  // 1. Linear page allocation and fast-path needs_sync check.
  KvPageTable a;
  kv_page_table_init(&a, backend, &layout);
  check(kv_page_table_needs_sync(&a, 0, 1) == 1,
        "empty table needs sync before first write");
  check(kv_page_table_prepare_write(&a, 0, 40) == 1,
        "prepare_write allocates 3 pages for 40 positions");
  check(a.npages == 3, "40 tokens occupy 3 pages of 16 tokens");
  check(kv_page_table_needs_sync(&a, 40, 8) == 0,
        "writing within existing private page needs no sync");
  check(kv_page_table_needs_sync(&a, 48, 1) == 1,
        "crossing into page 3 requires sync");

  for (int p = 0; p < 40; p++) {
    for (int s = 0; s < TEST_SLICE_COUNT; s++)
      write_row(&a, s, p, (uint64_t)(p * 100 + s + 1));
  }
  uint64_t *addrs_a = (uint64_t *)a.gpu_addrs.ptr;
  check(addrs_a[0] == a.pages[0]->buf.address &&
            addrs_a[2] == a.pages[2]->buf.address,
        "gpu_addrs mirrors allocated page device addresses");

  // 2. Fork at exact page boundary (32 tokens = 2 full pages): zero page alloc.
  int allocs_before = ctx.alloc_count;
  KvPageTable b;
  kv_page_table_init(&b, backend, &layout);
  size_t est_exact = kv_page_table_fork_delta_bytes(&a, 32, 32);
  check(est_exact == host_round_bytes(4 * sizeof(uint64_t)),
        "exact-boundary fork delta counts only the address table");
  check(kv_page_table_fork(&b, &a, 32) == 1,
        "fork at exact page boundary succeeds");
  check(b.npages == 2, "forked table holds 2 pages");
  check(b.pages[0] == a.pages[0] && b.pages[1] == a.pages[1],
        "full prefix pages are shared by pointer without copying");
  check(a.pages[0]->refs == 2 && a.pages[1]->refs == 2 && a.pages[2]->refs == 1,
        "shared prefix pages have refs == 2 while unshared tail stays 1");
  // Only 1 allocation occurred (the gpu_addrs buffer for b, zero page buffers).
  check(ctx.alloc_count == allocs_before + 1,
        "exact-boundary fork allocates zero page buffers");

  // 3. Fork at mid-page boundary (37 tokens = 2 full pages + 5 boundary rows).
  KvPageTable c;
  kv_page_table_init(&c, backend, &layout);
  size_t est_partial = kv_page_table_fork_delta_bytes(&a, 37, 48);
  check(est_partial ==
            page_alloc_bytes + host_round_bytes(4 * sizeof(uint64_t)),
        "partial-boundary fork delta counts 1 private boundary page");
  check(kv_page_table_fork(&c, &a, 37) == 1,
        "fork at mid-page boundary succeeds");
  check(c.npages == 3, "mid-page fork holds 3 pages");
  check(c.pages[0] == a.pages[0] && c.pages[1] == a.pages[1],
        "mid-page fork still shares full prefix pages");
  check(c.pages[2] != a.pages[2] && c.pages[2]->refs == 1,
        "mid-page fork allocates a private copy of the boundary page");
  int partial_ok = 1;
  for (int p = 0; p < 37; p++) {
    for (int s = 0; s < TEST_SLICE_COUNT; s++) {
      if (read_row(&c, s, p) != (uint64_t)(p * 100 + s + 1)) partial_ok = 0;
    }
  }
  check(partial_ok, "mid-page fork preserves all 37 prefix rows across slices");

  // 4. Proportional bytes across shared states sum to actual allocated bytes.
  size_t sum_bytes = kv_page_table_bytes(&a, &backend, &layout, 40) +
                     kv_page_table_bytes(&b, &backend, &layout, 32) +
                     kv_page_table_bytes(&c, &backend, &layout, 37);
  // 2 shared pages (refs == 3) + 2 private pages + 3 address tables.
  // Integer division of page_alloc_bytes (512) by 3 loses 2 bytes per shared
  // page, so sum_bytes is within 3 bytes per shared page of slot.bytes.
  check(sum_bytes <= slot.bytes && slot.bytes - sum_bytes <= 6,
        "proportional state bytes match actual slot bytes across 3 branches");

  // 5. Copy-on-write isolation when modifying a shared page.
  check(kv_page_table_needs_sync(&b, 20, 4) == 1,
        "writing to shared page 1 reports needs_sync == 1");
  check(kv_page_table_prepare_write(&b, 20, 4) == 1,
        "CoW prepare_write duplicates shared page 1");
  check(b.pages[0] == a.pages[0] && a.pages[0]->refs == 3,
        "untouched prefix page 0 stays shared");
  check(b.pages[1] != a.pages[1] && b.pages[1]->refs == 1 &&
            a.pages[1]->refs == 2,
        "written page 1 splits privately with updated refcounts");
  for (int s = 0; s < TEST_SLICE_COUNT; s++) write_row(&b, s, 20, 99999 + s);
  check(read_row(&a, 0, 20) == 2001 && read_row(&c, 0, 20) == 2001 &&
            read_row(&b, 0, 20) == 99999,
        "CoW write modifies only the writing branch");
  check(read_row(&b, 0, 16) == 1601,
        "CoW preserves unmodified rows in the duplicated page");

  // 6. OOM rollback leaves existing table intact.
  ctx.fail_after = ctx.alloc_count + 1;
  KvPage *orig_p0 = b.pages[0];
  int orig_npages = b.npages;
  uint32_t orig_refs = orig_p0->refs;
  check(kv_page_table_prepare_write(&b, 0, 64) == 0,
        "OOM during multi-page CoW/append returns 0");
  check(b.pages[0] == orig_p0 && b.npages == orig_npages &&
            orig_p0->refs == orig_refs,
        "OOM rollback preserves existing page pointers and refcounts");
  ctx.fail_after = -1;

  // 7. reset_writable replaces shared pages and frees excess pages.
  check(kv_page_table_reset_writable(&c, 16) == 1,
        "reset_writable shrinks and unshares table to 1 private page");
  check(c.npages == 1 && c.pages[0]->refs == 1 && a.pages[0]->refs == 2,
        "reset_writable yields private page 0 and decrements old shared refs");

  // 8. Freeing all tables releases every byte in the memory slot.
  kv_page_table_free(&a);
  kv_page_table_free(&b);
  kv_page_table_free(&c);
  check(slot.bytes == 0 && account.bytes == 0,
        "freeing all page tables releases all memory slot bytes");

  return failures != 0;
}
