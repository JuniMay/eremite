#include "engine/kv_pages.h"
#include "base/error.h"
#include <stdlib.h>
#include <string.h>

// Paged KV cache table implementation with reference-counted copy-on-write
// sharing across forked states.

// ---- Metal GPU backend ----

// Allocates a resident GPU buffer in `slot`, returning empty on refusal.
static Buf gpu_backend_alloc(void *ctx, MemorySlot *slot, size_t bytes) {
  Gpu *gpu = (Gpu *)ctx;
  Buf b = gpu_try_alloc_required(gpu, slot, bytes);
  if (b.obj) gpu_set_resident(&b, 1);
  return b;
}

// Marks a GPU buffer for deferred release after in-flight command buffers end.
static void gpu_backend_release(void *ctx, Buf *buf) {
  (void)ctx;
  if (buf->obj) gpu_mark_to_free(buf);
}

// Commits pending residency set updates on the GPU.
static void gpu_backend_commit(void *ctx) { gpu_commit_residency((Gpu *)ctx); }

// Rounds a GPU buffer size up to Metal's page granularity.
static size_t gpu_backend_round_bytes(size_t bytes) {
  return gpu_round_up_to_pages(bytes);
}

KvPageBackend kv_page_backend_gpu(Gpu *gpu, MemorySlot *slot) {
  return (KvPageBackend){
      .ctx = gpu,
      .slot = slot,
      .alloc = gpu_backend_alloc,
      .release = gpu_backend_release,
      .commit = gpu_backend_commit,
      .round_bytes = gpu_backend_round_bytes,
  };
}

// ---- layout helpers ----

// Returns the number of pages needed to cover `positions` tokens.
static int pages_for_positions(const KvPageLayout *layout, int positions) {
  if (positions <= 0) return 0;
  return (positions + layout->page_tokens - 1) / layout->page_tokens;
}

// Rounds `bytes` using `backend->round_bytes` when set.
static size_t round_backend_bytes(const KvPageBackend *backend, size_t bytes) {
  if (bytes == 0) return 0;
  return backend && backend->round_bytes ? backend->round_bytes(bytes) : bytes;
}

// Returns the next table capacity in pages covering `need_pages`.
static int grown_table_cap(const KvPageLayout *layout, int current_cap,
                           int need_pages) {
  if (need_pages <= current_cap) return current_cap;
  int cap = state_growth_next(&layout->table_growth, current_cap, need_pages);
  if (cap < 0)
    die("KV page table need %d exceeds max %d", need_pages,
        layout->table_growth.max_cap);
  return cap;
}

KvPageLayout kv_page_layout_init(int page_tokens, size_t slice_bytes,
                                 int slice_count,
                                 StateGrowthPolicy table_growth) {
  if (page_tokens <= 0 || (page_tokens & (page_tokens - 1)) != 0)
    die("KV page_tokens %d must be a positive power of two", page_tokens);
  if (slice_bytes == 0 || slice_count <= 0)
    die("invalid KV page slice geometry (%zu bytes x %d slices)", slice_bytes,
        slice_count);
  uint32_t shift = 0;
  for (int v = page_tokens; v > 1; v >>= 1) shift++;
  return (KvPageLayout){
      .page_tokens = page_tokens,
      .page_shift = shift,
      .slice_bytes = slice_bytes,
      .slice_count = slice_count,
      .page_bytes = slice_bytes * (size_t)slice_count,
      .table_growth = table_growth,
  };
}

KvPageLayout kv_page_layout_init_packed(int page_tokens, size_t page_bytes,
                                        StateGrowthPolicy table_growth) {
  if (page_tokens <= 0 || (page_tokens & (page_tokens - 1)) != 0)
    die("KV page_tokens %d must be a positive power of two", page_tokens);
  if (page_bytes == 0) die("invalid packed KV page size %zu", page_bytes);
  uint32_t shift = 0;
  for (int v = page_tokens; v > 1; v >>= 1) shift++;
  return (KvPageLayout){
      .page_tokens = page_tokens,
      .page_shift = shift,
      .slice_bytes = 0,
      .slice_count = 1,
      .page_bytes = page_bytes,
      .table_growth = table_growth,
  };
}

size_t kv_page_alloc_bytes(const KvPageBackend *backend,
                           const KvPageLayout *layout) {
  return round_backend_bytes(backend, layout->page_bytes);
}

// ---- internal page and table operations ----

// Allocates a new private KvPage (refs == 1), or returns null on refusal.
static KvPage *page_alloc(const KvPageBackend *backend,
                          const KvPageLayout *layout) {
  Buf buf = backend->alloc(backend->ctx, backend->slot, layout->page_bytes);
  if (!buf.ptr) return NULL;
  KvPage *page = xmalloc(sizeof *page);
  *page = (KvPage){.buf = buf, .refs = 1};
  return page;
}

// Decrements `page->refs` and releases its backing buffer when it reaches zero.
static void page_unref(const KvPageBackend *backend, KvPage *page) {
  if (!page) return;
  if (page->refs == 0) die("KV page reference count underflow");
  page->refs--;
  if (page->refs == 0) {
    backend->release(backend->ctx, &page->buf);
    free(page);
  }
}

// Ensures the host pointer arrays and GPU address buffer hold at least
// `need_pages` entries. Returns 1 on success or 0 on allocation refusal.
static int table_reserve(KvPageTable *t, int need_pages) {
  if (need_pages <= t->cap_pages) return 1;
  int next_cap = grown_table_cap(&t->layout, t->cap_pages, need_pages);
  Buf new_addrs = t->backend.alloc(t->backend.ctx, t->backend.slot,
                                   (size_t)next_cap * sizeof(uint64_t));
  if (!new_addrs.ptr) return 0;
  if (t->npages > 0 && t->gpu_addrs.ptr)
    memcpy(new_addrs.ptr, t->gpu_addrs.ptr,
           (size_t)t->npages * sizeof(uint64_t));
  if (t->gpu_addrs.ptr) t->backend.release(t->backend.ctx, &t->gpu_addrs);
  t->gpu_addrs = new_addrs;

  KvPage **new_pages = realloc(t->pages, (size_t)next_cap * sizeof *new_pages);
  void **new_host = realloc(t->host_ptrs, (size_t)next_cap * sizeof *new_host);
  if (!new_pages || !new_host) die("out of memory growing KV page table");
  t->pages = new_pages;
  t->host_ptrs = new_host;
  t->cap_pages = next_cap;
  return 1;
}

// Writes `page` into slot `idx` of `t`, updating host_ptrs and gpu_addrs.
static void table_set_slot(KvPageTable *t, int idx, KvPage *page) {
  t->pages[idx] = page;
  t->host_ptrs[idx] = page->buf.ptr;
  ((uint64_t *)t->gpu_addrs.ptr)[idx] = page->buf.address + page->buf.off;
}

// Commits backend residency updates when the backend provides a commit hook.
static void commit_backend(const KvPageBackend *backend) {
  if (backend->commit) backend->commit(backend->ctx);
}

// ---- public API ----

void kv_page_table_init(KvPageTable *t, KvPageBackend backend,
                        const KvPageLayout *layout) {
  *t = (KvPageTable){.backend = backend, .layout = *layout};
}

void kv_page_table_free(KvPageTable *t) {
  if (!t) return;
  for (int p = 0; p < t->npages; p++) page_unref(&t->backend, t->pages[p]);
  if (t->gpu_addrs.ptr) t->backend.release(t->backend.ctx, &t->gpu_addrs);
  free(t->pages);
  free(t->host_ptrs);
  t->pages = NULL;
  t->host_ptrs = NULL;
  t->npages = 0;
  t->cap_pages = 0;
}

int kv_page_table_needs_sync(const KvPageTable *t, int pos, int count) {
  if (count <= 0) return 0;
  if (pos < 0) die("negative KV write position %d", pos);
  int first = pos >> t->layout.page_shift;
  int last = (pos + count - 1) >> t->layout.page_shift;
  if (last >= t->npages) return 1;
  for (int p = first; p <= last; p++) {
    if (t->pages[p]->refs > 1) return 1;
  }
  return 0;
}

int kv_page_table_prepare_write(KvPageTable *t, int pos, int count) {
  if (count <= 0) return 1;
  if (pos < 0) die("negative KV write position %d", pos);
  int first = pos >> t->layout.page_shift;
  int last = (pos + count - 1) >> t->layout.page_shift;
  int need_pages = last + 1;
  if (need_pages > t->layout.table_growth.max_cap) return 0;

  int allocated_any = 0;
  int old_cap = t->cap_pages;
  if (!table_reserve(t, need_pages)) return 0;
  if (t->cap_pages != old_cap) allocated_any = 1;

  // Stage newly allocated pages for any shared slots in [first, min(last,
  // npages - 1)] and any missing slots [npages, last] before committing so a
  // mid-way allocation refusal leaves `t` untouched.
  int cow_end = last < t->npages ? last : t->npages - 1;
  int cow_count = 0;
  for (int p = first; p <= cow_end; p++) {
    if (t->pages[p]->refs > 1) cow_count++;
  }
  int append_count = need_pages > t->npages ? need_pages - t->npages : 0;
  int total_new = cow_count + append_count;
  if (total_new == 0) {
    if (allocated_any) commit_backend(&t->backend);
    return 1;
  }

  KvPage **staged = xmalloc((size_t)total_new * sizeof *staged);
  int made = 0;
  for (int i = 0; i < total_new; i++) {
    staged[i] = page_alloc(&t->backend, &t->layout);
    if (!staged[i]) {
      for (int j = 0; j < made; j++) page_unref(&t->backend, staged[j]);
      free(staged);
      if (allocated_any) commit_backend(&t->backend);
      return 0;
    }
    made++;
  }

  int si = 0;
  for (int p = first; p <= cow_end; p++) {
    if (t->pages[p]->refs > 1) {
      KvPage *old_page = t->pages[p];
      KvPage *copy = staged[si++];
      memcpy(copy->buf.ptr, old_page->buf.ptr, t->layout.page_bytes);
      table_set_slot(t, p, copy);
      page_unref(&t->backend, old_page);
    }
  }
  for (int p = t->npages; p < need_pages; p++) {
    table_set_slot(t, p, staged[si++]);
  }
  if (need_pages > t->npages) t->npages = need_pages;
  free(staged);
  commit_backend(&t->backend);
  return 1;
}

int kv_page_table_reset_writable(KvPageTable *t, int positions) {
  if (positions < 0) die("negative KV positions %d", positions);
  int need_pages = pages_for_positions(&t->layout, positions);
  if (need_pages > t->layout.table_growth.max_cap) return 0;

  int allocated_any = 0;
  int old_cap = t->cap_pages;
  if (need_pages > 0 && !table_reserve(t, need_pages)) return 0;
  if (t->cap_pages != old_cap) allocated_any = 1;

  int replace_end = need_pages < t->npages ? need_pages : t->npages;
  int need_new = 0;
  for (int p = 0; p < replace_end; p++) {
    if (t->pages[p]->refs > 1) need_new++;
  }
  if (need_pages > t->npages) need_new += need_pages - t->npages;

  KvPage **staged =
      need_new ? xmalloc((size_t)need_new * sizeof *staged) : NULL;
  for (int i = 0; i < need_new; i++) {
    staged[i] = page_alloc(&t->backend, &t->layout);
    if (!staged[i]) {
      for (int j = 0; j < i; j++) page_unref(&t->backend, staged[j]);
      free(staged);
      if (allocated_any) commit_backend(&t->backend);
      return 0;
    }
    allocated_any = 1;
  }

  int si = 0;
  for (int p = 0; p < replace_end; p++) {
    if (t->pages[p]->refs > 1) {
      KvPage *old_page = t->pages[p];
      table_set_slot(t, p, staged[si++]);
      page_unref(&t->backend, old_page);
    }
  }
  for (int p = t->npages; p < need_pages; p++) {
    table_set_slot(t, p, staged[si++]);
  }
  for (int p = need_pages; p < t->npages; p++) {
    page_unref(&t->backend, t->pages[p]);
    t->pages[p] = NULL;
    t->host_ptrs[p] = NULL;
  }
  t->npages = need_pages;
  free(staged);
  if (allocated_any) commit_backend(&t->backend);
  return 1;
}

int kv_page_table_fork(KvPageTable *dst, const KvPageTable *src,
                       int positions) {
  if (dst == src) die("cannot fork a KV page table onto itself");
  if (positions < 0) die("negative KV fork positions %d", positions);
  int full_pages = positions >> src->layout.page_shift;
  int rem_rows = positions & (src->layout.page_tokens - 1);
  int need_pages = full_pages + (rem_rows > 0 ? 1 : 0);
  if (need_pages > src->npages)
    die("KV fork of %d positions exceeds source page count %d", positions,
        src->npages);

  int allocated_any = 0;
  int old_cap = dst->cap_pages;
  if (need_pages > 0 && !table_reserve(dst, need_pages)) return 0;
  if (dst->cap_pages != old_cap) allocated_any = 1;

  // Allocate the private partial boundary page before mutating `dst` so any
  // allocation refusal leaves `dst` unchanged.
  KvPage *boundary = NULL;
  if (rem_rows > 0) {
    boundary = page_alloc(&dst->backend, &dst->layout);
    if (!boundary) {
      if (allocated_any) commit_backend(&dst->backend);
      return 0;
    }
    allocated_any = 1;
    const uint8_t *src_base = src->pages[full_pages]->buf.ptr;
    uint8_t *dst_base = boundary->buf.ptr;
    if (dst->layout.slice_bytes == 0) {
      memcpy(dst_base, src_base, dst->layout.page_bytes);
    } else {
      size_t row_bytes =
          dst->layout.slice_bytes / (size_t)dst->layout.page_tokens;
      size_t copy_bytes = (size_t)rem_rows * row_bytes;
      for (int s = 0; s < dst->layout.slice_count; s++) {
        size_t off = (size_t)s * dst->layout.slice_bytes;
        memcpy(dst_base + off, src_base + off, copy_bytes);
      }
    }
  }

  for (int p = 0; p < dst->npages; p++)
    page_unref(&dst->backend, dst->pages[p]);
  for (int p = 0; p < full_pages; p++) {
    KvPage *shared = src->pages[p];
    shared->refs++;
    table_set_slot(dst, p, shared);
  }
  if (boundary) table_set_slot(dst, full_pages, boundary);
  dst->npages = need_pages;
  if (allocated_any) commit_backend(&dst->backend);
  return 1;
}

size_t kv_page_table_bytes(const KvPageTable *t, const KvPageBackend *backend,
                           const KvPageLayout *layout, int target_positions) {
  int need_pages = pages_for_positions(layout, target_positions);
  size_t page_bytes = kv_page_alloc_bytes(backend, layout);
  if (!t) {
    int cap = need_pages > 0 ? grown_table_cap(layout, 0, need_pages) : 0;
    size_t addr_bytes =
        cap > 0 ? round_backend_bytes(backend, (size_t)cap * sizeof(uint64_t))
                : 0;
    return (size_t)need_pages * page_bytes + addr_bytes;
  }
  int pages_count = need_pages > t->npages ? need_pages : t->npages;
  int cap = pages_count > t->cap_pages
                ? grown_table_cap(layout, t->cap_pages, pages_count)
                : t->cap_pages;
  size_t addr_bytes =
      cap > 0 ? round_backend_bytes(backend, (size_t)cap * sizeof(uint64_t))
              : 0;
  size_t total = addr_bytes;
  for (int p = 0; p < t->npages; p++) {
    uint32_t refs = t->pages[p]->refs > 0 ? t->pages[p]->refs : 1;
    total += page_bytes / refs;
  }
  if (need_pages > t->npages)
    total += (size_t)(need_pages - t->npages) * page_bytes;
  return total;
}

size_t kv_page_table_fork_delta_bytes(const KvPageTable *src,
                                      int shared_positions,
                                      int target_positions) {
  int full_shared =
      shared_positions > 0 ? (shared_positions >> src->layout.page_shift) : 0;
  if (full_shared > src->npages) full_shared = src->npages;
  int target_pages = pages_for_positions(&src->layout, target_positions);
  int new_pages = target_pages > full_shared ? target_pages - full_shared : 0;
  int cap =
      target_pages > 0 ? grown_table_cap(&src->layout, 0, target_pages) : 0;
  size_t addr_bytes =
      cap > 0
          ? round_backend_bytes(&src->backend, (size_t)cap * sizeof(uint64_t))
          : 0;
  return (size_t)new_pages * kv_page_alloc_bytes(&src->backend, &src->layout) +
         addr_bytes;
}
