#ifndef EREMITE_ENGINE_KV_PAGES_H
#define EREMITE_ENGINE_KV_PAGES_H

#include "engine/memory.h"
#include "engine/state_growth.h"
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// Paged KV cache with reference-counted copy-on-write prefix sharing.
//
// Each page stores a fixed power-of-two number of token positions across a
// fixed set of packed slices (for example, K and V across global attention
// layers). States share full prefix pages by incrementing reference counts on
// fork, and duplicate a shared page only when a write modifies positions inside
// that page. Backend buffer allocation is abstracted behind KvPageBackend so
// Metal, host tests, and future discrete-memory backends share one table layer.

// Allocation backend for KV page buffers and GPU page-address tables.
typedef struct {
  // Opaque backend context (for example, a Gpu handle).
  void *ctx;
  // Memory slot counting allocated page and address-table buffers.
  MemorySlot *slot;
  // Allocates a buffer of `bytes` bytes; returns an empty Buf on refusal.
  Buf (*alloc)(void *ctx, MemorySlot *slot, size_t bytes);
  // Releases a buffer once any device work referencing it finishes.
  void (*release)(void *ctx, Buf *buf);
  // Commits residency updates after a batch of page allocations, or null.
  void (*commit)(void *ctx);
  // Rounds a buffer byte size up to the backend's allocation granularity.
  size_t (*round_bytes)(size_t bytes);
} KvPageBackend;

// Geometry and table growth parameters for a paged KV cache.
typedef struct {
  // Power-of-two token positions stored in each page.
  int page_tokens;
  // Base-2 logarithm of page_tokens for bit-shift indexing.
  uint32_t page_shift;
  // Byte size of one slice inside a page (page_tokens * row_bytes).
  size_t slice_bytes;
  // Number of slices packed sequentially inside each page.
  int slice_count;
  // Total byte size of one page buffer (slice_count * slice_bytes).
  size_t page_bytes;
  // Growth policy for the page-address table array, measured in pages.
  StateGrowthPolicy table_growth;
} KvPageLayout;

// One reference-counted fixed-size page shared across states via copy-on-write.
typedef struct {
  // Backing buffer holding all slices for page_tokens positions.
  Buf buf;
  // Number of KvPageTable instances referencing this page.
  uint32_t refs;
} KvPage;

// Per-state table mapping logical page indices to shared KvPage buffers.
typedef struct {
  // Allocation backend for pages and the device address table.
  KvPageBackend backend;
  // Page geometry and table growth policy.
  KvPageLayout layout;
  // Array of cap_pages pointers; indices 0..npages-1 are populated.
  KvPage **pages;
  // Host pointer array mirroring pages[p]->buf.ptr for state file I/O.
  void **host_ptrs;
  // Device buffer holding cap_pages uint64_t addresses (buf.address + off).
  Buf gpu_addrs;
  // Active page count and allocated capacity of pages, host_ptrs and gpu_addrs.
  int npages, cap_pages;
} KvPageTable;

// Returns a Metal GPU page backend counting allocations in `slot`.
KvPageBackend kv_page_backend_gpu(Gpu *gpu, MemorySlot *slot);

// Initializes a page layout for power-of-two `page_tokens`, `slice_count`
// slices of `slice_bytes` bytes each, and table growth policy `table_growth`.
KvPageLayout kv_page_layout_init(int page_tokens, size_t slice_bytes,
                                 int slice_count,
                                 StateGrowthPolicy table_growth);

// Returns the backend-rounded byte size of one page buffer.
size_t kv_page_alloc_bytes(const KvPageBackend *backend,
                           const KvPageLayout *layout);

// Initializes an empty page table with `backend` and `layout`.
void kv_page_table_init(KvPageTable *t, KvPageBackend backend,
                        const KvPageLayout *layout);

// Releases all page references and table buffers owned by `t`.
void kv_page_table_free(KvPageTable *t);

// Returns 1 if writing `count` positions starting at `pos` needs allocation or
// copy-on-write (requiring device synchronization first), or 0 if every target
// page is already allocated and privately owned.
int kv_page_table_needs_sync(const KvPageTable *t, int pos, int count);

// Ensures pages covering `[pos, pos + count)` are allocated and privately owned
// (refs == 1), duplicating any shared page in that range via copy-on-write.
// Returns 1 on success or 0 on allocation refusal (leaving `t` intact).
int kv_page_table_prepare_write(KvPageTable *t, int pos, int count);

// Resets `t` to hold enough privately owned pages for `positions` tokens,
// replacing shared pages and releasing excess pages. Used before reading a
// saved state from disk. Returns 1 on success or 0 on allocation refusal.
int kv_page_table_reset_writable(KvPageTable *t, int positions);

// Replaces `dst` with a copy-on-write fork of `src` covering `positions`
// tokens. Shares all complete prefix pages by incrementing reference counts and
// copies only the prefix rows of any partial trailing boundary page. Returns 1
// on success or 0 on allocation refusal.
int kv_page_table_fork(KvPageTable *dst, const KvPageTable *src, int positions);

// Returns the proportional allocated bytes of `t` (or a fresh table when `t` is
// null) at `target_positions` tokens, dividing shared pages by their reference
// counts and counting new pages beyond `t->npages` as private.
size_t kv_page_table_bytes(const KvPageTable *t, const KvPageBackend *backend,
                           const KvPageLayout *layout, int target_positions);

// Returns the additional bytes allocated when forking `shared_positions` tokens
// from `src` and growing the forked state to `target_positions` tokens.
size_t kv_page_table_fork_delta_bytes(const KvPageTable *src,
                                      int shared_positions,
                                      int target_positions);

#endif
