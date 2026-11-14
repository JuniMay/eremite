#include "check.h"
#include "gpu/gpu.h"

// Tests what runs after an eviction. The eviction happens while two command
// buffers are unfinished. The test checks that:
// - an allocation past the limit evicts an idle expert;
// - the evicted expert's buffer leaves its residency set;
// - both command buffers still run after the eviction;
// - gpu_return_freed_memory frees a buffer marked to be freed, if its work has
//   finished;
// - that buffer also leaves its residency set.

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add_one(device uint *y [[buffer(0)]]) { y[0] += 1; }\n";

// An idle expert holds most of the limit. So one more allocation evicts it.
static constexpr uint64_t LIMIT = 64ull << 20;
static constexpr size_t EXPERT_BYTES = 56 << 20, MORE_BYTES = 16 << 20;

// The add_one kernel's number.
static int add_one;

// An expert in the cache: its slot, its buffer, and its evictions.
typedef struct {
  MemorySlot slot;
  Buf b;
  int evicted;
} Expert;

// Frees the expert's buffer. The Memory object calls it to evict the expert.
static void evict(void *context) {
  Expert *e = context;
  gpu_free(&e->b);
  e->evicted++;
}

// The Memory object's eviction callback.
static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Encodes y += 1.
static void add_one_to(GpuCommandBuffer *c, const Buf *y) {
  gpu_select_kernel(c, add_one);
  gpu_bind_buffer(c, 0, y, 0);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

int main(void) {
  Memory *memory = memory_new();
  memory_set_limit(memory, LIMIT);
  MemorySlot *shared = memory_shared(memory);
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  add_one = gpu_find_kernel(gpu, "add_one");
  MemoryAccount cache;
  memory_account_init(memory, &cache, "expert cache", 0);
  static Expert expert;
  memory_slot_init(&expert.slot, &cache, MEMORY_MODEL, NULL, "routed expert",
                   evict, &expert);
  expert.b = gpu_alloc(gpu, &expert.slot, EXPERT_BYTES);
  gpu_set_resident(&expert.b, 1);
  gpu_commit_residency(gpu);
  Buf out = gpu_alloc(gpu, shared, 4);

  GpuCommandBuffer ahead = gpu_begin(gpu), behind = gpu_begin(gpu);
  add_one_to(&behind, &out);
  gpu_commit(&behind);
  Buf more = gpu_alloc(gpu, shared, MORE_BYTES);
  check(expert.evicted == 1 && more.obj,
        "an allocation past the limit evicts the idle expert");
  check(gpu_residency(gpu).buffers == 0,
        "after the eviction, the expert's buffer has left its residency set");
  add_one_to(&ahead, &out);
  gpu_commit(&ahead);
  gpu_sync(gpu);
  check(((uint32_t *)out.ptr)[0] == 2,
        "both command buffers still run after the eviction");

  Buf used = gpu_alloc(gpu, shared, 4);
  size_t page = used.allocated;
  gpu_set_resident(&used, 1);
  gpu_commit_residency(gpu);
  uint64_t held = memory_usage(memory).allocated;
  GpuCommandBuffer c = gpu_begin(gpu);
  add_one_to(&c, &used);
  gpu_commit(&c);
  gpu_mark_to_free(&used);
  // A command buffer can show as unfinished for a moment after its work is
  // done. So the test looks again until the buffer is freed.
  while (memory_usage(memory).allocated == held) gpu_return_freed_memory(gpu);
  check(memory_usage(memory).allocated == held - page,
        "the marked buffer is freed after its work has finished, without a "
        "sync");
  check(gpu_residency(gpu).buffers == 0,
        "the freed buffer leaves its residency set at once");
  return failures != 0;
}
