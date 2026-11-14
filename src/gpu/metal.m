#include "gpu/gpu.h"
#include "base/error.h"
#include "engine/memory.h"
#include <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Calls Metal in Objective-C; the rest of the file is C. Metal objects are
// retained and released by hand.

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

enum { MAX_KERNELS = 512 };

// Metal allows at most 32 residency sets per command queue.
enum { RESIDENCY_SETS = 32 };

// Marks a table entry whose buffer was removed. Searches go past it, and a
// rebuild of the table leaves it out.
#define REMOVED_BUFFER ((id)1)

// A wait for a signal ends the program after a minute by default. Nothing was
// measured to pick a minute: it only needs to be far longer than any command
// buffer runs, so that only a failed command buffer or a stopped GPU ends it.
enum { DEFAULT_SIGNAL_TIMEOUT_MS = 60000 };

struct GpuEvent {
  Gpu *gpu;
  id<MTLSharedEvent> event;
  // The last value given out.
  uint64_t signaled;
};

// A residency set, and the bytes of the buffers in it.
typedef struct {
  id<MTLResidencySet> set;
  uint64_t bytes;
} ResidencySet;

// An entry of the table of resident buffers.
typedef struct {
  // The Metal buffer; null if the entry is empty.
  id<MTLBuffer> buffer;
  // The bytes the buffer takes, and its residency set.
  uint64_t bytes;
  uint32_t set;
} ResidentBuffer;

struct Gpu {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  // The kernels' pipelines and names.
  id<MTLComputePipelineState> pipes[MAX_KERNELS];
  const char *names[MAX_KERNELS];
  int npipes;
  // Committed command buffers not yet checked, oldest first. Each stays
  // retained until it is checked, so that the autorelease pool can be drained
  // while it runs.
  id<MTLCommandBuffer> *done;
  int ndone, done_cap;
  // The last committed command buffer.
  id<MTLCommandBuffer> last;
  // The autorelease pool.
  void *pool;
  // The residency sets. Buffers are spread over several sets, since one very
  // large set made the GPU about ten times slower.
  ResidencySet residency[RESIDENCY_SETS];
  // The sets made, and one bit for each set changed since its last commit.
  uint32_t nresidency, residency_changed;
  // The most bytes a set holds.
  uint64_t residency_set_bytes;
  // The resident buffers, in a table of 2^resident_bits entries. The table is
  // keyed by the Metal buffer, so views of a buffer share its entry. Each
  // buffer has a start entry; if that is taken, the buffer goes in the next
  // free entry after it.
  ResidentBuffer *resident;
  uint32_t resident_bits;
  // The resident buffers, and the entries that are not empty: theirs and the
  // removed ones.
  size_t resident_buffers, resident_filled;
  // How long a wait for a signal lasts before it ends the program.
  uint64_t signal_timeout_ms;
};

// Returns the bits of the smallest table that count resident buffers fill at
// most half of.
static uint32_t table_bits(size_t count) {
  uint32_t bits = 0;
  while (((size_t)1 << bits) < 2 * count) bits++;
  return bits;
}

static const char *describe(NSError *err) {
  return [[err localizedDescription] UTF8String];
}

Gpu *gpu_new(const char *source, size_t len) {
  Gpu *gpu = xmalloc(sizeof *gpu);
  *gpu = (Gpu){0};
  gpu->pool = objc_autoreleasePoolPush();
  gpu->device = MTLCreateSystemDefaultDevice();
  if (!gpu->device) die("no Metal device");
  gpu->queue = [gpu->device newCommandQueue];
  gpu->signal_timeout_ms = DEFAULT_SIGNAL_TIMEOUT_MS;
  // A set holds a twentieth of the working set, so 20 sets cover it, within
  // Metal's limit of 32.
  gpu->residency_set_bytes = [gpu->device recommendedMaxWorkingSetSize] / 20;
  gpu->resident_bits = table_bits(0);
  gpu->resident =
      xmalloc(((size_t)1 << gpu->resident_bits) * sizeof *gpu->resident);
  MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
  [opts setMathMode:MTLMathModeSafe];  // IEEE: no fast math
  [opts setLanguageVersion:MTLLanguageVersion4_1];  // for tensors
  char *src = xmalloc(len + 1);
  memcpy(src, source, len);
  src[len] = 0;
  NSError *err = nil;
  gpu->library =
      [gpu->device newLibraryWithSource:[NSString stringWithUTF8String:src]
                                options:opts
                                  error:&err];
  if (!gpu->library) die("Metal compile: %s", describe(err));
  free(src);
  return gpu;
}

int gpu_find_kernel(Gpu *gpu, const char *name) {
  for (int i = 0; i < gpu->npipes; i++)
    if (!strcmp(gpu->names[i], name)) return i;
  if (gpu->npipes == MAX_KERNELS) die("more than %d kernels", MAX_KERNELS);
  id<MTLFunction> fn =
      [gpu->library newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) die("no kernel %s", name);
  NSError *err = nil;
  id<MTLComputePipelineState> p =
      [gpu->device newComputePipelineStateWithFunction:fn error:&err];
  if (!p) die("pipeline %s: %s", name, describe(err));
  gpu->names[gpu->npipes] = name;
  gpu->pipes[gpu->npipes] = p;
  return gpu->npipes++;
}

size_t gpu_round_up_to_pages(size_t size) {
  size_t page = (size_t)getpagesize();
  if (size > SIZE_MAX - page + 1) die("GPU allocation size overflow");
  return (size + page - 1) / page * page;
}

static Buf allocate(Gpu *gpu, MemorySlot *slot, size_t size) {
  if (!slot) die("a GPU buffer of %zu bytes without a memory slot", size);
  Buf b = {0};
  b.gpu = gpu;
  b.allocated = gpu_round_up_to_pages(size);
  b.memory = slot;
  if (!memory_claim(slot, b.allocated, MEMORY_REQUIRED)) return (Buf){0};
  id<MTLBuffer> buffer =
      [gpu->device newBufferWithLength:size
                               options:MTLResourceStorageModeShared];
  if (!buffer) die("cannot allocate %zu bytes on the GPU", size);
  b.obj = buffer;
  b.ptr = [buffer contents];
  b.size = size;
  return b;
}

Buf gpu_alloc(Gpu *gpu, MemorySlot *slot, size_t size) {
  Buf b = allocate(gpu, slot, size);
  if (!b.obj) die("%zu bytes of model memory do not fit", size);
  return b;
}

void gpu_free(Buf *b) {
  if (b->obj) {
    gpu_set_resident(b, 0);
    [(id)b->obj release];
    if (b->memory) memory_release(b->memory, b->allocated);
  }
  memset(b, 0, sizeof *b);
}

// Returns the entry of buffer in the table, or else the entry it would take:
// the first removed entry on the way, or the empty entry that ends the search.
// Metal buffer objects are 16-byte aligned, so the start entry comes from the
// address without its low 4 bits.
static size_t resident_entry(const Gpu *gpu, id<MTLBuffer> buffer) {
  size_t mask = ((size_t)1 << gpu->resident_bits) - 1;
  size_t at = ((uintptr_t)buffer >> 4) & mask, removed = SIZE_MAX;
  for (; gpu->resident[at].buffer; at = (at + 1) & mask) {
    if (gpu->resident[at].buffer == buffer) return at;
    if (gpu->resident[at].buffer == REMOVED_BUFFER && removed == SIZE_MAX)
      removed = at;
  }
  return removed != SIZE_MAX ? removed : at;
}

// Rebuilds the table with 2^bits entries, leaving out the removed ones.
static void rebuild_resident_table(Gpu *gpu, uint32_t bits) {
  ResidentBuffer *old = gpu->resident;
  size_t old_size = (size_t)1 << gpu->resident_bits;
  gpu->resident_bits = bits;
  gpu->resident = xmalloc(((size_t)1 << bits) * sizeof *gpu->resident);
  for (size_t i = 0; i < old_size; i++)
    if (old[i].buffer && old[i].buffer != REMOVED_BUFFER)
      gpu->resident[resident_entry(gpu, old[i].buffer)] = old[i];
  free(old);
  gpu->resident_filled = gpu->resident_buffers;
}

// Returns the set for a resident buffer of bytes: the first set that is empty
// or has room for it, or else a new set. Once all 32 sets exist, the set that
// holds the fewest bytes.
static uint32_t residency_set_for(Gpu *gpu, uint64_t bytes) {
  uint64_t most = gpu->residency_set_bytes;
  uint32_t fewest = 0;
  for (uint32_t i = 0; i < gpu->nresidency; i++) {
    uint64_t held = gpu->residency[i].bytes;
    if (!held || (held <= most && bytes <= most - held)) return i;
    if (held < gpu->residency[fewest].bytes) fewest = i;
  }
  if (gpu->nresidency == RESIDENCY_SETS) return fewest;
  MTLResidencySetDescriptor *descriptor =
      [[MTLResidencySetDescriptor alloc] init];
  NSError *err = nil;
  id<MTLResidencySet> set =
      [gpu->device newResidencySetWithDescriptor:descriptor error:&err];
  [descriptor release];
  if (!set) die("residency set: %s", describe(err));
  [gpu->queue addResidencySet:set];
  gpu->residency[gpu->nresidency].set = set;
  return gpu->nresidency++;
}

void gpu_set_resident(const Buf *b, int resident) {
  if (!b->obj) die("residency of an empty buffer");
  Gpu *gpu = b->gpu;
  id<MTLBuffer> buffer = b->obj;
  size_t at = resident_entry(gpu, buffer);
  if (resident) {
    if (gpu->resident[at].buffer == buffer) return;
    // Before the table is three quarters full, removed entries included, it
    // is rebuilt so that searches stay short: at the smallest size that the
    // resident buffers, this one included, fill at most half of.
    size_t size = (size_t)1 << gpu->resident_bits;
    if ((gpu->resident_filled + 1) * 4 >= size * 3) {
      rebuild_resident_table(gpu, table_bits(gpu->resident_buffers + 1));
      at = resident_entry(gpu, buffer);
    }
    uint64_t bytes = [buffer allocatedSize];
    uint32_t set = residency_set_for(gpu, bytes);
    [gpu->residency[set].set addAllocation:buffer];
    gpu->residency[set].bytes += bytes;
    gpu->residency_changed |= 1u << set;
    if (!gpu->resident[at].buffer) gpu->resident_filled++;
    gpu->resident[at] = (ResidentBuffer){buffer, bytes, set};
    gpu->resident_buffers++;
  } else if (gpu->resident[at].buffer == buffer) {
    ResidentBuffer *entry = &gpu->resident[at];
    [gpu->residency[entry->set].set removeAllocation:buffer];
    gpu->residency[entry->set].bytes -= entry->bytes;
    gpu->residency_changed |= 1u << entry->set;
    entry->buffer = REMOVED_BUFFER;
    gpu->resident_buffers--;
  }
}

void gpu_commit_residency(Gpu *gpu) {
  for (uint32_t i = 0; i < gpu->nresidency; i++)
    if (gpu->residency_changed & (1u << i)) [gpu->residency[i].set commit];
  gpu->residency_changed = 0;
}

void gpu_set_residency_set_bytes(Gpu *gpu, uint64_t bytes) {
  gpu->residency_set_bytes = bytes;
}

GpuResidency gpu_residency(Gpu *gpu) {
  GpuResidency r = {.sets = gpu->nresidency};
  // allAllocations keeps each buffer alive until its autorelease pool is
  // drained, so its arrays get a pool of their own.
  void *pool = objc_autoreleasePoolPush();
  for (uint32_t i = 0; i < gpu->nresidency; i++) {
    id<MTLResidencySet> set = gpu->residency[i].set;
    r.buffers += (uint32_t)[set allocationCount];
    NSArray *allocations = [set allAllocations];
    uint64_t bytes = 0;
    for (unsigned long j = 0; j < [allocations count]; j++)
      bytes += [[allocations objectAtIndex:j] allocatedSize];
    r.bytes += bytes;
    if (bytes > r.largest_set) r.largest_set = bytes;
  }
  objc_autoreleasePoolPop(pool);
  return r;
}

GpuCommandBuffer gpu_begin(Gpu *gpu) {
  id<MTLCommandBuffer> buffer =
      [gpu->queue commandBufferWithUnretainedReferences];
  [buffer enqueue];
  return (GpuCommandBuffer){gpu, buffer, [buffer computeCommandEncoder]};
}

// The encoder of a command buffer that is still being encoded.
static id<MTLComputeCommandEncoder> encoder(const GpuCommandBuffer *c) {
  if (!c->encoder) die("encoding into a command buffer after it was committed");
  return c->encoder;
}

static void move_to_done(Gpu *gpu, id<MTLCommandBuffer> c) {
  if (gpu->ndone == gpu->done_cap) {
    // The list doubles when it is full; 256 is only its first size.
    gpu->done_cap = gpu->done_cap ? 2 * gpu->done_cap : 256;
    gpu->done = realloc(gpu->done, (size_t)gpu->done_cap * sizeof *gpu->done);
    if (!gpu->done) die("out of memory");
  }
  gpu->done[gpu->ndone++] = [c retain];
  gpu->last = c;
}

void gpu_commit(GpuCommandBuffer *c) {
  [encoder(c) endEncoding];
  id<MTLCommandBuffer> buffer = c->buffer;
  [buffer commit];
  move_to_done(c->gpu, buffer);
  c->encoder = NULL;
}

void gpu_select_kernel(GpuCommandBuffer *c, int k) {
  if (k < 0 || k >= c->gpu->npipes) die("no kernel number %d", k);
  [encoder(c) setComputePipelineState:c->gpu->pipes[k]];
}

void gpu_bind_buffer(GpuCommandBuffer *c, int idx, const Buf *b, size_t off) {
  [encoder(c) setBuffer:b->obj offset:b->off + off atIndex:(unsigned long)idx];
}

void gpu_copy_bytes(GpuCommandBuffer *c, int idx, const void *p, size_t n) {
  [encoder(c) setBytes:p length:n atIndex:(unsigned long)idx];
}

void gpu_run_kernel(GpuCommandBuffer *c, uint32_t groups_x, uint32_t groups_y,
                    uint32_t groups_z, uint32_t threads_x, uint32_t threads_y,
                    uint32_t threads_z) {
  [encoder(c)
       dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z)
      threadsPerThreadgroup:MTLSizeMake(threads_x, threads_y, threads_z)];
}

GpuEvent *gpu_event_new(Gpu *gpu) {
  GpuEvent *event = xmalloc(sizeof *event);
  *event = (GpuEvent){gpu, [gpu->device newSharedEvent], 0};
  return event;
}

uint64_t gpu_signal(GpuCommandBuffer *c, GpuEvent *event) {
  if (event->gpu != c->gpu) die("GPU: a signal on an event of another GPU");
  [encoder(c) endEncoding];
  id<MTLCommandBuffer> buffer = c->buffer;
  [buffer encodeSignalEvent:event->event value:++event->signaled];
  c->encoder = [buffer computeCommandEncoder];
  return event->signaled;
}

// Checks the first count command buffers of done, all finished. Ends the
// program on an error.
static void check_finished(Gpu *gpu, int count) {
  if (!count) return;
  for (int i = 0; i < count; i++) {
    id<MTLCommandBuffer> c = gpu->done[i];
    if ([c status] == MTLCommandBufferStatusError)
      die("GPU: %s", describe([c error]));
    if (c == gpu->last) gpu->last = nil;
    [c release];
  }
  gpu->ndone -= count;
  memmove(gpu->done, gpu->done + count, (size_t)gpu->ndone * sizeof *gpu->done);
}

// Ends the program for a value not signaled in time: with the error of a
// failed command buffer if there is one, since its signals never come.
static void unsignaled(Gpu *gpu, uint64_t value) {
  for (int i = 0; i < gpu->ndone; i++)
    if ([gpu->done[i] status] == MTLCommandBufferStatusError)
      die("GPU: %s", describe([gpu->done[i] error]));
  die("GPU: value %llu not signaled in %.1f s", (unsigned long long)value,
      gpu->signal_timeout_ms / 1000.0);
}

void gpu_wait_signal(GpuEvent *event, uint64_t value) {
  if (value > event->signaled)
    die("GPU: waiting for value %llu, which the event never gave out",
        (unsigned long long)value);
  if (![event->event waitUntilSignaledValue:value
                                  timeoutMS:event->gpu->signal_timeout_ms])
    unsignaled(event->gpu, value);
}

void gpu_set_signal_timeout(Gpu *gpu, double seconds) {
  // Metal takes whole milliseconds, at most UINT64_MAX of them.
  double milliseconds = seconds * 1000;
  gpu->signal_timeout_ms = !(milliseconds > 0)     ? 0
                           : milliseconds < 0x1p64 ? (uint64_t)milliseconds
                                                   : UINT64_MAX;
}

void gpu_sync(Gpu *gpu) {
  if (!gpu->last) return;
  [gpu->last waitUntilCompleted];
  for (int i = 0; i < gpu->ndone; i++) [gpu->done[i] waitUntilCompleted];
  check_finished(gpu, gpu->ndone);
  gpu->last = nil;
  objc_autoreleasePoolPop(gpu->pool);
  gpu->pool = objc_autoreleasePoolPush();
}
