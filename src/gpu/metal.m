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
};

static const char *describe(NSError *err) { return [[err localizedDescription] UTF8String]; }

Gpu *gpu_new(const char *source, size_t len) {
  Gpu *gpu = xmalloc(sizeof *gpu);
  *gpu = (Gpu){0};
  gpu->pool = objc_autoreleasePoolPush();
  gpu->device = MTLCreateSystemDefaultDevice();
  if (!gpu->device) die("no Metal device");
  gpu->queue = [gpu->device newCommandQueue];
  MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
  [opts setMathMode:MTLMathModeSafe];  // IEEE: no fast math
  [opts setLanguageVersion:MTLLanguageVersion4_1];  // for tensors
  char *src = xmalloc(len + 1);
  memcpy(src, source, len);
  src[len] = 0;
  NSError *err = nil;
  gpu->library = [gpu->device newLibraryWithSource:[NSString stringWithUTF8String:src] options:opts error:&err];
  if (!gpu->library) die("Metal compile: %s", describe(err));
  free(src);
  return gpu;
}

int gpu_find_kernel(Gpu *gpu, const char *name) {
  for (int i = 0; i < gpu->npipes; i++)
    if (!strcmp(gpu->names[i], name)) return i;
  if (gpu->npipes == MAX_KERNELS) die("more than %d kernels", MAX_KERNELS);
  id<MTLFunction> fn = [gpu->library newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) die("no kernel %s", name);
  NSError *err = nil;
  id<MTLComputePipelineState> p = [gpu->device newComputePipelineStateWithFunction:fn error:&err];
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
  b.allocated = gpu_round_up_to_pages(size);
  b.memory = slot;
  if (!memory_claim(slot, b.allocated, MEMORY_REQUIRED)) return (Buf){0};
  id<MTLBuffer> buffer = [gpu->device newBufferWithLength:size options:MTLResourceStorageModeShared];
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
    [(id)b->obj release];
    if (b->memory) memory_release(b->memory, b->allocated);
  }
  memset(b, 0, sizeof *b);
}

GpuCommandBuffer gpu_begin(Gpu *gpu) {
  id<MTLCommandBuffer> buffer = [gpu->queue commandBufferWithUnretainedReferences];
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

void gpu_run_kernel(GpuCommandBuffer *c, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, uint32_t threads_x,
                    uint32_t threads_y, uint32_t threads_z) {
  [encoder(c) dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z)
             threadsPerThreadgroup:MTLSizeMake(threads_x, threads_y, threads_z)];
}

// Checks the first count command buffers of done, all finished. Ends the
// program on an error.
static void check_finished(Gpu *gpu, int count) {
  if (!count) return;
  for (int i = 0; i < count; i++) {
    id<MTLCommandBuffer> c = gpu->done[i];
    if ([c status] == MTLCommandBufferStatusError) die("GPU: %s", describe([c error]));
    if (c == gpu->last) gpu->last = nil;
    [c release];
  }
  gpu->ndone -= count;
  memmove(gpu->done, gpu->done + count, (size_t)gpu->ndone * sizeof *gpu->done);
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
