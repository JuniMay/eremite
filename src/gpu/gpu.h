#ifndef EREMITE_GPU_GPU_H
#define EREMITE_GPU_GPU_H
#include "engine/memory.h"
#include <stddef.h>
#include <stdint.h>

// Runs kernels on the GPU through Metal. Kernels are compiled from source at
// run time. Buffers are counted in model memory. Command buffers run in the
// order they are opened.

// The GPU, its queue and its compiled kernels.
typedef struct Gpu Gpu;

// A buffer, or a view of part of one.
typedef struct {
  // The Metal buffer.
  void *obj;
  // The CPU address of the view's first byte.
  uint8_t *ptr;
  size_t size;
  // Where the view starts in the Metal buffer.
  uint32_t off;
  // The slot that counts the buffer.
  MemorySlot *memory;
  // The bytes counted for the buffer. A view counts 0.
  size_t allocated;
} Buf;

// A command buffer being encoded. Encoding into it after gpu_commit ends the
// program.
typedef struct {
  Gpu *gpu;
  // The Metal command buffer.
  void *buffer;
  // Its compute encoder. It is null once the command buffer is committed.
  void *encoder;
} GpuCommandBuffer;

// Returns a new Gpu: opens the device and compiles source as Metal 4.1, without
// fast math. Ends the program if it does not compile.
Gpu *gpu_new(const char *source, size_t len);

// Finds the kernel called name, and returns its number. name must stay valid.
// Ends the program if there is no such kernel.
int gpu_find_kernel(Gpu *gpu, const char *name);

// Returns size rounded up to whole pages. That is the memory a buffer of size
// bytes takes.
size_t gpu_round_up_to_pages(size_t size);

// Allocates a buffer of size bytes, counted in slot. Ends the program if slot
// is null or the buffer does not fit.
Buf gpu_alloc(Gpu *gpu, MemorySlot *slot, size_t size);

// Frees the buffer, stops counting it, and empties *b. GPU work that uses the
// buffer must have finished.
void gpu_free(Buf *b);

// Opens a command buffer in the queue, behind those opened before, and starts
// encoding into it.
GpuCommandBuffer gpu_begin(Gpu *gpu);

// Ends the encoding and commits the command buffer.
void gpu_commit(GpuCommandBuffer *c);

// Selects kernel k for the next dispatches. Ends the program if there is no
// such kernel.
void gpu_select_kernel(GpuCommandBuffer *c, int k);

// Binds the buffer, from byte off, to argument idx.
void gpu_bind_buffer(GpuCommandBuffer *c, int idx, const Buf *b, size_t off);

// Copies n bytes from p into argument idx. The copy is made now, so p may be
// reused at once.
void gpu_copy_bytes(GpuCommandBuffer *c, int idx, const void *p, size_t n);

// Runs the selected kernel on a grid of groups_x by groups_y by groups_z
// threadgroups. Each threadgroup has threads_x by threads_y by threads_z
// threads, which run together and can share memory.
void gpu_run_kernel(GpuCommandBuffer *c, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, uint32_t threads_x,
                    uint32_t threads_y, uint32_t threads_z);

// Waits for the last committed command buffer, and so for all opened before
// it. Ends the program if one of them failed.
void gpu_sync(Gpu *gpu);

#endif
