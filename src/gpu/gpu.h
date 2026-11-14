#ifndef EREMITE_GPU_GPU_H
#define EREMITE_GPU_GPU_H
#include "engine/memory.h"
#include <stddef.h>
#include <stdint.h>

// Runs kernels on the GPU through Metal. Kernels are compiled from source at
// run time. Buffers are counted in model memory. Command buffers run in the
// order they are begun.

// The GPU, its queue and its compiled kernels.
typedef struct Gpu Gpu;

// A buffer, or a view of part of one.
typedef struct {
  // The GPU that made the buffer.
  Gpu *gpu;
  // The Metal buffer.
  void *obj;
  // The CPU address of the view's first byte.
  uint8_t *ptr;
  size_t size;
  // Where the view starts in the Metal buffer.
  uint32_t off;
  // The GPU address of the Metal buffer's first byte. Kernels reach some
  // buffers through addresses kept in other buffers.
  uint64_t address;
  // The slot that counts the buffer.
  MemorySlot *memory;
  // The bytes counted for the buffer. A view counts 0.
  size_t allocated;
} Buf;

// A command buffer being encoded. Encoding into it after gpu_commit ends the
// program.
typedef struct {
  Gpu *gpu;
  // The Metal command buffer. It is held until the commit, and null after it.
  void *buffer;
  // Its compute encoder. It is held until the commit, and null after it.
  void *encoder;
  // Its number: command buffers are numbered in the order they are begun.
  uint64_t number;
  // Whether it was begun in the queue, rather than detached.
  int in_queue;
} GpuCommandBuffer;

// An event: a counter on the GPU. Command buffers raise it as they run, and the
// CPU waits for it to reach a value.
typedef struct GpuEvent GpuEvent;

// A clock keeper: keeps the GPU's clock up while the GPU would idle. The GPU
// lowers its clock within milliseconds of idling. The work after the idle time
// then runs up to 3 times slower.
typedef struct GpuClockKeeper GpuClockKeeper;

// Returns a new Gpu: opens the device and compiles source as Metal 4.1, without
// fast math. Ends the program if it does not compile.
Gpu *gpu_new(const char *source, size_t len);

// Finds the kernel called name, and returns its number. name must stay valid.
// Ends the program if there is no such kernel.
int gpu_find_kernel(Gpu *gpu, const char *name);

// Returns size rounded up to whole pages. That is the memory a buffer of size
// bytes takes.
size_t gpu_round_up_to_pages(size_t size);

// Allocates a buffer of size bytes, counted in slot, as required memory: while
// it does not fit, it waits for released memory that comes back, as
// memory_wait_for_claim does. Returns an empty buffer if it does not fit. Ends
// the program if slot is null.
Buf gpu_try_alloc_required(Gpu *gpu, MemorySlot *slot, size_t size);

// Allocates a buffer as gpu_try_alloc_required does. Ends the program if the
// buffer does not fit.
Buf gpu_alloc(Gpu *gpu, MemorySlot *slot, size_t size);

// Allocates a buffer of size bytes, counted in slot, if it fits with the
// admission mode. Returns an empty buffer if it does not. Ends the program if
// slot is null.
Buf gpu_try_alloc(Gpu *gpu, MemorySlot *slot, size_t size,
                  MemoryAdmission mode);

// Returns Metal's recommended working set: the most memory, in bytes, that the
// GPU should keep mapped at once.
uint64_t gpu_recommended_working_set(Gpu *gpu);

// Frees the buffer, stops counting it, and empties *b. GPU work that uses the
// buffer must have finished. If the buffer is resident, it leaves its residency
// set at the next gpu_commit_residency. Metal frees its memory only then.
void gpu_free(Buf *b);

// Marks the buffer to be freed once the GPU has finished every command buffer
// begun before this call, since those may still use it. Empties *b at once. If
// they have all finished already, frees the buffer now. If not, gpu_sync or
// gpu_check_finished frees it later, once it finds them finished. Until then,
// the buffer is still counted.
void gpu_mark_to_free(Buf *b);

// Sets whether the buffer is resident: kept ready for the GPU between command
// buffers, in one of the GPU's residency sets. A view sets its whole buffer.
// The change takes effect at the next gpu_commit_residency. Ends the program
// if the buffer is empty.
void gpu_set_resident(const Buf *b, int resident);

// Commits the residency sets that changed since their last commit.
void gpu_commit_residency(Gpu *gpu);

// Sets the most bytes a residency set holds. A resident buffer goes to the
// first set with room for it, or else to a new set. Once Metal's limit of 32
// sets is reached, it goes to the set that holds the fewest bytes. The default
// is a twentieth of Metal's recommended working set.
void gpu_set_residency_set_bytes(Gpu *gpu, uint64_t bytes);

// What the residency sets hold, as Metal reports it.
typedef struct {
  // Residency sets, and the buffers they hold. A removed buffer is held until
  // the next gpu_commit_residency.
  uint32_t sets, buffers;
  // Bytes of the buffers that the sets hold after that commit, and of those in
  // the largest set.
  uint64_t bytes, largest_set;
} GpuResidency;

// Returns what the residency sets hold.
GpuResidency gpu_residency(Gpu *gpu);

// Begins a command buffer at the end of the queue and starts encoding into it.
// Command buffers in the queue run in the order they were begun, whenever each
// one is committed.
GpuCommandBuffer gpu_begin(Gpu *gpu);

// Begins a command buffer outside the queue and starts encoding into it. It
// joins the end of the queue when it is committed. So it runs after every
// command buffer committed before it, even one begun after it.
GpuCommandBuffer gpu_begin_detached(Gpu *gpu);

// Ends the encoding and commits the command buffer. Ends the program if it was
// committed before.
void gpu_commit(GpuCommandBuffer *c);

// Selects kernel k for the next dispatches. Ends the program if there is no
// such kernel.
void gpu_select_kernel(GpuCommandBuffer *c, int k);

// Binds the buffer, from byte off, to argument idx.
void gpu_bind_buffer(GpuCommandBuffer *c, int idx, const Buf *b, size_t off);

// Copies n bytes from p into argument idx. The copy is made now, so p may be
// reused at once.
void gpu_copy_bytes(GpuCommandBuffer *c, int idx, const void *p, size_t n);

// Declares that the command buffer's kernels read the buffer through its
// address. A buffer in a committed residency set needs no such call.
void gpu_declare_read(GpuCommandBuffer *c, const Buf *b);

// Runs the selected kernel on a grid of groups_x by groups_y by groups_z
// threadgroups. Each threadgroup has threads_x by threads_y by threads_z
// threads, which run together and can share memory.
void gpu_run_kernel(GpuCommandBuffer *c, uint32_t groups_x, uint32_t groups_y,
                    uint32_t groups_z, uint32_t threads_x, uint32_t threads_y,
                    uint32_t threads_z);

// Returns a new event at 0.
GpuEvent *gpu_event_new(Gpu *gpu);

// Frees the event. Command buffers that signal it must have finished.
void gpu_event_free(GpuEvent *event);

// Marks this point in the command buffer, and returns the next value of event.
// The GPU signals that value on event when it reaches this point, after it has
// finished the work encoded before it. The command buffer goes on with a new
// encoder, so the next dispatches need their kernel and arguments set again.
//
// The GPU never lowers an event, so a wait for a value returns as soon as any
// later value is signaled. Take an event's values in the order its command
// buffers run: a command buffer in the queue runs in the order it was begun,
// and a detached one in the order it was committed.
//
// Ends the program if event belongs to another GPU.
uint64_t gpu_signal(GpuCommandBuffer *c, GpuEvent *event);

// Waits until the GPU has signaled value on event. Ends the program if the
// event never gave out value, if a command buffer failed, or if value is not
// signaled within the GPU's signal timeout.
void gpu_wait_signal(GpuEvent *event, uint64_t value);

// Returns whether the GPU has signaled value on event, without waiting. Ends
// the program if the event never gave out value, or if a command buffer
// failed.
int gpu_has_signaled(GpuEvent *event, uint64_t value);

// Sets how long a wait for a signal lasts, in seconds, before it ends the
// program. The default is a minute.
void gpu_set_signal_timeout(Gpu *gpu, double seconds);

// Returns a clock keeper for the GPU, with its own queue and spin kernel. slot
// counts its small buffer. It takes up to a second, since it times its kernel.
GpuClockKeeper *gpu_clock_keeper_new(Gpu *gpu, MemorySlot *slot);

// Keeps the GPU's clock up for about seconds, and returns at once. It runs a
// spin kernel on one simdgroup, in the clock keeper's own queue. Work on the
// main queue runs beside the spin. A command buffer begun but not committed
// there does not hold the spin back.
void gpu_keep_clock_up(GpuClockKeeper *keeper, double seconds);

// Waits until the clock keeper's spins have finished.
void gpu_wait_for_clock_keeper(GpuClockKeeper *keeper);

// Waits for the last committed command buffer, and so for all begun before
// it. Ends the program if one of them failed.
void gpu_sync(Gpu *gpu);

// Checks which command buffers have finished, without waiting for the others.
// Ends the program if one of them failed. Frees the marked buffers whose
// command buffers have now all finished.
void gpu_check_finished(Gpu *gpu);

// Returns whether the command buffer numbered number has finished, without
// waiting. Ends the program if it was never begun, if it is not committed
// yet, or if it failed.
int gpu_has_finished(Gpu *gpu, uint64_t number);

// Waits until the command buffer numbered number has finished. Ends the
// program if it was never begun, if it is not committed yet, or if it failed.
void gpu_wait_until_finished(Gpu *gpu, uint64_t number);

// Returns how many command buffers begun in the queue are not committed yet.
// Each one holds back the command buffers begun after it.
int gpu_count_uncommitted_in_queue(Gpu *gpu);

// The GPU's times, counted when command buffers are checked.
typedef struct {
  // Seconds the GPU ran any checked command buffer.
  double busy_seconds;
  // Seconds the GPU sat idle between checked command buffers. The time across a
  // gpu_sync is not counted, since the CPU waited then.
  double idle_seconds;
} GpuTimes;

// Returns the GPU's times.
GpuTimes gpu_times(Gpu *gpu);

// Returns the memory of freed buffers to the system. It does three things:
// - It frees each buffer marked to be freed, if its GPU work has finished.
// - It commits the residency sets. A residency set holds a freed buffer's
//   memory until the set is committed.
// - It empties the autorelease pool. The pool can still hold freed Metal
//   objects.
// Call this right after an eviction. Then the freed memory is back before the
// next allocation.
void gpu_return_freed_memory(Gpu *gpu);

#endif
