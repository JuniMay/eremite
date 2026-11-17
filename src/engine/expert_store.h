#ifndef EREMITE_ENGINE_EXPERT_STORE_H
#define EREMITE_ENGINE_EXPERT_STORE_H
#include "engine/memory.h"
#include "engine/pass.h"
#include "gpu/gpu.h"
#include <stdint.h>

// Keeps one model's routed experts in memory. A store reads experts from the
// SSD into slots, reads ahead the experts that the model predicts, and evicts
// experts by score. It stays within the memory that other programs leave. The
// model gives the layout: its layers, their experts, and each expert's bytes
// in the files.
//
// The engine thread owns the stores. Threads that the engine thread waits for
// may also ask where experts are.

// One model's store of routed experts.
typedef struct ExpertStore ExpertStore;

// One part of an expert's bytes in a file. A slot holds an expert's parts one
// after the other.
typedef struct {
  // The file, where the part starts in it, and its bytes.
  int fd;
  uint64_t off, len;
} FileSpan;

// The most experts of a layer, and the most spans of an expert.
static constexpr int EXPERT_STORE_EXPERTS_MAX = 512, EXPERT_STORE_PARTS = 8;

// The layout of a model's experts. The layers below scored_layers share one
// pool of slots, evicted by score. Each later layer is a stage. A stage's
// experts are read at the start, into slots of their own.
typedef struct {
  // Every layer the store may hold.
  int layers;
  // The shared pool's layers.
  int scored_layers;
  // Experts per layer, at most EXPERT_STORE_EXPERTS_MAX.
  int experts;
  // Experts a row chooses in a scored layer.
  int chosen;
  // Experts per stage, at most experts.
  int stage_experts;
  // Spans per expert, at most EXPERT_STORE_PARTS.
  int parts;
  // [layers][experts][parts].
  const FileSpan *span;
  // [layers]: spans per expert of the layer, 0 while unknown.
  const int *spans;
  // An optional transform, with its context. It runs after an expert's spans
  // are read, before the expert counts as read. It writes extra bytes at the
  // end of the slot, from the spans' bytes. Span k is at at[k].
  uint64_t extra;
  void (*transform)(void *context, int layer, int expert, uint8_t *slot,
                    const uint32_t *at, uint64_t slot_bytes);
  void *context;
} ExpertLayout;

// A store's counts.
typedef struct {
  // Requested experts found in memory and not found, read-aheads that a
  // request found and read-aheads started, bytes read, and evictions.
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions;
  // Read-ahead hits still being read.
  uint64_t late;
  // Read-aheads cancelled before reading.
  uint64_t cancelled;
  // Experts replaced by substitutes.
  uint64_t substituted;
  // Slots mapped for the GPU again.
  uint64_t remaps;
  // Slots given back when the store shrank for memory.
  uint64_t given_back;
  // Uses that raise decoding's eviction priority. A prompt's uses count none.
  uint64_t decode_uses;
  // The decoded rows' selections, each counted for its row, by where its
  // expert was: in memory, read ahead, read ahead and still being read, a
  // substitute, or only on the SSD.
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;
  // The experts that each small pass's rows chose, counted once per pass.
  uint64_t row_unique;
  // Seconds spent waiting for reads.
  double read_s;
} ExpertStoreCounts;

// Returns a new store for the layout, whose arrays must stay valid. The slots'
// buffers are made on gpu.
ExpertStore *expert_store_new(Gpu *gpu, const ExpertLayout *layout);

// Returns the store's counts.
ExpertStoreCounts *expert_store_counts(ExpertStore *st);

// The GiB kept for other programs: the default, and the least and the most
// that may be set.
static constexpr int EXPERT_STORE_RESERVE_GIB_DEFAULT = 10,
                     EXPERT_STORE_RESERVE_GIB_MIN = 2,
                     EXPERT_STORE_RESERVE_GIB_MAX = 1024;

// The process's memory settings. A field of 0 takes its default.
typedef struct {
  // GiB of model memory at most; 0: no limit, the free memory decides.
  double model_gib;
  // GiB kept available to other programs (EXPERT_STORE_RESERVE_GIB_DEFAULT).
  double reserve_gib;
  // How the memory follows the system's counts (memory.h).
  MemoryTimes times;
  // Seconds after macOS swaps memory out during which an expert cache grows
  // only into free memory (default 300).
  double free_pages_only_after_swapout;
  // GiB that must stay free above the reserve for an expert cache to grow
  // (default 4).
  double expert_cache_growth_margin_gib;
  // GiB kept unused below the GPU's memory limit, for slots that the GPU
  // driver still counts after they leave its list (default 3).
  double gpu_memory_limit_margin_gib;
} MemorySettings;

// The settings that a field of 0 takes. model_gib has none: 0 sets no limit.
static constexpr MemorySettings EXPERT_STORE_DEFAULT_SETTINGS = {
    .reserve_gib = EXPERT_STORE_RESERVE_GIB_DEFAULT,
    .times = MEMORY_DEFAULT_TIMES,
    .free_pages_only_after_swapout = 300,
    .expert_cache_growth_margin_gib = 4,
    .gpu_memory_limit_margin_gib = 3};

// Sets the memory's limit, reserve and times, and the expert caches'
// settings. The slot bytes that reads have not written yet count as used.
// A cache gives slots back, and unmaps them, at most once per
// freed_memory_delay, and grows its pool at most once per
// available_memory_reading_interval.
void expert_store_set_memory(Memory *memory, MemorySettings settings);

// Sets the GiB that every store leaves to other programs. 0 sets
// EXPERT_STORE_RESERVE_GIB_DEFAULT.
void expert_store_set_reserve(double gib);

// Sets the store's backbone, before the store is initialized. The slots are
// then counted in the backbone's account, and each depends on the backbone's
// slot. The store needs a backbone before it makes buffers.
void expert_store_set_backbone(ExpertStore *st, MemorySlot *backbone);

// Sets whether a request runs. While one runs, the memory does not evict the
// slots that the request may still use.
void expert_store_set_request_running(ExpertStore *st, int running);

// Sets how many passes besides the running one have GPU work. While others
// run, the slot limit does not fall, demand reads do not hold read-aheads
// back, and each pass may read ahead as many experts.
void expert_store_set_other_passes(ExpertStore *st, int passes);

// Sets the rows of the coming passes. The slot limits rise to hold two
// layers' selections of that many rows.
void expert_store_set_pass_rows(ExpertStore *st, int rows);

// Returns the bytes that the store needs at least: the slots of two layers'
// selections, and the stages' slots if the layers in use include them.
uint64_t expert_store_minimum_bytes(ExpertStore *st, int layers);

// Returns the bytes that the store needs at least for passes of rows, as
// expert_store_minimum_bytes returns them after expert_store_set_pass_rows,
// without changing the store's pass rows or limits.
uint64_t expert_store_pass_minimum_bytes(ExpertStore *st, int rows, int layers);

// Starts the store. It sets the slot limits within budget bytes, or, for 0,
// within the available memory less the reserve, and at least to the minimum
// slots. A budget below the minimum ends the program. It starts the readers,
// and reads the stages' experts. layers is the number of layers in use: the
// scored ones, or all of them with the stages.
void expert_store_start(ExpertStore *st, uint64_t budget, int layers);

// Returns the expert's buffer, mapped for the GPU, or null if the expert has no
// slot. Sets at[k] to where span k starts. Sets *fresh while the slot is not
// in the committed residency sets. The command buffer must then be told that
// its kernels read the buffer.
const Buf *expert_store_get_buffer(ExpertStore *st, int layer, int expert,
                                   uint32_t at[EXPERT_STORE_PARTS], int *fresh);

// Starts reading the experts that are not in memory. Returns how many of the n
// are being read.
int expert_store_start_reads(ExpertStore *st, int layer, const int *experts,
                             int n);

// Waits for the experts' reads. Inside a pass, it yields while it waits.
void expert_store_wait_for_reads(ExpertStore *st, int layer, const int *experts,
                                 int n);

// Starts the experts' reads, and waits for them.
void expert_store_read_and_wait(ExpertStore *st, int layer, const int *experts,
                                int n);

// Counts the experts as used without reading any. It is for experts that a
// pass reached through a table of their addresses. Experts without a slot are
// skipped.
void expert_store_count_use(ExpertStore *st, int layer, const int *experts,
                            int n);

// Starts the experts' reads, and keeps the experts from eviction until they are
// released. Returns how many of the n are being read.
int expert_store_hold(ExpertStore *st, int layer, const int *experts, int n);

// Releases a hold of the experts.
void expert_store_release(ExpertStore *st, int layer, const int *experts,
                          int n);

// Keeps every expert of the layer from eviction until the layer is released.
// It is for GPU work that reaches them through the layer's table. Holds add up.
void expert_store_hold_layer(ExpertStore *st, int layer);

// Releases one hold of the layer.
void expert_store_release_layer(ExpertStore *st, int layer);

// The number of the store's values that belong to each pass.
static constexpr int EXPERT_STORE_PASS_VARIABLES = 2;

// Writes the store's values that belong to each pass: the layer that a prompt
// pass is at, and the layer whose experts it requests.
void expert_store_write_pass_variables(
    ExpertStore *st, PassVariable out[EXPERT_STORE_PASS_VARIABLES]);

// Returns whether the expert has a slot: in memory, or being read.
int expert_store_has_slot(ExpertStore *st, int layer, int expert);

// Returns whether the expert is in memory and read.
int expert_store_is_ready(ExpertStore *st, int layer, int expert);

// Returns whether the GPU can reach every expert of the layer through
// addresses alone: each expert's read has finished, and its slot is in the
// committed residency sets. *generation changes whenever an expert of the
// layer moves to or leaves a slot. A table of the layer's addresses is rebuilt
// then.
int expert_store_layer_is_reachable(ExpertStore *st, int layer,
                                    uint64_t *generation);

// Where an expert is.
typedef enum {
  // Only on the SSD.
  EXPERT_ON_SSD,
  // In memory, read on demand.
  EXPERT_IN_MEMORY,
  // In memory, read ahead.
  EXPERT_READ_AHEAD,
  // Read ahead, and still being read.
  EXPERT_BEING_READ_AHEAD
} ExpertPlace;

// Returns where the expert is.
ExpertPlace expert_store_locate(ExpertStore *st, int layer, int expert);

// Starts reading the predicted experts that are not in memory, in their order.
// It stops when inflight read-aheads per running pass are reading.
void expert_store_read_ahead(ExpertStore *st, int layer, const int *experts,
                             int n, int inflight);

// Cancels the running pass's read-aheads for the layer that no reader has
// started, except those of the n experts in keep.
void expert_store_cancel_read_aheads(ExpertStore *st, int layer,
                                     const int *keep, int n);

// After the layer's routing, takes away the protection of the running pass's
// read-aheads for the layer. They are then evicted by their scores.
void expert_store_unprotect_read_aheads(ExpertStore *st, int layer);

// After a decoded layer's routing, credits its near misses: the experts that
// its router ranked just below the selected ones. They then stay a little
// longer. Layers after the scored ones are ignored.
void expert_store_credit_near_misses(ExpertStore *st, int layer,
                                     const int *experts, int n);

// Sets the layer that a prompt pass is at, or -1 while decoding. The pass
// still needs that layer's experts and the next layer's.
void expert_store_set_prompt_layer(ExpertStore *st, int layer);

// Ages the scores, and resizes the store. It is for the time between tokens.
void expert_store_age_and_resize(ExpertStore *st);

// Resizes the store in a prompt pass at the layer. It releases only slots of
// the layers before the previous one.
void expert_store_resize_in_prompt(ExpertStore *st, int layer);

// Resizes the store while no request runs, without aging the scores.
void expert_store_resize_when_idle(ExpertStore *st);

// Waits for every store's reads in flight.
void expert_store_wait_for_all_reads(void);

// These return the store's bytes, its slots, and its slots mapped for the GPU.
uint64_t expert_store_bytes(ExpertStore *st);
int expert_store_slot_count(ExpertStore *st);
int expert_store_mapped_count(ExpertStore *st);

// These return the slots that the store may hold now, and at most. When the
// layers differ in size, the limits are bytes. These then count the allocated
// slots, plus as many slots of the largest size as the remaining bytes hold.
int expert_store_slot_limit(ExpertStore *st);
int expert_store_slot_maximum(ExpertStore *st);

// Returns a slot's bytes, or the allocated slots' average when the layers
// differ in size.
uint64_t expert_store_slot_bytes(ExpertStore *st);

// Returns how many buffers were given back for one of another size.
uint64_t expert_store_resized_buffer_count(ExpertStore *st);

// Returns the bytes of the scored layers' experts without a slot, and sets
// *experts to their count. They are read while the store changes, so they may
// be off by a few experts.
uint64_t expert_store_bytes_without_slot(ExpertStore *st, int *experts);

// Sets the read times of every store since the start: the seconds with an
// expert read in flight, and the summed seconds of demand reads and of
// background reads.
void expert_store_read_times(double *busy, double *demand, double *background);

#endif
