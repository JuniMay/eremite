#ifndef EREMITE_TESTS_MODEL_LOAD_H
#define EREMITE_TESTS_MODEL_LOAD_H
#include "engine/expert_store.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/deepseek/model.h"

// Opens the memory, the GPU and the model for the model tests, as the command
// line does.

// The Memory object's eviction callback.
static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Returns a Gpu with the kernel library. After each eviction, the Memory
// object returns the freed buffers' memory to the system.
[[maybe_unused]] static Gpu *open_gpu(Memory *memory) {
  size_t length;
  const char *source = kernel_library_source(&length);
  Gpu *gpu = gpu_new(source, length);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  return gpu;
}

// Loads the DeepSeek model of dir with model_gib of model memory, keeping
// reserve_gib for other programs. Returns whether it loaded.
[[maybe_unused]] static int load_deepseek(const char *dir, const Options *o,
                                          double model_gib,
                                          double reserve_gib) {
  Memory *memory = memory_new();
  expert_store_set_memory(memory, (MemorySettings){.model_gib = model_gib,
                                                   .reserve_gib = reserve_gib});
  Gpu *gpu = open_gpu(memory);
  static MemoryAccount account;
  static MemorySlot work;
  memory_account_init(memory, &account, dir, UINT64_MAX);
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "model workspace", NULL,
                   NULL);
  work.automatic = 1;
  return deepseek_load(gpu, &work, dir, o);
}

#endif
