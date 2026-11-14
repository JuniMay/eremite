#include "base/error.h"
#include "check.h"
#include "load.h"

// Tests that a model load that finds no room fails without ending the
// program, and that the model loads once there is room. The largest reserve
// leaves every byte to other programs, so no allocation fits.
//
// usage: model-load_room CHECKPOINT [GGUF]

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: model-load_room CHECKPOINT [GGUF]");
  Options o = {.ctx = 4096,
               .dump_fd = -1,
               .routes_fd = -1,
               .quiet = 1,
               .deepseek.experts = argc == 3 ? argv[2] : NULL};
  Memory *memory = memory_new();
  expert_store_set_memory(
      memory, (MemorySettings){.model_gib = 0,
                               .reserve_gib = EXPERT_STORE_RESERVE_GIB_MAX});
  Gpu *gpu = open_gpu(memory);
  static MemoryAccount account;
  static MemorySlot work;
  memory_account_init(memory, &account, argv[1], UINT64_MAX);
  memory_slot_init(&work, &account, MEMORY_MODEL, NULL, "model workspace", NULL,
                   NULL);
  work.automatic = 1;
  check(!deepseek_load(gpu, &work, argv[1], &o),
        "a load that finds no room fails, and the program goes on");
  expert_store_set_memory(memory,
                          (MemorySettings){.model_gib = 0, .reserve_gib = 0});
  check(deepseek_load(gpu, &work, argv[1], &o),
        "the model loads once there is room");
  return failures != 0;
}
