#include "base/error.h"
#include "check.h"
#include "engine/tokenizer.h"
#include "load.h"
#include "model/model.h"
#include "platform/system_memory.h"
#include <math.h>
#include <stdlib.h>

// Tests prompts while many states are kept, the expert cache holds fewer slots
// than a chunk of rows chooses experts, and the available memory is short of
// the reserve. Before each pass, the model sets the least memory that the pass
// needs, its minimum. The kept states must not fill that minimum: otherwise
// the pass runs chunks whose experts the cache cannot hold. A new state whose
// memory does not fit is refused, and the program goes on.
//
// usage: model-prompt_pressure CHECKPOINT [GGUF]

// The tokens of each kept state.
static constexpr int KEPT_TOKENS = 64;
// The most states that the test keeps.
static constexpr int MOST_KEPT = 128;
// The slots that stay in the cache.
static constexpr int STAYING_SLOTS = 20;
// The tokens of the short prompt: eight rows, which the pass halves to four
// when their minimum does not fit.
static constexpr int SHORT_TOKENS = 8;
// The rows of the chunk whose minimum the kept states' bytes would cover.
static constexpr int COVERED_ROWS = 4;
static constexpr double GIB = 1024.0 * 1024 * 1024;

// Sets the reserve to bytes more than the available memory.
static void raise_reserve(Memory *memory, uint64_t bytes) {
  uint64_t available;
  memory_read_system(memory, &available);
  double gib = ((double)available + (double)bytes) / GIB;
  expert_store_set_memory(memory, (MemorySettings){.reserve_gib = gib});
}

// Whether every logit is finite.
static int all_finite(const float *logits, int vocab) {
  int finite = logits != NULL;
  for (int i = 0; finite && i < vocab; i++) finite = isfinite(logits[i]);
  return finite;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: model-prompt_pressure CHECKPOINT [GGUF]");
  Memory *memory = memory_new();
  expert_store_set_memory(memory, (MemorySettings){0});
  Gpu *gpu = open_gpu(memory);
  Options options = {.ctx = 4096,
                     .dump_fd = -1,
                     .routes_fd = -1,
                     .quiet = 1,
                     .deepseek.experts = argc == 3 ? argv[2] : NULL};
  Model *model = model_open(memory, gpu, argv[1], &options);
  check(model != NULL, "the model loads");
  const ModelOps *ops = model->ops;
  Tokenizer *tokenizer = tokenizer_open(argv[1], NULL);
  int line[64], n = tokenizer_encode_with_added(
                    tokenizer, "The river runs past the old mill. ", line, 64);
  int ids[KEPT_TOKENS];
  for (int i = 0; i < KEPT_TOKENS; i++) ids[i] = line[i % n];
  ModelCounters loaded;
  ops->read_counters(&loaded);
  // The cache's minimum is the slots of two layers' experts for a chunk's
  // rows. The kept states, beside the slots that stay, would cover the
  // minimum of a chunk of COVERED_ROWS rows, which needs more slots than stay.
  uint64_t layer_experts =
      (uint64_t)loaded.experts_per_row / loaded.routed_layers;
  uint64_t staying = STAYING_SLOTS * loaded.slot_bytes,
           covered = 2 * COVERED_ROWS * layer_experts * loaded.slot_bytes,
           state_bytes = ops->state_bytes(NULL, KEPT_TOKENS);
  int kept_count = (int)((covered - staying) / state_bytes) + 1;
  check(COVERED_ROWS * layer_experts > STAYING_SLOTS && kept_count <= MOST_KEPT,
        "a few kept states cover the minimum of a chunk that needs more slots "
        "than stay");
  if (kept_count > MOST_KEPT) kept_count = MOST_KEPT;
  int prefilled = 1;
  for (int k = 0; k < kept_count; k++)
    prefilled &= all_finite(ops->prefill(ops->state_new(), ids, KEPT_TOKENS, 0,
                                         0, -1, NULL, NULL, NULL),
                            ops->vocab_size());
  check(prefilled, "the kept states' prompts prefill");
  ModelCounters full, small;
  ops->read_counters(&full);
  ModelState *next = ops->state_new();
  check(next != NULL, "a new state fits while memory is free");
  // Between requests, the server gives memory back to keep the reserve.
  ops->set_request_running(0);
  memory_set_minimum(&model->memory, 0, MEMORY_REQUIRED);
  raise_reserve(memory, full.cache_bytes - staying);
  memory_reclaim(memory, full.cache_bytes - staying);
  ops->set_request_running(1);
  ops->read_counters(&small);
  printf("%d kept states of %.3f GiB; the cache held %d slots, then %d\n",
         kept_count, state_bytes / GIB, full.cache_slots, small.cache_slots);
  check(all_finite(
            ops->prefill(next, ids, SHORT_TOKENS, 0, 0, -1, NULL, NULL, NULL),
            ops->vocab_size()),
        "a short prompt prefills beside the kept states with few slots");
  // Above the available memory by all that the cache held, the reserve leaves
  // no room for another state.
  raise_reserve(memory, full.cache_bytes);
  check(ops->state_new() == NULL,
        "a new state that does not fit is refused, and the program goes on");
  // With the reserve back at its default, states and prompts run as before.
  expert_store_set_memory(memory, (MemorySettings){0});
  ModelState *after = ops->state_new();
  check(after && all_finite(ops->prefill(after, ids, SHORT_TOKENS, 0, 0, -1,
                                         NULL, NULL, NULL),
                            ops->vocab_size()),
        "a state and its prompt fit once the reserve is back");
  return failures != 0;
}
