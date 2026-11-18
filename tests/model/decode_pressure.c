#include "base/error.h"
#include "check.h"
#include "engine/sampling.h"
#include "engine/tokenizer.h"
#include "load.h"
#include "model/model.h"
#include "platform/system_memory.h"
#include <math.h>

// Tests that a decoding step that finds the available memory short of the
// reserve gives expert slots back and goes on, instead of ending the program,
// also when the last step used every slot. Four rows decode a few tokens.
// Then the reserve rises, so that the cache keeps only a few layers' experts,
// and a step uses each of its slots. Then the reserve rises above the
// available memory again: the next step's workspace must come from a slot that
// the last step used.
//
// usage: model-decode_pressure CHECKPOINT [GGUF]

static constexpr int ROWS = 4;
// The decoding steps before the reserve rises.
static constexpr int STEPS = 8;
static constexpr double GIB = 1024.0 * 1024 * 1024;

// Decodes one step of the rows, and returns its logits.
static const float *decode_step(const ModelOps *ops, ModelState **states,
                                int *tokens, int *positions) {
  const float *rows = ops->decode_rows(states, tokens, positions, ROWS);
  for (int r = 0; rows && r < ROWS; r++) {
    tokens[r] =
        sample_greedy(rows + (size_t)r * ops->vocab_size(), ops->vocab_size());
    positions[r]++;
  }
  return rows;
}

// Sets the reserve to bytes more than the available memory.
static void raise_reserve(Memory *memory, uint64_t bytes) {
  uint64_t available;
  memory_read_system(memory, &available);
  double gib = ((double)available + (double)bytes) / GIB;
  expert_store_set_memory(memory, (MemorySettings){.reserve_gib = gib});
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: model-decode_pressure CHECKPOINT [GGUF]");
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
  ModelState *states[ROWS];
  int tokens[ROWS], positions[ROWS];
  for (int r = 0; r < ROWS; r++) {
    char prompt[64];
    snprintf(prompt, sizeof prompt, "[%d] Count from 1 to 100.", r);
    int ids[64], n = tokenizer_encode_with_added(tokenizer, prompt, ids, 64);
    states[r] = ops->state_new();
    const float *logits =
        ops->prefill(states[r], ids, n, 0, 0, -1, NULL, NULL, NULL);
    tokens[r] = sample_greedy(logits, ops->vocab_size()), positions[r] = n;
  }
  for (int step = 0; step < STEPS; step++)
    decode_step(ops, states, tokens, positions);
  ModelCounters full, small, last;
  ops->read_counters(&full);
  // The cache's minimum is the slots of two layers' experts for the step's
  // rows. Three times that many slots stay, more than the minimum and fewer
  // than one step uses.
  uint64_t layer_experts = (uint64_t)full.experts_per_row / full.routed_layers;
  uint64_t staying = 3 * 2 * ROWS * layer_experts * full.slot_bytes;
  check(full.cache_bytes > staying, "the cache holds more than stays");
  raise_reserve(memory, full.cache_bytes - staying);
  check(decode_step(ops, states, tokens, positions) != NULL,
        "a step decodes while the cache gives slots back");
  ops->read_counters(&small);
  // Above the available memory by all that the cache held, the reserve stays
  // out of reach, also while freed memory is still coming back.
  raise_reserve(memory, full.cache_bytes);
  const float *rows = decode_step(ops, states, tokens, positions);
  int finite = rows != NULL;
  for (int i = 0; finite && i < ROWS * ops->vocab_size(); i++)
    finite = isfinite(rows[i]);
  check(finite, "a step after one that used every slot decodes");
  ops->read_counters(&last);
  printf("the cache held %.2f GiB in %d slots, then %.2f GiB in %d, then "
         "%.2f GiB in %d\n",
         full.cache_bytes / GIB, full.cache_slots, small.cache_bytes / GIB,
         small.cache_slots, last.cache_bytes / GIB, last.cache_slots);
  check(small.cache_bytes < full.cache_bytes, "the cache gave slots back");
  return failures != 0;
}
