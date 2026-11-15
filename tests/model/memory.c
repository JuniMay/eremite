#include "base/error.h"
#include "engine/sampling.h"
#include "engine/tokenizer.h"
#include "load.h"
#include "model/model.h"
#include "platform/system_memory.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests that a model's dense weights are evicted for another model and
// reloaded while a state stays in memory: the state's next logits are the
// same afterwards. It makes no claim about speed.
//
// usage: model-memory CHECKPOINT [MEMORY_GIB]

#define CHECK(test)                                                            \
  do {                                                                         \
    if (!(test)) die("memory model check failed: %s", #test);                  \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: model-memory CHECKPOINT [MEMORY_GIB]");
  Memory *memory = memory_new();
  memory_set_limit(memory, (uint64_t)((argc == 3 ? strtod(argv[2], NULL) : 24) *
                                      1073741824.0));
  Gpu *gpu = open_gpu(memory);
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(memory, gpu, argv[1], &options);
  CHECK(model && model->ops->dense_weight_slot);
  const ModelOps *ops = model->ops;
  ModelState *state = ops->state_new();
  CHECK(model->memory.state_bytes > 0);
  Tokenizer *tokenizer = tokenizer_open(argv[1], NULL);
  int tokens[128],
      n = tokenizer_encode_with_added(
          tokenizer, "Describe a blue triangle in one sentence.", tokens, 128);
  const float *prefill =
      ops->prefill(state, tokens, n, 0, 0, -1, NULL, NULL, NULL);
  CHECK(prefill != NULL);
  if (ops->read_last_plan) {
    ModelPlan plan;
    ops->read_last_plan(&plan);
    printf("prefill chunks: %d for %d input tokens\n", plan.chunks, n);
  }
  for (int i = 0; i < ops->vocab_size(); i++) CHECK(isfinite(prefill[i]));
  int token = sample_greedy(prefill, ops->vocab_size()), position = n;
  ModelSnapshot *snapshot = ops->snapshot_take(state, n, 0);
  const float *row = ops->decode_rows(&state, &token, &position, 1);
  size_t bytes = (size_t)ops->vocab_size() * sizeof(float);
  float *expected = xmalloc(bytes);
  memcpy(expected, row, bytes);
  ops->snapshot_restore(state, snapshot);
  ops->set_history(state, tokens, n);

  MemorySlot *weights = ops->dense_weight_slot();
  uint64_t state_bytes = model->memory.state_bytes,
           weight_bytes = weights->bytes;
  MemoryAccount competitor;
  MemorySlot competing_weights;
  memory_account_init(memory, &competitor, "synthetic competing model", 0);
  memory_slot_init(&competing_weights, &competitor, MEMORY_MODEL, NULL,
                   "dense weights", NULL, NULL);
  // What no eviction frees: other accounts' memory, the model's workspace and
  // its state.
  uint64_t fixed = memory_usage(memory).allocated - model->memory.bytes +
                   model->work.payload + model->memory.state_bytes;
  uint64_t amount = memory_usage(memory).limit - fixed;
  MemorySlot *request[] = {&competing_weights};
  CHECK(!memory_admit(request, &amount, 1, MEMORY_REQUIRED));
  CHECK(memory_ready(weights));
  model_set_request_running(model, 0);
  uint64_t footprint_before = system_memory_footprint();
  CHECK(memory_admit(request, &amount, 1, MEMORY_REQUIRED));
  CHECK(!memory_ready(weights) && weights->payload == 0);
  CHECK(model->memory.state_bytes == state_bytes);
  uint64_t footprint_after = system_memory_footprint();
  // Metal's physical-memory ledger catches up after buffer release: the test
  // reads it until it has.
  uint64_t retained = footprint_before - weight_bytes + (1ull << 30);
  while (footprint_after > retained)
    footprint_after = system_memory_footprint();
  printf("process footprint around eviction: %.3f -> %.3f GiB\n",
         footprint_before / 1073741824.0, footprint_after / 1073741824.0);
  CHECK(footprint_after <= retained);
  memory_slot_remove(&competing_weights);
  CHECK(model_load(model, &options));
  model_set_request_running(model, 1);
  CHECK(memory_ready(weights) && model->memory.state_bytes == state_bytes);
  row = ops->decode_rows(&state, &token, &position, 1);
  CHECK(row != NULL && !memcmp(row, expected, bytes));
  printf("weight reload: all %d logits identical; %.3f GiB weights evicted "
         "together; %.3f GiB state retained\n",
         ops->vocab_size(), weight_bytes / 1073741824.0,
         state_bytes / 1073741824.0);
  ops->snapshot_free(snapshot);
  ops->state_free(state);
  free(expected);
  tokenizer_close(tokenizer);
  return 0;
}
