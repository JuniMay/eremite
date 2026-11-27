#include "check.h"
#include "engine/engine.c"
#include <stdint.h>
#include <stdio.h>

// Tests that per-model memory statistics split weights, experts, workspace and
// states canonically without double-counting state bytes in model_workspace.

// Sizes used to exercise each memory category with distinct values.
static constexpr uint64_t WEIGHTS_BYTES = 10ull << 20;
static constexpr uint64_t EXPERTS_BYTES = 20ull << 20;
static constexpr uint64_t WORKSPACE_BYTES = 3ull << 20;
static constexpr uint64_t INITIAL_STATE_BYTES = 16ull << 20;
static constexpr uint64_t GROWN_STATE_BYTES = 48ull << 20;

// Mock model counters returned to eremite_read_stats.
static uint64_t mock_dense_bytes, mock_cache_bytes;

// Populates mock model counters for eremite_read_stats.
static void mock_read_counters(ModelCounters *c) {
  *c = (ModelCounters){.dense_bytes = mock_dense_bytes,
                       .cache_bytes = mock_cache_bytes};
}

// Mock model operations exposing counter readings.
static const ModelOps mock_ops = {.read_counters = mock_read_counters};

int main(void) {
  memory = memory_new();
  Model model = {.ops = &mock_ops};
  memory_account_init(memory, &model.memory, "mock", 0);
  memory_slot_init(&model.work, &model.memory, MEMORY_MODEL, NULL,
                   "model workspace", NULL, NULL);
  model.work.automatic = 1;

  MemorySlot weights, experts, states;
  memory_slot_init(&weights, &model.memory, MEMORY_MODEL, NULL, "weights", NULL,
                   NULL);
  memory_slot_set_role(&weights, MEMORY_ROLE_WEIGHTS);
  memory_slot_init(&experts, &model.memory, MEMORY_MODEL, &weights,
                   "routed expert", NULL, NULL);
  memory_slot_set_role(&experts, MEMORY_ROLE_EXPERTS);
  memory_slot_init(&states, &model.memory, MEMORY_STATE, NULL, "states", NULL,
                   NULL);
  states.automatic = 1;

  check(memory_claim(&weights, WEIGHTS_BYTES, MEMORY_REQUIRED),
        "weights slot claims bytes");
  check(memory_claim(&experts, EXPERTS_BYTES, MEMORY_REQUIRED),
        "experts slot claims bytes");
  check(memory_claim(&model.work, WORKSPACE_BYTES, MEMORY_REQUIRED),
        "workspace slot claims bytes");
  check(memory_claim(&states, INITIAL_STATE_BYTES, MEMORY_REQUIRED),
        "state slot claims initial bytes");

  mock_dense_bytes = WEIGHTS_BYTES - 4096;
  mock_cache_bytes = EXPERTS_BYTES - 8192;

  eremite_model em = {.m = &model, .ops = &mock_ops};
  atomic_init(&em.loaded, 1);
  models[0] = &em;
  nmodels = 1;

  eremite_stats s = {0};
  eremite_read_stats(&em, &s);
  MemoryAccountBreakdown b = memory_account_breakdown(&model.memory);
  check(b.weights == WEIGHTS_BYTES && b.experts == EXPERTS_BYTES &&
            b.workspace == WORKSPACE_BYTES && b.states == INITIAL_STATE_BYTES &&
            b.total == model.memory.bytes,
        "memory_account_breakdown reports exact role totals");
  check(s.dense_bytes == b.weights, "dense_bytes matches canonical weights");
  check(s.cache_bytes == b.experts, "cache_bytes matches canonical experts");
  check(s.own_state_bytes == b.states,
        "own_state_bytes matches canonical states");
  check(s.model_workspace == WORKSPACE_BYTES,
        "model_workspace excludes state bytes");
  check(s.dense_bytes + s.cache_bytes + s.model_workspace + s.own_state_bytes ==
            model.memory.bytes,
        "category bytes sum to account total");

  check(memory_claim(&states, GROWN_STATE_BYTES - INITIAL_STATE_BYTES,
                     MEMORY_REQUIRED),
        "state slot grows");
  eremite_read_stats(&em, &s);
  check(s.own_state_bytes == GROWN_STATE_BYTES,
        "own_state_bytes tracks state growth");
  check(s.model_workspace == WORKSPACE_BYTES,
        "model_workspace stays unchanged when state grows");
  check(s.dense_bytes + s.cache_bytes + s.model_workspace + s.own_state_bytes ==
            model.memory.bytes,
        "category bytes still sum to account total after state growth");

  return failures != 0;
}
