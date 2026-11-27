#include "check.h"
#include "model/inkling2/model.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Unit tests for Inkling-2-Mini's 48-visit unrolled schedule, CPU expert tile
// grouping, chat turn formatting, and static ModelOps metadata (no GPU or
// checkpoint required).

// Deterministic test PRNG constants (xorshift32).
static constexpr uint32_t RNG_SEED = 0x9e3779b9u;
static constexpr uint32_t SHIFT_A = 13, SHIFT_B = 17, SHIFT_C = 5;

// Grouping test dimensions: 20 tokens x 8 experts with tile limit 4.
static constexpr int GROUP_ROWS = 20;
static constexpr int GROUP_TILE_LIMIT = 4;
static constexpr int GROUP_EXPERT_MOD = 6;
static constexpr size_t CHAT_BUF_CAP = 512;

// Advances xorshift32 state.
static uint32_t rng_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << SHIFT_A;
  x ^= x >> SHIFT_B;
  x ^= x << SHIFT_C;
  *state = x;
  return x;
}

// Verifies the 48-visit looped schedule geometry and cache slot numbering.
static void test_schedule(void) {
  Ink2Visit s[INK2_TOTAL_LAYERS];
  inkling2_build_schedule(s);

  int global_seen = 0, slide_seen = 0;
  int boundaries = 0, loop_msg_visits = 0;
  for (int v = 0; v < INK2_TOTAL_LAYERS; v++) {
    if (s[v].is_global) {
      check(s[v].global_slot == global_seen && s[v].slide_slot == -1,
            "global visit gets consecutive global_slot and no slide_slot");
      global_seen++;
    } else {
      check(s[v].slide_slot == slide_seen && s[v].global_slot == -1,
            "sliding visit gets consecutive slide_slot and no global_slot");
      slide_seen++;
    }
    if (s[v].is_boundary) boundaries++;
    if (s[v].loop_pass > 0) loop_msg_visits++;
  }
  check(global_seen == INK2_GLOBAL_COUNT, "8 global logical visits");
  check(slide_seen == INK2_SLIDE_COUNT, "40 sliding logical visits");
  check(boundaries == 8, "8 block boundaries across 48 visits");
  check(loop_msg_visits == 2 * INK2_CORE_LAYERS,
        "24 visits receive cross-pass messages");
  check(s[0].phys == 0 && s[5].phys == 5 && s[5].is_global,
        "prelude spans physical layers 0..5 with global layer 5");
  check(s[6].phys == 6 && s[6].loop_pass == 0 && s[6].loop_idx == 0,
        "visit 6 starts core pass 0");
  check(s[18].phys == 6 && s[18].loop_pass == 1 && s[18].loop_idx == 0,
        "visit 18 starts core pass 1");
  check(s[30].phys == 6 && s[30].loop_pass == 2 && s[30].loop_idx == 0,
        "visit 30 starts core pass 2");
  check(s[42].phys == 18 && s[42].loop_idx == -1 && s[47].phys == 23 &&
            s[47].is_global,
        "postlude spans physical layers 18..23 with global layer 23");
}

// Verifies expert tile grouping splits large experts and preserves inverse
// token-slot mappings.
static void test_tile_grouping(void) {
  uint32_t rng = RNG_SEED;
  int npairs = GROUP_ROWS * INK2_TOPK;
  uint32_t ids[GROUP_ROWS * INK2_TOPK];
  float w[GROUP_ROWS * INK2_TOPK];
  for (int r = 0; r < GROUP_ROWS; r++) {
    for (int k = 0; k < INK2_TOPK; k++) {
      ids[r * INK2_TOPK + k] =
          (uint32_t)((r + k * 3 + (rng_next(&rng) % GROUP_EXPERT_MOD)) %
                     INK2_EXPERTS);
      w[r * INK2_TOPK + k] = (float)(k + 1) * 0.1f;
    }
  }
  int ptok[GROUP_ROWS * INK2_TOPK], tp[GROUP_ROWS * INK2_TOPK];
  float pw[GROUP_ROWS * INK2_TOPK];
  Ink2MoeTile tiles[GROUP_ROWS * INK2_TOPK + INK2_EXPERTS];
  int ntiles = inkling2_group_expert_tiles(ids, w, GROUP_ROWS, GROUP_TILE_LIMIT,
                                           ptok, pw, tp, tiles);
  check(ntiles > 0, "grouping emits tiles");

  int covered = 0, bounds_ok = 1;
  uint32_t pair_expert[GROUP_ROWS * INK2_TOPK];
  for (int t = 0; t < ntiles; t++) {
    if (tiles[t].count == 0 || tiles[t].count > (uint32_t)GROUP_TILE_LIMIT)
      bounds_ok = 0;
    if (tiles[t].start != (uint32_t)covered) bounds_ok = 0;
    for (uint32_t j = 0; j < tiles[t].count; j++)
      pair_expert[tiles[t].start + j] = tiles[t].expert;
    covered += (int)tiles[t].count;
  }
  check(bounds_ok && covered == npairs,
        "tiles partition all pairs with counts within max_tile");

  int map_ok = 1;
  for (int p = 0; p < npairs; p++) {
    int slot = tp[p];
    if (slot < 0 || slot >= npairs) {
      map_ok = 0;
      break;
    }
    if (pair_expert[slot] != ids[p] || ptok[slot] != p / INK2_TOPK ||
        pw[slot] != w[p])
      map_ok = 0;
  }
  check(map_ok, "inverse tp map matches expert id, token row and weight");
}

// Verifies CLI chat turn formatting and static ModelOps properties.
static void test_chat_and_ops(void) {
  char buf[CHAT_BUF_CAP];
  inkling2_ops.render_chat_turn(buf, sizeof buf, "Hello", 0, 0);
  check(!strcmp(buf,
                "<|message_system|><|content_text|>Thinking effort level: 0"
                "<|end_message|><|message_user|><|content_text|>Hello"
                "<|end_message|>"),
        "render_chat_turn formats think=0 matching chat_template.jinja");

  inkling2_ops.render_chat_turn(buf, sizeof buf, "Look", 1, 90);
  check(!strcmp(buf,
                "<|message_system|><|content_text|>Thinking effort level: 0.9"
                "<|end_message|><|message_user|><|content_image|>"
                "<|unused_200054|><|end_message|><|message_user|>"
                "<|content_text|>Look<|end_message|>"),
        "render_chat_turn formats think=90 with image marker");

  inkling2_ops.render_chat_turn(buf, sizeof buf, "Max", 0, 100);
  check(strstr(buf, "Thinking effort level: 0.99<|end_message|>") != NULL,
        "render_chat_turn clamps think=100 to 0.99");

  check(inkling2_ops.vocab_size() == INK2_VOCAB, "vocab_size returns 200058");
  check(inkling2_ops.replay_rows(128) == -1, "replay_rows indicates snapshot");
  check(!strcmp(inkling2_ops.model_types[0], "inkling2_loop") &&
            !strcmp(inkling2_ops.model_types[1], "inkling2_loop_text"),
        "model_types registers inkling2_loop and inkling2_loop_text");
  check(inkling2_ops.set_request_running != NULL &&
            inkling2_ops.expert_store_bytes != NULL &&
            inkling2_ops.read_routes != NULL &&
            inkling2_ops.set_other_passes != NULL,
        "ModelOps exposes expert-store lifecycle, route and counter hooks");
  check(inkling2_ops.expert_store_bytes() == 0,
        "expert_store_bytes is 0 before load");
  ModelCounters c;
  inkling2_ops.read_counters(&c);
  check(c.layers == INK2_TOTAL_LAYERS && c.routed_layers == INK2_TOTAL_LAYERS &&
            c.experts_per_layer == INK2_EXPERTS &&
            c.experts_per_row == INK2_TOTAL_LAYERS * INK2_TOPK &&
            c.experts == INK2_PHYS_LAYERS * INK2_EXPERTS && c.dense_bytes == 0,
        "read_counters reports 48 logical routed visits and 3072 experts");
  check(INK2_BF16_KV_ROW_BYTES == (size_t)INK2_KV_DIM * 2 &&
            INK2_FP8_KV_SCALES_OFF == (size_t)INK2_KV_DIM &&
            INK2_FP8_KV_ROW_BYTES ==
                (size_t)INK2_KV_DIM + (size_t)INK2_KV_HEADS * 2 &&
            INK2_FP8_KV_ROW_BYTES % 16 == 0,
        "FP8 KV row geometry is 1040 bytes and 16-byte aligned");
}

int main(void) {
  test_schedule();
  test_tile_grouping();
  test_chat_and_ops();
  return failures != 0;
}
