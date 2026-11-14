#ifndef EREMITE_TESTS_UNIT_STORE_LAYOUT_H
#define EREMITE_TESTS_UNIT_STORE_LAYOUT_H
#include "engine/expert_store.h"

// The expert store's tests include its source and create a store with the
// layout of DeepSeek V4.1 Flash: 40 scored layers of 384 experts, of which a
// row chooses 6, then three stages of 128 experts. An expert has up to three
// spans.

enum {
  LAYERS = 40,
  EXPERTS = 384,
  TOPK = 6,
  DRAFTS = 3,
  DRAFT_EXPERTS = 128,
  CACHE_LAYERS = LAYERS + DRAFTS
};

#define EXPERT_W (3u * 5898240u)  // an MXFP4 expert's codes
#define EXPERT_S (3u * 368640u)  // and their scales

static FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
static int expert_spans[CACHE_LAYERS];
static ExpertStore *st;

static ExpertStore *store_create(void) {
  return expert_store_new(NULL, &(ExpertLayout){.layers = CACHE_LAYERS,
                                                .scored_layers = LAYERS,
                                                .experts = EXPERTS,
                                                .chosen = TOPK,
                                                .stage_experts = DRAFT_EXPERTS,
                                                .parts = 3,
                                                .span = &expert_span[0][0][0],
                                                .spans = expert_spans});
}

#endif
