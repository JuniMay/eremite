#ifndef EREMITE_MODEL_LFM3_MODEL_H
#define EREMITE_MODEL_LFM3_MODEL_H
#include "model/model.h"

// LFM3's language model, as LiquidAI's d1 decision models hold it, and its
// decision pass: one sequence holds a state and the questions after it, and
// the pass returns the logits of chosen tokens at chosen rows.

// The language model's shape (text_config in d1-2B's config.json).
static constexpr int LFM3_DIM = 2048, LFM3_LAYERS = 24, LFM3_VOCAB = 128000,
                     LFM3_HIDDEN = 7680, LFM3_HEADS = 16, LFM3_KV_HEADS = 4,
                     LFM3_HEAD_DIM = 128, LFM3_CONV_WIDTH = 7,
                     LFM3_WINDOW = 256, LFM3_POSITIONS = 128000;

// The epsilon of every RMSNorm (norm_eps).
static constexpr float LFM3_NORM_EPS = 1e-5f;

#endif
