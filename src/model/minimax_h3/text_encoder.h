#ifndef EREMITE_MODEL_MINIMAX_H3_TEXT_ENCODER_H
#define EREMITE_MODEL_MINIMAX_H3_TEXT_ENCODER_H
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <stdint.h>

// MiniMax-H3's shared text encoder: Qwen3-VL-32B truncated to layers 0 through
// 49, using Comfy-Org NVFP4 + AWQ weights and row-scaled INT8 embeddings.
// Prompts are tokenized as raw text without a chat template; empty prompts fall
// back to the pad token. Output is the unnormalized hidden state after
// layer 49.

// Width of each output hidden-state vector.
static constexpr int MINIMAX_H3_TEXT_DIM = 5120;

// Number of Qwen3-VL transformer layers executed for MiniMax-H3 conditioning.
static constexpr int MINIMAX_H3_TEXT_LAYERS = 50;

// Number of DeepStack visual feature tensors injected after layers 0, 1 and 2.
static constexpr int MINIMAX_H3_TEXT_DEEPSTACK_LAYERS = 3;

// Token IDs used by MiniMax-H3's Qwen3-VL prompt presentation.
static constexpr int MINIMAX_H3_TEXT_PAD_TOKEN = 151643,
                     MINIMAX_H3_TEXT_VISION_START_TOKEN = 151652,
                     MINIMAX_H3_TEXT_VISION_END_TOKEN = 151653;

// Loaded MiniMax-H3 Qwen3-VL-32B text encoder handle.
typedef struct MinimaxH3TextEncoder MinimaxH3TextEncoder;

// Opens the Qwen3-VL-32B NVFP4 safetensors checkpoint at weights_path and the
// tokenizer in tokenizer_dir. Does not load weight buffers until requested.
MinimaxH3TextEncoder *minimax_h3_text_encoder_open(Gpu *gpu,
                                                   MemoryAccount *account,
                                                   const char *weights_path,
                                                   const char *tokenizer_dir);

// Frees the encoder, its weight group, tokenizer and open file handle.
void minimax_h3_text_encoder_close(MinimaxH3TextEncoder *enc);

// Returns the open safetensors index shared with the Qwen3-VL vision tower.
const TensorIndex *
minimax_h3_text_encoder_tensors(const MinimaxH3TextEncoder *enc);

// Returns the encoder's GPU handle.
Gpu *minimax_h3_text_encoder_gpu(const MinimaxH3TextEncoder *enc);

// Returns the encoder's weight group so dependent groups can share its account.
WeightGroup *minimax_h3_text_encoder_weights(MinimaxH3TextEncoder *enc);

// Returns the GPU bytes required to keep the text encoder weights resident.
uint64_t
minimax_h3_text_encoder_required_bytes(const MinimaxH3TextEncoder *enc);

// Returns the GPU bytes currently resident for the text encoder weights.
uint64_t
minimax_h3_text_encoder_resident_bytes(const MinimaxH3TextEncoder *enc);

// Loads the text encoder weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_text_encoder_load(MinimaxH3TextEncoder *enc);

// Encodes raw prompt text into token IDs (no special-token parsing; empty
// prompt produces a single pad token). Returns the token count.
int minimax_h3_text_encoder_tokenize(const MinimaxH3TextEncoder *enc,
                                     const char *prompt, int *ids, int max_ids);

// Runs layers 0 through 49 on seq_len tokens and writes unnormalized FP32
// hidden states (seq_len by MINIMAX_H3_TEXT_DIM) into out_hidden.
// - ids: seq_len token IDs; entries < 0 keep rows pre-filled in
// embeds_override.
// - pos_ids: optional seq_len by 3 MRoPE (t, h, w) coordinates; when null,
//   position i receives (i, i, i).
// - embeds_override: optional FP32 buffer of seq_len by MINIMAX_H3_TEXT_DIM
//   copied into out_hidden before embedding lookup for ids >= 0.
// - deepstack: optional array of 3 FP32 buffers (visual_count by 5120) added at
//   visual_indices after layers 0, 1 and 2.
// Returns 1 on success or 0 if weight loading or workspace allocation fails.
int minimax_h3_text_encoder_encode_tokens(
    MinimaxH3TextEncoder *enc, const int *ids, const int *pos_ids,
    const Buf *embeds_override,
    const Buf *const deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS],
    const int *visual_indices, int visual_count, int seq_len, Buf *out_hidden);

// Convenience wrapper for text-only prompts: tokenizes prompt, allocates
// out_hidden in out_slot, runs the text encoder, and sets *out_seq_len.
// Returns 1 on success or 0 on failure.
int minimax_h3_text_encoder_encode_prompt(MinimaxH3TextEncoder *enc,
                                          const char *prompt,
                                          MemorySlot *out_slot, Buf *out_hidden,
                                          int *out_seq_len);

#endif
