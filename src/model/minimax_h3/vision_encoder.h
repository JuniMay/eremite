#ifndef EREMITE_MODEL_MINIMAX_H3_VISION_ENCODER_H
#define EREMITE_MODEL_MINIMAX_H3_VISION_ENCODER_H
#include "engine/memory.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/minimax_h3/text_encoder.h"
#include <stdint.h>

// MiniMax-H3 Qwen3-VL BF16 vision tower: 27 ViT layers (width 1152, 16 heads
// of 72 dims), bilinear 48x48 position embedding interpolation, 2D split-half
// RoPE, main patch merger and three DeepStack mergers (layers 8, 16, 24). Also
// provides reference image sizing helpers and multimodal prompt conditioning
// assembly for MiniMax-H3-Image edits.

// One 8-bit sRGB reference image passed for multimodal prompt encoding.
typedef struct {
  // Interleaved 8-bit sRGB pixels [height, width, 3].
  const uint8_t *rgb;
  // Image width and height in pixels.
  int width, height;
} MinimaxH3RefImage;

// Loaded MiniMax-H3 Qwen3-VL vision tower handle.
typedef struct MinimaxH3VisionEncoder MinimaxH3VisionEncoder;

// Computes the Qwen3-VL vision input size for a reference image: scales the
// short edge to 1024 rounded to 32 px, then enforces smart_resize bounds.
void minimax_h3_vision_ref_size(int width, int height, int *out_w, int *out_h);

// Computes the VAE encoder input size for a reference image: clamps the short
// edge into [1024, 2048] while keeping aspect ratio, rounded to 32 px.
void minimax_h3_vae_ref_size(int width, int height, int *out_w, int *out_h);

// The vision tower patches and prompt tokens of one reference image.
typedef struct {
  // 16 by 16 pixel patches that the vision tower reads.
  int patches;
  // Merged tokens that the image takes in the prompt, one per 2 by 2 patches.
  int tokens;
} MinimaxH3RefTokens;

// Returns the patches and tokens of a width by height reference image after
// its Qwen3-VL resize.
MinimaxH3RefTokens minimax_h3_vision_ref_tokens(int width, int height);

// Returns the GPU bytes of the vision tower's working buffers for one image of
// patches patches.
uint64_t minimax_h3_vision_encoder_workspace_bytes(int patches);

// Returns the token count of the multimodal prompt for prompt and ref_count
// reference images of refs[i].tokens tokens each: each image's label, vision
// tokens and two delimiters, then the prompt's text.
int minimax_h3_multimodal_prompt_tokens(const MinimaxH3TextEncoder *text_enc,
                                        const char *prompt,
                                        const MinimaxH3RefTokens *refs,
                                        int ref_count);

// Returns the most GPU bytes held at once while a multimodal prompt of seq_len
// tokens with ref_count reference images is encoded, its hidden rows included.
uint64_t minimax_h3_multimodal_prompt_peak_bytes(int seq_len,
                                                 const MinimaxH3RefTokens *refs,
                                                 int ref_count);

// Opens the Qwen3-VL vision tower sharing text_enc's open safetensors file and
// GPU device, registering its BF16 weights in account under a separate
// on-demand WeightGroup. Must be closed before text_enc is closed.
MinimaxH3VisionEncoder *
minimax_h3_vision_encoder_open(MinimaxH3TextEncoder *text_enc,
                               MemoryAccount *account);

// Frees the vision tower handle and its weight group.
void minimax_h3_vision_encoder_close(MinimaxH3VisionEncoder *vis);

// Returns the vision tower's weight group.
WeightGroup *minimax_h3_vision_encoder_weights(MinimaxH3VisionEncoder *vis);

// Returns the GPU bytes required to keep the vision tower weights resident.
uint64_t
minimax_h3_vision_encoder_required_bytes(const MinimaxH3VisionEncoder *vis);

// Returns the GPU bytes currently resident for the vision tower weights.
uint64_t
minimax_h3_vision_encoder_resident_bytes(const MinimaxH3VisionEncoder *vis);

// Loads the vision tower weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_vision_encoder_load(MinimaxH3VisionEncoder *vis);

// Encodes one pre-sized RGB image (width and height positive multiples of 32)
// through the 27-layer vision tower, main merger and 3 DeepStack mergers.
// Allocates out_merged and out_deepstack[0..2] (each n_merged by 5120 FP32) in
// out_slot and sets *out_grid_h = height / 16, *out_grid_w = width / 16.
// hooks may be null. Returns 1 on success or 0 on failure or stop.
int minimax_h3_vision_encoder_encode_image(
    MinimaxH3VisionEncoder *vis, const uint8_t *rgb, int width, int height,
    MemorySlot *out_slot, Buf *out_merged,
    Buf out_deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS], int *out_grid_h,
    int *out_grid_w, const MinimaxH3RunHooks *hooks);

// Encodes a MiniMax-H3-Image prompt with zero or more reference images:
// resizes each reference image to Qwen3-VL vision size, runs the vision tower,
// builds the "Picture 1: <|vision_start|>...<|vision_end|>...<prompt>" token
// sequence with MRoPE position IDs, DeepStack injections and AdaLN modality
// tags (0 on vision blocks including flanking start/end tokens, 1 on text),
// and runs Qwen3-VL layers 0..49.
// Allocates out_hidden in out_slot and a heap array in *out_tags (caller frees
// *out_tags with free() and out_hidden with gpu_free()).
// hooks may be null. Returns 1 on success or 0 on failure or stop.
int minimax_h3_encode_multimodal_prompt(
    MinimaxH3TextEncoder *text_enc, MinimaxH3VisionEncoder *vis,
    const char *prompt, const MinimaxH3RefImage *refs, int ref_count,
    MemorySlot *out_slot, Buf *out_hidden, int *out_seq_len, int **out_tags,
    const MinimaxH3RunHooks *hooks);

#endif
