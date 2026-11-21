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

// Computes Qwen3-VL smart_resize dimensions for an image or video frame:
// rounds both sides to 32 px and clamps area into [3136, 12845056] pixels.
void minimax_h3_vision_smart_size(int width, int height, int *out_w,
                                  int *out_h);

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

// Encodes a pair of pre-sized RGB video frames (width and height positive
// multiples of 32) as one 2-frame temporal patch through the vision tower,
// main merger and 3 DeepStack mergers. Allocates out_merged and
// out_deepstack[0..2] in out_slot and sets *out_grid_h = height / 16,
// *out_grid_w = width / 16. hooks may be null. Returns 1 on success or 0 on
// failure or stop.
int minimax_h3_vision_encoder_encode_video_block(
    MinimaxH3VisionEncoder *vis, const uint8_t *rgb0, const uint8_t *rgb1,
    int width, int height, MemorySlot *out_slot, Buf *out_merged,
    Buf out_deepstack[MINIMAX_H3_TEXT_DEEPSTACK_LAYERS], int *out_grid_h,
    int *out_grid_w, const MinimaxH3RunHooks *hooks);

// Kind of one conditioning item in a MiniMax-H3 video prompt presentation.
typedef enum {
  // Single RGB image labeled "<Picture i>: " followed by one vision block.
  MINIMAX_H3_VIDEO_COND_IMAGE,
  // Audio reference labeled "<Audio j>: " with no vision tokens.
  MINIMAX_H3_VIDEO_COND_AUDIO,
  // Video clip sampled at 2 fps labeled "<Video k>: " plus per-pair
  // "<T.T seconds>" timestamps and 2-frame vision blocks.
  MINIMAX_H3_VIDEO_COND_VIDEO
} MinimaxH3VideoCondKind;

// One conditioning item passed to MiniMax-H3 video prompt planning or encoding.
typedef struct {
  // Conditioning modality kind.
  MinimaxH3VideoCondKind kind;
  // Packed 8-bit sRGB frames [frame_count, height, width, 3] for image or
  // video items; may be null during planning or for audio items.
  const uint8_t *rgb;
  // Frame width and height in pixels for image or video items; 0 for audio.
  int width, height;
  // Frame count (1 for image, >= 1 for 2 fps video, 0 for audio).
  int frame_count;
  // Optional per-frame timestamps in seconds for video items; when null,
  // frame i uses i / 2.0 seconds.
  const double *timestamps;
} MinimaxH3VideoCondItem;

// Token, tag, MRoPE position and peak GPU memory plan for a MiniMax-H3 video
// prompt presentation.
typedef struct {
  // Token IDs [seq_len]; -1 marks merged visual embedding slots.
  int *ids;
  // AdaLN modality tags [seq_len]: 0 on vision blocks (including delimiters),
  // 1 on text and labels.
  int *tags;
  // Qwen3-VL MRoPE coordinates [seq_len * 3] as (t, h, w) triples.
  int *pos_ids;
  // Sequence indices [visual_tokens] of all merged visual embedding slots.
  int *vis_indices;
  // Start index in ids of each vision block [vision_blocks].
  int *block_starts;
  // Merged visual token count of each vision block [vision_blocks].
  int *block_merged;
  // Patch grid height and width of each vision block [vision_blocks].
  int *block_grid_h, *block_grid_w;
  // Total token count passed to Qwen3-VL.
  int seq_len;
  // Total vision blocks (one per image and one per 2-frame video pair).
  int vision_blocks;
  // Sum of merged visual tokens across all vision blocks.
  int visual_tokens;
  // Peak GPU bytes held while encoding all vision blocks and text layers.
  uint64_t peak_bytes;
} MinimaxH3VideoPromptLayout;

// Frees all heap arrays owned by layout and resets its fields to zero.
void minimax_h3_video_prompt_layout_free(MinimaxH3VideoPromptLayout *layout);

// Builds the token, tag, MRoPE position and peak-memory layout for a
// MiniMax-H3 video prompt (T2VA, FL2VA or REF2VA) without touching the GPU.
void minimax_h3_video_prompt_build_layout(Tokenizer *tokenizer,
                                          const char *prompt,
                                          const MinimaxH3VideoCondItem *items,
                                          int item_count,
                                          MinimaxH3VideoPromptLayout *out);

// Encodes a MiniMax-H3 video prompt (T2VA, FL2VA or REF2VA) with zero or more
// image, audio and 2 fps video conditioning items through Qwen3-VL layers
// 0..49. Allocates out_hidden in out_slot and *out_tags on the heap (caller
// frees *out_tags with free() and out_hidden with gpu_free()). hooks may be
// null. Returns 1 on success or 0 on failure or stop.
int minimax_h3_encode_video_prompt(
    MinimaxH3TextEncoder *text_enc, MinimaxH3VisionEncoder *vis,
    const char *prompt, const MinimaxH3VideoCondItem *items, int item_count,
    MemorySlot *out_slot, Buf *out_hidden, int *out_seq_len, int **out_tags,
    const MinimaxH3RunHooks *hooks);

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
