#ifndef EREMITE_MODEL_MINIMAX_H3_VIDEO_MODEL_H
#define EREMITE_MODEL_MINIMAX_H3_VIDEO_MODEL_H
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/model.h"
#include "model/model.h"
#include <stddef.h>
#include <stdint.h>

// MiniMax-H3 Video model integration: wraps the Qwen3-VL-32B NVFP4 text and
// vision encoders, pruned INT8-ConvRot 50-layer FL2VA/REF2VA DiT (with
// optional BF16 Turbo LoRA), INT8-ConvRot 3D Video VAE, FP32 Stereo Audio VAE
// and native macOS H.264/AAC MP4 muxer behind ModelOps.

// Spatial multiple in pixels required by the Video VAE (16x) and DiT (2x2
// patch); Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:936.
static constexpr int MINIMAX_H3_VIDEO_PATCH_PX = 32;

// Base short edge and default 16:9 768p canvas dimensions in pixels;
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:937-938,1024-1025.
static constexpr int MINIMAX_H3_VIDEO_BASE_SHORT_EDGE = 768,
                     MINIMAX_H3_VIDEO_DEFAULT_WIDTH = 1344,
                     MINIMAX_H3_VIDEO_DEFAULT_HEIGHT = 768;

// Default auto-adapted canvas pixel area (768 * 1344);
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:938.
static constexpr int64_t MINIMAX_H3_VIDEO_ADAPT_MAX_PIXELS = 768ll * 1344ll;

// Maximum single-axis output dimension and maximum explicit canvas area (up to
// 2048x1152 = 2.36 MP); MiniMaxAI/MiniMax-H3 README.md:56.
static constexpr int MINIMAX_H3_VIDEO_MAX_DIM = 2048;
static constexpr int64_t MINIMAX_H3_VIDEO_MAX_CANVAS_AREA = 2048ll * 1152ll;

// Minimum, default (~5 s) and maximum (~15 s) output frame counts on the 17k+5
// grid at 24 fps; MiniMaxAI/MiniMax-H3 README.md:54 and Comfy-Org/ComfyUI
// comfy_extras/nodes_minimax_h3.py:1026.
static constexpr int MINIMAX_H3_VIDEO_MIN_FRAMES = 5,
                     MINIMAX_H3_VIDEO_DEFAULT_FRAMES = 124,
                     MINIMAX_H3_VIDEO_MAX_FRAMES = 362;

// Minimum, default and maximum whole-second duration bounds.
static constexpr int MINIMAX_H3_VIDEO_MIN_SECONDS = 1,
                     MINIMAX_H3_VIDEO_DEFAULT_SECONDS = 5,
                     MINIMAX_H3_VIDEO_MAX_SECONDS = 15;

// Minimum and maximum accepted Euler denoising step counts.
static constexpr int MINIMAX_H3_VIDEO_MIN_STEPS = 1,
                     MINIMAX_H3_VIDEO_MAX_STEPS = 100;

// Maximum REF2VA reference image, video, standalone audio and total file
// counts; Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:1193,1197,1205
// and MiniMaxAI/MiniMax-H3 README.md.
static constexpr int MINIMAX_H3_VIDEO_MAX_REF_IMAGES = 9,
                     MINIMAX_H3_VIDEO_MAX_REF_VIDEOS = 3,
                     MINIMAX_H3_VIDEO_MAX_REF_AUDIOS = 3,
                     MINIMAX_H3_VIDEO_MAX_TOTAL_REFS = 12;

// Maximum short edge in pixels for REF2VA reference images in "max" sizing
// mode; Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:939.
static constexpr int MINIMAX_H3_VIDEO_REF_IMAGE_MAX_SHORT_EDGE = 2048;

// Required guidance scale for CFG-distilled MiniMax-H3 video checkpoints;
// MiniMaxAI/MiniMax-H3 README.md:190.
static constexpr double MINIMAX_H3_VIDEO_GUIDANCE_SCALE = 1.0;

// Resident GPU byte requirements of the eight MiniMax-H3 Video weight groups.
typedef struct {
  // Qwen3-VL text encoder weight bytes.
  uint64_t text;
  // Qwen3-VL vision tower weight bytes.
  uint64_t vision;
  // Pruned INT8-ConvRot DiT weight bytes.
  uint64_t dit;
  // Optional BF16 Turbo LoRA weight bytes (0 when unused).
  uint64_t lora;
  // INT8-ConvRot 3D Video VAE decoder weight bytes.
  uint64_t video_vae;
  // FP16 3D Video VAE encoder weight bytes.
  uint64_t video_vae_enc;
  // FP32 Audio VAE BigVGAN decoder weight bytes.
  uint64_t audio_vae;
  // FP32 Audio VAE DAC encoder weight bytes.
  uint64_t audio_vae_enc;
} MinimaxH3VideoWeightBytes;

// GPU working bytes of each stage of one video request, weights aside.
typedef struct {
  // Peak bytes while encoding prompt, vision blocks and condition latents.
  uint64_t encode;
  // Peak bytes while running DiT sampling beside prompt hidden rows.
  uint64_t sample;
  // Peak bytes while decoding video frames with the Video VAE.
  uint64_t decode_video;
  // Peak bytes while decoding stereo audio with the Audio VAE.
  uint64_t decode_audio;
} MinimaxH3VideoStageBytes;

// Adapts (width, height) to a 768-short-edge canvas capped at 768*1344 pixels
// with both axes rounded to 32 px; Comfy-Org/ComfyUI
// comfy_extras/nodes_minimax_h3.py:971-982.
void minimax_h3_video_adapt_canvas(int width, int height, int *out_w,
                                   int *out_h);

// Validates and rounds a requested video canvas size (0x0 uses 1344x768) to
// multiples of 32 px within canvas limits. Returns 1 on success or 0 with a
// message in error.
int minimax_h3_video_plan_canvas(int width, int height, int *out_w, int *out_h,
                                 char *error, size_t error_cap);

// Plans aligned video frame count (17k+5), video latent T (5k+2) and 40 Hz
// audio latent T from seconds or explicit frames (0/0 uses 124 frames).
// Returns 1 on success or 0 with a message in error.
int minimax_h3_video_plan_frames(int seconds, int frames, int *out_frames,
                                 int *out_t_lat, int *out_audio_t, char *error,
                                 size_t error_cap);

// Computes full-residence and staged peak GPU memory bytes for a video request
// given weight sizes, stage workspaces and on-demand encoder flags.
MinimaxH3RequestMemory minimax_h3_video_plan_request_memory(
    MinimaxH3VideoWeightBytes weights, MinimaxH3VideoStageBytes stages,
    int needs_vision, int needs_video_enc, int needs_audio_enc);

// Reserves GPU memory in account for a planned video before running the text
// encoder. Returns MINIMAX_H3_ADMITTED (setting *out_use_full),
// MINIMAX_H3_NOT_NOW or MINIMAX_H3_NEVER_FITS (writing error and keeping
// account->minimum unchanged on refusal).
MinimaxH3Admission minimax_h3_video_admit_request(
    MemoryAccount *account, int prefer_full, uint64_t staged_base, int width,
    int height, int frames, MinimaxH3RequestMemory req_mem, int *out_use_full,
    char *error, size_t error_cap);

#endif
