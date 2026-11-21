#ifndef EREMITE_MODEL_MINIMAX_H3_DIT_H
#define EREMITE_MODEL_MINIMAX_H3_DIT_H
#include "engine/memory.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/minimax_h3/lora.h"
#include "model/minimax_h3/quant.h"
#include <stdint.h>

// MiniMax-H3 DiT: a 50-layer INT8-ConvRot single-stream transformer with a
// 2-layer BF16 token refiner, pruned curve-form AdaLN and partial 3D
// split-half RoPE. Supports both MiniMax-H3-Image (6-step PDD head bank) and
// MiniMax-H3 Video FL2VA/REF2VA (joint video + stereo audio streams with
// optional runtime BF16 Turbo LoRA).

// Maximum number of PDD sampling steps supported by the head bank.
static constexpr int MINIMAX_H3_DIT_MAX_PDD_STEPS = 16;

// Native video frame rate in frames per second; Comfy-Org/ComfyUI
// comfy_extras/nodes_minimax_h3.py:940 (FPS = 24).
static constexpr int MINIMAX_H3_VIDEO_FPS = 24;

// Default video and audio flow-matching shift factors; MiniMaxAI/MiniMax-H3
// scheduler/scheduler_config.json:3 (12.0),
// audio_scheduler/scheduler_config.json:3 (3.0) and Comfy-Org/ComfyUI
// comfy/ldm/minimax/model.py:521.
static constexpr double MINIMAX_H3_VIDEO_SIGMA_SHIFT = 12.0,
                        MINIMAX_H3_AUDIO_SIGMA_SHIFT = 3.0;

// Default base-model denoising step count when no Turbo LoRA overrides it.
static constexpr int MINIMAX_H3_VIDEO_DEFAULT_STEPS = 20;

// One optional encoded reference image latent passed during image editing.
typedef struct {
  // Normalized channel-first FP32 latent [24, h_lat, w_lat].
  const float *latents;
  // Reference latent height and width in VAE cells (1/16 pixel dimensions).
  int h_lat, w_lat;
} MinimaxH3RefLatent;

// Loaded MiniMax-H3-Image DiT handle.
typedef struct MinimaxH3Dit MinimaxH3Dit;

// Opens the pruned INT8-ConvRot DiT safetensors file at weights_path and
// registers all weights in account. Does not load GPU buffers until requested.
MinimaxH3Dit *minimax_h3_dit_open(Gpu *gpu, MemoryAccount *account,
                                  const char *weights_path);

// Frees the DiT handle, its weight group and open safetensors file.
void minimax_h3_dit_close(MinimaxH3Dit *dit);

// Returns the DiT's weight group.
WeightGroup *minimax_h3_dit_weights(MinimaxH3Dit *dit);

// Returns the GPU bytes required to keep the DiT weights resident.
uint64_t minimax_h3_dit_required_bytes(const MinimaxH3Dit *dit);

// Returns the GPU bytes currently resident for the DiT weights.
uint64_t minimax_h3_dit_resident_bytes(const MinimaxH3Dit *dit);

// Returns the GPU workspace bytes needed by one sampling call for text_len
// prompt tokens, ref_patches reference patches and target_patches canvas
// patches.
uint64_t minimax_h3_dit_workspace_bytes(int text_len, int ref_patches,
                                        int target_patches);

// Returns the number of PDD sampling steps configured by the checkpoint.
int minimax_h3_dit_pdd_steps(const MinimaxH3Dit *dit);

// Returns non-zero when the loaded DiT checkpoint includes the stereo audio
// patch embedder and final audio velocity head (FL2VA / REF2VA video model).
int minimax_h3_dit_has_audio(const MinimaxH3Dit *dit);

// Rounds frame_count up to the MiniMax-H3 17k + 5 video frame grid (minimum 5);
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:955-958.
int minimax_h3_video_align_frames(int frame_count);

// Returns the video VAE latent temporal length (5k + 2) for frame_count video
// frames; Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:961-962.
int minimax_h3_video_latent_t(int frame_count);

// Returns the 40 Hz audio latent frame count matching frame_count frames at
// 24 fps; Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:965-968.
int minimax_h3_video_audio_t(int frame_count);

// Maps sigma from flow shift from_shift onto flow shift to_shift via the
// shared base grid; Comfy-Org/ComfyUI comfy/ldm/minimax/model.py:46-49.
double minimax_h3_time_shift_sigma(double sigma, double from_shift,
                                   double to_shift);

// One FL2VA or guide conditioning anchor placed at a target video frame index.
typedef struct {
  // Resolved 0-based pixel frame index along the target video timeline.
  int frame_index;
  // Optional normalized video latent [24, t_lat, h_lat, w_lat] matching the
  // target spatial canvas; null when anchoring audio only.
  const float *video_latents;
  // Latent temporal length of video_latents (1 for a still frame, or 5k + 2).
  int t_lat;
  // Optional normalized audio latent [32, 2, audio_t]; null for video only.
  const float *audio_latents;
  // Audio latent temporal length (40 Hz frames).
  int audio_t;
} MinimaxH3VideoKeyframe;

// Modality kind of one REF2VA conditioning block packed ahead of the target.
typedef enum {
  // Single reference image latent [24, 1, h_lat, w_lat].
  MINIMAX_H3_VIDEO_REF_IMAGE,
  // Standalone stereo reference audio latent [32, 2, audio_t].
  MINIMAX_H3_VIDEO_REF_AUDIO,
  // Silent reference video latent [24, t_lat, h_lat, w_lat].
  MINIMAX_H3_VIDEO_REF_VIDEO,
  // Reference video latent paired with its synchronized stereo audio latent.
  MINIMAX_H3_VIDEO_REF_VIDEO_AUDIO
} MinimaxH3VideoRefKind;

// One REF2VA conditioning block passed to video sampling.
typedef struct {
  // Reference block modality kind.
  MinimaxH3VideoRefKind kind;
  // Normalized video/image latent [24, t_lat, h_lat, w_lat]; null for audio.
  const float *video_latents;
  // Latent temporal length (1 for image, >= 1 for video, 0 for audio).
  int t_lat;
  // Latent height and width in VAE cells (1/16 pixel dimensions).
  int h_lat, w_lat;
  // Normalized audio latent [32, 2, audio_t]; null when no audio is attached.
  const float *audio_latents;
  // Audio latent temporal length (40 Hz frames; 0 when no audio is attached).
  int audio_t;
} MinimaxH3VideoRef;

// Kind of one contiguous segment inside a MiniMax-H3 packed DiT sequence.
typedef enum {
  // Refined Qwen3-VL text/vision prompt tokens.
  MINIMAX_H3_SEG_TEXT,
  // FL2VA/guide visual keyframe patch tokens.
  MINIMAX_H3_SEG_COND,
  // FL2VA/guide audio anchor tokens.
  MINIMAX_H3_SEG_COND_AUDIO,
  // REF2VA reference image or video patch tokens.
  MINIMAX_H3_SEG_REF_IMG,
  // REF2VA reference audio tokens.
  MINIMAX_H3_SEG_REF_AUDIO,
  // Target stereo audio tokens to denoise.
  MINIMAX_H3_SEG_AUDIO,
  // Target video patch tokens to denoise.
  MINIMAX_H3_SEG_VIDEO
} MinimaxH3PackedSegKind;

// One contiguous span inside a packed DiT sequence.
typedef struct {
  // Start token index in the packed sequence.
  int start;
  // Number of tokens in this segment.
  int count;
  // Segment modality and role.
  MinimaxH3PackedSegKind kind;
} MinimaxH3PackedSeg;

// Static packed-sequence structure for one audio-video DiT sampling shape.
typedef struct {
  // Contiguous segment table in packed order [seg_count].
  MinimaxH3PackedSeg *segments;
  // Number of segments in segments.
  int seg_count;
  // 3D MM-RoPE (t, h, w) coordinates [seq_len * 3].
  float *pos;
  // Per-token AdaLN modulation row indices [seq_len].
  uint32_t *mod_rows;
  // Total packed tokens, visual patch rows and audio rows.
  int seq_len, total_vis, total_audio;
  // Start offset and row count of the target video stream.
  int target_video_start, target_video_rows;
  // Start offset and row count of the target audio stream.
  int target_audio_start, target_audio_rows;
  // Distinct AdaLN timestep rows evaluated per block (2..4).
  int num_t;
} MinimaxH3PackedLayout;

// Frees heap arrays owned by layout and zeroes its fields.
void minimax_h3_packed_layout_free(MinimaxH3PackedLayout *layout);

// Builds the MiniMax-H3 packed sequence segments, 3D MM-RoPE coordinates and
// per-token AdaLN modulation rows for a video+audio target with optional
// keyframes and references; Comfy-Org/ComfyUI
// comfy/ldm/minimax/model.py:391-512.
void minimax_h3_packed_layout_build(
    int text_len, const int *text_tags, int t_lat, int h_lat, int w_lat,
    int audio_t, const MinimaxH3VideoKeyframe *keyframes, int keyframe_count,
    const MinimaxH3VideoRef *refs, int ref_count, MinimaxH3PackedLayout *out);

// Returns the GPU workspace bytes needed by one audio-video DiT sampling run
// for the given packed token counts and optional BF16 LoRA adapter.
uint64_t minimax_h3_dit_video_workspace_bytes(int text_len, int total_vis,
                                              int target_video_rows,
                                              int total_audio,
                                              int target_audio_rows,
                                              const MinimaxH3Lora *lora);

// Parameters for one joint audio-video DiT sampling call.
typedef struct {
  // FP32 GPU buffer of text_len by 5120 Qwen3-VL layer-49 hidden states.
  const Buf *text_hidden;
  // Token count of text_hidden.
  int text_len;
  // Optional text_len array of AdaLN modality tags (0=visual, 1=text).
  const int *text_tags;
  // Optional FL2VA / guide keyframes anchored on the target timeline.
  const MinimaxH3VideoKeyframe *keyframes;
  // Number of entries in keyframes.
  int keyframe_count;
  // Optional REF2VA reference image/audio/video blocks packed before targets.
  const MinimaxH3VideoRef *refs;
  // Number of entries in refs.
  int ref_count;
  // Target video latent shape [24, t_lat, h_lat, w_lat].
  int t_lat, h_lat, w_lat;
  // Target audio latent temporal length [32, 2, audio_t] (40 Hz frames).
  int audio_t;
  // Number of Euler denoising steps (> 0).
  int steps;
  // Video and audio flow-matching shift factors (defaults 12.0 and 3.0).
  double shift_video, shift_audio;
  // Deterministic RNG seed for initial video/audio noise and cond noise aug.
  uint64_t seed;
  // Optional runtime BF16 LoRA adapter (e.g., 4-step or 8-step Turbo LoRA).
  const MinimaxH3Lora *lora;
  // Caller-allocated output buffer for video latents [24, t_lat, h_lat, w_lat].
  float *out_video_latents;
  // Caller-allocated output buffer for audio latents [32, 2, audio_t].
  float *out_audio_latents;
  // Optional cooperative yield, cancellation and progress hooks.
  const MinimaxH3RunHooks *hooks;
} MinimaxH3VideoSampleParams;

// Runs joint audio-video flow-matching sampling on an audio-enabled DiT
// checkpoint, writing normalized channel-first video latents and stereo audio
// latents. Returns 1 on success or 0 on load, workspace or stop failure.
int minimax_h3_dit_sample_video(MinimaxH3Dit *dit,
                                const MinimaxH3VideoSampleParams *params);

// Loads the DiT weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_dit_load(MinimaxH3Dit *dit);

// Runs full PDD flow-matching sampling to produce a normalized channel-first
// latent [24, h_lat, w_lat] in out_latents.
// - text_hidden: FP32 GPU buffer of text_len by 5120 Qwen3-VL layer-49 states.
// - text_tags: optional text_len array of AdaLN modality tags (0=visual,
// 1=text);
//   when null, all text tokens receive tag 1.
// - refs: optional array of ref_count encoded reference image latents.
// - seed: deterministic RNG seed for initial latent noise (and ref noise aug).
// - hooks: optional cooperative yield and cancellation hooks; may be null.
// Returns 1 on success or 0 if loading, workspace allocation or stop fails.
int minimax_h3_dit_sample(MinimaxH3Dit *dit, const Buf *text_hidden,
                          int text_len, const int *text_tags,
                          const MinimaxH3RefLatent *refs, int ref_count,
                          int h_lat, int w_lat, uint64_t seed,
                          float *out_latents, const MinimaxH3RunHooks *hooks);

#endif
