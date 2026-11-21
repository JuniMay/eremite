#ifndef EREMITE_MODEL_MINIMAX_H3_VIDEO_VAE_H
#define EREMITE_MODEL_MINIMAX_H3_VIDEO_VAE_H
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/quant.h"
#include <stdint.h>

// MiniMax-H3 Video VAE: a 24-channel spatial-16x, temporal-4x causal video VAE
// with a 36-layer INT8-ConvRot 3D ViT decoder, a 6-level FP16 3D causal CNN
// encoder, spatial tiling (256 px tiles, min 64 px overlap) and temporal
// chunking (17-frame clips, 5-latent stride, 2-latent overlap).

// Temporal compression factor of the video VAE (comfy/ldm/minimax/vae.py:636).
static constexpr int MINIMAX_H3_VVAE_TEMPORAL_RATIO = 4;

// Frames per temporal encoding clip (comfy/ldm/minimax/vae.py:621).
static constexpr int MINIMAX_H3_VVAE_CLIP_LENGTH = 17;

// Trailing latent tokens dropped after temporal encoding
// (comfy/ldm/minimax/vae.py:622).
static constexpr int MINIMAX_H3_VVAE_TOKEN_DROP = 3;

// Latent tokens per decoded chunk stride and overlap
// (comfy/ldm/minimax/vae.py:642-643).
static constexpr int MINIMAX_H3_VVAE_CHUNK_TOKENS = 5,
                     MINIMAX_H3_VVAE_TOKEN_OVERLAP = 2,
                     MINIMAX_H3_VVAE_FRAME_PRE_PAD = 3,
                     MINIMAX_H3_VVAE_FRAME_OVERLAP = 5;

// Temporal decoding chunk plan for a latent sequence of z_len frames.
typedef struct {
  // Trailing duplicate latent frames padded before chunked decoding.
  int pad_tokens;
  // Number of 7-latent (chunk + overlap) decoder chunks.
  int num_chunks;
  // Final decoded RGB frame count after trimming temporal padding.
  int out_frames;
} MinimaxH3VvaeTemporalPlan;

// Loaded MiniMax-H3 Video VAE handle.
typedef struct MinimaxH3VideoVae MinimaxH3VideoVae;

// Computes the temporal decoding chunk plan and final frame count for z_len
// video latent frames (z_len >= 1).
MinimaxH3VvaeTemporalPlan minimax_h3_video_vae_decode_plan(int z_len);

// Returns the latent temporal length produced by encoding frame_count video
// frames (frame_count >= 1).
int minimax_h3_video_vae_encode_latent_t(int frame_count);

// Opens the INT8-ConvRot video VAE safetensors file at weights_path and
// registers decoder weights in account without loading GPU buffers.
MinimaxH3VideoVae *minimax_h3_video_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path);

// Frees the video VAE handle, its weight groups and open safetensors file.
void minimax_h3_video_vae_close(MinimaxH3VideoVae *vae);

// Returns the video decoder's weight group.
WeightGroup *minimax_h3_video_vae_weights(MinimaxH3VideoVae *vae);

// Returns the GPU bytes required to keep the video decoder weights resident.
uint64_t minimax_h3_video_vae_required_bytes(const MinimaxH3VideoVae *vae);

// Returns the GPU bytes currently resident for the video decoder weights.
uint64_t minimax_h3_video_vae_resident_bytes(const MinimaxH3VideoVae *vae);

// Returns the GPU workspace bytes needed to decode one 7-latent 256x256 tile.
uint64_t minimax_h3_video_vae_decode_workspace_bytes(void);

// Loads the video decoder weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_video_vae_load(MinimaxH3VideoVae *vae);

// Decodes normalized channel-first FP32 latents [24, t_lat, h_lat, w_lat] into
// interleaved 8-bit sRGB frames [out_frames, h_lat * 16, w_lat * 16, 3]
// written to out_rgb, where out_frames = decode_plan(t_lat).out_frames.
// hooks may be null. Returns 1 on success or 0 on load, workspace or stop fail.
int minimax_h3_video_vae_decode(MinimaxH3VideoVae *vae, const float *latents,
                                int t_lat, int h_lat, int w_lat,
                                uint8_t *out_rgb,
                                const MinimaxH3RunHooks *hooks);

#endif
