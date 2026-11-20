#ifndef EREMITE_MODEL_MINIMAX_H3_IMAGE_VAE_H
#define EREMITE_MODEL_MINIMAX_H3_IMAGE_VAE_H
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <stdint.h>

// MiniMax-H3-Image VAE: a 24-channel spatial-16x VAE with a 24-layer FP16 2D
// ViT decoder and spatial tiling (256 px tiles, minimum 64 px linear overlap).

// Number of channels in MiniMax-H3 image latents.
static constexpr int MINIMAX_H3_VAE_LATENT_CHANNELS = 24;

// Spatial compression factor of the VAE (pixels per latent cell).
static constexpr int MINIMAX_H3_VAE_SPATIAL_RATIO = 16;

// Maximum number of spatial tiles along one image axis.
static constexpr int MINIMAX_H3_VAE_MAX_TILES = 64;

// One axis's spatial tiling plan in pixels.
typedef struct {
  // Number of tiles along this axis.
  int count;
  // Starting pixel coordinate of each tile.
  int start[MINIMAX_H3_VAE_MAX_TILES];
  // Pixel length of each tile.
  int len[MINIMAX_H3_VAE_MAX_TILES];
  // Pixel overlap between tile i and tile i + 1 (count - 1 entries).
  int overlap[MINIMAX_H3_VAE_MAX_TILES];
} MinimaxH3VaeTilePlan;

// Loaded MiniMax-H3-Image VAE handle.
typedef struct MinimaxH3ImageVae MinimaxH3ImageVae;

// Computes the spatial tiling plan for an axis of length input_len pixels
// (must be a positive multiple of MINIMAX_H3_VAE_SPATIAL_RATIO).
MinimaxH3VaeTilePlan minimax_h3_image_vae_split_tiles(int input_len);

// Opens the FP16 VAE safetensors file at weights_path and registers decoder
// weights in account. Does not load GPU buffers until requested.
MinimaxH3ImageVae *minimax_h3_image_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path);

// Frees the VAE handle, its weight group and open safetensors file.
void minimax_h3_image_vae_close(MinimaxH3ImageVae *vae);

// Returns the open safetensors index shared with the VAE encoder.
const TensorIndex *minimax_h3_image_vae_tensors(const MinimaxH3ImageVae *vae);

// Returns the VAE's GPU handle.
Gpu *minimax_h3_image_vae_gpu(const MinimaxH3ImageVae *vae);

// Returns the VAE decoder's weight group.
WeightGroup *minimax_h3_image_vae_weights(MinimaxH3ImageVae *vae);

// Returns the GPU bytes required to keep the VAE decoder weights resident.
uint64_t minimax_h3_image_vae_required_bytes(const MinimaxH3ImageVae *vae);

// Returns the GPU bytes currently resident for the VAE decoder weights.
uint64_t minimax_h3_image_vae_resident_bytes(const MinimaxH3ImageVae *vae);

// Loads the VAE decoder weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_image_vae_load(MinimaxH3ImageVae *vae);

// Decodes normalized channel-first FP32 latents [24, h_lat, w_lat] into
// interleaved 8-bit sRGB pixels [h_lat * 16, w_lat * 16, 3] in out_rgb.
// Returns 1 on success or 0 if loading or workspace allocation fails.
int minimax_h3_image_vae_decode(MinimaxH3ImageVae *vae, const float *latents,
                                int h_lat, int w_lat, uint8_t *out_rgb);

#endif
