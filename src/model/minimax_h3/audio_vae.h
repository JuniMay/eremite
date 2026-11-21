#ifndef EREMITE_MODEL_MINIMAX_H3_AUDIO_VAE_H
#define EREMITE_MODEL_MINIMAX_H3_AUDIO_VAE_H
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/minimax_h3/quant.h"
#include <stdint.h>

// MiniMax-H3 Audio VAE: a 32 kHz stereo waveform codec with a BigVGAN decoder
// and a DAC encoder plus causal attention projection (40 latent frames/s,
// 800 samples per latent frame, 32 latent channels per stereo channel).

// Audio output sample rate in Hz (comfy/ldm/minimax/audio_vae.py:1493).
static constexpr int MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000;

// PCM samples per audio latent frame (comfy/ldm/minimax/audio_vae.py:1498).
static constexpr int MINIMAX_H3_AUDIO_HOP_LENGTH = 800;

// Audio latent frames per second (sample_rate / hop_length = 40).
static constexpr int MINIMAX_H3_AUDIO_LATENT_RATE =
    MINIMAX_H3_AUDIO_SAMPLE_RATE / MINIMAX_H3_AUDIO_HOP_LENGTH;

// Channels per stereo audio latent frame (comfy/ldm/minimax/audio_vae.py:1490).
static constexpr int MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32;

// Stereo waveform channel count (comfy/ldm/minimax/audio_vae.py:1478).
static constexpr int MINIMAX_H3_AUDIO_STEREO_CHANNELS = 2;

// Loaded MiniMax-H3 Audio VAE handle.
typedef struct MinimaxH3AudioVae MinimaxH3AudioVae;

// Opens the FP32 audio VAE safetensors file at weights_path and registers
// decoder and encoder weight groups in account without loading GPU buffers.
MinimaxH3AudioVae *minimax_h3_audio_vae_open(Gpu *gpu, MemoryAccount *account,
                                             const char *weights_path);

// Frees the audio VAE handle, its weight groups and open safetensors file.
void minimax_h3_audio_vae_close(MinimaxH3AudioVae *vae);

// Returns the audio decoder's weight group.
WeightGroup *minimax_h3_audio_vae_weights(MinimaxH3AudioVae *vae);

// Returns the GPU bytes required to keep the audio decoder weights resident.
uint64_t minimax_h3_audio_vae_required_bytes(const MinimaxH3AudioVae *vae);

// Returns the GPU bytes currently resident for the audio decoder weights.
uint64_t minimax_h3_audio_vae_resident_bytes(const MinimaxH3AudioVae *vae);

// Returns the GPU workspace bytes needed to decode audio_t latent frames.
uint64_t minimax_h3_audio_vae_decode_workspace_bytes(int audio_t);

// Loads the audio decoder weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_audio_vae_load(MinimaxH3AudioVae *vae);

// Returns the audio encoder's weight group.
WeightGroup *minimax_h3_audio_vae_encoder_weights(MinimaxH3AudioVae *vae);

// Returns the GPU bytes required to keep the audio encoder weights resident.
uint64_t
minimax_h3_audio_vae_encoder_required_bytes(const MinimaxH3AudioVae *vae);

// Returns the GPU bytes currently resident for the audio encoder weights.
uint64_t
minimax_h3_audio_vae_encoder_resident_bytes(const MinimaxH3AudioVae *vae);

// Returns the GPU workspace bytes needed to encode samples per channel.
uint64_t minimax_h3_audio_vae_encode_workspace_bytes(int samples);

// Loads the audio encoder weights into GPU memory if not already resident.
// Returns 1 on success or 0 if memory did not fit or shutdown interrupted.
int minimax_h3_audio_vae_load_encoder(MinimaxH3AudioVae *vae);

// Returns the number of latent frames produced for samples PCM frames per
// channel (ceil(samples / 800)).
int minimax_h3_audio_vae_encoded_frames(int samples);

// Decodes normalized audio latents [32, 2, audio_t] into planar stereo FP32
// PCM [2, audio_t * 800] in [-1, 1] written to out_pcm. hooks may be null.
// Returns 1 on success or 0 on load, workspace or stop failure.
int minimax_h3_audio_vae_decode(MinimaxH3AudioVae *vae, const float *latents,
                                int audio_t, float *out_pcm,
                                const MinimaxH3RunHooks *hooks);

// Encodes planar stereo FP32 PCM [2, samples] at 32 kHz (zero-padded to a
// multiple of 800 samples) into normalized latents [32, 2, audio_t] written to
// out_latents, where audio_t = ceil(samples / 800). hooks may be null.
// Returns 1 on success or 0 on load, workspace or stop failure.
int minimax_h3_audio_vae_encode(MinimaxH3AudioVae *vae, const float *pcm,
                                int samples, float *out_latents,
                                const MinimaxH3RunHooks *hooks);

#endif
