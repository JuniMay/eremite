#ifndef EREMITE_MODEL_NEMOTRON_AUDIO_H
#define EREMITE_MODEL_NEMOTRON_AUDIO_H
#include "engine/memory.h"
#include "engine/pass.h"
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// NVIDIA Nemotron-3.5-Transcribe Parallel Expert Perception Encoder:
// 16 kHz log-mel frontend, 32-layer FP8 ASR Transformer, 31-layer BF16
// Sortformer speaker branch, high-resolution subpixel upsampling, thresholded
// sinusoidal speaker fusion and modality projection into the language model's
// 2688-wide embedding space.

// Audio frontend sampling and STFT geometry (config.json perception).
static constexpr int NEMO_AUDIO_SAMPLE_RATE = 16000;
static constexpr int NEMO_AUDIO_N_FFT = 512;
static constexpr int NEMO_AUDIO_WIN_LEN = 400;
static constexpr int NEMO_AUDIO_HOP_LEN = 160;
static constexpr int NEMO_AUDIO_N_BINS = NEMO_AUDIO_N_FFT / 2 + 1;
static constexpr int NEMO_AUDIO_N_MELS = 128;
static constexpr int NEMO_AUDIO_SUBSAMPLE = 8;
static constexpr int NEMO_AUDIO_STACK_DIM =
    NEMO_AUDIO_N_MELS * NEMO_AUDIO_SUBSAMPLE;

// ASR Transformer branch geometry (config.json pe_encoder_config).
static constexpr int NEMO_AUDIO_ASR_DIM = 1280;
static constexpr int NEMO_AUDIO_ASR_HEADS = 10;
static constexpr int NEMO_AUDIO_ASR_HEAD_DIM = 128;
static constexpr int NEMO_AUDIO_ASR_FFN = 5120;
static constexpr int NEMO_AUDIO_ASR_LAYERS = 32;
static constexpr int NEMO_AUDIO_ASR_HALF_ROT = 32;

// Sortformer speaker branch geometry (config.json diarization_model_cfg).
static constexpr int NEMO_AUDIO_DIAR_DIM = 512;
static constexpr int NEMO_AUDIO_DIAR_HEADS = 8;
static constexpr int NEMO_AUDIO_DIAR_HEAD_DIM = 64;
static constexpr int NEMO_AUDIO_DIAR_FFN = 2048;
static constexpr int NEMO_AUDIO_DIAR_LAYERS = 31;
static constexpr int NEMO_AUDIO_DIAR_HALF_ROT = 32;
static constexpr int NEMO_AUDIO_DIAR_TF_DIM = 192;
static constexpr int NEMO_AUDIO_NUM_SPK = 8;
static constexpr int NEMO_AUDIO_UP_FACTOR = 8;
static constexpr int NEMO_AUDIO_CONV_K = 3;

// Modality projection output width matching Nemotron-H hidden_size.
static constexpr int NEMO_AUDIO_OUT_DIM = 2688;

// Long-form windowed inference and streaming Sortformer cache parameters.
static constexpr int NEMO_AUDIO_ONLINE_TOKENS = 500;
static constexpr int NEMO_AUDIO_CTX_LEFT_TOKENS = 50;
static constexpr int NEMO_AUDIO_CTX_RIGHT_TOKENS = 50;
static constexpr int NEMO_AUDIO_ONLINE_MEL_FRAMES =
    NEMO_AUDIO_ONLINE_TOKENS * NEMO_AUDIO_SUBSAMPLE;
static constexpr int NEMO_AUDIO_DIAR_SPKCACHE_LEN = 188;
static constexpr int NEMO_AUDIO_DIAR_FIFO_LEN = 40;
static constexpr int NEMO_AUDIO_DIAR_UPDATE_PERIOD = 300;
static constexpr int NEMO_AUDIO_DIAR_SIL_PER_SPK = 1;

// Loaded Parallel Expert Perception Encoder handle.
typedef struct NemoAudioEncoder NemoAudioEncoder;

// Returns the number of valid log-mel frames produced from pcm_samples at
// 16 kHz.
int nemo_audio_mel_frames(int pcm_samples);

// Returns the number of encoded audio tokens produced from pcm_samples at
// 16 kHz.
int nemo_audio_token_count(int pcm_samples);

// Computes unnormalized log-mel features [mel_frames, 128] from 16 kHz mono
// PCM using window [400] and mel filterbank fb [128, 257].
void nemo_audio_mel_spectrogram(const float *pcm, int pcm_samples,
                                const float *window, const float *fb,
                                float *out_mel);

// Applies per-feature zero-mean unit-variance normalization (ddof=1, +1e-5)
// across frames of [frames, 128] mel features into out [frames, 128].
void nemo_audio_normalize_per_feature(const float *mel, int frames, float *out);

// Stacks consecutive groups of 8 mel frames from [frames, 128] into
// [ceil(frames / 8), 1024], zero-padding a partial trailing group.
void nemo_audio_stack_frames(const float *mel, int frames, float *out_stacked);

// Builds the fixed BF16-rounded sinusoidal speaker fusion matrix [n_spk, dim].
void nemo_audio_build_sinusoid_kernel(float *out_kernel, int n_spk, int dim);

// Compresses n_frames (> spkcache_len) Sortformer pre-encoded embeddings
// [n_frames, emb_dim] and speaker probabilities [n_frames, n_spk] into
// out_embs [spkcache_len, emb_dim] and out_preds [spkcache_len, n_spk], using
// sil_emb [emb_dim] for silence placeholder slots.
void nemo_audio_sortformer_compress_spkcache(const float *emb_seq,
                                             const float *preds, int n_frames,
                                             int emb_dim, int n_spk,
                                             int spkcache_len, int sil_per_spk,
                                             const float *sil_emb,
                                             float *out_embs, float *out_preds);

// Opens perception.safetensors in dir, registers its weights in a reloadable
// weight group under account, and compiles perception GPU pipelines.
NemoAudioEncoder *nemo_audio_open(Gpu *gpu, MemoryAccount *account,
                                  MemorySlot *work, const char *dir);

// Releases all weights, file handles and host resources owned by enc.
void nemo_audio_close(NemoAudioEncoder *enc);

// Pins or unpins the perception weight group while a request runs.
void nemo_audio_pin(NemoAudioEncoder *enc, int active);

// Returns nonzero when the perception weights are currently resident in GPU
// memory.
int nemo_audio_is_loaded(const NemoAudioEncoder *enc);

// Returns the memory slot of the perception weight group.
MemorySlot *nemo_audio_weight_slot(NemoAudioEncoder *enc);

// Returns the byte footprint required to hold the perception weights.
uint64_t nemo_audio_weight_bytes(const NemoAudioEncoder *enc);

// Encodes 16 kHz mono PCM into heap-allocated FP32 (BF16-rounded) embeddings
// [*out_tokens, 2688] written to *out_embeds. wait yields between layers when
// non-null. Returns 1 on success or 0 with error populated on failure.
int nemo_audio_encode(NemoAudioEncoder *enc, const float *pcm, int pcm_samples,
                      const PassWait *wait, float **out_embeds, int *out_tokens,
                      char *error, size_t error_cap);

#endif
