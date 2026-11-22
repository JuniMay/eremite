#include "model/nemotron/audio.h"
#include "base/error.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "platform/threads.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Parallel Expert Perception Encoder (perception.safetensors):
// 1. CPU log-mel spectrogram frontend (16 kHz, preemph 0.97, 512-pt STFT,
//    400-sample Hann window, 160-sample hop, 128 mel bins, log(x + 2^-24)).
// 2. Per-feature normalization for ASR and raw features for Sortformer.
// 3. 32-layer FP8/BF16 ASR TransformerEncoder (d_model 1280, 10 heads, qk_norm,
//    RoPE half-rotated) and 31-layer BF16 Sortformer TransformerEncoder
//    (d_model 512, 8 heads, full RoPE) + subpixel high-resolution speaker head.
// 4. Single-pass encoding up to 40 s (4000 mel frames), and context-extended
//    windowed ASR + streaming speaker-cache Sortformer for recordings up to 1
//    h.
// 5. Thresholded sinusoidal speaker fusion and 1280->2688 linear projection.

// Frontend numerical constants from FilterbankFeatures and config.json.
static constexpr float PREEMPH_COEFF = 0.97f;
static constexpr float LOG_ZERO_GUARD = 5.9604644775390625e-8f;  // 2^-24
static constexpr float NORM_EPS = 1e-5f;
static constexpr float LN_EPS = 1e-5f;
static constexpr float SPK_THRESHOLD = 0.5f;
static constexpr float SPK_KERNEL_SCALE = 1.0f;
static constexpr double ROPE_BASE = 10000.0;
static constexpr double SINUSOID_BASE = 10000.0;
static constexpr double TWO_PI = 6.2831853071795864769;
static constexpr int STFT_PAD = NEMO_AUDIO_N_FFT / 2;
static constexpr int WIN_OFFSET = (NEMO_AUDIO_N_FFT - NEMO_AUDIO_WIN_LEN) / 2;

// Streaming Sortformer speaker-cache compression constants (config.json).
static constexpr float SPK_PRED_SCORE_THRESH = 0.25f;
static constexpr float SPK_BOOST_LATEST = 0.05f;
static constexpr float SPK_STRONG_BOOST_RATE = 0.75f;
static constexpr float SPK_WEAK_BOOST_RATE = 1.5f;
static constexpr float SPK_MIN_POS_RATE = 0.5f;
static constexpr float SPK_BOOST_OFFSET = 0.5f;
static constexpr float SPK_SPEECH_THRESH = 0.5f;
static constexpr int SPK_MAX_INDEX = 99999;

// GPU kernel tile, threadgroup and alignment constants matching
// src/kernels/nemotron_audio.metal.
static constexpr uint32_t AUD_TILE = 64;
static constexpr uint32_t AUD_FP8_M_TILE = 128;
static constexpr uint32_t AUD_MM_THREADS = 128;
static constexpr uint32_t AUD_ROW_THREADS = 256;
static constexpr uint32_t AUD_PACK_THREADS = 64;
static constexpr uint32_t AUD_UPSAMPLE_THREADS = 256;
static constexpr uint32_t AUD_ACT_NONE = 0;
static constexpr uint32_t AUD_ACT_GELU = 1;
static constexpr uint32_t AUD_ACT_RELU = 2;
static constexpr float FP8_UNDO_SHIFT = 256.0f;
static constexpr double LOG2_E = 1.4426950408889634074;
static constexpr size_t ALIGN_READ_MIN_BYTES = 1u << 20;
static constexpr size_t ALIGN_MASK = 15u;
static constexpr size_t PATH_CAP = 1024;
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_MASK = 0xFFFF0000u;

// Maximum token count of one Transformer forward (streaming Sortformer runs up
// to spkcache 188 + fifo 40 + window 600 = 828 tokens, rounded up with margin).
static constexpr int MAX_ENCODER_ROPE_LEN = 2048;

// Parameter structs matching src/kernels/nemotron_audio.metal.
typedef struct {
  // Row width in floats and nonzero to round outputs to BF16.
  uint32_t dim, round;
  // LayerNorm epsilon.
  float eps;
} AudLnParam;

typedef struct {
  // Valid rows, padded rows and width in floats.
  uint32_t rows, padded_rows, cols;
} AudPackParam;

typedef struct {
  // Active rows, output width, inner width and padded weight row count.
  uint32_t m, n, k, w_rows;
  // Activation identifier, bias flag, residual flag and BF16 round flag.
  uint32_t act, has_bias, has_res, round;
  // Per-tensor FP8 weight scale multiplied by 256.
  float scale;
} AudMmParam;

typedef struct {
  // Sequence rows, 64-aligned head-major stride, heads and head width.
  uint32_t rows, seq_stride, heads, head_dim;
  // Half of rotated dimensions and qk_norm flag.
  uint32_t half_rot, qk_norm;
  // Per-head LayerNorm epsilon.
  float eps;
} AudQkvParam;

typedef struct {
  // Sequence rows, 64-aligned head-major stride and head count.
  uint32_t rows, seq_stride, heads;
  // Attention scale multiplied by log2(e).
  float log2_scale;
} AudAttnParam;

typedef struct {
  // Encoder frame count, input channels and upsampling factor.
  uint32_t frames, in_ch, up_factor;
} AudUpsampleParam;

typedef struct {
  // Encoder frame count, temporal pooling factor and speaker count.
  uint32_t frames, factor, n_spk;
} AudPoolParam;

typedef struct {
  // Frame count, ASR feature width and speaker count.
  uint32_t rows, dim, n_spk;
  // LayerNorm epsilon, speaker activity threshold and kernel infusion scale.
  float eps, threshold, scale;
} AudFuseParam;

// One linear projection weight matrix (either per-tensor FP8 or BF16) with
// optional FP32 bias vector.
typedef struct {
  // Raw weight buffer (E4M3 [n, k] when is_fp8, or BF16 [w_rows, k]).
  Buf w;
  // Optional FP32 bias buffer [n].
  Buf bias;
  // Output rows, inner width and 64-aligned weight rows in w.
  int n, k, w_rows;
  // Nonzero for per-tensor FP8 weights, zero for BF16 weights.
  int is_fp8;
  // Nonzero when bias is present.
  int has_bias;
  // Per-tensor FP8 weight scale multiplied by 256.
  float fp8_scale;
} AudLinear;

// Weights for one ASR or Sortformer Transformer block.
typedef struct {
  // Pre-attention and pre-FFN LayerNorm weights and biases (FP32).
  Buf norm1_w, norm1_b, norm2_w, norm2_b;
  // Optional per-head Q and K LayerNorm weights and biases (FP32, ASR only).
  Buf q_norm_w, q_norm_b, k_norm_w, k_norm_b;
  // Fused QKV, output and FFN projections.
  AudLinear w_qkv, out_proj, ffn_up, ffn_down;
} AudBlock;

// Parameter bundle for filling a zero-padded BF16 weight matrix from disk.
typedef struct {
  // Source BF16 tensor on disk.
  Tensor src;
  // Padded row count and column count.
  int w_rows, k;
} PadBf16Param;

// Parameter bundle for filling a precomputed RoPE cosine or sine table.
typedef struct {
  // Maximum sequence length and half rotated dimension.
  int max_len, half_rot;
  // Nonzero to write sine values, zero to write cosine values.
  int is_sin;
} RopeFillParam;

// Parameter bundle for filling the sinusoidal speaker fusion matrix.
typedef struct {
  // Number of speaker channels and ASR feature width.
  int n_spk, dim;
} SinusoidFillParam;

// Parallel mel spectrogram job shared across worker threads.
typedef struct {
  // Pre-emphasized zero-padded waveform [pcm_samples + 2 * STFT_PAD].
  const float *padded;
  // Window weights [400].
  const float *window;
  // Mel filterbank [128, 257].
  const float *fb;
  // Destination log-mel matrix [frames, 128].
  float *out_mel;
  // Total mel frames and frames processed per parallel block.
  int frames, block;
} MelJob;

// Sortformer streaming cache state held on host between long-form windows.
typedef struct {
  // Retained speaker-cache embeddings [spkcache_len, 512] and probabilities
  // [spkcache_len, 8].
  float *spkcache, *spkcache_preds;
  // Number of valid frames currently in spkcache and whether compression ran.
  int spkcache_len, spkcache_compressed;
  // Rolling FIFO embeddings [cap, 512] and probabilities [cap, 8].
  float *fifo, *fifo_preds;
  // Number of valid frames currently in fifo and allocated capacity.
  int fifo_len, fifo_cap;
} DiarStreamState;

// Loaded Parallel Expert Perception Encoder state.
struct NemoAudioEncoder {
  // GPU handle and owning memory account and workspace slot.
  Gpu *gpu;
  MemoryAccount *account;
  MemorySlot *work;
  // Open perception.safetensors index.
  TensorIndex *index;
  // Reloadable weight group holding all GPU perception tensors.
  WeightGroup *weights;

  // Compiled Metal kernel identifiers from src/kernels/nemotron_audio.metal.
  int k_ln, k_pack, k_mm_fp8, k_mm_bf16, k_qkv, k_attn64, k_attn128;
  int k_upsample, k_pool, k_fuse;

  // Host-resident frontend Hann window [400], mel filterbank [128, 257] and
  // learnable Sortformer silence embedding [512].
  float window[NEMO_AUDIO_WIN_LEN];
  float fb[NEMO_AUDIO_N_MELS * NEMO_AUDIO_N_BINS];
  float learnable_sil_emb[NEMO_AUDIO_DIAR_DIM];

  // Shared RoPE cosine and sine tables (both branches rotate 64 dims).
  Buf rope_cos, rope_sin;
  // Fixed sinusoidal speaker fusion matrix [8, 1280] in FP32 (BF16-rounded).
  Buf diar_kernel;

  // ASR 32-layer Transformer weights.
  AudLinear asr_pre_proj;
  Buf asr_embed_nw, asr_embed_nb, asr_final_nw, asr_final_nb;
  AudBlock asr_layers[NEMO_AUDIO_ASR_LAYERS];

  // Sortformer 31-layer Transformer and speaker head weights.
  AudLinear diar_pre_proj;
  Buf diar_embed_nw, diar_embed_nb, diar_final_nw, diar_final_nb;
  AudBlock diar_layers[NEMO_AUDIO_DIAR_LAYERS];
  AudLinear diar_enc_proj;
  Buf diar_up_w, diar_up_b;
  AudLinear diar_h2h, diar_h2spks;

  // Speaker-ASR fusion norms and 1280->2688 modality projection.
  Buf asr_fuse_nw, asr_fuse_nb, diar_fuse_nw, diar_fuse_nb;
  AudLinear proj;
};

// Rounds v to nearest BF16 value (ties to even), matching GPU bf().
static inline float round_bf16(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Converts one IEEE BF16 bit pattern to FP32.
static inline float bf16_to_f32(uint16_t bits) {
  uint32_t u = (uint32_t)bits << BF16_SHIFT;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Rounds n up to a multiple of align (align must be positive).
static inline int align_up(int n, int align) {
  return ((n + align - 1) / align) * align;
}

// Reads exact byte count from a safetensors Tensor into dst.
static void read_tensor_bytes(Tensor t, void *dst) {
  uint8_t *p = dst;
  uint64_t off = t.off;
  uint64_t left = t.size;
  while (left > 0) {
    ssize_t n = pread(t.fd, p, (size_t)left, (off_t)off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("perception: short read on safetensors weight");
    p += n;
    off += (uint64_t)n;
    left -= (uint64_t)n;
  }
}

// Reads a BF16 or F32 tensor into an FP32 host array of count elements.
static void read_tensor_f32(Tensor t, float *dst, size_t count) {
  if (t.dtype == 'f') {
    if (t.size != count * sizeof(float))
      die("perception: unexpected F32 tensor size");
    read_tensor_bytes(t, dst);
  } else if (t.dtype == 'b') {
    if (t.size != count * sizeof(uint16_t))
      die("perception: unexpected BF16 tensor size");
    uint16_t *tmp = xmalloc(t.size);
    read_tensor_bytes(t, tmp);
    for (size_t i = 0; i < count; i++) dst[i] = bf16_to_f32(tmp[i]);
    free(tmp);
  } else {
    die("perception: unsupported tensor dtype %d", t.dtype);
  }
}

// Reads a scalar F32 tensor value from disk.
static float read_scalar_f32(Tensor t) {
  float v = 0.0f;
  read_tensor_f32(t, &v, 1);
  return v;
}

// ---- CPU Audio Frontend & Sortformer Streaming Helpers ----

int nemo_audio_mel_frames(int pcm_samples) {
  if (pcm_samples <= 0) return 0;
  return pcm_samples / NEMO_AUDIO_HOP_LEN;
}

int nemo_audio_token_count(int pcm_samples) {
  int frames = nemo_audio_mel_frames(pcm_samples);
  if (frames <= 0) return 0;
  return (frames + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
}

// In-place radix-2 Cooley-Tukey complex FFT of size 512 in FP64 using
// precomputed half-circle twiddle factors tw_r/tw_i[0..255].
static void fft512(double *re, double *im, const double *tw_r,
                   const double *tw_i) {
  static constexpr int N = NEMO_AUDIO_N_FFT;
  for (int i = 1, j = 0; i < N; i++) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      double ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (int len = 2; len <= N; len <<= 1) {
    int half = len >> 1;
    int stride = N / len;
    for (int i = 0; i < N; i += len) {
      for (int j = 0; j < half; j++) {
        double wr = tw_r[j * stride];
        double wi = tw_i[j * stride];
        int u_idx = i + j;
        int v_idx = i + j + half;
        double vr = re[v_idx] * wr - im[v_idx] * wi;
        double vi = re[v_idx] * wi + im[v_idx] * wr;
        re[v_idx] = re[u_idx] - vr;
        im[v_idx] = im[u_idx] - vi;
        re[u_idx] += vr;
        im[u_idx] += vi;
      }
    }
  }
}

// Worker callback computing log-mel frames for one block of frames.
static void mel_block_worker(void *ctx, size_t block_idx) {
  const MelJob *job = ctx;
  int start = (int)block_idx * job->block;
  int end = start + job->block;
  if (end > job->frames) end = job->frames;

  double tw_r[NEMO_AUDIO_N_FFT / 2], tw_i[NEMO_AUDIO_N_FFT / 2];
  for (int k = 0; k < NEMO_AUDIO_N_FFT / 2; k++) {
    double ang = -TWO_PI * (double)k / (double)NEMO_AUDIO_N_FFT;
    tw_r[k] = cos(ang);
    tw_i[k] = sin(ang);
  }

  double re[NEMO_AUDIO_N_FFT], im[NEMO_AUDIO_N_FFT];
  double power[NEMO_AUDIO_N_BINS];
  for (int f = start; f < end; f++) {
    const float *frame = job->padded + (size_t)f * NEMO_AUDIO_HOP_LEN;
    memset(re, 0, sizeof re);
    memset(im, 0, sizeof im);
    for (int i = 0; i < NEMO_AUDIO_WIN_LEN; i++)
      re[WIN_OFFSET + i] = (double)(frame[WIN_OFFSET + i] * job->window[i]);
    fft512(re, im, tw_r, tw_i);
    for (int k = 0; k < NEMO_AUDIO_N_BINS; k++)
      power[k] = re[k] * re[k] + im[k] * im[k];
    float *dst = job->out_mel + (size_t)f * NEMO_AUDIO_N_MELS;
    for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
      const float *fb_row = job->fb + (size_t)m * NEMO_AUDIO_N_BINS;
      double energy = 0.0;
      for (int k = 0; k < NEMO_AUDIO_N_BINS; k++)
        energy += (double)fb_row[k] * power[k];
      dst[m] = (float)log(energy + (double)LOG_ZERO_GUARD);
    }
  }
}

void nemo_audio_mel_spectrogram(const float *pcm, int pcm_samples,
                                const float *window, const float *fb,
                                float *out_mel) {
  int frames = nemo_audio_mel_frames(pcm_samples);
  if (frames <= 0) return;

  size_t pad_len = (size_t)pcm_samples + 2u * STFT_PAD;
  float *padded = xmalloc(pad_len * sizeof(float));
  memset(padded, 0, (size_t)STFT_PAD * sizeof(float));
  memset(padded + STFT_PAD + pcm_samples, 0, (size_t)STFT_PAD * sizeof(float));
  padded[STFT_PAD] = pcm[0];
  for (int i = 1; i < pcm_samples; i++)
    padded[STFT_PAD + i] = pcm[i] - PREEMPH_COEFF * pcm[i - 1];

  static constexpr int FRAMES_PER_BLOCK = 64;
  int nblocks = (frames + FRAMES_PER_BLOCK - 1) / FRAMES_PER_BLOCK;
  MelJob job = {.padded = padded,
                .window = window,
                .fb = fb,
                .out_mel = out_mel,
                .frames = frames,
                .block = FRAMES_PER_BLOCK};
  if (nblocks <= 1)
    mel_block_worker(&job, 0);
  else
    threads_apply((size_t)nblocks, THREADS_INITIATED, &job, mel_block_worker);
  free(padded);
}

void nemo_audio_normalize_per_feature(const float *mel, int frames,
                                      float *out) {
  if (frames <= 0) return;
  for (int m = 0; m < NEMO_AUDIO_N_MELS; m++) {
    double ref = (double)mel[m];
    double sum = 0.0;
    for (int f = 0; f < frames; f++)
      sum += (double)mel[(size_t)f * NEMO_AUDIO_N_MELS + m] - ref;
    double mean = ref + sum / (double)frames;
    double std_v = 0.0;
    if (frames > 1) {
      double var_sum = 0.0;
      for (int f = 0; f < frames; f++) {
        double d = (double)mel[(size_t)f * NEMO_AUDIO_N_MELS + m] - mean;
        var_sum += d * d;
      }
      std_v = sqrt(var_sum / (double)(frames - 1));
    }
    double inv = 1.0 / (std_v + (double)NORM_EPS);
    for (int f = 0; f < frames; f++) {
      double v = ((double)mel[(size_t)f * NEMO_AUDIO_N_MELS + m] - mean) * inv;
      out[(size_t)f * NEMO_AUDIO_N_MELS + m] = (float)v;
    }
  }
}

void nemo_audio_stack_frames(const float *mel, int frames, float *out_stacked) {
  if (frames <= 0) return;
  int out_frames = (frames + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
  memset(out_stacked, 0,
         (size_t)out_frames * NEMO_AUDIO_STACK_DIM * sizeof(float));
  for (int t = 0; t < out_frames; t++) {
    float *dst = out_stacked + (size_t)t * NEMO_AUDIO_STACK_DIM;
    for (int s = 0; s < NEMO_AUDIO_SUBSAMPLE; s++) {
      int src_f = t * NEMO_AUDIO_SUBSAMPLE + s;
      if (src_f >= frames) break;
      memcpy(dst + (size_t)s * NEMO_AUDIO_N_MELS,
             mel + (size_t)src_f * NEMO_AUDIO_N_MELS,
             (size_t)NEMO_AUDIO_N_MELS * sizeof(float));
    }
  }
}

void nemo_audio_build_sinusoid_kernel(float *out_kernel, int n_spk, int dim) {
  double log_base = log(SINUSOID_BASE);
  for (int s = 0; s < n_spk; s++) {
    float *row = out_kernel + (size_t)s * (size_t)dim;
    for (int d = 0; d < dim; d += 2) {
      double div_term = exp(-(double)d * (log_base / (double)dim));
      double angle = (double)s * div_term;
      row[d] = round_bf16((float)sin(angle));
      if (d + 1 < dim) row[d + 1] = round_bf16((float)cos(angle));
    }
  }
}

// Score-index pair used to select top-K Sortformer speaker-cache frames.
typedef struct {
  // Importance score.
  float score;
  // Flat frame-speaker index.
  int index;
} ScoreEntry;

// Sorts ScoreEntry records by descending score, preserving lower index on ties.
static int cmp_score_desc(const void *a, const void *b) {
  const ScoreEntry *x = a, *y = b;
  if (x->score > y->score) return -1;
  if (x->score < y->score) return 1;
  return (x->index > y->index) - (x->index < y->index);
}

// Sorts integer frame indices in ascending order.
static int cmp_int_asc(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

// Boosts the top n_boost scores for each speaker in scores [n_frames, n_spk].
static void boost_topk_scores(float *scores, int n_frames, int n_spk,
                              int n_boost, float scale_factor) {
  if (n_boost <= 0) return;
  if (n_boost > n_frames) n_boost = n_frames;
  float delta = -scale_factor * logf(SPK_BOOST_OFFSET);
  ScoreEntry *buf = xmalloc((size_t)n_frames * sizeof *buf);
  for (int s = 0; s < n_spk; s++) {
    for (int f = 0; f < n_frames; f++)
      buf[f] = (ScoreEntry){scores[(size_t)f * n_spk + s], f};
    qsort(buf, (size_t)n_frames, sizeof *buf, cmp_score_desc);
    for (int i = 0; i < n_boost; i++) {
      int f = buf[i].index;
      if (isfinite(scores[(size_t)f * n_spk + s]))
        scores[(size_t)f * n_spk + s] += delta;
    }
  }
  free(buf);
}

void nemo_audio_sortformer_compress_spkcache(
    const float *emb_seq, const float *preds, int n_frames, int emb_dim,
    int n_spk, int spkcache_len, int sil_per_spk, const float *sil_emb,
    float *out_embs, float *out_preds) {
  int spk_per = spkcache_len / n_spk - sil_per_spk;
  int strong_boost = (int)floorf((float)spk_per * SPK_STRONG_BOOST_RATE);
  int weak_boost = (int)floorf((float)spk_per * SPK_WEAK_BOOST_RATE);
  int min_pos = (int)floorf((float)spk_per * SPK_MIN_POS_RATE);

  int padded_frames = n_frames + sil_per_spk;
  float *scores = xmalloc((size_t)padded_frames * n_spk * sizeof(float));

  // Compute log-odds scores preferring confident single-speaker frames.
  float neg_log_half = -logf(SPK_SPEECH_THRESH);
  for (int f = 0; f < n_frames; f++) {
    const float *pr = preds + (size_t)f * n_spk;
    float log1p_sum = 0.0f;
    float lp[NEMO_AUDIO_NUM_SPK], l1p[NEMO_AUDIO_NUM_SPK];
    for (int s = 0; s < n_spk; s++) {
      float p = pr[s] < SPK_PRED_SCORE_THRESH ? SPK_PRED_SCORE_THRESH : pr[s];
      float q = 1.0f - pr[s];
      if (q < SPK_PRED_SCORE_THRESH) q = SPK_PRED_SCORE_THRESH;
      lp[s] = logf(p);
      l1p[s] = logf(q);
      log1p_sum += l1p[s];
    }
    for (int s = 0; s < n_spk; s++)
      scores[(size_t)f * n_spk + s] = lp[s] - l1p[s] + log1p_sum + neg_log_half;
  }

  // Disable non-speech frames and overlapped non-positive frames when enough
  // single-speaker frames exist.
  for (int s = 0; s < n_spk; s++) {
    int pos_cnt = 0;
    for (int f = 0; f < n_frames; f++) {
      float *sc = &scores[(size_t)f * n_spk + s];
      if (preds[(size_t)f * n_spk + s] <= SPK_SPEECH_THRESH)
        *sc = -INFINITY;
      else if (*sc > 0.0f)
        pos_cnt++;
    }
    if (pos_cnt >= min_pos) {
      for (int f = 0; f < n_frames; f++) {
        float *sc = &scores[(size_t)f * n_spk + s];
        if (isfinite(*sc) && *sc <= 0.0f) *sc = -INFINITY;
      }
    }
  }

  // Boost newly appended frames beyond spkcache_len.
  for (int f = spkcache_len; f < n_frames; f++) {
    for (int s = 0; s < n_spk; s++) {
      float *sc = &scores[(size_t)f * n_spk + s];
      if (isfinite(*sc)) *sc += SPK_BOOST_LATEST;
    }
  }

  // Strong and weak per-speaker boosting.
  boost_topk_scores(scores, n_frames, n_spk, strong_boost, 2.0f);
  boost_topk_scores(scores, n_frames, n_spk, weak_boost, 1.0f);

  // Append silence placeholder slots with +inf score.
  for (int f = n_frames; f < padded_frames; f++)
    for (int s = 0; s < n_spk; s++) scores[(size_t)f * n_spk + s] = INFINITY;

  // Flatten in speaker-major order [n_spk, padded_frames] and take top-K.
  int total_flat = n_spk * padded_frames;
  ScoreEntry *flat = xmalloc((size_t)total_flat * sizeof *flat);
  for (int s = 0; s < n_spk; s++) {
    for (int f = 0; f < padded_frames; f++) {
      int flat_idx = s * padded_frames + f;
      flat[flat_idx] = (ScoreEntry){scores[(size_t)f * n_spk + s], flat_idx};
    }
  }
  qsort(flat, (size_t)total_flat, sizeof *flat, cmp_score_desc);

  int *sel = xmalloc((size_t)spkcache_len * sizeof(int));
  for (int i = 0; i < spkcache_len; i++)
    sel[i] = (flat[i].score == -INFINITY) ? SPK_MAX_INDEX : flat[i].index;
  qsort(sel, (size_t)spkcache_len, sizeof(int), cmp_int_asc);

  for (int i = 0; i < spkcache_len; i++) {
    float *dst_e = out_embs + (size_t)i * emb_dim;
    float *dst_p = out_preds + (size_t)i * n_spk;
    int disabled = (sel[i] == SPK_MAX_INDEX);
    int frame = disabled ? 0 : (sel[i] % padded_frames);
    if (frame >= n_frames) disabled = 1;
    if (disabled) {
      memcpy(dst_e, sil_emb, (size_t)emb_dim * sizeof(float));
      memset(dst_p, 0, (size_t)n_spk * sizeof(float));
    } else {
      memcpy(dst_e, emb_seq + (size_t)frame * emb_dim,
             (size_t)emb_dim * sizeof(float));
      memcpy(dst_p, preds + (size_t)frame * n_spk,
             (size_t)n_spk * sizeof(float));
    }
  }

  free(sel);
  free(flat);
  free(scores);
}

// ---- Weight Registration & Loading ----

// WeightFill callback copying a BF16 matrix [n, k] into a zero-padded BF16
// buffer [w_rows, k] so tiled matrix kernels always see 64-aligned rows.
static void fill_padded_bf16(Buf *buf, const void *ctx) {
  const PadBf16Param *p = ctx;
  memset(buf->ptr, 0, buf->size);
  read_tensor_bytes(p->src, buf->ptr);
}

// WeightFill callback generating BF16-rounded RoPE cosine or sine tables
// [max_len, half_rot] matching RotaryPositionalEncoding.create_pe.
static void fill_rope_table(Buf *buf, const void *ctx) {
  const RopeFillParam *p = ctx;
  float *dst = (float *)buf->ptr;
  int d_rot = 2 * p->half_rot;
  double *inv_freq = xmalloc((size_t)p->half_rot * sizeof(double));
  for (int j = 0; j < p->half_rot; j++)
    inv_freq[j] = 1.0 / pow(ROPE_BASE, (double)(2 * j) / (double)d_rot);
  for (int pos = 0; pos < p->max_len; pos++) {
    float *row = dst + (size_t)pos * p->half_rot;
    for (int j = 0; j < p->half_rot; j++) {
      double angle = (double)pos * inv_freq[j];
      row[j] = round_bf16((float)(p->is_sin ? sin(angle) : cos(angle)));
    }
  }
  free(inv_freq);
}

// WeightFill callback generating the sinusoidal speaker fusion matrix.
static void fill_sinusoid_kernel(Buf *buf, const void *ctx) {
  const SinusoidFillParam *p = ctx;
  nemo_audio_build_sinusoid_kernel((float *)buf->ptr, p->n_spk, p->dim);
}

// Registers one linear projection (FP8 or BF16) and optional bias in weights.
static void register_linear(WeightGroup *wg, const TensorIndex *idx,
                            AudLinear *lin, const char *prefix, int n, int k,
                            int expect_bias) {
  char name[PATH_CAP];
  snprintf(name, sizeof name, "%s.weight", prefix);
  Tensor tw = tensor_index_get(idx, "%s", name);
  if (tw.ndim != 2 || tw.shape[0] != n || tw.shape[1] != k)
    die("perception: %s shape mismatch (%lldx%lld, expected %dx%d)", name,
        (long long)tw.shape[0], (long long)tw.shape[1], n, k);

  lin->n = n;
  lin->k = k;
  lin->w_rows = align_up(n, (int)AUD_TILE);
  lin->has_bias = expect_bias;

  if (tw.dtype == '8') {
    lin->is_fp8 = 1;
    snprintf(name, sizeof name, "%s.weight_scale", prefix);
    float w_scale = read_scalar_f32(tensor_index_get(idx, "%s", name));
    lin->fp8_scale = w_scale * FP8_UNDO_SHIFT;
    int aligned =
        tw.size >= ALIGN_READ_MIN_BYTES && (tw.off & ALIGN_MASK) == 0 ? 1 : 0;
    weight_group_add_tensor(wg, &lin->w, tw, aligned);
  } else if (tw.dtype == 'b') {
    lin->is_fp8 = 0;
    lin->fp8_scale = 0.0f;
    if (lin->w_rows == n) {
      int aligned =
          tw.size >= ALIGN_READ_MIN_BYTES && (tw.off & ALIGN_MASK) == 0 ? 1 : 0;
      weight_group_add_tensor(wg, &lin->w, tw, aligned);
    } else {
      PadBf16Param pp = {.src = tw, .w_rows = lin->w_rows, .k = k};
      weight_group_add_generated(
          wg, &lin->w, (size_t)lin->w_rows * (size_t)k * sizeof(uint16_t),
          fill_padded_bf16, &pp, sizeof pp);
    }
  } else {
    die("perception: %s unexpected dtype %d", name, tw.dtype);
  }

  if (expect_bias) {
    snprintf(name, sizeof name, "%s.bias", prefix);
    Tensor tb = tensor_index_get(idx, "%s", name);
    if (tb.ndim != 1 || tb.shape[0] != n)
      die("perception: %s bias shape mismatch", name);
    weight_group_add_vector(wg, &lin->bias, tb, 0.0f, 0);
  }
}

// Registers LayerNorm weight and bias vectors in wg.
static void register_ln(WeightGroup *wg, const TensorIndex *idx, Buf *w, Buf *b,
                        const char *prefix, int dim) {
  Tensor tw = tensor_index_get(idx, "%s.weight", prefix);
  Tensor tb = tensor_index_get(idx, "%s.bias", prefix);
  if (tw.ndim != 1 || tw.shape[0] != dim || tb.ndim != 1 || tb.shape[0] != dim)
    die("perception: %s norm shape mismatch (expected %d)", prefix, dim);
  weight_group_add_vector(wg, w, tw, 0.0f, 0);
  weight_group_add_vector(wg, b, tb, 0.0f, 0);
}

NemoAudioEncoder *nemo_audio_open(Gpu *gpu, MemoryAccount *account,
                                  MemorySlot *work, const char *dir) {
  char path[PATH_CAP];
  snprintf(path, sizeof path, "%s/perception.safetensors", dir);
  NemoAudioEncoder *enc = xmalloc(sizeof *enc);
  *enc = (NemoAudioEncoder){.gpu = gpu, .account = account, .work = work};
  enc->index = tensor_index_open_file(path);
  enc->weights = weight_group_new(gpu, account, "nemotron perception");

  // Resolve compiled Metal kernels.
  enc->k_ln = gpu_find_kernel(gpu, "nemo_aud_layernorm");
  enc->k_pack = gpu_find_kernel(gpu, "nemo_aud_f32_to_bf16");
  enc->k_mm_fp8 = gpu_find_kernel(gpu, "nemo_aud_mm_fp8");
  enc->k_mm_bf16 = gpu_find_kernel(gpu, "nemo_aud_mm_bf16");
  enc->k_qkv = gpu_find_kernel(gpu, "nemo_aud_qkv_prep");
  enc->k_attn64 = gpu_find_kernel(gpu, "nemo_aud_flash_attn_64");
  enc->k_attn128 = gpu_find_kernel(gpu, "nemo_aud_flash_attn_128");
  enc->k_upsample = gpu_find_kernel(gpu, "nemo_aud_subpixel_upsample");
  enc->k_pool = gpu_find_kernel(gpu, "nemo_aud_diar_pool");
  enc->k_fuse = gpu_find_kernel(gpu, "nemo_aud_fuse");

  // Load small CPU-resident frontend buffers and silence embedding.
  read_tensor_f32(
      tensor_index_get(enc->index, "perception.preprocessor.featurizer.window"),
      enc->window, NEMO_AUDIO_WIN_LEN);
  read_tensor_f32(
      tensor_index_get(enc->index, "perception.preprocessor.featurizer.fb"),
      enc->fb, (size_t)NEMO_AUDIO_N_MELS * NEMO_AUDIO_N_BINS);
  read_tensor_f32(tensor_index_get(
                      enc->index,
                      "perception.encoder.diarization_model.sortformer_modules."
                      "learnable_sil_emb"),
                  enc->learnable_sil_emb, NEMO_AUDIO_DIAR_DIM);

  // Register generated RoPE tables and sinusoidal speaker fusion kernel.
  RopeFillParam rcos = {.max_len = MAX_ENCODER_ROPE_LEN,
                        .half_rot = NEMO_AUDIO_ASR_HALF_ROT,
                        .is_sin = 0};
  RopeFillParam rsin = {.max_len = MAX_ENCODER_ROPE_LEN,
                        .half_rot = NEMO_AUDIO_ASR_HALF_ROT,
                        .is_sin = 1};
  size_t rope_bytes =
      (size_t)MAX_ENCODER_ROPE_LEN * NEMO_AUDIO_ASR_HALF_ROT * sizeof(float);
  weight_group_add_generated(enc->weights, &enc->rope_cos, rope_bytes,
                             fill_rope_table, &rcos, sizeof rcos);
  weight_group_add_generated(enc->weights, &enc->rope_sin, rope_bytes,
                             fill_rope_table, &rsin, sizeof rsin);

  SinusoidFillParam sp = {.n_spk = NEMO_AUDIO_NUM_SPK,
                          .dim = NEMO_AUDIO_ASR_DIM};
  weight_group_add_generated(enc->weights, &enc->diar_kernel,
                             (size_t)NEMO_AUDIO_NUM_SPK * NEMO_AUDIO_ASR_DIM *
                                 sizeof(float),
                             fill_sinusoid_kernel, &sp, sizeof sp);

  // Register 32-layer ASR Transformer weights.
  register_linear(enc->weights, enc->index, &enc->asr_pre_proj,
                  "perception.encoder.asr_encoder.pre_encode.proj",
                  NEMO_AUDIO_ASR_DIM, NEMO_AUDIO_STACK_DIM, 0);
  register_ln(enc->weights, enc->index, &enc->asr_embed_nw, &enc->asr_embed_nb,
              "perception.encoder.asr_encoder.embed_norm", NEMO_AUDIO_ASR_DIM);
  register_ln(enc->weights, enc->index, &enc->asr_final_nw, &enc->asr_final_nb,
              "perception.encoder.asr_encoder.final_norm", NEMO_AUDIO_ASR_DIM);
  for (int l = 0; l < NEMO_AUDIO_ASR_LAYERS; l++) {
    AudBlock *b = &enc->asr_layers[l];
    char pfx[PATH_CAP];
    snprintf(pfx, sizeof pfx, "perception.encoder.asr_encoder.layers.%d.norm1",
             l);
    register_ln(enc->weights, enc->index, &b->norm1_w, &b->norm1_b, pfx,
                NEMO_AUDIO_ASR_DIM);
    snprintf(pfx, sizeof pfx, "perception.encoder.asr_encoder.layers.%d.norm2",
             l);
    register_ln(enc->weights, enc->index, &b->norm2_w, &b->norm2_b, pfx,
                NEMO_AUDIO_ASR_DIM);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.attn.q_norm", l);
    register_ln(enc->weights, enc->index, &b->q_norm_w, &b->q_norm_b, pfx,
                NEMO_AUDIO_ASR_HEAD_DIM);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.attn.k_norm", l);
    register_ln(enc->weights, enc->index, &b->k_norm_w, &b->k_norm_b, pfx,
                NEMO_AUDIO_ASR_HEAD_DIM);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.attn.w_qkv", l);
    register_linear(enc->weights, enc->index, &b->w_qkv, pfx,
                    3 * NEMO_AUDIO_ASR_DIM, NEMO_AUDIO_ASR_DIM, 0);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.attn.out_proj", l);
    register_linear(enc->weights, enc->index, &b->out_proj, pfx,
                    NEMO_AUDIO_ASR_DIM, NEMO_AUDIO_ASR_DIM, 1);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.ffn.net.0", l);
    register_linear(enc->weights, enc->index, &b->ffn_up, pfx,
                    NEMO_AUDIO_ASR_FFN, NEMO_AUDIO_ASR_DIM, 1);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.asr_encoder.layers.%d.ffn.net.3", l);
    register_linear(enc->weights, enc->index, &b->ffn_down, pfx,
                    NEMO_AUDIO_ASR_DIM, NEMO_AUDIO_ASR_FFN, 1);
  }

  // Register 31-layer Sortformer Transformer and speaker head weights.
  register_linear(
      enc->weights, enc->index, &enc->diar_pre_proj,
      "perception.encoder.diarization_model.encoder.pre_encode.proj",
      NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_STACK_DIM, 0);
  register_ln(enc->weights, enc->index, &enc->diar_embed_nw,
              &enc->diar_embed_nb,
              "perception.encoder.diarization_model.encoder.embed_norm",
              NEMO_AUDIO_DIAR_DIM);
  register_ln(enc->weights, enc->index, &enc->diar_final_nw,
              &enc->diar_final_nb,
              "perception.encoder.diarization_model.encoder.final_norm",
              NEMO_AUDIO_DIAR_DIM);
  for (int l = 0; l < NEMO_AUDIO_DIAR_LAYERS; l++) {
    AudBlock *b = &enc->diar_layers[l];
    char pfx[PATH_CAP];
    snprintf(pfx, sizeof pfx,
             "perception.encoder.diarization_model.encoder.layers.%d.norm1", l);
    register_ln(enc->weights, enc->index, &b->norm1_w, &b->norm1_b, pfx,
                NEMO_AUDIO_DIAR_DIM);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.diarization_model.encoder.layers.%d.norm2", l);
    register_ln(enc->weights, enc->index, &b->norm2_w, &b->norm2_b, pfx,
                NEMO_AUDIO_DIAR_DIM);
    snprintf(
        pfx, sizeof pfx,
        "perception.encoder.diarization_model.encoder.layers.%d.attn.w_qkv", l);
    register_linear(enc->weights, enc->index, &b->w_qkv, pfx,
                    3 * NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_DIAR_DIM, 0);
    snprintf(
        pfx, sizeof pfx,
        "perception.encoder.diarization_model.encoder.layers.%d.attn.out_proj",
        l);
    register_linear(enc->weights, enc->index, &b->out_proj, pfx,
                    NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_DIAR_DIM, 1);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.diarization_model.encoder.layers.%d.ffn.net.0",
             l);
    register_linear(enc->weights, enc->index, &b->ffn_up, pfx,
                    NEMO_AUDIO_DIAR_FFN, NEMO_AUDIO_DIAR_DIM, 1);
    snprintf(pfx, sizeof pfx,
             "perception.encoder.diarization_model.encoder.layers.%d.ffn.net.3",
             l);
    register_linear(enc->weights, enc->index, &b->ffn_down, pfx,
                    NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_DIAR_FFN, 1);
  }

  register_linear(
      enc->weights, enc->index, &enc->diar_enc_proj,
      "perception.encoder.diarization_model.sortformer_modules.encoder_proj",
      NEMO_AUDIO_DIAR_TF_DIM, NEMO_AUDIO_DIAR_DIM, 1);
  Tensor tup_w = tensor_index_get(
      enc->index, "perception.encoder.diarization_model.sortformer_modules."
                  "subpixel_upsample.weight");
  Tensor tup_b = tensor_index_get(
      enc->index, "perception.encoder.diarization_model.sortformer_modules."
                  "subpixel_upsample.bias");
  int up_out = NEMO_AUDIO_UP_FACTOR * NEMO_AUDIO_DIAR_TF_DIM;
  if (tup_w.dtype != 'b' || tup_w.ndim != 3 || tup_w.shape[0] != up_out ||
      tup_w.shape[1] != NEMO_AUDIO_DIAR_TF_DIM ||
      tup_w.shape[2] != NEMO_AUDIO_CONV_K)
    die("perception: subpixel_upsample.weight shape mismatch");
  weight_group_add_tensor(enc->weights, &enc->diar_up_w, tup_w, 0);
  weight_group_add_vector(enc->weights, &enc->diar_up_b, tup_b, 0.0f, 0);

  register_linear(enc->weights, enc->index, &enc->diar_h2h,
                  "perception.encoder.diarization_model.sortformer_modules."
                  "first_hidden_to_hidden",
                  NEMO_AUDIO_DIAR_TF_DIM, NEMO_AUDIO_DIAR_TF_DIM, 1);
  register_linear(enc->weights, enc->index, &enc->diar_h2spks,
                  "perception.encoder.diarization_model.sortformer_modules."
                  "single_hidden_to_spks",
                  NEMO_AUDIO_NUM_SPK, NEMO_AUDIO_DIAR_TF_DIM, 1);

  // Register fusion norms and 1280->2688 modality projection.
  register_ln(enc->weights, enc->index, &enc->asr_fuse_nw, &enc->asr_fuse_nb,
              "perception.encoder.asr_norm", NEMO_AUDIO_ASR_DIM);
  register_ln(enc->weights, enc->index, &enc->diar_fuse_nw, &enc->diar_fuse_nb,
              "perception.encoder.diar_norm", NEMO_AUDIO_NUM_SPK);
  register_linear(enc->weights, enc->index, &enc->proj, "perception.proj",
                  NEMO_AUDIO_OUT_DIM, NEMO_AUDIO_ASR_DIM, 1);

  return enc;
}

void nemo_audio_close(NemoAudioEncoder *enc) {
  if (!enc) return;
  weight_group_free(enc->weights);
  tensor_index_close(enc->index);
  free(enc);
}

void nemo_audio_pin(NemoAudioEncoder *enc, int active) {
  if (!enc) return;
  memory_pin(weight_group_slot(enc->weights), active);
}

int nemo_audio_is_loaded(const NemoAudioEncoder *enc) {
  return enc && weight_group_resident_bytes(enc->weights) > 0;
}

MemorySlot *nemo_audio_weight_slot(NemoAudioEncoder *enc) {
  return enc ? weight_group_slot(enc->weights) : NULL;
}

uint64_t nemo_audio_weight_bytes(const NemoAudioEncoder *enc) {
  return enc ? weight_group_required_bytes(enc->weights) : 0;
}

// ---- GPU Dispatch Helpers ----

// Dispatches LayerNorm over rows of width dim.
static void run_ln(GpuCommandBuffer *cb, const NemoAudioEncoder *enc,
                   const Buf *x, const Buf *w, const Buf *b, const Buf *y,
                   int rows, int dim, int round) {
  AudLnParam p = {
      .dim = (uint32_t)dim, .round = (uint32_t)round, .eps = LN_EPS};
  gpu_select_kernel(cb, enc->k_ln);
  gpu_bind_buffer(cb, 0, x, 0);
  gpu_bind_buffer(cb, 1, w, 0);
  gpu_bind_buffer(cb, 2, b, 0);
  gpu_bind_buffer(cb, 3, y, 0);
  gpu_copy_bytes(cb, 4, &p, sizeof p);
  gpu_run_kernel(cb, (uint32_t)rows, 1, 1, AUD_ROW_THREADS, 1, 1);
}

// Dispatches a linear projection (per-tensor FP8 or BF16) with optional bias,
// activation and residual add. x_bf16 scratch is used only for FP8 weights.
static void run_linear(GpuCommandBuffer *cb, const NemoAudioEncoder *enc,
                       const AudLinear *lin, const Buf *x, const Buf *x_bf16,
                       const Buf *res, const Buf *y, int m, uint32_t act,
                       int round) {
  AudMmParam p = {.m = (uint32_t)m,
                  .n = (uint32_t)lin->n,
                  .k = (uint32_t)lin->k,
                  .w_rows = (uint32_t)lin->w_rows,
                  .act = act,
                  .has_bias = (uint32_t)lin->has_bias,
                  .has_res = res ? 1u : 0u,
                  .round = (uint32_t)round,
                  .scale = lin->fp8_scale};
  const Buf *bias_b = lin->has_bias ? &lin->bias : y;
  const Buf *res_b = res ? res : y;

  if (lin->is_fp8) {
    int m_pad = align_up(m, (int)AUD_FP8_M_TILE);
    AudPackParam pp = {.rows = (uint32_t)m,
                       .padded_rows = (uint32_t)m_pad,
                       .cols = (uint32_t)lin->k};
    gpu_select_kernel(cb, enc->k_pack);
    gpu_bind_buffer(cb, 0, x, 0);
    gpu_bind_buffer(cb, 1, x_bf16, 0);
    gpu_copy_bytes(cb, 2, &pp, sizeof pp);
    uint32_t cols4 = (uint32_t)(lin->k / 4);
    uint32_t gx = (cols4 + AUD_PACK_THREADS - 1u) / AUD_PACK_THREADS;
    gpu_run_kernel(cb, gx, (uint32_t)m_pad, 1, AUD_PACK_THREADS, 1, 1);

    uint32_t tiles_m = (uint32_t)(m_pad / (int)AUD_FP8_M_TILE);
    uint32_t tiles_n = (uint32_t)((lin->n + (int)AUD_TILE - 1) / (int)AUD_TILE);
    gpu_select_kernel(cb, enc->k_mm_fp8);
    gpu_bind_buffer(cb, 0, &lin->w, 0);
    gpu_bind_buffer(cb, 1, x_bf16, 0);
    gpu_bind_buffer(cb, 2, bias_b, 0);
    gpu_bind_buffer(cb, 3, res_b, 0);
    gpu_bind_buffer(cb, 4, y, 0);
    gpu_copy_bytes(cb, 5, &p, sizeof p);
    gpu_run_kernel(cb, tiles_m * tiles_n, 1, 1, AUD_MM_THREADS, 1, 1);
  } else {
    uint32_t tiles_m = (uint32_t)((m + (int)AUD_TILE - 1) / (int)AUD_TILE);
    uint32_t tiles_n = (uint32_t)((lin->n + (int)AUD_TILE - 1) / (int)AUD_TILE);
    gpu_select_kernel(cb, enc->k_mm_bf16);
    gpu_bind_buffer(cb, 0, &lin->w, 0);
    gpu_bind_buffer(cb, 1, x, 0);
    gpu_bind_buffer(cb, 2, bias_b, 0);
    gpu_bind_buffer(cb, 3, res_b, 0);
    gpu_bind_buffer(cb, 4, y, 0);
    gpu_copy_bytes(cb, 5, &p, sizeof p);
    gpu_run_kernel(cb, tiles_m * tiles_n, 1, 1, AUD_MM_THREADS, 1, 1);
  }
}

// Temporary GPU scratch buffers for one ASR or Sortformer encoder window.
typedef struct {
  // Stacked or pre-encoded input, residual stream, normalized activations and
  // FFN / QKV / attention buffers.
  Buf in, h, norm, qkv, q_hm, k_hm, v_hm, attn_out, ffn, x_bf16;
  // Sortformer head buffers (projected 192-wide states, upsampled ReLU states,
  // intermediate 192-wide states, 10 ms logits and 80 ms pooled probabilities).
  Buf diar_proj, diar_up, diar_h1, diar_logits, diar_probs;
  // Final fusion and 2688-wide output projection buffers.
  Buf fuse_asr, fuse_probs, fused, proj_out;
} AudScratch;

// Frees all buffers in s after GPU work completes.
static void free_scratch(AudScratch *s) {
  Buf *list[] = {&s->in,         &s->h,           &s->norm,       &s->qkv,
                 &s->q_hm,       &s->k_hm,        &s->v_hm,       &s->attn_out,
                 &s->ffn,        &s->x_bf16,      &s->diar_proj,  &s->diar_up,
                 &s->diar_h1,    &s->diar_logits, &s->diar_probs, &s->fuse_asr,
                 &s->fuse_probs, &s->fused,       &s->proj_out};
  for (size_t i = 0; i < sizeof list / sizeof *list; i++)
    if (list[i]->obj) gpu_free(list[i]);
}

// Tries to allocate one GPU scratch buffer under slot; returns 0 on failure.
static int try_buf(Gpu *gpu, MemorySlot *slot, Buf *b, size_t bytes) {
  *b = gpu_try_alloc_required(gpu, slot, bytes);
  return b->obj != NULL;
}

// Runs the N-layer TransformerEncoder (either ASR or Sortformer) on GPU over
// tokens rows in s->h (when bypass_pre_encode) or stacked mel frames in s->in.
// Leaves final_norm output in s->h.
static void run_transformer_window(
    const NemoAudioEncoder *enc, const AudScratch *s, int tokens, int d_model,
    int heads, int head_dim, int half_rot, int qk_norm, int n_layers,
    int bypass_pre_encode, const AudLinear *pre_proj, const Buf *embed_nw,
    const Buf *embed_nb, const Buf *final_nw, const Buf *final_nb,
    const AudBlock *layers, const PassWait *wait) {
  int seq_stride = align_up(tokens, (int)AUD_TILE);
  int attn_kernel =
      (head_dim == NEMO_AUDIO_ASR_HEAD_DIM) ? enc->k_attn128 : enc->k_attn64;
  float log2_scale = (float)(LOG2_E / sqrt((double)head_dim));

  GpuCommandBuffer cb = gpu_begin(enc->gpu);
  if (!bypass_pre_encode)
    run_linear(&cb, enc, pre_proj, &s->in, &s->x_bf16, NULL, &s->h, tokens,
               AUD_ACT_NONE, 1);
  run_ln(&cb, enc, &s->h, embed_nw, embed_nb, &s->h, tokens, d_model, 1);
  gpu_commit(&cb);

  for (int l = 0; l < n_layers; l++) {
    if (wait) pass_wait(wait);
    const AudBlock *blk = &layers[l];
    cb = gpu_begin(enc->gpu);

    // Pre-attention LayerNorm -> fused QKV -> optional QK-norm + RoPE ->
    // FlashAttention -> out_proj + residual.
    run_ln(&cb, enc, &s->h, &blk->norm1_w, &blk->norm1_b, &s->norm, tokens,
           d_model, 1);
    run_linear(&cb, enc, &blk->w_qkv, &s->norm, &s->x_bf16, NULL, &s->qkv,
               tokens, AUD_ACT_NONE, 1);

    AudQkvParam qp = {.rows = (uint32_t)tokens,
                      .seq_stride = (uint32_t)seq_stride,
                      .heads = (uint32_t)heads,
                      .head_dim = (uint32_t)head_dim,
                      .half_rot = (uint32_t)half_rot,
                      .qk_norm = (uint32_t)qk_norm,
                      .eps = LN_EPS};
    gpu_select_kernel(&cb, enc->k_qkv);
    gpu_bind_buffer(&cb, 0, &s->qkv, 0);
    gpu_bind_buffer(&cb, 1, qk_norm ? &blk->q_norm_w : &enc->rope_cos, 0);
    gpu_bind_buffer(&cb, 2, qk_norm ? &blk->q_norm_b : &enc->rope_cos, 0);
    gpu_bind_buffer(&cb, 3, qk_norm ? &blk->k_norm_w : &enc->rope_cos, 0);
    gpu_bind_buffer(&cb, 4, qk_norm ? &blk->k_norm_b : &enc->rope_cos, 0);
    gpu_bind_buffer(&cb, 5, &enc->rope_cos, 0);
    gpu_bind_buffer(&cb, 6, &enc->rope_sin, 0);
    gpu_bind_buffer(&cb, 7, &s->q_hm, 0);
    gpu_bind_buffer(&cb, 8, &s->k_hm, 0);
    gpu_bind_buffer(&cb, 9, &s->v_hm, 0);
    gpu_copy_bytes(&cb, 10, &qp, sizeof qp);
    gpu_run_kernel(&cb, (uint32_t)heads, (uint32_t)seq_stride, 1,
                   (uint32_t)head_dim, 1, 1);

    AudAttnParam ap = {.rows = (uint32_t)tokens,
                       .seq_stride = (uint32_t)seq_stride,
                       .heads = (uint32_t)heads,
                       .log2_scale = log2_scale};
    gpu_select_kernel(&cb, attn_kernel);
    gpu_bind_buffer(&cb, 0, &s->q_hm, 0);
    gpu_bind_buffer(&cb, 1, &s->k_hm, 0);
    gpu_bind_buffer(&cb, 2, &s->v_hm, 0);
    gpu_bind_buffer(&cb, 3, &s->attn_out, 0);
    gpu_copy_bytes(&cb, 4, &ap, sizeof ap);
    gpu_run_kernel(&cb, (uint32_t)(seq_stride / (int)AUD_TILE), (uint32_t)heads,
                   1, AUD_MM_THREADS, 1, 1);

    run_linear(&cb, enc, &blk->out_proj, &s->attn_out, &s->x_bf16, &s->h, &s->h,
               tokens, AUD_ACT_NONE, 1);

    // Pre-FFN LayerNorm -> up + exact GELU -> down + residual.
    run_ln(&cb, enc, &s->h, &blk->norm2_w, &blk->norm2_b, &s->norm, tokens,
           d_model, 1);
    run_linear(&cb, enc, &blk->ffn_up, &s->norm, &s->x_bf16, NULL, &s->ffn,
               tokens, AUD_ACT_GELU, 1);
    run_linear(&cb, enc, &blk->ffn_down, &s->ffn, &s->x_bf16, &s->h, &s->h,
               tokens, AUD_ACT_NONE, 1);

    gpu_commit(&cb);
  }

  cb = gpu_begin(enc->gpu);
  run_ln(&cb, enc, &s->h, final_nw, final_nb, &s->h, tokens, d_model, 1);
  gpu_commit(&cb);
}

// Runs the Sortformer post-encoder head (encoder_proj -> subpixel_upsample +
// ReLU -> first_hidden_to_hidden + ReLU -> single_hidden_to_spks -> sigmoid +
// 8x pooling) on s->h [tokens, 512], leaving [tokens, 8] probabilities in
// s->diar_probs.
static void run_sortformer_head(const NemoAudioEncoder *enc,
                                const AudScratch *s, int tokens) {
  int up_frames = tokens * NEMO_AUDIO_UP_FACTOR;
  int up_out_ch = NEMO_AUDIO_UP_FACTOR * NEMO_AUDIO_DIAR_TF_DIM;
  GpuCommandBuffer cb = gpu_begin(enc->gpu);

  run_linear(&cb, enc, &enc->diar_enc_proj, &s->h, &s->x_bf16, NULL,
             &s->diar_proj, tokens, AUD_ACT_NONE, 1);

  AudUpsampleParam up = {.frames = (uint32_t)tokens,
                         .in_ch = NEMO_AUDIO_DIAR_TF_DIM,
                         .up_factor = NEMO_AUDIO_UP_FACTOR};
  gpu_select_kernel(&cb, enc->k_upsample);
  gpu_bind_buffer(&cb, 0, &s->diar_proj, 0);
  gpu_bind_buffer(&cb, 1, &enc->diar_up_w, 0);
  gpu_bind_buffer(&cb, 2, &enc->diar_up_b, 0);
  gpu_bind_buffer(&cb, 3, &s->diar_up, 0);
  gpu_copy_bytes(&cb, 4, &up, sizeof up);
  uint32_t up_gx =
      ((uint32_t)up_out_ch + AUD_UPSAMPLE_THREADS - 1u) / AUD_UPSAMPLE_THREADS;
  gpu_run_kernel(&cb, up_gx, (uint32_t)tokens, 1, AUD_UPSAMPLE_THREADS, 1, 1);

  run_linear(&cb, enc, &enc->diar_h2h, &s->diar_up, &s->x_bf16, NULL,
             &s->diar_h1, up_frames, AUD_ACT_RELU, 1);
  run_linear(&cb, enc, &enc->diar_h2spks, &s->diar_h1, &s->x_bf16, NULL,
             &s->diar_logits, up_frames, AUD_ACT_NONE, 1);

  AudPoolParam pp = {.frames = (uint32_t)tokens,
                     .factor = NEMO_AUDIO_UP_FACTOR,
                     .n_spk = NEMO_AUDIO_NUM_SPK};
  gpu_select_kernel(&cb, enc->k_pool);
  gpu_bind_buffer(&cb, 0, &s->diar_logits, 0);
  gpu_bind_buffer(&cb, 1, &s->diar_probs, 0);
  gpu_copy_bytes(&cb, 2, &pp, sizeof pp);
  gpu_run_kernel(&cb, 1, (uint32_t)tokens, 1, NEMO_AUDIO_NUM_SPK, 1, 1);

  gpu_commit(&cb);
}

// Updates streaming Sortformer FIFO and speaker cache from one window's
// pre-encoded embeddings and predictions (synchronous batch=1 update matching
// SortformerModules.streaming_update).
static void diar_stream_update(DiarStreamState *st, const float *chunk_pre,
                               int chunk_len_total, const float *all_preds,
                               int lc_enc, int rc_enc, const float *sil_emb) {
  int s_len = st->spkcache_len;
  int f_len = st->fifo_len;
  int core = chunk_len_total - lc_enc - rc_enc;
  if (core <= 0) return;

  const float *fifo_preds_fresh =
      all_preds + (size_t)s_len * NEMO_AUDIO_NUM_SPK;
  const float *chunk_core_emb =
      chunk_pre + (size_t)lc_enc * NEMO_AUDIO_DIAR_DIM;
  const float *chunk_core_preds =
      all_preds + (size_t)(s_len + f_len + lc_enc) * NEMO_AUDIO_NUM_SPK;

  int new_fifo = f_len + core;
  if (new_fifo > st->fifo_cap) {
    st->fifo_cap = new_fifo * 2;
    float *nf = realloc(st->fifo, (size_t)st->fifo_cap * NEMO_AUDIO_DIAR_DIM *
                                      sizeof(float));
    float *np = realloc(st->fifo_preds, (size_t)st->fifo_cap *
                                            NEMO_AUDIO_NUM_SPK * sizeof(float));
    if (!nf || !np) die("out of memory");
    st->fifo = nf;
    st->fifo_preds = np;
  }
  if (f_len > 0)
    memcpy(st->fifo_preds, fifo_preds_fresh,
           (size_t)f_len * NEMO_AUDIO_NUM_SPK * sizeof(float));
  memcpy(st->fifo + (size_t)f_len * NEMO_AUDIO_DIAR_DIM, chunk_core_emb,
         (size_t)core * NEMO_AUDIO_DIAR_DIM * sizeof(float));
  memcpy(st->fifo_preds + (size_t)f_len * NEMO_AUDIO_NUM_SPK, chunk_core_preds,
         (size_t)core * NEMO_AUDIO_NUM_SPK * sizeof(float));
  st->fifo_len = new_fifo;

  if (st->fifo_len > NEMO_AUDIO_DIAR_FIFO_LEN) {
    int pop_len = NEMO_AUDIO_DIAR_UPDATE_PERIOD;
    int overflow = st->fifo_len - NEMO_AUDIO_DIAR_FIFO_LEN;
    if (overflow > pop_len) pop_len = overflow;
    if (pop_len > st->fifo_len) pop_len = st->fifo_len;

    int cand_len = s_len + pop_len;
    float *cand_e =
        xmalloc((size_t)cand_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
    float *cand_p =
        xmalloc((size_t)cand_len * NEMO_AUDIO_NUM_SPK * sizeof(float));
    if (s_len > 0) {
      memcpy(cand_e, st->spkcache,
             (size_t)s_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
      const float *old_p =
          st->spkcache_compressed ? st->spkcache_preds : all_preds;
      memcpy(cand_p, old_p, (size_t)s_len * NEMO_AUDIO_NUM_SPK * sizeof(float));
    }
    memcpy(cand_e + (size_t)s_len * NEMO_AUDIO_DIAR_DIM, st->fifo,
           (size_t)pop_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
    memcpy(cand_p + (size_t)s_len * NEMO_AUDIO_NUM_SPK, st->fifo_preds,
           (size_t)pop_len * NEMO_AUDIO_NUM_SPK * sizeof(float));

    int rem = st->fifo_len - pop_len;
    if (rem > 0) {
      memmove(st->fifo, st->fifo + (size_t)pop_len * NEMO_AUDIO_DIAR_DIM,
              (size_t)rem * NEMO_AUDIO_DIAR_DIM * sizeof(float));
      memmove(st->fifo_preds,
              st->fifo_preds + (size_t)pop_len * NEMO_AUDIO_NUM_SPK,
              (size_t)rem * NEMO_AUDIO_NUM_SPK * sizeof(float));
    }
    st->fifo_len = rem;

    if (cand_len > NEMO_AUDIO_DIAR_SPKCACHE_LEN) {
      nemo_audio_sortformer_compress_spkcache(
          cand_e, cand_p, cand_len, NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_NUM_SPK,
          NEMO_AUDIO_DIAR_SPKCACHE_LEN, NEMO_AUDIO_DIAR_SIL_PER_SPK, sil_emb,
          st->spkcache, st->spkcache_preds);
      st->spkcache_len = NEMO_AUDIO_DIAR_SPKCACHE_LEN;
      st->spkcache_compressed = 1;
    } else {
      memcpy(st->spkcache, cand_e,
             (size_t)cand_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
      memcpy(st->spkcache_preds, cand_p,
             (size_t)cand_len * NEMO_AUDIO_NUM_SPK * sizeof(float));
      st->spkcache_len = cand_len;
    }
    free(cand_e);
    free(cand_p);
  }
}

// Writes an error message into error when non-null.
static void set_audio_err(char *error, size_t cap, const char *msg) {
  if (error && cap > 0) snprintf(error, cap, "%s", msg);
}

// Packs count FP32 floats into a heap-allocated BF16 array (ties to even).
static uint16_t *pack_f32_to_bf16_alloc(const float *src, size_t count) {
  uint16_t *dst = xmalloc(count * sizeof(uint16_t));
  for (size_t i = 0; i < count; i++) {
    float r = round_bf16(src[i]);
    uint32_t u;
    memcpy(&u, &r, sizeof u);
    dst[i] = (uint16_t)(u >> BF16_SHIFT);
  }
  return dst;
}

int nemo_audio_encode_with_asr(NemoAudioEncoder *enc, const float *pcm,
                               int pcm_samples, const PassWait *wait,
                               float **out_embeds, uint16_t **out_asr_bf16,
                               int *out_tokens, char *error, size_t error_cap) {
  if (out_embeds) *out_embeds = NULL;
  if (out_asr_bf16) *out_asr_bf16 = NULL;
  if (out_tokens) *out_tokens = 0;
  if (!enc || !pcm || pcm_samples < NEMO_AUDIO_HOP_LEN) {
    set_audio_err(error, error_cap, "audio clip is too short to encode");
    return 0;
  }

  if (!weight_group_load(enc->weights)) {
    set_audio_err(error, error_cap,
                  "perception weights do not fit in available memory");
    return 0;
  }

  int mel_frames = nemo_audio_mel_frames(pcm_samples);
  int total_tokens = nemo_audio_token_count(pcm_samples);

  // 1. Compute raw log-mel spectrogram (for Sortformer) and per-feature
  //    normalized log-mel spectrogram (for ASR).
  float *mel_raw =
      xmalloc((size_t)mel_frames * NEMO_AUDIO_N_MELS * sizeof(float));
  float *mel_asr =
      xmalloc((size_t)mel_frames * NEMO_AUDIO_N_MELS * sizeof(float));
  nemo_audio_mel_spectrogram(pcm, pcm_samples, enc->window, enc->fb, mel_raw);
  nemo_audio_normalize_per_feature(mel_raw, mel_frames, mel_asr);

  int use_online = mel_frames > NEMO_AUDIO_ONLINE_MEL_FRAMES;
  int max_asr_win =
      use_online ? (NEMO_AUDIO_ONLINE_TOKENS + NEMO_AUDIO_CTX_LEFT_TOKENS +
                    NEMO_AUDIO_CTX_RIGHT_TOKENS)
                 : total_tokens;
  int max_diar_win = use_online ? (NEMO_AUDIO_DIAR_SPKCACHE_LEN +
                                   NEMO_AUDIO_DIAR_FIFO_LEN + max_asr_win)
                                : total_tokens;
  int max_tokens = max_asr_win > max_diar_win ? max_asr_win : max_diar_win;
  int max_seq_stride = align_up(max_tokens, (int)AUD_TILE);
  int max_fp8_pad = align_up(max_tokens, (int)AUD_FP8_M_TILE);
  int fuse_chunk = total_tokens < NEMO_AUDIO_ONLINE_TOKENS
                       ? total_tokens
                       : NEMO_AUDIO_ONLINE_TOKENS;

  AudScratch s = {0};
  int alloc_ok =
      try_buf(enc->gpu, enc->work, &s.in,
              (size_t)max_tokens * NEMO_AUDIO_STACK_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.h,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.norm,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.qkv,
              (size_t)max_tokens * 3u * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.q_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.k_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.v_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.attn_out,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.ffn,
              (size_t)max_tokens * NEMO_AUDIO_ASR_FFN * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.x_bf16,
              (size_t)max_fp8_pad * NEMO_AUDIO_ASR_FFN * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.diar_proj,
              (size_t)max_diar_win * NEMO_AUDIO_DIAR_TF_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.diar_up,
              (size_t)max_diar_win * NEMO_AUDIO_UP_FACTOR *
                  NEMO_AUDIO_DIAR_TF_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.diar_h1,
              (size_t)max_diar_win * NEMO_AUDIO_UP_FACTOR *
                  NEMO_AUDIO_DIAR_TF_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.diar_logits,
              (size_t)max_diar_win * NEMO_AUDIO_UP_FACTOR * NEMO_AUDIO_NUM_SPK *
                  sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.diar_probs,
              (size_t)max_diar_win * NEMO_AUDIO_NUM_SPK * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.fuse_asr,
              (size_t)fuse_chunk * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.fuse_probs,
              (size_t)fuse_chunk * NEMO_AUDIO_NUM_SPK * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.fused,
              (size_t)fuse_chunk * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.proj_out,
              (size_t)fuse_chunk * NEMO_AUDIO_OUT_DIM * sizeof(float));

  if (!alloc_ok) {
    free_scratch(&s);
    free(mel_raw);
    free(mel_asr);
    set_audio_err(error, error_cap,
                  "perception workspace does not fit in available memory");
    return 0;
  }

  float *asr_all =
      xmalloc((size_t)total_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float));
  float *probs_all =
      xmalloc((size_t)total_tokens * NEMO_AUDIO_NUM_SPK * sizeof(float));

  if (!use_online) {
    // 2A. Single-pass encoding (recordings <= 40 s).
    nemo_audio_stack_frames(mel_asr, mel_frames, (float *)s.in.ptr);
    run_transformer_window(enc, &s, total_tokens, NEMO_AUDIO_ASR_DIM,
                           NEMO_AUDIO_ASR_HEADS, NEMO_AUDIO_ASR_HEAD_DIM,
                           NEMO_AUDIO_ASR_HALF_ROT, 1, NEMO_AUDIO_ASR_LAYERS, 0,
                           &enc->asr_pre_proj, &enc->asr_embed_nw,
                           &enc->asr_embed_nb, &enc->asr_final_nw,
                           &enc->asr_final_nb, enc->asr_layers, wait);
    gpu_sync(enc->gpu);
    memcpy(asr_all, s.h.ptr,
           (size_t)total_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float));

    nemo_audio_stack_frames(mel_raw, mel_frames, (float *)s.in.ptr);
    run_transformer_window(enc, &s, total_tokens, NEMO_AUDIO_DIAR_DIM,
                           NEMO_AUDIO_DIAR_HEADS, NEMO_AUDIO_DIAR_HEAD_DIM,
                           NEMO_AUDIO_DIAR_HALF_ROT, 0, NEMO_AUDIO_DIAR_LAYERS,
                           0, &enc->diar_pre_proj, &enc->diar_embed_nw,
                           &enc->diar_embed_nb, &enc->diar_final_nw,
                           &enc->diar_final_nb, enc->diar_layers, wait);
    run_sortformer_head(enc, &s, total_tokens);
    gpu_sync(enc->gpu);
    memcpy(probs_all, s.diar_probs.ptr,
           (size_t)total_tokens * NEMO_AUDIO_NUM_SPK * sizeof(float));
  } else {
    // 2B. Long-form windowed ASR + streaming Sortformer (> 40 s).
    int chunk_feat = NEMO_AUDIO_ONLINE_MEL_FRAMES;
    int left_ctx_feat = NEMO_AUDIO_CTX_LEFT_TOKENS * NEMO_AUDIO_SUBSAMPLE;
    int right_ctx_feat = NEMO_AUDIO_CTX_RIGHT_TOKENS * NEMO_AUDIO_SUBSAMPLE;
    int num_chunks = (mel_frames + chunk_feat - 1) / chunk_feat;

    DiarStreamState st = {
        .spkcache = xmalloc((size_t)NEMO_AUDIO_DIAR_SPKCACHE_LEN *
                            NEMO_AUDIO_DIAR_DIM * sizeof(float)),
        .spkcache_preds = xmalloc((size_t)NEMO_AUDIO_DIAR_SPKCACHE_LEN *
                                  NEMO_AUDIO_NUM_SPK * sizeof(float)),
        .fifo_cap = NEMO_AUDIO_DIAR_FIFO_LEN + max_asr_win,
    };
    st.fifo =
        xmalloc((size_t)st.fifo_cap * NEMO_AUDIO_DIAR_DIM * sizeof(float));
    st.fifo_preds =
        xmalloc((size_t)st.fifo_cap * NEMO_AUDIO_NUM_SPK * sizeof(float));
    float *chunk_pre =
        xmalloc((size_t)max_asr_win * NEMO_AUDIO_DIAR_DIM * sizeof(float));

    for (int ci = 0; ci < num_chunks; ci++) {
      int start = ci * chunk_feat;
      int end = start + chunk_feat;
      if (end > mel_frames) end = mel_frames;
      int ctx_start = start - left_ctx_feat;
      if (ctx_start < 0) ctx_start = 0;
      int ctx_end = end + right_ctx_feat;
      if (ctx_end > mel_frames) ctx_end = mel_frames;

      int win_mel = ctx_end - ctx_start;
      int win_toks =
          (win_mel + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int left_off = start - ctx_start;
      int right_off = ctx_end - end;
      int left_drop = left_off / NEMO_AUDIO_SUBSAMPLE;
      int out_pos = (start + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int out_end = (end + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int core_len = out_end - out_pos;

      // ASR window.
      nemo_audio_stack_frames(mel_asr + (size_t)ctx_start * NEMO_AUDIO_N_MELS,
                              win_mel, (float *)s.in.ptr);
      run_transformer_window(enc, &s, win_toks, NEMO_AUDIO_ASR_DIM,
                             NEMO_AUDIO_ASR_HEADS, NEMO_AUDIO_ASR_HEAD_DIM,
                             NEMO_AUDIO_ASR_HALF_ROT, 1, NEMO_AUDIO_ASR_LAYERS,
                             0, &enc->asr_pre_proj, &enc->asr_embed_nw,
                             &enc->asr_embed_nb, &enc->asr_final_nw,
                             &enc->asr_final_nb, enc->asr_layers, wait);
      gpu_sync(enc->gpu);
      memcpy(asr_all + (size_t)out_pos * NEMO_AUDIO_ASR_DIM,
             (const float *)s.h.ptr + (size_t)left_drop * NEMO_AUDIO_ASR_DIM,
             (size_t)core_len * NEMO_AUDIO_ASR_DIM * sizeof(float));

      // Sortformer streaming step: pre_encode chunk, concat [spkcache, fifo,
      // chunk], run 31-layer encoder + head, slice core predictions, update
      // cache.
      nemo_audio_stack_frames(mel_raw + (size_t)ctx_start * NEMO_AUDIO_N_MELS,
                              win_mel, (float *)s.in.ptr);
      GpuCommandBuffer cb = gpu_begin(enc->gpu);
      run_linear(&cb, enc, &enc->diar_pre_proj, &s.in, &s.x_bf16, NULL, &s.h,
                 win_toks, AUD_ACT_NONE, 1);
      gpu_commit(&cb);
      gpu_sync(enc->gpu);
      memcpy(chunk_pre, s.h.ptr,
             (size_t)win_toks * NEMO_AUDIO_DIAR_DIM * sizeof(float));

      int cat_toks = st.spkcache_len + st.fifo_len + win_toks;
      float *cat_dst = (float *)s.h.ptr;
      if (st.spkcache_len > 0)
        memcpy(cat_dst, st.spkcache,
               (size_t)st.spkcache_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
      if (st.fifo_len > 0)
        memcpy(cat_dst + (size_t)st.spkcache_len * NEMO_AUDIO_DIAR_DIM, st.fifo,
               (size_t)st.fifo_len * NEMO_AUDIO_DIAR_DIM * sizeof(float));
      memcpy(cat_dst +
                 (size_t)(st.spkcache_len + st.fifo_len) * NEMO_AUDIO_DIAR_DIM,
             chunk_pre, (size_t)win_toks * NEMO_AUDIO_DIAR_DIM * sizeof(float));

      run_transformer_window(
          enc, &s, cat_toks, NEMO_AUDIO_DIAR_DIM, NEMO_AUDIO_DIAR_HEADS,
          NEMO_AUDIO_DIAR_HEAD_DIM, NEMO_AUDIO_DIAR_HALF_ROT, 0,
          NEMO_AUDIO_DIAR_LAYERS, 1, &enc->diar_pre_proj, &enc->diar_embed_nw,
          &enc->diar_embed_nb, &enc->diar_final_nw, &enc->diar_final_nb,
          enc->diar_layers, wait);
      run_sortformer_head(enc, &s, cat_toks);
      gpu_sync(enc->gpu);

      int lc_enc = left_off / NEMO_AUDIO_SUBSAMPLE;
      int rc_enc =
          (right_off + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int diar_core = win_toks - lc_enc - rc_enc;
      const float *all_p = (const float *)s.diar_probs.ptr;
      const float *win_p =
          all_p +
          (size_t)(st.spkcache_len + st.fifo_len + lc_enc) * NEMO_AUDIO_NUM_SPK;
      for (int t = 0; t < core_len; t++) {
        int src_t = t < diar_core ? t : (diar_core > 0 ? diar_core - 1 : 0);
        memcpy(probs_all + (size_t)(out_pos + t) * NEMO_AUDIO_NUM_SPK,
               win_p + (size_t)src_t * NEMO_AUDIO_NUM_SPK,
               (size_t)NEMO_AUDIO_NUM_SPK * sizeof(float));
      }

      if (ci < num_chunks - 1)
        diar_stream_update(&st, chunk_pre, win_toks, all_p, lc_enc, rc_enc,
                           enc->learnable_sil_emb);
    }

    free(chunk_pre);
    free(st.spkcache);
    free(st.spkcache_preds);
    free(st.fifo);
    free(st.fifo_preds);
  }

  free(mel_raw);
  free(mel_asr);

  // 3. Fuse thresholded speaker probabilities into ASR states and project
  //    1280 -> 2688 in chunks of fuse_chunk tokens.
  float *embeds =
      xmalloc((size_t)total_tokens * NEMO_AUDIO_OUT_DIM * sizeof(float));
  for (int pos = 0; pos < total_tokens; pos += fuse_chunk) {
    int rows = total_tokens - pos;
    if (rows > fuse_chunk) rows = fuse_chunk;
    memcpy(s.fuse_asr.ptr, asr_all + (size_t)pos * NEMO_AUDIO_ASR_DIM,
           (size_t)rows * NEMO_AUDIO_ASR_DIM * sizeof(float));
    memcpy(s.fuse_probs.ptr, probs_all + (size_t)pos * NEMO_AUDIO_NUM_SPK,
           (size_t)rows * NEMO_AUDIO_NUM_SPK * sizeof(float));

    GpuCommandBuffer cb = gpu_begin(enc->gpu);
    AudFuseParam fp = {.rows = (uint32_t)rows,
                       .dim = NEMO_AUDIO_ASR_DIM,
                       .n_spk = NEMO_AUDIO_NUM_SPK,
                       .eps = LN_EPS,
                       .threshold = SPK_THRESHOLD,
                       .scale = SPK_KERNEL_SCALE};
    gpu_select_kernel(&cb, enc->k_fuse);
    gpu_bind_buffer(&cb, 0, &s.fuse_asr, 0);
    gpu_bind_buffer(&cb, 1, &s.fuse_probs, 0);
    gpu_bind_buffer(&cb, 2, &enc->asr_fuse_nw, 0);
    gpu_bind_buffer(&cb, 3, &enc->asr_fuse_nb, 0);
    gpu_bind_buffer(&cb, 4, &enc->diar_fuse_nw, 0);
    gpu_bind_buffer(&cb, 5, &enc->diar_fuse_nb, 0);
    gpu_bind_buffer(&cb, 6, &enc->diar_kernel, 0);
    gpu_bind_buffer(&cb, 7, &s.fused, 0);
    gpu_copy_bytes(&cb, 8, &fp, sizeof fp);
    gpu_run_kernel(&cb, (uint32_t)rows, 1, 1, AUD_ROW_THREADS, 1, 1);

    run_linear(&cb, enc, &enc->proj, &s.fused, &s.x_bf16, NULL, &s.proj_out,
               rows, AUD_ACT_NONE, 1);
    gpu_commit(&cb);
    gpu_sync(enc->gpu);

    memcpy(embeds + (size_t)pos * NEMO_AUDIO_OUT_DIM, s.proj_out.ptr,
           (size_t)rows * NEMO_AUDIO_OUT_DIM * sizeof(float));
  }

  if (out_asr_bf16) {
    *out_asr_bf16 = pack_f32_to_bf16_alloc(asr_all, (size_t)total_tokens *
                                                        NEMO_AUDIO_ASR_DIM);
  }
  free(asr_all);
  free(probs_all);
  free_scratch(&s);

  *out_embeds = embeds;
  *out_tokens = total_tokens;
  return 1;
}

int nemo_audio_encode(NemoAudioEncoder *enc, const float *pcm, int pcm_samples,
                      const PassWait *wait, float **out_embeds, int *out_tokens,
                      char *error, size_t error_cap) {
  return nemo_audio_encode_with_asr(enc, pcm, pcm_samples, wait, out_embeds,
                                    NULL, out_tokens, error, error_cap);
}

int nemo_audio_encode_asr_only(NemoAudioEncoder *enc, const float *pcm,
                               int pcm_samples, const PassWait *wait,
                               uint16_t **out_asr_bf16, int *out_tokens,
                               char *error, size_t error_cap) {
  if (out_asr_bf16) *out_asr_bf16 = NULL;
  if (out_tokens) *out_tokens = 0;
  if (!enc || !pcm || pcm_samples < NEMO_AUDIO_HOP_LEN) {
    set_audio_err(error, error_cap, "audio clip is too short to encode");
    return 0;
  }
  if (!weight_group_load(enc->weights)) {
    set_audio_err(error, error_cap,
                  "perception weights do not fit in available memory");
    return 0;
  }

  int mel_frames = nemo_audio_mel_frames(pcm_samples);
  int total_tokens = nemo_audio_token_count(pcm_samples);

  float *mel_raw =
      xmalloc((size_t)mel_frames * NEMO_AUDIO_N_MELS * sizeof(float));
  float *mel_asr =
      xmalloc((size_t)mel_frames * NEMO_AUDIO_N_MELS * sizeof(float));
  nemo_audio_mel_spectrogram(pcm, pcm_samples, enc->window, enc->fb, mel_raw);
  nemo_audio_normalize_per_feature(mel_raw, mel_frames, mel_asr);
  free(mel_raw);

  int use_online = mel_frames > NEMO_AUDIO_ONLINE_MEL_FRAMES;
  int max_tokens =
      use_online ? (NEMO_AUDIO_ONLINE_TOKENS + NEMO_AUDIO_CTX_LEFT_TOKENS +
                    NEMO_AUDIO_CTX_RIGHT_TOKENS)
                 : total_tokens;
  int max_seq_stride = align_up(max_tokens, (int)AUD_TILE);
  int max_fp8_pad = align_up(max_tokens, (int)AUD_FP8_M_TILE);

  AudScratch s = {0};
  int alloc_ok =
      try_buf(enc->gpu, enc->work, &s.in,
              (size_t)max_tokens * NEMO_AUDIO_STACK_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.h,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.norm,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.qkv,
              (size_t)max_tokens * 3u * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.q_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.k_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.v_hm,
              (size_t)NEMO_AUDIO_ASR_HEADS * max_seq_stride *
                  NEMO_AUDIO_ASR_HEAD_DIM * sizeof(uint16_t)) &&
      try_buf(enc->gpu, enc->work, &s.attn_out,
              (size_t)max_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.ffn,
              (size_t)max_tokens * NEMO_AUDIO_ASR_FFN * sizeof(float)) &&
      try_buf(enc->gpu, enc->work, &s.x_bf16,
              (size_t)max_fp8_pad * NEMO_AUDIO_ASR_FFN * sizeof(uint16_t));
  if (!alloc_ok) {
    free_scratch(&s);
    free(mel_asr);
    set_audio_err(error, error_cap,
                  "perception workspace does not fit in available memory");
    return 0;
  }

  float *asr_all =
      xmalloc((size_t)total_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float));
  if (!use_online) {
    nemo_audio_stack_frames(mel_asr, mel_frames, (float *)s.in.ptr);
    run_transformer_window(enc, &s, total_tokens, NEMO_AUDIO_ASR_DIM,
                           NEMO_AUDIO_ASR_HEADS, NEMO_AUDIO_ASR_HEAD_DIM,
                           NEMO_AUDIO_ASR_HALF_ROT, 1, NEMO_AUDIO_ASR_LAYERS, 0,
                           &enc->asr_pre_proj, &enc->asr_embed_nw,
                           &enc->asr_embed_nb, &enc->asr_final_nw,
                           &enc->asr_final_nb, enc->asr_layers, wait);
    gpu_sync(enc->gpu);
    memcpy(asr_all, s.h.ptr,
           (size_t)total_tokens * NEMO_AUDIO_ASR_DIM * sizeof(float));
  } else {
    int chunk_feat = NEMO_AUDIO_ONLINE_MEL_FRAMES;
    int left_ctx_feat = NEMO_AUDIO_CTX_LEFT_TOKENS * NEMO_AUDIO_SUBSAMPLE;
    int right_ctx_feat = NEMO_AUDIO_CTX_RIGHT_TOKENS * NEMO_AUDIO_SUBSAMPLE;
    int num_chunks = (mel_frames + chunk_feat - 1) / chunk_feat;
    for (int ci = 0; ci < num_chunks; ci++) {
      int start = ci * chunk_feat;
      int end = start + chunk_feat;
      if (end > mel_frames) end = mel_frames;
      int ctx_start = start - left_ctx_feat;
      if (ctx_start < 0) ctx_start = 0;
      int ctx_end = end + right_ctx_feat;
      if (ctx_end > mel_frames) ctx_end = mel_frames;

      int win_mel = ctx_end - ctx_start;
      int win_toks =
          (win_mel + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int left_drop = (start - ctx_start) / NEMO_AUDIO_SUBSAMPLE;
      int out_pos = (start + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int out_end = (end + NEMO_AUDIO_SUBSAMPLE - 1) / NEMO_AUDIO_SUBSAMPLE;
      int core_len = out_end - out_pos;

      nemo_audio_stack_frames(mel_asr + (size_t)ctx_start * NEMO_AUDIO_N_MELS,
                              win_mel, (float *)s.in.ptr);
      run_transformer_window(enc, &s, win_toks, NEMO_AUDIO_ASR_DIM,
                             NEMO_AUDIO_ASR_HEADS, NEMO_AUDIO_ASR_HEAD_DIM,
                             NEMO_AUDIO_ASR_HALF_ROT, 1, NEMO_AUDIO_ASR_LAYERS,
                             0, &enc->asr_pre_proj, &enc->asr_embed_nw,
                             &enc->asr_embed_nb, &enc->asr_final_nw,
                             &enc->asr_final_nb, enc->asr_layers, wait);
      gpu_sync(enc->gpu);
      memcpy(asr_all + (size_t)out_pos * NEMO_AUDIO_ASR_DIM,
             (const float *)s.h.ptr + (size_t)left_drop * NEMO_AUDIO_ASR_DIM,
             (size_t)core_len * NEMO_AUDIO_ASR_DIM * sizeof(float));
    }
  }

  free(mel_asr);
  free_scratch(&s);
  uint16_t *packed = pack_f32_to_bf16_alloc(asr_all, (size_t)total_tokens *
                                                         NEMO_AUDIO_ASR_DIM);
  free(asr_all);

  *out_asr_bf16 = packed;
  *out_tokens = total_tokens;
  return 1;
}
