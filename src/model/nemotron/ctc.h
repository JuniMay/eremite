#ifndef EREMITE_MODEL_NEMOTRON_CTC_H
#define EREMITE_MODEL_NEMOTRON_CTC_H
#include "engine/memory.h"
#include "engine/pass.h"
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// NVIDIA Nemotron-3.5-Transcribe auxiliary CTC timestamp head
// (model-ctc-bf16.safetensors): 2x ConvTranspose1d temporal upsampling
// (80 ms -> 40 ms), 2-layer bridging Transformer (d_model=1280, 16 heads,
// head_dim=80, qk_norm, RoPE rotary_fraction=0.5, FFN=640), scaled residual
// blend, 23863-class CTC projection, embedded SentencePiece BPE tokenizer, and
// banded CTC Viterbi forced alignment.

// CTC decoder geometry matching model-ctc-bf16.safetensors and config.json.
static constexpr int NEMO_CTC_DIM = 1280;
static constexpr int NEMO_CTC_HEADS = 16;
static constexpr int NEMO_CTC_HEAD_DIM = 80;
static constexpr int NEMO_CTC_HALF_ROT = 20;
static constexpr int NEMO_CTC_FFN = 640;
static constexpr int NEMO_CTC_LAYERS = 2;
static constexpr int NEMO_CTC_UP_FACTOR = 2;
static constexpr int NEMO_CTC_VOCAB = 23862;
static constexpr int NEMO_CTC_BLANK_ID = 23862;
static constexpr int NEMO_CTC_CLASSES = NEMO_CTC_VOCAB + 1;
static constexpr double NEMO_CTC_FRAME_SECONDS = 0.04;

// One aligned word span referencing byte offsets in the input transcript.
typedef struct {
  // Start byte offset (inclusive) of this word in the UTF-8 transcript.
  int byte_start;
  // End byte offset (exclusive) of this word in the UTF-8 transcript.
  int byte_end;
  // Aligned word start time in seconds within [0, duration].
  double start;
  // Aligned word end time in seconds within [start, duration].
  double end;
} NemoCtcWord;

// Loaded SentencePiece BPE tokenizer parsed from a serialized ModelProto.
typedef struct NemoSpmTokenizer NemoSpmTokenizer;

// Loaded Nemotron CTC timestamp decoder and aligner handle.
typedef struct NemoCtcAligner NemoCtcAligner;

// Parses a SentencePiece BPE ModelProto byte buffer into a tokenizer handle.
// Returns null if the buffer is malformed or contains no pieces.
NemoSpmTokenizer *nemo_spm_open_bytes(const uint8_t *proto, size_t proto_len);

// Releases all memory owned by tk.
void nemo_spm_close(NemoSpmTokenizer *tk);

// Returns the number of vocabulary pieces in tk.
int nemo_spm_vocab_size(const NemoSpmTokenizer *tk);

// Returns the UTF-8 spelling of piece id, or empty string if out of range.
const char *nemo_spm_piece(const NemoSpmTokenizer *tk, int id);

// Applies the precompiled nmt_nfkc character map to UTF-8 text[0..len - 1].
// Writes a heap-allocated NUL-terminated UTF-8 string into *out and its byte
// length into *out_len. Caller frees *out.
void nemo_spm_normalize(const NemoSpmTokenizer *tk, const char *text,
                        size_t len, char **out, size_t *out_len);

// Encodes one UTF-8 word into BPE piece IDs written to out[0..max - 1].
// Prepends the SentencePiece word-boundary marker when add_prefix is nonzero
// and the word is not a user-defined special token. Returns the piece count.
int nemo_spm_encode_word(const NemoSpmTokenizer *tk, const char *word,
                         size_t len, int add_prefix, int *out, int max);

// Decodes piece IDs[0..count - 1] into a heap-allocated NUL-terminated UTF-8
// string, converting SentencePiece word-boundary markers to ASCII spaces.
char *nemo_spm_decode(const NemoSpmTokenizer *tk, const int *ids, int count);

// Splits UTF-8 transcript[0..len - 1] into word spans referencing byte offsets
// in transcript. Writes a heap-allocated array to *out_words and its length to
// *out_count. Caller frees *out_words.
void nemo_ctc_split_words(const char *transcript, size_t len,
                          NemoCtcWord **out_words, int *out_count);

// Runs CTC Viterbi forced alignment of target tokens[0..n_tokens - 1] against
// log-probabilities logp[frames, n_gather], where token_slot[i] maps token i
// to its column in logp and blank_slot is the blank column. Writes first and
// last frame indices into tok_first[0..n_tokens - 1] and
// tok_last[0..n_tokens - 1]. Returns 1 on success or 0 when unalignable.
int nemo_ctc_viterbi_align(const float *logp, int frames, int n_gather,
                           const int *tokens, const int *token_slot,
                           int n_tokens, int blank_slot, int *tok_first,
                           int *tok_last);

// Returns nonzero when dir contains model-ctc-bf16.safetensors.
int nemo_ctc_available(const char *dir);

// Opens model-ctc-bf16.safetensors in dir, parses its SentencePiece tokenizer,
// registers its weights in a reloadable weight group under account, and looks
// up its GPU pipelines. Returns null if the file is absent or invalid.
NemoCtcAligner *nemo_ctc_open(Gpu *gpu, MemoryAccount *account,
                              MemorySlot *work, const char *dir);

// Releases all weights, file handles and host resources owned by ctc.
void nemo_ctc_close(NemoCtcAligner *ctc);

// Pins or unpins the CTC weight group while an alignment call runs.
void nemo_ctc_pin(NemoCtcAligner *ctc, int active);

// Returns nonzero when the CTC weights are currently resident in GPU memory.
int nemo_ctc_is_loaded(const NemoCtcAligner *ctc);

// Returns the byte footprint required to hold the CTC weights.
uint64_t nemo_ctc_weight_bytes(const NemoCtcAligner *ctc);

// Aligns UTF-8 transcript[0..transcript_len - 1] against BF16-packed raw ASR
// encoder states asr_bf16[asr_tokens, 1280] for an audio clip of duration
// seconds. Writes aligned words to *out_words (caller frees) and count to
// *out_count. Returns 1 on success or 0 with error populated on failure.
int nemo_ctc_align(NemoCtcAligner *ctc, const uint16_t *asr_bf16,
                   int asr_tokens, double duration, const char *transcript,
                   size_t transcript_len, const PassWait *wait,
                   NemoCtcWord **out_words, int *out_count, char *error,
                   size_t error_cap);

// Runs greedy CTC decoding over BF16-packed raw ASR encoder states
// asr_bf16[asr_tokens, 1280] and writes a heap-allocated NUL-terminated UTF-8
// transcript to *out_text. Returns 1 on success or 0 on failure.
int nemo_ctc_greedy_decode(NemoCtcAligner *ctc, const uint16_t *asr_bf16,
                           int asr_tokens, const PassWait *wait,
                           char **out_text, char *error, size_t error_cap);

#endif
