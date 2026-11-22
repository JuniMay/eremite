#include "model/nemotron/ctc.h"
#include "base/error.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "model/nemotron/audio.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Nemotron-3.5-Transcribe CTC timestamp head and forced aligner:
// 1. Embedded SentencePiece BPE tokenizer parsed from
//    ctc_timestamp.tokenizer.model_proto, including precompiled Darts nmt_nfkc
//    character normalization and score-ordered BPE merging.
// 2. UTF-8 word splitting with CJK character segmentation preserving exact byte
//    offsets into the input transcript.
// 3. Reloadable BF16 GPU CTC decoder (~84 MiB): ConvTranspose1d 2x upsampling
//    (80 ms -> 40 ms), 2-layer bridging Transformer (16x80 qk_norm half-RoPE),
//    scaled residual blend, and chunked 23863-class log-softmax gather.
// 4. Streaming chunked banded CTC Viterbi forced aligner with 2-bit packed
//    backpointers and automatic full-band fallback for long leading or trailing
//    silence.

// Checkpoint filename and numerical constants matching config.json.
static constexpr char CTC_FILENAME[] = "model-ctc-bf16.safetensors";
static constexpr float CTC_LN_EPS = 1e-5f;
static constexpr double CTC_ROPE_BASE = 10000.0;
static constexpr double CTC_LOG2_E = 1.4426950408889634074;
static constexpr double CTC_MIN_WORD_SECONDS = 0.005;
static constexpr double CTC_END_OFFSET_SECONDS = 0.12;
static constexpr double CTC_BAND_RATIO = 0.12;
static constexpr int CTC_BAND_MIN_STATES = 512;
static constexpr int64_t CTC_FULL_VITERBI_CELLS = 16000000;
static constexpr int CTC_MAX_ROPE_LEN = 2048;
static constexpr int CTC_LOGIT_CHUNK = 256;
static constexpr int CTC_F32_LOAD_ROWS = 256;
static constexpr size_t CTC_PATH_CAP = 1024;
static constexpr int CTC_MAX_WORD_TOKENS = 256;

// GPU kernel tile and threadgroup sizes matching nemotron_audio.metal.
static constexpr uint32_t CTC_TILE = 64;
static constexpr uint32_t CTC_MM_THREADS = 128;
static constexpr uint32_t CTC_ROW_THREADS = 256;
static constexpr uint32_t CTC_PACK_THREADS = 64;
static constexpr uint32_t CTC_RESID_THREADS = 256;
static constexpr uint32_t CTC_ACT_NONE = 0;
static constexpr uint32_t CTC_ACT_GELU = 1;

// IEEE BF16 conversion bit masks and shifts.
static constexpr uint32_t BF16_SHIFT = 16;
static constexpr uint32_t BF16_HALF = 0x7FFFu;
static constexpr uint32_t BF16_MASK = 0xFFFF0000u;

// Protobuf wire types and SentencePiece ModelProto field numbers.
static constexpr uint32_t PB_WIRE_VARINT = 0;
static constexpr uint32_t PB_WIRE_FIXED64 = 1;
static constexpr uint32_t PB_WIRE_BYTES = 2;
static constexpr uint32_t PB_WIRE_FIXED32 = 5;
static constexpr uint32_t SPM_FIELD_PIECES = 1;
static constexpr uint32_t SPM_FIELD_NORM_SPEC = 3;
static constexpr uint32_t SPM_PIECE_STR = 1;
static constexpr uint32_t SPM_PIECE_SCORE = 2;
static constexpr uint32_t SPM_PIECE_TYPE = 3;
static constexpr uint32_t SPM_NORM_CHARSMAP = 2;
static constexpr int SPM_TYPE_NORMAL = 1;
static constexpr int SPM_TYPE_UNKNOWN = 2;
static constexpr int SPM_TYPE_USER_DEFINED = 4;

// SentencePiece UTF-8 word-boundary prefix (U+2581 LOWER ONE EIGHTH BLOCK).
static constexpr char SPM_SPACE_UTF8[] = "\xE2\x96\x81";
static constexpr size_t SPM_SPACE_LEN = 3;

// Darts double-array trie bit layout used by SentencePiece precompiled maps.
static constexpr uint32_t DARTS_LEAF_BIT = 1u << 31;
static constexpr uint32_t DARTS_VALUE_MASK = DARTS_LEAF_BIT - 1u;
static constexpr uint32_t DARTS_HAS_LEAF_SHIFT = 8;
static constexpr uint32_t DARTS_EXT_BIT = 1u << 9;
static constexpr uint32_t DARTS_EXT_SHIFT = 6;
static constexpr uint32_t DARTS_OFFSET_SHIFT = 10;
static constexpr uint32_t DARTS_LABEL_MASK = DARTS_LEAF_BIT | 0xFFu;

// FNV-1a 64-bit hash constants for vocabulary piece lookup.
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME = 1099511628211ULL;

// Two-bit Viterbi backpointer transition codes.
static constexpr uint8_t VIT_STAY = 0;
static constexpr uint8_t VIT_PREV1 = 1;
static constexpr uint8_t VIT_PREV2 = 2;
static constexpr uint8_t VIT_CODE_MASK = 3;

// GPU parameter structs matching src/kernels/nemotron_audio.metal.
typedef struct {
  // Row width in floats and BF16 rounding flag.
  uint32_t dim, round;
  // LayerNorm variance epsilon.
  float eps;
} CtcLnParam;

typedef struct {
  // Active rows, output width, inner width and padded weight rows.
  uint32_t m, n, k, w_rows;
  // Activation mode, bias flag, residual flag and BF16 rounding flag.
  uint32_t act, has_bias, has_res, round;
  // Unused FP8 scale field kept for layout compatibility.
  float scale;
} CtcMmParam;

typedef struct {
  // Rows, 64-aligned head-major stride, heads and head width.
  uint32_t rows, seq_stride, heads, head_dim;
  // Half of rotated dimensions and qk_norm flag.
  uint32_t half_rot, qk_norm;
  // Per-head LayerNorm epsilon.
  float eps;
} CtcQkvParam;

typedef struct {
  // Rows, 64-aligned head-major stride and head count.
  uint32_t rows, seq_stride, heads;
  // Attention scale multiplied by log2(e).
  float log2_scale;
} CtcAttnParam;

typedef struct {
  // Encoder frame count and feature width (multiple of 4).
  uint32_t frames, dim;
} CtcInterleaveParam;

typedef struct {
  // Total float4 elements and BF16 rounding flag.
  uint32_t count4, round;
  // Scalar multiplier on the bridged Transformer output.
  float scale;
} CtcResidParam;

typedef struct {
  // Row count, full CTC vocabulary size and gathered token count.
  uint32_t rows, vocab, n_gather;
} CtcLogSoftmaxParam;

// ---- Small numeric helpers ----

// Rounds v to nearest BF16 value (ties to even) and returns raw 16-bit bits.
static inline uint16_t f32_to_bf16(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof u);
  u = (u + BF16_HALF + ((u >> BF16_SHIFT) & 1u)) & BF16_MASK;
  return (uint16_t)(u >> BF16_SHIFT);
}

// Converts raw BF16 bits to FP32.
static inline float bf16_to_f32(uint16_t bits) {
  uint32_t u = (uint32_t)bits << BF16_SHIFT;
  float v;
  memcpy(&v, &u, sizeof v);
  return v;
}

// Rounds n up to a multiple of align.
static inline int align_up(int n, int align) {
  return ((n + align - 1) / align) * align;
}

// Computes FNV-1a hash over len bytes.
static uint64_t hash_bytes(const char *s, size_t len) {
  uint64_t h = FNV_OFFSET;
  for (size_t i = 0; i < len; i++) {
    h ^= (uint8_t)s[i];
    h *= FNV_PRIME;
  }
  return h;
}

// Decodes one UTF-8 codepoint starting at s[0..len - 1] into *cp and returns
// its byte length (at least 1).
static size_t utf8_decode(const char *s, size_t len, uint32_t *cp) {
  uint8_t b0 = (uint8_t)s[0];
  if (b0 < 0x80u) {
    *cp = b0;
    return 1;
  }
  if ((b0 & 0xE0u) == 0xC0u && len >= 2 && (((uint8_t)s[1] & 0xC0u) == 0x80u)) {
    *cp = ((uint32_t)(b0 & 0x1Fu) << 6) | ((uint32_t)(uint8_t)s[1] & 0x3Fu);
    return 2;
  }
  if ((b0 & 0xF0u) == 0xE0u && len >= 3 && (((uint8_t)s[1] & 0xC0u) == 0x80u) &&
      (((uint8_t)s[2] & 0xC0u) == 0x80u)) {
    *cp = ((uint32_t)(b0 & 0x0Fu) << 12) |
          (((uint32_t)(uint8_t)s[1] & 0x3Fu) << 6) |
          ((uint32_t)(uint8_t)s[2] & 0x3Fu);
    return 3;
  }
  if ((b0 & 0xF8u) == 0xF0u && len >= 4 && (((uint8_t)s[1] & 0xC0u) == 0x80u) &&
      (((uint8_t)s[2] & 0xC0u) == 0x80u) &&
      (((uint8_t)s[3] & 0xC0u) == 0x80u)) {
    *cp = ((uint32_t)(b0 & 0x07u) << 18) |
          (((uint32_t)(uint8_t)s[1] & 0x3Fu) << 12) |
          (((uint32_t)(uint8_t)s[2] & 0x3Fu) << 6) |
          ((uint32_t)(uint8_t)s[3] & 0x3Fu);
    return 4;
  }
  *cp = b0;
  return 1;
}

// Writes an error message into error when non-null.
static void set_ctc_err(char *error, size_t cap, const char *msg) {
  if (error && cap > 0) snprintf(error, cap, "%s", msg);
}

// ---- SentencePiece BPE Tokenizer ----

// One SentencePiece vocabulary entry.
typedef struct {
  // Heap-allocated NUL-terminated UTF-8 piece spelling.
  char *text;
  // Byte length of text.
  uint32_t len;
  // Piece merge priority score (higher merges earlier).
  float score;
  // SentencePiece piece type (NORMAL, UNKNOWN or USER_DEFINED).
  int type;
} SpmPiece;

// Open-addressed hash table slot mapping a piece string to its vocabulary ID.
typedef struct {
  // FNV-1a hash of the stored piece string.
  uint64_t hash;
  // Piece vocabulary ID, or -1 for an empty slot.
  int id;
} SpmHashSlot;

// Loaded SentencePiece BPE tokenizer and precompiled Darts normalizer.
struct NemoSpmTokenizer {
  // Vocabulary pieces indexed by piece ID.
  SpmPiece *pieces;
  // Total number of pieces in the vocabulary.
  int vocab_size;
  // Vocabulary ID of the unknown token piece.
  int unk_id;
  // Open-addressed lookup table for normal pieces.
  SpmHashSlot *table;
  // Bitmask equal to table capacity minus one.
  uint32_t table_mask;
  // Indices of user-defined special pieces.
  int *user_ids;
  // Number of user-defined special pieces.
  int n_user;
  // Darts double-array trie units for nmt_nfkc normalization, or null.
  uint32_t *trie;
  // Number of 32-bit units in trie.
  size_t trie_len;
  // NUL-terminated UTF-8 replacement string pool indexed by Darts leaf values.
  char *pool;
  // Byte length of pool.
  size_t pool_len;
};

// Reads a protobuf unsigned varint starting at *pos within buf[0..end - 1].
static int pb_read_varint(const uint8_t *buf, size_t end, size_t *pos,
                          uint64_t *out) {
  uint64_t v = 0;
  int shift = 0;
  while (*pos < end && shift < 64) {
    uint8_t b = buf[(*pos)++];
    v |= (uint64_t)(b & 0x7Fu) << shift;
    if ((b & 0x80u) == 0) {
      *out = v;
      return 1;
    }
    shift += 7;
  }
  return 0;
}

// Skips one protobuf field of wire_type at *pos within buf[0..end - 1].
static int pb_skip(const uint8_t *buf, size_t end, size_t *pos,
                   uint32_t wire_type) {
  if (wire_type == PB_WIRE_VARINT) {
    uint64_t dummy = 0;
    return pb_read_varint(buf, end, pos, &dummy);
  }
  if (wire_type == PB_WIRE_FIXED64) {
    if (*pos + 8 > end) return 0;
    *pos += 8;
    return 1;
  }
  if (wire_type == PB_WIRE_BYTES) {
    uint64_t len = 0;
    if (!pb_read_varint(buf, end, pos, &len) || len > end - *pos) return 0;
    *pos += (size_t)len;
    return 1;
  }
  if (wire_type == PB_WIRE_FIXED32) {
    if (*pos + 4 > end) return 0;
    *pos += 4;
    return 1;
  }
  return 0;
}

// Parses one SentencePiece submessage into piece. Returns 1 on success.
static int pb_parse_piece(const uint8_t *buf, size_t len, SpmPiece *piece) {
  *piece = (SpmPiece){.type = SPM_TYPE_NORMAL};
  size_t pos = 0;
  while (pos < len) {
    uint64_t tag = 0;
    if (!pb_read_varint(buf, len, &pos, &tag)) return 0;
    uint32_t fn = (uint32_t)(tag >> 3);
    uint32_t wt = (uint32_t)(tag & 7u);
    if (fn == SPM_PIECE_STR && wt == PB_WIRE_BYTES) {
      uint64_t slen = 0;
      if (!pb_read_varint(buf, len, &pos, &slen) || slen > len - pos) return 0;
      free(piece->text);
      piece->text = xmalloc((size_t)slen + 1);
      memcpy(piece->text, buf + pos, (size_t)slen);
      piece->text[slen] = '\0';
      piece->len = (uint32_t)slen;
      pos += (size_t)slen;
    } else if (fn == SPM_PIECE_SCORE && wt == PB_WIRE_FIXED32) {
      if (pos + 4 > len) return 0;
      memcpy(&piece->score, buf + pos, sizeof(float));
      pos += 4;
    } else if (fn == SPM_PIECE_TYPE && wt == PB_WIRE_VARINT) {
      uint64_t tv = 0;
      if (!pb_read_varint(buf, len, &pos, &tv)) return 0;
      piece->type = (int)tv;
    } else {
      if (!pb_skip(buf, len, &pos, wt)) return 0;
    }
  }
  if (!piece->text) {
    piece->text = xmalloc(1);
    piece->text[0] = '\0';
    piece->len = 0;
  }
  return 1;
}

// Parses a SentencePiece NormalizerSpec precompiled_charsmap blob into tk.
static int parse_precompiled_charsmap(NemoSpmTokenizer *tk, const uint8_t *buf,
                                      size_t len) {
  if (len < 4) return 0;
  uint32_t trie_bytes = 0;
  memcpy(&trie_bytes, buf, sizeof trie_bytes);
  if (trie_bytes < 4 || (trie_bytes & 3u) != 0 || 4u + trie_bytes > len)
    return 0;
  size_t pool_bytes = len - 4u - trie_bytes;
  free(tk->trie);
  free(tk->pool);
  tk->trie = xmalloc(trie_bytes);
  memcpy(tk->trie, buf + 4, trie_bytes);
  tk->trie_len = trie_bytes / sizeof(uint32_t);
  tk->pool = xmalloc(pool_bytes + 1);
  if (pool_bytes > 0) memcpy(tk->pool, buf + 4 + trie_bytes, pool_bytes);
  tk->pool[pool_bytes] = '\0';
  tk->pool_len = pool_bytes;
  return 1;
}

// Looks up a normal piece by exact UTF-8 bytes in tk's hash table, returning
// its vocabulary ID or -1 if absent.
static int spm_lookup_normal(const NemoSpmTokenizer *tk, const char *s,
                             size_t len) {
  if (!tk->table) return -1;
  uint64_t h = hash_bytes(s, len);
  uint32_t idx = (uint32_t)h & tk->table_mask;
  for (;;) {
    int id = tk->table[idx].id;
    if (id < 0) return -1;
    if (tk->table[idx].hash == h && tk->pieces[id].len == len &&
        memcmp(tk->pieces[id].text, s, len) == 0) {
      return id;
    }
    idx = (idx + 1u) & tk->table_mask;
  }
}

// Looks up a user-defined special piece by exact UTF-8 bytes, or returns -1.
static int spm_lookup_user(const NemoSpmTokenizer *tk, const char *s,
                           size_t len) {
  for (int i = 0; i < tk->n_user; i++) {
    int id = tk->user_ids[i];
    if (tk->pieces[id].len == len && memcmp(tk->pieces[id].text, s, len) == 0)
      return id;
  }
  return -1;
}

NemoSpmTokenizer *nemo_spm_open_bytes(const uint8_t *proto, size_t proto_len) {
  if (!proto || proto_len == 0) return NULL;
  NemoSpmTokenizer *tk = calloc(1, sizeof *tk);
  if (!tk) die("out of memory");
  tk->unk_id = 0;

  int cap = 256;
  tk->pieces = xmalloc((size_t)cap * sizeof(SpmPiece));
  size_t pos = 0;
  while (pos < proto_len) {
    uint64_t tag = 0;
    if (!pb_read_varint(proto, proto_len, &pos, &tag)) {
      nemo_spm_close(tk);
      return NULL;
    }
    uint32_t fn = (uint32_t)(tag >> 3);
    uint32_t wt = (uint32_t)(tag & 7u);
    if (fn == SPM_FIELD_PIECES && wt == PB_WIRE_BYTES) {
      uint64_t mlen = 0;
      if (!pb_read_varint(proto, proto_len, &pos, &mlen) ||
          mlen > proto_len - pos) {
        nemo_spm_close(tk);
        return NULL;
      }
      if (tk->vocab_size >= cap) {
        cap *= 2;
        SpmPiece *np = realloc(tk->pieces, (size_t)cap * sizeof(SpmPiece));
        if (!np) die("out of memory");
        tk->pieces = np;
      }
      SpmPiece piece;
      if (!pb_parse_piece(proto + pos, (size_t)mlen, &piece)) {
        free(piece.text);
        nemo_spm_close(tk);
        return NULL;
      }
      if (piece.type == SPM_TYPE_UNKNOWN) tk->unk_id = tk->vocab_size;
      tk->pieces[tk->vocab_size++] = piece;
      pos += (size_t)mlen;
    } else if (fn == SPM_FIELD_NORM_SPEC && wt == PB_WIRE_BYTES) {
      uint64_t mlen = 0;
      if (!pb_read_varint(proto, proto_len, &pos, &mlen) ||
          mlen > proto_len - pos) {
        nemo_spm_close(tk);
        return NULL;
      }
      const uint8_t *nbuf = proto + pos;
      size_t npos = 0, nend = (size_t)mlen;
      while (npos < nend) {
        uint64_t ntag = 0;
        if (!pb_read_varint(nbuf, nend, &npos, &ntag)) {
          nemo_spm_close(tk);
          return NULL;
        }
        uint32_t nfn = (uint32_t)(ntag >> 3);
        uint32_t nwt = (uint32_t)(ntag & 7u);
        if (nfn == SPM_NORM_CHARSMAP && nwt == PB_WIRE_BYTES) {
          uint64_t clen = 0;
          if (!pb_read_varint(nbuf, nend, &npos, &clen) || clen > nend - npos) {
            nemo_spm_close(tk);
            return NULL;
          }
          if (clen > 0 &&
              !parse_precompiled_charsmap(tk, nbuf + npos, (size_t)clen)) {
            nemo_spm_close(tk);
            return NULL;
          }
          npos += (size_t)clen;
        } else if (!pb_skip(nbuf, nend, &npos, nwt)) {
          nemo_spm_close(tk);
          return NULL;
        }
      }
      pos += (size_t)mlen;
    } else if (!pb_skip(proto, proto_len, &pos, wt)) {
      nemo_spm_close(tk);
      return NULL;
    }
  }

  if (tk->vocab_size == 0) {
    nemo_spm_close(tk);
    return NULL;
  }

  // Build open-addressed hash table for normal pieces and collect user-defined
  // piece indices.
  uint32_t tcap = 64;
  while (tcap < (uint32_t)tk->vocab_size * 2u) tcap <<= 1;
  tk->table = xmalloc((size_t)tcap * sizeof(SpmHashSlot));
  tk->table_mask = tcap - 1u;
  for (uint32_t i = 0; i < tcap; i++) tk->table[i] = (SpmHashSlot){0, -1};

  int user_count = 0;
  for (int i = 0; i < tk->vocab_size; i++) {
    if (tk->pieces[i].type == SPM_TYPE_USER_DEFINED) user_count++;
  }
  if (user_count > 0) tk->user_ids = xmalloc((size_t)user_count * sizeof(int));

  for (int i = 0; i < tk->vocab_size; i++) {
    if (tk->pieces[i].type == SPM_TYPE_USER_DEFINED) {
      tk->user_ids[tk->n_user++] = i;
      continue;
    }
    if (tk->pieces[i].type != SPM_TYPE_NORMAL) continue;
    uint64_t h = hash_bytes(tk->pieces[i].text, tk->pieces[i].len);
    uint32_t idx = (uint32_t)h & tk->table_mask;
    while (tk->table[idx].id >= 0) {
      int cur = tk->table[idx].id;
      if (tk->table[idx].hash == h &&
          tk->pieces[cur].len == tk->pieces[i].len &&
          memcmp(tk->pieces[cur].text, tk->pieces[i].text, tk->pieces[i].len) ==
              0) {
        break;
      }
      idx = (idx + 1u) & tk->table_mask;
    }
    if (tk->table[idx].id < 0) {
      tk->table[idx].hash = h;
      tk->table[idx].id = i;
    }
  }
  return tk;
}

void nemo_spm_close(NemoSpmTokenizer *tk) {
  if (!tk) return;
  for (int i = 0; i < tk->vocab_size; i++) free(tk->pieces[i].text);
  free(tk->pieces);
  free(tk->table);
  free(tk->user_ids);
  free(tk->trie);
  free(tk->pool);
  free(tk);
}

int nemo_spm_vocab_size(const NemoSpmTokenizer *tk) {
  return tk ? tk->vocab_size : 0;
}

const char *nemo_spm_piece(const NemoSpmTokenizer *tk, int id) {
  if (!tk || id < 0 || id >= tk->vocab_size) return "";
  return tk->pieces[id].text;
}

// Appends len bytes from src onto a growable buffer.
static void buf_append(char **dst, size_t *size, size_t *cap, const char *src,
                       size_t len) {
  if (*size + len + 1 > *cap) {
    size_t ncap = *cap > 0 ? *cap * 2 : 64;
    while (ncap < *size + len + 1) ncap *= 2;
    char *nb = realloc(*dst, ncap);
    if (!nb) die("out of memory");
    *dst = nb;
    *cap = ncap;
  }
  if (len > 0) memcpy(*dst + *size, src, len);
  *size += len;
  (*dst)[*size] = '\0';
}

// Finds the longest Darts prefix match starting at src[0..len - 1]. Returns
// the matched byte count and writes the replacement string to *repl.
static size_t darts_longest_match(const NemoSpmTokenizer *tk, const char *src,
                                  size_t len, const char **repl) {
  if (!tk->trie || tk->trie_len == 0 || len == 0) return 0;
  uint32_t node = 0;
  uint32_t unit = tk->trie[0];
  node ^= (unit >> DARTS_OFFSET_SHIFT)
          << ((unit & DARTS_EXT_BIT) >> DARTS_EXT_SHIFT);
  size_t best_len = 0;
  uint32_t best_val = 0;
  for (size_t i = 0; i < len; i++) {
    uint8_t b = (uint8_t)src[i];
    node ^= b;
    if (node >= tk->trie_len) break;
    unit = tk->trie[node];
    if ((unit & DARTS_LABEL_MASK) != b) break;
    node ^= (unit >> DARTS_OFFSET_SHIFT)
            << ((unit & DARTS_EXT_BIT) >> DARTS_EXT_SHIFT);
    if ((unit >> DARTS_HAS_LEAF_SHIFT) & 1u) {
      if (node >= tk->trie_len) break;
      best_len = i + 1;
      best_val = tk->trie[node] & DARTS_VALUE_MASK;
    }
  }
  if (best_len > 0 && best_val < tk->pool_len) {
    *repl = tk->pool + best_val;
    return best_len;
  }
  return 0;
}

void nemo_spm_normalize(const NemoSpmTokenizer *tk, const char *text,
                        size_t len, char **out, size_t *out_len) {
  char *dst = xmalloc(len + 16);
  size_t size = 0, cap = len + 16;
  dst[0] = '\0';
  size_t i = 0;
  while (i < len) {
    const char *repl = NULL;
    size_t mlen = tk ? darts_longest_match(tk, text + i, len - i, &repl) : 0;
    if (mlen > 0 && repl) {
      buf_append(&dst, &size, &cap, repl, strlen(repl));
      i += mlen;
    } else {
      uint32_t cp = 0;
      size_t clen = utf8_decode(text + i, len - i, &cp);
      buf_append(&dst, &size, &cap, text + i, clen);
      i += clen;
    }
  }
  *out = dst;
  if (out_len) *out_len = size;
}

// Doubly-linked symbol node during SentencePiece BPE merging.
typedef struct {
  // Preceding and following symbol indices, or -1 at list boundaries.
  int prev, next;
  // Byte offset and byte length in the normalized word buffer (0 if merged).
  int start, len;
} BpeSym;

// Candidate adjacent BPE merge ordered in a max-heap by score then position.
typedef struct {
  // Vocabulary merge score of the combined string.
  float score;
  // Left and right symbol indices in the BpeSym array.
  int left, right;
  // Combined byte length when this candidate was enqueued.
  int merged_len;
} BpeMerge;

// Binary max-heap of BpeMerge entries.
typedef struct {
  // Heap array of candidate merges.
  BpeMerge *items;
  // Current element count and allocated capacity.
  int len, cap;
} BpeHeap;

// Returns nonzero when candidate a has higher merge priority than b.
static int bpe_merge_better(BpeMerge a, BpeMerge b) {
  if (a.score != b.score) return a.score > b.score;
  return a.left < b.left;
}

// Pushes one merge candidate onto heap h.
static void bpe_heap_push(BpeHeap *h, BpeMerge m) {
  if (h->len >= h->cap) {
    h->cap = h->cap > 0 ? h->cap * 2 : 64;
    BpeMerge *ni = realloc(h->items, (size_t)h->cap * sizeof(BpeMerge));
    if (!ni) die("out of memory");
    h->items = ni;
  }
  int i = h->len++;
  h->items[i] = m;
  while (i > 0) {
    int p = (i - 1) / 2;
    if (!bpe_merge_better(h->items[i], h->items[p])) break;
    BpeMerge tmp = h->items[p];
    h->items[p] = h->items[i];
    h->items[i] = tmp;
    i = p;
  }
}

// Pops the highest-priority merge candidate from non-empty heap h.
static BpeMerge bpe_heap_pop(BpeHeap *h) {
  BpeMerge top = h->items[0];
  h->items[0] = h->items[--h->len];
  int i = 0;
  for (;;) {
    int l = 2 * i + 1, r = 2 * i + 2, best = i;
    if (l < h->len && bpe_merge_better(h->items[l], h->items[best])) best = l;
    if (r < h->len && bpe_merge_better(h->items[r], h->items[best])) best = r;
    if (best == i) break;
    BpeMerge tmp = h->items[i];
    h->items[i] = h->items[best];
    h->items[best] = tmp;
    i = best;
  }
  return top;
}

// Checks whether syms[left] + syms[right] forms a known piece and pushes it.
static void bpe_try_pair(const NemoSpmTokenizer *tk, const char *s,
                         const BpeSym *syms, int left, int right, BpeHeap *h) {
  if (left < 0 || right < 0 || syms[left].len <= 0 || syms[right].len <= 0)
    return;
  int mlen = syms[left].len + syms[right].len;
  int id = spm_lookup_normal(tk, s + syms[left].start, (size_t)mlen);
  if (id >= 0) {
    bpe_heap_push(h, (BpeMerge){.score = tk->pieces[id].score,
                                .left = left,
                                .right = right,
                                .merged_len = mlen});
  }
}

int nemo_spm_encode_word(const NemoSpmTokenizer *tk, const char *word,
                         size_t len, int add_prefix, int *out, int max) {
  if (!tk || !word || len == 0 || max <= 0) return 0;

  // User-defined special tokens match verbatim without normalization or prefix.
  int uid = spm_lookup_user(tk, word, len);
  if (uid >= 0) {
    out[0] = uid;
    return 1;
  }

  char *norm = NULL;
  size_t norm_len = 0;
  nemo_spm_normalize(tk, word, len, &norm, &norm_len);

  // Replace any normalized ASCII spaces with SentencePiece's word-boundary
  // marker and optionally prepend it.
  size_t cap = norm_len * SPM_SPACE_LEN + SPM_SPACE_LEN + 8;
  char *buf = xmalloc(cap);
  size_t blen = 0;
  buf[0] = '\0';
  if (add_prefix) buf_append(&buf, &blen, &cap, SPM_SPACE_UTF8, SPM_SPACE_LEN);
  for (size_t i = 0; i < norm_len; i++) {
    if (norm[i] == ' ')
      buf_append(&buf, &blen, &cap, SPM_SPACE_UTF8, SPM_SPACE_LEN);
    else
      buf_append(&buf, &blen, &cap, norm + i, 1);
  }
  free(norm);
  if (blen == 0) {
    free(buf);
    return 0;
  }

  // Split buf into UTF-8 codepoint symbols.
  int sym_cap = (int)blen + 1;
  BpeSym *syms = xmalloc((size_t)sym_cap * sizeof(BpeSym));
  int nsyms = 0;
  for (size_t pos = 0; pos < blen;) {
    uint32_t cp = 0;
    size_t clen = utf8_decode(buf + pos, blen - pos, &cp);
    syms[nsyms] = (BpeSym){
        .prev = nsyms - 1, .next = -1, .start = (int)pos, .len = (int)clen};
    if (nsyms > 0) syms[nsyms - 1].next = nsyms;
    nsyms++;
    pos += clen;
  }

  BpeHeap heap = {0};
  for (int i = 0; i + 1 < nsyms; i++)
    bpe_try_pair(tk, buf, syms, i, i + 1, &heap);

  while (heap.len > 0) {
    BpeMerge m = bpe_heap_pop(&heap);
    if (syms[m.left].len <= 0 || syms[m.right].len <= 0) continue;
    if (syms[m.left].next != m.right) continue;
    if (syms[m.left].len + syms[m.right].len != m.merged_len) continue;
    int after = syms[m.right].next;
    syms[m.left].len = m.merged_len;
    syms[m.left].next = after;
    if (after >= 0) syms[after].prev = m.left;
    syms[m.right].len = 0;
    bpe_try_pair(tk, buf, syms, syms[m.left].prev, m.left, &heap);
    bpe_try_pair(tk, buf, syms, m.left, syms[m.left].next, &heap);
  }

  int count = 0;
  for (int idx = 0; idx >= 0 && count < max; idx = syms[idx].next) {
    if (syms[idx].len <= 0) continue;
    int id =
        spm_lookup_normal(tk, buf + syms[idx].start, (size_t)syms[idx].len);
    out[count++] = id >= 0 ? id : tk->unk_id;
  }

  free(heap.items);
  free(syms);
  free(buf);
  return count;
}

char *nemo_spm_decode(const NemoSpmTokenizer *tk, const int *ids, int count) {
  char *out = xmalloc(64);
  size_t size = 0, cap = 64;
  out[0] = '\0';
  if (!tk || !ids) return out;
  for (int i = 0; i < count; i++) {
    int id = ids[i];
    if (id < 0 || id >= tk->vocab_size) continue;
    const char *p = tk->pieces[id].text;
    size_t plen = tk->pieces[id].len;
    for (size_t j = 0; j < plen;) {
      if (j + SPM_SPACE_LEN <= plen &&
          memcmp(p + j, SPM_SPACE_UTF8, SPM_SPACE_LEN) == 0) {
        if (size > 0) buf_append(&out, &size, &cap, " ", 1);
        j += SPM_SPACE_LEN;
      } else {
        buf_append(&out, &size, &cap, p + j, 1);
        j++;
      }
    }
  }
  return out;
}

// ---- Word Splitting ----

// Returns nonzero when Unicode codepoint cp is treated as word-breaking space.
static int is_space_cp(uint32_t cp) {
  return cp <= 0x20u || cp == 0x00A0u || cp == 0x3000u;
}

// Returns nonzero when Unicode codepoint cp is a CJK ideograph or Japanese
// kana character that is segmented individually when written without spaces.
static int is_cjk_char(uint32_t cp) {
  return (cp >= 0x3040u && cp <= 0x30FFu) || (cp >= 0x3400u && cp <= 0x4DBFu) ||
         (cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0xF900u && cp <= 0xFAFFu);
}

// Returns nonzero when cp is CJK sentence punctuation attached to the previous
// CJK character rather than forming its own timestamp word.
static int is_cjk_punct(uint32_t cp) {
  return (cp >= 0x3001u && cp <= 0x303Fu) || (cp >= 0xFF01u && cp <= 0xFF0Fu) ||
         (cp >= 0xFF1Au && cp <= 0xFF20u) || (cp >= 0xFF3Bu && cp <= 0xFF40u) ||
         (cp >= 0xFF5Bu && cp <= 0xFF65u);
}

// Appends one word byte span onto a growable NemoCtcWord list.
static void push_word(NemoCtcWord **words, int *count, int *cap, int start,
                      int end) {
  if (end <= start) return;
  if (*count >= *cap) {
    *cap = *cap > 0 ? *cap * 2 : 32;
    NemoCtcWord *nw = realloc(*words, (size_t)*cap * sizeof(NemoCtcWord));
    if (!nw) die("out of memory");
    *words = nw;
  }
  (*words)[(*count)++] = (NemoCtcWord){
      .byte_start = start, .byte_end = end, .start = 0.0, .end = 0.0};
}

void nemo_ctc_split_words(const char *transcript, size_t len,
                          NemoCtcWord **out_words, int *out_count) {
  *out_words = NULL;
  *out_count = 0;
  if (!transcript || len == 0) return;

  NemoCtcWord *words = NULL;
  int count = 0, cap = 0;
  size_t i = 0;
  while (i < len) {
    uint32_t cp = 0;
    size_t clen = utf8_decode(transcript + i, len - i, &cp);
    if (is_space_cp(cp)) {
      i += clen;
      continue;
    }
    // Consume one whitespace-delimited token, splitting CJK ideograph runs into
    // single characters with trailing CJK punctuation attached.
    size_t tok_start = i;
    int in_non_cjk = 0;
    size_t run_start = i;
    while (i < len) {
      clen = utf8_decode(transcript + i, len - i, &cp);
      if (is_space_cp(cp)) break;
      if (is_cjk_char(cp)) {
        if (in_non_cjk) {
          push_word(&words, &count, &cap, (int)run_start, (int)i);
          in_non_cjk = 0;
        }
        size_t c_start = i;
        i += clen;
        while (i < len) {
          uint32_t next_cp = 0;
          size_t nlen = utf8_decode(transcript + i, len - i, &next_cp);
          if (!is_cjk_punct(next_cp)) break;
          i += nlen;
        }
        push_word(&words, &count, &cap, (int)c_start, (int)i);
        run_start = i;
      } else if (is_cjk_punct(cp) && !in_non_cjk && count > 0 &&
                 words[count - 1].byte_end == (int)i) {
        i += clen;
        words[count - 1].byte_end = (int)i;
        run_start = i;
      } else {
        if (!in_non_cjk) {
          run_start = i;
          in_non_cjk = 1;
        }
        i += clen;
      }
    }
    if (in_non_cjk) push_word(&words, &count, &cap, (int)run_start, (int)i);
    if (i == tok_start) i++;
  }
  *out_words = words;
  *out_count = count;
}

// ---- Streaming Banded CTC Viterbi Forced Aligner ----

// State for streaming frame-by-frame CTC Viterbi forced alignment with 2-bit
// packed backpointers.
typedef struct {
  // Total CTC frames, target token count and expanded state count (2 * L + 1).
  int total_frames, n_tokens, n_states;
  // Column index of the CTC blank symbol inside each frame's gathered logp row.
  int blank_slot;
  // Band half-width around the diagonal, or n_states for full feasible search.
  int band_half;
  // Number of frames fed so far.
  int cur_frame;
  // Target token IDs [n_tokens] and their gathered logp column indices.
  const int *tokens;
  const int *token_slot;
  // Earliest reachable frame and minimum remaining frames per expanded state.
  int *min_t, *rem_t;
  // Per-frame active state bounds [total_frames] and bit offset into bp_bytes.
  int *row_lo, *row_hi;
  uint64_t *row_bit_off;
  // Dynamic 2-bit packed backpointer bit-stream.
  uint8_t *bp_bytes;
  uint64_t bp_bits, bp_cap_bytes;
  // Previous and current frame DP score arrays [n_states].
  float *prev_dp, *curr_dp;
} CtcViterbi;

// Returns nonzero when state s may skip the preceding blank from state s - 2.
static inline int viterbi_can_skip(const CtcViterbi *v, int s) {
  if ((s & 1) == 0 || s < 3) return 0;
  int tok_idx = s >> 1;
  return v->tokens[tok_idx] != v->tokens[tok_idx - 1];
}

// Returns the gathered logp column index emitted by expanded state s.
static inline int viterbi_state_col(const CtcViterbi *v, int s) {
  return (s & 1) ? v->token_slot[s >> 1] : v->blank_slot;
}

// Computes minimum frames needed to align tokens[0..n_tokens - 1].
static int viterbi_min_frames(const int *tokens, int n_tokens) {
  if (n_tokens <= 0) return 0;
  int need = n_tokens;
  for (int i = 1; i < n_tokens; i++) {
    if (tokens[i] == tokens[i - 1]) need++;
  }
  return need;
}

// Initializes a streaming CTC Viterbi aligner. band_half <= 0 selects the full
// feasible band. Returns 1 when alignment is feasible or 0 otherwise.
static int viterbi_init(CtcViterbi *v, int total_frames, const int *tokens,
                        const int *token_slot, int n_tokens, int blank_slot,
                        int band_half) {
  *v = (CtcViterbi){0};
  if (total_frames <= 0 || n_tokens <= 0) return 0;
  if (total_frames < viterbi_min_frames(tokens, n_tokens)) return 0;

  int S = 2 * n_tokens + 1;
  v->total_frames = total_frames;
  v->n_tokens = n_tokens;
  v->n_states = S;
  v->blank_slot = blank_slot;
  v->band_half = band_half > 0 ? band_half : S;
  v->tokens = tokens;
  v->token_slot = token_slot;

  v->min_t = xmalloc((size_t)S * sizeof(int));
  v->rem_t = xmalloc((size_t)S * sizeof(int));
  v->min_t[0] = 0;
  v->min_t[1] = 0;
  for (int s = 2; s < S; s++) {
    int best = v->min_t[s - 1] + 1;
    if (viterbi_can_skip(v, s) && v->min_t[s - 2] + 1 < best)
      best = v->min_t[s - 2] + 1;
    v->min_t[s] = best;
  }
  v->rem_t[S - 1] = 0;
  v->rem_t[S - 2] = 0;
  for (int s = S - 3; s >= 0; s--) {
    int best = v->rem_t[s + 1] + 1;
    if (viterbi_can_skip(v, s + 2) && v->rem_t[s + 2] + 1 < best)
      best = v->rem_t[s + 2] + 1;
    v->rem_t[s] = best;
  }

  v->row_lo = xmalloc((size_t)total_frames * sizeof(int));
  v->row_hi = xmalloc((size_t)total_frames * sizeof(int));
  v->row_bit_off = xmalloc((size_t)total_frames * sizeof(uint64_t));
  v->prev_dp = xmalloc((size_t)S * sizeof(float));
  v->curr_dp = xmalloc((size_t)S * sizeof(float));
  for (int s = 0; s < S; s++) v->prev_dp[s] = -INFINITY;

  // Pre-reserve initial backpointer storage based on expected band width.
  int est_w = v->band_half * 2 + 1;
  if (est_w > S) est_w = S;
  uint64_t est_bytes =
      ((uint64_t)total_frames * (uint64_t)est_w * 2u + 7u) / 8u;
  if (est_bytes < 256) est_bytes = 256;
  v->bp_bytes = calloc(1, (size_t)est_bytes);
  if (!v->bp_bytes) die("out of memory");
  v->bp_cap_bytes = est_bytes;
  return 1;
}

// Releases all host memory owned by v.
static void viterbi_free(CtcViterbi *v) {
  free(v->min_t);
  free(v->rem_t);
  free(v->row_lo);
  free(v->row_hi);
  free(v->row_bit_off);
  free(v->bp_bytes);
  free(v->prev_dp);
  free(v->curr_dp);
  *v = (CtcViterbi){0};
}

// Ensures bp_bytes holds at least total_bits bits.
static void viterbi_ensure_bp(CtcViterbi *v, uint64_t total_bits) {
  uint64_t need = (total_bits + 7u) / 8u;
  if (need <= v->bp_cap_bytes) return;
  uint64_t ncap = v->bp_cap_bytes * 2u;
  while (ncap < need) ncap *= 2u;
  uint8_t *nb = realloc(v->bp_bytes, (size_t)ncap);
  if (!nb) die("out of memory");
  memset(nb + v->bp_cap_bytes, 0, (size_t)(ncap - v->bp_cap_bytes));
  v->bp_bytes = nb;
  v->bp_cap_bytes = ncap;
}

// Advances Viterbi DP by one CTC frame using gathered log-probabilities logp.
static void viterbi_step(CtcViterbi *v, const float *logp) {
  int t = v->cur_frame++;
  int S = v->n_states;
  int rem_frames = v->total_frames - 1 - t;

  if (t == 0) {
    v->row_lo[0] = 0;
    v->row_hi[0] = 1;
    v->row_bit_off[0] = v->bp_bits;
    v->bp_bits += 4;
    viterbi_ensure_bp(v, v->bp_bits);
    v->prev_dp[0] = logp[viterbi_state_col(v, 0)];
    v->prev_dp[1] = logp[viterbi_state_col(v, 1)];
    return;
  }

  // Compute exact feasible bounds via binary search on monotonic min_t/rem_t.
  int lo_f = 0, hi_f = S - 1;
  {
    int a = 0, b = S - 1;
    while (a < b) {
      int m = a + (b - a) / 2;
      if (v->rem_t[m] > rem_frames)
        a = m + 1;
      else
        b = m;
    }
    lo_f = a;
  }
  {
    int a = lo_f, b = S - 1;
    while (a < b) {
      int m = a + (b - a + 1) / 2;
      if (v->min_t[m] <= t)
        a = m;
      else
        b = m - 1;
    }
    hi_f = a;
  }

  int s_lo = lo_f, s_hi = hi_f;
  if (v->band_half < S) {
    int center = (int)((int64_t)t * (S - 1) /
                       (v->total_frames > 1 ? v->total_frames - 1 : 1));
    int b_lo = center - v->band_half;
    int b_hi = center + v->band_half;
    if (b_lo > s_lo) s_lo = b_lo;
    if (b_hi < s_hi) s_hi = b_hi;
    if (s_lo > s_hi) {
      s_lo = lo_f;
      s_hi = hi_f;
    }
  }

  v->row_lo[t] = s_lo;
  v->row_hi[t] = s_hi;
  uint64_t bit_off = v->bp_bits;
  v->row_bit_off[t] = bit_off;
  int width = s_hi - s_lo + 1;
  v->bp_bits += (uint64_t)width * 2u;
  viterbi_ensure_bp(v, v->bp_bits);

  int prev_lo = v->row_lo[t - 1];
  int prev_hi = v->row_hi[t - 1];
  for (int s = s_lo; s <= s_hi; s++) {
    float best = (s >= prev_lo && s <= prev_hi) ? v->prev_dp[s] : -INFINITY;
    uint8_t code = VIT_STAY;
    if (s >= 1 && s - 1 >= prev_lo && s - 1 <= prev_hi &&
        v->prev_dp[s - 1] > best) {
      best = v->prev_dp[s - 1];
      code = VIT_PREV1;
    }
    if (viterbi_can_skip(v, s) && s - 2 >= prev_lo && s - 2 <= prev_hi &&
        v->prev_dp[s - 2] > best) {
      best = v->prev_dp[s - 2];
      code = VIT_PREV2;
    }
    v->curr_dp[s] =
        isfinite(best) ? best + logp[viterbi_state_col(v, s)] : -INFINITY;
    uint64_t bpos = bit_off + (uint64_t)(s - s_lo) * 2u;
    v->bp_bytes[bpos >> 3] =
        (uint8_t)((v->bp_bytes[bpos >> 3] & ~(VIT_CODE_MASK << (bpos & 7u))) |
                  (code << (bpos & 7u)));
  }

  // Clear stale entries outside the new active window and swap DP rows.
  for (int s = prev_lo; s <= prev_hi; s++) v->prev_dp[s] = -INFINITY;
  float *tmp = v->prev_dp;
  v->prev_dp = v->curr_dp;
  v->curr_dp = tmp;
}

// Backtracks the optimal CTC alignment path and populates tok_first/tok_last
// for each target token. Returns 1 on success or 0 when no finite path exists.
static int viterbi_finish(const CtcViterbi *v, int *tok_first, int *tok_last) {
  if (v->cur_frame != v->total_frames) return 0;
  int S = v->n_states;
  int last_t = v->total_frames - 1;
  int lo = v->row_lo[last_t], hi = v->row_hi[last_t];
  float s_end = (S - 1 >= lo && S - 1 <= hi) ? v->prev_dp[S - 1] : -INFINITY;
  float s_pen = (S - 2 >= lo && S - 2 <= hi) ? v->prev_dp[S - 2] : -INFINITY;
  if (!isfinite(s_end) && !isfinite(s_pen)) return 0;

  for (int i = 0; i < v->n_tokens; i++) {
    tok_first[i] = -1;
    tok_last[i] = -1;
  }

  int s = s_end >= s_pen ? (S - 1) : (S - 2);
  for (int t = last_t; t >= 0; t--) {
    if (s & 1) {
      int ti = s >> 1;
      tok_first[ti] = t;
      if (tok_last[ti] < 0) tok_last[ti] = t;
    }
    if (t == 0) break;
    int rlo = v->row_lo[t];
    uint64_t bpos = v->row_bit_off[t] + (uint64_t)(s - rlo) * 2u;
    uint8_t code = (v->bp_bytes[bpos >> 3] >> (bpos & 7u)) & VIT_CODE_MASK;
    if (code == VIT_PREV1)
      s -= 1;
    else if (code == VIT_PREV2)
      s -= 2;
  }

  for (int i = 0; i < v->n_tokens; i++) {
    if (tok_first[i] < 0 || tok_last[i] < tok_first[i]) return 0;
  }
  return 1;
}

int nemo_ctc_viterbi_align(const float *logp, int frames, int n_gather,
                           const int *tokens, const int *token_slot,
                           int n_tokens, int blank_slot, int *tok_first,
                           int *tok_last) {
  CtcViterbi vit;
  if (!viterbi_init(&vit, frames, tokens, token_slot, n_tokens, blank_slot, 0))
    return 0;
  for (int t = 0; t < frames; t++) {
    viterbi_step(&vit, logp + (size_t)t * n_gather);
  }
  int ok = viterbi_finish(&vit, tok_first, tok_last);
  viterbi_free(&vit);
  return ok;
}

// ---- GPU CTC Decoder ----

// Linear projection in BF16 [w_rows, k] with optional FP32 bias [n].
typedef struct {
  // Zero-padded BF16 weight matrix [w_rows, k].
  Buf w;
  // Optional FP32 bias [n].
  Buf bias;
  // Output width, inner width, padded weight rows and bias presence flag.
  int n, k, w_rows, has_bias;
} CtcLinear;

// Weights for one CTC bridging Transformer layer.
typedef struct {
  // Pre-attention and pre-FFN LayerNorm weights and biases (FP32).
  Buf norm1_w, norm1_b, norm2_w, norm2_b;
  // Per-head Q and K LayerNorm weights and biases (FP32 [80]).
  Buf q_norm_w, q_norm_b, k_norm_w, k_norm_b;
  // QKV, output and FFN linear projections.
  CtcLinear w_qkv, out_proj, ffn_up, ffn_down;
} CtcBlock;

// Parameter bundle for converting an F32 or BF16 2D matrix on disk to padded
// BF16 in GPU memory.
typedef struct {
  // Source tensor in model-ctc-bf16.safetensors.
  Tensor src;
  // Active rows, padded rows and columns.
  int n, w_rows, k;
} CtcMatFillParam;

// Parameter bundle for transposing one temporal offset of ConvTranspose1d
// weight [in_ch, out_ch, 2] into BF16 [out_ch, in_ch].
typedef struct {
  // Source ConvTranspose1d weight tensor on disk.
  Tensor src;
  // Feature width (1280) and temporal offset (0 for even, 1 for odd frames).
  int dim, k_idx;
} CtcUpFillParam;

// Parameter bundle for generating CTC RoPE cosine or sine tables.
typedef struct {
  // Maximum CTC sequence length and half rotated dimension (20).
  int max_len, half_rot;
  // Nonzero for sine table, zero for cosine table.
  int is_sin;
} CtcRopeFillParam;

// Transient GPU workspace buffers allocated for one CTC decoder pass.
typedef struct {
  // Input ASR frames [max_asr, 1280] and even/odd upsampled halves.
  Buf asr_in, up_even, up_odd;
  // Upsampled states, Transformer hidden states, norm scratch and bridged
  // states.
  Buf up, h, norm, bridged;
  // Fused QKV, head-major BF16 Q/K/V, attention output and FFN hidden state.
  Buf qkv, q_hm, k_hm, v_hm, attn_out, ffn;
  // Chunked vocabulary logits [chunk, 23863], gather IDs, gathered logp and
  // argmax.
  Buf logits, gather_ids, out_logp, out_argmax;
} CtcScratch;

// Loaded Nemotron CTC timestamp decoder and aligner handle.
struct NemoCtcAligner {
  // GPU handle, owning memory account and workspace slot.
  Gpu *gpu;
  MemoryAccount *account;
  MemorySlot *work;
  // Open model-ctc-bf16.safetensors index.
  TensorIndex *index;
  // Reloadable weight group holding all ~84 MiB of BF16 CTC decoder weights.
  WeightGroup *weights;
  // Embedded SentencePiece BPE tokenizer.
  NemoSpmTokenizer *spm;
  // Scalar residual blend scale (0.1).
  float residual_scale;

  // Compiled Metal kernel identifiers from src/kernels/nemotron_audio.metal.
  int k_ln, k_mm_bf16, k_qkv, k_attn80, k_interleave, k_resid, k_logsoftmax;

  // RoPE tables for head_dim=80, half_rot=20.
  Buf rope_cos, rope_sin;
  // 2x ConvTranspose1d even and odd linear projections [1280, 1280] + shared
  // bias.
  CtcLinear up0, up1;
  // Bridging Transformer embedding/final LayerNorms and 2 blocks.
  Buf embed_nw, embed_nb, final_nw, final_nb;
  CtcBlock layers[NEMO_CTC_LAYERS];
  // 23863-class CTC output head [23872, 1280] + bias [23863].
  CtcLinear head;
};

// Reads exact byte count from safetensors Tensor t at relative byte offset off.
static void read_tensor_slice(Tensor t, uint64_t rel_off, void *dst,
                              size_t bytes) {
  uint8_t *p = dst;
  uint64_t off = t.off + rel_off;
  size_t left = bytes;
  while (left > 0) {
    ssize_t n = pread(t.fd, p, left, (off_t)off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("ctc: short read on safetensors weight");
    p += n;
    off += (uint64_t)n;
    left -= (size_t)n;
  }
}

// WeightFill callback: reads F32 or BF16 matrix [n, k] in row chunks and
// writes zero-padded BF16 [w_rows, k] into buffer.
static void fill_mat_bf16_padded(Buf *buffer, const void *parameters) {
  const CtcMatFillParam *p = parameters;
  uint16_t *dst = (uint16_t *)buffer->ptr;
  size_t total = (size_t)p->w_rows * p->k;
  if (p->src.dtype == 'f') {
    float *tmp = xmalloc((size_t)CTC_F32_LOAD_ROWS * p->k * sizeof(float));
    for (int r = 0; r < p->n; r += CTC_F32_LOAD_ROWS) {
      int take = p->n - r;
      if (take > CTC_F32_LOAD_ROWS) take = CTC_F32_LOAD_ROWS;
      size_t elems = (size_t)take * p->k;
      read_tensor_slice(p->src, (uint64_t)r * p->k * sizeof(float), tmp,
                        elems * sizeof(float));
      uint16_t *row_dst = dst + (size_t)r * p->k;
      for (size_t i = 0; i < elems; i++) row_dst[i] = f32_to_bf16(tmp[i]);
    }
    free(tmp);
  } else if (p->src.dtype == 'b') {
    read_tensor_slice(p->src, 0, dst, (size_t)p->n * p->k * sizeof(uint16_t));
  } else {
    die("ctc: unsupported matrix dtype %d", p->src.dtype);
  }
  size_t valid = (size_t)p->n * p->k;
  if (valid < total) memset(dst + valid, 0, (total - valid) * sizeof(uint16_t));
}

// WeightFill callback: transposes ConvTranspose1d weight [dim, dim, 2] at
// offset k_idx into BF16 row-major [dim, dim].
static void fill_upsample_half(Buf *buffer, const void *parameters) {
  const CtcUpFillParam *p = parameters;
  size_t elems = (size_t)p->dim * p->dim * NEMO_CTC_UP_FACTOR;
  uint16_t *dst = (uint16_t *)buffer->ptr;
  if (p->src.dtype == 'f') {
    float *src = xmalloc(elems * sizeof(float));
    read_tensor_slice(p->src, 0, src, elems * sizeof(float));
    for (int ic = 0; ic < p->dim; ic++) {
      for (int oc = 0; oc < p->dim; oc++) {
        size_t sidx =
            ((size_t)ic * p->dim + oc) * NEMO_CTC_UP_FACTOR + p->k_idx;
        dst[(size_t)oc * p->dim + ic] = f32_to_bf16(src[sidx]);
      }
    }
    free(src);
  } else if (p->src.dtype == 'b') {
    uint16_t *src = xmalloc(elems * sizeof(uint16_t));
    read_tensor_slice(p->src, 0, src, elems * sizeof(uint16_t));
    for (int ic = 0; ic < p->dim; ic++) {
      for (int oc = 0; oc < p->dim; oc++) {
        size_t sidx =
            ((size_t)ic * p->dim + oc) * NEMO_CTC_UP_FACTOR + p->k_idx;
        dst[(size_t)oc * p->dim + ic] = src[sidx];
      }
    }
    free(src);
  } else {
    die("ctc: unsupported upsample dtype %d", p->src.dtype);
  }
}

// WeightFill callback: precomputes RoPE cosine or sine table [max_len,
// half_rot].
static void fill_ctc_rope(Buf *buffer, const void *parameters) {
  const CtcRopeFillParam *p = parameters;
  float *dst = (float *)buffer->ptr;
  int rot_dim = 2 * p->half_rot;
  for (int pos = 0; pos < p->max_len; pos++) {
    for (int j = 0; j < p->half_rot; j++) {
      double freq = 1.0 / pow(CTC_ROPE_BASE, (double)(2 * j) / (double)rot_dim);
      double angle = (double)pos * freq;
      dst[(size_t)pos * p->half_rot + j] =
          (float)(p->is_sin ? sin(angle) : cos(angle));
    }
  }
}

// Registers a padded BF16 linear layer (plus optional FP32 bias) in group.
static void add_ctc_linear(WeightGroup *g, TensorIndex *idx, CtcLinear *lin,
                           const char *w_name, const char *b_name, int n,
                           int k) {
  Tensor wt = tensor_index_get(idx, "%s", w_name);
  int w_rows = align_up(n, (int)CTC_TILE);
  lin->n = n;
  lin->k = k;
  lin->w_rows = w_rows;
  lin->has_bias = b_name != NULL;
  CtcMatFillParam mp = {.src = wt, .n = n, .w_rows = w_rows, .k = k};
  weight_group_add_generated(g, &lin->w, (size_t)w_rows * k * sizeof(uint16_t),
                             fill_mat_bf16_padded, &mp, sizeof mp);
  if (b_name) {
    weight_group_add_vector(g, &lin->bias, tensor_index_get(idx, "%s", b_name),
                            0.0f, 0);
  }
}

int nemo_ctc_available(const char *dir) {
  if (!dir) return 0;
  char path[CTC_PATH_CAP];
  int n = snprintf(path, sizeof path, "%s/%s", dir, CTC_FILENAME);
  if (n <= 0 || (size_t)n >= sizeof path) return 0;
  return access(path, R_OK) == 0;
}

NemoCtcAligner *nemo_ctc_open(Gpu *gpu, MemoryAccount *account,
                              MemorySlot *work, const char *dir) {
  if (!gpu || !account || !work || !dir || !nemo_ctc_available(dir))
    return NULL;
  char path[CTC_PATH_CAP];
  snprintf(path, sizeof path, "%s/%s", dir, CTC_FILENAME);
  TensorIndex *idx = tensor_index_open_file(path);
  if (!idx) return NULL;

  // Parse embedded SentencePiece BPE ModelProto.
  Tensor proto_t = tensor_index_get(idx, "ctc_timestamp.tokenizer.model_proto");
  uint8_t *proto_bytes = xmalloc((size_t)proto_t.size);
  read_tensor_slice(proto_t, 0, proto_bytes, (size_t)proto_t.size);
  NemoSpmTokenizer *spm =
      nemo_spm_open_bytes(proto_bytes, (size_t)proto_t.size);
  free(proto_bytes);
  if (!spm || nemo_spm_vocab_size(spm) != NEMO_CTC_VOCAB) {
    nemo_spm_close(spm);
    tensor_index_close(idx);
    return NULL;
  }

  NemoCtcAligner *ctc = calloc(1, sizeof *ctc);
  if (!ctc) die("out of memory");
  ctc->gpu = gpu;
  ctc->account = account;
  ctc->work = work;
  ctc->index = idx;
  ctc->spm = spm;

  // Read scalar residual blend factor (default 0.1).
  ctc->residual_scale = 0.1f;
  if (tensor_index_has(idx, "ctc_timestamp.decoder.residual_scale")) {
    Tensor rs = tensor_index_get(idx, "ctc_timestamp.decoder.residual_scale");
    if (rs.dtype == 'f' && rs.size == sizeof(float)) {
      read_tensor_slice(rs, 0, &ctc->residual_scale, sizeof(float));
    } else if (rs.dtype == 'b' && rs.size == sizeof(uint16_t)) {
      uint16_t bits = 0;
      read_tensor_slice(rs, 0, &bits, sizeof bits);
      ctc->residual_scale = bf16_to_f32(bits);
    }
  }

  ctc->k_ln = gpu_find_kernel(gpu, "nemo_aud_layernorm");
  ctc->k_mm_bf16 = gpu_find_kernel(gpu, "nemo_aud_mm_bf16");
  ctc->k_qkv = gpu_find_kernel(gpu, "nemo_aud_qkv_prep");
  ctc->k_attn80 = gpu_find_kernel(gpu, "nemo_aud_flash_attn_80");
  ctc->k_interleave = gpu_find_kernel(gpu, "nemo_ctc_interleave");
  ctc->k_resid = gpu_find_kernel(gpu, "nemo_ctc_residual_add");
  ctc->k_logsoftmax = gpu_find_kernel(gpu, "nemo_ctc_log_softmax_gather");

  ctc->weights = weight_group_new(gpu, account, "nemotron_ctc");
  WeightGroup *g = ctc->weights;

  // RoPE tables [2048, 20].
  size_t rope_bytes =
      (size_t)CTC_MAX_ROPE_LEN * NEMO_CTC_HALF_ROT * sizeof(float);
  CtcRopeFillParam cp = {
      .max_len = CTC_MAX_ROPE_LEN, .half_rot = NEMO_CTC_HALF_ROT, .is_sin = 0};
  CtcRopeFillParam sp = {
      .max_len = CTC_MAX_ROPE_LEN, .half_rot = NEMO_CTC_HALF_ROT, .is_sin = 1};
  weight_group_add_generated(g, &ctc->rope_cos, rope_bytes, fill_ctc_rope, &cp,
                             sizeof cp);
  weight_group_add_generated(g, &ctc->rope_sin, rope_bytes, fill_ctc_rope, &sp,
                             sizeof sp);

  // ConvTranspose1d(1280, 1280, k=2, stride=2) split into even/odd BF16 GEMMs.
  Tensor up_w = tensor_index_get(idx, "ctc_timestamp.decoder.upsample.weight");
  Tensor up_b = tensor_index_get(idx, "ctc_timestamp.decoder.upsample.bias");
  size_t up_mat_bytes = (size_t)NEMO_CTC_DIM * NEMO_CTC_DIM * sizeof(uint16_t);
  CtcUpFillParam u0 = {.src = up_w, .dim = NEMO_CTC_DIM, .k_idx = 0};
  CtcUpFillParam u1 = {.src = up_w, .dim = NEMO_CTC_DIM, .k_idx = 1};
  ctc->up0 = (CtcLinear){.n = NEMO_CTC_DIM,
                         .k = NEMO_CTC_DIM,
                         .w_rows = NEMO_CTC_DIM,
                         .has_bias = 1};
  ctc->up1 = ctc->up0;
  weight_group_add_generated(g, &ctc->up0.w, up_mat_bytes, fill_upsample_half,
                             &u0, sizeof u0);
  weight_group_add_generated(g, &ctc->up1.w, up_mat_bytes, fill_upsample_half,
                             &u1, sizeof u1);
  weight_group_add_vector(g, &ctc->up0.bias, up_b, 0.0f, 0);
  weight_group_add_alias(g, &ctc->up1.bias, &ctc->up0.bias);

  // Bridging 2-layer Transformer norms and blocks.
  weight_group_add_vector(
      g, &ctc->embed_nw,
      tensor_index_get(idx,
                       "ctc_timestamp.decoder.transformer.embed_norm.weight"),
      0.0f, 0);
  weight_group_add_vector(
      g, &ctc->embed_nb,
      tensor_index_get(idx,
                       "ctc_timestamp.decoder.transformer.embed_norm.bias"),
      0.0f, 0);
  weight_group_add_vector(
      g, &ctc->final_nw,
      tensor_index_get(idx,
                       "ctc_timestamp.decoder.transformer.final_norm.weight"),
      0.0f, 0);
  weight_group_add_vector(
      g, &ctc->final_nb,
      tensor_index_get(idx,
                       "ctc_timestamp.decoder.transformer.final_norm.bias"),
      0.0f, 0);

  for (int l = 0; l < NEMO_CTC_LAYERS; l++) {
    CtcBlock *blk = &ctc->layers[l];
    char n1[160], n2[160];
    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.norm1.weight", l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.norm1.bias", l);
    weight_group_add_vector(g, &blk->norm1_w, tensor_index_get(idx, "%s", n1),
                            0.0f, 0);
    weight_group_add_vector(g, &blk->norm1_b, tensor_index_get(idx, "%s", n2),
                            0.0f, 0);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.w_qkv.weight",
             l);
    add_ctc_linear(g, idx, &blk->w_qkv, n1, NULL, 3 * NEMO_CTC_DIM,
                   NEMO_CTC_DIM);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.q_norm.weight",
             l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.q_norm.bias", l);
    weight_group_add_vector(g, &blk->q_norm_w, tensor_index_get(idx, "%s", n1),
                            0.0f, 0);
    weight_group_add_vector(g, &blk->q_norm_b, tensor_index_get(idx, "%s", n2),
                            0.0f, 0);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.k_norm.weight",
             l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.k_norm.bias", l);
    weight_group_add_vector(g, &blk->k_norm_w, tensor_index_get(idx, "%s", n1),
                            0.0f, 0);
    weight_group_add_vector(g, &blk->k_norm_b, tensor_index_get(idx, "%s", n2),
                            0.0f, 0);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.out_proj.weight",
             l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.attn.out_proj.bias",
             l);
    add_ctc_linear(g, idx, &blk->out_proj, n1, n2, NEMO_CTC_DIM, NEMO_CTC_DIM);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.norm2.weight", l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.norm2.bias", l);
    weight_group_add_vector(g, &blk->norm2_w, tensor_index_get(idx, "%s", n1),
                            0.0f, 0);
    weight_group_add_vector(g, &blk->norm2_b, tensor_index_get(idx, "%s", n2),
                            0.0f, 0);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.ffn.net.0.weight", l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.ffn.net.0.bias", l);
    add_ctc_linear(g, idx, &blk->ffn_up, n1, n2, NEMO_CTC_FFN, NEMO_CTC_DIM);

    snprintf(n1, sizeof n1,
             "ctc_timestamp.decoder.transformer.layers.%d.ffn.net.3.weight", l);
    snprintf(n2, sizeof n2,
             "ctc_timestamp.decoder.transformer.layers.%d.ffn.net.3.bias", l);
    add_ctc_linear(g, idx, &blk->ffn_down, n1, n2, NEMO_CTC_DIM, NEMO_CTC_FFN);
  }

  // CTC vocabulary head [23863, 1280] + bias [23863].
  add_ctc_linear(g, idx, &ctc->head,
                 "ctc_timestamp.decoder.decoder_layers.0.weight",
                 "ctc_timestamp.decoder.decoder_layers.0.bias",
                 NEMO_CTC_CLASSES, NEMO_CTC_DIM);

  return ctc;
}

void nemo_ctc_close(NemoCtcAligner *ctc) {
  if (!ctc) return;
  if (ctc->weights) weight_group_free(ctc->weights);
  nemo_spm_close(ctc->spm);
  if (ctc->index) tensor_index_close(ctc->index);
  free(ctc);
}

void nemo_ctc_pin(NemoCtcAligner *ctc, int active) {
  if (!ctc || !ctc->weights) return;
  memory_pin(weight_group_slot(ctc->weights), active);
}

int nemo_ctc_is_loaded(const NemoCtcAligner *ctc) {
  return ctc && ctc->weights && weight_group_resident_bytes(ctc->weights) > 0;
}

uint64_t nemo_ctc_weight_bytes(const NemoCtcAligner *ctc) {
  return (ctc && ctc->weights) ? weight_group_required_bytes(ctc->weights) : 0;
}

// Allocates a GPU workspace buffer, returning 0 if memory does not fit.
static int try_ctc_buf(Gpu *gpu, MemorySlot *work, Buf *out, size_t bytes) {
  *out = gpu_try_alloc_required(gpu, work, bytes);
  return out->obj != NULL;
}

// Frees all allocated buffers inside CtcScratch.
static void free_ctc_scratch(CtcScratch *s) {
  Buf *list[] = {&s->out_argmax, &s->out_logp, &s->gather_ids, &s->logits,
                 &s->ffn,        &s->attn_out, &s->v_hm,       &s->k_hm,
                 &s->q_hm,       &s->qkv,      &s->bridged,    &s->norm,
                 &s->h,          &s->up,       &s->up_odd,     &s->up_even,
                 &s->asr_in};
  for (size_t i = 0; i < sizeof list / sizeof *list; i++) {
    if (list[i]->obj) gpu_free(list[i]);
  }
  *s = (CtcScratch){0};
}

// Dispatches LayerNorm over rows of width dim.
static void run_ctc_ln(GpuCommandBuffer *cb, const NemoCtcAligner *ctc,
                       const Buf *x, const Buf *w, const Buf *b, const Buf *y,
                       int rows, int dim, int round) {
  CtcLnParam p = {
      .dim = (uint32_t)dim, .round = (uint32_t)round, .eps = CTC_LN_EPS};
  gpu_select_kernel(cb, ctc->k_ln);
  gpu_bind_buffer(cb, 0, x, 0);
  gpu_bind_buffer(cb, 1, w, 0);
  gpu_bind_buffer(cb, 2, b, 0);
  gpu_bind_buffer(cb, 3, y, 0);
  gpu_copy_bytes(cb, 4, &p, sizeof p);
  gpu_run_kernel(cb, (uint32_t)rows, 1, 1, CTC_ROW_THREADS, 1, 1);
}

// Dispatches BF16 tiled matrix multiplication y = act(x @ W^T + bias) + res.
static void run_ctc_mm(GpuCommandBuffer *cb, const NemoCtcAligner *ctc,
                       const CtcLinear *lin, const Buf *x, size_t x_off,
                       const Buf *res, const Buf *y, int m, uint32_t act,
                       int round) {
  CtcMmParam p = {.m = (uint32_t)m,
                  .n = (uint32_t)lin->n,
                  .k = (uint32_t)lin->k,
                  .w_rows = (uint32_t)lin->w_rows,
                  .act = act,
                  .has_bias = (uint32_t)lin->has_bias,
                  .has_res = res != NULL ? 1u : 0u,
                  .round = (uint32_t)round,
                  .scale = 1.0f};
  uint32_t tiles_m = ((uint32_t)m + CTC_TILE - 1u) / CTC_TILE;
  uint32_t tiles_n = ((uint32_t)lin->n + CTC_TILE - 1u) / CTC_TILE;
  const Buf *bias = lin->has_bias ? &lin->bias : &lin->w;
  const Buf *rbuf = res ? res : y;
  gpu_select_kernel(cb, ctc->k_mm_bf16);
  gpu_bind_buffer(cb, 0, &lin->w, 0);
  gpu_bind_buffer(cb, 1, x, x_off);
  gpu_bind_buffer(cb, 2, bias, 0);
  gpu_bind_buffer(cb, 3, rbuf, 0);
  gpu_bind_buffer(cb, 4, y, 0);
  gpu_copy_bytes(cb, 5, &p, sizeof p);
  gpu_run_kernel(cb, tiles_m * tiles_n, 1, 1, CTC_MM_THREADS, 1, 1);
}

// Runs the CTC upsampler, 2-layer bridging Transformer and scaled residual
// blend on one window of win_toks ASR frames in s->asr_in, leaving
// [2 * win_toks, 1280] blended states in s->h.
static void run_ctc_backbone_window(const NemoCtcAligner *ctc,
                                    const CtcScratch *s, int win_toks,
                                    const PassWait *wait) {
  int up_len = win_toks * NEMO_CTC_UP_FACTOR;
  int seq_stride = align_up(up_len, (int)CTC_TILE);
  float log2_scale = (float)(CTC_LOG2_E / sqrt((double)NEMO_CTC_HEAD_DIM));

  // 1. 2x ConvTranspose1d upsampling via even/odd GEMMs + interleave -> s->up.
  GpuCommandBuffer cb = gpu_begin(ctc->gpu);
  run_ctc_mm(&cb, ctc, &ctc->up0, &s->asr_in, 0, NULL, &s->up_even, win_toks,
             CTC_ACT_NONE, 1);
  run_ctc_mm(&cb, ctc, &ctc->up1, &s->asr_in, 0, NULL, &s->up_odd, win_toks,
             CTC_ACT_NONE, 1);

  CtcInterleaveParam ip = {.frames = (uint32_t)win_toks, .dim = NEMO_CTC_DIM};
  gpu_select_kernel(&cb, ctc->k_interleave);
  gpu_bind_buffer(&cb, 0, &s->up_even, 0);
  gpu_bind_buffer(&cb, 1, &s->up_odd, 0);
  gpu_bind_buffer(&cb, 2, &s->up, 0);
  gpu_copy_bytes(&cb, 3, &ip, sizeof ip);
  uint32_t gx =
      ((uint32_t)(NEMO_CTC_DIM / 4) + CTC_PACK_THREADS - 1u) / CTC_PACK_THREADS;
  gpu_run_kernel(&cb, gx, (uint32_t)win_toks, 1, CTC_PACK_THREADS, 1, 1);

  // 2. Bridging Transformer embed_norm -> s->h.
  run_ctc_ln(&cb, ctc, &s->up, &ctc->embed_nw, &ctc->embed_nb, &s->h, up_len,
             NEMO_CTC_DIM, 1);
  gpu_commit(&cb);

  // 3. Two bridging Transformer blocks.
  for (int l = 0; l < NEMO_CTC_LAYERS; l++) {
    if (wait) pass_wait(wait);
    const CtcBlock *blk = &ctc->layers[l];
    cb = gpu_begin(ctc->gpu);

    run_ctc_ln(&cb, ctc, &s->h, &blk->norm1_w, &blk->norm1_b, &s->norm, up_len,
               NEMO_CTC_DIM, 1);
    run_ctc_mm(&cb, ctc, &blk->w_qkv, &s->norm, 0, NULL, &s->qkv, up_len,
               CTC_ACT_NONE, 1);

    CtcQkvParam qp = {.rows = (uint32_t)up_len,
                      .seq_stride = (uint32_t)seq_stride,
                      .heads = NEMO_CTC_HEADS,
                      .head_dim = NEMO_CTC_HEAD_DIM,
                      .half_rot = NEMO_CTC_HALF_ROT,
                      .qk_norm = 1,
                      .eps = CTC_LN_EPS};
    gpu_select_kernel(&cb, ctc->k_qkv);
    gpu_bind_buffer(&cb, 0, &s->qkv, 0);
    gpu_bind_buffer(&cb, 1, &blk->q_norm_w, 0);
    gpu_bind_buffer(&cb, 2, &blk->q_norm_b, 0);
    gpu_bind_buffer(&cb, 3, &blk->k_norm_w, 0);
    gpu_bind_buffer(&cb, 4, &blk->k_norm_b, 0);
    gpu_bind_buffer(&cb, 5, &ctc->rope_cos, 0);
    gpu_bind_buffer(&cb, 6, &ctc->rope_sin, 0);
    gpu_bind_buffer(&cb, 7, &s->q_hm, 0);
    gpu_bind_buffer(&cb, 8, &s->k_hm, 0);
    gpu_bind_buffer(&cb, 9, &s->v_hm, 0);
    gpu_copy_bytes(&cb, 10, &qp, sizeof qp);
    gpu_run_kernel(&cb, NEMO_CTC_HEADS, (uint32_t)seq_stride, 1,
                   NEMO_CTC_HEAD_DIM, 1, 1);

    CtcAttnParam ap = {.rows = (uint32_t)up_len,
                       .seq_stride = (uint32_t)seq_stride,
                       .heads = NEMO_CTC_HEADS,
                       .log2_scale = log2_scale};
    gpu_select_kernel(&cb, ctc->k_attn80);
    gpu_bind_buffer(&cb, 0, &s->q_hm, 0);
    gpu_bind_buffer(&cb, 1, &s->k_hm, 0);
    gpu_bind_buffer(&cb, 2, &s->v_hm, 0);
    gpu_bind_buffer(&cb, 3, &s->attn_out, 0);
    gpu_copy_bytes(&cb, 4, &ap, sizeof ap);
    gpu_run_kernel(&cb, (uint32_t)(seq_stride / (int)CTC_TILE), NEMO_CTC_HEADS,
                   1, CTC_MM_THREADS, 1, 1);

    run_ctc_mm(&cb, ctc, &blk->out_proj, &s->attn_out, 0, &s->h, &s->h, up_len,
               CTC_ACT_NONE, 1);

    run_ctc_ln(&cb, ctc, &s->h, &blk->norm2_w, &blk->norm2_b, &s->norm, up_len,
               NEMO_CTC_DIM, 1);
    run_ctc_mm(&cb, ctc, &blk->ffn_up, &s->norm, 0, NULL, &s->ffn, up_len,
               CTC_ACT_GELU, 1);
    run_ctc_mm(&cb, ctc, &blk->ffn_down, &s->ffn, 0, &s->h, &s->h, up_len,
               CTC_ACT_NONE, 1);
    gpu_commit(&cb);
  }

  // 4. final_norm -> s->bridged, then residual blend h = up + scale * bridged.
  cb = gpu_begin(ctc->gpu);
  run_ctc_ln(&cb, ctc, &s->h, &ctc->final_nw, &ctc->final_nb, &s->bridged,
             up_len, NEMO_CTC_DIM, 1);
  uint32_t count4 = (uint32_t)(((size_t)up_len * NEMO_CTC_DIM) / 4u);
  CtcResidParam rp = {
      .count4 = count4, .round = 1, .scale = ctc->residual_scale};
  gpu_select_kernel(&cb, ctc->k_resid);
  gpu_bind_buffer(&cb, 0, &s->up, 0);
  gpu_bind_buffer(&cb, 1, &s->bridged, 0);
  gpu_bind_buffer(&cb, 2, &s->h, 0);
  gpu_copy_bytes(&cb, 3, &rp, sizeof rp);
  gpu_run_kernel(&cb, (count4 + CTC_RESID_THREADS - 1u) / CTC_RESID_THREADS, 1,
                 1, CTC_RESID_THREADS, 1, 1);
  gpu_commit(&cb);
}

// Allocates the bounded GPU scratch buffers needed for CTC decoding over
// windows of up to max_asr ASR tokens and n_gather gathered vocabulary classes.
static int alloc_ctc_scratch(const NemoCtcAligner *ctc, CtcScratch *s,
                             int max_asr, int n_gather) {
  int max_up = max_asr * NEMO_CTC_UP_FACTOR;
  int seq_stride = align_up(max_up, (int)CTC_TILE);
  int chunk = max_up < CTC_LOGIT_CHUNK ? max_up : CTC_LOGIT_CHUNK;
  int gather_alloc = n_gather > 0 ? n_gather : 1;

  *s = (CtcScratch){0};
  int ok = try_ctc_buf(ctc->gpu, ctc->work, &s->asr_in,
                       (size_t)max_asr * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->up_even,
                       (size_t)max_asr * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->up_odd,
                       (size_t)max_asr * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->up,
                       (size_t)max_up * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->h,
                       (size_t)max_up * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->norm,
                       (size_t)max_up * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->bridged,
                       (size_t)max_up * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->qkv,
                       (size_t)max_up * 3u * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->q_hm,
                       (size_t)NEMO_CTC_HEADS * seq_stride * NEMO_CTC_HEAD_DIM *
                           sizeof(uint16_t)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->k_hm,
                       (size_t)NEMO_CTC_HEADS * seq_stride * NEMO_CTC_HEAD_DIM *
                           sizeof(uint16_t)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->v_hm,
                       (size_t)NEMO_CTC_HEADS * seq_stride * NEMO_CTC_HEAD_DIM *
                           sizeof(uint16_t)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->attn_out,
                       (size_t)max_up * NEMO_CTC_DIM * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->ffn,
                       (size_t)max_up * NEMO_CTC_FFN * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->logits,
                       (size_t)chunk * NEMO_CTC_CLASSES * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->gather_ids,
                       (size_t)gather_alloc * sizeof(uint32_t)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->out_logp,
                       (size_t)chunk * gather_alloc * sizeof(float)) &&
           try_ctc_buf(ctc->gpu, ctc->work, &s->out_argmax,
                       (size_t)chunk * sizeof(uint32_t));
  if (!ok) free_ctc_scratch(s);
  return ok;
}

// Unpacks BF16 ASR frames [start .. start + count) into FP32 s->asr_in.
static void unpack_asr_window(const uint16_t *asr_bf16, int start, int count,
                              float *dst) {
  const uint16_t *src = asr_bf16 + (size_t)start * NEMO_CTC_DIM;
  size_t elems = (size_t)count * NEMO_CTC_DIM;
  for (size_t i = 0; i < elems; i++) dst[i] = bf16_to_f32(src[i]);
}

// Callback invoked per chunk of CTC frames with gathered log-probabilities and
// per-frame argmax class indices.
typedef void (*CtcChunkCallback)(void *ctx, const float *logp,
                                 const uint32_t *argmax, int rows,
                                 int n_gather);

// Streams the entire ASR sequence through the CTC decoder (single-pass up to
// 500 ASR tokens, or 500-token core + 50-token left/right context windows above
// 500 tokens), invoking on_chunk for each contiguous chunk of up to 256 CTC
// frames in temporal order.
static int run_ctc_stream(NemoCtcAligner *ctc, const uint16_t *asr_bf16,
                          int asr_tokens, const uint32_t *gather_ids,
                          int n_gather, const PassWait *wait,
                          CtcChunkCallback on_chunk, void *ctx, char *error,
                          size_t error_cap) {
  if (!weight_group_load(ctc->weights)) {
    set_ctc_err(error, error_cap,
                "CTC timestamp weights do not fit in available memory");
    return 0;
  }

  int use_window = asr_tokens > NEMO_AUDIO_ONLINE_TOKENS;
  int max_asr = use_window
                    ? (NEMO_AUDIO_ONLINE_TOKENS + NEMO_AUDIO_CTX_LEFT_TOKENS +
                       NEMO_AUDIO_CTX_RIGHT_TOKENS)
                    : asr_tokens;
  CtcScratch s;
  if (!alloc_ctc_scratch(ctc, &s, max_asr, n_gather)) {
    set_ctc_err(error, error_cap,
                "CTC timestamp workspace does not fit in available memory");
    return 0;
  }

  if (n_gather > 0 && gather_ids) {
    memcpy(s.gather_ids.ptr, gather_ids, (size_t)n_gather * sizeof(uint32_t));
  } else {
    ((uint32_t *)s.gather_ids.ptr)[0] = NEMO_CTC_BLANK_ID;
  }
  int eff_gather = n_gather > 0 ? n_gather : 1;

  int num_windows = use_window ? (asr_tokens + NEMO_AUDIO_ONLINE_TOKENS - 1) /
                                     NEMO_AUDIO_ONLINE_TOKENS
                               : 1;
  for (int wi = 0; wi < num_windows; wi++) {
    int core_start = wi * NEMO_AUDIO_ONLINE_TOKENS;
    int core_end =
        use_window ? core_start + NEMO_AUDIO_ONLINE_TOKENS : asr_tokens;
    if (core_end > asr_tokens) core_end = asr_tokens;
    int ctx_start = use_window ? core_start - NEMO_AUDIO_CTX_LEFT_TOKENS : 0;
    if (ctx_start < 0) ctx_start = 0;
    int ctx_end =
        use_window ? core_end + NEMO_AUDIO_CTX_RIGHT_TOKENS : asr_tokens;
    if (ctx_end > asr_tokens) ctx_end = asr_tokens;

    int win_toks = ctx_end - ctx_start;
    int up_core_start = (core_start - ctx_start) * NEMO_CTC_UP_FACTOR;
    int up_core_len = (core_end - core_start) * NEMO_CTC_UP_FACTOR;

    unpack_asr_window(asr_bf16, ctx_start, win_toks, (float *)s.asr_in.ptr);
    run_ctc_backbone_window(ctc, &s, win_toks, wait);

    for (int off = 0; off < up_core_len; off += CTC_LOGIT_CHUNK) {
      if (wait) pass_wait(wait);
      int rows = up_core_len - off;
      if (rows > CTC_LOGIT_CHUNK) rows = CTC_LOGIT_CHUNK;
      size_t h_byte_off =
          (size_t)(up_core_start + off) * NEMO_CTC_DIM * sizeof(float);

      GpuCommandBuffer cb = gpu_begin(ctc->gpu);
      run_ctc_mm(&cb, ctc, &ctc->head, &s.h, h_byte_off, NULL, &s.logits, rows,
                 CTC_ACT_NONE, 0);
      CtcLogSoftmaxParam lp = {.rows = (uint32_t)rows,
                               .vocab = NEMO_CTC_CLASSES,
                               .n_gather = (uint32_t)eff_gather};
      gpu_select_kernel(&cb, ctc->k_logsoftmax);
      gpu_bind_buffer(&cb, 0, &s.logits, 0);
      gpu_bind_buffer(&cb, 1, &s.gather_ids, 0);
      gpu_bind_buffer(&cb, 2, &s.out_logp, 0);
      gpu_bind_buffer(&cb, 3, &s.out_argmax, 0);
      gpu_copy_bytes(&cb, 4, &lp, sizeof lp);
      gpu_run_kernel(&cb, (uint32_t)rows, 1, 1, CTC_ROW_THREADS, 1, 1);
      gpu_commit(&cb);
      gpu_sync(ctc->gpu);

      on_chunk(ctx, (const float *)s.out_logp.ptr,
               (const uint32_t *)s.out_argmax.ptr, rows, eff_gather);
    }
  }

  free_ctc_scratch(&s);
  return 1;
}

// Streaming Viterbi callback feeding chunk log-probabilities into CtcViterbi.
static void on_viterbi_chunk(void *ctx, const float *logp,
                             const uint32_t *argmax, int rows, int n_gather) {
  (void)argmax;
  CtcViterbi *vit = ctx;
  for (int r = 0; r < rows; r++) {
    viterbi_step(vit, logp + (size_t)r * n_gather);
  }
}

// Assigns proportional fallback timestamps across [0, duration] by byte length
// when CTC forced alignment cannot align the transcript.
static void assign_proportional_times(NemoCtcWord *words, int n_words,
                                      double duration) {
  if (n_words <= 0) return;
  if (duration < 0.0) duration = 0.0;
  double total_weight = 0.0;
  for (int i = 0; i < n_words; i++) {
    int w = words[i].byte_end - words[i].byte_start;
    total_weight += (double)(w > 0 ? w : 1);
  }
  double cur = 0.0;
  for (int i = 0; i < n_words; i++) {
    int w = words[i].byte_end - words[i].byte_start;
    double share = duration * ((double)(w > 0 ? w : 1) / total_weight);
    words[i].start = cur;
    words[i].end = (i + 1 == n_words) ? duration : (cur + share);
    cur = words[i].end;
  }
}

// Returns nonzero when a vocabulary piece consists solely of punctuation and
// optional leading SentencePiece word-boundary markers.
static int is_punct_piece(const char *text, size_t len) {
  if (!text || len == 0) return 0;
  size_t i = 0;
  int saw_punct = 0;
  while (i < len) {
    if (i + SPM_SPACE_LEN <= len &&
        memcmp(text + i, SPM_SPACE_UTF8, SPM_SPACE_LEN) == 0) {
      i += SPM_SPACE_LEN;
      continue;
    }
    uint32_t cp = 0;
    size_t clen = utf8_decode(text + i, len - i, &cp);
    i += clen;
    int ascii_p = (cp >= 0x21u && cp <= 0x2Fu) ||
                  (cp >= 0x3Au && cp <= 0x40u) ||
                  (cp >= 0x5Bu && cp <= 0x60u) || (cp >= 0x7Bu && cp <= 0x7Eu);
    int uni_p = (cp >= 0x2010u && cp <= 0x2027u) ||
                (cp >= 0x2030u && cp <= 0x205Eu) || is_cjk_punct(cp);
    if (!ascii_p && !uni_p) return 0;
    saw_punct = 1;
  }
  return saw_punct;
}

// Fills start/end seconds for any word that had zero non-unk CTC tokens by
// interpolating smoothly between its aligned neighbors within [0, duration].
static void fill_unaligned_words(NemoCtcWord *words, const int *has_time,
                                 int n_words, double duration) {
  int i = 0;
  while (i < n_words) {
    if (has_time[i]) {
      i++;
      continue;
    }
    int j = i;
    while (j < n_words && !has_time[j]) j++;
    double t0 = i > 0 ? words[i - 1].end : 0.0;
    double t1 = j < n_words ? words[j].start : duration;
    if (t1 < t0) t1 = t0;
    int run = j - i;
    for (int k = 0; k < run; k++) {
      words[i + k].start = t0 + (t1 - t0) * ((double)k / (double)run);
      words[i + k].end = t0 + (t1 - t0) * ((double)(k + 1) / (double)run);
    }
    i = j;
  }
}

int nemo_ctc_align(NemoCtcAligner *ctc, const uint16_t *asr_bf16,
                   int asr_tokens, double duration, const char *transcript,
                   size_t transcript_len, const PassWait *wait,
                   NemoCtcWord **out_words, int *out_count, char *error,
                   size_t error_cap) {
  if (out_words) *out_words = NULL;
  if (out_count) *out_count = 0;
  if (!ctc || !asr_bf16 || asr_tokens <= 0) {
    set_ctc_err(error, error_cap, "invalid CTC alignment input");
    return 0;
  }

  double clip_dur = duration > 0.0 ? duration
                                   : (double)(asr_tokens * NEMO_CTC_UP_FACTOR) *
                                         NEMO_CTC_FRAME_SECONDS;

  NemoCtcWord *words = NULL;
  int n_words = 0;
  nemo_ctc_split_words(transcript, transcript_len, &words, &n_words);
  if (n_words == 0) {
    *out_words = NULL;
    *out_count = 0;
    return 1;
  }

  // Tokenize each word with SentencePiece BPE and map each non-unk token to its
  // word index and unique gathered column.
  int *word_tok_start = xmalloc((size_t)n_words * sizeof(int));
  int *word_tok_end = xmalloc((size_t)n_words * sizeof(int));
  int tok_cap = n_words * 4 + 16;
  int *tokens = xmalloc((size_t)tok_cap * sizeof(int));
  int *token_slot = xmalloc((size_t)tok_cap * sizeof(int));
  int n_tokens = 0;

  int *vocab_to_slot = xmalloc((size_t)NEMO_CTC_CLASSES * sizeof(int));
  for (int i = 0; i < NEMO_CTC_CLASSES; i++) vocab_to_slot[i] = -1;
  uint32_t *gather_ids = xmalloc((size_t)NEMO_CTC_CLASSES * sizeof(uint32_t));
  int n_gather = 0;

  int piece_buf[CTC_MAX_WORD_TOKENS];
  for (int w = 0; w < n_words; w++) {
    int start_b = words[w].byte_start;
    int end_b = words[w].byte_end;
    // Prepend SentencePiece word-boundary marker at start of transcript or
    // after whitespace (avoid injecting markers between split CJK characters).
    int add_prefix = 1;
    if (start_b > 0) {
      size_t prev_i = (size_t)start_b - 1;
      while (prev_i > 0 && (((uint8_t)transcript[prev_i] & 0xC0u) == 0x80u))
        prev_i--;
      uint32_t prev_cp = 0;
      utf8_decode(transcript + prev_i, (size_t)start_b - prev_i, &prev_cp);
      add_prefix = is_space_cp(prev_cp);
    }
    int np = nemo_spm_encode_word(ctc->spm, transcript + start_b,
                                  (size_t)(end_b - start_b), add_prefix,
                                  piece_buf, CTC_MAX_WORD_TOKENS);
    word_tok_start[w] = n_tokens;
    for (int j = 0; j < np; j++) {
      int tid = piece_buf[j];
      if (tid == ctc->spm->unk_id || tid < 0 || tid >= NEMO_CTC_VOCAB) continue;
      if (n_tokens >= tok_cap) {
        tok_cap *= 2;
        int *nt = realloc(tokens, (size_t)tok_cap * sizeof(int));
        int *ns = realloc(token_slot, (size_t)tok_cap * sizeof(int));
        if (!nt || !ns) die("out of memory");
        tokens = nt;
        token_slot = ns;
      }
      int slot = vocab_to_slot[tid];
      if (slot < 0) {
        slot = n_gather;
        vocab_to_slot[tid] = slot;
        gather_ids[n_gather++] = (uint32_t)tid;
      }
      tokens[n_tokens] = tid;
      token_slot[n_tokens] = slot;
      n_tokens++;
    }
    word_tok_end[w] = n_tokens;
  }
  free(vocab_to_slot);

  int blank_slot = n_gather;
  gather_ids[n_gather++] = NEMO_CTC_BLANK_ID;

  int total_frames = asr_tokens * NEMO_CTC_UP_FACTOR;
  int n_states = 2 * n_tokens + 1;
  int min_frames = viterbi_min_frames(tokens, n_tokens);
  if (n_tokens == 0 || total_frames < min_frames) {
    assign_proportional_times(words, n_words, clip_dur);
    free(gather_ids);
    free(token_slot);
    free(tokens);
    free(word_tok_end);
    free(word_tok_start);
    *out_words = words;
    *out_count = n_words;
    return 1;
  }

  int64_t full_cells = (int64_t)total_frames * (int64_t)n_states;
  int band_half = 0;
  if (full_cells > CTC_FULL_VITERBI_CELLS) {
    band_half = (int)ceil(CTC_BAND_RATIO * (double)n_states);
    if (band_half < CTC_BAND_MIN_STATES) band_half = CTC_BAND_MIN_STATES;
  }

  int *tok_first = xmalloc((size_t)n_tokens * sizeof(int));
  int *tok_last = xmalloc((size_t)n_tokens * sizeof(int));
  int aligned = 0;

  // Run banded Viterbi first; if a long recording's leading/trailing silence
  // forces the path outside the diagonal band, retry once with the full band.
  for (int attempt = 0; attempt < 2; attempt++) {
    int cur_band = (attempt == 0) ? band_half : 0;
    CtcViterbi vit;
    if (!viterbi_init(&vit, total_frames, tokens, token_slot, n_tokens,
                      blank_slot, cur_band)) {
      break;
    }
    int ok = run_ctc_stream(ctc, asr_bf16, asr_tokens, gather_ids, n_gather,
                            wait, on_viterbi_chunk, &vit, error, error_cap);
    if (!ok) {
      viterbi_free(&vit);
      free(tok_last);
      free(tok_first);
      free(gather_ids);
      free(token_slot);
      free(tokens);
      free(word_tok_end);
      free(word_tok_start);
      free(words);
      return 0;
    }
    aligned = viterbi_finish(&vit, tok_first, tok_last);
    viterbi_free(&vit);
    if (aligned || cur_band == 0) break;
  }

  if (!aligned) {
    assign_proportional_times(words, n_words, clip_dur);
  } else {
    int *has_time = calloc((size_t)n_words, sizeof(int));
    if (!has_time) die("out of memory");
    for (int w = 0; w < n_words; w++) {
      int ts = word_tok_start[w], te = word_tok_end[w];
      if (ts < te) {
        // Prefer non-punctuation tokens for acoustic span endpoints so silent
        // trailing punctuation does not stretch a word into trailing silence.
        int first_i = ts, last_i = te - 1;
        while (first_i < te &&
               is_punct_piece(ctc->spm->pieces[tokens[first_i]].text,
                              ctc->spm->pieces[tokens[first_i]].len)) {
          first_i++;
        }
        while (last_i >= ts &&
               is_punct_piece(ctc->spm->pieces[tokens[last_i]].text,
                              ctc->spm->pieces[tokens[last_i]].len)) {
          last_i--;
        }
        if (first_i > last_i) {
          first_i = ts;
          last_i = te - 1;
        }
        int f0 = tok_first[first_i];
        int f1 = tok_last[last_i];
        double s_sec = (double)f0 * NEMO_CTC_FRAME_SECONDS;
        double e_sec =
            (double)(f1 + 1) * NEMO_CTC_FRAME_SECONDS + CTC_END_OFFSET_SECONDS;
        if (s_sec < 0.0) s_sec = 0.0;
        if (e_sec > clip_dur) e_sec = clip_dur;
        if (e_sec < s_sec) e_sec = s_sec;
        words[w].start = s_sec;
        words[w].end = e_sec;
        has_time[w] = 1;
      }
    }
    fill_unaligned_words(words, has_time, n_words, clip_dur);
    free(has_time);

    // Clamp each word's end to the next word's anchored start before enforcing
    // monotonic bounds and minimum word duration within [0, clip_dur].
    for (int w = 0; w < n_words; w++) {
      if (w + 1 < n_words && words[w].end > words[w + 1].start)
        words[w].end = words[w + 1].start;
      if (w > 0 && words[w].start < words[w - 1].end)
        words[w].start = words[w - 1].end;
      if (words[w].start > clip_dur) words[w].start = clip_dur;
      if (words[w].end < words[w].start + CTC_MIN_WORD_SECONDS) {
        double target = words[w].start + CTC_MIN_WORD_SECONDS;
        if (w + 1 < n_words && target > words[w + 1].start &&
            words[w + 1].start >= words[w].start) {
          target = words[w + 1].start;
        }
        words[w].end = target < clip_dur ? target : clip_dur;
      }
      if (words[w].end > clip_dur) words[w].end = clip_dur;
    }
  }

  free(tok_last);
  free(tok_first);
  free(gather_ids);
  free(token_slot);
  free(tokens);
  free(word_tok_end);
  free(word_tok_start);

  *out_words = words;
  *out_count = n_words;
  return 1;
}

// Collector state for greedy CTC decoding across streamed chunks.
typedef struct {
  // Collapsed non-blank token IDs.
  int *ids;
  // Number of collected IDs and allocated capacity.
  int len, cap;
  // Previous frame's argmax class ID for CTC repeat collapsing.
  int prev_id;
} CtcGreedyCollector;

// Streaming callback collapsing consecutive duplicate and blank argmax frames.
static void on_greedy_chunk(void *ctx, const float *logp,
                            const uint32_t *argmax, int rows, int n_gather) {
  (void)logp;
  (void)n_gather;
  CtcGreedyCollector *c = ctx;
  for (int r = 0; r < rows; r++) {
    int id = (int)argmax[r];
    if (id != c->prev_id && id != NEMO_CTC_BLANK_ID && id >= 0 &&
        id < NEMO_CTC_VOCAB) {
      if (c->len >= c->cap) {
        c->cap = c->cap > 0 ? c->cap * 2 : 128;
        int *ni = realloc(c->ids, (size_t)c->cap * sizeof(int));
        if (!ni) die("out of memory");
        c->ids = ni;
      }
      c->ids[c->len++] = id;
    }
    c->prev_id = id;
  }
}

int nemo_ctc_greedy_decode(NemoCtcAligner *ctc, const uint16_t *asr_bf16,
                           int asr_tokens, const PassWait *wait,
                           char **out_text, char *error, size_t error_cap) {
  if (out_text) *out_text = NULL;
  if (!ctc || !asr_bf16 || asr_tokens <= 0) {
    set_ctc_err(error, error_cap, "invalid CTC decode input");
    return 0;
  }
  uint32_t blank = NEMO_CTC_BLANK_ID;
  CtcGreedyCollector col = {.prev_id = -1};
  int ok = run_ctc_stream(ctc, asr_bf16, asr_tokens, &blank, 1, wait,
                          on_greedy_chunk, &col, error, error_cap);
  if (!ok) {
    free(col.ids);
    return 0;
  }
  *out_text = nemo_spm_decode(ctc->spm, col.ids, col.len);
  free(col.ids);
  return 1;
}
