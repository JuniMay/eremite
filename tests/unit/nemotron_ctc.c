#include "check.h"
#include "model/nemotron/ctc.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Unit tests for the Nemotron CTC timestamp module's pure-CPU components:
// SentencePiece ModelProto parsing, score-ordered BPE encoding and decoding,
// UTF-8/CJK word splitting, and CTC Viterbi forced alignment.

// Dynamic byte buffer for constructing synthetic protobuf test payloads.
typedef struct {
  // Heap-allocated byte array.
  uint8_t *data;
  // Current byte length and allocated capacity.
  size_t len, cap;
} PbBuf;

// Appends len raw bytes to pb.
static void pb_push(PbBuf *pb, const void *src, size_t len) {
  if (pb->len + len > pb->cap) {
    size_t ncap = pb->cap > 0 ? pb->cap * 2 : 128;
    while (ncap < pb->len + len) ncap *= 2;
    uint8_t *nd = realloc(pb->data, ncap);
    if (!nd) abort();
    pb->data = nd;
    pb->cap = ncap;
  }
  memcpy(pb->data + pb->len, src, len);
  pb->len += len;
}

// Appends a protobuf unsigned varint to pb.
static void pb_varint(PbBuf *pb, uint64_t v) {
  uint8_t b;
  do {
    b = (uint8_t)(v & 0x7Fu);
    v >>= 7;
    if (v) b |= 0x80u;
    pb_push(pb, &b, 1);
  } while (v);
}

// Appends a protobuf length-delimited field to pb.
static void pb_bytes(PbBuf *pb, uint32_t field, const void *data, size_t len) {
  pb_varint(pb, ((uint64_t)field << 3) | 2u);
  pb_varint(pb, len);
  pb_push(pb, data, len);
}

// Appends one SentencePiece submessage (field 1 of ModelProto) to pb.
static void pb_add_piece(PbBuf *pb, const char *str, float score, int type) {
  PbBuf sub = {0};
  pb_bytes(&sub, 1, str, strlen(str));
  pb_varint(&sub, (2u << 3) | 5u);
  pb_push(&sub, &score, sizeof(float));
  if (type != 1) {
    pb_varint(&sub, (3u << 3) | 0u);
    pb_varint(&sub, (uint64_t)type);
  }
  pb_bytes(pb, 1, sub.data, sub.len);
  free(sub.data);
}

// Verifies SentencePiece ModelProto parsing, BPE score-ordered merging,
// user-defined tokens, unknown fallback and decoding.
static void test_spm_bpe(void) {
  PbBuf pb = {0};
  // ID 0: <unk> (type=2)
  pb_add_piece(&pb, "<unk>", 0.0f, 2);
  // ID 1: <spk:0> (type=4 user-defined)
  pb_add_piece(&pb, "<spk:0>", 0.0f, 4);
  // ID 2..7: single characters
  pb_add_piece(&pb, "\xE2\x96\x81", -100.0f, 1);
  pb_add_piece(&pb, "h", -101.0f, 1);
  pb_add_piece(&pb, "e", -102.0f, 1);
  pb_add_piece(&pb, "l", -103.0f, 1);
  pb_add_piece(&pb, "o", -104.0f, 1);
  pb_add_piece(&pb, "!", -105.0f, 1);
  // ID 8..12: BPE merges (higher score merges earlier)
  pb_add_piece(&pb, "he", -1.0f, 1);  // id 8
  pb_add_piece(&pb,
               "\xE2\x96\x81"
               "he",
               -2.0f, 1);  // id 9
  pb_add_piece(&pb, "ll", -3.0f, 1);  // id 10
  pb_add_piece(&pb, "llo", -4.0f, 1);  // id 11
  pb_add_piece(&pb,
               "\xE2\x96\x81"
               "hello",
               -5.0f, 1);  // id 12

  NemoSpmTokenizer *tk = nemo_spm_open_bytes(pb.data, pb.len);
  check(tk != NULL, "parsed synthetic SentencePiece ModelProto");
  check(nemo_spm_vocab_size(tk) == 13, "vocab size matches 13 pieces");

  int ids[16];
  int n = nemo_spm_encode_word(tk, "hello!", 6, 1, ids, 16);
  // Expected merges: "he" (-1) -> "▁he" (-2) -> "ll" (-3) -> "llo" (-4) ->
  // "▁hello" (-5), leaving [12 ("▁hello"), 7 ("!")].
  check(n == 2 && ids[0] == 12 && ids[1] == 7,
        "BPE merges follow descending score priority");

  // Without prefix: "he" (-1), "ll" (-3), "llo" (-4) -> [8 ("he"), 11 ("llo")].
  n = nemo_spm_encode_word(tk, "hello", 5, 0, ids, 16);
  check(n == 2 && ids[0] == 8 && ids[1] == 11,
        "BPE without word-boundary prefix merges subwords cleanly");

  // User-defined token bypasses prefix and BPE.
  n = nemo_spm_encode_word(tk, "<spk:0>", 7, 1, ids, 16);
  check(n == 1 && ids[0] == 1, "user-defined special token matches verbatim");

  // Unmapped character falls back to <unk> (0).
  n = nemo_spm_encode_word(tk, "z", 1, 0, ids, 16);
  check(n == 1 && ids[0] == 0, "unmapped character maps to unk");

  // Decoding converts SentencePiece word-boundary markers to spaces.
  int dec_ids[3] = {12, 7, 12};
  char *decoded = nemo_spm_decode(tk, dec_ids, 3);
  check(strcmp(decoded, "hello! hello") == 0,
        "nemo_spm_decode restores spaces and strips leading marker");
  free(decoded);

  // Truncated/invalid protobuf returns NULL safely.
  uint8_t bad[3] = {0x0A, 0x7F, 0x00};
  check(nemo_spm_open_bytes(bad, sizeof bad) == NULL,
        "malformed ModelProto is rejected cleanly");

  nemo_spm_close(tk);
  free(pb.data);
}

// Verifies UTF-8 English, punctuation, whitespace and CJK word splitting.
static void test_word_splitting(void) {
  const char *en = "  The quick,  don't\n3.5!\t";
  NemoCtcWord *words = NULL;
  int count = 0;
  nemo_ctc_split_words(en, strlen(en), &words, &count);
  check(count == 4, "English sentence splits into 4 words");
  if (count == 4) {
    check(words[0].byte_end - words[0].byte_start == 3 &&
              memcmp(en + words[0].byte_start, "The", 3) == 0,
          "word 0 is The");
    check(words[1].byte_end - words[1].byte_start == 6 &&
              memcmp(en + words[1].byte_start, "quick,", 6) == 0,
          "word 1 preserves trailing punctuation");
    check(words[2].byte_end - words[2].byte_start == 5 &&
              memcmp(en + words[2].byte_start, "don't", 5) == 0,
          "word 2 keeps contraction intact");
    check(words[3].byte_end - words[3].byte_start == 4 &&
              memcmp(en + words[3].byte_start, "3.5!", 4) == 0,
          "word 3 keeps decimal number intact");
  }
  free(words);

  // Mixed CJK + punctuation + Latin: "你好，world。" -> ["你", "好，",
  // "world。"].
  const char *mix = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C"
                    "world\xE3\x80\x82";
  nemo_ctc_split_words(mix, strlen(mix), &words, &count);
  check(count == 3, "mixed CJK and Latin splits into 3 spans");
  if (count == 3) {
    check(words[0].byte_end - words[0].byte_start == 3,
          "first CJK char is 3 bytes");
    check(words[1].byte_end - words[1].byte_start == 6,
          "second CJK char attaches trailing CJK comma");
    check(memcmp(mix + words[2].byte_start, "world\xE3\x80\x82", 8) == 0,
          "Latin word attaches trailing CJK period");
  }
  free(words);

  // Empty and whitespace-only strings return zero words.
  nemo_ctc_split_words("   \t\n ", 6, &words, &count);
  check(count == 0 && words == NULL, "whitespace-only text returns 0 words");
}

// Verifies CTC Viterbi forced alignment on synthetic emission matrices.
static void test_viterbi_align(void) {
  static constexpr int BLANK = 3;
  static constexpr int GATHER = 4;

  // Case 1: 3 distinct tokens [10, 20, 30] placed at frames 2..3, 6, 9..10 of
  // 12 frames (slots 0, 1, 2; blank is slot 3).
  static constexpr int F1 = 12;
  float lp1[F1 * GATHER];
  for (int t = 0; t < F1; t++) {
    for (int c = 0; c < GATHER; c++) lp1[t * GATHER + c] = -10.0f;
    lp1[t * GATHER + BLANK] = -0.1f;
  }
  lp1[2 * GATHER + 0] = -0.05f;
  lp1[3 * GATHER + 0] = -0.05f;
  lp1[6 * GATHER + 1] = -0.05f;
  lp1[9 * GATHER + 2] = -0.05f;
  lp1[10 * GATHER + 2] = -0.05f;
  int toks1[3] = {10, 20, 30};
  int slots1[3] = {0, 1, 2};
  int first1[3], last1[3];
  int ok = nemo_ctc_viterbi_align(lp1, F1, GATHER, toks1, slots1, 3, BLANK,
                                  first1, last1);
  check(ok == 1 && first1[0] == 2 && last1[0] == 3 && first1[1] == 6 &&
            last1[1] == 6 && first1[2] == 9 && last1[2] == 10,
        "Viterbi recovers exact multi-frame spans for distinct tokens");

  // Case 2: Repeated identical tokens [10, 10] MUST be separated by at least
  // one blank frame. Plant strong token 10 at frames 1..5 and a weak dip to
  // blank at frame 3.
  static constexpr int F2 = 8;
  float lp2[F2 * GATHER];
  for (int t = 0; t < F2; t++) {
    for (int c = 0; c < GATHER; c++) lp2[t * GATHER + c] = -10.0f;
    lp2[t * GATHER + BLANK] = -0.1f;
  }
  lp2[1 * GATHER + 0] = -0.05f;
  lp2[2 * GATHER + 0] = -0.05f;
  lp2[3 * GATHER + 0] = -2.0f;
  lp2[3 * GATHER + BLANK] = -0.2f;
  lp2[4 * GATHER + 0] = -0.05f;
  lp2[5 * GATHER + 0] = -0.05f;
  int toks2[2] = {10, 10};
  int slots2[2] = {0, 0};
  int first2[2], last2[2];
  ok = nemo_ctc_viterbi_align(lp2, F2, GATHER, toks2, slots2, 2, BLANK, first2,
                              last2);
  check(ok == 1 && first2[0] == 1 && last2[0] == 2 && first2[1] == 4 &&
            last2[1] == 5,
        "Viterbi separates repeated identical tokens through a blank frame");

  // Case 3: Leading silence (speech only at the last 3 frames of 40 frames).
  static constexpr int F3 = 40;
  float lp3[F3 * GATHER];
  for (int t = 0; t < F3; t++) {
    for (int c = 0; c < GATHER; c++) lp3[t * GATHER + c] = -10.0f;
    lp3[t * GATHER + BLANK] = -0.05f;
  }
  lp3[37 * GATHER + 0] = -0.01f;
  lp3[38 * GATHER + 1] = -0.01f;
  lp3[39 * GATHER + 2] = -0.01f;
  ok = nemo_ctc_viterbi_align(lp3, F3, GATHER, toks1, slots1, 3, BLANK, first1,
                              last1);
  check(ok == 1 && first1[0] == 37 && first1[1] == 38 && first1[2] == 39,
        "Viterbi aligns speech accurately after long leading silence");

  // Case 4: Infeasible frame count for repeated token [10, 10] in 2 frames
  // (needs at least 3 frames: token, blank, token).
  ok = nemo_ctc_viterbi_align(lp2, 2, GATHER, toks2, slots2, 2, BLANK, first2,
                              last2);
  check(ok == 0, "Viterbi returns 0 when frame count is infeasible");
}

int main(void) {
  test_spm_bpe();
  test_word_splitting();
  test_viterbi_align();
  return failures != 0;
}
