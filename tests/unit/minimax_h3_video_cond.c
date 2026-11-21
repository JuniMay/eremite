#include "base/error.h"
#include "engine/tokenizer.h"
#include "model/minimax_h3/text_encoder.h"
#include "model/minimax_h3/vision_encoder.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Unit tests for MiniMax-H3 video prompt tokenization and multimodal layout
// assembly (T2VA, FL2VA and REF2VA).

// Number of single-byte entries in the synthetic GPT-2 byte-level vocabulary.
static constexpr int BYTE_VOCAB_SIZE = 256;

// JSON text buffer for the temporary synthetic tokenizer.json.
static char tok_json[1 << 16];
static size_t tok_json_len;

// Total failed assertions in this test binary.
static int failures;

// Records one assertion outcome and prints label on failure.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

// Appends formatted text to tok_json.
__attribute__((format(printf, 1, 2))) static void append_json(const char *fmt,
                                                              ...) {
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(tok_json + tok_json_len, sizeof tok_json - tok_json_len,
                    fmt, args);
  va_end(args);
  if (n < 0 || (size_t)n >= sizeof tok_json - tok_json_len)
    die("synthetic tokenizer.json overflow");
  tok_json_len += (size_t)n;
}

// Appends one Unicode code point as a JSON-escaped UTF-8 sequence.
static void append_json_cp(int c) {
  if (c == '"' || c == '\\')
    append_json("\\%c", c);
  else if (c < 0x80)
    append_json("%c", c);
  else
    append_json("%c%c", 0xC0 | (c >> 6), 0x80 | (c & 63));
}

// Writes a minimal byte-level BPE tokenizer.json into dir where every byte b
// encodes to token ID b.
static void write_byte_tokenizer(const char *dir) {
  tok_json_len = 0;
  append_json("{\"added_tokens\":[],\"normalizer\":null,"
              "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{"
              "\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\s+|\\\\S+\"},"
              "\"behavior\":\"Isolated\",\"invert\":false},{\"type\":"
              "\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,"
              "\"use_regex\":false}]},\"model\":{\"type\":\"BPE\","
              "\"dropout\":null,\"unk_token\":null,"
              "\"continuing_subword_prefix\":null,"
              "\"end_of_word_suffix\":null,\"fuse_unk\":false,"
              "\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{");
  for (int b = 0, extra = 0; b < BYTE_VOCAB_SIZE; b++) {
    int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    append_json("\"");
    append_json_cp(keep ? b : BYTE_VOCAB_SIZE + extra++);
    append_json("\":%d%s", b, b + 1 < BYTE_VOCAB_SIZE ? "," : "");
  }
  append_json("},\"merges\":[]}}");
  char path[256];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  FILE *f = fopen(path, "w");
  if (!f || fwrite(tok_json, 1, tok_json_len, f) != tok_json_len)
    die("cannot write %s", path);
  fclose(f);
}

// Checks whether ids[offset .. offset + strlen(s)) equals the ASCII bytes of s.
static int matches_ascii(const int *ids, int len, int offset, const char *s) {
  size_t slen = strlen(s);
  if (offset < 0 || offset + (int)slen > len) return 0;
  for (size_t i = 0; i < slen; i++) {
    if (ids[offset + (int)i] != (unsigned char)s[i]) return 0;
  }
  return 1;
}

int main(void) {
  char dir[] = "/tmp/eremite-video-cond-XXXXXX";
  if (!mkdtemp(dir)) die("mkdtemp failed");
  write_byte_tokenizer(dir);
  Tokenizer *tok = tokenizer_open(dir, NULL);

  // 1. Extra video control tokens parse cleanly; unknown special-like tags stay
  // plain text.
  int ids[256];
  const char *prompt_special =
      "A<d>hi</d><|cutoff|><|lyrics_start|>la<|lyrics_end|>"
      "<|caption_start|>cap<|caption_end|>";
  int n = minimax_h3_tokenize_video_text(tok, prompt_special, 0, ids, 256);
  check(n == 15, "all seven extra video tokens collapse to single IDs");
  check(ids[0] == 'A' && ids[1] == MINIMAX_H3_TEXT_DIALOGUE_START_TOKEN &&
            ids[2] == 'h' && ids[3] == 'i' &&
            ids[4] == MINIMAX_H3_TEXT_DIALOGUE_END_TOKEN &&
            ids[5] == MINIMAX_H3_TEXT_CUTOFF_TOKEN &&
            ids[6] == MINIMAX_H3_TEXT_LYRICS_START_TOKEN && ids[7] == 'l' &&
            ids[8] == 'a' && ids[9] == MINIMAX_H3_TEXT_LYRICS_END_TOKEN &&
            ids[10] == MINIMAX_H3_TEXT_CAPTION_START_TOKEN && ids[11] == 'c' &&
            ids[12] == 'a' && ids[13] == 'p' &&
            ids[14] == MINIMAX_H3_TEXT_CAPTION_END_TOKEN,
        "exact token IDs for all seven MiniMax-H3 video control tokens");

  // Empty prompt falls back to pad token when allow_empty == 0 and stays empty
  // when allow_empty == 1.
  n = minimax_h3_tokenize_video_text(tok, "", 0, ids, 256);
  check(n == 1 && ids[0] == MINIMAX_H3_TEXT_PAD_TOKEN,
        "empty video prompt falls back to pad token");
  n = minimax_h3_tokenize_video_text(tok, "", 1, ids, 256);
  check(n == 0, "empty video segment stays empty when allow_empty is set");

  // 2. Smart resize helper rounds to 32 px and enforces [min_pixels, max].
  int sw = 0, sh = 0;
  minimax_h3_vision_smart_size(1344, 768, &sw, &sh);
  check(sw == 1344 && sh == 768,
        "smart_size preserves valid 1344x768 multiple of 32");
  minimax_h3_vision_smart_size(64, 64, &sw, &sh);
  check(sw == 64 && sh == 64, "smart_size preserves 64x64 above min_pixels");
  minimax_h3_vision_smart_size(32, 32, &sw, &sh);
  check(sw == 64 && sh == 64,
        "smart_size scales 32x32 up to meet min_pixels 3136");

  // 3. T2VA layout (no conditioning items).
  MinimaxH3VideoPromptLayout layout;
  minimax_h3_video_prompt_build_layout(tok, "rain <d>go</d>", NULL, 0, &layout);
  check(layout.vision_blocks == 0 && layout.visual_tokens == 0 &&
            layout.seq_len == 9,
        "T2VA layout builds text-only token sequence with extra tokens");
  check(layout.ids[5] == MINIMAX_H3_TEXT_DIALOGUE_START_TOKEN &&
            layout.ids[8] == MINIMAX_H3_TEXT_DIALOGUE_END_TOKEN,
        "T2VA preserves dialogue start and end token IDs");
  int all_text_tags = 1, mono_pos = 1;
  for (int i = 0; i < layout.seq_len; i++) {
    if (layout.tags[i] != 1) all_text_tags = 0;
    if (layout.pos_ids[3 * i] != i || layout.pos_ids[3 * i + 1] != i ||
        layout.pos_ids[3 * i + 2] != i)
      mono_pos = 0;
  }
  check(all_text_tags && mono_pos,
        "T2VA assigns TAG_TEXT and monotonic (i, i, i) MRoPE coordinates");
  check(layout.peak_bytes ==
            minimax_h3_text_encoder_workspace_bytes(layout.seq_len, 0) +
                (uint64_t)layout.seq_len * (uint64_t)MINIMAX_H3_TEXT_DIM *
                    sizeof(float),
        "T2VA peak_bytes matches text-only workspace plus output hidden");
  minimax_h3_video_prompt_layout_free(&layout);

  // 4. FL2VA layout with first and last frames (64x96 -> 2x3 = 6 merged tokens
  // per image).
  MinimaxH3VideoCondItem fl2va[2] = {
      {.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
       .width = 96,
       .height = 64,
       .frame_count = 1},
      {.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
       .width = 96,
       .height = 64,
       .frame_count = 1},
  };
  minimax_h3_video_prompt_build_layout(tok, "go", fl2va, 2, &layout);
  const char *p1 = "<Picture 1>: ";
  const char *p2 = "<Picture 2>: ";
  int p1_len = (int)strlen(p1), p2_len = (int)strlen(p2);
  int expected_len = p1_len + 8 + p2_len + 8 + 2;
  check(layout.vision_blocks == 2 && layout.visual_tokens == 12 &&
            layout.seq_len == expected_len,
        "FL2VA counts two 6-token vision blocks, labels, delimiters and text");
  check(matches_ascii(layout.ids, layout.seq_len, 0, p1),
        "FL2VA emits angle-bracket <Picture 1>: label");
  check(layout.ids[p1_len] == MINIMAX_H3_TEXT_VISION_START_TOKEN &&
            layout.ids[p1_len + 1] == -1 && layout.ids[p1_len + 6] == -1 &&
            layout.ids[p1_len + 7] == MINIMAX_H3_TEXT_VISION_END_TOKEN,
        "FL2VA flanks block 0 merged slots with VISION_START and VISION_END");
  check(layout.tags[p1_len - 1] == 1 && layout.tags[p1_len] == 0 &&
            layout.tags[p1_len + 7] == 0 && layout.tags[p1_len + 8] == 1,
        "FL2VA tags vision block plus start/end as visual (0) and labels as "
        "text (1)");
  check(matches_ascii(layout.ids, layout.seq_len, p1_len + 8, p2),
        "FL2VA emits <Picture 2>: label before second frame");

  // Verify MRoPE continuity across both 2x3 vision blocks (mh=2, mw=3,
  // len_max=3).
  int b0_start = layout.block_starts[0];
  int b0_base = b0_start;
  check(layout.pos_ids[3 * b0_start] == b0_base &&
            layout.pos_ids[3 * b0_start + 1] == b0_base &&
            layout.pos_ids[3 * b0_start + 2] == b0_base &&
            layout.pos_ids[3 * (b0_start + 5)] == b0_base &&
            layout.pos_ids[3 * (b0_start + 5) + 1] == b0_base + 1 &&
            layout.pos_ids[3 * (b0_start + 5) + 2] == b0_base + 2,
        "FL2VA block 0 MRoPE spans (base, base + row, base + col)");
  int b0_end = b0_start + 6;
  check(layout.pos_ids[3 * b0_end] == b0_base + 3 &&
            layout.pos_ids[3 * b0_end + 1] == b0_base + 3 &&
            layout.pos_ids[3 * b0_end + 2] == b0_base + 3,
        "FL2VA token after block 0 advances by max(mh, mw)");
  int b1_start = layout.block_starts[1];
  int b1_base = layout.pos_ids[3 * (b1_start - 1)] + 1;
  check(layout.pos_ids[3 * b1_start] == b1_base &&
            layout.pos_ids[3 * (b1_start + 5) + 1] == b1_base + 1 &&
            layout.pos_ids[3 * (b1_start + 5) + 2] == b1_base + 2,
        "FL2VA block 1 MRoPE continues seamlessly from VISION_START");
  minimax_h3_video_prompt_layout_free(&layout);

  // 5. REF2VA layout: image + audio + 3-frame odd-length video + second audio.
  MinimaxH3VideoCondItem ref2va[4] = {
      {.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
       .width = 64,
       .height = 64,
       .frame_count = 1},
      {.kind = MINIMAX_H3_VIDEO_COND_AUDIO},
      {.kind = MINIMAX_H3_VIDEO_COND_VIDEO,
       .width = 64,
       .height = 64,
       .frame_count = 3},
      {.kind = MINIMAX_H3_VIDEO_COND_AUDIO},
  };
  minimax_h3_video_prompt_build_layout(tok, "", ref2va, 4, &layout);
  // 1 image block (4 tokens) + 2 video pair blocks (4 tokens each) = 3 blocks,
  // 12 visual tokens. Empty prompt beside items adds 0 tokens.
  check(layout.vision_blocks == 3 && layout.visual_tokens == 12,
        "REF2VA odd 3-frame video pads to 2 temporal pairs plus 1 image");
  int cur = 0;
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<Picture 1>: "),
        "REF2VA emits <Picture 1>: first");
  cur += (int)strlen("<Picture 1>: ") + 6;
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<Audio 1>: "),
        "REF2VA emits <Audio 1>: without vision tokens");
  cur += (int)strlen("<Audio 1>: ");
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<Video 1>: "),
        "REF2VA emits <Video 1>: before video pair timestamps");
  cur += (int)strlen("<Video 1>: ");
  // Default 2 fps timestamps: pair 0 is (0.0 + 0.5)/2 = 0.25 -> "<0.2
  // seconds>", pair 1 repeat-pads frame 2 -> (1.0 + 1.0)/2 = 1.0 -> "<1.0
  // seconds>".
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<0.2 seconds>"),
        "REF2VA formats first 2-frame pair midpoint as <0.2 seconds>");
  cur += (int)strlen("<0.2 seconds>") + 6;
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<1.0 seconds>"),
        "REF2VA repeat-pads odd final frame timestamp to <1.0 seconds>");
  cur += (int)strlen("<1.0 seconds>") + 6;
  check(matches_ascii(layout.ids, layout.seq_len, cur, "<Audio 2>: "),
        "REF2VA increments independent audio ordinal to <Audio 2>: ");
  cur += (int)strlen("<Audio 2>: ");
  check(cur == layout.seq_len,
        "REF2VA empty prompt alongside items emits no trailing pad token");
  check(layout.peak_bytes > 0, "REF2VA computes non-zero peak GPU bytes");
  minimax_h3_video_prompt_layout_free(&layout);

  // 6. Audio-only REF2VA uses the text-only peak-memory path (0 vision blocks).
  MinimaxH3VideoCondItem audio_only[1] = {
      {.kind = MINIMAX_H3_VIDEO_COND_AUDIO},
  };
  minimax_h3_video_prompt_build_layout(tok, "music", audio_only, 1, &layout);
  check(layout.vision_blocks == 0 && layout.visual_tokens == 0,
        "audio-only REF2VA requires zero vision blocks");
  check(matches_ascii(layout.ids, layout.seq_len, 0, "<Audio 1>: music"),
        "audio-only REF2VA concatenates <Audio 1>: and prompt");
  minimax_h3_video_prompt_layout_free(&layout);

  tokenizer_close(tok);
  char path[256];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  unlink(path);
  rmdir(dir);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("minimax_h3_video_cond: all checks passed\n");
  return 0;
}
