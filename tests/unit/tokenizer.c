#include "check.h"
#include "base/error.h"
#include "engine/tokenizer.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Tests the tokenizer on a tokenizer.json that the test writes: byte-level
// BPE whose merges join runs of spaces up to 256 spaces, so that the last
// merge's parts take 256 bytes each in the file's UTF-8.

// The most spaces that one token of the test's tokenizer holds.
static constexpr int MOST_SPACES = 256;

// The text of the tokenizer.json that the test writes, and its length.
static char text[1 << 20];
static size_t length;

// Appends formatted text to the file's text.
__attribute__((format(printf, 1, 2))) static void append(const char *format,
                                                         ...) {
  va_list args;
  va_start(args, format);
  int n = vsnprintf(text + length, sizeof text - length, format, args);
  va_end(args);
  if (n < 0 || (size_t)n >= sizeof text - length)
    die("the test's tokenizer.json does not fit");
  length += (size_t)n;
}

// Appends code point c as UTF-8, escaped for a JSON string.
static void append_json_char(int c) {
  if (c == '"' || c == '\\')
    append("\\%c", c);
  else if (c < 0x80)
    append("%c", c);
  else
    append("%c%c", 0xC0 | c >> 6, 0x80 | (c & 63));
}

// Appends the byte-level spelling of `count` spaces, U+0120 each.
static void append_spaces(int count) {
  for (int i = 0; i < count; i++) append_json_char(0x120);
}

// Writes dir/tokenizer.json: the 256 bytes as tokens 0 to 255 in GPT-2's
// byte-level alphabet, then runs of 2, 4, ... MOST_SPACES spaces, each the
// merge of two halves. Whitespace and other text split apart first.
static void write_tokenizer(const char *dir) {
  append("{\"added_tokens\":[],\"normalizer\":null,"
         "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{"
         "\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\s+|\\\\S+\"},"
         "\"behavior\":\"Isolated\",\"invert\":false},{\"type\":"
         "\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,"
         "\"use_regex\":false}]},\"model\":{\"type\":\"BPE\",\"dropout\":"
         "null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
         "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":"
         "false,\"ignore_merges\":false,\"vocab\":{");
  for (int b = 0, extra = 0; b < 256; b++) {
    int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    append("\"");
    append_json_char(keep ? b : 256 + extra++);
    append("\":%d,", b);
  }
  for (int spaces = 2, id = 256; spaces <= MOST_SPACES; spaces *= 2, id++) {
    append("\"");
    append_spaces(spaces);
    append("\":%d%s", id, spaces < MOST_SPACES ? "," : "");
  }
  append("},\"merges\":[");
  for (int half = 1; half < MOST_SPACES; half *= 2) {
    append("%s[\"", half > 1 ? "," : "");
    append_spaces(half);
    append("\",\"");
    append_spaces(half);
    append("\"]");
  }
  append("]}}");
  char path[256];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  FILE *f = fopen(path, "w");
  if (!f || fwrite(text, 1, length, f) != length) die("cannot write %s", path);
  fclose(f);
}

int main(void) {
  char dir[] = "/tmp/eremite-tokenizer-XXXXXX";
  if (!mkdtemp(dir)) return 1;
  write_tokenizer(dir);
  Tokenizer *t = tokenizer_open(dir, NULL);
  char spaces[MOST_SPACES + 1];
  memset(spaces, ' ', MOST_SPACES), spaces[MOST_SPACES] = 0;
  int ids[8];
  int n = tokenizer_encode_text(t, spaces, ids, 8);
  // The run of 256 spaces is the last token: 256 bytes, then 7 runs.
  check(n == 1 && ids[0] == 256 + 7,
        "256 spaces encode as the token that the last merge makes, whose "
        "parts take 256 bytes each in the file");
  n = tokenizer_encode_text(t, "a  b", ids, 8);
  check(n == 3 && ids[0] == 'a' && ids[1] == 256 && ids[2] == 'b',
        "two spaces between words encode as the first merge's token");
  int length;
  const char *bytes = tokenizer_token_bytes(t, 256 + 7, &length);
  check(length == MOST_SPACES && !memcmp(bytes, spaces, MOST_SPACES),
        "the longest token's bytes are its 256 spaces");
  tokenizer_close(t);
  char path[256];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  unlink(path), rmdir(dir);
  return failures != 0;
}
