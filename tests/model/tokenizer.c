#include "check.h"
#include "base/error.h"
#include "engine/tokenizer.h"
#include <stdlib.h>
#include <string.h>

// Tests a checkpoint's tokenizer against the reference tokenizer's ids for
// DeepSeek V4.1 Flash. Each case is a text and the ids that the reference
// tokenizer gives it. In a text, an added token's spelling is that token. A
// case may also ask that the ids' bytes spell the text again, or that the text
// is one added token.
//
// usage: model-tokenizer CHECKPOINT

// One case: a text, made of a piece repeated, and the reference ids for it.
typedef struct {
  // The piece of text. It may hold NUL bytes.
  const char *piece;
  // The piece's length in bytes.
  int piece_length;
  // How many times the piece repeats in the text.
  int repeat;
  // The reference ids: ids, repeated id_repeat times.
  const int *ids;
  // How many ids the ids array holds.
  int id_count;
  // How many times the ids repeat.
  int id_repeat;
  // Whether the ids' bytes must spell the text again.
  bool spells_text;
  // Whether the text must be one added token.
  bool is_added;
} Case;

// A text that is the string literal s once.
#define TEXT(s) .piece = (s), .piece_length = sizeof(s) - 1, .repeat = 1
// A text that is the string literal s, n times.
#define REPEAT(s, n) .piece = (s), .piece_length = sizeof(s) - 1, .repeat = (n)
// Reference ids that are the given ids once.
#define IDS(...) IDS_REPEAT(1, __VA_ARGS__)
// Reference ids that are the given ids, n times.
#define IDS_REPEAT(n, ...)                                                     \
  .ids = (const int[]){__VA_ARGS__},                                           \
  .id_count = sizeof((const int[]){__VA_ARGS__}) / sizeof(int),                \
  .id_repeat = (n)

static const Case cases[] = {
    {TEXT("")},
    {TEXT("Hello, world!"), IDS(19923, 14, 2058, 3)},
    {TEXT("I'm I'M she'll we've it'd isn't"),
     IDS(43, 4571, 342, 125845, 1225, 5922, 579, 5270, 436, 7485, 7367, 1664)},
    {TEXT("print(\"hello\")\n    return 42\n"),
     IDS(3098, 1698, 33310, 5225, 361, 1354, 223, 3180, 201)},
    {TEXT("1234567890 1.25 -4e+10"),
     IDS(6895, 18009, 25744, 18, 223, 19, 16, 1069, 565, 22, 71, 13, 553)},
    {TEXT("中文测试 日本語 한국어 العربية ह\u093fन\u094dद\u0940"),
     IDS(21134, 10251, 223, 88768, 51345, 9227, 55595, 17278, 112789, 8714)},
    {TEXT("Café e\u0301 Å A\u030a \u212b 한글"),
     IDS(37, 2797, 619, 312, 17793, 28000, 334, 139, 235, 14508, 107, 17382,
         91669, 225, 160, 230, 114, 77738, 110)},
    {TEXT("a\u0323\u0301 \u0301mark combining\u034f"),
     IDS(67, 100819, 17793, 223, 17793, 7747, 26179, 140, 240)},
    {TEXT("🙂🚀 👩🏽\u200d💻 🇺🇸"),
     IDS(80300, 227, 74287, 225, 52780, 105, 40965, 124, 46088, 23903, 122,
         77579, 121, 43754, 119)},
    {TEXT("\r\n\t  \n\n\r end\t"), IDS(204, 201, 200, 6776, 204, 1522, 200)},
    {TEXT(" \t x  y\n  "), IDS(49293, 1527, 223, 383, 201, 262)},
    {TEXT("\000\001\177\302\205\u00a0\u2028\u2029"),
     IDS(191, 192, 224, 129, 230, 2162, 300, 104, 300, 105)},
    {TEXT("<|im_start|>user\nhi<|im_end|>\n"),
     IDS(30, 94, 328, 37864, 94, 32, 5265, 201, 6366, 30, 94, 328, 42616, 94,
         1018)},
    {TEXT("<think>reason</think><tool_call>{}</tool_call>"),
     IDS(128821, 86512, 128822, 30, 72461, 112042, 31923, 24568, 72461, 112042,
         32)},
    {TEXT("<|im_start|><|im_start|>"),
     IDS(30, 94, 328, 37864, 94, 5451, 94, 328, 37864, 94, 32)},
    {TEXT("  <|im_end|>  "), IDS(223, 818, 94, 328, 42616, 94, 32, 262)},
    {TEXT("a<|endoftext|>b"), IDS(67, 30, 94, 523, 2154, 2067, 94, 32, 68)},
    {TEXT("<tool_response>done</tool_response>"),
     IDS(30, 72461, 64469, 32, 44629, 1718, 72461, 64469, 32)},
    {TEXT("∀x∈ℝ: x² ≥ 0. ±∞ ≠ NaN"),
     IDS(78728, 90, 20954, 119659, 28, 1527, 1628, 24985, 223, 18, 16, 10663,
         24219, 63014, 81409)},
    {TEXT("𐐀𐐨 𓂀 𠀀 ꙮ ᚠ"),
     IDS(120577, 241, 225, 120577, 241, 104, 86387, 244, 227, 225, 86387, 257,
         225, 225, 2297, 250, 109, 8751, 251, 257)},
    {REPEAT("a", 512), IDS_REPEAT(64, 89086)},
    {REPEAT(" ", 512), IDS_REPEAT(4, 69933)},
    {REPEAT("🤖", 100), IDS_REPEAT(100, 62377, 247)},
    {REPEAT("\n", 256), IDS_REPEAT(16, 93614)},
    {TEXT("\r\000? a2\u0323c-1\r"),
     IDS(204, 191, 33, 260, 20, 100819, 69, 15, 19, 204)},
    {TEXT("2\u00a0\ra//"), IDS(20, 2162, 204, 67, 835)},
    {TEXT("\u0301\000中\n\302\205"), IDS(17793, 191, 525, 201, 129, 230)},
    {TEXT("3\u2028"), IDS(21, 300, 104)},
    {TEXT("Xé\r\té"), IDS(58, 619, 204, 200, 619)},
    {TEXT("₹\n"), IDS(94296, 201)},
    {TEXT("a\u0345b \u0345 1\u0345!"),
     IDS(67, 140, 230, 68, 223, 140, 230, 223, 19, 140, 230, 3)},
    {TEXT("a\u05b0b \u05b0 1\u05b0!"),
     IDS(67, 18906, 68, 223, 18906, 223, 19, 18906, 3)},
    {TEXT("a١b ١ 1١!"), IDS(67, 63468, 68, 223, 63468, 223, 19, 63468, 3)},
    {TEXT("a²b ² 1²!"), IDS(67, 1628, 68, 223, 1628, 223, 19, 1628, 3)},
    {TEXT("aⅠb Ⅰ 1Ⅰ!"), IDS(67, 61151, 68, 223, 61151, 223, 19, 61151, 3)},
    {TEXT("a\u200cb \u200c 1\u200c!"),
     IDS(67, 300, 237, 68, 223, 300, 237, 223, 19, 300, 237, 3)},
    {TEXT("a\u2060b \u2060 1\u2060!"),
     IDS(67, 15371, 257, 68, 223, 15371, 257, 223, 19, 15371, 257, 3)},
    {TEXT("a\u202fb \u202f 1\u202f!"),
     IDS(67, 300, 110, 68, 663, 110, 223, 19, 300, 110, 3)},
    {TEXT("a\u3000b \u3000 1\u3000!"),
     IDS(67, 18524, 68, 223, 18524, 223, 19, 18524, 3)},
    {TEXT("aᲉb Ᲊ 1Ᲊ!"),
     IDS(67, 160, 113, 234, 68, 8751, 113, 234, 223, 19, 160, 113, 234, 3)},
    {TEXT("a𞓐b 𞓐 1𞓐!"), IDS(67, 175, 255, 244, 241, 68, 86387, 255, 244, 241,
                            223, 19, 175, 255, 244, 241, 3)},
    {TEXT("<｜begin▁of▁sentence｜>"), IDS(0), .spells_text = true,
     .is_added = true},
    {TEXT("<｜end▁of▁sentence｜>"), IDS(1), .spells_text = true,
     .is_added = true},
    {TEXT("<｜▁pad▁｜>"), IDS(2), .spells_text = true, .is_added = true},
    {TEXT("｜DSML｜"), IDS(128825), .spells_text = true, .is_added = true},
    {TEXT(" Review the queue. 文件 12345.\n"),
     IDS(6820, 270, 18505, 16, 223, 5791, 223, 6895, 1883, 603),
     .spells_text = true},
    {REPEAT("中", 400), IDS_REPEAT(400, 525), .spells_text = true},
    {REPEAT("thequickbrownfoxjumpsoverthelazydog", 40),
     IDS_REPEAT(40, 1805, 69582, 68, 4775, 47971, 76, 3798, 821, 432, 72016,
                1521, 2628, 520),
     .spells_text = true},
};

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-tokenizer CHECKPOINT");
  Tokenizer *tk = tokenizer_open(argv[1], nullptr);
  int count = sizeof cases / sizeof *cases, failed = 0;
  for (int c = 0; c < count; c++) {
    const Case *k = &cases[c];
    int length = k->piece_length * k->repeat;
    char *text = xmalloc((size_t)length);
    for (int r = 0; r < k->repeat; r++)
      memcpy(text + (size_t)r * k->piece_length, k->piece, k->piece_length);
    // A text gives at most one id per byte.
    int *got = xmalloc((size_t)length * sizeof *got);
    int n = tokenizer_encode_bytes(tk, text, (size_t)length, TOKENIZE_ADDED,
                                   got, length),
        expected = k->id_count * k->id_repeat, same = n == expected;
    for (int i = 0; same && i < n; i++)
      same = got[i] == k->ids[i % k->id_count];
    if (k->spells_text) {
      int offset = 0;
      for (int i = 0; same && i < n; i++) {
        int bytes;
        const char *piece = tokenizer_token_bytes(tk, got[i], &bytes);
        same = offset + bytes <= length && !memcmp(text + offset, piece, bytes);
        offset += bytes;
      }
      same &= offset == length;
    }
    if (k->is_added)
      same &= n == 1 && tokenizer_token_kind(tk, got[0]) != TOKEN_REGULAR;
    if (!same) {
      failed++;
      printf("case %d (%d bytes): %d ids, the reference's %d\n", c, length, n,
             expected);
    }
    free(got);
    free(text);
  }
  printf("%d of %d cases match\n", count - failed, count);
  check(!failed, "every case's ids match the reference's");
  tokenizer_close(tk);
  return failures != 0;
}
