#ifndef EREMITE_ENGINE_TOKENIZER_H
#define EREMITE_ENGINE_TOKENIZER_H
#include <stddef.h>

// A checkpoint's tokenizer from its tokenizer.json. It is byte-level BPE.
// Before BPE, the text goes through the file's normalizer, none or NFC, and its
// Split regexes. Each regex isolates its matches within the pieces of the one
// before. The regexes and NFC run in the system's ICU library. Any thread may
// use a tokenizer.

// A checkpoint's tokenizer.
typedef struct Tokenizer Tokenizer;

// The byte that marks a media part's place in a rendered prompt. No valid
// UTF-8 text holds it.
#define MEDIA_MARK "\xff"

// How text is encoded.
typedef enum {
  // Text alone. The added tokens' spellings are text too.
  TOKENIZE_TEXT,
  // Text in which the added tokens' spellings are those tokens.
  TOKENIZE_ADDED,
  // A rendered prompt: as TOKENIZE_ADDED, except that MEDIA_MARK is the image
  // token, and the image token's own spelling is text.
  TOKENIZE_PROMPT
} TokenizeMode;

// What kind of token an id is.
typedef enum {
  // A token of the vocabulary, or an unknown id.
  TOKEN_REGULAR,
  // An added token that is not special.
  TOKEN_ADDED,
  // A special added token, such as the start of a sequence.
  TOKEN_SPECIAL
} TokenKind;

// Opens dir/tokenizer.json. image, unless null, is the spelling of the model's
// image token. Ends the program when the file's pipeline has other steps.
Tokenizer *tokenizer_open(const char *dir, const char *image);

// Frees the tokenizer.
void tokenizer_close(Tokenizer *tokenizer);

// Returns the vocabulary's size: the highest id plus one.
int tokenizer_vocab_size(const Tokenizer *t);

// Encodes len bytes of text, NUL bytes included, into at most max ids in out.
// Returns their count. Ends the program when there are more.
int tokenizer_encode_bytes(Tokenizer *t, const char *text, size_t len,
                           TokenizeMode mode, int *out, int max);

// These encode a NUL-ended text as TOKENIZE_TEXT, TOKENIZE_ADDED and
// TOKENIZE_PROMPT.
int tokenizer_encode_text(Tokenizer *t, const char *text, int *out, int max);
int tokenizer_encode_with_added(Tokenizer *t, const char *text, int *out,
                                int max);
int tokenizer_encode_prompt(Tokenizer *t, const char *text, int *out, int max);

// Returns the bytes of the token id, and sets *len to their count. An unknown
// id gives an empty string.
const char *tokenizer_token_bytes(const Tokenizer *t, int id, int *len);

// Returns what kind of token id is.
TokenKind tokenizer_token_kind(const Tokenizer *t, int id);

// Returns the id of the added token that len bytes of text spell, or -1.
int tokenizer_find_added(const Tokenizer *t, const char *text, size_t len);

// Returns the id of the added token that special spells. Ends the program
// when there is none.
int tokenizer_require_added(const Tokenizer *t, const char *special);

#endif
