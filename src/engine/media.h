#ifndef EREMITE_ENGINE_MEDIA_H
#define EREMITE_ENGINE_MEDIA_H
#include "engine/memory.h"
#include <stdint.h>

// A media item in a prompt (image patches or mono audio waveform), and its
// feature rows once the model has encoded it. Its span in the prompt holds
// negative tokens, its keys. The keys come from the file's bytes, so that
// prefix reuse tells different media apart.
typedef struct {
  // The positions of its span.
  int ntokens;
  // Its grid of patches, and of tokens.
  int vh, vw, th, tw;
  // The patches' pixels in BF16.
  uint16_t *patch;
  // Decoded mono 16 kHz PCM samples for audio media, or null for images.
  float *pcm;
  // Number of samples in pcm.
  int pcm_samples;
  // The hash of the encoded file's bytes.
  uint64_t hash;
  // The position of its span in the prompt.
  int start;
  // BF16 feature rows, after encoding.
  uint16_t *rows;
  // The slot that counts the feature rows.
  MemorySlot *memory;
} Media;

// Returns the length of the image's span: its positions in the prompt.
int media_span_length(const Media *im);

// Writes the span's keys to keys.
void media_write_keys(const Media *im, int *keys);

// Copies the n tokens of in to out, and replaces the i-th media_token with the
// span of im[i], whose start it sets. out holds at most max tokens. Returns
// the count of tokens with the spans, or -1 if the tokens hold another number
// of placeholders than nim.
int media_expand_prompt(const int *in, int n, Media *im, int nim, int *out,
                        int max, int media_token);

// Frees the image's patches and feature rows, and clears it.
void media_free(Media *im);

#endif
