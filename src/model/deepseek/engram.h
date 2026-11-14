#ifndef EREMITE_MODEL_DEEPSEEK_ENGRAM_H
#define EREMITE_MODEL_DEEPSEEK_ENGRAM_H
#include <stdint.h>

// Engram's n-gram hashing: each token's normalized id, and the multipliers,
// primes and offsets of the hashes. Its token normalization is computed from
// the checkpoint's tokenizer. The supported Flash layout uses two layers,
// n-grams 2..4 and eight heads.

typedef struct {
  int64_t multipliers[8], primes[48], offsets[48];
  int32_t pad, count, *map;
} Engram;

// Computes the layout from the tokenizer of the checkpoint directory.
void engram_init(Engram *layout, const char *checkpoint);

void engram_free(Engram *layout);

#endif
