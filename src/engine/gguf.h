#ifndef EREMITE_ENGINE_GGUF_H
#define EREMITE_ENGINE_GGUF_H
#include <stdint.h>

// The tensors of GGUF files, llama.cpp's format. A set is one file, or the
// downloaded shards of a split set (NAME-00001-of-0000N.gguf). The files are
// opened without the page cache (F_NOCACHE).

// A tensor in a GGUF file.
typedef struct {
  // The file that holds the tensor, and the tensor's ggml type.
  int fd, type;
  // The tensor's offset in the file, and its dimensions, the fastest first.
  uint64_t off, ne[4];
} GgufTensor;

// The ggml types of the supported expert sets' tensors, with ggml's numbers.
// Each type packs 256 values into a block.
enum {
  // Q2_K: 84 bytes per block, 2.625 bits per value.
  GG_Q2_K = 10,
  // Q6_K: 210 bytes per block, 6.5625 bits per value.
  GG_Q6_K = 14,
  // IQ2_XXS: 66 bytes per block, 2.0625 bits per value.
  GG_IQ2_XXS = 16,
  // IQ2_XS: 74 bytes per block, 2.3125 bits per value.
  GG_IQ2_XS = 17,
  // IQ3_XXS: 98 bytes per block, 3.0625 bits per value.
  GG_IQ3_XXS = 18,
  // IQ1_S: 50 bytes per block, 1.5625 bits per value.
  GG_IQ1_S = 19,
  // IQ2_S: 82 bytes per block, 2.5625 bits per value.
  GG_IQ2_S = 22,
  // IQ1_M: 56 bytes per block, 1.75 bits per value.
  GG_IQ1_M = 29
};

// The tensors of the GGUF files added to it.
typedef struct GgufFiles GgufFiles;

// Returns an empty set of GGUF files.
GgufFiles *gguf_files_new(void);

// Adds the tensors of path, or of every downloaded shard of the split set
// that path names. Ends the program when a file is not GGUF version 3.
void gguf_files_add(GgufFiles *files, const char *path);

// Finds the added tensor called name, and writes it to *t. Returns 0 when there
// is none.
int gguf_files_find(const GgufFiles *files, const char *name, GgufTensor *t);

// Frees the set. Its files stay open, since the tensors found in them read
// them later.
void gguf_files_free(GgufFiles *files);

#endif
