#ifndef EREMITE_ENGINE_STATE_FILE_H
#define EREMITE_ENGINE_STATE_FILE_H
#include <CommonCrypto/CommonDigest.h>
#include <stddef.h>
#include <stdint.h>

// The identity of a checkpoint's weights for saved states, and CRC32C
// checksums.

// Adds to h what a checkpoint's weights are, for a saved state's identity: the
// config.json, .safetensors and .gguf files at any depth of dir, by name, size
// and first and last 64 KiB. Names that start with a dot are left out.
void identity_add_folder(CC_SHA256_CTX *h, const char *dir);
// Adds to h the same for a GGUF file, or for every file of its split set
// (NAME-00001-of-00005.gguf).
void identity_add_gguf_set(CC_SHA256_CTX *h, const char *path);

// Returns crc continued over the n bytes at data: CRC32C (Castagnoli), with
// the ARM crc32c instructions.
uint32_t crc32c(uint32_t crc, const void *data, size_t n);

#endif
