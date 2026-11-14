#ifndef EREMITE_ENGINE_STATE_FILE_H
#define EREMITE_ENGINE_STATE_FILE_H
#include "model/model.h"
#include <CommonCrypto/CommonDigest.h>

// Saved states: a state's rows in parts, and snapshots, in files. The caller
// names, opens, syncs and renames the files. The functions reach a model's
// rows through its family's state_regions, snapshot_parts and snapshot_alloc.

// Writes one part of a saved state, from a state that holds at least `to`
// positions. For each region, the part holds its rows from row
// region_stable(from) up to row region_rows(to).
// Returns the file's bytes, or -1 if a write failed or a second stop request
// ended it. With fd -1 it writes nothing and returns the bytes it would take.
// snapshot_file_write does the same with fd -1.
int64_t state_file_write(const ModelOps *ops, ModelState *s, int from, int to,
                         int fd);
// Fills s with the rows before `positions` from a saved state's parts, given
// in the order they were written. Each part's rows replace the earlier parts'
// from its own start. Returns 0, or -1 if a part is short, damaged or of
// another shape, or if the parts leave rows out. After -1, s is of no use.
int state_file_read(const ModelOps *ops, ModelState *s, int positions,
                    const int *fds, int count);
// Writes snapshot x without its logits, or for a null x, s's rolling rows as
// they are at pos. Returns the file's bytes, or -1.
int64_t snapshot_file_write(const ModelOps *ops, ModelSnapshot *x,
                            ModelState *s, int pos, int fd);
// Returns the snapshot that a file holds, or null if the file is short,
// damaged or of another shape.
ModelSnapshot *snapshot_file_read(const ModelOps *ops, int fd);

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
