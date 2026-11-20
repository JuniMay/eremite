#ifndef EREMITE_ENGINE_SAFETENSORS_H
#define EREMITE_ENGINE_SAFETENSORS_H
#include <stdint.h>

// A checkpoint's tensors in safetensors files: one file, model.safetensors, or
// the shards that model.safetensors.index.json names. The files are opened
// without the page cache (F_NOCACHE).

// The most dimensions a tensor shape records.
static constexpr int TENSOR_MAX_DIMS = 5;

// A tensor: its file, its bytes there, its type and its shape.
typedef struct {
  // The file that holds the tensor.
  int fd;
  // Where the tensor's bytes start in the file, and how many there are.
  uint64_t off, size;
  // The number type: 'b' BF16, 'h' F16, 'f' F32, '8' E4M3, 'e' E8M0, 'i' I8,
  // 'u' U8, 'U' U32, 'l' I64, and '?' for any other type.
  int dtype;
  // The dimensions, the slowest first, and how many there are.
  int64_t shape[TENSOR_MAX_DIMS];
  // The number of dimensions recorded in shape.
  int ndim;
} Tensor;

// A checkpoint's tensors, found by name.
typedef struct TensorIndex TensorIndex;

// Opens the tensors of the checkpoint in dir. Ends the program when a file
// cannot be read.
TensorIndex *tensor_index_open(const char *dir);

// Opens the tensors of one safetensors file at path. Ends the program when the
// file cannot be read.
TensorIndex *tensor_index_open_file(const char *path);

// Returns the tensor that the format names. Ends the program when there is
// none.
Tensor tensor_index_get(const TensorIndex *index, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

// Returns whether the checkpoint has the tensor that the format names.
int tensor_index_has(const TensorIndex *index, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

// Returns the string for key from the header's __metadata__ object, or null if
// absent.
const char *tensor_index_metadata(const TensorIndex *index, const char *key);

// Closes the index and its files, once none of its tensors is used.
void tensor_index_close(TensorIndex *index);

#endif
