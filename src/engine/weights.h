#ifndef EREMITE_ENGINE_WEIGHTS_H
#define EREMITE_ENGINE_WEIGHTS_H
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// Weight groups: weights in GPU buffers that load together. A group is counted
// in one memory slot. The memory may evict the group, and the model then loads
// it again. A weight is a checkpoint tensor as stored, a conversion of tensors,
// or the output of a function.

typedef struct WeightGroup WeightGroup;

// Writes a weight's buffer from the weight's parameters.
typedef void (*WeightFill)(Buf *buffer, const void *parameters);

// The operations of weight_group_add_vector, which may be combined: add a
// number, and then exponentiate.
static constexpr int WEIGHT_ADD = 1, WEIGHT_EXP = 2;

// Returns an empty group counted in the account, whose buffers are made on
// gpu.
WeightGroup *weight_group_new(Gpu *gpu, MemoryAccount *account,
                              const char *name);

// Returns an empty group of the backbone's account and GPU, evicted before the
// backbone.
WeightGroup *weight_group_new_dependent(WeightGroup *backbone,
                                        const char *name);

// Returns the group's memory slot.
MemorySlot *weight_group_slot(WeightGroup *group);

// Returns the bytes that loading the group takes: its buffers, and room for
// the largest tensors that a conversion reads first.
uint64_t weight_group_required_bytes(const WeightGroup *group);

// Returns the group's bytes while it is loaded, and 0 otherwise.
uint64_t weight_group_resident_bytes(const WeightGroup *group);

// Adds the tensor as stored. An aligned tensor is read straight into a buffer
// 32 KiB larger. The buffer's off says where the tensor starts.
void weight_group_add_tensor(WeightGroup *group, Buf *target, Tensor tensor,
                             int aligned);

// Decodes rows by cols scales in the compressed scale format. For each 16
// rows, the format holds 16 row bases and then the rows' offsets in nibbles.
// Byte exceptions follow. Returns 0 if the stream is malformed.
int weight_group_decode_compressed_scales(uint8_t *out, const uint8_t *fixed,
                                          size_t fixed_bytes,
                                          const uint32_t *exceptions,
                                          size_t exception_count, size_t rows,
                                          size_t cols);

// Adds rows by cols scales in the compressed scale format, from U8 codes and
// U32 exceptions.
void weight_group_add_compressed_scales(WeightGroup *group, Buf *target,
                                        Tensor fixed, Tensor exceptions,
                                        int rows, int cols);

// Adds a BF16 or FP32 vector as FP32 values. With WEIGHT_ADD, add is added to
// each value. With WEIGHT_EXP, each value is then exponentiated.
void weight_group_add_vector(WeightGroup *group, Buf *target, Tensor tensor,
                             float add, int operations);

// Adds the elementwise product of two vectors as FP32 values.
void weight_group_add_product(WeightGroup *group, Buf *target, Tensor first,
                              Tensor second);

// Adds a BF16 matrix with its rows padded to stride values.
void weight_group_add_padded(WeightGroup *group, Buf *target, Tensor tensor,
                             int stride);

// Adds a quantized matrix as BF16 rows padded to stride values, without any
// global scale. Kind 'n' is NVFP4 with E4M3 scales per 16 values, 'm' is E4M3
// with E8M0 scales per 32 values, and any other kind is E4M3 alone.
void weight_group_add_expanded(WeightGroup *group, Buf *target, Tensor weight,
                               Tensor scales, int kind, int cols, int stride);

// Adds a weight of the given bytes, which fill writes from a copy of the
// parameters.
void weight_group_add_generated(WeightGroup *group, Buf *target, size_t bytes,
                                WeightFill fill, const void *parameters,
                                size_t parameter_bytes);

// Makes target a copy of source whenever the group loads.
void weight_group_add_alias(WeightGroup *group, Buf *target, const Buf *source);

// Loads the group unless it is loaded. Returns 0 if its memory does not fit,
// or if a stop request ended a model's load.
int weight_group_load(WeightGroup *group);

// Sets whether a stop request ends loads, as it does while a model loads.
void weight_group_set_stop_on_quit(int on);

// Sets whether the loaded group's buffers are resident.
void weight_group_set_resident(WeightGroup *group, int resident);

// Frees the group and its buffers.
void weight_group_free(WeightGroup *group);

// Removes a weight from a group that is not loaded.
void weight_group_remove(WeightGroup *group, Buf *target);

#endif
