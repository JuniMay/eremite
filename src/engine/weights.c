#include "engine/weights.h"
#include "base/error.h"
#include "base/shutdown.h"
#include "engine/io.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  Buf *target;
  const Buf *alias;
  size_t bytes;
  WeightFill fill;
  void *parameters;
} Weight;

struct WeightGroup {
  // The GPU that the group's buffers are made on.
  Gpu *gpu;
  MemorySlot slot;
  Weight *items;
  size_t count, capacity;
  uint64_t required, scratch;
  uint64_t read_bytes;
  int read_error;
};

static _Thread_local WeightGroup *reading_group;

// While a model loads, a stop request ends its reads between weights and
// between parts of one; the load then frees what it took and fails. Weights
// that a pass loads complete with the pass.
static int stop_on_quit;

void weight_group_set_stop_on_quit(int on) { stop_on_quit = on; }

static int quitting(void) { return stop_on_quit && shutdown_requested(); }

static void unload(void *context) {
  WeightGroup *group = context;
  for (size_t i = 0; i < group->count; i++) {
    Weight *weight = &group->items[i];
    if (weight->alias)
      *weight->target = (Buf){0};
    else
      gpu_free(weight->target);
  }
  gpu_commit_residency(group->gpu);
}

WeightGroup *weight_group_new(Gpu *gpu, MemoryAccount *account,
                              const char *name) {
  WeightGroup *group = xmalloc(sizeof *group);
  group->gpu = gpu;
  memory_slot_init(&group->slot, account, MEMORY_MODEL, NULL, name, unload,
                   group);
  return group;
}

WeightGroup *weight_group_new_dependent(WeightGroup *backbone,
                                        const char *name) {
  WeightGroup *group = xmalloc(sizeof *group);
  group->gpu = backbone->gpu;
  memory_slot_init(&group->slot, backbone->slot.account, MEMORY_MODEL,
                   &backbone->slot, name, unload, group);
  return group;
}

MemorySlot *weight_group_slot(WeightGroup *group) { return &group->slot; }

uint64_t weight_group_required_bytes(const WeightGroup *group) {
  if (!group) return 0;
  return group->required + group->scratch;
}

uint64_t weight_group_resident_bytes(const WeightGroup *group) {
  return group && memory_ready(&group->slot) ? group->slot.payload : 0;
}

static Weight *append(WeightGroup *group, Buf *target) {
  if (group->count == group->capacity) {
    group->capacity = group->capacity ? group->capacity * 2 : 64;
    Weight *items = realloc(group->items, group->capacity * sizeof *items);
    if (!items) die("out of memory for weight metadata");
    group->items = items;
  }
  Weight *weight = &group->items[group->count++];
  *weight = (Weight){.target = target};
  return weight;
}

void weight_group_add_generated(WeightGroup *group, Buf *target, size_t bytes,
                                WeightFill fill, const void *parameters,
                                size_t parameter_bytes) {
  Weight *weight = append(group, target);
  weight->bytes = bytes;
  weight->fill = fill;
  if (parameter_bytes) {
    weight->parameters = xmalloc(parameter_bytes);
    memcpy(weight->parameters, parameters, parameter_bytes);
  }
  group->required += gpu_round_up_to_pages(bytes);
  target->size = bytes;
}

void weight_group_add_alias(WeightGroup *group, Buf *target,
                            const Buf *source) {
  Weight *weight = append(group, target);
  weight->alias = source;
}

// Checkpoint files are opened without the page cache, but an uncached read
// bypasses it only in whole 16 KiB sectors into page-aligned memory; other
// reads still fill it, and the kernel keeps that file cache. So a tensor at
// its own offset is read in whole sectors, up to READ_PIECE at a time, into a
// page-aligned buffer, and its bytes are copied out.
static constexpr size_t READ_ALIGN = 16384, READ_PIECE = 16 << 20;

static void read_bytes(Tensor tensor, size_t offset, void *target,
                       size_t bytes) {
  reading_group->read_bytes += bytes;
  uint64_t from = tensor.off + offset, end = from + bytes,
           at = from & ~(uint64_t)(READ_ALIGN - 1);
  uint64_t span = (end - at + READ_ALIGN - 1) & ~(uint64_t)(READ_ALIGN - 1);
  size_t room = span < READ_PIECE ? (size_t)span : READ_PIECE;
  uint8_t *buffer = NULL;
  if (posix_memalign((void **)&buffer, READ_ALIGN, room))
    die("out of memory for a %zu-byte read buffer", room);
  while (at < end) {
    if (quitting()) {
      reading_group->read_error = ECANCELED;
      break;
    }
    uint64_t left = (end - at + READ_ALIGN - 1) & ~(uint64_t)(READ_ALIGN - 1);
    size_t part = left < room ? (size_t)left : room;
    uint64_t first = at > from ? at : from,
             last = at + part < end ? at + part : end;  // the bytes wanted
    ssize_t n = pread(tensor.fd, buffer, part, (off_t)at);
    if (n < 0 && errno == EINTR) continue;
    if (n < (ssize_t)(last - at))
      die("cannot reload weight data at offset %llu", (unsigned long long)at);
    memcpy((uint8_t *)target + (first - from), buffer + (first - at),
           last - first);
    at += part;
  }
  free(buffer);
}

typedef struct {
  Tensor tensor;
  int aligned;
} Raw;

static void raw_fill(Buf *buffer, const void *context) {
  const Raw *raw = context;
  if (raw->aligned) {
    reading_group->read_bytes += raw->tensor.size;
    io_read(raw->tensor.fd, raw->tensor.off, raw->tensor.size, buffer);
  } else
    read_bytes(raw->tensor, 0, buffer->ptr, raw->tensor.size);
}

void weight_group_add_tensor(WeightGroup *group, Buf *target, Tensor tensor,
                             int aligned) {
  Raw raw = {tensor, aligned};
  weight_group_add_generated(group, target,
                             tensor.size + (aligned ? 2 * 16384 : 0), raw_fill,
                             &raw, sizeof raw);
}

int weight_group_decode_compressed_scales(uint8_t *out, const uint8_t *fixed,
                                          size_t fixed_bytes,
                                          const uint32_t *exceptions,
                                          size_t exception_count, size_t rows,
                                          size_t cols) {
  if (!rows || rows % 16 || !cols || cols % 2 || rows > (1u << 24) / cols)
    return 0;
  size_t stride = 16 + cols * 8;
  if (fixed_bytes != rows / 16 * stride) return 0;
  for (size_t row = 0; row < rows; row++) {
    const uint8_t *slab = fixed + row / 16 * stride;
    unsigned base = slab[row % 16];
    if (base > 240) return 0;
    const uint8_t *codes = slab + 16 + row % 16 * (cols / 2);
    for (size_t col = 0; col < cols; col++)
      out[row * cols + col] =
          (uint8_t)(base + ((codes[col / 2] >> (4 * (col % 2))) & 15));
  }
  uint32_t previous = 0;
  for (size_t i = 0; i < exception_count; i++) {
    uint32_t entry = exceptions[i], position = entry & 0xffffffu;
    if (position >= rows * cols || (i && position <= previous)) return 0;
    out[position] = (uint8_t)(entry >> 24);
    previous = position;
  }
  return 1;
}

typedef struct {
  Tensor fixed, exceptions;
  int rows, cols;
} CompressedScales;

static void csf_fill(Buf *buffer, const void *context) {
  const CompressedScales *scales = context;
  void *fixed =
           memory_host_alloc_required(&reading_group->slot, scales->fixed.size),
       *exceptions = memory_host_alloc_required(&reading_group->slot,
                                                scales->exceptions.size);
  read_bytes(scales->fixed, 0, fixed, scales->fixed.size);
  read_bytes(scales->exceptions, 0, exceptions, scales->exceptions.size);
  int decoded = !reading_group->read_error &&
                weight_group_decode_compressed_scales(
                    buffer->ptr, fixed, scales->fixed.size, exceptions,
                    scales->exceptions.size / 4, scales->rows, scales->cols);
  memory_host_free(&reading_group->slot, fixed);
  memory_host_free(&reading_group->slot, exceptions);
  if (!reading_group->read_error && !decoded)
    die("invalid compressed NVFP4 scale stream");
}

void weight_group_add_compressed_scales(WeightGroup *group, Buf *target,
                                        Tensor fixed, Tensor exceptions,
                                        int rows, int cols) {
  if (fixed.dtype != 'u' || exceptions.dtype != 'U' || exceptions.size % 4)
    die("compressed NVFP4 scales require U8 codes and U32 exceptions");
  uint64_t scratch = fixed.size + exceptions.size;
  if (scratch > group->scratch) group->scratch = scratch;
  CompressedScales scales = {fixed, exceptions, rows, cols};
  weight_group_add_generated(group, target, (size_t)rows * cols, csf_fill,
                             &scales, sizeof scales);
}

typedef struct {
  Tensor first, second;
  float add;
  int operations, product;
} Vector;

static float value(const uint8_t *bytes, int index, int dtype) {
  if (dtype == 'b') {
    uint16_t short_bits;
    memcpy(&short_bits, bytes + (size_t)index * 2, 2);
    uint32_t bits = (uint32_t)short_bits << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
  }
  if (dtype == 'h') {
    _Float16 half_value;
    memcpy(&half_value, bytes + (size_t)index * 2, 2);
    return (float)half_value;
  }
  float result;
  memcpy(&result, bytes + (size_t)index * 4, 4);
  return result;
}

static size_t vector_element_bytes(int dtype) {
  return dtype == 'b' || dtype == 'h' ? 2 : 4;
}

static void vector_fill(Buf *buffer, const void *context) {
  const Vector *vector = context;
  uint8_t first[16384], second[16384];
  size_t width = vector_element_bytes(vector->first.dtype);
  size_t count = vector->first.size / width;
  float *output = (float *)buffer->ptr;
  for (size_t at = 0; at < count; at += 4096) {
    size_t n = count - at < 4096 ? count - at : 4096;
    read_bytes(vector->first, at * width, first, n * width);
    if (vector->product) {
      size_t other_width = vector_element_bytes(vector->second.dtype);
      read_bytes(vector->second, at * other_width, second, n * other_width);
    }
    for (size_t i = 0; i < n; i++) {
      float v = value(first, (int)i, vector->first.dtype);
      if (vector->operations & WEIGHT_ADD) v += vector->add;
      if (vector->product) v *= value(second, (int)i, vector->second.dtype);
      if (vector->operations & WEIGHT_EXP) v = expf(v);
      output[at + i] = v;
    }
  }
}

void weight_group_add_vector(WeightGroup *group, Buf *target, Tensor tensor,
                             float add, int operations) {
  if (tensor.dtype != 'b' && tensor.dtype != 'h' && tensor.dtype != 'f')
    die("weight vector must use BF16, F16 or FP32");
  Vector vector = {.first = tensor, .add = add, .operations = operations};
  size_t bytes = tensor.size / vector_element_bytes(tensor.dtype) * 4;
  weight_group_add_generated(group, target, bytes, vector_fill, &vector,
                             sizeof vector);
}

void weight_group_add_product(WeightGroup *group, Buf *target, Tensor first,
                              Tensor second) {
  Vector vector = {.first = first, .second = second, .product = 1};
  size_t n = first.size / vector_element_bytes(first.dtype);
  if (n != second.size / vector_element_bytes(second.dtype))
    die("weight product sizes differ");
  weight_group_add_generated(group, target, n * 4, vector_fill, &vector,
                             sizeof vector);
}

typedef struct {
  Tensor tensor;
  int stride;
} Padded;

static void padded_fill(Buf *buffer, const void *context) {
  const Padded *padded = context;
  size_t rows = padded->tensor.shape[0], row_bytes = padded->tensor.size / rows;
  memset(buffer->ptr, 0, buffer->size);
  for (size_t row = 0; row < rows; row++)
    read_bytes(padded->tensor, row * row_bytes,
               buffer->ptr + row * padded->stride * 2, row_bytes);
}

void weight_group_add_padded(WeightGroup *group, Buf *target, Tensor tensor,
                             int stride) {
  if (tensor.dtype != 'b' || stride * 2u < tensor.size / tensor.shape[0])
    die("invalid padded BF16 weight");
  Padded padded = {tensor, stride};
  weight_group_add_generated(group, target,
                             (size_t)tensor.shape[0] * stride * 2, padded_fill,
                             &padded, sizeof padded);
}

typedef struct {
  Tensor weight, scales;
  int kind, cols, stride;
} Expanded;

static float e4m3(uint8_t code) {
  unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
  float value = exponent ? ldexpf(1 + mantissa * .125f, (int)exponent - 7)
                         : ldexpf((float)mantissa, -9);
  return code & 128 ? -value : value;
}

static void expanded_fill(Buf *buffer, const void *context) {
  const Expanded *p = context;
  uint8_t *weight =
      memory_host_alloc_required(&reading_group->slot, p->weight.size);
  uint8_t *scales =
      memory_host_alloc_required(&reading_group->slot, p->scales.size);
  read_bytes(p->weight, 0, weight, p->weight.size);
  if (p->scales.size) read_bytes(p->scales, 0, scales, p->scales.size);
  memset(buffer->ptr, 0, buffer->size);
  static const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int row = 0; row < p->weight.shape[0]; row++)
    for (int col = 0; col < p->cols; col++) {
      float value;
      if (p->kind == 'n') {
        uint8_t code =
            weight[(size_t)row * (p->cols / 2) + col / 2] >> ((col % 2) * 4);
        value = values[code & 7];
        if (code & 8) value = -value;
        value *= e4m3(scales[(size_t)row * (p->cols / 16) + col / 16]);
      } else {
        value = e4m3(weight[(size_t)row * p->cols + col]);
        if (p->kind == 'm')
          value *= ldexpf(
              1, (int)scales[(size_t)row * (p->cols / 32) + col / 32] - 127);
      }
      uint32_t bits;
      memcpy(&bits, &value, 4);
      ((uint16_t *)buffer->ptr)[(size_t)row * p->stride + col] =
          (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }
  memory_host_free(&reading_group->slot, weight);
  memory_host_free(&reading_group->slot, scales);
}

void weight_group_add_expanded(WeightGroup *group, Buf *target, Tensor weight,
                               Tensor scales, int kind, int cols, int stride) {
  Expanded p = {weight, scales, kind, cols, stride};
  uint64_t scratch = weight.size + scales.size;
  if (scratch > group->scratch) group->scratch = scratch;
  weight_group_add_generated(group, target,
                             (size_t)weight.shape[0] * stride * 2,
                             expanded_fill, &p, sizeof p);
}

static int allocate_group(WeightGroup *group) {
  MemorySlot *slots[] = {&group->slot};
  uint64_t required = weight_group_required_bytes(group);
  if (!memory_admit(slots, &required, 1, MEMORY_REQUIRED)) return 0;
  memory_pin(&group->slot, 1);
  group->read_bytes = 0;
  group->read_error = 0;
  for (size_t i = 0; i < group->count; i++) {
    Weight *weight = &group->items[i];
    if (weight->alias) continue;
    *weight->target = gpu_alloc(group->gpu, &group->slot, weight->bytes);
    gpu_set_resident(weight->target, 1);
  }
  return 1;
}

static void fill_group(WeightGroup *group) {
  WeightGroup *previous = reading_group;
  reading_group = group;
  for (size_t i = 0; i < group->count && !group->read_error; i++) {
    Weight *weight = &group->items[i];
    if (quitting())
      group->read_error = ECANCELED;
    else if (!weight->alias)
      weight->fill(weight->target, weight->parameters);
  }
  reading_group = previous;
}

static void finish_group(WeightGroup *group, unsigned pinned) {
  for (size_t i = 0; i < group->count; i++)
    if (group->items[i].alias) *group->items[i].target = *group->items[i].alias;
  memory_shrink(&group->slot);
  memory_set_ready(&group->slot, 1);
  memory_pin(&group->slot, pinned);
}

// A load that did not complete: its buffers and their count go back, and the
// group keeps the pin it had before the load.
static void discard(WeightGroup *group, unsigned pinned) {
  unload(group);
  memory_shrink(&group->slot);
  memory_pin(&group->slot, pinned);
}

int weight_group_load(WeightGroup *group) {
  if (memory_ready(&group->slot)) {
    memory_touch(&group->slot);
    return 1;
  }
  unsigned pinned = group->slot.pins;
  if (!allocate_group(group)) return 0;
  fill_group(group);
  if (group->read_error) {  // a stop request ended it; other read errors end
                            // the program
    discard(group, pinned);
    return 0;
  }
  finish_group(group, pinned);
  gpu_commit_residency(group->gpu);
  return 1;
}

void weight_group_set_resident(WeightGroup *group, int resident) {
  if (!group || !memory_ready(&group->slot)) return;
  for (size_t i = 0; i < group->count; i++)
    if (!group->items[i].alias)
      gpu_set_resident(group->items[i].target, resident);
}

void weight_group_free(WeightGroup *group) {
  if (!group) return;
  unload(group);
  memory_shrink(&group->slot);
  memory_slot_remove(&group->slot);
  for (size_t i = 0; i < group->count; i++) free(group->items[i].parameters);
  free(group->items);
  free(group);
}

void weight_group_remove(WeightGroup *group, Buf *target) {
  if (memory_ready(&group->slot)) die("cannot change a resident weight group");
  for (size_t i = 0; i < group->count; i++)
    if (group->items[i].target == target) {
      Weight *weight = &group->items[i];
      if (!weight->alias)
        group->required -= gpu_round_up_to_pages(weight->bytes);
      free(weight->parameters);
      memmove(weight, weight + 1, (group->count - i - 1) * sizeof *weight);
      group->count--;
      *target = (Buf){0};
      return;
    }
}
