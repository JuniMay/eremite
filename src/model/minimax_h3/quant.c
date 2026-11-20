#include "model/minimax_h3/quant.h"
#include "base/error.h"
#include "engine/io.h"
#include "engine/pass.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Number of sub-blocks of 32 rows inside each 128-row cuBLASLt scale tile.
static constexpr int SCALE_SUB_BLOCKS = 4;

// Number of rows inside each cuBLASLt scale sub-block.
static constexpr int SCALE_SUB_ROWS = 32;

// Bytes in one 128 by 4 cuBLASLt scale tile.
static constexpr size_t SCALE_TILE_BYTES =
    (size_t)MINIMAX_H3_SCALE_TILE_ROWS * MINIMAX_H3_SCALE_TILE_COLS;

// Maximum bytes read per uncached sector-aligned pread call.
static constexpr size_t READ_CHUNK_BYTES = 16u << 20;

// Parameters stored by a WeightGroup entry that unswizzles NVFP4 scales.
typedef struct {
  // Checkpoint tensor holding the swizzled F8_E4M3 scale bytes.
  Tensor tensor;
  // Output scale matrix height in rows.
  int rows;
  // Output scale matrix width in columns.
  int cols;
} Nvfp4ScalesParam;

void minimax_h3_unswizzle_nvfp4_scales(uint8_t *out, const uint8_t *blocked,
                                       int rows, int cols) {
  if (rows <= 0 || cols <= 0) die("invalid NVFP4 scale matrix shape");
  int col_blocks =
      (cols + MINIMAX_H3_SCALE_TILE_COLS - 1) / MINIMAX_H3_SCALE_TILE_COLS;
  for (int r = 0; r < rows; r++) {
    int rb = r / MINIMAX_H3_SCALE_TILE_ROWS;
    int r_in_tile = r % MINIMAX_H3_SCALE_TILE_ROWS;
    int sub = r_in_tile / SCALE_SUB_ROWS;
    int lane = r_in_tile % SCALE_SUB_ROWS;
    size_t lane_off =
        (size_t)lane * (SCALE_SUB_BLOCKS * MINIMAX_H3_SCALE_TILE_COLS) +
        (size_t)sub * MINIMAX_H3_SCALE_TILE_COLS;
    uint8_t *dst_row = out + (size_t)r * (size_t)cols;
    for (int c = 0; c < cols; c++) {
      int cb = c / MINIMAX_H3_SCALE_TILE_COLS;
      int c_in_tile = c % MINIMAX_H3_SCALE_TILE_COLS;
      size_t tile_base =
          ((size_t)rb * (size_t)col_blocks + (size_t)cb) * SCALE_TILE_BYTES;
      dst_row[c] = blocked[tile_base + lane_off + (size_t)c_in_tile];
    }
  }
}

void minimax_h3_read_tensor_bytes(Tensor tensor, void *out) {
  if (!tensor.size) return;
  uint64_t from = tensor.off, end = from + tensor.size;
  uint64_t at = from & ~(IO_SECTOR - 1);
  uint64_t span = (end - at + IO_SECTOR - 1) & ~(IO_SECTOR - 1);
  size_t room = span < READ_CHUNK_BYTES ? (size_t)span : READ_CHUNK_BYTES;
  uint8_t *scratch = NULL;
  if (posix_memalign((void **)&scratch, IO_SECTOR, room))
    die("out of memory for a %zu-byte read buffer", room);
  while (at < end) {
    uint64_t left = (end - at + IO_SECTOR - 1) & ~(IO_SECTOR - 1);
    size_t part = left < room ? (size_t)left : room;
    uint64_t first = at > from ? at : from;
    uint64_t last = at + part < end ? at + part : end;
    ssize_t n = pread(tensor.fd, scratch, part, (off_t)at);
    if (n < 0 && errno == EINTR) continue;
    if (n < (ssize_t)(last - at))
      die("cannot read tensor bytes at offset %llu", (unsigned long long)at);
    memcpy((uint8_t *)out + (first - from), scratch + (first - at),
           (size_t)(last - first));
    at += part;
  }
  free(scratch);
}

float minimax_h3_read_scalar_f32(Tensor tensor) {
  if (tensor.dtype != 'f' || tensor.size != sizeof(float))
    die("expected an F32 scalar tensor");
  float value = 0;
  minimax_h3_read_tensor_bytes(tensor, &value);
  return value;
}

// Reads and unswizzles a blocked NVFP4 scale tensor into a GPU buffer.
static void fill_nvfp4_scales(Buf *buffer, const void *context) {
  const Nvfp4ScalesParam *param = context;
  uint8_t *blocked = xmalloc(param->tensor.size);
  minimax_h3_read_tensor_bytes(param->tensor, blocked);
  minimax_h3_unswizzle_nvfp4_scales(buffer->ptr, blocked, param->rows,
                                    param->cols);
  free(blocked);
}

void minimax_h3_nvfp4_scales_add(WeightGroup *group, Buf *target, Tensor tensor,
                                 int rows, int cols) {
  if (tensor.dtype != '8' || rows <= 0 || cols <= 0)
    die("NVFP4 scales require an F8_E4M3 tensor with positive dimensions");
  int row_blocks =
      (rows + MINIMAX_H3_SCALE_TILE_ROWS - 1) / MINIMAX_H3_SCALE_TILE_ROWS;
  int col_blocks =
      (cols + MINIMAX_H3_SCALE_TILE_COLS - 1) / MINIMAX_H3_SCALE_TILE_COLS;
  uint64_t expected =
      (uint64_t)row_blocks * (uint64_t)col_blocks * SCALE_TILE_BYTES;
  if (tensor.size != expected)
    die("NVFP4 scale tensor has %llu bytes, expected %llu",
        (unsigned long long)tensor.size, (unsigned long long)expected);
  Nvfp4ScalesParam param = {.tensor = tensor, .rows = rows, .cols = cols};
  weight_group_add_generated(group, target, (size_t)rows * (size_t)cols,
                             fill_nvfp4_scales, &param, sizeof param);
}

// Context passed to PassWait callbacks while yielding for GPU completion.
typedef struct {
  // The GPU whose command buffer is running.
  Gpu *gpu;
  // The committed command buffer number to wait for.
  uint64_t number;
} GpuWaitContext;

// Returns whether the command buffer in context has finished.
static int gpu_wait_ready(void *context) {
  GpuWaitContext *wait = context;
  return gpu_has_finished(wait->gpu, wait->number);
}

// Blocks until the command buffer in context finishes.
static void gpu_wait_block(void *context) {
  GpuWaitContext *wait = context;
  gpu_wait_until_finished(wait->gpu, wait->number);
}

void minimax_h3_gpu_wait(Gpu *gpu, uint64_t number) {
  if (!number) return;
  if (!pass_current()) {
    gpu_wait_until_finished(gpu, number);
    return;
  }
  GpuWaitContext context = {.gpu = gpu, .number = number};
  pass_wait(&(PassWait){gpu_wait_ready, gpu_wait_block, &context, PASS_WAIT_GPU,
                        NULL});
  gpu_check_finished(gpu);
}
