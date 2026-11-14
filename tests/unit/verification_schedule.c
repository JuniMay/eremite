#include "check.h"
#include "model/deepseek/model.h"
#include <string.h>

// Tests the dispatches of wide verification passes on the CPU, without a Metal
// device.

static int kernel, dispatches, kernels[12], row_counts[12];
static size_t input_offsets[12], output_offsets[12];

static void spy_kernel(GpuCommandBuffer *c, int value) {
  (void)c;
  kernel = value;
}

static void spy_buffer(GpuCommandBuffer *c, int index, const Buf *b,
                       size_t offset) {
  (void)c, (void)b;
  if (index == 2) input_offsets[dispatches] = offset;
  if (index == 3) output_offsets[dispatches] = offset;
}

static void spy_bytes(GpuCommandBuffer *c, int index, const void *bytes,
                      size_t size) {
  (void)c, (void)size;
  if (index == 4 && kernel == 101)
    row_counts[dispatches] = ((const uint32_t *)bytes)[6];
}

static void spy_go(GpuCommandBuffer *c, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t tx, uint32_t ty, uint32_t tz) {
  (void)c, (void)x, (void)y, (void)z, (void)tx, (void)ty, (void)tz;
  kernels[dispatches++] = kernel;
}

#define gpu_select_kernel spy_kernel
#define gpu_bind_buffer spy_buffer
#define gpu_copy_bytes spy_bytes
#define gpu_run_kernel spy_go
#include "model/deepseek/model.c"
#undef gpu_select_kernel
#undef gpu_bind_buffer
#undef gpu_copy_bytes
#undef gpu_run_kernel

int main(void) {
  opt.lookup = 0;
  // the window and the rewind margin; the compressor two rows more at least
  check(RING == WINDOW + REWIND && PENDING >= REWIND + 2,
        "ordinary decoding keeps the window and the rewind margin");
  opt.draft = BLOCK;
  check(RING == WINDOW + REWIND,
        "DSpark's drafts fit within the rewind margin");
  opt.draft = 0, opt.lookup = 15;
  check(RING == 143 && PENDING >= 17,
        "wide verification retains overwritten local history");
  K_mvm_bf16 = 101;
  K_mm_bf16 = 102;
  Lin weights = {.n = EXPERTS, .k = DIM};
  Buf input = {0}, output = {0};
  Prod p = {&weights, &output, 128, EXPERTS};
  GpuCommandBuffer c = {0};
  verifying = 1;
  products(&c, 1, &p, 16, &input, 64, DIM, 0);
  check(dispatches == 3, "sixteen rows use three bounded vector batches");
  check(row_counts[0] == 6 && row_counts[1] == 6 && row_counts[2] == 4,
        "batch row counts are six, six and four");
  check(kernels[0] == 101 && kernels[1] == 101 && kernels[2] == 101,
        "all segments use the vector kernel");
  check(input_offsets[0] == 64 && input_offsets[1] == 64 + 6u * DIM * 4 &&
            input_offsets[2] == 64 + 12u * DIM * 4,
        "segments read their own input rows");
  check(output_offsets[0] == 128 &&
            output_offsets[1] == 128 + 6u * EXPERTS * 4 &&
            output_offsets[2] == 128 + 12u * EXPERTS * 4,
        "segments write their own output rows");
  dispatches = 0;
  verifying = 0;
  products(&c, 1, &p, 16, &input, 64, DIM, 0);
  check(dispatches == 1 && kernels[0] == 102,
        "ordinary prefill retains the matrix path");
  return failures != 0;
}
