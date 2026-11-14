#include "check.h"
#include "model/deepseek/model.h"
#include <stdio.h>

// Tests that candidate masks follow prompt row indices across decoder
// compaction: both the scoring and the selection of a later indexer take a
// row's bits. The dispatches' offsets are observed on the CPU, without
// weights or a GPU.

enum { SCORE = 102, SELECT = 101 };  // stand-ins for the kernels' numbers

static int dispatch_kernel, observed, masked;
static size_t
    candidate_offset[2];  // as the scoring and the selection receive them

static void spy_kernel(GpuCommandBuffer *c, int k) {
  (void)c;
  dispatch_kernel = k;
}

static void spy_buffer(GpuCommandBuffer *c, int index, const Buf *b,
                       size_t offset) {
  (void)c, (void)b;
  if (dispatch_kernel == SCORE && index == 6)
    candidate_offset[0] = offset, observed++;
  if (dispatch_kernel == SELECT && index == 3)
    candidate_offset[1] = offset, observed++;
}

static void spy_bytes(GpuCommandBuffer *c, int index, const void *data,
                      size_t size) {
  (void)c, (void)size;
  if (dispatch_kernel == SCORE && index == 5)
    masked = ((const uint32_t *)data)[3] != 0;  // Index.cand
}

static void spy_go(GpuCommandBuffer *c, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t tx, uint32_t ty, uint32_t tz) {
  (void)c, (void)x, (void)y, (void)z, (void)tx, (void)ty, (void)tz;
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
  opt.ctx = 32768, cand_words = 129, sc_rows = 256;
  L[24].ratio = L[24].index_source = 1, L[24].src = 20;
  K_index_score = SCORE, K_index_topk = SELECT;
  GpuCommandBuffer c = {0};
  for (int row = 4; row <= MB + 4; row += MB) {
    observed = masked = 0;
    attend_rows(&c, 24, 133, 16384 + row, 0, row);
    size_t expected = (size_t)row * cand_words * 4;
    printf("prompt row %d: scoring offset %zu, selection offset %zu, expected "
           "%zu\n",
           row, candidate_offset[0], candidate_offset[1], expected);
    char what[96];
    snprintf(what, sizeof what,
             "prompt row %d's scoring and selection read its candidate bits",
             row);
    check(observed == 2 && masked && candidate_offset[0] == expected &&
              candidate_offset[1] == expected,
          what);
  }
  return failures != 0;
}
