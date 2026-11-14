#ifndef EREMITE_MODEL_DEEPSEEK_PREFILL_H
#define EREMITE_MODEL_DEEPSEEK_PREFILL_H
#include <stdint.h>

// The plan of a pruned prefill: the rows of a prompt chunk that the decoder
// layers run. They run the rows from prefix_begin to prefix_end, around a
// saved prefix's end, and the rows from tail_begin to the chunk's end. kept is
// the rows they run, and suffix_rows the rows from tail_begin on.
typedef struct {
  int prefix_begin, prefix_end, tail_begin, kept, suffix_rows;
} PrefillPrune;

// Returns the plan for count rows from start, for a prompt that ends at end
// and a saved prefix that ends at prefix, or -1 for none. The decoder layers
// mix positions only through a causal window. So a row matters only within a
// halo of window + rewind + (stages + 1) * (window - 1) positions before the
// prompt's end or the prefix's end. The kept rows start and end at the matrix
// batches' bounds, every block rows, so that the batches keep their shapes.
static inline PrefillPrune prefill_prune_plan(int start, int count, int end,
                                              int prefix, int stages,
                                              int window, int rewind,
                                              int block) {
  if (count <= 0) return (PrefillPrune){0};
  if (stages < 0 || window < 1 || rewind < 0 || block < 1)
    return (PrefillPrune){.kept = count, .suffix_rows = count};
  int64_t halo =
      (int64_t)window + rewind + ((int64_t)stages + 1) * (window - 1);
  int64_t tail = (int64_t)end - halo - start, saved = (int64_t)prefix - start,
          before = saved - halo;
  int first = tail <= 0       ? 0
              : tail >= count ? count
                              : (int)(tail / block * block);
  int last = prefix < 0 || saved <= 0 ? 0
             : saved >= count ? count
                              : (int)((saved + block - 1) / block * block);
  if (last > count) last = count;
  int begin = !last || before <= 0 ? 0
              : before >= count    ? count
                                   : (int)(before / block * block);
  if (begin == last) begin = last = 0;
  if (last && last >= first) {
    if (begin < first) first = begin;
    begin = last = 0;
  }
  return (PrefillPrune){begin, last, first, last - begin + count - first,
                        count - first};
}

#endif
