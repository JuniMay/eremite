#include "engine/draft_choice.h"
#include <string.h>

void draft_choice_init(DraftChoice *c) { memset(c, 0, sizeof *c); }

// A running mean of the first DRAFT_CHOICE_MEMORY observations, then an
// exponential one with that weight, so that a figure starts at its first
// value and follows changes of the text.
static void running(double *value, int *n, double x) {
  if (*n < DRAFT_CHOICE_MEMORY) ++*n;
  *value += (x - *value) / *n;
}

int draft_choice_next_count(DraftChoice *c, int most) {
  if (most > DRAFT_CHOICE_MAX) most = DRAFT_CHOICE_MAX;
  if (most <= 0) return 0;
  for (int round = 1; round <= DRAFT_CHOICE_WARM; round++)
    for (int k = most; k >= 0; k--)
      if (c->seconds_n[k] < round) return k;
  int best = 0;
  double tokens = 1, rate = 1 / c->seconds[0];
  for (int k = 1; k <= most; k++) {
    tokens += c->kept[k];
    if (tokens / c->seconds[k] > rate) best = k, rate = tokens / c->seconds[k];
  }
  if (++c->passes % DRAFT_CHOICE_EXPLORE) return best;
  int up = (c->passes / DRAFT_CHOICE_EXPLORE) %
           2;  // alternately the count above and the one below
  return up ? (best < most ? best + 1 : best - 1)
            : (best > 0 ? best - 1 : best + 1);
}

void draft_choice_record_pass(DraftChoice *c, int count, int kept,
                              double seconds) {
  if (count < 0 || count > DRAFT_CHOICE_MAX || kept < 0 || kept > count ||
      !(seconds > 0))
    return;
  running(&c->seconds[count], &c->seconds_n[count], seconds);
  for (int j = 1; j <= count; j++)
    running(&c->kept[j], &c->kept_n[j], kept >= j);
}
