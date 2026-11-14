#include "check.h"
#include "engine/draft_choice.h"

// Tests the draft count: every count is measured DRAFT_CHOICE_WARM times, in
// rounds from the most down; then the count with the most expected tokens per
// second, and every DRAFT_CHOICE_EXPLORE-th pass a neighbour of it.

static void expect(const char *name, int ok) { check(ok, name); }

int main(void) {
  DraftChoice c;
  draft_choice_init(&c);
  int order[4 * DRAFT_CHOICE_WARM], rounds = 1;
  for (int i = 0; i < 4 * DRAFT_CHOICE_WARM; i++)
    order[i] = draft_choice_next_count(&c, 3),
    draft_choice_record_pass(&c, order[i], order[i], 0.04 + 0.01 * order[i]);
  for (int i = 0; i < 4 * DRAFT_CHOICE_WARM; i++)
    rounds &= order[i] == 3 - i % 4;
  expect("each count is measured DRAFT_CHOICE_WARM times, in rounds from the "
         "most down",
         rounds);
  expect("kept drafts make the most drafts the fastest",
         draft_choice_next_count(&c, 3) == 3);
  expect("a smaller limit bounds the count",
         draft_choice_next_count(&c, 2) == 2);
  for (int i = 0; i < 64; i++) draft_choice_record_pass(&c, 3, 0, 0.07);
  expect("rejected drafts make no drafts the fastest",
         draft_choice_next_count(&c, 3) == 0);
  expect("a figure follows its last observations",
         c.kept[1] < 0.01 && c.kept_n[1] == DRAFT_CHOICE_MEMORY);
  draft_choice_init(&c);
  for (int w = 0; w < DRAFT_CHOICE_WARM; w++)
    for (int k = 0; k <= 3; k++)
      draft_choice_record_pass(&c, k, k, 0.04 + 0.01 * k);
  int explored = 0, choices = 0;
  for (int i = 0; i < 4 * DRAFT_CHOICE_EXPLORE; i++, choices++)
    explored += draft_choice_next_count(&c, 3) != 3;
  expect("every DRAFT_CHOICE_EXPLORE-th pass tries a neighbour",
         explored == choices / DRAFT_CHOICE_EXPLORE);
  draft_choice_init(&c);
  draft_choice_record_pass(&c, 2, 3, 0.05);
  draft_choice_record_pass(&c, 16, 0, 0.05);
  draft_choice_record_pass(&c, 1, 0, 0);
  expect("invalid feedback is ignored",
         !c.seconds_n[1] && !c.seconds_n[2] && !c.kept_n[1]);
  expect("no room means no drafts", draft_choice_next_count(&c, 0) == 0);

  return failures != 0;
}
