#ifndef EREMITE_ENGINE_DRAFT_CHOICE_H
#define EREMITE_ENGINE_DRAFT_CHOICE_H

// Chooses how many of a draft model's drafts a verification pass runs. The
// outputs do not depend on it: a pass keeps only the drafts that the target's
// own samples confirm. The speed does, since every verified row costs GPU work
// and expert reads, kept or not.
//
// The choice is the count k with the most expected tokens per second. The
// expected tokens are 1, plus the rate at which drafts 1 to j were all kept,
// for each j up to k. The time is the measured time of a pass with k drafts.
//
// Before it chooses, it measures every count DRAFT_CHOICE_WARM times, in
// rounds from the most down. So no figure rests on one pass, or on the cold
// cache after a prompt. Afterwards, every DRAFT_CHOICE_EXPLORE-th pass tries a
// count next to the best. So the figures follow the text.

// The most drafts of a pass.
static constexpr int DRAFT_CHOICE_MAX = 15;
// How many times each count is measured before the choice starts.
static constexpr int DRAFT_CHOICE_WARM = 3;
// Every DRAFT_CHOICE_EXPLORE-th pass tries a count next to the best.
static constexpr int DRAFT_CHOICE_EXPLORE = 8;
// The most observations that each figure keeps.
static constexpr int DRAFT_CHOICE_MEMORY = 8;

// The measurements that the choice rests on.
typedef struct DraftChoice {
  // [j], for j from 1: the rate at which drafts 1 to j were all kept.
  double kept[DRAFT_CHOICE_MAX + 1];
  // [k]: the time of a pass with k drafts.
  double seconds[DRAFT_CHOICE_MAX + 1];
  // The observations behind each figure, up to DRAFT_CHOICE_MEMORY.
  int kept_n[DRAFT_CHOICE_MAX + 1], seconds_n[DRAFT_CHOICE_MAX + 1];
  // The passes recorded.
  int passes;
} DraftChoice;

// Starts a choice with no measurements.
void draft_choice_init(DraftChoice *choice);

// Returns the drafts for the next pass, from 0 to most, at most
// DRAFT_CHOICE_MAX.
int draft_choice_next_count(DraftChoice *choice, int most);

// Records that a pass with count drafts kept the first kept of them and took
// seconds.
void draft_choice_record_pass(DraftChoice *choice, int count, int kept,
                              double seconds);

#endif
