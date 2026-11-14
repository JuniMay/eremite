#ifndef EREMITE_ENGINE_SAMPLING_H
#define EREMITE_ENGINE_SAMPLING_H
#include <stdint.h>

// Samples the next token, and verifies drafts. Verification keeps the drafts
// that the target's own samples confirm. So drafts never change what is
// sampled. A token sampled at a temperature above 0 takes one random draw. The
// workspace belongs to the engine thread.

// Sampling settings and the random stream. At temperature 0, sampling is
// greedy. Otherwise it draws from the softmax at the temperature, within the
// top_k highest-ranked tokens and the smallest share top_p of the
// probability.
typedef struct {
  double temperature, top_p;
  // The random stream's state.
  unsigned short seed[3];
  // The most tokens to draw from, or 0 for every token.
  int top_k;
} Sampler;

// Returns whether the settings are valid.
int sampler_is_valid(const Sampler *s);

// Seeds the random stream.
void sampler_set_seed(Sampler *s, uint64_t seed);

// Returns the token with the highest logit, the lowest id among equals.
int sample_greedy(const float *logits, int count);

// Samples a token from count logits.
int sample_token(const float *logits, int count, Sampler *s);

// Samples a token, and writes the probabilities that it was drawn from to out.
int sample_proposal(const float *logits, int count, Sampler *s, double *out);

// Samples ndraft + 1 rows of count logits in turn, while the samples match
// the drafts. Returns the next token, and sets *accepted to the drafts kept. A
// token for which stop returns nonzero ends the drafts kept. The random stream
// advances as serial sampling's would.
int sample_verify_drafts(const float *logits, int count, const int *drafts,
                         int ndraft, int (*stop)(int token), Sampler *s,
                         int *accepted);

// Verifies sampled drafts. q holds their proposal probabilities, a row of
// count per draft. A draft is kept with probability min(1, p / q). A rejected
// one is sampled again from the normalized max(0, p - q). So the target's
// distribution is unchanged.
int sample_verify_sampled_drafts(const float *logits, int count,
                                 const int *drafts, int ndraft, const double *q,
                                 int (*stop)(int token), Sampler *s,
                                 int *accepted);

#endif
