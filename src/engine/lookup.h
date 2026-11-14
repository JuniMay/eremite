#ifndef EREMITE_ENGINE_LOOKUP_H
#define EREMITE_ENGINE_LOOKUP_H
#include <stdint.h>

// Drafts from the context: the tokens that followed an earlier occurrence of
// the history's last tokens. The history holds only committed tokens. The
// target model verifies every proposal.

// The most drafts of a proposal.
static constexpr int CONTEXT_LOOKUP_DRAFTS = 15;
// The longest match that is counted, in tokens.
static constexpr int CONTEXT_LOOKUP_MATCH = 64;
// The most earlier occurrences examined per index.
static constexpr int CONTEXT_LOOKUP_SEARCH = 64;

// A history of committed tokens, indexed for proposals.
typedef struct ContextLookup ContextLookup;

// A proposal.
typedef struct {
  // The drafts, the length of the matched suffix, where the drafts start in
  // the history, and the occurrences examined.
  int count, matched, source, examined;
  int32_t tokens[CONTEXT_LOOKUP_DRAFTS];
} ContextDraft;

// Returns a lookup for up to capacity tokens, from 1 to 1048576, or null.
ContextLookup *context_lookup_new(int capacity);

// Frees the lookup.
void context_lookup_free(ContextLookup *lookup);

// Forgets the history.
void context_lookup_reset(ContextLookup *lookup);

// Appends count tokens to the history. Returns 0, and appends none, when they
// do not fit. Negative ids are a media part's places. They are never drafted.
int context_lookup_append(ContextLookup *lookup, const int32_t *tokens,
                          int count);

// Proposes up to max_drafts tokens. A match needs 12 tokens. After a proposal
// that was kept whole, it needs 4. After two proposals of which nothing was
// kept, it needs 16.
ContextDraft context_lookup_propose(const ContextLookup *lookup,
                                    int max_drafts);

// Records how many drafts a pass offered, and how many the target kept. An
// empty proposal counts as neither. Returns 0 for invalid counts.
int context_lookup_record_verification(ContextLookup *lookup, int offered,
                                       int accepted);

#endif
