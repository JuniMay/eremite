#ifndef EREMITE_MODEL_MODEL_H
#define EREMITE_MODEL_MODEL_H
#include "engine/expert_store.h"
#include "engine/media.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/sampling.h"
#include "gpu/gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Every model behind one interface. A model family fills a ModelOps table.
// model_open reads a checkpoint's config.json, picks the family that runs its
// model_type, and loads the model.
//
// A process holds at most one model of each family, since a family keeps its
// model in globals. Every call on models and their states runs on one thread.

// DeepSeek's options.
typedef struct {
  // When above 0, a missing expert is replaced by a resident one whose
  // selection key is at most this far below.
  float substitute;
  // A GGUF file (set) whose routed experts replace the checkpoint's, or null.
  const char *experts;
  // The tokens of approximate replay for the decoder layers, at least WINDOW;
  // 0 for none.
  int ced;
  // Whether prefill skips the decoder batches that neither the last row nor a
  // saved prefix depends on.
  int prefill_prune;
} DeepSeekOptions;

// Common execution settings and family-specific options.
typedef struct {
  // Capacity in tokens.
  int ctx;
  // A pipe that receives each layer's streams, or -1.
  int dump_fd;
  // A pipe that receives each decoded token's routes, predictions and router
  // logits, or -1.
  int routes_fd;
  // The drafts per pass of the model's own draft heads; 0 for none.
  int draft;
  // The context-copy drafts per pass, at most the model's verification
  // capacity.
  int lookup;
  // No progress lines on stderr.
  int quiet;

  union {
    DeepSeekOptions deepseek;
  };
} Options;

// One sequence's state, of a family's own type.
typedef struct ModelState ModelState;
// A state's rolling part at one position.
typedef struct ModelSnapshot ModelSnapshot;

// One array of a state's rows, which only grows. state_copy copies these
// arrays, and saved states hold them. Row 0 is at data, and each row takes unit
// bytes. A state of p positions holds its first (p - lag) / ratio + guard rows,
// and none while p < lag. The rows before (p - lag) / ratio do not change at
// later positions. A guard row is computed again where the state continues:
// the last row of a compressed pair, or with lag 1, a row that needs the next
// position's token.
typedef struct {
  void *data;
  uint32_t unit, ratio, guard, lag;
} StateRegion;

// One part of a snapshot's bytes, in its family's order.
typedef struct {
  void *data;
  size_t bytes;
} SnapshotPart;

// The most regions of a state, and the most parts and integers of a snapshot.
static constexpr int STATE_REGIONS_MAX = 128, SNAPSHOT_PARTS_MAX = 160,
                     SNAPSHOT_META_MAX = 8;

static inline int64_t region_stable(const StateRegion *r, int positions) {
  return positions < (int)r->lag ? 0 : (positions - (int)r->lag) / r->ratio;
}

static inline int64_t region_rows(const StateRegion *r, int positions) {
  return positions < (int)r->lag ? 0 : region_stable(r, positions) + r->guard;
}

// Copies the rows before `positions` of n regions, from src's to dst's. Both
// lists are for that many positions.
static inline void regions_copy(const StateRegion *dst, const StateRegion *src,
                                int n, int positions) {
  for (int i = 0; i < n; i++)
    memcpy(dst[i].data, src[i].data,
           (size_t)region_rows(&src[i], positions) * src[i].unit);
}

// A model's counters since it loaded, for the server's status. Only a model
// with routed experts (experts_per_row > 0) has the expert cache's.
typedef struct {
  // Rows of decoding and prompt passes.
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  // The prompt pass running: its layer (0: none) and rows.
  int prefill_layer, prefill_rows_now;
  // The layers, and the experts a row chooses: 0 without routed experts.
  int layers, experts_per_row;
  // The layers with routed experts, and each one's experts.
  int routed_layers, experts_per_layer;
  // The weights always resident.
  uint64_t dense_bytes;
  // The expert cache's counts of decoded rows and of prompt rows.
  ExpertStoreCounts decode, prefill;
  ExpertStoreCounts cache;
  uint64_t cache_bytes, slot_bytes, uncached_bytes;
  int cache_slots, mapped, cap, max_slots, experts, uncached;
} ModelCounters;

// The last prefill's plan: its chunks, the rows it ran from its first row, and
// the rows its later layers ran. Also whether approximate or pruned prefill
// applied: DeepSeek's CED and dependency pruning.
typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} ModelPlan;

typedef struct {
  // The family's name, for the log.
  const char *family;
  // The config.json model_type values it runs, ending with a null.
  const char *const *model_types;
  // The model's encoded-media placeholder, or null.
  const char *media_token;
  // Loads the model of dir on gpu, its workspace counted in work.
  int (*load)(Gpu *gpu, MemorySlot *work, const char *dir, const Options *o);
  // Returns the slot of its dense weights, which it loads again after an
  // eviction.
  MemorySlot *(*dense_weight_slot)(void);
  // Sets whether a request runs. While one runs, the model keeps what the
  // request may still use, such as its expert slots.
  void (*set_request_running)(int running);
  // Writes the model's counters to c.
  void (*read_counters)(ModelCounters *c);
  // Returns the bytes in the expert store. It is null for a model without one.
  uint64_t (*expert_store_bytes)(void);
  // Writes the last prefill's plan. It is null for a family that does not plan
  // its own.
  void (*read_last_plan)(ModelPlan *p);
  // These return the limits after loading. vocab_size: the vocabulary's size.
  // max_rows: the rows of different states that one decoding pass takes.
  // max_verify: the rows that one verify pass takes, the token and its drafts.
  // drafts_per_pass: the drafts per pass that the model makes itself, at most
  // what its draft head makes, and 0 without one. rewind_margin: the positions
  // a state can continue before its end.
  int (*vocab_size)(void);
  int (*max_rows)(void);
  int (*max_verify)(void);
  int (*drafts_per_pass)(void);
  int (*rewind_margin)(void);
  // Returns whether a sampled token ends a reply.
  int (*is_stop_token)(int token);

  // Returns a new state, or null when its memory does not fit.
  ModelState *(*state_new)(void);
  void (*state_free)(ModelState *s);
  // Returns the bytes of s, or of a new state for a null s, when it holds that
  // many positions.
  size_t (*state_bytes)(const ModelState *s, int positions);
  // Copies the rows of src before `positions` that later positions do not
  // change into dst. src may be in use by a running pass.
  void (*state_copy)(ModelState *dst, const ModelState *src, int positions);

  ModelSnapshot *(*snapshot_take)(ModelState *s, int pos, const float *logits);
  void (*snapshot_restore)(ModelState *s, const ModelSnapshot *x);
  void (*snapshot_free)(ModelSnapshot *x);
  size_t (*snapshot_bytes)(int with_logits);
  // Returns the snapshot's logits, or null if it has none.
  const float *(*snapshot_logits)(const ModelSnapshot *x);

  // The version of the layout of the regions and snapshot parts below. A saved
  // state of another version is not read.
  int state_version;
  // Writes s's regions for `positions` positions, at most STATE_REGIONS_MAX:
  // the arrays that state_copy copies. Returns their count. With fill, s is
  // first made to hold them: it holds `positions` positions and continues as
  // after a state_copy of that many. A snapshot restored into it, or a replay,
  // then rebuilds the rest.
  int (*write_state_regions)(ModelState *s, int positions, int fill,
                             StateRegion *out);
  // Writes the parts of snapshot x, or for a null x, of s's rolling rows as
  // they are at pos, at most SNAPSHOT_PARTS_MAX and always in one order. A
  // snapshot's logits, if it has them, come last. Writes its integers to meta,
  // at most SNAPSHOT_META_MAX, and their count to *nmeta. Returns the count of
  // parts, or -1 if s's rolling rows cannot be as at pos: a family whose state
  // knows its own position.
  int (*write_snapshot_parts)(ModelSnapshot *x, ModelState *s, int pos,
                              SnapshotPart *out, int32_t *meta, int *nmeta);
  // Returns an empty snapshot with those integers, for its parts to be read
  // into, or null if they do not suit the model.
  ModelSnapshot *(*snapshot_new_empty)(const int32_t *meta, int nmeta,
                                       int with_logits);

  // Prefills positions from to n - 1 of a prompt in s, and returns the last
  // one's logits. tokens holds all n of the prompt's tokens. With replay, the
  // state holds only the rows that state_copy copies before from, and the
  // prefill rebuilds the rest; replay_rows returns -1 for a model whose rolling
  // rows cannot be rebuilt from the rows it keeps: it resumes only from a
  // snapshot.
  // A capture of 0 or more takes a snapshot at that position into *captured.
  // A non-null every gets the logits of every position from from to n - 1.
  const float *(*prefill)(ModelState *s, const int *tokens, int n, int from,
                          int replay, int capture, ModelSnapshot **captured,
                          float *every);
  int (*replay_rows)(int k);
  int (*replay_start)(int k);
  // Runs one token for each of m states, m at most max_rows, and returns rows
  // of logits. Row r is tokens[r] at pos[r] of states[r], that state's next
  // position.
  const float *(*decode_rows)(ModelState *const *states, const int *tokens,
                              const int *pos, int m);
  // Drafts. propose_drafts writes up to drafts_per_pass() drafts after token
  // at pos, once every position before it has run. It returns the sampled
  // proposals' probabilities, n vocabulary rows, or null for proposals without
  // sampling. verify runs m rows from pos, the token and then its drafts, and
  // returns each row's logits. accept keeps the first n rows.
  const double *(*propose_drafts)(ModelState *s, int token, int pos,
                                  int *drafts, int n, Sampler *sampler);
  const float *(*verify)(ModelState *s, const int *tokens, int m, int pos);
  void (*accept)(ModelState *s, const int *tokens, int n, int pos);
  // Sets the state's token history at pos from the tokens before it, for a
  // model that keeps one; otherwise it does nothing.
  void (*set_history)(ModelState *s, const int *tokens, int pos);
  // Runs about once a second while no request runs, and gives memory back to
  // other programs if they need it. Returns the model's expert slots, or 0
  // without an expert store.
  int (*run_idle_work)(void);
  // Prints the model's own lines of the command line's report. At 0, when
  // decoding starts, it notes its counters. At 1, after decoding, it prints
  // what it measured meanwhile, per token of `tokens`. May be null.
  void (*print_report)(FILE *out, int at, int tokens);
  // These prepare and encode the model's media, and set their feature spans
  // for the next prefill. Null functions mean the model takes text only.
  int (*prepare_media)(const char *dir, Media *im, const void *data, size_t len,
                       const char **error);
  // Returns 0 if the memory does not fit the encoding.
  int (*encode_media)(Media *im);
  void (*set_prefill_media)(const Media *im, int n);
  // Writes the experts that row `row` of the latest decoding pass chose, and
  // where each one was; null for a model without routed experts.
  int (*read_routes)(int row, uint16_t *id, uint8_t *from);
  // Writes one user turn of the command line, with media placeholders and the
  // chosen reasoning effort.
  void (*render_chat_turn)(char *text, size_t cap, const char *prompt,
                           int media_count, int think);
  // Runs the command line's --bench: times the kernels on the model's own
  // matrices. It may be null.
  void (*run_bench)(void);
  // pass_new returns a pass of this model that will run body(argument).
  // Inside the pass, pass_begin starts its work on buffers of its own, and
  // pass_end ends it.
  Pass *(*pass_new)(void (*body)(void *), void *argument);
  void (*pass_begin)(void);
  void (*pass_end)(void);
  // Sets how many passes besides the running one have GPU work. The engine
  // sets it before it runs passes. Null where a family has no use for it.
  void (*set_other_passes)(int passes);
  // Between two calls of a pass, releases what binds the states the pass used
  // to the model's variables, so that code outside the pass may use those
  // states; or null if nothing is bound.
  void (*pass_release)(void);
} ModelOps;

typedef struct {
  const ModelOps *ops;
  const char *dir;
  // The GPU that the model runs on.
  Gpu *gpu;
  MemoryAccount memory;
  MemorySlot work;
} Model;

// The family that runs the checkpoint directory dir (its config.json's
// model_type), and the model of dir, loaded on gpu and counted in an account
// of memory. Both end the process with a message when no family runs it.
const ModelOps *model_find_family(const char *dir);
Model *model_create(Memory *memory, Gpu *gpu, const char *dir);
int model_load(Model *model, const Options *options);
void model_set_request_running(Model *model, int active);
Model *model_open(Memory *memory, Gpu *gpu, const char *dir, const Options *o);

extern const ModelOps deepseek_ops;

#endif
