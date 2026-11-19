#include "eremite.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include "base/shutdown.h"
#include "engine/draft_choice.h"
#include "engine/expert_store.h"
#include "engine/lookup.h"
#include "engine/media.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/sampling.h"
#include "engine/state_file.h"
#include "engine/tokenizer.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/deepseek/model.h"
#include "model/model.h"
#include "platform/system_memory.h"
#include "platform/threads.h"
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// The engine behind the C ABI of eremite.h: the process's log and stop, and
// for each model its loading, its work on states and snapshots, sampling,
// context-copy drafts, images, statistics and its tokenizer. Requests, their
// scheduling and the prefix cache are the server's.

struct eremite_sampler {
  Sampler sp;
};

struct eremite_model {
  Model *m;
  const ModelOps *ops;
  Tokenizer *tk;
  Options opt;
  char *dir;
  atomic_int loaded;
  // eremite_state_identity's, once known.
  uint8_t identity[32];
  int identified;
  // A decision model's turn beside decoding calls: when its decisions' latest
  // turn began, and how long it may run, which is how long the decoding calls
  // before it took. Decoding calls that have not started wait while it runs.
  double turn_start, turn_length;
};

struct eremite_media {
  eremite_model *m;
  Media *items;
  int n;
};

// The bytes in one GiB.
static constexpr double GiB = 1024.0 * 1024 * 1024;

static eremite_process proc;
static int initialized;
// The models registered, one of each family.
static eremite_model *models[8];
static int nmodels;
// The memory that every model is counted in, made by eremite_init, and the
// GPU that the models run on, made when the first model loads.
static Memory *memory;
static Gpu *gpu;
// The calls posted to streams since the process started. Each stream keeps
// the number of its call.
static uint64_t calls_posted;
// What the engine thread blocks on while no stream's call can go on, made by
// eremite_init, and whether eremite_wake asked eremite_run_streams to return.
static ThreadsEvent *wake;
static atomic_int woken;

// What the engine thread does. The GPU's idle times are split by it.
enum {
  // Outside eremite_run_streams.
  THREAD_OUTSIDE,
  // Running a pass.
  THREAD_PASS,
  // No pass can go on, and one waits for the GPU.
  THREAD_WAIT_GPU,
  // No pass can go on, none waits for the GPU, and one waits for reads.
  THREAD_WAIT_READ,
  // No pass can go on, and none waits for the GPU or for reads.
  THREAD_WAIT_OTHER,
  // The number of states.
  THREAD_STATES
};

static_assert(THREAD_STATES <= GPU_THREAD_STATES,
              "the GPU splits its idle times by every engine-thread state");

// The Memory object's eviction callback.
static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Returns the GPU, made with the kernel library at the first call.
static Gpu *models_gpu(void) {
  if (!gpu) {
    size_t length;
    const char *source = kernel_library_source(&length);
    gpu = gpu_new(source, length);
    if (proc.gpu_timeout_seconds > 0)
      gpu_set_signal_timeout(gpu, proc.gpu_timeout_seconds);
    memory_set_eviction_callback(memory, after_evictions, gpu);
  }
  return gpu;
}

// Records what the engine thread does now, once the GPU is made.
static void set_thread_state(int state) {
  if (gpu) gpu_set_thread_state(gpu, state);
}

// The state memory of every model.
static uint64_t state_bytes(void) {
  uint64_t bytes = 0;
  for (int i = 0; i < nmodels; i++) bytes += models[i]->m->memory.state_bytes;
  return bytes;
}

// The log is say(): one line per event, to the caller's on_log (log_hook) or
// stderr.
void eremite_log(const char *line) { say("%s", line); }

// ---- the process ----

int eremite_init(const eremite_process *p) {
  if (!p || p->abi != EREMITE_ABI || initialized) return -1;
  if (p->reserve_gb && !(p->reserve_gb >= EXPERT_STORE_RESERVE_GIB_MIN &&
                         p->reserve_gb <= EXPERT_STORE_RESERVE_GIB_MAX))
    return -1;
  const double settings[] = {p->gpu_timeout_seconds,
                             p->freed_memory_delay_seconds,
                             p->available_memory_reading_interval_seconds,
                             p->available_memory_reading_max_age_seconds,
                             p->keep_recently_used_weights_seconds,
                             p->free_pages_only_after_swapout_seconds,
                             p->expert_cache_growth_margin_gb,
                             p->gpu_memory_limit_margin_gb};
  for (size_t i = 0; i < sizeof settings / sizeof *settings; i++)
    if (!isfinite(settings[i]) || settings[i] < 0) return -1;
  memory = memory_new();
  expert_store_set_memory(
      memory,
      (MemorySettings){
          .model_gib = p->memory_gb,
          .reserve_gib = p->reserve_gb,
          .times = {.freed_memory_delay = p->freed_memory_delay_seconds,
                    .available_memory_reading_interval =
                        p->available_memory_reading_interval_seconds,
                    .available_memory_reading_max_age =
                        p->available_memory_reading_max_age_seconds,
                    .keep_recently_used_weights =
                        p->keep_recently_used_weights_seconds},
          .free_pages_only_after_swapout =
              p->free_pages_only_after_swapout_seconds,
          .expert_cache_growth_margin_gib = p->expert_cache_growth_margin_gb,
          .gpu_memory_limit_margin_gib = p->gpu_memory_limit_margin_gb});
  proc = *p;
  die_hook = p->on_die;
  log_hook = p->on_log;
  threads_set_priority(
      THREADS_INTERACTIVE);  // the models' GPU work goes before background work
  // SIGINT and SIGTERM request a stop, which ends a load and lets the server
  // save its states.
  shutdown_watch(p->on_stop);
  wake = threads_event_new();
  initialized = 1;
  return 0;
}

int eremite_is_stopping(void) { return shutdown_requested(); }

int eremite_count_stop_requests(void) { return shutdown_requests(); }

void eremite_request_stop(void) { shutdown_request(); }

// ---- models ----

static char *copy(const char *s) { return s ? strdup(s) : 0; }

eremite_model *eremite_open(const eremite_config *c) {
  if (!initialized || !c || c->abi != EREMITE_ABI || !c->model) return NULL;
  const ModelOps *ops = model_find_family(c->model);
  const eremite_deepseek_config *d = &c->deepseek;
  Options options = {.ctx = c->ctx,
                     .lookup = c->lookup,
                     .dump_fd = -1,
                     .routes_fd = -1,
                     .quiet = 1,
                     .draft = d->draft,
                     .deepseek.substitute = d->substitute,
                     .deepseek.experts = d->experts,
                     .deepseek.ced = d->ced,
                     .deepseek.prefill_prune = d->prefill_prune,
                     .deepseek.packed_state_rows = d->packed_state_rows};
  const char *bad =
      options.ctx < 1 || options.ctx > 1 << 20 ? "ctx takes 1 to 1048576 tokens"
      : ops->decide && (options.draft || options.lookup || d->experts ||
                        d->ced || d->substitute || d->packed_state_rows)
          ? "a decision model takes no drafts, lookup, experts, "
            "ced, substitute or packed state rows"
      : options.draft < 0 || options.draft > BLOCK
          ? "draft exceeds this model's proposal limit"
      : options.lookup < 0 || options.lookup > VERIFY_DRAFTS
          ? "lookup takes off or 1 to 15 proposals"
      : d->ced && (d->ced < WINDOW || d->ced > 1 << 20)
          ? "ced takes off or 128 to 1048576 replay tokens"
      : !isfinite(c->memory_gb) || c->memory_gb < 0
          ? "memory takes 0 (none) or a GiB limit"
          : NULL;
  if (bad) {
    say("%s: %s", c->model, bad);
    return NULL;
  }
  for (int i = 0; i < nmodels; i++)
    if (models[i]->ops == ops) {
      say("%s: a %s model is registered already", c->model, ops->family);
      return NULL;
    }
  eremite_model *x = xmalloc(sizeof *x);
  atomic_init(&x->loaded, 0);
  x->ops = ops;
  x->opt = options;
  x->opt.deepseek.experts = copy(options.deepseek.experts);
  x->tk = tokenizer_open(c->model, ops->media_token);
  x->dir = copy(c->model);
  x->m = model_create(memory, NULL, x->dir, (uint64_t)(c->memory_gb * GiB));
  models[nmodels++] = x;
  say("%s: registered %s; weights load on first inference request", c->model,
      ops->family);
  return x;
}

int eremite_is_loaded(const eremite_model *m) {
  if (!atomic_load_explicit(&m->loaded, memory_order_acquire)) return 0;
  MemorySlot *slot =
      m->ops->dense_weight_slot ? m->ops->dense_weight_slot() : NULL;
  return !slot || memory_ready(slot);
}

void eremite_set_request_running(eremite_model *m, int active) {
  model_set_request_running(m->m, active);
}

int eremite_load(eremite_model *m) {
  int was_loaded = eremite_is_loaded(m);
  m->m->gpu = models_gpu();
  if (!model_load(m->m, &m->opt)) {
    model_set_request_running(m->m, 0);
    MemoryUsage u = memory_usage(memory);
    uint64_t limit = memory_current_limit(memory);
    if (shutdown_requested())
      say("%s: loading stopped", m->dir);
    else
      say("%s: not loaded: it does not fit; %.1f GiB more model memory fits "
          "now (%.1f GiB held, %.1f GiB of state "
          "memory)",
          m->dir,
          (limit > u.committed ? limit - u.committed : 0) / 1073741824.0,
          u.committed / 1073741824.0, state_bytes() / 1073741824.0);
    return 0;
  }
  model_set_request_running(m->m, 1);
  atomic_store_explicit(&m->loaded, 1, memory_order_release);
  if (!was_loaded && m->ops->decide)
    say("%s: %s decision model, context %d", m->dir, m->ops->family,
        m->opt.ctx);
  else if (!was_loaded)
    say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d; "
        "prefill: ced_depth=%d (%s), dependency "
        "pruning=%s; packed state rows %s",
        m->dir, m->ops->family, m->opt.ctx, m->ops->drafts_per_pass(),
        m->ops->rewind_margin(), m->opt.lookup, m->opt.deepseek.ced,
        m->opt.deepseek.ced ? "approximate" : "off",
        m->opt.deepseek.prefill_prune ? "on" : "off",
        m->opt.deepseek.packed_state_rows ? "on" : "off");
  return 1;
}

int eremite_run_idle_work(eremite_model *m) {
  if (!atomic_load_explicit(&m->loaded, memory_order_acquire)) return 0;
  ModelCounters c;
  m->ops->read_counters(&c);
  int before = c.cache_slots, after = m->ops->run_idle_work();
  if (after < before)
    say("idle: memory short, expert cache %d -> %d slots", before, after);
  return after;
}

double eremite_idle_work_interval_seconds(void) {
  return memory ? memory_times(memory).available_memory_reading_interval : 0;
}

int eremite_max_rows(const eremite_model *m) {
  return m->ops->max_rows ? m->ops->max_rows() : 0;
}

int eremite_drafts_per_pass(const eremite_model *m) {
  if (!m->ops->drafts_per_pass) return 0;
  return eremite_is_loaded(m) ? m->ops->drafts_per_pass() : m->opt.draft;
}

int eremite_max_verify(const eremite_model *m) {
  return m->ops->max_verify ? m->ops->max_verify() : 0;
}

int eremite_rewind_margin(const eremite_model *m) {
  return m->ops->rewind_margin ? m->ops->rewind_margin() : 0;
}

int eremite_vocab_size(const eremite_model *m) { return m->ops->vocab_size(); }

int eremite_is_stop_token(const eremite_model *m, int32_t token) {
  return m->ops->is_stop_token && m->ops->is_stop_token(token);
}

int eremite_is_decision_model(const eremite_model *m) {
  return m->ops->decide != NULL;
}

// Runs a decision model's pass with its images, if any. A non-null wait is
// waited for in each layer.
static const float *run_decide(eremite_model *m, const int32_t *tokens, int n,
                               int state_tokens, const int32_t *question_ends,
                               int questions, const int32_t *rows,
                               const int32_t *ids, int count,
                               const eremite_media *images,
                               const PassWait *wait) {
  if (!m->ops->decide) die("%s: not a decision model", m->dir);
  for (int i = 0; images && i < images->n; i++)
    if (!images->items[i].rows)
      die("%s: a decision's images must be encoded first", m->dir);
  return m->ops->decide(tokens, n, state_tokens, question_ends, questions, rows,
                        ids, count, images ? images->items : NULL,
                        images ? images->n : 0, wait);
}

const float *eremite_decide(eremite_model *m, const int32_t *tokens, int n,
                            int state_tokens, const int32_t *question_ends,
                            int questions, const int32_t *rows,
                            const int32_t *ids, int count,
                            const eremite_media *images) {
  return run_decide(m, tokens, n, state_tokens, question_ends, questions, rows,
                    ids, count, images, NULL);
}

// Ends the program for a decision model: it keeps no states and runs no
// prefills, decoding or drafts.
static void require_states(const eremite_model *m) {
  if (m->ops->decide)
    die("%s: a decision model keeps no states and runs no prefills or "
        "decoding",
        m->dir);
}

// ---- attention states and snapshots ----

eremite_state *eremite_state_new(eremite_model *m) {
  require_states(m);
  return m->ops->state_new();
}

void eremite_state_free(eremite_model *m, eremite_state *s) {
  require_states(m);
  m->ops->state_free(s);
}

size_t eremite_state_bytes(const eremite_model *m, const eremite_state *s,
                           int positions) {
  require_states(m);
  return m->ops->state_bytes(s, positions);
}

void eremite_state_copy(eremite_model *m, eremite_state *dst,
                        const eremite_state *src, int positions) {
  require_states(m);
  m->ops->state_copy(dst, src, positions);
}

eremite_snapshot *eremite_snapshot_take(eremite_model *m, eremite_state *s,
                                        int pos, const float *logits) {
  require_states(m);
  return m->ops->snapshot_take(s, pos, logits);
}

void eremite_snapshot_restore(eremite_model *m, eremite_state *s,
                              const eremite_snapshot *x) {
  require_states(m);
  m->ops->snapshot_restore(s, x);
}

void eremite_snapshot_free(eremite_model *m, eremite_snapshot *x) {
  require_states(m);
  m->ops->snapshot_free(x);
}

size_t eremite_snapshot_bytes(const eremite_model *m, int with_logits) {
  require_states(m);
  return m->ops->snapshot_bytes(with_logits);
}

const float *eremite_snapshot_logits(const eremite_model *m,
                                     const eremite_snapshot *x) {
  require_states(m);
  return m->ops->snapshot_logits(x);
}

// ---- saved states ----

int64_t eremite_state_write(eremite_model *m, eremite_state *s, int from,
                            int to, int fd) {
  require_states(m);
  return state_file_write(m->ops, s, from, to, fd);
}

int eremite_state_read(eremite_model *m, eremite_state *s, int positions,
                       const int *fds, int count) {
  require_states(m);
  return state_file_read(m->ops, s, positions, fds, count);
}

int64_t eremite_snapshot_write(eremite_model *m, const eremite_snapshot *x,
                               eremite_state *s, int pos, int fd) {
  require_states(m);
  return snapshot_file_write(m->ops, (eremite_snapshot *)x, s, pos, fd);
}

eremite_snapshot *eremite_snapshot_read(eremite_model *m, int fd) {
  require_states(m);
  return snapshot_file_read(m->ops, fd);
}

static void identity_number(CC_SHA256_CTX *h, int64_t v) {
  CC_SHA256_Update(h, &v, sizeof v);
}

void eremite_state_identity(eremite_model *m, uint8_t out[32]) {
  if (!m->identified) {
    CC_SHA256_CTX h;
    CC_SHA256_Init(&h);
    static const char tag[] = "eremite saved state";
    CC_SHA256_Update(&h, tag, sizeof tag);
    CC_SHA256_Update(&h, m->ops->family, (CC_LONG)strlen(m->ops->family) + 1);
    identity_number(&h, m->ops->state_version);
    identity_add_folder(&h, m->dir);
    identity_number(&h, m->opt.draft);
    identity_number(&h, m->opt.lookup);
    const DeepSeekOptions *d = &m->opt.deepseek;
    uint32_t substitute;
    memcpy(&substitute, &d->substitute, sizeof substitute);
    identity_number(&h, d->ced);
    identity_number(&h, d->prefill_prune);
    identity_number(&h, d->packed_state_rows != 0);
    identity_number(&h, substitute);
    identity_number(&h, d->experts != NULL);
    if (d->experts) identity_add_gguf_set(&h, d->experts);
    CC_SHA256_Final(m->identity, &h);
    m->identified = 1;
  }
  memcpy(out, m->identity, 32);
}

// ---- running a model ----

// Prefills positions from to n - 1 of the prompt in s, and returns the last
// one's logits. A non-null wait is waited for in each layer.
static const float *
run_prefill(eremite_model *m, eremite_state *s, const int32_t *tokens, int n,
            int from, int replay, int capture, eremite_snapshot **captured,
            eremite_prefill_info *info, const eremite_media *images,
            const PassWait *wait) {
  if (captured) *captured = 0;
  int start = replay && from > 0 ? m->ops->replay_start(from) : from;
  if (m->ops->set_prefill_media)
    m->ops->set_prefill_media(images ? images->items : NULL,
                              images ? images->n : 0);
  const float *lg =
      m->ops->prefill(s, tokens, n, from, replay, capture, captured, 0, wait);
  if (info) {  // the family's plan, or every row through every layer
    ModelPlan p = {
        .chunks = 1, .tokens = n - start, .decoder_tokens = n - start};
    if (m->ops->read_last_plan) m->ops->read_last_plan(&p);
    *info = (eremite_prefill_info){.chunks = p.chunks,
                                   .tokens = p.tokens,
                                   .decoder_tokens = p.decoder_tokens,
                                   .ced_applied = p.ced_applied,
                                   .prune_applied = p.prune_applied,
                                   .prefix_tokens = p.prefix_tokens};
  }
  return lg;
}

const float *eremite_prefill(eremite_model *m, eremite_state *s,
                             const int32_t *tokens, int n, int from, int replay,
                             int capture, eremite_snapshot **captured,
                             eremite_prefill_info *info,
                             const eremite_media *images) {
  require_states(m);
  return run_prefill(m, s, tokens, n, from, replay, capture, captured, info,
                     images, NULL);
}

// ---- streams ----

// The most rows of a stream's decoding call.
static constexpr int STREAM_ROWS = 16;

typedef enum {
  STREAM_NONE,
  STREAM_PREFILL,
  STREAM_DECODE,
  STREAM_VERIFY,
  STREAM_DECIDE,
  STREAM_QUIT
} StreamCall;

// The streams that one eremite_run_streams call advances.
typedef struct {
  eremite_stream *const *streams;
  int count;
} StreamRun;

struct eremite_stream {
  eremite_model *m;
  Pass *pass;
  // The call posted and not finished, and its number among all posted calls.
  StreamCall call;
  uint64_t call_number;
  // Its pass has started the posted call.
  int started;
  // It finished, and eremite_run_streams has not reported that yet.
  int finished;
  // the call's arguments
  eremite_state *states[STREAM_ROWS];
  int32_t row_tokens[STREAM_ROWS], row_pos[STREAM_ROWS];
  // A prefill's prompt; a verify's rows (the token, then the drafts); a
  // decision's sequence.
  const int32_t *tokens;
  int n, from, replay, capture, count, drafts, proposed, pos;
  // A decision's state rows, its questions' ends, and the rows and token ids
  // of the logits it returns (count of them).
  int state_tokens, questions;
  const int32_t *question_ends, *logit_rows, *logit_ids;
  const eremite_media *images;
  eremite_sampler *sampler;
  // its results
  const float *logits;
  eremite_snapshot *captured;
  eremite_prefill_info info;
  int32_t next;
  int accepted;
  // What the stream's prefill waits for in each layer: the decoding calls in
  // its run that were posted at the wait's first look. The number of the last
  // call posted then, or 0 between waits.
  PassWait wait_for_decoding;
  uint64_t calls_at_first_look;
  // What the stream's decision waits for in each layer: its turn
  // (decision_may_go_on). The number of the last command buffer committed at
  // the wait's first look, and when it was seen finished, or 0 between waits.
  PassWait wait_for_turn;
  uint64_t work_at_first_look;
  double work_finished_at;
  // The run that advances the stream, while one does.
  const StreamRun *run;
};

static int stream_has_call(void *context) {
  return ((eremite_stream *)context)->call != STREAM_NONE;
}

// Whether the stream's posted call is a decoding call: a decode or a verify.
static int has_decoding_call(const eremite_stream *x) {
  return x->call == STREAM_DECODE || x->call == STREAM_VERIFY;
}

// Whether decoding calls in the stream's run that were posted at its wait's
// first look have not finished. The first look is the wait's first question.
static int decoding_calls_remain(eremite_stream *x) {
  if (!x->run) die("a stream's pass waited outside eremite_run_streams");
  if (!x->calls_at_first_look) x->calls_at_first_look = calls_posted;
  for (int k = 0; k < x->run->count; k++) {
    const eremite_stream *other = x->run->streams[k];
    if (has_decoding_call(other) &&
        other->call_number <= x->calls_at_first_look)
      return 1;
  }
  return 0;
}

// Whether the decoding calls that the stream's prefill waits for have
// finished. The prefill waits for those in its run that were posted at the
// wait's first look.
static int decoding_calls_finished(void *context) {
  eremite_stream *x = context;
  if (decoding_calls_remain(x)) return 0;
  x->calls_at_first_look = 0;
  return 1;
}

// Ends the stream's decision's wait in a layer.
static void end_turn_wait(eremite_stream *x) {
  x->calls_at_first_look = 0;
  x->work_at_first_look = 0;
  x->work_finished_at = 0;
}

// Whether the stream's decision may run its next layer. Without decoding calls
// in its run posted at the wait's first look, it goes on at once. Beside them,
// its GPU work committed by then finishes first, so that its turn counts the
// GPU's time. Its turn goes on while it has run for less time than the
// decoding calls before the turn took. Then those calls go first, and the next
// turn may run as long as they took after the decision's GPU work finished.
static int decision_may_go_on(void *context) {
  eremite_stream *x = context;
  eremite_model *m = x->m;
  if (!x->calls_at_first_look) {
    x->work_at_first_look = gpu_last_committed(gpu);
    if (!decoding_calls_remain(x)) return end_turn_wait(x), 1;
  }
  if (!x->work_finished_at) {
    if (x->work_at_first_look && !gpu_has_finished(gpu, x->work_at_first_look))
      return 0;
    double at = now();
    if (at - m->turn_start < m->turn_length) return end_turn_wait(x), 1;
    m->turn_length = 0;
    x->work_finished_at = at;
  }
  if (decoding_calls_remain(x)) return 0;
  m->turn_start = now();
  m->turn_length = m->turn_start - x->work_finished_at;
  return end_turn_wait(x), 1;
}

// Waits until the GPU work that the stream's decision committed by its wait's
// first look has finished. Its pass blocks the engine thread only as the one
// pass in its run that may go on: in its turn, while the decoding calls that
// have not started wait.
static void wait_for_decision_work(void *context) {
  eremite_stream *x = context;
  if (x->work_at_first_look)
    gpu_wait_until_finished(gpu, x->work_at_first_look);
}

// Whether a decision's turn runs among the streams.
static int decision_turn_runs(eremite_stream *const *streams, int n) {
  for (int k = 0; k < n; k++) {
    const eremite_model *m = streams[k]->m;
    if (streams[k]->call == STREAM_DECIDE && m->turn_length > 0 &&
        now() - m->turn_start < m->turn_length)
      return 1;
  }
  return 0;
}

// Whether the stream's call waits while a decision's turn runs among the
// streams: a decoding call that has not started.
static int waits_for_turn(eremite_stream *x, eremite_stream *const *streams,
                          int n) {
  return has_decoding_call(x) && !x->started && decision_turn_runs(streams, n);
}

static void stream_run(eremite_stream *x) {
  eremite_model *m = x->m;
  x->logits = NULL;
  switch (x->call) {
  case STREAM_PREFILL:
    x->logits = run_prefill(m, x->states[0], x->tokens, x->n, x->from,
                            x->replay, x->capture, &x->captured, &x->info,
                            x->images, &x->wait_for_decoding);
    break;
  case STREAM_DECODE:
    x->logits =
        eremite_decode_rows(m, x->states, x->row_tokens, x->row_pos, x->count);
    break;
  case STREAM_DECIDE:
    x->logits = run_decide(
        m, x->tokens, x->n, x->state_tokens, x->question_ends, x->questions,
        x->logit_rows, x->logit_ids, x->count, x->images, &x->wait_for_turn);
    break;
  case STREAM_VERIFY: {
    int32_t *rows = (int32_t *)x->tokens;
    const double *q =
        x->proposed ? NULL
                    : eremite_propose_drafts(m, x->states[0], rows[0], x->pos,
                                             rows + 1, x->drafts, x->sampler);
    x->logits = eremite_verify(m, x->states[0], rows, x->drafts + 1, x->pos);
    if (x->logits) {
      x->next = eremite_sample_verify_drafts(m, x->logits, rows + 1, x->drafts,
                                             q, x->sampler, &x->accepted);
      eremite_accept(m, x->states[0], rows, x->accepted + 1, x->pos);
    }
    break;
  }
  default: break;
  }
}

// The stream's pass: a call after another, with its own buffers; between
// calls no state stays bound to it.
static void stream_body(void *argument) {
  eremite_stream *x = argument;
  x->m->ops->pass_begin();
  for (;;) {
    pass_wait(&(PassWait){stream_has_call, NULL, x, PASS_WAIT_OTHER, NULL});
    if (x->call == STREAM_QUIT) break;
    x->started = 1;
    stream_run(x);
    if (x->m->ops->pass_release) x->m->ops->pass_release();
    x->call = STREAM_NONE;
    x->started = 0;
    x->finished = 1;
  }
  x->m->ops->pass_end();
}

eremite_stream *eremite_stream_new(eremite_model *m) {
  eremite_stream *x = xmalloc(sizeof *x);
  x->m = m;
  x->wait_for_decoding =
      (PassWait){decoding_calls_finished, NULL, x, PASS_WAIT_OTHER, NULL};
  x->wait_for_turn = (PassWait){decision_may_go_on, wait_for_decision_work, x,
                                PASS_WAIT_GPU, NULL};
  x->pass = m->ops->pass_new(stream_body, x);
  return x;
}

void eremite_stream_free(eremite_stream *x) {
  if (!x) return;
  x->call = STREAM_QUIT;
  pass_run(x->pass);
  pass_free(x->pass);
  free(x);
}

// Posts the call whose arguments the stream holds, with the next call number.
static void post_call(eremite_stream *x, StreamCall call) {
  x->call = call;
  x->call_number = ++calls_posted;
}

void eremite_stream_prefill(eremite_stream *x, eremite_state *s,
                            const int32_t *tokens, int n, int from, int replay,
                            int capture, const eremite_media *images) {
  require_states(x->m);
  x->states[0] = s, x->tokens = tokens, x->n = n, x->from = from,
  x->replay = replay, x->capture = capture, x->images = images;
  x->captured = NULL;
  post_call(x, STREAM_PREFILL);
}

void eremite_stream_decode_rows(eremite_stream *x, eremite_state **states,
                                const int32_t *tokens, const int32_t *pos,
                                int count) {
  require_states(x->m);
  if (count < 1 || count > STREAM_ROWS)
    die("a stream's decoding call takes 1 to %d rows", STREAM_ROWS);
  memcpy(x->states, states, (size_t)count * sizeof *states);
  memcpy(x->row_tokens, tokens, (size_t)count * sizeof *tokens);
  memcpy(x->row_pos, pos, (size_t)count * sizeof *pos);
  x->count = count;
  post_call(x, STREAM_DECODE);
}

void eremite_stream_verify(eremite_stream *x, eremite_state *s, int32_t *rows,
                           int drafts, int proposed, int pos,
                           eremite_sampler *sampler) {
  require_states(x->m);
  x->states[0] = s, x->tokens = rows, x->drafts = drafts,
  x->proposed = proposed, x->pos = pos, x->sampler = sampler;
  x->accepted = 0;
  post_call(x, STREAM_VERIFY);
}

void eremite_stream_decide(eremite_stream *x, const int32_t *tokens, int n,
                           int state_tokens, const int32_t *question_ends,
                           int questions, const int32_t *rows,
                           const int32_t *ids, int count,
                           const eremite_media *images) {
  if (!x->m->ops->decide) die("%s: not a decision model", x->m->dir);
  x->tokens = tokens, x->n = n, x->state_tokens = state_tokens;
  x->question_ends = question_ends, x->questions = questions;
  x->logit_rows = rows, x->logit_ids = ids, x->count = count;
  x->images = images;
  post_call(x, STREAM_DECIDE);
}

// What the streams with a call wait for while none can go on (THREAD_*): the
// GPU if any waits for it, else reads if any waits for them, else something
// else.
static int streams_wait_for(eremite_stream *const *streams, int n) {
  int reads = 0;
  for (int k = 0; k < n; k++) {
    if (streams[k]->call == STREAM_NONE) continue;
    int kind = pass_waiting_for(streams[k]->pass);
    if (kind == PASS_WAIT_GPU) return THREAD_WAIT_GPU;
    reads |= kind == PASS_WAIT_READ;
  }
  return reads ? THREAD_WAIT_READ : THREAD_WAIT_OTHER;
}

// Sets the run that advances each of the streams, or null.
static void set_stream_run(eremite_stream *const *streams, int n,
                           const StreamRun *run) {
  for (int k = 0; k < n; k++) streams[k]->run = run;
}

// The streams' calls advance on this thread: each stream that can go on runs
// until it would wait. When none can go on, the thread waits for any of them.
int eremite_run_streams(eremite_stream *const *streams, int n) {
  static unsigned first;
  int posted = 0;
  for (int k = 0; k < n; k++) posted += streams[k]->call != STREAM_NONE;
  for (int i = 0; i < nmodels; i++)
    if (models[i]->ops->set_other_passes)
      models[i]->ops->set_other_passes(posted - 1);
  const StreamRun run = {streams, n};
  set_stream_run(streams, n, &run);
  Pass *waiting[n];
  int finished = -1;
  for (;;) {
    int ran = 0;
    for (int k = 0; k < n && finished < 0; k++) {
      int i = (int)(((unsigned)first + (unsigned)k) % (unsigned)n);
      eremite_stream *x = streams[i];
      if (waits_for_turn(x, streams, n)) continue;
      if (pass_ready(x->pass))
        set_thread_state(THREAD_PASS), pass_resume(x->pass), ran = 1;
      if (x->finished) x->finished = 0, first = (unsigned)i + 1, finished = i;
    }
    if (finished >= 0 || atomic_exchange(&woken, 0)) break;
    if (ran) continue;
    int count = 0;
    for (int k = 0; k < n; k++)
      if (streams[k]->call != STREAM_NONE &&
          !waits_for_turn(streams[k], streams, n))
        waiting[count++] = streams[k]->pass;
    if (!count) break;
    set_thread_state(streams_wait_for(streams, n));
    if (!pass_wait_for_any(waiting, count, wake))
      die("the streams' calls wait for each other: none can go on");
  }
  set_thread_state(THREAD_OUTSIDE);
  set_stream_run(streams, n, NULL);
  return finished;
}

void eremite_wake(void) {
  atomic_store(&woken, 1);
  if (wake) threads_event_signal(wake);
}

const float *eremite_stream_logits(const eremite_stream *x) {
  return x->logits;
}

eremite_snapshot *eremite_stream_captured_snapshot(const eremite_stream *x) {
  return x->captured;
}

void eremite_stream_prefill_info(const eremite_stream *x,
                                 eremite_prefill_info *info) {
  *info = x->info;
}

int32_t eremite_stream_next_token(const eremite_stream *x, int *accepted) {
  if (accepted) *accepted = x->accepted;
  return x->next;
}

int eremite_replay_rows(const eremite_model *m, int k) {
  require_states(m);
  return m->ops->replay_rows(k);
}

int eremite_replay_start(const eremite_model *m, int k) {
  require_states(m);
  return m->ops->replay_start(k);
}

const float *eremite_decode_rows(eremite_model *m, eremite_state **states,
                                 const int32_t *tokens, const int32_t *pos,
                                 int count) {
  require_states(m);
  return m->ops->decode_rows(states, tokens, pos, count);
}

const double *eremite_propose_drafts(eremite_model *m, eremite_state *s,
                                     int32_t token, int pos, int32_t *drafts,
                                     int n, eremite_sampler *sampler) {
  require_states(m);
  return m->ops->propose_drafts(s, token, pos, drafts, n, &sampler->sp);
}

const float *eremite_verify(eremite_model *m, eremite_state *s,
                            const int32_t *rows, int count, int pos) {
  require_states(m);
  return m->ops->verify(s, rows, count, pos);
}

void eremite_accept(eremite_model *m, eremite_state *s, const int32_t *rows,
                    int n, int pos) {
  require_states(m);
  m->ops->accept(s, rows, n, pos);
}

void eremite_set_history(eremite_model *m, eremite_state *s,
                         const int32_t *tokens, int pos) {
  require_states(m);
  m->ops->set_history(s, tokens, pos);
}

// ---- sampling ----

eremite_sampler *eremite_sampler_new(double temperature, double top_p,
                                     int top_k, uint64_t seed, int seed_set) {
  Sampler sp = {.temperature = temperature, .top_p = top_p, .top_k = top_k};
  if (!sampler_is_valid(&sp)) return 0;
  eremite_sampler *s = xmalloc(sizeof *s);
  s->sp = sp;
  if (seed_set)
    sampler_set_seed(&s->sp, seed);
  else
    arc4random_buf(s->sp.seed, sizeof s->sp.seed);
  return s;
}

void eremite_sampler_free(eremite_sampler *s) { free(s); }

int32_t eremite_sample(const eremite_model *m, const float *logits,
                       eremite_sampler *s) {
  return sample_token(logits, m->ops->vocab_size(), &s->sp);
}

int32_t eremite_sample_verify_drafts(const eremite_model *m,
                                     const float *logits, const int32_t *drafts,
                                     int n, const double *q, eremite_sampler *s,
                                     int *accepted) {
  return sample_verify_sampled_drafts(logits, m->ops->vocab_size(), drafts, n,
                                      q, m->ops->is_stop_token, &s->sp,
                                      accepted);
}

// ---- context-copy drafts ----

eremite_lookup *eremite_lookup_new(int capacity) {
  return context_lookup_new(capacity);
}

void eremite_lookup_free(eremite_lookup *x) { context_lookup_free(x); }

int eremite_lookup_append(eremite_lookup *x, const int32_t *tokens, int n) {
  return context_lookup_append(x, tokens, n);
}

int eremite_lookup_propose(const eremite_lookup *x, int max, int32_t *drafts) {
  ContextDraft d = context_lookup_propose(x, max);
  memcpy(drafts, d.tokens, (size_t)d.count * sizeof *drafts);
  return d.count;
}

void eremite_lookup_record_verification(eremite_lookup *x, int offered,
                                        int accepted) {
  context_lookup_record_verification(x, offered, accepted);
}

eremite_draft_choice *eremite_draft_choice_new(void) {
  eremite_draft_choice *x = xmalloc(sizeof *x);
  draft_choice_init(x);
  return x;
}

void eremite_draft_choice_free(eremite_draft_choice *x) { free(x); }

int eremite_draft_choice_next_count(eremite_draft_choice *x, int most) {
  return draft_choice_next_count(x, most);
}

void eremite_draft_choice_record_pass(eremite_draft_choice *x, int count,
                                      int kept, double seconds) {
  draft_choice_record_pass(x, count, kept, seconds);
}

// ---- images ----

eremite_media *eremite_media_open(eremite_model *m, const uint8_t *const *data,
                                  const size_t *len, int n, char *message,
                                  size_t cap) {
  if (!m->ops->encode_media || !m->ops->prepare_media) {
    snprintf(message, cap, "this model has no media encoder");
    return 0;
  }
  eremite_media *x = xmalloc(sizeof *x);
  x->m = m;
  x->items = xmalloc((size_t)n * sizeof *x->items), x->n = 0;
  for (int i = 0; i < n; i++) {
    const char *why;
    if (!m->ops->prepare_media(m->dir, &x->items[i], data[i], len[i], &why)) {
      snprintf(message, cap, "media %d: %s", i + 1, why);
      eremite_media_close(x);
      return 0;
    }
    x->n = i + 1;
  }
  return x;
}

int eremite_media_expand(eremite_media *x, const int32_t *in, int n,
                         int32_t *out, int max) {
  int token = tokenizer_require_added(x->m->tk, x->m->ops->media_token);
  return media_expand_prompt(in, n, x->items, x->n, out, max, token);
}

int eremite_media_encode(eremite_media *x, int from) {
  int encoded = 0;  // the encoder runs for the images the prefill reaches; the
                    // others are in the state's rows
  for (int i = 0; i < x->n; i++)
    if (x->items[i].start + media_span_length(&x->items[i]) > from) {
      if (!x->m->ops->encode_media(&x->items[i])) return -1;
      encoded++;
    }
  return encoded;
}

void eremite_media_close(eremite_media *x) {
  if (!x) return;
  if (x->m->ops->set_prefill_media) x->m->ops->set_prefill_media(0, 0);
  for (int i = 0; i < x->n; i++) media_free(&x->items[i]);
  free(x->items), free(x);
}

// ---- statistics ----

static eremite_cache_counts counts(const ExpertStoreCounts *c) {
  return (eremite_cache_counts){.hits = c->hits,
                                .misses = c->misses,
                                .prefetch_hits = c->prefetch_hits,
                                .prefetches = c->prefetches,
                                .bytes = c->bytes,
                                .evictions = c->evictions,
                                .late = c->late,
                                .cancelled = c->cancelled,
                                .substituted = c->substituted,
                                .remaps = c->remaps,
                                .given_back = c->given_back,
                                .row_own = c->row_own,
                                .row_ahead = c->row_ahead,
                                .row_late = c->row_late,
                                .row_sub = c->row_sub,
                                .row_ssd = c->row_ssd,
                                .read_s = c->read_s};
}

int eremite_read_routes(const eremite_model *m, int row, uint16_t *experts,
                        uint8_t *sources) {
  return m->ops->read_routes ? m->ops->read_routes(row, experts, sources) : 0;
}

static eremite_stats process_stats;

void eremite_refresh_stats(void) {
  SystemMemoryInfo mi;
  system_memory_read_info(&mi);
  uint64_t available = 0;
  if (memory) memory_read_system(memory, &available);
  uint64_t target =
      gpu ? gpu_recommended_working_set(gpu) : 0;  // 0 before the GPU is made
  process_stats =
      (eremite_stats){.footprint = system_memory_footprint(),
                      .gpu_in_use = system_memory_gpu_driver(),
                      .available = available,
                      .gpu_limit = target ? (int64_t)target : INT64_MAX,
                      .total = mi.total,
                      .free = mi.free,
                      .file = mi.file,
                      .file_kept = memory ? memory_kept_file_cache(memory) : 0,
                      .wired = mi.wired,
                      .compressor = mi.compressor,
                      .compressed = mi.compressed,
                      .swap_used = mi.swap_used,
                      .swapouts = mi.swapouts,
                      .pressure = mi.pressure};
}

uint64_t eremite_reclaim(uint64_t bytes) {
  return memory_reclaim(memory, bytes);
}

uint64_t eremite_unload(eremite_model *m) {
  return memory_unload(&m->m->memory);
}

eremite_memory_info eremite_read_memory(void) {
  MemoryUsage usage = memory_usage(memory);
  return (eremite_memory_info){usage.limit, usage.allocated, state_bytes()};
}

void eremite_set_memory_limit(uint64_t bytes) {
  memory_set_limit(memory, bytes);
}

void eremite_drain(void) {
  if (gpu) gpu_sync(gpu), gpu_commit_residency(gpu);
  expert_store_wait_for_all_reads();
}

void eremite_read_stats(const eremite_model *m, eremite_stats *s) {
  ModelCounters c = {0};
  if (atomic_load_explicit(&m->loaded, memory_order_acquire))
    m->ops->read_counters(&c);
  *s = process_stats;
  eremite_stats model = {.decode = counts(&c.decode),
                         .prefill = counts(&c.prefill),
                         .decode_rows = c.decode_rows,
                         .prefill_rows = c.prefill_rows,
                         .decode_seconds = c.decode_seconds,
                         .decode_gpu_seconds = c.decode_gpu_seconds,
                         .prefill_layer = c.prefill_layer,
                         .prefill_rows_now = c.prefill_rows_now,
                         .dense_bytes = c.dense_bytes,
                         .layers = c.layers,
                         .experts_per_row = c.experts_per_row,
                         .routed_layers = c.routed_layers,
                         .experts_per_layer = c.experts_per_layer};
  s->decode = model.decode;
  s->prefill = model.prefill;
  s->decode_rows = model.decode_rows;
  s->prefill_rows = model.prefill_rows;
  s->decode_seconds = model.decode_seconds;
  s->decode_gpu_seconds = model.decode_gpu_seconds;
  s->prefill_layer = model.prefill_layer;
  s->prefill_rows_now = model.prefill_rows_now;
  s->dense_bytes = model.dense_bytes;
  s->layers = model.layers;
  s->experts_per_row = model.experts_per_row;
  s->routed_layers = model.routed_layers;
  s->experts_per_layer = model.experts_per_layer;
  MemoryUsage usage = memory_usage(memory);
  s->model_memory_limit = memory_current_limit(memory);
  s->model_memory_used = usage.allocated;
  s->model_memory_committed = usage.committed;
  s->model_memory_peak = usage.peak;
  s->state_allocated = state_bytes();
  s->model_workspace = m->m->memory.bytes - c.dense_bytes - c.cache_bytes;
  const MemoryAccount *own = &m->m->memory;
  s->own_memory_limit = own->limit == UINT64_MAX ? 0 : own->limit;
  s->own_memory_counted = memory_counted(own);
  s->own_state_bytes = own->state_bytes;
  s->required_bytes = c.required_bytes;
  s->cache = counts(&c.cache);
  s->slots = c.cache_slots;
  s->mapped = c.mapped;
  s->cap = c.cap;
  s->max_slots = c.max_slots;
  s->experts = c.experts;
  s->slot_bytes = c.slot_bytes;
  s->cache_bytes = c.cache_bytes;
  s->uncached_bytes = c.uncached_bytes;
  s->uncached = c.uncached;
  _Static_assert(THREAD_STATES == sizeof s->thread_s / sizeof *s->thread_s,
                 "one time per engine-thread state");
  GpuTimes times = gpu ? gpu_times(gpu) : (GpuTimes){0};
  for (int k = 0; k < THREAD_STATES; k++)
    s->thread_s[k] = times.thread_seconds[k],
    s->gpu_idle_s[k] = times.idle_by_state[k];
  s->gpu_busy_s = times.busy_seconds;
  expert_store_read_times(&s->ssd_busy_s, &s->demand_read_s,
                          &s->background_read_s);
}

// ---- tokenizers, from any thread ----

eremite_tokenizer *eremite_model_tokenizer(const eremite_model *m) {
  return m->tk;
}

eremite_tokenizer *eremite_tokenizer_open(const char *dir, const char *image) {
  return tokenizer_open(dir, image);
}

int eremite_tokenize(eremite_tokenizer *t, const char *text, size_t len,
                     int mode, int32_t *out, int max) {
  if (mode < TOKENIZE_TEXT || mode > TOKENIZE_PROMPT) return -1;
  int *p = xmalloc((len + 2) * sizeof *p);  // a token takes at least one byte
  int n =
      tokenizer_encode_bytes(t, text, len, (TokenizeMode)mode, p, (int)len + 2);
  if (n <= max) memcpy(out, p, (size_t)n * sizeof *p);
  free(p);
  return n <= max ? n : -1;
}

const char *eremite_token_bytes(const eremite_tokenizer *t, int32_t id,
                                size_t *len) {
  int n;
  const char *s = tokenizer_token_bytes(t, id, &n);
  *len = (size_t)n;
  return s;
}

int eremite_token_kind(const eremite_tokenizer *t, int32_t id) {
  return tokenizer_token_kind(t, id);
}

int32_t eremite_find_added_token(const eremite_tokenizer *t, const char *text,
                                 size_t len) {
  return tokenizer_find_added(t, text, len);
}
