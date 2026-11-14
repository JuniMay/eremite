#ifndef EREMITE_H
#define EREMITE_H
#include <stddef.h>
#include <stdint.h>

// eremite's inference engine behind a C ABI, for a server. The engine runs the
// models of every checkpoint that the model interface runs (DeepSeek V4.1
// Flash), several of them in one process, at most one of each family. For
// each model: loading, sampling, context-copy drafts, the draft-count chooser,
// the identity of its saved states, statistics, and its tokenizer; for the
// process: the log and the stop on SIGINT or SIGTERM. The server owns
// everything about requests: their queue and scheduling, the prefix cache
// (which states and snapshots to keep, and how a prompt reuses them), HTTP,
// and the status it builds from eremite_read_stats.
//
// Threads: eremite_init, eremite_load and every function on models, samplers
// and context lookups run on one thread, the server's engine thread: the
// engine, the models and the GPU keep their state without locks, and one
// command queue carries every model's passes. The tokenizer's functions,
// eremite_log, eremite_is_stopping and eremite_request_stop may be called from
// any thread. eremite_read_stats runs on the engine thread between passes; the
// caller publishes a copy for its HTTP handlers and terminal.

#define EREMITE_ABI 1

// ---- the process ----

typedef struct {
  // EREMITE_ABI.
  int abi;
  // Upper limit of model memory, state memory included; 0: none (the free
  // memory decides).
  double memory_gb;
  // Memory kept available to other programs; 0: default 10 GiB.
  double reserve_gb;
  // Seconds that a wait for the GPU may last before the program ends: only a
  // failed command buffer or a stopped GPU takes that long. 0: default 60.
  double gpu_timeout_seconds;
  // How the memory follows the system's counts, and how the expert caches
  // give memory back and grow; 0 takes each default. Seconds that freed memory
  // takes to show in the system's memory counts: no reading within this time
  // of a release, a cache's steps of giving back at most once per this time,
  // and a load's waits for memory that comes back (default 1.5).
  double freed_memory_delay_seconds;
  // Seconds between two readings of the system's available memory (default
  // 1); a request of 1 GiB or more reads at once. A cache's pool grows at most
  // once per this time.
  double available_memory_reading_interval_seconds;
  // Seconds after which a reading is taken again, even right after memory was
  // freed (default 5).
  double available_memory_reading_max_age_seconds;
  // Seconds that used weights count as recently used: a growing expert cache
  // does not evict them (default 30).
  double keep_recently_used_weights_seconds;
  // Seconds after macOS swaps memory out during which an expert cache grows
  // only into free memory (default 300).
  double free_pages_only_after_swapout_seconds;
  // GiB that must stay free above the reserve for an expert cache to grow
  // (default 4).
  double expert_cache_growth_margin_gb;
  // GiB kept unused below the GPU's memory limit, for slots that the GPU
  // driver still counts after they leave its list (default 3).
  double gpu_memory_limit_margin_gb;
  // The log's lines, each with its time, go to on_log instead of stderr when
  // it is set. It runs on the thread that logs, one line at a time under the
  // log's lock: it must return quickly and must not call eremite_log.
  void (*on_log)(const char *line);
  // Runs before a fatal error's message on stderr and the process's exit
  // (the live view restores the terminal), on the failing thread; or null.
  void (*on_die)(void);
  // Runs after each stop request, from SIGINT, SIGTERM or
  // eremite_request_stop, on a thread of its own; or null. The server wakes
  // its waiting threads there.
  void (*on_stop)(void);
} eremite_process;

// Installs the log, the stop's signal handlers and the failure hook, on the
// calling thread, which from then on is the engine's: it needs a stack of at
// least 16 MiB, and it gets the user-interactive QoS class, so that the
// models' GPU work goes before background work. 0, or -1 when called twice,
// with another ABI or with a setting it refuses.
int eremite_init(const eremite_process *p);

// Whether a stop was requested (SIGINT, SIGTERM or eremite_request_stop), and
// a request for one. Each request counts: eremite_count_stop_requests says how
// many there were (the server stops saving states at quit at the second).
int eremite_is_stopping(void);
void eremite_request_stop(void);
int eremite_count_stop_requests(void);

// A line for the log, with the time: on stderr, or to on_log.
void eremite_log(const char *line);

// ---- models ----

typedef struct {
  // GGUF file or shard; null uses checkpoint experts.
  const char *experts;
  float substitute;
  // DSpark proposals, 0 to 5.
  int draft;
  int ced;
  int prefill_prune;
} eremite_deepseek_config;

// Common settings and DeepSeek's options.
typedef struct {
  int abi;
  const char *model;
  int ctx;
  int lookup;
  eremite_deepseek_config deepseek;
} eremite_config;

typedef struct eremite_model eremite_model;
typedef struct eremite_media eremite_media;
typedef struct eremite_sampler eremite_sampler;

// Registers a checkpoint and opens its tokenizer, without loading weights.
// Returns null for invalid settings, or for a second model of the same
// family.
eremite_model *eremite_open(const eremite_config *c);
// Loads the registered model on the engine thread, and loads its weights again
// after an eviction. Returns whether it loaded.
int eremite_load(eremite_model *m);
// Returns whether the model is loaded and its weights are in memory.
int eremite_is_loaded(const eremite_model *m);
// Sets whether the model runs admitted requests. While it does, its weights
// stay in memory.
void eremite_set_request_running(eremite_model *m, int active);

// Runs once per available_memory_reading_interval_seconds while the model runs
// no request, and gives memory back to other programs if they need it. Returns
// the model's expert cache slots after that, or 0 without a cache.
int eremite_run_idle_work(eremite_model *m);
// Returns the seconds between two calls of eremite_run_idle_work: the
// available-memory reading interval in use after eremite_init, or 0 before it.
double eremite_idle_work_interval_seconds(void);
// Gives back every idle part of the model's memory while it runs no request:
// its expert cache before its weights. Its states stay, and the next
// eremite_load reads the weights again. Returns the bytes given back.
uint64_t eremite_unload(eremite_model *m);
// Evicts idle memory until bytes are freed or none is left. Returns the bytes
// freed.
uint64_t eremite_reclaim(uint64_t bytes);

// Process controls run on the engine thread, between completed model calls.
// eremite_read_memory: the memory limit, the bytes counted for the models
// (their state memory included), and their state memory's bytes.
typedef struct {
  uint64_t limit, model_bytes, state_bytes;
} eremite_memory_info;

eremite_memory_info eremite_read_memory(void);
// Sets the memory limit in bytes.
void eremite_set_memory_limit(uint64_t bytes);
// Waits for the GPU's work and the expert reads, at a boundary of the engine
// thread.
void eremite_drain(void);

// The model's limits: rows of different states that eremite_decode_rows takes
// at once, the drafts per pass it makes itself (the selected model options'
// draft field, at most its draft head's), the most rows one verify pass takes,
// the positions a state can continue before its end (the rewind margin), and
// the vocabulary. And whether a token ends a reply.
int eremite_max_rows(const eremite_model *m);
int eremite_drafts_per_pass(const eremite_model *m);
int eremite_max_verify(const eremite_model *m);
int eremite_rewind_margin(const eremite_model *m);
int eremite_vocab_size(const eremite_model *m);
int eremite_is_stop_token(const eremite_model *m, int32_t token);

// ---- saved states ----

// Writes the identity of m's saved states: what they depend on. It is known
// before the weights load. It covers the family's state layout, the
// checkpoint's weights, the experts' GGUF set, and the options that change a
// state's rows or shapes: draft, lookup, ced, prefill_prune and substitute.
// The weights are the folder's config.json, .safetensors and .gguf files, by
// name, size, and first and last 64 KiB. A state saved with another identity
// is not read.
void eremite_state_identity(eremite_model *m, uint8_t out[32]);

// ---- sampling ----

// Returns a sampler, or null for invalid settings. temperature must be finite
// and not negative, and 0 is greedy. top_p must be in [0, 1]. top_k must not
// be negative, and 0 takes every token. Without seed_set, the seed is random.
eremite_sampler *eremite_sampler_new(double temperature, double top_p,
                                     int top_k, uint64_t seed, int seed_set);
void eremite_sampler_free(eremite_sampler *s);
// Returns a token sampled from the logits.
int32_t eremite_sample(const eremite_model *m, const float *logits,
                       eremite_sampler *s);
// Samples the rows of logits of the token and its n drafts. Returns the next
// token, and sets *accepted to how many drafts it kept. A token that ends a
// reply ends the kept drafts. q is eremite_propose_drafts's result, or null for
// context-copy proposals, which are not sampled.
int32_t eremite_sample_verify_drafts(const eremite_model *m,
                                     const float *logits, const int32_t *drafts,
                                     int n, const double *q, eremite_sampler *s,
                                     int *accepted);

// ---- context-copy drafts (lookup) ----

typedef struct ContextLookup eremite_lookup;
// Returns a lookup for capacity tokens, or null if it cannot hold that many.
eremite_lookup *eremite_lookup_new(int capacity);
void eremite_lookup_free(eremite_lookup *x);
// Appends the tokens. Returns 0 if the lookup is full.
int eremite_lookup_append(eremite_lookup *x, const int32_t *tokens, int n);
// Writes up to max drafts copied from the context. Returns how many.
int eremite_lookup_propose(const eremite_lookup *x, int max, int32_t *drafts);
// Records how many of the offered drafts were accepted.
void eremite_lookup_record_verification(eremite_lookup *x, int offered,
                                        int accepted);

// ---- how many draft-model drafts a pass runs ----
typedef struct DraftChoice eremite_draft_choice;
eremite_draft_choice *eremite_draft_choice_new(void);
void eremite_draft_choice_free(eremite_draft_choice *x);
// Returns how many drafts the next pass runs, 0 to most.
int eremite_draft_choice_next_count(eremite_draft_choice *x, int most);
// A pass with `count` drafts kept the first `kept` and took `seconds`, with no
// other call beside it.
void eremite_draft_choice_record_pass(eremite_draft_choice *x, int count,
                                      int kept, double seconds);

// ---- statistics, for the server's status ----

// An expert cache's counts, as ExpertStoreCounts holds them.
typedef struct {
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions, late,
      cancelled, substituted, remaps, given_back;
  // Decoded rows' selections by where the expert was.
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;
  // Seconds spent waiting for reads.
  double read_s;
} eremite_cache_counts;

// A model's counters (its expert cache's: zero without one) and the
// process's memory.
typedef struct {
  // All of the cache's; of decoded rows; of prompt rows.
  eremite_cache_counts cache, decode, prefill;
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  // The prefill running: its layer (0: none) and rows.
  int prefill_layer, prefill_rows_now;
  // The expert cache's slots, and experts on the SSD only.
  int slots, mapped, cap, max_slots, experts, uncached;
  uint64_t slot_bytes, uncached_bytes, cache_bytes, dense_bytes;
  uint64_t footprint, gpu_in_use, available;
  // Metal's working-set recommendation; INT64_MAX before GPU initialization.
  int64_t gpu_limit;
  uint64_t total, free, file, file_kept, wired, compressor, compressed,
      swap_used, swapouts;
  // 1 normal, 2 warn, 4 critical.
  int pressure;
  // The layers, and the experts a row chooses: 0 without routed experts.
  int layers, experts_per_row;
  // The layers with routed experts, with experts_per_row / routed_layers
  // chosen in each, and each one's experts.
  int routed_layers, experts_per_layer;
  // The most model memory now within --memory and the free memory above the
  // reserve, what the models hold (state memory included), what is counted,
  // and the most ever counted.
  uint64_t model_memory_limit, model_memory_used, model_memory_committed,
      model_memory_peak;
  uint64_t state_allocated, model_workspace;
  // Process-wide since the start, to show where the time goes:
  // - thread_s: the engine thread's seconds outside eremite_run_streams,
  // running a
  //   pass, and in eremite_run_streams with no pass ready while one waits for
  //   the GPU, or else for reads, or else for something else.
  // - gpu_busy_s and gpu_idle_s: the GPU main queue's busy seconds, and its
  //   idle seconds between command buffers in each of those five states of
  //   the engine thread.
  // - ssd_busy_s: the seconds with an expert read in flight. demand_read_s and
  //   background_read_s: the summed seconds of demand and background reads.
  double thread_s[5], gpu_busy_s, gpu_idle_s[5], ssd_busy_s, demand_read_s,
      background_read_s;
} eremite_stats;

// Reads the process's memory once, for every model's statistics.
void eremite_refresh_stats(void);
void eremite_read_stats(const eremite_model *m, eremite_stats *s);
// Writes the experts that row `row` of the model's latest decoding pass chose
// in each layer with routed experts, experts_per_row / routed_layers per
// layer, and where each one was: 0 in memory, 1 read ahead, 2 late, 3
// substituted, 4 from the SSD. Returns 1. Returns 0, and writes nothing, for a
// model without routed experts or a row that did not run every layer. It runs
// on the engine thread, after the pass.
int eremite_read_routes(const eremite_model *m, int row, uint16_t *experts,
                        uint8_t *sources);

// ---- tokenizers (tokenizer.json), from any thread ----

// A checkpoint's tokenizer: a model's own (eremite_model_tokenizer), or one
// opened from a checkpoint directory without its model. `image` is the
// spelling of the model's image token, or null. A tokenizer.json that the
// engine cannot run ends the process with a message.
typedef struct Tokenizer eremite_tokenizer;
eremite_tokenizer *eremite_model_tokenizer(const eremite_model *m);
eremite_tokenizer *eremite_tokenizer_open(const char *dir, const char *image);
// Writes the token ids of len bytes of text, at most max. Mode 0 takes the text
// alone. Mode 1 also takes the added tokens' spellings as those tokens. Mode 2
// takes a rendered prompt: as mode 1, with the byte 0xFF for an image's place.
// Returns the count, or -1 if there are more than max.
int eremite_tokenize(eremite_tokenizer *t, const char *text, size_t len,
                     int mode, int32_t *out, int max);
// Returns a token's bytes, and sets *len to their length: 0 for an unknown id.
const char *eremite_token_bytes(const eremite_tokenizer *t, int32_t id,
                                size_t *len);
// Returns whether the token is one of the tokenizer's added tokens: 2 for one
// that tokenizer.json marks special, 1 for another, 0 for none.
int eremite_token_kind(const eremite_tokenizer *t, int32_t id);
// Returns the id of the added token spelled by len bytes of text, or -1.
int32_t eremite_find_added_token(const eremite_tokenizer *t, const char *text,
                                 size_t len);

#endif
