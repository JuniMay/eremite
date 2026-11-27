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
#include "platform/image.h"
#include "platform/memory_policy.h"
#include "platform/system_memory.h"
#include "platform/threads.h"
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
  // Accumulated active seconds during multi-model GPU contention, rebased so
  // the lagging active model stays at 0.
  double fair_time;
  // Whether this model was contending for the GPU in the previous turn check.
  int fair_active;
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
// The memory that every model is counted in, made by eremite_init, its
// platform memory policy backend, and the GPU that the models run on, made
// when the first model loads.
static Memory *memory;
static PlatformMemory *memory_policy;
static Gpu *gpu;
// The calls posted to streams since the process started. Each stream keeps
// the number of its call.
static uint64_t calls_posted;
// The model that owns the GPU turn while two or more distinct models contend
// for the GPU, or null when at most one model contends.
static eremite_model *turn_owner;
// Wall-clock time when turn_owner's current accounted span began, or 0 outside
// eremite_run_streams.
static double turn_started_at;
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
  memory_policy = platform_memory_new_macos(&(PlatformMemoryMacosConfig){
      .enable_occupy = p->occupy_free_memory != 0});
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
  memory_set_policy(memory, memory_policy);
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

// Maximum path buffer length in bytes when locating a model's tokenizer.
static constexpr size_t MODEL_PATH_CAP = 2048;

// Resolves dir, dir/tokenizer or dir/processor for a model's tokenizer.json.
static Tokenizer *open_model_tokenizer(const char *dir, const char *media) {
  char path[MODEL_PATH_CAP];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  if (access(path, R_OK) == 0) return tokenizer_open(dir, media);
  snprintf(path, sizeof path, "%s/tokenizer/tokenizer.json", dir);
  if (access(path, R_OK) == 0) {
    snprintf(path, sizeof path, "%s/tokenizer", dir);
    return tokenizer_open(path, media);
  }
  snprintf(path, sizeof path, "%s/processor", dir);
  return tokenizer_open(path, media);
}

eremite_model *eremite_open(const eremite_config *c) {
  if (!initialized || !c || c->abi != EREMITE_ABI || !c->model) return NULL;
  const ModelOps *ops = model_find_family(c->model);
  const eremite_deepseek_config *d = &c->deepseek;
  const eremite_inkling2_config *ik = &c->inkling2;
  const eremite_minimax_h3_config *h3 = &c->minimax_h3;
  const eremite_minimax_h3_video_config *hv = &c->minimax_h3_video;
  int has_deepseek = d->experts || d->ced || d->substitute != 0 ||
                     d->no_prefill_prune || d->unpacked_state_rows;
  int has_inkling2 = ik->fp8_kv != 0;
  int has_minimax_h3 = h3->text_encoder || h3->dit || h3->vae;
  int has_minimax_h3_video =
      hv->text_encoder || hv->dit || hv->vae || hv->audio_vae || hv->lora ||
      hv->lora_strength != 0 || hv->default_steps != 0 ||
      hv->shift_video != 0 || hv->shift_audio != 0 || hv->variant != 0;
  Options options = {.ctx = c->ctx,
                     .lookup = c->lookup,
                     .dump_fd = -1,
                     .routes_fd = -1,
                     .quiet = 1,
                     .draft = d->draft};
  ImageLimits img_limits = {0};
  int has_image_limits = 0;
  if (ops->read_image_limits) {
    ops->read_image_limits(&img_limits);
    has_image_limits = 1;
    if (options.ctx == 0) options.ctx = img_limits.max_prompt_tokens;
  }
  VideoLimits vid_limits = {0};
  int has_video_limits = 0;
  if (ops->read_video_limits) {
    ops->read_video_limits(&vid_limits);
    has_video_limits = 1;
    if (options.ctx == 0) options.ctx = vid_limits.max_prompt_tokens;
  }
  if (ops->generate_image) {
    options.minimax_h3_image = (MinimaxH3ImageOptions){
        .text_encoder = h3->text_encoder, .dit = h3->dit, .vae = h3->vae};
  } else if (ops->generate_video) {
    options.minimax_h3_video =
        (MinimaxH3VideoOptions){.text_encoder = hv->text_encoder,
                                .dit = hv->dit,
                                .vae = hv->vae,
                                .audio_vae = hv->audio_vae,
                                .lora = hv->lora,
                                .lora_strength = hv->lora_strength,
                                .default_steps = hv->default_steps,
                                .shift_video = hv->shift_video,
                                .shift_audio = hv->shift_audio,
                                .variant = (VideoVariant)hv->variant};
  } else if (ops == &inkling2_ops) {
    options.inkling2 = (Inkling2Options){.fp8_kv = ik->fp8_kv != 0};
  } else {
    options.deepseek =
        (DeepSeekOptions){.substitute = d->substitute,
                          .experts = d->experts,
                          .ced = d->ced,
                          .prefill_prune = !d->ced && !d->no_prefill_prune,
                          .packed_state_rows = !d->unpacked_state_rows};
  }
  const char *bad =
      options.ctx < 1 || options.ctx > 1 << 20 ? "ctx takes 1 to 1048576 tokens"
      : has_image_limits && options.ctx > img_limits.max_prompt_tokens
          ? "ctx exceeds this model's prompt token limit"
      : has_video_limits && options.ctx > vid_limits.max_prompt_tokens
          ? "ctx exceeds this model's prompt token limit"
      : ops->decide && (options.draft || options.lookup || has_deepseek ||
                        has_inkling2 || has_minimax_h3 || has_minimax_h3_video)
          ? "a decision model takes no drafts, lookup, experts, "
            "ced, substitute, packed state rows, fp8_kv or media checkpoint "
            "paths"
      : ops->generate_image &&
              (options.draft || options.lookup || has_deepseek ||
               has_inkling2 || has_minimax_h3_video)
          ? "an image model takes no drafts, lookup, experts, "
            "ced, substitute, packed state rows, fp8_kv or video checkpoint "
            "paths"
      : ops->generate_image &&
              (!h3->text_encoder || !h3->text_encoder[0] || !h3->dit ||
               !h3->dit[0] || !h3->vae || !h3->vae[0])
          ? "an image model requires text_encoder, dit and vae paths"
      : !ops->generate_image && has_minimax_h3
          ? "image text_encoder, dit and vae apply to image models only"
      : ops->generate_video && (options.draft || options.lookup ||
                                has_deepseek || has_inkling2 || has_minimax_h3)
          ? "a video model takes no drafts, lookup, experts, "
            "ced, substitute, packed state rows, fp8_kv or image checkpoint "
            "paths"
      : ops->generate_video &&
              (!hv->text_encoder || !hv->text_encoder[0] || !hv->dit ||
               !hv->dit[0] || !hv->vae || !hv->vae[0] || !hv->audio_vae ||
               !hv->audio_vae[0])
          ? "a video model requires text_encoder, dit, vae and audio_vae paths"
      : ops->generate_video && (hv->variant != EREMITE_VIDEO_VARIANT_FL2VA &&
                                hv->variant != EREMITE_VIDEO_VARIANT_REF2VA)
          ? "video variant must be FL2VA or REF2VA"
      : ops->generate_video &&
              (!isfinite(hv->lora_strength) || hv->lora_strength < 0 ||
               !isfinite(hv->shift_video) || hv->shift_video < 0 ||
               !isfinite(hv->shift_audio) || hv->shift_audio < 0)
          ? "video lora_strength, shift_video and shift_audio must be "
            "non-negative"
      : ops->generate_video && hv->default_steps != 0 &&
              (hv->default_steps < vid_limits.min_steps ||
               hv->default_steps > vid_limits.max_steps)
          ? "video default_steps is out of range"
      : !ops->generate_video && has_minimax_h3_video
          ? "video checkpoint paths apply to video models only"
      : ops != &inkling2_ops && has_inkling2
          ? "fp8_kv applies to Inkling-2-Mini models only"
      : ops != &deepseek_ops && !ops->decide && !ops->generate_image &&
              !ops->generate_video && has_deepseek
          ? "experts, ced, substitute and packed state rows apply to "
            "DeepSeek models only"
      : options.draft < 0 || options.draft > ops->max_drafts
          ? "draft exceeds this model's proposal limit"
      : options.lookup < 0 || options.lookup > VERIFY_DRAFTS
          ? "lookup takes off or 1 to 15 proposals"
      : !ops->generate_image && !ops->generate_video && d->ced &&
              (d->ced < WINDOW || d->ced > 1 << 20)
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
  if (ops->generate_image) {
    x->opt.minimax_h3_image.text_encoder =
        copy(options.minimax_h3_image.text_encoder);
    x->opt.minimax_h3_image.dit = copy(options.minimax_h3_image.dit);
    x->opt.minimax_h3_image.vae = copy(options.minimax_h3_image.vae);
  } else if (ops->generate_video) {
    x->opt.minimax_h3_video.text_encoder =
        copy(options.minimax_h3_video.text_encoder);
    x->opt.minimax_h3_video.dit = copy(options.minimax_h3_video.dit);
    x->opt.minimax_h3_video.vae = copy(options.minimax_h3_video.vae);
    x->opt.minimax_h3_video.audio_vae =
        copy(options.minimax_h3_video.audio_vae);
    x->opt.minimax_h3_video.lora = copy(options.minimax_h3_video.lora);
  } else if (ops == &deepseek_ops) {
    x->opt.deepseek.experts = copy(options.deepseek.experts);
  }
  x->tk = open_model_tokenizer(c->model, ops->media_token);
  x->dir = copy(c->model);
  x->m = model_create(memory, NULL, x->dir, (uint64_t)(c->memory_gb * GiB));
  models[nmodels++] = x;
  say("%s: registered %s; weights load on first inference request", c->model,
      ops->family);
  return x;
}

int eremite_is_loaded(const eremite_model *m) {
  if (!atomic_load_explicit(&m->loaded, memory_order_acquire)) return 0;
  if (m->ops->is_loaded) return m->ops->is_loaded();
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
  else if (!was_loaded && m->ops->generate_image)
    say("%s: %s image model, prompt limit %d tokens", m->dir, m->ops->family,
        m->opt.ctx);
  else if (!was_loaded && m->ops->generate_video)
    say("%s: %s video model, prompt limit %d tokens", m->dir, m->ops->family,
        m->opt.ctx);
  else if (!was_loaded && m->ops == &inkling2_ops)
    say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d; "
        "fp8 kv %s",
        m->dir, m->ops->family, m->opt.ctx, m->ops->drafts_per_pass(),
        m->ops->rewind_margin(), m->opt.lookup,
        m->opt.inkling2.fp8_kv ? "on (approximate)" : "off");
  else if (!was_loaded && m->ops == &deepseek_ops)
    say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d; "
        "prefill: ced_depth=%d (%s), dependency "
        "pruning=%s; packed state rows %s",
        m->dir, m->ops->family, m->opt.ctx, m->ops->drafts_per_pass(),
        m->ops->rewind_margin(), m->opt.lookup, m->opt.deepseek.ced,
        m->opt.deepseek.ced ? "approximate" : "off",
        m->opt.deepseek.prefill_prune ? "on" : "off",
        m->opt.deepseek.packed_state_rows ? "on" : "off");
  else if (!was_loaded)
    say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d",
        m->dir, m->ops->family, m->opt.ctx, m->ops->drafts_per_pass(),
        m->ops->rewind_margin(), m->opt.lookup);
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

int eremite_is_image_model(const eremite_model *m) {
  return m->ops->generate_image != NULL;
}

int eremite_image_model_limits(const eremite_model *m,
                               eremite_image_limits *out) {
  if (!m || !out || !m->ops->read_image_limits) return 0;
  ImageLimits lim = {0};
  m->ops->read_image_limits(&lim);
  *out = (eremite_image_limits){
      .max_prompt_tokens = lim.max_prompt_tokens,
      .max_dim = lim.max_dim,
      .default_width = lim.default_width,
      .default_height = lim.default_height,
      .min_generation_area = lim.min_generation_area,
      .max_generation_area = lim.max_generation_area,
      .default_generation_area = lim.default_generation_area,
      .max_canvas_area = lim.max_canvas_area,
      .max_request_pixels = lim.max_request_pixels,
      .max_images_per_request = lim.max_images_per_request,
      .max_references = lim.max_references,
      .min_quality = lim.min_quality,
      .max_quality = lim.max_quality,
      .default_quality = lim.default_quality,
      .inference_steps = lim.inference_steps,
      .guidance_scale = lim.guidance_scale};
  return 1;
}

int eremite_image_read_size(const uint8_t *data, size_t len, int *width,
                            int *height, char *error, size_t error_cap) {
  const char *why = 0;
  if (image_read_size(data, len, width, height, &why)) return 1;
  snprintf(error, error_cap, "%s", why);
  return 0;
}

int eremite_image_fit_size(const eremite_model *m, int width, int height,
                           int *out_width, int *out_height, char *error,
                           size_t error_cap) {
  if (!m->ops->fit_image_size) die("%s: not an image model", m->dir);
  return m->ops->fit_image_size(width, height, out_width, out_height, error,
                                error_cap);
}

// Converts an eremite_image_request and runs the image model's generation pass.
static int run_image_generation(eremite_model *m,
                                const eremite_image_request *request,
                                const PassWait *wait, const int *cancelled,
                                eremite_image_result *out) {
  if (!m->ops->generate_image) die("%s: not an image model", m->dir);
  static_assert(sizeof(eremite_image_input) == sizeof(ImageInput),
                "image input structs match");
  static_assert(sizeof(out->error) == IMAGE_ERROR_BYTES,
                "image error buffers match");
  ImageRequest req = {
      .prompt = request->prompt,
      .width = request->width,
      .height = request->height,
      .min_generation_area = request->min_generation_area,
      .format = (ImageFormat)request->format,
      .quality = request->quality,
      .background = (ImageBackground)request->background,
      .seed = request->seed,
      .refs = (const ImageInput *)request->refs,
      .ref_count = request->ref_count,
      .mask = {.data = request->mask.data, .len = request->mask.len},
      .partial_images = request->partial_images,
      .on_partial_image = request->on_partial_image,
      .partial_image_context = request->partial_image_context};
  ImageResult res = {0};
  int ok = m->ops->generate_image(&req, wait, cancelled, &res);
  out->data = res.data;
  out->len = res.len;
  memcpy(out->error, res.error, sizeof out->error);
  out->invalid_input = res.invalid_input;
  out->prompt_tokens = res.prompt_tokens;
  out->prefill_seconds = res.prefill_seconds;
  out->image_seconds = res.image_seconds;
  return ok;
}

int eremite_generate_image(eremite_model *m,
                           const eremite_image_request *request,
                           eremite_image_result *out) {
  return run_image_generation(m, request, NULL, NULL, out);
}

void eremite_image_result_free(eremite_image_result *result) {
  if (!result) return;
  free(result->data);
  memset(result, 0, sizeof *result);
}

int eremite_is_video_model(const eremite_model *m) {
  return m->ops->generate_video != NULL;
}

int eremite_video_model_limits(const eremite_model *m,
                               eremite_video_limits *out) {
  if (!m || !out || !m->ops->read_video_limits) return 0;
  VideoLimits lim = {0};
  m->ops->read_video_limits(&lim);
  const MinimaxH3VideoOptions *vo = &m->opt.minimax_h3_video;
  lim.variant = vo->variant;
  if (vo->default_steps > 0) lim.default_steps = vo->default_steps;
  if (vo->shift_video > 0.0) lim.default_shift_video = vo->shift_video;
  if (vo->shift_audio > 0.0) lim.default_shift_audio = vo->shift_audio;
  *out = (eremite_video_limits){.max_prompt_tokens = lim.max_prompt_tokens,
                                .patch_px = lim.patch_px,
                                .max_dim = lim.max_dim,
                                .default_width = lim.default_width,
                                .default_height = lim.default_height,
                                .max_canvas_area = lim.max_canvas_area,
                                .fps = lim.fps,
                                .min_frames = lim.min_frames,
                                .max_frames = lim.max_frames,
                                .default_frames = lim.default_frames,
                                .min_seconds = lim.min_seconds,
                                .max_seconds = lim.max_seconds,
                                .default_seconds = lim.default_seconds,
                                .min_steps = lim.min_steps,
                                .max_steps = lim.max_steps,
                                .default_steps = lim.default_steps,
                                .max_ref_images = lim.max_ref_images,
                                .max_ref_videos = lim.max_ref_videos,
                                .max_ref_audios = lim.max_ref_audios,
                                .max_total_refs = lim.max_total_refs,
                                .variant = (int)lim.variant,
                                .guidance_scale = lim.guidance_scale,
                                .default_shift_video = lim.default_shift_video,
                                .default_shift_audio = lim.default_shift_audio};
  return 1;
}

// Converts an eremite_video_request and runs the video model's generation pass.
static int run_video_generation(eremite_model *m,
                                const eremite_video_request *request,
                                const PassWait *wait, const int *cancelled,
                                eremite_video_result *out) {
  if (!m->ops->generate_video) die("%s: not a video model", m->dir);
  static_assert(sizeof(eremite_video_media_input) == sizeof(VideoMediaInput),
                "video media input structs match");
  static_assert(sizeof(out->error) == VIDEO_ERROR_BYTES,
                "video error buffers match");
  VideoRequest req = {
      .prompt = request->prompt,
      .width = request->width,
      .height = request->height,
      .seconds = request->seconds,
      .frames = request->frames,
      .steps = request->steps,
      .shift_video = request->shift_video,
      .shift_audio = request->shift_audio,
      .seed = request->seed,
      .first_frame = {.data = request->first_frame.data,
                      .len = request->first_frame.len},
      .last_frame = {.data = request->last_frame.data,
                     .len = request->last_frame.len},
      .ref_images = (const VideoMediaInput *)request->ref_images,
      .ref_image_count = request->ref_image_count,
      .ref_videos = (const VideoMediaInput *)request->ref_videos,
      .ref_video_audios = (const VideoMediaInput *)request->ref_video_audios,
      .ref_video_count = request->ref_video_count,
      .ref_audios = (const VideoMediaInput *)request->ref_audios,
      .ref_audio_count = request->ref_audio_count,
      .ref_image_max = request->ref_image_max};
  VideoResult res = {0};
  int ok = m->ops->generate_video(&req, wait, cancelled, &res);
  out->data = res.data;
  out->len = res.len;
  out->width = res.width;
  out->height = res.height;
  out->frames = res.frames;
  out->fps = res.fps;
  memcpy(out->error, res.error, sizeof out->error);
  out->invalid_input = res.invalid_input;
  out->prompt_tokens = res.prompt_tokens;
  out->prefill_seconds = res.prefill_seconds;
  out->video_seconds = res.video_seconds;
  return ok;
}

int eremite_generate_video(eremite_model *m,
                           const eremite_video_request *request,
                           eremite_video_result *out) {
  return run_video_generation(m, request, NULL, NULL, out);
}

void eremite_video_result_free(eremite_video_result *result) {
  if (!result) return;
  free(result->data);
  memset(result, 0, sizeof *result);
}

// Ends the program for a decision, image or video model: none keeps attention
// states or runs prefills, decoding or drafts.
static void require_states(const eremite_model *m) {
  if (m->ops->decide)
    die("%s: a decision model keeps no states and runs no prefills or "
        "decoding",
        m->dir);
  if (m->ops->generate_image)
    die("%s: an image model keeps no states and runs no prefills or decoding",
        m->dir);
  if (m->ops->generate_video)
    die("%s: a video model keeps no states and runs no prefills or decoding",
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

size_t eremite_state_copy_bytes(const eremite_model *m,
                                const eremite_state *src, int positions,
                                int target_positions) {
  require_states(m);
  if (m->ops->state_copy_bytes)
    return m->ops->state_copy_bytes(src, positions, target_positions);
  return m->ops->state_bytes(NULL, target_positions);
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
    if (m->ops == &inkling2_ops) {
      identity_number(&h, m->opt.inkling2.fp8_kv != 0);
    } else if (m->ops == &deepseek_ops) {
      const DeepSeekOptions *d = &m->opt.deepseek;
      uint32_t substitute;
      memcpy(&substitute, &d->substitute, sizeof substitute);
      identity_number(&h, d->ced);
      identity_number(&h, d->prefill_prune);
      identity_number(&h, d->packed_state_rows != 0);
      identity_number(&h, substitute);
      identity_number(&h, d->experts != NULL);
      if (d->experts) identity_add_gguf_set(&h, d->experts);
    }
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
  STREAM_IMAGE,
  STREAM_VIDEO,
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
  // An image call's request, cooperative cancellation flag and result.
  eremite_image_request image_request;
  eremite_image_result image_result;
  // A video call's request and result.
  eremite_video_request video_request;
  eremite_video_result video_result;
  int cancelled;
  // its results
  const float *logits;
  eremite_snapshot *captured;
  eremite_prefill_info info;
  int32_t next;
  int accepted;
  // Unified cooperative turn wait checked between layers or slices: enforces
  // same-model decode priority for prefills and balanced measured-time turns
  // across distinct models.
  PassWait wait_for_turn;
  // Last posted call number seen at the first look of a same-model prefill
  // wait, or 0 outside such a wait.
  uint64_t calls_at_first_look;
  // Last committed GPU command buffer number seen at the first look of a turn
  // checkpoint, or 0 outside a checkpoint.
  uint64_t work_at_first_look;
  // Nonzero while waiting for work_at_first_look to finish before accounting
  // the current turn checkpoint.
  int in_turn_wait;
  // Nonzero while yielded waiting for turn_owner to return to this stream's
  // model.
  int waiting_for_turn;
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

// Whether the stream's started call waits for off-GPU background work (such as
// MP4 or image encoding) rather than GPU or expert-read work.
static int is_background_waiting(const eremite_stream *x) {
  return x->call != STREAM_NONE && x->started &&
         pass_waiting_for(x->pass) == PASS_WAIT_OTHER &&
         pass_has_notify(x->pass) && !pass_ready(x->pass);
}

// Whether the stream currently contends for the GPU or expert-read I/O.
static int is_contending(const eremite_stream *x) {
  return x->call != STREAM_NONE && !is_background_waiting(x);
}

// Whether the stream has started work currently in flight on the GPU or
// waiting on expert reads, rather than paused at a turn or decode yield.
static int is_in_flight(const eremite_stream *x) {
  return is_contending(x) && x->started && !x->waiting_for_turn &&
         !x->calls_at_first_look;
}

// Whether decoding calls of the same model posted by the wait's first look
// have not finished yet.
static int own_decoding_calls_remain(eremite_stream *x) {
  if (!x->run) die("a stream's pass waited outside eremite_run_streams");
  if (!x->calls_at_first_look) x->calls_at_first_look = calls_posted;
  for (int k = 0; k < x->run->count; k++) {
    const eremite_stream *other = x->run->streams[k];
    if (other->m == x->m && has_decoding_call(other) &&
        other->call_number <= x->calls_at_first_look)
      return 1;
  }
  return 0;
}

// Returns whether model m has at least one contending stream in streams.
static int model_is_contending(const eremite_model *m,
                               eremite_stream *const *streams, int n) {
  for (int k = 0; k < n; k++)
    if (streams[k]->m == m && is_contending(streams[k])) return 1;
  return 0;
}

// Returns the earliest call number among contending streams of model m.
static uint64_t model_earliest_call(const eremite_model *m,
                                    eremite_stream *const *streams, int n) {
  uint64_t best = UINT64_MAX;
  for (int k = 0; k < n; k++)
    if (streams[k]->m == m && is_contending(streams[k]) &&
        streams[k]->call_number < best)
      best = streams[k]->call_number;
  return best;
}

// Charges elapsed wall-clock time up to at against the current turn owner.
static void charge_turn_owner(double at) {
  if (turn_owner && turn_started_at > 0.0) {
    double dt = at - turn_started_at;
    if (dt > 0.0) turn_owner->fair_time += dt;
    turn_started_at = at;
  }
}

// Rebases fair_time across all contending models so the lagging model sits at
// 0.
static void rebase_fair_times(eremite_stream *const *streams, int n) {
  int found = 0;
  double min_t = 0.0;
  for (int i = 0; i < nmodels; i++) {
    if (!model_is_contending(models[i], streams, n)) continue;
    if (!found || models[i]->fair_time < min_t) {
      min_t = models[i]->fair_time;
      found = 1;
    }
  }
  if (!found || min_t <= 0.0) return;
  for (int i = 0; i < nmodels; i++) {
    if (!model_is_contending(models[i], streams, n)) continue;
    models[i]->fair_time -= min_t;
    if (models[i]->fair_time < 0.0) models[i]->fair_time = 0.0;
  }
}

// Picks the contending model with the smallest fair_time (breaking ties by
// earliest posted call number). When prefer_other is non-null, yields to any
// other contending model whose fair_time is at most prefer_other's.
static eremite_model *pick_lowest_fair_model(eremite_stream *const *streams,
                                             int n,
                                             eremite_model *prefer_other) {
  eremite_model *best = NULL;
  uint64_t best_call = UINT64_MAX;
  for (int i = 0; i < nmodels; i++) {
    eremite_model *m = models[i];
    if (m == prefer_other || !model_is_contending(m, streams, n)) continue;
    uint64_t call = model_earliest_call(m, streams, n);
    if (!best || m->fair_time < best->fair_time ||
        (m->fair_time == best->fair_time && call < best_call)) {
      best = m;
      best_call = call;
    }
  }
  if (prefer_other) {
    if (best && best->fair_time <= prefer_other->fair_time) return best;
    return prefer_other;
  }
  return best;
}

// Publishes each model's remaining measured turn budget in seconds (0 when
// running alone, not owning the turn, or before any peer turn is measured).
static void update_turn_budgets(eremite_stream *const *streams, int n) {
  double peer_min = 0.0;
  int has_peer = 0;
  if (turn_owner) {
    for (int i = 0; i < nmodels; i++) {
      eremite_model *m = models[i];
      if (m == turn_owner || !model_is_contending(m, streams, n)) continue;
      if (!has_peer || m->fair_time < peer_min) {
        peer_min = m->fair_time;
        has_peer = 1;
      }
    }
  }
  double owner_budget =
      (turn_owner && has_peer && peer_min > turn_owner->fair_time)
          ? peer_min - turn_owner->fair_time
          : 0.0;
  for (int i = 0; i < nmodels; i++) {
    if (!models[i]->ops->set_turn_budget) continue;
    models[i]->ops->set_turn_budget(models[i] == turn_owner ? owner_budget
                                                            : 0.0);
  }
}

// Synchronizes multi-model fairness state, other-passes counts and turn
// budgets with the current contending streams.
static void sync_fair_state(eremite_stream *const *streams, int n) {
  int contending_streams = 0;
  for (int k = 0; k < n; k++) contending_streams += is_contending(streams[k]);
  int other = contending_streams > 1 ? contending_streams - 1 : 0;
  for (int i = 0; i < nmodels; i++)
    if (models[i]->ops->set_other_passes)
      models[i]->ops->set_other_passes(other);

  int now_active[8] = {0};
  int active_models = 0;
  for (int i = 0; i < nmodels; i++) {
    now_active[i] = model_is_contending(models[i], streams, n);
    active_models += now_active[i];
  }

  if (active_models <= 1) {
    turn_owner = NULL;
    turn_started_at = 0.0;
    for (int i = 0; i < nmodels; i++) {
      models[i]->fair_time = 0.0;
      models[i]->fair_active = now_active[i];
    }
    update_turn_budgets(streams, n);
    return;
  }

  // Rebase already-active models first so newly joining models start at 0,
  // matching the lagging peer rather than monopolizing or starving.
  int kept = 0;
  double kept_min = 0.0;
  for (int i = 0; i < nmodels; i++) {
    if (models[i]->fair_active && now_active[i]) {
      if (!kept || models[i]->fair_time < kept_min) {
        kept_min = models[i]->fair_time;
        kept = 1;
      }
    }
  }
  for (int i = 0; i < nmodels; i++) {
    if (models[i]->fair_active && now_active[i]) {
      if (kept && kept_min > 0.0) {
        models[i]->fair_time -= kept_min;
        if (models[i]->fair_time < 0.0) models[i]->fair_time = 0.0;
      }
    } else if (models[i]->fair_active && !now_active[i]) {
      models[i]->fair_time = 0.0;
      if (turn_owner == models[i]) turn_owner = NULL;
    } else if (!models[i]->fair_active && now_active[i]) {
      models[i]->fair_time = 0.0;
    }
    models[i]->fair_active = now_active[i];
  }
  rebase_fair_times(streams, n);

  // Preserve ownership for any model with GPU or expert-read work already in
  // flight so its remaining GPU execution is charged to it on a clean queue.
  eremite_model *inflight_model = NULL;
  int owner_inflight = 0;
  for (int k = 0; k < n; k++) {
    if (!is_in_flight(streams[k])) continue;
    if (streams[k]->m == turn_owner) owner_inflight = 1;
    if (!inflight_model) inflight_model = streams[k]->m;
  }
  if (!owner_inflight) {
    if (inflight_model)
      turn_owner = inflight_model;
    else
      turn_owner = pick_lowest_fair_model(streams, n, NULL);
  }
  if (turn_started_at == 0.0) turn_started_at = now();
  update_turn_budgets(streams, n);
}

// Marks the stream's turn wait as PASS_WAIT_OTHER so the engine thread sleeps
// while another model waits on reads or background work.
static void set_wait_other(eremite_stream *x) {
  x->wait_for_turn.kind = PASS_WAIT_OTHER;
  pass_set_wait_kind(x->pass, PASS_WAIT_OTHER);
}

// Clears the stream's turn wait state when resuming execution.
static void end_turn_wait(eremite_stream *x) {
  x->calls_at_first_look = 0;
  x->work_at_first_look = 0;
  x->in_turn_wait = 0;
  x->waiting_for_turn = 0;
  x->wait_for_turn.kind = PASS_WAIT_GPU;
}

// Whether the stream may run its next layer or slice. Enforces same-model
// decoding priority for prefills and symmetric measured-time turns across
// distinct contending models, with zero gating when running alone.
static int stream_may_go_on(void *context) {
  eremite_stream *x = context;
  eremite_model *m = x->m;
  if (!x->run) die("a stream's pass waited outside eremite_run_streams");

  if (x->call == STREAM_PREFILL && own_decoding_calls_remain(x)) {
    set_wait_other(x);
    return 0;
  }
  x->calls_at_first_look = 0;

  if (!turn_owner) {
    end_turn_wait(x);
    return 1;
  }
  if (turn_owner != m) {
    x->waiting_for_turn = 1;
    x->in_turn_wait = 0;
    x->work_at_first_look = 0;
    set_wait_other(x);
    return 0;
  }
  if (x->waiting_for_turn) {
    update_turn_budgets(x->run->streams, x->run->count);
    end_turn_wait(x);
    return 1;
  }

  if (!x->in_turn_wait) {
    x->in_turn_wait = 1;
    x->work_at_first_look = gpu ? gpu_last_committed(gpu) : 0;
  }
  if (x->work_at_first_look && !gpu_has_finished(gpu, x->work_at_first_look))
    return 0;

  double at = now();
  charge_turn_owner(at);
  rebase_fair_times(x->run->streams, x->run->count);
  eremite_model *next =
      pick_lowest_fair_model(x->run->streams, x->run->count, m);
  if (next != m) {
    turn_owner = next;
    turn_started_at = at;
    x->waiting_for_turn = 1;
    x->in_turn_wait = 0;
    x->work_at_first_look = 0;
    update_turn_budgets(x->run->streams, x->run->count);
    set_wait_other(x);
    return 0;
  }
  update_turn_budgets(x->run->streams, x->run->count);
  end_turn_wait(x);
  return 1;
}

// Blocks until the GPU work committed by this stream before its turn
// checkpoint has finished.
static void wait_for_turn_work(void *context) {
  eremite_stream *x = context;
  if (x->work_at_first_look && gpu)
    gpu_wait_until_finished(gpu, x->work_at_first_look);
}

// Whether the stream waits while another model owns the GPU turn.
static int waits_for_turn(const eremite_stream *x) {
  return turn_owner && x->m != turn_owner && !is_background_waiting(x);
}

static void stream_run(eremite_stream *x) {
  eremite_model *m = x->m;
  x->logits = NULL;
  switch (x->call) {
  case STREAM_PREFILL:
    x->logits = run_prefill(m, x->states[0], x->tokens, x->n, x->from,
                            x->replay, x->capture, &x->captured, &x->info,
                            x->images, &x->wait_for_turn);
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
  case STREAM_IMAGE:
    run_image_generation(m, &x->image_request, &x->wait_for_turn, &x->cancelled,
                         &x->image_result);
    break;
  case STREAM_VIDEO:
    run_video_generation(m, &x->video_request, &x->wait_for_turn, &x->cancelled,
                         &x->video_result);
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
  x->wait_for_turn =
      (PassWait){stream_may_go_on, wait_for_turn_work, x, PASS_WAIT_GPU, NULL};
  x->pass = m->ops->pass_new(stream_body, x);
  return x;
}

void eremite_stream_free(eremite_stream *x) {
  if (!x) return;
  x->call = STREAM_QUIT;
  pass_run(x->pass);
  pass_free(x->pass);
  eremite_image_result_free(&x->image_result);
  eremite_video_result_free(&x->video_result);
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

void eremite_stream_generate_image(eremite_stream *x,
                                   const eremite_image_request *request) {
  if (!x->m->ops->generate_image) die("%s: not an image model", x->m->dir);
  eremite_image_result_free(&x->image_result);
  x->image_request = *request;
  x->cancelled = 0;
  post_call(x, STREAM_IMAGE);
}

void eremite_stream_generate_video(eremite_stream *x,
                                   const eremite_video_request *request) {
  if (!x->m->ops->generate_video) die("%s: not a video model", x->m->dir);
  eremite_video_result_free(&x->video_result);
  x->video_request = *request;
  x->cancelled = 0;
  post_call(x, STREAM_VIDEO);
}

void eremite_stream_cancel(eremite_stream *x) {
  if (x) x->cancelled = 1;
}

// What the streams with a call wait for while none can go on (THREAD_*): the
// GPU if any active waiter waits for it, else reads if any waits for them,
// else something else.
static int streams_wait_for(eremite_stream *const *streams, int n) {
  int reads = 0;
  for (int k = 0; k < n; k++) {
    if (streams[k]->call == STREAM_NONE || waits_for_turn(streams[k])) continue;
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
  const StreamRun run = {streams, n};
  set_stream_run(streams, n, &run);
  sync_fair_state(streams, n);
  Pass *waiting[n];
  int finished = -1;
  for (;;) {
    int ran = 0;
    eremite_model *owner_before = turn_owner;
    for (int k = 0; k < n && finished < 0; k++) {
      int i = (int)(((unsigned)first + (unsigned)k) % (unsigned)n);
      eremite_stream *x = streams[i];
      if (waits_for_turn(x)) continue;
      if (pass_ready(x->pass)) {
        set_thread_state(THREAD_PASS);
        pass_resume(x->pass);
        ran = 1;
        if (!x->finished && is_background_waiting(x)) {
          charge_turn_owner(now());
          sync_fair_state(streams, n);
        }
      }
      if (x->finished) x->finished = 0, first = (unsigned)i + 1, finished = i;
    }
    if (finished >= 0 || atomic_exchange(&woken, 0)) break;
    if (ran || turn_owner != owner_before) continue;
    int count = 0;
    for (int k = 0; k < n; k++)
      if (streams[k]->call != STREAM_NONE && !waits_for_turn(streams[k]))
        waiting[count++] = streams[k]->pass;
    if (!count) break;
    set_thread_state(streams_wait_for(streams, n));
    if (!pass_wait_for_any(waiting, count, wake))
      die("the streams' calls wait for each other: none can go on");
  }
  charge_turn_owner(now());
  turn_started_at = 0.0;
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

int eremite_stream_image_result(eremite_stream *x, eremite_image_result *out) {
  *out = x->image_result;
  memset(&x->image_result, 0, sizeof x->image_result);
  return out->data != NULL;
}

int eremite_stream_video_result(eremite_stream *x, eremite_video_result *out) {
  *out = x->video_result;
  memset(&x->video_result, 0, sizeof x->video_result);
  return out->data != NULL;
}

int eremite_replay_rows(const eremite_model *m, int k) {
  require_states(m);
  return m->ops->replay_rows(k);
}

int eremite_replay_start(const eremite_model *m, int k) {
  require_states(m);
  return m->ops->replay_start ? m->ops->replay_start(k) : k;
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
  if (m->ops->set_history) m->ops->set_history(s, tokens, pos);
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

int eremite_can_align_audio(const eremite_model *m) {
  if (!m || !m->ops->can_align_audio || !m->ops->align_audio) return 0;
  return m->ops->can_align_audio(m->dir);
}

int eremite_align_audio(eremite_model *m, const uint8_t *audio_data,
                        size_t audio_len, const char *transcript,
                        size_t transcript_len, eremite_audio_word **out_words,
                        int *out_count, double *out_duration, char *error,
                        size_t error_cap) {
  if (out_words) *out_words = NULL;
  if (out_count) *out_count = 0;
  if (out_duration) *out_duration = 0.0;
  if (!m || !m->ops->align_audio ||
      (m->ops->can_align_audio && !m->ops->can_align_audio(m->dir))) {
    if (error && error_cap > 0)
      snprintf(error, error_cap,
               "this model does not support word-level audio timestamps");
    return 0;
  }
  AudioWord *raw = NULL;
  int count = 0;
  double duration = 0.0;
  if (!m->ops->align_audio(audio_data, audio_len, transcript, transcript_len,
                           NULL, &raw, &count, &duration, error, error_cap)) {
    free(raw);
    return 0;
  }
  eremite_audio_word *words = NULL;
  if (count > 0) {
    words = xmalloc((size_t)count * sizeof *words);
    for (int i = 0; i < count; i++) {
      words[i] = (eremite_audio_word){.byte_start = raw[i].byte_start,
                                      .byte_end = raw[i].byte_end,
                                      .start = raw[i].start,
                                      .end = raw[i].end};
    }
  }
  free(raw);
  if (out_words)
    *out_words = words;
  else
    free(words);
  if (out_count) *out_count = count;
  if (out_duration) *out_duration = duration;
  return 1;
}

void eremite_audio_words_free(eremite_audio_word *words) { free(words); }

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
                      .memory_occupied = memory ? memory_occupied(memory) : 0,
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

uint64_t eremite_maintain_memory(uint64_t max_grow_bytes) {
  return memory ? memory_maintain(memory, max_grow_bytes) : 0;
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
  s->generated_images = c.generated_images;
  s->image_steps = c.image_steps;
  s->image_seconds = c.image_seconds;
  s->image_stage = c.image_stage;
  s->image_step = c.image_step;
  s->image_steps_total = c.image_steps_total;
  s->generated_videos = c.generated_videos;
  s->video_steps = c.video_steps;
  s->video_seconds = c.video_seconds;
  s->video_stage = c.video_stage;
  s->video_step = c.video_step;
  s->video_steps_total = c.video_steps_total;
  const MemoryAccount *own = &m->m->memory;
  MemoryAccountBreakdown breakdown = memory_account_breakdown(own);
  s->dense_bytes = breakdown.weights;
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
  s->model_workspace = breakdown.workspace;
  s->own_memory_limit = own->limit == UINT64_MAX ? 0 : own->limit;
  s->own_memory_counted = memory_counted(own);
  s->own_state_bytes = breakdown.states;
  s->required_bytes = c.required_bytes;
  s->cache = counts(&c.cache);
  s->slots = c.cache_slots;
  s->mapped = c.mapped;
  s->cap = c.cap;
  s->max_slots = c.max_slots;
  s->experts = c.experts;
  s->slot_bytes = c.slot_bytes;
  s->cache_bytes = breakdown.experts;
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
