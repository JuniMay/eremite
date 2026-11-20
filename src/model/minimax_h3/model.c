#include "model/minimax_h3/model.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include "base/shutdown.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/resample.h"
#include "engine/weights.h"
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include "model/minimax_h3/vision_encoder.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// MiniMax-H3-Image behind ModelOps. Core T2I weight groups (Qwen3-VL text
// encoder, INT8-ConvRot DiT and FP16 VAE decoder) stay resident together when
// loaded; the vision tower and VAE encoder load on demand for editing requests
// and remain resident until evicted or unloaded.

// Stack size in bytes for one MiniMax-H3-Image engine pass.
static constexpr size_t PASS_STACK_BYTES = 1u << 20;

// Maximum path buffer length in bytes when resolving checkpoint files.
static constexpr size_t PATH_CAP = 2048;

// Guard factor above MINIMAX_H3_IMAGE_MAX_DIM before rounding scaled canvas
// axes: rejects extreme aspect ratios early while preventing int overflow.
static constexpr double CANVAS_ASPECT_OVERSIZE_FACTOR = 2.0;

// Bytes in one gibibyte for log messages.
static constexpr double BYTES_PER_GIB = (double)(1ull << 30);

// DiT spatial patch side in latent cells (32 px canvas patch / 16 px VAE cell).
static constexpr int DIT_PATCH_SIDE =
    MINIMAX_H3_IMAGE_PATCH_PX / MINIMAX_H3_VAE_SPATIAL_RATIO;

// Shared GPU device for the active MiniMax-H3-Image model.
static Gpu *model_gpu;

// Workspace memory slot owned by the model's MemoryAccount.
static MemorySlot *work_slot;

// Sub-model handles owned by the single MiniMax-H3-Image instance.
static MinimaxH3TextEncoder *text_enc;
static MinimaxH3VisionEncoder *vision_enc;
static MinimaxH3Dit *dit;
static MinimaxH3ImageVae *vae;

// Least bytes needed for full residence (all three core groups plus workspace)
// and for staged fallback (largest single stage plus workspace).
static uint64_t required_full_bytes, required_staged_bytes;

// Nonzero when all three core T2I weight groups fit resident together.
static int full_resident;

// Total prompt tokens encoded, images produced and denoising steps completed
// since the model opened.
static uint64_t total_prefill_rows, generated_images, total_image_steps;

// Total seconds spent denoising, decoding and encoding images since open.
static double total_image_seconds;

// Live prompt-encoder layer (1-based) and token count while prefilling.
static int active_prefill_layer, active_prefill_rows;

// Live image generation stage, 1-based denoising step and total step count.
static int active_image_stage, active_image_step, active_image_steps_total;

// Clears live per-call progress indicators back to idle.
static void clear_active_progress(void) {
  active_image_stage = IMAGE_STAGE_NONE;
  active_prefill_layer = 0;
  active_prefill_rows = 0;
  active_image_step = 0;
  active_image_steps_total = 0;
}

// Writes message into dst (when non-null and non-empty).
static void write_error(char *dst, size_t cap, const char *message) {
  if (dst && cap > 0) snprintf(dst, cap, "%s", message);
}

int minimax_h3_image_plan_canvas(int width, int height, int min_area,
                                 int *out_canvas_w, int *out_canvas_h,
                                 char *error, size_t error_cap) {
  if (width < 1 || height < 1 || width > MINIMAX_H3_IMAGE_MAX_DIM ||
      height > MINIMAX_H3_IMAGE_MAX_DIM) {
    if (error && error_cap > 0)
      snprintf(error, error_cap, "size dimensions must be between 1 and %d",
               MINIMAX_H3_IMAGE_MAX_DIM);
    return 0;
  }
  int area = min_area == 0 ? MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA : min_area;
  if (area < MINIMAX_H3_IMAGE_MIN_AREA || area > MINIMAX_H3_IMAGE_MAX_AREA) {
    if (error && error_cap > 0)
      snprintf(error, error_cap,
               "min_generation_area must be between %d and %d",
               MINIMAX_H3_IMAGE_MIN_AREA, MINIMAX_H3_IMAGE_MAX_AREA);
    return 0;
  }

  double req_area = (double)width * (double)height;
  double scale = req_area < (double)area ? sqrt((double)area / req_area) : 1.0;
  double sw = (double)width * scale;
  double sh = (double)height * scale;
  if (sw > (double)MINIMAX_H3_IMAGE_MAX_DIM * CANVAS_ASPECT_OVERSIZE_FACTOR ||
      sh > (double)MINIMAX_H3_IMAGE_MAX_DIM * CANVAS_ASPECT_OVERSIZE_FACTOR) {
    write_error(error, error_cap,
                "aspect ratio exceeds the maximum canvas dimension");
    return 0;
  }

  int cw = (int)llround(sw / (double)MINIMAX_H3_IMAGE_PATCH_PX) *
           MINIMAX_H3_IMAGE_PATCH_PX;
  int ch = (int)llround(sh / (double)MINIMAX_H3_IMAGE_PATCH_PX) *
           MINIMAX_H3_IMAGE_PATCH_PX;
  if (cw < MINIMAX_H3_IMAGE_PATCH_PX) cw = MINIMAX_H3_IMAGE_PATCH_PX;
  if (ch < MINIMAX_H3_IMAGE_PATCH_PX) ch = MINIMAX_H3_IMAGE_PATCH_PX;

  double target_ar = (double)width / (double)height;
  while ((int64_t)cw * (int64_t)ch < (int64_t)area) {
    int nw = cw + MINIMAX_H3_IMAGE_PATCH_PX;
    int nh = ch + MINIMAX_H3_IMAGE_PATCH_PX;
    double err_w = fabs((double)nw / (double)ch - target_ar);
    double err_h = fabs((double)cw / (double)nh - target_ar);
    if (err_w <= err_h)
      cw = nw;
    else
      ch = nh;
    if (cw > MINIMAX_H3_IMAGE_MAX_DIM || ch > MINIMAX_H3_IMAGE_MAX_DIM) break;
  }

  if (cw > MINIMAX_H3_IMAGE_MAX_DIM || ch > MINIMAX_H3_IMAGE_MAX_DIM) {
    if (error && error_cap > 0)
      snprintf(error, error_cap, "generation canvas %dx%d exceeds %d px limit",
               cw, ch, MINIMAX_H3_IMAGE_MAX_DIM);
    return 0;
  }
  if ((int64_t)cw * (int64_t)ch > MINIMAX_H3_IMAGE_MAX_CANVAS_AREA) {
    if (error && error_cap > 0)
      snprintf(error, error_cap,
               "generation canvas %dx%d exceeds %lld pixel limit", cw, ch,
               (long long)MINIMAX_H3_IMAGE_MAX_CANVAS_AREA);
    return 0;
  }
  if (out_canvas_w) *out_canvas_w = cw;
  if (out_canvas_h) *out_canvas_h = ch;
  return 1;
}

// Returns the larger of two byte counts.
static uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }

// Returns the DiT patches of the latent of a width by height image.
static int latent_patches(int width, int height) {
  int h_lat = height / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = width / MINIMAX_H3_VAE_SPATIAL_RATIO;
  return ((h_lat + DIT_PATCH_SIDE - 1) / DIT_PATCH_SIDE) *
         ((w_lat + DIT_PATCH_SIDE - 1) / DIT_PATCH_SIDE);
}

MinimaxH3RefSizes minimax_h3_image_ref_sizes(int width, int height) {
  MinimaxH3RefSizes sizes = {.vision =
                                 minimax_h3_vision_ref_tokens(width, height)};
  minimax_h3_vae_ref_size(width, height, &sizes.vae_width, &sizes.vae_height);
  return sizes;
}

MinimaxH3StageBytes minimax_h3_image_stage_bytes(int seq_len,
                                                 const MinimaxH3RefSizes *refs,
                                                 int ref_count, int canvas_w,
                                                 int canvas_h) {
  if (ref_count < 0 || ref_count > MINIMAX_H3_IMAGE_MAX_REFS)
    die("%d reference images is outside 0 to %d", ref_count,
        MINIMAX_H3_IMAGE_MAX_REFS);
  uint64_t hidden =
      (uint64_t)seq_len * (uint64_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  MinimaxH3RefTokens tokens[MINIMAX_H3_IMAGE_MAX_REFS];
  uint64_t vae_encode = 0;
  int ref_patches = 0;
  for (int i = 0; i < ref_count; i++) {
    tokens[i] = refs[i].vision;
    vae_encode =
        max_u64(vae_encode, minimax_h3_image_vae_encoder_workspace_bytes(
                                refs[i].vae_width, refs[i].vae_height));
    ref_patches += latent_patches(refs[i].vae_width, refs[i].vae_height);
  }
  // The references' VAE encoding runs after the prompt's, beside its hidden
  // rows.
  uint64_t encode =
      minimax_h3_multimodal_prompt_peak_bytes(seq_len, tokens, ref_count);
  if (ref_count > 0) encode = max_u64(encode, vae_encode + hidden);
  return (MinimaxH3StageBytes){
      .encode = encode,
      .sample = minimax_h3_dit_workspace_bytes(
                    seq_len, ref_patches, latent_patches(canvas_w, canvas_h)) +
                hidden,
      .decode = minimax_h3_image_vae_workspace_bytes()};
}

MinimaxH3RequestMemory minimax_h3_image_plan_request_memory(
    MinimaxH3WeightBytes weights, MinimaxH3StageBytes stages, int ref_count) {
  uint64_t core = weights.text + weights.dit + weights.vae;
  uint64_t edit = ref_count > 0 ? weights.vision + weights.vae_encoder : 0;
  uint64_t full =
      max_u64(max_u64(core + edit + stages.encode, core + stages.sample),
              core + stages.decode);
  uint64_t staged = max_u64(
      max_u64(weights.text + edit + stages.encode, weights.dit + stages.sample),
      weights.vae + stages.decode);
  return (MinimaxH3RequestMemory){.full_bytes = full, .staged_bytes = staged};
}

// Sets account's minimum to bytes when they fit within the current memory
// limit, waiting while released system memory comes back. Returns whether
// they fit.
static int try_admit_bytes(MemoryAccount *account, uint64_t bytes) {
  if (bytes <= memory_counted(account) &&
      bytes > memory_current_limit(account->memory))
    return 0;
  return memory_wait_for_minimum(account, bytes);
}

MinimaxH3Admission minimax_h3_image_admit_request(
    MemoryAccount *account, int prefer_full, uint64_t staged_base, int canvas_w,
    int canvas_h, MinimaxH3RequestMemory req_mem, int *out_use_full,
    char *error, size_t error_cap) {
  uint64_t ceiling = memory_ceiling(account->memory);
  if (account->limit < ceiling) ceiling = account->limit;
  uint64_t staged_target = max_u64(req_mem.staged_bytes, staged_base);
  if (staged_target > ceiling) {
    if (error && error_cap > 0)
      snprintf(error, error_cap,
               "generation canvas %dx%d needs %.1f GiB of memory, more than "
               "the %.1f GiB that this model may use",
               canvas_w, canvas_h, (double)staged_target / BYTES_PER_GIB,
               (double)ceiling / BYTES_PER_GIB);
    return MINIMAX_H3_NEVER_FITS;
  }
  if (prefer_full) {
    uint64_t full_target = max_u64(req_mem.full_bytes, account->minimum);
    if (full_target <= ceiling && try_admit_bytes(account, full_target)) {
      if (out_use_full) *out_use_full = 1;
      return MINIMAX_H3_ADMITTED;
    }
    if (shutdown_requested()) return MINIMAX_H3_NOT_NOW;
  }
  if (try_admit_bytes(account, staged_target)) {
    if (out_use_full) *out_use_full = 0;
    return MINIMAX_H3_ADMITTED;
  }
  if (!shutdown_requested() && error && error_cap > 0)
    snprintf(error, error_cap,
             "generation canvas %dx%d needs %.1f GiB of memory, which does not "
             "fit now",
             canvas_w, canvas_h, (double)staged_target / BYTES_PER_GIB);
  return MINIMAX_H3_NOT_NOW;
}

// Returns the weight bytes of the five MiniMax-H3-Image weight groups.
static MinimaxH3WeightBytes
read_weight_bytes(const MinimaxH3TextEncoder *text,
                  const MinimaxH3VisionEncoder *vision, const MinimaxH3Dit *dit,
                  const MinimaxH3ImageVae *vae) {
  return (MinimaxH3WeightBytes){
      .text = minimax_h3_text_encoder_required_bytes(text),
      .vision = minimax_h3_vision_encoder_required_bytes(vision),
      .dit = minimax_h3_dit_required_bytes(dit),
      .vae = minimax_h3_image_vae_required_bytes(vae),
      .vae_encoder = minimax_h3_image_vae_encoder_required_bytes(vae)};
}

// Resolves the directory containing tokenizer.json under dir into out.
static void resolve_tokenizer_dir(const char *dir, char *out, size_t cap) {
  char probe[PATH_CAP];
  snprintf(probe, sizeof probe, "%s/tokenizer.json", dir);
  if (access(probe, R_OK) == 0) {
    snprintf(out, cap, "%s", dir);
    return;
  }
  snprintf(out, cap, "%s/tokenizer", dir);
}

// Pins or unpins the three core T2I weight groups together.
static void pin_core_groups(int pinned) {
  if (text_enc)
    memory_pin(weight_group_slot(minimax_h3_text_encoder_weights(text_enc)),
               pinned);
  if (dit) memory_pin(weight_group_slot(minimax_h3_dit_weights(dit)), pinned);
  if (vae)
    memory_pin(weight_group_slot(minimax_h3_image_vae_weights(vae)), pinned);
}

// Pins or unpins the two on-demand editing weight groups together.
static void pin_edit_groups(int pinned) {
  if (vision_enc)
    memory_pin(weight_group_slot(minimax_h3_vision_encoder_weights(vision_enc)),
               pinned);
  if (vae)
    memory_pin(weight_group_slot(minimax_h3_image_vae_encoder_weights(vae)),
               pinned);
}

// Opens MiniMax-H3-Image sub-models on first call and loads core T2I weights
// in full-residence mode, falling back to staged loading if memory is short.
static int minimax_h3_image_load(Gpu *gpu, MemorySlot *work, const char *dir,
                                 const Options *o) {
  if (!text_enc) {
    if (!o || !o->minimax_h3_image.text_encoder || !o->minimax_h3_image.dit ||
        !o->minimax_h3_image.vae)
      die("%s: MiniMax-H3-Image requires text_encoder, dit and vae paths", dir);
    model_gpu = gpu;
    work_slot = work;
    char tok_dir[PATH_CAP];
    resolve_tokenizer_dir(dir, tok_dir, sizeof tok_dir);
    text_enc = minimax_h3_text_encoder_open(
        gpu, work->account, o->minimax_h3_image.text_encoder, tok_dir);
    vision_enc = minimax_h3_vision_encoder_open(text_enc, work->account);
    dit = minimax_h3_dit_open(gpu, work->account, o->minimax_h3_image.dit);
    vae =
        minimax_h3_image_vae_open(gpu, work->account, o->minimax_h3_image.vae);
    uint64_t text_req = minimax_h3_text_encoder_required_bytes(text_enc);
    uint64_t vis_req = minimax_h3_vision_encoder_required_bytes(vision_enc);
    uint64_t dit_req = minimax_h3_dit_required_bytes(dit);
    uint64_t vae_req = minimax_h3_image_vae_required_bytes(vae);
    uint64_t vae_enc_req = minimax_h3_image_vae_encoder_required_bytes(vae);
    int default_canvas_patches =
        MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA /
        (MINIMAX_H3_IMAGE_PATCH_PX * MINIMAX_H3_IMAGE_PATCH_PX);
    uint64_t encode_ws = minimax_h3_text_encoder_workspace_bytes(
                             MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS, 0) +
                         (uint64_t)MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS *
                             (uint64_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
    uint64_t dit_ws =
        minimax_h3_dit_workspace_bytes(0, 0, default_canvas_patches);
    uint64_t vae_ws = minimax_h3_image_vae_workspace_bytes();
    uint64_t sample_ws = dit_ws > vae_ws ? dit_ws : vae_ws;
    uint64_t peak_ws = sample_ws > encode_ws ? sample_ws : encode_ws;
    uint64_t encode_stage = text_req + vis_req + vae_enc_req + encode_ws;
    uint64_t sample_stage = dit_req + vae_req + sample_ws;
    required_full_bytes = text_req + dit_req + vae_req + peak_ws;
    required_staged_bytes =
        sample_stage > encode_stage ? sample_stage : encode_stage;
  }

  // Primary mode: load and keep all three core T2I groups resident together.
  pin_core_groups(1);
  if (memory_wait_for_minimum(work_slot->account, required_full_bytes) &&
      minimax_h3_text_encoder_load(text_enc) && minimax_h3_dit_load(dit) &&
      minimax_h3_image_vae_load(vae)) {
    full_resident = 1;
    return 1;
  }
  pin_core_groups(0);
  memory_set_minimum(work_slot->account, 0, MEMORY_REQUIRED);
  if (shutdown_requested()) return 0;

  // Fallback mode when full residence does not fit: confirm the largest single
  // stage fits, then warm the text encoder so the first request's Stage 1 runs
  // immediately without a redundant DiT load/evict cycle.
  MemorySlot *text_slot =
      weight_group_slot(minimax_h3_text_encoder_weights(text_enc));
  memory_pin(text_slot, 1);
  if (!memory_wait_for_minimum(work_slot->account, required_staged_bytes) ||
      !minimax_h3_text_encoder_load(text_enc)) {
    memory_pin(text_slot, 0);
    memory_set_minimum(work_slot->account, 0, MEMORY_REQUIRED);
    return 0;
  }
  memory_pin(text_slot, 0);
  full_resident = 0;
  say("%s: full residence (%.1f GiB) did not fit; running staged (%.1f GiB)",
      dir, (double)required_full_bytes / BYTES_PER_GIB,
      (double)required_staged_bytes / BYTES_PER_GIB);
  return 1;
}

// Returns null because MiniMax-H3-Image manages multiple weight groups.
static MemorySlot *minimax_h3_image_weight_slot(void) { return NULL; }

// Returns nonzero when the required weight groups for the active residency
// mode are resident in GPU memory.
static int minimax_h3_image_is_loaded(void) {
  if (!text_enc || !dit || !vae) return 0;
  int text_ready = memory_ready(
      weight_group_slot(minimax_h3_text_encoder_weights(text_enc)));
  int dit_ready = memory_ready(weight_group_slot(minimax_h3_dit_weights(dit)));
  int vae_ready =
      memory_ready(weight_group_slot(minimax_h3_image_vae_weights(vae)));
  if (full_resident) return text_ready && dit_ready && vae_ready;
  return text_ready || dit_ready || vae_ready;
}

// Sets account's minimum back to base after a request, or to staged_base when
// base does not fit.
static void restore_base_minimum(MemoryAccount *account, uint64_t base,
                                 uint64_t staged_base) {
  if (!memory_set_minimum(account, base, MEMORY_REQUIRED))
    memory_set_minimum(account, staged_base, MEMORY_REQUIRED);
}

// Pins core weight groups while a request runs and unpins all groups when idle.
static void minimax_h3_image_set_request_running(int running) {
  if (running) {
    if (full_resident) pin_core_groups(1);
    return;
  }
  pin_core_groups(0);
  pin_edit_groups(0);
}

// Copies MiniMax-H3-Image counters, live stage progress and resident weight
// bytes into c.
static void minimax_h3_image_read_counters(ModelCounters *c) {
  uint64_t resident = 0;
  if (text_enc) resident += minimax_h3_text_encoder_resident_bytes(text_enc);
  if (vision_enc)
    resident += minimax_h3_vision_encoder_resident_bytes(vision_enc);
  if (dit) resident += minimax_h3_dit_resident_bytes(dit);
  if (vae) {
    resident += minimax_h3_image_vae_resident_bytes(vae);
    resident += minimax_h3_image_vae_encoder_resident_bytes(vae);
  }
  *c = (ModelCounters){.prefill_rows = total_prefill_rows,
                       .prefill_layer = active_prefill_layer,
                       .prefill_rows_now = active_prefill_rows,
                       .generated_images = generated_images,
                       .image_steps = total_image_steps,
                       .image_seconds = total_image_seconds,
                       .image_stage = active_image_stage,
                       .image_step = active_image_step,
                       .image_steps_total = active_image_steps_total,
                       .layers = MINIMAX_H3_TEXT_LAYERS,
                       .dense_bytes = resident,
                       .required_bytes = required_staged_bytes};
}

// Returns the Qwen3-VL tokenizer vocabulary size.
static int minimax_h3_image_vocab_size(void) { return MINIMAX_H3_IMAGE_VOCAB; }

// Returns 0 because MiniMax-H3-Image has no routed expert cache.
static int minimax_h3_image_run_idle_work(void) { return 0; }

// Writes a runtime failure message matching cancellation, shutdown or OOM.
static void set_runtime_error(ImageResult *out,
                              const MinimaxH3RunHooks *hooks) {
  if (hooks && hooks->cancelled && *hooks->cancelled)
    write_error(out->error, sizeof out->error, "request cancelled");
  else if (shutdown_requested())
    write_error(out->error, sizeof out->error, "the server is stopping");
  else
    write_error(out->error, sizeof out->error,
                "image generation failed: memory is short");
}

// Runs one text-to-image or reference editing pass and writes the encoded PNG
// or JPEG bytes and stage timings into out.
static int minimax_h3_image_generate(const ImageRequest *req,
                                     const PassWait *wait, const int *cancelled,
                                     ImageResult *out) {
  memset(out, 0, sizeof *out);
  if (!req || !req->prompt) {
    write_error(out->error, sizeof out->error, "prompt is required");
    out->invalid_input = 1;
    return 0;
  }
  if (req->ref_count < 0 || req->ref_count > MINIMAX_H3_IMAGE_MAX_REFS) {
    snprintf(out->error, sizeof out->error,
             "at most %d reference images are supported",
             MINIMAX_H3_IMAGE_MAX_REFS);
    out->invalid_input = 1;
    return 0;
  }
  if (req->ref_count > 0 && !req->refs) {
    write_error(out->error, sizeof out->error, "missing reference image list");
    out->invalid_input = 1;
    return 0;
  }
  if (req->format != IMAGE_FORMAT_PNG && req->format != IMAGE_FORMAT_JPEG) {
    write_error(out->error, sizeof out->error,
                "output_format must be png or jpeg");
    out->invalid_input = 1;
    return 0;
  }
  int quality =
      req->quality == 0 ? MINIMAX_H3_IMAGE_DEFAULT_JPEG_QUALITY : req->quality;
  if (req->format == IMAGE_FORMAT_JPEG &&
      (quality < IMAGE_JPEG_QUALITY_MIN || quality > IMAGE_JPEG_QUALITY_MAX)) {
    snprintf(out->error, sizeof out->error,
             "JPEG quality must be between %d and %d", IMAGE_JPEG_QUALITY_MIN,
             IMAGE_JPEG_QUALITY_MAX);
    out->invalid_input = 1;
    return 0;
  }

  int canvas_w = 0, canvas_h = 0;
  if (!minimax_h3_image_plan_canvas(req->width, req->height,
                                    req->min_generation_area, &canvas_w,
                                    &canvas_h, out->error, sizeof out->error)) {
    out->invalid_input = 1;
    return 0;
  }

  // Decode all reference images before touching GPU weights so invalid image
  // uploads return HTTP 400 immediately.
  uint8_t *decoded_rgb[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  MinimaxH3RefImage ref_images[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  for (int i = 0; i < req->ref_count; i++) {
    const char *why = NULL;
    int rw = 0, rh = 0;
    if (!req->refs[i].data || req->refs[i].len == 0) {
      snprintf(out->error, sizeof out->error, "reference image %d is empty",
               i + 1);
      out->invalid_input = 1;
      for (int k = 0; k < i; k++) free(decoded_rgb[k]);
      return 0;
    }
    decoded_rgb[i] =
        image_decode_rgb(req->refs[i].data, req->refs[i].len, &rw, &rh, &why);
    if (!decoded_rgb[i]) {
      snprintf(out->error, sizeof out->error, "reference image %d: %s", i + 1,
               why ? why : "cannot decode image");
      out->invalid_input = 1;
      for (int k = 0; k < i; k++) free(decoded_rgb[k]);
      return 0;
    }
    ref_images[i] =
        (MinimaxH3RefImage){.rgb = decoded_rgb[i], .width = rw, .height = rh};
  }

  MinimaxH3RunHooks hooks = {.yield_wait = wait,
                             .cancelled = cancelled,
                             .prefill_layer = &active_prefill_layer,
                             .prefill_rows_now = &active_prefill_rows,
                             .prefill_rows_total = &total_prefill_rows,
                             .image_step = &active_image_step,
                             .image_steps_total = &active_image_steps_total,
                             .image_steps_completed = &total_image_steps};
  if (minimax_h3_should_stop(&hooks) || !text_enc || !dit || !vae) {
    for (int i = 0; i < req->ref_count; i++) free(decoded_rgb[i]);
    set_runtime_error(out, &hooks);
    return 0;
  }

  // Reserve the GPU memory needed for the planned canvas and reference images
  // before running the text encoder so oversized canvases refuse early (or
  // fall back cleanly from full residence to staged execution).
  MinimaxH3RefSizes ref_sizes[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  MinimaxH3RefTokens ref_tokens[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  for (int i = 0; i < req->ref_count; i++) {
    ref_sizes[i] =
        minimax_h3_image_ref_sizes(ref_images[i].width, ref_images[i].height);
    ref_tokens[i] = ref_sizes[i].vision;
  }
  int planned_seq_len = minimax_h3_multimodal_prompt_tokens(
      text_enc, req->prompt, ref_tokens, req->ref_count);
  MinimaxH3RequestMemory req_mem = minimax_h3_image_plan_request_memory(
      read_weight_bytes(text_enc, vision_enc, dit, vae),
      minimax_h3_image_stage_bytes(planned_seq_len, ref_sizes, req->ref_count,
                                   canvas_w, canvas_h),
      req->ref_count);
  int request_full = full_resident;
  MinimaxH3Admission admission = minimax_h3_image_admit_request(
      work_slot->account, full_resident, required_staged_bytes, canvas_w,
      canvas_h, req_mem, &request_full, out->error, sizeof out->error);
  if (admission != MINIMAX_H3_ADMITTED) {
    for (int i = 0; i < req->ref_count; i++) free(decoded_rgb[i]);
    if (minimax_h3_should_stop(&hooks)) set_runtime_error(out, &hooks);
    out->invalid_input = admission == MINIMAX_H3_NEVER_FITS;
    return 0;
  }
  if (full_resident && !request_full) pin_core_groups(0);

  Buf text_hidden = {0};
  int text_len = 0;
  int *text_tags = NULL;
  float *ref_lat_bufs[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  MinimaxH3RefLatent dit_refs[MINIMAX_H3_IMAGE_MAX_REFS] = {0};
  float *out_lat = NULL;
  uint8_t *canvas_rgb = NULL;
  uint8_t *final_rgb = NULL;
  MemorySlot *text_slot =
      weight_group_slot(minimax_h3_text_encoder_weights(text_enc));
  MemorySlot *dit_slot = weight_group_slot(minimax_h3_dit_weights(dit));
  MemorySlot *vae_slot = weight_group_slot(minimax_h3_image_vae_weights(vae));
  int ok = 1;

  // Stage 1 (prefill): prompt text and reference vision encoding.
  // In full-residence mode all three core groups remain pinned; in staged mode
  // each stage pins only its own weights so Memory can swap if room is tight.
  active_image_stage = IMAGE_STAGE_PREFILL;
  double prefill_start = now();
  memory_pin(text_slot, 1);
  if (!minimax_h3_text_encoder_load(text_enc)) ok = 0;
  if (ok && req->ref_count > 0) {
    pin_edit_groups(1);
    if (!minimax_h3_vision_encoder_load(vision_enc) ||
        !minimax_h3_image_vae_load_encoder(vae))
      ok = 0;
  }

  if (ok) {
    ok = minimax_h3_encode_multimodal_prompt(
        text_enc, vision_enc, req->prompt, req->ref_count ? ref_images : NULL,
        req->ref_count, work_slot, &text_hidden, &text_len, &text_tags, &hooks);
    if (ok && text_len != planned_seq_len)
      die("MiniMax-H3-Image: the prompt has %d tokens, but its memory plan "
          "counted %d",
          text_len, planned_seq_len);
  }
  double prefill_s = now() - prefill_start;

  // Transition from prompt prefill to image generation for reference VAE
  // encoding, DiT denoising, VAE decoding and output encoding.
  double gen_start = 0.0;
  if (ok) {
    active_image_stage = IMAGE_STAGE_GENERATING;
    gen_start = now();
  }

  for (int i = 0; ok && i < req->ref_count; i++) {
    int vw = 0, vh = 0;
    minimax_h3_vae_ref_size(ref_images[i].width, ref_images[i].height, &vw,
                            &vh);
    const uint8_t *vae_src = ref_images[i].rgb;
    uint8_t *resized = NULL;
    if (vw != ref_images[i].width || vh != ref_images[i].height) {
      resized = image_resize_rgb_lanczos(ref_images[i].rgb, ref_images[i].width,
                                         ref_images[i].height, vw, vh);
      vae_src = resized;
    }
    int rh_lat = vh / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int rw_lat = vw / MINIMAX_H3_VAE_SPATIAL_RATIO;
    size_t floats = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)rh_lat *
                    (size_t)rw_lat;
    ref_lat_bufs[i] = xmalloc(floats * sizeof(float));
    ok = minimax_h3_image_vae_encode(vae, vae_src, vw, vh, ref_lat_bufs[i],
                                     &hooks);
    free(resized);
    dit_refs[i] = (MinimaxH3RefLatent){
        .latents = ref_lat_bufs[i], .h_lat = rh_lat, .w_lat = rw_lat};
  }

  // Raw reference pixels and encoder pins are no longer needed after Stage 1.
  for (int i = 0; i < req->ref_count; i++) {
    free(decoded_rgb[i]);
    decoded_rgb[i] = NULL;
  }
  pin_edit_groups(0);
  if (!request_full) memory_pin(text_slot, 0);

  // Stage 2: 6-step PDD DiT sampling.
  int h_lat = canvas_h / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = canvas_w / MINIMAX_H3_VAE_SPATIAL_RATIO;
  if (ok) {
    memory_pin(dit_slot, 1);
    ok = minimax_h3_dit_load(dit);
  }
  if (ok) {
    size_t lat_floats =
        (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h_lat * (size_t)w_lat;
    out_lat = xmalloc(lat_floats * sizeof(float));
    ok = minimax_h3_dit_sample(dit, &text_hidden, text_len, text_tags,
                               req->ref_count ? dit_refs : NULL, req->ref_count,
                               h_lat, w_lat, req->seed, out_lat, &hooks);
  }
  if (!request_full) memory_pin(dit_slot, 0);

  gpu_free(&text_hidden);
  free(text_tags);
  text_tags = NULL;
  for (int i = 0; i < req->ref_count; i++) {
    free(ref_lat_bufs[i]);
    ref_lat_bufs[i] = NULL;
  }

  // Stage 3: FP16 2D ViT VAE decode.
  if (ok) {
    memory_pin(vae_slot, 1);
    ok = minimax_h3_image_vae_load(vae);
  }
  if (ok) {
    canvas_rgb = xmalloc((size_t)canvas_w * (size_t)canvas_h *
                         (size_t)MINIMAX_H3_RGB_CHANNELS);
    ok = minimax_h3_image_vae_decode(vae, out_lat, h_lat, w_lat, canvas_rgb,
                                     &hooks);
  }
  if (!request_full) memory_pin(vae_slot, 0);
  free(out_lat);
  out_lat = NULL;
  // The GPU work of the request is over: its reservation goes back to the
  // model's base.
  restore_base_minimum(work_slot->account,
                       full_resident && minimax_h3_image_is_loaded()
                           ? required_full_bytes
                           : required_staged_bytes,
                       required_staged_bytes);

  if (!ok) {
    if (gen_start > 0.0) total_image_seconds += now() - gen_start;
    clear_active_progress();
    free(canvas_rgb);
    set_runtime_error(out, &hooks);
    return 0;
  }

  const uint8_t *encode_src = canvas_rgb;
  if (canvas_w != req->width || canvas_h != req->height) {
    final_rgb = image_resize_rgb_lanczos(canvas_rgb, canvas_w, canvas_h,
                                         req->width, req->height);
    free(canvas_rgb);
    canvas_rgb = NULL;
    encode_src = final_rgb;
  }

  const char *enc_err = NULL;
  if (req->format == IMAGE_FORMAT_PNG)
    out->data = image_encode_png(encode_src, req->width, req->height, &out->len,
                                 &enc_err);
  else
    out->data = image_encode_jpeg(encode_src, req->width, req->height, quality,
                                  &out->len, &enc_err);
  free(canvas_rgb);
  free(final_rgb);

  double gen_s = now() - gen_start;
  total_image_seconds += gen_s;
  clear_active_progress();

  if (!out->data) {
    snprintf(out->error, sizeof out->error, "failed to encode output image: %s",
             enc_err ? enc_err : "unknown error");
    return 0;
  }
  out->prompt_tokens = text_len;
  out->prefill_seconds = prefill_s;
  out->image_seconds = gen_s;
  generated_images++;
  return 1;
}

// Writes MiniMax-H3-Image's input and output limits into out.
static void minimax_h3_image_read_limits(ImageLimits *out) {
  *out = (ImageLimits){
      .max_prompt_tokens = MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS,
      .max_dim = MINIMAX_H3_IMAGE_MAX_DIM,
      .min_generation_area = MINIMAX_H3_IMAGE_MIN_AREA,
      .max_generation_area = MINIMAX_H3_IMAGE_MAX_AREA,
      .default_generation_area = MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA,
      .max_canvas_area = MINIMAX_H3_IMAGE_MAX_CANVAS_AREA,
      .max_request_pixels = MINIMAX_H3_IMAGE_MAX_REQUEST_PIXELS,
      .max_images_per_request = MINIMAX_H3_IMAGE_MAX_IMAGES,
      .max_references = MINIMAX_H3_IMAGE_MAX_REFS,
      .min_jpeg_quality = IMAGE_JPEG_QUALITY_MIN,
      .max_jpeg_quality = IMAGE_JPEG_QUALITY_MAX,
      .default_jpeg_quality = MINIMAX_H3_IMAGE_DEFAULT_JPEG_QUALITY,
      .inference_steps = MINIMAX_H3_IMAGE_STEPS,
      .guidance_scale = MINIMAX_H3_IMAGE_GUIDANCE_SCALE};
}

// Allocates a cooperative engine pass with stack space for image generation.
static Pass *minimax_h3_image_pass_new(void (*body)(void *), void *argument) {
  return pass_new(body, argument, PASS_STACK_BYTES);
}

// Starts a cooperative image pass (no per-pass bindings required).
static void minimax_h3_image_pass_begin(void) {}

// Ends a cooperative image pass (no per-pass bindings to release).
static void minimax_h3_image_pass_end(void) {}

// Checkpoint _class_name and model_type identifiers handled by
// MiniMax-H3-Image.
static const char *const minimax_h3_image_types[] = {"MiniMaxH3ImagePipeline",
                                                     "minimax_h3_image", 0};

// ModelOps table registering MiniMax-H3-Image with the engine.
const ModelOps minimax_h3_image_ops = {
    .family = "MiniMax-H3-Image",
    .model_types = minimax_h3_image_types,
    .load = minimax_h3_image_load,
    .dense_weight_slot = minimax_h3_image_weight_slot,
    .is_loaded = minimax_h3_image_is_loaded,
    .set_request_running = minimax_h3_image_set_request_running,
    .read_counters = minimax_h3_image_read_counters,
    .vocab_size = minimax_h3_image_vocab_size,
    .generate_image = minimax_h3_image_generate,
    .read_image_limits = minimax_h3_image_read_limits,
    .run_idle_work = minimax_h3_image_run_idle_work,
    .pass_new = minimax_h3_image_pass_new,
    .pass_begin = minimax_h3_image_pass_begin,
    .pass_end = minimax_h3_image_pass_end};
