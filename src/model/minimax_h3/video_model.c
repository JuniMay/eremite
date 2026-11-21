#include "model/minimax_h3/video_model.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/log.h"
#include "base/shutdown.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/resample.h"
#include "engine/weights.h"
#include "model/minimax_h3/audio_vae.h"
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/lora.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include "model/minimax_h3/video_vae.h"
#include "model/minimax_h3/vision_encoder.h"
#include "platform/image.h"
#include "platform/video.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// MiniMax-H3 Video behind ModelOps: supports T2VA, FL2VA and REF2VA with
// optional BF16 LoRA, native H.264/AAC MP4 output, exact per-request
// memory planning and strict stage separation (Qwen3-VL is prefill; DiT and
// VAEs are video generation, never reported as prefill or token decoding).

// Stack size in bytes for one MiniMax-H3 Video engine pass.
static constexpr size_t PASS_STACK_BYTES = 1u << 20;

// Maximum path buffer length in bytes when resolving checkpoint files.
static constexpr size_t PATH_CAP = 2048;

// Bytes in one gibibyte for admission and log messages.
static constexpr double BYTES_PER_GIB = (double)(1ull << 30);

// 2 fps Qwen3-VL sampling stride from 24 fps reference video frames;
// Comfy-Org/ComfyUI comfy_extras/nodes_minimax_h3.py:1259 (FPS // 2).
static constexpr int REF_VIDEO_QWEN_FPS = 2,
                     REF_VIDEO_QWEN_STRIDE =
                         MINIMAX_H3_VIDEO_FPS / REF_VIDEO_QWEN_FPS;

// Video VAE clip grid constants used to crop reference videos down to 17k + 5.
static constexpr int REF_VIDEO_CLIP_STEP = 17, REF_VIDEO_CLIP_MIN = 5;

// Qwen3-VL items contributed per reference video (video block + optional audio
// soundtrack label).
static constexpr int REF_VIDEO_MAX_QWEN_ITEMS = 2;

// Maximum conditioning items passed to Qwen3-VL in one REF2VA request:
// up to 9 images + 3 videos (each with an optional audio label) + 3 audios.
static constexpr int MAX_QWEN_COND_ITEMS =
    MINIMAX_H3_VIDEO_MAX_REF_IMAGES +
    MINIMAX_H3_VIDEO_MAX_REF_VIDEOS * REF_VIDEO_MAX_QWEN_ITEMS +
    MINIMAX_H3_VIDEO_MAX_REF_AUDIOS;

// Maximum FL2VA keyframe anchors (first frame and last frame).
static constexpr int MAX_FL2VA_KEYFRAMES = 2;

// Shared GPU device and workspace slot for the active MiniMax-H3 Video model.
static Gpu *video_gpu;
static MemorySlot *video_work_slot;

// Sub-model handles owned by the single MiniMax-H3 Video instance.
static MinimaxH3TextEncoder *v_text_enc;
static MinimaxH3VisionEncoder *v_vision_enc;
static MinimaxH3Dit *v_dit;
static MinimaxH3Lora *v_lora;
static MinimaxH3VideoVae *v_video_vae;
static MinimaxH3AudioVae *v_audio_vae;

// Configured checkpoint variant and sampling defaults.
static VideoVariant active_variant = VIDEO_VARIANT_FL2VA;
static int configured_default_steps = MINIMAX_H3_VIDEO_DEFAULT_STEPS;
static double configured_shift_video = MINIMAX_H3_VIDEO_SIGMA_SHIFT;
static double configured_shift_audio = MINIMAX_H3_AUDIO_SIGMA_SHIFT;

// Baseline full-residence and staged memory minimums computed at open.
static uint64_t required_full_bytes, required_staged_bytes;

// Nonzero when all core weight groups fit resident together.
static int full_resident;

// Cumulative counters since open (strictly separating prompt prefill from
// video generation).
static uint64_t total_prefill_rows, generated_videos, total_video_steps;
static double total_video_seconds;

// Live progress indicators read by eremite_read_stats.
static int active_prefill_layer, active_prefill_rows;
static int active_video_stage, active_video_step, active_video_steps_total;

// Clears live per-call progress indicators back to idle.
static void clear_active_progress(void) {
  active_video_stage = VIDEO_STAGE_NONE;
  active_prefill_layer = 0;
  active_prefill_rows = 0;
  active_video_step = 0;
  active_video_steps_total = 0;
}

// Writes message into dst when non-null and non-empty.
static void write_error(char *dst, size_t cap, const char *message) {
  if (dst && cap > 0) snprintf(dst, cap, "%s", message);
}

// Returns the larger of two unsigned 64-bit byte counts.
static uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }

// Rounds value to the nearest positive multiple of MINIMAX_H3_VIDEO_PATCH_PX.
static int round_patch_px(double value) {
  int r = (int)llround(value / (double)MINIMAX_H3_VIDEO_PATCH_PX) *
          MINIMAX_H3_VIDEO_PATCH_PX;
  return r < MINIMAX_H3_VIDEO_PATCH_PX ? MINIMAX_H3_VIDEO_PATCH_PX : r;
}

void minimax_h3_video_adapt_canvas(int width, int height, int *out_w,
                                   int *out_h) {
  double ratio = (double)width / (double)height;
  double nom_w = ratio >= 1.0 ? (double)MINIMAX_H3_VIDEO_BASE_SHORT_EDGE * ratio
                              : (double)MINIMAX_H3_VIDEO_BASE_SHORT_EDGE;
  double nom_h = ratio >= 1.0
                     ? (double)MINIMAX_H3_VIDEO_BASE_SHORT_EDGE
                     : (double)MINIMAX_H3_VIDEO_BASE_SHORT_EDGE / ratio;
  if (nom_w * nom_h > (double)MINIMAX_H3_VIDEO_ADAPT_MAX_PIXELS) {
    double s =
        sqrt((double)MINIMAX_H3_VIDEO_ADAPT_MAX_PIXELS / (nom_w * nom_h));
    nom_w *= s;
    nom_h *= s;
  }
  *out_w = round_patch_px(nom_w);
  *out_h = round_patch_px(nom_h);
}

int minimax_h3_video_plan_canvas(int width, int height, int *out_w, int *out_h,
                                 char *error, size_t error_cap) {
  if (width == 0 && height == 0) {
    if (out_w) *out_w = MINIMAX_H3_VIDEO_DEFAULT_WIDTH;
    if (out_h) *out_h = MINIMAX_H3_VIDEO_DEFAULT_HEIGHT;
    return 1;
  }
  if (width < 1 || height < 1 || width > MINIMAX_H3_VIDEO_MAX_DIM ||
      height > MINIMAX_H3_VIDEO_MAX_DIM) {
    if (error && error_cap > 0)
      snprintf(error, error_cap, "video dimensions must be between 1 and %d",
               MINIMAX_H3_VIDEO_MAX_DIM);
    return 0;
  }
  int cw = round_patch_px((double)width);
  int ch = round_patch_px((double)height);
  if (cw > MINIMAX_H3_VIDEO_MAX_DIM || ch > MINIMAX_H3_VIDEO_MAX_DIM) {
    if (error && error_cap > 0)
      snprintf(error, error_cap, "video canvas %dx%d exceeds %d px limit", cw,
               ch, MINIMAX_H3_VIDEO_MAX_DIM);
    return 0;
  }
  if ((int64_t)cw * (int64_t)ch > MINIMAX_H3_VIDEO_MAX_CANVAS_AREA) {
    if (error && error_cap > 0)
      snprintf(error, error_cap, "video canvas %dx%d exceeds %lld pixel limit",
               cw, ch, (long long)MINIMAX_H3_VIDEO_MAX_CANVAS_AREA);
    return 0;
  }
  if (out_w) *out_w = cw;
  if (out_h) *out_h = ch;
  return 1;
}

int minimax_h3_video_plan_frames(int seconds, int frames, int *out_frames,
                                 int *out_t_lat, int *out_audio_t, char *error,
                                 size_t error_cap) {
  int raw = 0;
  if (frames < 0 || seconds < 0) {
    write_error(error, error_cap, "video duration must be positive");
    return 0;
  }
  if (frames > 0) {
    if (frames > MINIMAX_H3_VIDEO_MAX_FRAMES) {
      if (error && error_cap > 0)
        snprintf(error, error_cap, "video frame count %d exceeds %d limit",
                 frames, MINIMAX_H3_VIDEO_MAX_FRAMES);
      return 0;
    }
    raw = frames;
  } else if (seconds > 0) {
    if (seconds < MINIMAX_H3_VIDEO_MIN_SECONDS ||
        seconds > MINIMAX_H3_VIDEO_MAX_SECONDS) {
      if (error && error_cap > 0)
        snprintf(error, error_cap, "video seconds must be between %d and %d",
                 MINIMAX_H3_VIDEO_MIN_SECONDS, MINIMAX_H3_VIDEO_MAX_SECONDS);
      return 0;
    }
    raw = seconds * MINIMAX_H3_VIDEO_FPS;
  } else {
    raw = MINIMAX_H3_VIDEO_DEFAULT_FRAMES;
  }
  int aligned = minimax_h3_video_align_frames(raw);
  if (aligned > MINIMAX_H3_VIDEO_MAX_FRAMES)
    aligned = MINIMAX_H3_VIDEO_MAX_FRAMES;
  if (out_frames) *out_frames = aligned;
  if (out_t_lat) *out_t_lat = minimax_h3_video_latent_t(aligned);
  if (out_audio_t) *out_audio_t = minimax_h3_video_audio_t(aligned);
  return 1;
}

MinimaxH3RequestMemory minimax_h3_video_plan_request_memory(
    MinimaxH3VideoWeightBytes weights, MinimaxH3VideoStageBytes stages,
    int needs_vision, int needs_video_enc, int needs_audio_enc) {
  uint64_t core = weights.text + weights.dit + weights.lora +
                  weights.video_vae + weights.audio_vae;
  uint64_t enc_extra = (needs_vision ? weights.vision : 0) +
                       (needs_video_enc ? weights.video_vae_enc : 0) +
                       (needs_audio_enc ? weights.audio_vae_enc : 0);
  uint64_t decode_ws = max_u64(stages.decode_video, stages.decode_audio);
  uint64_t full =
      max_u64(max_u64(core + enc_extra + stages.encode, core + stages.sample),
              core + decode_ws);

  uint64_t stage1 = weights.text + enc_extra + stages.encode;
  uint64_t stage2 = weights.dit + weights.lora + stages.sample;
  uint64_t stage3 = max_u64(weights.video_vae + stages.decode_video,
                            weights.audio_vae + stages.decode_audio);
  uint64_t staged = max_u64(max_u64(stage1, stage2), stage3);
  return (MinimaxH3RequestMemory){.full_bytes = full, .staged_bytes = staged};
}

// Waits for account minimum bytes if they fit within the current memory limit.
static int try_admit_bytes(MemoryAccount *account, uint64_t bytes) {
  if (bytes <= memory_counted(account) &&
      bytes > memory_current_limit(account->memory))
    return 0;
  return memory_wait_for_minimum(account, bytes);
}

MinimaxH3Admission minimax_h3_video_admit_request(
    MemoryAccount *account, int prefer_full, uint64_t staged_base, int width,
    int height, int frames, MinimaxH3RequestMemory req_mem, int *out_use_full,
    char *error, size_t error_cap) {
  uint64_t ceiling = memory_ceiling(account->memory);
  if (account->limit < ceiling) ceiling = account->limit;
  uint64_t staged_target = max_u64(req_mem.staged_bytes, staged_base);
  if (staged_target > ceiling) {
    if (error && error_cap > 0)
      snprintf(error, error_cap,
               "video %dx%d (%d frames) needs %.1f GiB of memory, more than "
               "the %.1f GiB that this model may use",
               width, height, frames, (double)staged_target / BYTES_PER_GIB,
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
             "video %dx%d (%d frames) needs %.1f GiB of memory, which does "
             "not fit now",
             width, height, frames, (double)staged_target / BYTES_PER_GIB);
  return MINIMAX_H3_NOT_NOW;
}

// Reads resident weight requirements of all eight MiniMax-H3 Video groups.
static MinimaxH3VideoWeightBytes read_weight_bytes(void) {
  return (MinimaxH3VideoWeightBytes){
      .text = minimax_h3_text_encoder_required_bytes(v_text_enc),
      .vision = minimax_h3_vision_encoder_required_bytes(v_vision_enc),
      .dit = minimax_h3_dit_required_bytes(v_dit),
      .lora = minimax_h3_lora_required_bytes(v_lora),
      .video_vae = minimax_h3_video_vae_required_bytes(v_video_vae),
      .video_vae_enc = minimax_h3_video_vae_encoder_required_bytes(v_video_vae),
      .audio_vae = minimax_h3_audio_vae_required_bytes(v_audio_vae),
      .audio_vae_enc =
          minimax_h3_audio_vae_encoder_required_bytes(v_audio_vae)};
}

// Resolves the directory holding tokenizer.json under dir into out.
static void resolve_tokenizer_dir(const char *dir, char *out, size_t cap) {
  char probe[PATH_CAP];
  snprintf(probe, sizeof probe, "%s/tokenizer.json", dir);
  if (access(probe, R_OK) == 0) {
    snprintf(out, cap, "%s", dir);
    return;
  }
  snprintf(probe, sizeof probe, "%s/tokenizer/tokenizer.json", dir);
  if (access(probe, R_OK) == 0) {
    snprintf(out, cap, "%s/tokenizer", dir);
    return;
  }
  snprintf(out, cap, "%s/processor", dir);
}

// Pins or unpins the core video generation weight groups together.
static void pin_core_groups(int pinned) {
  if (v_text_enc)
    memory_pin(weight_group_slot(minimax_h3_text_encoder_weights(v_text_enc)),
               pinned);
  if (v_dit)
    memory_pin(weight_group_slot(minimax_h3_dit_weights(v_dit)), pinned);
  if (v_lora)
    memory_pin(weight_group_slot(minimax_h3_lora_weights(v_lora)), pinned);
  if (v_video_vae)
    memory_pin(weight_group_slot(minimax_h3_video_vae_weights(v_video_vae)),
               pinned);
  if (v_audio_vae)
    memory_pin(weight_group_slot(minimax_h3_audio_vae_weights(v_audio_vae)),
               pinned);
}

// Pins or unpins the on-demand conditioning encoder weight groups.
static void pin_cond_groups(int pinned) {
  if (v_vision_enc)
    memory_pin(
        weight_group_slot(minimax_h3_vision_encoder_weights(v_vision_enc)),
        pinned);
  if (v_video_vae)
    memory_pin(
        weight_group_slot(minimax_h3_video_vae_encoder_weights(v_video_vae)),
        pinned);
  if (v_audio_vae)
    memory_pin(
        weight_group_slot(minimax_h3_audio_vae_encoder_weights(v_audio_vae)),
        pinned);
}

// Restores account minimum to base, or staged_base if base no longer fits.
static void restore_base_minimum(MemoryAccount *account, uint64_t base,
                                 uint64_t staged_base) {
  if (!memory_set_minimum(account, base, MEMORY_REQUIRED))
    memory_set_minimum(account, staged_base, MEMORY_REQUIRED);
}

// Resizes an RGB image with Lanczos keeping aspect ratio to cover (tw, th),
// then center-crops to exact (tw, th); matches comfy.utils.common_upscale
// "center" crop (comfy_extras/nodes_minimax_h3.py:1073).
static uint8_t *resize_cover_crop_rgb(const uint8_t *rgb, int sw, int sh,
                                      int tw, int th) {
  if (sw == tw && sh == th) {
    size_t bytes = (size_t)tw * (size_t)th * (size_t)MINIMAX_H3_RGB_CHANNELS;
    uint8_t *copy = xmalloc(bytes);
    memcpy(copy, rgb, bytes);
    return copy;
  }
  double scale = fmax((double)tw / (double)sw, (double)th / (double)sh);
  int rw = (int)llround((double)sw * scale);
  int rh = (int)llround((double)sh * scale);
  if (rw < tw) rw = tw;
  if (rh < th) rh = th;
  uint8_t *scaled = image_resize_rgb_lanczos(rgb, sw, sh, rw, rh);
  if (rw == tw && rh == th) return scaled;
  uint8_t *cropped =
      xmalloc((size_t)tw * (size_t)th * (size_t)MINIMAX_H3_RGB_CHANNELS);
  int x0 = (rw - tw) / 2, y0 = (rh - th) / 2;
  size_t dst_stride = (size_t)tw * (size_t)MINIMAX_H3_RGB_CHANNELS;
  size_t src_stride = (size_t)rw * (size_t)MINIMAX_H3_RGB_CHANNELS;
  for (int y = 0; y < th; y++)
    memcpy(cropped + (size_t)y * dst_stride,
           scaled + (size_t)(y0 + y) * src_stride +
               (size_t)x0 * (size_t)MINIMAX_H3_RGB_CHANNELS,
           dst_stride);
  free(scaled);
  return cropped;
}

// Opens MiniMax-H3 Video sub-models on first call and loads core weights in
// full-residence mode, falling back to staged loading when memory is tight.
static int minimax_h3_video_load(Gpu *gpu, MemorySlot *work, const char *dir,
                                 const Options *o) {
  if (!v_text_enc) {
    if (!o || !o->minimax_h3_video.text_encoder || !o->minimax_h3_video.dit ||
        !o->minimax_h3_video.vae || !o->minimax_h3_video.audio_vae)
      die("%s: MiniMax-H3 Video requires text_encoder, dit, vae and audio_vae "
          "paths",
          dir);
    video_gpu = gpu;
    video_work_slot = work;
    active_variant = o->minimax_h3_video.variant;
    configured_default_steps = o->minimax_h3_video.default_steps > 0
                                   ? o->minimax_h3_video.default_steps
                                   : MINIMAX_H3_VIDEO_DEFAULT_STEPS;
    configured_shift_video = o->minimax_h3_video.shift_video > 0.0
                                 ? o->minimax_h3_video.shift_video
                                 : MINIMAX_H3_VIDEO_SIGMA_SHIFT;
    configured_shift_audio = o->minimax_h3_video.shift_audio > 0.0
                                 ? o->minimax_h3_video.shift_audio
                                 : MINIMAX_H3_AUDIO_SIGMA_SHIFT;

    char tok_dir[PATH_CAP];
    resolve_tokenizer_dir(dir, tok_dir, sizeof tok_dir);
    v_text_enc = minimax_h3_text_encoder_open(
        gpu, work->account, o->minimax_h3_video.text_encoder, tok_dir);
    v_vision_enc = minimax_h3_vision_encoder_open(v_text_enc, work->account);
    v_dit = minimax_h3_dit_open(gpu, work->account, o->minimax_h3_video.dit);
    if (!minimax_h3_dit_has_audio(v_dit))
      die("%s: DiT checkpoint has no audio stream (expected FL2VA or REF2VA "
          "video checkpoint)",
          o->minimax_h3_video.dit);
    if (o->minimax_h3_video.lora && o->minimax_h3_video.lora[0]) {
      float strength = o->minimax_h3_video.lora_strength > 0.0f
                           ? o->minimax_h3_video.lora_strength
                           : 1.0f;
      v_lora = minimax_h3_lora_open(gpu, work->account,
                                    o->minimax_h3_video.lora, strength);
    }
    v_video_vae =
        minimax_h3_video_vae_open(gpu, work->account, o->minimax_h3_video.vae);
    v_audio_vae = minimax_h3_audio_vae_open(gpu, work->account,
                                            o->minimax_h3_video.audio_vae);

    // Full-residence budget covers the default 1344x768 124-frame T2VA clip.
    // Staged baseline covers the smallest valid clip so smaller videos can load
    // and run when memory is tighter; each request still reserves its exact
    // canvas and frame budget before Qwen3-VL runs.
    int def_t = minimax_h3_video_latent_t(MINIMAX_H3_VIDEO_DEFAULT_FRAMES);
    int def_at = minimax_h3_video_audio_t(MINIMAX_H3_VIDEO_DEFAULT_FRAMES);
    int def_frame_rows =
        (MINIMAX_H3_VIDEO_DEFAULT_WIDTH / MINIMAX_H3_VIDEO_PATCH_PX) *
        (MINIMAX_H3_VIDEO_DEFAULT_HEIGHT / MINIMAX_H3_VIDEO_PATCH_PX);
    int def_vis = def_t * def_frame_rows;
    int def_aud = def_at * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
    uint64_t text_hidden_bytes = (uint64_t)MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS *
                                 (uint64_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
    MinimaxH3VideoStageBytes def_stages = {
        .encode = minimax_h3_text_encoder_workspace_bytes(
                      MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS, 0) +
                  text_hidden_bytes,
        .sample = minimax_h3_dit_video_workspace_bytes(
                      0, def_vis, def_vis, def_aud, def_aud, v_lora) +
                  text_hidden_bytes,
        .decode_video = minimax_h3_video_vae_decode_workspace_bytes(),
        .decode_audio = minimax_h3_audio_vae_decode_workspace_bytes(def_at)};
    MinimaxH3VideoWeightBytes wb = read_weight_bytes();
    MinimaxH3RequestMemory full_mem =
        minimax_h3_video_plan_request_memory(wb, def_stages, 0, 0, 0);
    required_full_bytes = full_mem.full_bytes;

    int min_t = minimax_h3_video_latent_t(MINIMAX_H3_VIDEO_MIN_FRAMES);
    int min_at = minimax_h3_video_audio_t(MINIMAX_H3_VIDEO_MIN_FRAMES);
    int min_aud = min_at * MINIMAX_H3_AUDIO_STEREO_CHANNELS;
    MinimaxH3VideoStageBytes min_stages = {
        .encode = minimax_h3_text_encoder_workspace_bytes(1, 0),
        .sample = minimax_h3_dit_video_workspace_bytes(0, min_t, min_t, min_aud,
                                                       min_aud, v_lora),
        .decode_video = minimax_h3_video_vae_decode_workspace_bytes(),
        .decode_audio = minimax_h3_audio_vae_decode_workspace_bytes(min_at)};
    MinimaxH3RequestMemory min_mem =
        minimax_h3_video_plan_request_memory(wb, min_stages, 0, 0, 0);
    required_staged_bytes = min_mem.staged_bytes;
  }

  // Primary mode: load and keep all core video weight groups resident.
  pin_core_groups(1);
  if (memory_wait_for_minimum(video_work_slot->account, required_full_bytes) &&
      minimax_h3_text_encoder_load(v_text_enc) && minimax_h3_dit_load(v_dit) &&
      minimax_h3_lora_load(v_lora) && minimax_h3_video_vae_load(v_video_vae) &&
      minimax_h3_audio_vae_load(v_audio_vae)) {
    full_resident = 1;
    return 1;
  }
  pin_core_groups(0);
  memory_set_minimum(video_work_slot->account, 0, MEMORY_REQUIRED);
  if (shutdown_requested()) return 0;

  // Fallback mode: confirm largest single stage fits, then warm text encoder.
  MemorySlot *text_slot =
      weight_group_slot(minimax_h3_text_encoder_weights(v_text_enc));
  memory_pin(text_slot, 1);
  if (!memory_wait_for_minimum(video_work_slot->account,
                               required_staged_bytes) ||
      !minimax_h3_text_encoder_load(v_text_enc)) {
    memory_pin(text_slot, 0);
    memory_set_minimum(video_work_slot->account, 0, MEMORY_REQUIRED);
    return 0;
  }
  memory_pin(text_slot, 0);
  full_resident = 0;
  say("%s: full residence (%.1f GiB) did not fit; running staged (%.1f GiB)",
      dir, (double)required_full_bytes / BYTES_PER_GIB,
      (double)required_staged_bytes / BYTES_PER_GIB);
  return 1;
}

// Returns null because MiniMax-H3 Video manages multiple weight groups.
static MemorySlot *minimax_h3_video_weight_slot(void) { return NULL; }

// Returns nonzero when the active residency mode's weight groups are resident.
static int minimax_h3_video_is_loaded(void) {
  if (!v_text_enc || !v_dit || !v_video_vae || !v_audio_vae) return 0;
  int t_ok = memory_ready(
      weight_group_slot(minimax_h3_text_encoder_weights(v_text_enc)));
  int d_ok = memory_ready(weight_group_slot(minimax_h3_dit_weights(v_dit)));
  int l_ok = !v_lora ||
             memory_ready(weight_group_slot(minimax_h3_lora_weights(v_lora)));
  int vv_ok = memory_ready(
      weight_group_slot(minimax_h3_video_vae_weights(v_video_vae)));
  int av_ok = memory_ready(
      weight_group_slot(minimax_h3_audio_vae_weights(v_audio_vae)));
  if (full_resident) return t_ok && d_ok && l_ok && vv_ok && av_ok;
  return t_ok || d_ok || vv_ok || av_ok;
}

// Pins core weight groups while a request runs and unpins all groups when idle.
static void minimax_h3_video_set_request_running(int running) {
  if (running) {
    if (full_resident) pin_core_groups(1);
    return;
  }
  pin_core_groups(0);
  pin_cond_groups(0);
}

// Copies MiniMax-H3 Video counters, live stage progress and resident bytes.
static void minimax_h3_video_read_counters(ModelCounters *c) {
  uint64_t resident = 0;
  if (v_text_enc)
    resident += minimax_h3_text_encoder_resident_bytes(v_text_enc);
  if (v_vision_enc)
    resident += minimax_h3_vision_encoder_resident_bytes(v_vision_enc);
  if (v_dit) resident += minimax_h3_dit_resident_bytes(v_dit);
  if (v_lora) resident += minimax_h3_lora_resident_bytes(v_lora);
  if (v_video_vae) {
    resident += minimax_h3_video_vae_resident_bytes(v_video_vae);
    resident += minimax_h3_video_vae_encoder_resident_bytes(v_video_vae);
  }
  if (v_audio_vae) {
    resident += minimax_h3_audio_vae_resident_bytes(v_audio_vae);
    resident += minimax_h3_audio_vae_encoder_resident_bytes(v_audio_vae);
  }
  *c = (ModelCounters){.prefill_rows = total_prefill_rows,
                       .prefill_layer = active_prefill_layer,
                       .prefill_rows_now = active_prefill_rows,
                       .generated_videos = generated_videos,
                       .video_steps = total_video_steps,
                       .video_seconds = total_video_seconds,
                       .video_stage = active_video_stage,
                       .video_step = active_video_step,
                       .video_steps_total = active_video_steps_total,
                       .layers = MINIMAX_H3_TEXT_LAYERS,
                       .dense_bytes = resident,
                       .required_bytes = required_staged_bytes};
}

// Returns the Qwen3-VL tokenizer vocabulary size.
static int minimax_h3_video_vocab_size(void) { return MINIMAX_H3_IMAGE_VOCAB; }

// Returns 0 because MiniMax-H3 Video has no routed expert cache.
static int minimax_h3_video_run_idle_work(void) { return 0; }

// Writes MiniMax-H3 Video limits into out.
static void minimax_h3_video_read_limits(VideoLimits *out) {
  *out = (VideoLimits){.max_prompt_tokens = MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS,
                       .patch_px = MINIMAX_H3_VIDEO_PATCH_PX,
                       .max_dim = MINIMAX_H3_VIDEO_MAX_DIM,
                       .default_width = MINIMAX_H3_VIDEO_DEFAULT_WIDTH,
                       .default_height = MINIMAX_H3_VIDEO_DEFAULT_HEIGHT,
                       .max_canvas_area = MINIMAX_H3_VIDEO_MAX_CANVAS_AREA,
                       .fps = MINIMAX_H3_VIDEO_FPS,
                       .min_frames = MINIMAX_H3_VIDEO_MIN_FRAMES,
                       .max_frames = MINIMAX_H3_VIDEO_MAX_FRAMES,
                       .default_frames = MINIMAX_H3_VIDEO_DEFAULT_FRAMES,
                       .min_seconds = MINIMAX_H3_VIDEO_MIN_SECONDS,
                       .max_seconds = MINIMAX_H3_VIDEO_MAX_SECONDS,
                       .default_seconds = MINIMAX_H3_VIDEO_DEFAULT_SECONDS,
                       .min_steps = MINIMAX_H3_VIDEO_MIN_STEPS,
                       .max_steps = MINIMAX_H3_VIDEO_MAX_STEPS,
                       .default_steps = configured_default_steps,
                       .max_ref_images = MINIMAX_H3_VIDEO_MAX_REF_IMAGES,
                       .max_ref_videos = MINIMAX_H3_VIDEO_MAX_REF_VIDEOS,
                       .max_ref_audios = MINIMAX_H3_VIDEO_MAX_REF_AUDIOS,
                       .max_total_refs = MINIMAX_H3_VIDEO_MAX_TOTAL_REFS,
                       .variant = active_variant,
                       .guidance_scale = MINIMAX_H3_VIDEO_GUIDANCE_SCALE,
                       .default_shift_video = configured_shift_video,
                       .default_shift_audio = configured_shift_audio};
}

// Writes a runtime failure message matching cancellation, shutdown or OOM.
static void set_runtime_error(VideoResult *out,
                              const MinimaxH3RunHooks *hooks) {
  if (hooks && hooks->cancelled && *hooks->cancelled)
    write_error(out->error, sizeof out->error, "request cancelled");
  else if (shutdown_requested())
    write_error(out->error, sizeof out->error, "the server is stopping");
  else
    write_error(out->error, sizeof out->error,
                "video generation failed: memory is short");
}

// Decoded and resized conditioning inputs owned for one video request.
typedef struct {
  // FL2VA resized first/last frame RGB buffers [canvas_h, canvas_w, 3].
  uint8_t *kf_rgb[MAX_FL2VA_KEYFRAMES];
  // Resolved target frame index of each FL2VA keyframe.
  int kf_frame_idx[MAX_FL2VA_KEYFRAMES];
  // Encoded Video VAE latents [24, 1, h_lat, w_lat] for each FL2VA keyframe.
  float *kf_latents[MAX_FL2VA_KEYFRAMES];
  // Active FL2VA keyframe count (0..2).
  int kf_count;

  // REF2VA resized reference image RGB buffers.
  uint8_t *ref_img_rgb[MINIMAX_H3_VIDEO_MAX_REF_IMAGES];
  // REF2VA resized reference image widths in pixels.
  int ref_img_w[MINIMAX_H3_VIDEO_MAX_REF_IMAGES];
  // REF2VA resized reference image heights in pixels.
  int ref_img_h[MINIMAX_H3_VIDEO_MAX_REF_IMAGES];
  // Encoded Video VAE latents for each REF2VA reference image.
  float *ref_img_lat[MINIMAX_H3_VIDEO_MAX_REF_IMAGES];
  // Active REF2VA reference image count.
  int ref_img_count;

  // REF2VA resized 24 fps reference video RGB buffers.
  uint8_t *ref_vid_rgb[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA 2 fps subsampled reference video RGB buffers for Qwen3-VL.
  uint8_t *ref_vid_qwen_rgb[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA resized reference video widths in pixels.
  int ref_vid_w[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA resized reference video heights in pixels.
  int ref_vid_h[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA reference video frame counts at 24 fps.
  int ref_vid_frames[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA reference video frame counts after 2 fps subsampling.
  int ref_vid_qwen_frames[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // REF2VA reference video 32 kHz planar stereo PCM soundtracks.
  float *ref_vid_pcm[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // Per-channel PCM sample counts of ref_vid_pcm.
  int ref_vid_pcm_samples[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // Encoded Video VAE latents for each REF2VA reference video.
  float *ref_vid_lat[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // Encoded Audio VAE latents for each REF2VA reference video soundtrack.
  float *ref_vid_aud_lat[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // Temporal latent frame counts for each REF2VA reference video.
  int ref_vid_t_lat[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // 40 Hz audio latent frame counts for each REF2VA reference video soundtrack.
  int ref_vid_audio_t[MINIMAX_H3_VIDEO_MAX_REF_VIDEOS];
  // Active REF2VA reference video count.
  int ref_vid_count;

  // REF2VA standalone reference audio 32 kHz planar stereo PCM buffers.
  float *ref_aud_pcm[MINIMAX_H3_VIDEO_MAX_REF_AUDIOS];
  // Per-channel PCM sample counts of ref_aud_pcm.
  int ref_aud_samples[MINIMAX_H3_VIDEO_MAX_REF_AUDIOS];
  // Encoded Audio VAE latents for each standalone reference audio clip.
  float *ref_aud_lat[MINIMAX_H3_VIDEO_MAX_REF_AUDIOS];
  // 40 Hz audio latent frame counts for each standalone reference audio clip.
  int ref_aud_t[MINIMAX_H3_VIDEO_MAX_REF_AUDIOS];
  // Active REF2VA standalone reference audio count.
  int ref_aud_count;

  // Qwen3-VL conditioning item array referencing buffers above.
  MinimaxH3VideoCondItem qwen_items[MAX_QWEN_COND_ITEMS];
  // Active entry count in qwen_items.
  int qwen_item_count;

  // DiT FL2VA keyframe descriptors referencing encoded latents above.
  MinimaxH3VideoKeyframe dit_kfs[MAX_FL2VA_KEYFRAMES];
  // DiT REF2VA reference descriptors referencing encoded latents above.
  MinimaxH3VideoRef dit_refs[MINIMAX_H3_VIDEO_MAX_TOTAL_REFS];
  // Active entry count in dit_refs.
  int dit_ref_count;
} VideoCondState;

// Frees all heap buffers owned by VideoCondState.
static void video_cond_free(VideoCondState *st) {
  for (int i = 0; i < MAX_FL2VA_KEYFRAMES; i++) {
    free(st->kf_rgb[i]);
    free(st->kf_latents[i]);
  }
  for (int i = 0; i < MINIMAX_H3_VIDEO_MAX_REF_IMAGES; i++) {
    free(st->ref_img_rgb[i]);
    free(st->ref_img_lat[i]);
  }
  for (int i = 0; i < MINIMAX_H3_VIDEO_MAX_REF_VIDEOS; i++) {
    free(st->ref_vid_rgb[i]);
    free(st->ref_vid_qwen_rgb[i]);
    free(st->ref_vid_pcm[i]);
    free(st->ref_vid_lat[i]);
    free(st->ref_vid_aud_lat[i]);
  }
  for (int i = 0; i < MINIMAX_H3_VIDEO_MAX_REF_AUDIOS; i++) {
    free(st->ref_aud_pcm[i]);
    free(st->ref_aud_lat[i]);
  }
  memset(st, 0, sizeof *st);
}

// Decodes and prepares all FL2VA or REF2VA conditioning inputs on CPU. Returns
// 1 on success or 0 with invalid_input and error set on client input error.
static int prepare_conditioning(const VideoRequest *req, int canvas_w,
                                int canvas_h, int frame_count,
                                VideoCondState *st, VideoResult *out) {
  int has_first = req->first_frame.data != NULL && req->first_frame.len > 0;
  int has_last = req->last_frame.data != NULL && req->last_frame.len > 0;
  int has_refs = req->ref_image_count > 0 || req->ref_video_count > 0 ||
                 req->ref_audio_count > 0;

  if (active_variant == VIDEO_VARIANT_FL2VA && has_refs) {
    out->invalid_input = 1;
    write_error(out->error, sizeof out->error,
                "reference images, videos and audio require a REF2VA model");
    return 0;
  }
  if (active_variant == VIDEO_VARIANT_REF2VA && (has_first || has_last)) {
    out->invalid_input = 1;
    write_error(out->error, sizeof out->error,
                "first_frame and last_frame require an FL2VA model");
    return 0;
  }
  if (req->ref_image_count < 0 ||
      req->ref_image_count > MINIMAX_H3_VIDEO_MAX_REF_IMAGES ||
      req->ref_video_count < 0 ||
      req->ref_video_count > MINIMAX_H3_VIDEO_MAX_REF_VIDEOS ||
      req->ref_audio_count < 0 ||
      req->ref_audio_count > MINIMAX_H3_VIDEO_MAX_REF_AUDIOS ||
      req->ref_image_count + req->ref_video_count + req->ref_audio_count >
          MINIMAX_H3_VIDEO_MAX_TOTAL_REFS) {
    out->invalid_input = 1;
    snprintf(out->error, sizeof out->error,
             "reference counts exceed limits (images <= %d, videos <= %d, "
             "audios <= %d, total <= %d)",
             MINIMAX_H3_VIDEO_MAX_REF_IMAGES, MINIMAX_H3_VIDEO_MAX_REF_VIDEOS,
             MINIMAX_H3_VIDEO_MAX_REF_AUDIOS, MINIMAX_H3_VIDEO_MAX_TOTAL_REFS);
    return 0;
  }

  // FL2VA first and last keyframes
  // (comfy_extras/nodes_minimax_h3.py:1066-1076).
  if (has_first) {
    int iw = 0, ih = 0;
    const char *why = NULL;
    uint8_t *raw = image_decode_rgb(req->first_frame.data, req->first_frame.len,
                                    &iw, &ih, &why);
    if (!raw) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "first_frame: %s",
               why ? why : "decode failed");
      return 0;
    }
    uint8_t *resized =
        (iw == canvas_w && ih == canvas_h)
            ? raw
            : image_resize_rgb_lanczos(raw, iw, ih, canvas_w, canvas_h);
    if (resized != raw) free(raw);
    int k = st->kf_count++;
    st->kf_rgb[k] = resized;
    st->kf_frame_idx[k] = 0;
    st->dit_kfs[k] = (MinimaxH3VideoKeyframe){.frame_index = 0, .t_lat = 1};
    st->qwen_items[st->qwen_item_count++] =
        (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
                                 .rgb = resized,
                                 .width = canvas_w,
                                 .height = canvas_h,
                                 .frame_count = 1};
  }
  if (has_last) {
    int iw = 0, ih = 0;
    const char *why = NULL;
    uint8_t *raw = image_decode_rgb(req->last_frame.data, req->last_frame.len,
                                    &iw, &ih, &why);
    if (!raw) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "last_frame: %s",
               why ? why : "decode failed");
      return 0;
    }
    uint8_t *cropped = resize_cover_crop_rgb(raw, iw, ih, canvas_w, canvas_h);
    free(raw);
    int k = st->kf_count++;
    st->kf_rgb[k] = cropped;
    st->kf_frame_idx[k] = frame_count - 1;
    st->dit_kfs[k] =
        (MinimaxH3VideoKeyframe){.frame_index = frame_count - 1, .t_lat = 1};
    st->qwen_items[st->qwen_item_count++] =
        (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
                                 .rgb = cropped,
                                 .width = canvas_w,
                                 .height = canvas_h,
                                 .frame_count = 1};
  }

  // REF2VA images (comfy_extras/nodes_minimax_h3.py:1218-1234).
  for (int i = 0; i < req->ref_image_count; i++) {
    if (!req->ref_images || !req->ref_images[i].data ||
        req->ref_images[i].len == 0) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_image %d is empty", i + 1);
      return 0;
    }
    int iw = 0, ih = 0;
    const char *why = NULL;
    uint8_t *raw = image_decode_rgb(req->ref_images[i].data,
                                    req->ref_images[i].len, &iw, &ih, &why);
    if (!raw) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_image %d: %s", i + 1,
               why ? why : "decode failed");
      return 0;
    }
    double scale = 1.0;
    if (req->ref_image_max) {
      int short_edge = iw < ih ? iw : ih;
      scale = fmin(1.0, (double)MINIMAX_H3_VIDEO_REF_IMAGE_MAX_SHORT_EDGE /
                            (double)short_edge);
    } else {
      scale = fmin(1.0, sqrt(((double)canvas_w * (double)canvas_h) /
                             ((double)iw * (double)ih)));
    }
    int tw = round_patch_px((double)iw * scale);
    int th = round_patch_px((double)ih * scale);
    uint8_t *resized = (iw == tw && ih == th)
                           ? raw
                           : image_resize_rgb_lanczos(raw, iw, ih, tw, th);
    if (resized != raw) free(raw);
    int idx = st->ref_img_count++;
    st->ref_img_rgb[idx] = resized;
    st->ref_img_w[idx] = tw;
    st->ref_img_h[idx] = th;
    st->qwen_items[st->qwen_item_count++] =
        (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_IMAGE,
                                 .rgb = resized,
                                 .width = tw,
                                 .height = th,
                                 .frame_count = 1};
    st->dit_refs[st->dit_ref_count++] =
        (MinimaxH3VideoRef){.kind = MINIMAX_H3_VIDEO_REF_IMAGE,
                            .t_lat = 1,
                            .h_lat = th / MINIMAX_H3_VAE_SPATIAL_RATIO,
                            .w_lat = tw / MINIMAX_H3_VAE_SPATIAL_RATIO};
  }

  // REF2VA videos (comfy_extras/nodes_minimax_h3.py:1236-1272).
  for (int i = 0; i < req->ref_video_count; i++) {
    if (!req->ref_videos || !req->ref_videos[i].data ||
        req->ref_videos[i].len == 0) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_video %d is empty", i + 1);
      return 0;
    }
    VideoDecoded dec = {0};
    const char *why = NULL;
    if (!video_decode_mp4(req->ref_videos[i].data, req->ref_videos[i].len,
                          MINIMAX_H3_VIDEO_FPS, MINIMAX_H3_AUDIO_SAMPLE_RATE,
                          &dec, &why)) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_video %d: %s", i + 1,
               why ? why : "decode failed");
      return 0;
    }
    int n = dec.frame_count > frame_count ? frame_count : dec.frame_count;
    if (n < REF_VIDEO_CLIP_MIN) {
      video_decoded_free(&dec);
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error,
               "ref_video %d needs at least %d frames at %d fps", i + 1,
               REF_VIDEO_CLIP_MIN, MINIMAX_H3_VIDEO_FPS);
      return 0;
    }
    while (n % REF_VIDEO_CLIP_STEP != REF_VIDEO_CLIP_MIN) n--;

    int cw = 0, ch = 0;
    minimax_h3_video_adapt_canvas(dec.width, dec.height, &cw, &ch);
    if ((int64_t)dec.width * (int64_t)dec.height < (int64_t)cw * (int64_t)ch) {
      cw = round_patch_px((double)dec.width);
      ch = round_patch_px((double)dec.height);
    }
    size_t dst_frame_bytes =
        (size_t)cw * (size_t)ch * (size_t)MINIMAX_H3_RGB_CHANNELS;
    size_t src_frame_bytes = (size_t)dec.width * (size_t)dec.height *
                             (size_t)MINIMAX_H3_RGB_CHANNELS;
    uint8_t *resized = xmalloc((size_t)n * dst_frame_bytes);
    for (int f = 0; f < n; f++) {
      const uint8_t *sf = dec.rgb + (size_t)f * src_frame_bytes;
      uint8_t *df = resized + (size_t)f * dst_frame_bytes;
      if (dec.width == cw && dec.height == ch) {
        memcpy(df, sf, dst_frame_bytes);
      } else {
        uint8_t *rf =
            image_resize_rgb_lanczos(sf, dec.width, dec.height, cw, ch);
        memcpy(df, rf, dst_frame_bytes);
        free(rf);
      }
    }

    // Paired soundtrack: explicit ref_video_audios[i] overrides container
    // audio.
    float *pcm = NULL;
    int pcm_samples = 0;
    if (req->ref_video_audios && req->ref_video_audios[i].data &&
        req->ref_video_audios[i].len > 0) {
      pcm = audio_decode_pcm(req->ref_video_audios[i].data,
                             req->ref_video_audios[i].len,
                             MINIMAX_H3_AUDIO_SAMPLE_RATE, &pcm_samples, &why);
      if (!pcm) {
        free(resized);
        video_decoded_free(&dec);
        out->invalid_input = 1;
        snprintf(out->error, sizeof out->error, "ref_video_audio %d: %s", i + 1,
                 why ? why : "decode failed");
        return 0;
      }
    } else if (dec.audio_pcm && dec.audio_samples > 0) {
      pcm = dec.audio_pcm;
      pcm_samples = dec.audio_samples;
      dec.audio_pcm = NULL;
      dec.audio_samples = 0;
    }
    video_decoded_free(&dec);

    // Subsample reference video at 2 fps for Qwen3-VL conditioning.
    int qwen_n = (n + REF_VIDEO_QWEN_STRIDE - 1) / REF_VIDEO_QWEN_STRIDE;
    uint8_t *qwen_rgb = xmalloc((size_t)qwen_n * dst_frame_bytes);
    for (int q = 0; q < qwen_n; q++)
      memcpy(qwen_rgb + (size_t)q * dst_frame_bytes,
             resized + (size_t)(q * REF_VIDEO_QWEN_STRIDE) * dst_frame_bytes,
             dst_frame_bytes);

    int idx = st->ref_vid_count++;
    st->ref_vid_rgb[idx] = resized;
    st->ref_vid_qwen_rgb[idx] = qwen_rgb;
    st->ref_vid_w[idx] = cw;
    st->ref_vid_h[idx] = ch;
    st->ref_vid_frames[idx] = n;
    st->ref_vid_qwen_frames[idx] = qwen_n;
    st->ref_vid_pcm[idx] = pcm;
    st->ref_vid_pcm_samples[idx] = pcm_samples;
    st->ref_vid_t_lat[idx] = minimax_h3_video_vae_encode_latent_t(n);
    st->ref_vid_audio_t[idx] =
        pcm ? minimax_h3_audio_vae_encoded_frames(pcm_samples) : 0;

    // Paired soundtrack gets <Audio j>: immediately before <Video k>:.
    if (pcm)
      st->qwen_items[st->qwen_item_count++] =
          (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_AUDIO};
    st->qwen_items[st->qwen_item_count++] =
        (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_VIDEO,
                                 .rgb = qwen_rgb,
                                 .width = cw,
                                 .height = ch,
                                 .frame_count = qwen_n};
    st->dit_refs[st->dit_ref_count++] = (MinimaxH3VideoRef){
        .kind = st->ref_vid_audio_t[idx] > 0 ? MINIMAX_H3_VIDEO_REF_VIDEO_AUDIO
                                             : MINIMAX_H3_VIDEO_REF_VIDEO,
        .t_lat = st->ref_vid_t_lat[idx],
        .h_lat = ch / MINIMAX_H3_VAE_SPATIAL_RATIO,
        .w_lat = cw / MINIMAX_H3_VAE_SPATIAL_RATIO,
        .audio_t = st->ref_vid_audio_t[idx]};
  }

  // REF2VA standalone audios (comfy_extras/nodes_minimax_h3.py:1273-1280).
  for (int i = 0; i < req->ref_audio_count; i++) {
    if (!req->ref_audios || !req->ref_audios[i].data ||
        req->ref_audios[i].len == 0) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_audio %d is empty", i + 1);
      return 0;
    }
    int samples = 0;
    const char *why = NULL;
    float *pcm =
        audio_decode_pcm(req->ref_audios[i].data, req->ref_audios[i].len,
                         MINIMAX_H3_AUDIO_SAMPLE_RATE, &samples, &why);
    if (!pcm) {
      out->invalid_input = 1;
      snprintf(out->error, sizeof out->error, "ref_audio %d: %s", i + 1,
               why ? why : "decode failed");
      return 0;
    }
    int idx = st->ref_aud_count++;
    st->ref_aud_pcm[idx] = pcm;
    st->ref_aud_samples[idx] = samples;
    st->ref_aud_t[idx] = minimax_h3_audio_vae_encoded_frames(samples);
    st->qwen_items[st->qwen_item_count++] =
        (MinimaxH3VideoCondItem){.kind = MINIMAX_H3_VIDEO_COND_AUDIO};
    st->dit_refs[st->dit_ref_count++] = (MinimaxH3VideoRef){
        .kind = MINIMAX_H3_VIDEO_REF_AUDIO, .audio_t = st->ref_aud_t[idx]};
  }
  return 1;
}

// Computes exact stage workspace bytes for a planned video request.
static MinimaxH3VideoStageBytes
plan_stage_bytes(const MinimaxH3VideoPromptLayout *qwen_layout,
                 const MinimaxH3PackedLayout *dit_layout,
                 const VideoCondState *st, int canvas_w, int canvas_h,
                 int audio_t) {
  uint64_t hidden = (uint64_t)qwen_layout->seq_len *
                    (uint64_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  uint64_t encode_ws = qwen_layout->peak_bytes;

  // Video VAE and Audio VAE condition encoding run beside text hidden rows.
  for (int i = 0; i < st->kf_count; i++)
    encode_ws = max_u64(encode_ws, minimax_h3_video_vae_encode_workspace_bytes(
                                       1, canvas_w, canvas_h) +
                                       hidden);
  for (int i = 0; i < st->ref_img_count; i++)
    encode_ws = max_u64(encode_ws, minimax_h3_video_vae_encode_workspace_bytes(
                                       1, st->ref_img_w[i], st->ref_img_h[i]) +
                                       hidden);
  for (int i = 0; i < st->ref_vid_count; i++) {
    encode_ws = max_u64(encode_ws, minimax_h3_video_vae_encode_workspace_bytes(
                                       st->ref_vid_frames[i], st->ref_vid_w[i],
                                       st->ref_vid_h[i]) +
                                       hidden);
    if (st->ref_vid_pcm[i])
      encode_ws =
          max_u64(encode_ws, minimax_h3_audio_vae_encode_workspace_bytes(
                                 st->ref_vid_pcm_samples[i]) +
                                 hidden);
  }
  for (int i = 0; i < st->ref_aud_count; i++)
    encode_ws = max_u64(encode_ws, minimax_h3_audio_vae_encode_workspace_bytes(
                                       st->ref_aud_samples[i]) +
                                       hidden);

  return (MinimaxH3VideoStageBytes){
      .encode = encode_ws,
      .sample = minimax_h3_dit_video_workspace_bytes(
                    qwen_layout->seq_len, dit_layout->total_vis,
                    dit_layout->target_video_rows, dit_layout->total_audio,
                    dit_layout->target_audio_rows, v_lora) +
                hidden,
      .decode_video = minimax_h3_video_vae_decode_workspace_bytes(),
      .decode_audio = minimax_h3_audio_vae_decode_workspace_bytes(audio_t)};
}

// Encodes all visual and audio condition clips via Video VAE and Audio VAE
// encoders and binds the resulting latent pointers into st->dit_kfs/dit_refs.
static int encode_condition_latents(VideoCondState *st, int canvas_w,
                                    int canvas_h, int use_full,
                                    const MinimaxH3RunHooks *hooks) {
  int needs_v_enc =
      st->kf_count > 0 || st->ref_img_count > 0 || st->ref_vid_count > 0;
  int needs_a_enc = st->ref_aud_count > 0;
  for (int i = 0; i < st->ref_vid_count; i++)
    if (st->ref_vid_pcm[i]) needs_a_enc = 1;

  if (needs_v_enc) {
    MemorySlot *ve_slot =
        weight_group_slot(minimax_h3_video_vae_encoder_weights(v_video_vae));
    memory_pin(ve_slot, 1);
    int h_lat = canvas_h / MINIMAX_H3_VAE_SPATIAL_RATIO;
    int w_lat = canvas_w / MINIMAX_H3_VAE_SPATIAL_RATIO;
    for (int i = 0; i < st->kf_count; i++) {
      size_t n = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)h_lat *
                 (size_t)w_lat;
      st->kf_latents[i] = xmalloc(n * sizeof(float));
      if (!minimax_h3_video_vae_encode(v_video_vae, st->kf_rgb[i], 1, canvas_w,
                                       canvas_h, st->kf_latents[i], hooks)) {
        memory_pin(ve_slot, 0);
        return 0;
      }
      st->dit_kfs[i].video_latents = st->kf_latents[i];
    }
    int ref_idx = 0;
    for (int i = 0; i < st->ref_img_count; i++) {
      int rh = st->ref_img_h[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int rw = st->ref_img_w[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      size_t n =
          (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS * (size_t)rh * (size_t)rw;
      st->ref_img_lat[i] = xmalloc(n * sizeof(float));
      if (!minimax_h3_video_vae_encode(v_video_vae, st->ref_img_rgb[i], 1,
                                       st->ref_img_w[i], st->ref_img_h[i],
                                       st->ref_img_lat[i], hooks)) {
        memory_pin(ve_slot, 0);
        return 0;
      }
      st->dit_refs[ref_idx++].video_latents = st->ref_img_lat[i];
    }
    for (int i = 0; i < st->ref_vid_count; i++) {
      int rh = st->ref_vid_h[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      int rw = st->ref_vid_w[i] / MINIMAX_H3_VAE_SPATIAL_RATIO;
      size_t n = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                 (size_t)st->ref_vid_t_lat[i] * (size_t)rh * (size_t)rw;
      st->ref_vid_lat[i] = xmalloc(n * sizeof(float));
      if (!minimax_h3_video_vae_encode(
              v_video_vae, st->ref_vid_rgb[i], st->ref_vid_frames[i],
              st->ref_vid_w[i], st->ref_vid_h[i], st->ref_vid_lat[i], hooks)) {
        memory_pin(ve_slot, 0);
        return 0;
      }
      st->dit_refs[ref_idx++].video_latents = st->ref_vid_lat[i];
    }
    memory_pin(ve_slot, 0);
  }

  if (needs_a_enc) {
    MemorySlot *ae_slot =
        weight_group_slot(minimax_h3_audio_vae_encoder_weights(v_audio_vae));
    memory_pin(ae_slot, 1);
    int ref_idx = st->ref_img_count;
    for (int i = 0; i < st->ref_vid_count; i++) {
      if (st->ref_vid_pcm[i]) {
        size_t n = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                   (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                   (size_t)st->ref_vid_audio_t[i];
        st->ref_vid_aud_lat[i] = xmalloc(n * sizeof(float));
        if (!minimax_h3_audio_vae_encode(v_audio_vae, st->ref_vid_pcm[i],
                                         st->ref_vid_pcm_samples[i],
                                         st->ref_vid_aud_lat[i], hooks)) {
          memory_pin(ae_slot, 0);
          return 0;
        }
        st->dit_refs[ref_idx].audio_latents = st->ref_vid_aud_lat[i];
      }
      ref_idx++;
    }
    for (int i = 0; i < st->ref_aud_count; i++) {
      size_t n = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                 (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                 (size_t)st->ref_aud_t[i];
      st->ref_aud_lat[i] = xmalloc(n * sizeof(float));
      if (!minimax_h3_audio_vae_encode(v_audio_vae, st->ref_aud_pcm[i],
                                       st->ref_aud_samples[i],
                                       st->ref_aud_lat[i], hooks)) {
        memory_pin(ae_slot, 0);
        return 0;
      }
      st->dit_refs[ref_idx++].audio_latents = st->ref_aud_lat[i];
    }
    memory_pin(ae_slot, 0);
  }
  (void)use_full;
  return 1;
}

// Runs one MiniMax-H3 audio-video generation call (T2VA, FL2VA or REF2VA).
static int minimax_h3_video_generate(const VideoRequest *req,
                                     const PassWait *wait, const int *cancelled,
                                     VideoResult *out) {
  memset(out, 0, sizeof *out);
  if (!req || !req->prompt) {
    out->invalid_input = 1;
    write_error(out->error, sizeof out->error, "prompt is required");
    return 0;
  }

  int steps = req->steps > 0 ? req->steps : configured_default_steps;
  if (steps < MINIMAX_H3_VIDEO_MIN_STEPS ||
      steps > MINIMAX_H3_VIDEO_MAX_STEPS) {
    out->invalid_input = 1;
    snprintf(out->error, sizeof out->error, "steps must be between %d and %d",
             MINIMAX_H3_VIDEO_MIN_STEPS, MINIMAX_H3_VIDEO_MAX_STEPS);
    return 0;
  }
  double shift_v =
      req->shift_video > 0.0 ? req->shift_video : configured_shift_video;
  double shift_a =
      req->shift_audio > 0.0 ? req->shift_audio : configured_shift_audio;

  // Plan canvas dimensions; if unset and an input reference image is provided,
  // adapt canvas aspect ratio from that reference image (768p short edge).
  int req_w = req->width, req_h = req->height;
  if (req_w == 0 && req_h == 0) {
    const VideoMediaInput *anchor = NULL;
    if (req->first_frame.data && req->first_frame.len > 0)
      anchor = &req->first_frame;
    else if (req->ref_image_count > 0 && req->ref_images &&
             req->ref_images[0].data && req->ref_images[0].len > 0)
      anchor = &req->ref_images[0];
    if (anchor) {
      int iw = 0, ih = 0;
      const char *why = NULL;
      if (image_read_size(anchor->data, anchor->len, &iw, &ih, &why))
        minimax_h3_video_adapt_canvas(iw, ih, &req_w, &req_h);
    }
  }
  int canvas_w = 0, canvas_h = 0;
  if (!minimax_h3_video_plan_canvas(req_w, req_h, &canvas_w, &canvas_h,
                                    out->error, sizeof out->error)) {
    out->invalid_input = 1;
    return 0;
  }

  int frame_count = 0, t_lat = 0, audio_t = 0;
  if (!minimax_h3_video_plan_frames(req->seconds, req->frames, &frame_count,
                                    &t_lat, &audio_t, out->error,
                                    sizeof out->error)) {
    out->invalid_input = 1;
    return 0;
  }

  VideoCondState cond = {0};
  if (!prepare_conditioning(req, canvas_w, canvas_h, frame_count, &cond, out)) {
    video_cond_free(&cond);
    return 0;
  }

  if (!v_text_enc || !v_dit || !v_video_vae || !v_audio_vae) {
    video_cond_free(&cond);
    write_error(out->error, sizeof out->error,
                "video model weights are not loaded");
    return 0;
  }

  // Build exact CPU token and DiT layouts before reserving memory so oversized
  // requests fail early with a clear message before running Qwen3-VL.
  MinimaxH3VideoPromptLayout qwen_layout;
  minimax_h3_video_prompt_build_layout(
      minimax_h3_text_encoder_tokenizer(v_text_enc), req->prompt,
      cond.qwen_items, cond.qwen_item_count, &qwen_layout);

  int h_lat = canvas_h / MINIMAX_H3_VAE_SPATIAL_RATIO;
  int w_lat = canvas_w / MINIMAX_H3_VAE_SPATIAL_RATIO;
  MinimaxH3PackedLayout dit_layout;
  minimax_h3_packed_layout_build(qwen_layout.seq_len, qwen_layout.tags, t_lat,
                                 h_lat, w_lat, audio_t, cond.dit_kfs,
                                 cond.kf_count, cond.dit_refs,
                                 cond.dit_ref_count, &dit_layout);

  int needs_vision = qwen_layout.vision_blocks > 0;
  int needs_v_enc =
      cond.kf_count > 0 || cond.ref_img_count > 0 || cond.ref_vid_count > 0;
  int needs_a_enc = cond.ref_aud_count > 0;
  for (int i = 0; i < cond.ref_vid_count; i++)
    if (cond.ref_vid_pcm[i]) needs_a_enc = 1;

  MinimaxH3VideoStageBytes stages = plan_stage_bytes(
      &qwen_layout, &dit_layout, &cond, canvas_w, canvas_h, audio_t);
  MinimaxH3RequestMemory req_mem = minimax_h3_video_plan_request_memory(
      read_weight_bytes(), stages, needs_vision, needs_v_enc, needs_a_enc);
  minimax_h3_packed_layout_free(&dit_layout);
  minimax_h3_video_prompt_layout_free(&qwen_layout);

  uint64_t base_min = video_work_slot->account->minimum;
  int use_full = 0;
  MinimaxH3Admission adm = minimax_h3_video_admit_request(
      video_work_slot->account, full_resident, required_staged_bytes, canvas_w,
      canvas_h, frame_count, req_mem, &use_full, out->error, sizeof out->error);
  if (adm != MINIMAX_H3_ADMITTED) {
    out->invalid_input = (adm == MINIMAX_H3_NEVER_FITS);
    video_cond_free(&cond);
    return 0;
  }

  if (use_full) {
    pin_core_groups(1);
  } else {
    pin_core_groups(0);
    pin_cond_groups(0);
  }

  MinimaxH3RunHooks hooks = {.yield_wait = wait,
                             .cancelled = cancelled,
                             .prefill_layer = &active_prefill_layer,
                             .prefill_rows_now = &active_prefill_rows,
                             .prefill_rows_total = &total_prefill_rows,
                             .image_step = &active_video_step,
                             .image_steps_total = &active_video_steps_total,
                             .image_steps_completed = &total_video_steps};

  MemorySlot *text_slot =
      weight_group_slot(minimax_h3_text_encoder_weights(v_text_enc));
  MemorySlot *vis_slot =
      weight_group_slot(minimax_h3_vision_encoder_weights(v_vision_enc));
  MemorySlot *dit_slot = weight_group_slot(minimax_h3_dit_weights(v_dit));
  MemorySlot *lora_slot =
      v_lora ? weight_group_slot(minimax_h3_lora_weights(v_lora)) : NULL;
  MemorySlot *vvae_slot =
      weight_group_slot(minimax_h3_video_vae_weights(v_video_vae));
  MemorySlot *avae_slot =
      weight_group_slot(minimax_h3_audio_vae_weights(v_audio_vae));

  // Stage 1: Qwen3-VL text/vision prompt encoding (the ONLY prefill stage).
  clear_active_progress();
  active_video_stage = VIDEO_STAGE_PREFILL;
  double t_prefill_start = now();

  memory_pin(text_slot, 1);
  if (needs_vision) memory_pin(vis_slot, 1);
  Buf text_hidden = {0};
  int seq_len = 0;
  int *text_tags = NULL;
  int ok = minimax_h3_encode_video_prompt(v_text_enc, v_vision_enc, req->prompt,
                                          cond.qwen_items, cond.qwen_item_count,
                                          video_work_slot, &text_hidden,
                                          &seq_len, &text_tags, &hooks);
  double prefill_s = now() - t_prefill_start;
  active_prefill_layer = 0;
  active_prefill_rows = 0;

  memory_pin(vis_slot, 0);
  if (!use_full) memory_pin(text_slot, 0);
  if (!ok) {
    clear_active_progress();
    gpu_free(&text_hidden);
    free(text_tags);
    video_cond_free(&cond);
    restore_base_minimum(video_work_slot->account, base_min,
                         required_staged_bytes);
    set_runtime_error(out, &hooks);
    return 0;
  }

  // Stage 2 & 3: conditioning VAE encoding, DiT sampling, Video/Audio VAE
  // decoding and MP4 muxing (reported strictly as video generation).
  active_video_stage = VIDEO_STAGE_GENERATING;
  double t_gen_start = now();

  ok = encode_condition_latents(&cond, canvas_w, canvas_h, use_full, &hooks);
  if (!ok) {
    clear_active_progress();
    gpu_free(&text_hidden);
    free(text_tags);
    video_cond_free(&cond);
    restore_base_minimum(video_work_slot->account, base_min,
                         required_staged_bytes);
    set_runtime_error(out, &hooks);
    return 0;
  }

  size_t video_lat_floats = (size_t)MINIMAX_H3_VAE_LATENT_CHANNELS *
                            (size_t)t_lat * (size_t)h_lat * (size_t)w_lat;
  size_t audio_lat_floats = (size_t)MINIMAX_H3_AUDIO_LATENT_CHANNELS *
                            (size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                            (size_t)audio_t;
  float *video_latents = xmalloc(video_lat_floats * sizeof(float));
  float *audio_latents = xmalloc(audio_lat_floats * sizeof(float));

  memory_pin(dit_slot, 1);
  if (lora_slot) memory_pin(lora_slot, 1);
  MinimaxH3VideoSampleParams sp = {.text_hidden = &text_hidden,
                                   .text_len = seq_len,
                                   .text_tags = text_tags,
                                   .keyframes = cond.dit_kfs,
                                   .keyframe_count = cond.kf_count,
                                   .refs = cond.dit_refs,
                                   .ref_count = cond.dit_ref_count,
                                   .t_lat = t_lat,
                                   .h_lat = h_lat,
                                   .w_lat = w_lat,
                                   .audio_t = audio_t,
                                   .steps = steps,
                                   .shift_video = shift_v,
                                   .shift_audio = shift_a,
                                   .seed = req->seed,
                                   .lora = v_lora,
                                   .out_video_latents = video_latents,
                                   .out_audio_latents = audio_latents,
                                   .hooks = &hooks};
  ok = minimax_h3_dit_sample_video(v_dit, &sp);

  gpu_free(&text_hidden);
  free(text_tags);
  video_cond_free(&cond);
  if (!use_full) {
    memory_pin(dit_slot, 0);
    if (lora_slot) memory_pin(lora_slot, 0);
  }
  if (!ok) {
    free(audio_latents);
    free(video_latents);
    clear_active_progress();
    restore_base_minimum(video_work_slot->account, base_min,
                         required_staged_bytes);
    set_runtime_error(out, &hooks);
    return 0;
  }

  // Decode video latents -> RGB frames.
  MinimaxH3VvaeTemporalPlan tplan = minimax_h3_video_vae_decode_plan(t_lat);
  int decoded_frames = tplan.out_frames;
  size_t rgb_bytes = (size_t)decoded_frames * (size_t)canvas_h *
                     (size_t)canvas_w * (size_t)MINIMAX_H3_RGB_CHANNELS;
  uint8_t *rgb = xmalloc(rgb_bytes);
  memory_pin(vvae_slot, 1);
  ok = minimax_h3_video_vae_decode(v_video_vae, video_latents, t_lat, h_lat,
                                   w_lat, rgb, &hooks);
  free(video_latents);
  if (!use_full) memory_pin(vvae_slot, 0);
  if (!ok) {
    free(rgb);
    free(audio_latents);
    clear_active_progress();
    restore_base_minimum(video_work_slot->account, base_min,
                         required_staged_bytes);
    set_runtime_error(out, &hooks);
    return 0;
  }

  // Decode audio latents -> 32 kHz planar stereo FP32 PCM.
  int total_pcm_samples = audio_t * MINIMAX_H3_AUDIO_HOP_LENGTH;
  float *pcm = xmalloc((size_t)MINIMAX_H3_AUDIO_STEREO_CHANNELS *
                       (size_t)total_pcm_samples * sizeof(float));
  memory_pin(avae_slot, 1);
  ok = minimax_h3_audio_vae_decode(v_audio_vae, audio_latents, audio_t, pcm,
                                   &hooks);
  free(audio_latents);
  if (!use_full) memory_pin(avae_slot, 0);
  if (!ok) {
    free(pcm);
    free(rgb);
    clear_active_progress();
    restore_base_minimum(video_work_slot->account, base_min,
                         required_staged_bytes);
    set_runtime_error(out, &hooks);
    return 0;
  }

  // Trim planar stereo PCM to match the exact decoded video duration.
  int target_samples =
      (int)llround(((double)decoded_frames / (double)MINIMAX_H3_VIDEO_FPS) *
                   (double)MINIMAX_H3_AUDIO_SAMPLE_RATE);
  int keep_samples =
      target_samples < total_pcm_samples ? target_samples : total_pcm_samples;
  if (keep_samples < total_pcm_samples)
    memmove(pcm + keep_samples, pcm + total_pcm_samples,
            (size_t)keep_samples * sizeof(float));

  // Mux H.264 video + 32 kHz stereo AAC into MP4.
  size_t mp4_len = 0;
  const char *enc_err = NULL;
  uint8_t *mp4 = video_encode_mp4(
      rgb, decoded_frames, canvas_w, canvas_h, MINIMAX_H3_VIDEO_FPS, pcm,
      keep_samples, MINIMAX_H3_AUDIO_SAMPLE_RATE, &mp4_len, &enc_err);
  free(pcm);
  free(rgb);
  double video_s = now() - t_gen_start;
  clear_active_progress();
  restore_base_minimum(video_work_slot->account, base_min,
                       required_staged_bytes);

  if (!mp4) {
    snprintf(out->error, sizeof out->error, "MP4 encoding failed: %s",
             enc_err ? enc_err : "unknown error");
    return 0;
  }

  generated_videos++;
  total_video_seconds += video_s;
  out->data = mp4;
  out->len = mp4_len;
  out->width = canvas_w;
  out->height = canvas_h;
  out->frames = decoded_frames;
  out->fps = MINIMAX_H3_VIDEO_FPS;
  out->prompt_tokens = seq_len;
  out->prefill_seconds = prefill_s;
  out->video_seconds = video_s;
  return 1;
}

// Allocates a Pass running body(argument) on a 1 MiB coroutine stack.
static Pass *minimax_h3_video_pass_new(void (*body)(void *), void *argument) {
  return pass_new(body, argument, PASS_STACK_BYTES);
}

// No-op per-pass begin hook: video passes manage GPU buffers explicitly.
static void minimax_h3_video_pass_begin(void) {}

// No-op per-pass end hook: video passes manage GPU buffers explicitly.
static void minimax_h3_video_pass_end(void) {}

// Pipeline class names from MiniMaxAI/MiniMax-H3 model_index.json and
// {FL2VA,Ref2VA}/model_index.json.
static const char *const minimax_h3_video_types[] = {"MiniMaxH3ModularPipeline",
                                                     "MiniMaxH3Pipeline", NULL};

const ModelOps minimax_h3_video_ops = {
    .family = "MiniMax-H3-Video",
    .model_types = minimax_h3_video_types,
    .load = minimax_h3_video_load,
    .dense_weight_slot = minimax_h3_video_weight_slot,
    .is_loaded = minimax_h3_video_is_loaded,
    .set_request_running = minimax_h3_video_set_request_running,
    .read_counters = minimax_h3_video_read_counters,
    .vocab_size = minimax_h3_video_vocab_size,
    .generate_video = minimax_h3_video_generate,
    .read_video_limits = minimax_h3_video_read_limits,
    .run_idle_work = minimax_h3_video_run_idle_work,
    .pass_new = minimax_h3_video_pass_new,
    .pass_begin = minimax_h3_video_pass_begin,
    .pass_end = minimax_h3_video_pass_end,
};
