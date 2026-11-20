#include "engine/memory.h"
#include "eremite.h"
#include "model/minimax_h3/dit.h"
#include "model/minimax_h3/image_vae.h"
#include "model/minimax_h3/model.h"
#include "model/minimax_h3/quant.h"
#include "model/minimax_h3/text_encoder.h"
#include "platform/image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Unit tests for MiniMax-H3-Image canvas planning and run-hook cancellation.

static int failures;

// Records a check outcome and prints a failure line when cond is zero.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

int main(void) {
  int cw = 0, ch = 0;
  char err[256] = {0};

  // Default min_generation_area (4194304 = 2048x2048) upscales small canvases.
  check(minimax_h3_image_plan_canvas(512, 512, 0, &cw, &ch, err, sizeof err) &&
            cw == 2048 && ch == 2048,
        "512x512 with default min_generation_area upscales to 2048x2048");

  // Non-square small size preserves aspect ratio and reaches >= min_area.
  check(minimax_h3_image_plan_canvas(1000, 500, 0, &cw, &ch, err, sizeof err) &&
            cw % 32 == 0 && ch % 32 == 0 &&
            (int64_t)cw * (int64_t)ch >= MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA &&
            fabs((double)cw / (double)ch - 2.0) < 0.05,
        "1000x500 with default min_area reaches >= 4 MP at ~2:1 aspect");

  // Standard 16 MP sizes remain exact multiples of 32 without upscaling.
  check(
      minimax_h3_image_plan_canvas(4096, 4096, 0, &cw, &ch, err, sizeof err) &&
          cw == 4096 && ch == 4096,
      "4096x4096 stays exact 4096x4096");
  check(
      minimax_h3_image_plan_canvas(4736, 3552, 0, &cw, &ch, err, sizeof err) &&
          cw == 4736 && ch == 3552,
      "4736x3552 stays exact 4736x3552");
  check(
      minimax_h3_image_plan_canvas(6272, 2688, 0, &cw, &ch, err, sizeof err) &&
          cw == 6272 && ch == 2688,
      "6272x2688 stays exact 6272x2688");

  // Passing min_generation_area = 1024 disables automatic 4 MP upscaling.
  check(
      minimax_h3_image_plan_canvas(512, 512, 1024, &cw, &ch, err, sizeof err) &&
          cw == 512 && ch == 512,
      "512x512 with min_area=1024 stays 512x512");
  check(
      minimax_h3_image_plan_canvas(500, 500, 1024, &cw, &ch, err, sizeof err) &&
          cw == 512 && ch == 512,
      "500x500 with min_area=1024 rounds canvas to 512x512");
  check(minimax_h3_image_plan_canvas(32, 32, 1024, &cw, &ch, err, sizeof err) &&
            cw == 32 && ch == 32,
        "32x32 with min_area=1024 stays 32x32");
  check(minimax_h3_image_plan_canvas(16, 16, 1024, &cw, &ch, err, sizeof err) &&
            cw == 32 && ch == 32,
        "16x16 with min_area=1024 expands to 32x32 minimum patch");

  // Invalid dimensions and min_generation_area bounds fail cleanly with text.
  err[0] = '\0';
  check(
      !minimax_h3_image_plan_canvas(0, 512, 1024, &cw, &ch, err, sizeof err) &&
          err[0] != '\0',
      "zero width is rejected");
  err[0] = '\0';
  check(
      !minimax_h3_image_plan_canvas(512, -1, 1024, &cw, &ch, err, sizeof err) &&
          err[0] != '\0',
      "negative height is rejected");
  err[0] = '\0';
  check(!minimax_h3_image_plan_canvas(9000, 512, 1024, &cw, &ch, err,
                                      sizeof err) &&
            err[0] != '\0',
        "width above 8192 is rejected");
  err[0] = '\0';
  check(
      !minimax_h3_image_plan_canvas(512, 512, 512, &cw, &ch, err, sizeof err) &&
          err[0] != '\0',
      "min_generation_area below 1024 is rejected");
  err[0] = '\0';
  check(!minimax_h3_image_plan_canvas(512, 512, 20000000, &cw, &ch, err,
                                      sizeof err) &&
            err[0] != '\0',
        "min_generation_area above 16777216 is rejected");
  err[0] = '\0';
  check(!minimax_h3_image_plan_canvas(1, 8192, 0, &cw, &ch, err, sizeof err) &&
            err[0] != '\0',
        "extreme aspect ratio exceeding max canvas dimension is rejected");
  err[0] = '\0';
  check(!minimax_h3_image_plan_canvas(6000, 6000, 1024, &cw, &ch, err,
                                      sizeof err) &&
            err[0] != '\0',
        "canvas above 24 MP pixel limit is rejected");

  // Cooperative stop hooks honor the cancellation flag.
  int cancelled = 0;
  MinimaxH3RunHooks hooks = {.yield_wait = NULL, .cancelled = &cancelled};
  check(!minimax_h3_should_stop(NULL), "null hooks do not request stop");
  check(!minimax_h3_should_stop(&hooks), "uncancelled hooks do not stop");
  cancelled = 1;
  check(minimax_h3_should_stop(&hooks), "cancelled hooks request stop");

  // Null model returns 0 (no hardcoded family fallback); MiniMax-H3's ops
  // export the exact model.h limits directly.
  eremite_image_limits api_lim = {0};
  check(eremite_image_model_limits(NULL, &api_lim) == 0,
        "eremite_image_model_limits rejects a null model");
  ImageLimits lim = {0};
  minimax_h3_image_ops.read_image_limits(&lim);
  check(lim.max_prompt_tokens == MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS &&
            lim.max_dim == MINIMAX_H3_IMAGE_MAX_DIM &&
            lim.default_width == MINIMAX_H3_IMAGE_DEFAULT_SIDE &&
            lim.default_height == MINIMAX_H3_IMAGE_DEFAULT_SIDE &&
            lim.min_generation_area == MINIMAX_H3_IMAGE_MIN_AREA &&
            lim.max_generation_area == MINIMAX_H3_IMAGE_MAX_AREA &&
            lim.default_generation_area == MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA &&
            lim.max_canvas_area == MINIMAX_H3_IMAGE_MAX_CANVAS_AREA &&
            lim.max_request_pixels == MINIMAX_H3_IMAGE_MAX_REQUEST_PIXELS &&
            lim.max_images_per_request == MINIMAX_H3_IMAGE_MAX_IMAGES &&
            lim.max_references == MINIMAX_H3_IMAGE_MAX_REFS &&
            lim.min_quality == IMAGE_QUALITY_MIN &&
            lim.max_quality == IMAGE_QUALITY_MAX &&
            lim.default_quality == MINIMAX_H3_IMAGE_DEFAULT_QUALITY &&
            lim.inference_steps == MINIMAX_H3_IMAGE_STEPS &&
            lim.guidance_scale == MINIMAX_H3_IMAGE_GUIDANCE_SCALE,
        "MiniMax-H3 read_image_limits exports exact C model limits");

  // An image's size that fits stays as it is. A larger one scales down to the
  // largest canvas with its shape kept, and a very thin one fits no canvas.
  int fit_w = 0, fit_h = 0;
  check(
      minimax_h3_image_fit_size(1024, 1024, &fit_w, &fit_h, err, sizeof err) &&
          fit_w == 1024 && fit_h == 1024,
      "an image size that fits stays as it is");
  check(
      minimax_h3_image_fit_size(8000, 6000, &fit_w, &fit_h, err, sizeof err) &&
          (int64_t)fit_w * fit_h <= MINIMAX_H3_IMAGE_MAX_CANVAS_AREA &&
          fabs((double)fit_w / fit_h - 8000.0 / 6000.0) < 0.01 &&
          minimax_h3_image_plan_canvas(fit_w, fit_h, 0, &cw, &ch, err,
                                       sizeof err),
      "a large image size scales down to the largest canvas, shape kept");
  check(
      minimax_h3_image_fit_size(12000, 4000, &fit_w, &fit_h, err, sizeof err) &&
          fit_w <= MINIMAX_H3_IMAGE_MAX_DIM &&
          fabs((double)fit_w / fit_h - 3.0) < 0.01,
      "a wide image size scales down to the longest side");
  err[0] = '\0';
  check(
      !minimax_h3_image_fit_size(100, 8000, &fit_w, &fit_h, err, sizeof err) &&
          !fit_w && !fit_h && strstr(err, "100x8000") != NULL,
      "a very thin image fits no canvas");

  // Workspace byte estimates scale monotonically with canvas and prompt sizes.
  uint64_t dit_small = minimax_h3_dit_workspace_bytes(64, 0, 256);
  uint64_t dit_default = minimax_h3_dit_workspace_bytes(
      64, 0,
      MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA /
          (MINIMAX_H3_IMAGE_PATCH_PX * MINIMAX_H3_IMAGE_PATCH_PX));
  uint64_t text_small = minimax_h3_text_encoder_workspace_bytes(64, 0);
  uint64_t text_max = minimax_h3_text_encoder_workspace_bytes(
      MINIMAX_H3_TEXT_MAX_PROMPT_TOKENS, 0);
  uint64_t vae_tile = minimax_h3_image_vae_workspace_bytes();
  check(dit_small > 0 && dit_default > dit_small && text_small > 0 &&
            text_max > text_small && vae_tile > 0,
        "workspace size estimates are positive and grow with canvas/prompt");

  // Shared SplitMix64 + Box-Muller Gaussian RNG is deterministic and ~N(0, 1).
  float g1[4096], g2[4096], g3[4096];
  minimax_h3_fill_gaussian(g1, 4096, 42);
  minimax_h3_fill_gaussian(g2, 4096, 42);
  minimax_h3_fill_gaussian(g3, 4096, 43);
  double sum = 0.0, sq = 0.0;
  for (int i = 0; i < 4096; i++) {
    sum += g1[i];
    sq += (double)g1[i] * (double)g1[i];
  }
  double mean = sum / 4096.0;
  double std = sqrt(sq / 4096.0 - mean * mean);
  check(memcmp(g1, g2, sizeof g1) == 0 && memcmp(g1, g3, sizeof g1) != 0 &&
            fabs(mean) < 0.1 && fabs(std - 1.0) < 0.1,
        "shared Gaussian RNG is deterministic, seed-sensitive and ~N(0, 1)");

  // Per-request canvas memory planning and early admission under Memory.
  MinimaxH3WeightBytes w = {.text = 18ull << 30,
                            .vision = 1ull << 30,
                            .dit = 15ull << 30,
                            .vae = 2ull << 30,
                            .vae_encoder = 1ull << 30};
  MinimaxH3RequestMemory mem_default = minimax_h3_image_plan_request_memory(
      w, minimax_h3_image_stage_bytes(64, NULL, 0, 2048, 2048), 0);
  MinimaxH3RequestMemory mem_huge = minimax_h3_image_plan_request_memory(
      w, minimax_h3_image_stage_bytes(64, NULL, 0, 4896, 4896), 0);
  uint64_t dit_delta = minimax_h3_dit_workspace_bytes(64, 0, 153 * 153) -
                       minimax_h3_dit_workspace_bytes(64, 0, 64 * 64);
  check(mem_huge.full_bytes == mem_default.full_bytes + dit_delta &&
            mem_huge.staged_bytes > mem_default.staged_bytes &&
            mem_huge.full_bytes > mem_huge.staged_bytes,
        "planned request memory scales exactly by DiT workspace growth");

  // An edit's encode stage holds the vision tower's workspace beside the image
  // outputs, the prompt's buffers beside the text layers' workspace, and the
  // VAE encoder's workspace beside the hidden rows.
  MinimaxH3RefSizes ref = minimax_h3_image_ref_sizes(1024, 1024);
  int edit_seq_len = 64 + ref.vision.tokens;
  MinimaxH3StageBytes edit =
      minimax_h3_image_stage_bytes(edit_seq_len, &ref, 1, 2048, 2048);
  uint64_t row = (uint64_t)MINIMAX_H3_TEXT_DIM * sizeof(float);
  uint64_t image_outputs = (uint64_t)(1 + MINIMAX_H3_TEXT_DEEPSTACK_LAYERS) *
                           (uint64_t)ref.vision.tokens * row;
  uint64_t prompt_buffers = 2 * (uint64_t)edit_seq_len * row +
                            (uint64_t)MINIMAX_H3_TEXT_DEEPSTACK_LAYERS *
                                (uint64_t)ref.vision.tokens * row;
  check(ref.vision.patches == 4096 && ref.vision.tokens == 1024 &&
            ref.vae_width == 1024 && ref.vae_height == 1024 &&
            edit.encode >=
                minimax_h3_vision_encoder_workspace_bytes(ref.vision.patches) +
                    image_outputs &&
            edit.encode >= minimax_h3_text_encoder_workspace_bytes(
                               edit_seq_len, ref.vision.tokens) +
                               prompt_buffers &&
            edit.encode >= minimax_h3_image_vae_encoder_workspace_bytes(
                               ref.vae_width, ref.vae_height) +
                               (uint64_t)edit_seq_len * row,
        "an edit's encode stage counts the vision and VAE encoder workspaces");

  Memory *mem = memory_new();
  MemoryAccount acct;
  memory_account_init(mem, &acct, "minimax-h3-test", 0);
  int use_full = -1;

  // Under a 50 GiB limit, 4896x4896 fits in full residence.
  memory_set_limit(mem, 50ull << 30);
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 1, mem_default.staged_bytes, 4896,
                                       4896, mem_huge, &use_full, err,
                                       sizeof err) == MINIMAX_H3_ADMITTED &&
            use_full == 1 && acct.minimum == mem_huge.full_bytes,
        "4896x4896 admits in full residence when limit covers full_bytes");

  // Under a 40 GiB limit, 4896x4896 falls back cleanly from full to staged.
  memory_set_minimum(&acct, mem_default.full_bytes, MEMORY_REQUIRED);
  memory_set_limit(mem, 40ull << 30);
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 1, mem_default.staged_bytes, 4896,
                                       4896, mem_huge, &use_full, err,
                                       sizeof err) == MINIMAX_H3_ADMITTED &&
            use_full == 0 && acct.minimum == mem_huge.staged_bytes,
        "4896x4896 falls back to staged when only staged_bytes fits");

  // Under a 22 GiB limit, 4896x4896 can never fit: it is refused early with a
  // clear message, and the account's baseline minimum stays.
  memory_set_minimum(&acct, mem_default.staged_bytes, MEMORY_REQUIRED);
  memory_set_limit(mem, 22ull << 30);
  uint64_t min_before = acct.minimum;
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 0, mem_default.staged_bytes, 4896,
                                       4896, mem_huge, &use_full, err,
                                       sizeof err) == MINIMAX_H3_NEVER_FITS &&
            acct.minimum == min_before && strstr(err, "4896x4896") != NULL &&
            strstr(err, "more than") != NULL,
        "4896x4896 never fits when staged_bytes exceeds the limit");

  // Under a 40 GiB limit with 20 GiB reserved by another model, 4896x4896
  // does not fit now, but it would fit once the other model frees its memory.
  MemoryAccount other;
  memory_account_init(mem, &other, "other-model", 0);
  memory_set_limit(mem, 40ull << 30);
  int other_reserved = memory_set_minimum(&other, 20ull << 30, MEMORY_REQUIRED);
  err[0] = '\0';
  check(other_reserved &&
            minimax_h3_image_admit_request(&acct, 0, mem_default.staged_bytes,
                                           4896, 4896, mem_huge, &use_full, err,
                                           sizeof err) == MINIMAX_H3_NOT_NOW &&
            acct.minimum == min_before && strstr(err, "now") != NULL,
        "4896x4896 does not fit now while another model holds the memory");
  memory_set_minimum(&other, 0, MEMORY_REQUIRED);

  // A mask is checked before any GPU work: it needs an image to edit, an
  // alpha channel and the first image's size, and the output keeps the first
  // image's shape. These checks run without the model's weights.
  enum { REF_W = 64, REF_H = 48 };

  static uint8_t ref_rgb[REF_H][REF_W][3], mask_rgba[REF_H][REF_W][4];
  for (int y = 0; y < REF_H; y++)
    for (int x = 0; x < REF_W; x++) {
      ref_rgb[y][x][0] = (uint8_t)(3 * x);
      mask_rgba[y][x][3] = y < REF_H / 2 ? 0 : 255;
    }
  size_t ref_len = 0, mask_len = 0, flat_len = 0, small_len = 0;
  const char *encode_error = 0;
  uint8_t *ref_png = image_encode_png(&ref_rgb[0][0][0], REF_W, REF_H, &ref_len,
                                      &encode_error);
  uint8_t *mask_png = image_encode_png_rgba(&mask_rgba[0][0][0], REF_W, REF_H,
                                            &mask_len, &encode_error);
  uint8_t *flat_png = image_encode_png(&ref_rgb[0][0][0], REF_W, REF_H,
                                       &flat_len, &encode_error);
  uint8_t *small_png = image_encode_png_rgba(
      &mask_rgba[0][0][0], REF_W / 2, REF_H / 2, &small_len, &encode_error);
  ImageInput ref_input = {.data = ref_png, .len = ref_len};

  struct {
    ImageInput mask;
    int ref_count, width, height;
    const char *message;
  } mask_cases[] = {
      {{mask_png, mask_len}, 0, REF_W, REF_H, "a mask needs an image"},
      {{flat_png, flat_len}, 1, REF_W, REF_H, "no alpha channel"},
      {{small_png, small_len}, 1, REF_W, REF_H, "the mask is 32x24"},
      {{mask_png, mask_len}, 1, REF_W, REF_W, "keep the first image's 64x48"},
  };

  int masks_refused = 1;
  for (size_t i = 0; i < sizeof mask_cases / sizeof mask_cases[0]; i++) {
    ImageRequest masked = {.prompt = "edit",
                           .width = mask_cases[i].width,
                           .height = mask_cases[i].height,
                           .format = IMAGE_FORMAT_PNG,
                           .refs = &ref_input,
                           .ref_count = mask_cases[i].ref_count,
                           .mask = mask_cases[i].mask};
    ImageResult refused = {0};
    int ok = minimax_h3_image_ops.generate_image(&masked, NULL, NULL, &refused);
    masks_refused &= !ok && refused.invalid_input &&
                     strstr(refused.error, mask_cases[i].message) != NULL;
  }
  check(masks_refused, "a mask without an image, without alpha, of another "
                       "size or with another output shape is invalid input");
  ImageRequest fitting = {.prompt = "edit",
                          .width = REF_W,
                          .height = REF_H,
                          .format = IMAGE_FORMAT_PNG,
                          .refs = &ref_input,
                          .ref_count = 1,
                          .mask = {mask_png, mask_len}};
  ImageResult unloaded = {0};
  check(!minimax_h3_image_ops.generate_image(&fitting, NULL, NULL, &unloaded) &&
            !unloaded.invalid_input,
        "a fitting mask passes its checks, and then needs the model's weights");
  free(ref_png);
  free(mask_png);
  free(flat_png);
  free(small_png);

  // A staged request keeps at least the staged base reserved while it runs.
  memory_set_limit(mem, 50ull << 30);
  uint64_t staged_base = mem_default.staged_bytes + (1ull << 30);
  memory_set_minimum(&acct, staged_base, MEMORY_REQUIRED);
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 0, staged_base, 2048, 2048,
                                       mem_default, &use_full, err,
                                       sizeof err) == MINIMAX_H3_ADMITTED &&
            use_full == 0 && acct.minimum == staged_base,
        "a staged request keeps the staged base reserved");

  printf("minimax_h3_image unit checks: %d failures\n", failures);
  return failures != 0;
}
