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
            lim.min_generation_area == MINIMAX_H3_IMAGE_MIN_AREA &&
            lim.max_generation_area == MINIMAX_H3_IMAGE_MAX_AREA &&
            lim.default_generation_area == MINIMAX_H3_IMAGE_DEFAULT_MIN_AREA &&
            lim.max_canvas_area == MINIMAX_H3_IMAGE_MAX_CANVAS_AREA &&
            lim.max_request_pixels == MINIMAX_H3_IMAGE_MAX_REQUEST_PIXELS &&
            lim.max_images_per_request == MINIMAX_H3_IMAGE_MAX_IMAGES &&
            lim.max_references == MINIMAX_H3_IMAGE_MAX_REFS &&
            lim.min_jpeg_quality == IMAGE_JPEG_QUALITY_MIN &&
            lim.max_jpeg_quality == IMAGE_JPEG_QUALITY_MAX &&
            lim.default_jpeg_quality == MINIMAX_H3_IMAGE_DEFAULT_JPEG_QUALITY &&
            lim.inference_steps == MINIMAX_H3_IMAGE_STEPS &&
            lim.guidance_scale == MINIMAX_H3_IMAGE_GUIDANCE_SCALE,
        "MiniMax-H3 read_image_limits exports exact C model limits");

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
                                       sizeof err) == 1 &&
            use_full == 1 && acct.minimum == mem_huge.full_bytes,
        "4896x4896 admits in full residence when limit covers full_bytes");

  // Under a 40 GiB limit, 4896x4896 falls back cleanly from full to staged.
  memory_set_minimum(&acct, mem_default.full_bytes, MEMORY_REQUIRED);
  memory_set_limit(mem, 40ull << 30);
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 1, mem_default.staged_bytes, 4896,
                                       4896, mem_huge, &use_full, err,
                                       sizeof err) == 1 &&
            use_full == 0 && acct.minimum == mem_huge.staged_bytes,
        "4896x4896 falls back to staged when only staged_bytes fits");

  // Under a 22 GiB limit, 4896x4896 refuses early with a clear message and
  // leaves the account's baseline minimum intact.
  memory_set_minimum(&acct, mem_default.staged_bytes, MEMORY_REQUIRED);
  memory_set_limit(mem, 22ull << 30);
  uint64_t min_before = acct.minimum;
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 0, mem_default.staged_bytes, 4896,
                                       4896, mem_huge, &use_full, err,
                                       sizeof err) == 0 &&
            acct.minimum == min_before && strstr(err, "4896x4896") != NULL &&
            strstr(err, "GiB") != NULL,
        "4896x4896 refuses early when staged_bytes exceeds limit");

  // A staged request keeps at least the staged base reserved while it runs.
  memory_set_limit(mem, 50ull << 30);
  uint64_t staged_base = mem_default.staged_bytes + (1ull << 30);
  memory_set_minimum(&acct, staged_base, MEMORY_REQUIRED);
  err[0] = '\0';
  check(minimax_h3_image_admit_request(&acct, 0, staged_base, 2048, 2048,
                                       mem_default, &use_full, err,
                                       sizeof err) == 1 &&
            use_full == 0 && acct.minimum == staged_base,
        "a staged request keeps the staged base reserved");

  printf("minimax_h3_image unit checks: %d failures\n", failures);
  return failures != 0;
}
