#include "model/minimax_h3/model.h"
#include "model/minimax_h3/quant.h"
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

  printf("minimax_h3_image unit checks: %d failures\n", failures);
  return failures != 0;
}
