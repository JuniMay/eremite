#include "engine/memory.h"
#include "model/minimax_h3/video_model.h"
#include "model/model.h"
#include <stdio.h>
#include <string.h>

// Unit tests for MiniMax-H3 Video canvas/frame planning, memory admission and
// request validation without GPU or checkpoints.

// Failed assertion counter.
static int failures;

// Records one assertion outcome and prints label on failure.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

int main(void) {
  char err[256] = {0};
  int w = 0, h = 0;

  // 1. Canvas adaptation (768 short edge, 768*1344 pixel cap, multiple of 32).
  minimax_h3_video_adapt_canvas(1920, 1080, &w, &h);
  check(w == 1344 && h == 768, "16:9 1080p adapts to 1344x768");
  minimax_h3_video_adapt_canvas(1080, 1920, &w, &h);
  check(w == 768 && h == 1344, "9:16 portrait adapts to 768x1344");
  minimax_h3_video_adapt_canvas(1024, 1024, &w, &h);
  check(w == 768 && h == 768, "1:1 square adapts to 768x768");
  minimax_h3_video_adapt_canvas(2560, 1080, &w, &h);
  check(w % 32 == 0 && h % 32 == 0 &&
            (int64_t)w * (int64_t)h <=
                MINIMAX_H3_VIDEO_ADAPT_MAX_PIXELS + (int64_t)w * 32,
        "ultrawide 21:9 caps area near 768*1344 on 32 px grid");

  // 2. Canvas planning and limit checks.
  check(minimax_h3_video_plan_canvas(0, 0, &w, &h, err, sizeof err) &&
            w == 1344 && h == 768,
        "0x0 plans default 1344x768 canvas");
  check(minimax_h3_video_plan_canvas(832, 480, &w, &h, err, sizeof err) &&
            w == 832 && h == 480,
        "832x480 stays exact 832x480");
  check(minimax_h3_video_plan_canvas(500, 300, &w, &h, err, sizeof err) &&
            w == 512 && h == 288,
        "500x300 rounds to 32 px multiples 512x288");
  err[0] = '\0';
  check(!minimax_h3_video_plan_canvas(4096, 4096, &w, &h, err, sizeof err) &&
            err[0] != '\0',
        "4096x4096 exceeds video max dimension");
  err[0] = '\0';
  check(!minimax_h3_video_plan_canvas(2048, 2048, &w, &h, err, sizeof err) &&
            err[0] != '\0',
        "2048x2048 exceeds video max canvas area");
  err[0] = '\0';
  check(!minimax_h3_video_plan_canvas(-1, 768, &w, &h, err, sizeof err) &&
            err[0] != '\0',
        "negative width is rejected");

  // 3. Frame and duration planning (17k + 5 frames, 5k + 2 video T, 40 Hz audio
  // T).
  int frames = 0, t_lat = 0, audio_t = 0;
  check(minimax_h3_video_plan_frames(0, 0, &frames, &t_lat, &audio_t, err,
                                     sizeof err) &&
            frames == 124 && t_lat == 37 && audio_t == 207,
        "default duration plans 124 frames, t_lat=37, audio_t=207");
  check(minimax_h3_video_plan_frames(1, 0, &frames, &t_lat, &audio_t, err,
                                     sizeof err) &&
            frames == 39 && t_lat == 12 && audio_t == 65,
        "1 second snaps 24 frames up to 39 frames");
  check(minimax_h3_video_plan_frames(0, 22, &frames, &t_lat, &audio_t, err,
                                     sizeof err) &&
            frames == 22 && t_lat == 7 && audio_t == 37,
        "explicit 22 frames stays 22 frames, t_lat=7, audio_t=37");
  check(minimax_h3_video_plan_frames(15, 0, &frames, &t_lat, &audio_t, err,
                                     sizeof err) &&
            frames == 362 && t_lat == 107 && audio_t == 603,
        "15 seconds snaps 360 frames to 362 frames");
  err[0] = '\0';
  check(!minimax_h3_video_plan_frames(20, 0, &frames, &t_lat, &audio_t, err,
                                      sizeof err) &&
            err[0] != '\0',
        "20 seconds exceeds 15 second limit");
  err[0] = '\0';
  check(!minimax_h3_video_plan_frames(0, 500, &frames, &t_lat, &audio_t, err,
                                      sizeof err) &&
            err[0] != '\0',
        "500 frames exceeds 362 frame limit");

  // 4. Memory planning and early admission (full, staged, never fits, not now).
  MinimaxH3VideoWeightBytes wb = {
      .text = 16ull << 30,
      .vision = 1ull << 30,
      .dit = 21ull << 30,
      .lora = 2ull << 30,
      .video_vae = 2ull << 30,
      .video_vae_enc = 1ull << 30,
      .audio_vae = 1ull << 30,
      .audio_vae_enc = 1ull << 30,
  };
  MinimaxH3VideoStageBytes sb = {
      .encode = 2ull << 30,
      .sample = 10ull << 30,
      .decode_video = 1ull << 30,
      .decode_audio = 1ull << 30,
  };
  MinimaxH3RequestMemory rm =
      minimax_h3_video_plan_request_memory(wb, sb, 1, 1, 1);
  // Core = 16 + 21 + 2 + 2 + 1 = 42 GiB; sample = 42 + 10 = 52 GiB;
  // staged = max(16 + 3 + 2 = 21, 21 + 2 + 10 = 33, 2 + 1 = 3) = 33 GiB.
  check(rm.full_bytes == (52ull << 30) && rm.staged_bytes == (33ull << 30),
        "video memory plan computes exact full (52 GiB) and staged (33 GiB) "
        "peaks");

  Memory *mem = memory_new();
  MemoryAccount acct = {0};
  memory_account_init(mem, &acct, "video-test", 0);
  int use_full = -1;

  memory_set_limit(mem, 64ull << 30);
  check(minimax_h3_video_admit_request(&acct, 1, 30ull << 30, 1344, 768, 124,
                                       rm, &use_full, err,
                                       sizeof err) == MINIMAX_H3_ADMITTED &&
            use_full == 1,
        "64 GiB limit admits video in full residence");

  memory_set_minimum(&acct, 30ull << 30, MEMORY_REQUIRED);
  memory_set_limit(mem, 40ull << 30);
  check(minimax_h3_video_admit_request(&acct, 1, 30ull << 30, 1344, 768, 124,
                                       rm, &use_full, err,
                                       sizeof err) == MINIMAX_H3_ADMITTED &&
            use_full == 0,
        "40 GiB limit falls back cleanly to staged execution");

  memory_set_minimum(&acct, 20ull << 30, MEMORY_REQUIRED);
  memory_set_limit(mem, 25ull << 30);
  err[0] = '\0';
  check(minimax_h3_video_admit_request(&acct, 0, 20ull << 30, 1344, 768, 124,
                                       rm, &use_full, err,
                                       sizeof err) == MINIMAX_H3_NEVER_FITS &&
            acct.minimum == (20ull << 30) && strstr(err, "1344x768") != NULL,
        "25 GiB limit refuses early with NEVER_FITS and preserves minimum");

  MemoryAccount other = {0};
  memory_account_init(mem, &other, "other-model", 0);
  memory_set_limit(mem, 40ull << 30);
  check(memory_set_minimum(&other, 15ull << 30, MEMORY_REQUIRED) &&
            minimax_h3_video_admit_request(&acct, 0, 20ull << 30, 1344, 768,
                                           124, rm, &use_full, err,
                                           sizeof err) == MINIMAX_H3_NOT_NOW,
        "another model holding 15 GiB of 40 GiB returns NOT_NOW");

  // 5. Limits export and client input checks without weights loaded.
  VideoLimits lim = {0};
  minimax_h3_video_ops.read_video_limits(&lim);
  check(lim.fps == 24 && lim.default_width == 1344 &&
            lim.default_height == 768 && lim.max_frames == 362 &&
            lim.max_total_refs == 12 && lim.guidance_scale == 1.0,
        "read_video_limits exports exact C model limits");

  VideoRequest bad_steps = {.prompt = "test", .steps = 500};
  VideoResult res = {0};
  check(!minimax_h3_video_ops.generate_video(&bad_steps, NULL, NULL, &res) &&
            res.invalid_input,
        "steps > 100 is rejected as invalid_input");

  VideoRequest fl_with_refs = {.prompt = "test", .ref_image_count = 1};
  res = (VideoResult){0};
  check(!minimax_h3_video_ops.generate_video(&fl_with_refs, NULL, NULL, &res) &&
            res.invalid_input && strstr(res.error, "REF2VA") != NULL,
        "FL2VA model rejects REF2VA reference images as invalid_input");

  VideoRequest unloaded = {.prompt = "test", .width = 512, .height = 288};
  res = (VideoResult){0};
  check(!minimax_h3_video_ops.generate_video(&unloaded, NULL, NULL, &res) &&
            !res.invalid_input,
        "valid request passes validation before checking loaded weights");

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("minimax_h3_video_model: all checks passed\n");
  return 0;
}
