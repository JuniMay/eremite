#include "base/clock.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/minimax_h3/video_model.h"
#include "model/model.h"
#include "platform/image.h"
#include "platform/video.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Real-checkpoint MiniMax-H3 Video test: runs T2VA and FL2VA generation with
// optional Turbo LoRA, verifies decoded H.264/AAC MP4 frames and stereo audio,
// and checks that prefill and video generation counters stay strictly separate.

// Test clip geometry (512x288, 22 frames ~= 0.92 s at 24 fps, 4 Turbo steps).
static constexpr int TEST_W = 512, TEST_H = 288, TEST_FRAMES = 22,
                     TEST_STEPS = 4;

// Deterministic RNG seeds for T2VA and FL2VA runs.
static constexpr uint64_t SEED_T2VA = 42, SEED_FL2VA = 99;

// Failed assertion counter.
static int failures;

// Records one check result and prints label on failure.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

// Computes RGB standard deviation across decoded video frames.
static double rgb_stddev(const uint8_t *rgb, size_t count) {
  double sum = 0.0, sq = 0.0;
  for (size_t i = 0; i < count; i++) {
    double v = (double)rgb[i];
    sum += v;
    sq += v * v;
  }
  double mean = sum / (double)count;
  double var = sq / (double)count - mean * mean;
  return var > 0.0 ? sqrt(var) : 0.0;
}

// Computes RMS amplitude across planar stereo FP32 PCM samples.
static double pcm_rms(const float *pcm, size_t count) {
  double sq = 0.0;
  for (size_t i = 0; i < count; i++) sq += (double)pcm[i] * (double)pcm[i];
  return count > 0 ? sqrt(sq / (double)count) : 0.0;
}

// Synthesizes a colorful PNG first frame for the FL2VA test.
static uint8_t *make_test_first_frame_png(int w, int h, size_t *out_len) {
  uint8_t *rgb = xmalloc((size_t)w * (size_t)h * 3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      uint8_t *p = rgb + ((size_t)y * (size_t)w + (size_t)x) * 3;
      p[0] = (uint8_t)(40 + (x * 180) / w);
      p[1] = (uint8_t)(60 + (y * 160) / h);
      p[2] = (uint8_t)(180 - (y * 120) / h);
    }
  }
  const char *err = NULL;
  uint8_t *png = image_encode_png(rgb, w, h, out_len, &err);
  free(rgb);
  if (!png) die("image_encode_png failed: %s", err ? err : "unknown");
  return png;
}

// Synthesizes a short 5-frame MP4 clip with stereo audio for REF2VA testing.
static uint8_t *make_test_ref_video_mp4(int w, int h, size_t *out_len) {
  int frames = 5;
  uint8_t *rgb = xmalloc((size_t)frames * (size_t)w * (size_t)h * 3);
  for (int f = 0; f < frames; f++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        uint8_t *p =
            rgb + (((size_t)f * (size_t)h + (size_t)y) * (size_t)w + x) * 3;
        p[0] = (uint8_t)((x * 200) / w + f * 8);
        p[1] = (uint8_t)((y * 200) / h);
        p[2] = (uint8_t)(120 + f * 20);
      }
    }
  }
  int samples = 6667;
  float *pcm = xmalloc((size_t)2 * (size_t)samples * sizeof(float));
  for (int i = 0; i < samples; i++) {
    double t = (double)i / 32000.0;
    pcm[i] = 0.2f * (float)sin(2.0 * M_PI * 440.0 * t);
    pcm[samples + i] = 0.2f * (float)sin(2.0 * M_PI * 660.0 * t);
  }
  const char *err = NULL;
  uint8_t *mp4 = video_encode_mp4(rgb, frames, w, h, 24, pcm, samples, 32000,
                                  out_len, &err);
  free(pcm);
  free(rgb);
  if (!mp4) die("video_encode_mp4 failed: %s", err ? err : "unknown");
  return mp4;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    printf("usage: %s <model_dir> <text_encoder> <dit> <video_vae> "
           "<audio_vae> [lora] [steps] [fl2va|ref2va]\n",
           argv[0]);
    return 1;
  }
  const char *model_dir = argv[1];
  const char *text_enc = argv[2];
  const char *dit_path = argv[3];
  const char *vvae_path = argv[4];
  const char *avae_path = argv[5];
  const char *lora_path = argc > 6 && argv[6][0] ? argv[6] : NULL;
  int steps = argc > 7 ? atoi(argv[7]) : TEST_STEPS;
  int is_ref2va = argc > 8 && strcmp(argv[8], "ref2va") == 0;

  Memory *memory = memory_new();
  size_t source_bytes = 0;
  const char *source = kernel_library_source(&source_bytes);
  Gpu *gpu = gpu_new(source, source_bytes);

  Options opt = {
      .ctx = 4096,
      .dump_fd = -1,
      .routes_fd = -1,
      .quiet = 0,
      .minimax_h3_video = {.text_encoder = text_enc,
                           .dit = dit_path,
                           .vae = vvae_path,
                           .audio_vae = avae_path,
                           .lora = lora_path,
                           .lora_strength = 1.0f,
                           .default_steps = steps,
                           .shift_video = 12.0,
                           .shift_audio = 3.0,
                           .variant = is_ref2va ? VIDEO_VARIANT_REF2VA
                                                : VIDEO_VARIANT_FL2VA},
  };

  double t0 = now();
  Model *model = model_open(memory, gpu, model_dir, &opt);
  check(model != NULL, "MiniMax-H3 Video model opened and loaded");
  if (!model) return 1;
  printf("loaded MiniMax-H3 Video (%s) in %.2f s\n",
         is_ref2va ? "ref2va" : "fl2va", now() - t0);

  if (is_ref2va) {
    size_t img_len = 0, vid_len = 0;
    uint8_t *img = make_test_first_frame_png(256, 256, &img_len);
    uint8_t *vid = make_test_ref_video_mp4(256, 160, &vid_len);
    VideoMediaInput ref_imgs[1] = {{.data = img, .len = img_len}};
    VideoMediaInput ref_vids[1] = {{.data = vid, .len = vid_len}};
    VideoRequest ref_req = {
        .prompt = "A cinematic shot featuring <Picture 1> moving smoothly as "
                  "in <Video 1> with matching soundtrack from <Audio 1>.",
        .width = TEST_W,
        .height = TEST_H,
        .frames = TEST_FRAMES,
        .steps = steps,
        .seed = SEED_T2VA,
        .ref_images = ref_imgs,
        .ref_image_count = 1,
        .ref_videos = ref_vids,
        .ref_video_count = 1,
    };
    VideoResult res = {0};
    int ok = model->ops->generate_video(&ref_req, NULL, NULL, &res);
    free(vid);
    free(img);
    check(ok && res.data != NULL && res.len > 0,
          "REF2VA produced MP4 bytes with ref image, video and audio");
    if (!ok) fprintf(stderr, "REF2VA error: %s\n", res.error);
    printf("REF2VA: %zu bytes MP4, %d prompt tokens, prefill %.2f s, video "
           "%.2f s\n",
           res.len, res.prompt_tokens, res.prefill_seconds, res.video_seconds);
    if (ok) {
      VideoDecoded dec = {0};
      const char *derr = NULL;
      int dok = video_decode_mp4(res.data, res.len, 24, 32000, &dec, &derr);
      double std = dok ? rgb_stddev(dec.rgb, (size_t)dec.frame_count *
                                                 (size_t)dec.width *
                                                 (size_t)dec.height * 3)
                       : 0.0;
      double arms = (dok && dec.audio_pcm)
                        ? pcm_rms(dec.audio_pcm, (size_t)dec.audio_samples * 2)
                        : 0.0;
      printf("REF2VA decoded: %d frames (rgb std %.1f), %d audio samples (rms "
             "%.4f)\n",
             dec.frame_count, std, dec.audio_samples, arms);
      check(dok && dec.frame_count == TEST_FRAMES && std > 5.0 && arms > 1e-4,
            "REF2VA MP4 decodes cleanly with non-trivial video and audio");
      video_decoded_free(&dec);
    }
    ModelCounters c = {0};
    model->ops->read_counters(&c);
    check(c.decode_rows == 0 && c.generated_videos == 1 &&
              c.video_steps == (uint64_t)steps &&
              c.prefill_rows == (uint64_t)res.prompt_tokens,
          "REF2VA counters strictly separate prefill and video generation");
    free(res.data);
    if (failures) return 1;
    printf("minimax_h3_video (ref2va): all checks passed\n");
    return 0;
  }

  // 1. T2VA generation.
  VideoRequest t2va = {
      .prompt = "A golden retriever running across a sunny green meadow, "
                "barking happily with birds chirping in the background.",
      .width = TEST_W,
      .height = TEST_H,
      .frames = TEST_FRAMES,
      .steps = steps,
      .seed = SEED_T2VA,
  };
  VideoResult res1 = {0};
  int ok1 = model->ops->generate_video(&t2va, NULL, NULL, &res1);
  check(ok1 && res1.data != NULL && res1.len > 0, "T2VA produced MP4 bytes");
  if (!ok1) fprintf(stderr, "T2VA error: %s\n", res1.error);
  check(res1.width == TEST_W && res1.height == TEST_H &&
            res1.frames == TEST_FRAMES && res1.fps == 24,
        "T2VA output geometry matches 512x288 22 frames @ 24 fps");
  check(res1.prompt_tokens > 0 && res1.prefill_seconds > 0.0 &&
            res1.video_seconds > 0.0,
        "T2VA reports separate prompt tokens, prefill_seconds and "
        "video_seconds");
  printf("T2VA: %zu bytes MP4, %d prompt tokens, prefill %.2f s, video %.2f "
         "s\n",
         res1.len, res1.prompt_tokens, res1.prefill_seconds,
         res1.video_seconds);

  // Decode T2VA MP4 back and verify non-trivial frames and stereo audio.
  if (ok1) {
    VideoDecoded dec = {0};
    const char *derr = NULL;
    int dok = video_decode_mp4(res1.data, res1.len, 24, 32000, &dec, &derr);
    check(dok && dec.frame_count == TEST_FRAMES && dec.width == TEST_W &&
              dec.height == TEST_H,
          "T2VA MP4 decodes cleanly back to 22 RGB frames");
    double std =
        rgb_stddev(dec.rgb, (size_t)dec.frame_count * (size_t)dec.width *
                                (size_t)dec.height * 3);
    double arms = dec.audio_pcm
                      ? pcm_rms(dec.audio_pcm, (size_t)dec.audio_samples * 2)
                      : 0.0;
    printf("T2VA decoded: %d frames (rgb std %.1f), %d audio samples (rms "
           "%.4f)\n",
           dec.frame_count, std, dec.audio_samples, arms);
    check(std > 5.0, "T2VA decoded RGB frames have non-trivial visual detail");
    check(dec.audio_pcm != NULL && dec.audio_samples > 20000 && arms > 1e-4,
          "T2VA decoded stereo AAC track has non-silent audio");
    video_decoded_free(&dec);
  }
  int t2va_tokens = res1.prompt_tokens;
  free(res1.data);

  // 2. FL2VA generation with a synthetic first frame.
  size_t png_len = 0;
  uint8_t *png = make_test_first_frame_png(TEST_W, TEST_H, &png_len);
  VideoRequest fl2va = {
      .prompt = "Camera slowly zooms forward as gentle wind blows.",
      .width = TEST_W,
      .height = TEST_H,
      .frames = TEST_FRAMES,
      .steps = steps,
      .seed = SEED_FL2VA,
      .first_frame = {.data = png, .len = png_len},
  };
  VideoResult res2 = {0};
  int ok2 = model->ops->generate_video(&fl2va, NULL, NULL, &res2);
  free(png);
  check(ok2 && res2.data != NULL && res2.len > 0, "FL2VA produced MP4 bytes");
  if (!ok2) fprintf(stderr, "FL2VA error: %s\n", res2.error);
  check(res2.prompt_tokens > t2va_tokens,
        "FL2VA prompt_tokens includes Qwen3-VL first-frame vision tokens");
  printf("FL2VA: %zu bytes MP4, %d prompt tokens, prefill %.2f s, video %.2f "
         "s\n",
         res2.len, res2.prompt_tokens, res2.prefill_seconds,
         res2.video_seconds);
  if (ok2) {
    VideoDecoded dec = {0};
    const char *derr = NULL;
    int dok = video_decode_mp4(res2.data, res2.len, 24, 32000, &dec, &derr);
    check(dok && dec.frame_count == TEST_FRAMES,
          "FL2VA MP4 decodes cleanly back to 22 frames");
    video_decoded_free(&dec);
  }
  int fl2va_tokens = res2.prompt_tokens;
  free(res2.data);

  // 3. Verify strict separation of prefill vs video generation counters.
  ModelCounters c = {0};
  model->ops->read_counters(&c);
  check(c.decode_rows == 0 && c.decode_seconds == 0.0,
        "video generation never increments token decode counters");
  check(c.prefill_rows == (uint64_t)(t2va_tokens + fl2va_tokens),
        "prefill_rows equals exact Qwen3-VL text+vision tokens");
  check(c.generated_videos == 2 && c.video_steps == (uint64_t)(steps * 2) &&
            c.video_seconds > 0.0 && c.video_stage == VIDEO_STAGE_NONE,
        "video counters record 2 generated videos, exact denoising steps and "
        "idle stage");

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("minimax_h3_video: all checks passed\n");
  return 0;
}
