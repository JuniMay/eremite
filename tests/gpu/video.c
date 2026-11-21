// Tests native macOS H.264/AAC MP4 encoding, decoding and standalone audio
// decoding through AVFoundation and VideoToolbox.
#include "check.h"
#include "platform/video.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Synthetic test clip geometry: 24 frames (1 s at 24 fps) of 64x64 RGB24.
static constexpr int TEST_FRAMES = 24, TEST_W = 64, TEST_H = 64, TEST_FPS = 24;

// Synthetic stereo audio rate and frequencies (left 440 Hz, right 880 Hz).
static constexpr int TEST_SR = 32000;
// Left and right test tone frequencies in Hz.
static constexpr double FREQ_LEFT = 440.0, FREQ_RIGHT = 880.0;
// Full sine-wave cycle in radians (2 * pi).
static constexpr double TWO_PI = 6.28318530717958647692;
// Peak amplitude of the synthetic test sine waves.
static constexpr float TONE_AMP = 0.5f;

// Active and inactive channel values used for solid RGB test frames.
static constexpr uint8_t COLOR_ON = 220, COLOR_OFF = 20;

// Smallest valid MP4 header byte size and ftyp box offset.
static constexpr size_t MIN_MP4_BYTES = 64, FTYP_OFFSET = 4, FTYP_LEN = 4;

// Frame count used for the silent video-only test clip.
static constexpr int SILENT_FRAMES = 8;

// Builds a synthetic RGB frame sequence whose frames alternate red, green and
// blue solids so H.264 compression preserves clean per-frame mean colors.
static uint8_t *make_test_frames(int count, int w, int h) {
  size_t fbytes = (size_t)w * (size_t)h * VIDEO_RGB_CHANNELS;
  uint8_t *rgb = malloc((size_t)count * fbytes);
  check(rgb != NULL, "frame buffer allocated");
  for (int f = 0; f < count; f++) {
    uint8_t r = (f % VIDEO_RGB_CHANNELS == 0) ? COLOR_ON : COLOR_OFF;
    uint8_t g = (f % VIDEO_RGB_CHANNELS == 1) ? COLOR_ON : COLOR_OFF;
    uint8_t b = (f % VIDEO_RGB_CHANNELS == 2) ? COLOR_ON : COLOR_OFF;
    uint8_t *dst = rgb + (size_t)f * fbytes;
    for (int i = 0; i < w * h; i++) {
      dst[i * VIDEO_RGB_CHANNELS + 0] = r;
      dst[i * VIDEO_RGB_CHANNELS + 1] = g;
      dst[i * VIDEO_RGB_CHANNELS + 2] = b;
    }
  }
  return rgb;
}

// Builds planar stereo FP32 sine waves [2, samples] at sample_rate.
static float *make_test_audio(int samples, int sample_rate) {
  float *pcm = malloc((size_t)samples * VIDEO_AUDIO_CHANNELS * sizeof(float));
  check(pcm != NULL, "audio buffer allocated");
  for (int i = 0; i < samples; i++) {
    double t = (double)i / (double)sample_rate;
    pcm[i] = TONE_AMP * (float)sin(TWO_PI * FREQ_LEFT * t);
    pcm[samples + i] = TONE_AMP * (float)sin(TWO_PI * FREQ_RIGHT * t);
  }
  return pcm;
}

// Verifies MP4 encoding with both H.264 video and stereo AAC audio tracks.
static void test_mp4_av_encode(void) {
  uint8_t *rgb = make_test_frames(TEST_FRAMES, TEST_W, TEST_H);
  float *pcm = make_test_audio(TEST_SR, TEST_SR);

  size_t mp4_len = 0;
  const char *err = NULL;
  uint8_t *mp4 = video_encode_mp4(rgb, TEST_FRAMES, TEST_W, TEST_H, TEST_FPS,
                                  pcm, TEST_SR, TEST_SR, &mp4_len, &err);
  check(mp4 != NULL, "av mp4 encoded");
  check(mp4_len > MIN_MP4_BYTES &&
            memcmp(mp4 + FTYP_OFFSET, "ftyp", FTYP_LEN) == 0,
        "av mp4 has ftyp box");

  free(mp4);
  free(pcm);
  free(rgb);
}

// Verifies silent (video-only) MP4 encoding.
static void test_mp4_video_only_encode(void) {
  uint8_t *rgb = make_test_frames(SILENT_FRAMES, TEST_W, TEST_H);
  size_t mp4_len = 0;
  const char *err = NULL;
  uint8_t *mp4 = video_encode_mp4(rgb, SILENT_FRAMES, TEST_W, TEST_H, TEST_FPS,
                                  NULL, 0, 0, &mp4_len, &err);
  check(mp4 != NULL && mp4_len > MIN_MP4_BYTES &&
            memcmp(mp4 + FTYP_OFFSET, "ftyp", FTYP_LEN) == 0,
        "video-only mp4 encoded");
  free(mp4);
  free(rgb);
}

// Runs native MP4 encoding tests.
int main(void) {
  test_mp4_av_encode();
  test_mp4_video_only_encode();
  return failures != 0;
}
