// Tests CPU-side validation of platform video and audio helpers.
#include "check.h"
#include "platform/video.h"
#include <stdint.h>

// Verifies invalid encoding and decoding arguments are refused before touching
// AVFoundation or VideoToolbox hardware services.
int main(void) {
  const char *err = NULL;
  size_t len = 0;
  uint8_t pixel[12] = {0};

  check(video_encode_mp4(NULL, 1, 16, 16, 24, NULL, 0, 0, &len, &err) == NULL &&
            err != NULL,
        "null rgb refused");
  err = NULL;
  check(video_encode_mp4(pixel, 0, 16, 16, 24, NULL, 0, 0, &len, &err) ==
                NULL &&
            err != NULL,
        "zero frames refused");
  err = NULL;
  check(video_encode_mp4(pixel, 1, 3, 4, 24, NULL, 0, 0, &len, &err) == NULL &&
            err != NULL,
        "odd width refused");
  err = NULL;
  check(video_encode_mp4(pixel, 1, 16, 16, 0, NULL, 0, 0, &len, &err) == NULL &&
            err != NULL,
        "zero fps refused");
  err = NULL;
  float sample[2] = {0.0f, 0.0f};
  check(video_encode_mp4(pixel, 1, 16, 16, 24, sample, 1, 100, &len, &err) ==
                NULL &&
            err != NULL,
        "bad audio sample rate refused");
  return failures != 0;
}
