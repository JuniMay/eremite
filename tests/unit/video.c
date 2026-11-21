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

  err = NULL;
  VideoDecoded dec = {0};
  check(video_decode_mp4(NULL, 0, 24, 32000, &dec, &err) == 0 && err != NULL,
        "empty video refused");
  err = NULL;
  check(video_decode_mp4(pixel, sizeof pixel, 0, 32000, &dec, &err) == 0 &&
            err != NULL,
        "bad decode fps refused");

  err = NULL;
  int samples = 0;
  check(audio_decode_pcm(NULL, 0, 32000, &samples, &err) == NULL && err != NULL,
        "empty audio refused");
  err = NULL;
  check(audio_decode_pcm(pixel, sizeof pixel, 100, &samples, &err) == NULL &&
            err != NULL,
        "bad decode sample rate refused");

  video_decoded_free(&dec);
  return failures != 0;
}
