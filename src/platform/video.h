#ifndef EREMITE_PLATFORM_VIDEO_H
#define EREMITE_PLATFORM_VIDEO_H
#include <stddef.h>
#include <stdint.h>

// Channels per RGB video frame pixel.
static constexpr int VIDEO_RGB_CHANNELS = 3;

// Audio channels (stereo) encoded and decoded by the platform video layer.
static constexpr int VIDEO_AUDIO_CHANNELS = 2;

// Smallest and largest frame rates accepted by video encoding and decoding.
static constexpr int VIDEO_MIN_FPS = 1, VIDEO_MAX_FPS = 120;

// Smallest and largest audio sample rates accepted in Hz.
static constexpr int VIDEO_MIN_SAMPLE_RATE = 8000,
                     VIDEO_MAX_SAMPLE_RATE = 96000;

// Encodes frame_count RGB24 frames (width by height) at fps as an H.264 MP4
// file in a new buffer and sets *length. When audio_pcm is non-null and
// audio_samples > 0, muxes planar stereo FP32 PCM [2, audio_samples] at
// sample_rate as AAC stereo. Returns null with *error set on failure.
uint8_t *video_encode_mp4(const uint8_t *rgb, int frame_count, int width,
                          int height, int fps, const float *audio_pcm,
                          int audio_samples, int sample_rate, size_t *length,
                          const char **error);

#endif
