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

// Maximum frame count decoded from one uploaded video clip (150 s at 24 fps).
static constexpr int VIDEO_MAX_DECODE_FRAMES = 3600;

// Maximum audio duration decoded from one uploaded clip in seconds.
static constexpr int VIDEO_MAX_DECODE_AUDIO_SECONDS = 60;

// Maximum mono audio duration decoded for speech transcription in seconds.
static constexpr int AUDIO_MAX_DECODE_SECONDS = 3600;

// Decoded video frames and optional stereo soundtrack from an input clip.
typedef struct {
  // Heap-allocated 8-bit RGB frames [frame_count, height, width, 3].
  uint8_t *rgb;
  // Number of decoded frames and frame dimensions in pixels.
  int frame_count, width, height;
  // Heap-allocated planar stereo FP32 PCM [2, audio_samples] in [-1, 1], or
  // null when the input clip carries no audio track.
  float *audio_pcm;
  // Samples per channel and sample rate in Hz of audio_pcm.
  int audio_samples, audio_sample_rate;
} VideoDecoded;

// Releases decoded frame and audio buffers and zeroes all fields of decoded.
void video_decoded_free(VideoDecoded *decoded);

// Encodes frame_count RGB24 frames (width by height) at fps as an H.264 MP4
// file in a new buffer and sets *length. When audio_pcm is non-null and
// audio_samples > 0, muxes planar stereo FP32 PCM [2, audio_samples] at
// sample_rate as AAC stereo. Returns null with *error set on failure.
uint8_t *video_encode_mp4(const uint8_t *rgb, int frame_count, int width,
                          int height, int fps, const float *audio_pcm,
                          int audio_samples, int sample_rate, size_t *length,
                          const char **error);

// Decodes an encoded video clip into target_fps RGB24 frames and optional
// planar stereo FP32 audio at target_sample_rate (when > 0). Returns 1 on
// success or 0 with *error set on failure.
int video_decode_mp4(const void *data, size_t len, int target_fps,
                     int target_sample_rate, VideoDecoded *out,
                     const char **error);

// Decodes an encoded audio clip into planar stereo FP32 PCM [2, *out_samples]
// at target_sample_rate in a new buffer. Returns null with *error set on
// failure.
float *audio_decode_pcm(const void *data, size_t len, int target_sample_rate,
                        int *out_samples, const char **error);

// Decodes an encoded audio or video clip into mono FP32 PCM [*out_samples] in
// [-1, 1] at target_sample_rate in a new buffer, rejecting clips longer than
// max_seconds. Returns null with *error set on failure.
float *audio_decode_pcm_mono(const void *data, size_t len,
                             int target_sample_rate, int max_seconds,
                             int *out_samples, const char **error);

#endif
