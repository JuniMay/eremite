#include "platform/video.h"
#include "base/error.h"
#include <AVFoundation/AVFoundation.h>
#include <AudioToolbox/AudioToolbox.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Uses AVFoundation and CoreMedia in manual-retain Objective-C. Every entry
// point wraps its work in an autorelease pool and removes temporary container
// files before returning.

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

// Smallest and largest frame dimensions accepted by H.264 encoding and video
// decoding.
static constexpr int MIN_VIDEO_DIM = 2, MAX_VIDEO_DIM = 8192;

// Bytes per pixel in 32-bit BGRA CoreVideo pixel buffers.
static constexpr int BGRA_BYTES = 4;

// Opaque alpha byte written into BGRA pixel buffers.
static constexpr uint8_t ALPHA_OPAQUE = 255;

// Bits per pixel per frame used to size the H.264 average bitrate target.
static constexpr double H264_BITS_PER_PIXEL = 0.25;

// Lower and upper H.264 average bitrate bounds in bits per second.
static constexpr int H264_MIN_BITRATE = 256000, H264_MAX_BITRATE = 32000000;

// Stereo AAC output bitrate in bits per second.
static constexpr int AAC_BITRATE = 192000;

// Samples per stereo audio buffer appended to AVAssetWriter.
static constexpr int AUDIO_CHUNK_SAMPLES = 1024;

// Float32 bits per sample for linear PCM CoreAudio descriptors.
static constexpr int PCM_FLOAT_BITS = 32;

// Temporary file path buffer capacity.
static constexpr size_t TEMP_PATH_CAP = 512;

// Sets *error when non-null.
static void set_error(const char **error, const char *message) {
  if (error) *error = message;
}

// Clamps v to [-1, 1] and replaces non-finite values with 0.
static float clamp_pcm(float v) {
  if (!isfinite(v)) return 0.0f;
  if (v < -1.0f) return -1.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

// Creates a unique temporary file ending in suffix and writes its path into
// out_path. Returns an open file descriptor or -1 on failure.
static int make_temp_file(const char *suffix, char *out_path, size_t cap) {
  const char *base = getenv("TMPDIR");
  if (!base || !base[0]) base = "/tmp";
  size_t blen = strlen(base);
  while (blen > 1 && base[blen - 1] == '/') blen--;
  int n = snprintf(out_path, cap, "%.*s/eremite-av-XXXXXX%s", (int)blen, base,
                   suffix);
  if (n < 0 || (size_t)n >= cap) return -1;
  return mkstemps(out_path, (int)strlen(suffix));
}

// Reads all bytes of path into a newly allocated buffer and sets *out_len.
static uint8_t *read_whole_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseeko(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  off_t size = ftello(f);
  if (size <= 0 || fseeko(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *buf = malloc((size_t)size);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_len = (size_t)size;
  return buf;
}

// Creates a CoreVideo BGRA pixel buffer populated from one RGB24 frame.
static CVPixelBufferRef make_bgra_buffer(CVPixelBufferPoolRef pool,
                                         const uint8_t *rgb, int width,
                                         int height) {
  CVPixelBufferRef px = NULL;
  CVReturn rc = pool
                    ? CVPixelBufferPoolCreatePixelBuffer(NULL, pool, &px)
                    : CVPixelBufferCreate(NULL, (size_t)width, (size_t)height,
                                          kCVPixelFormatType_32BGRA, NULL, &px);
  if (rc != kCVReturnSuccess || !px) return NULL;
  CVPixelBufferLockBaseAddress(px, 0);
  uint8_t *dst = CVPixelBufferGetBaseAddress(px);
  size_t stride = CVPixelBufferGetBytesPerRow(px);
  for (int y = 0; y < height; y++) {
    const uint8_t *srow = rgb + (size_t)y * (size_t)width * VIDEO_RGB_CHANNELS;
    uint8_t *drow = dst + (size_t)y * stride;
    for (int x = 0; x < width; x++) {
      drow[x * BGRA_BYTES + 0] = srow[x * VIDEO_RGB_CHANNELS + 2];
      drow[x * BGRA_BYTES + 1] = srow[x * VIDEO_RGB_CHANNELS + 1];
      drow[x * BGRA_BYTES + 2] = srow[x * VIDEO_RGB_CHANNELS + 0];
      drow[x * BGRA_BYTES + 3] = ALPHA_OPAQUE;
    }
  }
  CVPixelBufferUnlockBaseAddress(px, 0);
  return px;
}

// Builds an interleaved FP32 stereo CMSampleBuffer for [offset, offset + count)
// from planar stereo PCM [2, total_samples].
static CMSampleBufferRef make_audio_sample_buffer(const float *planar_pcm,
                                                  int total_samples, int offset,
                                                  int count, int sample_rate) {
  AudioStreamBasicDescription asbd = {0};
  asbd.mSampleRate = (Float64)sample_rate;
  asbd.mFormatID = kAudioFormatLinearPCM;
  asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  asbd.mFramesPerPacket = 1;
  asbd.mChannelsPerFrame = VIDEO_AUDIO_CHANNELS;
  asbd.mBitsPerChannel = PCM_FLOAT_BITS;
  asbd.mBytesPerFrame = VIDEO_AUDIO_CHANNELS * sizeof(float);
  asbd.mBytesPerPacket = asbd.mBytesPerFrame;

  CMAudioFormatDescriptionRef fmt = NULL;
  if (CMAudioFormatDescriptionCreate(NULL, &asbd, 0, NULL, 0, NULL, NULL,
                                     &fmt) != noErr)
    return NULL;

  size_t bytes = (size_t)count * (size_t)asbd.mBytesPerFrame;
  CMBlockBufferRef block = NULL;
  if (CMBlockBufferCreateWithMemoryBlock(
          NULL, NULL, bytes, NULL, NULL, 0, bytes,
          kCMBlockBufferAssureMemoryNowFlag, &block) != kCMBlockBufferNoErr) {
    CFRelease(fmt);
    return NULL;
  }
  char *dst_raw = NULL;
  if (CMBlockBufferGetDataPointer(block, 0, NULL, NULL, &dst_raw) !=
      kCMBlockBufferNoErr) {
    CFRelease(block);
    CFRelease(fmt);
    return NULL;
  }
  float *dst = (float *)dst_raw;
  const float *left = planar_pcm + offset;
  const float *right = planar_pcm + total_samples + offset;
  for (int i = 0; i < count; i++) {
    dst[i * VIDEO_AUDIO_CHANNELS + 0] = clamp_pcm(left[i]);
    dst[i * VIDEO_AUDIO_CHANNELS + 1] = clamp_pcm(right[i]);
  }

  CMSampleTimingInfo timing = {0};
  timing.duration = CMTimeMake(1, sample_rate);
  timing.presentationTimeStamp = CMTimeMake(offset, sample_rate);
  timing.decodeTimeStamp = kCMTimeInvalid;
  size_t sample_size = (size_t)asbd.mBytesPerFrame;

  CMSampleBufferRef sb = NULL;
  OSStatus st = CMSampleBufferCreate(NULL, block, true, NULL, NULL, fmt, count,
                                     1, &timing, 1, &sample_size, &sb);
  CFRelease(block);
  CFRelease(fmt);
  return st == noErr ? sb : NULL;
}

uint8_t *video_encode_mp4(const uint8_t *rgb, int frame_count, int width,
                          int height, int fps, const float *audio_pcm,
                          int audio_samples, int sample_rate, size_t *length,
                          const char **error) {
  if (length) *length = 0;
  if (!rgb || frame_count < 1 || width < MIN_VIDEO_DIM ||
      height < MIN_VIDEO_DIM || width > MAX_VIDEO_DIM ||
      height > MAX_VIDEO_DIM || (width & 1) != 0 || (height & 1) != 0) {
    set_error(error, "video dimensions must be even and between 2 and 8192");
    return NULL;
  }
  if (fps < VIDEO_MIN_FPS || fps > VIDEO_MAX_FPS) {
    set_error(error, "video frame rate is out of range");
    return NULL;
  }
  int has_audio = audio_pcm != NULL && audio_samples > 0;
  if (has_audio && (sample_rate < VIDEO_MIN_SAMPLE_RATE ||
                    sample_rate > VIDEO_MAX_SAMPLE_RATE)) {
    set_error(error, "audio sample rate is out of range");
    return NULL;
  }

  char path[TEMP_PATH_CAP];
  int fd = make_temp_file(".mp4", path, sizeof path);
  if (fd < 0) {
    set_error(error, "cannot create temporary MP4 file");
    return NULL;
  }
  close(fd);
  unlink(path);

  void *pool = objc_autoreleasePoolPush();
  uint8_t *result = NULL;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  NSError *ns_err = nil;
  AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:url
                                                    fileType:AVFileTypeMPEG4
                                                       error:&ns_err];
  if (!writer) {
    objc_autoreleasePoolPop(pool);
    set_error(error, "cannot create MP4 writer");
    return NULL;
  }

  double raw_bitrate =
      (double)width * (double)height * (double)fps * H264_BITS_PER_PIXEL;
  int bitrate = (int)llround(raw_bitrate);
  if (bitrate < H264_MIN_BITRATE) bitrate = H264_MIN_BITRATE;
  if (bitrate > H264_MAX_BITRATE) bitrate = H264_MAX_BITRATE;

  NSDictionary *video_settings = @{
    AVVideoCodecKey : AVVideoCodecTypeH264,
    AVVideoWidthKey : @(width),
    AVVideoHeightKey : @(height),
    AVVideoColorPropertiesKey : @{
      AVVideoColorPrimariesKey : AVVideoColorPrimaries_ITU_R_709_2,
      AVVideoTransferFunctionKey : AVVideoTransferFunction_ITU_R_709_2,
      AVVideoYCbCrMatrixKey : AVVideoYCbCrMatrix_ITU_R_709_2,
    },
    AVVideoCompressionPropertiesKey : @{
      AVVideoAverageBitRateKey : @(bitrate),
      AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
    },
  };
  AVAssetWriterInput *v_in =
      [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeVideo
                                     outputSettings:video_settings];
  v_in.expectsMediaDataInRealTime = NO;
  NSDictionary *px_attrs = @{
    (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
    (NSString *)kCVPixelBufferWidthKey : @(width),
    (NSString *)kCVPixelBufferHeightKey : @(height),
  };
  AVAssetWriterInputPixelBufferAdaptor *adaptor =
      [[AVAssetWriterInputPixelBufferAdaptor alloc]
             initWithAssetWriterInput:v_in
          sourcePixelBufferAttributes:px_attrs];

  AVAssetWriterInput *a_in = nil;
  if (has_audio) {
    NSDictionary *audio_settings = @{
      AVFormatIDKey : @(kAudioFormatMPEG4AAC),
      AVSampleRateKey : @(sample_rate),
      AVNumberOfChannelsKey : @(VIDEO_AUDIO_CHANNELS),
      AVEncoderBitRateKey : @(AAC_BITRATE),
    };
    a_in = [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeAudio
                                          outputSettings:audio_settings];
    a_in.expectsMediaDataInRealTime = NO;
  }

  if (![writer canAddInput:v_in] || (a_in && ![writer canAddInput:a_in])) {
    [a_in release];
    [adaptor release];
    [v_in release];
    [writer release];
    objc_autoreleasePoolPop(pool);
    set_error(error, "cannot configure MP4 tracks");
    return NULL;
  }
  [writer addInput:v_in];
  if (a_in) [writer addInput:a_in];
  if (![writer startWriting]) {
    [a_in release];
    [adaptor release];
    [v_in release];
    [writer release];
    objc_autoreleasePoolPop(pool);
    set_error(error, "cannot start MP4 writing");
    return NULL;
  }
  [writer startSessionAtSourceTime:kCMTimeZero];

  dispatch_queue_t v_queue =
      dispatch_queue_create("eremite.video.encode", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t v_done = dispatch_semaphore_create(0);
  __block int v_idx = 0;
  __block int v_ok = 1;
  size_t frame_bytes = (size_t)width * (size_t)height * VIDEO_RGB_CHANNELS;

  [v_in requestMediaDataWhenReadyOnQueue:v_queue
                              usingBlock:^{
                                while (v_in.isReadyForMoreMediaData) {
                                  if (v_idx >= frame_count || !v_ok) {
                                    [v_in markAsFinished];
                                    dispatch_semaphore_signal(v_done);
                                    return;
                                  }
                                  CVPixelBufferRef px = make_bgra_buffer(
                                      adaptor.pixelBufferPool,
                                      rgb + (size_t)v_idx * frame_bytes, width,
                                      height);
                                  CMTime pts = CMTimeMake(v_idx, fps);
                                  if (!px || ![adaptor appendPixelBuffer:px
                                                    withPresentationTime:pts]) {
                                    v_ok = 0;
                                  }
                                  if (px) CVPixelBufferRelease(px);
                                  v_idx++;
                                }
                              }];

  dispatch_queue_t a_queue = NULL;
  dispatch_semaphore_t a_done = NULL;
  __block int a_off = 0;
  __block int a_ok = 1;
  if (a_in) {
    a_queue =
        dispatch_queue_create("eremite.audio.encode", DISPATCH_QUEUE_SERIAL);
    a_done = dispatch_semaphore_create(0);
    [a_in requestMediaDataWhenReadyOnQueue:a_queue
                                usingBlock:^{
                                  while (a_in.isReadyForMoreMediaData) {
                                    if (a_off >= audio_samples || !a_ok) {
                                      [a_in markAsFinished];
                                      dispatch_semaphore_signal(a_done);
                                      return;
                                    }
                                    int chunk = audio_samples - a_off;
                                    if (chunk > AUDIO_CHUNK_SAMPLES)
                                      chunk = AUDIO_CHUNK_SAMPLES;
                                    CMSampleBufferRef sb =
                                        make_audio_sample_buffer(
                                            audio_pcm, audio_samples, a_off,
                                            chunk, sample_rate);
                                    if (!sb || ![a_in appendSampleBuffer:sb])
                                      a_ok = 0;
                                    if (sb) CFRelease(sb);
                                    a_off += chunk;
                                  }
                                }];
  }

  dispatch_semaphore_wait(v_done, DISPATCH_TIME_FOREVER);
  if (a_done) dispatch_semaphore_wait(a_done, DISPATCH_TIME_FOREVER);

  dispatch_semaphore_t fin = dispatch_semaphore_create(0);
  [writer
      finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(fin); }];
  dispatch_semaphore_wait(fin, DISPATCH_TIME_FOREVER);

  int ok = v_ok && a_ok && writer.status == AVAssetWriterStatusCompleted;
  if (ok) {
    size_t bytes = 0;
    result = read_whole_file(path, &bytes);
    if (result && length) *length = bytes;
    if (!result) set_error(error, "cannot read encoded MP4 bytes");
  } else {
    set_error(error, "MP4 encoding failed");
  }

  unlink(path);
  dispatch_release(fin);
  dispatch_release(v_done);
  dispatch_release(v_queue);
  if (a_done) dispatch_release(a_done);
  if (a_queue) dispatch_release(a_queue);
  [a_in release];
  [adaptor release];
  [v_in release];
  [writer release];
  objc_autoreleasePoolPop(pool);
  return result;
}
