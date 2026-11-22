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

// Half-frame rounding offset for resampling decoded video frames.
static constexpr double HALF_STEP = 0.5;

// Minimum input byte count and container header tag lengths for format
// sniffing.
static constexpr size_t MAGIC_MIN_BYTES = 12, FOURCC_BYTES = 4,
                        RIFF_TAG_OFFSET = 8, ID3_TAG_BYTES = 3;

// MPEG audio frame sync header bytes used to identify raw MP3 streams.
static constexpr uint8_t MP3_SYNC_BYTE0 = 0xff, MP3_SYNC_MASK = 0xe0,
                         MP3_LAYER_MASK = 0x06;

// EBML container magic bytes used to identify WebM and Matroska streams.
static constexpr uint8_t EBML_MAGIC[4] = {0x1a, 0x45, 0xdf, 0xa3};

// Mono channel count for single-channel PCM decoding.
static constexpr int MONO_AUDIO_CHANNELS = 1;

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

// Writes len bytes of data into a new temporary file with suffix and closes
// its descriptor. Returns 1 on success or 0 on failure.
static int write_temp_file(const void *data, size_t len, const char *suffix,
                           char *out_path, size_t cap) {
  int fd = make_temp_file(suffix, out_path, cap);
  if (fd < 0) return 0;
  const uint8_t *p = data;
  size_t left = len;
  while (left > 0) {
    ssize_t wrote = write(fd, p, left);
    if (wrote <= 0) {
      close(fd);
      unlink(out_path);
      return 0;
    }
    p += wrote;
    left -= (size_t)wrote;
  }
  if (close(fd) != 0) {
    unlink(out_path);
    return 0;
  }
  return 1;
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

// Picks a file extension hint for AVURLAsset from container magic bytes.
static const char *sniff_media_suffix(const void *data, size_t len,
                                      int is_video) {
  const uint8_t *b = data;
  if (len >= MAGIC_MIN_BYTES) {
    if (memcmp(b, "RIFF", FOURCC_BYTES) == 0 &&
        memcmp(b + RIFF_TAG_OFFSET, "WAVE", FOURCC_BYTES) == 0)
      return ".wav";
    if (memcmp(b, "fLaC", FOURCC_BYTES) == 0) return ".flac";
    if (memcmp(b, "FORM", FOURCC_BYTES) == 0 &&
        (memcmp(b + RIFF_TAG_OFFSET, "AIFF", FOURCC_BYTES) == 0 ||
         memcmp(b + RIFF_TAG_OFFSET, "AIFC", FOURCC_BYTES) == 0))
      return ".aiff";
    if (memcmp(b, "caff", FOURCC_BYTES) == 0) return ".caf";
    if (memcmp(b, "OggS", FOURCC_BYTES) == 0) return ".ogg";
    if (memcmp(b, EBML_MAGIC, FOURCC_BYTES) == 0) return ".webm";
    if (memcmp(b + FOURCC_BYTES, "ftyp", FOURCC_BYTES) == 0)
      return is_video ? ".mp4" : ".m4a";
    if (memcmp(b, "ID3", ID3_TAG_BYTES) == 0 ||
        (b[0] == MP3_SYNC_BYTE0 && (b[1] & MP3_SYNC_MASK) == MP3_SYNC_MASK &&
         (b[1] & MP3_LAYER_MASK) != 0))
      return ".mp3";
  }
  return is_video ? ".mp4" : ".m4a";
}

// Synchronously loads asset's tracks of media_type using AVFoundation's async
// loader. Returns a retained NSArray or nil on failure.
static NSArray<AVAssetTrack *> *load_tracks(AVAsset *asset,
                                            AVMediaType media_type) {
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block NSArray<AVAssetTrack *> *result = nil;
  [asset
      loadTracksWithMediaType:media_type
            completionHandler:^(NSArray<AVAssetTrack *> *tracks, NSError *err) {
              (void)err;
              if (tracks) result = [tracks retain];
              dispatch_semaphore_signal(sem);
            }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  dispatch_release(sem);
  return result;
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

// Copies a 32-bit BGRA CVPixelBuffer into an RGB24 frame buffer.
static void copy_bgra_to_rgb(CVPixelBufferRef px, uint8_t *dst, int width,
                             int height) {
  CVPixelBufferLockBaseAddress(px, kCVPixelBufferLock_ReadOnly);
  const uint8_t *src = CVPixelBufferGetBaseAddress(px);
  size_t stride = CVPixelBufferGetBytesPerRow(px);
  for (int y = 0; y < height; y++) {
    const uint8_t *srow = src + (size_t)y * stride;
    uint8_t *drow = dst + (size_t)y * (size_t)width * VIDEO_RGB_CHANNELS;
    for (int x = 0; x < width; x++) {
      drow[x * VIDEO_RGB_CHANNELS + 0] = srow[x * BGRA_BYTES + 2];
      drow[x * VIDEO_RGB_CHANNELS + 1] = srow[x * BGRA_BYTES + 1];
      drow[x * VIDEO_RGB_CHANNELS + 2] = srow[x * BGRA_BYTES + 0];
    }
  }
  CVPixelBufferUnlockBaseAddress(px, kCVPixelBufferLock_ReadOnly);
}

// Reads all FP32 PCM samples from an audio track of asset at
// target_sample_rate with the requested channel count (1 for mono, 2 for planar
// stereo) up to max_seconds.
static float *read_audio_track(AVAsset *asset, AVAssetTrack *track,
                               int target_sample_rate, int channels,
                               int max_seconds, int *out_samples,
                               const char **error) {
  NSError *ns_err = nil;
  AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset
                                                         error:&ns_err];
  if (!reader) {
    set_error(error, "cannot open audio reader");
    return NULL;
  }
  NSDictionary *settings = @{
    AVFormatIDKey : @(kAudioFormatLinearPCM),
    AVSampleRateKey : @(target_sample_rate),
    AVNumberOfChannelsKey : @(channels),
    AVLinearPCMBitDepthKey : @(PCM_FLOAT_BITS),
    AVLinearPCMIsFloatKey : @YES,
    AVLinearPCMIsBigEndianKey : @NO,
    AVLinearPCMIsNonInterleaved : @NO,
  };
  AVAssetReaderTrackOutput *output =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:track
                                       outputSettings:settings];
  output.alwaysCopiesSampleData = NO;
  if (![reader canAddOutput:output]) {
    [output release];
    [reader release];
    set_error(error, "cannot configure audio decoder");
    return NULL;
  }
  [reader addOutput:output];
  if (![reader startReading]) {
    [output release];
    [reader release];
    set_error(error, "cannot decode audio track");
    return NULL;
  }

  size_t max_samples = (size_t)target_sample_rate * (size_t)max_seconds;
  size_t cap = (size_t)target_sample_rate;
  size_t count = 0;
  float *left = malloc(cap * sizeof(float));
  float *right =
      channels == VIDEO_AUDIO_CHANNELS ? malloc(cap * sizeof(float)) : NULL;
  if (!left || (channels == VIDEO_AUDIO_CHANNELS && !right)) {
    free(left);
    free(right);
    [reader cancelReading];
    [output release];
    [reader release];
    set_error(error, "audio buffer allocation failed");
    return NULL;
  }

  int overflow = 0;
  CMSampleBufferRef sb = NULL;
  while ((sb = [output copyNextSampleBuffer]) != NULL) {
    CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sb);
    size_t n_frames = (size_t)CMSampleBufferGetNumSamples(sb);
    if (block && n_frames > 0) {
      size_t byte_len = 0;
      char *raw = NULL;
      if (CMBlockBufferGetDataPointer(block, 0, NULL, &byte_len, &raw) ==
          kCMBlockBufferNoErr) {
        size_t avail = byte_len / ((size_t)channels * sizeof(float));
        if (n_frames > avail) n_frames = avail;
        if (count + n_frames > max_samples) {
          overflow = 1;
          CFRelease(sb);
          break;
        }
        if (count + n_frames > cap) {
          size_t next = cap * 2;
          while (next < count + n_frames) next *= 2;
          float *nl = realloc(left, next * sizeof(float));
          float *nr = channels == VIDEO_AUDIO_CHANNELS
                          ? realloc(right, next * sizeof(float))
                          : NULL;
          if (!nl || (channels == VIDEO_AUDIO_CHANNELS && !nr)) {
            if (nl) left = nl;
            if (nr) right = nr;
            free(left);
            free(right);
            CFRelease(sb);
            [reader cancelReading];
            [output release];
            [reader release];
            set_error(error, "audio buffer allocation failed");
            return NULL;
          }
          left = nl;
          right = nr;
          cap = next;
        }
        const float *inter = (const float *)raw;
        if (channels == MONO_AUDIO_CHANNELS) {
          for (size_t i = 0; i < n_frames; i++)
            left[count + i] = clamp_pcm(inter[i]);
        } else {
          for (size_t i = 0; i < n_frames; i++) {
            left[count + i] = clamp_pcm(inter[i * VIDEO_AUDIO_CHANNELS + 0]);
            right[count + i] = clamp_pcm(inter[i * VIDEO_AUDIO_CHANNELS + 1]);
          }
        }
        count += n_frames;
      }
    }
    CFRelease(sb);
  }

  AVAssetReaderStatus status = reader.status;
  [output release];
  [reader release];
  if (overflow) {
    free(left);
    free(right);
    set_error(error, "audio clip exceeds maximum supported duration");
    return NULL;
  }
  if (status == AVAssetReaderStatusFailed || count == 0) {
    free(left);
    free(right);
    set_error(error, "cannot decode audio track");
    return NULL;
  }

  if (channels == MONO_AUDIO_CHANNELS) {
    *out_samples = (int)count;
    return left;
  }

  float *planar = malloc(count * VIDEO_AUDIO_CHANNELS * sizeof(float));
  if (!planar) {
    free(left);
    free(right);
    set_error(error, "audio buffer allocation failed");
    return NULL;
  }
  memcpy(planar, left, count * sizeof(float));
  memcpy(planar + count, right, count * sizeof(float));
  free(left);
  free(right);
  *out_samples = (int)count;
  return planar;
}

void video_decoded_free(VideoDecoded *decoded) {
  if (!decoded) return;
  free(decoded->rgb);
  free(decoded->audio_pcm);
  memset(decoded, 0, sizeof *decoded);
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

int video_decode_mp4(const void *data, size_t len, int target_fps,
                     int target_sample_rate, VideoDecoded *out,
                     const char **error) {
  if (!out) {
    set_error(error, "missing video output buffer");
    return 0;
  }
  memset(out, 0, sizeof *out);
  if (!data || len == 0) {
    set_error(error, "video data is empty");
    return 0;
  }
  if (target_fps < VIDEO_MIN_FPS || target_fps > VIDEO_MAX_FPS) {
    set_error(error, "target video frame rate is out of range");
    return 0;
  }
  if (target_sample_rate != 0 && (target_sample_rate < VIDEO_MIN_SAMPLE_RATE ||
                                  target_sample_rate > VIDEO_MAX_SAMPLE_RATE)) {
    set_error(error, "target audio sample rate is out of range");
    return 0;
  }

  char path[TEMP_PATH_CAP];
  const char *suffix = sniff_media_suffix(data, len, 1);
  if (!write_temp_file(data, len, suffix, path, sizeof path)) {
    set_error(error, "cannot stage video data");
    return 0;
  }

  void *pool = objc_autoreleasePoolPush();
  int ok = 0;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  AVURLAsset *asset = [[AVURLAsset alloc] initWithURL:url options:nil];
  NSArray<AVAssetTrack *> *v_tracks = load_tracks(asset, AVMediaTypeVideo);
  if (!v_tracks || v_tracks.count == 0) {
    [v_tracks release];
    set_error(error, "file contains no video track");
    goto done;
  }
  AVAssetTrack *v_track = v_tracks[0];
  CGSize nat = v_track.naturalSize;
  int width = (int)llround(fabs(nat.width));
  int height = (int)llround(fabs(nat.height));
  if (width < 1 || height < 1 || width > MAX_VIDEO_DIM ||
      height > MAX_VIDEO_DIM) {
    [v_tracks release];
    set_error(error, "video dimensions are out of range");
    goto done;
  }

  NSError *ns_err = nil;
  AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset
                                                         error:&ns_err];
  if (!reader) {
    [v_tracks release];
    set_error(error, "cannot open video reader");
    goto done;
  }
  NSDictionary *v_settings = @{
    (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
    (NSString *)kCVPixelBufferWidthKey : @(width),
    (NSString *)kCVPixelBufferHeightKey : @(height),
  };
  AVAssetReaderTrackOutput *v_out =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:v_track
                                       outputSettings:v_settings];
  [v_tracks release];
  v_out.alwaysCopiesSampleData = NO;
  if (![reader canAddOutput:v_out] ||
      ([reader addOutput:v_out], ![reader startReading])) {
    [v_out release];
    [reader release];
    set_error(error, "cannot start decoding video track");
    goto done;
  }

  size_t frame_bytes = (size_t)width * (size_t)height * VIDEO_RGB_CHANNELS;
  int cap = target_fps;
  int count = 0;
  uint8_t *frames = malloc((size_t)cap * frame_bytes);
  uint8_t *prev = malloc(frame_bytes);
  if (!frames || !prev) {
    free(frames);
    free(prev);
    [reader cancelReading];
    [v_out release];
    [reader release];
    set_error(error, "video buffer allocation failed");
    goto done;
  }

  int have_prev = 0;
  double prev_pts = 0.0;
  double first_pts = 0.0;
  int overflow = 0;
  CMSampleBufferRef sb = NULL;
  while ((sb = [v_out copyNextSampleBuffer]) != NULL) {
    CVImageBufferRef img = CMSampleBufferGetImageBuffer(sb);
    CMTime pts_t = CMSampleBufferGetPresentationTimeStamp(sb);
    double pts = CMTIME_IS_VALID(pts_t) ? CMTimeGetSeconds(pts_t) : 0.0;
    if (img) {
      if (!have_prev) {
        first_pts = pts;
        prev_pts = 0.0;
        copy_bgra_to_rgb(img, prev, width, height);
        have_prev = 1;
        memcpy(frames, prev, frame_bytes);
        count = 1;
      } else {
        double rel = pts - first_pts;
        if (rel < prev_pts) rel = prev_pts;
        while ((double)count / (double)target_fps <=
               rel + HALF_STEP / target_fps) {
          if (count >= VIDEO_MAX_DECODE_FRAMES) {
            overflow = 1;
            break;
          }
          if (count >= cap) {
            int next = cap * 2;
            if (next > VIDEO_MAX_DECODE_FRAMES) next = VIDEO_MAX_DECODE_FRAMES;
            uint8_t *nf = realloc(frames, (size_t)next * frame_bytes);
            if (!nf) {
              free(frames);
              free(prev);
              CFRelease(sb);
              [reader cancelReading];
              [v_out release];
              [reader release];
              set_error(error, "video buffer allocation failed");
              goto done;
            }
            frames = nf;
            cap = next;
          }
          double target_t = (double)count / (double)target_fps;
          if (fabs(target_t - prev_pts) < fabs(rel - target_t))
            memcpy(frames + (size_t)count * frame_bytes, prev, frame_bytes);
          else
            copy_bgra_to_rgb(img, frames + (size_t)count * frame_bytes, width,
                             height);
          count++;
        }
        copy_bgra_to_rgb(img, prev, width, height);
        prev_pts = rel;
      }
    }
    CFRelease(sb);
    if (overflow) break;
  }

  AVAssetReaderStatus status = reader.status;
  [v_out release];
  [reader release];
  free(prev);

  if (overflow) {
    free(frames);
    set_error(error, "video clip exceeds maximum supported frame count");
    goto done;
  }
  if (status == AVAssetReaderStatusFailed || count == 0) {
    free(frames);
    set_error(error, "cannot decode video frames");
    goto done;
  }

  out->rgb = frames;
  out->frame_count = count;
  out->width = width;
  out->height = height;

  if (target_sample_rate > 0) {
    NSArray<AVAssetTrack *> *a_tracks = load_tracks(asset, AVMediaTypeAudio);
    if (a_tracks && a_tracks.count > 0) {
      int a_samples = 0;
      float *pcm = read_audio_track(
          asset, a_tracks[0], target_sample_rate, VIDEO_AUDIO_CHANNELS,
          VIDEO_MAX_DECODE_AUDIO_SECONDS, &a_samples, error);
      [a_tracks release];
      if (!pcm) {
        video_decoded_free(out);
        goto done;
      }
      out->audio_pcm = pcm;
      out->audio_samples = a_samples;
      out->audio_sample_rate = target_sample_rate;
    } else {
      [a_tracks release];
    }
  }
  ok = 1;

done:
  [asset release];
  objc_autoreleasePoolPop(pool);
  unlink(path);
  return ok;
}

// Decodes an encoded audio clip into FP32 PCM with the requested channel count
// and duration limit in seconds.
static float *decode_audio_pcm_common(const void *data, size_t len,
                                      int target_sample_rate, int channels,
                                      int max_seconds, int *out_samples,
                                      const char **error) {
  if (out_samples) *out_samples = 0;
  if (!data || len == 0) {
    set_error(error, "audio data is empty");
    return NULL;
  }
  if (target_sample_rate < VIDEO_MIN_SAMPLE_RATE ||
      target_sample_rate > VIDEO_MAX_SAMPLE_RATE) {
    set_error(error, "target audio sample rate is out of range");
    return NULL;
  }
  if (max_seconds < 1 || max_seconds > AUDIO_MAX_DECODE_SECONDS) {
    set_error(error, "maximum audio duration is out of range");
    return NULL;
  }

  char path[TEMP_PATH_CAP];
  const char *suffix = sniff_media_suffix(data, len, 0);
  if (!write_temp_file(data, len, suffix, path, sizeof path)) {
    set_error(error, "cannot stage audio data");
    return NULL;
  }

  void *pool = objc_autoreleasePoolPush();
  float *pcm = NULL;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  AVURLAsset *asset = [[AVURLAsset alloc] initWithURL:url options:nil];
  NSArray<AVAssetTrack *> *a_tracks = load_tracks(asset, AVMediaTypeAudio);
  if (!a_tracks || a_tracks.count == 0) {
    set_error(error, "file contains no audio track");
  } else {
    int samples = 0;
    pcm = read_audio_track(asset, a_tracks[0], target_sample_rate, channels,
                           max_seconds, &samples, error);
    if (pcm && out_samples) *out_samples = samples;
  }
  [a_tracks release];
  [asset release];
  objc_autoreleasePoolPop(pool);
  unlink(path);
  return pcm;
}

float *audio_decode_pcm(const void *data, size_t len, int target_sample_rate,
                        int *out_samples, const char **error) {
  return decode_audio_pcm_common(
      data, len, target_sample_rate, VIDEO_AUDIO_CHANNELS,
      VIDEO_MAX_DECODE_AUDIO_SECONDS, out_samples, error);
}

float *audio_decode_pcm_mono(const void *data, size_t len,
                             int target_sample_rate, int max_seconds,
                             int *out_samples, const char **error) {
  return decode_audio_pcm_common(data, len, target_sample_rate,
                                 MONO_AUDIO_CHANNELS, max_seconds, out_samples,
                                 error);
}
