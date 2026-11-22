#ifndef EREMITE_MODEL_MODEL_H
#define EREMITE_MODEL_MODEL_H
#include "engine/expert_store.h"
#include "engine/media.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/sampling.h"
#include "gpu/gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Every model behind one interface. A model family fills a ModelOps table.
// model_open reads a checkpoint's config.json, picks the family that runs its
// model_type, and loads the model.
//
// A process holds at most one model of each family, since a family keeps its
// model in globals. Every call on models and their states runs on one thread.

// DeepSeek's options.
typedef struct {
  // When above 0, a missing expert is replaced by a resident one whose
  // selection key is at most this far below.
  float substitute;
  // A GGUF file (set) whose routed experts replace the checkpoint's, or null.
  const char *experts;
  // The tokens of approximate replay for the decoder layers, at least WINDOW;
  // 0 for none.
  int ced;
  // Whether prefill skips the decoder batches that neither the last row nor a
  // saved prefix depends on.
  int prefill_prune;
  // Whether a state keeps its compressed rows and index keys packed, as the
  // reference rounds them: the same values in about 28% of BF16's bytes.
  int packed_state_rows;
} DeepSeekOptions;

// MiniMax-H3-Image's checkpoint files.
typedef struct {
  // Safetensors file of the Qwen3-VL text and vision encoder.
  const char *text_encoder;
  // Safetensors file of the pruned INT8-ConvRot DiT.
  const char *dit;
  // Safetensors file of the FP16 image VAE.
  const char *vae;
} MinimaxH3ImageOptions;

// MiniMax-H3 Video task variant selected by the DiT checkpoint.
typedef enum {
  // T2VA and first/last-frame-to-audio-video checkpoint (FL2VA).
  VIDEO_VARIANT_FL2VA = 0,
  // Reference-to-audio-video checkpoint (REF2VA).
  VIDEO_VARIANT_REF2VA = 1
} VideoVariant;

// MiniMax-H3 Video checkpoint files and sampling defaults.
typedef struct {
  // Safetensors file of the Qwen3-VL text and vision encoder.
  const char *text_encoder;
  // Safetensors file of the pruned INT8-ConvRot FL2VA or REF2VA DiT.
  const char *dit;
  // Safetensors file of the INT8-ConvRot 3D video VAE.
  const char *vae;
  // Safetensors file of the FP32 stereo audio VAE.
  const char *audio_vae;
  // Optional safetensors file of a BF16 LoRA adapter; null for none.
  const char *lora;
  // Runtime LoRA multiplier; <= 0 uses 1.0.
  float lora_strength;
  // Default denoising steps when a request leaves steps unset; 0 uses default.
  int default_steps;
  // Video and audio flow-matching shift factors; <= 0 use 12.0 and 3.0.
  double shift_video, shift_audio;
  // DiT task variant (FL2VA or REF2VA).
  VideoVariant variant;
} MinimaxH3VideoOptions;

// Common execution settings and family-specific options.
typedef struct {
  // Capacity in tokens.
  int ctx;
  // A pipe that receives each layer's streams, or -1.
  int dump_fd;
  // A pipe that receives each decoded token's routes, predictions and router
  // logits, or -1.
  int routes_fd;
  // The drafts per pass of the model's own draft heads; 0 for none.
  int draft;
  // The context-copy drafts per pass, at most the model's verification
  // capacity.
  int lookup;
  // No progress lines on stderr.
  int quiet;

  union {
    DeepSeekOptions deepseek;
    MinimaxH3ImageOptions minimax_h3_image;
    MinimaxH3VideoOptions minimax_h3_video;
  };
} Options;

// One sequence's state, of a family's own type.
typedef struct ModelState ModelState;
// A state's rolling part at one position.
typedef struct ModelSnapshot ModelSnapshot;

// One array of a state's rows, which only grows. state_copy copies these
// arrays, and saved states hold them. Row 0 is at data, and each row takes unit
// bytes. A state of p positions holds its first (p - lag) / ratio + guard rows,
// and none while p < lag. The rows before (p - lag) / ratio do not change at
// later positions. A guard row is computed again where the state continues:
// the last row of a compressed pair, or with lag 1, a row that needs the next
// position's token.
typedef struct {
  void *data;
  uint32_t unit, ratio, guard, lag;
} StateRegion;

// One part of a snapshot's bytes, in its family's order.
typedef struct {
  void *data;
  size_t bytes;
} SnapshotPart;

// The most regions of a state, and the most parts and integers of a snapshot.
static constexpr int STATE_REGIONS_MAX = 128, SNAPSHOT_PARTS_MAX = 160,
                     SNAPSHOT_META_MAX = 8;

static inline int64_t region_stable(const StateRegion *r, int positions) {
  return positions < (int)r->lag ? 0 : (positions - (int)r->lag) / r->ratio;
}

static inline int64_t region_rows(const StateRegion *r, int positions) {
  return positions < (int)r->lag ? 0 : region_stable(r, positions) + r->guard;
}

// Copies the rows before `positions` of n regions, from src's to dst's. Both
// lists are for that many positions.
static inline void regions_copy(const StateRegion *dst, const StateRegion *src,
                                int n, int positions) {
  for (int i = 0; i < n; i++)
    memcpy(dst[i].data, src[i].data,
           (size_t)region_rows(&src[i], positions) * src[i].unit);
}

// Running stage of an image generation call.
typedef enum {
  // No image call is running.
  IMAGE_STAGE_NONE = 0,
  // The prompt encoder runs over text and reference media tokens.
  IMAGE_STAGE_PREFILL = 1,
  // The denoiser and image decoder generate pixels.
  IMAGE_STAGE_GENERATING = 2
} ImageStage;

// Running stage of a video generation call.
typedef enum {
  // No video call is running.
  VIDEO_STAGE_NONE = 0,
  // The prompt encoder runs over text and reference media tokens.
  VIDEO_STAGE_PREFILL = 1,
  // The denoiser, video/audio VAEs and MP4 muxer generate audio-video output.
  VIDEO_STAGE_GENERATING = 2
} VideoStage;

// A model's counters since it loaded, for the server's status. Only a model
// with routed experts (experts_per_row > 0) has the expert cache's.
typedef struct {
  // Rows of decoding and prompt passes.
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  // The prompt pass running: its layer (0: none) and rows.
  int prefill_layer, prefill_rows_now;
  // Images generated and denoising steps finished since the model opened.
  uint64_t generated_images, image_steps;
  // Seconds spent denoising, decoding and encoding images.
  double image_seconds;
  // Active image generation stage (ImageStage).
  int image_stage;
  // Active 1-based denoising step, or 0 outside denoising.
  int image_step;
  // Total denoising steps in the active image call, or 0 outside denoising.
  int image_steps_total;
  // Videos generated and denoising steps finished since the model opened.
  uint64_t generated_videos, video_steps;
  // Seconds spent denoising, VAE decoding and MP4 encoding videos.
  double video_seconds;
  // Active video generation stage (VideoStage).
  int video_stage;
  // Active 1-based video denoising step, or 0 outside denoising.
  int video_step;
  // Total denoising steps in the active video call, or 0 outside denoising.
  int video_steps_total;
  // The layers, and the experts a row chooses: 0 without routed experts.
  int layers, experts_per_row;
  // The layers with routed experts, and each one's experts.
  int routed_layers, experts_per_layer;
  // The weights always resident.
  uint64_t dense_bytes;
  // The least memory that the model needs to run a full prompt pass: its
  // weights, the pass's workspace and its smallest expert cache. 0 before it
  // loads.
  uint64_t required_bytes;
  // The expert cache's counts of decoded rows and of prompt rows.
  ExpertStoreCounts decode, prefill;
  ExpertStoreCounts cache;
  uint64_t cache_bytes, slot_bytes, uncached_bytes;
  int cache_slots, mapped, cap, max_slots, experts, uncached;
} ModelCounters;

// The last prefill's plan: its chunks, the rows it ran from its first row, and
// the rows its later layers ran. Also whether approximate or pruned prefill
// applied: DeepSeek's CED and dependency pruning.
typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} ModelPlan;

// Output image encoding format.
typedef enum {
  // Lossless PNG image.
  IMAGE_FORMAT_PNG = 0,
  // Lossy JPEG image.
  IMAGE_FORMAT_JPEG = 1,
  // Lossy WebP image.
  IMAGE_FORMAT_WEBP = 2
} ImageFormat;

// Output image background.
typedef enum {
  // The model's own background, which is opaque.
  IMAGE_BACKGROUND_AUTO = 0,
  // An opaque background.
  IMAGE_BACKGROUND_OPAQUE = 1,
  // A transparent background, made from a flat key-colored one. It needs PNG
  // or WebP.
  IMAGE_BACKGROUND_TRANSPARENT = 2
} ImageBackground;

// One encoded input image passed for reference-based image editing.
typedef struct {
  // Encoded image file bytes (PNG, JPEG or another format ImageIO decodes).
  const uint8_t *data;
  // Byte length of data.
  size_t len;
} ImageInput;

// Parameters of one image generation or editing pass.
typedef struct {
  // Prompt text (UTF-8, NUL-terminated).
  const char *prompt;
  // Exact requested output width and height in pixels.
  int width, height;
  // Minimum generation canvas area in pixels; 0 uses the model default.
  int min_generation_area;
  // Output image encoding format.
  ImageFormat format;
  // JPEG or WebP quality in 0 to 100; PNG ignores it.
  int quality;
  // Output background.
  ImageBackground background;
  // Deterministic RNG seed for initial latent noise.
  uint64_t seed;
  // Optional reference images; null when ref_count is 0.
  const ImageInput *refs;
  // Number of reference images (0 for text-to-image).
  int ref_count;
  // An edit's optional mask: an encoded image the size of the first reference
  // image, whose alpha marks what changes. Transparent pixels change, and
  // opaque pixels keep the first image's. data is null for no mask.
  ImageInput mask;
  // Partial images to send while the image generates, spread evenly over the
  // denoising steps; 0 sends none. A model sends at most one fewer than its
  // steps.
  int partial_images;
  // Receives each partial image: its encoded bytes, len bytes, and its
  // 0-based index. Null when partial_images is 0.
  void (*on_partial_image)(void *context, const uint8_t *data, size_t len,
                           int index);
  // The context passed to on_partial_image.
  void *partial_image_context;
} ImageRequest;

// Capacity in bytes of ImageResult.error, including its NUL terminator.
static constexpr size_t IMAGE_ERROR_BYTES = 256;

// Output of one image generation or editing pass.
typedef struct {
  // Heap-allocated PNG or JPEG bytes; null on failure.
  uint8_t *data;
  // Byte length of data.
  size_t len;
  // Error message when data is null; empty on success.
  char error[IMAGE_ERROR_BYTES];
  // Nonzero when the failure was a client input error; zero for load, memory
  // or stop failures.
  int invalid_input;
  // Encoded prompt tokens processed by the text encoder, including reference
  // image tokens.
  int prompt_tokens;
  // Seconds spent in the prompt encoder stage.
  double prefill_seconds;
  // Seconds spent denoising, decoding and encoding the output image.
  double image_seconds;
} ImageResult;

// One word aligned to an audio clip by a model's CTC timestamp head.
typedef struct {
  // Byte start (inclusive) and end (exclusive) offsets in the UTF-8 transcript.
  int byte_start, byte_end;
  // Aligned start and end times in seconds within [0, duration].
  double start, end;
} AudioWord;

// Input and output limits of an image generation model.
typedef struct {
  // Maximum prompt tokens retained before encoding.
  int max_prompt_tokens;
  // Maximum single-axis output or canvas dimension in pixels.
  int max_dim;
  // Output width and height in pixels of a text-to-image call whose size is
  // auto.
  int default_width, default_height;
  // Minimum and maximum accepted min_generation_area in pixels.
  int min_generation_area, max_generation_area;
  // Default min_generation_area in pixels when omitted.
  int default_generation_area;
  // Maximum generation canvas area in pixels.
  int64_t max_canvas_area;
  // Maximum total output pixels across all images of one request.
  int64_t max_request_pixels;
  // Maximum images returned by one request.
  int max_images_per_request;
  // Maximum reference images accepted by one edit request.
  int max_references;
  // Minimum, maximum and default JPEG and WebP quality values.
  int min_quality, max_quality, default_quality;
  // Required inference step count for distilled checkpoints.
  int inference_steps;
  // Required guidance scale for CFG-free checkpoints.
  double guidance_scale;
} ImageLimits;

// Capacity in bytes of VideoResult.error, including its NUL terminator.
static constexpr size_t VIDEO_ERROR_BYTES = 256;

// One encoded media file buffer passed to video generation (image, video or
// audio clip).
typedef struct {
  // Encoded file bytes (PNG/JPEG, MP4/MOV or WAV/AAC/MP3/M4A).
  const uint8_t *data;
  // Byte length of data.
  size_t len;
} VideoMediaInput;

// Parameters of one video generation pass (T2VA, FL2VA or REF2VA).
typedef struct {
  // Prompt text (UTF-8, NUL-terminated).
  const char *prompt;
  // Requested output width and height in pixels; 0 uses default or adapts to
  // the first input reference.
  int width, height;
  // Requested duration in whole seconds (0 uses frames or model default).
  int seconds;
  // Explicit requested frame count at 24 fps (0 uses seconds or default);
  // snapped up to the 17k + 5 grid.
  int frames;
  // Requested denoising steps (0 uses model default).
  int steps;
  // Optional video and audio flow-shift overrides (<= 0 use model defaults).
  double shift_video, shift_audio;
  // Deterministic RNG seed for initial video/audio noise and cond noise aug.
  uint64_t seed;
  // Optional FL2VA first and last keyframe images (data null when omitted).
  VideoMediaInput first_frame, last_frame;
  // Optional REF2VA reference images, videos and standalone audio clips.
  const VideoMediaInput *ref_images;
  // Number of reference images in ref_images.
  int ref_image_count;
  // Optional REF2VA reference video clips (each may carry a soundtrack).
  const VideoMediaInput *ref_videos;
  // Optional explicit soundtracks index-paired with ref_videos; null or zero
  // len falls back to the video container's own audio track.
  const VideoMediaInput *ref_video_audios;
  // Number of reference videos in ref_videos.
  int ref_video_count;
  // Optional REF2VA standalone reference audio clips.
  const VideoMediaInput *ref_audios;
  // Number of standalone audio clips in ref_audios.
  int ref_audio_count;
  // Nonzero in REF2VA to scale reference images to 2048 short edge instead of
  // matching the generation pixel area.
  int ref_image_max;
} VideoRequest;

// Output of one video generation pass.
typedef struct {
  // Heap-allocated MP4 bytes (H.264 + 32 kHz stereo AAC); null on failure.
  uint8_t *data;
  // Byte length of data.
  size_t len;
  // Actual generated video width, height, frame count and fps.
  int width, height, frames, fps;
  // Error message when data is null; empty on success.
  char error[VIDEO_ERROR_BYTES];
  // Nonzero when failure was a client input error; zero for load, memory or
  // stop failures.
  int invalid_input;
  // Encoded prompt tokens processed by the text encoder, including reference
  // media tokens.
  int prompt_tokens;
  // Seconds spent in the prompt encoder stage.
  double prefill_seconds;
  // Seconds spent denoising, VAE decoding and MP4 muxing.
  double video_seconds;
} VideoResult;

// Input and output limits of a video generation model.
typedef struct {
  // Maximum prompt tokens retained before encoding.
  int max_prompt_tokens;
  // Spatial multiple in pixels required for width and height.
  int patch_px;
  // Maximum single-axis output dimension in pixels.
  int max_dim;
  // Default output width and height in pixels.
  int default_width, default_height;
  // Maximum generation canvas area in pixels.
  int64_t max_canvas_area;
  // Output frame rate in frames per second.
  int fps;
  // Minimum, maximum and default video frame counts.
  int min_frames, max_frames, default_frames;
  // Minimum, maximum and default durations in whole seconds.
  int min_seconds, max_seconds, default_seconds;
  // Minimum, maximum and default denoising step counts.
  int min_steps, max_steps, default_steps;
  // Maximum reference images, videos, standalone audios and total references.
  int max_ref_images, max_ref_videos, max_ref_audios, max_total_refs;
  // Active checkpoint task variant (FL2VA or REF2VA).
  VideoVariant variant;
  // Required guidance scale for CFG-free checkpoints.
  double guidance_scale;
  // Default video and audio flow-matching shift factors.
  double default_shift_video, default_shift_audio;
} VideoLimits;

typedef struct {
  // The family's name, for the log.
  const char *family;
  // The config.json model_type values it runs, ending with a null.
  const char *const *model_types;
  // The model's encoded-media placeholder, or null.
  const char *media_token;
  // Maximum draft proposals per pass this family's draft head produces, or 0
  // without one.
  int max_drafts;
  // Loads the model of dir on gpu, its workspace counted in work.
  int (*load)(Gpu *gpu, MemorySlot *work, const char *dir, const Options *o);
  // Returns the slot of its dense weights, which it loads again after an
  // eviction.
  MemorySlot *(*dense_weight_slot)(void);
  // Returns whether every required weight group is resident. Null checks
  // dense_weight_slot alone.
  int (*is_loaded)(void);
  // Sets whether a request runs. While one runs, the model keeps what the
  // request may still use, such as its expert slots.
  void (*set_request_running)(int running);
  // Writes the model's counters to c.
  void (*read_counters)(ModelCounters *c);
  // Returns the bytes in the expert store. It is null for a model without one.
  uint64_t (*expert_store_bytes)(void);
  // Writes the last prefill's plan. It is null for a family that does not plan
  // its own.
  void (*read_last_plan)(ModelPlan *p);
  // These return the limits after loading. vocab_size: the vocabulary's size.
  // max_rows: the rows of different states that one decoding pass takes.
  // max_verify: the rows that one verify pass takes, the token and its drafts.
  // drafts_per_pass: the drafts per pass that the model makes itself, at most
  // what its draft head makes, and 0 without one. rewind_margin: the positions
  // a state can continue before its end.
  int (*vocab_size)(void);
  int (*max_rows)(void);
  int (*max_verify)(void);
  int (*drafts_per_pass)(void);
  int (*rewind_margin)(void);
  // Returns whether a sampled token ends a reply.
  int (*is_stop_token)(int token);

  // Returns a new state, or null when its memory does not fit.
  ModelState *(*state_new)(void);
  void (*state_free)(ModelState *s);
  // Returns the bytes of s, or of a new state for a null s, when it holds that
  // many positions.
  size_t (*state_bytes)(const ModelState *s, int positions);
  // Copies the rows of src before `positions` that later positions do not
  // change into dst. src may be in use by a running pass.
  void (*state_copy)(ModelState *dst, const ModelState *src, int positions);

  ModelSnapshot *(*snapshot_take)(ModelState *s, int pos, const float *logits);
  void (*snapshot_restore)(ModelState *s, const ModelSnapshot *x);
  void (*snapshot_free)(ModelSnapshot *x);
  size_t (*snapshot_bytes)(int with_logits);
  // Returns the snapshot's logits, or null if it has none.
  const float *(*snapshot_logits)(const ModelSnapshot *x);

  // The version of the layout of the regions and snapshot parts below. A saved
  // state of another version is not read.
  int state_version;
  // Writes s's regions for `positions` positions, at most STATE_REGIONS_MAX:
  // the arrays that state_copy copies. Returns their count. With fill, s is
  // first made to hold them: it holds `positions` positions and continues as
  // after a state_copy of that many. A snapshot restored into it, or a replay,
  // then rebuilds the rest.
  int (*write_state_regions)(ModelState *s, int positions, int fill,
                             StateRegion *out);
  // Writes the parts of snapshot x, or for a null x, of s's rolling rows as
  // they are at pos, at most SNAPSHOT_PARTS_MAX and always in one order. A
  // snapshot's logits, if it has them, come last. Writes its integers to meta,
  // at most SNAPSHOT_META_MAX, and their count to *nmeta. Returns the count of
  // parts, or -1 if s's rolling rows cannot be as at pos: a family whose state
  // knows its own position.
  int (*write_snapshot_parts)(ModelSnapshot *x, ModelState *s, int pos,
                              SnapshotPart *out, int32_t *meta, int *nmeta);
  // Returns an empty snapshot with those integers, for its parts to be read
  // into, or null if they do not suit the model.
  ModelSnapshot *(*snapshot_new_empty)(const int32_t *meta, int nmeta,
                                       int with_logits);

  // Prefills positions from to n - 1 of a prompt in s, and returns the last
  // one's logits. tokens holds all n of the prompt's tokens. With replay, the
  // state holds only the rows that state_copy copies before from, and the
  // prefill rebuilds the rest; replay_rows returns -1 for a model whose rolling
  // rows cannot be rebuilt from the rows it keeps: it resumes only from a
  // snapshot.
  // A capture of 0 or more takes a snapshot at that position into *captured.
  // A non-null every gets the logits of every position from from to n - 1.
  // A non-null wait is waited for in each layer, at a point where the prefill
  // holds no experts.
  const float *(*prefill)(ModelState *s, const int *tokens, int n, int from,
                          int replay, int capture, ModelSnapshot **captured,
                          float *every, const PassWait *wait);
  int (*replay_rows)(int k);
  int (*replay_start)(int k);
  // Runs one token for each of m states, m at most max_rows, and returns rows
  // of logits. Row r is tokens[r] at pos[r] of states[r], that state's next
  // position.
  const float *(*decode_rows)(ModelState *const *states, const int *tokens,
                              const int *pos, int m);
  // Runs a decision model's pass over one sequence of n tokens: the state's
  // first state_tokens, then the questions, question q's ending before
  // question_ends[q]. The rows whose tokens are negative are the media's: they
  // take the encoded rows of the nmedia media in turn, every row of each.
  // Returns count logits: logit i is token ids[i]'s at row rows[i]. Returns
  // null when the pass's memory does not fit. A non-null wait is waited for in
  // each layer. Null for a model that runs no decisions.
  const float *(*decide)(const int *tokens, int n, int state_tokens,
                         const int *question_ends, int questions,
                         const int *rows, const int *ids, int count,
                         const Media *media, int nmedia, const PassWait *wait);
  // Generates or edits one image from request, writing the encoded bytes or
  // error message into out. wait yields between layers to decoding calls, and
  // *cancelled requests early stop when nonzero; either may be null. Returns 1
  // on success or 0 on failure. Null for a model that generates no images.
  int (*generate_image)(const ImageRequest *request, const PassWait *wait,
                        const int *cancelled, ImageResult *out);
  // Writes the image model's request and output limits into out. Null for a
  // model that generates no images.
  void (*read_image_limits)(ImageLimits *out);
  // Writes into *out_width and *out_height the largest size of a width by
  // height image's shape that the model generates: the size itself when it
  // fits. Returns 1, or returns 0 with a message in error when no size of that
  // shape fits. Null for a model that generates no images.
  int (*fit_image_size)(int width, int height, int *out_width, int *out_height,
                        char *error, size_t error_cap);
  // Generates one audio-video MP4 clip from request, writing the MP4 bytes or
  // error message into out. wait yields between layers to decoding calls, and
  // *cancelled requests early stop when nonzero; either may be null. Returns 1
  // on success or 0 on failure. Null for a model that generates no videos.
  int (*generate_video)(const VideoRequest *request, const PassWait *wait,
                        const int *cancelled, VideoResult *out);
  // Writes the video model's request and output limits into out. Null for a
  // model that generates no videos.
  void (*read_video_limits)(VideoLimits *out);
  // Drafts. propose_drafts writes up to drafts_per_pass() drafts after token
  // at pos, once every position before it has run. It returns the sampled
  // proposals' probabilities, n vocabulary rows, or null for proposals without
  // sampling. verify runs m rows from pos, the token and then its drafts, and
  // returns each row's logits. accept keeps the first n rows.
  const double *(*propose_drafts)(ModelState *s, int token, int pos,
                                  int *drafts, int n, Sampler *sampler);
  const float *(*verify)(ModelState *s, const int *tokens, int m, int pos);
  void (*accept)(ModelState *s, const int *tokens, int n, int pos);
  // Sets the state's token history at pos from the tokens before it, for a
  // model that keeps one; otherwise it does nothing.
  void (*set_history)(ModelState *s, const int *tokens, int pos);
  // Runs about once a second while no request runs, and gives memory back to
  // other programs if they need it. Returns the model's expert slots, or 0
  // without an expert store.
  int (*run_idle_work)(void);
  // Prints the model's own lines of the command line's report. At 0, when
  // decoding starts, it notes its counters. At 1, after decoding, it prints
  // what it measured meanwhile, per token of `tokens`. May be null.
  void (*print_report)(FILE *out, int at, int tokens);
  // These prepare and encode the model's media, and set their feature spans
  // for the next prefill. Null functions mean the model takes text only.
  int (*prepare_media)(const char *dir, Media *im, const void *data, size_t len,
                       const char **error);
  // Returns 0 if the memory does not fit the encoding.
  int (*encode_media)(Media *im);
  void (*set_prefill_media)(const Media *im, int n);
  // Returns nonzero when checkpoint dir supports word-level audio timestamp
  // alignment. Null for models without an audio alignment head.
  int (*can_align_audio)(const char *dir);
  // Aligns UTF-8 transcript text against encoded audio file bytes. Writes a
  // heap-allocated array of *out_count words to *out_words and clip duration
  // in seconds to *out_duration when non-null. Returns 1 on success or 0 with
  // error populated. Null for models without an audio alignment head.
  int (*align_audio)(const void *audio_data, size_t audio_len,
                     const char *transcript, size_t transcript_len,
                     const PassWait *wait, AudioWord **out_words,
                     int *out_count, double *out_duration, char *error,
                     size_t error_cap);
  // Writes the experts that row `row` of the latest decoding pass chose, and
  // where each one was; null for a model without routed experts.
  int (*read_routes)(int row, uint16_t *id, uint8_t *from);
  // Writes one user turn of the command line, with media placeholders and the
  // chosen reasoning effort.
  void (*render_chat_turn)(char *text, size_t cap, const char *prompt,
                           int media_count, int think);
  // Runs the command line's --bench: times the kernels on the model's own
  // matrices. It may be null.
  void (*run_bench)(void);
  // pass_new returns a pass of this model that will run body(argument).
  // Inside the pass, pass_begin starts its work on buffers of its own, and
  // pass_end ends it.
  Pass *(*pass_new)(void (*body)(void *), void *argument);
  void (*pass_begin)(void);
  void (*pass_end)(void);
  // Sets how many passes besides the running one have GPU work. The engine
  // sets it before it runs passes. Null where a family has no use for it.
  void (*set_other_passes)(int passes);
  // Sets the remaining measured turn budget in seconds while sharing the GPU
  // with other models, or 0 when alone or before a peer turn is measured. Null
  // where a family has no use for it.
  void (*set_turn_budget)(double seconds);
  // Between two calls of a pass, releases what binds the states the pass used
  // to the model's variables, so that code outside the pass may use those
  // states; or null if nothing is bound.
  void (*pass_release)(void);
} ModelOps;

typedef struct {
  const ModelOps *ops;
  const char *dir;
  // The GPU that the model runs on.
  Gpu *gpu;
  MemoryAccount memory;
  MemorySlot work;
} Model;

// The family that runs the checkpoint directory dir (its config.json's
// model_type), and the model of dir, loaded on gpu and counted in an account
// of memory whose own limit is limit bytes (0: none). Both end the process
// with a message when no family runs it.
const ModelOps *model_find_family(const char *dir);
Model *model_create(Memory *memory, Gpu *gpu, const char *dir, uint64_t limit);
int model_load(Model *model, const Options *options);
void model_set_request_running(Model *model, int active);
Model *model_open(Memory *memory, Gpu *gpu, const char *dir, const Options *o);

extern const ModelOps deepseek_ops, inkling2_ops, lfm3_ops,
    minimax_h3_image_ops, minimax_h3_video_ops, nemotron_ops;

#endif
