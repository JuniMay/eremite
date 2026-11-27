#ifndef EREMITE_H
#define EREMITE_H
#include <stddef.h>
#include <stdint.h>

// eremite's inference engine behind a C ABI, for a server. The engine runs the
// models of every checkpoint that the model interface runs (DeepSeek V4.1
// Flash, and LiquidAI's d1 decision models), several of them in one process,
// at most one of each family. For each model: loading, the attention states
// and their snapshots, prefill and decoding, the model's drafts and their
// verification, sampling, context-copy drafts, the vision encoder, a decision
// model's passes, and its tokenizer; for the process: the log and the stop on
// SIGINT or SIGTERM. The server owns everything about requests: their queue
// and scheduling, the prefix cache (which states and snapshots to keep, and
// how a prompt reuses them), HTTP, and the status it builds from
// eremite_read_stats.
//
// Threads: eremite_init, eremite_load and every function on models, states,
// snapshots, prefill, decoding, drafts, samplers, context lookups and images
// run on one thread, the server's engine thread: the engine, the models and the
// GPU keep their state without locks, and one command queue carries every
// model's passes. The tokenizer's functions, eremite_log, eremite_is_stopping
// and eremite_request_stop may be called from any thread. eremite_read_stats
// runs on the engine thread between passes; the caller publishes a copy for its
// HTTP handlers and terminal.

#define EREMITE_ABI 15

// ---- the process ----

typedef struct {
  // EREMITE_ABI.
  int abi;
  // Upper limit of model memory, state memory included; 0: none (the free
  // memory decides).
  double memory_gb;
  // Memory kept available to other programs; 0: default 10 GiB.
  double reserve_gb;
  // Seconds that a wait for the GPU may last before the program ends: only a
  // failed command buffer or a stopped GPU takes that long. 0: default 60.
  double gpu_timeout_seconds;
  // How the memory follows the system's counts, and how the expert caches
  // give memory back and grow; 0 takes each default. Seconds that freed memory
  // takes to show in the system's memory counts: no reading within this time
  // of a release, a cache's steps of giving back at most once per this time,
  // and a load's waits for memory that comes back (default 1.5).
  double freed_memory_delay_seconds;
  // Seconds between two readings of the system's available memory (default
  // 1); a request of 1 GiB or more reads at once. A cache's pool grows at most
  // once per this time.
  double available_memory_reading_interval_seconds;
  // Seconds after which a reading is taken again, even right after memory was
  // freed (default 5).
  double available_memory_reading_max_age_seconds;
  // Seconds that used weights count as recently used: a growing expert cache
  // does not evict them (default 30).
  double keep_recently_used_weights_seconds;
  // Seconds after macOS swaps memory out during which an expert cache grows
  // only into free memory (default 300).
  double free_pages_only_after_swapout_seconds;
  // GiB that must stay free above the reserve for an expert cache to grow
  // (default 4).
  double expert_cache_growth_margin_gb;
  // GiB kept unused below the GPU's memory limit, for slots that the GPU
  // driver still counts after they leave its list (default 3).
  double gpu_memory_limit_margin_gb;
  // Nonzero enables the platform backend's releasable guard pool so released
  // and idle headroom above reserve cannot be absorbed by OS file cache.
  int occupy_free_memory;
  // The log's lines, each with its time, go to on_log instead of stderr when
  // it is set. It runs on the thread that logs, one line at a time under the
  // log's lock: it must return quickly and must not call eremite_log.
  void (*on_log)(const char *line);
  // Runs before a fatal error's message on stderr and the process's exit
  // (the live view restores the terminal), on the failing thread; or null.
  void (*on_die)(void);
  // Runs after each stop request, from SIGINT, SIGTERM or
  // eremite_request_stop, on a thread of its own; or null. The server wakes
  // its waiting threads there.
  void (*on_stop)(void);
} eremite_process;

// Installs the log, the stop's signal handlers and the failure hook, on the
// calling thread, which from then on is the engine's: it needs a stack of at
// least 16 MiB, and it gets the user-interactive QoS class, so that the
// models' GPU work goes before background work. 0, or -1 when called twice,
// with another ABI or with a setting it refuses.
int eremite_init(const eremite_process *p);

// Whether a stop was requested (SIGINT, SIGTERM or eremite_request_stop), and
// a request for one. Each request counts: eremite_count_stop_requests says how
// many there were (the server stops saving states at quit at the second).
int eremite_is_stopping(void);
void eremite_request_stop(void);
int eremite_count_stop_requests(void);

// A line for the log, with the time: on stderr, or to on_log.
void eremite_log(const char *line);

// ---- models ----

typedef struct {
  // GGUF file or shard; null uses checkpoint experts.
  const char *experts;
  float substitute;
  // DSpark proposals, 0 to 5.
  int draft;
  int ced;
  int prefill_prune;
  // Whether a state keeps its compressed rows and index keys packed, as the
  // reference rounds them: the same values in about 28% of BF16's bytes.
  int packed_state_rows;
} eremite_deepseek_config;

// Inkling-2-Mini options.
typedef struct {
  // Whether a state stores global, sliding-window and draft-head KV caches as
  // approximate FP8 E4M3 with per-head scales instead of BF16.
  int fp8_kv;
} eremite_inkling2_config;

// MiniMax-H3-Image checkpoint files.
typedef struct {
  // Safetensors file of the Qwen3-VL text and vision encoder.
  const char *text_encoder;
  // Safetensors file of the pruned INT8-ConvRot DiT.
  const char *dit;
  // Safetensors file of the FP16 image VAE.
  const char *vae;
} eremite_minimax_h3_config;

// MiniMax-H3 Video checkpoint files and sampling defaults.
typedef struct {
  // Safetensors file of the Qwen3-VL text and vision encoder.
  const char *text_encoder;
  // Safetensors file of the pruned INT8-ConvRot FL2VA or REF2VA DiT.
  const char *dit;
  // Safetensors file of the INT8-ConvRot 3D Video VAE.
  const char *vae;
  // Safetensors file of the FP32 stereo Audio VAE.
  const char *audio_vae;
  // Optional safetensors file of a BF16 LoRA adapter, or null.
  const char *lora;
  // LoRA scaling multiplier (<= 0 uses 1.0).
  float lora_strength;
  // Default Euler denoising steps (<= 0 uses 50).
  int default_steps;
  // Default video flow-matching shift factor (<= 0 uses 12.0).
  double shift_video;
  // Default audio flow-matching shift factor (<= 0 uses 3.0).
  double shift_audio;
  // Checkpoint task variant: EREMITE_VIDEO_VARIANT_FL2VA or
  // EREMITE_VIDEO_VARIANT_REF2VA.
  int variant;
} eremite_minimax_h3_video_config;

// Common settings and family-specific options.
typedef struct {
  int abi;
  const char *model;
  int ctx;
  int lookup;
  // Upper limit of the model's own memory: its weights, expert cache,
  // workspaces and states; 0: none of its own.
  double memory_gb;
  eremite_deepseek_config deepseek;
  eremite_inkling2_config inkling2;
  eremite_minimax_h3_config minimax_h3;
  eremite_minimax_h3_video_config minimax_h3_video;
} eremite_config;

typedef struct eremite_model eremite_model;
typedef struct eremite_media eremite_media;
typedef struct eremite_sampler eremite_sampler;

// Registers a checkpoint and opens its tokenizer, without loading weights.
// Returns null for invalid settings, or for a second model of the same
// family.
eremite_model *eremite_open(const eremite_config *c);
// Loads the registered model on the engine thread, and loads its weights again
// after an eviction. Returns whether it loaded.
int eremite_load(eremite_model *m);
// Returns whether the model is loaded and its weights are in memory.
int eremite_is_loaded(const eremite_model *m);
// Sets whether the model runs admitted requests. While it does, its weights
// stay in memory.
void eremite_set_request_running(eremite_model *m, int active);

// Runs once per available_memory_reading_interval_seconds while the model runs
// no request, and gives memory back to other programs if they need it. Returns
// the model's expert cache slots after that, or 0 without a cache.
int eremite_run_idle_work(eremite_model *m);
// Returns the seconds between two calls of eremite_run_idle_work: the
// available-memory reading interval in use after eremite_init, or 0 before it.
double eremite_idle_work_interval_seconds(void);
// Gives back every idle part of the model's memory while it runs no request:
// its expert cache before its weights. Its states stay, and the next
// eremite_load reads the weights again. Returns the bytes given back.
uint64_t eremite_unload(eremite_model *m);
// Evicts idle memory until bytes are freed or none is left. Returns the bytes
// freed.
uint64_t eremite_reclaim(uint64_t bytes);
// Maintains the platform memory policy's releasable guard pool: yields under
// pressure or below reserve, or occupies idle headroom above reserve +
// hysteresis within the limit, growing by at most max_grow_bytes (0:
// unbounded). Returns occupied bytes afterward.
uint64_t eremite_maintain_memory(uint64_t max_grow_bytes);

// Process controls run on the engine thread, between completed model calls.
// eremite_read_memory: the memory limit, the bytes counted for the models
// (their state memory included), and their state memory's bytes.
typedef struct {
  uint64_t limit, model_bytes, state_bytes;
} eremite_memory_info;

eremite_memory_info eremite_read_memory(void);
// Sets the memory limit in bytes.
void eremite_set_memory_limit(uint64_t bytes);
// Waits for the GPU's work and the expert reads, at a boundary of the engine
// thread.
void eremite_drain(void);

// The model's limits: rows of different states that eremite_decode_rows takes
// at once, the drafts per pass it makes itself (the selected model options'
// draft field, at most its draft head's), the most rows one verify pass takes,
// the positions a state can continue before its end (the rewind margin), and
// the vocabulary. And whether a token ends a reply. A decision model's limits
// are 0, but its vocabulary, and no token ends a reply.
int eremite_max_rows(const eremite_model *m);
int eremite_drafts_per_pass(const eremite_model *m);
int eremite_max_verify(const eremite_model *m);
int eremite_rewind_margin(const eremite_model *m);
int eremite_vocab_size(const eremite_model *m);
int eremite_is_stop_token(const eremite_model *m, int32_t token);

// ---- decision models ----

// A decision model, such as LiquidAI's d1, reads one sequence in one pass: a
// state, then questions after it. It answers each question with the logits of
// a few tokens at the question's last row, and keeps nothing between passes.
// It runs no states, snapshots, prefills, decoding or drafts: those calls end
// the program. Its eremite_config takes no drafts, lookup or DeepSeek options.

// Returns whether the model is a decision model.
int eremite_is_decision_model(const eremite_model *m);
// Runs a decision model's pass over n tokens: the first state_tokens are the
// state, and question q's tokens end before question_ends[q], the last
// question's at n. Its images, or null for none, are encoded
// (eremite_media_encode), and each one's span (eremite_media_expand) holds
// its rows. Returns count logits, valid until the next call that runs the
// model: logit i is token ids[i]'s at row rows[i]. Returns null when the
// pass's memory does not fit.
const float *eremite_decide(eremite_model *m, const int32_t *tokens, int n,
                            int state_tokens, const int32_t *question_ends,
                            int questions, const int32_t *rows,
                            const int32_t *ids, int count,
                            const eremite_media *images);

// ---- image models ----

// An image model, such as MiniMax-H3-Image, generates or edits one image per
// call from a text prompt and optional reference images, and keeps no
// attention states between calls. It runs no states, snapshots, prefills,
// decoding, drafts or decisions: those calls end the program. Its
// eremite_config takes minimax_h3 checkpoint paths and no drafts, lookup or
// DeepSeek options.

// Running stage of an image generation call.
enum {
  // No image call is running.
  EREMITE_IMAGE_STAGE_NONE = 0,
  // The prompt encoder runs over text and reference media tokens.
  EREMITE_IMAGE_STAGE_PREFILL = 1,
  // The denoiser and image decoder generate pixels.
  EREMITE_IMAGE_STAGE_GENERATING = 2
};

// Output image encoding format.
enum {
  // Lossless PNG encoding.
  EREMITE_IMAGE_PNG = 0,
  // Lossy JPEG encoding.
  EREMITE_IMAGE_JPEG = 1,
  // Lossy WebP encoding.
  EREMITE_IMAGE_WEBP = 2
};

// Output image background.
enum {
  // The model's own background, which is opaque.
  EREMITE_IMAGE_BACKGROUND_AUTO = 0,
  // An opaque background.
  EREMITE_IMAGE_BACKGROUND_OPAQUE = 1,
  // A transparent background, made from a flat key-colored one. It needs
  // EREMITE_IMAGE_PNG or EREMITE_IMAGE_WEBP.
  EREMITE_IMAGE_BACKGROUND_TRANSPARENT = 2
};

// Capacity in bytes of eremite_image_result.error, including its NUL
// terminator.
enum { EREMITE_IMAGE_ERROR_BYTES = 256 };

// One encoded input image passed for image editing.
typedef struct {
  // Encoded image bytes (PNG, JPEG or another format macOS decodes).
  const uint8_t *data;
  // Byte length of data.
  size_t len;
} eremite_image_input;

// Receives one partial image of an image call: the encoded bytes of an early
// estimate of the image, len bytes, and its 0-based index among the call's
// partial images. It runs inside the call, on the thread that runs the call:
// it copies what it keeps and returns quickly.
typedef void (*eremite_partial_image_fn)(void *context, const uint8_t *data,
                                         size_t len, int index);

// Parameters of one image generation or editing call.
typedef struct {
  // Prompt text (UTF-8, NUL-terminated).
  const char *prompt;
  // Exact requested output width and height in pixels.
  int width, height;
  // Minimum generation canvas area in pixels (1024 to 16777216; 0 uses
  // 4194304).
  int min_generation_area;
  // Output format: EREMITE_IMAGE_PNG, EREMITE_IMAGE_JPEG or EREMITE_IMAGE_WEBP.
  int format;
  // JPEG or WebP quality in 0 to 100; PNG ignores it.
  int quality;
  // Output background: one of EREMITE_IMAGE_BACKGROUND_*.
  int background;
  // Deterministic RNG seed for initial latent noise.
  uint64_t seed;
  // Optional reference images; null when ref_count is 0.
  const eremite_image_input *refs;
  // Number of reference images (0 for text-to-image, 1 to 16 for editing).
  int ref_count;
  // An edit's optional mask: an encoded image the size of the first reference
  // image, whose alpha marks what changes. Transparent pixels change, and
  // opaque pixels keep the first image's. data is null for no mask.
  eremite_image_input mask;
  // Partial images to send while the image generates, spread evenly over the
  // denoising steps, each encoded like the image; 0 sends none. A model sends
  // at most one fewer than its steps.
  int partial_images;
  // Receives each partial image; null when partial_images is 0.
  eremite_partial_image_fn on_partial_image;
  // The context passed to on_partial_image.
  void *partial_image_context;
} eremite_image_request;

// Output of one image generation or editing call.
typedef struct {
  // Heap-allocated PNG or JPEG bytes; null on failure.
  uint8_t *data;
  // Byte length of data.
  size_t len;
  // Error message when data is null; empty on success.
  char error[EREMITE_IMAGE_ERROR_BYTES];
  // Nonzero when the request inputs were invalid; zero for load, memory or
  // stop failures.
  int invalid_input;
  // Encoded prompt tokens processed by the text encoder, including reference
  // image tokens.
  int prompt_tokens;
  // Seconds spent in the prompt encoder stage.
  double prefill_seconds;
  // Seconds spent denoising, decoding and encoding the output image.
  double image_seconds;
} eremite_image_result;

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
} eremite_image_limits;

// Returns whether the model is an image generation model.
int eremite_is_image_model(const eremite_model *m);
// Writes the image limits of m into out and returns 1, or returns 0 when m is
// null, out is null, or m generates no images.
int eremite_image_model_limits(const eremite_model *m,
                               eremite_image_limits *out);
// Reads the width and height of the first image of the encoded bytes data, of
// len bytes, without decoding its pixels. Returns 1, or returns 0 with the
// reason in error.
int eremite_image_read_size(const uint8_t *data, size_t len, int *width,
                            int *height, char *error, size_t error_cap);
// Writes into *out_width and *out_height the largest size of a width by height
// image's shape that m generates: the size itself when it fits. Returns 1, or
// returns 0 with the reason in error when no size of that shape fits.
int eremite_image_fit_size(const eremite_model *m, int width, int height,
                           int *out_width, int *out_height, char *error,
                           size_t error_cap);
// Runs one image generation or editing call synchronously and writes the
// result into out. Caller releases out->data with eremite_image_result_free.
// Returns 1 on success or 0 on failure.
int eremite_generate_image(eremite_model *m,
                           const eremite_image_request *request,
                           eremite_image_result *out);
// Frees the encoded image buffer held by result and clears its fields.
void eremite_image_result_free(eremite_image_result *result);

// ---- video models ----

// A video model, such as MiniMax-H3 Video, generates one audio-video MP4 clip
// per call from a text prompt and optional keyframe or reference media, and
// keeps no attention states between calls. It runs no states, snapshots,
// prefills, decoding, drafts, decisions or image calls: those calls end the
// program.

// Running stage of a video generation call.
enum {
  // No video call is running.
  EREMITE_VIDEO_STAGE_NONE = 0,
  // The prompt encoder runs over text and reference media tokens.
  EREMITE_VIDEO_STAGE_PREFILL = 1,
  // Conditioning VAEs, DiT denoiser, Video/Audio VAEs and MP4 muxer run.
  EREMITE_VIDEO_STAGE_GENERATING = 2
};

// Task variant of a MiniMax-H3 Video checkpoint.
enum {
  // First/last keyframe-to-video-and-audio (and text-to-video-and-audio).
  EREMITE_VIDEO_VARIANT_FL2VA = 0,
  // Reference image, video and audio conditioning (and
  // text-to-video-and-audio).
  EREMITE_VIDEO_VARIANT_REF2VA = 1
};

// Capacity in bytes of eremite_video_result.error, including its NUL
// terminator.
enum { EREMITE_VIDEO_ERROR_BYTES = 256 };

// One encoded conditioning image, video or audio buffer passed to a video call.
typedef struct {
  // Encoded media bytes (PNG/JPEG/WebP, MP4/MOV or WAV/M4A/MP3).
  const uint8_t *data;
  // Byte length of data.
  size_t len;
} eremite_video_media_input;

// Parameters of one audio-video generation call.
typedef struct {
  // Prompt text (UTF-8, NUL-terminated).
  const char *prompt;
  // Requested output width in pixels (0 uses adapted or default width).
  int width;
  // Requested output height in pixels (0 uses adapted or default height).
  int height;
  // Requested duration in whole seconds (0 uses frames or default duration).
  int seconds;
  // Explicit requested frame count (0 uses seconds or default duration).
  int frames;
  // Requested Euler denoising steps (0 uses the model's configured default).
  int steps;
  // Video flow-matching shift override (<= 0 uses model default).
  double shift_video;
  // Audio flow-matching shift override (<= 0 uses model default).
  double shift_audio;
  // Deterministic RNG seed for initial video/audio noise and cond noise aug.
  uint64_t seed;
  // Optional FL2VA first keyframe image (data null when omitted).
  eremite_video_media_input first_frame;
  // Optional FL2VA last keyframe image (data null when omitted).
  eremite_video_media_input last_frame;
  // Optional REF2VA reference images; null when ref_image_count is 0.
  const eremite_video_media_input *ref_images;
  // Number of reference images in ref_images.
  int ref_image_count;
  // Optional REF2VA reference video clips; null when ref_video_count is 0.
  const eremite_video_media_input *ref_videos;
  // Optional explicit soundtracks index-paired with ref_videos; null or zero
  // len falls back to the video container's own audio track.
  const eremite_video_media_input *ref_video_audios;
  // Number of reference videos in ref_videos.
  int ref_video_count;
  // Optional REF2VA standalone reference audio clips; null when
  // ref_audio_count is 0.
  const eremite_video_media_input *ref_audios;
  // Number of standalone audio clips in ref_audios.
  int ref_audio_count;
  // Nonzero in REF2VA to scale reference images to 2048 short edge instead of
  // matching the generation pixel area.
  int ref_image_max;
} eremite_video_request;

// Output of one audio-video generation call.
typedef struct {
  // Heap-allocated MP4 bytes (H.264 + 32 kHz stereo AAC); null on failure.
  uint8_t *data;
  // Byte length of data.
  size_t len;
  // Actual generated video width in pixels.
  int width;
  // Actual generated video height in pixels.
  int height;
  // Actual generated video frame count.
  int frames;
  // Actual generated video frame rate in frames per second.
  int fps;
  // Error message when data is null; empty on success.
  char error[EREMITE_VIDEO_ERROR_BYTES];
  // Nonzero when the request inputs were invalid; zero for load, memory or
  // stop failures.
  int invalid_input;
  // Encoded prompt tokens processed by the text encoder, including reference
  // media tokens.
  int prompt_tokens;
  // Seconds spent in the prompt encoder stage.
  double prefill_seconds;
  // Seconds spent denoising, VAE decoding and MP4 muxing.
  double video_seconds;
} eremite_video_result;

// Input and output limits of a video generation model.
typedef struct {
  // Maximum prompt tokens retained before encoding.
  int max_prompt_tokens;
  // Spatial multiple in pixels required for width and height.
  int patch_px;
  // Maximum single-axis output dimension in pixels.
  int max_dim;
  // Default output width in pixels.
  int default_width;
  // Default output height in pixels.
  int default_height;
  // Maximum generation canvas area in pixels.
  int64_t max_canvas_area;
  // Output frame rate in frames per second.
  int fps;
  // Minimum video frame count.
  int min_frames;
  // Maximum video frame count.
  int max_frames;
  // Default video frame count.
  int default_frames;
  // Minimum duration in whole seconds.
  int min_seconds;
  // Maximum duration in whole seconds.
  int max_seconds;
  // Default duration in whole seconds.
  int default_seconds;
  // Minimum denoising step count.
  int min_steps;
  // Maximum denoising step count.
  int max_steps;
  // Default denoising step count configured on the model.
  int default_steps;
  // Maximum reference images accepted by one REF2VA request.
  int max_ref_images;
  // Maximum reference videos accepted by one REF2VA request.
  int max_ref_videos;
  // Maximum standalone reference audios accepted by one REF2VA request.
  int max_ref_audios;
  // Maximum total reference files accepted by one REF2VA request.
  int max_total_refs;
  // Active checkpoint task variant (EREMITE_VIDEO_VARIANT_*).
  int variant;
  // Required guidance scale for CFG-free checkpoints.
  double guidance_scale;
  // Default video flow-matching shift factor configured on the model.
  double default_shift_video;
  // Default audio flow-matching shift factor configured on the model.
  double default_shift_audio;
} eremite_video_limits;

// Returns whether the model is a video generation model.
int eremite_is_video_model(const eremite_model *m);
// Writes the video limits of m into out and returns 1, or returns 0 when m is
// null, out is null, or m generates no videos.
int eremite_video_model_limits(const eremite_model *m,
                               eremite_video_limits *out);
// Runs one audio-video generation call synchronously and writes the result
// into out. Caller releases out->data with eremite_video_result_free. Returns
// 1 on success or 0 on failure.
int eremite_generate_video(eremite_model *m,
                           const eremite_video_request *request,
                           eremite_video_result *out);
// Frees the encoded MP4 buffer held by result and clears its fields.
void eremite_video_result_free(eremite_video_result *result);

// ---- attention states and snapshots ----

// One sequence's state, of one model.
typedef struct ModelState eremite_state;
// Returns a new, empty state, or null when its memory does not fit.
eremite_state *eremite_state_new(eremite_model *m);
void eremite_state_free(eremite_model *m, eremite_state *s);
// Returns the bytes that s, or a new state for a null s, takes when it holds
// that many positions.
size_t eremite_state_bytes(const eremite_model *m, const eremite_state *s,
                           int positions);
// Copies the rows of src before positions that later positions do not change
// into dst. src may be a running request's state.
void eremite_state_copy(eremite_model *m, eremite_state *dst,
                        const eremite_state *src, int positions);

// A state's rolling part at one position: the rows that later positions
// overwrite. A state continues at the position of a snapshot restored into it,
// or up to eremite_rewind_margin() positions before it. The rows before that
// position must first be copied into it from the snapshot's own state. For
// that, the state's rings hold eremite_rewind_margin() positions' rows before
// each window.
//
// A row written past where the state then continues takes the place of one
// of those rows, the oldest. Such stale rows are a verify pass's rows after
// the tokens that eremite_accept keeps, and a state's rows after the end of a
// prompt that continued it before them. Each row written at the state's end
// replaces one again. A state, and a snapshot of it, then continues at most
// eremite_rewind_margin() less the stale rows before its position. The caller
// counts the stale rows.
typedef struct ModelSnapshot eremite_snapshot;
// Returns a snapshot of s's rolling rows now, as they are at pos. If logits
// are given, the snapshot also holds them: the logits that pos's token is
// sampled from, as a prefill or decoding call returned them.
eremite_snapshot *eremite_snapshot_take(eremite_model *m, eremite_state *s,
                                        int pos, const float *logits);
void eremite_snapshot_restore(eremite_model *m, eremite_state *s,
                              const eremite_snapshot *x);
void eremite_snapshot_free(eremite_model *m, eremite_snapshot *x);
size_t eremite_snapshot_bytes(const eremite_model *m, int with_logits);
// Returns the snapshot's logits, or null if it has none.
const float *eremite_snapshot_logits(const eremite_model *m,
                                     const eremite_snapshot *x);

// ---- saved states ----

// The server names, opens, syncs and renames the files. These calls write
// and read their contents, past the page cache (F_NOCACHE), with a CRC32C for
// each 4 MiB block. A saved state is a list of parts.
//
// eremite_state_write writes one part: the rows of s that positions from
// `from` on wrote, up to `to`. s holds at least `to` positions. The rows that
// the first `from` positions do not change are in earlier parts. Returns the
// bytes written, or -1 if a write failed or a second stop request ended it.
// With fd -1, it writes nothing and returns the bytes it would take.
// eremite_snapshot_write does the same with fd -1.
int64_t eremite_state_write(eremite_model *m, eremite_state *s, int from,
                            int to, int fd);
// Fills s, a new state, with the rows before `positions` from a saved state's
// parts, given in the order they were written. Each part's rows replace the
// earlier parts' from its own start. s then holds what eremite_state_copy of
// that many positions would give, and a snapshot is restored into it next, or
// a prefill replays. Returns 0, or -1 if a part is short, damaged or of
// another shape, or if the parts leave rows out. After -1, s is of no use.
int eremite_state_read(eremite_model *m, eremite_state *s, int positions,
                       const int *fds, int count);
// Writes snapshot x without its logits, or for a null x, s's rolling rows as
// they are at pos, its end. Returns the bytes written, or -1.
int64_t eremite_snapshot_write(eremite_model *m, const eremite_snapshot *x,
                               eremite_state *s, int pos, int fd);
// Returns the snapshot that a file holds, or null if the file is short,
// damaged or of another shape.
eremite_snapshot *eremite_snapshot_read(eremite_model *m, int fd);
// Writes the identity of m's saved states: what they depend on. It is known
// before the weights load. It covers the family's state layout, the
// checkpoint's weights, the experts' GGUF set, and the options that change a
// state's rows or shapes: draft, lookup, ced, prefill_prune, substitute,
// packed_state_rows and fp8_kv.
// The weights are the folder's config.json, .safetensors and .gguf files, by
// name, size, and first and last 64 KiB. A state saved with another identity
// is not read.
void eremite_state_identity(eremite_model *m, uint8_t out[32]);

// ---- running a model ----

typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} eremite_prefill_info;

// Prefills positions from to n - 1 of a prompt in s, and returns the last
// one's logits: vocabulary floats, valid until the next call that runs a
// model. tokens holds all n of the prompt's tokens.
//
// Without replay, the state holds every position before from: its own, or
// copied rows and a snapshot restored at from. With replay, it holds only the
// rows that eremite_state_copy copies before from, and the prefill rebuilds
// the rest: it computes earlier positions again. eremite_replay_rows returns -1
// for a model whose rolling rows cannot be rebuilt from the rows it keeps. Such
// a model resumes only from a snapshot, and its prefill computes the positions
// after the snapshot again. For DeepSeek with --ced off, the replay is exact:
// the prefill starts earlier, each layer where its window needs. With --ced,
// the replay is bounded and approximate, as DeepInfer's bounded replay of the
// encoder's sliding window.
//
// With a capture of 0 or more and a non-null captured, *captured gets the
// rolling rows at that position as a snapshot, taken during the prefill. The
// position is a replay's branch point, from, or a position after from and
// before n. *captured gets null if the model cannot take a whole state's
// snapshot there, as for DeepSeek under CED compaction. A non-null info gets
// the prefill's plan.
//
// The server gives valid token IDs, positions and row counts, within the
// limits above. Prompt IDs may include the negative image keys from
// eremite_media_expand. The calls that return logits return null when they
// are cancelled, or when the model's workspace does not fit in the memory.
const float *eremite_prefill(eremite_model *m, eremite_state *s,
                             const int32_t *tokens, int n, int from, int replay,
                             int capture, eremite_snapshot **captured,
                             eremite_prefill_info *info,
                             const eremite_media *images);
// What a replay at k costs beyond the prompt's new rows. eremite_replay_rows
// returns the rows before k that it computes again, per layer on average, or
// -1 for a model that resumes only from snapshots. eremite_replay_start
// returns the first position that it computes again: the prompt's images from
// there on must be encoded (eremite_media_encode).
int eremite_replay_rows(const eremite_model *m, int k);
int eremite_replay_start(const eremite_model *m, int k);
// Runs one token for each of count states at once, count at most
// eremite_max_rows(), without drafts. Row r is tokens[r] at pos[r], states[r]'s
// next position. Returns rows of logits.
const float *eremite_decode_rows(eremite_model *m, eremite_state **states,
                                 const int32_t *tokens, const int32_t *pos,
                                 int count);
// The model's drafts. eremite_propose_drafts writes n drafts after token at pos
// of s, n at most eremite_drafts_per_pass(). It returns the sampled proposals'
// probabilities, or null for proposals without sampling. eremite_verify runs
// count rows from pos, the token and then its drafts, and returns rows of
// logits. eremite_accept keeps the first n of the rows.
const double *eremite_propose_drafts(eremite_model *m, eremite_state *s,
                                     int32_t token, int pos, int32_t *drafts,
                                     int n, eremite_sampler *sampler);
const float *eremite_verify(eremite_model *m, eremite_state *s,
                            const int32_t *rows, int count, int pos);
void eremite_accept(eremite_model *m, eremite_state *s, const int32_t *rows,
                    int n, int pos);
// Sets s's token history at pos from the tokens before it, which DeepSeek's
// Engram uses.
void eremite_set_history(eremite_model *m, eremite_state *s,
                         const int32_t *tokens, int pos);

// ---- streams ----

// A stream runs one model's calls, one after another, on its own stack and
// with its own working buffers. The calls of several streams, of any models,
// run at the same time: whichever stream can continue runs while the others
// wait for the GPU or the SSD. Posting a call returns at once, and
// eremite_run_streams advances the streams. A state takes part in one call at a
// time; other states may be used between the calls of a stream. Pointers given
// to a call stay valid until it finishes.
typedef struct eremite_stream eremite_stream;
eremite_stream *eremite_stream_new(eremite_model *m);
// Frees a stream that has no call posted.
void eremite_stream_free(eremite_stream *x);
// Post the calls of eremite_prefill and eremite_decode_rows. Their results come
// from the functions below.
void eremite_stream_prefill(eremite_stream *x, eremite_state *s,
                            const int32_t *tokens, int n, int from, int replay,
                            int capture, const eremite_media *images);
void eremite_stream_decode_rows(eremite_stream *x, eremite_state **states,
                                const int32_t *tokens, const int32_t *pos,
                                int count);
// Posts a token and its drafts, decoded together. rows[0] is the token at pos
// of s. With proposed 0, the model's drafts fill rows[1] to rows[drafts];
// otherwise they are already there. The verified rows are then sampled, and
// the kept ones accepted.
void eremite_stream_verify(eremite_stream *x, eremite_state *s, int32_t *rows,
                           int drafts, int proposed, int pos,
                           eremite_sampler *sampler);
// Posts a decision model's pass (eremite_decide). Its logits come from
// eremite_stream_logits.
void eremite_stream_decide(eremite_stream *x, const int32_t *tokens, int n,
                           int state_tokens, const int32_t *question_ends,
                           int questions, const int32_t *rows,
                           const int32_t *ids, int count,
                           const eremite_media *images);
// Posts an image model's generation or editing call (eremite_generate_image).
// Pointers in request stay valid until the call finishes. Its result comes
// from eremite_stream_image_result.
void eremite_stream_generate_image(eremite_stream *x,
                                   const eremite_image_request *request);
// Posts a video model's generation call (eremite_generate_video). Pointers in
// request stay valid until the call finishes. Its result comes from
// eremite_stream_video_result.
void eremite_stream_generate_video(eremite_stream *x,
                                   const eremite_video_request *request);
// Requests cooperative cancellation of a running stream call at its next
// layer boundary.
void eremite_stream_cancel(eremite_stream *x);
// Advances the streams until a call finishes, and returns that stream's index.
// It returns -1 when no stream has a call posted, and when eremite_wake asks it
// to. Only this function runs the streams. A prefill lets decoding calls
// (decodes and verifies) go first: in each layer, it waits until those posted
// to the streams before its wait have finished. A decision takes turns with
// them: it lets those posted before its wait go first, then runs for as long
// as they took, while decoding calls that have not started wait. A run whose
// calls can only wait for each other ends the program.
int eremite_run_streams(eremite_stream *const *streams, int n);
// Makes eremite_run_streams return -1 soon: at once if it waits, or else after
// the call that runs now waits. If its only call waits for the GPU, it returns
// after that wait. A wake that comes outside eremite_run_streams ends its next
// run after one look at the streams. Any thread may call it.
void eremite_wake(void);
// A finished call's results. The logits stay valid until the stream's next
// call. Null logits: the call did not fit in memory.
const float *eremite_stream_logits(const eremite_stream *x);
eremite_snapshot *eremite_stream_captured_snapshot(const eremite_stream *x);
void eremite_stream_prefill_info(const eremite_stream *x,
                                 eremite_prefill_info *info);
// For a verify call: returns the next token, and sets *accepted to the drafts
// kept.
int32_t eremite_stream_next_token(const eremite_stream *x, int *accepted);
// Moves the finished image call's result into out. Caller releases out->data
// with eremite_image_result_free. Returns 1 when out->data is non-null.
int eremite_stream_image_result(eremite_stream *x, eremite_image_result *out);
// Moves the finished video call's result into out. Caller releases out->data
// with eremite_video_result_free. Returns 1 when out->data is non-null.
int eremite_stream_video_result(eremite_stream *x, eremite_video_result *out);

// ---- sampling ----

// Returns a sampler, or null for invalid settings. temperature must be finite
// and not negative, and 0 is greedy. top_p must be in [0, 1]. top_k must not
// be negative, and 0 takes every token. Without seed_set, the seed is random.
eremite_sampler *eremite_sampler_new(double temperature, double top_p,
                                     int top_k, uint64_t seed, int seed_set);
void eremite_sampler_free(eremite_sampler *s);
// Returns a token sampled from the logits.
int32_t eremite_sample(const eremite_model *m, const float *logits,
                       eremite_sampler *s);
// Samples the rows of logits of the token and its n drafts. Returns the next
// token, and sets *accepted to how many drafts it kept. A token that ends a
// reply ends the kept drafts. q is eremite_propose_drafts's result, or null for
// context-copy proposals, which are not sampled.
int32_t eremite_sample_verify_drafts(const eremite_model *m,
                                     const float *logits, const int32_t *drafts,
                                     int n, const double *q, eremite_sampler *s,
                                     int *accepted);

// ---- context-copy drafts (lookup) ----

typedef struct ContextLookup eremite_lookup;
// Returns a lookup for capacity tokens, or null if it cannot hold that many.
eremite_lookup *eremite_lookup_new(int capacity);
void eremite_lookup_free(eremite_lookup *x);
// Appends the tokens. Returns 0 if the lookup is full.
int eremite_lookup_append(eremite_lookup *x, const int32_t *tokens, int n);
// Writes up to max drafts copied from the context. Returns how many.
int eremite_lookup_propose(const eremite_lookup *x, int max, int32_t *drafts);
// Records how many of the offered drafts were accepted.
void eremite_lookup_record_verification(eremite_lookup *x, int offered,
                                        int accepted);

// ---- how many draft-model drafts a pass runs ----
typedef struct DraftChoice eremite_draft_choice;
eremite_draft_choice *eremite_draft_choice_new(void);
void eremite_draft_choice_free(eremite_draft_choice *x);
// Returns how many drafts the next pass runs, 0 to most.
int eremite_draft_choice_next_count(eremite_draft_choice *x, int most);
// A pass with `count` drafts kept the first `kept` and took `seconds`, with no
// other call beside it.
void eremite_draft_choice_record_pass(eremite_draft_choice *x, int count,
                                      int kept, double seconds);

// ---- encoded media ----

// Opens a prompt's media inputs: each placeholder becomes its feature span.
// Returns null, with a message, if an input cannot be decoded.
eremite_media *eremite_media_open(eremite_model *m, const uint8_t *const *data,
                                  const size_t *len, int n, char *message,
                                  size_t cap);
// Writes the prompt with the spans in place to out, at most max tokens.
// Returns its length, or -1 if its places and images differ or it does not
// fit.
int eremite_media_expand(eremite_media *x, const int32_t *in, int n,
                         int32_t *out, int max);
// Encodes the inputs at positions from `from` on, and gives all of them to the
// model's next prefill. Returns the count encoded.
int eremite_media_encode(eremite_media *x, int from);
// Closes the inputs, after that prefill.
void eremite_media_close(eremite_media *x);

// One word aligned to an audio clip by a model's CTC timestamp head.
typedef struct {
  // Byte start (inclusive) and end (exclusive) offsets in the UTF-8 transcript.
  int byte_start, byte_end;
  // Aligned start and end times in seconds within [0, duration].
  double start, end;
} eremite_audio_word;

// Returns nonzero when model m supports word-level audio timestamp alignment.
int eremite_can_align_audio(const eremite_model *m);
// Aligns UTF-8 transcript[0..transcript_len - 1] against encoded audio bytes
// audio_data[0..audio_len - 1]. Writes a heap-allocated array of *out_count
// words to *out_words (caller frees with eremite_audio_words_free) and audio
// duration in seconds to *out_duration when non-null. Returns 1 on success or
// 0 with an error message in error.
int eremite_align_audio(eremite_model *m, const uint8_t *audio_data,
                        size_t audio_len, const char *transcript,
                        size_t transcript_len, eremite_audio_word **out_words,
                        int *out_count, double *out_duration, char *error,
                        size_t error_cap);
// Releases a word array returned by eremite_align_audio.
void eremite_audio_words_free(eremite_audio_word *words);

// ---- statistics, for the server's status ----

// An expert cache's counts, as ExpertStoreCounts holds them.
typedef struct {
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions, late,
      cancelled, substituted, remaps, given_back;
  // Decoded rows' selections by where the expert was.
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;
  // Seconds spent waiting for reads.
  double read_s;
} eremite_cache_counts;

// A model's counters (its expert cache's: zero without one) and the
// process's memory.
typedef struct {
  // All of the cache's; of decoded rows; of prompt rows.
  eremite_cache_counts cache, decode, prefill;
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  // The prefill running: its layer (0: none) and rows.
  int prefill_layer, prefill_rows_now;
  // Images generated and denoising steps finished since the model opened.
  uint64_t generated_images, image_steps;
  // Seconds spent denoising, decoding and encoding images.
  double image_seconds;
  // Active image generation stage (EREMITE_IMAGE_STAGE_*).
  int image_stage;
  // Active 1-based denoising step, or 0 outside denoising.
  int image_step;
  // Total denoising steps in the active image call, or 0 outside denoising.
  int image_steps_total;
  // Videos generated since the model opened.
  uint64_t generated_videos;
  // Video denoising steps finished since the model opened.
  uint64_t video_steps;
  // Seconds spent denoising, VAE decoding and muxing videos.
  double video_seconds;
  // Active video generation stage (EREMITE_VIDEO_STAGE_*).
  int video_stage;
  // Active 1-based video denoising step, or 0 outside denoising.
  int video_step;
  // Total denoising steps in the active video call, or 0 outside denoising.
  int video_steps_total;
  // The expert cache's slots, and experts on the SSD only.
  int slots, mapped, cap, max_slots, experts, uncached;
  uint64_t slot_bytes, uncached_bytes, cache_bytes, dense_bytes;
  // This process's footprint, the GPU driver's memory, and the bytes
  // available to this process: the free pages, and the file cache above the
  // file cache that the kernel keeps.
  uint64_t footprint, gpu_in_use, available;
  // Metal's working-set recommendation; INT64_MAX before GPU initialization.
  int64_t gpu_limit;
  // The system's memory. file_kept is the pre-existing file cache respected
  // until drained or purged (the smallest file cache read). memory_occupied is
  // the bytes currently held in the platform policy's releasable guard pool.
  uint64_t total, free, file, file_kept, memory_occupied, wired, compressor,
      compressed, swap_used, swapouts;
  // 1 normal, 2 warn, 4 critical.
  int pressure;
  // The layers, and the experts a row chooses: 0 without routed experts.
  int layers, experts_per_row;
  // The layers with routed experts, with experts_per_row / routed_layers
  // chosen in each, and each one's experts.
  int routed_layers, experts_per_layer;
  // The most model memory now within --memory and the free memory above the
  // reserve, what the models hold (state memory included), what is counted,
  // and the most ever counted.
  uint64_t model_memory_limit, model_memory_used, model_memory_committed,
      model_memory_peak;
  uint64_t state_allocated, model_workspace;
  // The model's own memory: its own limit (0: none), the bytes counted for it
  // and for its states, and the least that it needs to run a full prompt
  // pass: its weights, the pass's workspace and its smallest expert cache (0
  // before it loads).
  uint64_t own_memory_limit, own_memory_counted, own_state_bytes,
      required_bytes;
  // Process-wide since the start, to show where the time goes:
  // - thread_s: the engine thread's seconds outside eremite_run_streams,
  // running a
  //   pass, and in eremite_run_streams with no pass ready while one waits for
  //   the GPU, or else for reads, or else for something else.
  // - gpu_busy_s and gpu_idle_s: the GPU main queue's busy seconds, and its
  //   idle seconds between command buffers in each of those five states of
  //   the engine thread.
  // - ssd_busy_s: the seconds with an expert read in flight. demand_read_s and
  //   background_read_s: the summed seconds of demand and background reads.
  double thread_s[5], gpu_busy_s, gpu_idle_s[5], ssd_busy_s, demand_read_s,
      background_read_s;
} eremite_stats;

// Reads the process's memory once, for every model's statistics.
void eremite_refresh_stats(void);
void eremite_read_stats(const eremite_model *m, eremite_stats *s);
// Writes the experts that row `row` of the model's latest decoding pass chose
// in each layer with routed experts, experts_per_row / routed_layers per
// layer, and where each one was: 0 in memory, 1 read ahead, 2 late, 3
// substituted, 4 from the SSD. Returns 1. Returns 0, and writes nothing, for a
// model without routed experts or a row that did not run every layer. It runs
// on the engine thread, after the pass.
int eremite_read_routes(const eremite_model *m, int row, uint16_t *experts,
                        uint8_t *sources);

// ---- tokenizers (tokenizer.json), from any thread ----

// A checkpoint's tokenizer: a model's own (eremite_model_tokenizer), or one
// opened from a checkpoint directory without its model. `image` is the
// spelling of the model's image token, or null. A tokenizer.json that the
// engine cannot run ends the process with a message.
typedef struct Tokenizer eremite_tokenizer;
eremite_tokenizer *eremite_model_tokenizer(const eremite_model *m);
eremite_tokenizer *eremite_tokenizer_open(const char *dir, const char *image);
// Writes the token ids of len bytes of text, at most max. Mode 0 takes the text
// alone. Mode 1 also takes the added tokens' spellings as those tokens. Mode 2
// takes a rendered prompt: as mode 1, with the byte 0xFF for an image's place.
// Returns the count, or -1 if there are more than max.
int eremite_tokenize(eremite_tokenizer *t, const char *text, size_t len,
                     int mode, int32_t *out, int max);
// Returns a token's bytes, and sets *len to their length: 0 for an unknown id.
const char *eremite_token_bytes(const eremite_tokenizer *t, int32_t id,
                                size_t *len);
// Returns whether the token is one of the tokenizer's added tokens: 2 for one
// that tokenizer.json marks special, 1 for another, 0 for none.
int eremite_token_kind(const eremite_tokenizer *t, int32_t id);
// Returns the id of the added token spelled by len bytes of text, or -1.
int32_t eremite_find_added_token(const eremite_tokenizer *t, const char *text,
                                 size_t len);

#endif
