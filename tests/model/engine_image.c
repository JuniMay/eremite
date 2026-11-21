#include "base/clock.h"
#include "base/error.h"
#include "eremite.h"
#include "platform/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// End-to-end C engine API test for MiniMax-H3-Image through eremite.h:
// registration, full-residence load, synchronous and stream generation, PNG
// and JPEG output, non-32 exact resizing, >4096-token prompt truncation,
// reference editing, masks, transparent backgrounds, partial images, invalid
// input reporting, cancellation, and full unload.
//
// Usage:
//   build/tests/model-engine_image <model_dir> <text_enc.safetensors>
//       <dit.safetensors> <vae.safetensors> [ref_image.png]

static int failures;

// Records a test assertion and prints a failure line when cond is zero.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

// The partial images that a call sent.
typedef struct {
  // Width and height in pixels that each must decode to.
  int width, height;
  // Partial images received.
  int count;
  // Whether each came with the next index and decoded to the size.
  int in_order_and_sized;
} PartialImages;

// Counts one partial image of a call and checks its index and size.
static void keep_partial_image(void *context, const uint8_t *data, size_t len,
                               int index) {
  PartialImages *partials = context;
  int width = 0, height = 0;
  const char *why = NULL;
  uint8_t *pixels = image_decode_rgb(data, len, &width, &height, &why);
  if (!pixels || index != partials->count || width != partials->width ||
      height != partials->height)
    partials->in_order_and_sized = 0;
  free(pixels);
  partials->count++;
}

// Reads a binary file into a heap buffer and sets *out_len.
static uint8_t *read_binary_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = xmalloc((size_t)sz);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("read failed: %s", path);
  fclose(f);
  *out_len = (size_t)sz;
  return buf;
}

int main(int argc, char **argv) {
  const char *model_dir =
      argc > 1 ? argv[1] : getenv("EREMITE_MINIMAX_H3_MODEL_DIR");
  const char *te_path =
      argc > 2 ? argv[2] : getenv("EREMITE_MINIMAX_H3_TEXT_ENCODER");
  const char *dit_path = argc > 3 ? argv[3] : getenv("EREMITE_MINIMAX_H3_DIT");
  const char *vae_path = argc > 4 ? argv[4] : getenv("EREMITE_MINIMAX_H3_VAE");
  const char *ref_path =
      argc > 5 ? argv[5] : getenv("EREMITE_MINIMAX_H3_REF_IMAGE");
  if (!model_dir || !te_path || !dit_path || !vae_path) {
    printf("skipping engine_image: pass <model_dir> <text_enc> <dit> <vae>\n");
    return 0;
  }

  eremite_process proc = {.abi = EREMITE_ABI, .reserve_gb = 4.0};
  check(eremite_init(&proc) == 0, "eremite_init succeeds");

  // Reject invalid configs before registering the real model.
  eremite_config bad_missing = {
      .abi = EREMITE_ABI,
      .model = model_dir,
      .ctx = 4096,
      .minimax_h3 = {.text_encoder = te_path, .dit = NULL, .vae = vae_path}};
  check(eremite_open(&bad_missing) == NULL,
        "eremite_open rejects image model missing dit path");

  eremite_config bad_draft = {.abi = EREMITE_ABI,
                              .model = model_dir,
                              .ctx = 4096,
                              .deepseek = {.draft = 2},
                              .minimax_h3 = {.text_encoder = te_path,
                                             .dit = dit_path,
                                             .vae = vae_path}};
  check(eremite_open(&bad_draft) == NULL,
        "eremite_open rejects image model with draft option");

  eremite_config bad_ctx = {.abi = EREMITE_ABI,
                            .model = model_dir,
                            .ctx = 4097,
                            .minimax_h3 = {.text_encoder = te_path,
                                           .dit = dit_path,
                                           .vae = vae_path}};
  check(eremite_open(&bad_ctx) == NULL,
        "eremite_open rejects image model ctx above max_prompt_tokens");

  eremite_config cfg = {.abi = EREMITE_ABI,
                        .model = model_dir,
                        .ctx = 0,
                        .minimax_h3 = {.text_encoder = te_path,
                                       .dit = dit_path,
                                       .vae = vae_path}};
  eremite_model *m = eremite_open(&cfg);
  check(m != NULL, "eremite_open registers MiniMax-H3-Image with default ctx");
  if (!m) return 1;
  check(eremite_is_image_model(m) == 1 && eremite_is_decision_model(m) == 0,
        "model reports image kind and not decision kind");
  eremite_image_limits lim = {0};
  check(eremite_image_model_limits(NULL, &lim) == 0 &&
            eremite_image_model_limits(m, &lim) == 1 &&
            lim.max_prompt_tokens == 4096 && lim.inference_steps == 6 &&
            lim.max_request_pixels == (32ll << 20),
        "eremite_image_model_limits requires an opened image model");
  check(eremite_is_loaded(m) == 0,
        "weights are not loaded before eremite_load");
  check(eremite_model_tokenizer(m) != NULL,
        "model tokenizer opens from tokenizer subfolder");

  double t0 = now();
  int loaded = eremite_load(m);
  printf("eremite_load: %s in %.2f s\n", loaded ? "ok" : "failed", now() - t0);
  check(loaded == 1 && eremite_is_loaded(m) == 1,
        "eremite_load brings core T2I weights resident");

  eremite_refresh_stats();
  eremite_stats st = {0};
  eremite_read_stats(m, &st);
  printf("loaded stats: dense=%.2f GiB required=%.2f GiB workspace=%.2f MiB\n",
         (double)st.dense_bytes / (double)(1ull << 30),
         (double)st.required_bytes / (double)(1ull << 30),
         (double)st.model_workspace / (double)(1ull << 20));
  check(st.dense_bytes > (13ull << 30) && st.required_bytes > (20ull << 30),
        "stats report resident weights and > 20 GiB minimum stage requirement");

  // 1. Synchronous T2I at non-32 size 500x500 (min_area=1024 -> 512x512 canvas
  //    resized back to exact 500x500 PNG).
  eremite_image_request req_png = {
      .prompt = "A red ceramic mug on a white studio table.",
      .width = 500,
      .height = 500,
      .min_generation_area = 1024,
      .format = EREMITE_IMAGE_PNG,
      .seed = 11};
  eremite_image_result res = {0};
  double t1 = now();
  int ok_png = eremite_generate_image(m, &req_png, &res);
  printf("sync T2I 500x500 PNG: ok=%d len=%zu in %.2f s\n", ok_png, res.len,
         now() - t1);
  int dec_w = 0, dec_h = 0;
  const char *derr = NULL;
  uint8_t *dec =
      ok_png ? image_decode_rgb(res.data, res.len, &dec_w, &dec_h, &derr)
             : NULL;
  check(ok_png && dec != NULL && dec_w == 500 && dec_h == 500 &&
            res.prompt_tokens > 0 && res.prefill_seconds > 0.0 &&
            res.image_seconds > 0.0,
        "sync T2I PNG decodes to exact 500x500 with prompt tokens and timings");
  int first_prompt_tokens = res.prompt_tokens;
  free(dec);
  eremite_image_result_free(&res);

  eremite_refresh_stats();
  eremite_read_stats(m, &st);
  check(st.prefill_rows == (uint64_t)first_prompt_tokens &&
            st.generated_images == 1 && st.image_steps == 6 &&
            st.image_seconds > 0.0 &&
            st.image_stage == EREMITE_IMAGE_STAGE_NONE &&
            st.prefill_layer == 0 && st.image_step == 0 && st.decode_rows == 0,
        "stats separate prompt prefill tokens from image counts and steps");

  // 2. Stream T2I at 512x512 JPEG (quality 90).
  eremite_stream *stream = eremite_stream_new(m);
  eremite_image_request req_jpg = {
      .prompt = "A blue glass bottle in bright sunlight.",
      .width = 512,
      .height = 512,
      .min_generation_area = 1024,
      .format = EREMITE_IMAGE_JPEG,
      .quality = 90,
      .seed = 22};
  eremite_stream_generate_image(stream, &req_jpg);
  double t2 = now();
  int done = eremite_run_streams(&stream, 1);
  int ok_jpg = eremite_stream_image_result(stream, &res);
  printf("stream T2I 512x512 JPEG: done=%d ok=%d len=%zu in %.2f s\n", done,
         ok_jpg, res.len, now() - t2);
  dec_w = dec_h = 0;
  derr = NULL;
  dec = ok_jpg ? image_decode_rgb(res.data, res.len, &dec_w, &dec_h, &derr)
               : NULL;
  check(done == 0 && ok_jpg && res.len > 2 && res.data[0] == 0xFF &&
            res.data[1] == 0xD8 && dec != NULL && dec_w == 512 && dec_h == 512,
        "stream T2I JPEG has SOI marker and decodes to 512x512");
  free(dec);
  eremite_image_result_free(&res);

  // 3. Invalid client inputs report invalid_input = 1 without crashing.
  static const uint8_t garbage[] = {0x00, 0x01, 0x02, 0x03};
  eremite_image_input bad_ref = {.data = garbage, .len = sizeof garbage};
  eremite_image_request req_bad = {.prompt = "edit",
                                   .width = 512,
                                   .height = 512,
                                   .min_generation_area = 1024,
                                   .format = EREMITE_IMAGE_PNG,
                                   .refs = &bad_ref,
                                   .ref_count = 1};
  int ok_bad = eremite_generate_image(m, &req_bad, &res);
  check(!ok_bad && res.data == NULL && res.invalid_input == 1 &&
            res.error[0] != '\0',
        "corrupt reference image fails cleanly with invalid_input=1");
  eremite_image_result_free(&res);

  // 4. Long prompt (> 4096 tokens) truncates cleanly instead of aborting.
  const char *phrase = "warm sunlit wooden table ";
  size_t plen = strlen(phrase);
  char *long_prompt = xmalloc(plen * 1200 + 1);
  for (int i = 0; i < 1200; i++) memcpy(long_prompt + i * plen, phrase, plen);
  long_prompt[1200 * plen] = '\0';
  eremite_image_request req_long = {.prompt = long_prompt,
                                    .width = 256,
                                    .height = 256,
                                    .min_generation_area = 1024,
                                    .format = EREMITE_IMAGE_PNG,
                                    .seed = 33};
  int ok_long = eremite_generate_image(m, &req_long, &res);
  free(long_prompt);
  check(ok_long && res.data != NULL && res.len > 0,
        "prompt longer than 4096 tokens truncates and generates cleanly");
  eremite_image_result_free(&res);

  // 5. Cooperative stream cancellation stops cleanly with invalid_input = 0.
  eremite_stream_generate_image(stream, &req_png);
  eremite_stream_cancel(stream);
  int done_c = eremite_run_streams(&stream, 1);
  int ok_c = eremite_stream_image_result(stream, &res);
  check(done_c == 0 && !ok_c && res.data == NULL && res.invalid_input == 0,
        "cancelled image stream returns cleanly without invalid_input");
  eremite_image_result_free(&res);

  // Between calls the model counts its weights only: each call's workspace
  // buffers stop counting when they are freed.
  eremite_refresh_stats();
  eremite_read_stats(m, &st);
  printf("idle workspace after the calls: %.2f MiB\n",
         (double)st.model_workspace / (double)(1ull << 20));
  check(st.model_workspace == 0,
        "an idle image model counts no workspace after its calls");

  // 6. Optional reference image editing through stream API.
  if (ref_path) {
    size_t rlen = 0;
    uint8_t *rdata = read_binary_file(ref_path, &rlen);
    eremite_image_input ref_in = {.data = rdata, .len = rlen};
    int ref_w = 0, ref_h = 0, fit_w = 0, fit_h = 0;
    char size_error[EREMITE_IMAGE_ERROR_BYTES] = {0};
    check(eremite_image_read_size(rdata, rlen, &ref_w, &ref_h, size_error,
                                  sizeof size_error) &&
              eremite_image_fit_size(m, ref_w, ref_h, &fit_w, &fit_h,
                                     size_error, sizeof size_error) &&
              fit_w == ref_w && fit_h == ref_h,
          "the reference image's size reads and fits the model as it is");
    eremite_image_request req_edit = {
        .prompt = "Keep <Picture 1> unchanged with warm golden lighting.",
        .width = 512,
        .height = 512,
        .min_generation_area = 1024,
        .format = EREMITE_IMAGE_PNG,
        .seed = 42,
        .refs = &ref_in,
        .ref_count = 1};
    eremite_stream_generate_image(stream, &req_edit);
    double t3 = now();
    int done_e = eremite_run_streams(&stream, 1);
    int ok_e = eremite_stream_image_result(stream, &res);
    printf("stream edit 512x512 PNG: done=%d ok=%d len=%zu in %.2f s%s%s\n",
           done_e, ok_e, res.len, now() - t3, ok_e ? "" : ": ", res.error);
    dec_w = dec_h = 0;
    derr = NULL;
    dec = ok_e ? image_decode_rgb(res.data, res.len, &dec_w, &dec_h, &derr)
               : NULL;
    check(
        done_e == 0 && ok_e && dec != NULL && dec_w == 512 && dec_h == 512 &&
            res.prompt_tokens > 256 && res.prefill_seconds > 0.0 &&
            res.image_seconds > 0.0,
        "stream edit produces 512x512 PNG and counts reference vision tokens");
    free(dec);
    eremite_image_result_free(&res);

    eremite_refresh_stats();
    eremite_read_stats(m, &st);
    printf("after edit dense=%.2f GiB prefill_rows=%llu images=%llu "
           "steps=%llu\n",
           (double)st.dense_bytes / (double)(1ull << 30),
           (unsigned long long)st.prefill_rows,
           (unsigned long long)st.generated_images,
           (unsigned long long)st.image_steps);
    check(st.dense_bytes > (20ull << 30) && st.generated_images == 4 &&
              st.image_steps == 24 && st.decode_rows == 0,
          "stats report edit weights, 4 images, 24 steps and 0 decode rows");

    // A masked edit keeps the reference's pixels where the mask is opaque:
    // here its bottom half, which must come back exactly.
    int orig_w = 0, orig_h = 0;
    const char *orig_error = NULL;
    uint8_t *orig =
        image_decode_rgb(rdata, rlen, &orig_w, &orig_h, &orig_error);
    uint8_t *mask_px = xmalloc((size_t)orig_w * (size_t)orig_h * 4);
    for (int y = 0; y < orig_h; y++)
      for (int x = 0; x < orig_w; x++) {
        uint8_t *px = mask_px + ((size_t)y * (size_t)orig_w + (size_t)x) * 4;
        px[0] = px[1] = px[2] = 0;
        px[3] = y < orig_h / 2 ? 0 : 255;
      }
    size_t mask_len = 0;
    const char *mask_error = NULL;
    uint8_t *mask_png =
        image_encode_png_rgba(mask_px, orig_w, orig_h, &mask_len, &mask_error);
    eremite_image_request req_masked = {
        .prompt = "Turn <Picture 1> into a pencil sketch.",
        .width = orig_w,
        .height = orig_h,
        .min_generation_area = 1024,
        .format = EREMITE_IMAGE_PNG,
        .seed = 5,
        .refs = &ref_in,
        .ref_count = 1,
        .mask = {.data = mask_png, .len = mask_len}};
    int ok_masked = eremite_generate_image(m, &req_masked, &res);
    int out_w = 0, out_h = 0;
    const char *out_error = NULL;
    uint8_t *out_px = ok_masked ? image_decode_rgb(res.data, res.len, &out_w,
                                                   &out_h, &out_error)
                                : NULL;
    size_t top = (size_t)orig_w * (size_t)(orig_h / 2) * 3,
           all = (size_t)orig_w * (size_t)orig_h * 3;
    printf("masked edit %dx%d: ok=%d %s\n", orig_w, orig_h, ok_masked,
           ok_masked ? "" : res.error);
    check(
        orig && mask_png && ok_masked && out_px && out_w == orig_w &&
            out_h == orig_h && !memcmp(out_px + top, orig + top, all - top),
        "a masked edit keeps the reference's pixels where the mask is opaque");
    free(out_px);
    eremite_image_result_free(&res);

    // A transparent masked edit: the matte reads the edit's own key-colored
    // border, and then the mask keeps the reference's pixels, opaque.
    req_masked.background = EREMITE_IMAGE_BACKGROUND_TRANSPARENT;
    int ok_masked_clear = eremite_generate_image(m, &req_masked, &res);
    out_px = ok_masked_clear ? image_decode_rgb(res.data, res.len, &out_w,
                                                &out_h, &out_error)
                             : NULL;
    uint8_t *out_alpha =
        ok_masked_clear
            ? image_decode_alpha(res.data, res.len, &out_w, &out_h, &out_error)
            : NULL;
    // The mask keeps the rows from the middle one down.
    size_t pixels = (size_t)orig_w * (size_t)orig_h,
           first_kept = (size_t)orig_w * (size_t)(orig_h / 2), kept_opaque = 0;
    for (size_t i = first_kept; out_alpha && i < pixels; i++)
      kept_opaque += out_alpha[i] == 255;
    printf("transparent masked edit %dx%d: ok=%d %s\n", orig_w, orig_h,
           ok_masked_clear, ok_masked_clear ? "" : res.error);
    check(out_px && out_alpha && out_w == orig_w && out_h == orig_h &&
              !memcmp(out_px + top, orig + top, all - top) &&
              kept_opaque == pixels - first_kept && !out_alpha[0],
          "a transparent masked edit clears the edit's background and keeps "
          "the reference's pixels opaque under the mask");
    free(out_alpha);
    free(out_px);
    free(mask_png);
    free(mask_px);
    free(orig);
    eremite_image_result_free(&res);
    // The masked edits above read the reference's bytes, so they stay until
    // here.
    free(rdata);
  }

  // A transparent background: the key-colored background turns clear, so the
  // corners have alpha 0 and some pixels stay opaque.
  eremite_image_request req_clear = {.prompt = "A red apple in the middle of "
                                               "the frame.",
                                     .width = 512,
                                     .height = 512,
                                     .min_generation_area = 1024,
                                     .format = EREMITE_IMAGE_PNG,
                                     .background =
                                         EREMITE_IMAGE_BACKGROUND_TRANSPARENT,
                                     .seed = 21};
  int ok_clear = eremite_generate_image(m, &req_clear, &res);
  int clear_w = 0, clear_h = 0;
  const char *clear_error = NULL;
  uint8_t *clear_alpha = ok_clear
                             ? image_decode_alpha(res.data, res.len, &clear_w,
                                                  &clear_h, &clear_error)
                             : NULL;
  size_t opaque_pixels = 0;
  for (int i = 0; clear_alpha && i < clear_w * clear_h; i++)
    opaque_pixels += clear_alpha[i] == 255;
  printf("transparent 512x512: ok=%d opaque pixels %zu %s\n", ok_clear,
         opaque_pixels, ok_clear ? "" : res.error);
  check(clear_alpha && clear_w == 512 && clear_h == 512 && !clear_alpha[0] &&
            !clear_alpha[511] && !clear_alpha[511 * 512] &&
            !clear_alpha[512 * 512 - 1] && opaque_pixels > 0,
        "a transparent background clears the corners and keeps the subject");
  free(clear_alpha);
  eremite_image_result_free(&res);

  // The key sentence goes before the prompt, so a prompt past the token limit
  // keeps it, and the corners still turn clear.
  enum { BALL_REPEATS = 1200 };  // about 4800 tokens, past the 4096 limit

  const char *ball = "a small red ball ";
  size_t ball_len = strlen(ball);
  char *long_clear = xmalloc(ball_len * BALL_REPEATS + 1);
  for (int i = 0; i < BALL_REPEATS; i++)
    memcpy(long_clear + i * ball_len, ball, ball_len);
  long_clear[BALL_REPEATS * ball_len] = '\0';
  req_clear.prompt = long_clear;
  req_clear.width = req_clear.height = 256;
  int ok_long_clear = eremite_generate_image(m, &req_clear, &res);
  free(long_clear);
  uint8_t *long_alpha = ok_long_clear
                            ? image_decode_alpha(res.data, res.len, &clear_w,
                                                 &clear_h, &clear_error)
                            : NULL;
  printf("transparent with a long prompt: ok=%d %s\n", ok_long_clear,
         ok_long_clear ? "" : res.error);
  check(long_alpha && clear_w == 256 && clear_h == 256 && !long_alpha[0] &&
            !long_alpha[256 * 256 - 1],
        "a transparent prompt past the token limit still clears the corners");
  free(long_alpha);
  eremite_image_result_free(&res);

  // Partial images: two estimates come before the image, in order and of the
  // output size, and the image is the same as one without them.
  eremite_image_request req_partial = req_png;
  int ok_plain = eremite_generate_image(m, &req_partial, &res);
  int plain_w = 0, plain_h = 0;
  const char *plain_error = NULL;
  uint8_t *plain = ok_plain ? image_decode_rgb(res.data, res.len, &plain_w,
                                               &plain_h, &plain_error)
                            : NULL;
  eremite_image_result_free(&res);
  PartialImages partials = {.width = req_png.width,
                            .height = req_png.height,
                            .in_order_and_sized = 1};
  req_partial.partial_images = 2;
  req_partial.on_partial_image = keep_partial_image;
  req_partial.partial_image_context = &partials;
  int ok_partial = eremite_generate_image(m, &req_partial, &res);
  int partial_w = 0, partial_h = 0;
  const char *partial_error = NULL;
  uint8_t *with_partials = ok_partial
                               ? image_decode_rgb(res.data, res.len, &partial_w,
                                                  &partial_h, &partial_error)
                               : NULL;
  printf("partial images: ok=%d sent %d %s\n", ok_partial, partials.count,
         ok_partial ? "" : res.error);
  check(
      plain && with_partials && partials.count == 2 &&
          partials.in_order_and_sized && plain_w == partial_w &&
          plain_h == partial_h &&
          !memcmp(plain, with_partials, (size_t)plain_w * (size_t)plain_h * 3),
      "two partial images come in order at the output size, and the image "
      "matches one without them");
  free(with_partials);
  free(plain);
  eremite_image_result_free(&res);

  eremite_stream_free(stream);

  // 7. Idle unload frees all resident weight groups and clears is_loaded.
  uint64_t staged_required = st.required_bytes;
  eremite_set_request_running(m, 0);
  uint64_t freed = eremite_unload(m);
  printf("eremite_unload freed %.2f GiB, is_loaded=%d\n",
         (double)freed / (double)(1ull << 30), eremite_is_loaded(m));
  check(freed > (20ull << 30) && eremite_is_loaded(m) == 0,
        "eremite_unload releases all weight groups cleanly");

  // 8. A canvas that needs more than the memory limit is refused before the
  //    text encoder runs, as invalid input, with a message naming the canvas
  //    and the memory.
  uint64_t tight_limit = staged_required + (512ull << 20);
  eremite_set_memory_limit(tight_limit);
  int staged_loaded = eremite_load(m);
  check(staged_loaded == 1 && eremite_is_loaded(m) == 1,
        "staged reload succeeds under tight memory limit");
  eremite_refresh_stats();
  eremite_read_stats(m, &st);
  uint64_t prefill_before = st.prefill_rows;
  eremite_image_request req_huge = {.prompt = "A wide mountain landscape.",
                                    .width = 4896,
                                    .height = 4896,
                                    .min_generation_area = 1024,
                                    .format = EREMITE_IMAGE_PNG,
                                    .seed = 55};
  int ok_huge = eremite_generate_image(m, &req_huge, &res);
  eremite_refresh_stats();
  eremite_read_stats(m, &st);
  printf("huge canvas under %.2f GiB limit: ok=%d prefill_delta=%llu err=%s\n",
         (double)tight_limit / (double)(1ull << 30), ok_huge,
         (unsigned long long)(st.prefill_rows - prefill_before), res.error);
  check(!ok_huge && res.invalid_input == 1 &&
            st.prefill_rows == prefill_before &&
            strstr(res.error, "4896x4896") != NULL &&
            strstr(res.error, "more than") != NULL,
        "a canvas over the memory limit is refused early as invalid input");
  eremite_image_result_free(&res);
  eremite_set_request_running(m, 0);
  eremite_unload(m);
  eremite_set_memory_limit(0);

  printf("engine_image checks: %d failures\n", failures);
  return failures != 0;
}
