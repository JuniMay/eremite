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
// reference editing, invalid input reporting, cancellation, and full unload.
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

  eremite_config cfg = {.abi = EREMITE_ABI,
                        .model = model_dir,
                        .ctx = 4096,
                        .minimax_h3 = {.text_encoder = te_path,
                                       .dit = dit_path,
                                       .vae = vae_path}};
  eremite_model *m = eremite_open(&cfg);
  check(m != NULL, "eremite_open registers MiniMax-H3-Image");
  if (!m) return 1;
  check(eremite_is_image_model(m) == 1 && eremite_is_decision_model(m) == 0,
        "model reports image kind and not decision kind");
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

  // 6. Optional reference image editing through stream API.
  if (ref_path) {
    size_t rlen = 0;
    uint8_t *rdata = read_binary_file(ref_path, &rlen);
    eremite_image_input ref_in = {.data = rdata, .len = rlen};
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
    printf("stream edit 512x512 PNG: done=%d ok=%d len=%zu in %.2f s\n", done_e,
           ok_e, res.len, now() - t3);
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
    free(rdata);

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
  }

  eremite_stream_free(stream);

  // 7. Idle unload frees all resident weight groups and clears is_loaded.
  eremite_set_request_running(m, 0);
  uint64_t freed = eremite_unload(m);
  printf("eremite_unload freed %.2f GiB, is_loaded=%d\n",
         (double)freed / (double)(1ull << 30), eremite_is_loaded(m));
  check(freed > (20ull << 30) && eremite_is_loaded(m) == 0,
        "eremite_unload releases all weight groups cleanly");

  printf("engine_image checks: %d failures\n", failures);
  return failures != 0;
}
