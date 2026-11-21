#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "eremite.h"
#include "platform/video.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Real-checkpoint fairness and speed test for MiniMax-H3 Video beside a
// language model (Inkling-2-Mini or DeepSeek V4.1 Flash) in one process:
// 1. Measures solo LLM prefill speed, solo LLM decode speed, and solo video
//    generation time.
// 2. Runs video generation beside continuous LLM decoding and checks balanced
//    throughput and bounded per-token latency.
// 3. Runs video generation beside LLM prompt prefill and checks that prefill
//    makes steady progress and finishes promptly while video is running.
// 4. Verifies that concurrent LLM logits match direct solo calls and that the
//    concurrent MP4 decodes cleanly.
//
// usage: model-video_llm_turns LLM_DIR VIDEO_DIR TEXT_ENC DIT VVAE AVAE
//        [GGUF] [--size W H FRAMES STEPS]

// Language-model context holding prompt prefill and all decoded tokens.
static constexpr int CONTEXT = 4096;

// Engine memory budget and reserve in GiB (fits DeepSeek/Inkling + MiniMax-H3
// Video together in full residence).
static constexpr double MEMORY_GB = 96.0, RESERVE_GB = 4.0;

// Solo timing runs for decode and prompt prefill length in tokens.
static constexpr int SOLO_DECODE_TOKENS = 24, PREFILL_TOKENS = 1024;

// Video clip geometry: 768x448, 22 frames (~0.92 s at 24 fps), 4 steps.
// Large enough that unsplit DiT blocks visibly starve concurrent LLM decoding.
static constexpr int VIDEO_W = 768, VIDEO_H = 448, VIDEO_FRAMES = 22,
                     VIDEO_STEPS = 4;

// Deterministic video RNG seeds.
static constexpr uint64_t SEED_SOLO = 42, SEED_CONCURRENT = 42;

// Minimum acceptable concurrent LLM throughput as a fraction of solo speed
// (50/50 sharing minus scheduling overhead; unfixed engine drops below 10%).
static constexpr double MIN_CONCURRENT_LLM_RATIO = 0.25;

// Maximum acceptable single-token decode latency multiplier over solo decode
// when sharing the GPU with video generation.
static constexpr double MAX_CONCURRENT_TOKEN_MULTIPLE = 8.0;

// Maximum acceptable concurrent prefill wall time as a multiple of solo
// prefill time (at 50/50 sharing, concurrent prefill takes ~2x solo).
static constexpr double MAX_CONCURRENT_PREFILL_MULTIPLE = 4.0;

// Prompt decoded by the language model.
static const char DECODE_PROMPT[] =
    "Write a detailed story about a lighthouse keeper watching a storm.";

// Prompt encoded by the video model.
static const char VIDEO_PROMPT[] =
    "A golden retriever running across a sunny green meadow with birds.";

// Returns the token id with the highest logit.
static int32_t argmax(const float *logits, int vocab) {
  int32_t best = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[best]) best = i;
  return best;
}

// Builds a deterministic token buffer of target_len tokens by repeating seed.
static int32_t *make_long_prompt(const int32_t *seed, int seed_n,
                                 int target_len) {
  int32_t *buf = xmalloc((size_t)target_len * sizeof *buf);
  for (int i = 0; i < target_len; i++) buf[i] = seed[i % seed_n];
  return buf;
}

int main(int argc, char **argv) {
  int video_w = VIDEO_W, video_h = VIDEO_H, video_frames = VIDEO_FRAMES,
      video_steps = VIDEO_STEPS;
  const char *gguf = NULL;
  if (argc < 7)
    die("usage: model-video_llm_turns LLM_DIR VIDEO_DIR TEXT_ENC DIT VVAE "
        "AVAE [GGUF] [--size W H FRAMES STEPS]");
  for (int i = 7; i < argc; i++) {
    if (!strcmp(argv[i], "--size") && i + 4 < argc) {
      video_w = atoi(argv[++i]);
      video_h = atoi(argv[++i]);
      video_frames = atoi(argv[++i]);
      video_steps = atoi(argv[++i]);
    } else if (!gguf) {
      gguf = argv[i];
    } else {
      die("unexpected argument: %s", argv[i]);
    }
  }

  check(eremite_init(&(eremite_process){.abi = EREMITE_ABI,
                                        .memory_gb = MEMORY_GB,
                                        .reserve_gb = RESERVE_GB}) == 0,
        "engine initializes");

  eremite_model *llm = eremite_open(&(eremite_config){
      .abi = EREMITE_ABI,
      .model = argv[1],
      .ctx = CONTEXT,
      .deepseek.experts = gguf,
  });
  eremite_model *vid = eremite_open(&(eremite_config){
      .abi = EREMITE_ABI,
      .model = argv[2],
      .minimax_h3_video = {.text_encoder = argv[3],
                           .dit = argv[4],
                           .vae = argv[5],
                           .audio_vae = argv[6],
                           .default_steps = video_steps,
                           .variant = EREMITE_VIDEO_VARIANT_FL2VA},
  });
  check(llm && vid && eremite_load(llm) && eremite_load(vid),
        "both models load");
  if (!llm || !vid || !eremite_is_loaded(llm) || !eremite_is_loaded(vid))
    return 1;

  int vocab = eremite_vocab_size(llm);
  int32_t base_tokens[256];
  int base_n = eremite_tokenize(eremite_model_tokenizer(llm), DECODE_PROMPT,
                                strlen(DECODE_PROMPT), 0, base_tokens, 256);
  int32_t *long_tokens = make_long_prompt(base_tokens, base_n, PREFILL_TOKENS);

  eremite_stream *sl = eremite_stream_new(llm);
  eremite_stream *sv = eremite_stream_new(vid);
  eremite_video_request vreq = {
      .prompt = VIDEO_PROMPT,
      .width = video_w,
      .height = video_h,
      .frames = video_frames,
      .steps = video_steps,
      .seed = SEED_SOLO,
  };

  // 1. Solo LLM prefill and decode baseline.
  eremite_state *st_solo = eremite_state_new(llm);
  eremite_stream_prefill(sl, st_solo, long_tokens, PREFILL_TOKENS, 0, 0, -1,
                         NULL);
  double t0 = now();
  check(eremite_run_streams(&sl, 1) == 0, "solo LLM prefill completes");
  double solo_prefill_s = now() - t0;
  double solo_prefill_tps = (double)PREFILL_TOKENS / solo_prefill_s;
  const float *solo_prefill_logits = eremite_stream_logits(sl);
  check(solo_prefill_logits != NULL, "solo LLM prefill produced logits");
  int32_t expected_prefill_tok = argmax(solo_prefill_logits, vocab);
  eremite_state_free(llm, st_solo);

  eremite_state *st_dec = eremite_state_new(llm);
  eremite_stream_prefill(sl, st_dec, base_tokens, base_n, 0, 0, -1, NULL);
  eremite_run_streams(&sl, 1);
  int32_t tok = argmax(eremite_stream_logits(sl), vocab);
  int32_t pos = base_n;
  // Warm up one decode step before timing.
  eremite_stream_decode_rows(sl, &st_dec, &tok, &pos, 1);
  eremite_run_streams(&sl, 1);
  tok = argmax(eremite_stream_logits(sl), vocab);
  pos++;

  double solo_dec_start = now();
  double solo_dec_max = 0.0;
  for (int i = 0; i < SOLO_DECODE_TOKENS; i++) {
    double s0 = now();
    eremite_stream_decode_rows(sl, &st_dec, &tok, &pos, 1);
    eremite_run_streams(&sl, 1);
    double dt = now() - s0;
    if (dt > solo_dec_max) solo_dec_max = dt;
    tok = argmax(eremite_stream_logits(sl), vocab);
    pos++;
  }
  double solo_dec_s = now() - solo_dec_start;
  double solo_dec_tps = (double)SOLO_DECODE_TOKENS / solo_dec_s;
  double solo_dec_ms = (solo_dec_s / (double)SOLO_DECODE_TOKENS) * 1e3;
  eremite_state_free(llm, st_dec);

  // 2. Solo video baseline.
  eremite_stream_generate_video(sv, &vreq);
  t0 = now();
  check(eremite_run_streams(&sv, 1) == 0, "solo video completes");
  double solo_video_s = now() - t0;
  eremite_video_result solo_vres = {0};
  check(eremite_stream_video_result(sv, &solo_vres) && solo_vres.len > 0,
        "solo video produced MP4");
  eremite_video_result_free(&solo_vres);

  printf("SOLO: llm prefill %.2f s (%.1f tok/s), llm decode %.1f tok/s "
         "(avg %.1f ms, max %.1f ms), video %.2f s\n",
         solo_prefill_s, solo_prefill_tps, solo_dec_tps, solo_dec_ms,
         solo_dec_max * 1e3, solo_video_s);

  // 3. Concurrent video + continuous LLM decode.
  st_dec = eremite_state_new(llm);
  eremite_stream_prefill(sl, st_dec, base_tokens, base_n, 0, 0, -1, NULL);
  eremite_run_streams(&sl, 1);
  tok = argmax(eremite_stream_logits(sl), vocab);
  pos = base_n;

  vreq.seed = SEED_CONCURRENT;
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_decode_rows(sl, &st_dec, &tok, &pos, 1);
  eremite_stream *streams[2] = {sv, sl};

  int conc_tokens = 0, video_done = 0, llm_idle = 0;
  double conc_start = now();
  double tok_start = conc_start;
  double conc_tok_max = 0.0;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) {
      video_done = 1;
      break;
    }
    if (done == 1) {
      double t1 = now();
      double dt = t1 - tok_start;
      if (dt > conc_tok_max) conc_tok_max = dt;
      tok_start = t1;
      conc_tokens++;
      tok = argmax(eremite_stream_logits(sl), vocab);
      pos++;
      if (pos < CONTEXT) {
        eremite_stream_decode_rows(sl, &st_dec, &tok, &pos, 1);
      } else {
        llm_idle = 1;
        break;
      }
    }
  }
  double conc_video_s = now() - conc_start;
  if (!video_done) eremite_run_streams(&sv, 1);
  if (!llm_idle) eremite_run_streams(&sl, 1);
  eremite_video_result conc_vres = {0};
  check(eremite_stream_video_result(sv, &conc_vres) && conc_vres.len > 0,
        "concurrent video produced MP4");
  if (conc_vres.data) {
    VideoDecoded dec = {0};
    const char *derr = NULL;
    int dok =
        video_decode_mp4(conc_vres.data, conc_vres.len, 24, 32000, &dec, &derr);
    check(dok && dec.frame_count == video_frames,
          "concurrent video MP4 decodes cleanly");
    video_decoded_free(&dec);
  }
  eremite_video_result_free(&conc_vres);
  eremite_state_free(llm, st_dec);

  double conc_dec_tps = (double)conc_tokens / conc_video_s;
  double dec_ratio = conc_dec_tps / solo_dec_tps;
  double max_tok_mult = (conc_tok_max * 1e3) / solo_dec_ms;
  printf("CONCURRENT VIDEO+DECODE: %d tokens in %.2f s (%.1f tok/s, %.1f%% "
         "of solo, max tok %.1f ms = %.1fx solo), video %.2f s (%.2fx solo)\n",
         conc_tokens, conc_video_s, conc_dec_tps, dec_ratio * 100.0,
         conc_tok_max * 1e3, max_tok_mult, conc_video_s,
         conc_video_s / solo_video_s);
  check(dec_ratio >= MIN_CONCURRENT_LLM_RATIO,
        "LLM decode maintains balanced throughput during video generation");
  check(max_tok_mult <= MAX_CONCURRENT_TOKEN_MULTIPLE,
        "LLM decode per-token latency stays bounded during video generation");

  // 4. Concurrent video + LLM prompt prefill.
  eremite_state *st_pre = eremite_state_new(llm);
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_prefill(sl, st_pre, long_tokens, PREFILL_TOKENS, 0, 0, -1,
                         NULL);
  double pre_start = now();
  double conc_prefill_s = 0.0;
  int prefill_done_first = 0, pre_video_done = 0;
  const float *conc_prefill_logits = NULL;
  int32_t conc_prefill_tok = -1;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 1) {
      conc_prefill_s = now() - pre_start;
      prefill_done_first = 1;
      conc_prefill_logits = eremite_stream_logits(sl);
      if (conc_prefill_logits)
        conc_prefill_tok = argmax(conc_prefill_logits, vocab);
      break;
    }
    if (done == 0) {
      pre_video_done = 1;
      eremite_video_result tmp = {0};
      eremite_stream_video_result(sv, &tmp);
      eremite_video_result_free(&tmp);
      break;
    }
  }
  if (!prefill_done_first) {
    eremite_run_streams(&sl, 1);
    conc_prefill_s = now() - pre_start;
    conc_prefill_logits = eremite_stream_logits(sl);
    if (conc_prefill_logits)
      conc_prefill_tok = argmax(conc_prefill_logits, vocab);
  }
  double post_pre_video_start = now();
  if (!pre_video_done) eremite_run_streams(&sv, 1);
  double post_pre_video_s = now() - post_pre_video_start;
  eremite_video_result pre_vres = {0};
  eremite_stream_video_result(sv, &pre_vres);
  eremite_video_result_free(&pre_vres);
  eremite_state_free(llm, st_pre);

  double prefill_mult = conc_prefill_s / solo_prefill_s;
  printf("CONCURRENT VIDEO+PREFILL: prefill %.2f s (%.1f tok/s, %.2fx solo, "
         "finished_first=%d), video finished %.2f s later\n",
         conc_prefill_s, (double)PREFILL_TOKENS / conc_prefill_s, prefill_mult,
         prefill_done_first, post_pre_video_s);
  // Note: when solo_prefill_s > solo_video_s (e.g. DeepSeek 1024-token prefill
  // vs a short 22-frame video), video legitimately finishes first while sharing
  // evenly, so check bounded slowdown in both cases.
  check(conc_prefill_logits != NULL && conc_prefill_tok == expected_prefill_tok,
        "concurrent LLM prefill matches solo prefill prediction");
  check((prefill_done_first || solo_prefill_s > solo_video_s) &&
            prefill_mult <= MAX_CONCURRENT_PREFILL_MULTIPLE,
        "LLM prefill makes balanced progress during video generation");

  free(long_tokens);
  eremite_stream_free(sv);
  eremite_stream_free(sl);
  eremite_drain();
  return failures != 0;
}
