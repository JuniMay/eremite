#include "check.h"
#include "base/error.h"
#include "eremite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests the engine's run path on a DeepSeek V4.1 Flash checkpoint: the model
// loads, and two streams prefill two prompts and decode them through
// eremite_run_streams, their passes interleaved. After a wake, the first run
// returns -1 and the prefills stay posted. Each stream's logits must be
// the same bytes as those of the same calls made directly on a fresh state.
// Then the statistics count the engine thread's time in passes, the GPU's busy
// time and the states' memory, and the engine drains.
//
// usage: model-engine_run CHECKPOINT GGUF

// Tokens decoded after each prompt, and the streams.
static constexpr int DECODE = 4, STREAMS = 2;

// The engine thread's state while it runs a pass, as the engine numbers it.
static constexpr int THREAD_PASS = 1;

static const char *const prompts[STREAMS] = {
    "Write a short story about a lighthouse keeper.",
    "Explain in two sentences why the sky is blue."};

// The token with the highest logit.
static int32_t best(const float *logits, int vocab) {
  int32_t b = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[b]) b = i;
  return b;
}

int main(int argc, char **argv) {
  if (argc != 3) die("usage: model-engine_run CHECKPOINT GGUF");
  check(eremite_init(&(eremite_process){
            .abi = EREMITE_ABI, .memory_gb = 40, .reserve_gb = 4}) == 0,
        "the process starts");
  eremite_model *m =
      eremite_open(&(eremite_config){.abi = EREMITE_ABI,
                                     .model = argv[1],
                                     .ctx = 4096,
                                     .deepseek.experts = argv[2]});
  check(m && eremite_load(m) && eremite_is_loaded(m), "the model loads");
  if (!m || !eremite_is_loaded(m)) return 1;
  int vocab = eremite_vocab_size(m);
  size_t row = (size_t)vocab * sizeof(float);

  // The direct calls: each prompt's prefill, then its decoded tokens.
  int32_t tokens[STREAMS][256];
  int n[STREAMS];
  float *expected[STREAMS];
  for (int k = 0; k < STREAMS; k++) {
    n[k] = eremite_tokenize(eremite_model_tokenizer(m), prompts[k],
                            strlen(prompts[k]), 0, tokens[k], 256);
    expected[k] = xmalloc((DECODE + 1) * row);
    eremite_state *s = eremite_state_new(m);
    const float *logits =
        eremite_prefill(m, s, tokens[k], n[k], 0, 0, -1, NULL, NULL, NULL);
    memcpy(expected[k], logits, row);
    for (int i = 0; i < DECODE; i++) {
      int32_t token = best(logits, vocab), pos = n[k] + i;
      logits = eremite_decode_rows(m, &s, &token, &pos, 1);
      memcpy(expected[k] + (size_t)(i + 1) * vocab, logits, row);
    }
    eremite_state_free(m, s);
  }

  // The same calls in two streams, run together.
  eremite_stream *streams[STREAMS];
  eremite_state *states[STREAMS];
  int step[STREAMS] = {0}, different = 0, calls = 0;
  for (int k = 0; k < STREAMS; k++) {
    streams[k] = eremite_stream_new(m);
    states[k] = eremite_state_new(m);
    eremite_stream_prefill(streams[k], states[k], tokens[k], n[k], 0, 0, -1,
                           NULL);
  }
  // A wake ends the next run before a call finishes. The prefills stay posted,
  // and the runs after it finish them.
  eremite_wake();
  check(eremite_run_streams(streams, STREAMS) == -1,
        "after a wake, the next run returns -1 before a call finishes");
  for (int left = STREAMS; left > 0;) {
    int k = eremite_run_streams(streams, STREAMS);
    if (k < 0) die("a run returned -1 with calls posted and no wake");
    const float *logits = eremite_stream_logits(streams[k]);
    different +=
        !logits || memcmp(logits, expected[k] + (size_t)step[k] * vocab, row);
    calls++;
    if (step[k] == DECODE) {
      left--;
      continue;
    }
    int32_t token = best(logits, vocab), pos = n[k] + step[k]++;
    eremite_stream_decode_rows(streams[k], &states[k], &token, &pos, 1);
  }
  printf("%d stream calls, %d with logits other than the direct calls'\n",
         calls, different);
  check(calls == STREAMS * (DECODE + 1) && different == 0,
        "every stream call gives the logits of the same call made directly");

  eremite_refresh_stats();
  eremite_stats s;
  eremite_read_stats(m, &s);
  eremite_memory_info memory = eremite_read_memory();
  printf("engine thread in passes %.2f s, GPU busy %.2f s, state memory "
         "%.1f MiB\n",
         s.thread_s[THREAD_PASS], s.gpu_busy_s, memory.state_bytes / 1048576.0);
  check(s.thread_s[THREAD_PASS] > 0 && s.gpu_busy_s > 0 &&
            s.gpu_limit < INT64_MAX,
        "the statistics count the engine thread's passes and the GPU's work");
  check(memory.state_bytes > 0 && s.state_allocated == memory.state_bytes &&
            memory.model_bytes > memory.state_bytes,
        "the states' memory is counted, within the model memory");
  for (int k = 0; k < STREAMS; k++) {
    eremite_stream_free(streams[k]);
    eremite_state_free(m, states[k]);
  }
  eremite_drain();
  return failures != 0;
}
