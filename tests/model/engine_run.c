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
// Then one stream decodes without stopping while the other prefills a long
// prompt. The prefill finishes at most one layer between two decoding calls,
// and it finishes. Both streams give the direct calls' logits. Then a prefill
// runs alone while a decoding call waits in a stream that is not run. The
// prefill finishes, and the decoding call finishes once its stream runs. Then
// the statistics count the engine thread's time in passes, the GPU's busy time
// and the states' memory, and the engine drains. Without GGUF, the model uses
// the checkpoint's own experts.
//
// usage: model-engine_run CHECKPOINT [GGUF]

// Tokens decoded after each prompt, and the streams.
static constexpr int DECODE = 4, STREAMS = 2;

// The model's context, and the long prompt's tokens: the whole context, so
// that its prefill takes seconds.
static constexpr int CONTEXT = 4096, LONG_PROMPT = CONTEXT;

// The 64-bit FNV-1a hash's start value and multiplier.
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ull,
                          FNV_PRIME = 1099511628211ull;

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

// Returns the FNV-1a hash of a row of logits. Null logits end the test: the
// call did not fit in memory.
static uint64_t hash_logits(const float *logits, size_t bytes) {
  if (!logits) die("a call did not fit in memory");
  const unsigned char *p = (const unsigned char *)logits;
  uint64_t h = FNV_OFFSET;
  for (size_t i = 0; i < bytes; i++) h = (h ^ p[i]) * FNV_PRIME;
  return h;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: model-engine_run CHECKPOINT [GGUF]");
  check(eremite_init(&(eremite_process){
            .abi = EREMITE_ABI, .memory_gb = 40, .reserve_gb = 4}) == 0,
        "the process starts");
  eremite_model *m = eremite_open(
      &(eremite_config){.abi = EREMITE_ABI,
                        .model = argv[1],
                        .ctx = CONTEXT,
                        .deepseek.experts = argc == 3 ? argv[2] : NULL});
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
  for (int k = 0; k < STREAMS; k++) eremite_state_free(m, states[k]);

  // One stream decodes without stopping while the other prefills a long
  // prompt: the second prompt's tokens, repeated.
  int32_t *long_tokens = xmalloc(LONG_PROMPT * sizeof *long_tokens);
  for (int i = 0; i < LONG_PROMPT; i++) long_tokens[i] = tokens[1][i % n[1]];
  eremite_state *direct = eremite_state_new(m);
  uint64_t long_expected =
      hash_logits(eremite_prefill(m, direct, long_tokens, LONG_PROMPT, 0, 0, -1,
                                  NULL, NULL, NULL),
                  row);
  eremite_state_free(m, direct);
  for (int k = 0; k < STREAMS; k++) states[k] = eremite_state_new(m);
  eremite_stream_prefill(streams[0], states[0], tokens[0], n[0], 0, 0, -1,
                         NULL);
  // The decoding stream's calls, its prefill first: the hash of each one's
  // logits, and the token that the next one decodes. It decodes within its
  // context.
  uint64_t *hashes = xmalloc(CONTEXT * sizeof *hashes);
  int32_t *next = xmalloc(CONTEXT * sizeof *next);
  // during: the decoding calls that finished while the long prompt was
  // prefilled. layers_done: its layers that had finished at the last of them.
  // most_layers: the most layers of it that finished between two, or after the
  // last one.
  int finished = 0, during = 0, layers_done = 0, most_layers = 0,
      long_posted = 0, long_done = 0;
  uint64_t long_got = 0;
  // Each finished call of the decoding stream posts the next one, until the
  // long prompt's prefill has finished.
  for (int decoding = 1; decoding;) {
    int k = eremite_run_streams(streams, STREAMS);
    const float *got = eremite_stream_logits(streams[k]);
    if (k == 1) {
      long_got = hash_logits(got, row);
      long_done = 1;
      eremite_stats st;
      eremite_read_stats(m, &st);
      if (st.layers - layers_done > most_layers)
        most_layers = st.layers - layers_done;
      continue;
    }
    if (long_posted && !long_done) {
      eremite_stats st;
      eremite_read_stats(m, &st);
      if (st.prefill_layer - layers_done > most_layers)
        most_layers = st.prefill_layer - layers_done;
      layers_done = st.prefill_layer;
      during++;
    }
    hashes[finished] = hash_logits(got, row);
    next[finished] = best(got, vocab);
    finished++;
    if (long_done) {
      decoding = 0;
      continue;
    }
    int32_t pos = n[0] + finished - 1;
    if (pos == CONTEXT)
      die("the decoding stream filled its context before the prefill ended");
    eremite_stream_decode_rows(streams[0], &states[0], &next[finished - 1],
                               &pos, 1);
    if (!long_posted) {
      eremite_stream_prefill(streams[1], states[1], long_tokens, LONG_PROMPT, 0,
                             0, -1, NULL);
      long_posted = 1;
    }
  }
  // The same calls made directly.
  different = long_got != long_expected;
  direct = eremite_state_new(m);
  const float *logits =
      eremite_prefill(m, direct, tokens[0], n[0], 0, 0, -1, NULL, NULL, NULL);
  different += hash_logits(logits, row) != hashes[0];
  for (int i = 1; i < finished; i++) {
    int32_t pos = n[0] + i - 1;
    logits = eremite_decode_rows(m, &direct, &next[i - 1], &pos, 1);
    different += hash_logits(logits, row) != hashes[i];
  }
  eremite_state_free(m, direct);
  printf("%d decoding calls finished during a %d-token prefill, at most %d of "
         "its layers between two or after the last; %d calls with logits other "
         "than the direct calls'\n",
         during, LONG_PROMPT, most_layers, different);
  check(most_layers <= 1,
        "a prefill finishes at most one layer between two decoding calls");
  check(different == 0, "a prefill beside a stream that decodes without "
                        "stopping finishes, and both give the direct calls' "
                        "logits");

  // A prefill runs alone while a decoding call waits in a stream that is not
  // run. A prefill that waited for that decoding call could not go on, and the
  // run would end the program.
  for (int k = 0; k < STREAMS; k++) {
    eremite_state_free(m, states[k]);
    states[k] = eremite_state_new(m);
  }
  eremite_stream_prefill(streams[0], states[0], tokens[0], n[0], 0, 0, -1,
                         NULL);
  eremite_run_streams(streams, 1);
  int32_t token = best(eremite_stream_logits(streams[0]), vocab), pos = n[0];
  eremite_stream_decode_rows(streams[0], &states[0], &token, &pos, 1);
  eremite_stream_prefill(streams[1], states[1], tokens[1], n[1], 0, 0, -1,
                         NULL);
  eremite_run_streams(streams + 1, 1);
  const float *prefilled = eremite_stream_logits(streams[1]);
  different = !prefilled || memcmp(prefilled, expected[1], row);
  eremite_run_streams(streams, 1);
  const float *decoded = eremite_stream_logits(streams[0]);
  different += !decoded || memcmp(decoded, expected[0] + vocab, row);
  check(different == 0, "a prefill run alone, and the decoding call that "
                        "waited for its stream, give the direct calls' logits");

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
  free(long_tokens), free(hashes), free(next);
  eremite_drain();
  return failures != 0;
}
