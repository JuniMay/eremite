#include "check.h"
#include "base/clock.h"
#include "base/error.h"
#include "eremite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Tests how a decision takes turns with decoding calls, on DeepSeek V4.1 Flash
// and d1-2B in one process: one stream decodes without stopping while another
// runs a decision. The decision lets the decoding calls posted before it go
// first, and then runs for as long as they took before decoding goes again. So
// a short decision, which takes less time alone than a decoding call, lets at
// most two decoding calls finish while it runs: those posted before it, and
// those after a first turn that ended early because the decoding call before
// it was nearly done. A long decision, which takes the time of several
// decoding calls alone, lets decoding calls finish between its layers. Both
// give the logits of the same decision made directly. While the long decision
// waits for decoding calls that read experts, the engine thread sleeps until
// the reads end: its time waiting for reads grows. The memory limit leaves
// room for part of the experts only, so that decoding calls read some.
//
// usage: model-decision_turns CHECKPOINT D1_CHECKPOINT [GGUF]

// DeepSeek's context, which holds every token that the decoding stream decodes.
static constexpr int CONTEXT = 4096;

// The decoding calls timed alone, of which the test takes the fastest and the
// slowest, and the runs of each decision timed alone, of which the fastest is
// its time.
static constexpr int TIMED_DECODES = 8, TIMED_RUNS = 3;

// The rows of the long decision: its state fills them up to its question. It
// then takes longer alone than two decoding calls, as the test checks.
static constexpr int LONG_ROWS = 16384;

// The engine thread's state while it waits for reads, as the engine numbers
// it.
static constexpr int THREAD_WAIT_READ = 3;

// The prompt that the decoding stream decodes after.
static const char prompt[] = "Write a short story about a lighthouse keeper.";

// A JSON state and one yes-or-no question, as d1's prompt holds them: the
// short decision. Its first TURN_OPEN tokens open the user's turn, its state's
// text follows them, and its question follows its first STATE tokens.
static const int32_t short_tokens[] = {
    124894, 124899, 5922,   207,   2718,   229,   496,   99489,  6380,  827,
    360,    496,    6929,   6380,  496,    16308, 52966, 278,    48505, 7285,
    1917,   229,    10016,  229,   496,    3166,  6380,  827,    360,   496,
    16414,  273,    6380,   19843, 733,    229,   3255,  920,    733,   229,
    3255,   22,     29,     207,   360,    40439, 360,   496,    37507, 6380,
    2846,   920,    360,    496,   14700,  6380,  3495,  207,    229,   1130,
    64024,  63509,  833,    29088, 278,    29936, 7986,  267,    48505, 7285,
    7450,   467,    278,    2064,  64929,  57922, 415,   18171,  432,   794,
    1049,   22,     124900, 207,   124899, 63514, 207,   124901, 124902};
static constexpr int SHORT_ROWS = sizeof short_tokens / sizeof *short_tokens,
                     TURN_OPEN = 4, STATE = 63;

// The tokens of yes, Yes, no, No and NO, read at the question's last row.
static const int32_t ids[] = {11683, 12447, 2243, 4547, 19598};
static constexpr int LOGITS = sizeof ids / sizeof *ids;

// A decision: its tokens, its state's rows, its logits made directly, and its
// time alone on a stream.
typedef struct {
  const char *what;
  const int32_t *tokens;
  int n, state;
  float expected[LOGITS];
  double alone;
} Decision;

// The decoding stream: its state, the token it decodes next and that token's
// position.
typedef struct {
  eremite_stream *stream;
  eremite_state *state;
  int32_t token, pos;
} Decoding;

// Returns the calling thread's CPU seconds: the engine thread's, since the
// test runs the streams.
static double read_thread_cpu_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// Reads the process's statistics.
static eremite_stats read_stats(const eremite_model *m) {
  eremite_stats s;
  eremite_refresh_stats();
  eremite_read_stats(m, &s);
  return s;
}

// The token with the highest logit.
static int32_t best(const float *logits, int vocab) {
  int32_t b = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[b]) b = i;
  return b;
}

// Posts the decoding stream's next call.
static void post_decode(Decoding *d) {
  if (d->pos == CONTEXT) die("the decoding stream filled its context");
  eremite_stream_decode_rows(d->stream, &d->state, &d->token, &d->pos, 1);
}

// Takes the next token from the decoding stream's finished call.
static void take_token(Decoding *d, int vocab) {
  const float *logits = eremite_stream_logits(d->stream);
  if (!logits) die("a decoding call did not fit in memory");
  d->token = best(logits, vocab);
  d->pos++;
}

// Posts the decision on stream x.
static void post_decision(eremite_stream *x, const Decision *c) {
  int32_t end = c->n, rows[LOGITS];
  for (int i = 0; i < LOGITS; i++) rows[i] = c->n - 1;
  eremite_stream_decide(x, c->tokens, c->n, c->state, &end, 1, rows, ids,
                        LOGITS, NULL);
}

// Makes the decision directly and keeps its logits; then times it alone on
// stream x.
static void prepare_decision(eremite_model *d1, eremite_stream *x,
                             Decision *c) {
  int32_t end = c->n, rows[LOGITS];
  for (int i = 0; i < LOGITS; i++) rows[i] = c->n - 1;
  const float *direct = eremite_decide(d1, c->tokens, c->n, c->state, &end, 1,
                                       rows, ids, LOGITS, NULL);
  if (!direct) die("%s did not fit in memory", c->what);
  memcpy(c->expected, direct, sizeof c->expected);
  c->alone = 0;
  for (int run = 0; run < TIMED_RUNS; run++) {
    post_decision(x, c);
    double start = now();
    eremite_run_streams(&x, 1);
    double took = now() - start;
    if (!run || took < c->alone) c->alone = took;
  }
}

// Runs the decision on stream x while the decoding stream decodes without
// stopping, and returns how many decoding calls finished while it ran. The
// decoding stream has a call posted before and after. Checks the decision's
// logits against the direct call's.
static int run_beside_decoding(eremite_stream *x, const Decision *c,
                               Decoding *d, int vocab) {
  eremite_stream *streams[] = {d->stream, x};
  post_decision(x, c);
  int during = 0;
  for (;;) {
    int k = eremite_run_streams(streams, 2);
    if (k == 0) {
      during++;
      take_token(d, vocab);
      post_decode(d);
      continue;
    }
    if (k != 1) die("a run returned -1 with calls posted and no wake");
    break;
  }
  const float *got = eremite_stream_logits(x);
  check(got && !memcmp(got, c->expected, sizeof c->expected),
        "the decision beside decoding gives the direct call's logits");
  return during;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4)
    die("usage: model-decision_turns CHECKPOINT D1_CHECKPOINT [GGUF]");
  check(eremite_init(&(eremite_process){
            .abi = EREMITE_ABI, .memory_gb = 48, .reserve_gb = 4}) == 0,
        "the process starts");
  eremite_model *deepseek = eremite_open(
      &(eremite_config){.abi = EREMITE_ABI,
                        .model = argv[1],
                        .ctx = CONTEXT,
                        .deepseek.experts = argc == 4 ? argv[3] : NULL});
  eremite_model *d1 = eremite_open(&(eremite_config){
      .abi = EREMITE_ABI, .model = argv[2], .ctx = LONG_ROWS});
  check(deepseek && d1 && eremite_load(deepseek) && eremite_load(d1),
        "both models load");
  if (!deepseek || !d1 || !eremite_is_loaded(deepseek) ||
      !eremite_is_loaded(d1))
    return 1;
  int vocab = eremite_vocab_size(deepseek);

  // The long decision: the short one's state text, repeated up to LONG_ROWS
  // rows with its question.
  int question = SHORT_ROWS - STATE,
      repeats = (LONG_ROWS - TURN_OPEN - question) / (STATE - TURN_OPEN);
  int long_state = TURN_OPEN + repeats * (STATE - TURN_OPEN);
  int32_t *long_tokens =
      xmalloc((size_t)(long_state + question) * sizeof *long_tokens);
  memcpy(long_tokens, short_tokens, TURN_OPEN * sizeof *long_tokens);
  for (int r = 0; r < repeats; r++)
    memcpy(long_tokens + TURN_OPEN + r * (STATE - TURN_OPEN),
           short_tokens + TURN_OPEN, (STATE - TURN_OPEN) * sizeof *long_tokens);
  memcpy(long_tokens + long_state, short_tokens + STATE,
         (size_t)question * sizeof *long_tokens);
  Decision short_decision = {.what = "the short decision",
                             .tokens = short_tokens,
                             .n = SHORT_ROWS,
                             .state = STATE},
           long_decision = {.what = "the long decision",
                            .tokens = long_tokens,
                            .n = long_state + question,
                            .state = long_state};
  eremite_stream *x = eremite_stream_new(d1);
  prepare_decision(d1, x, &short_decision);
  prepare_decision(d1, x, &long_decision);

  // The decoding stream: the prompt's prefill, then decoding calls timed alone.
  int32_t tokens[256];
  int n = eremite_tokenize(eremite_model_tokenizer(deepseek), prompt,
                           strlen(prompt), 0, tokens, 256);
  Decoding decoding = {eremite_stream_new(deepseek),
                       eremite_state_new(deepseek), 0, n - 1};
  eremite_stream_prefill(decoding.stream, decoding.state, tokens, n, 0, 0, -1,
                         NULL);
  eremite_run_streams(&decoding.stream, 1);
  take_token(&decoding, vocab);
  double fastest = 0, slowest = 0;
  for (int i = 0; i < TIMED_DECODES; i++) {
    post_decode(&decoding);
    double start = now();
    eremite_run_streams(&decoding.stream, 1);
    double took = now() - start;
    take_token(&decoding, vocab);
    if (!i || took < fastest) fastest = took;
    if (took > slowest) slowest = took;
  }
  printf("alone: the short decision %.1f ms, the long one %.1f ms, a decoding "
         "call %.1f to %.1f ms\n",
         short_decision.alone * 1e3, long_decision.alone * 1e3, fastest * 1e3,
         slowest * 1e3);

  post_decode(&decoding);
  int during_short = run_beside_decoding(x, &short_decision, &decoding, vocab);
  eremite_stats before = read_stats(deepseek);
  double cpu_start = read_thread_cpu_seconds(), start = now();
  int during_long = run_beside_decoding(x, &long_decision, &decoding, vocab);
  double cpu = read_thread_cpu_seconds() - cpu_start, took = now() - start;
  eremite_stats after = read_stats(deepseek);
  double expert_reads = after.demand_read_s - before.demand_read_s,
         read_waits = after.thread_s[THREAD_WAIT_READ] -
                      before.thread_s[THREAD_WAIT_READ];
  printf("decoding calls that finished while a decision ran: %d during the "
         "short one, %d during the long one\n",
         during_short, during_long);
  printf("while the long decision ran: %.2f s, the engine thread's CPU time "
         "%.2f s, expert reads %.2f s, the engine thread's waits for reads "
         "%.2f s\n",
         took, cpu, expert_reads, read_waits);
  check(short_decision.alone < fastest,
        "the short decision takes less time alone than a decoding call");
  check(during_short <= 2,
        "at most two decoding calls finish while the short decision runs");
  check(long_decision.alone > 2 * slowest,
        "the long decision takes more time alone than two decoding calls");
  check(during_long >= 2,
        "decoding calls finish between the long decision's layers");
  check(expert_reads > 0,
        "decoding calls read experts while the long decision runs");
  check(read_waits > 0,
        "while the long decision waits for decoding calls that read experts, "
        "the engine thread sleeps until the reads end");

  eremite_run_streams(&decoding.stream, 1);
  eremite_stream_free(decoding.stream);
  eremite_state_free(deepseek, decoding.state);
  eremite_stream_free(x);
  free(long_tokens);
  eremite_drain();
  return failures != 0;
}
