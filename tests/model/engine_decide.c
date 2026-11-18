#include "check.h"
#include "base/error.h"
#include "eremite.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests a decision model through the engine's C interface, on d1-2B's
// checkpoint: the options that eremite_open refuses for it, its limits, a
// decision run directly and on a stream, its counters, and a state's call,
// which ends the program for it (in a child process).
//
// usage: model-engine_decide CHECKPOINT

// A JSON state and one yes-or-no question, as d1's prompt holds them.
static const int32_t tokens[] = {
    124894, 124899, 5922,   207,   2718,   229,   496,   99489,  6380,  827,
    360,    496,    6929,   6380,  496,    16308, 52966, 278,    48505, 7285,
    1917,   229,    10016,  229,   496,    3166,  6380,  827,    360,   496,
    16414,  273,    6380,   19843, 733,    229,   3255,  920,    733,   229,
    3255,   22,     29,     207,   360,    40439, 360,   496,    37507, 6380,
    2846,   920,    360,    496,   14700,  6380,  3495,  207,    229,   1130,
    64024,  63509,  833,    29088, 278,    29936, 7986,  267,    48505, 7285,
    7450,   467,    278,    2064,  64929,  57922, 415,   18171,  432,   794,
    1049,   22,     124900, 207,   124899, 63514, 207,   124901, 124902};
static constexpr int ROWS = sizeof tokens / sizeof *tokens, STATE = 63;
// The tokens of yes, Yes, no, No and NO at the question's last row, their
// logits in LiquidAI's own code in FP32, and the largest difference of its
// BF16 run from them.
static const int32_t rows[] = {ROWS - 1, ROWS - 1, ROWS - 1, ROWS - 1,
                               ROWS - 1},
                     ids[] = {11683, 12447, 2243, 4547, 19598};
static const double reference[] = {8.003815, 10.504749, 5.083753, 7.219336,
                                   1.851632};
static constexpr double BOUND = 0.0619;
static constexpr int LOGITS = sizeof ids / sizeof *ids;
// The token that ends LFM3's chat turns, <|im_end|>.
static constexpr int32_t TURN_END = 124900;

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-engine_decide CHECKPOINT");
  check(eremite_init(&(eremite_process){.abi = EREMITE_ABI}) == 0,
        "the process starts");
  eremite_config bad[] = {
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .deepseek.draft = 1},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .lookup = 3},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .deepseek.ced = 128},
      {.abi = EREMITE_ABI,
       .model = argv[1],
       .ctx = 4096,
       .deepseek.packed_state_rows = 1}};
  int refused = 0;
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
    refused += !eremite_open(&bad[i]);
  check(refused == 4, "a decision model refuses drafts, lookup, ced and "
                      "packed state rows");
  eremite_model *m = eremite_open(
      &(eremite_config){.abi = EREMITE_ABI, .model = argv[1], .ctx = 128000});
  check(m && eremite_is_decision_model(m), "d1-2B opens as a decision model");
  if (!m) return 1;
  check(eremite_max_rows(m) == 0 && eremite_max_verify(m) == 0 &&
            eremite_drafts_per_pass(m) == 0 && eremite_rewind_margin(m) == 0 &&
            eremite_vocab_size(m) == 128000 &&
            !eremite_is_stop_token(m, TURN_END),
        "its decoding limits are 0, its vocabulary 128,000 tokens, and no "
        "token ends a reply");
  check(eremite_load(m) && eremite_is_loaded(m), "it loads");

  int32_t question_ends[] = {ROWS};
  const float *direct = eremite_decide(m, tokens, ROWS, STATE, question_ends, 1,
                                       rows, ids, LOGITS);
  check(direct != NULL, "a decision runs");
  if (!direct) return 1;
  float first[LOGITS];
  memcpy(first, direct, sizeof first);
  double difference = 0;
  for (int i = 0; i < LOGITS; i++)
    difference = fmax(difference, fabs(first[i] - reference[i]));
  printf("the logits differ from FP32 by %.4f\n", difference);
  check(difference <= BOUND, "its logits are as close to FP32 as LiquidAI's "
                             "BF16 run");

  eremite_stream *x = eremite_stream_new(m);
  eremite_stream_decide(x, tokens, ROWS, STATE, question_ends, 1, rows, ids,
                        LOGITS);
  int finished = eremite_run_streams(&x, 1);
  const float *streamed = eremite_stream_logits(x);
  check(finished == 0 && streamed && !memcmp(streamed, first, sizeof first),
        "the same decision on a stream gives the same bits");
  eremite_stream_free(x);

  eremite_stats stats;
  eremite_read_stats(m, &stats);
  check(stats.prefill_rows == 2 * ROWS && stats.dense_bytes > 0 &&
            stats.required_bytes > stats.dense_bytes,
        "the counters count both decisions' rows, the weights, and the "
        "memory a full block's pass needs");

  fflush(stdout);
  pid_t child = fork();
  if (child == 0) {
    eremite_state_new(m);
    _exit(0);
  }
  int status = 0;
  waitpid(child, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
        "asking a decision model for a state ends the program with a "
        "message");
  return failures != 0;
}
