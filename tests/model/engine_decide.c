#include "check.h"
#include "bmp.h"
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
// which ends the program for it (in a child process). Decisions about images:
// each image's place in the tokens takes its span, the images are encoded, and
// the pass reads their rows there, for one image and for two in turn, as
// LiquidAI's own code does.
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

// The most images, questions, logits and expanded tokens of an image case.
static constexpr int MOST_IMAGES = 2, MOST_QUESTIONS = 3, MOST_LOGITS = 16,
                     MOST_EXPANDED = 256;

// A decision about images: its tokens, where each image's place holds the
// image token (<|image|>), its images (BMP test patterns of these sizes), its
// tokens and its state's rows and its questions' ends once the images' spans
// are in place, the rows and ids of its logits, their values in LiquidAI's own
// code in FP32, and the largest difference of its BF16 run from them.
typedef struct {
  const char *what;
  const int32_t *tokens;
  int n, images;
  int widths[MOST_IMAGES], heights[MOST_IMAGES];
  int expanded, state, questions;
  int32_t ends[MOST_QUESTIONS];
  int count;
  int32_t rows[MOST_LOGITS], ids[MOST_LOGITS];
  double logits[MOST_LOGITS], bound;
} ImageCase;

// A text state after a 400 x 208 image, and three questions: yes or no, a
// choice of three, and a score of three levels.
static const int32_t card[] = {
    124894, 124899, 5922,   207,   125009, 125030, 125010, 41,    1426,
    4559,   34,     3599,   376,   11775,  309,    267,    5322,  10590,
    6542,   20242,  64673,  63509, 833,    4439,   278,    7628,  8286,
    6542,   64929,  57922,  415,   18171,  432,    794,    1049,  22,
    124900, 207,    124899, 63514, 207,    124901, 124902, 63509, 833,
    28327,  3599,   355,    31211, 462,    278,    1583,   8527,  64929,
    13314,  833,    41,     2959,  207,    42,     4752,   207,   43,
    6542,   8,      57922,  415,   278,    7331,   4713,   1049,  22,
    124900, 207,    124899, 63514, 207,    124901, 124902, 63509, 833,
    3522,   34880,  355,    278,   7628,   64929,  24,     37787, 207,
    25,     15175,  207,    26,    9398,   8,      57922,  415,   267,
    2936,   19763,  229,    24,    21,     26,     1049,   22,    124900,
    207,    124899, 63514,  207,   124901, 124902};

// A text state after a 200 x 150 image and a 400 x 208 one, and a yes-or-no
// question.
static const int32_t two_images[] = {
    124894, 124899, 5922,  207,    125009, 125030, 125010, 125009,
    125030, 125010, 15279, 1426,   13092,  64673,  63509,  833,
    17464,  278,    1046,  11333,  278,    1542,   64929,  57922,
    415,    18171,  432,   794,    1049,   22,     124900, 207,
    124899, 63514,  207,   124901, 124902};

static const ImageCase image_cases[] = {
    {"one image and three questions",
     card,
     sizeof card / sizeof *card,
     1,
     {400},
     {208},
     185,
     94,
     3,
     {114, 150, 185},
     14,
     {113, 113, 113, 113, 113, 149, 149, 149, 149, 149, 149, 184, 184, 184},
     {11683, 12447, 2243, 4547, 19598, 41, 334, 42, 378, 43, 340, 24, 25, 26},
     {2.908662, 7.315761, 5.763090, 10.504194, 3.416532, 13.300821, 8.844008,
      7.259619, 2.441981, 8.001862, 3.481334, 6.407514, 7.811986, 12.584948},
     0.0583},
    {"two images in turn",
     two_images,
     sizeof two_images / sizeof *two_images,
     2,
     {200, 400},
     {150, 208},
     177,
     156,
     1,
     {177},
     5,
     {176, 176, 176, 176, 176},
     {11683, 12447, 2243, 4547, 19598},
     {6.193096, 10.140717, 4.268310, 8.358241, 2.249351},
     0.1082},
};

// Runs the case's decision on m, directly and on a stream, and checks its
// logits against the reference's.
static void check_image_case(eremite_model *m, const ImageCase *c) {
  printf("%s:\n", c->what);
  const uint8_t *files[MOST_IMAGES];
  size_t lens[MOST_IMAGES];
  for (int i = 0; i < c->images; i++)
    files[i] = make_test_bmp(c->widths[i], c->heights[i], &lens[i]);
  char message[256];
  eremite_media *media =
      eremite_media_open(m, files, lens, c->images, message, sizeof message);
  check(media != NULL, "its images open");
  if (!media) return;
  int32_t expanded[MOST_EXPANDED];
  int n = eremite_media_expand(media, c->tokens, c->n, expanded, MOST_EXPANDED);
  check(n == c->expanded,
        "each image's place takes a row for each square of its patches");
  check(eremite_media_encode(media, 0) == c->images, "its images are encoded");
  const float *direct =
      n == c->expanded
          ? eremite_decide(m, expanded, n, c->state, c->ends, c->questions,
                           c->rows, c->ids, c->count, media)
          : NULL;
  check(direct != NULL, "its decision runs");
  if (direct) {
    float first[MOST_LOGITS];
    memcpy(first, direct, (size_t)c->count * sizeof *first);
    double difference = 0;
    for (int i = 0; i < c->count; i++)
      difference = fmax(difference, fabs(first[i] - c->logits[i]));
    printf("the logits differ from FP32 by %.4f\n", difference);
    check(difference <= c->bound,
          "its logits are as close to FP32 as LiquidAI's BF16 run");
    eremite_stream *x = eremite_stream_new(m);
    eremite_stream_decide(x, expanded, n, c->state, c->ends, c->questions,
                          c->rows, c->ids, c->count, media);
    int finished = eremite_run_streams(&x, 1);
    const float *streamed = eremite_stream_logits(x);
    check(finished == 0 && streamed &&
              !memcmp(streamed, first, (size_t)c->count * sizeof *first),
          "the same decision on a stream gives the same bits");
    eremite_stream_free(x);
  }
  eremite_media_close(media);
  for (int i = 0; i < c->images; i++) free((void *)files[i]);
}

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
                                       rows, ids, LOGITS, NULL);
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
                        LOGITS, NULL);
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

  for (size_t i = 0; i < sizeof image_cases / sizeof *image_cases; i++)
    check_image_case(m, &image_cases[i]);

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
