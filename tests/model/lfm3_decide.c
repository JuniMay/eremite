#include "check.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "engine/memory.h"
#include "engine/tokenizer.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/lfm3/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests d1-2B's decision pass on its checkpoint against LiquidAI's own code
// in FP32 (transformers 5.19.0 and torch 2.15 on the CPU): the logits of the
// answer tokens at each question's last row, and the probabilities that d1
// reads from them. Each case's bound is how far LiquidAI's code in BF16 is
// from FP32 on the same case. The cases: a JSON state with one question, a
// text state with three, and a state of 10,806 tokens, which the pass takes
// in three blocks of rows; the test tokenizes that one itself.
//
// usage: model-lfm3_decide CHECKPOINT

// The most answer tokens of a case.
static constexpr int MOST_TOKENS = 16;

// A case: its sequence, the answer tokens' rows, ids, reference logits and
// option (the question's options in order, each its tokens' best logit), and
// the reference's probabilities, option by option. The bounds are LiquidAI's
// BF16 run's largest differences from FP32 on the case.
typedef struct {
  const char *what;
  const int *tokens;
  int n, state, questions;
  int ends[4];
  int count;
  int rows[MOST_TOKENS], ids[MOST_TOKENS], options[MOST_TOKENS];
  double logits[MOST_TOKENS], probabilities[MOST_TOKENS];
  double logit_bound, probability_bound;
} Case;

static const int json_state[] = {
    124894, 124899, 5922,   207,   2718,   229,   496,   99489,  6380,  827,
    360,    496,    6929,   6380,  496,    16308, 52966, 278,    48505, 7285,
    1917,   229,    10016,  229,   496,    3166,  6380,  827,    360,   496,
    16414,  273,    6380,   19843, 733,    229,   3255,  920,    733,   229,
    3255,   22,     29,     207,   360,    40439, 360,   496,    37507, 6380,
    2846,   920,    360,    496,   14700,  6380,  3495,  207,    229,   1130,
    64024,  63509,  833,    29088, 278,    29936, 7986,  267,    48505, 7285,
    7450,   467,    278,    2064,  64929,  57922, 415,   18171,  432,   794,
    1049,   22,     124900, 207,   124899, 63514, 207,   124901, 124902};

static const int three_questions[] = {
    124894, 124899, 5922,   207,   49,     452,    14917,  10625, 309,
    1226,   60525,  2966,   10922, 22,     13330,  52966,  734,   7285,
    64673,  63509,  833,    4439,  278,    13107,  13564,  374,   267,
    52966,  64929,  57922,  415,   18171,  432,    794,    1049,  22,
    124900, 207,    124899, 63514, 207,    124901, 124902, 63509, 833,
    28327,  2909,   1252,   9435,  532,    64929,  13314,  833,   41,
    120315, 20,     52966,  91,    20,     60326,  1363,   207,   42,
    57615,  309,    18830,  1205,  207,    43,     4634,   432,   2974,
    49521,  8,      57922,  415,   278,    7331,   4713,   1049,  22,
    124900, 207,    124899, 63514, 207,    124901, 124902, 63509, 833,
    3522,   30648,  355,    532,   29936,  64929,  24,     2419,  207,
    25,     8057,   207,    26,    1068,   8,      57922,  415,   267,
    2936,   19763,  229,    24,    21,     26,     1049,   22,    124900,
    207,    124899, 63514,  207,   124901, 124902};

// The tokens of yes and no, each written three ways, of the option codes A,
// B and C plain and after a space, and of the digits 0 to 2.
enum {
  YES = 11683,
  CAPITAL_YES = 12447,
  NO = 2243,
  CAPITAL_NO = 4547,
  UPPER_NO = 19598,
  CODE_A = 41,
  SPACED_A = 334,
  CODE_B = 42,
  SPACED_B = 378,
  CODE_C = 43,
  SPACED_C = 340,
  DIGIT_0 = 24,
  DIGIT_1 = 25,
  DIGIT_2 = 26
};

static Case cases[] = {
    {"a JSON state and one question",
     json_state,
     89,
     63,
     1,
     {89},
     5,
     {88, 88, 88, 88, 88},
     {YES, CAPITAL_YES, NO, CAPITAL_NO, UPPER_NO},
     {0, 0, 1, 1, 1},
     {8.003815, 10.504749, 5.083753, 7.219336, 1.851632},
     {0.921041, 0.078959},
     0.0619,
     0.0037},
    {"a text state and three questions",
     three_questions,
     123,
     21,
     3,
     {43, 88, 123},
     14,
     {42, 42, 42, 42, 42, 87, 87, 87, 87, 87, 87, 122, 122, 122},
     {YES, CAPITAL_YES, NO, CAPITAL_NO, UPPER_NO, CODE_A, SPACED_A, CODE_B,
      SPACED_B, CODE_C, SPACED_C, DIGIT_0, DIGIT_1, DIGIT_2},
     {0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2, 0, 1, 2},
     {10.752959, 15.612402, 3.119531, 7.634193, 1.623971, 11.242476, 7.2931,
      9.904803, 5.925263, 7.652206, 3.69891, 10.271042, 12.042607, 10.583084},
     {0.997441, 0.002559, 0.696351, 0.256121, 0.047529, 0.166014, 0.624347,
      0.20964},
     0.0665,
     0.00485},
};

static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Returns calibration.json's default temperature, which d1 divides its
// option logits by.
static double read_temperature(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/calibration.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *t = json_get(root, "default");
  if (!t || t->type != J_NUM) die("%s: no default temperature", path);
  double temperature = t->num;
  json_free(root);
  free(text);
  return temperature;
}

// Runs the case's pass, and checks its logits, and the probabilities that d1
// reads from them, against the reference's.
static void run_case(const Model *m, const Case *c, double temperature) {
  const float *got =
      m->ops->decide(c->tokens, c->n, c->state, c->ends, c->questions, c->rows,
                     c->ids, c->count, NULL);
  if (!got) die("%s: the pass did not fit in memory", c->what);
  double logit_difference = 0, probability_difference = 0;
  for (int i = 0; i < c->count; i++)
    logit_difference = fmax(logit_difference, fabs(got[i] - c->logits[i]));
  // Each question's options: its tokens' best logit over the temperature,
  // then a softmax.
  for (int i = 0, option = 0; i < c->count;) {
    int j = i;
    while (j < c->count && c->rows[j] == c->rows[i]) j++;
    double best[MOST_TOKENS], most = -INFINITY, sum = 0;
    int options = c->options[j - 1] + 1;
    for (int o = 0; o < options; o++) best[o] = -INFINITY;
    for (int k = i; k < j; k++)
      best[c->options[k]] = fmax(best[c->options[k]], got[k] / temperature);
    for (int o = 0; o < options; o++) most = fmax(most, best[o]);
    for (int o = 0; o < options; o++) sum += exp(best[o] - most);
    for (int o = 0; o < options; o++, option++)
      probability_difference =
          fmax(probability_difference,
               fabs(exp(best[o] - most) / sum - c->probabilities[option]));
    i = j;
  }
  printf("%s: logits differ by %.4f, probabilities by %.5f\n", c->what,
         logit_difference, probability_difference);
  char what[256];
  snprintf(what, sizeof what,
           "%s: the logits and probabilities are as close to FP32 as "
           "LiquidAI's BF16 run",
           c->what);
  check(logit_difference <= c->logit_bound &&
            probability_difference <= c->probability_bound,
        what);
}

// The long case's state: 600 numbered lines of parcels and days.
static int write_long_state(char *out, size_t room) {
  size_t at = 0;
  for (int i = 0; i < 600 && at < room; i++)
    at += (size_t)snprintf(
        out + at, room - at,
        "Line %d: the parcel number %d left the depot on day %d.\n", i,
        i * 7 % 1000, i % 31 + 1);
  return (int)at;
}

// The long case: a state of 10,806 tokens that the test tokenizes as d1's
// prompt holds it, and two questions, a yes or no and a score.
static void run_long_case(const Model *m, const char *dir, double temperature) {
  static char state[1 << 16], first[1 << 17];
  static int tokens[1 << 15];
  write_long_state(state, sizeof state);
  const char *head = "<|startoftext|><|im_start|>user\n",
             *question = "Did any parcel leave the depot on day 31?\n\nReply "
                         "with yes or no only.",
             *turn = "<|im_end|>\n<|im_start|>assistant\n<think></think>",
             *second = "QUESTION:\nHow many lines are there?\n\n0 few\n1 "
                       "some\n2 many\n\nReply with a single digit 0-2 only.";
  snprintf(first, sizeof first, "%s%s\n\n\nQUESTION:\n%s%s", head, state,
           question, turn);
  Tokenizer *t = tokenizer_open(dir, NULL);
  int n1 = tokenizer_encode_with_added(t, first, tokens, 1 << 15);
  char rest[512];
  snprintf(rest, sizeof rest, "%s%s", second, turn);
  int n =
      n1 + tokenizer_encode_with_added(t, rest, tokens + n1, (1 << 15) - n1);
  char suffix[512];
  snprintf(suffix, sizeof suffix, "%s%s", question, turn);
  int suffix_tokens[256];
  int state_rows =
      n1 - tokenizer_encode_with_added(t, suffix, suffix_tokens, 256);
  check(n == 10867 && state_rows == 10806,
        "the long case's prompt takes the reference's 10,867 tokens, 10,806 "
        "of them the state's");
  Case c = {
      "a state of 10,806 tokens and two questions",
      tokens,
      n,
      state_rows,
      2,
      {n1, n},
      8,
      {n1 - 1, n1 - 1, n1 - 1, n1 - 1, n1 - 1, n - 1, n - 1, n - 1},
      {YES, CAPITAL_YES, NO, CAPITAL_NO, UPPER_NO, DIGIT_0, DIGIT_1, DIGIT_2},
      {0, 0, 1, 1, 1, 0, 1, 2},
      {6.883955, 13.150933, 3.107337, 9.147763, 2.858903, 2.619674, 5.131567,
       7.250163},
      {0.952268, 0.047732, 0.02536, 0.165897, 0.808744},
      0.0314,
      0.00259};
  run_case(m, &c, temperature);
  tokenizer_close(t);
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-lfm3_decide CHECKPOINT");
  Memory *memory = memory_new();
  size_t length;
  const char *source = kernel_library_source(&length);
  Gpu *gpu = gpu_new(source, length);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  Options options = {
      .ctx = LFM3_POSITIONS, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *m = model_open(memory, gpu, argv[1], &options);
  if (!m) die("%s: not loaded", argv[1]);
  ModelCounters counters;
  m->ops->read_counters(&counters);
  check(counters.required_bytes > counters.dense_bytes,
        "a pass over a full block of rows needs memory beyond the weights");
  double temperature = read_temperature(argv[1]);
  for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
    run_case(m, &cases[i], temperature);
  run_long_case(m, argv[1], temperature);
  return failures != 0;
}
