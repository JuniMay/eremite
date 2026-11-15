#include "model/deepseek/model.h"
#include "model/deepseek/model.c"
#include "load.h"
#include <math.h>

// Tests that passes of a model interleaved at every yield each compute what
// they compute alone, byte for byte: every logit and, with drafts, every
// proposed token. As the engine does, the model is told how many other
// passes run beside each one. Two prompts are prefilled and decoded greedily
// alone, then together, each on its own stack with its own buffers and signals.
// For DeepSeek, whose passes share some work buffers, those buffers are filled
// with garbage at every switch: nothing a pass needs may be left in them
// across a wait. Each prefill also yields once in each layer, where the engine
// lets decoding calls go first.
//
// usage: model-passes CHECKPOINT --text FILE [--experts GGUF] [--memory GIB]
//        [--tokens N] [--decode N] [--rows N] [--draft N] [--no-garbage]
//        [--release] [--phases] [--verbose]

enum { MAX_DRAFTS = 15, MAX_ROWS = 8 };

static int use_garbage = 1, drafts_wanted, vocab, rows_each = 1,
           release_between, phases;
static uint64_t garbage_filled;
static const ModelOps *ops;
// What the test blocks on while no pass can continue. It lives as long as the
// process: reads may still signal it after their pass has ended.
static ThreadsEvent *wake;

// As the server's streams do between calls.
static void between_calls(void) {
  if (release_between && ops->pass_release) ops->pass_release();
}

// A job's sequences: its first prompt, and each further one later in the
// text and a little shorter, so that they decode at different positions.
typedef struct {
  const int *tokens;
  int n, decode;
  ModelState *state[MAX_ROWS];
  uint64_t digest, proposals;
  int resumes, last, accepted;
  int pos[MAX_ROWS], token[MAX_ROWS];  // what the decoding continues from
  // What the job's prefills wait for in each layer, and whether it was looked
  // at once since it was last over.
  PassWait layer_wait;
  int looked;
} Job;

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = data;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static int best(const float *logits) {
  int b = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[b]) b = i;
  return b;
}

// Whether this is the second look since the wait was last over. The first
// look yields.
static int is_second_look(void *context) {
  int *looked = context;
  *looked = !*looked;
  return !*looked;
}

static void job_prefill(Job *j) {
  j->digest = 14695981039346656037ull;
  j->looked = 0;
  j->layer_wait =
      (PassWait){is_second_look, NULL, &j->looked, PASS_WAIT_OTHER, NULL};
  for (int r = 0; r < rows_each; r++) {
    j->pos[r] = j->n - r * 17;
    const float *logits =
        ops->prefill(j->state[r], j->tokens + r * 211, j->pos[r], 0, 0, -1, 0,
                     0, &j->layer_wait);
    if (!logits) die("the prefill did not fit in memory");
    between_calls();
    j->digest = fnv(j->digest, logits, (size_t)vocab * 4);
    j->token[r] = best(logits);
  }
}

static void job_decode(Job *j) {
  const float *logits;
  int *pos = j->pos, *token = j->token;
  Sampler greedy = {.temperature = 0, .top_p = 1};
  for (int done = 0; done < j->decode;) {
    if (drafts_wanted) {
      int rows[MAX_DRAFTS + 1] = {token[0]}, kept;
      ops->propose_drafts(j->state[0], token[0], pos[0], rows + 1,
                          drafts_wanted, &greedy);
      j->proposals =
          fnv(j->proposals, rows + 1, (size_t)drafts_wanted * sizeof *rows);
      logits = ops->verify(j->state[0], rows, drafts_wanted + 1, pos[0]);
      int next = sample_verify_sampled_drafts(
          logits, vocab, rows + 1, drafts_wanted, NULL, ops->is_stop_token,
          &greedy, &kept);
      ops->accept(j->state[0], rows, kept + 1, pos[0]);
      between_calls();
      j->digest = fnv(j->digest, logits, (size_t)(kept + 1) * vocab * 4);
      j->accepted += kept;
      done += kept + 1, pos[0] += kept + 1;
      logits += (size_t)kept * vocab;
      token[0] = next;
    } else {
      logits = ops->decode_rows(j->state, token, pos, rows_each);
      if (!logits) die("a decoded token did not fit in memory");
      between_calls();
      j->digest = fnv(j->digest, logits, (size_t)rows_each * vocab * 4);
      for (int r = 0; r < rows_each; r++)
        pos[r]++, token[r] = best(logits + (size_t)r * vocab);
      done++;
    }
  }
  j->last = token[0];
}

static void job_body(void *argument) {
  ops->pass_begin();
  job_prefill(argument);
  job_decode(argument);
  ops->pass_end();
}

static void prefill_body(void *argument) {
  ops->pass_begin();
  job_prefill(argument);
  ops->pass_end();
}

static void decode_body(void *argument) {
  ops->pass_begin();
  job_decode(argument);
  ops->pass_end();
}

// Between two passes' turns, once queued work has run, the buffers that
// DeepSeek's passes share hold nothing that a pass needs.
static void garbage(void) {
  gpu_sync(gpu);
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) {
    Buf *b = work_buffers[i];
    if (is_pass_work(b) || !b->obj) continue;
    size_t n = b->size < (1u << 20) ? b->size : (1u << 20);
    memset(b->ptr, 0xAB, n);
    garbage_filled += n;
  }
}

// Runs the jobs' passes, resuming each as soon as it can continue; alone
// (count 1) each waits as a single pass does.
static void interleave(Job **jobs, void (*const *bodies)(void *), int count) {
  Pass *passes[4];
  int ended[4] = {0}, alive = count;
  for (int i = 0; i < count; i++) passes[i] = ops->pass_new(bodies[i], jobs[i]);
  // As the engine tells the model before it runs passes together.
  if (ops->set_other_passes) ops->set_other_passes(count - 1);
  if (count == 1) {
    while (!pass_resume(passes[0])) jobs[0]->resumes++, pass_block(passes[0]);
    jobs[0]->resumes++;
    ended[0] = 1, alive = 0;
  }
  while (alive) {
    int ran = 0;
    for (int i = 0; i < count; i++) {
      if (ended[i] || !pass_ready(passes[i])) continue;
      if (use_garbage && ops == &deepseek_ops) garbage();
      jobs[i]->resumes++;
      ended[i] = pass_resume(passes[i]);
      alive -= ended[i];
      ran = 1;
    }
    if (ran || !alive) continue;
    Pass *waiting[4];
    int n = 0;
    for (int i = 0; i < count; i++)
      if (!ended[i]) waiting[n++] = passes[i];
    if (!pass_wait_for_any(waiting, n, wake))
      die("the passes wait for each other");
  }
  for (int i = 0; i < count; i++) pass_free(passes[i]);
}

static void states_new(Job *jobs, int count) {
  for (int i = 0; i < count; i++) {
    for (int r = 0; r < rows_each; r++) jobs[i].state[r] = ops->state_new();
    jobs[i].resumes = 0, jobs[i].accepted = 0, jobs[i].proposals = 0;
  }
}

static void states_free(Job *jobs, int count) {
  for (int i = 0; i < count; i++)
    for (int r = 0; r < rows_each; r++) ops->state_free(jobs[i].state[r]);
}

static void run(Job *jobs, int count) {
  states_new(jobs, count);
  Job *list[2] = {&jobs[0], count > 1 ? &jobs[1] : NULL};
  void (*const bodies[2])(void *) = {job_body, job_body};
  interleave(list, bodies, count);
  states_free(jobs, count);
}

// As the server runs two requests: the first prompt in a pass of its own, the
// second prompt beside the first request's decoding, each in a new pass, and
// the second request's decoding after them.
static void run_phased(Job *jobs) {
  states_new(jobs, 2);
  void (*const prefill_only[1])(void *) = {prefill_body},
                                     (*const decode_only[1])(void *) = {
                                         decode_body};
  void (*const both[2])(void *) = {prefill_body, decode_body};
  Job *first[1] = {&jobs[0]}, *second[1] = {&jobs[1]},
      *pair[2] = {&jobs[1], &jobs[0]};
  interleave(first, prefill_only, 1);
  interleave(pair, both, 2);
  interleave(second, decode_only, 1);
  states_free(jobs, 2);
}

int main(int argc, char **argv) {
  if (argc < 4)
    die("usage: model-passes CHECKPOINT --text FILE [--experts GGUF] [--memory "
        "GIB] [--tokens N] [--decode N] "
        "[--rows N] [--draft N] [--no-garbage] [--release] [--phases] "
        "[--verbose]");
  double memory_gib = 40;
  wake = threads_event_new();
  Options o = {.ctx = 32768, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  const char *text = 0;
  int tokens_each = 600, decode = 16;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--text") && i + 1 < argc)
      text = argv[++i];
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc)
      o.deepseek.experts = argv[++i];
    else if (!strcmp(argv[i], "--memory") && i + 1 < argc)
      memory_gib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc)
      tokens_each = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--decode") && i + 1 < argc)
      decode = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc)
      o.draft = drafts_wanted = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rows") && i + 1 < argc)
      rows_each = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-garbage"))
      use_garbage = 0;
    else if (!strcmp(argv[i], "--release"))
      release_between = 1;
    else if (!strcmp(argv[i], "--phases"))
      phases = 1;
    else if (!strcmp(argv[i], "--verbose"))
      o.quiet = 0;
    else
      die("invalid argument: %s", argv[i]);
  }
  if (!text) die("--text FILE is required");
  if (drafts_wanted > MAX_DRAFTS) die("--draft takes at most %d", MAX_DRAFTS);
  if (rows_each < 1 || rows_each > MAX_ROWS || (drafts_wanted && rows_each > 1))
    die("--rows takes 1 to %d, and 1 with drafts", MAX_ROWS);
  ops = model_find_family(argv[1]);
  Tokenizer *tk = tokenizer_open(argv[1], ops->media_token);
  size_t size;
  char *t = read_file(text, &size);
  int *tokens = xmalloc((size + 2) * sizeof *tokens);
  int count = tokenizer_encode_with_added(tk, t, tokens, (int)size + 2);
  free(t);
  int needed = 2 * tokens_each + rows_each * 211;
  if (count < needed)
    die("the text has %d tokens; %d are needed", count, needed);
  Memory *memory = memory_new();
  expert_store_set_memory(
      memory, (MemorySettings){.model_gib = memory_gib, .reserve_gib = 4});
  if (!model_open(memory, open_gpu(memory), argv[1], &o))
    die("the model did not load");
  vocab = ops->vocab_size();

  Job alone[2], together[2];
  for (int i = 0; i < 2; i++) {
    alone[i] = (Job){.tokens = tokens + (size_t)i * tokens_each,
                     .n = tokens_each - i * tokens_each / 3,
                     .decode = decode};
    together[i] = alone[i];
  }
  // An idle run first: the expert cache and the GPU's clock start warm for
  // every timed run below.
  run(&alone[0], 1);
  double t0 = now(), busy0 = gpu_times(gpu).busy_seconds;
  run(&alone[0], 1);
  double a = now() - t0, a_busy = gpu_times(gpu).busy_seconds - busy0;
  t0 = now(), busy0 = gpu_times(gpu).busy_seconds;
  run(&alone[1], 1);
  double b = now() - t0, b_busy = gpu_times(gpu).busy_seconds - busy0;
  t0 = now(), busy0 = gpu_times(gpu).busy_seconds;
  ModelCounters before, after;
  ops->read_counters(&before);
  if (phases)
    run_phased(together);
  else
    run(together, 2);
  ops->read_counters(&after);
  double both = now() - t0, both_busy = gpu_times(gpu).busy_seconds - busy0;
  printf("alone: %.2f s and %.2f s (GPU busy %.2f and %.2f s), one after the "
         "other %.2f s; interleaved %.2f s (GPU "
         "busy %.2f s)\n",
         a, b, a_busy, b_busy, a + b, both, both_busy);
  printf("while interleaved: %llu experts evicted\n",
         (unsigned long long)(after.cache.evictions - before.cache.evictions));
  int failures = 0;
  for (int i = 0; i < 2; i++) {
    printf("pass %d: %d prompt tokens, %d decoded", i, alone[i].n, decode);
    if (drafts_wanted)
      printf(" (%d drafts accepted alone, %d together; proposals %016llx and "
             "%016llx)",
             alone[i].accepted, together[i].accepted,
             (unsigned long long)alone[i].proposals,
             (unsigned long long)together[i].proposals);
    printf("; alone %016llx in %d resumes, together %016llx in %d resumes; "
           "last token %d\n",
           (unsigned long long)alone[i].digest, alone[i].resumes,
           (unsigned long long)together[i].digest, together[i].resumes,
           alone[i].last);
    if (alone[i].digest != together[i].digest ||
        alone[i].last != together[i].last ||
        alone[i].proposals != together[i].proposals) {
      printf("FAIL: pass %d's logits differ when interleaved\n", i);
      failures++;
    }
  }
  if (use_garbage && ops == &deepseek_ops)
    printf("garbage written between passes: %.1f GiB\n",
           (double)garbage_filled / (1u << 30));
  printf("%s\n",
         failures ? "interleaved passes: FAILED"
                  : "interleaved passes: every logit equal to the pass alone");
  return failures != 0;
}
