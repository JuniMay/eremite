#include "base/clock.h"
#include "base/error.h"
#include "base/file.h"
#include "base/shutdown.h"
#include "engine/draft_choice.h"
#include "engine/expert_store.h"
#include "engine/lookup.h"
#include "engine/media.h"
#include "engine/sampling.h"
#include "engine/tokenizer.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/deepseek/model.h"
#include "model/model.h"
#include "platform/system_memory.h"
#include "platform/threads.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// The command line: generates text with a checkpoint that the engine runs, or
// runs token ids for checks.

static void usage(void) {
  fputs(
      "usage: eremite -m MODEL [-p PROMPT | -f FILE | --tokens IDS] [options]\n"
      "  -m MODEL      a supported model checkpoint directory\n"
      "  -n N          tokens to generate (default 128)\n"
      "  -t T, --temperature T  sampling temperature, 0 = greedy (default 0)\n"
      "  --top-p P     nucleus probability in [0, 1] (default 1, full "
      "distribution)\n"
      "  --seed N      nonnegative integer sampling seed\n"
      "  --raw         use the prompt as is, without the chat template\n"
      "  --rendered    use a server-rendered chat prompt; preserve literal "
      "image-marker text\n"
      "  --think N     thinking effort 1-100; 0 disables it (default off)\n"
      "  --image FILE  an image (any format macOS reads) before the prompt's "
      "text, up to 16; with --raw\n"
      "                each takes a <｜deepseek_image｜> of the prompt "
      "instead "
      "(models with a vision encoder)\n"
      "  --features-fd FD  write encoded BF16 feature rows to FD and exit\n"
      "  --ctx N       context capacity in tokens, at most 1048576 (default "
      "16384)\n"
      "  --prefill-prune  skip decoder batches outside required history "
      "(default off); CED and\n"
      "                substitution must be off\n"
      "  --ced off|N   approximate prefill: replay N decoder tokens, "
      "128-1048576 (default off);\n"
      "                keeps the verification rewind margin; decoding uses all "
      "layers\n"
      "  --memory GB   upper limit of model memory, state memory included "
      "(default: none; the free\n"
      "                memory decides)\n"
      "  --reserve GB  memory kept available to other programs, 2-1024 "
      "(default 10)\n"
      "  --tokens IDS  comma-separated token ids, or @FILE (commas, spaces or "
      "newlines between them); run\n"
      "                without generation\n"
      "  --dump-fd FD  write every layer's streams and the logits to FD\n"
      "  --routes-fd FD  write every decoded token's selected experts, "
      "predictions and router logits to FD\n"
      "                (per layer, int32: 6 selected, the top 16 predicted for "
      "each of the next two layers,\n"
      "                the top 16 of this layer predicted from the residual "
      "before attention; 384 float32)\n"
      "  --tokenize    print the prompt's token ids and exit\n"
      "  --bench       time the kernels on the model's matrices\n"
      "  --check-prefill  compare the prompt's batched prefill with "
      "token-by-token decoding\n"
      "  --check-batch  compare a token decoded alone with the same token "
      "decoded beside another sequence\n"
      "                (and, with --draft, in a verify pass): the logits must "
      "be equal\n"
      "  --draft N     at most N DSpark proposals per pass, 0-5; each pass "
      "runs the count that has given the\n"
      "                most tokens per second\n"
      "  --lookup off|N  verify up to N (1-15) context-copy drafts; --draft "
      "supplies fallback (default off)\n"
      "  --substitute D  for tests only (the server does not offer it); "
      "approximate: when decoding, replace\n"
      "                selected experts that are not in memory by the resident "
      "experts with the highest selection\n"
      "                keys at most D below theirs, the lowest-ranked first, "
      "never the two with the highest scores\n"
      "  --experts GGUF  read the routed experts from a GGUF file, or a set of "
      "shards given by one\n"
      "                (llama.cpp's layout, e.g. unsloth's UD-IQ2_XXS); "
      "approximate: smaller codes\n"
      "  --perplexity  decode the text token by token; print the perplexity of "
      "each next token\n"
      "  --prefill N   with --perplexity: prefill the text's first N tokens in "
      "one pass, then score the rest\n"
      "  --logits-fd FD  with --perplexity: write the logits (the vocabulary's "
      "float32) each token is\n"
      "                scored on to FD\n",
      stderr);
  exit(2);
}

static Model *model;

// A prefill's logits. Null means quitting stopped the prefill, and the run
// ends, or its working memory did not fit.
static const float *prefilled(const float *logits) {
  if (logits) return logits;
  if (shutdown_requested()) exit(0);
  die("the prompt's working memory does not fit the free memory less "
      "--reserve, or --memory");
}

static double sampling_option(const char *name, const char *value,
                              double upper) {
  char *end;
  errno = 0;
  double n = strtod(value, &end);
  if (end == value || *end || errno || !isfinite(n) || n < 0 || n > upper)
    die("%s must be finite and in %s", name,
        isfinite(upper) ? "[0, 1]" : "[0, infinity)");
  return n;
}

// Whether the model reads routed experts through the expert cache.
static ModelCounters counters(void) {
  ModelCounters c;
  model->ops->read_counters(&c);
  return c;
}

static int cached(void) { return counters().experts_per_row > 0; }

static uint64_t model_cache_bytes(void) {
  return model->ops->expert_store_bytes ? model->ops->expert_store_bytes() : 0;
}

// The Memory object's eviction callback.
static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Makes the memory and the GPU, sets the memory's limits, and opens the model.
static Model *open_model(const char *dir, const Options *o, double memory_gb,
                         double reserve_gb) {
  Memory *memory = memory_new();
  expert_store_set_memory(memory, (MemorySettings){.model_gib = memory_gb,
                                                   .reserve_gib = reserve_gb});
  size_t length;
  const char *source = kernel_library_source(&length);
  Gpu *gpu = gpu_new(source, length);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  return model_open(memory, gpu, dir, o);
}

int main(int argc, char **argv) {
  // Interactive work: the scheduler and the GPU's arbitration favor it over
  // background processes.
  threads_set_priority(THREADS_INTERACTIVE);
  const char *dir = 0, *prompt = 0, *ids = 0;
  int ntokens = 128, raw = 0, think = 0, tokenize = 0, bench = 0, check = 0,
      check_batch = 0, perplexity = 0, prefill = 1, logits_fd = -1;
  double temp = 0, top_p = 1;
  uint64_t seed = 0;
  int seed_set = 0;
  double memory_gb = 0, reserve_gb = 10;
  Options o = {.ctx = 16384, .dump_fd = -1, .routes_fd = -1};
  const char *media_path[16];
  int nmedia = 0, vision_fd = -1;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : 0;
    if (!strcmp(a, "--raw")) {
      raw = 1;
      continue;
    }
    if (!strcmp(a, "--rendered")) {
      raw = 2;
      continue;
    }
    if (!strcmp(a, "--tokenize")) {
      tokenize = 1;
      if (!raw) raw = 1;
      continue;
    }
    if (!strcmp(a, "--bench")) {
      bench = 1;
      continue;
    }
    if (!strcmp(a, "--check-prefill")) {
      check = 1;
      continue;
    }
    if (!strcmp(a, "--check-batch")) {
      check_batch = 1;
      continue;
    }
    if (!strcmp(a, "--perplexity")) {
      perplexity = 1;
      if (!raw) raw = 1;
      continue;
    }
    if (!strcmp(a, "--prefill-prune")) {
      o.deepseek.prefill_prune = 1;
      continue;
    }
    if (!v) usage();
    i++;
    if (!strcmp(a, "-m"))
      dir = v;
    else if (!strcmp(a, "-p"))
      prompt = v;
    else if (!strcmp(a, "-f"))
      prompt = read_file(v, 0);
    else if (!strcmp(a, "-n"))
      ntokens = atoi(v);
    else if (!strcmp(a, "-t") || !strcmp(a, "--temperature"))
      temp = sampling_option("temperature", v, INFINITY);
    else if (!strcmp(a, "--top-p"))
      top_p = sampling_option("top-p", v, 1);
    else if (!strcmp(a, "--seed")) {
      char *end;
      errno = 0;
      unsigned long long n = strtoull(v, &end, 10);
      if (*v < '0' || *v > '9' || *end || errno)
        die("--seed needs a nonnegative integer");
      seed = n;
      seed_set = 1;
    } else if (!strcmp(a, "--think")) {
      char *end;
      long n = strtol(v, &end, 10);
      if (end == v || *end || n < 0 || n > 100)
        die("--think takes 0 (off) or 1 to 100");
      think = (int)n;
    } else if (!strcmp(a, "--ced")) {
      char *end;
      errno = 0;
      long n = !strcmp(v, "off") ? 0 : strtol(v, &end, 10);
      if (strcmp(v, "off") &&
          (end == v || *end || errno || n < WINDOW || n > 1 << 20))
        die("--ced takes off or 128 to 1048576 replay tokens");
      o.deepseek.ced = (int)n;
    } else if (!strcmp(a, "--ctx"))
      o.ctx = atoi(v);
    else if (!strcmp(a, "--memory"))
      memory_gb = sampling_option("memory", v, 1024);
    else if (!strcmp(a, "--reserve")) {
      char *end;
      double gib = strtod(v, &end);
      if (end == v || *end ||
          !(gib >= EXPERT_STORE_RESERVE_GIB_MIN &&
            gib <= EXPERT_STORE_RESERVE_GIB_MAX))
        die("--reserve takes %d to %d GiB", EXPERT_STORE_RESERVE_GIB_MIN,
            EXPERT_STORE_RESERVE_GIB_MAX);
      reserve_gb = gib;
    } else if (!strcmp(a, "--tokens"))
      ids = v;
    else if (!strcmp(a, "--dump-fd"))
      o.dump_fd = atoi(v);
    else if (!strcmp(a, "--routes-fd"))
      o.routes_fd = atoi(v);
    else if (!strcmp(a, "--draft"))
      o.draft = atoi(v);
    else if (!strcmp(a, "--lookup")) {
      char *end;
      errno = 0;
      long value = !strcmp(v, "off") ? 0 : strtol(v, &end, 10);
      if (strcmp(v, "off") &&
          (end == v || *end || errno || value < 0 || value > VERIFY_DRAFTS))
        die("--lookup takes off or 1 to %d drafts", VERIFY_DRAFTS);
      o.lookup = (int)value;
    } else if (!strcmp(a, "--substitute"))
      o.deepseek.substitute = atof(v);
    else if (!strcmp(a, "--experts"))
      o.deepseek.experts = v;
    else if (!strcmp(a, "--prefill"))
      prefill = atoi(v);
    else if (!strcmp(a, "--logits-fd"))
      logits_fd = atoi(v);
    else if (!strcmp(a, "--image")) {
      if (nmedia == 16) die("at most 16 media inputs");
      media_path[nmedia++] = v;
    } else if (!strcmp(a, "--features-fd"))
      vision_fd = atoi(v);
    else
      usage();
  }
  if (vision_fd >= 0 && !prompt) prompt = "";
  if (!dir || (!prompt && !ids && !bench)) usage();
  if (nmedia &&
      (ids || bench || check || check_batch || perplexity || o.dump_fd >= 0))
    die("media inputs need generation or --features-fd");
  if (o.ctx < 1 || o.ctx > 1 << 20)
    die("--ctx takes 1 to %d tokens (the model's longest context)", 1 << 20);
  if (o.draft < 0 || o.draft > VERIFY_DRAFTS)
    die("--draft takes 0 to %d drafts", VERIFY_DRAFTS);
  if (o.deepseek.ced && (o.dump_fd >= 0 || check))
    die("--dump-fd and --check-prefill require --ced off; they inspect every "
        "prompt row");
  const ModelOps *ops = model_find_family(dir);
  if (o.draft > BLOCK) die("this model accepts at most %d proposals", BLOCK);
  if (nmedia && !ops->encode_media) die("this model does not encode images");
  if (bench && !ops->run_bench)
    die("--bench: this model has no kernel timings");
  if (bench) {
    o.ctx = 2048;
    model = open_model(dir, &o, memory_gb, reserve_gb);
    if (shutdown_requested()) return 0;
    if (!model)
      die("model weights do not fit the free memory less --reserve, or "
          "--memory");
    ops->run_bench();
    return 0;
  }
  Tokenizer *tk = tokenizer_open(dir, ops->media_token);
  int *tokens = xmalloc((size_t)o.ctx * sizeof *tokens), n = 0;
  if (ids) {
    const char *list = *ids == '@' ? read_file(ids + 1, 0) : ids,
               *sep = ", \t\r\n";
    for (const char *s = list + strspn(list, sep); *s && n < o.ctx;
         s += strspn(s, sep)) {
      char *end;
      tokens[n++] = (int)strtol(s, &end, 10);
      if (end == s) die("--tokens: not a token id at \"%.20s\"", s);
      s = end;
    }
  } else {
    size_t len = strlen(prompt) + 512 + 128 * (size_t)nmedia;
    char *text = xmalloc(len);
    if (raw)
      snprintf(text, len, "%s", prompt);
    else if (ops->render_chat_turn)
      ops->render_chat_turn(text, len, prompt, nmedia, think);
    else
      die("-p and -f: this model's chat template is not in the CLI; use --raw "
          "or --tokens");
    n = raw == 1 ? tokenizer_encode_with_added(tk, text, tokens, o.ctx)
                 : tokenizer_encode_prompt(
                       tk, text, tokens,
                       o.ctx);  // --raw: the placeholders place images
  }
  Media ims[16] = {0};
  if (nmedia) {  // each placeholder becomes its image's span
    int total = n;
    for (int i = 0; i < nmedia; i++) {
      size_t len;
      const char *err;
      char *d = read_file(media_path[i], &len);
      if (!ops->prepare_media ||
          !ops->prepare_media(dir, &ims[i], d, len, &err))
        die("%s: %s", media_path[i],
            ops->prepare_media ? err : "this model has no media processor");
      free(d);
      total += media_span_length(&ims[i]) - 1;
    }
    int *p = xmalloc((size_t)(total + 1) * sizeof *p),
        m = media_expand_prompt(tokens, n, ims, nmedia, p, total,
                                tokenizer_require_added(tk, ops->media_token));
    if (m < 0) die("the prompt placeholders and media inputs differ");
    if (m > o.ctx)
      die("the prompt has %d tokens with its media; --ctx holds %d", m, o.ctx);
    free(tokens), tokens = p, n = m;
  }
  if (tokenize) {
    for (int i = 0; i < n; i++)
      printf("%d%c", tokens[i], i + 1 < n ? ',' : '\n');
    return 0;
  }
  model = open_model(dir, &o, memory_gb, reserve_gb);
  if (shutdown_requested()) return 0;
  if (!model)
    die("model weights do not fit the free memory less --reserve, or --memory");
  const int V = ops->vocab_size();
  ModelState *st = ops->state_new();
  for (int i = 0; i < nmedia; i++) {
    if (shutdown_requested()) return 0;
    double t = now();
    ops->encode_media(&ims[i]);
    fprintf(stderr, "image %s: %d x %d patches, %d tokens, encoded in %.2f s\n",
            media_path[i], ims[i].vh, ims[i].vw, media_span_length(&ims[i]),
            now() - t);
    if (vision_fd >= 0 &&
        write(vision_fd, ims[i].rows, (size_t)ims[i].ntokens * DIM * 2) < 0)
      die("vision: write failed");
  }
  if (vision_fd >= 0) return 0;
  if (ops->set_prefill_media) ops->set_prefill_media(ims, nmedia);
  if (check) {  // every position's logits: the batched prefill against
                // token-by-token decoding
    float *a = xmalloc((size_t)n * V * 4);
    prefilled(ops->prefill(st, tokens, n, 0, 0, -1, 0, a));
    if (shutdown_requested()) return 0;
    if (n >= 4) {  // two prompt batches, the first of odd length, against one
      ModelState *s2 = ops->state_new();
      int n1 = (n / 2) | 1, diff = 0;
      prefilled(ops->prefill(s2, tokens, n1, 0, 0, -1, 0, 0));
      const float *c = prefilled(ops->prefill(s2, tokens, n, n1, 0, -1, 0, 0)),
                  *an = a + (size_t)(n - 1) * V;
      for (int k = 0; k < V; k++) diff += c[k] != an[k];
      printf(
          "two batches (%d + %d) against one: %d of %d final logits differ\n",
          n1, n - n1, diff, V);
    }
    if (shutdown_requested()) return 0;
    ModelState *s3 = ops->state_new();
    int agree[2] = {0}, count[2] = {0};
    double snr[2] = {0};
    for (int i = 0; i < n && !shutdown_requested(); i++) {
      const float *b = ops->decode_rows(&s3, tokens + i, &i, 1),
                  *ai = a + (size_t)i * V;
      double err = 0, ref = 0;
      for (int k = 0; k < V; k++)
        err += (ai[k] - b[k]) * (double)(ai[k] - b[k]),
            ref += (double)b[k] * b[k];
      int part = i >= WINDOW;
      count[part]++, agree[part] += sample_greedy(ai, V) == sample_greedy(b, V),
          snr[part] += 10 * log10(ref / err);
    }
    for (int part = 0; part < 2; part++)
      if (count[part])
        printf("positions %s 128: argmax equal at %d of %d, mean logit SNR "
               "%.1f dB\n",
               part ? ">=" : "<", agree[part], count[part],
               snr[part] / count[part]);
    return 0;
  }
  if (check_batch) {  // the next token of the prompt (a) and of the prompt less
                      // 3 tokens (b), alone and together
    if (n < 5) die("--check-batch needs a prompt of at least 5 tokens");
    ModelState *both_st[2] = {st, ops->state_new()};
    ModelSnapshot *at[2];
    int len[2] = {n, n - 3}, next[2];
    float *alone = xmalloc((size_t)2 * V * 4);
    for (int s = 0; s < 2; s++) {
      if (shutdown_requested()) return 0;
      next[s] = sample_greedy(
          prefilled(ops->prefill(both_st[s], tokens, len[s], 0, 0, -1, 0, 0)),
          V);
      at[s] = ops->snapshot_take(both_st[s], len[s], 0);
      memcpy(alone + (size_t)s * V,
             ops->decode_rows(&both_st[s], &next[s], &len[s], 1),
             (size_t)V * 4);
      ops->snapshot_restore(both_st[s], at[s]);
    }
    if (shutdown_requested()) return 0;
    const float *both = ops->decode_rows(both_st, next, len, 2);
    int diff[2] = {0};
    for (int s = 0; s < 2; s++)
      for (int k = 0; k < V; k++)
        diff[s] += both[(size_t)s * V + k] != alone[(size_t)s * V + k];
    printf("two sequences decoded together against alone: %d and %d of %d "
           "logits differ\n",
           diff[0], diff[1], V);
    int k = ops->drafts_per_pass();
    if (k) {  // a verify pass's first row is the token alone
      if (shutdown_requested()) return 0;
      ops->snapshot_restore(st, at[0]);
      int rows[VERIFY_DRAFTS + 1] = {next[0]}, d = 0;
      Sampler greedy = {.temperature = 0, .top_p = 1};
      ops->propose_drafts(st, next[0], n, rows + 1, k, &greedy);
      const float *v = ops->verify(st, rows, k + 1, n);
      for (int j = 0; j < V; j++) d += v[j] != alone[j];
      printf("a verify pass of %d drafts against the token alone: %d of %d "
             "first-row logits differ\n",
             k, d, V);
    }
    return 0;
  }
  if (ids && !perplexity) {  // replay: one token at a time, every position's
                             // logits dumped
    for (int i = 0; i < n && !shutdown_requested(); i++)
      ops->decode_rows(&st, tokens + i, &i, 1);
    return 0;
  }
  if (perplexity) {  // tokens prefill.. scored, each on the logits after the
                     // token before it, decoded one at a time
    if (prefill < 1 || prefill >= n)
      die("--prefill takes 1 to %d tokens", n - 1);
    const float *lg =
        prefill > 1
            ? prefilled(ops->prefill(st, tokens, prefill, 0, 0, -1, 0, 0))
            : 0;  // after token prefill - 1
    double nll = 0, t0 = now();
    int top = 0, steps = 0, scored = 0;
    ExpertStoreCounts c0 = counters().cache;
    for (int i = prefill; i < n && !shutdown_requested(); i++, lg = 0) {
      int p = i - 1;
      if (!lg) lg = ops->decode_rows(&st, tokens + p, &p, 1), steps++;
      if (logits_fd >= 0 && write(logits_fd, lg, (size_t)V * 4) < 0)
        die("logits: write failed");
      double mx = lg[0], sum = 0;
      for (int k = 1; k < V; k++) mx = lg[k] > mx ? lg[k] : mx;
      for (int k = 0; k < V; k++) sum += exp(lg[k] - mx);
      nll += mx + log(sum) - lg[tokens[i]];
      top += sample_greedy(lg, V) == tokens[i];
      scored++;
    }
    if (scored)
      printf("perplexity %.4f over %d tokens; the next token is the most "
             "likely at %d of them\n",
             exp(nll / scored), scored, top);
    else if (shutdown_requested())
      fprintf(stderr, "perplexity interrupted before scoring\n");
    fprintf(stderr, "%.2f tokens/s; %.1f experts substituted per token\n",
            steps / (now() - t0),
            (counters().cache.substituted - c0.substituted) /
                (double)(steps ? steps : 1));
    return 0;
  }
  Sampler sampler = {.temperature = temp, .top_p = top_p};
  if (seed_set)
    sampler_set_seed(&sampler, seed);
  else
    arc4random_buf(sampler.seed, sizeof sampler.seed);
  ExpertStoreCounts c0 = counters().cache;
  SystemMemoryCounters m0 = system_memory_read_counters();
  double g0 = gpu_times(model->gpu).busy_seconds, t0 = now();
  const float *logits = prefilled(ops->prefill(st, tokens, n, 0, 0, -1, 0, 0));
  if (ops->print_report) ops->print_report(stderr, 0, n);
  double t1 = now();
  ExpertStoreCounts c1 = counters().cache;
  double g1 = gpu_times(model->gpu).busy_seconds;
  int drafts = ops->drafts_per_pass(), verify_rows = ops->max_verify();
  int pos = n, gen = 0, passes = 0, accepted = 0,
      t = sample_token(logits, V, &sampler), len;
  ContextLookup *lookup = NULL;
  lookup = o.lookup ? context_lookup_new(o.ctx) : NULL;
  if (o.lookup && (!lookup || !context_lookup_append(lookup, tokens, n)))
    die("cannot initialize context lookup");
  int lookup_attempts = 0, lookup_passes = 0, lookup_offered = 0,
      lookup_kept = 0;
  DraftChoice choice;  // how many of the draft model's drafts each pass runs
  draft_choice_init(&choice);
  int chosen[DRAFT_CHOICE_MAX + 1] = {0};
  uint64_t cache_start = model_cache_bytes(), cache_min = cache_start,
           expert_store_slot_maximum = cache_start;
  while (gen < ntokens && pos < o.ctx && !ops->is_stop_token(t) &&
         !shutdown_requested()) {
    // A verification pass runs the current token and its proposed successors.
    // Target samples confirm a prefix and supply the next token, consuming
    // exactly the draws used by ordinary sampling.
    int rows[VERIFY_DRAFTS + 1] = {t}, k = 0, a = 0, next;
    double pass_start = now();  // the draft call counts toward the pass's time
    const double *proposal_probabilities = NULL;
    int room = ntokens - gen - 1;
    if (room > o.ctx - pos - 1) room = o.ctx - pos - 1;
    if (room > verify_rows - 1) room = verify_rows - 1;
    ContextDraft proposal = {0};
    if (lookup) {
      if (!context_lookup_append(lookup, &t, 1))
        die("context lookup exceeds the configured context");
      proposal =
          context_lookup_propose(lookup, o.lookup < room ? o.lookup : room);
      lookup_attempts++;
    }
    if (proposal.count) {
      k = proposal.count;
      memcpy(rows + 1, proposal.tokens, (size_t)k * sizeof *rows);
      lookup_passes++;
      lookup_offered += k;
    } else {
      k = drafts < room ? drafts : room;
      if (drafts) k = draft_choice_next_count(&choice, k);
      if (drafts) chosen[k]++;
      if (k)
        proposal_probabilities =
            ops->propose_drafts(st, t, pos, rows + 1, k, &sampler);
    }
    if (k > 0) {
      logits = ops->verify(st, rows, k + 1, pos);
      next = sample_verify_sampled_drafts(logits, V, rows + 1, k,
                                          proposal_probabilities,
                                          ops->is_stop_token, &sampler, &a);
      ops->accept(st, rows, a + 1, pos);
      logits += (size_t)a * V;
      passes++, accepted += a;
    } else {
      logits = ops->decode_rows(&st, &t, &pos, 1);
      next = sample_token(logits, V, &sampler);
    }
    if (drafts && !proposal.count)
      draft_choice_record_pass(&choice, k, a, now() - pass_start);
    if (lookup && a && !context_lookup_append(lookup, rows + 1, a))
      die("context lookup exceeds the configured context");
    if (proposal.count) {
      if (lookup) context_lookup_record_verification(lookup, k, a);
      lookup_kept += a;
    }
    uint64_t current_cache = model_cache_bytes();
    if (current_cache < cache_min) cache_min = current_cache;
    if (current_cache > expert_store_slot_maximum)
      expert_store_slot_maximum = current_cache;
    int i = 0;
    for (; i <= a && !ops->is_stop_token(rows[i]); i++, gen++) {
      const char *s = tokenizer_token_bytes(tk, rows[i], &len);
      fwrite(s, 1, len, stdout);
    }
    fflush(stdout);
    if (i <= a) break;  // a kept draft ends the text
    pos += a + 1;
    t = next;
  }
  context_lookup_free(lookup);
  if (!gen && shutdown_requested()) {
    fprintf(stderr, "generation interrupted before decoding\n");
    return 0;
  }
  double t2 = now();
  int verify_limit = drafts;
  if (verify_limit < o.lookup) verify_limit = o.lookup;
  if (passes)
    fprintf(stderr,
            "speculative: %d passes, %.2f of up to %d drafts kept per pass\n",
            passes, (double)accepted / passes, verify_limit);
  if (drafts) {
    fprintf(stderr, "draft counts chosen:");
    for (int k = 0; k <= drafts && k <= DRAFT_CHOICE_MAX; k++)
      fprintf(stderr, " %d drafts %d times%s", k, chosen[k],
              k < drafts ? "," : "\n");
  }
  if (o.lookup)
    fprintf(stderr,
            "{\"context_lookup\":{\"attempts\":%d,\"passes\":%d,\"offered\":%d,"
            "\"accepted\":%d}}\n",
            lookup_attempts, lookup_passes, lookup_offered, lookup_kept);
  ModelCounters final_counts = counters();
  ExpertStoreCounts c2 = final_counts.cache;
  double g2 = gpu_times(model->gpu).busy_seconds;
  SystemMemoryCounters m2 = system_memory_read_counters();
  fputc('\n', stdout);
  if (cached()) {
    fprintf(stderr,
            "{\"decode_cache\":{\"bytes_at_start\":%llu,\"min_bytes\":%llu,"
            "\"max_bytes\":%llu,\"slots_released\":%llu}}\n",
            (unsigned long long)cache_start, (unsigned long long)cache_min,
            (unsigned long long)expert_store_slot_maximum,
            (unsigned long long)(c2.given_back - c1.given_back));
    // hits: requested experts that needed no read on demand (resident, or
    // prefetched)
    double hit1 = 1 - (double)(c1.misses - c0.misses) /
                          fmax(1, c1.hits - c0.hits + c1.misses - c0.misses +
                                      c1.prefetch_hits - c0.prefetch_hits);
    double hit2 = 1 - (double)(c2.misses - c1.misses) /
                          fmax(1, c2.hits - c1.hits + c2.misses - c1.misses +
                                      c2.prefetch_hits - c1.prefetch_hits);
    fprintf(stderr,
            "prefill: %d tokens in %.2f s, %.2f tok/s, expert hits %.1f%%, "
            "%.2f GB read in %.2f s, GPU %.2f s\n"
            "decode:  %d tokens in %.2f s, %.2f tok/s, expert hits %.1f%%, "
            "%.2f GB read in %.2f s, GPU %.2f s\n"
            "cache:   %d slots, %.1f GiB, %llu evictions; available memory "
            "%.1f GiB, "
            "footprint %.1f GiB, GPU resident memory (all processes) %.1f GiB; "
            "%llu pages swapped out and %llu "
            "compressed by the system meanwhile, this process's memory "
            "compressed at most %.0f MiB\n",
            n, t1 - t0, n / (t1 - t0), 100 * hit1, (c1.bytes - c0.bytes) / 1e9,
            c1.read_s - c0.read_s, g1 - g0, gen, t2 - t1, gen / (t2 - t1),
            100 * hit2, (c2.bytes - c1.bytes) / 1e9, c2.read_s - c1.read_s,
            g2 - g1, final_counts.cache_slots,
            final_counts.cache_bytes / (1024.0 * 1024 * 1024),
            (unsigned long long)c2.evictions,
            system_memory_available() / (1024.0 * 1024 * 1024),
            system_memory_footprint() / (1024.0 * 1024 * 1024),
            system_memory_gpu_driver() / (1024.0 * 1024 * 1024),
            (unsigned long long)(m2.swapouts - m0.swapouts),
            (unsigned long long)(m2.compressions - m0.compressions),
            system_memory_compressed(1) / (1024.0 * 1024));
    fprintf(stderr,
            "decode reads per token: %.1f on demand, %.1f ahead (%.1f used, "
            "%.1f of them late), %.1f cancelled; "
            "%.1f experts substituted, %.1f mapped for the GPU again\n",
            (c2.misses - c1.misses) / (double)gen,
            (c2.prefetches - c1.prefetches) / (double)gen,
            (c2.prefetch_hits - c1.prefetch_hits) / (double)gen,
            (c2.late - c1.late) / (double)gen,
            (c2.cancelled - c1.cancelled) / (double)gen,
            (c2.substituted - c1.substituted) / (double)gen,
            (c2.remaps - c1.remaps) / (double)gen);
  } else {
    fprintf(stderr,
            "prefill: %d tokens in %.2f s, %.2f tok/s, GPU %.2f s\n"
            "decode:  %d tokens in %.2f s, %.2f tok/s, GPU %.2f s\n"
            "memory:  available %.1f GiB, footprint %.1f GiB, GPU resident "
            "memory (all processes) %.1f GiB\n",
            n, t1 - t0, n / (t1 - t0), g1 - g0, gen, t2 - t1, gen / (t2 - t1),
            g2 - g1, system_memory_available() / (1024.0 * 1024 * 1024),
            system_memory_footprint() / (1024.0 * 1024 * 1024),
            system_memory_gpu_driver() / (1024.0 * 1024 * 1024));
  }
  if (ops->print_report) ops->print_report(stderr, 1, gen);
  return 0;
}
