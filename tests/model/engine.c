#include "check.h"
#include "base/error.h"
#include "eremite.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the engine's C interface on a DeepSeek V4.1 Flash checkpoint without
// loading its weights: the process's start and its idle work's interval, the
// options eremite_open refuses, one model per family, the model's limits and
// tokenizer, samplers, context lookups, the draft-count chooser, a saved
// state's identity, and the statistics before a load. A child process opens
// the same model with packed state rows, since a process holds one model of
// each family: its identity must differ.
//
// usage: model-engine CHECKPOINT

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-engine CHECKPOINT");
  uint8_t packed_identity[32] = {0};
  int identity_pipe[2];
  if (pipe(identity_pipe)) die("cannot make a pipe");
  fflush(stdout);
  pid_t child = fork();
  if (child == 0) {
    eremite_config unpacked = {
        .abi = EREMITE_ABI,
        .model = argv[1],
        .ctx = 4096,
        .lookup = 15,
        .deepseek = {.draft = 2, .unpacked_state_rows = 1}};
    eremite_model *p = eremite_init(&(eremite_process){.abi = EREMITE_ABI}) == 0
                           ? eremite_open(&unpacked)
                           : NULL;
    if (p) eremite_state_identity(p, packed_identity);
    _exit(write(identity_pipe[1], packed_identity, 32) == 32 ? 0 : 1);
  }
  close(identity_pipe[1]);
  int got = child > 0 && read(identity_pipe[0], packed_identity, 32) == 32;
  close(identity_pipe[0]);
  if (child > 0) waitpid(child, NULL, 0);
  check(eremite_init(&(eremite_process){.abi = EREMITE_ABI + 1}) == -1,
        "a process of another ABI is refused");
  check(eremite_idle_work_interval_seconds() == 0,
        "the idle work has no interval before the process starts");
  check(eremite_init(&(eremite_process){
            .abi = EREMITE_ABI,
            .available_memory_reading_interval_seconds = 2.5}) == 0,
        "the process starts");
  check(eremite_idle_work_interval_seconds() == 2.5,
        "the idle work runs once per available-memory reading interval");
  check(eremite_init(&(eremite_process){.abi = EREMITE_ABI}) == -1,
        "a second start is refused");

  eremite_config bad[] = {
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 0},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .deepseek.draft = 6},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .lookup = 16},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .deepseek.ced = 100},
      {.abi = EREMITE_ABI, .model = argv[1], .ctx = 4096, .memory_gb = -1},
      {.abi = EREMITE_ABI + 1, .model = argv[1], .ctx = 4096}};
  int refused = 0;
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
    refused += !eremite_open(&bad[i]);
  check(refused == 6, "a context of 0, 6 drafts, 16 lookups, a CED of 100, a "
                      "memory limit below 0 and another ABI are refused");
  eremite_config c = {.abi = EREMITE_ABI,
                      .model = argv[1],
                      .ctx = 4096,
                      .lookup = 15,
                      .deepseek.draft = 2};
  eremite_model *m = eremite_open(&c);
  check(m != NULL, "a valid configuration registers the model");
  check(!eremite_open(&c), "a second model of the same family is refused");
  check(m && !eremite_is_loaded(m), "a registered model is not loaded");
  if (!m) return 1;
  check(eremite_vocab_size(m) == 129280 && eremite_max_rows(m) == 6 &&
            eremite_max_verify(m) == 16 && eremite_drafts_per_pass(m) == 2,
        "the model's limits are known before it loads");

  eremite_tokenizer *t = eremite_model_tokenizer(m);
  int32_t ids[64];
  const char *text = "Hello, world";
  int n = eremite_tokenize(t, text, strlen(text), 0, ids, 64);
  char back[64];
  size_t total = 0;
  for (int i = 0; i < n && total < sizeof back; i++) {
    size_t len;
    const char *piece = eremite_token_bytes(t, ids[i], &len);
    if (total + len <= sizeof back) memcpy(back + total, piece, len);
    total += len;
  }
  check(n > 0 && total == strlen(text) && !memcmp(back, text, total),
        "a text's pieces spell it again");
  check(eremite_tokenize(t, text, strlen(text), 0, ids, 1) == -1,
        "a text with more tokens than places gives -1");
  const char *bos = "<｜begin▁of▁sentence｜>";
  int32_t id = eremite_find_added_token(t, bos, strlen(bos));
  check(id >= 0 && eremite_token_kind(t, id) > 0,
        "the beginning of a sentence is an added token");
  check(eremite_tokenize(t, "\xff", 1, 2, ids, 64) == 1 && ids[0] == 129264,
        "an image's place in a rendered prompt is the image token");

  check(!eremite_sampler_new(-1, 1, 0, 0, 1) &&
            !eremite_sampler_new(1, 1.5, 0, 0, 1) &&
            !eremite_sampler_new(1, 1, -1, 0, 1),
        "a negative temperature, a top-p above 1 and a negative top-k are "
        "refused");
  eremite_sampler *greedy = eremite_sampler_new(0, 1, 0, 0, 1);
  float *logits = calloc(129280, sizeof *logits);
  logits[1234] = 5;
  check(greedy && eremite_sample(m, logits, greedy) == 1234,
        "a greedy sampler takes the highest logit");
  eremite_sampler_free(greedy);
  free(logits);

  int32_t pattern[20], history[40], drafts[15];
  for (int i = 0; i < 20; i++) pattern[i] = 1000 + i * 7;
  for (int i = 0; i < 40; i++) history[i] = pattern[i % 20];
  eremite_lookup *lookup = eremite_lookup_new(1024);
  int appended = lookup && eremite_lookup_append(lookup, history, 40);
  int proposed = appended ? eremite_lookup_propose(lookup, 15, drafts) : 0;
  check(proposed > 0 && drafts[0] == pattern[0],
        "a context lookup proposes what followed the last tokens before");
  eremite_lookup_free(lookup);

  eremite_draft_choice *choice = eremite_draft_choice_new();
  int count = eremite_draft_choice_next_count(choice, 2);
  eremite_draft_choice_record_pass(choice, count, count, 0.01);
  check(count >= 0 && count <= 2,
        "the draft-count chooser picks 0 to the most drafts");
  eremite_draft_choice_free(choice);

  uint8_t first[32], second[32], zero[32] = {0};
  eremite_state_identity(m, first);
  eremite_state_identity(m, second);
  check(!memcmp(first, second, 32) && memcmp(first, zero, 32),
        "a saved state's identity is known and stable");
  check(got && memcmp(packed_identity, zero, 32) &&
            memcmp(packed_identity, first, 32),
        "packed state rows give another identity");

  eremite_refresh_stats();
  eremite_stats s;
  eremite_read_stats(m, &s);
  check(s.total > 0 && s.gpu_limit == INT64_MAX && s.decode_rows == 0,
        "before the GPU starts, the statistics hold the system's memory and no "
        "work");
  return failures != 0;
}
