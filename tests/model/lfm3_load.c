#include "check.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/lfm3/model.h"
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests loading d1-2B's language model from its checkpoint: the family that
// runs it, the weights counted, and that the vision tower's weights do not
// load with it. A child process checks that a checkpoint whose config.json
// lacks d1's question layout is refused.
//
// usage: model-lfm3_load CHECKPOINT

// d1-2B's language model: 154 BF16 matrices, the embedding table among them,
// and 132 vectors, which load as FP32.
static constexpr uint64_t MATRIX_BYTES = 3456827392, VECTOR_BYTES = 1551360;
// The bytes of all of the checkpoint's tensors, the vision tower's included.
static constexpr uint64_t CHECKPOINT_BYTES = 3836363776;

static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Writes dir/config.json: the checkpoint's config.json without its
// d1_attention key.
static void write_config_without_layout(const char *checkpoint,
                                        const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", checkpoint);
  FILE *in = fopen(path, "r");
  snprintf(path, sizeof path, "%s/config.json", dir);
  FILE *out = fopen(path, "w");
  if (!in || !out) die("cannot copy config.json");
  char line[4096];
  while (fgets(line, sizeof line, in))
    if (!strstr(line, "\"d1_attention\"")) fputs(line, out);
  fclose(in), fclose(out);
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-lfm3_load CHECKPOINT");
  check(model_find_family(argv[1]) == &lfm3_ops,
        "the checkpoint's model_type lfm3_vl is LFM3's family");
  char dir[] = "/tmp/eremite-lfm3-XXXXXX";
  if (!mkdtemp(dir)) die("cannot make a folder");
  write_config_without_layout(argv[1], dir);
  fflush(stdout);
  pid_t child = fork();
  if (child == 0) {
    Memory *memory = memory_new();
    size_t length;
    const char *source = kernel_library_source(&length);
    Model *m = model_create(memory, gpu_new(source, length), dir, 0);
    int loaded = model_load(m, &(Options){.ctx = 1});
    _exit(loaded ? 0 : 2);
  }
  int status = 0;
  waitpid(child, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
        "a config.json without d1's question layout ends the load with a "
        "message");
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", dir);
  unlink(path), rmdir(dir);

  Memory *memory = memory_new();
  size_t length;
  const char *source = kernel_library_source(&length);
  Gpu *gpu = gpu_new(source, length);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  Options options = {
      .ctx = LFM3_POSITIONS, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *m = model_open(memory, gpu, argv[1], &options);
  check(m != NULL, "the language model loads");
  if (!m) return 1;
  ModelCounters c;
  m->ops->read_counters(&c);
  check(c.layers == LFM3_LAYERS && m->ops->vocab_size() == LFM3_VOCAB,
        "the model has 24 layers and a vocabulary of 128,000 tokens");
  check(c.dense_bytes >= MATRIX_BYTES + VECTOR_BYTES &&
            c.dense_bytes < CHECKPOINT_BYTES,
        "its matrices and FP32 vectors are counted, and the vision tower's "
        "weights are not loaded");
  return failures != 0;
}
