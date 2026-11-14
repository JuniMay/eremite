#include "check.h"
#include "model/model.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that model_open runs the loader of the family whose model types
// include the checkpoint's config.json model_type, and that a model_type no
// family runs ends the program. The loader is a stand-in; no weights are read.

static int loads;
static const char *loaded;

static int load_stub(Gpu *gpu, MemorySlot *work, const char *dir,
                     const Options *options) {
  (void)gpu, (void)work, (void)options;
  loads++, loaded = dir;
  return 1;
}

static const char *const deepseek_types[] = {"deepseek_v41", 0};
const ModelOps deepseek_ops = {
    .family = "DeepSeek", .model_types = deepseek_types, .load = load_stub};

static int write_config(const char *dir, const char *type) {
  char path[256], text[128];
  snprintf(path, sizeof path, "%s/config.json", dir);
  int length = snprintf(text, sizeof text,
                        "{\"model_type\": \"%s\", \"eos_token_id\": 1}", type);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  int written = fd >= 0 && write(fd, text, length) == length;
  if (fd >= 0) close(fd);
  return written;
}

int main(void) {
  char dir[] = "/tmp/eremite-model-XXXXXX", path[256];
  if (!mkdtemp(dir) || !write_config(dir, "deepseek_v41")) return 1;
  check(model_find_family(dir) == &deepseek_ops,
        "the checkpoint's model_type selects its family");
  Model *model = model_open(memory_new(), NULL, dir, &(Options){.ctx = 16});
  check(model && loads == 1 && loaded == dir,
        "model_open runs the family's loader on the checkpoint");

  if (!write_config(dir, "llama")) return 1;
  fflush(stdout);
  pid_t child = fork();
  if (!child) model_find_family(dir), _exit(0);
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "a model_type that no family runs ends the program");

  snprintf(path, sizeof path, "%s/config.json", dir);
  unlink(path);
  rmdir(dir);
  return failures != 0;
}
