#include "check.h"
#include "base/error.h"
#include "base/str.h"
#include "engine/gguf.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the GGUF reader on files that the test writes: header values that it
// skips, the data's alignment, tensors' dimensions, types and offsets, and a
// split set whose second shard is missing.

static char dir[] = "/tmp/eremite-gguf-XXXXXX";

static void put(Str *s, const void *bytes, size_t n) { str_put(s, bytes, n); }

static void put_u32(Str *s, uint32_t v) { put(s, &v, 4); }

static void put_u64(Str *s, uint64_t v) { put(s, &v, 8); }

static void put_string(Str *s, const char *text) {
  put_u64(s, strlen(text));
  put(s, text, strlen(text));
}

// Writes a GGUF file with a 64-byte alignment and one tensor, called name, of
// 2 by 3 values of the type, at offset 128 of the data.
static void write_gguf(const char *file, const char *name, uint32_t type) {
  Str s = {0};
  put(&s, "GGUF", 4);
  put_u32(&s, 3);
  put_u64(&s, 1);  // tensors
  put_u64(&s, 3);  // key-value pairs
  put_string(&s, "general.name"), put_u32(&s, 8), put_string(&s, "a test");
  put_string(&s, "general.list"), put_u32(&s, 9), put_u32(&s, 4),
      put_u64(&s, 3);
  for (uint32_t i = 0; i < 3; i++) put_u32(&s, i);
  put_string(&s, "general.alignment"), put_u32(&s, 4), put_u32(&s, 64);
  put_string(&s, name);
  put_u32(&s, 2), put_u64(&s, 2), put_u64(&s, 3);
  put_u32(&s, type), put_u64(&s, 128);
  char path[256];
  snprintf(path, sizeof path, "%s/%s", dir, file);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0 || write(fd, s.p, s.n) != (ssize_t)s.n)
    die("cannot write %s", path);
  close(fd);
  free(s.p);
}

static uint64_t data_start(
    const char *name) {  // the header's bytes, rounded up to the alignment
  uint64_t header = 4 + 4 + 8 + 8;
  header += 8 + strlen("general.name") + 4 + 8 + strlen("a test");
  header += 8 + strlen("general.list") + 4 + 4 + 8 + 3 * 4;
  header += 8 + strlen("general.alignment") + 4 + 4;
  header += 8 + strlen(name) + 4 + 2 * 8 + 4 + 8;
  return (header + 63) / 64 * 64;
}

int main(void) {
  if (!mkdtemp(dir)) return 1;
  write_gguf("single.gguf", "blk.0.ffn_up_exps.weight", GG_IQ2_XXS);
  write_gguf("set-00001-of-00003.gguf", "first", GG_IQ1_M);
  write_gguf("set-00003-of-00003.gguf", "third", GG_Q2_K);
  char path[256];
  snprintf(path, sizeof path, "%s/single.gguf", dir);
  GgufFiles *files = gguf_files_new();
  gguf_files_add(files, path);
  GgufTensor t;
  check(gguf_files_find(files, "blk.0.ffn_up_exps.weight", &t),
        "the tensor is found");
  check(t.type == GG_IQ2_XXS && t.ne[0] == 2 && t.ne[1] == 3 && t.ne[2] == 0,
        "its type and dimensions come from the header");
  check(t.off == data_start("blk.0.ffn_up_exps.weight") + 128,
        "its offset counts from the aligned start of the data, past the "
        "skipped values");
  check(!gguf_files_find(files, "missing", &t),
        "a missing tensor is not found");

  snprintf(path, sizeof path, "%s/set-00001-of-00003.gguf", dir);
  gguf_files_add(files, path);
  GgufTensor first, third;
  check(gguf_files_find(files, "first", &first) &&
            gguf_files_find(files, "third", &third) && first.type == GG_IQ1_M &&
            third.type == GG_Q2_K && first.fd != third.fd,
        "a split set's shards are all opened, each in its own file");
  check(third.off == data_start("third") + 128,
        "a shard's offsets count within its own file");
  check(gguf_files_find(files, "blk.0.ffn_up_exps.weight", &t),
        "the earlier file's tensors stay");

  snprintf(path, sizeof path, "%s/not.gguf", dir);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0 || write(fd, "GGML....", 8) != 8) return 1;
  close(fd);
  fflush(stdout);
  pid_t child = fork();
  if (!child) gguf_files_add(files, path), _exit(0);
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "a file that is not GGUF ends the program");

  snprintf(path, sizeof path, "%s/single.gguf", dir);
  GgufFiles *other = gguf_files_new();
  gguf_files_add(other, path);
  check(gguf_files_find(other, "blk.0.ffn_up_exps.weight", &t) &&
            !gguf_files_find(other, "first", &t),
        "two sets keep their own tensors");
  gguf_files_free(other);
  gguf_files_find(files, "third", &third);
  gguf_files_free(files);
  check(fcntl(third.fd, F_GETFD) != -1,
        "freeing a set leaves its files open for the tensors found in them");

  const char *names[] = {"single.gguf", "set-00001-of-00003.gguf",
                         "set-00003-of-00003.gguf", "not.gguf"};
  for (int i = 0; i < 4; i++)
    snprintf(path, sizeof path, "%s/%s", dir, names[i]), unlink(path);
  rmdir(dir);
  return failures != 0;
}
