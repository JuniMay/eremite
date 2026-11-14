#include "check.h"
#include "base/error.h"
#include "engine/safetensors.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the tensor index on checkpoints that the test writes: one
// model.safetensors, and shards that an index file names. Also tests that
// closing one index leaves another's tensors readable.

// Writes a safetensors file: the header's length, the header, then data.
static void write_file(const char *dir, const char *name, const char *header,
                       const void *data, size_t bytes) {
  char path[256];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  uint64_t length = strlen(header);
  if (fd < 0 || write(fd, &length, 8) != 8 ||
      write(fd, header, length) != (ssize_t)length ||
      write(fd, data, bytes) != (ssize_t)bytes)
    die("cannot write %s", path);
  close(fd);
}

static void remove_files(const char *dir, const char *const *names, int count) {
  char path[256];
  for (int i = 0; i < count; i++)
    snprintf(path, sizeof path, "%s/%s", dir, names[i]), unlink(path);
}

int main(void) {
  char dir[] = "/tmp/eremite-safetensors-XXXXXX";
  if (!mkdtemp(dir)) return 1;
  const char single[] =
      "{\"__metadata__\":{\"format\":\"pt\"},"
      "\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]},"
      "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[12,28]}}";
  uint8_t data[28];
  for (int i = 0; i < 28; i++) data[i] = (uint8_t)(i * 7);
  write_file(dir, "model.safetensors", single, data, sizeof data);
  TensorIndex *index = tensor_index_open(dir);
  Tensor a = tensor_index_get(index, "a"),
         b = tensor_index_get(index, "%s", "b");
  uint64_t start = 8 + strlen(single);
  check(a.dtype == 'b' && a.ndim == 2 && a.shape[0] == 2 && a.shape[1] == 3 &&
            a.off == start && a.size == 12,
        "a tensor's type, shape and bytes come from the header");
  check(b.dtype == 'f' && b.ndim == 1 && b.off == start + 12 && b.size == 16,
        "the next tensor's bytes follow");
  check(tensor_index_has(index, "a") && !tensor_index_has(index, "c"),
        "the index has the header's tensors and no others");
  uint8_t read_back[16];
  check(pread(b.fd, read_back, 16, (off_t)b.off) == 16 &&
            !memcmp(read_back, data + 12, 16),
        "the tensor's file holds its bytes at its offset");

  // A sharded checkpoint, opened while the first stays open.
  char sharded[256];
  snprintf(sharded, sizeof sharded, "%s/sharded", dir);
  if (mkdir(sharded, 0755)) return 1;
  const char *index_json = "{\"metadata\":{},\"weight_map\":{\"x\":\"one."
                           "safetensors\",\"y\":\"two.safetensors\","
                           "\"z\":\"one.safetensors\"}}";
  char path[256];
  snprintf(path, sizeof path, "%s/model.safetensors.index.json", sharded);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0 ||
      write(fd, index_json, strlen(index_json)) != (ssize_t)strlen(index_json))
    return 1;
  close(fd);
  write_file(sharded, "one.safetensors",
             "{\"x\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
             "\"z\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[4,12]}}",
             data, 12);
  write_file(sharded, "two.safetensors",
             "{\"y\":{\"dtype\":\"U32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
             data, 8);
  TensorIndex *shards = tensor_index_open(sharded);
  Tensor x = tensor_index_get(shards, "x"), y = tensor_index_get(shards, "y"),
         z = tensor_index_get(shards, "z");
  check(x.dtype == 'u' && z.dtype == 'l' && y.dtype == 'U' && x.fd == z.fd &&
            x.fd != y.fd,
        "the shards' tensors come from the files the index names");
  check(!tensor_index_has(shards, "a") && !tensor_index_has(index, "x"),
        "two open indexes keep their own tensors");
  tensor_index_close(shards);
  Tensor again = tensor_index_get(index, "b");
  check(again.fd == b.fd && again.off == b.off &&
            pread(again.fd, read_back, 16, (off_t)again.off) == 16 &&
            !memcmp(read_back, data + 12, 16),
        "closing one index leaves the other's tensors readable");

  fflush(stdout);
  pid_t child = fork();
  if (!child) tensor_index_get(index, "missing"), _exit(0);
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "asking for a missing tensor ends the program");
  tensor_index_close(index);

  remove_files(sharded,
               (const char *const[]){"model.safetensors.index.json",
                                     "one.safetensors", "two.safetensors"},
               3);
  rmdir(sharded);
  remove_files(dir, (const char *const[]){"model.safetensors"}, 1);
  rmdir(dir);
  return failures != 0;
}
