#include "check.h"
#include "engine/state_file.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// Tests the identity of a checkpoint's weights and CRC32C. The identity must
// follow the weights and nothing else. CRC32C must give its standard check
// value, also in pieces. The temporary folder is removed before exit.

static char directory[4096];

static void remove_tree(const char *path) {
  DIR *d = opendir(path);
  if (d) {
    for (struct dirent *e; (e = readdir(d));) {
      if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
      char sub[8192];
      snprintf(sub, sizeof sub, "%s/%s", path, e->d_name);
      struct stat st;
      if (!lstat(sub, &st) && S_ISDIR(st.st_mode))
        remove_tree(sub);
      else
        unlink(sub);
    }
    closedir(d);
  }
  rmdir(path);
}

static void cleanup(void) {
  if (*directory) remove_tree(directory);
}

// ---- identity ----

static void write_file(const char *name, const char *text, size_t size) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(1);
  }
  for (size_t i = 0; i < size; i++) fputc(text[i % strlen(text)], f);
  fclose(f);
}

static void folder_identity(const char *folder, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, folder);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_add_folder(&h, path);
  CC_SHA256_Final(out, &h);
}

static void set_identity(const char *first, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, first);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_add_gguf_set(&h, path);
  CC_SHA256_Final(out, &h);
}

static void identity_checks(void) {
  char path[8192];
  snprintf(path, sizeof path, "%s/model", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/sub", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/.cache", directory), mkdir(path, 0700);
  write_file("model/config.json", "{\"a\": 1}", 8);
  write_file("model/a.safetensors", "weights-a", 300000);
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/README.md", "read me", 7);
  write_file("model/.cache/c.safetensors", "hidden", 6);
  uint8_t base[32], again[32];
  folder_identity("model", base);
  write_file("model/README.md", "read me again", 13);
  write_file("model/.cache/c.safetensors", "hidden, changed", 15);
  write_file("model/.DS_Store", "finder", 6);
  folder_identity("model", again);
  check(!memcmp(base, again, 32),
        "documents and dot files leave the identity as it is");
  write_file("model/a.safetensors", "weights-A",
             300000);  // the first 64 KiB differ
  folder_identity("model", again);
  check(memcmp(base, again, 32),
        "a weights file's first bytes change the identity");
  write_file("model/a.safetensors", "weights-a", 300000);
  folder_identity("model", again);
  check(!memcmp(base, again, 32),
        "the same weights give the same identity again");
  write_file("model/sub/b.safetensors", "weights-b", 1001);  // its size
  folder_identity("model", again);
  check(memcmp(base, again, 32), "a weights file's size changes the identity");
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/config.json", "{\"a\": 2}", 8);
  folder_identity("model", again);
  check(memcmp(base, again, 32), "config.json changes the identity");
  write_file("x-00001-of-00002.gguf", "first", 5000);
  write_file("x-00002-of-00002.gguf", "second", 5000);
  set_identity("x-00001-of-00002.gguf", base);
  write_file("x-00002-of-00002.gguf", "SECOND", 5000);
  set_identity("x-00001-of-00002.gguf", again);
  check(memcmp(base, again, 32), "every file of a GGUF set counts");
}

int main(void) {
  const char *base = getenv("TMPDIR");
  if (!base || !*base) base = "/tmp";
  if (snprintf(directory, sizeof directory,
               "%s/eremite-state-file-check.XXXXXX",
               base) >= (int)sizeof directory ||
      !mkdtemp(directory)) {
    perror("mkdtemp");
    return 1;
  }
  atexit(cleanup);

  identity_checks();
  check(crc32c(0, "123456789", 9) == 0xe3069283,
        "CRC32C gives the standard check value of \"123456789\"");
  check(crc32c(crc32c(0, "1234", 4), "56789", 5) == 0xe3069283,
        "CRC32C continues across pieces");
  return failures != 0;
}
