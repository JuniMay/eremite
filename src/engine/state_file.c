#include "engine/state_file.h"
#include "base/error.h"
#include <arm_acle.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

uint32_t crc32c(uint32_t crc, const void *data, size_t n) {
  const uint8_t *p = data;
  crc = ~crc;
  for (; n >= 8; p += 8, n -= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}

static int read_all(int fd, void *p, size_t n, off_t at) {
  for (size_t done = 0; done < n;) {
    ssize_t r = pread(fd, (uint8_t *)p + done, n - done, at + (off_t)done);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    done += (size_t)r;
  }
  return 0;
}

// ---- identity ----

static int weights_file(const char *name) {
  size_t n = strlen(name);
  return !strcmp(name, "config.json") ||
         (n > 12 && !strcmp(name + n - 12, ".safetensors")) ||
         (n > 5 && !strcmp(name + n - 5, ".gguf"));
}

// The file at path, as `name`: its name, size, and first and last 64 KiB.
static void identity_file(CC_SHA256_CTX *h, const char *path,
                          const char *name) {
  // The bytes hashed at each end of the file.
  constexpr size_t EDGE = 65536;

  CC_SHA256_Update(h, name, (CC_LONG)strlen(name) + 1);
  struct stat st;
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  uint64_t size =
      fd >= 0 && !fstat(fd, &st) ? (uint64_t)st.st_size : UINT64_MAX;
  CC_SHA256_Update(h, &size, sizeof size);
  if (fd < 0) return;
  fcntl(fd, F_NOCACHE, 1);
  uint8_t *buf = xmalloc(EDGE);
  size_t head = size < EDGE ? (size_t)size : EDGE;
  if (!read_all(fd, buf, head, 0)) CC_SHA256_Update(h, buf, (CC_LONG)head);
  if (size > EDGE) {
    size_t tail = size - EDGE < EDGE ? (size_t)(size - EDGE) : EDGE;
    if (!read_all(fd, buf, tail, (off_t)(size - tail)))
      CC_SHA256_Update(h, buf, (CC_LONG)tail);
  }
  free(buf);
  close(fd);
}

typedef struct {
  char **names;
  int n, cap;
} Names;

static void collect(const char *root, const char *rel, Names *out, int depth) {
  char path[4096];
  snprintf(path, sizeof path, "%s%s%s", root, *rel ? "/" : "", rel);
  DIR *d = opendir(path);
  if (!d) return;
  for (struct dirent *e; (e = readdir(d));) {
    if (e->d_name[0] == '.') continue;
    char sub[4096], full[4096];
    snprintf(sub, sizeof sub, "%s%s%s", rel, *rel ? "/" : "", e->d_name);
    snprintf(full, sizeof full, "%s/%s", root, sub);
    struct stat st;
    if (lstat(full, &st)) continue;
    if (S_ISDIR(st.st_mode)) {
      if (depth < 8) collect(root, sub, out, depth + 1);
      continue;
    }
    if (!weights_file(e->d_name)) continue;
    if (out->n == out->cap &&
        !(out->names = realloc(
              out->names, (size_t)(out->cap = out->cap ? 2 * out->cap : 64) *
                              sizeof *out->names)))
      die("out of memory listing %s", root);
    out->names[out->n++] = strdup(sub);
  }
  closedir(d);
}

static int by_name(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

void identity_add_folder(CC_SHA256_CTX *h, const char *dir) {
  Names names = {0};
  collect(dir, "", &names, 0);
  qsort(names.names, (size_t)names.n, sizeof *names.names, by_name);
  CC_SHA256_Update(h, &names.n, sizeof names.n);
  for (int i = 0; i < names.n; i++) {
    char full[4096];
    snprintf(full, sizeof full, "%s/%s", dir, names.names[i]);
    identity_file(h, full, names.names[i]);
    free(names.names[i]);
  }
  free(names.names);
}

void identity_add_gguf_set(CC_SHA256_CTX *h, const char *path) {
  const char *slash = strrchr(path, '/'), *base = slash ? slash + 1 : path;
  size_t n = strlen(base), dir = (size_t)(base - path);
  int k, total;
  char tail[32];
  // NAME-00001-of-00005.gguf: every file of the set
  if (n > 20 &&
      sscanf(base + n - 20, "-%5d-of-%5d%31s", &k, &total, tail) == 3 &&
      !strcmp(tail, ".gguf") && total > 0 && total < 100000) {
    for (int i = 1; i <= total; i++) {
      char sibling[4096];
      snprintf(sibling, sizeof sibling, "%.*s%.*s-%05d-of-%05d.gguf", (int)dir,
               path, (int)(n - 20), base, i, total);
      identity_file(h, sibling, sibling + dir);
    }
  } else
    identity_file(h, path, base);
}
