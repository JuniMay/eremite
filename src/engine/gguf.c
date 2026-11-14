#include "engine/gguf.h"
#include "base/error.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// A file's header holds key-value pairs, which are skipped except for the
// data's alignment, and then the tensors' names, dimensions, types and
// offsets from the aligned start of the data.

// A tensor and its name.
typedef struct {
  char *name;
  GgufTensor t;
} GgEntry;

struct GgufFiles {
  // The added tensors, and how many there are.
  GgEntry *gg;
  int ngg;
};

// A header reader: the bytes read so far, in growing chunks, and the position.
typedef struct {
  int fd;
  uint8_t *b;
  uint64_t n, at;
} Rd;

static const uint8_t *take_bytes(Rd *r, uint64_t n) {
  while (r->at + n > r->n) {
    uint64_t want = r->n ? 2 * r->n : 1 << 20;
    r->b = realloc(r->b, want);
    ssize_t got = pread(r->fd, r->b + r->n, want - r->n, r->n);
    if (got <= 0) die("GGUF header truncated");
    r->n += got;
  }
  r->at += n;
  return r->b + r->at - n;
}

static uint64_t rd_u(Rd *r, int n) {
  uint64_t v = 0;
  memcpy(&v, take_bytes(r, n), n);
  return v;
}

static char *rd_str(Rd *r) {
  uint64_t n = rd_u(r, 8);
  char *s = xmalloc(n + 1);
  memcpy(s, take_bytes(r, n), n);
  s[n] = 0;
  return s;
}

// Skips a value of the type: a scalar, a string (8), or an array (9).
static void rd_skip(Rd *r, uint32_t type) {
  static const uint8_t size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  if (type == 8)
    take_bytes(r, rd_u(r, 8));
  else if (type == 9) {
    uint32_t t = rd_u(r, 4);
    uint64_t n = rd_u(r, 8);
    for (uint64_t i = 0; i < n; i++)
      if (t == 8 || t == 9)
        rd_skip(r, t);
      else
        take_bytes(r, size[t]);
  } else if (type < 13)
    take_bytes(r, size[type]);
  else
    die("GGUF: unknown value type %u", type);
}

static void add_file(GgufFiles *files, const char *path) {
  Rd r = {open(path, O_RDONLY), 0, 0, 0};
  if (r.fd < 0) die("cannot open %s", path);
  if (memcmp(take_bytes(&r, 4), "GGUF", 4) || rd_u(&r, 4) != 3)
    die("%s: not a GGUF file of version 3", path);
  uint64_t nt = rd_u(&r, 8), nkv = rd_u(&r, 8), align = 32;
  for (uint64_t i = 0; i < nkv; i++) {
    char *key = rd_str(&r);
    uint32_t type = rd_u(&r, 4);
    if (!strcmp(key, "general.alignment") && type == 4)
      align = rd_u(&r, 4);
    else
      rd_skip(&r, type);
    free(key);
  }
  files->gg = realloc(files->gg, (files->ngg + nt) * sizeof *files->gg);
  if (!files->gg) die("out of memory for the tensors of %s", path);
  GgEntry *first = files->gg + files->ngg;
  for (uint64_t i = 0; i < nt; i++) {
    GgEntry *e = &files->gg[files->ngg++];
    e->name = rd_str(&r);
    uint32_t nd = rd_u(&r, 4);
    memset(e->t.ne, 0, sizeof e->t.ne);
    for (uint32_t d = 0; d < nd; d++) {
      uint64_t v = rd_u(&r, 8);
      if (d < 4) e->t.ne[d] = v;
    }
    e->t.type = rd_u(&r, 4);
    e->t.off = rd_u(&r, 8);
    e->t.fd = r.fd;
  }
  uint64_t data = (r.at + align - 1) / align * align;
  for (GgEntry *e = first; e < files->gg + files->ngg; e++) e->t.off += data;
  free(r.b);
  fcntl(r.fd, F_NOCACHE, 1);
}

GgufFiles *gguf_files_new(void) { return xmalloc(sizeof(GgufFiles)); }

void gguf_files_add(GgufFiles *files, const char *path) {
  const char *of = strstr(path, "-of-");
  int n = of ? atoi(of + 4) : 0;
  if (!of || of - path < 5 || n <= 0) {
    add_file(files, path);
    return;
  }
  // A shard that was not downloaded is left out, such as the one with the
  // Engram tables.
  char p[4096];
  for (int i = 1; i <= n; i++) {
    snprintf(p, sizeof p, "%.*s%05d%s", (int)(of - path - 5), path, i, of);
    if (!access(p, R_OK)) add_file(files, p);
  }
}

int gguf_files_find(const GgufFiles *files, const char *name, GgufTensor *t) {
  for (int i = 0; i < files->ngg; i++)
    if (!strcmp(files->gg[i].name, name)) return *t = files->gg[i].t, 1;
  return 0;
}

void gguf_files_free(GgufFiles *files) {
  for (int i = 0; i < files->ngg; i++) free(files->gg[i].name);
  free(files->gg);
  free(files);
}
