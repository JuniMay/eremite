#include "engine/safetensors.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Every tensor of every file, by name, in a hash table with linear probing.

// The initial capacity when storing header metadata entries.
static constexpr size_t METADATA_INITIAL_CAPACITY = 8;

// A tensor entry in the hash table.
typedef struct {
  // The tensor's name.
  const char *name;
  // The tensor's location, type and shape.
  Tensor t;
} Entry;

// A string entry from a header's __metadata__ object.
typedef struct {
  // The entry's key.
  char *key;
  // The entry's string value.
  char *value;
} Metadata;

// A checkpoint's tensor table, open files and header metadata.
struct TensorIndex {
  // Open-addressed table of tensors by name.
  Entry *table;
  // Number of slots in table, always a power of two.
  size_t capacity;
  // Open file descriptors for the checkpoint's safetensors files.
  int *fds;
  // Number of open file descriptors in fds.
  int nfiles;
  // Header metadata entries read from the opened files.
  Metadata *metadata;
  // Number of stored metadata entries and the allocated capacity.
  size_t metadata_count, metadata_capacity;
};

static uint64_t hash(const char *s) {
  uint64_t h = 1469598103934665603ull;
  while (*s) h = (h ^ (uint8_t)*s++) * 1099511628211ull;
  return h;
}

static Entry *slot(const TensorIndex *index, const char *name) {
  size_t i = hash(name) & (index->capacity - 1);
  while (index->table[i].name && strcmp(index->table[i].name, name))
    i = (i + 1) & (index->capacity - 1);
  return &index->table[i];
}

// A file's header: an 8-byte length, then that many bytes of JSON.
static Json *tensor_header(const char *path, int *fd, uint64_t *bytes) {
  *fd = open(path, O_RDONLY);
  if (*fd < 0 || pread(*fd, bytes, 8, 0) != 8) die("cannot read %s", path);
  char *text = xmalloc(*bytes + 1);
  if (pread(*fd, text, *bytes, 8) != (ssize_t)*bytes)
    die("cannot read %s", path);
  Json *header = json_parse(text);
  free(text);
  fcntl(*fd, F_NOCACHE, 1);
  return header;
}

static TensorIndex *tensor_index(size_t count) {
  TensorIndex *index = xmalloc(sizeof *index);
  size_t slots = count ? count : 1;
  index->capacity = 1;
  while (index->capacity < 2 * slots) index->capacity *= 2;
  index->table = xmalloc(index->capacity * sizeof *index->table);
  index->fds = xmalloc(slots * sizeof *index->fds);
  return index;
}

// Records string entries from a header's __metadata__ object; an earlier file's
// value for the same key is kept.
static void index_metadata(TensorIndex *index, const Json *meta) {
  if (meta->type != J_OBJ) return;
  for (int i = 0; i < meta->n; i++) {
    const Json *entry = &meta->kid[i];
    if (entry->type != J_STR || !entry->key || !entry->s) continue;
    if (tensor_index_metadata(index, entry->key)) continue;
    if (index->metadata_count == index->metadata_capacity) {
      size_t next = index->metadata_capacity ? index->metadata_capacity * 2
                                             : METADATA_INITIAL_CAPACITY;
      Metadata *grown = realloc(index->metadata, next * sizeof *grown);
      if (!grown) die("out of memory for safetensors metadata");
      index->metadata = grown;
      index->metadata_capacity = next;
    }
    char *key = strdup(entry->key), *value = strdup(entry->s);
    if (!key || !value) die("out of memory for safetensors metadata");
    index->metadata[index->metadata_count++] = (Metadata){key, value};
  }
}

// Adds a file's tensors; a name that an earlier file gave keeps its tensor.
static void index_tensors(TensorIndex *index, Json *hdr, int fd, uint64_t n) {
  index->fds[index->nfiles++] = fd;
  for (int k = 0; k < hdr->n; k++) {
    Json *e = &hdr->kid[k];
    if (!strcmp(e->key, "__metadata__")) {
      index_metadata(index, e);
      continue;
    }
    Entry *en = slot(index, e->key);
    if (en->name) continue;
    en->name = strdup(e->key);
    if (!en->name) die("out of memory for tensor name");
    const char *dt = json_get(e, "dtype")->s;
    Json *sh = json_get(e, "shape"), *of = json_get(e, "data_offsets");
    Tensor *t = &en->t;
    t->fd = fd;
    t->off = 8 + n + (uint64_t)of->kid[0].num;
    t->size = (uint64_t)of->kid[1].num - (uint64_t)of->kid[0].num;
    t->dtype = !strcmp(dt, "BF16")      ? 'b'
               : !strcmp(dt, "F16")     ? 'h'
               : !strcmp(dt, "F32")     ? 'f'
               : !strcmp(dt, "F8_E4M3") ? '8'
               : !strcmp(dt, "F8_E8M0") ? 'e'
               : !strcmp(dt, "I8")      ? 'i'
               : !strcmp(dt, "U8")      ? 'u'
               : !strcmp(dt, "U32")     ? 'U'
               : !strcmp(dt, "I64")     ? 'l'
                                        : '?';
    t->ndim = sh->n;
    for (int d = 0; d < sh->n && d < TENSOR_MAX_DIMS; d++)
      t->shape[d] = (int64_t)sh->kid[d].num;
  }
  json_free(hdr);
}

TensorIndex *tensor_index_open_file(const char *path) {
  int fd;
  uint64_t bytes;
  Json *header = tensor_header(path, &fd, &bytes);
  TensorIndex *index = tensor_index((size_t)header->n);
  index_tensors(index, header, fd, bytes);
  return index;
}

TensorIndex *tensor_index_open(const char *dir) {
  char path[4096];
  snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
  if (access(path, R_OK)) {
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    return tensor_index_open_file(path);
  }
  char *text = read_file(path, NULL);
  Json *root = json_parse(text), *map = json_get(root, "weight_map");
  if (!map) die("%s has no weight_map", path);
  TensorIndex *index = tensor_index(map->n);
  const char **done = xmalloc((size_t)map->n * sizeof *done);
  int files = 0;
  for (int i = 0; i < map->n; i++) {
    const char *file = map->kid[i].s;
    int seen = 0;
    for (int j = 0; j < files; j++) seen |= !strcmp(done[j], file);
    if (seen) continue;
    done[files++] = file;
    snprintf(path, sizeof path, "%s/%s", dir, file);
    int fd;
    uint64_t bytes;
    Json *header = tensor_header(path, &fd, &bytes);
    index_tensors(index, header, fd, bytes);
  }
  json_free(root);
  free(text);
  free(done);
  return index;
}

void tensor_index_close(TensorIndex *index) {
  if (!index) return;
  for (size_t i = 0; i < index->capacity; i++)
    free((char *)index->table[i].name);
  for (size_t i = 0; i < index->metadata_count; i++) {
    free(index->metadata[i].key);
    free(index->metadata[i].value);
  }
  for (int i = 0; i < index->nfiles; i++) close(index->fds[i]);
  free(index->metadata), free(index->table), free(index->fds), free(index);
}

const char *tensor_index_metadata(const TensorIndex *index, const char *key) {
  for (size_t i = 0; i < index->metadata_count; i++)
    if (!strcmp(index->metadata[i].key, key)) return index->metadata[i].value;
  return NULL;
}

Tensor tensor_index_get(const TensorIndex *index, const char *fmt, ...) {
  char name[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(name, sizeof name, fmt, ap);
  va_end(ap);
  Entry *e = slot(index, name);
  if (!e->name) die("checkpoint has no tensor %s", name);
  return e->t;
}

int tensor_index_has(const TensorIndex *index, const char *fmt, ...) {
  char name[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(name, sizeof name, fmt, ap);
  va_end(ap);
  return slot(index, name)->name != 0;
}
