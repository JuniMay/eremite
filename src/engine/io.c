#include "engine/io.h"
#include "base/error.h"
#include "platform/threads.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void io_read_piece(void *pieces, size_t i) {
  IoPiece *p = (IoPiece *)pieces + i;
  for (uint64_t got = 0; got < p->len;) {
    ssize_t r = pread(p->fd, p->dst + got, p->len - got, p->off + got);
    if (r < 0) die("read failed at offset %llu", p->off + got);
    if (!r) break;  // the sectors may extend past the end of the file
    got += r;
  }
}

int io_split(IoPiece *out, int fd, uint64_t off, uint64_t len, uint8_t *dst,
             uint32_t *at) {
  uint64_t a = off & ~(IO_SECTOR - 1),
           e = (off + len + IO_SECTOR - 1) & ~(IO_SECTOR - 1);
  int n = 0;
  *at = off - a;
  for (uint64_t p = a; p < e; p += IO_PIECE)
    out[n++] =
        (IoPiece){fd, p, e - p < IO_PIECE ? e - p : IO_PIECE, dst + (p - a)};
  return n;
}

void io_read(int fd, uint64_t off, uint64_t len, Buf *dst) {
  int n = (len + 2 * IO_SECTOR) / IO_PIECE + 2;
  IoPiece *p = xmalloc(n * sizeof *p);
  n = io_split(p, fd, off, len, dst->ptr, &dst->off);
  if (dst->off + len > dst->size) die("io_read: buffer too small");
  threads_apply(n, THREADS_INITIATED, p, io_read_piece);
  free(p);
}

typedef struct {
  int fd;
  uint64_t off;
  uint32_t len;
  uint8_t *dst;
} Row;

// A row is read in whole 4 KiB blocks (the file system's block size), which
// uncached reads also take past the page cache.
static constexpr uint64_t FILE_BLOCK = 4096;

static void read_row(void *ctx, size_t i) {
  Row *r = (Row *)ctx + i;
  _Alignas(16384) uint8_t tmp[3 * IO_SECTOR];
  uint64_t a = r->off & ~(FILE_BLOCK - 1),
           e = (r->off + r->len + FILE_BLOCK - 1) & ~(FILE_BLOCK - 1);
  if (e - a > sizeof tmp) die("io_read_rows: row too long");
  if (pread(r->fd, tmp, e - a, a) < (ssize_t)(r->off + r->len - a))
    die("read failed at offset %llu", a);
  memcpy(r->dst, tmp + (r->off - a), r->len);
}

// Rows are small random reads, whose speed is the number in flight: 64 lanes
// read them, and the threads layer adds threads while lanes wait in pread.
static constexpr int LANES = 64;

typedef struct {
  Row *r;
  int n, lane, lanes;
} Lane;

uint64_t io_read_rows_memory(int rows) {
  return (uint64_t)rows * (sizeof(uint64_t) + sizeof(Row));
}

static void read_lane(void *ctx) {
  Lane *l = ctx;
  for (int i = l->lane; i < l->n; i += l->lanes) read_row(l->r, i);
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

// A row requested more than once is read once and copied: a prompt repeats
// many of its n-grams.
void io_read_rows(int fd, uint64_t base, uint32_t rowbytes, const int64_t *rows,
                  int n, uint8_t *dst) {
  if (n >= 1 << 20) die("io_read_rows: too many rows");
  uint64_t *key =
      xmalloc((size_t)n * 8);  // row, then request index in the low 20 bits
  for (int i = 0; i < n; i++) key[i] = (uint64_t)rows[i] << 20 | i;
  qsort(key, n, 8, cmp_u64);
  Row *r = xmalloc((size_t)n * sizeof *r);
  int u = 0;
  for (int i = 0; i < n; i++)
    if (!i || key[i] >> 20 != key[i - 1] >> 20) {
      uint64_t at = key[i] & 0xfffff;
      r[u++] = (Row){fd, base + rows[at] * (uint64_t)rowbytes, rowbytes,
                     dst + at * rowbytes};
    }
  Lane lane[LANES];
  int k = u < LANES ? u : LANES;
  ThreadsGroup *g = threads_group_new();
  for (int j = 0; j < k; j++) {
    lane[j] = (Lane){r, u, j, k};
    threads_async(g, THREADS_INITIATED, &lane[j], read_lane);
  }
  threads_group_wait(g);
  threads_group_free(g);
  for (int i = 1, first = key[0] & 0xfffff; i < n; i++) {
    int at = key[i] & 0xfffff;
    if (key[i] >> 20 == key[i - 1] >> 20)
      memcpy(dst + (uint64_t)at * rowbytes, dst + (uint64_t)first * rowbytes,
             rowbytes);
    else
      first = at;
  }
  free(key);
  free(r);
}
