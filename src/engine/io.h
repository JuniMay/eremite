#ifndef EREMITE_ENGINE_IO_H
#define EREMITE_ENGINE_IO_H
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// Reads files without the page cache (F_NOCACHE). A read covers whole 16 KiB
// sectors, since reading part of a sector still fills the page cache. A read
// is split into pieces of at most 2 MiB, which can run in parallel.

// The bytes of a sector, and the most bytes of a piece.
static constexpr uint64_t IO_SECTOR = 16384, IO_PIECE = 2ull << 20;

// The most bytes of the sectors that hold len bytes.
#define IO_REGION(len) (((len) + 3 * IO_SECTOR - 1) / IO_SECTOR * IO_SECTOR)

// A piece of a read: len bytes at off in the file fd, to dst.
typedef struct {
  int fd;
  uint64_t off, len;
  uint8_t *dst;
} IoPiece;

// Splits the sectors that hold len bytes at off into pieces, which are read to
// dst. Returns the number of pieces, and sets *at to where the bytes start in
// dst.
int io_split(IoPiece *out, int fd, uint64_t off, uint64_t len, uint8_t *dst,
             uint32_t *at);

// Reads piece i of the array pieces. Ends the program if the read fails.
void io_read_piece(void *pieces, size_t i);

// Reads len bytes at off in the file fd into dst, its pieces in parallel. Sets
// dst->off to where the bytes start. Ends the program if dst is too small.
void io_read(int fd, uint64_t off, uint64_t len, Buf *dst);

// Reads n rows of rowbytes bytes each from a table at base in the file fd. Row
// rows[i] goes to dst + i * rowbytes.
void io_read_rows(int fd, uint64_t base, uint32_t rowbytes, const int64_t *rows,
                  int n, uint8_t *dst);

// Returns the host memory that io_read_rows takes for that many rows. The
// caller counts it, since io_read_rows does not.
uint64_t io_read_rows_memory(int rows);

#endif
