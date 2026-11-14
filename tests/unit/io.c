#include "check.h"
#include "engine/io.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests reads past the page cache. A read splits into whole sectors, in pieces
// of at most IO_PIECE that follow each other. A read at any offset gives the
// file's bytes, also where its last sector runs past the file's end. Row reads
// give each requested row, also a row asked for twice, and a read of no rows
// reads nothing. A read into a buffer that is too small ends the program.

// The test file's bytes: a little over 6 MiB, so that reads take several
// pieces and the file ends inside a sector.
static constexpr size_t FILE_BYTES = (6u << 20) + 12345;

// The bytes of each row in the row table, and where the table starts.
static constexpr uint32_t ROW_BYTES = 300;
static constexpr uint64_t TABLE_START = 4096 + 17;

// Returns the test file's byte at offset.
static uint8_t byte_at(uint64_t offset) {
  return (uint8_t)(offset * 131 + (offset >> 13));
}

// Reads len bytes at off with io_read. Returns whether they are the file's.
static int read_matches(int fd, uint64_t off, uint64_t len) {
  size_t size = IO_REGION(len);
  Buf dst = {.ptr = aligned_alloc(IO_SECTOR, size), .size = size};
  io_read(fd, off, len, &dst);
  int same = 1;
  for (uint64_t i = 0; i < len; i++)
    same &= dst.ptr[dst.off + i] == byte_at(off + i);
  free(dst.ptr);
  return same;
}

int main(void) {
  // How a read of 5 MiB at an offset inside a sector splits.
  uint64_t off = 3 * IO_SECTOR + 100, len = 5u << 20;
  uint64_t start = 3 * IO_SECTOR,
           end = (off + len + IO_SECTOR - 1) / IO_SECTOR * IO_SECTOR;
  IoPiece pieces[8];
  uint8_t *region = malloc(IO_REGION(len));
  uint32_t at = 0;
  int count = io_split(pieces, 7, off, len, region, &at);
  int follow = count > 0;
  uint64_t covered = 0;
  for (int i = 0; i < count; i++) {
    follow &= pieces[i].fd == 7 && pieces[i].len <= IO_PIECE &&
              pieces[i].off == start + covered &&
              pieces[i].dst == region + covered;
    covered += pieces[i].len;
  }
  check(follow && covered == end - start,
        "a read splits into whole sectors, in pieces of at most 2 MiB that "
        "follow each other");
  check(at == 100,
        "the requested bytes start at their offset in the first sector");
  check(end - start <= IO_REGION(len),
        "the sectors fit in IO_REGION(len) bytes");
  free(region);

  char path[] = "/tmp/eremite-io-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) return 1;
  uint8_t *bytes = malloc(FILE_BYTES);
  for (size_t i = 0; i < FILE_BYTES; i++) bytes[i] = byte_at(i);
  if (write(fd, bytes, FILE_BYTES) != (ssize_t)FILE_BYTES) return 1;
  free(bytes);
  fcntl(fd, F_NOCACHE, 1);

  check(read_matches(fd, 0, 1), "a read of the first byte gives that byte");
  check(read_matches(fd, 12345, (5u << 20) + 777),
        "a read across several pieces gives the file's bytes");
  check(read_matches(fd, FILE_BYTES - 1000, 1000),
        "a read whose last sector runs past the file's end gives the bytes");

  // Rows asked for in any order, two of them twice.
  const int64_t rows[] = {5, 2, 5, 9, 2, 0};
  constexpr int ROWS = sizeof rows / sizeof *rows;
  uint8_t got[ROWS * ROW_BYTES];
  io_read_rows(fd, TABLE_START, ROW_BYTES, rows, ROWS, got);
  int rows_match = 1;
  for (int i = 0; i < ROWS; i++)
    for (uint32_t b = 0; b < ROW_BYTES; b++)
      rows_match &= got[i * ROW_BYTES + b] ==
                    byte_at(TABLE_START + (uint64_t)rows[i] * ROW_BYTES + b);
  check(rows_match, "row reads give each requested row, also a row asked "
                    "for twice");
  memset(got, 0xa5, sizeof got);
  io_read_rows(fd, TABLE_START, ROW_BYTES, rows, 0, got);
  int untouched = 1;
  for (size_t i = 0; i < sizeof got; i++) untouched &= got[i] == 0xa5;
  check(untouched, "a read of no rows writes nothing");

  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    Buf small = {.ptr = malloc(IO_SECTOR), .size = IO_SECTOR};
    io_read(fd, 100, IO_SECTOR, &small);
    _exit(0);
  }
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "a read into a buffer that is too small ends the program");

  close(fd);
  unlink(path);
  return failures != 0;
}
