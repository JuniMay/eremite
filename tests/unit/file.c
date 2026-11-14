#include "base/file.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Tests that read_file returns every byte of a file, its length and a
// trailing NUL.

int main(void) {
  char path[] = "/tmp/eremite-file-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) return 1;
  const char bytes[] = {'a', 'b', 0, 'c'};
  if (write(fd, bytes, sizeof bytes) != (ssize_t)sizeof bytes) return 1;
  close(fd);
  size_t len = 0;
  char *p = read_file(path, &len);
  unlink(path);
  check(len == sizeof bytes && !memcmp(p, bytes, sizeof bytes),
        "every byte comes back, a NUL among them");
  check(p[len] == 0, "a NUL follows the bytes");
  free(p);
  return failures != 0;
}
