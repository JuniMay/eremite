#include "base/file.h"
#include "base/error.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

char *read_file(const char *path, size_t *len) {
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) die("cannot open %s", path);
  char *p = xmalloc(st.st_size + 1);
  for (off_t got = 0; got < st.st_size;) {
    ssize_t r = pread(fd, p + got, st.st_size - got, got);
    if (r <= 0) die("cannot read %s", path);
    got += r;
  }
  close(fd);
  if (len) *len = st.st_size;
  return p;
}
