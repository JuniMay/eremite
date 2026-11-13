#include "base/error.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void (*die_hook)(void);

[[noreturn]] void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (die_hook) die_hook();
  fputs("eremite: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  exit(1);
}

void *xmalloc(size_t n) {
  void *p = calloc(1, n ? n : 1);
  if (!p) die("out of memory (%zu bytes)", n);
  return p;
}
