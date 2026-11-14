#include "base/str.h"
#include "base/error.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void str_put(Str *s, const char *t, size_t n) {
  if (s->n + n + 1 > s->cap) {
    s->cap = (s->n + n + 1) * 2;
    if (!(s->p = realloc(s->p, s->cap)))
      die("out of memory (%zu bytes)", s->cap);
  }
  memcpy(s->p + s->n, t, n);
  s->n += n;
  s->p[s->n] = 0;
}

void str_cat(Str *s, const char *t) { str_put(s, t, strlen(t)); }

void str_printf(Str *s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(0, 0, fmt, ap);
  va_end(ap);
  str_put(s, "", 0);
  if (s->n + n + 1 > s->cap &&
      !(s->p = realloc(s->p, s->cap = (s->n + n + 1) * 2)))
    die("out of memory");
  va_start(ap, fmt);
  vsnprintf(s->p + s->n, n + 1, fmt, ap);
  va_end(ap);
  s->n += n;
}
