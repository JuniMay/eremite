#ifndef EREMITE_BASE_STR_H
#define EREMITE_BASE_STR_H
#include <stddef.h>

// A string that grows as text is appended, always NUL-terminated.
typedef struct {
  char *p;
  size_t n, cap;
} Str;

// Puts n bytes of t at the end.
void str_put(Str *s, const char *t, size_t n);

// Concatenates the NUL-terminated string t.
void str_cat(Str *s, const char *t);

// Prints formatted text at the end.
void str_printf(Str *s, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

#endif
