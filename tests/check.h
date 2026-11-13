#ifndef EREMITE_TESTS_CHECK_H
#define EREMITE_TESTS_CHECK_H
#include <stdio.h>

// Test results: each check prints ok or FAIL, and a test exits with status 1
// when any check failed.

static int failures;

static inline void check(int ok, const char *what) {
  printf("%s: %s\n", ok ? "ok" : "FAIL", what);
  failures += !ok;
}

#endif
