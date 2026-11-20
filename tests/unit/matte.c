#include "engine/matte.h"
#include <stdio.h>
#include <string.h>

// Alpha mattes: keeping an original's pixels where a mask is opaque.

// The number of failed checks.
static int failures;

// Prints label with ok or FAIL by cond, and counts the failures.
static void check(int cond, const char *label) {
  printf("%s: %s\n", cond ? "ok" : "FAIL", label);
  if (!cond) failures++;
}

int main(void) {
  // Three pixels: the mask keeps the first, changes the second, and mixes the
  // third halfway.
  uint8_t edited[] = {10, 20, 30, 40, 50, 60, 200, 100, 0};
  const uint8_t original[] = {110, 120, 130, 140, 150, 160, 0, 100, 255};
  const uint8_t mask[] = {255, 0, 128};
  matte_keep_masked(edited, original, mask, 3);
  const uint8_t expected[] = {110, 120, 130, 40, 50, 60, 100, 100, 128};
  check(!memcmp(edited, expected, sizeof expected),
        "an opaque mask keeps the original, a clear one keeps the edit, and "
        "half alpha mixes them");
  return failures != 0;
}
