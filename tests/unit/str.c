#include "base/str.h"
#include "check.h"
#include <stdlib.h>
#include <string.h>

// Tests that a Str keeps every appended piece in order, NUL-terminated, as it
// grows.

int main(void) {
  Str s = {0};
  str_cat(&s, "layer");
  str_put(&s, " 12 of 40", 3);
  str_printf(&s, ": %d experts", 6);
  check(s.n == strlen("layer 12: 6 experts") &&
            !strcmp(s.p, "layer 12: 6 experts"),
        "pieces append in order");
  for (int i = 0; i < 1000; i++) str_printf(&s, "%d,", i % 10);
  int ok = s.n == strlen("layer 12: 6 experts") + 2000 && s.p[s.n] == 0;
  for (int i = 0; i < 1000 && ok; i++)
    ok = s.p[19 + 2 * i] == '0' + i % 10 && s.p[20 + 2 * i] == ',';
  check(ok, "1,000 more pieces grow it without losing a byte");
  free(s.p);
  return failures != 0;
}
