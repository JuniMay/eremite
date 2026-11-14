#include "base/clock.h"
#include "check.h"
#include <math.h>

// Tests that the clock never goes back, that it measures a 20 ms sleep, and
// that a sleep of zero, negative or NaN seconds returns at once.

int main(void) {
  double start = now(), last = start;
  int forward = 1;
  for (int i = 0; i < 100000; i++) {
    double t = now();
    forward &= t >= last;
    last = t;
  }
  check(forward, "the clock never goes back");
  start = now();
  sleep_seconds(0.02);
  double slept = now() - start;
  printf("a 20 ms sleep measured %.1f ms\n", slept * 1000);
  check(slept >= 0.019 && slept < 0.2, "the clock measures a 20 ms sleep");
  start = now();
  sleep_seconds(0), sleep_seconds(-1), sleep_seconds(NAN);
  check(now() - start < slept / 2,
        "a sleep of zero, negative or NaN seconds returns at once");
  return failures != 0;
}
