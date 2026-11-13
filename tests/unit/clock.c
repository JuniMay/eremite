#include "base/clock.h"
#include "check.h"
#include <unistd.h>

// Tests that the clock never goes back and measures a 20 ms sleep.

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
  usleep(20000);
  double slept = now() - start;
  printf("a 20 ms sleep measured %.1f ms\n", slept * 1000);
  check(slept >= 0.019 && slept < 0.2, "the clock measures a 20 ms sleep");
  return failures != 0;
}
