#include "base/clock.h"
#include <errno.h>
#include <time.h>

double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

void sleep_seconds(double seconds) {
  if (!(seconds > 0)) return;
  struct timespec left = {.tv_sec = (time_t)seconds,
                          .tv_nsec = (long)((seconds - (time_t)seconds) * 1e9)};
  while (nanosleep(&left, &left) && errno == EINTR) {}
}
