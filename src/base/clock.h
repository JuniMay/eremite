#ifndef EREMITE_BASE_CLOCK_H
#define EREMITE_BASE_CLOCK_H

// Seconds on the monotonic clock, from an arbitrary start.
double now(void);

// Sleeps for seconds on the monotonic clock.
void sleep_seconds(double seconds);

#endif
