#ifndef EREMITE_TESTS_UNIT_DEFERRED_READS_H
#define EREMITE_TESTS_UNIT_DEFERRED_READS_H
#include "base/error.h"
#include "platform/threads.h"

// Included before the expert store's source: the demand reads that the store
// starts wait in a list until the test runs them, in the order they started.

enum { DEFERRED_MAX = 1024 };

static struct {
  void *context;
  void (*fn)(void *);
} deferred_calls[DEFERRED_MAX];

static int deferred_count;

static void deferred_async(ThreadsGroup *group, ThreadsPriority priority,
                           void *context, void (*fn)(void *)) {
  (void)group, (void)priority;
  if (deferred_count == DEFERRED_MAX) die("too many deferred reads");
  deferred_calls[deferred_count].context = context,
  deferred_calls[deferred_count++].fn = fn;
}

// Runs the deferred reads.
static void deferred_run(void) {
  for (int i = 0; i < deferred_count; i++)
    deferred_calls[i].fn(deferred_calls[i].context);
  deferred_count = 0;
}

#define threads_async deferred_async

#endif
