#include "platform/threads.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

// Tests the threads layer with the backend this build selects.

static int failures;

static void check(int ok, const char *what) {
  printf("%s: %s\n", ok ? "ok" : "FAIL", what);
  failures += !ok;
}

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static atomic_int counter;
static void count(void *) { atomic_fetch_add(&counter, 1); }
static void add_index(void *context, size_t i) { atomic_fetch_add((atomic_size_t *)context, i); }
static void block(void *context) { usleep((useconds_t)(uintptr_t)context); }
static void *mark(void *context) {
  atomic_store((atomic_int *)context, 1);
  return NULL;
}

int main(void) {
  printf("backend: %s\n", threads_backend());
  ThreadsGroup *group = threads_group_new();
  check(threads_group_done(group), "a new group has nothing to wait for");
  threads_group_enter(group);
  check(!threads_group_done(group), "a group with an entered task is not done");
  threads_group_leave(group);
  check(threads_group_done(group), "a group whose tasks all left is done");

  for (int i = 0; i < 1000; i++) threads_async(group, i % 2 ? THREADS_INTERACTIVE : THREADS_INITIATED, NULL, count);
  threads_group_wait(group);
  check(atomic_load(&counter) == 1000, "waiting for a group returns after all of its 1,000 tasks ran");

  atomic_size_t sum = 0;
  threads_apply(10000, THREADS_INITIATED, &sum, add_index);
  check(atomic_load(&sum) == (size_t)10000 * 9999 / 2, "a parallel loop runs each of 10,000 indexes once");
  threads_apply(0, THREADS_INITIATED, &sum, add_index);
  check(atomic_load(&sum) == (size_t)10000 * 9999 / 2, "a parallel loop of no indexes runs nothing");

  double start = seconds();
  for (int i = 0; i < 32; i++) threads_async(group, THREADS_INITIATED, (void *)(uintptr_t)50000, block);
  threads_group_wait(group);
  double took = seconds() - start;
  printf("32 tasks that each block for 50 ms took %.0f ms\n", took * 1000);
  check(took < 0.4, "tasks that block run side by side");

  atomic_int started = 0;
  threads_spawn(THREADS_INITIATED, mark, &started);
  for (int i = 0; i < 1000 && !atomic_load(&started); i++) usleep(1000);
  check(atomic_load(&started), "threads_spawn starts a thread");

  threads_set_priority(THREADS_INTERACTIVE);
#ifdef __APPLE__
  check(qos_class_self() == QOS_CLASS_USER_INTERACTIVE, "the calling thread takes the interactive class");
#endif
  threads_group_free(group);
  return failures != 0;
}
