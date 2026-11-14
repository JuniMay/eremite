#include "check.h"
#include "platform/threads.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

// Tests the threads layer with the backend this build selects.

// A spawned thread's mark, and the event it signals after setting it.
typedef struct {
  atomic_int ran;
  ThreadsEvent *event;
} Spawned;

static atomic_int counter;

static void count(void *) { atomic_fetch_add(&counter, 1); }

static void add_index(void *context, size_t i) {
  atomic_fetch_add((atomic_size_t *)context, i);
}

// Leaves the group, then blocks until every task has left it. The tasks end
// only if they run side by side.
static void meet(void *group) {
  threads_group_leave(group);
  threads_group_wait(group);
}

static void leave(void *group) { threads_group_leave(group); }

static void signal_event(void *event) { threads_event_signal(event); }

static void *mark(void *context) {
  Spawned *spawned = context;
  atomic_store(&spawned->ran, 1);
  threads_event_signal(spawned->event);
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

  for (int i = 0; i < 1000; i++)
    threads_async(group, i % 2 ? THREADS_INTERACTIVE : THREADS_INITIATED, NULL,
                  count);
  threads_group_wait(group);
  check(atomic_load(&counter) == 1000,
        "waiting for a group returns after all of its 1,000 tasks ran");

  atomic_size_t sum = 0;
  threads_apply(10000, THREADS_INITIATED, &sum, add_index);
  check(atomic_load(&sum) == (size_t)10000 * 9999 / 2,
        "a parallel loop runs each of 10,000 indexes once");
  threads_apply(0, THREADS_INITIATED, &sum, add_index);
  check(atomic_load(&sum) == (size_t)10000 * 9999 / 2,
        "a parallel loop of no indexes runs nothing");

  ThreadsGroup *meeting = threads_group_new();
  for (int i = 0; i < 32; i++) threads_group_enter(meeting);
  for (int i = 0; i < 32; i++)
    threads_async(group, THREADS_INITIATED, meeting, meet);
  threads_group_wait(group);
  check(threads_group_done(meeting),
        "tasks that block run side by side: 32 tasks each wait until all 32 "
        "have started");
  threads_group_free(meeting);

  ThreadsEvent *event = threads_event_new();
  check(!threads_event_take(event), "a new event has no signal to take");
  threads_event_signal(event);
  threads_event_signal(event);
  check(threads_event_take(event) && !threads_event_take(event),
        "two signals before a take count as one");
  threads_event_signal(event);
  threads_event_wait(event);
  check(!threads_event_take(event),
        "a signal that comes before the wait stays for it, and the wait "
        "takes it");
  threads_async(NULL, THREADS_INITIATED, event, signal_event);
  threads_event_wait(event);
  check(1, "a wait ends when another thread signals the event");

  ThreadsGroup *notified = threads_group_new();
  threads_group_notify(notified, event);
  threads_event_wait(event);
  check(1, "a group whose count is zero signals the event soon");
  threads_group_enter(notified);
  threads_group_notify(notified, event);
  threads_async(NULL, THREADS_INITIATED, notified, leave);
  threads_event_wait(event);
  check(threads_group_done(notified),
        "a group signals the event once its last task has left");
  threads_group_free(notified);

  Spawned spawned = {.event = event};
  threads_spawn(THREADS_INITIATED, mark, &spawned);
  threads_event_wait(event);
  check(atomic_load(&spawned.ran), "threads_spawn starts a thread");
  threads_event_free(event);

  threads_set_priority(THREADS_INTERACTIVE);
#ifdef __APPLE__
  check(qos_class_self() == QOS_CLASS_USER_INTERACTIVE,
        "the calling thread takes the interactive class");
#endif
  threads_group_free(group);
  return failures != 0;
}
