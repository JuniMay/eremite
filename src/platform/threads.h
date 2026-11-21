#ifndef EREMITE_PLATFORM_THREADS_H
#define EREMITE_PLATFORM_THREADS_H
#include <stddef.h>

#if EREMITE_THREADS_GCD
#include <dispatch/dispatch.h>
#define THREADS_API static inline
#elif EREMITE_THREADS_PTHREAD
#define THREADS_API
#else
#error "define EREMITE_THREADS_GCD or EREMITE_THREADS_PTHREAD"
#endif

// Running tasks on background threads and waiting for them: background
// tasks, parallel loops, and groups that count tasks so that a caller can
// wait for them. The backend is chosen at build time: Grand Central Dispatch
// (EREMITE_THREADS_GCD), whose calls are inline below, or a pool of POSIX
// threads (EREMITE_THREADS_PTHREAD, threads_pthread.c).

// Interactive tasks are what a decoding token waits for; initiated tasks are
// what a prompt or a load waits for.
typedef enum { THREADS_INTERACTIVE, THREADS_INITIATED } ThreadsPriority;

// A count of unfinished tasks that a caller can wait for.
typedef struct ThreadsGroup ThreadsGroup;

// Returns the backend's name: "gcd" or "pthread".
THREADS_API const char *threads_backend(void);

// Returns a new group whose count is zero.
THREADS_API ThreadsGroup *threads_group_new(void);

// Frees the group.
THREADS_API void threads_group_free(ThreadsGroup *group);

// Enters the group: adds one to its count.
THREADS_API void threads_group_enter(ThreadsGroup *group);

// Leaves the group: takes one from its count.
THREADS_API void threads_group_leave(ThreadsGroup *group);
// Returns whether the group is done: whether its count is zero. Does not
// block.
THREADS_API int threads_group_done(ThreadsGroup *group);
// Blocks until the count is zero.
THREADS_API void threads_group_wait(ThreadsGroup *group);

// An event that one thread waits for and other threads signal. A signal that
// comes while no thread waits stays for the next wait, and several signals
// before a wait count as one. So after each wait, the waiter looks again at
// what it waits for.
typedef struct ThreadsEvent ThreadsEvent;

// Returns a new event that is not signaled.
ThreadsEvent *threads_event_new(void);
// Frees the event. No group may still have to signal it.
void threads_event_free(ThreadsEvent *event);
// Signals the event. Any thread may call it.
void threads_event_signal(ThreadsEvent *event);
// Blocks until the event is signaled, then takes the signal.
void threads_event_wait(ThreadsEvent *event);
// Blocks until the event is signaled or seconds elapse, then takes any signal.
// Returns 1 if signaled, or 0 if seconds elapsed first.
int threads_event_wait_for(ThreadsEvent *event, double seconds);
// Takes the event's signal if there is one, without blocking. Returns whether
// there was one.
int threads_event_take(ThreadsEvent *event);
// Signals the event once the group's count is zero, on another thread. If the
// count is zero already, it signals the event soon.
THREADS_API void threads_group_notify(ThreadsGroup *group, ThreadsEvent *event);

// Calls fn(context) once on a background thread and returns without waiting
// for it. Unless group is null, group counts the call until fn returns.
THREADS_API void threads_async(ThreadsGroup *group, ThreadsPriority priority,
                               void *context, void (*fn)(void *));

// Applies fn to each index: calls fn(context, i) once for each i from 0 to
// n - 1, and blocks until every call has returned.
THREADS_API void threads_apply(size_t n, ThreadsPriority priority,
                               void *context, void (*fn)(void *, size_t));

// Spawns a thread: starts a new thread that runs fn(context).
void threads_spawn(ThreadsPriority priority, void *(*fn)(void *),
                   void *context);

// Sets the calling thread's priority.
void threads_set_priority(ThreadsPriority priority);

#if EREMITE_THREADS_GCD
// A group is a dispatch group, and a priority selects a global queue.
static inline dispatch_group_t threads_dispatch(ThreadsGroup *group) {
  return (dispatch_group_t)(void *)group;
}

static inline dispatch_queue_t threads_queue(ThreadsPriority priority) {
  return dispatch_get_global_queue(priority == THREADS_INTERACTIVE
                                       ? QOS_CLASS_USER_INTERACTIVE
                                       : QOS_CLASS_USER_INITIATED,
                                   0);
}

THREADS_API const char *threads_backend(void) { return "gcd"; }

THREADS_API ThreadsGroup *threads_group_new(void) {
  return (ThreadsGroup *)(void *)dispatch_group_create();
}

THREADS_API void threads_group_free(ThreadsGroup *group) {
  dispatch_release(threads_dispatch(group));
}

THREADS_API void threads_group_enter(ThreadsGroup *group) {
  dispatch_group_enter(threads_dispatch(group));
}

THREADS_API void threads_group_leave(ThreadsGroup *group) {
  dispatch_group_leave(threads_dispatch(group));
}

THREADS_API int threads_group_done(ThreadsGroup *group) {
  return dispatch_group_wait(threads_dispatch(group), DISPATCH_TIME_NOW) == 0;
}

THREADS_API void threads_group_wait(ThreadsGroup *group) {
  dispatch_group_wait(threads_dispatch(group), DISPATCH_TIME_FOREVER);
}

// Signals the event that threads_group_notify passes.
static inline void threads_signal_event(void *event) {
  threads_event_signal(event);
}

THREADS_API void threads_group_notify(ThreadsGroup *group,
                                      ThreadsEvent *event) {
  dispatch_group_notify_f(threads_dispatch(group),
                          threads_queue(THREADS_INTERACTIVE), event,
                          threads_signal_event);
}

THREADS_API void threads_async(ThreadsGroup *group, ThreadsPriority priority,
                               void *context, void (*fn)(void *)) {
  if (group)
    dispatch_group_async_f(threads_dispatch(group), threads_queue(priority),
                           context, fn);
  else
    dispatch_async_f(threads_queue(priority), context, fn);
}

THREADS_API void threads_apply(size_t n, ThreadsPriority priority,
                               void *context, void (*fn)(void *, size_t)) {
  dispatch_apply_f(n, threads_queue(priority), context, fn);
}
#endif

#endif
