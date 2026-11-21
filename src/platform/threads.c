#include "platform/threads.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

// In both backends, threads_spawn starts its thread with pthread_create. On
// macOS, a priority sets the thread's quality-of-service class through
// Apple's pthread extensions; on other systems, priorities have no effect.

#ifdef __APPLE__
static qos_class_t qos_class(ThreadsPriority priority) {
  return priority == THREADS_INTERACTIVE ? QOS_CLASS_USER_INTERACTIVE
                                         : QOS_CLASS_USER_INITIATED;
}
#endif

void threads_spawn(ThreadsPriority priority, void *(*fn)(void *),
                   void *context) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
#ifdef __APPLE__
  pthread_attr_set_qos_class_np(&attributes, qos_class(priority), 0);
#endif
  pthread_t thread;
  if (pthread_create(&thread, &attributes, fn, context)) {
    fputs("threads: cannot create a thread\n", stderr);
    abort();
  }
  pthread_attr_destroy(&attributes);
}

struct ThreadsEvent {
  pthread_mutex_t lock;
  pthread_cond_t changed;
  int signaled;
};

ThreadsEvent *threads_event_new(void) {
  ThreadsEvent *event = calloc(1, sizeof *event);
  if (!event) {
    fputs("threads: out of memory\n", stderr);
    abort();
  }
  pthread_mutex_init(&event->lock, NULL);
  pthread_cond_init(&event->changed, NULL);
  return event;
}

void threads_event_free(ThreadsEvent *event) {
  if (!event) return;
  pthread_cond_destroy(&event->changed);
  pthread_mutex_destroy(&event->lock);
  free(event);
}

void threads_event_signal(ThreadsEvent *event) {
  pthread_mutex_lock(&event->lock);
  event->signaled = 1;
  pthread_cond_signal(&event->changed);
  pthread_mutex_unlock(&event->lock);
}

// Nanoseconds in one second for timed condition waits.
static constexpr long NS_PER_SECOND = 1000000000L;

void threads_event_wait(ThreadsEvent *event) {
  pthread_mutex_lock(&event->lock);
  while (!event->signaled) pthread_cond_wait(&event->changed, &event->lock);
  event->signaled = 0;
  pthread_mutex_unlock(&event->lock);
}

int threads_event_wait_for(ThreadsEvent *event, double seconds) {
  if (seconds <= 0.0) return threads_event_take(event);
  pthread_mutex_lock(&event->lock);
  if (!event->signaled) {
    time_t sec = (time_t)seconds;
    long nsec = (long)((seconds - (double)sec) * (double)NS_PER_SECOND);
    if (nsec < 0) nsec = 0;
    if (nsec >= NS_PER_SECOND) sec++, nsec -= NS_PER_SECOND;
#ifdef __APPLE__
    struct timespec rel = {.tv_sec = sec, .tv_nsec = nsec};
    pthread_cond_timedwait_relative_np(&event->changed, &event->lock, &rel);
#else
    struct timespec abs;
    clock_gettime(CLOCK_REALTIME, &abs);
    abs.tv_sec += sec;
    abs.tv_nsec += nsec;
    if (abs.tv_nsec >= NS_PER_SECOND)
      abs.tv_sec++, abs.tv_nsec -= NS_PER_SECOND;
    pthread_cond_timedwait(&event->changed, &event->lock, &abs);
#endif
  }
  int signaled = event->signaled;
  event->signaled = 0;
  pthread_mutex_unlock(&event->lock);
  return signaled;
}

int threads_event_take(ThreadsEvent *event) {
  pthread_mutex_lock(&event->lock);
  int signaled = event->signaled;
  event->signaled = 0;
  pthread_mutex_unlock(&event->lock);
  return signaled;
}

void threads_set_priority(ThreadsPriority priority) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(qos_class(priority), 0);
#else
  (void)priority;
#endif
}
