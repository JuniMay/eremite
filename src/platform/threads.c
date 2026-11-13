#include "platform/threads.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

// In both backends, threads_spawn starts its thread with pthread_create. On
// macOS, a priority sets the thread's quality-of-service class through
// Apple's pthread extensions; on other systems, priorities have no effect.

#ifdef __APPLE__
static qos_class_t qos_class(ThreadsPriority priority) {
  return priority == THREADS_INTERACTIVE ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED;
}
#endif

void threads_spawn(ThreadsPriority priority, void *(*fn)(void *), void *context) {
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

void threads_set_priority(ThreadsPriority priority) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(qos_class(priority), 0);
#else
  (void)priority;
#endif
}
