#if EREMITE_THREADS_PTHREAD
#include "platform/threads.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// The threads layer on POSIX threads.

enum { MAX_WORKERS = 64 };

struct ThreadsGroup {
  pthread_mutex_t lock;
  pthread_cond_t finished;
  long count;
};

// A task waits in its priority's queue until a worker of that priority takes it.
typedef struct Task {
  struct Task *next;
  ThreadsGroup *group;
  void *context;
  void (*fn)(void *);
} Task;

// One priority's queue and its workers. A worker that finds the queue empty
// waits, counted as idle, until a task arrives.
typedef struct {
  pthread_mutex_t lock;
  pthread_cond_t ready;
  Task *head, *tail;
  int waiting;  // tasks no worker has taken
  int workers, idle;
} Pool;

static Pool pools[2] = {
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, 0, 0},
    {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, 0, 0},
};

static void *xmalloc(size_t bytes) {
  void *p = malloc(bytes);
  if (!p) {
    fputs("threads: out of memory\n", stderr);
    abort();
  }
  return p;
}

const char *threads_backend(void) { return "pthread"; }

ThreadsGroup *threads_group_new(void) {
  ThreadsGroup *group = xmalloc(sizeof *group);
  pthread_mutex_init(&group->lock, NULL);
  pthread_cond_init(&group->finished, NULL);
  group->count = 0;
  return group;
}

void threads_group_free(ThreadsGroup *group) {
  pthread_cond_destroy(&group->finished);
  pthread_mutex_destroy(&group->lock);
  free(group);
}

void threads_group_enter(ThreadsGroup *group) {
  pthread_mutex_lock(&group->lock);
  group->count++;
  pthread_mutex_unlock(&group->lock);
}

void threads_group_leave(ThreadsGroup *group) {
  pthread_mutex_lock(&group->lock);
  if (--group->count == 0) pthread_cond_broadcast(&group->finished);
  pthread_mutex_unlock(&group->lock);
}

int threads_group_done(ThreadsGroup *group) {
  pthread_mutex_lock(&group->lock);
  int done = group->count == 0;
  pthread_mutex_unlock(&group->lock);
  return done;
}

void threads_group_wait(ThreadsGroup *group) {
  pthread_mutex_lock(&group->lock);
  while (group->count) pthread_cond_wait(&group->finished, &group->lock);
  pthread_mutex_unlock(&group->lock);
}

static void *worker(void *p) {
  Pool *pool = p;
  for (;;) {
    pthread_mutex_lock(&pool->lock);
    while (!pool->head) {
      pool->idle++;
      pthread_cond_wait(&pool->ready, &pool->lock);
      pool->idle--;
    }
    Task *task = pool->head;
    pool->head = task->next;
    if (!pool->head) pool->tail = NULL;
    pool->waiting--;
    pthread_mutex_unlock(&pool->lock);
    task->fn(task->context);
    if (task->group) threads_group_leave(task->group);
    free(task);
  }
  return NULL;
}

// Queues the task and wakes an idle worker. When more tasks wait than workers
// are idle, the pool also starts another worker, up to MAX_WORKERS, so that
// tasks that block, such as SSD reads, do not hold up the tasks behind them.
// A worker that the signal has woken still counts as idle until it takes a
// task.
void threads_async(ThreadsGroup *group, ThreadsPriority priority, void *context, void (*fn)(void *)) {
  Task *task = xmalloc(sizeof *task);
  *task = (Task){.group = group, .context = context, .fn = fn};
  if (group) threads_group_enter(group);
  Pool *pool = &pools[priority];
  pthread_mutex_lock(&pool->lock);
  if (pool->tail)
    pool->tail->next = task;
  else
    pool->head = task;
  pool->tail = task;
  pool->waiting++;
  int grow = pool->waiting > pool->idle && pool->workers < MAX_WORKERS;
  if (grow) pool->workers++;
  pthread_cond_signal(&pool->ready);
  pthread_mutex_unlock(&pool->lock);
  if (grow) threads_spawn(priority, worker, pool);
}

typedef struct {
  atomic_size_t next;
  size_t n;
  void *context;
  void (*fn)(void *, size_t);
} Loop;

static void run_loop(void *p) {
  Loop *loop = p;
  for (size_t i; (i = atomic_fetch_add(&loop->next, 1)) < loop->n;) loop->fn(loop->context, i);
}

// The calling thread and one helper task for each further CPU take indexes
// from a shared counter until none are left.
void threads_apply(size_t n, ThreadsPriority priority, void *context, void (*fn)(void *, size_t)) {
  if (!n) return;
  Loop loop = {.n = n, .context = context, .fn = fn};
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  size_t helpers = cpus > 1 ? (size_t)cpus - 1 : 0;
  if (helpers > n - 1) helpers = n - 1;
  ThreadsGroup *group = threads_group_new();
  for (size_t k = 0; k < helpers; k++) threads_async(group, priority, &loop, run_loop);
  run_loop(&loop);
  threads_group_wait(group);
  threads_group_free(group);
}
#endif
