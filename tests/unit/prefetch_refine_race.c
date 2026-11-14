#include "check.h"
#include "platform/threads.h"
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

// Tests a reader that takes a read-ahead's piece while a refinement decides
// whether to cancel the read-ahead: the refinement leaves the piece to the
// reader and returns without waiting for its read. Hooks on the queue's lock,
// the piece's claim, the read and the load's group let the reader win at a
// fixed point, without depending on timing. Reads come from /dev/zero.

static int watched_lock(pthread_mutex_t *lock);
static char watched_exchange(_Atomic char *object, char value);
static ssize_t watched_pread(int fd, void *buffer, size_t length, off_t offset);
static void watched_wait(ThreadsGroup *group);
static void watched_leave(ThreadsGroup *group);

#define pthread_mutex_lock(lock) watched_lock(lock)
#undef atomic_exchange
#define atomic_exchange(object, value) watched_exchange(object, value)
#define pread(fd, buffer, length, offset)                                      \
  watched_pread(fd, buffer, length, offset)
#define threads_group_wait(group) watched_wait(group)
#define threads_group_leave(group) watched_leave(group)
#include "engine/io.c"
#include "engine/expert_store.c"
#include "store_layout.h"
#undef pthread_mutex_lock
#undef atomic_exchange
#undef pread
#undef threads_group_wait
#undef threads_group_leave

enum { CONTROLLER, REFINER, READER };

static _Thread_local int role;
static pthread_mutex_t control = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static int armed, injection, reader_go, reader_abort, reader_in_io, allow_io,
    refine_done, refine_waited;
static Slot fixture[2];
static double priorities[2];
static ThreadsGroup *watched_group;

// The reader takes the piece at the refinement's first hook: the queue's lock
// for a refinement that decides under it, else the piece's claim after its
// unlocked scan.
static void let_reader_win(int point) {
  if (!armed) return;
  armed = 0, injection = point;
  pthread_mutex_lock(&control);
  reader_go = 1;
  pthread_cond_broadcast(&changed);
  while (!reader_in_io) pthread_cond_wait(&changed, &control);
  pthread_mutex_unlock(&control);
}

static int watched_lock(pthread_mutex_t *lock) {
  if (role == REFINER && lock == &st->qlock) let_reader_win(1);
  return pthread_mutex_lock(lock);
}

static char watched_exchange(_Atomic char *object, char value) {
  if (role == REFINER && object == &fixture[0].taken[0]) let_reader_win(2);
  return atomic_exchange_explicit(object, value, memory_order_seq_cst);
}

static ssize_t watched_pread(int fd, void *buffer, size_t length,
                             off_t offset) {
  if (role == READER && buffer == fixture[0].piece[0].dst) {
    pthread_mutex_lock(&control);
    reader_in_io = 1;
    pthread_cond_broadcast(&changed);
    while (!allow_io) pthread_cond_wait(&changed, &control);
    pthread_mutex_unlock(&control);
  }
  return pread(fd, buffer, length, offset);
}

static void watched_wait(ThreadsGroup *group) {
  if (role == REFINER && group == watched_group) {
    pthread_mutex_lock(&control);
    if (reader_in_io && !allow_io) {
      refine_waited = 1;
      pthread_cond_broadcast(&changed);
    }
    pthread_mutex_unlock(&control);
  }
  threads_group_wait(group);
}

// The real reader's last step on its piece; the fixture's reader thread then
// stops instead of serving other entries.
static void watched_leave(ThreadsGroup *group) {
  threads_group_leave(group);
  if (role == READER && group == watched_group) pthread_exit(NULL);
}

static void *reader_thread(void *unused) {
  (void)unused;
  role = READER;
  pthread_mutex_lock(&control);
  while (!reader_go) pthread_cond_wait(&changed, &control);
  int abort = reader_abort;
  pthread_mutex_unlock(&control);
  if (abort) return NULL;
  return reader(st);
}

static void *refine_thread(void *unused) {
  (void)unused;
  role = REFINER;
  expert_store_cancel_read_aheads(st, 0, NULL, 0);
  pthread_mutex_lock(&control);
  refine_done = 1;
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&control);
  return NULL;
}

int main(void) {
  st = store_create();
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  st->slots = fixture, st->score = priorities;
  st->max_slots = st->cap = st->nslots = 2, st->nfree = st->npinned = 0;
  st->head = 1, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  expert_spans[0] = 1;
  for (int i = 0; i < 2; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(65536), s->b.size = 65536;
    memset(s->b.ptr, 0xa5, s->b.size);
    s->layer = 0, s->expert = i, s->mapped = 1, s->prefetched = 1,
    s->used = st->token;
    s->prev = i ? -1 : 1, s->next = i ? 0 : -1;
    WHERE(0, i) = i, priorities[i] = st->escore[i] = i + 1;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
    start(st, i, 0);
  }
  watched_group = fixture[0].loading;
  armed = 1;
  pthread_t reading, refining;
  if (pthread_create(&reading, NULL, reader_thread, NULL) ||
      pthread_create(&refining, NULL, refine_thread, NULL))
    die("cannot start the race's threads");
  pthread_mutex_lock(&control);
  while (!refine_done && !refine_waited) pthread_cond_wait(&changed, &control);
  int returned = refine_done, waited = refine_waited;
  pthread_mutex_unlock(&control);

  printf("the reader took the piece at %s\n", injection == 1
                                                  ? "the queue's lock"
                                              : injection == 2 ? "the claim"
                                                               : "no hook");
  check(injection && reader_in_io && atomic_load(&fixture[0].taken[0]),
        "the reader took the piece during the refinement");
  check(returned, "the refinement returns before the reader's read ends");
  check(!waited, "the refinement does not wait for the reader");
  check(returned && WHERE(0, 0) == 0 && fixture[0].loading == watched_group &&
            atomic_load(&fixture[0].unfinished) == 1,
        "the started read-ahead stays cached while its read is held");
  check(returned && WHERE(0, 1) == -1 && fixture[1].expert == -1 &&
            !slot_reading(st, 1),
        "the unstarted read-ahead is still cancelled");

  pthread_mutex_lock(&control);
  if (!reader_go)
    reader_abort = reader_go =
        1;  // a missing hook fails the checks without stranding the reader
  allow_io = 1;
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&control);
  pthread_join(reading, NULL);
  pthread_join(refining, NULL);
  check(WHERE(0, 0) == 0 && expert_store_is_ready(st, 0, 0),
        "the reader's read completes the started expert");
  check(st->stats.bytes == 16384, "only the started piece was read");
  check(!atomic_load(&st->prefetching) && !atomic_load(&st->demand),
        "every admission is released");
  for (int i = 0; i < 2; i++) cancel(st, i), free(fixture[i].b.ptr);
  close(fd);
  st->slots = NULL, st->score = NULL;
  return failures != 0;
}
