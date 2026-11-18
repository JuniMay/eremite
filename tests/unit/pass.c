#include "base/clock.h"
#include "check.h"
#include "engine/pass.h"
#include <string.h>

// Tests passes on their own stacks: interleaving, variables, stack depth,
// waiting for a group, waiting until any of several passes can continue, and
// what a yield and a resume cost.

// The sum of the ids of the passes that ran.
static int counter;
// The variable that each pass makes its own.
static int shared;
// The yields that measure what a yield and a resume cost.
static constexpr int SWITCHES = 2000000;

static int always_ready(void *) { return 1; }

static int flag_ready(void *flag) { return *(int *)flag; }

// The wait ends when the resumer blocks.
static void flag_block(void *flag) { *(int *)flag = 1; }

typedef struct {
  int id, yields, flag;
  char *log;
} Args;

// Yields `yields` times, appending its id each time.
static void worker(void *argument) {
  Args *a = argument;
  volatile double kept = a->id * 1.5;
  for (int i = 0; i < a->yields; i++) {
    a->flag = 0;
    pass_wait(&(PassWait){
        .ready = flag_ready, .block = flag_block, .context = &a->flag});
    a->log[strlen(a->log)] = (char)('0' + a->id);
    counter += a->id;
  }
  check(kept == a->id * 1.5, "a floating-point register survives the yields");
}

static void waits_for_nothing(void *argument) {
  Pass *self = pass_current();
  pass_wait(&(PassWait){.ready = always_ready});
  check(pass_current() == self, "a wait that is already over does not yield");
  *(int *)argument = 1;
}

static int deep(int depth) {
  volatile char block[64 * 1024];
  block[0] = (char)depth;
  return depth ? deep(depth - 1) + block[0] : block[0];
}

static void deep_body(void *argument) { *(int *)argument = deep(100); }

static void uses_variable(void *argument) {
  int value = *(int *)argument;
  check(shared == 0, "a pass starts from the value its variable had");
  shared = value;
  int never = 0;
  pass_wait(
      &(PassWait){.ready = flag_ready, .block = flag_block, .context = &never});
  check(shared == value,
        "a pass does not see what another pass wrote to its variable");
}

static void leave_group(void *group) { threads_group_leave(group); }

static void signal_event(void *event) { threads_event_signal(event); }

static void waits_for_group(void *group) { pass_wait_group(group); }

// Over at its third look, for a wait that stands for the GPU's.
static int third_look(void *looks) { return ++*(int *)looks >= 3; }

static void waits_for_gpu(void *looks) {
  pass_wait(&(PassWait){
      .ready = third_look, .context = looks, .kind = PASS_WAIT_GPU});
}

// The looks after which a lone pass's wait for the GPU is over without a
// block: enough that a thread that looks again instead of blocking is caught.
static constexpr int LOOKS_UNTIL_OVER = 1000;

// A lone pass's wait for the GPU: its block ends it.
typedef struct {
  int looks, blocked;
} LoneGpuWait;

static int lone_gpu_wait_over(void *wait) {
  LoneGpuWait *w = wait;
  return w->blocked || ++w->looks > LOOKS_UNTIL_OVER;
}

static void block_lone_gpu_wait(void *wait) {
  ((LoneGpuWait *)wait)->blocked = 1;
}

static void waits_for_gpu_alone(void *wait) {
  pass_wait(&(PassWait){.ready = lone_gpu_wait_over,
                        .block = block_lone_gpu_wait,
                        .context = wait,
                        .kind = PASS_WAIT_GPU});
}

static int never_ready(void *) { return 0; }

// A wait that only another pass could end, and that no pass ends.
static void waits_for_another_pass(void *) {
  pass_wait(&(PassWait){.ready = never_ready, .kind = PASS_WAIT_OTHER});
}

static void yielding(void *) {
  int flag = 0;
  for (int i = 0; i < SWITCHES; i++)
    flag = 0, pass_wait(&(PassWait){.ready = flag_ready, .context = &flag});
}

int main(void) {
  char log[32] = {0};
  Args a = {1, 3, 0, log}, b = {2, 3, 0, log};
  Pass *p = pass_new(worker, &a, 1 << 20), *q = pass_new(worker, &b, 1 << 20);
  check(!pass_current(), "no pass runs outside a resume");
  int ends = 0;
  for (int round = 0; round < 4; round++) {
    ends += pass_resume(p);
    pass_block(p);
    ends += pass_resume(q);
    pass_block(q);
  }
  check(!strcmp(log, "121212") && counter == 9 && ends == 2,
        "two passes take turns, each where it yielded");
  pass_free(p), pass_free(q);

  int done = 0;
  p = pass_new(waits_for_nothing, &done, 1 << 20);
  check(pass_resume(p) == 1 && done == 1,
        "a pass that never yields ends in one resume");
  pass_free(p);

  int deep_result = -1;
  p = pass_new(deep_body, &deep_result, 16u << 20);
  pass_run(p);
  check(deep_result > 0,
        "a pass recurses through about 6 MiB of its own 16 MiB stack");
  pass_free(p);

  shared = 0;
  int one = 11, two = 22;
  static const PassVariable variables[] = {{&shared, sizeof shared}};
  p = pass_new(uses_variable, &one, 1 << 20),
  q = pass_new(uses_variable, &two, 1 << 20);
  pass_set_variables(p, variables, 1), pass_set_variables(q, variables, 1);
  pass_resume(p), pass_resume(q);
  pass_block(p), pass_block(q);
  check(pass_resume(p) == 1 && pass_resume(q) == 1 && shared == 0,
        "the resumer's value is back after each resume");
  pass_free(p), pass_free(q);

  ThreadsGroup *group = threads_group_new();
  threads_group_enter(group);
  p = pass_new(waits_for_group, group, 1 << 20);
  check(!pass_resume(p) && !pass_ready(p) &&
            pass_waiting_for(p) == PASS_WAIT_READ,
        "a pass yields while its group has an unfinished task");
  threads_async(NULL, THREADS_INITIATED, group, leave_group);
  pass_block(p);
  check(pass_ready(p) && pass_resume(p) == 1,
        "the pass ends once the group's task has finished");
  pass_free(p);

  ThreadsEvent *event = threads_event_new();
  threads_group_enter(group);
  p = pass_new(waits_for_group, group, 1 << 20);
  pass_resume(p);
  threads_async(NULL, THREADS_INITIATED, group, leave_group);
  check(pass_wait_for_any(&p, 1, event) && pass_ready(p) && pass_resume(p) == 1,
        "a wait for any pass ends when another thread finishes the group that "
        "a pass waits for");
  pass_free(p);

  threads_group_enter(group);
  p = pass_new(waits_for_group, group, 1 << 20);
  pass_resume(p);
  threads_async(NULL, THREADS_INITIATED, event, signal_event);
  check(pass_wait_for_any(&p, 1, event) && !pass_ready(p),
        "a wait for any pass ends when another thread signals the event");
  threads_group_leave(group);
  // The group's notify still signals the event: the next wait takes it.
  threads_event_wait(event);
  pass_block(p);
  check(pass_resume(p) == 1, "the pass then ends once its group is done");
  pass_free(p);
  threads_group_free(group);

  int looks = 0;
  Pass *pair[2] = {pass_new(waits_for_gpu, &looks, 1 << 20),
                   pass_new(waits_for_another_pass, NULL, 1 << 20)};
  pass_resume(pair[0]), pass_resume(pair[1]);
  check(pass_wait_for_any(pair, 2, event) && looks == 3 && pass_ready(pair[0]),
        "while a pass waits for the GPU, a wait for any pass looks again "
        "until the GPU's wait is over");
  check(pass_resume(pair[0]) == 1 && !pass_wait_for_any(pair + 1, 1, event),
        "a wait for any pass returns 0 when the passes wait only for each "
        "other");
  pass_free(pair[0]), pass_free(pair[1]);

  LoneGpuWait alone = {0};
  p = pass_new(waits_for_gpu_alone, &alone, 1 << 20);
  pass_resume(p);
  check(pass_wait_for_any(&p, 1, event) && alone.blocked,
        "a wait for any pass blocks in the GPU's wait of the only pass");
  check(pass_resume(p) == 1, "the pass then ends");
  pass_free(p);
  threads_event_free(event);

  p = pass_new(yielding, NULL, 1 << 20);
  double start = now();
  while (!pass_resume(p)) {}
  printf("a yield and a resume take %.0f ns\n",
         (now() - start) / SWITCHES * 1e9);
  pass_free(p);
  return failures != 0;
}
