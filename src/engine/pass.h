#ifndef EREMITE_ENGINE_PASS_H
#define EREMITE_ENGINE_PASS_H
#include "platform/threads.h"
#include <stddef.h>

// A pass is a piece of the engine's work that runs on its own stack and
// yields where it would block, so that one thread can advance several passes.

typedef struct Pass Pass;

// What a pass waits for: ready says without blocking whether the wait is
// over, and block waits until it is. Both run outside the pass. For a wait
// that another thread ends, notify makes the event be signaled when the wait
// may be over. It is null for a wait that the GPU or another pass ends.
enum { PASS_WAIT_OTHER, PASS_WAIT_GPU, PASS_WAIT_READ };

typedef struct {
  int (*ready)(void *context);
  void (*block)(void *context);
  void *context;
  // PASS_WAIT_*: what the wait is for. The engine thread's time is measured by
  // it, and a wait for the GPU beside other passes is looked at again at once.
  int kind;
  void (*notify)(void *context, ThreadsEvent *event);
} PassWait;

// Memory that holds the running pass's values, and the resumer's otherwise.
typedef struct {
  void *address;
  size_t bytes;
} PassVariable;

// Returns a pass that will run body(argument) on a stack of at least
// stack_bytes, whose pages are committed as they are used.
Pass *pass_new(void (*body)(void *), void *argument, size_t stack_bytes);

// Sets the pass's variables: makes them the pass's own, starting from their
// current values.
void pass_set_variables(Pass *pass, const PassVariable *variables, int count);

// Sets a function that runs after each swap of the pass's variables.
void pass_set_resume_callback(Pass *pass, void (*resumed)(void));

void pass_free(Pass *pass);

// Runs the pass until it yields or ends; returns whether it has ended.
int pass_resume(Pass *pass);

// Whether the pass is ready: whether it can continue without blocking.
int pass_ready(const Pass *pass);

// The kind of the pass's wait, PASS_WAIT_OTHER when it waits for nothing.
int pass_waiting_for(const Pass *pass);

// Whether the pass's wait has a background notify callback.
int pass_has_notify(const Pass *pass);

// Sets the kind of the pass's wait until the wait ends: a wait's ready function
// calls it when the wait turns to something else.
void pass_set_wait_kind(Pass *pass, int kind);

// Blocks until the pass can continue.
void pass_block(const Pass *pass);

// Waits until one of the n passes can continue, or the event is signaled.
// If the only pass waits for the GPU, it blocks in that wait: looking would
// keep a core busy. While one of several passes waits for the GPU, it looks at
// them and at the event again at once: a thread that looks notices the GPU's
// end sooner than a blocked thread wakes. Otherwise it has each wait with a
// notify signal the event, and blocks on the event.
// Returns 0 at once if no pass can continue and none waits for the GPU or has
// a notify: the passes then wait only for each other.
int pass_wait_for_any(Pass *const *passes, int n, ThreadsEvent *event);

// Runs the pass to its end, blocking at each of its yields.
void pass_run(Pass *pass);

// The pass running now, or null.
Pass *pass_current(void);

// Inside a pass: yields until the wait is over, unless it already is.
void pass_wait(const PassWait *wait);

// Waits for a read group: inside a pass, by yielding with PASS_WAIT_READ.
void pass_wait_group(ThreadsGroup *group);

// Waits for an off-GPU background group: inside a pass, by yielding with
// PASS_WAIT_OTHER so other models may run freely on the GPU meanwhile.
void pass_wait_background(ThreadsGroup *group);

// When set, runs before a pass yields.
extern void (*pass_before_yield)(void);

#endif
