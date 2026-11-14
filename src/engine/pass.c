#include "engine/pass.h"
#include "base/error.h"
#include "platform/context.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/common_interface_defs.h>
#define PASS_ASAN 1
#endif
#endif

// A pass and the code that resumes it take turns on one thread, so nothing
// here races.

struct Pass {
  // The mapping, and the saved stack pointer while not running.
  void *stack, *sp;
  size_t mapped;
  // The stack above the guard page, and its size.
  char *bottom;
  size_t size;
  // AddressSanitizer's state of the stack while the pass does not run.
  void *fake;
  void (*body)(void *);
  void *argument;
  // What it waits for; ready is null when it waits for nothing.
  PassWait wait;
  int ended;
  // Copied from pass_set_variables' list.
  PassVariable *variables;
  int nvariables;
  // The pass's values while it does not run; the resumer's while it does.
  char *saved, *outer;
  void (*resumed)(void);
};

static Pass *running;
static void *resumer_sp, *resumer_fake;
static const void *resumer_bottom;
static size_t resumer_size;

// Before a switch, tells AddressSanitizer which stack is about to run.
static void fiber_switch_start(void **fake, const void *bottom, size_t size) {
#if PASS_ASAN
  __sanitizer_start_switch_fiber(fake, bottom, size);
#else
  (void)fake, (void)bottom, (void)size;
#endif
}

// After a switch, tells AddressSanitizer that it is done, and stores the
// previous stack's bottom and size unless they are null.
static void fiber_switch_finish(void *fake, const void **bottom, size_t *size) {
#if PASS_ASAN
  __sanitizer_finish_switch_fiber(fake, bottom, size);
#else
  (void)fake, (void)bottom, (void)size;
#endif
}

// A new pass's first resume continues here, on the pass's own stack.
static void pass_entry(void) {
  fiber_switch_finish(NULL, &resumer_bottom, &resumer_size);
  Pass *p = running;
  p->body(p->argument);
  p->ended = 1;
  fiber_switch_start(NULL, resumer_bottom, resumer_size);
  context_switch(&p->sp, resumer_sp);
  die("a finished pass was resumed");
}

// The stack's lowest page stays unmapped, so that an overflow faults.
Pass *pass_new(void (*body)(void *), void *argument, size_t stack_bytes) {
  Pass *p = xmalloc(sizeof *p);
  size_t page = (size_t)getpagesize(),
         bytes = (stack_bytes + page - 1) / page * page + page;
  void *stack =
      mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (stack == MAP_FAILED) die("cannot map a pass's stack");
  if (mprotect(stack, page, PROT_NONE)) die("cannot protect a pass's stack");
  void *sp = context_new((char *)stack + bytes, pass_entry);
  *p = (Pass){.stack = stack,
              .mapped = bytes,
              .bottom = (char *)stack + page,
              .size = bytes - page,
              .sp = sp,
              .body = body,
              .argument = argument};
  return p;
}

void pass_set_variables(Pass *p, const PassVariable *variables, int count) {
  size_t bytes = 0;
  for (int i = 0; i < count; i++) bytes += variables[i].bytes;
  free(p->saved), free(p->outer);
  p->saved = xmalloc(bytes ? bytes : 1), p->outer = xmalloc(bytes ? bytes : 1);
  free(p->variables);
  p->variables = xmalloc((size_t)count * sizeof *p->variables + 1);
  memcpy(p->variables, variables, (size_t)count * sizeof *p->variables);
  p->nvariables = count;
  char *at = p->saved;
  for (int i = 0; i < count; i++)
    memcpy(at, variables[i].address, variables[i].bytes),
        at += variables[i].bytes;
}

void pass_set_resume_callback(Pass *p, void (*resumed)(void)) {
  p->resumed = resumed;
}

void pass_free(Pass *p) {
  if (!p) return;
  if (running == p) die("a pass cannot free itself");
  munmap(p->stack, p->mapped);
  free(p->saved), free(p->outer), free(p->variables);
  free(p);
}

int pass_resume(Pass *p) {
  if (running) die("a pass cannot resume another");
  if (p->ended) die("a finished pass was resumed");
  size_t at = 0;
  for (int i = 0; i < p->nvariables; i++) {
    memcpy(p->outer + at, p->variables[i].address, p->variables[i].bytes);
    memcpy(p->variables[i].address, p->saved + at, p->variables[i].bytes);
    at += p->variables[i].bytes;
  }
  if (p->resumed) p->resumed();
  p->wait = (PassWait){0};
  running = p;
  fiber_switch_start(&resumer_fake, p->bottom, p->size);
  context_switch(&resumer_sp, p->sp);
  fiber_switch_finish(resumer_fake, NULL, NULL);
  running = NULL;
  at = 0;
  for (int i = 0; i < p->nvariables; i++) {
    memcpy(p->saved + at, p->variables[i].address, p->variables[i].bytes);
    memcpy(p->variables[i].address, p->outer + at, p->variables[i].bytes);
    at += p->variables[i].bytes;
  }
  if (p->resumed) p->resumed();
  return p->ended;
}

int pass_ready(const Pass *p) {
  return !p->wait.ready || p->wait.ready(p->wait.context);
}

int pass_waiting_for(const Pass *p) {
  return p->wait.ready ? p->wait.kind : PASS_WAIT_OTHER;
}

void pass_block(const Pass *p) {
  if (p->wait.block) p->wait.block(p->wait.context);
}

void pass_run(Pass *p) {
  while (!pass_resume(p)) pass_block(p);
}

int pass_wait_for_any(Pass *const *passes, int n, ThreadsEvent *event) {
  for (;;) {
    int gpu = 0, notified = 0;
    for (int k = 0; k < n; k++) {
      if (pass_ready(passes[k])) return 1;
      gpu |= pass_waiting_for(passes[k]) == PASS_WAIT_GPU;
      notified |= passes[k]->wait.notify != NULL;
    }
    if (gpu) {
      if (threads_event_take(event)) return 1;
      __asm__ __volatile__("yield");  // a hint to the core while it looks again
      continue;
    }
    if (!notified) return 0;
    for (int k = 0; k < n; k++)
      if (passes[k]->wait.notify)
        passes[k]->wait.notify(passes[k]->wait.context, event);
    threads_event_wait(event);
    return 1;
  }
}

Pass *pass_current(void) { return running; }

void (*pass_before_yield)(void);

void pass_wait(const PassWait *wait) {
  Pass *p = running;
  if (!p) die("pass_wait outside a pass");
  if (wait->ready && wait->ready(wait->context)) return;
  if (pass_before_yield) pass_before_yield();
  p->wait = *wait;
  fiber_switch_start(&p->fake, resumer_bottom, resumer_size);
  context_switch(&p->sp, resumer_sp);
  fiber_switch_finish(p->fake, &resumer_bottom, &resumer_size);
}

static int group_ready(void *group) { return threads_group_done(group); }

static void group_block(void *group) { threads_group_wait(group); }

static void group_notify(void *group, ThreadsEvent *event) {
  threads_group_notify(group, event);
}

void pass_wait_group(ThreadsGroup *group) {
  if (running)
    pass_wait(&(PassWait){group_ready, group_block, group, PASS_WAIT_READ,
                          group_notify});
  else
    group_block(group);
}
