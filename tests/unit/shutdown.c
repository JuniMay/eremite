#include "base/shutdown.h"
#include "check.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that SIGINT and SIGTERM request a stop, that installing the handlers
// again keeps the request, and that requests are counted. After each request,
// the hook runs once, on a thread of its own.

// The hook's runs and the thread of the last one, under the lock.
static pthread_mutex_t hook_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t hook_ran = PTHREAD_COND_INITIALIZER;
static int hook_runs;
static pthread_t hook_thread;

static void count_run(void) {
  pthread_mutex_lock(&hook_lock);
  hook_runs++, hook_thread = pthread_self();
  pthread_cond_signal(&hook_ran);
  pthread_mutex_unlock(&hook_lock);
}

// Waits until the hook has run that many times. Returns its runs.
static int wait_for_runs(int runs) {
  pthread_mutex_lock(&hook_lock);
  while (hook_runs < runs) pthread_cond_wait(&hook_ran, &hook_lock);
  int done = hook_runs;
  pthread_mutex_unlock(&hook_lock);
  return done;
}

static int child(int signal_number) {
  shutdown_watch(count_run);
  check(!shutdown_requested(), "no stop is requested at first");
  check(raise(signal_number) == 0 && shutdown_requested(),
        "the signal requests a stop");
  wait_for_runs(1);
  check(!pthread_equal(hook_thread, pthread_self()),
        "the hook runs after the signal, on a thread of its own");
  shutdown_watch(NULL);
  check(shutdown_requested(),
        "installing the handlers again keeps the request");
  shutdown_request();
  check(shutdown_requests() == 2, "each request counts");
  check(wait_for_runs(2) == 2, "the hook runs once for each request");
  fflush(stdout);
  return failures;
}

int main(void) {
  const int signals[] = {SIGINT, SIGTERM};
  const char *names[] = {"SIGINT", "SIGTERM"};
  for (int i = 0; i < 2; i++) {
    printf("%s:\n", names[i]);
    fflush(stdout);
    pid_t pid = fork();
    if (!pid) _exit(child(signals[i]));
    int status;
    waitpid(pid, &status, 0);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "every check in the child passed");
  }
  return failures != 0;
}
