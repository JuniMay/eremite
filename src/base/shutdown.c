#include "base/shutdown.h"
#include "base/error.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

// Each request counts, so that a second one can skip work that the first
// lets finish.
static volatile sig_atomic_t stop_requested;
// The hook, and the pipe that tells its thread of each request: the read end,
// and the write end that the handlers use.
static void (*request_hook)(void);
static int hook_fd = -1;
static volatile sig_atomic_t request_fd = -1;

void shutdown_request(void) {
  if (stop_requested < 1000) stop_requested++;
  // A signal handler calls this, so it keeps errno. The write end does not
  // block.
  int saved = errno;
  if (request_fd >= 0) (void)!write(request_fd, "", 1);
  errno = saved;
}

static void request_shutdown(int) { shutdown_request(); }

// Runs the hook once for each byte that a request wrote.
static void *run_hook(void *) {
  for (char byte;;) {
    ssize_t n = read(hook_fd, &byte, 1);
    if (n == 1)
      request_hook();
    else if (n == 0 || errno != EINTR)
      return NULL;
  }
}

// Makes the pipe and starts the hook's thread.
static void start_hook(void (*hook)(void)) {
  int fds[2];
  pthread_t thread;
  if (pipe(fds) || fcntl(fds[0], F_SETFD, FD_CLOEXEC) ||
      fcntl(fds[1], F_SETFD, FD_CLOEXEC) || fcntl(fds[1], F_SETFL, O_NONBLOCK))
    die("cannot make the stop's pipe");
  request_hook = hook, hook_fd = fds[0];
  if (pthread_create(&thread, NULL, run_hook, NULL) || pthread_detach(thread))
    die("cannot start the stop's thread");
  request_fd = fds[1];
}

// Installing the handlers again keeps a request that is already pending.
void shutdown_watch(void (*hook)(void)) {
  if (hook && !request_hook) start_hook(hook);
  struct sigaction action = {.sa_handler = request_shutdown,
                             .sa_flags = SA_RESTART};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, 0) || sigaction(SIGTERM, &action, 0))
    die("cannot install clean shutdown handlers");
}

int shutdown_requested(void) { return stop_requested != 0; }

int shutdown_requests(void) { return stop_requested; }
