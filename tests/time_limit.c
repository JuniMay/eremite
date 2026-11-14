#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Runs a program with a time limit, in a process group of its own: after the
// given seconds, SIGKILL ends the group, the program's children too. 0 sets no
// limit. SIGINT and SIGTERM pass on to the group. It exits with the program's
// status, or with 124 when the limit ended the program.
// usage: time-limit SECONDS PROGRAM [ARGUMENTS...]

// The program's process group, and whether the limit ended it.
static volatile sig_atomic_t group, expired;

static void end_group(int signal_number) {
  if (signal_number == SIGALRM) expired = 1;
  if (group > 0)
    kill(-group, signal_number == SIGALRM ? SIGKILL : signal_number);
}

int main(int argc, char **argv) {
  char *end = NULL;
  long seconds = argc >= 3 ? strtol(argv[1], &end, 10) : -1;
  if (argc < 3 || *end || seconds < 0) {
    fprintf(stderr, "usage: time-limit SECONDS PROGRAM [ARGUMENTS...]\n");
    return 2;
  }
  pid_t child = fork();
  if (child < 0) {
    perror("fork");
    return 2;
  }
  if (child == 0) {
    setpgid(0, 0);
    execvp(argv[2], argv + 2);
    perror(argv[2]);
    _exit(127);
  }
  // Both sides set the group, so that it exists before a signal comes.
  setpgid(child, child);
  group = child;
  struct sigaction action = {.sa_handler = end_group};
  sigemptyset(&action.sa_mask);
  sigaction(SIGALRM, &action, NULL);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
  alarm((unsigned)seconds);
  int status;
  while (waitpid(child, &status, 0) < 0)
    if (errno != EINTR) {
      perror("waitpid");
      return 2;
    }
  if (expired) {
    fprintf(stderr, "time-limit: %s ran past %ld s\n", argv[2], seconds);
    return 124;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
