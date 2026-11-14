#ifndef EREMITE_TESTS_GPU_COPY_H
#define EREMITE_TESTS_GPU_COPY_H
#include "base/error.h"
#include <spawn.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Runs the cases that must end the program in a copy of the test program,
// because Metal does not work after fork.

extern char **environ;

// Runs a copy of this program with the argument how. Returns whether the copy
// ended with an error within 20 s, saying expected. Every case ends within a
// second; a copy still running after 20 s is stopped.
static inline int ends_with(const char *self, const char *how,
                            const char *expected) {
  int out[2];
  if (pipe(out)) die("pipe");
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out[1], 1);
  posix_spawn_file_actions_adddup2(&actions, out[1], 2);
  posix_spawn_file_actions_addclose(&actions, out[0]);
  char *args[] = {(char *)self, (char *)how, NULL};
  pid_t pid;
  if (posix_spawn(&pid, self, &actions, NULL, args, environ))
    die("cannot run %s", self);
  posix_spawn_file_actions_destroy(&actions);
  close(out[1]);
  int status = 0, done = waitpid(pid, &status, 0) == pid;
  char said[512] = {0};
  for (ssize_t n = 0, k; n + 1 < (ssize_t)sizeof said &&
                         (k = read(out[0], said + n, sizeof said - 1 - n)) > 0;)
    n += k;
  close(out[0]);
  return done && WIFEXITED(status) && WEXITSTATUS(status) != 0 &&
         strstr(said, expected);
}

#endif
