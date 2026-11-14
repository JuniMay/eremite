#include "base/error.h"
#include "check.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that die runs its hook, prints its message and exits with status 1,
// and that xmalloc returns zeroed memory.

static void hook(void) { fputs("hook\n", stderr); }

int main(void) {
  int pipe_fds[2];
  if (pipe(pipe_fds)) return 1;
  pid_t child = fork();
  if (!child) {
    dup2(pipe_fds[1], 2);
    die_hook = hook;
    die("bad layer %d", 7);
  }
  close(pipe_fds[1]);
  char out[256] = {0};
  size_t got = 0;
  for (ssize_t r; (r = read(pipe_fds[0], out + got, sizeof out - 1 - got)) > 0;)
    got += (size_t)r;
  int status;
  waitpid(child, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
        "die exits with status 1");
  check(!strcmp(out, "hook\neremite: bad layer 7\n"),
        "die runs its hook, then prints its message");

  unsigned char *p = xmalloc(64);
  int zero = 1;
  for (int i = 0; i < 64; i++) zero &= p[i] == 0;
  check(zero, "xmalloc returns zeroed memory");
  free(p);
  p = xmalloc(0);
  check(p != NULL, "xmalloc of no bytes returns memory");
  free(p);
  return failures != 0;
}
