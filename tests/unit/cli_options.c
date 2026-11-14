#include "check.h"
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests the command line's messages for numbers out of range: each one names
// the range that its option takes. The command line is the eremite program
// beside the tests' folder.

// Runs the command line with an option and its value, and returns what it
// wrote to stderr.
static const char *run_option(const char *cli, const char *option,
                              const char *value) {
  static char said[512];
  int out[2];
  if (pipe(out)) return "";
  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    dup2(out[1], STDERR_FILENO);
    close(out[0]), close(out[1]);
    execl(cli, "eremite", "-m", "/nonexistent", option, value, "-p", "hi",
          (char *)NULL);
    _exit(127);
  }
  close(out[1]);
  ssize_t n = 0, k;
  while (n < (ssize_t)sizeof said - 1 &&
         (k = read(out[0], said + n, sizeof said - 1 - n)) > 0)
    n += k;
  said[n] = 0;
  close(out[0]);
  waitpid(child, NULL, 0);
  return said;
}

int main(int argc, char **argv) {
  (void)argc;
  char folder[PATH_MAX], cli[PATH_MAX];
  snprintf(folder, sizeof folder, "%s", argv[0]);
  snprintf(cli, sizeof cli, "%s/../eremite", dirname(folder));
  const char *said = run_option(cli, "--memory", "2000");
  check(strstr(said, "[0, 1024]") != NULL,
        "--memory names its range of 0 to 1024 GiB");
  said = run_option(cli, "--top-p", "2");
  check(strstr(said, "[0, 1]") != NULL, "--top-p names its range of 0 to 1");
  said = run_option(cli, "--gpu-timeout", "-1");
  check(strstr(said, "[0, infinity)") != NULL,
        "--gpu-timeout names its range of 0 and more");
  return failures != 0;
}
