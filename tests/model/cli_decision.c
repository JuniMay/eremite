#include "check.h"
#include "base/error.h"
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that the command line refuses a decision model's checkpoint: it
// generates text, and a decision model generates none. The program ends with
// a message that says so, and does not crash. The command line is the eremite
// program beside the tests' folder.
//
// usage: model-cli_decision CHECKPOINT

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-cli_decision CHECKPOINT");
  char folder[PATH_MAX], cli[PATH_MAX];
  snprintf(folder, sizeof folder, "%s", argv[0]);
  snprintf(cli, sizeof cli, "%s/../eremite", dirname(folder));
  int out[2];
  if (pipe(out)) die("pipe failed");
  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    dup2(out[1], STDERR_FILENO);
    close(out[0]), close(out[1]);
    execl(cli, "eremite", "-m", argv[1], "--raw", "-p", "hi", (char *)NULL);
    _exit(127);
  }
  close(out[1]);
  static char said[4096];
  ssize_t n = 0, k;
  while (n < (ssize_t)sizeof said - 1 &&
         (k = read(out[0], said + n, sizeof said - 1 - n)) > 0)
    n += k;
  said[n] = 0;
  close(out[0]);
  int status = 0;
  waitpid(child, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
        "the command line ends with status 1, without a crash");
  check(strstr(said, "decision model") != NULL,
        "its message says that the checkpoint is a decision model's");
  return failures != 0;
}
