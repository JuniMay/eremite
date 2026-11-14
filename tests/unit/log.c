#include "base/log.h"
#include "check.h"
#include <stdio.h>
#include <string.h>

// Tests that a log line is the time of day followed by the message.

static char last[1024];

static void keep(const char *line) { snprintf(last, sizeof last, "%s", line); }

int main(void) {
  log_hook = keep;
  say("loaded %d layers", 40);
  printf("line: %s\n", last);
  check(strlen(last) == 13 + strlen("loaded 40 layers") && last[2] == ':' &&
            last[5] == ':' && last[8] == '.' && last[12] == ' ',
        "a line starts with HH:MM:SS.mmm and a space");
  check(!strcmp(last + 13, "loaded 40 layers"),
        "the formatted message follows the time");
  return failures != 0;
}
