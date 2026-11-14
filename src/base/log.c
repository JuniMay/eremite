#include "base/log.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

void (*log_hook)(const char *line);

static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;

void say(const char *fmt, ...) {
  struct timeval tv;
  struct tm tm;
  char t[16];
  gettimeofday(&tv, 0);
  localtime_r(&tv.tv_sec, &tm);
  strftime(t, sizeof t, "%H:%M:%S", &tm);
  va_list ap;
  va_start(ap, fmt);
  pthread_mutex_lock(&log_lock);
  if (log_hook) {
    char line[1024];
    int k =
        snprintf(line, sizeof line, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vsnprintf(line + k, sizeof line - k, fmt, ap);
    log_hook(line);
  } else {
    fprintf(stderr, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
  }
  pthread_mutex_unlock(&log_lock);
  va_end(ap);
}
