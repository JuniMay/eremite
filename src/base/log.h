#ifndef EREMITE_BASE_LOG_H
#define EREMITE_BASE_LOG_H

// Timestamped log lines.

// When set, receives each line instead of stderr.
extern void (*log_hook)(const char *line);

// Says one line: writes it, prefixed with the time of day. Any thread may call
// it.
void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#endif
