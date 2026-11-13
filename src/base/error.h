#pragma once
#include <stddef.h>

// Fatal errors, and allocation that cannot fail.

// When set, runs before a fatal error's message is printed.
extern void (*die_hook)(void);

// Ends the program: prints "eremite: " and the message to stderr, then exits
// with status 1.
_Noreturn void die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// Returns n zeroed bytes; dies when memory runs out.
void *xmalloc(size_t n);
