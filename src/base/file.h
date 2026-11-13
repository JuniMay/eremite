#ifndef EREMITE_BASE_FILE_H
#define EREMITE_BASE_FILE_H
#include <stddef.h>

// Returns the file's bytes followed by a NUL, and their count in *len unless
// len is null; dies when the file cannot be read.
char *read_file(const char *path, size_t *len);

#endif
