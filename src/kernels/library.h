#ifndef EREMITE_KERNELS_LIBRARY_H
#define EREMITE_KERNELS_LIBRARY_H
#include <stddef.h>

// The kernel library: the Metal sources that the program embeds, joined in a
// fixed order, the shared number formats first.

// Returns the library's source, and its length in *length.
const char *kernel_library_source(size_t *length);

#endif
