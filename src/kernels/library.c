#include "kernels/library.h"

static const char source[] = {
#embed "formats.metal"
    , '\n',
#embed "kernels.metal"
    , '\n',
#embed "gguf_grids.metal"
    , '\n',
#embed "gguf.metal"
    , '\n',
#embed "vision.metal"
    , '\n',
#embed "lfm3.metal"
    , '\n',
#embed "lfm3_vision.metal"
};

const char *kernel_library_source(size_t *length) {
  *length = sizeof source;
  return source;
}
