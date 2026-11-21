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
    , '\n',
#embed "lfm3_projector.metal"
    , '\n',
#embed "minimax_h3_quant.metal"
    , '\n',
#embed "minimax_h3_qwen.metal"
    , '\n',
#embed "minimax_h3_vae.metal"
    , '\n',
#embed "minimax_h3_dit.metal"
    , '\n',
#embed "minimax_h3_vision.metal"
    , '\n',
#embed "minimax_h3_audio_vae.metal"
};

const char *kernel_library_source(size_t *length) {
  *length = sizeof source;
  return source;
}
