#ifndef EREMITE_MODEL_INKLING2_VISION_H
#define EREMITE_MODEL_INKLING2_VISION_H

#include "engine/media.h"
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

// Inkling-2-Mini vision encoder (config.json vision_config,
// image_processing_inkling2.py, modeling_inkling2.py Inkling2VisionModel):
// 23-layer Vision Transformer with bilinear 32x32 position embedding, 2D axial
// pairwise RoPE (width then height), LayerScale, and a 2x2 spatial-merge GELU
// projector into the 2304-wide language stream.

// Vision geometry constants (config.json vision_config and
// processor_config.json).
static constexpr int INK2_VIS_LAYERS = 23;
static constexpr int INK2_VIS_DIM = 1024;
static constexpr int INK2_VIS_HEADS = 16;
static constexpr int INK2_VIS_HEAD_DIM = INK2_VIS_DIM / INK2_VIS_HEADS;
static constexpr int INK2_VIS_MLP_DIM = 4096;
static constexpr int INK2_VIS_PROJ_DIM = 4096;
static constexpr int INK2_VIS_PATCH = 20;
static constexpr int INK2_VIS_MERGE = 2;
static constexpr int INK2_VIS_CELL = INK2_VIS_PATCH * INK2_VIS_MERGE;
static constexpr int INK2_VIS_CHANNELS = 3;
static constexpr int INK2_VIS_PATCH_K =
    INK2_VIS_PATCH * INK2_VIS_PATCH * INK2_VIS_CHANNELS;
static constexpr int INK2_VIS_PATCH_K_PAD = 1216;
static constexpr int INK2_VIS_POS_GRID = 32;
static constexpr int INK2_VIS_POS_ENTRIES =
    INK2_VIS_POS_GRID * INK2_VIS_POS_GRID;
static constexpr int INK2_VIS_TAPS = 4;
static constexpr int INK2_VIS_ROPE_PAIRS = INK2_VIS_HEAD_DIM / 2;

// Composites an 8-bit straight-RGB image and its 8-bit alpha channel onto the
// 8x8 checkerboard background (204 / 153) matching
// Inkling2ImageProcessor.convert_to_rgb.
void inkling2_vision_composite_checkerboard(uint8_t *rgb, const uint8_t *alpha,
                                            int width, int height);

// Computes the 4-tap bilinear (align_corners=False, border clamp) indices and
// weights into the 32x32 position embedding table for a `vh x vw` patch grid
// ordered in 2x2 spatial-merge blocks (tr, tc, mr, mc).
void inkling2_vision_bilinear_taps(int vh, int vw, uint32_t *indices,
                                   float *weights);

// Computes 2D axial pairwise RoPE (cos, sin) pairs `[1 + vh*vw, 32, 2]` with
// CLS at (0, 0) and 1-indexed patches (row + 1, col + 1), width frequencies in
// pairs 0..15 and height frequencies in pairs 16..31.
void inkling2_vision_rope_table(int vh, int vw, float *cos_sin);

// Initializes the vision tower's lazy dependent weight group and kernel handles
// from the loaded language backbone.
void inkling2_vision_init(Gpu *gpu, MemorySlot *work_slot,
                          TensorIndex *checkpoint, WeightGroup *text_weights);

// Decodes an image file, composites alpha if present, pads with -1.0 to whole
// 40x40 cells, rescales/normalizes to [-1, 1], and packs padded BF16 patches
// into `im`. Returns 1 on success or 0 with `*error` set.
int inkling2_prepare_media(const char *dir, Media *im, const void *data,
                           size_t len, const char **error);

// Runs the 23-layer vision encoder and 2x2 merge projector on `im->patch` and
// stores `im->ntokens` BF16 feature rows of width 2304 in `im->rows`. Returns
// 1 on success or 0 when memory does not fit.
int inkling2_encode_media(Media *im);

// Returns the resident bytes of the lazy vision WeightGroup, or 0 while
// unloaded.
uint64_t inkling2_vision_resident_bytes(void);

#endif
