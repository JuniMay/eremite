#include "check.h"
#include "bmp.h"
#include "base/error.h"
#include "engine/memory.h"
#include "gpu/gpu.h"
#include "kernels/library.h"
#include "model/lfm3/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests d1-2B's vision tower and projector on its checkpoint against
// LiquidAI's own code in FP32 (transformers 5.19.0 and torch 2.15 on the CPU):
// the rows of two test images, one for each square of 2 x 2 patches. They are
// as close to the FP32 rows as LiquidAI's code in BF16: in their mean
// difference over 128 sampled values, and in their largest difference, which
// stays within the BF16 run's largest over the whole image. The same image
// gives the same bits again.
//
// usage: model-lfm3_vision CHECKPOINT

// The values sampled from an image's rows, each its own stride apart.
static constexpr int SAMPLES = 128;

// A test image: its size, its grid of tokens, the stride of the samples, the
// reference's FP32 values there, and LiquidAI's BF16 run's mean difference
// from them and its largest difference over the whole image.
typedef struct {
  int width, height, tokens_w, tokens_h, stride;
  float values[SAMPLES];
  double mean_bound, worst_bound;
} Image;

static const Image images[] = {
    {400,
     208,
     12,
     6,
     1152,
     {0.0135956,   -0.00142571, -0.0166572,  -0.015745,    0.00666103,
      0.0240101,   -0.014574,   0.0282432,   0.0320965,    -0.00504347,
      0.00211777,  -0.0192115,  -0.0152688,  -0.0341557,   -0.0240451,
      0.0192181,   0.0214213,   -0.0210704,  0.0014734,    0.0119309,
      0.0296341,   0.0272808,   -0.00184461, -0.000630207, 0.0321727,
      -0.0342847,  -0.00703217, -0.0581535,  -0.00942718,  0.0448539,
      -0.0289145,  -0.00151504, -0.0163352,  0.0116225,    -0.00733845,
      -0.0147432,  -0.0263853,  -0.0266061,  -0.0289179,   0.020789,
      -0.0018096,  -0.047141,   -0.0125086,  -0.0498354,   -0.00197609,
      -0.00862573, -0.0274051,  -0.0147396,  -0.0150996,   0.0292988,
      -0.00662842, -0.0153306,  -0.0152538,  0.0318348,    -0.00232759,
      0.0114768,   0.00858574,  -0.0250703,  0.0111316,    -0.0413127,
      -0.00981365, 0.0274935,   -0.0206068,  -0.00951996,  0.0157848,
      -0.00553557, -0.0230113,  0.00567814,  0.0223793,    0.00047986,
      -0.031226,   0.00872695,  0.0148669,   -0.0491887,   0.0171234,
      -0.0731146,  0.011745,    0.0296879,   -0.0238343,   -0.00613105,
      0.00320364,  -0.002642,   -0.00873373, -0.00816775,  -0.00723907,
      0.013409,    -0.0153512,  0.0288765,   0.0535319,    -0.0515078,
      -0.027573,   -0.00746432, 0.00100769,  0.0244456,    -0.0187997,
      -0.0235942,  0.00397921,  0.0164588,   0.0136807,    -0.0218048,
      -0.00717347, 0.0170457,   -0.035963,   0.0298156,    -0.000788522,
      0.00146942,  0.0391224,   -0.0341718,  -0.00612441,  -0.00611631,
      -0.0297398,  0.0105893,   0.0270152,   -0.011686,    -0.0221098,
      -0.0114495,  0.000694203, -0.00904383, -0.0250264,   0.0693601,
      0.0226454,   -0.031453,   0.0365996,   -0.0359589,   -0.00121637,
      0.0126883,   -0.0184326,  -0.0311431},
     0.000824,
     0.0256},
    {200,
     150,
     10,
     7,
     1120,
     {0.00679424, -0.00992905, -0.0202182,  -0.0293853, -0.0416713,  -0.0225678,
      0.0356307,  -0.0105566,  -0.0234326,  0.0196551,  0.0383293,   0.00600307,
      0.0129947,  -0.0292402,  0.00612009,  0.00221999, -0.00977436, 0.0381125,
      -0.0575862, 0.0738785,   0.0322413,   -0.0383559, 0.0552693,   0.0168121,
      0.00122484, -0.0115944,  -0.0271575,  0.0329451,  -0.0354351,  -0.0187192,
      0.0141671,  -0.00148462, 0.0299433,   -0.0105274, -0.0146809,  -0.0315299,
      -0.0107566, -0.0120732,  -0.00892794, 0.00779274, 0.0174645,   0.0077962,
      -0.0228686, 0.0110745,   -0.0281201,  -0.0209616, 0.0704084,   0.0353973,
      0.0740411,  0.0163236,   0.0150959,   0.00187357, 0.00983706,  0.0238406,
      0.00325731, 0.0187169,   0.0432997,   0.0310887,  0.0118104,   0.0060274,
      -0.0271656, 0.0269188,   0.0711578,   0.0530375,  -0.00894031, -0.0289205,
      -0.0177039, -0.0303474,  -0.0416749,  -0.0148597, 0.0310911,   -0.0555991,
      -0.0216729, 0.0037135,   0.0922614,   0.0637696,  0.00116366,  0.0260681,
      0.00660965, 0.034935,    0.020429,    0.019134,   -0.00172729, 0.0482118,
      0.0485241,  0.0183088,   0.0320244,   0.0316112,  -0.00642096, -0.028559,
      -0.0369652, 0.0427937,   0.0314137,   0.00213295, 0.0148779,   -0.040976,
      0.0358082,  -0.0197259,  -0.00790986, -0.0163662, -0.0458798,  -0.0144456,
      -0.0165871, 0.0253588,   -0.0696044,  0.0198342,  0.00787061,  0.0132858,
      -0.0542352, -0.0425308,  -0.0277308,  0.0441605,  0.0113419,   -0.0256224,
      -0.0393539, 0.00656789,  0.00553383,  -0.0134886, 0.0165252,   0.0333178,
      0.0322991,  -0.00717095, 0.0122533,   -0.0123991, 0.0036433,   0.0225503,
      0.0131558,  -0.00940459},
     0.000520,
     0.0388},
};

static void after_evictions(void *gpu) { gpu_return_freed_memory(gpu); }

// Returns the value of a BF16 row.
static float value_of(uint16_t bits) {
  uint32_t wide = (uint32_t)bits << 16;
  float f;
  memcpy(&f, &wide, sizeof f);
  return f;
}

// Prepares and encodes the image's test pattern in im. Returns whether both
// worked.
static int encode(Model *m, const char *dir, const Image *image, Media *im) {
  size_t len;
  uint8_t *bmp = make_test_bmp(image->width, image->height, &len);
  const char *error = 0;
  int encoded = m->ops->prepare_media(dir, im, bmp, len, &error) &&
                m->ops->encode_media(im);
  free(bmp);
  return encoded;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-lfm3_vision CHECKPOINT");
  Memory *memory = memory_new();
  size_t length;
  const char *source = kernel_library_source(&length);
  Gpu *gpu = gpu_new(source, length);
  memory_set_eviction_callback(memory, after_evictions, gpu);
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *m = model_open(memory, gpu, argv[1], &options);
  if (!m) die("%s: not loaded", argv[1]);
  for (size_t i = 0; i < sizeof images / sizeof *images; i++) {
    const Image *image = &images[i];
    printf("a %d x %d image:\n", image->width, image->height);
    Media im = {0};
    int encoded = encode(m, argv[1], image, &im);
    check(encoded, "it is encoded");
    if (!encoded) continue;
    check(im.tw == image->tokens_w && im.th == image->tokens_h,
          "it takes a row for each square of patches");
    double sum = 0, worst = 0;
    for (int k = 0; k < SAMPLES; k++) {
      double d =
          fabs(value_of(im.rows[(size_t)k * image->stride]) - image->values[k]);
      sum += d, worst = fmax(worst, d);
    }
    printf("mean difference %.6f (BF16 run %.6f), largest %.5f\n",
           sum / SAMPLES, image->mean_bound, worst);
    check(sum / SAMPLES <= image->mean_bound && worst <= image->worst_bound,
          "its rows are as close to FP32 as LiquidAI's BF16 run");
    Media again = {0};
    size_t bytes = (size_t)im.th * im.tw * LFM3_DIM * sizeof *im.rows;
    check(encode(m, argv[1], image, &again) &&
              !memcmp(again.rows, im.rows, bytes),
          "the same image gives the same bits again");
    media_free(&again), media_free(&im);
  }
  return failures != 0;
}
