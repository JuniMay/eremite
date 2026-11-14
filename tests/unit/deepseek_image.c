#include "check.h"
#include "base/error.h"
#include "model/deepseek/image.h"
#include "model/deepseek/model.h"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Tests the images for DeepSeek's vision encoder. A small image is scaled up
// before its grid is planned. A wide image's grid stays within the token
// limit, at the image's shape. The patches hold the pixels from -1 to 1, and
// gray padding fills the rest. Each patch's RoPE angles turn by its row and
// its column. The hash follows the file's bytes. Bytes that hold no image are
// refused.

// The vision encoder's patch side, the tokens an image may take at most, and
// the fewest pixels an image is scaled up to.
static constexpr int PATCH = 14, MAX_TOKENS = 1024, MIN_PIXELS = 295936;

// A patch's values: 3 channels of PATCH x PATCH pixels. VIT_K pads them with
// zeros.
static constexpr int PATCH_VALUES = 3 * PATCH * PATCH;

// Returns a PNG file of width x height pixels of one gray level.
static CFDataRef gray_png(int width, int height, uint8_t level) {
  uint8_t *pixels = malloc((size_t)width * height * 3);
  memset(pixels, level, (size_t)width * height * 3);
  CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGDataProviderRef provider =
      CGDataProviderCreateWithData(0, pixels, (size_t)width * height * 3, 0);
  CGImageRef image = CGImageCreate(width, height, 8, 24, (size_t)width * 3,
                                   space, kCGImageAlphaNone, provider, 0, false,
                                   kCGRenderingIntentDefault);
  CFMutableDataRef file = CFDataCreateMutable(0, 0);
  CGImageDestinationRef destination =
      CGImageDestinationCreateWithData(file, CFSTR("public.png"), 1, 0);
  if (!image || !destination) die("cannot make a PNG file");
  CGImageDestinationAddImage(destination, image, 0);
  if (!CGImageDestinationFinalize(destination)) die("cannot encode a PNG file");
  CFRelease(destination);
  CGImageRelease(image);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(space);
  free(pixels);
  return file;
}

// Opens a gray PNG image of width x height pixels.
static int open_png(Media *im, int width, int height, uint8_t level) {
  CFDataRef png = gray_png(width, height, level);
  const char *error = 0;
  int opened = image_open(im, CFDataGetBytePtr(png),
                          (size_t)CFDataGetLength(png), &error);
  CFRelease(png);
  return opened && !error;
}

// Returns the BF16 bits that a pixel level becomes: from -1 to 1, rounded.
static uint16_t level_bits(uint8_t level) {
  float value = ((float)level / 255.0f - 0.5f) / 0.5f;
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Returns the patch value of channel c of the pixel at x, y.
static uint16_t pixel(const Media *im, int x, int y, int c) {
  size_t patch = (size_t)(y / PATCH) * im->vw + x / PATCH;
  return im->patch[patch * VIT_K + c * PATCH * PATCH + (y % PATCH) * PATCH +
                   x % PATCH];
}

// Returns whether the image's grid of tokens follows its grid of patches.
static int grids_agree(const Media *im) {
  return im->th == (im->vh + 2) / 3 && im->tw == (im->vw + 2) / 3 &&
         im->ntokens == im->th * (im->tw + 1) + 2;
}

int main(void) {
  // A small white square, scaled up before its grid is planned.
  Media square = {0};
  check(open_png(&square, 28, 28, 255), "a small PNG image opens");
  check(square.vh == square.vw && grids_agree(&square) &&
            square.vh * square.vw * PATCH * PATCH >= MIN_PIXELS,
        "a small square is scaled up to at least the fewest pixels");
  int white = 1, zeros = 1;
  for (size_t i = 0; i < (size_t)square.vh * square.vw * VIT_K; i++) {
    if ((int)(i % VIT_K) < PATCH_VALUES)
      white &= square.patch[i] == level_bits(255);
    else
      zeros &= square.patch[i] == 0;
  }
  check(white, "a white image's patches all hold 1");
  check(zeros, "each patch's values are padded with zeros to VIT_K");

  // A wide white image, which gray padding fills to its grid.
  Media wide = {0};
  check(open_png(&wide, 56, 28, 255), "a wide PNG image opens");
  int width = wide.vw * PATCH, height = wide.vh * PATCH;
  check(grids_agree(&wide) && fabs((double)width / height - 2) < 0.1,
        "a wide image keeps its shape");
  check(pixel(&wide, 0, 0, 0) == level_bits(127) &&
            pixel(&wide, width - 1, height - 1, 2) == level_bits(127) &&
            pixel(&wide, width / 2, height / 2, 1) == level_bits(255),
        "gray padding fills the rows above and below a wide image");

  // A large wide image, whose grid must shrink to the token limit.
  Media large = {0};
  check(open_png(&large, 2000, 1000, 40), "a large PNG image opens");
  check(grids_agree(&large) && large.ntokens <= MAX_TOKENS &&
            fabs((double)large.vw / large.vh - 2) < 0.1,
        "a large image's grid stays within the token limit, at its shape");

  // The RoPE angles of patch (0, 0) and of patch (1, 2).
  float *angles = malloc((size_t)large.vh * large.vw * 64 * sizeof *angles);
  image_rope(&large, angles);
  int origin = 1;
  for (int i = 0; i < 32; i++) origin &= angles[i] == 1 && angles[32 + i] == 0;
  const float *turned = angles + ((size_t)1 * large.vw + 2) * 64;
  check(origin, "the first patch's angles are all 0");
  check(turned[0] == (float)cos(1.0) && turned[32] == (float)sin(1.0) &&
            turned[16] == (float)cos(2.0) && turned[48] == (float)sin(2.0),
        "a patch's angles turn by its row in the first half, and by its "
        "column in the second");
  free(angles);

  check(square.hash != large.hash, "different files have different hashes");
  Media again = {0};
  open_png(&again, 28, 28, 255);
  check(again.hash == square.hash, "the same file has the same hash");

  static const char text[] = "not an image";
  Media none = {0};
  const char *error = 0;
  check(!image_open(&none, text, sizeof text - 1, &error) && error,
        "bytes that hold no image are refused with an error");

  media_free(&square);
  media_free(&wide);
  media_free(&large);
  media_free(&again);
  return failures != 0;
}
