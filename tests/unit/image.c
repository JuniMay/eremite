#include "check.h"
#include "base/error.h"
#include "platform/image.h"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <string.h>

// Tests decoding image files into 8-bit RGB pixels. A PNG file decodes to its
// exact pixels: alpha is dropped, and gray is repeated in each channel. A JPEG
// file decodes close to the pixels it was made from, and exactly as libjpeg
// decodes it with its default settings. Bytes that hold no image give an
// error.

// The PNG images' size.
static constexpr int WIDTH = 5, HEIGHT = 3;

// The JPEG images' side, and how far the smooth image's decoded channels may
// be from the pixels it was made from.
static constexpr int JPEG_SIDE = 16, JPEG_TOLERANCE = 3;

// libjpeg's default quality.
static constexpr int JPEG_DEFAULT_QUALITY = 75;

// Encodes WIDTH x HEIGHT pixels as a PNG file. Each pixel has components
// bytes in space. The last byte is alpha, unless alpha is kCGImageAlphaNone.
static CFDataRef encode_png(const uint8_t *pixels, int components,
                            CGColorSpaceRef space, CGImageAlphaInfo alpha) {
  size_t row = (size_t)WIDTH * components;
  CGDataProviderRef provider =
      CGDataProviderCreateWithData(0, pixels, row * HEIGHT, 0);
  CGImageRef image = CGImageCreate(WIDTH, HEIGHT, 8, 8 * (size_t)components,
                                   row, space, (CGBitmapInfo)alpha, provider, 0,
                                   false, kCGRenderingIntentDefault);
  CFMutableDataRef file = CFDataCreateMutable(0, 0);
  CGImageDestinationRef destination =
      CGImageDestinationCreateWithData(file, CFSTR("public.png"), 1, 0);
  if (!image || !destination) die("cannot make a PNG file");
  CGImageDestinationAddImage(destination, image, 0);
  if (!CGImageDestinationFinalize(destination)) die("cannot encode a PNG file");
  CFRelease(destination);
  CGImageRelease(image);
  CGDataProviderRelease(provider);
  return file;
}

// Encodes JPEG_SIDE x JPEG_SIDE RGB pixels as a JPEG file at quality. With
// full_chroma, the chroma channels keep their full size. Sets *length to the
// file's bytes.
static uint8_t *encode_jpeg(const uint8_t *pixels, int quality, int full_chroma,
                            unsigned long *length) {
  struct jpeg_compress_struct c;
  struct jpeg_error_mgr errors;
  c.err = jpeg_std_error(&errors);
  jpeg_create_compress(&c);
  unsigned char *file = 0;
  jpeg_mem_dest(&c, &file, length);
  c.image_width = c.image_height = JPEG_SIDE;
  c.input_components = 3;
  c.in_color_space = JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, quality, TRUE);
  for (int i = 0; full_chroma && i < c.num_components; i++)
    c.comp_info[i].h_samp_factor = c.comp_info[i].v_samp_factor = 1;
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    JSAMPROW row = (JSAMPROW)(pixels + (size_t)c.next_scanline * JPEG_SIDE * 3);
    jpeg_write_scanlines(&c, &row, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  return file;
}

// Decodes a JPEG file with libjpeg's default settings, into RGB pixels.
static uint8_t *decode_with_libjpeg(const uint8_t *file, unsigned long length) {
  struct jpeg_decompress_struct d;
  struct jpeg_error_mgr errors;
  d.err = jpeg_std_error(&errors);
  jpeg_create_decompress(&d);
  jpeg_mem_src(&d, file, length);
  jpeg_read_header(&d, TRUE);
  d.out_color_space = JCS_RGB;
  jpeg_start_decompress(&d);
  uint8_t *pixels = xmalloc((size_t)JPEG_SIDE * JPEG_SIDE * 3);
  while (d.output_scanline < d.output_height) {
    JSAMPROW row = pixels + (size_t)d.output_scanline * JPEG_SIDE * 3;
    jpeg_read_scanlines(&d, &row, 1);
  }
  jpeg_finish_decompress(&d);
  jpeg_destroy_decompress(&d);
  return pixels;
}

// Decodes a file's bytes. Returns the pixels if there are width x height of
// them and no error, or else null.
static uint8_t *decode(const void *bytes, size_t length, int width,
                       int height) {
  int w = -1, h = -1;
  const char *error = "unset";
  uint8_t *pixels = image_decode_rgb(bytes, length, &w, &h, &error);
  if (!pixels || error || w != width || h != height) {
    free(pixels);
    return 0;
  }
  return pixels;
}

int main(void) {
  CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB),
                  gray_space = CGColorSpaceCreateDeviceGray();

  // RGB pixels, each one different.
  uint8_t rgb[HEIGHT][WIDTH][3];
  for (int y = 0; y < HEIGHT; y++)
    for (int x = 0; x < WIDTH; x++) {
      rgb[y][x][0] = (uint8_t)(40 * x + y);
      rgb[y][x][1] = (uint8_t)(200 - 30 * y);
      rgb[y][x][2] = (uint8_t)(7 * x * y + 3);
    }
  CFDataRef png = encode_png(&rgb[0][0][0], 3, srgb, kCGImageAlphaNone);
  uint8_t *pixels = decode(CFDataGetBytePtr(png), (size_t)CFDataGetLength(png),
                           WIDTH, HEIGHT);
  check(pixels && !memcmp(pixels, rgb, sizeof rgb),
        "an RGB PNG file decodes to its exact pixels");
  free(pixels);
  CFRelease(png);

  // The same pixels, partly transparent.
  uint8_t rgba[HEIGHT][WIDTH][4];
  for (int y = 0; y < HEIGHT; y++)
    for (int x = 0; x < WIDTH; x++) {
      memcpy(rgba[y][x], rgb[y][x], 3);
      rgba[y][x][3] = (uint8_t)(60 + 40 * x);
    }
  png = encode_png(&rgba[0][0][0], 4, srgb, kCGImageAlphaLast);
  pixels = decode(CFDataGetBytePtr(png), (size_t)CFDataGetLength(png), WIDTH,
                  HEIGHT);
  check(pixels && !memcmp(pixels, rgb, sizeof rgb),
        "a PNG file with alpha decodes to its colors, with alpha dropped");
  free(pixels);
  int alpha_w = 0, alpha_h = 0;
  const char *alpha_error = "unset";
  uint8_t *alpha =
      image_decode_alpha(CFDataGetBytePtr(png), (size_t)CFDataGetLength(png),
                         &alpha_w, &alpha_h, &alpha_error);
  int alpha_exact =
      alpha && !alpha_error && alpha_w == WIDTH && alpha_h == HEIGHT;
  for (int y = 0; alpha_exact && y < HEIGHT; y++)
    for (int x = 0; x < WIDTH; x++)
      alpha_exact &= alpha[y * WIDTH + x] == rgba[y][x][3];
  check(alpha_exact, "a PNG file's alpha decodes to its exact values");
  free(alpha);
  CFRelease(png);
  png = encode_png(&rgb[0][0][0], 3, srgb, kCGImageAlphaNone);
  alpha_error = 0;
  alpha =
      image_decode_alpha(CFDataGetBytePtr(png), (size_t)CFDataGetLength(png),
                         &alpha_w, &alpha_h, &alpha_error);
  check(!alpha && alpha_error && !alpha_w && !alpha_h,
        "a PNG file without alpha gives no alpha and an error");
  CFRelease(png);
  size_t rgba_length = 0;
  const char *rgba_error = "unset";
  uint8_t *encoded_rgba = image_encode_png_rgba(&rgba[0][0][0], WIDTH, HEIGHT,
                                                &rgba_length, &rgba_error);
  pixels = encoded_rgba ? decode(encoded_rgba, rgba_length, WIDTH, HEIGHT) : 0;
  alpha = encoded_rgba ? image_decode_alpha(encoded_rgba, rgba_length, &alpha_w,
                                            &alpha_h, &alpha_error)
                       : 0;
  alpha_exact = alpha && alpha_w == WIDTH && alpha_h == HEIGHT;
  for (int y = 0; alpha_exact && y < HEIGHT; y++)
    for (int x = 0; x < WIDTH; x++)
      alpha_exact &= alpha[y * WIDTH + x] == rgba[y][x][3];
  check(encoded_rgba && !rgba_error && pixels &&
            !memcmp(pixels, rgb, sizeof rgb) && alpha_exact,
        "encoded RGBA PNG bytes round-trip to their colors and alpha");
  free(alpha);
  free(pixels);
  free(encoded_rgba);

  // Gray pixels.
  uint8_t gray[HEIGHT][WIDTH];
  for (int y = 0; y < HEIGHT; y++)
    for (int x = 0; x < WIDTH; x++) gray[y][x] = (uint8_t)(50 * x + 10 * y);
  png = encode_png(&gray[0][0], 1, gray_space, kCGImageAlphaNone);
  pixels = decode(CFDataGetBytePtr(png), (size_t)CFDataGetLength(png), WIDTH,
                  HEIGHT);
  int repeated = pixels != 0;
  for (int i = 0; repeated && i < WIDTH * HEIGHT; i++)
    for (int c = 0; c < 3; c++)
      repeated &= pixels[3 * i + c] == (&gray[0][0])[i];
  check(repeated, "a gray PNG file decodes with its gray in each channel");
  free(pixels);
  CFRelease(png);

  // A smooth JPEG image.
  uint8_t smooth[JPEG_SIDE][JPEG_SIDE][3];
  for (int y = 0; y < JPEG_SIDE; y++)
    for (int x = 0; x < JPEG_SIDE; x++) {
      smooth[y][x][0] = (uint8_t)(8 * x + 4);
      smooth[y][x][1] = (uint8_t)(8 * y + 4);
      smooth[y][x][2] = (uint8_t)(128 + 2 * (x - y));
    }
  unsigned long jpeg_length = 0;
  uint8_t *jpeg = encode_jpeg(&smooth[0][0][0], 100, 1, &jpeg_length);
  pixels = decode(jpeg, jpeg_length, JPEG_SIDE, JPEG_SIDE);
  int close = pixels != 0;
  for (size_t i = 0; close && i < sizeof smooth; i++)
    close &= abs(pixels[i] - (&smooth[0][0][0])[i]) <= JPEG_TOLERANCE;
  check(close, "a JPEG file decodes within 3 levels of its pixels");
  free(pixels);
  free(jpeg);

  // A noisy JPEG image at libjpeg's default settings, with half-size chroma.
  uint8_t noisy[JPEG_SIDE * JPEG_SIDE * 3];
  for (size_t i = 0; i < sizeof noisy; i++)
    noisy[i] = (uint8_t)((i * 2654435761u) >> 24);
  jpeg = encode_jpeg(noisy, JPEG_DEFAULT_QUALITY, 0, &jpeg_length);
  pixels = decode(jpeg, jpeg_length, JPEG_SIDE, JPEG_SIDE);
  uint8_t *reference = decode_with_libjpeg(jpeg, jpeg_length);
  check(pixels && !memcmp(pixels, reference, sizeof noisy),
        "a JPEG file decodes exactly as libjpeg decodes it by default");
  free(reference);
  free(pixels);
  free(jpeg);

  // Bytes that hold no image, and a JPEG file whose first marker is broken.
  static const char text[] = "this is not an image";
  static const uint8_t broken[] = {0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0};
  int width = -1, height = -1;
  const char *error = 0;
  pixels = image_decode_rgb(text, sizeof text - 1, &width, &height, &error);
  check(!pixels && error && !width && !height,
        "bytes that hold no image give an error and no pixels");
  error = 0;
  pixels = image_decode_rgb(broken, sizeof broken, &width, &height, &error);
  check(!pixels && error && !width && !height,
        "a broken JPEG file gives an error and no pixels");

  // Encoding RGB pixels as PNG and JPEG, and rejecting invalid inputs.
  size_t encoded_length = 0;
  error = "unset";
  uint8_t *encoded_png =
      image_encode_png(&rgb[0][0][0], WIDTH, HEIGHT, &encoded_length, &error);
  pixels = encoded_png ? decode(encoded_png, encoded_length, WIDTH, HEIGHT) : 0;
  check(encoded_png && encoded_length > 8 &&
            !memcmp(encoded_png, "\x89PNG\r\n\x1a\n", 8) && !error && pixels &&
            !memcmp(pixels, rgb, sizeof rgb),
        "encoded PNG bytes round-trip to the exact RGB pixels");
  free(pixels);
  int read_w = 0, read_h = 0;
  error = "unset";
  check(
      image_read_size(encoded_png, encoded_length, &read_w, &read_h, &error) &&
          read_w == WIDTH && read_h == HEIGHT && !error,
      "a PNG file's size reads without decoding its pixels");
  free(encoded_png);

  error = "unset";
  uint8_t *encoded_jpeg =
      image_encode_jpeg(&smooth[0][0][0], JPEG_SIDE, JPEG_SIDE,
                        JPEG_DEFAULT_QUALITY, &encoded_length, &error);
  jpeg = encode_jpeg(&smooth[0][0][0], JPEG_DEFAULT_QUALITY, 0, &jpeg_length);
  pixels = encoded_jpeg
               ? decode(encoded_jpeg, encoded_length, JPEG_SIDE, JPEG_SIDE)
               : 0;
  check(encoded_jpeg && !error && encoded_length == jpeg_length &&
            !memcmp(encoded_jpeg, jpeg, jpeg_length) && pixels != 0,
        "encoded JPEG bytes match libjpeg defaults and decode back");
  error = "unset";
  check(
      image_read_size(encoded_jpeg, encoded_length, &read_w, &read_h, &error) &&
          read_w == JPEG_SIDE && read_h == JPEG_SIDE && !error,
      "a JPEG file's size reads without decoding its pixels");
  error = 0;
  check(!image_read_size(text, sizeof text - 1, &read_w, &read_h, &error) &&
            error && !read_w && !read_h,
        "bytes that hold no image give no size and an error");
  free(pixels);
  free(jpeg);
  free(encoded_jpeg);

  error = 0;
  encoded_png =
      image_encode_png(&rgb[0][0][0], 0, HEIGHT, &encoded_length, &error);
  check(!encoded_png && !encoded_length && error,
        "encoding an empty PNG gives an error");
  error = 0;
  encoded_jpeg =
      image_encode_jpeg(&smooth[0][0][0], JPEG_SIDE, JPEG_SIDE,
                        IMAGE_QUALITY_MAX + 1, &encoded_length, &error);
  check(!encoded_jpeg && !encoded_length && error,
        "encoding a JPEG with quality outside 0 to 100 gives an error");
  error = "unset";
  encoded_jpeg = image_encode_jpeg(&smooth[0][0][0], JPEG_SIDE, JPEG_SIDE,
                                   IMAGE_QUALITY_MIN, &encoded_length, &error);
  check(encoded_jpeg && encoded_length > 0 && !error,
        "a JPEG encodes at the lowest quality, 0");
  free(encoded_jpeg);

  // WebP: lossy, so the check is that it decodes to the same size.
  error = "unset";
  uint8_t *encoded_webp =
      image_encode_webp(&smooth[0][0][0], JPEG_SIDE, JPEG_SIDE,
                        JPEG_DEFAULT_QUALITY, &encoded_length, &error);
  pixels = encoded_webp
               ? decode(encoded_webp, encoded_length, JPEG_SIDE, JPEG_SIDE)
               : 0;
  check(encoded_webp && !error && encoded_length > 12 &&
            !memcmp(encoded_webp, "RIFF", 4) &&
            !memcmp(encoded_webp + 8, "WEBP", 4) && pixels,
        "encoded WebP bytes decode back to the same size");
  free(pixels);
  free(encoded_webp);
  error = 0;
  encoded_webp =
      image_encode_webp(&smooth[0][0][0], JPEG_SIDE, JPEG_SIDE,
                        IMAGE_QUALITY_MAX + 1, &encoded_length, &error);
  check(!encoded_webp && !encoded_length && error,
        "encoding a WebP with quality outside 0 to 100 gives an error");
  error = "unset";
  encoded_webp =
      image_encode_webp_rgba(&rgba[0][0][0], WIDTH, HEIGHT, IMAGE_QUALITY_MAX,
                             &encoded_length, &error);
  alpha = encoded_webp ? image_decode_alpha(encoded_webp, encoded_length,
                                            &alpha_w, &alpha_h, &alpha_error)
                       : 0;
  check(encoded_webp && !error && alpha && alpha_w == WIDTH &&
            alpha_h == HEIGHT,
        "an RGBA WebP file keeps an alpha channel of its size");
  free(alpha);
  free(encoded_webp);

  CGColorSpaceRelease(gray_space);
  CGColorSpaceRelease(srgb);
  return failures != 0;
}
