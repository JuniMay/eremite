#include "platform/image.h"
#include "base/error.h"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <jpeglib.h>
#include <webp/encode.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define BOMB                                                                   \
  178956970  // Pillow refuses images of more pixels (twice
             // Media.MAX_IMAGE_PIXELS)

// From the image's own pixel data: alpha dropped (not composited), gray
// repeated, palettes looked up, 16-bit channels by their high byte; 0 for
// formats outside these.
static int raw_rgb(CGImageRef img, uint8_t *out, size_t w, size_t h) {
  CGColorSpaceRef cs = CGImageGetColorSpace(img);
  CGColorSpaceModel model =
      cs ? CGColorSpaceGetModel(cs) : kCGColorSpaceModelUnknown;
  CGBitmapInfo info = CGImageGetBitmapInfo(img);
  CGImageAlphaInfo alpha = (CGImageAlphaInfo)(info & kCGBitmapAlphaInfoMask);
  size_t bpc = CGImageGetBitsPerComponent(img),
         bpp = CGImageGetBitsPerPixel(img), stride = CGImageGetBytesPerRow(img);
  int indexed = model == kCGColorSpaceModelIndexed;
  int color = model == kCGColorSpaceModelRGB                     ? 3
              : model == kCGColorSpaceModelMonochrome || indexed ? 1
                                                                 : 0;
  int extra = alpha != kCGImageAlphaNone;  // an alpha or padding component
  if (!color || (info & kCGBitmapFloatComponents) ||
      alpha == kCGImageAlphaOnly || bpp != (color + extra) * bpc)
    return 0;
  if (indexed ? extra || (bpc != 1 && bpc != 2 && bpc != 4 && bpc != 8)
              : bpc != 8 && bpc != 16)
    return 0;
  uint8_t table[256 * 3] = {0};
  if (indexed) {
    CGColorSpaceRef base = CGColorSpaceGetBaseColorSpace(cs);
    if (!base || CGColorSpaceGetModel(base) != kCGColorSpaceModelRGB ||
        CGColorSpaceGetColorTableCount(cs) > 256)
      return 0;
    CGColorSpaceGetColorTable(cs, table);
  }
  CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(img));
  if (!data) return 0;
  if ((size_t)CFDataGetLength(data) < stride * (h - 1) + (w * bpp + 7) / 8) {
    CFRelease(data);
    return 0;
  }
  const uint8_t *p = CFDataGetBytePtr(data);
  uint32_t order = info & kCGBitmapByteOrderMask;
  int first = alpha == kCGImageAlphaFirst ||
              alpha == kCGImageAlphaPremultipliedFirst ||
              alpha == kCGImageAlphaNoneSkipFirst;
  int premul = alpha == kCGImageAlphaPremultipliedFirst ||
               alpha == kCGImageAlphaPremultipliedLast;
  int comps = color + extra, bytes = (int)bpc / 8;
  // little-endian 16- and 32-bit pixels of 8-bit components hold them in
  // reverse
  int reversed =
      bpc == 8 && ((bpp == 32 && order == kCGBitmapByteOrder32Little) ||
                   (bpp == 16 && order == kCGBitmapByteOrder16Little));
  int little = bpc == 16 && order == kCGBitmapByteOrder16Little;
  for (size_t y = 0; y < h; y++) {
    const uint8_t *row = p + y * stride;
    for (size_t x = 0; x < w; x++) {
      uint8_t *o = out + (y * w + x) * 3;
      if (indexed) {
        size_t bit = x * bpc;
        memcpy(o,
               table + 3 * ((row[bit / 8] >> (8 - bpc - bit % 8)) &
                            ((1u << bpc) - 1)),
               3);
        continue;
      }
      int v[4];  // the components in their declared order
      for (int c = 0; c < comps; c++) {
        const uint8_t *q =
            row + (x * comps + (reversed ? comps - 1 - c : c)) * bytes;
        int u = bytes == 1 ? q[0]
                : little   ? q[0] | q[1] << 8
                           : q[0] << 8 | q[1];
        // 16-bit gray without alpha opens in Pillow as I;16, whose conversion
        // clamps at 255; the other 16-bit images keep their high byte
        v[c] = bytes == 1             ? u
               : color == 1 && !extra ? (u > 255 ? 255 : u)
                                      : u >> 8;
      }
      int a = extra ? v[first ? 0 : color] : 255, *cv = v + (extra && first);
      for (int c = 0; c < 3; c++) {
        int u = cv[color == 3 ? c : 0];
        if (premul) u = a ? (u * 255 + a / 2) / a : 0;
        o[c] = (uint8_t)(u > 255 ? 255 : u);
      }
    }
  }
  CFRelease(data);
  return 1;
}

// Other formats (floating-point, CMYK, packed): drawn into sRGB.
static void draw_rgb(CGImageRef img, uint8_t *out, size_t w, size_t h) {
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  uint8_t *buf = calloc(w * h, 4);
  CGContextRef ctx = buf ? CGBitmapContextCreate(buf, w, h, 8, w * 4, cs,
                                                 kCGImageAlphaNoneSkipLast |
                                                     kCGBitmapByteOrder32Big)
                         : 0;
  if (!ctx) die("cannot draw an image of %zu x %zu pixels", w, h);
  CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
  for (size_t i = 0; i < w * h; i++) memcpy(out + i * 3, buf + i * 4, 3);
  CGContextRelease(ctx);
  CGColorSpaceRelease(cs);
  free(buf);
}

typedef struct {
  int w, h;
  uint8_t *p;
} Rgb;  // rows of w RGB pixels

// JPEG files through libjpeg-turbo with its defaults (the accurate integer
// IDCT, smooth chroma upsampling), as Pillow decodes them: macOS's decoder
// gave pixels up to 4 levels apart on photos. CMYK and YCCK files are left
// to ImageIO (w = 0).
typedef struct {
  struct jpeg_error_mgr mgr;
  jmp_buf back;
} JpegErr;

static void jpeg_fail(j_common_ptr c) { longjmp(((JpegErr *)c->err)->back, 1); }

static Rgb decode_jpeg(const void *bytes, size_t len, const char **err) {
  struct jpeg_decompress_struct c;
  JpegErr je;
  static Rgb r;  // static: kept across the longjmp (images are decoded on one
                 // thread at a time)
  r = (Rgb){0};
  c.err = jpeg_std_error(&je.mgr);
  je.mgr.error_exit = jpeg_fail;
  if (setjmp(je.back)) {
    jpeg_destroy_decompress(&c);
    free(r.p);
    *err = "the JPEG image could not be decoded";
    return (Rgb){0};
  }
  jpeg_create_decompress(&c);
  jpeg_mem_src(&c, bytes, (unsigned long)len);
  jpeg_read_header(&c, TRUE);
  if (c.jpeg_color_space == JCS_CMYK || c.jpeg_color_space == JCS_YCCK) {
    jpeg_destroy_decompress(&c);
    return (Rgb){0};
  }
  if ((double)c.image_width * c.image_height > BOMB) {
    jpeg_destroy_decompress(&c);
    *err = "the image has more than 178,956,970 pixels";
    return (Rgb){0};
  }
  c.out_color_space = JCS_RGB;
  jpeg_start_decompress(&c);
  r = (Rgb){(int)c.output_width, (int)c.output_height,
            xmalloc((size_t)c.output_width * c.output_height * 3)};
  while (c.output_scanline < c.output_height) {
    JSAMPROW row = r.p + (size_t)c.output_scanline * c.output_width * 3;
    jpeg_read_scanlines(&c, &row, 1);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  return r;
}

// The 8 bytes that open every PNG file.
static const uint8_t PNG_SIGNATURE[8] = {0x89, 'P',  'N',  'G',
                                         '\r', '\n', 0x1a, '\n'};

// Bytes in each of a PNG chunk's length, type and CRC fields.
static constexpr size_t PNG_FIELD_BYTES = 4;

// Bytes of a PNG chunk besides its data: its length, type and CRC.
static constexpr size_t PNG_CHUNK_OVERHEAD = 3 * PNG_FIELD_BYTES;

// The reversed polynomial of the CRC-32 that PNG chunks carry.
static constexpr uint32_t PNG_CRC_POLYNOMIAL = 0xEDB88320u;

// Bits in a byte, which the CRC takes one at a time.
static constexpr int BYTE_BITS = 8;

// Returns the CRC-32 of the len bytes at p, as a PNG chunk carries it.
static uint32_t png_crc(const uint8_t *p, size_t len) {
  uint32_t crc = ~0u;
  for (size_t i = 0; i < len; i++) {
    crc ^= p[i];
    for (int bit = 0; bit < BYTE_BITS; bit++)
      crc = crc & 1 ? (crc >> 1) ^ PNG_CRC_POLYNOMIAL : crc >> 1;
  }
  return ~crc;
}

// Returns the big-endian 32-bit number at p.
static uint32_t read_big_endian_32(const uint8_t *p) {
  return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 |
         (uint32_t)p[3];
}

// Returns an error message when the len bytes at data are a PNG file that is
// cut short or damaged: a chunk runs past the file's end, a chunk's CRC does
// not match it, or the chunks end before IEND. Returns null for a whole PNG
// file, and for bytes that are not PNG.
static const char *check_png(const void *data, size_t len) {
  const uint8_t *bytes = data;
  if (len < sizeof PNG_SIGNATURE ||
      memcmp(bytes, PNG_SIGNATURE, sizeof PNG_SIGNATURE))
    return 0;
  size_t at = sizeof PNG_SIGNATURE;
  while (len - at >= PNG_CHUNK_OVERHEAD) {
    uint32_t chunk = read_big_endian_32(bytes + at);
    if (chunk > len - at - PNG_CHUNK_OVERHEAD) break;
    const uint8_t *type = bytes + at + PNG_FIELD_BYTES;
    if (png_crc(type, PNG_FIELD_BYTES + chunk) !=
        read_big_endian_32(type + PNG_FIELD_BYTES + chunk))
      break;
    if (!memcmp(type, "IEND", PNG_FIELD_BYTES)) return 0;
    at += PNG_CHUNK_OVERHEAD + chunk;
  }
  return "the PNG file is cut short or damaged";
}

static Rgb decode(const void *bytes, size_t len, const char **err) {
  *err = 0;
  if (len > 3 && !memcmp(bytes, "\xff\xd8\xff", 3)) {
    Rgb r = decode_jpeg(bytes, len, err);
    if (r.p || *err) return r;
  }
  if ((*err = check_png(bytes, len))) return (Rgb){0};
  CFDataRef d = CFDataCreate(0, bytes, (CFIndex)len);
  CGImageSourceRef src = d ? CGImageSourceCreateWithData(d, 0) : 0;
  CGImageRef img = src && CGImageSourceGetCount(src)
                       ? CGImageSourceCreateImageAtIndex(src, 0, 0)
                       : 0;
  Rgb r = {0};
  size_t w = img ? CGImageGetWidth(img) : 0,
         h = img ? CGImageGetHeight(img) : 0;
  if (!img)
    *err = "the image could not be decoded";
  else if (!w || !h)
    *err = "the image is empty";
  else if ((double)w * h > BOMB)
    *err = "the image has more than 178,956,970 pixels";
  else {
    r = (Rgb){(int)w, (int)h, xmalloc(w * h * 3)};
    if (!raw_rgb(img, r.p, w, h)) draw_rgb(img, r.p, w, h);
  }
  if (img) CGImageRelease(img);
  if (src) CFRelease(src);
  if (d) CFRelease(d);
  return r;
}

// Number of 8-bit channels in an RGB pixel, and bits per channel.
static constexpr size_t RGB_CHANNELS = 3, RGB_CHANNEL_BITS = 8;

// Number of 8-bit channels in an RGBA pixel.
static constexpr size_t RGBA_CHANNELS = 4;

uint8_t *image_decode_rgb(const void *data, size_t len, int *width, int *height,
                          const char **err) {
  Rgb image = decode(data, len, err);
  *width = image.w, *height = image.h;
  return image.p;
}

uint8_t *image_decode_alpha(const void *data, size_t len, int *width,
                            int *height, const char **err) {
  *width = *height = 0;
  if ((*err = check_png(data, len))) return 0;
  CFDataRef d = CFDataCreate(0, data, (CFIndex)len);
  CGImageSourceRef src = d ? CGImageSourceCreateWithData(d, 0) : 0;
  CGImageRef img = src && CGImageSourceGetCount(src)
                       ? CGImageSourceCreateImageAtIndex(src, 0, 0)
                       : 0;
  size_t w = img ? CGImageGetWidth(img) : 0,
         h = img ? CGImageGetHeight(img) : 0;
  CGImageAlphaInfo alpha = img ? CGImageGetAlphaInfo(img) : kCGImageAlphaNone;
  uint8_t *out = 0;
  if (!img)
    *err = "the image could not be decoded";
  else if (!w || !h)
    *err = "the image is empty";
  else if ((double)w * h > BOMB)
    *err = "the image has more than 178,956,970 pixels";
  else if (alpha == kCGImageAlphaNone || alpha == kCGImageAlphaNoneSkipFirst ||
           alpha == kCGImageAlphaNoneSkipLast)
    *err = "the image has no alpha channel";
  else {
    out = xmalloc(w * h);
    memset(out, 0, w * h);
    CGContextRef ctx =
        CGBitmapContextCreate(out, w, h, 8, w, 0, kCGImageAlphaOnly);
    if (!ctx) die("cannot draw an image of %zu x %zu pixels", w, h);
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    CGContextRelease(ctx);
    *width = (int)w, *height = (int)h;
  }
  if (img) CGImageRelease(img);
  if (src) CFRelease(src);
  if (d) CFRelease(d);
  return out;
}

int image_read_size(const void *data, size_t len, int *width, int *height,
                    const char **error) {
  *width = *height = 0;
  if ((*error = check_png(data, len))) return 0;
  CFDataRef d = CFDataCreate(0, data, (CFIndex)len);
  CGImageSourceRef src = d ? CGImageSourceCreateWithData(d, 0) : 0;
  CFDictionaryRef props = src && CGImageSourceGetCount(src)
                              ? CGImageSourceCopyPropertiesAtIndex(src, 0, 0)
                              : 0;
  CFNumberRef w =
      props ? CFDictionaryGetValue(props, kCGImagePropertyPixelWidth) : 0;
  CFNumberRef h =
      props ? CFDictionaryGetValue(props, kCGImagePropertyPixelHeight) : 0;
  int read_w = 0, read_h = 0;
  if (!w || !h || !CFNumberGetValue(w, kCFNumberIntType, &read_w) ||
      !CFNumberGetValue(h, kCFNumberIntType, &read_h))
    *error = "the image could not be read";
  else if (read_w <= 0 || read_h <= 0)
    *error = "the image is empty";
  else if ((double)read_w * read_h > BOMB)
    *error = "the image has more than 178,956,970 pixels";
  else
    *width = read_w, *height = read_h;
  if (props) CFRelease(props);
  if (src) CFRelease(src);
  if (d) CFRelease(d);
  return *error == 0;
}

// Validates dimensions and pixel pointer for encoding. Returns 0 with *err set
// if invalid.
static int check_encode_input(const uint8_t *pixels, int width, int height,
                              const char **err) {
  if (!pixels || width <= 0 || height <= 0) {
    *err = "the image is empty";
    return 0;
  }
  if ((double)width * height > BOMB) {
    *err = "the image has more than 178,956,970 pixels";
    return 0;
  }
  return 1;
}

// Encodes width by height rows of channels-byte pixels as a PNG file: RGB, or
// RGBA with straight alpha when alpha is set.
static uint8_t *encode_png(const uint8_t *pixels, int width, int height,
                           int alpha, size_t *length, const char **err) {
  *length = 0;
  *err = 0;
  if (!check_encode_input(pixels, width, height, err)) return 0;
  size_t channels = alpha ? RGBA_CHANNELS : RGB_CHANNELS;
  size_t row = (size_t)width * channels;
  CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGDataProviderRef provider =
      CGDataProviderCreateWithData(0, pixels, row * (size_t)height, 0);
  CGImageRef image =
      space && provider
          ? CGImageCreate(
                (size_t)width, (size_t)height, RGB_CHANNEL_BITS,
                RGB_CHANNEL_BITS * channels, row, space,
                (CGBitmapInfo)(alpha ? kCGImageAlphaLast : kCGImageAlphaNone),
                provider, 0, false, kCGRenderingIntentDefault)
          : 0;
  CFMutableDataRef file = image ? CFDataCreateMutable(0, 0) : 0;
  CGImageDestinationRef dest =
      file ? CGImageDestinationCreateWithData(file, CFSTR("public.png"), 1, 0)
           : 0;
  uint8_t *out = 0;
  if (dest) {
    CGImageDestinationAddImage(dest, image, 0);
    if (CGImageDestinationFinalize(dest)) {
      size_t n = (size_t)CFDataGetLength(file);
      out = xmalloc(n);
      memcpy(out, CFDataGetBytePtr(file), n);
      *length = n;
    }
  }
  if (!out) *err = "the PNG image could not be encoded";
  if (dest) CFRelease(dest);
  if (file) CFRelease(file);
  if (image) CGImageRelease(image);
  if (provider) CGDataProviderRelease(provider);
  if (space) CGColorSpaceRelease(space);
  return out;
}

uint8_t *image_encode_png(const uint8_t *pixels, int width, int height,
                          size_t *length, const char **err) {
  return encode_png(pixels, width, height, 0, length, err);
}

uint8_t *image_encode_png_rgba(const uint8_t *pixels, int width, int height,
                               size_t *length, const char **err) {
  return encode_png(pixels, width, height, 1, length, err);
}

uint8_t *image_encode_jpeg(const uint8_t *pixels, int width, int height,
                           int quality, size_t *length, const char **err) {
  *length = 0;
  *err = 0;
  if (!check_encode_input(pixels, width, height, err)) return 0;
  if (quality < IMAGE_QUALITY_MIN || quality > IMAGE_QUALITY_MAX) {
    *err = "the JPEG quality is out of range";
    return 0;
  }
  struct jpeg_compress_struct c;
  JpegErr je;
  unsigned char *volatile file = 0;
  unsigned long file_length = 0;
  c.err = jpeg_std_error(&je.mgr);
  je.mgr.error_exit = jpeg_fail;
  if (setjmp(je.back)) {
    jpeg_destroy_compress(&c);
    free(file);
    *err = "the JPEG image could not be encoded";
    return 0;
  }
  jpeg_create_compress(&c);
  jpeg_mem_dest(&c, (unsigned char **)&file, &file_length);
  c.image_width = (JDIMENSION)width;
  c.image_height = (JDIMENSION)height;
  c.input_components = (int)RGB_CHANNELS;
  c.in_color_space = JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, quality, TRUE);
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    JSAMPROW row = (JSAMPROW)(pixels + (size_t)c.next_scanline * (size_t)width *
                                           RGB_CHANNELS);
    jpeg_write_scanlines(&c, &row, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  *length = (size_t)file_length;
  return file;
}

// Encodes width by height pixels as a lossy WebP file: RGB, or RGBA with
// straight alpha when alpha is set.
static uint8_t *encode_webp(const uint8_t *pixels, int width, int height,
                            int alpha, int quality, size_t *length,
                            const char **err) {
  *length = 0;
  *err = 0;
  if (!check_encode_input(pixels, width, height, err)) return 0;
  if (quality < IMAGE_QUALITY_MIN || quality > IMAGE_QUALITY_MAX) {
    *err = "the WebP quality is out of range";
    return 0;
  }
  if (width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION) {
    *err = "the image is larger than WebP allows";
    return 0;
  }
  uint8_t *encoded = 0;
  size_t n =
      alpha ? WebPEncodeRGBA(pixels, width, height, width * (int)RGBA_CHANNELS,
                             (float)quality, &encoded)
            : WebPEncodeRGB(pixels, width, height, width * (int)RGB_CHANNELS,
                            (float)quality, &encoded);
  uint8_t *out = 0;
  if (n) {
    out = xmalloc(n);
    memcpy(out, encoded, n);
    *length = n;
  } else {
    *err = "the WebP image could not be encoded";
  }
  WebPFree(encoded);
  return out;
}

uint8_t *image_encode_webp(const uint8_t *pixels, int width, int height,
                           int quality, size_t *length, const char **err) {
  return encode_webp(pixels, width, height, 0, quality, length, err);
}

uint8_t *image_encode_webp_rgba(const uint8_t *pixels, int width, int height,
                                int quality, size_t *length, const char **err) {
  return encode_webp(pixels, width, height, 1, quality, length, err);
}
