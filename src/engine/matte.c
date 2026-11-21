#include "engine/matte.h"
#include "base/error.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Channels of an RGB pixel.
static constexpr size_t RGB_CHANNELS = 3;

// Channels of an RGBA pixel.
static constexpr size_t RGBA_CHANNELS = 4;

// The alpha of an opaque pixel.
static constexpr unsigned OPAQUE = 255;

void matte_keep_masked(uint8_t *edited, const uint8_t *original,
                       const uint8_t *mask, size_t pixels) {
  for (size_t i = 0; i < pixels; i++) {
    unsigned keep = mask[i], change = OPAQUE - keep;
    for (size_t c = 0; c < RGB_CHANNELS; c++) {
      size_t at = i * RGB_CHANNELS + c;
      // Rounds to the nearest level.
      edited[at] =
          (uint8_t)((edited[at] * change + original[at] * keep + OPAQUE / 2) /
                    OPAQUE);
    }
  }
}

void matte_keep_masked_rgba(uint8_t *edited, const uint8_t *original,
                            const uint8_t *mask, size_t pixels) {
  for (size_t i = 0; i < pixels; i++) {
    unsigned keep = mask[i], change = OPAQUE - keep;
    uint8_t *px = edited + i * RGBA_CHANNELS;
    unsigned edit_alpha = px[RGB_CHANNELS];
    // The mixed alpha, in units of 1/255 of a level.
    unsigned alpha = edit_alpha * change + OPAQUE * keep;
    if (alpha == 0) continue;
    // Colors mix weighted by their alpha, and the sum is divided by the
    // mixed alpha to give a straight color again. Rounds to the nearest level.
    for (size_t c = 0; c < RGB_CHANNELS; c++)
      px[c] = (uint8_t)((px[c] * edit_alpha * change +
                         original[i * RGB_CHANNELS + c] * OPAQUE * keep +
                         alpha / 2) /
                        alpha);
    px[RGB_CHANNELS] = (uint8_t)((alpha + OPAQUE / 2) / OPAQUE);
  }
}

// Levels of an 8-bit channel.
static constexpr int LEVELS = 256;

// A channel's lead ranges from -255 to 255; this offset makes it an index.
static constexpr int LEAD_OFFSET = LEVELS - 1;

// The number of different leads.
static constexpr int LEADS = 2 * LEVELS - 1;

// The ratio of a normal distribution's standard deviation to its median
// absolute deviation: 1 over the normal quantile at three quarters.
static constexpr double MAD_TO_SIGMA = 1.4826;

// The smallest lead that a pixel can have: one level.
static constexpr double SMALLEST_LEAD = 1.0;

// Returns how far pixel's channel strong leads its other two channels: its
// level less the larger of theirs.
static int lead_of(const uint8_t *pixel, int strong) {
  int other = 0;
  for (int c = 0; c < (int)RGB_CHANNELS; c++)
    if (c != strong && pixel[c] > other) other = pixel[c];
  return pixel[strong] - other;
}

// Returns the median of the values counted in histogram, whose counts add up
// to total.
static int median_of(const size_t *histogram, int values, size_t total) {
  size_t seen = 0;
  for (int v = 0; v < values; v++) {
    seen += histogram[v];
    if (2 * seen >= total) return v;
  }
  return values - 1;
}

// Returns the median of the values counted in histogram, whose counts add up
// to total, where each whole value stands for the values that round to it:
// value 0 for those from 0 to one half, and value v for those within one half
// of v.
static double interpolated_median_of(const size_t *histogram, int values,
                                     size_t total) {
  double half = (double)total / 2.0, seen = 0.0;
  for (int v = 0; v < values; v++) {
    double count = (double)histogram[v];
    if (count > 0.0 && seen + count >= half) {
      double low = v == 0 ? 0.0 : v - 0.5, width = v == 0 ? 0.5 : 1.0;
      return low + width * (half - seen) / count;
    }
    seen += count;
  }
  return values - 1;
}

// Returns whether pixel (x, y) of a width by height image is on its border.
static int on_border(int x, int y, int width, int height) {
  return x == 0 || y == 0 || x == width - 1 || y == height - 1;
}

int matte_remove_key(const uint8_t *rgb, int width, int height,
                     const uint8_t requested[3], uint8_t *rgba) {
  if (width < 1 || height < 1) return 0;
  // The key: the median of each channel over the border.
  size_t channel_counts[RGB_CHANNELS][LEVELS] = {{0}};
  size_t border = 0;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      if (!on_border(x, y, width, height)) continue;
      const uint8_t *px = rgb + ((size_t)y * width + x) * RGB_CHANNELS;
      for (size_t c = 0; c < RGB_CHANNELS; c++) channel_counts[c][px[c]]++;
      border++;
    }
  uint8_t key[RGB_CHANNELS];
  int strong = 0;
  for (size_t c = 0; c < RGB_CHANNELS; c++) {
    key[c] = (uint8_t)median_of(channel_counts[c], LEVELS, border);
    if (requested[c] > requested[strong]) strong = (int)c;
  }
  if (lead_of(key, strong) <= 0) return 0;

  // The key's lead, and its spread: the median lead of the border pixels that
  // lead in the key's channel, and the median distance of their leads from it.
  size_t *lead_counts = xmalloc(LEADS * sizeof *lead_counts);
  memset(lead_counts, 0, LEADS * sizeof *lead_counts);
  size_t keyed = 0;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      if (!on_border(x, y, width, height)) continue;
      int lead = lead_of(rgb + ((size_t)y * width + x) * RGB_CHANNELS, strong);
      if (lead <= 0) continue;
      lead_counts[lead + LEAD_OFFSET]++;
      keyed++;
    }
  if (!keyed) {
    free(lead_counts);
    return 0;
  }
  int key_lead = median_of(lead_counts, LEADS, keyed) - LEAD_OFFSET;
  size_t spread_counts[LEADS] = {0};
  for (int v = 0; v < LEADS; v++) {
    int distance = abs(v - LEAD_OFFSET - key_lead);
    spread_counts[distance] += lead_counts[v];
  }
  free(lead_counts);
  // The background's noise, as a normal spread: its standard deviation comes
  // from the median distance of the border's leads from the key's lead.
  double sigma =
      MAD_TO_SIGMA * interpolated_median_of(spread_counts, LEADS, keyed);
  // How far below the key's lead the noise of the image's background reaches:
  // the farthest of n normal samples lies about sqrt(2 ln n) standard
  // deviations from their mean.
  size_t pixels = (size_t)width * (size_t)height;
  double reach = sigma * sqrt(2.0 * log((double)pixels));
  // A pixel whose lead is within that reach of the key's lead is transparent,
  // and a pixel with no lead is opaque. When the noise reaches below the
  // smallest lead, every pixel that leads at all is transparent.
  double clear_lead =
      key_lead - reach > SMALLEST_LEAD ? key_lead - reach : SMALLEST_LEAD;

  for (size_t i = 0; i < pixels; i++) {
    const uint8_t *px = rgb + i * RGB_CHANNELS;
    uint8_t *out = rgba + i * RGBA_CHANNELS;
    int lead = lead_of(px, strong);
    double alpha = lead <= 0            ? 1.0
                   : lead >= clear_lead ? 0.0
                                        : 1.0 - (double)lead / clear_lead;
    uint8_t level = (uint8_t)lround(alpha * OPAQUE);
    out[RGB_CHANNELS] = level;
    for (size_t c = 0; c < RGB_CHANNELS; c++) {
      if (level == 0) {
        out[c] = 0;
        continue;
      }
      // The observed color mixes the pixel's own color with the key's by
      // alpha; this unmixes it.
      double own = (px[c] - (1.0 - alpha) * key[c]) / alpha;
      out[c] = (uint8_t)lround(own < 0.0 ? 0.0 : own > OPAQUE ? OPAQUE : own);
    }
  }
  return 1;
}
