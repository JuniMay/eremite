#include "engine/media.h"
#include <stdlib.h>

static uint64_t mix(uint64_t x) {  // splitmix64's finalizer
  x ^= x >> 30, x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27, x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

int media_span_length(const Media *im) { return im->ntokens; }

int media_expand_prompt(const int *in, int n, Media *im, int nim, int *out,
                        int max, int media_token) {
  int k = 0, m = 0;
  for (int i = 0; i < n; i++) {
    if (in[i] != media_token) {
      if (m < max) out[m] = in[i];
      m++;
      continue;
    }
    if (k == nim) return -1;
    int s = media_span_length(&im[k]);
    im[k].start = m;
    if (m + s <= max) media_write_keys(&im[k], out + m);
    m += s, k++;
  }
  return k == nim ? m : -1;
}

void media_write_keys(const Media *im, int *keys) {
  for (int j = 0, n = media_span_length(im); j < n; j++)
    keys[j] = -1 - (int)(mix(im->hash + (uint64_t)j) & 0x3fffffff);
}

void media_free(Media *im) {
  free(im->patch);
  if (im->rows) memory_host_free(im->memory, im->rows);
  *im = (Media){0};
}
