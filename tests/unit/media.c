#include "check.h"
#include "engine/media.h"
#include <stdlib.h>
#include <string.h>

// Tests the images in a prompt: an image's span has its own negative keys,
// which follow from its file's hash; expanding a prompt replaces each
// placeholder with the next image's span and records where it starts; a
// prompt with another number of placeholders than images is refused; and
// freeing an image gives back its feature rows' memory.

// The placeholder token, and the tokens around the placeholders.
static constexpr int PLACEHOLDER = 99, BEFORE = 5, BETWEEN = 6, AFTER = 7;

int main(void) {
  Media images[2] = {{.ntokens = 3, .hash = 1234}, {.ntokens = 2, .hash = 99}};
  check(media_span_length(&images[0]) == 3,
        "an image's span has its tokens' count");

  int first[3], again[3], other[3];
  media_write_keys(&images[0], first);
  media_write_keys(&images[0], again);
  Media same_count = {.ntokens = 3, .hash = 4321};
  media_write_keys(&same_count, other);
  int negative = 1, distinct = first[0] != first[1] && first[1] != first[2] &&
                               first[0] != first[2];
  for (int i = 0; i < 3; i++) negative &= first[i] < 0;
  check(negative && distinct, "a span's keys are negative and distinct");
  check(!memcmp(first, again, sizeof first) &&
            memcmp(first, other, sizeof first),
        "the keys follow from the file's hash");

  const int prompt[] = {BEFORE, PLACEHOLDER, BETWEEN, PLACEHOLDER, AFTER};
  int out[16], keys0[3], keys1[2];
  int count = media_expand_prompt(prompt, 5, images, 2, out, 16, PLACEHOLDER);
  media_write_keys(&images[0], keys0);
  media_write_keys(&images[1], keys1);
  check(count == 8 && out[0] == BEFORE &&
            !memcmp(out + 1, keys0, sizeof keys0) && out[4] == BETWEEN &&
            !memcmp(out + 5, keys1, sizeof keys1) && out[7] == AFTER,
        "each placeholder becomes the next image's span");
  check(images[0].start == 1 && images[1].start == 5,
        "each image records where its span starts");

  check(media_expand_prompt(prompt, 5, images, 1, out, 16, PLACEHOLDER) == -1 &&
            media_expand_prompt(prompt, 3, images, 2, out, 16, PLACEHOLDER) ==
                -1,
        "a prompt with another number of placeholders than images is refused");

  int small[4];
  memset(small, 0, sizeof small);
  int needed = media_expand_prompt(prompt, 5, images, 2, small, 3, PLACEHOLDER);
  check(needed == 8 && small[3] == 0,
        "with too little room, the count is still returned, and nothing is "
        "written past the room");

  Memory *memory = memory_new();
  memory_set_limit(memory, 1 << 20);
  MemorySlot *shared = memory_shared(memory);
  Media image = {.ntokens = 4, .patch = malloc(64), .memory = shared};
  image.rows = memory_host_alloc(shared, 4096);
  uint64_t held = memory_usage(memory).allocated;
  media_free(&image);
  check(held > 0 && memory_usage(memory).allocated == 0 && !image.patch &&
            !image.rows && !image.ntokens,
        "freeing an image gives back its rows' memory and clears it");
  return failures != 0;
}
