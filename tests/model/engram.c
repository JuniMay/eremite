#include "check.h"
#include "base/error.h"
#include "model/deepseek/engram.h"
#include <stdint.h>
#include <stdio.h>

// Tests Engram's layout, which a DeepSeek V4.1 Flash checkpoint's tokenizer
// gives, against the reference table: an FNV-1a digest of the multipliers,
// primes, offsets, pad, count and every token's normalized id.
//
// usage: model-engram CHECKPOINT

static uint64_t digest(uint64_t hash, const void *data, size_t size) {
  const uint8_t *bytes = data;
  for (size_t i = 0; i < size; i++) hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: model-engram CHECKPOINT");
  Engram layout;
  engram_init(&layout, argv[1]);
  uint64_t hash = digest(1469598103934665603ull, layout.multipliers,
                         sizeof layout.multipliers);
  hash = digest(hash, layout.primes, sizeof layout.primes);
  hash = digest(hash, layout.offsets, sizeof layout.offsets);
  hash = digest(hash, &layout.pad, sizeof layout.pad);
  hash = digest(hash, &layout.count, sizeof layout.count);
  hash = digest(hash, layout.map, (size_t)layout.count * 4);
  engram_free(&layout);
  printf("digest %016llx\n", (unsigned long long)hash);
  check(hash == UINT64_C(0x25662602af90df34),
        "the layout matches the reference table");
  return failures != 0;
}
