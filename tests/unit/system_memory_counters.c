#include "check.h"
#include <stdint.h>

// Tests the file cache in the kernel's page counters: the file pages other
// than the speculative ones, which the free pages include.

#include "platform/system_memory.c"

int main(void) {
  vm_statistics64_data_t pages = {
      .free_count = 100, .external_page_count = 500, .speculative_count = 20};
  check(file_of(pages) == 480 * vm_page_size,
        "the file cache leaves out the speculative pages");
  pages.external_page_count = 10;
  check(file_of(pages) == 0,
        "speculative pages cannot make the file cache negative");
  return failures != 0;
}
