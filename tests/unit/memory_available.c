#include "check.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/sysctl.h>

// Tests the available memory: the free pages, which include the speculative
// ones, and the file cache beyond the share the kernel keeps
// (vm.vm_page_filecache_min). When the share cannot be read, only the free
// pages count. The share comes from a sysctl that the test replaces.

static uint64_t reported_floor;
static int query_failed, queries;

static int fixture_sysctl(const char *name, void *old, size_t *length,
                          void *value, size_t value_length) {
  queries++;
  if (query_failed || strcmp(name, "vm.vm_page_filecache_min")) {
    errno = ENOENT;
    return -1;
  }
  if (!old || !length || *length < sizeof reported_floor || value ||
      value_length)
    return -1;
  memcpy(old, &reported_floor, sizeof reported_floor);
  *length = sizeof reported_floor;
  return 0;
}

#define sysctlbyname fixture_sysctl
#include "platform/system_memory.c"
#undef sysctlbyname

int main(void) {
  vm_statistics64_data_t pages = {
      .free_count = 100, .external_page_count = 500, .speculative_count = 20};
  query_failed = 1;
  check(available_of(pages) == 100 * vm_page_size,
        "an unreadable share counts only the free pages");
  query_failed = 0, reported_floor = 200;
  check(available_of(pages) == (100 + 480 - 200) * vm_page_size,
        "the file cache beyond the kept share counts");
  reported_floor = 450;
  check(available_of(pages) == (100 + 30) * vm_page_size,
        "a larger kept share leaves less");
  reported_floor = 600;
  check(available_of(pages) == 100 * vm_page_size,
        "file cache within the kept share does not count");
  reported_floor = 0;
  check(available_of(pages) == 100 * vm_page_size,
        "a share read as zero counts only the free pages");
  reported_floor = 200, pages.external_page_count = 10;
  check(available_of(pages) == 100 * vm_page_size,
        "speculative pages cannot make the file count negative");
  pages.external_page_count = 220;
  check(available_of(pages) == 100 * vm_page_size,
        "speculative pages count only among the free pages");
  pages.free_count = 0, pages.external_page_count = 500;
  check(available_of(pages) == 280 * vm_page_size,
        "the file cache beyond the share stays available without free pages");
  check(queries > 0, "the available memory reads the kept share");
  return failures != 0;
}
