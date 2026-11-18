#include "check.h"
#include "platform/system_memory.h"
#include <stdlib.h>
#include <string.h>
#include <mach/vm_page_size.h>
#include <sys/sysctl.h>

// Tests the memory statistics against this Mac's own figures: the totals
// agree, the counts fit in the memory, a touched allocation raises this
// process's footprint, and the pressure is one of the kernel's levels.

// The bytes the test touches to raise its footprint, and the least rise it
// accepts. The kernel may count a few pages late.
static constexpr size_t TOUCHED = 64 << 20, LEAST_RISE = 48 << 20;

static uint64_t memsize(void) {
  uint64_t bytes = 0;
  size_t length = sizeof bytes;
  sysctlbyname("hw.memsize", &bytes, &length, 0, 0);
  return bytes;
}

// Returns whether level is one of the kernel's pressure levels.
static int pressure_level(int level) {
  return level == SYSTEM_MEMORY_NORMAL || level == SYSTEM_MEMORY_WARNING ||
         level == SYSTEM_MEMORY_CRITICAL;
}

int main(void) {
  SystemMemoryInfo info;
  system_memory_read_info(&info);
  uint64_t total = memsize();
  check(total > 0 && info.total == total, "the total is the Mac's RAM");
  check(info.free <= total && info.file <= total,
        "the free pages and the file pages fit in the RAM");
  check(info.wired > 0 && info.wired <= total,
        "the wired pages are some of the RAM");
  check(pressure_level(info.pressure) &&
            pressure_level(system_memory_pressure()),
        "the pressure is one of the kernel's levels");

  SystemMemoryCounters first = system_memory_read_counters();
  check(first.free <= total && first.file <= total &&
            first.free + first.file <= total,
        "the free pages and the file cache fit in the RAM together");
  SystemMemoryCounters second = system_memory_read_counters();
  check(second.swapouts >= first.swapouts &&
            second.compressions >= first.compressions,
        "the swap-outs and compressions since boot never fall");

  uint64_t before = system_memory_footprint();
  char *touched = malloc(TOUCHED);
  memset(touched, 1, TOUCHED);
  // Reading one byte of each page keeps the compiler from removing the
  // allocation.
  size_t pages = 0;
  for (size_t i = 0; i < TOUCHED; i += vm_page_size) pages += touched[i];
  uint64_t after = system_memory_footprint();
  check(pages == TOUCHED / vm_page_size, "every touched page holds its byte");
  check(before > 0 && after >= before + LEAST_RISE,
        "touching 64 MiB raises the footprint by at least 48 MiB");
  free(touched);

  check(system_memory_compressed(1) >= system_memory_compressed(0),
        "the compressed peak is at least the compressed bytes now");
  check(system_memory_gpu_driver() > 0, "the GPU driver holds some memory");
  return failures != 0;
}
