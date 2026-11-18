#include "platform/system_memory.h"
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

// The kernel's page counts. Its free count includes the speculative pages,
// file pages read ahead, which it takes back first.
static vm_statistics64_data_t vm(void) {
  vm_statistics64_data_t s = {0};
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s, &n);
  return s;
}

// A sysctl value, or 0 when there is no such value.
static uint64_t sysctl_u64(const char *name) {
  uint64_t v = 0;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, 0, 0) ? 0 : v;
}

// The file pages other than the speculative ones, which free pages include.
static uint64_t file_of(vm_statistics64_data_t s) {
  uint64_t file = s.external_page_count > s.speculative_count
                      ? s.external_page_count - s.speculative_count
                      : 0;
  return file * vm_page_size;
}

SystemMemoryCounters system_memory_read_counters(void) {
  vm_statistics64_data_t s = vm();
  return (SystemMemoryCounters){(uint64_t)s.free_count * vm_page_size,
                                file_of(s), s.swapouts, s.compressions};
}

void system_memory_read_info(SystemMemoryInfo *m) {
  vm_statistics64_data_t s = vm();
  uint64_t p = vm_page_size;
  struct xsw_usage sw = {0};
  size_t n = sizeof sw;
  sysctlbyname("vm.swapusage", &sw, &n, 0, 0);
  *m = (SystemMemoryInfo){.total = sysctl_u64("hw.memsize"),
                          .free = s.free_count * p,
                          .file = file_of(s),
                          .wired = s.wire_count * p,
                          .compressor = s.compressor_page_count * p,
                          .compressed =
                              s.total_uncompressed_pages_in_compressor * p,
                          .swap_used = sw.xsu_used,
                          .swapouts = s.swapouts,
                          .compressions = s.compressions,
                          .pressure = system_memory_pressure()};
}

SystemMemoryPressure system_memory_pressure(void) {
  int level = SYSTEM_MEMORY_NORMAL;
  size_t len = sizeof level;
  sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, 0, 0);
  return (SystemMemoryPressure)level;
}

uint64_t system_memory_footprint(void) {
  task_vm_info_data_t v = {0};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n);
  return v.phys_footprint;
}

uint64_t system_memory_compressed(int peak) {
  task_vm_info_data_t v = {0};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n);
  return peak ? v.compressed_peak : v.compressed;
}

// The accelerator's "In use system memory" statistic.
uint64_t system_memory_gpu_driver(void) {
  static io_service_t acc;
  if (!acc)
    acc = IOServiceGetMatchingService(kIOMainPortDefault,
                                      IOServiceMatching("IOAccelerator"));
  int64_t v = 0;
  CFDictionaryRef d =
      acc ? IORegistryEntryCreateCFProperty(acc, CFSTR("PerformanceStatistics"),
                                            kCFAllocatorDefault, 0)
          : 0;
  if (d) {
    CFNumberRef n = CFGetTypeID(d) == CFDictionaryGetTypeID()
                        ? CFDictionaryGetValue(d, CFSTR("In use system memory"))
                        : 0;
    if (n) CFNumberGetValue(n, kCFNumberSInt64Type, &v);
    CFRelease(d);
  }
  return v > 0 ? (uint64_t)v : 0;
}
