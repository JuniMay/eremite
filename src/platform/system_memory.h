#ifndef EREMITE_PLATFORM_SYSTEM_MEMORY_H
#define EREMITE_PLATFORM_SYSTEM_MEMORY_H
#include <stdint.h>

// The memory of the system as macOS reports it: the kernel's page counters
// and pressure level, this process's footprint, and the GPU driver's memory.

// The kernel's page counters at one moment, in bytes.
typedef struct {
  // The free pages. They include the speculative file pages.
  uint64_t free;
  // The file cache: the file pages other than the speculative ones.
  uint64_t file;
  // The pages swapped out since boot.
  uint64_t swapouts;
  // The pages compressed since boot.
  uint64_t compressions;
  // The machine's RAM; 0 when unknown.
  uint64_t total;
} SystemMemoryCounters;

// Reads the kernel's page counters.
SystemMemoryCounters system_memory_read_counters(void);

// The kernel's memory pressure levels, with the kernel's own numbers.
typedef enum {
  // Memory is plentiful.
  SYSTEM_MEMORY_NORMAL = 1,
  // Memory runs short. The kernel compresses and swaps more.
  SYSTEM_MEMORY_WARNING = 2,
  // Memory is almost gone. The kernel may end processes.
  SYSTEM_MEMORY_CRITICAL = 4
} SystemMemoryPressure;

// The system's memory at one moment, for a status display. The fields are
// bytes, except the page counts since boot and the pressure.
typedef struct {
  // The RAM, the free pages and the file cache.
  uint64_t total, free, file;
  // The wired pages, the compressor's pages, and the uncompressed bytes that
  // the compressor holds.
  uint64_t wired, compressor, compressed;
  // The bytes in swap files.
  uint64_t swap_used;
  // The pages swapped out and the pages compressed since boot.
  uint64_t swapouts, compressions;
  // The kernel's memory pressure.
  SystemMemoryPressure pressure;
} SystemMemoryInfo;

// Reads the system's memory into m.
void system_memory_read_info(SystemMemoryInfo *m);

// Returns the kernel's memory pressure.
SystemMemoryPressure system_memory_pressure(void);

// Returns this process's physical footprint, in bytes.
uint64_t system_memory_footprint(void);

// Returns this process's memory in the compressor, in bytes. With peak set, it
// returns the most so far.
uint64_t system_memory_compressed(int peak);

// Returns the GPU driver's resident memory for all processes, in bytes.
uint64_t system_memory_gpu_driver(void);

#endif
