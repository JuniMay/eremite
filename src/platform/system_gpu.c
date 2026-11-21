#include "platform/system_gpu.h"
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <IOKit/IOKitLib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

// Reads external GPU activity (excluding this process), display refresh period
// and user input idle time from documented macOS CoreGraphics and IOKit APIs.

// Nanoseconds per second for converting IOKit uptime timestamps.
static constexpr double NS_TO_SECONDS = 1e-9;

// Buffer capacity in bytes for parsing IOUserClientCreator strings.
static constexpr size_t CREATOR_CAP = 128;

// Returns the active refresh period in seconds of display, or 0 if unavailable.
static double read_display_frame_seconds(CGDirectDisplayID display) {
  double hz = 0.0;
  CGDisplayModeRef mode = CGDisplayCopyDisplayMode(display);
  if (mode) {
    hz = CGDisplayModeGetRefreshRate(mode);
    CGDisplayModeRelease(mode);
  }
  return hz > 0.0 ? 1.0 / hz : 0.0;
}

// Parses the process id from an IOUserClientCreator string ("pid N, name").
// Returns -1 when the string does not start with a valid pid.
static int parse_creator_pid(CFStringRef creator) {
  if (!creator || CFGetTypeID(creator) != CFStringGetTypeID()) return -1;
  char buf[CREATOR_CAP];
  if (!CFStringGetCString(creator, buf, sizeof buf, kCFStringEncodingUTF8))
    return -1;
  int pid = -1;
  return sscanf(buf, "pid %d", &pid) == 1 ? pid : -1;
}

// Updates *max_last_ns with the largest lastSubmittedTime found in an
// AGXDeviceUserClient AppUsage array.
static void scan_app_usage(CFArrayRef usage, uint64_t *max_last_ns) {
  if (!usage || CFGetTypeID(usage) != CFArrayGetTypeID()) return;
  CFIndex count = CFArrayGetCount(usage);
  for (CFIndex i = 0; i < count; i++) {
    CFDictionaryRef entry = CFArrayGetValueAtIndex(usage, i);
    if (!entry || CFGetTypeID(entry) != CFDictionaryGetTypeID()) continue;
    CFNumberRef num = CFDictionaryGetValue(entry, CFSTR("lastSubmittedTime"));
    if (!num || CFGetTypeID(num) != CFNumberGetTypeID()) continue;
    int64_t submitted = 0;
    if (CFNumberGetValue(num, kCFNumberSInt64Type, &submitted) &&
        submitted > 0 && (uint64_t)submitted > *max_last_ns)
      *max_last_ns = (uint64_t)submitted;
  }
}

// Reads elapsed seconds since any process other than self_pid last submitted
// GPU work to IOAccelerator, or INFINITY if unavailable or never submitted.
static double read_external_gpu_idle_seconds(int self_pid) {
  static io_service_t acc;
  if (!acc)
    acc = IOServiceGetMatchingService(kIOMainPortDefault,
                                      IOServiceMatching("IOAccelerator"));
  if (!acc) return INFINITY;

  io_iterator_t iter = IO_OBJECT_NULL;
  if (IORegistryEntryGetChildIterator(acc, kIOServicePlane, &iter) !=
      KERN_SUCCESS)
    return INFINITY;

  uint64_t max_last_ns = 0;
  io_registry_entry_t client;
  while ((client = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
    CFTypeRef creator = IORegistryEntryCreateCFProperty(
        client, CFSTR("IOUserClientCreator"), kCFAllocatorDefault, 0);
    int pid = parse_creator_pid(creator);
    if (creator) CFRelease(creator);

    if (pid != self_pid) {
      CFTypeRef usage = IORegistryEntryCreateCFProperty(
          client, CFSTR("AppUsage"), kCFAllocatorDefault, 0);
      scan_app_usage(usage, &max_last_ns);
      if (usage) CFRelease(usage);
    }
    IOObjectRelease(client);
  }
  IOObjectRelease(iter);

  if (!max_last_ns) return INFINITY;
  uint64_t now_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  if (now_ns <= max_last_ns) return 0.0;
  return (double)(now_ns - max_last_ns) * NS_TO_SECONDS;
}

double system_gpu_input_idle_seconds(int *display_asleep) {
  int asleep = 1;
  double idle = INFINITY;
  CGDirectDisplayID main_disp = CGMainDisplayID();
  if (main_disp != kCGNullDirectDisplay) {
    asleep = CGDisplayIsAsleep(main_disp) ? 1 : 0;
    if (!asleep) {
      double s = CGEventSourceSecondsSinceLastEventType(
          kCGEventSourceStateCombinedSessionState, kCGAnyInputEventType);
      if (isfinite(s) && s >= 0.0) idle = s;
    }
  }
  if (display_asleep) *display_asleep = asleep;
  return idle;
}

SystemGpuActivity system_gpu_read_activity(void) {
  SystemGpuActivity out = {.display_frame_seconds = 0.0,
                           .input_idle_seconds = INFINITY,
                           .external_gpu_idle_seconds = INFINITY,
                           .display_asleep = 1};
  out.input_idle_seconds = system_gpu_input_idle_seconds(&out.display_asleep);
  if (!out.display_asleep) {
    CGDirectDisplayID main_disp = CGMainDisplayID();
    if (main_disp != kCGNullDirectDisplay)
      out.display_frame_seconds = read_display_frame_seconds(main_disp);
  }
  out.external_gpu_idle_seconds = read_external_gpu_idle_seconds((int)getpid());
  return out;
}
