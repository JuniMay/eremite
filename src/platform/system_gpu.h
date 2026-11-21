#ifndef EREMITE_PLATFORM_SYSTEM_GPU_H
#define EREMITE_PLATFORM_SYSTEM_GPU_H

// Measured external GPU, display and user-input activity from macOS, used to
// yield GPU breathing room while other apps or the desktop UI are active and
// return to full GPU capacity otherwise.

// Snapshot of system display, input and external GPU activity at one moment.
typedef struct {
  // Active main display refresh period in seconds (1 / Hz), or 0 when no awake
  // display refresh rate is available.
  double display_frame_seconds;
  // Seconds since the last keyboard, mouse, trackpad or scroll event, or
  // INFINITY when unavailable or when the main display is asleep.
  double input_idle_seconds;
  // Seconds since any other process (excluding this process) last submitted
  // GPU work to the accelerator, or INFINITY when unavailable or none exists.
  double external_gpu_idle_seconds;
  // Nonzero when the main display is asleep or absent.
  int display_asleep;
} SystemGpuActivity;

// Reads elapsed seconds since the last user input event (or INFINITY when
// unavailable or asleep), writing whether the main display is asleep to
// *display_asleep when non-null.
double system_gpu_input_idle_seconds(int *display_asleep);

// Reads current display refresh, user-input idle and external GPU activity.
SystemGpuActivity system_gpu_read_activity(void);

#endif
