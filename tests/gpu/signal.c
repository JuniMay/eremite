#include "check.h"
#include "base/error.h"
#include "gpu/gpu.h"
#include <signal.h>
#include <spawn.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests signals and waits: after a wait for a signal, the CPU reads what the
// GPU wrote before it, and a wait that cannot end ends the program instead of
// returning. Those cases run in a copy of this program, because Metal does not
// work after fork.

extern char **environ;

static const char library[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void add(device const uint *x [[buffer(0)]],\n"
    "                device uint *y [[buffer(1)]],\n"
    "                constant uint &n [[buffer(2)]]) {\n"
    "  y[0] = x[0] + n;\n"
    "}\n";

// Encodes y = x + 1.
static void add_one(GpuCommandBuffer *c, int add, const Buf *x, const Buf *y) {
  uint32_t one = 1;
  gpu_select_kernel(c, add);
  gpu_bind_buffer(c, 0, x, 0);
  gpu_bind_buffer(c, 1, y, 0);
  gpu_copy_bytes(c, 2, &one, sizeof one);
  gpu_run_kernel(c, 1, 1, 1, 1, 1, 1);
}

// Runs one case that must end the program; the copy returns if it does not.
static void end_case(const char *how) {
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuEvent *event = gpu_event_new(gpu);
  if (!strcmp(how, "--never-given")) gpu_wait_signal(event, 1);
  if (!strcmp(how, "--never-committed")) {
    GpuCommandBuffer c = gpu_begin(gpu);
    uint64_t value = gpu_signal(&c, event);
    gpu_set_signal_timeout(gpu, 0.5);
    gpu_wait_signal(event, value);
  }
  if (!strcmp(how, "--another-gpu")) {
    GpuCommandBuffer c = gpu_begin(gpu_new(library, sizeof library - 1));
    gpu_signal(&c, event);
  }
}

// Runs a copy of this program with the argument how. Returns whether the copy
// ended with an error within 20 s, saying expected. Every case ends within a
// second; a copy still running after 20 s is stopped.
static int ends_with(const char *self, const char *how, const char *expected) {
  int out[2];
  if (pipe(out)) die("pipe");
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out[1], 1);
  posix_spawn_file_actions_adddup2(&actions, out[1], 2);
  posix_spawn_file_actions_addclose(&actions, out[0]);
  char *args[] = {(char *)self, (char *)how, NULL};
  pid_t pid;
  if (posix_spawn(&pid, self, &actions, NULL, args, environ))
    die("cannot run %s", self);
  posix_spawn_file_actions_destroy(&actions);
  close(out[1]);
  int status = 0, done = 0;
  for (int tenths = 0; tenths < 200 && !done; tenths++) {
    done = waitpid(pid, &status, WNOHANG) == pid;
    if (!done) usleep(100000);
  }
  if (!done) kill(pid, SIGKILL), waitpid(pid, &status, 0);
  char said[512] = {0};
  for (ssize_t n = 0, k; n + 1 < (ssize_t)sizeof said &&
                         (k = read(out[0], said + n, sizeof said - 1 - n)) > 0;)
    n += k;
  close(out[0]);
  return done && WIFEXITED(status) && WEXITSTATUS(status) != 0 &&
         strstr(said, expected);
}

int main(int argc, char **argv) {
  if (argc == 2) {
    end_case(argv[1]);
    return 0;
  }
  Memory *memory = memory_new();
  Gpu *gpu = gpu_new(library, sizeof library - 1);
  GpuEvent *event = gpu_event_new(gpu);
  int add = gpu_find_kernel(gpu, "add");
  Buf x = gpu_alloc(gpu, memory_shared(memory), 4),
      y = gpu_alloc(gpu, memory_shared(memory), 4),
      z = gpu_alloc(gpu, memory_shared(memory), 4);
  ((uint32_t *)x.ptr)[0] = 41;
  GpuCommandBuffer c = gpu_begin(gpu);
  add_one(&c, add, &x, &y);
  uint64_t value = gpu_signal(&c, event);
  add_one(&c, add, &y, &z);
  gpu_commit(&c);
  gpu_wait_signal(event, value);
  check(
      value == 1 && ((uint32_t *)y.ptr)[0] == 42,
      "after a wait for a signal, the CPU reads what the GPU wrote before it");
  gpu_sync(gpu);
  check(((uint32_t *)z.ptr)[0] == 43,
        "the work after a signal runs in the same command buffer");

  check(ends_with(argv[0], "--never-given", "never gave out"),
        "a wait for a value the event never gave out ends the program without "
        "waiting");
  check(ends_with(argv[0], "--never-committed", "not signaled in 0.5 s"),
        "a wait for a signal that does not come ends the program after the "
        "timeout");
  check(ends_with(argv[0], "--another-gpu", "another GPU"),
        "a signal on an event of another GPU ends the program");
  return failures != 0;
}
