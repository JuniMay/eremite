#pragma once

// Switching between stacks, keeping the registers that a function call keeps.
// macOS deprecates POSIX's ucontext, so the switch is written in assembly for
// each architecture (context_arm64.S).

// Saves the registers on the current stack and its stack pointer in *save,
// then continues from the stack pointer load.
void context_switch(void **save, void *load);

// Prepares the stack that ends at top so that the first switch to the
// returned stack pointer calls entry, which must not return.
void *context_new(void *top, void (*entry)(void));
