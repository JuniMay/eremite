# eremite

eremite runs large language models locally on Apple silicon. It streams the
weights of mixture-of-experts models from the SSD into an expert cache that
grows and shrinks with free memory, schedules the requests of several models
on one GPU, and serves an OpenAI- and Codex-compatible API.

## Build

Requirements: macOS 27 with Xcode 27, and `pkgconf` and `yyjson` from
Homebrew. The GPU code compiles its kernels at run time as Metal 4.1, which
needs macOS 27.

```sh
make            # the library, the command-line program and the test programs
make test       # tests that need neither a GPU nor model files
make gpu-test   # tests that need a GPU but no model files
```

`make THREADS=pthread` builds the threads layer on POSIX threads instead of
Grand Central Dispatch. `make SANITIZE=1` builds with AddressSanitizer and
UndefinedBehaviorSanitizer. Each test ends after `TEST_SECONDS` (default 600),
so that a hang fails the run; `TEST_SECONDS=0` sets no limit.

The tests in `tests/model` need a checkpoint's files and the GPU, and run by
hand. Each one's usage is at the top of its source. For example,
`build/tests/model-tokenizer CHECKPOINT` checks DeepSeek V4.1 Flash's tokenizer
against the reference tokenizer's ids.
