# eremite

eremite runs large language models locally on Apple silicon. It streams the
weights of mixture-of-experts models from the SSD into an expert cache that
grows and shrinks with free memory, schedules the requests of several models
on one GPU, and serves an OpenAI- and Codex-compatible API.

## Build

Requirements: macOS 27 with Xcode 27.

```sh
make            # the library, the command-line program and the test programs
make test       # tests that need neither a GPU nor model files
make gpu-test   # tests that need a GPU but no model files
```
