# eremite

eremite runs large language models locally on Apple silicon, with a C and
Metal engine and a Rust server. README.md covers building, testing and use,
`docs/server.md` the server's settings, and `docs/performance/` measured speed.

## Build and test

- `make`, `make test`, `make gpu-test`, `make server` and `make server-test`.
  The model tests need a checkpoint; README.md says how to run them.
- `make format-check` and `cargo clippy --release --locked --all-targets` must
  be clean. Never format `.metal` files.
- Run one model process at a time.

## Modules

- `src/`: `base/`, `platform/`, `gpu/` and `kernels/` (Metal), `engine/`
  (memory, expert cache, passes, weights, tokenizer, sampling, lookup, saved
  states; its C interface is `eremite.h`), `model/` (`deepseek/`, and `lfm3/`
  for d1) and `cli/`.
- `server/src/`: configuration, engine binding, worker, state cache, HTTP,
  chat formats, Codex and web search, decisions and the terminal UI.
- `tests/`, `server/tests/`, and `scripts/` for SearXNG and model downloads.

## Requirements

- Settings come from one JSON file. Requests set their own sampling.
- No fixed times or thresholds in logic: use events or measured values. A
  limit on a wait that may never end is a setting with a default.
- Judge policies by measurement: the worst slowdown of any request. Changes
  must not slow down decoding or prefill. Speed results come from measured
  runs.
- A speedup keeps results the same, unless its setting says it approximates.
- When a feature fails in measurement, fix it. Do not switch it off.
- Assume nothing is installed: scripts set things up, and defaults name only
  eremite's own folders. No machine-specific paths.
- A model's "state" is what it keeps between calls; its "status" is what it
  is doing.
- Docs and comments describe eremite as it is, not its history.

## Style

- Plain English: short, flat sentences and familiar words.
- Every function, type, field, constant and enum variant has a short comment.
  A field's comment goes above it. No examples, no references to other
  functions.
- A name says exactly what the function does, in the words of its comment.
- No unexplained numbers: name each one and give its reason.
- C: include guards, `static constexpr` constants, few globals, and no hidden
  state: pass the handles a function uses as arguments.

## Commits

- Docs change in the same commit as the code they describe.
- Bug fixes: a test that fails first, then the fix.
- Messages: `type(scope): subject`, with a type such as feat, fix or docs,
  then the problem, the change, results, tests and limits.
