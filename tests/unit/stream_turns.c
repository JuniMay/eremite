#include "check.h"
#include "base/clock.h"
#include "engine/pass.h"
#include "gpu/gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests multi-model turn scheduling in the engine's stream loop on a virtual
// clock without a Metal device: symmetric time sharing between asymmetric
// video, decision and language-model workloads, dynamic solo-to-shared
// transitions, intra-model decode priority, and off-GPU background waits.

// Stack size in bytes for each mock coroutine pass.
static constexpr size_t MOCK_STACK_BYTES = 1u << 20;

// Maximum committed virtual GPU jobs tracked at once.
static constexpr int MAX_GPU_JOBS = 16384;

// Sentinel address assigned to the engine's global gpu handle in unit tests.
static int fake_gpu_sentinel;

// Current virtual time in seconds.
static double fake_clock;

// Last committed virtual GPU command buffer number.
static uint64_t fake_committed;

// Completion timestamp of each virtual GPU command buffer.
static double fake_done_at[MAX_GPU_JOBS];

// Number of readiness polls seen for each virtual GPU command buffer.
static int fake_polls[MAX_GPU_JOBS];

// Engine-thread state transitions recorded by the test.
static int saw_wait_read_state;

// Returns the virtual clock time in seconds.
static double fake_now(void) { return fake_clock; }

// Returns the latest committed virtual command buffer number.
static uint64_t fake_gpu_last_committed(Gpu *g) {
  (void)g;
  return fake_committed;
}

// Returns whether virtual command buffer number has finished. Yields once on
// the initial check inside pass_wait, then advances virtual time to completion
// when polled by the scheduler.
static int fake_gpu_has_finished(Gpu *g, uint64_t number) {
  (void)g;
  if (!number) return 1;
  uint64_t slot = number % MAX_GPU_JOBS;
  if (fake_clock >= fake_done_at[slot]) return 1;
  if (++fake_polls[slot] >= 2) {
    fake_clock = fake_done_at[slot];
    return 1;
  }
  return 0;
}

// Advances the virtual clock to the completion time of command buffer number.
static void fake_gpu_wait_until_finished(Gpu *g, uint64_t number) {
  (void)g;
  if (!number) return;
  double target = fake_done_at[number % MAX_GPU_JOBS];
  if (fake_clock < target) fake_clock = target;
}

// Records engine-thread states observed during scheduling.
static void fake_gpu_set_thread_state(Gpu *g, int state) {
  (void)g;
  if (state == 3) saw_wait_read_state = 1;
}

// Commits a virtual GPU command buffer taking duration seconds and returns its
// command buffer number.
static uint64_t fake_gpu_submit(double duration) {
  double start = fake_clock;
  if (fake_committed) {
    double tail = fake_done_at[fake_committed % MAX_GPU_JOBS];
    if (tail > start) start = tail;
  }
  uint64_t number = ++fake_committed;
  fake_done_at[number % MAX_GPU_JOBS] = start + duration;
  fake_polls[number % MAX_GPU_JOBS] = 0;
  return number;
}

// Context for waiting on a virtual GPU command buffer inside a mock pass.
typedef struct {
  // Command buffer number to wait for.
  uint64_t number;
} FakeGpuWait;

// Returns whether the virtual GPU command buffer in context is finished.
static int fake_gpu_wait_ready(void *context) {
  FakeGpuWait *w = context;
  return fake_gpu_has_finished(NULL, w->number);
}

// Advances the virtual clock until the virtual GPU command buffer finishes.
static void fake_gpu_wait_block(void *context) {
  FakeGpuWait *w = context;
  fake_gpu_wait_until_finished(NULL, w->number);
}

// Submits a virtual GPU job of duration seconds and yields until it finishes.
static void mock_run_gpu(double duration) {
  uint64_t number = fake_gpu_submit(duration);
  FakeGpuWait wait = {.number = number};
  pass_wait(&(PassWait){.ready = fake_gpu_wait_ready,
                        .block = fake_gpu_wait_block,
                        .context = &wait,
                        .kind = PASS_WAIT_GPU});
}

#define now fake_now
#define gpu_last_committed fake_gpu_last_committed
#define gpu_has_finished fake_gpu_has_finished
#define gpu_wait_until_finished fake_gpu_wait_until_finished
#define gpu_set_thread_state fake_gpu_set_thread_state
#include "engine/engine.c"
#undef now
#undef gpu_last_committed
#undef gpu_has_finished
#undef gpu_wait_until_finished
#undef gpu_set_thread_state

// Role of a mock model in a test scenario.
typedef enum {
  // Language model supporting prefill and decode.
  MOCK_ROLE_LLM = 0,
  // Video generation model supporting generate_video.
  MOCK_ROLE_VIDEO = 1,
  // Decision model supporting decide.
  MOCK_ROLE_DECIDE = 2
} MockRole;

// Per-model configuration and accounting for mock models.
typedef struct {
  // Number of layers or slices per call.
  int steps;
  // Virtual GPU seconds consumed per layer, slice or decode call.
  double step_s;
  // Nonzero to commit GPU work without waiting before calling wait (like d1).
  int deferred_gpu;
  // Nonzero to include a simulated read wait inside a decode call.
  int with_read_wait;
  // Nonzero to wait on a background off-GPU task after GPU steps finish.
  int with_background_wait;
  // Background task completion flag toggled by the test driver.
  int background_done;
  // Event registered by the background wait's notify callback.
  ThreadsEvent *background_event;
  // Total virtual GPU seconds executed by this model.
  double total_gpu_s;
  // Number of layers or decode calls completed by this model.
  int completed_steps;
  // Latest other-passes count delivered by the engine.
  int other_passes;
  // Latest remaining measured turn budget in seconds delivered by the engine.
  double turn_budget;
  // Largest positive turn budget observed during contended steps.
  double max_turn_budget;
  // Nonzero if this model ever observed other_passes == 0 during a step.
  int saw_solo;
  // Nonzero if this model ever observed other_passes > 0 during a step.
  int saw_contended;
  // Nonzero if the model returned to solo after having been contended.
  int saw_solo_after_contended;
  // Dummy logits buffer returned by mock prefill, decode and decide calls.
  float logits[4];
} MockModel;

// Maximum mock models active in one test run.
static constexpr int MOCK_MAX = 4;

// State for each mock model slot.
static MockModel mocks[MOCK_MAX];

// Returns whether the mock background off-GPU task has completed.
static int mock_bg_ready(void *context) {
  MockModel *mm = context;
  return mm->background_done;
}

// Registers the wake event for the mock background off-GPU task.
static void mock_bg_notify(void *context, ThreadsEvent *event) {
  MockModel *mm = context;
  mm->background_event = event;
}

// Context for a simulated read wait that finishes on its second check.
typedef struct {
  // Number of readiness checks performed.
  int checks;
  // Virtual time added when the simulated read finishes.
  double duration;
} MockReadWait;

// Returns whether the simulated read wait has finished.
static int mock_read_ready(void *context) {
  MockReadWait *rw = context;
  return rw->checks > 0;
}

// Simulates notifying the wake event once a background read completes.
static void mock_read_notify(void *context, ThreadsEvent *event) {
  MockReadWait *rw = context;
  rw->checks++;
  fake_clock += rw->duration;
  threads_event_signal(event);
}

// Runs a mock pass's multi-step GPU loop for slot idx.
static void mock_run_steps(int idx, const PassWait *wait) {
  MockModel *mm = &mocks[idx];
  for (int i = 0; i < mm->steps; i++) {
    if (mm->other_passes == 0 && mm->turn_budget <= 0.0) {
      mm->saw_solo = 1;
      if (mm->saw_contended) mm->saw_solo_after_contended = 1;
    } else {
      mm->saw_contended = 1;
    }
    if (mm->deferred_gpu) {
      if (wait) pass_wait(wait);
      fake_gpu_submit(mm->step_s);
      mm->total_gpu_s += mm->step_s;
      mm->completed_steps++;
    } else {
      mock_run_gpu(mm->step_s);
      mm->total_gpu_s += mm->step_s;
      mm->completed_steps++;
      if (wait) pass_wait(wait);
    }
  }
  if (mm->deferred_gpu && fake_committed) {
    FakeGpuWait gw = {.number = fake_committed};
    pass_wait(&(PassWait){.ready = fake_gpu_wait_ready,
                          .block = fake_gpu_wait_block,
                          .context = &gw,
                          .kind = PASS_WAIT_GPU});
  }
  if (mm->with_background_wait) {
    pass_wait(&(PassWait){.ready = mock_bg_ready,
                          .context = mm,
                          .kind = PASS_WAIT_OTHER,
                          .notify = mock_bg_notify});
  }
}

// Runs a mock decoding call for slot idx.
static const float *mock_decode_slot(int idx) {
  MockModel *mm = &mocks[idx];
  if (mm->other_passes == 0 && mm->turn_budget <= 0.0) {
    mm->saw_solo = 1;
    if (mm->saw_contended) mm->saw_solo_after_contended = 1;
  } else {
    mm->saw_contended = 1;
  }
  if (mm->with_read_wait) {
    MockReadWait rw = {.checks = 0, .duration = mm->step_s * 0.5};
    pass_wait(&(PassWait){.ready = mock_read_ready,
                          .context = &rw,
                          .kind = PASS_WAIT_READ,
                          .notify = mock_read_notify});
    mock_run_gpu(mm->step_s * 0.5);
  } else {
    mock_run_gpu(mm->step_s);
  }
  mm->total_gpu_s += mm->step_s;
  mm->completed_steps++;
  return mm->logits;
}

// Allocates a coroutine pass for a mock model.
static Pass *mock_pass_new(void (*body)(void *), void *argument) {
  return pass_new(body, argument, MOCK_STACK_BYTES);
}

// No-op pass begin callback for mock models.
static void mock_pass_begin(void) {}

// No-op pass end callback for mock models.
static void mock_pass_end(void) {}

// Returns a fixed vocabulary size for mock language models.
static int mock_vocab_size(void) { return 4; }

// Returns a zero replay start offset for mock language models.
static int mock_replay_start(int from) { return from; }

#define DEFINE_MOCK_OPS(IDX)                                                   \
  static void mock_set_other_##IDX(int passes) {                               \
    mocks[IDX].other_passes = passes;                                          \
  }                                                                            \
  static void mock_set_budget_##IDX(double seconds) {                          \
    mocks[IDX].turn_budget = seconds;                                          \
    if (seconds > mocks[IDX].max_turn_budget)                                  \
      mocks[IDX].max_turn_budget = seconds;                                    \
  }                                                                            \
  static const float *mock_prefill_##IDX(                                      \
      ModelState *s, const int *tokens, int n, int from, int replay,           \
      int capture, ModelSnapshot **captured, float *every,                     \
      const PassWait *wait) {                                                  \
    (void)s, (void)tokens, (void)n, (void)from, (void)replay, (void)capture,   \
        (void)captured, (void)every;                                           \
    mock_run_steps(IDX, wait);                                                 \
    return mocks[IDX].logits;                                                  \
  }                                                                            \
  static const float *mock_decode_##IDX(                                       \
      ModelState *const *states, const int *tokens, const int *pos, int m) {   \
    (void)states, (void)tokens, (void)pos, (void)m;                            \
    return mock_decode_slot(IDX);                                              \
  }                                                                            \
  static const float *mock_decide_##IDX(                                       \
      const int *tokens, int n, int state_tokens, const int *question_ends,    \
      int questions, const int *rows, const int *ids, int count,               \
      const Media *media, int nmedia, const PassWait *wait) {                  \
    (void)tokens, (void)n, (void)state_tokens, (void)question_ends,            \
        (void)questions, (void)rows, (void)ids, (void)count, (void)media,      \
        (void)nmedia;                                                          \
    mock_run_steps(IDX, wait);                                                 \
    return mocks[IDX].logits;                                                  \
  }                                                                            \
  static int mock_video_##IDX(const VideoRequest *req, const PassWait *wait,   \
                              const int *cancelled, VideoResult *out) {        \
    (void)req, (void)cancelled;                                                \
    memset(out, 0, sizeof *out);                                               \
    mock_run_steps(IDX, wait);                                                 \
    out->data = xmalloc(1);                                                    \
    out->len = 1;                                                              \
    return 1;                                                                  \
  }                                                                            \
  static const ModelOps mock_llm_ops_##IDX = {                                 \
      .family = "MockLLM",                                                     \
      .vocab_size = mock_vocab_size,                                           \
      .replay_start = mock_replay_start,                                       \
      .prefill = mock_prefill_##IDX,                                           \
      .decode_rows = mock_decode_##IDX,                                        \
      .pass_new = mock_pass_new,                                               \
      .pass_begin = mock_pass_begin,                                           \
      .pass_end = mock_pass_end,                                               \
      .set_other_passes = mock_set_other_##IDX,                                \
      .set_turn_budget = mock_set_budget_##IDX,                                \
  };                                                                           \
  static const ModelOps mock_video_ops_##IDX = {                               \
      .family = "MockVideo",                                                   \
      .vocab_size = mock_vocab_size,                                           \
      .generate_video = mock_video_##IDX,                                      \
      .pass_new = mock_pass_new,                                               \
      .pass_begin = mock_pass_begin,                                           \
      .pass_end = mock_pass_end,                                               \
      .set_other_passes = mock_set_other_##IDX,                                \
      .set_turn_budget = mock_set_budget_##IDX,                                \
  };                                                                           \
  static const ModelOps mock_decide_ops_##IDX = {                              \
      .family = "MockDecide",                                                  \
      .vocab_size = mock_vocab_size,                                           \
      .decide = mock_decide_##IDX,                                             \
      .pass_new = mock_pass_new,                                               \
      .pass_begin = mock_pass_begin,                                           \
      .pass_end = mock_pass_end,                                               \
      .set_other_passes = mock_set_other_##IDX,                                \
      .set_turn_budget = mock_set_budget_##IDX,                                \
  };

DEFINE_MOCK_OPS(0)
DEFINE_MOCK_OPS(1)
DEFINE_MOCK_OPS(2)
DEFINE_MOCK_OPS(3)

// Returns the mock ModelOps table for slot idx and role.
static const ModelOps *mock_ops_for(int idx, MockRole role) {
  static const ModelOps *const llm[MOCK_MAX] = {
      &mock_llm_ops_0, &mock_llm_ops_1, &mock_llm_ops_2, &mock_llm_ops_3};
  static const ModelOps *const video[MOCK_MAX] = {
      &mock_video_ops_0, &mock_video_ops_1, &mock_video_ops_2,
      &mock_video_ops_3};
  static const ModelOps *const decide[MOCK_MAX] = {
      &mock_decide_ops_0, &mock_decide_ops_1, &mock_decide_ops_2,
      &mock_decide_ops_3};
  if (role == MOCK_ROLE_VIDEO) return video[idx];
  if (role == MOCK_ROLE_DECIDE) return decide[idx];
  return llm[idx];
}

// Static backing storage for registered mock models.
static eremite_model mock_models[MOCK_MAX];

// Resets the virtual clock, virtual GPU and engine model registry for one
// scenario with count mock models and their roles.
static void reset_world(int count, const MockRole *roles) {
  fake_clock = 1.0;
  fake_committed = 0;
  memset(fake_done_at, 0, sizeof fake_done_at);
  memset(fake_polls, 0, sizeof fake_polls);
  saw_wait_read_state = 0;
  calls_posted = 0;
  turn_owner = NULL;
  turn_started_at = 0.0;
  proc = (eremite_process){0};
  atomic_store(&woken, 0);
  if (!wake) wake = threads_event_new();
  gpu = (Gpu *)&fake_gpu_sentinel;
  nmodels = count;
  memset(mocks, 0, sizeof mocks);
  memset(mock_models, 0, sizeof mock_models);
  for (int i = 0; i < count; i++) {
    mock_models[i].ops = mock_ops_for(i, roles[i]);
    mock_models[i].dir = "mock";
    atomic_store(&mock_models[i].loaded, 1);
    models[i] = &mock_models[i];
  }
}

// Tolerance bounds for 50/50 two-model time sharing.
static constexpr double SHARE_TWO_LOW = 0.40, SHARE_TWO_HIGH = 0.60;

// Tolerance bounds for 33/33/33 three-model time sharing.
static constexpr double SHARE_THREE_LOW = 0.25, SHARE_THREE_HIGH = 0.42;

// Tests long video slices (100 ms) beside short LLM decode calls (20 ms).
static void test_video_heavy_vs_llm_decode(void) {
  MockRole roles[2] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM};
  reset_world(2, roles);
  mocks[0].steps = 20;
  mocks[0].step_s = 0.100;
  mocks[1].step_s = 0.020;

  eremite_stream *sv = eremite_stream_new(&mock_models[0]);
  eremite_stream *sl = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sv, sl};

  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tok = 1, pos = 0;
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);

  int decodes = 0;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) break;
    if (done == 1) {
      decodes++;
      eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
    }
  }

  double total = mocks[0].total_gpu_s + mocks[1].total_gpu_s;
  double video_share = mocks[0].total_gpu_s / total;
  double llm_share = mocks[1].total_gpu_s / total;
  printf("video-heavy vs llm-decode: video %.2f s (%.1f%%), llm %.2f s "
         "(%.1f%%, %d decodes)\n",
         mocks[0].total_gpu_s, video_share * 100.0, mocks[1].total_gpu_s,
         llm_share * 100.0, decodes);
  check(video_share >= SHARE_TWO_LOW && video_share <= SHARE_TWO_HIGH &&
            llm_share >= SHARE_TWO_LOW && llm_share <= SHARE_TWO_HIGH,
        "heavy video slices and short LLM decodes share GPU time evenly");

  eremite_run_streams(&sl, 1);
  eremite_stream_free(sv);
  eremite_stream_free(sl);
}

// Tests the inverse: short video slices (5 ms) beside heavy LLM decodes
// (50 ms).
static void test_video_light_vs_llm_heavy(void) {
  MockRole roles[2] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM};
  reset_world(2, roles);
  mocks[0].steps = 200;
  mocks[0].step_s = 0.005;
  mocks[1].step_s = 0.050;

  eremite_stream *sv = eremite_stream_new(&mock_models[0]);
  eremite_stream *sl = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sv, sl};

  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tok = 1, pos = 0;
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);

  int decodes = 0;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) break;
    if (done == 1) {
      decodes++;
      eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
    }
  }

  double total = mocks[0].total_gpu_s + mocks[1].total_gpu_s;
  double video_share = mocks[0].total_gpu_s / total;
  double llm_share = mocks[1].total_gpu_s / total;
  printf("video-light vs llm-heavy: video %.2f s (%.1f%%), llm %.2f s "
         "(%.1f%%, %d decodes)\n",
         mocks[0].total_gpu_s, video_share * 100.0, mocks[1].total_gpu_s,
         llm_share * 100.0, decodes);
  check(video_share >= SHARE_TWO_LOW && video_share <= SHARE_TWO_HIGH &&
            llm_share >= SHARE_TWO_LOW && llm_share <= SHARE_TWO_HIGH,
        "short video slices and heavy LLM decodes share GPU time evenly");

  eremite_run_streams(&sl, 1);
  eremite_stream_free(sv);
  eremite_stream_free(sl);
}

// Tests video generation beside LLM prompt prefill (both multi-step yielding
// calls with asymmetric layer durations).
static void test_video_vs_llm_prefill(void) {
  MockRole roles[2] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM};
  reset_world(2, roles);
  mocks[0].steps = 20;
  mocks[0].step_s = 0.120;
  mocks[1].steps = 120;
  mocks[1].step_s = 0.020;

  eremite_stream *sv = eremite_stream_new(&mock_models[0]);
  eremite_stream *sp = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sv, sp};

  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t prompt[4] = {1, 2, 3, 4};
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_prefill(sp, st, prompt, 4, 0, 0, -1, NULL);

  int first = eremite_run_streams(streams, 2);
  double v_at_first = mocks[0].total_gpu_s;
  double p_at_first = mocks[1].total_gpu_s;
  double total = v_at_first + p_at_first;
  double video_share = v_at_first / total;
  double prefill_share = p_at_first / total;
  printf("video vs llm-prefill (first finished=%d): video %.2f s (%.1f%%), "
         "prefill %.2f s (%.1f%%)\n",
         first, v_at_first, video_share * 100.0, p_at_first,
         prefill_share * 100.0);
  check(video_share >= SHARE_TWO_LOW && video_share <= SHARE_TWO_HIGH &&
            prefill_share >= SHARE_TWO_LOW && prefill_share <= SHARE_TWO_HIGH,
        "video and LLM prefill interleave evenly instead of starving each "
        "other");

  eremite_run_streams(streams, 2);
  eremite_stream_free(sv);
  eremite_stream_free(sp);
}

// Tests dynamic solo -> contended -> solo transitions in both directions.
static void test_dynamic_solo_contended_solo(void) {
  MockRole roles[2] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM};
  // Direction 1: video runs solo, LLM joins for 10 decodes, then video finishes
  // solo.
  reset_world(2, roles);
  mocks[0].steps = 30;
  mocks[0].step_s = 0.050;
  mocks[1].step_s = 0.025;

  eremite_stream *sv = eremite_stream_new(&mock_models[0]);
  eremite_stream *sl = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sv, sl};
  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tok = 1, pos = 0;

  eremite_stream_generate_video(sv, &vreq);
  eremite_wake();
  eremite_run_streams(&sv, 1);

  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
  int llm_done = 0;
  while (llm_done < 10) {
    int done = eremite_run_streams(streams, 2);
    if (done == 1) {
      llm_done++;
      if (llm_done < 10) eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
    } else if (done == 0) {
      break;
    }
  }
  if (sv->call != STREAM_NONE) eremite_run_streams(&sv, 1);

  check(mocks[0].saw_solo && mocks[0].saw_contended &&
            mocks[0].saw_solo_after_contended &&
            mocks[0].max_turn_budget > 0.0 && mocks[0].turn_budget == 0.0,
        "video transitions dynamically from solo to contended and back to "
        "solo with measured turn budgets");
  eremite_stream_free(sv);
  eremite_stream_free(sl);

  // Direction 2: LLM runs solo, short video joins and finishes, LLM continues
  // solo.
  reset_world(2, roles);
  mocks[0].steps = 6;
  mocks[0].step_s = 0.040;
  mocks[1].step_s = 0.020;

  sv = eremite_stream_new(&mock_models[0]);
  sl = eremite_stream_new(&mock_models[1]);
  streams[0] = sv;
  streams[1] = sl;

  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
  eremite_run_streams(&sl, 1);
  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) break;
    if (done == 1) eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
  }
  eremite_run_streams(&sl, 1);
  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
  eremite_run_streams(&sl, 1);

  check(mocks[1].saw_solo && mocks[1].saw_contended &&
            mocks[1].saw_solo_after_contended,
        "LLM transitions dynamically from solo to contended and back to solo");
  eremite_stream_free(sv);
  eremite_stream_free(sl);
}

// Tests three concurrently active models with asymmetric step durations
// sharing GPU time evenly.
static void test_three_models_balanced(void) {
  MockRole roles[3] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM, MOCK_ROLE_LLM};
  reset_world(3, roles);
  mocks[0].steps = 20;
  mocks[0].step_s = 0.100;
  mocks[1].steps = 70;
  mocks[1].step_s = 0.030;
  mocks[2].step_s = 0.015;

  eremite_stream *s0 = eremite_stream_new(&mock_models[0]);
  eremite_stream *s1 = eremite_stream_new(&mock_models[1]);
  eremite_stream *s2 = eremite_stream_new(&mock_models[2]);
  eremite_stream *streams[3] = {s0, s1, s2};

  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t prompt[4] = {1, 2, 3, 4}, tok = 1, pos = 0;
  eremite_stream_generate_video(s0, &vreq);
  eremite_stream_prefill(s1, st, prompt, 4, 0, 0, -1, NULL);
  eremite_stream_decode_rows(s2, &st, &tok, &pos, 1);

  for (;;) {
    int done = eremite_run_streams(streams, 3);
    if (done == 2) {
      eremite_stream_decode_rows(s2, &st, &tok, &pos, 1);
      continue;
    }
    break;
  }

  double total =
      mocks[0].total_gpu_s + mocks[1].total_gpu_s + mocks[2].total_gpu_s;
  double r0 = mocks[0].total_gpu_s / total;
  double r1 = mocks[1].total_gpu_s / total;
  double r2 = mocks[2].total_gpu_s / total;
  printf("three models: m0 %.1f%%, m1 %.1f%%, m2 %.1f%%\n", r0 * 100.0,
         r1 * 100.0, r2 * 100.0);
  check(r0 >= SHARE_THREE_LOW && r0 <= SHARE_THREE_HIGH &&
            r1 >= SHARE_THREE_LOW && r1 <= SHARE_THREE_HIGH &&
            r2 >= SHARE_THREE_LOW && r2 <= SHARE_THREE_HIGH,
        "three concurrent models each receive a balanced share of GPU time");

  while (s0->call != STREAM_NONE || s1->call != STREAM_NONE ||
         s2->call != STREAM_NONE)
    eremite_run_streams(streams, 3);
  eremite_stream_free(s0);
  eremite_stream_free(s1);
  eremite_stream_free(s2);
}

// Tests that decision turns preserve short-decision latency, long-decision
// interleaving and read-wait sleeping, and that intra-model decode priority
// stays intact.
static void test_decision_and_intra_model_rules(void) {
  MockRole roles[2] = {MOCK_ROLE_LLM, MOCK_ROLE_DECIDE};
  reset_world(2, roles);
  mocks[0].step_s = 0.040;
  mocks[0].with_read_wait = 1;
  mocks[1].steps = 4;
  mocks[1].step_s = 0.005;
  mocks[1].deferred_gpu = 1;

  eremite_stream *sd = eremite_stream_new(&mock_models[0]);
  eremite_stream *sc = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sd, sc};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tok = 1, pos = 0, qend = 4, rows[1] = {3}, ids[1] = {1};
  int32_t seq[4] = {1, 2, 3, 4};

  eremite_stream_decode_rows(sd, &st, &tok, &pos, 1);
  eremite_stream_decide(sc, seq, 4, 2, &qend, 1, rows, ids, 1, NULL);
  int during_short = 0;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) {
      during_short++;
      eremite_stream_decode_rows(sd, &st, &tok, &pos, 1);
      continue;
    }
    break;
  }
  check(during_short <= 2,
        "a short decision beside decoding lets at most two decodes finish");

  mocks[1].steps = 40;
  saw_wait_read_state = 0;
  eremite_stream_decide(sc, seq, 4, 2, &qend, 1, rows, ids, 1, NULL);
  int during_long = 0;
  for (;;) {
    int done = eremite_run_streams(streams, 2);
    if (done == 0) {
      during_long++;
      eremite_stream_decode_rows(sd, &st, &tok, &pos, 1);
      continue;
    }
    break;
  }
  check(during_long >= 2 && saw_wait_read_state,
        "a long decision interleaves with decoding and sleeps on read waits");

  eremite_run_streams(&sd, 1);
  eremite_stream_free(sd);
  eremite_stream_free(sc);

  MockRole single[1] = {MOCK_ROLE_LLM};
  reset_world(1, single);
  mocks[0].steps = 6;
  mocks[0].step_s = 0.020;

  eremite_stream *sdec = eremite_stream_new(&mock_models[0]);
  eremite_stream *spre = eremite_stream_new(&mock_models[0]);
  eremite_stream *same[2] = {sdec, spre};
  eremite_stream_prefill(spre, st, seq, 4, 0, 0, -1, NULL);
  eremite_stream_decode_rows(sdec, &st, &tok, &pos, 1);
  int same_decodes = 0;
  for (;;) {
    int done = eremite_run_streams(same, 2);
    if (done == 0) {
      same_decodes++;
      eremite_stream_decode_rows(sdec, &st, &tok, &pos, 1);
      continue;
    }
    break;
  }
  check(same_decodes >= 5,
        "same-model prefill still yields to its own model's decoding calls");
  eremite_run_streams(&sdec, 1);
  eremite_stream_free(sdec);
  eremite_stream_free(spre);
}

// Tests that an off-GPU background wait lets another model run at full solo
// power without being blocked.
static void test_background_wait_frees_gpu(void) {
  MockRole roles[2] = {MOCK_ROLE_VIDEO, MOCK_ROLE_LLM};
  reset_world(2, roles);
  mocks[0].steps = 1;
  mocks[0].step_s = 0.010;
  mocks[0].with_background_wait = 1;
  mocks[0].background_done = 0;
  mocks[1].step_s = 0.020;

  eremite_stream *sv = eremite_stream_new(&mock_models[0]);
  eremite_stream *sl = eremite_stream_new(&mock_models[1]);
  eremite_stream *streams[2] = {sv, sl};
  eremite_video_request vreq = {.prompt = "test"};
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tok = 1, pos = 0;

  eremite_stream_generate_video(sv, &vreq);
  eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);

  int decodes = 0;
  for (int i = 0; i < 8; i++) {
    int done = eremite_run_streams(streams, 2);
    if (done == 1) {
      decodes++;
      if (i < 7) eremite_stream_decode_rows(sl, &st, &tok, &pos, 1);
    }
  }
  check(decodes == 8 && mocks[1].saw_solo,
        "LLM runs freely at solo power while video waits on background MP4 "
        "encoding");

  mocks[0].background_done = 1;
  if (mocks[0].background_event)
    threads_event_signal(mocks[0].background_event);
  int finished = eremite_run_streams(&sv, 1);
  check(finished == 0,
        "video finishes once its off-GPU background wait completes");

  eremite_stream_free(sv);
  eremite_stream_free(sl);
}

// Tests that optional ModelOps callbacks (set_history and replay_start) are
// safely tolerated when a snapshot-only model family leaves them null.
static void test_optional_history_and_replay_ops(void) {
  MockRole roles[1] = {MOCK_ROLE_LLM};
  reset_world(1, roles);
  eremite_state *st = (eremite_state *)&fake_gpu_sentinel;
  int32_t tokens[3] = {10, 20, 30};
  eremite_set_history(&mock_models[0], st, tokens, 3);
  int start = eremite_replay_start(&mock_models[0], 135);
  check(start == 135,
        "optional set_history and replay_start tolerate null ops callbacks");
}

int main(void) {
  test_video_heavy_vs_llm_decode();
  test_video_light_vs_llm_heavy();
  test_video_vs_llm_prefill();
  test_dynamic_solo_contended_solo();
  test_three_models_balanced();
  test_decision_and_intra_model_rules();
  test_background_wait_frees_gpu();
  test_optional_history_and_replay_ops();
  if (wake) threads_event_free(wake), wake = NULL;
  return failures != 0;
}
