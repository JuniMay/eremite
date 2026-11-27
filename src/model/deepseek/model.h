#ifndef EREMITE_MODEL_DEEPSEEK_MODEL_H
#define EREMITE_MODEL_DEEPSEEK_MODEL_H
#include "engine/expert_store.h"
#include "engine/safetensors.h"
#include "model/model.h"

// DeepSeek V4.1 Flash shapes, execution and routed-expert cache.

// The image token, <｜deepseek_image｜>.
static constexpr int IMAGE_TOKEN = 129264;
// A patch's 3 x 14 x 14 values, zero-padded to a multiple of 64.
static constexpr int VIT_K = 640;

// ---- model shape (config.json of DeepSeek-V4.1-Flash) ----
static constexpr int DIM = 5120, LAYERS = 40, VOCAB = 129280, HC = 4, MIX = 24;
// Residual streams store the reference's BF16 values.
static constexpr int STREAM_BYTES = 2;
static constexpr int HEADS = 64, HEAD_DIM = 512, ROPE_DIM = 64, QRANK = 1280,
                     OGROUPS = 8, ORANK = 1024, WINDOW = 128;
static constexpr int EXPERTS = 384, TOPK = 6, INTER = 2304;
static constexpr int IDX_HEADS = 32, IDX_DIM = 128, IDX_TOPK = 512;
static constexpr int ENGRAM_ROWS = 24, ENGRAM_DIM = 256;
// The DSpark draft (inference/config.json: n_mtp_layers, dspark_*): three
// stages after the backbone, each with 128 routed experts of which a token
// uses 3, drafting BLOCK tokens from the attention inputs of the last three
// layers (37 to 39); the drafted rows after the first are the NOISE token.
static constexpr int DRAFTS = 3, DRAFT_EXPERTS = 128, DRAFT_TOPK = 3, BLOCK = 5,
                     NOISE = 128799;
// The target's verification capacity, independent of DSpark.
static constexpr int VERIFY_DRAFTS = 15;
// The positions a state can continue before its end or a snapshot's
// position, for the KV prefix cache's resumes: each layer's window ring
// keeps this many rows beyond its window. The rewind margin is at least
// this, and at least the drafts or --lookup proposals that a verify pass
// writes after the token (deepseek_rewind_margin).
static constexpr int REWIND = 8;
// Rows of different sequences that one decode pass takes
// (deepseek_decode_rows, --batch).
static constexpr int BATCH_ROWS = 6;
// Fixed power-of-two token span per shared copy-on-write KV page (128 ratio-2
// rows and 256 ratio-1 rows per page across all 4 KV sources).
static constexpr int DS_KV_PAGE_TOKENS = 256;
// The expert cache's layers: the backbone's, then the stages'.
static constexpr int CACHE_LAYERS = LAYERS + DRAFTS;

static constexpr float NORM_EPS = 1e-20f, HC_EPS = 1e-6f;

// One expert as stored: w1, w2, w3 codes back to back, then their scales.
static constexpr uint32_t EXPERT_W = 3u * 5898240u, EXPERT_S = 3u * 368640u;

// An expert's bytes in the files: an MXFP4 expert's codes (w1, w2, w3 back
// to back) and scales, or with --experts the GGUF rows of w1, w3 and w2.
extern FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
extern int
    expert_spans[CACHE_LAYERS];  // spans per expert of the layer, 0: not set

// Every layer without spans yet (--experts sets the backbone's) gets the
// checkpoint's MXFP4 codes and scales.
static inline void deepseek_mxfp4_spans(TensorIndex *index) {
  for (int l = 0; l < CACHE_LAYERS; l++)
    for (int e = 0;
         !expert_spans[l] && e < (l < LAYERS ? EXPERTS : DRAFT_EXPERTS); e++) {
      Tensor w =
          l < LAYERS
              ? tensor_index_get(index, "layers.%d.ffn.experts.%d.w1.weight", l,
                                 e)
              : tensor_index_get(index, "mtp.%d.ffn.experts.%d.w1.weight",
                                 l - LAYERS, e);
      Tensor s = l < LAYERS
                     ? tensor_index_get(
                           index, "layers.%d.ffn.experts.%d.w1.scale", l, e)
                     : tensor_index_get(index, "mtp.%d.ffn.experts.%d.w1.scale",
                                        l - LAYERS, e);
      expert_span[l][e][0] = (FileSpan){w.fd, w.off, EXPERT_W};
      expert_span[l][e][1] = (FileSpan){s.fd, s.off, EXPERT_S};
      if (e == (l < LAYERS ? EXPERTS : DRAFT_EXPERTS) - 1) expert_spans[l] = 2;
    }
}

// Returns DeepSeek's expert store: the backbone's routed experts, then the
// DSpark stages'.
static inline ExpertStore *deepseek_expert_store_new(Gpu *gpu) {
  return expert_store_new(gpu, &(ExpertLayout){.layers = CACHE_LAYERS,
                                               .scored_layers = LAYERS,
                                               .experts = EXPERTS,
                                               .chosen = TOPK,
                                               .stage_experts = DRAFT_EXPERTS,
                                               .parts = 3,
                                               .span = &expert_span[0][0][0],
                                               .spans = expert_spans});
}

// A routed expert as the expert kernels read it: the GPU addresses of its
// three matrices, or 0 for no expert, and its routing weight.
typedef struct {
  uint64_t p[3];
  float weight, pad;
} ExpertRef;

static_assert(sizeof(ExpertRef) == 32, "Metal expert record layout");

// Loads the model of dir on gpu. Its workspace is counted in work, and its
// state in a state slot of work's account.
int deepseek_load(Gpu *gpu, MemorySlot *work, const char *dir,
                  const Options *o);
// Runs the n tokens from position pos, and returns the last one's logits.
const float *deepseek_forward(const int *tokens, int n, int pos);

// What the last prefill ran: its tokens and chunks, the tokens of the decoder
// layers, the CED replay's tokens, and whether CED and pruning applied.
typedef struct {
  int tokens, chunks, decoder_tokens, ced_tokens, ced_applied, prune_applied;
  // The token rows that layers 21 to 39 ran, summed over those layers.
  int decoder_row_layers;
  // The position at which the prefill captured a snapshot, or -1 for none,
  // and whether that snapshot holds the encoder layers only.
  int prefix_tokens, prefix_encoder_only;
} DeepSeekForwardInfo;

// Returns what the last prefill ran.
DeepSeekForwardInfo deepseek_last_forward_info(void);
// Writes the first n of the DSpark draft's BLOCK drafts for `token` at
// position pos: the tokens for pos + 1 to pos + n. Every position before pos
// must have run, and Options.draft must be set.
void deepseek_propose_drafts(int token, int pos, int *drafts, int n);
// Speculative decoding. deepseek_verify runs the token at pos and its drafts,
// m rows, and returns each row's logits. deepseek_accept keeps the first n of
// the rows.
const float *deepseek_verify(const int *tokens, int m, int pos);
void deepseek_accept(const int *tokens, int n, int pos);
// Times the kernels on the model's own matrices, for the command line's
// --bench.
void deepseek_run_bench(void);
// Zeroes the state in use: its rings, compressed rows, index keys, compressor
// rows and token history.
void deepseek_zero_state(void);
// One sequence's attention state: the layers' window rings, compressed rows,
// index keys and compressor rows, the draft's windows, and the Engram history.
// The layers work on one state at a time.
typedef struct State State;
// Returns a new, zeroed state, or null when its memory does not fit. The first
// one takes over the state that deepseek_load made.
State *deepseek_state_new(void);
// Frees a state that is not in use.
void deepseek_state_free(State *s);
// Returns the bytes of s, or of a new state for a null s, when it holds that
// many positions.
size_t deepseek_state_bytes(const State *s, int positions);
// Returns the extra bytes allocated when forking `positions` tokens from `src`
// into a new state that grows to `target_positions`.
size_t deepseek_state_copy_bytes(const State *src, int positions,
                                 int target_positions);
// Makes s the state that the layers work on.
void deepseek_set_state_in_use(State *s);
// deepseek_pass_new returns a pass of this model that runs body(argument).
// Inside the pass, deepseek_pass_begin starts its work on buffers of its own,
// and deepseek_pass_end ends it.
struct Pass;
struct Pass *deepseek_pass_new(void (*body)(void *), void *argument);
void deepseek_pass_begin(void);
void deepseek_pass_end(void);
// Saves the rows of the state in use that later positions overwrite, in the
// state, and restores them: a rewind point, for the command line's checks. A
// restored state continues at the saved position, or up to
// deepseek_rewind_margin() positions before it.
void deepseek_save_rewind_point(void);
void deepseek_restore_rewind_point(void);
// For the server's KV prefix cache. A snapshot holds a state's rolling part
// at one position: the window rings, compressor rows and draft rings that
// later positions overwrite, and the Engram history.
typedef struct Snapshot Snapshot;
// Returns a snapshot of s's rolling rows now, as they are at pos, with these
// logits if there are any.
Snapshot *deepseek_snapshot_take(State *s, int pos, const float *logits);
void deepseek_snapshot_restore(State *s, const Snapshot *x);
void deepseek_snapshot_free(Snapshot *x);
size_t deepseek_snapshot_bytes(int with_logits);
int deepseek_snapshot_pos(const Snapshot *x);
// Returns the last layer that the snapshot holds: LAYERS - 1, or 20 with CED.
int deepseek_snapshot_extent(const Snapshot *x);
// Returns the snapshot's logits, or null if it has none.
const float *deepseek_snapshot_logits(const Snapshot *x);
// Copies the global rows of src before positions, its compressed rows and
// index keys, into dst. dst must not be in use, and its rewind point is
// dropped. src may be in use by a running pass, since no later step changes
// those rows.
void deepseek_state_copy(State *dst, const State *src, int positions);
// Prefills positions from .. n - 1 of a prompt (tokens: all n of them) in the
// state in use, and returns the last one's logits. Without replay, the state
// holds every position before from (its own, or a copy and a restored
// snapshot at from). With replay it holds only the global rows before from
// and the prefill rebuilds the rolling rows at from. With --ced off the
// replay is exact: the prefill starts earlier, each layer where its window
// needs. With --ced it is bounded and approximate (as DeepInfer's encoder
// SWA bounded replay): every layer computes again the ring's positions before
// from, with windows cut off at the first of them, and the copied global
// rows of groups completed before from stay. With capture >= 0, *captured
// gets the rolling rows at that position as a snapshot, taken during the
// prefill: after where it starts (from, or a replay's first row) and before
// n; a replay's branch point is from. Under CED compaction, unless the
// decoder layers start a ring or more before it, the capture holds layers
// 0-20 only (extent 20). A non-null wait is waited for before each batch's
// routed experts, while the prefill holds none.
const float *deepseek_prefill(const int *tokens, int n, int from, int replay,
                              int capture, Snapshot **captured,
                              const PassWait *wait);
// Returns the positions that a replay at k computes again before k, per layer
// on average, rounded up. With --ced off it is at most k, about 2,613 for long
// prefixes. With --ced it is the ring: WINDOW plus the rewind margin, 136
// positions, and at most k.
int deepseek_replay_rows(int k);
// Returns the rewind margin: the positions beyond its window that each layer's
// ring keeps, and so how far before its end a state can continue. It is
// REWIND, or more with --draft or --lookup, since a verify pass writes their
// rows after the token.
int deepseek_rewind_margin(void);
// Returns the first position that a replay at k computes again, layer 0's.
// The prompt's images from there on must be encoded.
int deepseek_replay_start(int k);
// Writes the experts that row `row` of the latest decode step chose, TOPK for
// each layer in turn, and where each one was: 0 in memory, 1 read ahead, 2
// late, 3 substituted, 4 from the SSD. Returns 1, or 0 if the row did not run
// in every layer of that step.
int deepseek_read_routes(int row, uint16_t *id, uint8_t *from);
// Runs one token for each of m sequences at once, m at most BATCH_ROWS, without
// drafts. Row r is tokens[r] at position pos[r] of state st[r], once every
// position before it has run. Each row is computed as deepseek_forward
// computes a token alone. Returns the m rows' logits.
const float *deepseek_decode_rows(State **st, const int *tokens, const int *pos,
                                  int m);
// Continues the state in use at pos, given the tokens before it. pos is at most
// deepseek_rewind_margin() positions before the last one that ran, or 0 after a
// reset.
void deepseek_continue_at(const int *tokens, int pos);
// Runs the vision encoder (inference/vision.py's ViT and Aligner), and writes
// the image's feature rows to im->rows. Its weights load at the first use.
int deepseek_encode_image(Media *im);
// Sets the images of the prompts that deepseek_forward runs next. Each image's
// span, whose start is set, holds negative tokens (media_write_keys). A span's
// positions may fall in any of the prompt's batches.
void deepseek_set_prefill_images(const Media *im, int n);

// The statistics of decoded rows and of prompt rows since the start.
extern ExpertStoreCounts decode_stats, prefill_stats;
extern uint64_t decode_rows, prefill_rows;
extern double decode_gpu_s, decode_s;
// A prompt pass's layer, 0 between prompts, and its rows.
extern int prefill_layer, prefill_n;
// Returns the bytes of the dense weights.
uint64_t deepseek_dense_bytes(void);
// Returns the bytes of the request and layer buffers that resize, without the
// fixed scratch buffers, the weights and the states.
uint64_t deepseek_workspace_bytes(void);

#endif
