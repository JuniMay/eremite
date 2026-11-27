#ifndef EREMITE_MODEL_INKLING2_MTP_H
#define EREMITE_MODEL_INKLING2_MTP_H

#include "engine/kv_pages.h"
#include "engine/memory.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include "model/inkling2/model.h"
#include <stddef.h>
#include <stdint.h>

// Inkling-2-Mini multi-token prediction (DSpark/DFlare + Markov RNN) draft
// head loaded from mtp.safetensors when opt.draft > 0.

// Draft head layer and tap topology (config.json mtp_config).
static constexpr int INK2_MTP_LAYERS = 3;
static constexpr int INK2_MTP_TAPS = 4;
static constexpr int INK2_MTP_TAP_0 = 5;
static constexpr int INK2_MTP_TAP_1 = 23;
static constexpr int INK2_MTP_TAP_2 = 41;
static constexpr int INK2_MTP_TAP_3 = 48;
static constexpr int INK2_MTP_BLOCK_SIZE = 8;
static constexpr int INK2_MTP_MAX_DRAFTS = INK2_MTP_BLOCK_SIZE - 1;
static constexpr int INK2_MTP_INTER = 9216;
static constexpr int INK2_MTP_WINDOW = 1024;
static constexpr int INK2_MTP_MARKOV_RANK = 256;
static constexpr int INK2_MTP_RNN_IN = 2 * INK2_MTP_MARKOV_RANK + 2 * INK2_DIM;
static constexpr int INK2_MTP_RNN_OUT = 3 * INK2_MTP_MARKOV_RANK;
static constexpr int INK2_MTP_LOCAL_COUNT = 2;
static constexpr int INK2_MTP_GLOBAL_LAYER = 1;

// Rolling ring capacities and fixed byte sizes for per-state MTP context
// caches.
static constexpr int INK2_MTP_SLIDE_RING = INK2_MTP_WINDOW + INK2_REWIND_MARGIN;
static constexpr int INK2_MTP_FRNN_RING = INK2_REWIND_MARGIN + INK2_MAX_VERIFY;
static constexpr size_t INK2_MTP_FRNN_BYTES =
    (size_t)INK2_MTP_FRNN_RING * INK2_DIM * sizeof(float);

// Per-sequence MTP draft context caches (two SWA rings, the rolling Markov
// context feature ring, and a pointer to the state's shared paged global KV).
typedef struct {
  // Parent state's paged global KV table holding MTP global K/V slices 16..17.
  const KvPageTable *global_kv;
  // Rolling context K and V rings (BF16 or packed FP8) for SWA draft layers 0
  // and 2.
  Buf slide_kv;
  // Rolling FP32 Markov context features f_rnn indexed by pos % frnn_ring.
  Buf frnn;
} Ink2MtpState;

// Per-pass reusable GPU scratch buffers for MTP context injection and drafting.
typedef struct {
  // Row capacity allocated for tap capture and context injection buffers.
  int cap_rows;
  // Target layer taps [4, cap_rows, INK2_DIM] FP32.
  Buf taps;
  // Fused context features [cap_rows, INK2_DIM] FP32 and raw K/V [cap_rows,
  // INK2_KV_DIM] FP32.
  Buf ctx, k_ctx_raw, v_ctx_raw;
  // Batched per-row device-address tables for multi-state decode injection.
  Buf kv_tgt, frnn_tgt;
  // Fixed block-8 drafting activation buffers.
  Buf cur, xn, q_raw, k_raw, v_raw, q_blk, k_blk, v_blk;
  Buf part, attn_out, o_out, gate, up, h_mlp, mlp_out, h_final;
  // Markov RNN sequential head buffers, argmax scratch and draft output tokens.
  Buf rnn_state, joint, proj, latent, corr, h_eff, argmax_parts, tok;
} Ink2MtpWork;

// Initializes MTP configuration, kernels and weight registration from
// dir/mtp.safetensors when requested_drafts > 0. Selects packed FP8 E4M3 KV
// caches when fp8_kv is nonzero. Returns the active draft limit (0 when
// disabled or mtp.safetensors is absent).
int inkling2_mtp_init(Gpu *gpu, WeightGroup *weights, const char *dir,
                      int requested_drafts, int fp8_kv);

// Returns the active maximum drafts per pass, or 0 when MTP is disabled.
int inkling2_mtp_active_drafts(void);

// Returns the byte width of one MTP KV cache row (BF16 or packed FP8).
size_t inkling2_mtp_kv_row_bytes(void);

// Returns the total byte size of one state's MTP sliding-window KV rings.
size_t inkling2_mtp_slide_kv_bytes(void);

// Returns the rolling snapshot byte size contributed by MTP (0 when disabled).
size_t inkling2_mtp_snap_rolling_bytes(void);

// Estimates GPU workspace bytes needed by MTP for `rows` forward tokens.
uint64_t inkling2_mtp_estimate_work_bytes(int rows);

// Releases all buffers in w after prior GPU commands finish.
void inkling2_mtp_free_work(Ink2MtpWork *w);

// Ensures w holds at least `rows` context injection rows and fixed drafting
// scratch buffers.
void inkling2_mtp_ensure_work(Gpu *gpu, MemorySlot *work_slot, Ink2MtpWork *w,
                              int rows);

// Allocates MTP rolling per-sequence state buffers (slide_kv and frnn) and
// binds `global_kv` for draft layer 1's global context KV slices. Returns 1 on
// success or 0 on memory refusal.
int inkling2_mtp_state_init(Gpu *gpu, MemorySlot *state_slot, Ink2MtpState *ms,
                            const KvPageTable *global_kv);

// Frees all GPU buffers owned by ms.
void inkling2_mtp_state_free(Ink2MtpState *ms);

// Returns the fixed rolling byte footprint of one Ink2MtpState (excluding
// global KV slices stored in the parent state's page table).
size_t inkling2_mtp_state_rolling_bytes(void);

// Copies target hidden state `src` ([rows, INK2_DIM] FP32) into tap slot
// `tap_idx` (0..3) of w->taps on the GPU.
void inkling2_mtp_capture_tap(GpuCommandBuffer *cmd, const Buf *src,
                              const Ink2MtpWork *w, int tap_idx, int rows);

// Injects the 4 captured target taps into each draft layer's context KV cache
// and the rolling f_rnn ring for one sequence (batched == 0) or multiple
// states (batched == 1).
void inkling2_mtp_inject(GpuCommandBuffer *cmd, Ink2MtpWork *w,
                         Ink2MtpState *const *states, const int *pos, int rows,
                         int batched, int is_prefill);

// Runs the 3-layer DSpark draft block and sequential Markov RNN head after
// `token` at `pos`, writing `n` draft token IDs into `drafts`.
void inkling2_mtp_propose(Gpu *gpu, MemorySlot *work_slot, Ink2MtpWork *w,
                          const Ink2MtpState *ms, const Buf *embed_w,
                          const Buf *embed_norm, const Buf *unembed_w,
                          int token, int pos, int *drafts, int n,
                          uint64_t *last_committed, void (*sync_fn)(void));

#endif
