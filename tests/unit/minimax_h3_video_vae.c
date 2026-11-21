// Tests MiniMax-H3 Video VAE temporal chunking geometry and workspace sizing.
#include "check.h"
#include "model/minimax_h3/video_vae.h"

// Expected aligned frame counts (17*k + 5) and latent counts (5*k + 2).
static constexpr int ALIGN_FRAME_STEP = 17, ALIGN_FRAME_BASE = 5,
                     ALIGN_LATENT_STEP = 5, ALIGN_LATENT_BASE = 2,
                     MAX_TEST_K = 22;

// Tests temporal chunking decode and encode length formulas.
int main(void) {
  MinimaxH3VvaeTemporalPlan still = minimax_h3_video_vae_decode_plan(1);
  check(still.out_frames == 1 && still.num_chunks == 1 && still.pad_tokens == 0,
        "single latent decodes to 1 frame");
  check(minimax_h3_video_vae_encode_latent_t(1) == 1,
        "single frame encodes to 1 latent");

  int aligned_ok = 1;
  for (int k = 0; k <= MAX_TEST_K; k++) {
    int t_lat = ALIGN_LATENT_STEP * k + ALIGN_LATENT_BASE;
    int frames = ALIGN_FRAME_STEP * k + ALIGN_FRAME_BASE;
    MinimaxH3VvaeTemporalPlan plan = minimax_h3_video_vae_decode_plan(t_lat);
    if (plan.out_frames != frames) aligned_ok = 0;
    if (k > 0 && (plan.pad_tokens != 0 || plan.num_chunks != k)) aligned_ok = 0;
    if (minimax_h3_video_vae_encode_latent_t(frames) != t_lat) aligned_ok = 0;
  }
  check(aligned_ok,
        "aligned latent lengths 5*k+2 round-trip exactly to 17*k+5 frames");

  // Verify non-aligned latent lengths remain monotonic and positive.
  int mono_ok = 1, prev = 0;
  for (int z = 1; z <= 60; z++) {
    MinimaxH3VvaeTemporalPlan p = minimax_h3_video_vae_decode_plan(z);
    if (p.out_frames <= prev || p.num_chunks < 1) mono_ok = 0;
    prev = p.out_frames;
  }
  check(mono_ok, "decode frame counts stay strictly increasing across z=1..60");

  check(minimax_h3_video_vae_decode_workspace_bytes() > 0,
        "decode workspace estimate is positive");
  return failures != 0;
}
