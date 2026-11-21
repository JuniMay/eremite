#include "model/minimax_h3/dit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Unit tests for MiniMax-H3 audio-video DiT frame alignment, flow-shift math,
// PackedLayout segment ordering, 3D MM-RoPE coordinates and AdaLN mod rows.

// Floating-point tolerance for RoPE coordinate and flow-shift comparisons.
static constexpr double TOL = 1e-4;

// Failed check counter.
static int failures;

// Records one assertion outcome and prints label on failure.
static void check(int cond, const char *label) {
  if (cond) return;
  fprintf(stderr, "FAIL: %s\n", label);
  failures++;
}

// Returns non-zero when a and b are within tol.
static int near(double a, double b, double tol) { return fabs(a - b) <= tol; }

int main(void) {
  // 1. Frame alignment (17k + 5), video latent T (5k + 2) and audio T (40 Hz).
  check(minimax_h3_video_align_frames(1) == 5 &&
            minimax_h3_video_align_frames(5) == 5 &&
            minimax_h3_video_align_frames(6) == 22 &&
            minimax_h3_video_align_frames(22) == 22 &&
            minimax_h3_video_align_frames(120) == 124 &&
            minimax_h3_video_align_frames(124) == 124 &&
            minimax_h3_video_align_frames(362) == 362,
        "video_align_frames snaps to 17k + 5 grid");
  check(minimax_h3_video_latent_t(5) == 2 &&
            minimax_h3_video_latent_t(22) == 7 &&
            minimax_h3_video_latent_t(39) == 12 &&
            minimax_h3_video_latent_t(124) == 37 &&
            minimax_h3_video_latent_t(362) == 107,
        "video_latent_t maps 17k + 5 frames to 5k + 2 latents");
  check(minimax_h3_video_audio_t(5) == 8 &&
            minimax_h3_video_audio_t(22) == 37 &&
            minimax_h3_video_audio_t(124) == 207 &&
            minimax_h3_video_audio_t(362) == 603,
        "video_audio_t maps aligned 24 fps durations to 40 Hz audio latents");

  // 2. Flow-matching sigma time shift and exact round-trip invertibility.
  check(near(minimax_h3_time_shift_sigma(0.0, 1.0, 12.0), 0.0, 1e-12) &&
            near(minimax_h3_time_shift_sigma(1.0, 1.0, 12.0), 1.0, 1e-12) &&
            near(minimax_h3_time_shift_sigma(0.5, 1.0, 12.0), 12.0 / 13.0,
                 1e-12) &&
            near(minimax_h3_time_shift_sigma(0.5, 1.0, 3.0), 3.0 / 4.0, 1e-12),
        "time_shift_sigma matches s*b / (1 + (s - 1)*b)");
  double sv = minimax_h3_time_shift_sigma(0.25, 1.0, 12.0);
  double sa = minimax_h3_time_shift_sigma(sv, 12.0, 3.0);
  double sa_direct = minimax_h3_time_shift_sigma(0.25, 1.0, 3.0);
  double sv_back = minimax_h3_time_shift_sigma(sa, 3.0, 12.0);
  check(near(sa, sa_direct, 1e-12) && near(sv_back, sv, 1e-12),
        "time_shift_sigma round-trips cleanly between video shift 12 and audio "
        "shift 3");

  // 3. T2VA PackedLayout: [text | audio | video], target 4x8 latents (2x4=8
  // patches/frame), t_lat=7 (22 frames), audio_t=10.
  MinimaxH3PackedLayout lay;
  minimax_h3_packed_layout_build(5, NULL, 7, 4, 8, 10, NULL, 0, NULL, 0, &lay);
  check(lay.seg_count == 3 && lay.segments[0].kind == MINIMAX_H3_SEG_TEXT &&
            lay.segments[1].kind == MINIMAX_H3_SEG_AUDIO &&
            lay.segments[2].kind == MINIMAX_H3_SEG_VIDEO,
        "T2VA packs segments in [text | audio | video] order");
  check(lay.num_t == 2 && lay.target_audio_start == 5 &&
            lay.target_audio_rows == 20 && lay.target_video_start == 25 &&
            lay.target_video_rows == 56 && lay.seq_len == 81,
        "T2VA counts exact rows and uses 2 AdaLN timestep slots");
  check(lay.mod_rows[0] == 1 && lay.mod_rows[4] == 1 && lay.mod_rows[5] == 5 &&
            lay.mod_rows[24] == 5 && lay.mod_rows[25] == 0 &&
            lay.mod_rows[80] == 0,
        "T2VA assigns mod_rows text=1, audio=5 (1*3+2), video=0");

  // Check audio stereo RoPE (t = cursor + 0..9, h = 0, ch0 w = w_low, ch1 w =
  // w_high) and video temporal cycle spans (1, 4, 4, 4, 4, 1, 4) * 5/3.
  float w_low = lay.pos[3 * 25 + 2];
  float w_high = lay.pos[3 * (25 + 3) + 2];
  check(near(lay.pos[3 * 5 + 0], 5.0, TOL) &&
            near(lay.pos[3 * 5 + 1], 0.0, TOL) &&
            near(lay.pos[3 * 5 + 2], w_low, TOL) &&
            near(lay.pos[3 * 14 + 0], 14.0, TOL) &&
            near(lay.pos[3 * 15 + 0], 5.0, TOL) &&
            near(lay.pos[3 * 15 + 2], w_high, TOL),
        "T2VA stereo audio RoPE pins left/right channels to w_low/w_high");
  // Frame 0: t=5, Frame 1: 5 + 5/3, Frame 2: 5 + 25/3, Frame 5: 5 + 85/3,
  // Frame 6: 5 + 90/3 = 35.
  check(near(lay.pos[3 * (25 + 0 * 8) + 0], 5.0, TOL) &&
            near(lay.pos[3 * (25 + 1 * 8) + 0], 5.0 + 5.0 / 3.0, TOL) &&
            near(lay.pos[3 * (25 + 2 * 8) + 0], 5.0 + 25.0 / 3.0, TOL) &&
            near(lay.pos[3 * (25 + 5 * 8) + 0], 5.0 + 85.0 / 3.0, TOL) &&
            near(lay.pos[3 * (25 + 6 * 8) + 0], 35.0, TOL),
        "T2VA video RoPE advances by FRAME_RESCALE * (1, 4, 4, 4, 4) per "
        "latent token");
  minimax_h3_packed_layout_free(&lay);

  // 4. FL2VA PackedLayout: first frame (idx 0) and last frame (idx 21) on a
  // 22-frame target, with mixed vision/text prompt tags.
  int fl_tags[6] = {1, 0, 0, 0, 1, 1};
  MinimaxH3VideoKeyframe kfs[2] = {
      {.frame_index = 0, .t_lat = 1},
      {.frame_index = 21, .t_lat = 1},
  };
  minimax_h3_packed_layout_build(6, fl_tags, 7, 4, 4, 10, kfs, 2, NULL, 0,
                                 &lay);
  // Target frame_rows = 2*2 = 4. Each keyframe adds 4 rows.
  check(lay.seg_count == 5 && lay.segments[0].kind == MINIMAX_H3_SEG_TEXT &&
            lay.segments[1].kind == MINIMAX_H3_SEG_COND &&
            lay.segments[2].kind == MINIMAX_H3_SEG_COND &&
            lay.segments[3].kind == MINIMAX_H3_SEG_AUDIO &&
            lay.segments[4].kind == MINIMAX_H3_SEG_VIDEO,
        "FL2VA packs [text | cond0 | cond1 | audio | video]");
  check(lay.num_t == 3 && lay.mod_rows[0] == 1 && lay.mod_rows[1] == 0 &&
            lay.mod_rows[6] == 6 && lay.mod_rows[13] == 6,
        "FL2VA preserves mixed text vision tags (0) and sets cond mod_row=6");
  // cond0 anchored at cursor + 0 = 6.0; cond1 anchored at 6 + 21 * 5/3 = 41.0.
  check(near(lay.pos[3 * 6 + 0], 6.0, TOL) &&
            near(lay.pos[3 * 10 + 0], 6.0 + 21.0 * (5.0 / 3.0), TOL),
        "FL2VA anchors keyframe RoPE time at cursor + (5/3) * frame_index");
  minimax_h3_packed_layout_free(&lay);

  // 5. REF2VA PackedLayout: image ref + standalone audio ref + paired
  // video+audio ref (audio rows packed BEFORE video rows!).
  MinimaxH3VideoRef refs[3] = {
      {.kind = MINIMAX_H3_VIDEO_REF_IMAGE, .t_lat = 1, .h_lat = 4, .w_lat = 4},
      {.kind = MINIMAX_H3_VIDEO_REF_AUDIO, .audio_t = 6},
      {.kind = MINIMAX_H3_VIDEO_REF_VIDEO_AUDIO,
       .t_lat = 2,
       .h_lat = 4,
       .w_lat = 16,
       .audio_t = 10},
  };
  minimax_h3_packed_layout_build(4, NULL, 2, 4, 4, 8, NULL, 0, refs, 3, &lay);
  // Segment order: text(4), ref_img(4), ref_audio(12), ref_audio(20),
  // ref_img(2*2*8=32), audio(16), video(2*4=8).
  check(lay.seg_count == 7 && lay.segments[0].kind == MINIMAX_H3_SEG_TEXT &&
            lay.segments[1].kind == MINIMAX_H3_SEG_REF_IMG &&
            lay.segments[2].kind == MINIMAX_H3_SEG_REF_AUDIO &&
            lay.segments[3].kind == MINIMAX_H3_SEG_REF_AUDIO &&
            lay.segments[4].kind == MINIMAX_H3_SEG_REF_IMG &&
            lay.segments[5].kind == MINIMAX_H3_SEG_AUDIO &&
            lay.segments[6].kind == MINIMAX_H3_SEG_VIDEO,
        "REF2VA packs video_audio ref's audio rows immediately before its "
        "video rows");
  check(lay.num_t == 4, "REF2VA with visual and audio refs uses 4 AdaLN slots");
  // Check mod_rows on ref_audio (3*3 + 2 = 11) and ref_img (2*3 + 0 = 6).
  int r_aud1 = lay.segments[2].start;
  int r_aud2 = lay.segments[3].start;
  int r_vid2 = lay.segments[4].start;
  check(lay.mod_rows[r_aud1] == 11 && lay.mod_rows[r_aud2] == 11 &&
            lay.mod_rows[r_vid2] == 6,
        "REF2VA assigns mod_row=11 to ref_audio and mod_row=6 to ref_img");
  // Check cursor progression:
  // ref0 (image): cursor 4 -> span 1 -> 5
  // ref1 (audio, t=6): cursor 5 -> span 6 -> 11
  // ref2 (video_audio, t_lat=2 -> span 25/3=8.333, audio_t=10 -> max=10):
  //   cursor 11 -> span 10 -> target_cursor = 21!
  check(near(lay.pos[3 * lay.segments[1].start + 0], 4.0, TOL) &&
            near(lay.pos[3 * r_aud1 + 0], 5.0, TOL) &&
            near(lay.pos[3 * r_aud2 + 0], 11.0, TOL) &&
            near(lay.pos[3 * r_vid2 + 0], 11.0, TOL) &&
            near(lay.pos[3 * lay.target_audio_start + 0], 21.0, TOL) &&
            near(lay.pos[3 * lay.target_video_start + 0], 21.0, TOL),
        "REF2VA advances time cursor across image, audio and video_audio "
        "spans");
  // Paired video_audio ref (4x16) uses its own wider w_bounds, whereas
  // standalone audio ref uses the square target's w_bounds.
  float target_w_low = lay.pos[3 * lay.target_audio_start + 2];
  float standalone_w_low = lay.pos[3 * r_aud1 + 2];
  float paired_w_low = lay.pos[3 * r_aud2 + 2];
  float paired_vid_w_low = lay.pos[3 * r_vid2 + 2];
  check(near(standalone_w_low, target_w_low, TOL) &&
            near(paired_w_low, paired_vid_w_low, TOL) &&
            !near(paired_w_low, target_w_low, 0.1),
        "REF2VA pins standalone audio to target w_bounds and paired audio to "
        "its own video w_bounds");

  uint64_t ws_bytes = minimax_h3_dit_video_workspace_bytes(
      4, lay.total_vis, lay.target_video_rows, lay.total_audio,
      lay.target_audio_rows, NULL);
  check(ws_bytes > 0, "dit_video_workspace_bytes returns positive size");
  minimax_h3_packed_layout_free(&lay);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("minimax_h3_video_dit: all checks passed\n");
  return 0;
}
