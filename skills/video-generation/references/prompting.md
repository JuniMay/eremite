# Base mode prompts

Use this format for text-to-video (T2VA), first-frame (I2VA), first-and-last-frame
(FL2VA) and last-frame (L2VA) requests. Write the prompt in English, keeping
spoken words, lyrics and on-screen text verbatim in their original language.

## Prompt layout

A base-mode prompt has an optional alignment line for keyframes, one blank line,
and three core fields in strict order separated by blank lines.

### 1. Alignment line (keyframes only)

Omit this line for text-only generation (T2VA). For keyframe tasks, put the
matching line first, followed by one blank line:

- **First frame only (`--first-frame` or `--input-reference`, I2VA)**:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

- **First and last frames (`--first-frame` and `--last-frame`, FL2VA)**:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

- **Last frame only (`--last-frame`, L2VA)**:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

Here `N` is the index of the final shot, and `S.SS` is the target clip duration
formatted to two decimal places (for example `5.00` or `8.00`).

### 2. Three core fields

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- `integrated_multimodal_description`: visuals, subjects, setting, lighting,
  actions, camera movement, shot cuts, speakers, dialogue, singing and diegetic
  sound in playback order.
- `overall_soundscape`: 1 to 4 English sentences in one paragraph summarizing
  ambient sound, physical action sounds and non-verbal human sounds (wind, rain,
  traffic, footsteps, fabric, impacts, breathing, laughter). Do not repeat
  dialogue, singing or diegetic music here. Write `N/A` only when complete
  silence is explicitly requested.
- `non_diegetic_music`: 1 to 3 English sentences describing score that only the
  audience hears. State instrumentation, tempo, rhythm and volume changes. Do
  not use abstract mood labels or explain emotional intent. Music that
  characters hear (radio, phone, live instruments, singing) belongs in
  `integrated_multimodal_description`. Write `N/A` when there is no score.

## Keyframe pacing

- **I2VA (`--first-frame`)**: `<Picture 1>` is frame 0 of `[Shot 1]`. Establish
  its style, subjects, clothing, colors, props and layout first, then move
  forward: first-frame anchor -> action onset -> continuous motion -> result.
- **FL2VA (`--first-frame` and `--last-frame`)**: `Picture 1` opens the clip and
  `Picture 2` ends it. Prefer one single shot unless cuts are requested.
  Describe the physical motion, pose changes, object manipulation and lighting
  shifts that carry the scene from `Picture 1` to `Picture 2` by the end of
  `[Shot N]`.
- **L2VA (`--last-frame`)**: `<Picture 1>` is the final frame of `[Shot N]`.
  Start from a compatible earlier state and describe how subjects, objects,
  camera and lighting converge onto `<Picture 1>` at the end.

## Shots and cuts

Start `[Shot 1]` with no timestamp, naming the visual style (such as
`Live-action, cinematic`, `2D-animated`, `3D CG`, `claymation`, `watercolor` or
`vintage film`) and initial framing. For keyframe tasks, match the style of the
reference image.

Number later shots in sequence with strictly increasing `MM:SS.mmm` cut times
inside the clip duration:

```text
[Shot 2] At 00:03.500, the camera cuts to a close-up of...
```

Use ordinary transition phrases (`the camera cuts to`, `the shot cuts to`,
`the shot transitions to`, `the shot changes to`, `the shot switches to`). Use
cross-dissolve, fade or wipe only when asked. Use a cut only when introducing a
new subject, space, viewpoint or moment; use camera motion for small shifts in
distance or angle.

## Camera motion

Write camera motion as natural English inside the shot sentence, combining up to
three dimensions. Omit amplitude when medium and speed when normal.

| Dimension | Expression | Meaning |
| --- | --- | --- |
| Motion type | `Zoom In` / `Zoom Out` | Focal length changes; camera stays still |
| Motion type | `Push In` / `Pull Out` | Camera moves forward / backward |
| Motion type | `Pan Left` / `Pan Right` | Camera pivots horizontally in place |
| Motion type | `Truck Left` / `Truck Right` | Camera translates horizontally |
| Motion type | `Tilt Up` / `Tilt Down` | Camera pivots vertically in place |
| Motion type | `Pedestal Up` / `Pedestal Down` | Camera moves vertically |
| Motion type | `Arc Shot` | Camera moves in an arc around the subject |
| Motion type | `Tracking Shot` | Camera follows a moving subject |
| Motion type | `Static Shot` | Camera position and lens stay still |
| Motion type | `Shake Slightly` / `Shake Strongly` | Slight / strong camera shake |
| Motion type | `POV` | Subject's point of view |
| Motion type | `Roll Clockwise` / `Roll Counterclockwise` | Camera rolls around the lens axis |
| Amplitude | `with small amplitude` / `with large amplitude` | Small / large compositional change |
| Speed | `at slow speed` / `at fast speed` | Slow / fast movement |

Example:

```text
The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
```

## Speakers, dialogue, singing and voiceover

- Assign each speaking or singing character a stable ID: `(S1)`, `(S2)`, and so
  on. Use a compound ID like `(S1,S2)` when numbered speakers talk or sing
  together. Silent characters get no ID. Keep the same ID across shots.
- On a speaker's first appearance, establish visible traits and voice traits
  (age, gender, pitch, timbre, pace, accent, on-screen or off-screen) outside
  `<d>`.
- Inside `<d>...</d>`, put only `[Language]` and the exact spoken words. Keep
  the requested words, language and punctuation verbatim.
- For voiceover, use the exact phrase `says in an off-screen voiceover`, and
  state right after `</d>` that the on-screen character's lips remain completely
  closed:

```text
The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
```

- When one line crosses a cut, place `<scenetrans>` at both connecting points
  and state continuity (`continues seamlessly across the cut`, `continues
  uninterrupted into the next shot`, `carries over from the previous shot` or
  `remains audible across the transition`).
- When speech is cut short by the clip ending, mark the cut point with
  `<cutoff>`.

## On-screen text

Put every visible sign, label, banner or subtitle in English double quotes,
verbatim in its original language:

```text
A red neon sign reading "OPEN" glows above the doorway.
```

## Examples

### Text-to-video (T2VA)

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening wooden shutters before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. Light footsteps cross the floor, followed by the crisp slice of a knife through crust.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes that fade out at the end.
```

### First and last frames (FL2VA, 6 seconds)

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a cyclist begins in the pose and framing established by Picture 1, holding a closed black umbrella beside a silver bicycle on a wet sidewalk. The camera pulls out with small amplitude at slow speed as she releases the handlebar, raises the umbrella above her shoulder and pushes the runner upward until the canopy opens. Drops roll off the fabric as she steps under it, rotates the handle and settles into the exact pose, spacing and composition established by Picture 2 at the end of the shot.

overall_soundscape: Steady rain hits the pavement, followed by the metallic click of the umbrella runner and the snap of the fabric opening. Water drips from the bicycle frame as distant traffic passes.

non_diegetic_music: N/A
```
