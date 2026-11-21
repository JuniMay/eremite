# Full-reference mode prompts

Use this format when calling a full-reference video model with reference images,
reference videos or reference audio (character or scene consistency, video
editing, video continuation, camera transfer, audio reuse or voice/music
reference).

Write all six sections in English. Keep the original language only for spoken
words and lyrics inside `<d>` and visible text inside double quotes `"..."`.
Shots, cuts, camera motion, speaker IDs, voiceover, `<scenetrans>`, `<cutoff>`
and on-screen text follow [prompting.md](prompting.md).

## How uploaded files map to labels

The server labels uploaded reference media in upload order:

- `--ref-image` files (up to 9; `--input-reference` prepends to them) become
  `<Picture 1>`, `<Picture 2>`, and so on.
- `--ref-video` files (up to 3, each at least 5 frames at 24 fps) become
  `<Video 1>`, `<Video 2>`, and so on.
- `<Audio 1>`, `<Audio 2>`, and so on are numbered independently of `<Video N>`:
  each reference video that has audio (either an explicit `--ref-video-audio`
  override or an audio track inside the video container) receives an `<Audio N>`
  label in upload order first, followed by standalone `--ref-audio` files (up to
  3) in upload order.

Total reference images, videos and standalone audio files in one request may not
exceed 12.

## Six sections in strict order

1. `subject_definitions:`
2. `summary:`
3. `retention_analysis:`
4. `detailed_description:`
5. `overall_soundscape:`
6. `non_diegetic_music:`

Once you assign a label in `subject_definitions`, keep its exact meaning in
every section that follows.

### 1. `subject_definitions`

Define each tracked reference unit on its own line, naming its source asset when
needed:

- `<Subject N>`: reusable visible content — a person, animal, object, scene,
  outfit, prop, visual effect, style, action, expression or pose. One subject
  can combine several sources (for example appearance from `<Picture 1>` and
  motion from `<Video 1>`), and one source can provide several subjects.
- `<Picture N>`: add a standalone line only when the image serves as a concrete
  target frame (first frame, keyframe, last frame, edited keyframe) or a
  storyboard/shot-planning anchor. When an image only supplies a subject or
  style, cite it inside `<Subject N>` instead.
- `<Video N>`: whole-video source for editing, continuation or camera, cut and
  pacing structure. Reusable characters or objects taken from a video still get
  `<Subject N>`.
- `<Audio N>`: standalone audio or an enabled soundtrack from a reference video.
  When an audio label binds to a target speaker, reuse that speaker's global
  `(Sx)` from `detailed_description` (`<Subject N> (Sx)` or a voice description
  with `(Sx)`). Never invent a separate speaker numbering here.

### 2. `summary`

Write one short English paragraph opening with one or more task types in square
brackets, joined by ` + ` without duplicates:

| Task type | Use when |
| --- | --- |
| `keyframe completion` | An image anchors a concrete first, middle or last frame |
| `reference generation` | An image, video or audio guides character, scene, style, motion, camera or storyboard |
| `video editing` | An existing source video is modified directly |
| `video continuation` | New footage extends or transitions from an existing source video |
| `audio reuse` | Source audio is copied directly in full or in part |
| `audio reference` | Only timbre, music style, rhythm, beat, wording or texture is referenced |

Rules for `summary`:

- A reference video used only for camera motion, cuts or pacing belongs to
  `reference generation`, not `video editing` or `video continuation`.
- When editing a video whose original audio stays audible, add `audio reuse`.
  When continuing a video's audio feel without copying the signal, add
  `audio reference`.
- For `video editing`, begin the first sentence after the bracketed tag with:
  `The target video is an edited version of <Video 1>.`
- Do not introduce new reference labels in `summary`.

### 3. `retention_analysis`

Write one line per defined label using these fixed markers:

- Visual labels (`<Subject N>`, `<Picture N>`, `<Video N>`):
  - `fully_preserved`: the defined role is kept intact.
  - `partially_preserved`: used with specific traits changed or partially kept.
  - `attribute_transfer`: traits move onto a different identifiable subject.
  - `weak_reference`: only broad style, category, layout or atmosphere carries over.
- Audio labels (`<Audio N>`):
  - `fully_copy`: the whole source audio becomes the final audio track.
  - `partially_copy`: part of the timeline or selected layers are copied, or
    other sounds are added, removed or replaced.
  - `reference`: timbre, rhythm, music style, lyrics or texture is referenced
    without copying the signal.
  - `weak_reference`: only broad category or atmosphere carries over.

Name the shots where each visual label applies (for example
`<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - ...`). Judge
fidelity only within the role defined in `subject_definitions`; new actions or
backgrounds do not count as losses. Never write `(Sx)` in `retention_analysis`.

### 4. `detailed_description`

This section differs from base mode's `integrated_multimodal_description` in
four ways:

1. Establish the overall visual style and lighting in 1 or 2 English sentences
   **before** `[Shot 1]`.
2. Insert `<Subject N>`, `<Picture N>`, `<Video N>` and `<Audio N>` at their
   first clear appearance and wherever they act. Describe a subject's key
   traits, frame position and action on first appearance; reuse the label later
   without redefining it.
3. For concrete frame anchors, use natural phrasing: `the shot begins from
   <Picture 1>`, `the shot's keyframe corresponds to <Picture 2>`, `the shot
   ends on <Picture 3>`.
4. Cite `<Audio N>` in the active shot or phase and state whether it is copied
   or referenced.

Target 350 to 500 English words for generation tasks; scale freely for dense
dialogue or complex video edits.

Speaker and audio attribution rules:

- Assign `(S1)`, `(S2)`, and so on once in target-video playback order.
- When a defined subject physically speaks, write `<Subject N> (Sx)` (add
  `off-screen` when off-screen). If a speaker has no subject label, use a stable
  voice description with `(Sx)`.
- When lyrics or spoken cues exist only inside a directly reused music track or
  soundtrack with no independent character or narrator speaking them, cite
  `<Audio N>` directly and do not invent an `(Sx)`.
- When reusing dialogue or lyrics verbatim, keep the exact words and original
  language inside `<d>`, write `[unclear]` for unintelligible words, use plain
  punctuation (`,`, `.`, `?`, `!`) and end complete sentences with `.`, `?` or
  `!` before `</d>`. When referencing only voice timbre or rhythm, do not carry
  the source clip's words into the target video.

### 5. `overall_soundscape` and 6. `non_diegetic_music`

Follow the base-mode rules in [prompting.md](prompting.md). When reference audio
supplies ambience or sound effects, state that in `overall_soundscape`. When
reference audio supplies the audience-only score, state that in
`non_diegetic_music`. Never repeat dialogue or lyrics in these two sections.

## Example

```text
subject_definitions:
<Subject 1> is the bright ceramic studio in <Picture 1>, with white plaster walls, wooden shelves of clay bowls and a sunlit worktable.
<Subject 2> is the potter in <Picture 2>, a woman in her thirties with short dark hair and a faded denim apron.
<Audio 1> is the voice-timbre reference for <Subject 2> (S1), containing clear spoken English.

summary:
[reference generation + audio reference] The target video shows <Subject 2> centering wet clay on a wheel inside <Subject 1>. <Audio 1> guides the vocal timbre of <Subject 2> across two shots.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the plaster walls, wooden shelves of clay bowls and sunlit worktable are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the potter's face, short dark hair and faded denim apron are retained.
<Audio 1>: reference - its vocal timbre and steady cadence guide <Subject 2>'s spoken line without copying the original recording.

detailed_description:
The target video uses a natural documentary style with soft daylight and crisp hand-held framing.
[Shot 1] A medium shot establishes <Subject 1>, the ceramic studio with white plaster walls, wooden shelves of clay bowls and a sunlit worktable. <Subject 2> (S1), the woman in her thirties with short dark hair and a faded denim apron, sits at a spinning pottery wheel, pressing wet grey clay between her palms. The camera pushes in with small amplitude at slow speed as clay rises into a smooth cylinder. Looking down at her hands, <Subject 2> (S1) says in the clear, steady voice referenced from <Audio 1>: <d>[English] Keep the pressure even from the base.</d>
[Shot 2] At 00:03.000, the camera cuts to a close-up of her wet fingers pulling the rim upward while sunlight reflects off the glaze of water along the clay wall. The wheel slows gently as she lifts her hands clear of the rim.

overall_soundscape:
The steady low hum of the electric pottery wheel runs throughout the clip, paired with the soft wet slide of palms over clay and faint studio room tone.

non_diegetic_music:
N/A
```
