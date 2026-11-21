# Transparent backgrounds

Ask the tool for a transparent background. The server then asks the model
for a flat pure green (#00FF00) background and makes that green clear. For a
clean cutout:

- Describe only the subject, whole and centered, with empty space around it.
  Do not describe a background, a floor or a scene.
- Keep green out of the subject, since green parts turn clear.
- Avoid soft shadows, glass, smoke and fine hair: their edges come out rough.
- When the call fails because the image has no flat green border, try once
  more with a simpler subject on an empty background.
- When an edit should keep an image's transparency, ask for a transparent
  background again.

For example:

```text
A game item icon: a single red apple with a short brown stem, seen from the front, whole and centered, with empty space around it. Flat cartoon style with thick dark outlines and two tones of red shading. No words, letters, numbers or logos appear anywhere in the image.
```
