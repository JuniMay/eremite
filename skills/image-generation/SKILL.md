---
name: image-generation
description: "Make or edit raster images with the image tool, such as photos, illustrations, textures, sprites, product shots, posters, mockups and cutouts on a transparent background, and write each prompt as a full description of the image. Use for a new image, an edit of an existing image, or variants from reference images. Do not use when the result should be vector or code, such as an SVG icon set or a chart that code can draw."
---

# Image generation

The image tool makes a new image from a prompt, or edits the images that it
is given. Its model draws what the prompt says. No second model rewrites or
lengthens the prompt on the way, so the prompt you send is the whole brief.

## When to use the tool

Use it for raster images. When the result should be vector or code, such as
an SVG icon set, a logo system to extend, or a chart that HTML, canvas or a
plotting library draws better, make it directly instead.

A new image is the default. Treat a request as an edit only when the user
wants an existing image changed and parts of it kept; then read
[references/editing.md](references/editing.md). For a transparent background,
read [references/transparency.md](references/transparency.md).

## Workflow

1. For several different images, make one call per image, each with its own
   prompt.
2. To edit an image file on disk, first look at it with the tool that shows
   images, then pass its path to the image tool.
3. Write the prompt as the next section says, and make the image without
   asking again, unless a missing fact would make the image wrong.
4. Look at each result: the counts, the positions, left and right, the
   spelling of every word, and what an edit had to keep. Change one thing per
   retry, and keep the rest of the prompt as it was.
5. An image that the project uses belongs in the project: copy or move it
   there from the tool's folder, and update what refers to it. Do not
   overwrite an existing file unless the user asked for that; write a new
   name, such as `hero-v2.png`. Tell the user each path and the prompt that
   made it.

## What the server controls

- The server picks the size and the quality. A new image comes out at its
  default size, 2048 by 2048 pixels unless its settings give another, so
  compose for that frame. An edit comes out at the size of its first image.
  When the user needs another shape, tell them that the server's default size
  sets it, and that an edit keeps the shape of its first image.
- Quality changes nothing: the model always takes the same 6 steps.
- The server makes no animated GIFs. Do not ask the tool for one.
- The model reads the first 4096 tokens of a prompt and drops the rest. Put
  what matters first.

## Write the prompt

- Write in English, in sentences that run as one or more paragraphs. Do not
  send labeled lines, lists of keywords or tags.
- Build the description from the whole down to the parts: the kind of image
  and its use; who or what is in it and what each one does; where things are,
  how many, the camera's position and which areas stay empty; then how it
  looks, in colors, materials, light, surfaces and the lens or the medium.
  For a busy layout, divide the frame into areas first, then fill each one.
- Describe what a viewer would see, not how good it is: "late sun from the
  right, a scratched leather strap" helps, and "stunning" or "4K" does not.
- Keep every detail that the user asked for, choose plain defaults where the
  request is silent, and add no people, objects, brands or words that the
  request does not call for. Exclude things with plain sentences: there is no
  negative prompt.
- Put each word that must appear in the image in quotes, exactly as spelled
  and in its own language, and say where it sits, which line is largest and
  where each line ends. When no lettering is wanted, say so outright, or the
  model tends to invent captions.
- Give charts and diagrams their true values, with units and the period they
  cover. If the user gave no numbers, use sample values and say so.
- Spell out a grid so that rows and columns cannot be mixed up, such as "two
  rows, with three panels in each row", and point at a panel by its row and
  its place in the row.

For example:

```text
A product photo for an online shop: one white ceramic coffee mug on a pale oak table, seen from the front at table height. The mug stands slightly left of center with its handle to the right, and the right third of the frame is an empty, soft gray wall. Soft daylight comes from a window on the left and casts a short shadow to the right. The glaze is glossy, with a faint reflection of the window. No words, letters, numbers or logos appear anywhere in the image.
```
