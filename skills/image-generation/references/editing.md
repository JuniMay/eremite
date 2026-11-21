# Editing images

- The model knows the images as `<Picture 1>`, `<Picture 2>` and so on, in the
  order that the tool sends them: the order of the paths that you give. The
  server takes up to 16 images.
- Give each picture a job: the scene being changed, a face to keep, a style
  to copy, or a part to put in. When a picture lends only one trait, such as
  its color, add that its other traits do not carry over. Images given only
  as a style or subject to follow make a new image that refers to them, not
  an edit.
- Point at the part to change by where it is or how it looks, and then say
  what it turns into: "the curtains in <Picture 1> turn dark green; the
  window, the rod and the light through them stay as they are."
- Write what the finished image shows, not only the operation: where a moved
  thing now stands, and what now fills the place that it left.
- The tool sends no mask, so the model may redraw any part of the image. Name
  everything that has to survive the edit, and name it again in each round.
- Change one thing per call, and name the picture that the call changes.

For example, with the kitchen photo's path first and the kettle photo's
second:

```text
<Picture 1> is a photo of a kitchen counter. <Picture 2> is a photo of a red enamel kettle. Place the kettle from <Picture 2> on the counter in <Picture 1>, to the right of the sink, at the size that a kettle has beside that sink. Keep its color, its handle and its spout. Light it like the kitchen, from the window behind the sink. Everything else in <Picture 1> stays the same: the sink, the tiles, the cups and the window.
```
