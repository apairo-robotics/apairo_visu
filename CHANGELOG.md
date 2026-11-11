# Changelog

## 0.1.1

### Studio

- The studio refuses requests addressed to a host name it is not served under, so a web page you visit cannot read it through DNS rebinding (`--host 0.0.0.0` still answers to any name).
- A transform whose result is not numeric (an object array, a whole Sample) now shows an error in the try panel instead of garbage bytes from server memory.
- The try panel's snippet now matches what it displays when a transform only runs on the whole sample (e.g. `AccumulateFrames`): pasting it no longer fails.
- A preprocessor that returns the sample (e.g. `ApplyPose`) now shows the channel it writes in the try panel.
- The graph links steps reading a preprocessor's output to that preprocessor, not to the dataset.
- The graph of an `.inference()` view shows the chain it runs instead of a single node.
- int64 / uint64 channels use their full value range: histograms and rasters no longer collapse to two bars or two colours.
- float16 and bool channels show their values; a dtype the browser cannot decode is refused rather than drawn from raw bytes.
- Float images with values in [0, 1] display instead of being drawn black.
- Malformed `/api/try` requests get a 400 error instead of a server error.
- `apairo studio --sync` on a channel the dataset does not have prints an error instead of a traceback.

### Desktop viewer (Open3D)

- A pipeline that raises on a frame says so in its timing line and clears its viewport, instead of leaving the previous frame's cloud under the new frame number.
- Semantic mode no longer hides points whose label has no entry in `semantic_map` (a `color_map`-only config showed an empty viewport).
- A `TypeError` raised inside a pipeline step is reported instead of the step being silently re-run without `frame_idx`.

### Rerun backend

- `apairo rerun --range` with `--labels` filters the labels with their points, so every point keeps its own class.
- `max_points` keeps each point's colour from a `KeyColormap`; a colormap or a label channel whose length does not match the cloud is an error.
- The trajectory and robot marker (`pose_key` / `poses`) now appear in the 3D views.
