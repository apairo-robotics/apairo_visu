# apairo_visu

Interactive 3D LiDAR visualisation for [apairo](../apairo) datasets.

`apairo_visu` extends apairo with interactive viewers that work natively with any `AbstractDataset`: a built-in Open3D window (default) and an optional [Rerun](https://rerun.io) backend (`apairo_visu.rerun`). Features:

- Semantic label colouring, height (viridis), and intensity display modes
- Per-class filter and distribution panel
- Trajectory overlay from pose matrices
- **Multi-pipeline comparison**: run preprocessing and/or model inference on each frame and compare N viewports side-by-side, with pipelines executing in parallel

## Installation

```bash
cd ~/dev/apairo_visu
python -m venv .venv && source .venv/bin/activate
pip install -e ../apairo   # local dependency
pip install -e .
```

## Quick start

```python
import apairo
import apairo_visu

ds  = apairo.Goose3DDataset("/data/goose", split="val")
cfg = apairo_visu.load_label_config("goose")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

## Pipeline comparison

Compare preprocessing strategies or model predictions side-by-side.  
Each `Pipeline` is a named sequence of `(pts, labels) -> (pts, labels)` callables.

```python
from apairo_visu import Pipeline

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg, pipelines=[
    Pipeline("Ground truth"),
    Pipeline("Model A", [preprocess, model_a]),
    Pipeline("Model B", [preprocess, model_b]),
])
```

Pipelines run in parallel -- each viewport updates as soon as its pipeline finishes.  
See [`examples/view_pipelines.py`](examples/view_pipelines.py) for a full runnable example.

## Pipeline structure graph

Where the viewer shows the *result* of a pipeline, `apairo_visu.graph` shows
its *structure*: how a dataset was composed (raw sources, `synchronize` /
`filter` / `window` / concat / zip wrappers, registered transforms), rendered
as a Graphviz graph. Pure introspection -- nothing is executed, no frame is
loaded, apairo core is untouched.

```python
from apairo_visu import graph

ds = raw.synchronize(reference="lidar", method="nearest", tolerance=0.05)
ds.transform("lidar", ground_height, output="ground")
train = ds.filter(valid_indices)

graph.show(train)                    # open an interactive page (pan/zoom, themes)
graph.export(train, "pipeline.html") # same page as a self-contained file
graph.export(train, "pipeline.svg")  # styled SVG, built-in renderer, no Graphviz
graph.export(train, "pipeline.mmd")  # Mermaid text -- renders natively on
                                     # GitHub / VS Code / Jupyter
graph.export(train, "pipeline.dot")  # Graphviz DOT text
graph.export(train, "pipeline.png")  # raster -- the one format needing `dot`
spec = graph.describe(train, val)    # merged GraphSpec (shared roots appear once)
```

Transforms are laid out as a *dataflow* graph, Kedro-Viz style: a per-channel
transform depends only on the latest writer of the channel it reads, so
independent transforms appear as parallel branches with named channel edges;
whole-sample callables (which may touch anything) join all open branches.
Factory-made closures are labelled with the factory's name.

Limitation: `.cache()` materialises its parent and drops the reference, so the
chain upstream of a cached dataset cannot be recovered -- call
`graph.describe()` **before** `.cache()` if you need the full graph.

## Rerun backend (optional)

Besides the Open3D window, `apairo_visu.rerun` logs any apairo dataset to the
[Rerun](https://rerun.io) viewer: one 3D view per LiDAR channel, one 2D view per
camera channel, all scrubbing together on a shared timeline. It handles semantic
labels, trajectory overlays, image channels, and async multi-rate rigs (each
sensor ticks at its own rate).

```bash
pip install -e ".[rerun]"
```

```python
import apairo
import apairo_visu.rerun as vr

ds = apairo.Rellis3DDataset("/data/RELLIS", keys=["lidar", "labels"])
vr.view(ds, label_cfgs=[vr.load_label_config("rellis")])
```

Or straight from the shell -- installing the package registers `rerun` as an
ecosystem subcommand of the core `apairo` CLI:

```bash
apairo rerun /data/ds --lidar ouster_points --camera zed_rgb
apairo rerun /data/ds --lidar velodyne_0 --labels labels --label-config semantic_kitti
apairo rerun /data/ds                       # discover the channel names
```

Full flag reference, image channels, pose/height colouring and async handling:
see the [Rerun backend guide](apairo_visu/rerun/README.md) and `examples/rerun/`.

## Studio (interactive pipeline environment)

`apairo_visu.studio` serves the pipeline graph of **live** dataset objects
with a click-to-inspect panel: node parameters, docstring, and the channel
table read from a real sample. Same architecture as its siblings
(projector / toaster / splasher): FastAPI server, vanilla zero-build front
shipped in the wheel, graph SVG server-rendered with CSS-variable tokens
(light / dark).

```bash
pip install -e ".[studio]"
```

```python
from apairo_visu import studio

studio.launch(train_ds)          # serves on 127.0.0.1:8710, opens the browser
studio.launch(train_ds, val_ds)  # merged graph, shared upstream appears once
```

Or straight from the shell on any dataset directory -- installing this
package registers `studio` as an ecosystem subcommand of the core `apairo`
CLI (entry-point group `apairo.cli_plugins`), which loads it as a
`RawDataset` and serves:

```bash
apairo studio /data/barakuda_kitti                 # every declared channel
apairo studio /data/barakuda_kitti --keys lidar labels --port 9000
apairo studio /data/barakuda_kitti --sync lidar    # all channels per frame
apairo studio . --no-browser --host 0.0.0.0        # headless / remote
```

An asynchronous dataset serves its raw event timeline (one channel per
frame); `--sync KEY` synchronizes on that reference channel (nearest match,
optional `--tolerance` seconds) so every frame carries all channels.

The viewer is exposed the same way: `apairo visu --dataset goose --root ...`
(equivalent to `python -m apairo_visu`).

Shipped so far:

- **Panel workbench** -- dockable panels (split-tree layout adapted from
  projector): drag a panel's header onto another to dock left/right/top/
  bottom; layout persists in the browser. Core panels: pipeline graph, node
  inspector, frame metrics; any channel opens as its own data panel, frame
  listing or series chart (the "open" / "list" / "plot" buttons in the
  inspector's channel table).
- **Structure inspector** -- click any node: parameters, docstring, channel
  table read from real samples. On asynchronous timelines (one channel per
  frame) the table still lists **every declared channel** -- camera next to
  lidar -- by locating one frame per channel through `frame_info`.
- **Data panels** -- a global frame slider drives every panel; each data
  panel is bound to one node+channel with a selectable presentation (auto,
  3D, BEV scatter, image, raster heatmap, histogram, values). Every stage
  is sized to the panel, so a data panel taking half the page renders at
  half-page resolution instead of a fixed thumbnail, and redraws when a
  gutter moves. Point clouds with xyz columns open in the **3D viewer** by
  default (three.js engine shared with toaster: trackball/orbit camera,
  motion LOD, WASD/QE fly keys, **shift+arrows** to step-rotate the view --
  roll and pitch, to turn a scan that came in tilted upright). Those camera
  keys act on the **clicked** panel, not the hovered one: click a panel to
  give it the keys and it keeps them, outlined in the accent colour, until
  another panel is clicked -- so the pointer is free to reach a control or
  another panel mid-gesture. The BEV stays one select away. A **view**
  popover carries the display settings, shared by every cloud panel and
  remembered across sessions -- they say how you want to *look* at clouds,
  not something about one channel: **background** -- match the page theme
  (the default, so a viridis cloud is never dark-on-dark under a light UI),
  pin it to dark or light, or pick a colour -- **point shape** round or
  square, **point size** in
  pixels or in **metres** (attenuated with distance), the **ground grid** on
  or off with its lines re-tinted to whatever background is in play, the
  **camera style** (free trackball or upright orbit), and a **frame cloud**
  button for when you have flown off into the void. The **BEV
  zooms**: scroll to zoom at the cursor, drag a box to zoom on a selection,
  shift-drag to pan, double-click to reset. **Images and rasters zoom the
  same way** (scroll at the cursor, drag to pan, double-click to reset):
  pixels are kept in an offscreen source at native resolution and blitted
  under the current view, so zooming into a corner of a camera frame shows
  real pixels, not an upscaled thumbnail. Clouds color by any of their own columns **or by a
  per-point sibling channel**: pick `color: labels` and the cloud renders
  as a labeled point cloud (categorical palette for integer labels, viridis
  for continuous channels), in both BEV and 3D. The view popover's point
  size drives both renderers; images get an R/B swap toggle (rosbag frames
  are BGR).
  Each panel also carries its **own frame input**, counted in **its
  channel's own frames** -- a lidar panel sits at `lidar 000850`, not at
  the interleaved global index that no label file is named after (and that
  shifts the moment a label is written to another channel). Type a stem
  (`000850`, or `850` without the padding) and press Enter to seek this
  channel. Panel metas read the same way:
  `lidar 000850 · frame 17206 · Full_rural_and_semi-urban`.
  A per-panel **sync** checkbox says where that seek lands. Ticked (the
  default) the panel and the global timeline drive **each other**: seeking
  here moves the topbar slider and every other synced panel onto the
  matching frame, and a global move brings this panel along -- so calling
  up `lidar 000850` puts the camera on the image next to it. Unticked, the
  panel is an island: seek and arrow it freely without disturbing anything,
  and the global timeline leaves it alone -- two moments of the same
  channel compare side by side. Tick it again and the frame it wandered to
  becomes everyone's.
  Previews work *at any node and after any transform
  step* (pipeline prefix). Raw arrays travel as `{dtype, shape,
  data(base64)}` and all rendering is client-side: viridis BEV with robust
  percentile framing and color-by column, RGB/BGR toggle for rosbag images,
  histograms + stats. Per-point channels are stride-decimated above
  `max_points` (default 150k). Panels on the same node share one request
  per frame.
- **Per-channel timeline** -- on asynchronous datasets the global slider
  walks the interleaved event timeline; the topbar channel select locks it
  to one channel's frames (lidar events only, camera events only). The
  counter then leads with the **file the frame came from**, the
  channel-relative position and the global timeline index following as
  context (`camera 000011 · 12/450 · frame 5031`), and the restriction
  combines with a sequence range. Data panel metas carry the same reading
  per frame (`frame 5031 (seq_b · camera 000011)`). Meanwhile data panels
  bound to a channel absent at the current frame **hold the nearest
  available data** and name *that* frame, annotated with its distance
  (`camera 000004 · frame 16 · held (0.083 s old)` from timestamps, frame
  distance without them) -- the camera panel keeps showing the latest image
  while you scrub lidar frames. Before a channel's first event the panel
  holds the *first* frame ahead instead (`first (0.025 s ahead)`), so
  opening a channel on an asynchronous root never lands on an empty panel
  just because its events start later than frame 0.
- **Per-sequence inspection** -- dataset nodes that carry sequence
  structure list their sequences in the inspector (id, frame count, global
  range); "view" restricts the frame slider to that sequence ("all frames"
  in the topbar clears it). Data panels show each frame's provenance
  (`frame 12 (seq_b · lidar 000850)`). Reads apairo's provenance contract
  (`frame_sequence_ids` / `frame_stems` / `frame_info`), which views
  forward through any chain -- e.g. a synchronized multi-sequence root.
- **Keyboard timeline** -- left/right step one frame, up/down ten (up goes
  forward: the arrows read as a throttle, not a list cursor). A step lands
  on a real event of **one channel** -- the next *lidar* scan, not the next
  interleaved imu message. Which channel, in order: the topbar track when
  one is set (the slider walks it, so the arrows agree with the counter),
  else the **focused panel's own channel**, so clicking a lidar view and
  pressing right steps lidar without touching the topbar at all. It moves
  the **global** frame either way, so every other panel follows: the camera
  panel holds its nearest image, the metrics and the series marker follow
  along, all in step with the lidar view. Focused text controls keep their
  own arrows.
- **Go to a file** -- a flat frame index says nothing about which file a
  frame came from, and every sequence of a root restarts its numbering at
  `000000`. The topbar **go to** box takes the on-disk stem a label is
  named after (`000850`, or `850`) and jumps to it, scoped to the selected
  channel and sequence; when the stem is not there it widens the search and
  reports where it actually lives. Without a channel track a bare number is
  read as a frame index. Server-side via
  `/api/node/{id}/locate/{stem}?channel=&sequence=`, so it works whether or
  not the channel's timeline is loaded.
- **Frame metrics** -- per-channel metrics of the selected node computed
  client-side each frame: per-column min/mean/max for point matrices,
  non-finite counts, and value distributions for label-like channels.
- **Channel listing** -- "list" on any inspector channel row opens the
  channel's **contents**, one row per frame: channel row, the on-disk file
  backing it, the global frame index, the time since the channel's first
  frame, and the size. Every channel names a file, whichever way it is
  stored: one file per frame (`000850.npy`) for a per-frame layout, or the
  single array the whole channel lives in (`imu_odometry.npy`, flagged *one
  file for every row*, sized by that row's own bytes) -- an unnamed frame
  would read as one with no home on disk. That is the view that answers
  "which frames
  have I already labelled and which are left" -- a `ground_truth` listing
  is exactly the label set. Filter by sequence, page through with
  prev/next, jump to a stem with *find*, page to wherever the timeline sits
  with *here*, and click any row to seek. Paged server-side
  (`/api/node/{id}/entries/{channel}`, capped per request because every row
  costs an `os.stat`); file and timestamp are best-effort, so a view or an
  exotic loader lists what it can rather than failing.
- **Series panels** -- "plot" on any inspector channel row opens a chart of
  that channel **across frames** (the whole dataset, or the sequence
  selected in the inspector): imu components, speeds, any per-frame scalar.
  Nothing is computed until you press **plot** -- reducing a
  million-frame channel is a decision, not a side effect of opening a
  panel -- and the component is picked from a **named list** derived from
  the channel's shape: `c0…c9` for a 1-D imu frame, `c3 (row 0, col 3)` for
  a 4x4 pose, `col 2 (z)` for a point cloud. 1-D frames index directly,
  matrix frames index flat (a 4x4 pose trajectory is components 3 and 7 in
  path mode), per-point clouds reduce a column to their mean. Changing the
  component or the range primes the button for a **re-plot** instead of
  refetching behind your back; the **rolling mean/variance window** and the
  stat pick redraw immediately, being pure client-side arithmetic on the
  values already fetched. **Path mode** plots x against y for trajectories.
  The chart doubles as a navigator: a dashed marker tracks the global
  frame, clicking seeks to it. Reductions run server-side
  (`/api/node/{id}/series/{channel}`, paged, channel loading narrowed with
  `select`).
- **Transform catalog & try** -- the catalog panel lists the public
  callables of `apairo_transform` / `apairo_preprocess` (introspected:
  signature, docstring, kwargs rendered as typed controls). Designate the
  example to apply on with the **try** button (any data panel or inspector
  channel row): the catalog targets that node+channel+sample. "Try on
  target" opens a TRY panel: before / after on the **designated sample** --
  pinned by default (editable frame input), or unpinned to follow the
  global slider -- **plus the exact code snippet to paste into the
  script**: the GUI tries, the script stays the source of truth. Only
  catalog entries execute, applied to in-memory arrays; the served datasets
  are never mutated.
- **Shareable state** -- `?frame=N` presets the frame,
  `?open=node:channel,...` pre-opens data panels,
  `?try=entry:node:channel[:k=v;...]` pre-opens a try panel.
- Stateful transforms (e.g. accumulators) preview a single frame in
  isolation, which may differ from a sequential pass.

Next: per-step bypass toggles for A/B comparison, side-by-side branch
views (see the studio plan).

## CLI

Installing the package registers three ecosystem subcommands on the core
`apairo` command (entry-point group `apairo.cli_plugins`):

```bash
# interactive studio on any dataset directory (see the Studio section)
apairo studio /data/barakuda_kitti --sync lidar

# Open3D viewer on a known dataset
apairo visu --dataset goose --root /data/goose --split val
apairo visu --dataset rellis --root /data/rellis
apairo visu --dataset semantic_kitti --root /data/kitti --split train --idx 50

# Rerun replay of any dataset directory (see the Rerun backend section)
apairo rerun /data/barakuda_kitti --lidar lidar --camera camera
```

`python -m apairo_visu` remains equivalent to `apairo visu`.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `->` / `L` | Next frame |
| `<-` / `H` | Previous frame |
| `T` | Cycle colour mode (Semantic -> Intensity -> Height) |
| `B` | Bird's-eye (top-down) view |
| `R` | Reset camera (all viewports) |
| `J` | Toggle trajectory overlay |

## Project layout

```
apairo_visu/
  config.py     ViewConfig + load_label_config
  pipeline.py   Pipeline + apairo FramePreprocessor bridge
  graph.py      pipeline structure graph (introspection -> GraphSpec + DOT/Mermaid)
  graph_render.py  built-in layout + styled SVG / interactive HTML renderer
  studio/       interactive pipeline environment (FastAPI + vanilla front)
  rerun/        optional Rerun backend: view() + `apairo rerun` CLI (own colormaps/pipeline)
  colors.py     colour maps (semantic / intensity / height)
  geometry.py   pure-numpy helpers (hover projection, trajectory) + Open3D builders
  poses.py      load_poses: pose channel -> 4x4 matrices
  viewer.py     LidarViewer (Open3D GUI)
  __main__.py   CLI
```

The light layer (`config`, `pipeline`, `graph`, `colors`, `geometry`, `poses`) imports
only numpy/PyYAML, so `import apairo_visu` works headless; Open3D is pulled in
lazily the first time `LidarViewer` is used, and `rerun-sdk` only when you import
`apairo_visu.rerun`.

## Documentation

- [Getting started](docs/getting_started.md)
- [LidarViewer API](docs/viewer.md)
- [Rerun backend](apairo_visu/rerun/README.md)
- [Label configurations](docs/label_configs.md)
- [Synchronising async datasets](docs/sync.md)
- [Examples](docs/examples.md)
