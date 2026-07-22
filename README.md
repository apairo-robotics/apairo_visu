# apairo_visu

Interactive 3D LiDAR visualisation for [apairo](../apairo) datasets.

`apairo_visu` extends apairo with an Open3D-based viewer that works natively with any `AbstractDataset`. Features:

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
  inspector, frame metrics; any channel opens as its own data panel (the
  "open" button in the inspector's channel table).
- **Structure inspector** -- click any node: parameters, docstring, channel
  table read from real samples. On asynchronous timelines (one channel per
  frame) the table still lists **every declared channel** -- camera next to
  lidar -- by locating one frame per channel through `frame_info`.
- **Data panels** -- a global frame slider drives every panel; each data
  panel is bound to one node+channel with a selectable presentation (auto,
  3D, BEV scatter, image, raster heatmap, histogram, values). Point clouds
  with xyz columns open in the **3D viewer** by default (three.js engine
  shared with toaster: trackball/orbit camera, motion LOD, fly keys); the
  BEV stays one select away. The **BEV zooms**: scroll to zoom at the
  cursor, drag a box to zoom on a selection, shift-drag to pan,
  double-click to reset. Clouds color by any of their own columns **or by a
  per-point sibling channel**: pick `color: labels` and the cloud renders
  as a labeled point cloud (categorical palette for integer labels, viridis
  for continuous channels), in both BEV and 3D. A point-size control drives
  both renderers; images get an R/B swap toggle (rosbag frames are BGR).
  Each panel also carries its **own frame input**: type an index to diverge
  from the global slider, **lock** to keep it -- moving the global slider
  resynchronizes every panel except the locked ones, so two moments of the
  same channel compare side by side.
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
  counter then leads with the channel-relative index, the global timeline
  index following as context (`camera 12/450 · frame 5031`), and the
  restriction combines with a sequence range. Data panel metas carry the
  same reading per frame (`frame 5031 (seq_b · camera 11)`). Meanwhile data panels bound
  to a channel absent at the current frame **hold the last available
  data**, annotated with its age (`held from frame 16 (0.083 s old)` from
  timestamps, frame distance without them) -- the camera panel keeps
  showing the latest image while you scrub lidar frames.
- **Per-sequence inspection** -- dataset nodes that carry sequence
  structure list their sequences in the inspector (id, frame count, global
  range); "view" restricts the frame slider to that sequence ("all frames"
  in the topbar clears it). Data panels show each frame's provenance
  (`frame 12 (seq_b)`). Reads apairo's provenance contract
  (`frame_sequence_ids` / `frame_info`), which views forward through any
  chain -- e.g. a synchronized multi-sequence root.
- **Frame metrics** -- per-channel metrics of the selected node computed
  client-side each frame: per-column min/mean/max for point matrices,
  non-finite counts, and value distributions for label-like channels.
- **Series panels** -- "plot" on any inspector channel row charts that
  channel **across frames** (the whole dataset, or the sequence selected in
  the inspector): imu components, speeds, any per-frame scalar. 1-D frames
  index directly, matrix frames index flat (a 4x4 pose trajectory is
  components 3 and 7 in path mode), per-point clouds reduce a column to its
  mean. An optional **rolling mean/variance window** reproduces
  roughness-style signals (imu variance over N frames) without touching the
  data; **path mode** plots x against y for trajectories. The chart doubles
  as a navigator: a dashed marker tracks the global frame, clicking seeks
  to it. Reductions run server-side (`/api/node/{id}/series/{channel}`,
  paged, channel loading narrowed with `select`).
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

Installing the package registers two ecosystem subcommands on the core
`apairo` command (entry-point group `apairo.cli_plugins`):

```bash
# interactive studio on any dataset directory (see the Studio section)
apairo studio /data/barakuda_kitti --sync lidar

# Open3D viewer on a known dataset
apairo visu --dataset goose --root /data/goose --split val
apairo visu --dataset rellis --root /data/rellis
apairo visu --dataset semantic_kitti --root /data/kitti --split train --idx 50
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
  colors.py     colour maps (semantic / intensity / height)
  geometry.py   pure-numpy helpers (hover projection, trajectory) + Open3D builders
  poses.py      load_poses: pose channel -> 4x4 matrices
  viewer.py     LidarViewer (Open3D GUI)
  __main__.py   CLI
```

The light layer (`config`, `pipeline`, `graph`, `colors`, `geometry`, `poses`) imports
only numpy/PyYAML, so `import apairo_visu` works headless; Open3D is pulled in
lazily the first time `LidarViewer` is used.

## Documentation

- [Getting started](docs/getting_started.md)
- [LidarViewer API](docs/viewer.md)
- [Label configurations](docs/label_configs.md)
- [Synchronising async datasets](docs/sync.md)
- [Examples](docs/examples.md)
