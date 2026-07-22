# apairo-rr

[Rerun](https://rerun.io) visualisation layer for [apairo](../apairo) datasets.

Logs LiDAR point clouds, semantic labels, robot trajectories, and **camera / image channels** to the Rerun viewer.  Supports multiple side-by-side pipelines, image channels updating along the timeline, sequence-aware navigation, and any `apairo` dataset (RELLIS-3D, SemanticKITTI, GOOSE, TartanKitti…).

---

## Installation

```bash
pip install -e .
```

Requires `apairo` and `apairo_preprocess` (resolved from local paths in `pyproject.toml`).

---

## Command line

Replay any apairo dataset straight from the shell — registered as the `rerun`
subcommand of the apairo CLI (also available as the standalone `apairo-rerun`):

```bash
# One 3D view per --lidar channel, one 2D view per --camera channel,
# all updating together along the timeline. Channel names are on-disk names;
# aliases (e.g. ouster_points -> lidar in channels.yaml) are resolved for you.
apairo rerun /path/to/ds --lidar ouster_points --camera zed_rgb

# Restrict to one sequence; several channels, comma-separated.
apairo rerun /path/to/ds --sequence 00000 --lidar velodyne_0,velodyne_1 --camera image_left,image_right

# Colour by true height: lift the (tilted) sensor cloud upright with the static
# mount TF, and into the world frame with a per-frame pose channel.
apairo rerun /path/to/ds --lidar ouster_points --color height \
    --lidar-frame os_sensor --base-frame base_link --pose dlio_odom_node_odom

# Ground-truth labels colouring the lidar points -- only frames the labels
# channel actually covers are replayed.
apairo rerun /path/to/ds --lidar velodyne_0 --labels labels --label-config semantic_kitti

# Discover the channel names (run with no --lidar/--camera).
apairo rerun /path/to/ds
```

| Flag | Description |
|------|-------------|
| `--sequence ID` | Restrict to one sequence (default: the whole dataset) |
| `--lidar A,B` | Point-cloud channel(s) shown as 3D views (aliases resolved) |
| `--camera A,B` | Image channel(s) shown as 2D views |
| `--labels A,B` | Labelled point channel(s) (e.g. ground-truth semantic labels), coloured onto the matching `--lidar` channel's points rather than shown as their own view. Paired by position with `--lidar`, or one channel shared by all. Frames without a match are dropped, so the viewer replays only the labelled subset |
| `--label-config` | Optional class colour/name legend for `--labels` (`rellis`, `semantic_kitti`, `goose`; default: inferred from `--as`). Any label id colours fine without one -- it only adds named classes for a known semantic table |
| `--color` | Point colouring: `flat` (default), `height` (z), or `range` |
| `--lidar-frame` / `--base-frame` | Override the static mount TF (e.g. `os_sensor`→`base_link`) |
| `--raw` | Keep clouds in their native sensor frame (disable the automatic upright mount TF) |
| `--pose CHANNEL` | Lift each cloud into the world frame with its per-frame pose and draw the trajectory |
| `--range M` | Drop points farther than `M` metres |
| `--as CLASS` | Dataset class to load with (default: `RawDataset`) |
| `--start S` / `--end E` | Frame window — an integer position (`100`, `-50` from the end) or a fraction of the sequence as a float in `[0,1]` (`0.5` = halfway) |
| `--every N` | Log every Nth frame |
| `--max-points N` / `--point-radius R` | Point-cloud density / radius |
| `--save FILE.rrd` / `--web` | Write a recording / serve a web viewer instead of spawning |

**Upright by default.**  Lidars are usually mounted tilted, so a raw scan "looks
up" instead of forward.  The CLI auto-resolves a static mount TF from the
calibration tree (a conventional sensor frame → `base_link`) and applies it so
the cloud sits upright; pass `--lidar-frame/--base-frame` to override it or
`--raw` to keep the native sensor frame.

Colouring is **flat** by default (no per-point scalar).  `--color height` colours
by `z` — true height once the cloud is upright (the auto / explicit mount), and
relative to the world once `--pose` lifts it into the odometry frame.

For **async / multi-rate** rigs (each sensor has its own `timestamps.txt`), the
channels update independently in time order: scrub the **time** axis and watch
each sensor refresh at its own rate while the others hold their last value.  A
**Frames** panel shows each channel's own frame index (e.g. `lidar: 1234` /
`camera: 567`), each line refreshing only when that sensor ticks — so you always
know which scan / image is on screen, even though the global `frame` timeline
just counts interleaved events.

---

## Quickstart

```python
import apairo
import apairo_rr
from apairo_rr import Pipeline

ds = apairo.Rellis3DDataset("/data/RELLIS", keys=["lidar", "labels"])

apairo_rr.view(
    ds,
    label_cfgs=[apairo_rr.load_label_config("rellis")],
    pipelines=[Pipeline("Semantic GT")],
)
```

---

## API

### `view(dataset, *, ...)`

Main entry point.  Logs a dataset to the Rerun viewer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | apairo dataset | Any dataset supporting `dataset[idx]` -> `Sample` |
| `label_cfg` | `dict \| None` | Label config applied to all pipelines |
| `label_cfgs` | `list[dict \| None]` | Per-pipeline label configs (overrides `label_cfg`) |
| `poses` | `list[np.ndarray]` | 4×4 pose matrices — logs a trajectory overlay |
| `pipelines` | `list[Pipeline]` | Processing pipelines, one 3D view each |
| `images` | `list[str \| ImageChannel]` | Image channels shown as 2D views beside the clouds, updating along the timeline |
| `point_key` | `str` | Sample key for the point cloud (default `"lidar"`) |
| `label_key` | `str \| None` | Sample key for semantic labels (default `"labels"`) |
| `frames` | `Iterable[int] \| None` | Frame indices to log (default: all) |
| `spawn` | `bool` | Spawn the Rerun viewer automatically (default `True`) |

If the dataset exposes `sequence_ids` (all `ProfiledDataset` subclasses), a **Sequence** panel appears in the viewer and updates as you scrub the timeline.

### `Pipeline(name, steps=[])`

Named sequence of per-frame transforms.

```python
Pipeline("Raw")
Pipeline("Range filter", [range_filter])
Pipeline("Trav — labels", [range_filter, TraversabilityFromLabels()])
```

Each step must have signature `(pts, labels) -> (pts, labels)` or `(pts, labels, frame_idx=...) -> (pts, labels)`.  `apairo_preprocess` `FramePreprocessor` objects are accepted directly.

### `ImageChannel(key, name=None, colormap=None)`

A 2D image channel to display beside the point clouds, updating per frame.  Pass
plain channel-key strings or `ImageChannel` instances to `view(images=...)`:

```python
from apairo_rr import ImageChannel, colorize

apairo_rr.view(
    ds,
    pipelines=[Pipeline("LiDAR", point_key="velodyne_0", label_key=None)],
    images=[
        "image_left_color",                                  # RGB, logged as-is
        ImageChannel("image_right", name="Right camera"),
        ImageChannel("depth_left", name="Depth",             # scalar map -> colour
                     colormap=lambda a: colorize(a, vmin=0.0, vmax=30.0)),
    ],
)
```

RGB `img` channels are logged directly; use `colormap=` with `colorize()` to turn
a single-channel map (depth, height, cost) into a colour image.  A channel
missing from a frame keeps its last value on screen, so async sensors stay in
sync.  See `examples/view_image_channels.py`.

### `load_label_config(name)`

Load a built-in label config by name.

```python
cfg = apairo_rr.load_label_config("rellis")          # RELLIS-3D (20 classes)
cfg = apairo_rr.load_label_config("semantic_kitti")  # SemanticKITTI (28 classes)
cfg = apairo_rr.load_label_config("goose")           # GOOSE (64 classes)
```

Returns a dict with `color_map` and `semantic_map` keys, compatible with `view(label_cfgs=...)`.

---

## Examples

### RELLIS-3D — three-way traversability comparison

```bash
# All sequences
python examples/view_rellis_traversability.py --root ~/data/rellis

# Single sequence, every 5th frame
python examples/view_rellis_traversability.py --sequence 00000 --every 5

# Custom robot radius for trajectory-based traversability
python examples/view_rellis_traversability.py --sequence 00001 --radius 0.8
```

**CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | `~/data/rellis` | Dataset root directory |
| `--sequence` | *(all)* | Sequence ID to visualise (`00000`, `00001`, …) |
| `--every N` | `1` | Log every Nth frame |
| `--idx N` | `0` | Start at frame N (within the selected sequence) |
| `--radius R` | `1.0` | Robot radius for `TraversabilityFromTrajectory` |

---

## Sequence navigation

When `--sequence` is omitted, all sequences are concatenated on the same timeline.  Scrub the Rerun timeline to move between frames — the **Sequence** panel at the top of the viewer shows the current sequence ID.

To jump directly to a sequence, relaunch with `--sequence <id>`.  Available IDs are printed at startup:

```
  1832 scans — sequences: ['00000', '00001', '00002', '00003', '00004']
```

---

## Adding a custom dataset

Any object supporting `__getitem__(int) -> Sample` and `__len__` works with `view()`.  For sequence-aware navigation, the dataset must expose:

```python
dataset.sequence_ids          # list[str]
dataset.sequence(seq_id)      # SequenceView with ._indices (global frame indices)
```

All `apairo.ProfiledDataset` subclasses (RELLIS, SemanticKITTI, GOOSE) implement this automatically.
