# LidarViewer

The main class of `apairo_visu`. Builds an Open3D GUI window that lets you navigate a dataset frame by frame, with optional per-frame transforms (preprocessing, model inference) shown side-by-side.

## API

### `LidarViewer.launch` (static)

```python
LidarViewer.launch(
    dataset,
    view_cfg: ViewConfig | None = None,
    label_cfg: dict | None = None,
    poses: list[np.ndarray] | None = None,
    start_idx: int = 0,
    pipelines: list[Pipeline] | None = None,
)
```

Creates the application and **blocks** until the window is closed.

| Parameter | Description |
|---|---|
| `dataset` | Any apairo dataset -- must support `dataset[idx]` returning a `Sample`. |
| `view_cfg` | Which keys to read from each `Sample`. Defaults to `point_key="lidar"`, `label_key="labels"`. |
| `label_cfg` | Dict with `color_map`, `semantic_map`, optional `traversable_map`. Use `load_label_config()` for built-in configs. |
| `poses` | Optional list of 4x4 `np.ndarray` (T_world_sensor), one per frame. Enables the trajectory overlay (viewport 0 only). |
| `start_idx` | Frame index to display first. |
| `pipelines` | List of `Pipeline` objects -- one viewport per entry. Defaults to `[Pipeline("Raw", [])]`. |

---

### `ViewConfig`

```python
@dataclass
class ViewConfig:
    point_key: str = "lidar"          # key for the point cloud in sample.data
    label_key: str | None = "labels"  # key for per-point labels, or None
    intensity_channel: int = 3        # column index for the intensity channel
```

Tells the viewer which tensors to extract from `sample.data` on each frame.  
The point cloud tensor must have shape `(N, C)` float32 with X, Y, Z in the first three columns.

---

### `Pipeline`

```python
@dataclass
class Pipeline:
    name: str
    steps: list[Callable] = field(default_factory=list)
```

A named sequence of per-frame transforms.  Each step is either a callable:

```python
step(pts: np.ndarray, labels: np.ndarray | None)
    -> tuple[np.ndarray, np.ndarray | None]
```

or an apairo `FramePreprocessor` (its `process` output becomes the new labels).
Steps are applied in order.  Pass an empty `steps` list to display the raw frame.

One `Pipeline` = one viewport.  When multiple pipelines are provided, they run **in parallel** on a thread pool and each viewport updates as soon as its pipeline finishes.

```python
Pipeline("Raw")                                # display raw data
Pipeline("Preprocessed", [my_preprocess])      # preprocessing only
Pipeline("Model A", [preprocess, model_a])     # preprocess -> inference
```

---

## Display modes

Three colour modes, cycled with `T` or via the combo box:

| Mode | Description | Requires |
|---|---|---|
| **Semantic** | Per-class colour from `label_cfg` | `label_key` + `label_cfg` |
| **Intensity** | Grayscale from the intensity column | `intensity_channel < point_cloud.shape[1]` |
| **Height** | Viridis colormap on Z | always available |

When `label_key=None` the viewer starts in **Height** mode.  When the intensity channel is out of bounds (e.g. `(N, 3)` clouds), Intensity mode falls back to Height silently.

The selected mode is applied to **all viewports simultaneously**.

---

## Panel layout

### Single pipeline (default)

```
┌──────────────────┬──────────────────────────────────────┐
│  NAVIGATION      │                                      │
│  frame / total   │                                      │
│  point count     │                                      │
│  [< Prev][Next>] │                                      │
│  [BEV]  [Reset]  │         3-D scene                    │
│                  │                                      │
│  COLOUR MODE [T] │                                      │
│  [Semantic ▼]    │                                      │
│                  │                                      │
│  OVERLAYS        │                                      │
│  [x] Trajectory[J] │                                      │
│                  │                                      │
│  CLASS DISTRIB.  │                                      │
│  asphalt  34.2%  │                                      │
│  ...             │                                      │
│  FILTER CLASSES  │                                      │
│  [Show][Hide]    │                                      │
│  ■ 23: asphalt   │                                      │
│  ...             │                                      │
└──────────────────┴──────────────────────────────────────┘
```

### Multiple pipelines

```
┌──────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  NAVIGATION      │   Raw           │   Preprocessed  │   Model A       │
│  [< Prev][Next>] ├─────────────────┼─────────────────┼─────────────────┤
│  [BEV]  [Reset]  │                 │                 │                 │
│  [Sync cam]      │                 │                 │                 │
│                  │   3-D scene 0   │   3-D scene 1   │   3-D scene 2   │
│  COLOUR MODE [T] │                 │                 │                 │
│  [Semantic ▼]    │                 │                 │                 │
│                  │                 │                 │                 │
│  ACTIVE PIPELINES│                 │                 │                 │
│  [x] Raw           │                 │                 │                 │
│  [x] Preprocessed  └─────────────────┴─────────────────┴─────────────────┘
│  [x] Model A
│
│  PIPELINES
│  Raw: 0 ms
│  Preprocessed: 4 ms
│  Model A: 42 ms
│
│  CLASS DISTRIB.
│  (from pipeline 0)
│  FILTER CLASSES
│  (all viewports)
└──────────────────
```

The **Overlays** and **Filter classes** sections are only shown when the corresponding data is available (poses and labels, respectively).  The **Active pipelines**, **Pipelines** timing, and **Sync cam** button are only shown in multi-pipeline mode.

---

## Active pipeline toggles

In multi-pipeline mode, the **Active pipelines** section shows one checkbox per pipeline.  Unchecking a pipeline:

- immediately hides its viewport and redistributes the available width among the remaining active ones,
- skips its computation on subsequent frame navigations (useful when inference is slow),
- re-runs it on the current frame when re-checked (using the cached raw data -- no dataset re-read).

This lets you compare any subset of pipelines without restarting the script.

```
# Start with 4 pipelines, then uncheck "Model B" in the panel
# -> 3 viewports fill the screen, Model B stops computing
# -> Re-check "Model B" -> it re-runs on the current frame and reappears
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `->` or `L` | Next frame |
| `<-` or `H` | Previous frame |
| `T` | Cycle colour mode (all viewports) |
| `B` | Bird's-eye (top-down) view (all viewports) |
| `R` | Reset camera to bounding box (all viewports) |
| `J` | Toggle trajectory overlay (when poses provided) |

---

## Trajectory overlay

When `poses` is provided, the viewer draws in viewport 0:
- **Blue line** -- past trajectory (frames 0 to current)
- **Orange line** -- future trajectory (frames current to end)

Both are rendered in the **current sensor frame**: all trajectory waypoints (world-frame origins from the pose matrices) are transformed into the coordinate system of the current LiDAR scan using `T_sensor_world = inv(poses[current_idx])`.

```python
# From a dataset pose channel (3x4 / 4x4 / [x y z qx qy qz qw] all handled):
poses = apairo_visu.load_poses(ds, key="poses")     # list of (4, 4) float64
apairo_visu.LidarViewer.launch(ds, poses=poses)
```

---

## Camera sync (multi-pipeline)

In multi-pipeline mode each viewport has an independent camera, allowing you to inspect different parts of different pipeline outputs simultaneously.  Use the **Sync cam** button or press `R` to reset all cameras to the same bounding-box view.

---

## Programmatic construction

```python
app = gui.Application.instance
app.initialize()

viewer = apairo_visu.LidarViewer(
    dataset, view_cfg=view_cfg, label_cfg=label_cfg, poses=poses, pipelines=pipelines
)
viewer._build_window()
app.run()
```
