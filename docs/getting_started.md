# Getting started

## Installation

`apairo_visu` depends on [apairo](../../apairo) (local package). The desktop viewer renders with [Open3D](https://www.open3d.org/), an optional extra.

```bash
cd ~/dev/apairo_visu
python -m venv .venv && source .venv/bin/activate

pip install -e ../apairo   # install apairo from local sources
pip install -e ".[open3d]" # install apairo_visu with the Open3D desktop viewer
```

Open3D is a large package (~200 MB), and only the desktop viewer (`LidarViewer`,
`apairo visu`) needs it: leave out `[open3d]` if you only use the studio
(`.[studio]`) or the Rerun backend (`.[rerun]`).

## Concepts

### Dataset

`apairo_visu` works with any apairo dataset. The viewer calls `dataset[idx]` and reads `sample.data[key]` to obtain the point cloud (and optionally labels). No custom wrappers are needed.

The built-in datasets that work out of the box are:

| Dataset class | `point_key` | `label_key` | Built-in config |
|---|---|---|---|
| `Goose3DDataset` | `"lidar"` | `"labels"` | `"goose"` |
| `Rellis3DDataset` | `"lidar"` | `"labels"` | `"rellis"` |
| `SemanticKittiDataset` | `"lidar"` | `"labels"` | `"semantic_kitti"` |
| `TartanKittiDataset` | `"velodyne_0"` | -- | -- |

### ViewConfig

`ViewConfig` tells the viewer which keys to read from each `Sample`:

```python
view_cfg = apairo_visu.ViewConfig(
    point_key="lidar",       # key for the point cloud tensor (float32, shape [N, C])
    label_key="labels",      # key for per-point labels (int64, shape [N]) -- or None
    intensity_channel=3,     # column index for intensity in the point cloud
)
```

The point cloud tensor is expected to have at least 3 columns (X, Y, Z). A 4th column (intensity) is optional.

### Label config

A label config is a plain dict (usually loaded from YAML) that describes how to colour each class:

```python
label_cfg = {
    "color_map":    {0: [0, 0, 0], 23: "#ff2f80", ...},   # class id -> RGB
    "semantic_map": {0: "unlabeled", 23: "asphalt", ...},  # class id -> name
    "traversable_map": [23, 31, 50, 51],                   # optional
}
```

Use `load_label_config(name)` to load a built-in config, or pass a path to a custom YAML file. See [Label configurations](label_configs.md) for details.

## Minimal example

```python
import apairo
import apairo_visu

ds  = apairo.Goose3DDataset("/data/goose", split="val")
cfg = apairo_visu.load_label_config("goose")

apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)
```

## Without labels

For datasets without semantic labels, omit `label_cfg` and set `label_key=None`. The viewer defaults to Height coloring (viridis on Z):

```python
view_cfg = apairo_visu.ViewConfig(point_key="velodyne_0", label_key=None)
apairo_visu.LidarViewer.launch(ds, view_cfg=view_cfg)
```
