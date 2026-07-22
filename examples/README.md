# Examples

Runnable scripts for both visualisation backends. Each takes a dataset root and
common flags (`--sequence`, `--every`, ...); pass `--help` for the full list.

## Default backend — interactive viewer

```
python examples/view_pipelines.py ~/data/rellis
```

- `view_dataset.py` — minimal one-dataset viewer
- `view_pipelines.py` — chaining colormap / filter pipelines
- `view_rellis.py` — RELLIS-3D semantic labels
- `view_rellis_traversability.py` — three-way traversability comparison
- `view_tartandrive.py` — async multi-rate rig via `synchronize()`

## Rerun backend — `rerun/`

Needs the optional extra: `pip install "apairo-visu[rerun]"`.

```
python examples/rerun/view_ground_height.py ~/data/rellis
```

Same datasets through `apairo_visu.rerun` (odometry, augmentation, voxelisation,
image channels, traversability inference, ...). Shared helpers live in
`rerun/utils.py`; run each script from the repo root.
