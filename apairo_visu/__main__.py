"""CLI entry-point for apairo_visu.

Examples:
    python -m apairo_visu --dataset goose --root /data/goose --split train
    python -m apairo_visu --dataset rellis --root /data/rellis
    python -m apairo_visu --dataset semantic_kitti --root /data/kitti --split val
    python -m apairo_visu --dataset goose --root /data/goose --cfg my_colors.yaml --idx 50
"""

from __future__ import annotations

import argparse
import importlib

from .config import ViewConfig, load_label_config

# dataset name -> (apairo class, point key, label key)
_DATASETS = {
    "goose":          ("Goose3DDataset",       "lidar", "labels"),
    "rellis":         ("Rellis3DDataset",      "lidar", "labels"),
    "semantic_kitti": ("SemanticKittiDataset", "lidar", "labels"),
}


def _load_dataset(name: str, root: str, split: str | None, keys: list[str]):
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(_DATASETS)}")
    class_name = _DATASETS[name][0]
    cls = getattr(importlib.import_module("apairo"), class_name)
    kwargs = {"keys": keys}
    if split is not None:
        kwargs["split"] = split
    return cls(root, **kwargs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="apairo visu", description="apairo_visu -- LiDAR dataset viewer"
    )
    p.add_argument("--dataset", required=True, help=f"Dataset name ({' | '.join(_DATASETS)})")
    p.add_argument("--root", required=True, help="Path to dataset root directory")
    p.add_argument("--split", default=None, help="Dataset split (train | val | test)")
    p.add_argument("--cfg", default=None, help="Path to a custom label YAML config (default: built-in)")
    p.add_argument("--idx", type=int, default=0, help="Starting frame index")
    p.add_argument("--no-labels", action="store_true", help="Disable label loading")
    args = p.parse_args(argv)

    if args.dataset not in _DATASETS:
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Known: {sorted(_DATASETS)}")
    _, point_key, label_key = _DATASETS[args.dataset]

    keys = [point_key] if args.no_labels else [point_key, label_key]
    dataset = _load_dataset(args.dataset, args.root, args.split, keys)

    label_cfg = None if args.no_labels else load_label_config(args.cfg or args.dataset)
    view_cfg = ViewConfig(
        point_key=point_key,
        label_key=None if args.no_labels else label_key,
    )

    print(f"Dataset: {args.dataset}  ({len(dataset)} frames)")
    print(f"Starting at frame {args.idx}")

    # Imported here so `--help` works without an Open3D / display dependency.
    from .viewer import LidarViewer
    LidarViewer.launch(dataset, view_cfg=view_cfg, label_cfg=label_cfg, start_idx=args.idx)


if __name__ == "__main__":
    main()
