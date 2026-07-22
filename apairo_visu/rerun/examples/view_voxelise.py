"""Visualise voxel-grid downsampling on RELLIS-3D with Rerun (in-place, no disk write).

Compares the original point cloud (range-filtered) with the voxelised version
using VoxelisePointCloud + VoxeliseLabels — no data written to disk.

Requires: apairo-rr  (pip install apairo-rr)

Usage::

    python examples/view_voxelise.py ~/data/rellis
    python examples/view_voxelise.py ~/data/rellis --voxel-size 0.2 --sequence 00000
    python examples/view_voxelise.py ~/data/rellis --voxel-size 0.5 --every 5
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import VoxeliseLabels, VoxelisePointCloud
from apairo_transform import RangeFilter

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


def make_voxelise_step(voxel_size: float, max_range: float | None = None):
    """Wrap VoxelisePointCloud + VoxeliseLabels into a single pipeline callable."""
    pc_proc = VoxelisePointCloud(voxel_size=voxel_size, max_range=max_range)
    lb_proc = VoxeliseLabels(voxel_size=voxel_size, max_range=max_range)

    def step(pts, labels):
        sample = types.SimpleNamespace(data={"lidar": pts, "labels": labels})
        new_pts = pc_proc.process(sample)
        new_labels = lb_proc.process(sample) if labels is not None else None
        return new_pts, new_labels

    return step


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root",          nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--voxel-size",  type=float, default=0.1,
                   help="Voxel edge length in metres (default: 0.1).")
    p.add_argument("--max-range",   type=float, default=50.0,
                   help="Range limit applied before voxelisation (default: 50.0 m).")
    p.add_argument("--reduction",   choices=["centroid", "first", "random"],
                   default="centroid", help="Point selection strategy per voxel.")
    p.add_argument("--aggregation", choices=["majority", "max"],
                   default="majority", help="Label aggregation per voxel.")
    p.add_argument("--sequence",    default=None, help="Restrict to one sequence ID.")
    p.add_argument("--every",       type=int, default=1,  help="Log every Nth frame.")
    p.add_argument("--idx",         type=int, default=0,  help="First frame index.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    print(f"  voxel_size={args.voxel_size} m, max_range={args.max_range} m")
    print(f"  reduction={args.reduction}, label aggregation={args.aggregation}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    voxelise_step = make_voxelise_step(args.voxel_size, max_range=args.max_range)
    rellis_cfg = apairo_rr.load_label_config("rellis")
    rf = RangeFilter(max=args.max_range)

    apairo_rr.view(
        ds,
        label_cfgs=[rellis_cfg, rellis_cfg],
        frames=frames,
        pipelines=[
            Pipeline("Original",
                     [rf]),
            Pipeline(f"Voxelised  {args.voxel_size} m",
                     [voxelise_step]),
        ],
    )


if __name__ == "__main__":
    main()
