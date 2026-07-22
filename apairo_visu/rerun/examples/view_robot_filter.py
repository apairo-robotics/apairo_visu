"""Visualise the effect of RangeFilter on RELLIS-3D with Rerun.

Compares the raw point cloud with the range-filtered version side by side.

Usage::

    python examples/view_robot_filter.py ~/data/rellis
    python examples/view_robot_filter.py ~/data/rellis --min-range 0.5
    python examples/view_robot_filter.py ~/data/rellis --min-range 1.0 --norm L2    
    python examples/view_robot_filter.py ~/data/rellis --sequence 00000 --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_transform import RangeFilter

from utils import get_generic_argparser_rellis

def L1(x):
    """Norm L1"""
    return np.sum(np.abs(x), axis=1)

def L2(x):
    """Norm L2"""
    return np.linalg.norm(x, axis=1)

def Linf(x):
    """Norm L inf"""
    return np.max(np.abs(x), axis=1)


_MAP_NORM = {
    "L1" : L1,
    "L2" : L2,
    "Linf": Linf
}

def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--min-range",   type=float, default=1.0,
                   help="Primary range limit in metres (default: 1.0).")
    p.add_argument("--norm", type=str, default="Linf",
                   help="Choose the norm between L1, L2, Linf (default Linf : cube)")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    rellis_cfg = apairo_rr.load_label_config("rellis")

    norm = _MAP_NORM.get(args.norm) if args.norm in _MAP_NORM.keys() else L2

    pipelines = [
        Pipeline("Raw",                                          []),
        Pipeline(f"Range < {args.min_range} m ({args.norm})", [RangeFilter(min=args.min_range, norm=norm)]),
    ]
    label_cfgs = [rellis_cfg, rellis_cfg]

    apairo_rr.view(
        ds,
        label_cfgs=label_cfgs,
        frames=frames,
        pipelines=pipelines,
    )


if __name__ == "__main__":
    main()
