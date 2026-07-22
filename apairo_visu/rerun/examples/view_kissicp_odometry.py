"""Visualise KISS-ICP odometry on RELLIS-3D with Rerun (in-place, no disk write).

Computes the KISS-ICP trajectory in memory and streams the estimated poses to
Rerun alongside the semantic labels — no data written to disk.

Compare the estimated trajectory (orange marker + blue line) with the map
structure to verify that the poses are geometrically consistent.

Requires: apairo-rr, kiss-icp  (pip install apairo-rr kiss-icp)

Usage::

    python examples/view_kissicp_odometry.py ~/data/rellis --sequence 00000
    python examples/view_kissicp_odometry.py ~/data/rellis --sequence 00000 --voxel-size 0.5
    python examples/view_kissicp_odometry.py ~/data/rellis --sequence 00000 --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline, Preprocess
from apairo_preprocess import KissICPOdometry
from apairo_transform import RangeFilter

from utils import get_generic_argparser_rellis

def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--voxel-size", type=float, default=1.0,
                   help="KISS-ICP map voxel size (default: 1.0 m).")
    p.add_argument("--max-range",  type=float, default=50.0,
                   help="Maximum point range kept (default: 50.0 m).")
    p.add_argument("--min-range",  type=float, default=1.0,
                   help="Minimum point range kept (default: 1.0 m).")
    p.add_argument("--deskew", action="store_true",
                   help="Enable motion deskewing (requires timestamps in column 3).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    print(f"  voxel_size={args.voxel_size} m, range=[{args.min_range}, {args.max_range}] m")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq_frames = list(ds.sequence(args.sequence)._indices)
    else:
        seq_frames = list(range(n))

    ds_session = Preprocess(
        KissICPOdometry(
            voxel_size=args.voxel_size,
            max_range=args.max_range,
            min_range=args.min_range,
            deskew=args.deskew,
        )
    ).run(ds, seq_frames, key="pose")

    apairo_rr.view(
        ds_session,
        label_cfgs=[apairo_rr.load_label_config("rellis")],
        pose_key="pose",
        frames=seq_frames[args.idx::args.every],
        pipelines=[
            Pipeline("KISS-ICP — semantic labels", [RangeFilter(max=args.max_range)]),
        ],
    )


if __name__ == "__main__":
    main()
