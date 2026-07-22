"""Visualise GICP odometry on RELLIS-3D with Rerun (in-place, no disk write).

Computes the GICP (Generalised ICP via Open3D) trajectory in memory and streams
the estimated poses to Rerun alongside the semantic labels — no data written to disk.

Compare the estimated trajectory (orange marker + blue line) with the map
structure to verify that scan-to-scan registration is geometrically consistent.

Requires: apairo-rr, open3d  (pip install apairo-rr open3d)

Usage::

    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000
    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000 --voxel-size 0.3
    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000 --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline, Preprocess
from apairo_preprocess import GICPOdometry
from apairo_transform import RangeFilter

from utils import get_generic_argparser_rellis

def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--voxel-size", type=float, default=0.3,
                   help="Down-sampling voxel size for ICP (default: 0.3 m).")
    p.add_argument("--max-range",  type=float, default=50.0,
                   help="Maximum point range kept before registration (default: 50.0 m).")
    p.add_argument("--max-corr",   type=float, default=1.0,
                   help="Maximum ICP correspondence distance (default: 1.0 m).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    print(f"  voxel_size={args.voxel_size} m, max_range={args.max_range} m, max_corr={args.max_corr} m")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq_frames = list(ds.sequence(args.sequence)._indices)
    else:
        seq_frames = list(range(n))

    ds_session = Preprocess(
        GICPOdometry(
            voxel_size=args.voxel_size,
            max_range=args.max_range,
            max_corr=args.max_corr,
        )
    ).run(ds, seq_frames, key="pose")

    apairo_rr.view(
        ds_session,
        label_cfgs=[apairo_rr.load_label_config("rellis")],
        pose_key="pose",
        frames=seq_frames[args.idx::args.every],
        pipelines=[
            Pipeline("GICP — semantic labels", [RangeFilter(max=args.max_range)]),
        ],
    )


if __name__ == "__main__":
    main()
