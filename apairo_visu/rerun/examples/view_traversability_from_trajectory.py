"""Visualise TraversabilityFromTrajectory on RELLIS-3D with Rerun (in-place, no disk write).

Loads GT poses from RELLIS and computes per-point traversability in memory using
TraversabilityFromTrajectory — no data written to disk.

Usage::

    python examples/view_traversability_from_trajectory.py
    python examples/view_traversability_from_trajectory.py --root ~/data/rellis --sequence 00000
    python examples/view_traversability_from_trajectory.py --sequence 00000 \\
        --robot-radius 1.5 --every 5
"""

from __future__ import annotations

from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline, Preprocess
from apairo_preprocess import TraversabilityFromTrajectory
from apairo_transform import RangeFilter

from utils import get_generic_argparser_rellis


def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--robot-radius",   type=float, default=1.0,  help="Robot XY half-width (m).")
    p.add_argument("--height-min",     type=float, default=-5.0, help="Min point height (m).")
    p.add_argument("--height-max",     type=float, default=0.5,  help="Max point height (m).")
    p.add_argument("--forward-window", type=int,   default=None, help="Look-ahead pose count.")
    p.add_argument("--sequence-gap",   type=float, default=5.0,  help="Boundary gap (m).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels", "poses"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq_frames = list(ds.sequence(args.sequence)._indices)
    else:
        seq_frames = list(range(n))

    display_frames = seq_frames[args.idx::args.every]

    trav_proc = TraversabilityFromTrajectory(
        robot_radius=args.robot_radius,
        height_min=args.height_min,
        height_max=args.height_max,
        forward_window=args.forward_window,
        sequence_gap=args.sequence_gap,
    )
    ds = Preprocess(trav_proc, default=None).run(ds, seq_frames, key="trav_traj")

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "non-traversable", 1: "traversable"},
    }

    rf = RangeFilter(max=50.0)

    print(f"[params] robot_radius={args.robot_radius}, height=[{args.height_min}, {args.height_max}]")

    apairo_rr.view(
        ds,
        label_cfgs=[apairo_rr.load_label_config("rellis"), cfg_trav],
        pose_key="poses",
        frames=display_frames,
        pipelines=[
            Pipeline("Semantic GT",                 [rf]),
            Pipeline("Traversability — trajectory", [rf], label_key="trav_traj"),
        ],
    )


if __name__ == "__main__":
    main()
