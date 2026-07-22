"""Visualise each augmentation type independently on pre-voxelised RELLIS-3D frames.

Shows one panel per augmentation type so you can assess the effect of each in
isolation, using the individual apairo_transform classes directly:
  - Original
  - Yaw        RandomRotation(axis="z")
  - Flip       RandomFlip(axis="x") + RandomFlip(axis="y")
  - Pitch      RandomRotation(axis="y", max_angle=...)
  - Combined   all three

Each panel uses an independent transform instance — no yaw bleeds into flip or pitch.

Requires pre-voxelised channels (voxelised, voxelised_trav_gt) on disk.

Usage::

    python examples/view_augmentation.py ~/data/rellis
    python examples/view_augmentation.py ~/data/rellis --pitch-angle 0.4
    python examples/view_augmentation.py ~/data/rellis --sequence 00001 --every 10
"""

from __future__ import annotations

from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_transform import RandomFlip, RandomRotation

from utils import get_generic_argparser_rellis

_BINARY_CFG = {
    "color_map":    {0: [220, 60, 60], 1: [60, 220, 60]},
    "semantic_map": {0: "non-trav",    1: "traversable"},
}


def _wrap(t):
    """Make a Pipeline step from an apairo_transform point-cloud transform."""
    def step(pts, labels):
        return t(pts), labels
    return step


def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--pitch-angle", type=float, default=0.2,
                   help="Max pitch angle in radians (default: 0.2 ~ 11 deg).")
    p.add_argument("--save", type=str, default=None, metavar="PATH",
                   help="Save the Rerun recording to a .rrd file instead of opening the viewer.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["voxelised", "voxelised_trav_gt"])
    print(f"  {len(ds)} voxelised frames — sequences: {ds.sequence_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, len(ds), args.every)

    pa = args.pitch_angle

    yaw   = _wrap(RandomRotation(axis="z"))
    flipx = _wrap(RandomFlip(axis="x", p=0.5))
    flipy = _wrap(RandomFlip(axis="y", p=0.5))
    pitch = _wrap(RandomRotation(axis="y", max_angle=pa))

    pipelines = [
        Pipeline("Original",                          point_key="voxelised", label_key="voxelised_trav_gt"),
        Pipeline("Yaw",          [yaw],               point_key="voxelised", label_key="voxelised_trav_gt"),
        Pipeline("Flip",         [flipx, flipy],      point_key="voxelised", label_key="voxelised_trav_gt"),
        Pipeline(f"Pitch {pa}r", [pitch],             point_key="voxelised", label_key="voxelised_trav_gt"),
        Pipeline("Combined",     [yaw, flipx, flipy, pitch], point_key="voxelised", label_key="voxelised_trav_gt"),
    ]

    apairo_rr.view(
        ds,
        label_cfgs=[_BINARY_CFG] * len(pipelines),
        frames=frames,
        pipelines=pipelines,
        save=args.save,
    )


if __name__ == "__main__":
    main()
