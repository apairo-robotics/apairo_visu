"""Fixed vs per-frame colormap range on RELLIS-3D LiDAR (colour by range).

Colours each point by its range (distance from the sensor) two ways, side by side:

  - "Range (per-frame)"  — normalised on each frame's own min/max.  As you scrub
    the timeline the colours shift: the same 20 m wall changes hue depending on
    what else is in view, so colours are *not* comparable over time.
  - "Range (0–rmax m)"   — anchored to a fixed ``[0, --rmax]`` range, so a given
    distance always maps to the same colour, in every frame.

This is exactly the footgun that fixed ``vmin``/``vmax`` solve (see
``apairo_rr.colormaps``): without a fixed range every frame renormalises on its
own extent, and colours are not comparable across frames.

Usage::

    python examples/view_colormap_range.py ~/data/rellis
    python examples/view_colormap_range.py ~/data/rellis --sequence 00000 --every 5
    python examples/view_colormap_range.py ~/data/rellis --rmax 30
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline, red_blue

from utils import get_generic_argparser_rellis


def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--rmax", type=float, default=50.0,
                   help="Upper bound (m) of the fixed range colour scale (default: 50).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar"])
    print(f"  {len(ds)} scans — sequences: {ds.sequence_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = list(ds.sequence(args.sequence)._indices)[args.idx::args.every]
    else:
        frames = range(args.idx, len(ds), args.every)

    rmax = args.rmax

    def _range(pts: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pts[:, :3], axis=1)

    # Same scalar, two normalisation strategies — scrub the timeline to see the
    # per-frame panel's colours drift while the fixed-range panel stays stable.
    pipelines = [
        Pipeline("Range (per-frame)", label_key=None,
                 colormap_fn=lambda pts: red_blue(_range(pts))),
        Pipeline(f"Range (0–{rmax:g} m)", label_key=None,
                 colormap_fn=lambda pts: red_blue(_range(pts), vmin=0.0, vmax=rmax)),
    ]

    apairo_rr.view(
        ds,
        pipelines=pipelines,
        frames=frames,
        application_id="colormap_range",
    )


if __name__ == "__main__":
    main()
