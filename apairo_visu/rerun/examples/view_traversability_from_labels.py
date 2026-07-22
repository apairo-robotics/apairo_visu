"""Visualise TraversabilityFromLabels on RELLIS-3D with Rerun (in-place, no disk write).

Compares the original semantic labels with the binary traversable/non-traversable
mapping produced by TraversabilityFromLabels — no data written to disk.

Requires: apairo-rr  (pip install apairo-rr)

Usage::

    python examples/view_traversability_from_labels.py ~/data/rellis
    python examples/view_traversability_from_labels.py ~/data/rellis --sequence 00000 --every 5
    python examples/view_traversability_from_labels.py ~/data/rellis --ids 1 3 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import TraversabilityFromLabels
from apairo_transform import RangeFilter

from utils import get_generic_argparser_rellis


def main() -> None:
    p = get_generic_argparser_rellis()

    p.add_argument("--ids", nargs="+", type=int, default=None,
                   help="Traversable class IDs. Defaults to RELLIS built-in set.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    traversable_ids = frozenset(args.ids) if args.ids else None
    trav_proc = TraversabilityFromLabels(traversable_ids=traversable_ids)
    print(f"  traversable IDs: {trav_proc._trav_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "not traversable", 1: "traversable"},
    }

    rf = RangeFilter(max=50.0)

    apairo_rr.view(
        ds,
        label_cfgs=[apairo_rr.load_label_config("rellis"), cfg_trav],
        frames=frames,
        pipelines=[
            Pipeline("Semantic GT",              [rf]),
            Pipeline("Traversability — labels",  [rf, trav_proc]),
        ],
    )


if __name__ == "__main__":
    main()
