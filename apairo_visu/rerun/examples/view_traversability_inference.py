"""Visualise traversability **inference** as a confusion map on RELLIS-3D.

Colours each point by its confusion-matrix cell (prediction vs. ground truth):

    TP  green   predicted traversable, and it is        (correct)
    TN  gray    predicted non-traversable, and it isn't (correct)
    FP  red     predicted traversable, but it ISN'T     (unsafe error)
    FN  amber   predicted non-traversable, but it IS    (over-cautious error)

The ground truth comes from ``TraversabilityFromLabels`` (RELLIS semantics).
The *prediction* here is a **placeholder** (the GT with a fraction of points
flipped) so the example runs without a model — see ``_PlaceholderModel``.  To
visualise a real model, inject its per-point binary prediction into the sample
under a channel and point ``--pred-key`` at it; everything else is unchanged.

Usage::

    python examples/view_traversability_inference.py
    python examples/view_traversability_inference.py ~/data/rellis --every 10
    python examples/view_traversability_inference.py --sequence 00000 --flip 0.2
    python examples/view_traversability_inference.py --pred-key my_model_pred
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline, Preprocess
from apairo_rr.confusion import CONFUSION_CFG, TraversabilityConfusion
from apairo_preprocess import TraversabilityFromLabels

from utils import get_generic_argparser_rellis


class _PlaceholderModel:
    """Stand-in for a real traversability model — DEMO ONLY.

    Emulates a model by starting from the semantic ground truth and flipping a
    fraction of points, so the confusion view exercises all four cells.  Replace
    with your model's per-point binary prediction channel via ``--pred-key``.
    """

    input_keys = ["labels"]

    def __init__(self, trav_ids, flip: float = 0.12, seed: int = 0) -> None:
        self._trav_ids = list(trav_ids)
        self._flip = flip
        self._rng = np.random.default_rng(seed)

    def process(self, sample) -> np.ndarray:
        gt = np.isin(np.asarray(sample.data["labels"]), self._trav_ids)
        flip = self._rng.random(len(gt)) < self._flip
        return np.where(flip, ~gt, gt).astype(np.uint8)


def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--flip", type=float, default=0.12,
                   help="Placeholder model error rate (ignored with --pred-key).")
    p.add_argument("--pred-key", default=None,
                   help="Use a real prediction channel already in the sample.")
    p.add_argument("--save", default=None, help="Write a .rrd file instead of spawning.")
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
        frames = list(ds.sequence(args.sequence)._indices[args.idx::args.every])
    else:
        frames = list(range(args.idx, n, args.every))

    gt_proc = TraversabilityFromLabels()
    ds = Preprocess(gt_proc, default=None).run(ds, frames, key="trav_gt")

    pred_key = args.pred_key
    if pred_key is None:
        pred_key = "trav_pred"
        model = _PlaceholderModel(gt_proc._trav_ids, flip=args.flip)
        ds = Preprocess(model, default=None).run(ds, frames, key=pred_key)
        print(f"  [demo] placeholder prediction (flip={args.flip}); "
              f"pass --pred-key to use a real model channel")

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "non-traversable", 1: "traversable"},
    }

    apairo_rr.view(
        ds,
        label_cfgs=[cfg_trav, cfg_trav, CONFUSION_CFG],
        frames=frames,
        pipelines=[
            Pipeline("Ground truth",       label_key="trav_gt"),
            Pipeline("Inference (model)",  label_key=pred_key),
            Pipeline(
                "Inference (confusion)",
                [TraversabilityConfusion(pred_key, "trav_gt")],
                label_key=None,
            ),
        ],
        save=args.save,
        spawn=args.save is None,
    )


if __name__ == "__main__":
    main()
