"""Confusion-matrix colouring for traversability inference.

Compare a model's per-point traversability **prediction** against the
**ground truth** and colour each point by its confusion-matrix cell:

    TP  green   predicted traversable, and it is        (correct)
    TN  gray    predicted non-traversable, and it isn't (correct)
    FP  red     predicted traversable, but it ISN'T     (unsafe error)
    FN  amber   predicted non-traversable, but it IS    (over-cautious error)

Both inputs are binary masks (1 = traversable, 0 = non-traversable).  Drop
:class:`TraversabilityConfusion` into a :class:`~apairo_rr.Pipeline` as a step
and pass :data:`CONFUSION_CFG` as the pipeline's label config to get the legend.
"""

from __future__ import annotations

import numpy as np

# Class ids — ordered so the legend reads TN, TP, FP, FN.
TN, TP, FP, FN = 0, 1, 2, 3

CONFUSION_CFG: dict = {
    "color_map": {
        TN: [120, 120, 120],   # gray  — correctly non-traversable
        TP: [60, 200, 90],     # green — correctly traversable
        FP: [220, 60, 50],     # red   — predicted traversable, isn't (unsafe)
        FN: [245, 175, 40],    # amber — missed traversable (over-cautious)
    },
    "semantic_map": {
        TN: "TN — non-traversable",
        TP: "TP — traversable",
        FP: "FP — false traversable (unsafe)",
        FN: "FN — missed traversable",
    },
}

# (4, 3) uint8 palette indexable by class id, for direct RGB colouring
# (e.g. ``CONFUSION_COLORS[confusion_class(pred, gt)]`` inside a colormap_fn).
CONFUSION_COLORS = np.array(
    [CONFUSION_CFG["color_map"][i] for i in (TN, TP, FP, FN)], dtype=np.uint8
)


def confusion_class(pred, gt) -> np.ndarray:
    """Per-point confusion class id from two binary traversability masks.

    Args:
        pred: Predicted traversability (1 = traversable). Shape ``(N,)``.
        gt:   Ground-truth traversability (1 = traversable). Shape ``(N,)``.

    Returns:
        ``(N,)`` uint8 array of class ids — ``0=TN, 1=TP, 2=FP, 3=FN``.
        Index :data:`CONFUSION_COLORS` for RGB, or log as ``class_ids`` with
        :data:`CONFUSION_CFG` for a labelled legend.

    Raises:
        ValueError: If *pred* and *gt* have different shapes.
    """
    pred = np.asarray(pred).astype(bool).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}"
        )
    # gt ? (pred ? TP : FN) : (pred ? FP : TN)
    return np.where(gt, np.where(pred, TP, FN), np.where(pred, FP, TN)).astype(np.uint8)


class TraversabilityConfusion:
    """Pipeline step: prediction + ground truth → per-point confusion class.

    Drop into a :class:`~apairo_rr.Pipeline` as a step.  It reads two binary
    traversability channels from the sample and returns the per-point confusion
    class id (see :func:`confusion_class`), so the viewer colours the cloud with
    :data:`CONFUSION_CFG` — TP green / TN gray / FP red / FN amber.

    Both channels must be aligned with the current point cloud, so place this
    step **before** any spatial filter (e.g. ``RangeFilter``) — a filter applied
    afterwards subsets the points and the confusion ids together.  If your
    ground truth is *semantic* rather than binary, run
    ``TraversabilityFromLabels`` first to produce the binary ``gt_key`` channel.

    Args:
        pred_key: Sample channel holding the model's binary prediction.
        gt_key:   Sample channel holding the binary ground-truth mask.

    Example::

        from apairo_rr import Pipeline, view
        from apairo_rr.confusion import TraversabilityConfusion, CONFUSION_CFG

        view(
            ds,
            label_cfgs=[CONFUSION_CFG],
            pipelines=[
                Pipeline(
                    "Inference (TP/TN/FP/FN)",
                    [TraversabilityConfusion("trav_pred", "trav_gt")],
                    label_key=None,
                ),
            ],
        )
    """

    def __init__(self, pred_key: str = "trav_pred", gt_key: str = "trav_gt") -> None:
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.input_keys = [pred_key, gt_key]

    def process(self, sample) -> np.ndarray:
        return confusion_class(
            sample.data[self.pred_key], sample.data[self.gt_key]
        )
