"""Example: side-by-side pipeline comparison with LidarViewer.

Demonstrates three comparison scenarios (select with --demo):

  preprocess   Raw data vs. a range-filtered, height-clipped cloud.
  models       Two stub models predicting labels on the same frame.
  full         Four-way: raw | preprocess | preprocess+model A | preprocess+model B.

Each pipeline step is either a callable with signature::

    step(pts: np.ndarray, labels: np.ndarray | None)
        -> tuple[np.ndarray, np.ndarray | None]

or an apairo ``FramePreprocessor`` (its output becomes the new
labels -- handy for dropping a real segmentation preprocessor straight in).
Replace the stub models below with your actual inference calls.

**Active pipeline toggles** -- the left panel has an "Active pipelines" section with one
checkbox per pipeline.  Uncheck a pipeline to hide its viewport and stop computing it;
re-check to bring it back on the current frame.  This lets you focus on any subset of
pipelines without restarting the script.

Usage::

    python examples/view_pipelines.py --root /data/goose --split val
    python examples/view_pipelines.py --root /data/goose --split val --demo models
    python examples/view_pipelines.py --root /data/goose --split val --demo full
"""

from __future__ import annotations

import argparse

import numpy as np

import apairo
import apairo_visu
from apairo_visu import Pipeline



# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------

def range_filter(
    pts: np.ndarray,
    labels: np.ndarray | None,
    min_range: float = 1.0,
    max_range: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove points closer than *min_range* or farther than *max_range* metres."""
    r = np.linalg.norm(pts[:, :3], axis=1)
    mask = (r >= min_range) & (r <= max_range)
    return pts[mask], labels[mask] if labels is not None else None


def height_clip(
    pts: np.ndarray,
    labels: np.ndarray | None,
    z_min: float = -2.0,
    z_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove points outside the given Z-axis range."""
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    return pts[mask], labels[mask] if labels is not None else None


def preprocess(
    pts: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Combination of range filter and height clip -- typical LiDAR preprocessing."""
    pts, labels = range_filter(pts, labels)
    pts, labels = height_clip(pts, labels)
    return pts, labels


# ---------------------------------------------------------------------------
# Stub models
#
# Replace these with real inference calls, e.g.:
#
#   def model_a(pts, labels):
#       logits = net_a(torch.from_numpy(pts).cuda())
#       return pts, logits.argmax(dim=-1).cpu().numpy()
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def model_a(
    pts: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Stub: random label predictions from a fixed set of class IDs."""
    classes = [0, 1, 3, 5, 7, 9, 12, 14, 17, 20, 23, 31]
    pred = _RNG.choice(classes, size=len(pts)).astype(np.int64)
    return pts, pred


def model_b(
    pts: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Stub: different random label predictions (biased toward fewer classes)."""
    classes = [0, 3, 23, 31]
    pred = _RNG.choice(classes, size=len(pts), p=[0.1, 0.2, 0.5, 0.2]).astype(np.int64)
    return pts, pred


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

DEMOS: dict[str, list[Pipeline]] = {
    # Compare raw data against a filtered cloud to validate preprocessing.
    "preprocess": [
        Pipeline("Raw"),
        Pipeline("Range + height clip", [preprocess]),
    ],

    # Run two models on the same raw input and compare their predictions.
    "models": [
        Pipeline("Ground truth"),
        Pipeline("Model A",  [model_a]),
        Pipeline("Model B",  [model_b]),
    ],

    # Full pipeline: also compare the effect of preprocessing on model output.
    "full": [
        Pipeline("Raw"),
        Pipeline("Preprocessed",      [preprocess]),
        Pipeline("Preprocess + A",    [preprocess, model_a]),
        Pipeline("Preprocess + B",    [preprocess, model_b]),
    ],
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline comparison demo")
    parser.add_argument("--root",  default="/data/goose", help="Dataset root directory")
    parser.add_argument("--split", default="val",         help="Dataset split")
    parser.add_argument("--idx",   type=int, default=0,   help="Starting frame index")
    parser.add_argument(
        "--demo",
        default="preprocess",
        choices=list(DEMOS),
        help="Comparison scenario to run (default: preprocess)",
    )
    args = parser.parse_args()

    ds  = apairo.Goose3DDataset(args.root, keys=["lidar", "labels"], split=args.split)
    cfg = apairo_visu.load_label_config("goose")

    print(f"Demo: {args.demo!r} -- {len(DEMOS[args.demo])} viewports")
    print("Navigation: <- -> (or H / L)  |  T: colour mode  |  B: BEV  |  Sync cam: panel button")

    apairo_visu.LidarViewer.launch(
        ds,
        label_cfg=cfg,
        start_idx=args.idx,
        pipelines=DEMOS[args.demo],
    )


if __name__ == "__main__":
    main()
