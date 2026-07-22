from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from .colormaps import Colormap


def _accepts_frame_idx(fn: Callable) -> bool:
    """Whether *fn* accepts a ``frame_idx`` keyword argument.

    Determined by introspecting the signature once, rather than catching
    ``TypeError`` from a speculative call — which would silently swallow a
    genuine ``TypeError`` raised *inside* the step and re-run it (masking the
    bug and risking double-applied side effects).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "frame_idx" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


@dataclass
class Pipeline:
    """Named sequence of per-frame transforms applied before logging.

    Each step is a callable with signature::

        step(pts: np.ndarray, labels: np.ndarray | None)
            -> tuple[np.ndarray, np.ndarray | None]

    Steps are applied in order.  apairo_preprocess ``FramePreprocessor``
    objects are also accepted directly as steps — their ``process(sample)``
    method is called automatically.  Unlike the old (pts, labels)-only
    interface, steps now receive the **full Sample** so any input key
    declared in ``step.input_keys`` is resolved from the sample without
    hardcoding.

    *point_key* and *label_key* name the sample channels that map to the
    pts / labels pair returned by :meth:`run`.  They default to the apairo
    conventions (``"lidar"`` / ``"labels"``) but can be overridden per
    pipeline when the dataset uses different key names (e.g. ``"voxelised"``).

    For continuous scalar visualisation, pass a *colormap_fn* that receives
    the full ``pts`` array (after all steps) and returns an ``(N, 3)`` uint8
    RGB array::

        def my_colormap(pts: np.ndarray) -> np.ndarray:
            v = pts[:, 4]
            t = (v - v.min()) / (v.ptp() + 1e-6)
            rgb = np.zeros((len(v), 3), dtype=np.uint8)
            rgb[:, 0] = (255 * (1 - t)).astype(np.uint8)
            rgb[:, 2] = (255 * t).astype(np.uint8)
            return rgb

        Pipeline("Scalar channel", colormap_fn=my_colormap)

    When *colormap_fn* is set the default ``_height_colors(pts[:,2])`` fallback
    is skipped, so point positions stay correct regardless of the scalar.

    Examples::

        Pipeline("Raw")
        Pipeline("Range filter", [range_filter])
        Pipeline("Trav — labels", [range_filter, TraversabilityFromLabels()])
        Pipeline("Ground height (voxelised)", point_key="voxelised",
                 label_key=None, colormap_fn=lambda pts: height_colors(pts[:, 4]))
    """

    name: str
    steps: list[Callable] = field(default_factory=list)
    colormap_fn: Optional[Union[Colormap, Callable[[np.ndarray], np.ndarray]]] = None
    point_key: str = "lidar"
    label_key: Optional[str] = "labels"

    # Memo of whether each plain-callable step accepts frame_idx (keyed by id).
    _frame_idx_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def run(self, sample, frame_idx: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        """Run all steps on *sample* and return ``(pts, labels)``.

        Args:
            sample: An apairo ``Sample`` (or any object with a ``data`` dict).
                    The viewer passes the dataset sample directly so that
                    FramePreprocessor steps can access any channel they declare
                    in their ``input_keys`` — no hardcoded key mapping required.
            frame_idx: Global frame index forwarded to steps that need it.
        """
        pts = np.asarray(sample.data[self.point_key], dtype=np.float32).copy()
        _lv = sample.data.get(self.label_key) if self.label_key else None
        labels = np.asarray(_lv, dtype=np.int64).copy() if _lv is not None else None

        for step in self.steps:
            if hasattr(step, "process") and hasattr(step, "input_keys"):
                if hasattr(step, "_idx"):
                    step._idx = frame_idx
                # Build sub-sample from step.input_keys.
                # point_key / label_key map to the current (post-step) pts / labels;
                # every other declared key is looked up in the original sample.
                sub_data: dict = {}
                for k in step.input_keys:
                    if k == self.point_key:
                        sub_data[k] = pts
                    elif k == self.label_key and labels is not None:
                        sub_data[k] = labels
                    elif k in sample.data:
                        sub_data[k] = sample.data[k]
                result = step.process(types.SimpleNamespace(data=sub_data))
                if result is not None:
                    labels = np.asarray(result, dtype=np.int64)
            elif hasattr(step, "compute_mask"):
                # apairo_transform-style filter: compute mask on pts, apply to both
                mask = step.compute_mask(pts)
                pts = pts[mask]
                if labels is not None:
                    labels = labels[mask]
            else:
                accepts = self._frame_idx_cache.get(id(step))
                if accepts is None:
                    accepts = self._frame_idx_cache[id(step)] = _accepts_frame_idx(step)
                if accepts:
                    pts, labels = step(pts, labels, frame_idx=frame_idx)
                else:
                    pts, labels = step(pts, labels)

        return pts, labels
