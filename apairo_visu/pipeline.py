"""Per-frame transform pipelines for :mod:`apairo_visu`.

A :class:`Pipeline` is a named sequence of transforms applied to one frame
before rendering.  One pipeline maps to one viewport; the viewer runs several
in parallel to compare them side by side.

Two kinds of step are accepted, and may be freely mixed in the same pipeline:

* a **plain callable** ``(pts, labels) -> (pts, labels)`` (optionally taking a
  ``frame_idx`` keyword) -- a filter, an augmentation, a model wrapper;
* an **apairo** :class:`~apairo.core.preprocessor.FramePreprocessor` -- its
  declared ``input_keys`` are fed from ``pts`` / ``labels`` and the array it
  returns becomes the new per-point labels.

This module imports only numpy; apairo is imported lazily, the first time a
``FramePreprocessor`` step is actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# A step is either a plain callable or an apairo FramePreprocessor (duck-typed
# below via ``input_keys``).
Step = Callable


def is_preprocessor(step) -> bool:
    """True for an apairo preprocessor-like object: it declares ``input_keys``.

    Detected structurally rather than by import so apairo stays an optional,
    lazily-loaded dependency. Not by ``process``: apairo keeps that name only
    as a deprecated alias of calling the preprocessor.
    """
    return hasattr(step, "input_keys") and (callable(step) or hasattr(step, "process"))


def call_preprocessor(step, sample):
    """Run a preprocessor on *sample*: call it, as apairo does since 0.5.

    An object with only a ``process`` method, the protocol before, still runs.
    """
    return step(sample) if callable(step) else step.process(sample)


def _run_preprocessor(
    step,
    pts: np.ndarray,
    labels: np.ndarray | None,
    frame_idx: int,
) -> np.ndarray | None:
    """Run an apairo ``FramePreprocessor`` step and return its labels output.

    Builds a real :class:`apairo.Sample` carrying only the channels the
    processor declares in ``input_keys`` (mapped from the viewer's ``lidar`` /
    ``labels`` arrays), then returns its output cast to int64 labels.
    """
    from apairo import Sample

    # Stateful processors that need random access expose a settable frame index.
    if hasattr(step, "_idx"):
        step._idx = frame_idx

    data: dict[str, np.ndarray] = {}
    for key in step.input_keys:
        if key == "lidar":
            data["lidar"] = pts
        elif key == "labels" and labels is not None:
            data["labels"] = labels
    result = call_preprocessor(step, Sample(data=data))
    if result is None:
        return labels
    return np.asarray(result, dtype=np.int64)


@dataclass
class Pipeline:
    """Named sequence of per-frame transforms applied before rendering.

    Steps run in order, the output of each feeding the next.  Pass an empty
    ``steps`` list (the default) to display the raw frame untouched.

    One :class:`Pipeline` maps to one viewport.  When several are given to
    :meth:`apairo_visu.LidarViewer.launch`, they run concurrently and each
    viewport refreshes as soon as its pipeline finishes -- handy when inference
    is slow.

    Attributes:
        name: Label shown in the viewport banner and the timing panel.
        steps: Ordered list of transform steps (callables and/or apairo
            ``FramePreprocessor`` objects).

    Examples::

        Pipeline("Raw")                              # no transform
        Pipeline("Range filter", [range_filter])     # preprocessing only
        Pipeline("Model A", [preprocess, model_a])   # preprocess -> inference
        Pipeline("Seg", [MySegPreprocessor()])       # apairo FramePreprocessor
    """

    name: str
    steps: list[Step] = field(default_factory=list)

    def run(
        self,
        pts: np.ndarray,
        labels: np.ndarray | None,
        frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply every step to ``(pts, labels)`` and return the result."""
        for step in self.steps:
            if is_preprocessor(step):
                labels = _run_preprocessor(step, pts, labels, frame_idx)
            else:
                try:
                    pts, labels = step(pts, labels, frame_idx=frame_idx)
                except TypeError:
                    pts, labels = step(pts, labels)
        return pts, labels
