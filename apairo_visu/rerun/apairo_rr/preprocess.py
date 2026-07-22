"""In-memory preprocess cache for visualization sessions.

Run a stateful preprocessor lazily over a set of frames and expose the results
as a new key on a thin dataset wrapper — no data written to disk.

For FramePreprocessor subclasses, results are computed on first access per
frame (lazy).  For SequencePreprocessor subclasses, all frames are computed
together on first access to any frame, then served from cache.
"""

from __future__ import annotations

import types
from typing import Any, Iterable

import numpy as np

from apairo.core.preprocessor import SequencePreprocessor

_SENTINEL = object()


class _SessionDataset:
    """Thin dataset wrapper that injects a lazily-computed key into each sample.

    Frame-by-frame preprocessors are called once per frame on first access.
    Sequence preprocessors are called once for the whole frame set on first
    access to any frame (they need global context — e.g. the full trajectory).
    All results are cached; subsequent accesses return the cached value.
    """

    def __init__(
        self,
        dataset,
        key: str,
        preprocessor,
        frames: set[int],
        default: Any,
    ) -> None:
        self._dataset = dataset
        self._key = key
        self._preprocessor = preprocessor
        self._frames = frames
        self._cache: dict[int, Any] = {}
        self._default = default
        self._input_keys = getattr(preprocessor, "input_keys", None)
        self._is_sequence = isinstance(preprocessor, SequencePreprocessor)
        self._sequence_computed = False

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        sample = self._dataset[idx]
        if idx in self._frames and idx not in self._cache:
            if self._is_sequence:
                if not self._sequence_computed:
                    self._precompute_all()
            else:
                if self._input_keys is not None:
                    sub = types.SimpleNamespace(
                        data={
                            ik: np.asarray(sample.data[ik])
                            for ik in self._input_keys
                            if ik in sample.data
                        }
                    )
                else:
                    sub = sample
                self._cache[idx] = self._preprocessor.process(sub)
        sample.data[self._key] = self._cache.get(idx, self._default)
        return sample

    def _precompute_all(self) -> None:
        sorted_frames = sorted(self._frames)

        def _make_sub(i: int):
            s = self._dataset[i]
            if self._input_keys is not None:
                return types.SimpleNamespace(data={
                    ik: np.asarray(s.data[ik])
                    for ik in self._input_keys
                    if ik in s.data
                })
            return s

        results = self._preprocessor.process(_make_sub(i) for i in sorted_frames)
        for fi, r in zip(sorted_frames, results):
            self._cache[fi] = r
        self._sequence_computed = True

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)


class Preprocess:
    """Run a stateful preprocessor in memory and expose its output as a dataset key.

    Unlike ``apairo_preprocess`` pipelines that write to disk, this is intended
    for visualization sessions only — results live in RAM and disappear when the
    session ends.

    For :class:`~apairo.core.preprocessor.FramePreprocessor` subclasses,
    computation is **lazy**: the preprocessor is called on first access to each
    frame, so the viewer opens immediately.

    For :class:`~apairo.core.preprocessor.SequencePreprocessor` subclasses,
    all frames are processed together on first access to any frame (the
    algorithm requires global context such as the full trajectory).

    Args:
        preprocessor: A ``FramePreprocessor`` or ``SequencePreprocessor``
                      instance from ``apairo_preprocess``.
        default:      Value injected for frames that were not included in
                      :meth:`run`.  Defaults to a 4×4 identity matrix (the
                      natural fallback for odometry poses).  Pass
                      ``default=None`` to inject ``None`` for non-processed
                      frames (useful for label-producing preprocessors).

    Example — odometry (FramePreprocessor, lazy)::

        ds = Preprocess(GICPOdometry(voxel_size=0.3)).run(ds, seq_frames, key="pose")
        apairo_rr.view(ds, pose_key="pose", ...)

    Example — trajectory traversability (SequencePreprocessor, eager-on-first-access)::

        trav = TraversabilityFromTrajectory(robot_radius=1.0)
        ds = Preprocess(trav, default=None).run(ds, seq_frames, key="trav_traj")
        apairo_rr.view(ds, pipelines=[Pipeline("Trav", label_key="trav_traj")], ...)
    """

    def __init__(self, preprocessor, default: Any = _SENTINEL) -> None:
        self._preprocessor = preprocessor
        self._default = np.eye(4) if default is _SENTINEL else default

    def run(
        self,
        dataset,
        frames: Iterable[int],
        *,
        key: str,
    ) -> _SessionDataset:
        """Register the preprocessor for *frames* and return a wrapped dataset.

        For frame-by-frame preprocessors, no computation happens here.
        For sequence preprocessors, computation is triggered on first access.

        Args:
            dataset: Any apairo dataset.
            frames:  Frame indices to process.  For stateful preprocessors,
                     access them in this order.
            key:     Name of the new channel injected into each sample.

        Returns:
            A :class:`_SessionDataset` wrapping *dataset* with *key* available
            in every ``sample.data``.
        """
        return _SessionDataset(
            dataset, key, self._preprocessor, set(frames), self._default
        )
