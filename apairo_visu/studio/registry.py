"""Node-id -> live-object registry backing the studio API.

Built once per served session from :func:`apairo_visu.graph.describe`: every
graph node maps back to the object it describes -- the dataset itself for
dataset nodes, ``(owner_dataset, step_index)`` for transform nodes.  All
detail extraction is defensive: a node that cannot be inspected returns what
it can plus an ``error`` field, never an exception (never crash, never show
garbage).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ..graph import GraphSpec, Node, describe

_PREVIEW_INDEX = 0  # sample used to derive the channel table
_CHANNEL_SCAN_CAP = 512  # frames probed to locate channels absent at index 0
_SERIES_CAP = 2000  # frames per series request (the front pages via start/stop)
_TRACK_SCAN_CAP = 5_000_000  # frame_info fallback bound (vector path has none)


def _step_callable(step) -> Any:
    """The object whose doc/signature describes a pipeline step."""
    if hasattr(step, "key") and hasattr(step, "fn"):  # _ChannelStep
        return step.fn
    if hasattr(step, "preprocessor"):  # _PreprocessorStep
        return step.preprocessor
    return step


def _timestamp_of(sample) -> float | None:
    """The sample's timestamp in seconds, JSON-safe (None iff synchronous)."""
    ts = getattr(sample, "timestamp", None)
    try:
        return None if ts is None else float(ts)
    except Exception:
        return None


def _doc_of(obj) -> str | None:
    doc = inspect.getdoc(obj)
    if doc is None and not inspect.isfunction(obj):
        doc = inspect.getdoc(type(obj))
    return doc


def _signature_of(obj) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


class StudioRegistry:
    """Live graph of the served datasets, queryable by node id."""

    def __init__(self, datasets) -> None:
        if not datasets:
            raise ValueError("studio needs at least one dataset")
        self.datasets = list(datasets)  # strong refs for the session lifetime
        self.objects: dict[str, Any] = {}
        self.spec: GraphSpec = describe(*self.datasets, registry=self.objects)
        self._nodes: dict[str, Node] = {n.id: n for n in self.spec.nodes}
        self._channel_cache: dict[str, list[dict]] = {}
        self._track_cache: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------- queries

    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def detail(self, node_id: str) -> dict | None:
        """Everything the inspector shows for one node, JSON-ready."""
        node = self._nodes.get(node_id)
        if node is None:
            return None
        base = {
            "id": node.id,
            "kind": node.kind,
            "label": node.label,
            "params": node.params,
        }
        target = self.objects.get(node_id)
        if target is None:
            return base
        if node.kind == "dataset":
            return {**base, **self._dataset_detail(target)}
        owner, step_index = target
        return {**base, **self._step_detail(owner, step_index)}

    # ------------------------------------------------------------- details

    def _dataset_detail(self, ds) -> dict:
        detail: dict = {"doc": _doc_of(type(ds))}
        try:
            detail["len"] = len(ds)
        except Exception as exc:  # pragma: no cover - defensive
            detail["error"] = f"len() failed: {exc}"
        detail["channels"] = self._channels(ds)
        sequences = self._sequences(ds)
        if sequences:
            detail["sequences"] = sequences
        return detail

    @staticmethod
    def _sequences(ds) -> list[dict] | None:
        """Per-sequence frame ranges ``[{id, start, stop}]``.

        Contiguous runs of the dataset's frame -> sequence-id mapping.
        Views forward ``frame_sequence_ids`` through their index mapping
        (apairo >= 0.6), so this works on the final dataset of a chain --
        e.g. a synchronized root. ``None`` when the dataset carries no
        sequence structure.
        """
        try:
            ids = list(ds.frame_sequence_ids)
        except Exception:
            return None
        runs: list[dict] = []
        for i, sid in enumerate(ids):
            if runs and runs[-1]["id"] == str(sid):
                runs[-1]["stop"] = i + 1
            else:
                runs.append({"id": str(sid), "start": i, "stop": i + 1})
        return runs or None

    def _step_detail(self, owner, step_index: int) -> dict:
        try:
            step = owner._pipeline[step_index]
        except (AttributeError, IndexError) as exc:
            return {"error": f"step no longer present: {exc}"}
        fn = _step_callable(step)
        detail = {
            "owner": type(owner).__name__,
            "step_index": step_index,
            "callable": getattr(fn, "__qualname__", type(fn).__name__),
            "signature": _signature_of(fn),
            "doc": _doc_of(fn),
        }
        try:
            detail["len"] = len(owner)
        except Exception:
            pass
        return detail

    # ------------------------------------------------------------- sampling

    def sample(self, node_id: str, index: int) -> dict | None:
        """Channel arrays of frame *index* as seen AT this node.

        Dataset node: the dataset's own ``__getitem__`` (all its transforms
        applied).  Transform node: the owner's raw sample with the pipeline
        prefix up to and including that step -- intermediate channels are
        visible (``_drop_keys`` is deliberately not applied), that is the
        point of previewing between steps.

        Returns ``None`` for an unknown node; raises ``IndexError`` when the
        index is out of range.
        """
        node = self._nodes.get(node_id)
        target = self.objects.get(node_id)
        if node is None or target is None:
            return None
        if node.kind == "dataset":
            ds = target
            n = len(ds)
            if not 0 <= index < n:
                raise IndexError(f"index {index} out of range [0, {n})")
            sample = ds[index]
            out = {"len": n, "data": sample.data}
            ts = _timestamp_of(sample)
            if ts is not None:
                out["timestamp"] = ts
            # Frame provenance (sequence / source channel / row) when the
            # dataset can name it -- the front shows it next to the index.
            try:
                ref = ds.frame_info(index)
                # Base datasets return a ref with sequence=None -- only a
                # named sequence is worth showing.
                if getattr(ref, "sequence", None) is not None:
                    out["frame"] = {
                        "sequence": ref.sequence,
                        "channel": getattr(ref, "channel", None),
                        "row": getattr(ref, "row", None),
                    }
            except Exception:
                pass
            return out
        else:
            owner, step_index = target
            n = len(owner)
            if not 0 <= index < n:
                raise IndexError(f"index {index} out of range [0, {n})")
            sample = owner._load(index)
            for step in owner._pipeline[: step_index + 1]:
                sample = step(sample)
        out = {"len": n, "data": sample.data}
        ts = _timestamp_of(sample)
        if ts is not None:
            out["timestamp"] = ts
        return out

    def series(
        self,
        node_id: str,
        channel: str,
        cols: list[int],
        start: int = 0,
        stop: int | None = None,
    ) -> dict | None:
        """Per-frame scalar series of one channel over ``[start, stop)``.

        Per frame and per requested index *c*: 1-D frames read ``a[c]``
        (an imu component, a speed), small 2-D frames (matrices like 4x4
        poses) read ``a.flat[c]`` (so a trajectory is cols 3,7,11), and
        per-point ``(N, C)`` frames reduce column *c* to its finite mean.
        Missing channel or out-of-range index yields ``None`` for that
        frame. Ranges are capped at ``_SERIES_CAP`` frames per request
        (``truncated`` flags it; the front pages with start/stop).

        Dataset nodes only -- returns ``None`` for unknown or transform
        nodes. Channel loading is narrowed with ``select`` when the dataset
        supports it, so a lidar+camera frame is not loaded to plot the imu.
        """
        node = self._nodes.get(node_id)
        ds = self.objects.get(node_id)
        if node is None or node.kind != "dataset" or ds is None:
            return None
        n = len(ds)
        start = max(0, start)
        stop = n if stop is None else min(stop, n)
        truncated = stop - start > _SERIES_CAP
        if truncated:
            stop = start + _SERIES_CAP
        src = ds
        try:
            src = ds.select([channel])
        except Exception:
            pass
        values: dict[int, list] = {c: [] for c in cols}
        for i in range(start, stop):
            try:
                frame = np.asarray(src[i].data[channel])
            except Exception:
                for c in cols:
                    values[c].append(None)
                continue
            for c in cols:
                values[c].append(self._reduce(frame, c))
        return {
            "start": start,
            "stop": stop,
            "truncated": truncated,
            "cols": {str(c): values[c] for c in cols},
        }

    def frames(self, node_id: str, channel: str) -> dict | None:
        """Global frame indices whose frame carries *channel* (cached).

        The per-channel timeline of an asynchronous dataset: scrubbing over
        these indices steps through lidar events only, camera events only…
        Derived from ``frame_info``; synchronous datasets (every frame
        carries every channel, ``frame_info().channel is None``) yield an
        empty list -- the global slider already is their timeline. Dataset
        nodes only; ``None`` for unknown or transform nodes.
        """
        node = self._nodes.get(node_id)
        ds = self.objects.get(node_id)
        if node is None or node.kind != "dataset" or ds is None:
            return None
        key = (str(id(ds)), str(channel))
        if key in self._track_cache:
            return self._track_cache[key]
        # Vectorized frame -> channel mapping (apairo >= 0.6), instant even
        # on multi-million-event roots; frame_info scan for older datasets.
        ids = getattr(ds, "frame_channel_ids", None)
        if ids is not None:
            hits = np.nonzero(np.asarray(ids) == str(channel))[0]
            result = {"indices": [int(i) for i in hits], "truncated": False}
        else:
            indices: list[int] = []
            truncated = False
            try:
                n = len(ds)
                truncated = n > _TRACK_SCAN_CAP
                for i in range(min(n, _TRACK_SCAN_CAP)):
                    if getattr(ds.frame_info(i), "channel", None) == channel:
                        indices.append(i)
            except Exception:
                indices, truncated = [], False
            result = {"indices": indices, "truncated": truncated}
        self._track_cache[key] = result
        return result

    @staticmethod
    def _reduce(a: np.ndarray, c: int) -> float | None:
        try:
            if a.ndim == 1:
                val = float(a[c]) if 0 <= c < a.shape[0] else None
            elif a.ndim == 2 and a.shape[0] <= 16:
                val = float(a.flat[c]) if 0 <= c < a.size else None
            elif a.ndim == 2:
                col = a[:, min(c, a.shape[1] - 1)].astype(np.float64)
                col = col[np.isfinite(col)]
                val = float(col.mean()) if col.size else None
            else:
                val = float(np.nanmean(a.astype(np.float64)))
        except Exception:
            return None
        return val if val is not None and np.isfinite(val) else None

    def _channels(self, ds) -> list[dict]:
        """Channel table: name, dtype, shape -- ALL declared channels.

        Reads sample ``_PREVIEW_INDEX`` first (cached). Asynchronous
        timelines carry one channel per frame, so frame 0 alone would hide
        the others (camera next to lidar): declared keys still missing are
        located by scanning ``frame_info`` (cheap index arithmetic) and read
        from the first frame that carries them. Keys not found within the
        scan cap are listed anyway with unknown dtype/shape.
        """
        key = str(id(ds))
        if key in self._channel_cache:
            return self._channel_cache[key]

        rows: list[dict] = []
        seen: set[str] = set()

        def add_from(sample) -> None:
            for name, value in sample.data.items():
                if str(name) in seen:
                    continue
                seen.add(str(name))
                dtype = getattr(value, "dtype", None)
                shape = getattr(value, "shape", None)
                rows.append({
                    "key": str(name),
                    "dtype": str(dtype) if dtype is not None else type(value).__name__,
                    "shape": list(shape) if shape is not None else None,
                })

        try:
            add_from(ds[_PREVIEW_INDEX])
        except Exception as exc:
            self._channel_cache[key] = [{"error": str(exc)}]
            return self._channel_cache[key]

        declared = [str(k) for k in (getattr(ds, "keys", None) or [])]
        missing = [k for k in declared if k not in seen]
        if missing:
            try:
                for idx in range(1, min(len(ds), _CHANNEL_SCAN_CAP)):
                    if not missing:
                        break
                    ch = getattr(ds.frame_info(idx), "channel", None)
                    if ch is not None and str(ch) in missing:
                        add_from(ds[idx])
                        missing = [k for k in declared if k not in seen]
            except Exception:
                pass
        for k in missing:
            rows.append({"key": k, "dtype": "?", "shape": None})

        self._channel_cache[key] = rows
        return rows
