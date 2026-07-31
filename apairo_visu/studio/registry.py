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
import os
from typing import Any

import numpy as np

from ..graph import GraphSpec, Node, describe

_PREVIEW_INDEX = 0  # sample used to derive the channel table
_CHANNEL_SCAN_CAP = 512  # frames probed to locate channels absent at index 0
_SERIES_CAP = 2000  # frames per series request (the front pages via start/stop)
_TRACK_SCAN_CAP = 5_000_000  # frame_info fallback bound (vector path has none)
_ENTRIES_CAP = 500  # rows per listing request -- each one costs an os.stat


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


def _is_sidecar(name: str) -> bool:
    """Is this a channel directory's bookkeeping rather than its data?

    A channel folder holds its frames next to a clock (``timestamps.txt``)
    and its manifests. Only what is left over is the data, which is how a
    single-array channel's one file gets identified by elimination.
    """
    return (
        name.startswith(".")
        or name == "timestamps.txt"
        or name.rsplit(".", 1)[-1].lower() in {"yaml", "yml", "json"}
    )


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
        self._stems_cache: dict[str, np.ndarray | None] = {}
        self._file_cache: dict[tuple[str, str], str | None] = {}

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
            # Frame provenance (sequence / source channel / row / on-disk
            # stem) when the dataset can name it -- the front shows it next
            # to the index. The stem is what the user labels by: it names the
            # file (``000850.npy``), which the flat index never does.
            try:
                ref = ds.frame_info(index)
                stem = self._stem(ds, index)
                # Base datasets return a ref with sequence=None -- only a
                # named sequence (or a nameable file) is worth showing.
                if getattr(ref, "sequence", None) is not None or stem is not None:
                    out["frame"] = {
                        "sequence": getattr(ref, "sequence", None),
                        "channel": getattr(ref, "channel", None),
                        "row": getattr(ref, "row", None),
                    }
                    if stem is not None:
                        out["frame"]["stem"] = stem
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

        ``stems`` runs parallel to ``indices``: the on-disk filename stem of
        each event (``"000850"``). That is the name a label file carries, so
        the front can show it while scrubbing and jump straight to it --
        a flat index alone cannot be matched back to a file.
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
        stems = self._frame_stems(ds)
        if stems is not None and result["indices"]:
            try:
                result["stems"] = [str(s) for s in stems[result["indices"]]]
            except Exception:
                pass
        self._track_cache[key] = result
        return result

    def entries(
        self,
        node_id: str,
        channel: str,
        start: int = 0,
        stop: int | None = None,
        sequence: str | None = None,
    ) -> dict | None:
        """One row per frame of *channel*: where it is and what backs it.

        The channel table says a channel exists and what one sample looks
        like; this says what is actually *in* it -- row, sequence, on-disk
        file and its size, timestamp, and the global frame index to seek to.
        That is the view you need to answer "which frames do I still have to
        label", which no single sample can show.

        Paged over the channel's own timeline (``start``/``stop`` count
        frames of the channel, not global indices), capped at
        ``_ENTRIES_CAP`` per request because every row costs an ``os.stat``.
        Optionally narrowed to one *sequence*. File and timestamp are
        best-effort: a view or an exotic loader simply reports ``None`` for
        them rather than failing the listing.
        """
        node = self._nodes.get(node_id)
        ds = self.objects.get(node_id)
        if node is None or node.kind != "dataset" or ds is None:
            return None
        track = self.frames(node_id, channel)
        indices: list[int] = list(track["indices"]) if track else []
        stems = (track or {}).get("stems")
        # A synchronous dataset has no per-channel timeline: every frame
        # carries the channel, so the listing is the whole frame axis.
        if not indices:
            try:
                indices = list(range(len(ds)))
            except Exception:
                indices = []
            stems = None
        seq_ids = getattr(ds, "frame_sequence_ids", None)
        seq_ids = None if seq_ids is None else np.asarray(seq_ids)
        if sequence is not None and seq_ids is not None:
            keep = [k for k, i in enumerate(indices) if str(seq_ids[i]) == sequence]
            indices = [indices[k] for k in keep]
            stems = None if stems is None else [stems[k] for k in keep]

        total = len(indices)
        start = max(0, min(start, total))
        stop = total if stop is None else min(stop, total)
        truncated = stop - start > _ENTRIES_CAP
        if truncated:
            stop = start + _ENTRIES_CAP

        rows: list[dict] = []
        for k in range(start, stop):
            index = int(indices[k])
            entry: dict = {"index": index, "pos": k}
            if stems is not None:
                entry["stem"] = stems[k]
            try:
                ref = ds.frame_info(index)
                seq = getattr(ref, "sequence", None)
                row = getattr(ref, "row", None)
                entry["sequence"] = None if seq is None else str(seq)
                entry["row"] = None if row is None else int(row)
                owner = self._owner(ds, entry["sequence"])
                if owner is not None and entry["row"] is not None:
                    name, size, shared = self._file_of(owner, channel, entry["row"])
                    entry["file"] = name
                    entry["bytes"] = size
                    if shared:
                        entry["shared"] = True
                    entry["timestamp"] = self._stamp_of(owner, channel, entry["row"])
            except Exception:
                pass
            rows.append(entry)
        return {
            "total": total,
            "start": start,
            "stop": stop,
            "truncated": truncated,
            "entries": rows,
        }

    @staticmethod
    def _owner(ds, sequence: str | None):
        """The sequence dataset that actually holds the files of *sequence*.

        A root delegates to its sub-datasets, which are the ones carrying
        ``loaders`` / ``timestamps``; a single-sequence dataset is its own
        owner. ``None`` when the id matches nothing.
        """
        ids = getattr(ds, "sequence_ids", None)
        seqs = getattr(ds, "sequences", None)
        if not ids or not seqs:
            return ds
        for sid, sub in zip(ids, seqs):
            if str(sid) == str(sequence):
                return sub
        return None

    def _file_of(self, owner, channel: str, row: int) -> tuple:
        """``(filename, bytes, shared)`` backing one row of a channel.

        Two on-disk shapes, and both deserve a name. A per-frame loader
        (``npys``, ``bin``, images) gives one file per row and its size on
        disk. A single-array loader (``npy``: one file holding the whole
        channel) gives that one file for every row, flagged *shared*, with
        the row's own byte count rather than the whole array's -- saying
        "no file" there would read as "this frame has no home on disk",
        which is exactly wrong. ``(None, None, False)`` when nothing can be
        named at all: a view, a zarr store, a transform node.
        """
        loader = (getattr(owner, "loaders", None) or {}).get(channel)
        files = getattr(loader, "files", None)
        directory = getattr(loader, "directory", None)
        if files and directory is not None and 0 <= row < len(files):
            name = str(files[row])
            try:
                return name, os.stat(os.path.join(str(directory), name)).st_size, False
            except OSError:
                return name, None, False

        name = self._channel_file(owner, channel)
        if name is None:
            return None, None, False
        array = getattr(loader, "array", None)
        try:
            size = int(array[row].nbytes) if array is not None and row < len(array) else None
        except Exception:
            size = None
        return name, size, True

    def _channel_file(self, owner, channel: str) -> str | None:
        """The lone data file of a channel directory, cached per channel.

        Mirrors what the ``npy`` loader itself does -- it keeps the array
        and forgets the path it came from, so the name has to be recovered
        from the layout. ``None`` when the directory holds more than one
        data file, since then no single name is *the* file.
        """
        key = (str(id(owner)), str(channel))
        if key in self._file_cache:
            return self._file_cache[key]
        directory = (getattr(owner, "_files", None) or {}).get(channel)
        name = None
        if directory is not None:
            try:
                names = sorted(
                    e.name for e in os.scandir(str(directory))
                    if e.is_file() and not _is_sidecar(e.name)
                )
                name = names[0] if len(names) == 1 else None
            except OSError:
                name = None
        self._file_cache[key] = name
        return name

    @staticmethod
    def _stamp_of(owner, channel: str, row: int) -> float | None:
        stamps = (getattr(owner, "timestamps", None) or {}).get(channel)
        try:
            return float(stamps[row]) if stamps is not None else None
        except Exception:
            return None

    def locate(
        self,
        node_id: str,
        stem: str,
        channel: str | None = None,
        sequence: str | None = None,
    ) -> dict | None:
        """Global frame indices whose on-disk file stem is *stem*.

        The inverse of the flat index: a label lives in ``000850.npy``, and
        this is what turns that name back into something the frame slider
        understands. Narrow with *channel* (asynchronous timelines interleave
        one stem sequence per channel) and *sequence* (every sequence of a
        root restarts its numbering at ``000000``), otherwise every match is
        returned in order, each tagged with where it came from.

        Dataset nodes only; ``None`` for unknown or transform nodes.
        """
        node = self._nodes.get(node_id)
        ds = self.objects.get(node_id)
        if node is None or node.kind != "dataset" or ds is None:
            return None
        stems = self._frame_stems(ds)
        if stems is None:
            return {"matches": [], "error": "this dataset carries no file stems"}
        hits = np.nonzero(stems == str(stem))[0]
        seq_ids = getattr(ds, "frame_sequence_ids", None)
        chan_ids = getattr(ds, "frame_channel_ids", None)
        seq_ids = None if seq_ids is None else np.asarray(seq_ids)
        chan_ids = None if chan_ids is None else np.asarray(chan_ids)
        matches: list[dict] = []
        for i in hits:
            seq = None if seq_ids is None else str(seq_ids[i])
            chan = None if chan_ids is None else str(chan_ids[i])
            if channel is not None and chan is not None and chan != channel:
                continue
            if sequence is not None and seq is not None and seq != sequence:
                continue
            matches.append({"index": int(i), "sequence": seq, "channel": chan})
        return {"matches": matches}

    def _frame_stems(self, ds) -> np.ndarray | None:
        """Filename stem per global frame, materialized once per dataset.

        ``frame_stems`` is a *property*: on a root it re-concatenates every
        sequence's table on each access -- ~0.7 s over a 1.8 M-event root,
        which a per-frame request cannot pay. Frames are immutable while
        serving, so the array is cached like the channel and track tables.
        ``None`` when the dataset exposes no stems.
        """
        key = str(id(ds))
        if key not in self._stems_cache:
            stems = getattr(ds, "frame_stems", None)
            try:
                self._stems_cache[key] = None if stems is None else np.asarray(stems)
            except Exception:
                self._stems_cache[key] = None
        return self._stems_cache[key]

    def _stem(self, ds, index: int) -> str | None:
        """On-disk filename stem of global frame *index* (``None`` if none)."""
        stems = self._frame_stems(ds)
        if stems is None:
            return None
        try:
            return str(stems[index])
        except Exception:
            return None

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
