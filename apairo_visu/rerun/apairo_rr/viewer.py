"""Rerun-based LiDAR viewer for apairo datasets."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .colormaps import ColumnColormap
from .images import ImageChannel
from .pipeline import Pipeline

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_label_config(name: str) -> dict:
    """Load a built-in label config by name.

    Args:
        name: One of ``"rellis"``, ``"semantic_kitti"``, ``"goose"``.

    Returns:
        Dict with ``color_map`` (``{class_id: [R, G, B]}``) and
        ``semantic_map`` (``{class_id: label_str}``) keys, ready to pass
        to :func:`view` as ``label_cfg`` or ``label_cfgs``.

    Raises:
        FileNotFoundError: If *name* does not match any built-in config.
    """
    import yaml
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No built-in config '{name}'. Available: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}")
    with path.open() as f:
        return yaml.safe_load(f)


def _normalize_color_map(raw: dict) -> dict[int, list[int]]:
    return {int(k): [int(c) for c in v] for k, v in raw.items()}


_default_colormap = ColumnColormap(2)


def _frame_sequence_ids(dataset):
    """``dataset.frame_sequence_ids`` (per global index) if available, else None.

    apairo's public provenance API — correct global indexing for both
    ``ProfiledDataset`` (Rellis…) and a ``RawDataset`` multi-sequence root, unlike
    ``sequence()._indices`` whose base frame differs between the two.
    """
    try:
        return list(dataset.frame_sequence_ids)
    except Exception:  # noqa: BLE001 -- not an apairo dataset / no keys loaded
        return None


def _channel_frame(dataset, idx: int):
    """``(channel, row)`` for global *idx* via ``frame_info``, or None.

    ``channel`` is ``None`` for synchronous frames (all channels at one row), so
    a per-channel readout only applies when it is a real channel name.
    """
    fi = getattr(dataset, "frame_info", None)
    if fi is None:
        return None
    try:
        ref = fi(idx)
    except Exception:  # noqa: BLE001
        return None
    return (ref.channel, int(ref.row)) if ref.channel is not None else None


def _annotation_context(cfg: dict) -> list[rr.ClassDescription]:
    color_map   = _normalize_color_map(cfg["color_map"])
    semantic_map = {int(k): v for k, v in cfg.get("semantic_map", {}).items()}
    return [
        rr.ClassDescription(info=rr.AnnotationInfo(
            id    = cls_id,
            color = tuple(color_map.get(cls_id, [128, 128, 128])),
            label = semantic_map.get(cls_id, str(cls_id)),
        ))
        for cls_id in sorted(semantic_map.keys())
    ]


def view(
    dataset,
    *,
    label_cfg:  dict | None = None,
    label_cfgs: list[dict | None] | None = None,
    poses:      list[np.ndarray] | None = None,
    pose_key:   str | None = None,
    pipelines:  list[Pipeline] | None = None,
    images:     list[str | ImageChannel] | None = None,
    point_key:  str = "lidar",
    label_key:  str | None = "labels",
    frames:     Iterable[int] | None = None,
    application_id: str = "apairo_rr",
    spawn: bool = True,
    save: str | Path | None = None,
    web: bool = False,
    web_port: int = 9090,
    grpc_port: int = 9876,
    point_radius: float | None = None,
    max_points: int | None = None,
) -> None:
    """Log an apairo dataset to the Rerun viewer.

    Opens the Rerun viewer with one :class:`~rerun.blueprint.Spatial3DView` per
    pipeline, arranged side-by-side.  If the dataset exposes ``sequence_ids``
    (all :class:`~apairo.core.ProfiledDataset` subclasses), a **Sequence**
    text panel is added above the 3D views and updates as you scrub the timeline.

    Args:
        dataset:        Any apairo dataset supporting ``dataset[idx]`` -> ``Sample``.
        label_cfg:      Single label config applied to all pipelines.
        label_cfgs:     Per-pipeline label configs; overrides ``label_cfg``.
                        Use :func:`load_label_config` to obtain built-in configs.
        poses:          List of 4×4 pose matrices (one per dataset frame).
                        Logs a static trajectory and a per-frame robot marker.
                        Mutually exclusive with *pose_key*.
        pose_key:       Sample key containing the per-frame 4×4 pose matrix.
                        Use this when the dataset was enriched with
                        :class:`~apairo_rr.Preprocess` (e.g. ``key="pose"``).
                        Mutually exclusive with *poses*.
        pipelines:      Ordered list of :class:`Pipeline` objects.
                        Defaults to a single ``Pipeline("Raw")`` — unless
                        *images* is given and *pipelines* is left ``None``, in
                        which case it defaults to ``[]`` (images-only) so the
                        viewer never assumes a ``"lidar"`` channel exists.
                        Pass ``pipelines=[Pipeline("Raw")]`` explicitly to show
                        both point clouds and images.
        images:         Image channels to display as 2D views beside the point
                        clouds, stacked vertically and updating along the
                        timeline.  Each item is either a sample-key string or an
                        :class:`~apairo_rr.ImageChannel` (for a custom title or a
                        colormap on a scalar map).  RGB ``img`` channels are
                        logged as-is; use ``ImageChannel(..., colormap=...)`` with
                        :func:`~apairo_rr.colorize` for depth / height / cost
                        maps.  A channel missing from a frame keeps its last value
                        on screen, so async (multi-rate) sensors stay in sync.
        point_key:      Sample key for the point cloud array (default ``"lidar"``).
        label_key:      Sample key for per-point semantic labels (default ``"labels"``).
                        Set to ``None`` to disable label colouring.
        frames:         Iterable of global frame indices to log.
                        Defaults to all frames in *dataset*.
        application_id: Rerun recording name shown in the viewer.
        spawn:          If ``True`` (default), launch the Rerun viewer process.
                        Set to ``False`` when saving to a ``.rrd`` file instead.
                        Ignored when ``web=True`` or ``save`` is set.
        save:           Path to a ``.rrd`` file.  When set, the recording is
                        written to disk instead of opening the viewer.
                        Implies ``spawn=False``.
        web:            If ``True``, serve a web viewer instead of spawning the
                        desktop app.  Any browser on the same local network can
                        open the URL that is printed to the console.
        web_port:       HTTP port for the web viewer (default 9090).
        grpc_port:      gRPC port for the data stream (default 9876).
        point_radius:   World-space radius for all logged point clouds.
                        Overrides the per-case defaults (``ui_points(2.0)`` for
                        labelled clouds, ``0.02`` m otherwise).
        max_points:     If set, randomly subsample each frame to at most this
                        many points before logging.  Reduces Rerun memory usage
                        at the cost of visual density (e.g. ``max_points=5000``).
    """
    image_channels = [
        ic if isinstance(ic, ImageChannel) else ImageChannel(str(ic))
        for ic in (images or [])
    ]

    if pipelines is None:
        # Don't assume a "lidar" channel exists when the caller only asked for
        # images; default to the Raw point cloud only when no images are given.
        pipelines = [] if image_channels else [Pipeline("Raw")]

    n_pipe = len(pipelines)

    # Resolve per-pipeline label configs
    if label_cfgs is not None:
        resolved: list[dict | None] = list(label_cfgs)
        while len(resolved) < n_pipe:
            resolved.append(label_cfg)
    else:
        resolved = [label_cfg] * n_pipe

    # ------------------------------------------------------------------ init
    if web:
        import socket
        rr.init(application_id, spawn=False)
        rr.serve_grpc(grpc_port=grpc_port, cors_allow_origin=["*"])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
            try:
                _s.connect(("10.255.255.255", 1))
                _host_ip = _s.getsockname()[0]
            except OSError:
                _host_ip = "127.0.0.1"
        _grpc_url = f"rerun+http://{_host_ip}:{grpc_port}/proxy"
        rr.serve_web_viewer(connect_to=_grpc_url, web_port=web_port, open_browser=False)
        print(f"Web viewer -> http://{_host_ip}:{web_port}?url={_grpc_url}")
    elif save is not None:
        rr.init(application_id, spawn=False)
        rr.save(str(save))
        print(f"Saving recording to {save} …")
    else:
        rr.init(application_id, spawn=spawn)

    # Build frame->sequence mapping from the public provenance API (global
    # indexing). Only worth a panel when the dataset spans several sequences.
    seq_map: dict[int, str] = {}
    _fsi = _frame_sequence_ids(dataset)
    if _fsi is not None and len({s for s in _fsi if s is not None}) > 1:
        seq_map = {i: s for i, s in enumerate(_fsi) if s is not None}
    elif _fsi is None and hasattr(dataset, "sequence_ids"):
        # Legacy fallback for datasets exposing sequence()/_indices only.
        try:
            for sid in dataset.sequence_ids:
                for i in dataset.sequence(sid)._indices:
                    seq_map[i] = sid
        except Exception:  # noqa: BLE001
            seq_map = {}

    # Per-channel frame index: async (multi-rate) datasets interleave one channel
    # event per global index, so the "frame" timeline alone can't tell you which
    # lidar scan / which camera frame is showing. ``frame_info`` reports the
    # channel + its row — surface it in a "Frames" panel.
    show_frames = len(dataset) > 0 and _channel_frame(dataset, 0) is not None

    # Blueprint: one Spatial3DView per pipeline side-by-side, with image channels
    # stacked in a 2D column to their right.
    views = [
        rrb.Spatial3DView(origin=f"/{pipe.name}", name=pipe.name)
        for pipe in pipelines
    ]
    spatial = rrb.Horizontal(*views) if len(views) > 1 else (views[0] if views else None)

    img_views = [
        rrb.Spatial2DView(origin=f"/images/{ic.key}", name=ic.title)
        for ic in image_channels
    ]
    img_panel = rrb.Vertical(*img_views) if len(img_views) > 1 else (img_views[0] if img_views else None)

    if spatial is not None and img_panel is not None:
        main = rrb.Horizontal(spatial, img_panel, column_shares=[2, 1])
    elif spatial is not None:
        main = spatial
    elif img_panel is not None:
        main = img_panel
    else:
        raise ValueError(
            "Nothing to display: provide at least one pipeline or image channel."
        )

    info_views = []
    if seq_map:
        info_views.append(rrb.TextDocumentView(origin="/info/sequence", name="Sequence"))
    if show_frames:
        info_views.append(rrb.TextDocumentView(origin="/info/frames", name="Frames"))
    if info_views:
        info_bar = rrb.Horizontal(*info_views) if len(info_views) > 1 else info_views[0]
        layout = rrb.Vertical(info_bar, main, row_shares=[1, 12])
    else:
        layout = main
    rr.send_blueprint(rrb.Blueprint(layout, auto_layout=False, auto_views=False))

    # Annotation contexts (static — logged once, apply to all frames)
    for pipe, cfg in zip(pipelines, resolved):
        if cfg is not None:
            rr.log(
                f"/{pipe.name}",
                rr.AnnotationContext(_annotation_context(cfg)),
                static=True,
            )

    # Full trajectory (static)
    if poses is not None:
        positions = np.array([p[:3, 3] for p in poses])
        rr.log(
            "/world/trajectory",
            rr.LineStrips3D([positions], colors=[[80, 140, 255]]),
            static=True,
        )

    # Apply view()-level defaults to pipelines that still carry the generic values.
    # Pipelines that set their own point_key / label_key explicitly are not affected.
    for pipe in pipelines:
        if pipe.point_key == "lidar":
            pipe.point_key = point_key
        if pipe.label_key == "labels":
            pipe.label_key = label_key

    # Pre-compute whether each colormap_fn accepts a sample argument.
    _cm_takes_sample = [
        len(inspect.signature(pipe.colormap_fn).parameters) >= 2
        if pipe.colormap_fn is not None else False
        for pipe in pipelines
    ]

    # ----------------------------------------------------------------- frames
    frame_indices = list(frames) if frames is not None else range(len(dataset))
    n = len(frame_indices)
    _img_note = f" + {len(image_channels)} image channel(s)" if image_channels else ""
    print(f"Logging {n} frames × {n_pipe} pipeline(s){_img_note} …")
    _traj_positions: list[np.ndarray] = []  # incremental trajectory for pose_key
    _per_channel_frame: dict[str, int] = {}  # last per-channel frame index, by channel

    for count, frame_idx in enumerate(frame_indices):
        rr.set_time("frame", sequence=frame_idx)

        sample = dataset[frame_idx]

        # Real time axis (seconds) for async / multi-rate datasets, so the
        # timeline scrubs by actual sensor time rather than just event order.
        # Synchronous samples have no timestamp -> only the "frame" axis.
        _ts = getattr(sample, "timestamp", None)
        if _ts is not None:
            rr.set_time("time", duration=float(_ts))

        # Sequence label
        if seq_map:
            rr.log("/info/sequence", rr.TextDocument(seq_map.get(frame_idx, "?")))

        # Per-channel frame index: the channel(s) updated at this event keep their
        # own counter, so the panel shows e.g. "lidar: 1234 / camera: 567" and each
        # line refreshes only when that sensor ticks.
        if show_frames:
            cr = _channel_frame(dataset, frame_idx)
            if cr is not None:
                _per_channel_frame[cr[0]] = cr[1]
                rr.log("/info/frames", rr.TextDocument(
                    "\n".join(f"{c}: {v}" for c, v in _per_channel_frame.items())
                ))

        # Robot position along trajectory
        if poses is not None and frame_idx < len(poses):
            rr.log("/world/robot", rr.Points3D(
                [poses[frame_idx][:3, 3]],
                radii=0.4,
                colors=[[255, 200, 0]],
            ))
        elif pose_key is not None:
            p = sample.data.get(pose_key)
            if p is not None:
                pos = p[:3, 3]
                _traj_positions.append(pos)
                if len(_traj_positions) > 1:
                    rr.log("/world/trajectory", rr.LineStrips3D(
                        [np.array(_traj_positions)], colors=[[80, 140, 255]]
                    ))
                rr.log("/world/robot", rr.Points3D(
                    [pos], radii=0.4, colors=[[255, 200, 0]]
                ))

        # Per-pipeline point clouds — each pipeline resolves its own keys from sample
        for pipe, cfg, cm_takes_sample in zip(pipelines, resolved, _cm_takes_sample):
            # Async/multi-rate: this sensor didn't update on this frame — keep its
            # last cloud on the timeline (symmetric with missing image channels).
            if pipe.point_key not in sample.data:
                continue
            pts, labels = pipe.run(sample, frame_idx=frame_idx)

            if max_points is not None and len(pts) > max_points:
                idx = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[idx]
                if labels is not None:
                    labels = labels[idx]

            xyz = pts[:, :3].astype(np.float64)

            # Resolve per-point colouring once (class ids from labels, or RGB).
            # class_ids colour from an AnnotationContext when cfg is given
            # (named classes, e.g. semantic segmentation); with no cfg, Rerun
            # still assigns each id a stable colour automatically -- any
            # per-point label works, not just a known semantic class table.
            class_ids = colors = None
            if labels is not None:
                class_ids = labels.astype(np.uint16)
                radius = point_radius if point_radius is not None else rr.Radius.ui_points(2.0)
            elif pipe.colormap_fn is not None:
                colors = np.asarray(
                    pipe.colormap_fn(pts, sample) if cm_takes_sample else pipe.colormap_fn(pts)
                )
                radius = point_radius if point_radius is not None else 0.02
            else:
                colors = np.asarray(_default_colormap(pts))
                radius = point_radius if point_radius is not None else 0.02

            # Drop invalid (NaN/inf) returns. Organized clouds (e.g. Ouster) pad
            # non-returns with NaN xyz, which otherwise blow up the 3D view's
            # auto-bounds (nothing frames) and the colormap's min/max.
            finite = np.isfinite(xyz).all(axis=1)
            if not finite.all():
                xyz = xyz[finite]
                if class_ids is not None:
                    class_ids = class_ids[finite]
                if colors is not None:
                    colors = colors[finite]

            if class_ids is not None:
                rr.log(f"/{pipe.name}/lidar",
                       rr.Points3D(xyz, class_ids=class_ids, radii=radius))
            else:
                rr.log(f"/{pipe.name}/lidar",
                       rr.Points3D(xyz, colors=colors, radii=radius))

        # Image channels — log the latest frame for each; missing channels keep
        # their previous value on the timeline (async sensors stay in sync).
        for ic in image_channels:
            arr = ic.resolve(sample)
            if arr is not None:
                rr.log(f"/images/{ic.key}", rr.Image(arr))

        if (count + 1) % 100 == 0:
            print(f"  {count + 1}/{n}")

    if web:
        print("Done — press Ctrl-C to stop the server.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    elif save is not None:
        print(f"Done — recording saved to {save}.")
    else:
        print("Done — open the Rerun viewer to explore.")
