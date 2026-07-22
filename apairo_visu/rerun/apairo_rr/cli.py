"""apairo_rr command-line interface.

Exposed two ways, both calling :func:`main`:

* ``apairo rerun ...`` -- ecosystem subcommand discovered by the apairo CLI
  through the ``apairo.cli_plugins`` entry-point group.  apairo never depends on
  apairo_rr; it dispatches to whatever is installed (same mechanism as
  ``apairo extractor``).
* ``apairo-rerun ...``  -- standalone console script.

Replay a sequence and watch the chosen sensors evolve along the timeline: one 3D
view per ``--lidar`` channel, one 2D view per ``--camera`` channel, side by side.

    apairo rerun /path/to/ds --lidar ouster_points --camera zed_rgb
    apairo rerun /path/to/ds --lidar ouster_points --color height \\
        --lidar-frame os_sensor --base-frame base_link
    apairo rerun /path/to/ds --lidar ouster_points --color height \\
        --lidar-frame os_sensor --base-frame base_link --pose dlio_odom_node_odom
    apairo rerun /path/to/ds --lidar velodyne_0 --labels labels \\
        --label-config semantic_kitti
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import apairo

from apairo_rr import ColumnColormap, ImageChannel, Pipeline, load_label_config, red_blue, view

# --label-config default per --as: the profiled datasets each have one
# canonical class table; RawDataset/TartanKittiDataset have none, so --labels
# on those needs an explicit --label-config.
_LABEL_CONFIG_BY_CLASS = {
    "Rellis3DDataset": "rellis",
    "SemanticKittiDataset": "semantic_kitti",
    "Goose3DDataset": "goose",
}

# Dataset classes selectable with ``--as``: the profile-free generic loader plus
# the profiled datasets whose layout maps canonical channel names from a profile.
_DATASETS = {
    name: getattr(apairo, name)
    for name in (
        "RawDataset", "TartanKittiDataset",
        "SemanticKittiDataset", "Rellis3DDataset", "Goose3DDataset",
    )
    if hasattr(apairo, name)
}

_COLORS = ("flat", "height", "range")


def _split_keys(value: Optional[str]) -> list[str]:
    """Parse a ``a,b,c`` channel list into stripped, non-empty keys."""
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]


def _resolve_pos(spec, n: int, default: int) -> int:
    """Resolve a position *spec* into an index in ``[0, n]``.

    Convention: an **integer** is a frame position (``100``; negative counts from
    the end), a **float in ``[0, 1]``** is a fraction of the *n* candidate frames
    (``0.5`` → halfway). ``None`` → *default*. Raises ``ValueError`` on garbage.
    """
    if spec is None:
        return default
    s = str(spec).strip()
    try:
        pos = int(s)                       # integer literal -> frame position
        if pos < 0:
            pos += n
    except ValueError:
        f = float(s)                       # float literal (raises on garbage)
        if 0.0 <= f <= 1.0:
            pos = round(f * n)             # fraction of the sequence
        else:
            pos = int(f)                   # float outside [0, 1]: positional
    return max(0, min(pos, n))


def _alias_map(path: Path) -> dict[str, str]:
    """Map on-disk channel name -> exposed alias, from ``.apairo/channels.yaml``.

    Channels can be aliased (e.g. ``ouster_points`` -> ``lidar``); a sample then
    exposes the data under the *alias*, not the directory name, so the names
    passed to ``--lidar`` / ``--camera`` / ``--pose`` must be resolved to it
    before they are used as sample keys -- otherwise every frame is silently
    skipped and nothing is logged.
    """
    try:
        from apairo.core.config import config_exists, read_config
    except Exception:
        return {}
    roots = [path]
    if not config_exists(path):  # root of sequences: peek the first with a config
        roots = [d for d in sorted(path.iterdir())
                 if d.is_dir() and config_exists(d)][:1]
    amap: dict[str, str] = {}
    for r in roots:
        if config_exists(r):
            for name, meta in read_config(r).get("channels", {}).items():
                if meta.get("alias"):
                    amap[name] = meta["alias"]
    return amap


# Conventional frame names used to auto-resolve the lidar mount TF. The cloud's
# own frame is rarely recorded in channel metadata, so we match a known sensor
# frame present in the calibration tree against a body frame (preferring
# base_link). Order = preference; first exact match in the tree wins.
_LIDAR_FRAME_HINTS = ("os_sensor", "velodyne", "hesai", "rslidar", "livox", "os_lidar")
_BASE_FRAME_HINTS = ("base_link", "base_footprint")


def _calib_nodes(cal) -> set[str]:
    """Frame names (graph nodes) in a Calibration, parsed from its ``A_to_B`` edges."""
    nodes: set[str] = set()
    for edge in cal.keys():
        head, sep, tail = str(edge).partition("_to_")
        if sep:
            nodes.add(head)
            nodes.add(tail)
    return nodes


def _auto_mount(ds):
    """Best-effort static mount TF ``<lidar frame> -> <base frame>``.

    Most rigs mount the lidar tilted, so a raw scan "looks up" instead of forward.
    Returns ``(lidar_frame, base_frame, T)`` or ``None`` when no sensible pair is
    found (e.g. no calibration, or no base_link) — in which case the cloud is left
    in its native frame.
    """
    cal = getattr(ds, "calibration", None)
    if not cal:
        return None
    try:
        nodes = _calib_nodes(cal)
    except Exception:  # noqa: BLE001 -- never let auto-resolution break the viewer
        return None
    base = next((b for b in _BASE_FRAME_HINTS if b in nodes), None)
    if base is None:
        return None
    for lf in _LIDAR_FRAME_HINTS:
        if lf in nodes and lf != base:
            try:
                return lf, base, np.asarray(cal.get_tf(lf, base), dtype=np.float64)
            except Exception:  # noqa: BLE001 -- no path: try the next candidate
                continue
    return None


def _make_colormap(mode: str):
    """Per-point colouring callable for ``Pipeline.colormap_fn`` (``pts -> (N,3)``)."""
    if mode == "height":
        return ColumnColormap(2)  # colour by z (needs the cloud upright/world — see --pose)
    if mode == "range":
        return lambda pts: red_blue(np.linalg.norm(pts[:, :3], axis=1))
    grey = np.array([190, 192, 196], np.uint8)  # flat: one neutral colour, no scalar
    return lambda pts: np.repeat(grey[None, :], len(pts), axis=0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apairo rerun",
        description="Replay an apairo dataset in Rerun: 3D point clouds and image "
                    "channels updating together along the timeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="dataset directory")
    p.add_argument("--sequence", "-s", metavar="ID", default=None,
                   help="restrict to one sequence id (default: the whole dataset)")
    p.add_argument("--lidar", metavar="A,B", default=None,
                   help="point-cloud channel(s) to show as 3D views "
                        "(comma-separated; on-disk names — aliases are resolved)")
    p.add_argument("--camera", metavar="A,B", default=None,
                   help="image channel(s) to show as 2D views (comma-separated)")
    p.add_argument("--labels", metavar="A,B", default=None,
                   help="labelled point channel(s) (e.g. ground-truth semantic "
                        "labels) -- coloured onto the matching --lidar channel's "
                        "points, not shown as a view of their own. Paired by "
                        "position with --lidar, or one channel shared by all "
                        "--lidar channels. These are typically annotated on only "
                        "a subset of lidar frames -- the viewer replays just "
                        "that subset.")
    p.add_argument("--label-config", choices=("rellis", "semantic_kitti", "goose"),
                   default=None, dest="label_config",
                   help="optional class colour/name legend for --labels, for a "
                        "known semantic table (default: inferred from --as, else "
                        "none -- any label id still colours fine without one)")
    p.add_argument("--color", choices=_COLORS, default="flat",
                   help="point colouring: flat (default), height (z), or range. "
                        "height needs the cloud upright/world — see --lidar-frame / --pose")
    p.add_argument("--lidar-frame", default=None, dest="lidar_frame",
                   help="frame the raw clouds live in (e.g. os_sensor); with "
                        "--base-frame, applies the static mount TF to view upright")
    p.add_argument("--base-frame", default=None, dest="base_frame",
                   help="upright frame to lift the clouds into (e.g. base_link)")
    p.add_argument("--raw", action="store_true",
                   help="show clouds in their native sensor frame (disable the "
                        "automatic upright mount TF)")
    p.add_argument("--pose", default=None,
                   help="pose / odometry channel — lift the clouds into the world "
                        "frame per frame and draw the trajectory")
    p.add_argument("--range", type=float, default=None, dest="range_max",
                   help="drop points farther than this many metres before logging")
    p.add_argument("--as", dest="as_", metavar="CLASS", choices=list(_DATASETS),
                   default="RawDataset",
                   help="dataset class to load with (default: RawDataset)")
    p.add_argument("--every", type=int, default=1, help="log every Nth frame")
    p.add_argument("--start", default=None,
                   help="first frame: an integer position (e.g. 100) or a fraction "
                        "of the sequence as a float in [0,1] (e.g. 0.5). Overrides --idx.")
    p.add_argument("--end", default=None,
                   help="last frame (exclusive): an integer position or a fraction "
                        "(e.g. 0.8). Default: end of the sequence.")
    p.add_argument("--idx", type=int, default=0,
                   help="first frame position (legacy alias of --start)")
    p.add_argument("--max-points", type=int, default=None, dest="max_points",
                   help="subsample each cloud to at most this many points")
    p.add_argument("--point-radius", type=float, default=None, dest="point_radius",
                   help="world-space radius for all logged points")
    p.add_argument("--save", default=None, metavar="FILE.rrd",
                   help="write a .rrd file instead of spawning the viewer")
    p.add_argument("--web", action="store_true",
                   help="serve a web viewer instead of spawning the desktop app")
    p.add_argument("--application-id", default="apairo_rr", dest="application_id",
                   help="Rerun recording name shown in the viewer")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2

    lidar_req = _split_keys(args.lidar)
    camera_req = _split_keys(args.camera)
    labels_req = _split_keys(args.labels)
    cls = _DATASETS[args.as_]

    if labels_req and len(labels_req) not in (1, len(lidar_req)):
        print(f"--labels must give one channel (shared by every --lidar channel) "
              f"or exactly as many as --lidar ({len(lidar_req)}); got "
              f"{len(labels_req)}.", file=sys.stderr)
        return 1

    # --sequence naming an actual sub-directory: load *that* sequence directly
    # instead of the whole root. A root only exposes the channel intersection
    # across every sequence (see RootSequenceMixin.available), so a channel
    # that exists in just a few sequences (e.g. a partial ground-truth label
    # set) would otherwise make the root load fail outright before we ever get
    # to filter down to the requested sequence.
    single_sequence = False
    load_path = path
    if args.sequence is not None and (path / args.sequence).is_dir():
        load_path = path / args.sequence
        single_sequence = True

    # No channels requested: load to discover the available keys and list them.
    if not lidar_req and not camera_req:
        try:
            ds = cls(str(load_path))
        except Exception as exc:  # noqa: BLE001 -- surface the loader's own message
            print(f"Could not load dataset: {exc}", file=sys.stderr)
            return 1
        avail = sorted(getattr(ds, "available_keys", []) or ds[0].data.keys())
        print("Pass --lidar and/or --camera with one of these channels:")
        print(f"  {', '.join(avail) if avail else '(none found)'}")
        return 2

    # Resolve aliases: --lidar ouster_points is exposed under its alias "lidar".
    amap = _alias_map(load_path)
    def exposed(k: str) -> str:
        return amap.get(k, k)

    pose_req = args.pose
    keys = list(dict.fromkeys(
        lidar_req + camera_req + labels_req + ([pose_req] if pose_req else [])
    ))
    try:
        ds = cls(str(load_path), keys=keys)
    except Exception as exc:  # noqa: BLE001 -- surface the loader's own message
        print(f"Could not load dataset: {exc}", file=sys.stderr)
        return 1

    lidar_x = [exposed(k) for k in lidar_req]
    camera_x = [exposed(k) for k in camera_req]
    labels_x = [exposed(k) for k in labels_req]
    pose_x = exposed(pose_req) if pose_req else None

    # Pair each --lidar channel with the --labels channel to colour its points
    # from -- one channel shared by all, or matched 1:1 by position.
    label_for: dict[str, str] = {}
    label_cfg = None
    if labels_x:
        pairs = labels_x if len(labels_x) == len(lidar_x) else labels_x * len(lidar_x)
        label_for = dict(zip(lidar_x, pairs))
        # --label-config is only a legend (class names + fixed colours) for a
        # known semantic table; any label id colours fine without one -- Rerun
        # auto-assigns a stable colour per id.
        label_cfg_name = args.label_config or _LABEL_CONFIG_BY_CLASS.get(args.as_)
        if label_cfg_name is not None:
            try:
                label_cfg = load_label_config(label_cfg_name)
            except FileNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                return 1

    # Static mount TF (tilted sensor frame -> upright body frame). Explicit
    # --lidar-frame/--base-frame win; otherwise resolve it automatically so the
    # cloud faces forward by default instead of "looking up". --raw keeps the
    # native sensor frame.
    mount = None
    mount_desc = None
    if args.lidar_frame and args.base_frame:
        try:
            mount = np.asarray(
                ds.calibration.get_tf(args.lidar_frame, args.base_frame), dtype=np.float64
            )
            mount_desc = f"{args.lidar_frame}->{args.base_frame}"
        except Exception as exc:  # noqa: BLE001
            print(f"Could not resolve TF {args.lidar_frame} -> {args.base_frame}: {exc}",
                  file=sys.stderr)
            return 1
    elif lidar_x and not args.raw:
        auto = _auto_mount(ds)
        if auto is not None:
            lf, bf, mount = auto
            mount_desc = f"{lf}->{bf} (auto — --raw to keep sensor frame)"

    if args.color == "height" and mount is None and not pose_x:
        print("warning: --color height without an upright mount TF or --pose: "
              "z is the raw (possibly tilted) sensor axis, not true height. "
              "Pass --lidar-frame/--base-frame.", file=sys.stderr)

    if (mount is not None or args.range_max is not None) or pose_x:
        # apairo_transform supplies the rigid-body / range / pose primitives.
        try:
            from apairo_transform import ApplyPose, PoseTo4x4, RangeFilter, TransformPoints
        except ImportError:
            # An auto-resolved mount is a best-effort nicety — degrade to the raw
            # frame rather than failing. Only an *explicit* request is fatal.
            if args.lidar_frame or args.pose or args.range_max is not None:
                print("--lidar-frame/--base-frame, --pose and --range need apairo_transform "
                      "(pip install apairo-transform).", file=sys.stderr)
                return 1
            print("note: apairo_transform not installed — showing clouds in their native "
                  "(tilted) sensor frame; `pip install apairo-transform` for the upright "
                  "mount TF.", file=sys.stderr)
            mount = None
            mount_desc = None

    # Frame window: --start/--end take an integer position or a float fraction
    # in [0,1] of the sequence's frames; --idx is the legacy start; --every strides.
    start_spec = args.start if args.start is not None else args.idx

    def _valid(spec) -> bool:
        if spec is None:
            return True
        try:
            float(str(spec).strip())
            return True
        except ValueError:
            return False

    if not (_valid(start_spec) and _valid(args.end)):
        print("Invalid --start/--end: use an integer position (e.g. 100) or a "
              "fraction in [0,1] (e.g. 0.5).", file=sys.stderr)
        return 1

    def _pick(base: list[int]) -> list[int]:
        n = len(base)
        lo = _resolve_pos(start_spec, n, 0)
        hi = _resolve_pos(args.end, n, n)
        return base[lo:max(lo, hi):args.every]

    # ----------------------------------------------------------------- transforms
    labels_synced = False
    if labels_x:
        # Ground-truth label channels are typically annotated on only a subset
        # of lidar scans and, being async, carry no xyz of their own -- resample
        # onto the *label* clock so each labelled tick's nearest lidar scan lands
        # in the same sample (Pipeline.run then colours those points straight
        # from the paired label channel) and every un-annotated lidar frame is
        # dropped rather than shown without labels.
        ref = labels_x[0]
        others = [k for k in lidar_x + camera_x + labels_x[1:]
                  + ([pose_x] if pose_x else []) if k != ref]
        try:
            ds = ds.synchronize(reference=ref, method={k: "nearest" for k in others})
            labels_synced = True
        except ValueError as exc:
            # Already-synchronous dataset (e.g. a profiled dataset): labels are
            # already one dense array per frame, nothing to resample onto.
            if "already synchronous" not in str(exc):
                raise

    if labels_synced:
        if pose_x:
            ds = ds.transform(pose_x, PoseTo4x4())
        for lk in lidar_x:
            if args.range_max is not None:
                ds = ds.transform(lk, RangeFilter(max=args.range_max))
            if mount is not None:
                ds = ds.transform(lk, TransformPoints(mount))
            if pose_x:
                ds = ds.transform(ApplyPose(lidar=lk, pose=pose_x))
        frames = _pick(list(range(len(ds))))
    elif pose_x:
        if not lidar_x:
            print("--pose lifts point clouds into the world frame — pass --lidar too.",
                  file=sys.stderr)
            return 1
        ref = lidar_x[0]
        others = [k for k in lidar_x + camera_x + [pose_x] if k != ref]
        ds = ds.synchronize(reference=ref, method={k: "nearest" for k in others})
        ds = ds.transform(pose_x, PoseTo4x4())  # pose channel -> 4x4 (passthrough if already)
        for lk in lidar_x:
            if args.range_max is not None:
                ds = ds.transform(lk, RangeFilter(max=args.range_max))
            if mount is not None:
                ds = ds.transform(lk, TransformPoints(mount))
            ds = ds.transform(ApplyPose(lidar=lk, pose=pose_x))  # dynamic pose -> world frame
        frames = _pick(list(range(len(ds))))
    else:
        for lk in lidar_x:
            if args.range_max is not None:
                ds = ds.transform(lk, RangeFilter(max=args.range_max))
            if mount is not None:
                ds = ds.transform(lk, TransformPoints(mount))
        if single_sequence:
            # Already loaded from the sequence's own directory (see load_path
            # above) -- ds *is* the requested sequence, nothing left to filter.
            base = list(range(len(ds)))
        else:
            seq_ids = list(getattr(ds, "sequence_ids", []) or [])
            if args.sequence is not None:
                if args.sequence not in seq_ids:
                    avail = f" Available: {', '.join(seq_ids)}." if seq_ids else ""
                    print(f"Sequence '{args.sequence}' not found.{avail}", file=sys.stderr)
                    return 1
                # Global frame indices for the sequence via the public provenance
                # API. ``sequence()._indices`` is local on a RawDataset root (base
                # frame = the sub-dataset), so feeding it as global indices showed
                # the wrong sequences — frame_sequence_ids is correctly global on
                # root and profiled.
                fsi = np.asarray(ds.frame_sequence_ids)
                base = np.flatnonzero(fsi == args.sequence).tolist()
            else:
                base = list(range(len(ds)))
        frames = _pick(base)

    print(f"  {len(ds)} frames"
          f"\n  lidar: {lidar_x or '-'}   camera: {camera_x or '-'}"
          f"   colour: {args.color}"
          + (f"   labels: {label_for}" if label_for else "")
          + (f"   pose: {pose_x}" if pose_x else "")
          + (f"   mount: {mount_desc}" if mount is not None else "")
          + f"\n  {len(frames)} frames to log"
          + (f" (global {frames[0]}–{frames[-1]})" if frames else ""))

    colormap_fn = _make_colormap(args.color)
    pipelines = [
        Pipeline(lk, point_key=lk, label_key=label_for.get(lk), colormap_fn=colormap_fn)
        for lk in lidar_x
    ]
    label_cfgs = [label_cfg if lk in label_for else None for lk in lidar_x]
    images = [ImageChannel(ck) for ck in camera_x]

    view(
        ds,
        pipelines=pipelines,
        label_cfgs=label_cfgs,
        images=images,
        frames=frames,
        pose_key=pose_x,
        application_id=args.application_id,
        save=args.save,
        web=args.web,
        spawn=args.save is None and not args.web,
        max_points=args.max_points,
        point_radius=args.point_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
