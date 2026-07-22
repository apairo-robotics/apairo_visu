"""``apairo studio`` -- ecosystem CLI plugin.

Registered under the ``apairo.cli_plugins`` entry-point group, so the core
``apairo`` command dispatches ``apairo studio <args>`` here (see
``apairo.cli``); the plugin parses its own arguments.

Examples::

    apairo studio /data/barakuda_kitti
    apairo studio /data/barakuda_kitti --keys lidar labels --port 9000
    apairo studio /data/barakuda_kitti --sync lidar --tolerance 0.05
    apairo studio . --no-browser --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apairo studio",
        description=(
            "Serve the interactive pipeline studio (browser UI) for a "
            "dataset directory."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="dataset directory, sequence or root (default: .)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        default=None,
        help="channels to load (default: every channel in channels.yaml)",
    )
    parser.add_argument(
        "--sync",
        metavar="KEY",
        default=None,
        help="synchronize on this reference channel (method: nearest) so "
        "every frame carries all channels; without it an asynchronous "
        "dataset serves its raw event timeline (one channel per frame)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="max timestamp distance in seconds for --sync matches",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8710, help="HTTP port (default: 8710)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the default browser on the studio URL",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=150_000,
        help="decimate per-point channels above this many rows "
        "before sending them to the browser (default: 150000)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="page title (default: 'apairo studio -- <directory name>')",
    )
    return parser


def _declared_keys(root: Path) -> list[str]:
    """Channel keys declared under *root*, read straight from the ``.apairo``
    manifests without building any loader -- so a malformed channel cannot raise
    here. Handles both a single sequence (its own ``channels.yaml``) and a root
    (the union of channels across its sequence sub-directories)."""
    import yaml

    def keys_of(directory: Path) -> set[str]:
        manifest = directory / ".apairo" / "channels.yaml"
        if not manifest.is_file():
            return set()
        data = yaml.safe_load(manifest.read_text()) or {}
        return set((data.get("channels") or {}).keys())

    own = keys_of(root)
    if own:
        return sorted(own)
    union: set[str] = set()
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            union |= keys_of(sub)
    return sorted(union)


def _load_tolerant(
    apairo, root: Path, keys: list[str] | None
) -> tuple[object, list[tuple[str, str]]]:
    """Build a ``RawDataset`` over *root*, skipping channels that fail to load
    instead of aborting the whole studio.

    Returns ``(dataset, dropped)`` where *dropped* lists ``(key, reason)`` for
    every channel left out. The requested set (explicit ``--keys`` or, by
    default, every declared channel) is loaded as one call first; only if that
    fails is each channel probed in isolation, so a single malformed channel --
    a missing ``timestamps.txt``, an incomplete suffix, a stray directory --
    no longer takes the whole viewer down with it."""
    try:
        return apairo.RawDataset(str(root), keys=keys), []
    except Exception as combined_error:
        requested = keys if keys is not None else _declared_keys(root)
        if not requested:
            raise combined_error
        print(
            "apairo studio: loading every channel at once failed; probing "
            "channels individually to serve the ones that load...",
            file=sys.stderr,
        )

    good: list[str] = []
    dropped: list[tuple[str, str]] = []
    for key in requested:
        try:
            apairo.RawDataset(str(root), keys=[key])
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            dropped.append((key, f"{type(exc).__name__}: {reason}"))
        else:
            good.append(key)

    if not good:
        raise RuntimeError(
            f"no channel under {root} could be loaded; run `apairo status "
            f"{root}` to inspect the dataset"
        )
    return apairo.RawDataset(str(root), keys=good), dropped


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"apairo studio: not a directory: {root}", file=sys.stderr)
        return 1

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "apairo studio requires the 'studio' extra: "
            "pip install apairo_visu[studio]",
            file=sys.stderr,
        )
        return 1

    import apairo

    try:
        dataset, dropped = _load_tolerant(apairo, root, args.keys)
    except Exception as exc:
        print(f"apairo studio: {exc}", file=sys.stderr)
        return 1

    if dropped:
        print(
            f"apairo studio: skipped {len(dropped)} channel(s) that failed to "
            f"load (the rest are served):",
            file=sys.stderr,
        )
        for key, reason in dropped:
            print(f"  - {key}: {reason}", file=sys.stderr)

    if args.sync and args.sync in dict(dropped):
        print(
            f"apairo studio: cannot synchronize on '{args.sync}' -- that "
            f"channel failed to load (see above)",
            file=sys.stderr,
        )
        return 1

    if args.sync:
        dataset = dataset.synchronize(
            reference=args.sync, method="nearest", tolerance=args.tolerance
        )

    from . import launch

    launch(
        dataset,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        title=args.title or f"apairo studio -- {root.resolve().name}",
        max_points=args.max_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
