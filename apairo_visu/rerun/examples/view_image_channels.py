"""Watch image channels evolve over time, beside the LiDAR point cloud.

apairo_rr can now display 2D image channels as Rerun 2D views that update along
the timeline, so you can replay a sequence and see the cameras (and any other
image-like sensor) evolve frame by frame next to the point cloud.

This example uses the TartanDrive layout (``TartanKittiDataset``), which carries
several camera channels (``image_left_color``, ``image_right``) plus a depth map
(``depth_left``).  RGB channels are logged as-is; the depth map is colourised on
the fly with :func:`apairo_rr.colorize`.

Pass ``--no-lidar`` to view the image channels alone (no point cloud assumed).

Usage::

    python examples/view_image_channels.py ~/data/tartan_kitti
    python examples/view_image_channels.py ~/data/tartan_kitti --every 5
    python examples/view_image_channels.py ~/data/tartan_kitti \
        --images image_left_color image_right
    python examples/view_image_channels.py ~/data/tartan_kitti --no-lidar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import apairo
import apairo_rr
from apairo_rr import ImageChannel, Pipeline, colorize

_DEFAULT_ROOT = Path.home() / "data" / "tartan_kitti"
_DEFAULT_IMAGES = ["image_left_color", "image_right"]
_POINT_KEY = "velodyne_0"
_DEPTH_KEY = "depth_left"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--every", type=int, default=1, help="Log every Nth frame.")
    p.add_argument("--idx", type=int, default=0, help="First frame index.")
    p.add_argument("--images", nargs="+", default=_DEFAULT_IMAGES,
                   help="Image channel keys to display as 2D views.")
    p.add_argument("--depth-key", default=_DEPTH_KEY,
                   help="Single-channel depth map to colourise (skipped if absent).")
    p.add_argument("--depth-max", type=float, default=30.0,
                   help="Upper bound (m) of the fixed depth colour scale.")
    p.add_argument("--no-lidar", action="store_true",
                   help="Show only the image channels, no point cloud.")
    p.add_argument("--save", default=None, help="Write a .rrd file instead of spawning.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    keys = list(args.images)
    if not args.no_lidar:
        keys.append(_POINT_KEY)
    if args.depth_key:
        keys.append(args.depth_key)

    ds = apairo.TartanKittiDataset(str(root), keys=keys)
    print(f"  {len(ds)} frames — channels: {keys}")

    frames = range(args.idx, len(ds), args.every)

    # RGB cameras logged as-is; the depth map colourised with a fixed scale so
    # the same distance keeps the same colour across the whole sequence.
    image_channels: list[ImageChannel] = [
        ImageChannel(key) for key in args.images
    ]
    if args.depth_key:
        image_channels.append(ImageChannel(
            args.depth_key, name=f"{args.depth_key} (0–{args.depth_max:g} m)",
            colormap=lambda a: colorize(a, vmin=0.0, vmax=args.depth_max),
        ))

    pipelines = [] if args.no_lidar else [
        Pipeline("LiDAR", point_key=_POINT_KEY, label_key=None)
    ]

    apairo_rr.view(
        ds,
        pipelines=pipelines,
        images=image_channels,
        frames=frames,
        application_id="image_channels",
        save=args.save,
        spawn=args.save is None,
    )


if __name__ == "__main__":
    main()
