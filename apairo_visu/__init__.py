"""apairo_visu -- interactive 3-D LiDAR visualisation for apairo datasets.

The light layer (``Pipeline``, ``ViewConfig``, ``load_label_config``,
``load_poses``, ``graph``) imports only numpy / PyYAML, so it can be used in a
headless context.  :class:`LidarViewer` pulls in Open3D and is therefore loaded lazily,
the first time it is accessed::

    import apairo_visu                      # no Open3D import yet
    cfg = apairo_visu.load_label_config("goose")
    apairo_visu.LidarViewer.launch(ds, label_cfg=cfg)   # Open3D loaded here
"""

from __future__ import annotations

from . import graph
from .config import BUILTIN_CONFIGS, ViewConfig, load_label_config
from .pipeline import Pipeline
from .poses import load_poses, pose_to_matrix

__all__ = [
    "LidarViewer",
    "ViewConfig",
    "Pipeline",
    "graph",
    "load_label_config",
    "load_poses",
    "pose_to_matrix",
    "BUILTIN_CONFIGS",
]


def __getattr__(name: str):
    # PEP 562: defer the Open3D import until LidarViewer is actually used.
    if name == "LidarViewer":
        from .viewer import LidarViewer
        return LidarViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
