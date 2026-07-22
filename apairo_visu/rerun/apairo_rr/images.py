"""Image-channel viewing for the apairo_rr viewer.

Where :class:`~apairo_rr.Pipeline` describes how a point-cloud channel becomes a
3D view, :class:`ImageChannel` describes how a 2D image channel becomes a Rerun
2D view.  Pass a list of them (or plain channel-key strings) to
:func:`~apairo_rr.view` via ``images=`` to watch camera / depth / BEV channels
evolve along the timeline next to the point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ImageChannel:
    """A 2D image channel to display in the viewer, updating per frame.

    Args:
        key:      Sample key holding the image array.  RGB images load as
                  ``(H, W, 3)`` uint8 (apairo ``img`` channels); single-channel
                  maps load as ``(H, W)``.
        name:     Display title for the 2D view (defaults to *key*).
        colormap: Optional ``(array) -> (H, W, 3) uint8`` callable applied before
                  logging — use it to colourise a scalar map (depth, height,
                  cost) with :func:`~apairo_rr.colorize`.  ``None`` logs the raw
                  array (RGB or grayscale) directly.

    Examples::

        ImageChannel("image_left_color")
        ImageChannel("image_right", name="Right camera")
        ImageChannel("depth_left", name="Depth",
                     colormap=lambda a: colorize(a, vmin=0.0, vmax=30.0))
    """

    key: str
    name: Optional[str] = None
    colormap: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @property
    def title(self) -> str:
        return self.name or self.key

    def resolve(self, sample) -> Optional[np.ndarray]:
        """Fetch the channel from *sample* and apply *colormap* if set.

        Returns ``None`` when the channel is absent in this frame.  Async sensors
        update at different rates, so a missing image simply leaves the previously
        logged one on screen — the Rerun timeline holds the last value until the
        channel updates again, which is exactly the "evolution over time" view.
        """
        arr = sample.data.get(self.key)
        if arr is None:
            return None
        arr = np.asarray(arr)
        if self.colormap is not None:
            arr = np.asarray(self.colormap(arr))
        return arr
