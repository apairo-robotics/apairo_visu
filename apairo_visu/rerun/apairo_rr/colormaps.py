"""Colormap abstractions and built-in implementations for Pipeline.colormap_fn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# A gradient maps a 1-D scalar array to (N, 3) uint8 RGB.
# Use as building blocks inside custom Colormap subclasses.
# Built-in gradients also accept optional ``vmin``/``vmax`` keywords to anchor
# the scale; a plain ``(v) -> rgb`` callable remains a valid Gradient.
Gradient = Callable[..., np.ndarray]


def _normalize(v: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
    """Scale *v* to ``[0, 1]``.

    When *vmin*/*vmax* are ``None`` the range is taken from the array itself
    (per-call normalisation — **not comparable across frames**).  Pass fixed
    bounds to anchor the scale so the same value maps to the same colour in
    every frame; out-of-range values are clipped.
    """
    # nan-aware bounds: organized clouds (e.g. Ouster) carry NaN for non-returns;
    # a plain min()/max() would poison the whole scale to NaN.  The NaN points
    # themselves are dropped before logging (see viewer), so only valid points
    # are coloured here.
    finite = np.isfinite(v)
    lo = (float(np.min(v[finite])) if finite.any() else 0.0) if vmin is None else float(vmin)
    hi = (float(np.max(v[finite])) if finite.any() else 1.0) if vmax is None else float(vmax)
    t = (v.astype(np.float64) - lo) / max(hi - lo, 1e-6)
    # Map NaN/inf to 0 so the downstream uint8 cast is clean; those points are
    # dropped before logging anyway.
    return np.nan_to_num(np.clip(t, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


class Colormap(ABC):
    """Interface for ``Pipeline.colormap_fn``.

    Subclass this to define any coloring logic from a full point array::

        class RangeColormap(Colormap):
            def __call__(self, pts: np.ndarray) -> np.ndarray:
                r = np.linalg.norm(pts[:, :3], axis=1)
                return red_blue(r)

        Pipeline("Range", colormap_fn=RangeColormap())
    """

    @abstractmethod
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Map an (N, D) point array to an (N, 3) uint8 RGB array."""


# ---------------------------------------------------------------------------
# Gradient primitives  —  Gradient = (N,) scalar → (N, 3) uint8
# ---------------------------------------------------------------------------

def red_blue(v: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Red (low) → blue (high) gradient.

    By default normalised over *v* itself, so colours are **not comparable
    across frames**.  Pass fixed *vmin*/*vmax* to anchor the scale.
    """
    t = _normalize(v, vmin, vmax)
    rgb = np.zeros((len(v), 3), dtype=np.uint8)
    rgb[:, 0] = (255 * (1 - t)).astype(np.uint8)
    rgb[:, 2] = (255 * t).astype(np.uint8)
    return rgb


def green_red(v: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Green (low) → red (high) gradient.

    By default normalised over *v* itself, so colours are **not comparable
    across frames**.  Pass fixed *vmin*/*vmax* to anchor the scale.
    """
    t = _normalize(v, vmin, vmax)
    rgb = np.zeros((len(v), 3), dtype=np.uint8)
    rgb[:, 1] = (255 * (1 - t)).astype(np.uint8)
    rgb[:, 0] = (255 * t).astype(np.uint8)
    return rgb


def colorize(
    scalar: np.ndarray,
    gradient: Gradient = red_blue,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map a 2-D scalar image to an ``(H, W, 3)`` uint8 RGB image via a gradient.

    Reuses the 1-D :data:`Gradient` primitives (:func:`red_blue`, :func:`green_red`)
    by flattening the image, colouring each pixel, then restoring the 2-D shape.
    Use it to turn a single-channel sensor map (depth, height, cost) into a colour
    image for :class:`~apairo_rr.ImageChannel`.

    Args:
        scalar:   ``(H, W)`` (or ``(H, W, 1)``) array of per-pixel scalars.
        gradient: A :data:`Gradient` applied to the flattened pixels.
        vmin:     Fixed lower bound for normalisation.  Leave ``None`` to
                  normalise on each frame's own min/max (colours not comparable
                  across frames — the same footgun as the point gradients).
        vmax:     Fixed upper bound for normalisation.

    Example::

        ImageChannel("depth_left", colormap=lambda a: colorize(a, vmin=0, vmax=30))
    """
    arr = np.asarray(scalar)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    flat = arr.reshape(-1)
    if vmin is None and vmax is None:
        rgb = gradient(flat)
    else:
        rgb = gradient(flat, vmin=vmin, vmax=vmax)
    return rgb.reshape(*arr.shape, 3)


# ---------------------------------------------------------------------------
# Built-in Colormap implementations
# ---------------------------------------------------------------------------

class KeyColormap(Colormap):
    """Colour by a per-point scalar stored in a dedicated sample key.

    Unlike :class:`ColumnColormap`, which reads a column from the point array,
    ``KeyColormap`` reads a separate channel from the sample directly — no
    embedding into the point array is needed.

    Args:
        key:      Sample key containing the per-point scalar array
                  (e.g. ``"ground_height_csf"``).
        gradient: :data:`Gradient` applied to the scalar.  Defaults to
                  :func:`red_blue`.
        vmin:     Fixed lower bound for normalisation.  Leave ``None`` to
                  normalise per-frame (colours not comparable across frames).
        vmax:     Fixed upper bound for normalisation.

    Example::

        Pipeline("Height CSF", point_key="voxelised", label_key=None,
                 colormap_fn=KeyColormap("ground_height_csf", vmin=0.0, vmax=2.0))
    """

    def __init__(self, key: str, gradient: Gradient = red_blue,
                 vmin: float | None = None, vmax: float | None = None):
        self.key = key
        self.gradient = gradient
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, pts: np.ndarray, sample=None) -> np.ndarray:
        if sample is not None and self.key in sample.data:
            v = np.asarray(sample.data[self.key], dtype=np.float32).ravel()
        else:
            v = pts[:, 0]
        if self.vmin is None and self.vmax is None:
            return self.gradient(v)
        return self.gradient(v, vmin=self.vmin, vmax=self.vmax)


class ColumnColormap(Colormap):
    """Colour each point by a single scalar column, via a :data:`Gradient`.

    Args:
        col:      Column index to extract from the (N, D) pts array.
        gradient: A :data:`Gradient` function applied to the extracted column.
                  Defaults to :func:`red_blue`.
        vmin:     Fixed lower bound for normalisation.  Leave ``None`` to
                  normalise per-frame (colours not comparable across frames).
        vmax:     Fixed upper bound for normalisation.

    Examples::

        Pipeline("Height Z",      colormap_fn=ColumnColormap(2))
        Pipeline("Intensity",     colormap_fn=ColumnColormap(3, gradient=green_red))
        Pipeline("Ground height", colormap_fn=ColumnColormap(4, vmin=0.0, vmax=2.0))
    """

    def __init__(self, col: int, gradient: Gradient = red_blue,
                 vmin: float | None = None, vmax: float | None = None):
        self.col = col
        self.gradient = gradient
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        v = pts[:, self.col]
        if self.vmin is None and self.vmax is None:
            return self.gradient(v)
        return self.gradient(v, vmin=self.vmin, vmax=self.vmax)
