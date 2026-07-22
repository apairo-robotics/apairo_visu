"""numpy -> JSON array codec + preview decimation (no FastAPI dependency).

Arrays travel as ``{dtype, shape, data(base64)}`` -- compact, lossless, and
trivially decodable on the JS side into a TypedArray over the buffer (house
codec, mirrored from splasher).  The server sends raw semantic arrays, never
pixels: all colorization happens client-side.
"""

from __future__ import annotations

import base64

import numpy as np


def encode_array(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict) -> np.ndarray:
    buf = base64.b64decode(payload["data"])
    return np.frombuffer(buf, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def decimate_rows(arr: np.ndarray, cap: int) -> tuple[np.ndarray, int]:
    """Stride-subsample the first axis above *cap* rows.

    Only meant for per-point data (clouds, per-point labels): strided rows
    keep the spatial structure readable.  Returns ``(array, original_rows)``.
    """
    rows = arr.shape[0]
    if rows <= cap:
        return arr, rows
    step = -(-rows // cap)  # ceil division
    return arr[::step], rows


def _is_per_point(arr: np.ndarray) -> bool:
    """Row-wise data it is safe to decimate: long 1-D, or (N, few-columns).

    Images ((H, W) rasters, (H, W, C)) must never lose rows.
    """
    if arr.ndim == 1:
        return True
    return arr.ndim == 2 and arr.shape[1] <= 32


def encode_channels(data: dict, max_points: int) -> dict:
    """Encode a sample's channel dict, decimating per-point channels."""
    channels: dict[str, dict] = {}
    for name, value in data.items():
        try:
            arr = np.asarray(value)
        except Exception:
            channels[str(name)] = {"repr": repr(value)}
            continue
        if arr.dtype == object or arr.ndim == 0:
            channels[str(name)] = {"repr": repr(value)}
            continue
        full_rows = None
        if _is_per_point(arr):
            arr, original = decimate_rows(arr, max_points)
            if original != arr.shape[0]:
                full_rows = original
        entry = encode_array(arr)
        if full_rows is not None:
            entry["full_rows"] = full_rows
        channels[str(name)] = entry
    return channels
