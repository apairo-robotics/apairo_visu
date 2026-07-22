"""apairo studio -- interactive pipeline environment (browser UI).

Serves the structure graph of live apairo dataset objects with a
click-to-inspect panel: node parameters, docstring, channel table.  Follows
the house architecture shared by projector / toaster / splasher: numpy-only
introspection core, FastAPI server, vanilla-JS zero-build front shipped as
package data.

Requires the ``studio`` extra: ``pip install apairo_visu[studio]``.

Example::

    import apairo
    from apairo_visu import studio

    ds = apairo.RawDataset(root, keys=[...]).synchronize(...)
    ds.transform("lidar", fn, output="lidar_f")
    studio.launch(ds)            # serves on 127.0.0.1:8710 and opens a browser
"""

from __future__ import annotations

__all__ = ["launch"]


def launch(
    *datasets,
    host: str = "127.0.0.1",
    port: int = 8710,
    open_browser: bool = True,
    title: str = "apairo studio",
    max_points: int = 150_000,
) -> None:
    """Serve the studio for one or more datasets (blocks until Ctrl-C).

    Args:
        datasets: Live dataset objects, as built by a training script.
            Several datasets are merged into one graph (shared upstream
            objects appear once).
        host: Bind address -- keep the localhost default unless you know
            you want the studio reachable from other machines.
        port: HTTP port.
        open_browser: Open the default browser on the studio URL.
        title: Window / page title.
        max_points: Per-point channels (clouds, per-point labels) are
            stride-decimated above this many rows before being sent to the
            browser.
    """
    from .server import serve  # lazy: fastapi/uvicorn are optional deps

    serve(
        datasets,
        host=host,
        port=port,
        open_browser=open_browser,
        title=title,
        max_points=max_points,
    )
