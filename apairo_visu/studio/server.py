"""FastAPI app serving the studio: graph + node inspector + static front.

REST/JSON only in phase 1 (the inspector is click-driven; no frame streaming
yet).  The graph SVG is rendered server-side by
:mod:`apairo_visu.graph_render` with CSS-variable color tokens, so the front
themes it live and the same renderer stays the single source of graph pixels.

fastapi is imported at module level on purpose (annotation resolution --
see projector's server for the rationale); only :func:`apairo_visu.studio.launch`
imports this module, so the light layer stays importable without fastapi.
"""

from __future__ import annotations

import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import graph_render
from .catalog import Catalog
from .protocol import _is_per_point, decimate_rows, encode_array, encode_channels
from .registry import StudioRegistry

# Web front (vanilla, zero build) served as-is -- shipped as package data.
WEB_DIR = Path(__file__).resolve().parent / "web"


# Names a browser on this machine reaches a loopback-bound studio by.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# Bind addresses that listen on every interface: the user chose to expose the
# studio, so the Host header cannot be checked against a known name.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})


def _allowed_hosts(host: str) -> frozenset[str] | None:
    """Host names the studio answers to when bound to *host* (None: any)."""
    host = host.strip("[]").lower()
    if host in _WILDCARD_HOSTS:
        return None
    return _LOOPBACK_HOSTS | {host}


def _host_name(header: str) -> str:
    """The name part of a ``Host`` header (``[::1]:8710`` -> ``::1``)."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:].split("]", 1)[0]
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def _encode_preview(arr, max_points: int) -> dict:
    arr = np.asarray(arr)
    if _is_per_point(arr):
        arr, original = decimate_rows(arr, max_points)
        encoded = encode_array(arr)
        if original != arr.shape[0]:
            encoded["full_rows"] = original
        return encoded
    return encode_array(arr)


def create_app(
    datasets,
    *,
    title: str = "apairo studio",
    max_points: int = 150_000,
    catalog: Catalog | None = None,
    host: str = "127.0.0.1",
) -> FastAPI:
    registry = StudioRegistry(datasets)
    catalog = catalog if catalog is not None else Catalog()
    tokens = {key: f"var(--{key})" for key in graph_render._LIGHT}
    svg, _, _ = graph_render._svg_body(registry.spec, tokens)

    app = FastAPI(title=f"{title} API", version="0.1")
    app.state.registry = registry
    allowed_hosts = _allowed_hosts(host)

    # DNS rebinding: a page from another site can resolve its own name to
    # 127.0.0.1 and then read this API as same-origin. Its requests still
    # carry that foreign name in the Host header, so a studio bound to a
    # known address answers only to the names it is reachable by.
    @app.middleware("http")
    async def known_host_only(request, call_next):
        if allowed_hosts is not None:
            name = _host_name(request.headers.get("host", ""))
            if name not in allowed_hosts:
                return PlainTextResponse(
                    f"unknown Host {name!r}: this studio answers to "
                    f"{', '.join(sorted(allowed_hosts))}",
                    status_code=400,
                )
        return await call_next(request)

    # The front ships with the package and changes with it: force
    # revalidation (cheap: StaticFiles answers 304 via ETag) so a reload
    # after an upgrade never serves a stale module from the browser cache.
    @app.middleware("http")
    async def no_stale_cache(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.get("/api/graph")
    def get_graph() -> dict:
        return {
            "title": title,
            "svg": svg,
            "nodes": [asdict(n) for n in registry.spec.nodes],
            "edges": [asdict(e) for e in registry.spec.edges],
        }

    @app.get("/api/node/{node_id}")
    def get_node(node_id: str) -> dict:
        detail = registry.detail(node_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
        return detail

    @app.get("/api/node/{node_id}/sample/{index}")
    def get_sample(node_id: str, index: int, channels: str | None = None) -> dict:
        try:
            result = registry.sample(node_id, index)
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
        data = result["data"]
        if channels:
            wanted = set(channels.split(","))
            data = {k: v for k, v in data.items() if str(k) in wanted}
        return {
            "index": index,
            "len": result["len"],
            "channels": encode_channels(data, max_points),
            **{k: result[k] for k in ("frame", "timestamp") if k in result},
        }

    @app.get("/api/node/{node_id}/series/{channel}")
    def get_series(
        node_id: str,
        channel: str,
        cols: str = "0",
        start: int = 0,
        stop: int | None = None,
    ) -> dict:
        try:
            cols_list = [int(c) for c in cols.split(",") if c.strip() != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail="cols must be comma-separated integers")
        if not cols_list:
            raise HTTPException(status_code=400, detail="cols is empty")
        result = registry.series(node_id, channel, cols_list, start, stop)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"unknown dataset node {node_id!r}"
            )
        return result

    @app.get("/api/node/{node_id}/frames/{channel}")
    def get_frames(node_id: str, channel: str) -> dict:
        result = registry.frames(node_id, channel)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"unknown dataset node {node_id!r}"
            )
        return result

    @app.get("/api/node/{node_id}/entries/{channel}")
    def get_entries(
        node_id: str,
        channel: str,
        start: int = 0,
        stop: int | None = None,
        sequence: str | None = None,
    ) -> dict:
        result = registry.entries(node_id, channel, start, stop, sequence)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"unknown dataset node {node_id!r}"
            )
        return result

    @app.get("/api/node/{node_id}/locate/{stem}")
    def get_locate(
        node_id: str,
        stem: str,
        channel: str | None = None,
        sequence: str | None = None,
    ) -> dict:
        result = registry.locate(node_id, stem, channel, sequence)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"unknown dataset node {node_id!r}"
            )
        return result

    @app.get("/api/catalog")
    def get_catalog() -> dict:
        return catalog.to_dict()

    @app.post("/api/try")
    def try_entry(payload: dict = Body(...)) -> dict:
        # A malformed request is the client's fault: answer 400, never 500.
        for field in ("entry_id", "node_id", "channel"):
            if not isinstance(payload.get(field), str):
                raise HTTPException(status_code=400, detail=f"{field} must be a string")
        index = payload.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int):
            raise HTTPException(status_code=400, detail="index must be an integer")
        kwargs = payload.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise HTTPException(status_code=400, detail="kwargs must be an object")
        entry_id = payload["entry_id"]
        if entry_id not in catalog.entries:
            raise HTTPException(status_code=400, detail=f"unknown catalog entry {entry_id!r}")
        node_id = payload["node_id"]
        channel = payload["channel"]
        try:
            result = registry.sample(node_id, index)
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
        channels = result["data"]
        if channel not in channels:
            raise HTTPException(status_code=400, detail=f"channel {channel!r} absent at this node")

        try:
            out: dict = {"before": _encode_preview(channels[channel], max_points)}
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"channel {channel!r}: {exc}")
        # Coercion/application errors are data, not server faults: the panel
        # shows them verbatim (and retries on the next frame).
        try:
            out["snippet"] = catalog.snippet(entry_id, kwargs, channel)
            # The form that ran (per-channel or sample-level) decides the
            # snippet: re-render it so pasting it gives what is displayed.
            after, out["snippet"] = catalog.try_apply(entry_id, kwargs, channels, channel)
            out["after"] = _encode_preview(after, max_points)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    # SSR-lite: the graph is baked into the page so it shows on first paint;
    # the front script only binds interactivity on the existing SVG.
    index_html = (WEB_DIR / "index.html").read_text().replace("<!--GRAPH-->", svg)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return index_html

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def serve(
    datasets,
    *,
    host: str = "127.0.0.1",
    port: int = 8710,
    open_browser: bool = True,
    title: str = "apairo studio",
    max_points: int = 150_000,
) -> None:
    import uvicorn

    app = create_app(datasets, title=title, max_points=max_points, host=host)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    print(f"apairo studio serving on {url}  (Ctrl-C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
