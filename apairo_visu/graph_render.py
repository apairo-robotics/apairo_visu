"""Built-in SVG / HTML renderer for pipeline :class:`~apairo_visu.graph.GraphSpec`.

Modern-looking alternative to the Graphviz backend: the layered layout is
computed here in pure Python (pipeline graphs are small DAGs -- a Sugiyama-style
pass is enough), and the output is a styled SVG in the spirit of Kedro-Viz.
No external dependency: ``to_svg`` writes a standalone SVG file and ``to_html``
wraps it in a self-contained page with pan / zoom and light / dark themes.

Graphviz remains the backend for raster formats (``.png``, ``.pdf``) via
:func:`apairo_visu.graph.export`.
"""

from __future__ import annotations

from .graph import GraphSpec, Node

__all__ = ["to_svg", "to_html"]

# ------------------------------------------------------------------- palettes

_LIGHT = {
    "canvas": "#f6f7f9",
    "dataset-fill": "#ffffff",
    "dataset-stroke": "#c7d3e2",
    "transform-fill": "#fdf6e8",
    "transform-stroke": "#e0cda2",
    "accent": "#4f7cac",
    "text": "#22303f",
    "muted": "#6d7889",
    "edge": "#a3aebc",
}

_DARK = {
    "canvas": "#161b24",
    "dataset-fill": "#222b3a",
    "dataset-stroke": "#3b4c66",
    "transform-fill": "#2d2921",
    "transform-stroke": "#5e5339",
    "accent": "#7fb3e8",
    "text": "#e7edf6",
    "muted": "#94a2b6",
    "edge": "#556074",
}

_FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"

# ------------------------------------------------------------------ layout

_PAD_X = 16
_PAD_Y = 12
_TITLE_H = 18
_LINE_H = 15
_RANK_GAP = 58
_NODE_GAP = 36
_MARGIN = 28
_MAX_PARAM_CHARS = 64


def _lines(node: Node) -> list[str]:
    lines = []
    for k, v in node.params.items():
        line = f"{k}: {v}"
        if len(line) > _MAX_PARAM_CHARS:
            line = line[: _MAX_PARAM_CHARS - 1] + "…"
        lines.append(line)
    return lines


def _box_size(node: Node) -> tuple[float, float]:
    lines = _lines(node)
    width = max(
        [len(node.label) * 7.8] + [len(line) * 6.1 for line in lines] + [70.0]
    ) + 2 * _PAD_X
    if node.kind == "dataset":
        width += 14  # room for the accent bar
    height = 2 * _PAD_Y + _TITLE_H + _LINE_H * len(lines)
    return width, height


def _layout(spec: GraphSpec) -> tuple[dict[str, tuple[float, float, float, float]], float, float]:
    """Assign a (x, y, w, h) box to every node; returns (boxes, width, height).

    Layered layout: rank = longest path from a root, order within a rank by
    the mean order of the parents (one barycenter pass is plenty at this
    scale), ranks centered on the vertical axis.
    """
    parents: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    for e in spec.edges:
        parents[e.dst].append(e.src)

    rank: dict[str, int] = {}
    pending = [n.id for n in spec.nodes]
    while pending:
        progressed = False
        remaining = []
        for nid in pending:
            if all(p in rank for p in parents[nid]):
                rank[nid] = 1 + max((rank[p] for p in parents[nid]), default=-1)
                progressed = True
            else:
                remaining.append(nid)
        if not progressed:  # cycle guard -- flatten whatever is left
            for nid in remaining:
                rank[nid] = 0
            break
        pending = remaining

    ranks: list[list[str]] = [[] for _ in range(max(rank.values(), default=0) + 1)]
    for n in spec.nodes:
        ranks[rank[n.id]].append(n.id)

    order = {nid: i for r in ranks for i, nid in enumerate(r)}
    for r in ranks[1:]:
        r.sort(key=lambda nid: sum(order[p] for p in parents[nid]) / len(parents[nid])
               if parents[nid] else order[nid])
        for i, nid in enumerate(r):
            order[nid] = i

    sizes = {n.id: _box_size(n) for n in spec.nodes}
    row_widths = [
        sum(sizes[nid][0] for nid in r) + _NODE_GAP * (len(r) - 1) for r in ranks
    ]
    canvas_w = max(row_widths, default=0.0)

    boxes: dict[str, tuple[float, float, float, float]] = {}
    y = float(_MARGIN)
    for r, row_w in zip(ranks, row_widths):
        x = _MARGIN + (canvas_w - row_w) / 2
        row_h = max(sizes[nid][1] for nid in r)
        for nid in r:
            w, h = sizes[nid]
            boxes[nid] = (x, y + (row_h - h) / 2, w, h)
            x += w + _NODE_GAP
        y += row_h + _RANK_GAP
    return boxes, canvas_w + 2 * _MARGIN, y - _RANK_GAP + _MARGIN


# ------------------------------------------------------------------ svg

def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_body(spec: GraphSpec, c: dict[str, str]) -> tuple[str, float, float]:
    """Render the SVG document. *c* maps palette tokens to CSS colors -- pass
    hex values for a standalone file or ``var(--...)`` references for HTML."""
    boxes, width, height = _layout(spec)
    parts: list[str] = []

    parts.append(
        f'<svg id="pipeline" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{_FONT}">'
    )
    parts.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,1.5 L9,5 L0,8.5 Z" fill="{c["edge"]}"/></marker></defs>'
    )
    parts.append(f'<rect width="100%" height="100%" fill="{c["canvas"]}"/>')

    for e in spec.edges:
        x1, y1, w1, h1 = boxes[e.src]
        x2, y2, w2, h2 = boxes[e.dst]
        sx, sy = x1 + w1 / 2, y1 + h1
        tx, ty = x2 + w2 / 2, y2 - 1.5
        bend = min(36.0, (ty - sy) / 2) if ty > sy else 24.0
        parts.append(
            f'<path d="M{sx:.1f},{sy:.1f} C{sx:.1f},{sy + bend:.1f} '
            f'{tx:.1f},{ty - bend:.1f} {tx:.1f},{ty:.1f}" fill="none" '
            f'stroke="{c["edge"]}" stroke-width="1.6" marker-end="url(#arrow)"/>'
        )
        if e.label:
            mx, my = (sx + tx) / 2, (sy + ty) / 2
            parts.append(
                f'<text x="{mx + 9:.1f}" y="{my + 3.5:.1f}" font-size="10.5" '
                f'font-style="italic" fill="{c["muted"]}" '
                f'style="paint-order:stroke" stroke="{c["canvas"]}" '
                f'stroke-width="3">{_esc(e.label)}</text>'
            )

    for node in spec.nodes:
        x, y, w, h = boxes[node.id]
        is_ds = node.kind != "transform"
        fill = c["dataset-fill"] if is_ds else c["transform-fill"]
        stroke = c["dataset-stroke"] if is_ds else c["transform-stroke"]
        rx = 10 if is_ds else h / 2
        # data-node + kind class let an interactive host (the studio front)
        # bind clicks and tell node kinds apart without a metadata request.
        parts.append(f'<g class="node {node.kind}" data-node="{node.id}">')
        parts.append(
            f'<rect class="box" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.3"/>'
        )
        tx = x + _PAD_X
        if is_ds:
            parts.append(
                f'<rect x="{x + 9:.1f}" y="{y + 9:.1f}" width="4" '
                f'height="{h - 18:.1f}" rx="2" fill="{c["accent"]}"/>'
            )
            tx += 14
        ty = y + _PAD_Y + 13
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="13" font-weight="600" '
            f'fill="{c["text"]}">{_esc(node.label)}</text>'
        )
        for i, line in enumerate(_lines(node)):
            parts.append(
                f'<text x="{tx:.1f}" y="{ty + _TITLE_H + i * _LINE_H:.1f}" '
                f'font-size="11" fill="{c["muted"]}">{_esc(line)}</text>'
            )
        parts.append("</g>")

    parts.append("</svg>")
    return "".join(parts), width, height


def to_svg(spec: GraphSpec, theme: str = "light") -> str:
    """Render a :class:`GraphSpec` as a standalone styled SVG document."""
    palette = {"light": _LIGHT, "dark": _DARK}.get(theme)
    if palette is None:
        raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
    body, _, _ = _svg_body(spec, palette)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def to_html(spec: GraphSpec, title: str = "apairo pipeline") -> str:
    """Render a :class:`GraphSpec` as a self-contained interactive HTML page.

    Pan (drag) and zoom (wheel), light / dark following the OS theme.  No
    server, no external resource -- a single file you can open or share.
    """
    tokens = {key: f"var(--{key})" for key in _LIGHT}
    body, width, height = _svg_body(spec, tokens)

    def css_vars(palette: dict[str, str]) -> str:
        return ";".join(f"--{k}:{v}" for k, v in palette.items())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ {css_vars(_LIGHT)} }}
  @media (prefers-color-scheme: dark) {{ :root {{ {css_vars(_DARK)} }} }}
  :root[data-theme="light"] {{ {css_vars(_LIGHT)} }}
  :root[data-theme="dark"] {{ {css_vars(_DARK)} }}
  html, body {{ margin: 0; height: 100%; background: var(--canvas); }}
  svg {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  svg:active {{ cursor: grabbing; }}
  .node:hover .box {{ stroke: var(--accent); stroke-width: 2; }}
  .hint {{ position: fixed; right: 14px; bottom: 10px; color: var(--muted);
          font: 11px {_FONT}; user-select: none; }}
</style>
</head>
<body>
{body}
<div class="hint">drag to pan &middot; scroll to zoom</div>
<script>
  const svg = document.getElementById('pipeline');
  const vb = svg.viewBox.baseVal;
  const toWorld = (e) => {{
    const r = svg.getBoundingClientRect();
    return {{ x: vb.x + (e.clientX - r.left) / r.width * vb.width,
              y: vb.y + (e.clientY - r.top) / r.height * vb.height }};
  }};
  svg.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const k = e.deltaY > 0 ? 1.12 : 1 / 1.12;
    const p = toWorld(e);
    vb.x = p.x - (p.x - vb.x) * k;  vb.y = p.y - (p.y - vb.y) * k;
    vb.width *= k;  vb.height *= k;
  }}, {{ passive: false }});
  let drag = null;
  svg.addEventListener('pointerdown', (e) => {{
    drag = {{ x: e.clientX, y: e.clientY, vx: vb.x, vy: vb.y }};
    svg.setPointerCapture(e.pointerId);
  }});
  svg.addEventListener('pointermove', (e) => {{
    if (!drag) return;
    const r = svg.getBoundingClientRect();
    vb.x = drag.vx - (e.clientX - drag.x) / r.width * vb.width;
    vb.y = drag.vy - (e.clientY - drag.y) / r.height * vb.height;
  }});
  svg.addEventListener('pointerup', () => {{ drag = null; }});
</script>
</body>
</html>
"""
