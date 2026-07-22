"""Pipeline-structure graphs for apairo dataset chains.

Where the 3-D viewer shows the *result* of a pipeline (the point cloud), this
module shows its *structure*: how a dataset was composed -- raw sources,
``synchronize`` / ``filter`` / ``select`` / ``window`` / concat / zip wrappers
and registered transforms -- rendered as a Graphviz graph.

The structure is recovered purely by introspecting the objects apairo already
builds: every wrapper keeps a reference to the dataset it wraps, and
``transform()`` records its steps on the dataset's ``_pipeline`` list.
Nothing is executed and no frame is loaded.

.. warning::
    ``.cache()`` materialises its parent and drops the reference, so the chain
    upstream of a ``CachedDataset`` cannot be recovered.  Describe the dataset
    *before* caching if you need the full graph; a cached dataset is rendered
    as a root annotated ``upstream unknown``.

This module imports only the standard library.  ``.svg`` and interactive
``.html`` are rendered by the built-in engine (:mod:`apairo_visu.graph_render`,
no dependency); ``.dot`` and Mermaid ``.mmd`` are dependency-free text formats;
only raster output (``.png``, ``.pdf``) needs the Graphviz ``dot`` executable
on ``PATH``.

Examples::

    from apairo_visu import graph

    spec = graph.describe(ds)          # GraphSpec, for inspection / testing
    graph.export(ds, "pipeline.svg")   # styled SVG, built-in renderer
    graph.export(ds, "pipeline.html")  # interactive page (pan/zoom, themes)
    graph.export(ds, "pipeline.dot")   # Graphviz DOT text
    graph.show(ds)                     # open the interactive page
"""

from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Node",
    "Edge",
    "GraphSpec",
    "describe",
    "to_dot",
    "to_mermaid",
    "export",
    "show",
]


@dataclass
class Node:
    """One box in the graph: a dataset object or a registered transform step.

    Attributes:
        id: Graph-unique identifier (``n0``, ``n1``, ...).
        kind: ``"dataset"`` (wrapper / source object) or ``"transform"``
            (one ``_pipeline`` step).
        label: Class name for datasets, callable / preprocessor name for
            transforms.
        params: Human-readable key facts (reference channel, tolerance, kept
            frame count, ...) shown under the label.
    """

    id: str
    kind: str
    label: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed data-flow edge, optionally labelled with the channel name."""

    src: str
    dst: str
    label: str | None = None


@dataclass
class GraphSpec:
    """Renderer-agnostic description of a pipeline graph."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


# --------------------------------------------------------------- introspection
#
# Everything below reads apairo internals (`_parent`, `_pipeline`, wrapper
# parameters).  All that access is confined to the three helpers `_upstream`,
# `_dataset_params` and `_step_info`, and is defensive (getattr with defaults)
# so a core refactor degrades the graph instead of breaking it.


def _upstream(ds) -> list:
    """Datasets *ds* was built from (empty for roots and cached datasets)."""
    parent = getattr(ds, "_parent", None)  # every single-parent wrapper
    if parent is not None:
        return [parent]
    for attr in ("datasets", "_datasets"):  # ConcatDataset (public) / ZipDataset
        children = getattr(ds, attr, None)
        if isinstance(children, (list, tuple)):
            return list(children)
    return []


def _dataset_params(ds) -> dict[str, str]:
    """Key displayable facts for a dataset node, all extracted defensively."""
    params: dict[str, str] = {}

    if hasattr(ds, "_cache") and getattr(ds, "_parent", None) is None:
        params["note"] = "upstream unknown (severed by .cache())"

    try:
        params["frames"] = str(len(ds))
    except Exception:
        pass

    # SynchronizedView
    if hasattr(ds, "_index_map"):
        params["reference"] = str(getattr(ds, "_reference", None) or "external clock")
        method = getattr(ds, "_method", None)
        if method is not None:
            params["method"] = method if isinstance(method, str) else str(method)
        tolerance = getattr(ds, "_tolerance", None)
        if tolerance is not None:
            params["tolerance"] = str(tolerance)

    # FilteredView
    indices = getattr(ds, "_indices", None)
    if indices is not None and getattr(ds, "_parent", None) is not None:
        try:
            params["kept"] = f"{len(indices)}/{len(ds._parent)}"
            params.pop("frames", None)
        except Exception:
            pass

    # WindowView
    if hasattr(ds, "_windows"):
        for attr in ("_size", "_stride", "_boundary"):
            value = getattr(ds, attr, None)
            if value is not None:
                params[attr.lstrip("_")] = str(value)

    for attr in ("root", "root_dir", "directory", "_sequence_dir"):
        root = getattr(ds, attr, None)
        if isinstance(root, (str, Path)):
            params["root"] = str(root)
            break

    try:
        keys = getattr(ds, "keys", None)
        if isinstance(keys, (list, tuple)) and keys:
            params["keys"] = ", ".join(str(k) for k in keys)
    except Exception:
        pass

    return params


_GENERIC_NAMES = frozenset({"fn", "step", "wrapper", "inner", "closure", "_"})


def _callable_name(fn) -> str:
    name = getattr(fn, "__name__", None)
    if name is None:
        return type(fn).__name__
    # Closures built by a factory are often named `fn` / `<lambda>`; the
    # factory's name in __qualname__ ("embed_labels.<locals>.fn") is what a
    # reader recognises.
    if name in _GENERIC_NAMES or name.startswith("<"):
        parts = getattr(fn, "__qualname__", "").split(".<locals>.")
        if len(parts) >= 2:  # innermost enclosing function = the factory
            return parts[-2].split(".")[-1]
    return name


def _step_info(step) -> tuple[str, list[str] | None, str | None]:
    """``(label, read_channels, written_channel)`` for one ``_pipeline`` step.

    ``read_channels`` is ``None`` for a sample-level callable: it receives the
    whole sample and may touch anything, so it must be treated as a barrier in
    the dataflow graph.
    """
    # _ChannelStep: transform(key, fn, output=...) -- provably reads only *key*
    if hasattr(step, "key") and hasattr(step, "fn"):
        output = step.output if step.output is not None else step.key
        return _callable_name(step.fn), [step.key], output
    # _PreprocessorStep: transform(preprocessor) -- reads declared input_keys
    if hasattr(step, "preprocessor") and hasattr(step, "output"):
        inputs = list(getattr(step.preprocessor, "input_keys", None) or [])
        return type(step.preprocessor).__name__, inputs or None, step.output
    # Sample-level: a bare callable, unknown reads/writes.
    return _callable_name(step), None, None


# --------------------------------------------------------------------- walker


def describe(*datasets, registry: dict | None = None) -> GraphSpec:
    """Build the structure graph of one or more apairo datasets.

    Walks each dataset's wrapper chain back to its roots and lists the
    transform steps registered on every object along the way.  Objects shared
    between the given datasets (e.g. two branches filtered from the same raw
    dataset) appear once, so passing several datasets yields one merged graph.

    Args:
        datasets: Final dataset objects, as built by a training script.
        registry: Optional dict filled with ``node_id -> live object``: the
            dataset object for dataset nodes, ``(owner_dataset, step_index)``
            for transform nodes.  Lets an interactive host (the studio) map a
            clicked node back to the object it describes.

    Returns:
        A :class:`GraphSpec` -- pure data, render it with :func:`to_dot`,
        :func:`export` or :func:`show`.
    """
    if not datasets:
        raise ValueError("describe() needs at least one dataset")
    spec = GraphSpec()
    memo: dict[int, list[tuple[str, str | None]]] = {}
    counter = itertools.count()
    for ds in datasets:
        _visit(ds, spec, memo, counter, registry)
    return spec


def _visit(ds, spec, memo, counter, registry=None) -> list[tuple[str, str | None]]:
    """Add *ds* and everything upstream of it to *spec*.

    Registered transforms are laid out as a *dataflow* graph, not the flat
    registration order: a per-channel step depends only on the latest writer
    of the channel it reads, so steps touching independent channels appear as
    parallel branches (execution stays sequential -- the graph shows data
    dependencies).  Sample-level callables receive the whole sample and may
    read or write anything, so they join all open branches (a barrier).

    Returns the open leaves ``[(node_id, channel), ...]`` downstream consumers
    connect from, each with the channel it published (``None`` if unknown).
    """
    if id(ds) in memo:
        return memo[id(ds)]

    node = Node(
        id=f"n{next(counter)}",
        kind="dataset",
        label=type(ds).__name__,
        params=_dataset_params(ds),
    )
    spec.nodes.append(node)
    if registry is not None:
        registry[node.id] = ds

    for parent in _upstream(ds):
        for src, channel in _visit(parent, spec, memo, counter, registry):
            spec.edges.append(Edge(src, node.id, channel))

    base = node.id  # where channels nobody rewrote yet come from
    last_writer: dict[str, str] = {}
    leaves: dict[str, str | None] = {node.id: None}

    def connect(src: str, dst: str, label: str | None) -> None:
        spec.edges.append(Edge(src, dst, label))
        leaves.pop(src, None)

    for step_index, step in enumerate(getattr(ds, "_pipeline", None) or []):
        label, reads, writes = _step_info(step)
        step_node = Node(id=f"n{next(counter)}", kind="transform", label=label)
        spec.nodes.append(step_node)
        if registry is not None:
            registry[step_node.id] = (ds, step_index)
        if reads is None:
            for src, channel in list(leaves.items()):
                connect(src, step_node.id, channel)
            last_writer.clear()
            base = step_node.id
        else:
            deps: dict[str, list[str]] = {}
            for key in reads:
                deps.setdefault(last_writer.get(key, base), []).append(key)
            for src, keys in deps.items():
                connect(src, step_node.id, ", ".join(keys))
            if writes:
                last_writer[writes] = step_node.id
        leaves[step_node.id] = writes

    memo[id(ds)] = list(leaves.items())
    return memo[id(ds)]


# ------------------------------------------------------------------ rendering

_DATASET_STYLE = 'shape=box, style="rounded,filled", fillcolor="#dbe7f5"'
_TRANSFORM_STYLE = 'shape=ellipse, style="filled", fillcolor="#fdf0d5"'


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(spec: GraphSpec) -> str:
    """Render a :class:`GraphSpec` as Graphviz DOT text."""
    lines = [
        "digraph apairo_pipeline {",
        "  rankdir=TB;",
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9, color="#707070",'
        ' fontcolor="#707070"];',
    ]
    for node in spec.nodes:
        parts = [node.label] + [f"{k}: {v}" for k, v in node.params.items()]
        label = "\\n".join(_escape(p) for p in parts)
        style = _TRANSFORM_STYLE if node.kind == "transform" else _DATASET_STYLE
        lines.append(f'  {node.id} [label="{label}", {style}];')
    for edge in spec.edges:
        attrs = f' [label="{_escape(edge.label)}"]' if edge.label else ""
        lines.append(f"  {edge.src} -> {edge.dst}{attrs};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _mermaid_label(node: Node) -> str:
    parts = [node.label] + [f"{k}: {v}" for k, v in node.params.items()]
    # Mermaid node text is quoted; single quotes are safe, double quotes end it.
    return "<br/>".join(p.replace('"', "'") for p in parts)


def to_mermaid(spec: GraphSpec) -> str:
    """Render a :class:`GraphSpec` as Mermaid ``flowchart`` text.

    Mermaid needs no local renderer: GitHub / GitLab READMEs, VS Code,
    Obsidian and Jupyter render it natively, so this is the zero-install
    output -- paste it in a fenced ``mermaid`` block.  Use :func:`to_dot` /
    :func:`export` when you want a standalone SVG/PNG file instead.
    """
    lines = ["flowchart TB"]
    for node in spec.nodes:
        label = _mermaid_label(node)
        shape = f'(["{label}"])' if node.kind == "transform" else f'("{label}")'
        lines.append(f"  {node.id}{shape}")
    for edge in spec.edges:
        arrow = f"-- {edge.label} -->" if edge.label else "-->"
        lines.append(f"  {edge.src} {arrow} {edge.dst}")
    for kind, style in (
        ("dataset", "fill:#dbe7f5,stroke:#5b7ea8"),
        ("transform", "fill:#fdf0d5,stroke:#c9a15a"),
    ):
        ids = [n.id for n in spec.nodes if n.kind == kind]
        if ids:
            lines.append(f"  classDef {kind} {style}")
            lines.append(f"  class {','.join(ids)} {kind}")
    return "\n".join(lines) + "\n"


def export(dataset, path: str | Path) -> Path:
    """Write the pipeline graph of *dataset* to *path*.

    The output format follows the file suffix.  Dependency-free formats,
    rendered by :mod:`apairo_visu.graph_render` or plain templating:

    - ``.svg`` -- styled standalone SVG (built-in layout, no Graphviz needed)
    - ``.html`` -- self-contained interactive page (pan / zoom, light / dark)
    - ``.dot`` / ``.gv`` -- Graphviz DOT text
    - ``.mmd`` / ``.mermaid`` -- Mermaid text (renders on GitHub / VS Code)

    Any other suffix (``.png``, ``.pdf``, ...) is rasterised with the
    Graphviz ``dot`` executable, which must be on ``PATH``.

    Args:
        dataset: A dataset object, or an already-built :class:`GraphSpec`.
        path: Destination file.

    Returns:
        The written path.
    """
    spec = dataset if isinstance(dataset, GraphSpec) else describe(dataset)
    path = Path(path)

    if path.suffix == ".svg":
        from . import graph_render

        path.write_text(graph_render.to_svg(spec))
        return path

    if path.suffix in (".html", ".htm"):
        from . import graph_render

        path.write_text(graph_render.to_html(spec))
        return path

    if path.suffix in (".mmd", ".mermaid"):
        path.write_text(to_mermaid(spec))
        return path

    dot = to_dot(spec)
    if path.suffix in (".dot", ".gv"):
        path.write_text(dot)
        return path

    fmt = path.suffix.lstrip(".") or "png"
    try:
        proc = subprocess.run(
            ["dot", f"-T{fmt}", "-o", str(path)],
            input=dot.encode(),
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Graphviz 'dot' executable not found on PATH -- install graphviz "
            "(e.g. `apt install graphviz`), or export to a .dot file instead."
        ) from None
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Graphviz rendering failed: {stderr}")
    return path


def show(dataset) -> Path:
    """Render *dataset*'s pipeline graph to a temporary interactive HTML page
    (pan / zoom, light / dark theme) and open it in the browser."""
    fd, tmp = tempfile.mkstemp(suffix=".html", prefix="apairo_pipeline_")
    os.close(fd)
    path = export(dataset, tmp)
    webbrowser.open(path.as_uri())
    return path
