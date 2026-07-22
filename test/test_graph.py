"""Tests for graph -- introspection of apairo dataset chains and DOT export."""

import shutil

import numpy as np
import pytest

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample

from apairo_visu import graph


class ArrayDataset(AbstractDataset):
    """Minimal in-memory synchronous dataset for exercising the walker."""

    def __init__(self, n: int = 5, keys: tuple[str, ...] = ("lidar",)):
        self._set_keys(list(keys))
        self._n = n

    def __len__(self) -> int:
        return self._n

    def _load(self, idx: int) -> Sample:
        data = {k: np.full((4, 3), idx, dtype=np.float32) for k in self.keys}
        return Sample(data=data)

    @property
    def is_synchronous(self) -> bool:
        return True


def node_labels(spec):
    return [n.label for n in spec.nodes]


def test_single_root_dataset():
    spec = graph.describe(ArrayDataset(n=3))
    assert len(spec.nodes) == 1
    assert spec.edges == []
    (node,) = spec.nodes
    assert node.kind == "dataset"
    assert node.label == "ArrayDataset"
    assert node.params["frames"] == "3"
    assert node.params["keys"] == "lidar"


def test_filter_chain_has_parent_edge_and_kept_count():
    ds = ArrayDataset(n=5)
    view = ds.filter(np.array([0, 2]))

    spec = graph.describe(view)
    assert node_labels(spec) == ["FilteredView", "ArrayDataset"]
    (edge,) = spec.edges
    root = next(n for n in spec.nodes if n.label == "ArrayDataset")
    filt = next(n for n in spec.nodes if n.label == "FilteredView")
    assert (edge.src, edge.dst) == (root.id, filt.id)
    assert filt.params["kept"] == "2/5"


def test_per_channel_transform_becomes_labelled_step():
    def double(arr):
        return arr * 2

    ds = ArrayDataset()
    ds.transform("lidar", double, output="lidar_2x")
    view = ds.filter(np.array([0]))

    spec = graph.describe(view)
    step = next(n for n in spec.nodes if n.kind == "transform")
    assert step.label == "double"

    ds_node = next(n for n in spec.nodes if n.label == "ArrayDataset")
    filt_node = next(n for n in spec.nodes if n.label == "FilteredView")
    in_edge = next(e for e in spec.edges if e.dst == step.id)
    out_edge = next(e for e in spec.edges if e.src == step.id)
    assert in_edge.src == ds_node.id
    assert in_edge.label == "lidar"          # channel the step consumes
    assert out_edge.dst == filt_node.id
    assert out_edge.label == "lidar_2x"      # channel the step publishes


def test_sample_level_transform_has_no_channel_labels():
    def jitter(sample):
        return sample

    ds = ArrayDataset()
    ds.transform(jitter)

    spec = graph.describe(ds)
    step = next(n for n in spec.nodes if n.kind == "transform")
    assert step.label == "jitter"
    (edge,) = spec.edges
    assert edge.label is None


def test_independent_channel_transforms_are_parallel_branches():
    def fix_lidar(arr):
        return arr

    def fix_imu(arr):
        return arr

    ds = ArrayDataset(keys=("lidar", "imu"))
    ds.transform("lidar", fix_lidar)
    ds.transform("imu", fix_imu)
    view = ds.filter(np.array([0]))

    spec = graph.describe(view)
    ds_node = next(n for n in spec.nodes if n.label == "ArrayDataset")
    filt = next(n for n in spec.nodes if n.label == "FilteredView")
    t_lidar = next(n for n in spec.nodes if n.label == "fix_lidar")
    t_imu = next(n for n in spec.nodes if n.label == "fix_imu")

    # Fork: both steps read directly from the dataset, not from each other.
    assert {e.dst for e in spec.edges if e.src == ds_node.id} == {t_lidar.id, t_imu.id}
    # Join: the downstream view consumes both branches, with channel labels.
    in_edges = {e.src: e.label for e in spec.edges if e.dst == filt.id}
    assert in_edges == {t_lidar.id: "lidar", t_imu.id: "imu"}


def test_sample_level_step_is_a_barrier_joining_branches():
    ds = ArrayDataset(keys=("lidar", "imu"))
    ds.transform("lidar", lambda a: a, output="lidar_f")
    ds.transform("imu", lambda a: a, output="imu_f")
    ds.transform(lambda s: s)  # whole-sample: must join both branches
    ds.transform("lidar_f", lambda a: a)  # reads from the barrier, not the fork

    spec = graph.describe(ds)
    barrier = spec.nodes[3]
    after = spec.nodes[4]
    assert len([e for e in spec.edges if e.dst == barrier.id]) == 2
    (edge,) = [e for e in spec.edges if e.dst == after.id]
    assert edge.src == barrier.id and edge.label == "lidar_f"


def test_factory_closures_show_the_factory_name():
    def embed_labels(key):
        def fn(sample):
            return sample
        return fn

    ds = ArrayDataset()
    ds.transform(embed_labels("lidar"))
    ds.transform(lambda s: s)

    spec = graph.describe(ds)
    labels = [n.label for n in spec.nodes if n.kind == "transform"]
    assert labels[0] == "embed_labels"
    assert labels[1] == "test_factory_closures_show_the_factory_name"  # lambda


def test_concat_yields_two_roots():
    import apairo

    ds = apairo.ConcatDataset([ArrayDataset(n=2), ArrayDataset(n=3)])
    spec = graph.describe(ds)
    assert sorted(node_labels(spec)) == [
        "ArrayDataset",
        "ArrayDataset",
        "ConcatDataset",
    ]
    concat = next(n for n in spec.nodes if n.label == "ConcatDataset")
    assert {e.dst for e in spec.edges} == {concat.id}
    assert len(spec.edges) == 2


def test_cache_severs_the_chain_and_is_annotated():
    cached = ArrayDataset(n=4).filter(np.array([0, 1])).cache()
    spec = graph.describe(cached)
    assert node_labels(spec) == ["CachedDataset"]
    (node,) = spec.nodes
    assert "cache" in node.params["note"]


def test_shared_upstream_is_merged_across_datasets():
    base = ArrayDataset(n=5)
    v1 = base.filter(np.array([0, 1]))
    v2 = base.filter(np.array([2, 3, 4]))

    spec = graph.describe(v1, v2)
    assert node_labels(spec).count("ArrayDataset") == 1
    assert node_labels(spec).count("FilteredView") == 2
    assert len(spec.edges) == 2


def test_describe_requires_a_dataset():
    with pytest.raises(ValueError):
        graph.describe()


def test_to_dot_renders_nodes_edges_and_escapes_quotes():
    spec = graph.GraphSpec(
        nodes=[
            graph.Node("n0", "dataset", 'Raw"quoted"', {"frames": "2"}),
            graph.Node("n1", "transform", "fn"),
        ],
        edges=[graph.Edge("n0", "n1", "lidar")],
    )
    dot = graph.to_dot(spec)
    assert dot.startswith("digraph apairo_pipeline {")
    assert '\\"quoted\\"' in dot
    assert "frames: 2" in dot
    assert 'n0 -> n1 [label="lidar"];' in dot


def test_to_mermaid_renders_shapes_labels_and_classes():
    spec = graph.GraphSpec(
        nodes=[
            graph.Node("n0", "dataset", 'Raw"q"', {"frames": "2"}),
            graph.Node("n1", "transform", "fn"),
        ],
        edges=[graph.Edge("n0", "n1", "lidar")],
    )
    mmd = graph.to_mermaid(spec)
    assert mmd.startswith("flowchart TB")
    assert 'n0("Raw\'q\'<br/>frames: 2")' in mmd   # dataset box, quotes sanitised
    assert 'n1(["fn"])' in mmd                     # transform stadium
    assert "n0 -- lidar --> n1" in mmd
    assert "class n0 dataset" in mmd
    assert "class n1 transform" in mmd


def test_export_mermaid_file(tmp_path):
    out = graph.export(ArrayDataset(), tmp_path / "pipeline.mmd")
    assert out.read_text().startswith("flowchart TB")


def test_export_dot_file(tmp_path):
    out = graph.export(ArrayDataset(), tmp_path / "pipeline.dot")
    text = out.read_text()
    assert text.startswith("digraph apairo_pipeline {")
    assert "ArrayDataset" in text


def test_export_svg_uses_builtin_renderer(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")  # no graphviz needed for SVG anymore
    out = graph.export(ArrayDataset().filter(np.array([0])), tmp_path / "p.svg")
    text = out.read_text()
    assert text.startswith("<?xml")
    assert "FilteredView" in text and "ArrayDataset" in text


def test_export_html_is_selfcontained_and_interactive(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    ds = ArrayDataset()
    ds.transform("lidar", lambda a: a, output="lidar_f")
    out = graph.export(ds.filter(np.array([0])), tmp_path / "p.html")
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "lidar_f" in text                      # channel edge label present
    assert "prefers-color-scheme" in text         # dark theme
    assert "viewBox" in text and "wheel" in text  # pan/zoom wiring
    assert "http://" not in text.replace("http://www.w3.org", "")  # no CDN


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz not installed")
def test_export_png_via_dot_binary(tmp_path):
    out = graph.export(ArrayDataset().filter(np.array([0])), tmp_path / "p.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_png_without_dot_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="dot"):
        graph.export(ArrayDataset(), tmp_path / "p.png")


def test_builtin_layout_no_overlap_and_both_themes():
    from apairo_visu import graph_render

    base = ArrayDataset(n=4)
    merged = graph.describe(base.filter(np.array([0])), base.filter(np.array([1])))
    boxes, width, height = graph_render._layout(merged)
    assert len(boxes) == 3 and width > 0 and height > 0
    rects = list(boxes.values())
    for i, (x1, y1, w1, h1) in enumerate(rects):
        for x2, y2, w2, h2 in rects[i + 1:]:
            overlap = (x1 < x2 + w2 and x2 < x1 + w1
                       and y1 < y2 + h2 and y2 < y1 + h1)
            assert not overlap

    light = graph_render.to_svg(merged, theme="light")
    dark = graph_render.to_svg(merged, theme="dark")
    assert light != dark and "<svg" in light and "<svg" in dark
    with pytest.raises(ValueError):
        graph_render.to_svg(merged, theme="sepia")
