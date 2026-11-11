"""LidarViewer logic, with Open3D replaced by stubs (nothing is rendered).

The viewer's state handling -- what each viewport holds after a frame, what the
panel says -- is plain Python; only its widgets and scene come from Open3D.
"""

import sys
from concurrent.futures import Future
from unittest.mock import MagicMock

import numpy as np
import pytest

from apairo_visu import Pipeline


class _Label:
    """Stand-in for gui.Label: keeps its own text."""

    def __init__(self, text=""):
        self.text = text


class _InlineExecutor:
    """Runs jobs at once but, like ThreadPoolExecutor, keeps a job's exception
    inside the returned future -- which the viewer never reads."""

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


@pytest.fixture()
def viewer_module(monkeypatch):
    o3d, gui, rendering = MagicMock(), MagicMock(), MagicMock()
    visualization = MagicMock(gui=gui, rendering=rendering)
    o3d.visualization = visualization
    for name, module in (
        ("open3d", o3d),
        ("open3d.visualization", visualization),
        ("open3d.visualization.gui", gui),
        ("open3d.visualization.rendering", rendering),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    gui.Application.instance.post_to_main_thread.side_effect = lambda win, fn: fn()
    gui.Label.side_effect = _Label
    sys.modules.pop("apairo_visu.viewer", None)
    import apairo_visu.viewer as viewer

    drawn = []
    monkeypatch.setattr(viewer, "make_point_cloud", lambda xyz, colors: drawn.append(len(xyz)))
    viewer.drawn = drawn
    yield viewer
    # Never leave the stubbed module behind for other tests.
    sys.modules.pop("apairo_visu.viewer", None)
    import apairo_visu
    if "viewer" in vars(apairo_visu):
        delattr(apairo_visu, "viewer")


class _Frames:
    """Three frames; every lidar value equals the frame index."""

    def __len__(self):
        return 3

    def __getitem__(self, i):
        from apairo import Sample

        return Sample(data={
            "lidar": np.full((5, 4), float(i), np.float32),
            "labels": np.array([0, 1, 1, 2, 255]),
        })


def _open(viewer_module, **kwargs):
    viewer = viewer_module.LidarViewer(_Frames(), **kwargs)
    viewer._executor = _InlineExecutor()
    viewer._build_window()
    return viewer


def test_a_failing_pipeline_says_so_and_drops_the_previous_cloud(viewer_module):
    def fails_on_frame_1(pts, labels):
        if pts[0, 0] == 1.0:
            raise RuntimeError("model failed on frame 1")
        return pts, labels

    viewer = _open(viewer_module, label_cfg=None,
                   pipelines=[Pipeline("model", [fails_on_frame_1])])
    assert viewer._cached_pts[0][0, 0] == 0.0
    viewer._on_next()
    assert viewer._lbl_frame.text == "2 / 3"
    # Not frame 0's cloud under frame 1's number, and not a silent "…".
    assert viewer._cached_pts[0] is None
    assert "RuntimeError: model failed on frame 1" in viewer._timing_labels[0].text
    viewer._scenes[0].scene.remove_geometry.assert_called_with("cloud")


@pytest.mark.parametrize("label_cfg", [
    {"color_map": {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}},
    {"color_map": {0: [255, 0, 0], 1: [0, 255, 0]}, "semantic_map": {0: "road", 1: "grass"}},
])
def test_semantic_mode_keeps_labels_the_class_list_does_not_offer(viewer_module, label_cfg):
    # Labels 2 and 255 are absent from semantic_map: the class filter has no
    # checkbox for them, so they must not be hidden.
    viewer = _open(viewer_module, label_cfg=label_cfg)
    assert viewer_module.DISPLAY_MODES[viewer._display_mode] == "Semantic"
    assert viewer_module.drawn == [5]


def test_semantic_mode_hides_an_unticked_class(viewer_module):
    viewer = _open(viewer_module, label_cfg={
        "color_map": {0: [255, 0, 0], 1: [0, 255, 0]},
        "semantic_map": {0: "road", 1: "grass"},
    })
    viewer_module.drawn.clear()
    viewer._on_class_toggle(1, False)
    assert viewer_module.drawn == [3]  # the two points of class 1 are hidden
