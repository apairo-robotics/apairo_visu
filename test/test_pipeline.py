"""Tests for Pipeline -- plain callables and the apairo FramePreprocessor bridge."""

import warnings

import numpy as np
import pytest

from apairo_visu import Pipeline


def test_empty_pipeline_is_identity():
    pts = np.random.rand(10, 4).astype(np.float32)
    labels = np.zeros(10, dtype=np.int64)
    out_pts, out_labels = Pipeline("Raw").run(pts, labels)
    np.testing.assert_array_equal(out_pts, pts)
    np.testing.assert_array_equal(out_labels, labels)


def test_callable_steps_run_in_order():
    def drop_first(pts, labels):
        return pts[1:], (labels[1:] if labels is not None else None)

    pts = np.arange(12).reshape(4, 3).astype(np.float32)
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    out_pts, out_labels = Pipeline("p", [drop_first, drop_first]).run(pts, labels)
    assert len(out_pts) == 2
    np.testing.assert_array_equal(out_labels, [2, 3])


def test_frame_idx_passed_when_step_accepts_it():
    seen = {}

    def record(pts, labels, frame_idx=0):
        seen["idx"] = frame_idx
        return pts, labels

    Pipeline("p", [record]).run(np.zeros((2, 3)), None, frame_idx=7)
    assert seen["idx"] == 7


def test_step_without_frame_idx_still_works():
    def no_kw(pts, labels):
        return pts, labels

    out, _ = Pipeline("p", [no_kw]).run(np.ones((3, 3)), None, frame_idx=5)
    assert out.shape == (3, 3)


def test_frame_preprocessor_bridge_replaces_labels():
    # A FramePreprocessor-like object: detected structurally via input_keys,
    # called on a real apairo.Sample, its returned array becomes the labels.
    class FakeSegPreprocessor:
        input_keys = ["lidar"]

        def __call__(self, sample):
            pts = sample.data["lidar"]
            # label points by sign of x
            return (pts[:, 0] > 0).astype(np.int64)

    pts = np.array([[-1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    out_pts, out_labels = Pipeline("seg", [FakeSegPreprocessor()]).run(pts, None)
    np.testing.assert_array_equal(out_pts, pts)          # points untouched
    np.testing.assert_array_equal(out_labels, [0, 1, 1])  # labels from the call


def test_frame_preprocessor_receives_frame_index():
    class StatefulProc:
        input_keys = ["lidar"]
        _idx = -1

        def __call__(self, sample):
            return np.full(len(sample.data["lidar"]), self._idx, dtype=np.int64)

    proc = StatefulProc()
    _, labels = Pipeline("p", [proc]).run(np.zeros((4, 3)), None, frame_idx=42)
    assert proc._idx == 42
    np.testing.assert_array_equal(labels, [42, 42, 42, 42])


def test_process_only_step_still_runs():
    # The protocol before apairo 0.5: input_keys and a process method, no __call__.
    class LegacyProc:
        input_keys = ["lidar"]

        def process(self, sample):
            return np.ones(len(sample.data["lidar"]), dtype=np.int64)

    _, labels = Pipeline("p", [LegacyProc()]).run(np.zeros((2, 3)), None)
    np.testing.assert_array_equal(labels, [1, 1])


def test_apairo_preprocessor_runs_without_deprecation():
    # Calling .process() on an apairo preprocessor goes through its deprecated
    # alias: the pipeline calls the instance instead.
    pytest.importorskip("apairo")
    from apairo import FramePreprocessor

    class Positive(FramePreprocessor):
        output_key = "positive"
        input_keys = ["lidar"]

        def __call__(self, sample):
            return (sample.data["lidar"][:, 0] > 0).astype(np.int64)

    pts = np.array([[-1, 0, 0], [2, 0, 0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _, labels = Pipeline("p", [Positive()]).run(pts, None)
    np.testing.assert_array_equal(labels, [0, 1])
