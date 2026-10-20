"""The Rerun backend runs apairo preprocessors by calling them."""

import types
import warnings

import numpy as np
import pytest

pytest.importorskip("rerun")
from apairo import FramePreprocessor, Sample, SequencePreprocessor  # noqa: E402

from apairo_visu.rerun import Pipeline, Preprocess  # noqa: E402


class Positive(FramePreprocessor):
    output_key = "positive"
    input_keys = ["lidar"]

    def __call__(self, sample):
        return (np.asarray(sample.data["lidar"])[:, 0] > 0).astype(np.int64)


class FrameIndex(SequencePreprocessor):
    output_key = "index"
    input_keys = ["lidar"]

    def __call__(self, frames):
        return np.arange(sum(1 for _ in frames))


def _samples(n=3):
    return [Sample(data={"lidar": np.full((2, 3), i - 1, np.float32)}) for i in range(n)]


def test_pipeline_calls_an_apairo_preprocessor():
    sample = Sample(data={"lidar": np.array([[-1, 0, 0], [2, 0, 0]], np.float32)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _, labels = Pipeline("p", [Positive()], label_key=None).run(sample)
    np.testing.assert_array_equal(labels, [0, 1])


def test_pipeline_still_runs_a_process_only_step():
    legacy = types.SimpleNamespace(
        input_keys=["lidar"],
        process=lambda s: np.ones(len(s.data["lidar"]), dtype=np.int64),
    )
    sample = Sample(data={"lidar": np.zeros((2, 3), np.float32)})
    _, labels = Pipeline("p", [legacy], label_key=None).run(sample)
    np.testing.assert_array_equal(labels, [1, 1])


def test_preprocess_calls_frame_and_sequence_preprocessors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        frame = Preprocess(Positive(), default=None).run(_samples(), [0, 2], key="pos")
        assert frame[0].data["pos"].tolist() == [0, 0]
        assert frame[1].data["pos"] is None  # not in the frames run
        assert frame[2].data["pos"].tolist() == [1, 1]

        seq = Preprocess(FrameIndex(), default=None).run(_samples(), [0, 1, 2], key="idx")
        assert [seq[i].data["idx"] for i in range(3)] == [0, 1, 2]
