"""What the Rerun backend logs: each point keeps its own class id and colour,
and the views include every entity they are meant to show."""

import numpy as np
import pytest

rr = pytest.importorskip("rerun")
from apairo import Sample  # noqa: E402

import apairo_visu.rerun.viewer as rerun_viewer  # noqa: E402
from apairo_visu.rerun import KeyColormap, Pipeline, view  # noqa: E402


@pytest.fixture()
def logged(monkeypatch):
    """Every (entity path, archetype) the backend logs."""
    records = []
    real_log = rr.log

    def record(path, obj, *args, **kwargs):
        records.append((path, obj))
        return real_log(path, obj, *args, **kwargs)

    monkeypatch.setattr(rerun_viewer.rr, "log", record)
    return records


def _clouds(records):
    return [obj for _, obj in records if isinstance(obj, rr.Points3D)]


def _column(batch):
    return np.asarray(batch.as_arrow_array().to_pylist())


class _Scalars:
    """Two frames of 100 points; point k sits at x = k and its "height"
    channel holds k."""

    def __len__(self):
        return 2

    def __getitem__(self, i):
        pts = np.zeros((100, 4), np.float32)
        pts[:, 0] = np.arange(100)
        return Sample(data={"lidar": pts, "height": np.arange(100, dtype=np.float32)})


def test_max_points_keeps_each_points_own_colour(logged, tmp_path):
    view(_Scalars(), save=str(tmp_path / "rec.rrd"), max_points=10,
         pipelines=[Pipeline("p", label_key=None,
                             colormap_fn=KeyColormap("height", vmin=0, vmax=99))])
    cloud = _clouds(logged)[0]
    x = _column(cloud.positions)[:, 0]
    colours = _column(cloud.colors)
    assert len(x) == len(colours) == 10
    # red_blue: the blue byte grows with the scalar, which equals x here.
    rgba = np.array([[(c >> s) & 0xFF for s in (24, 16, 8, 0)] for c in colours.ravel()])
    expected_blue = (255 * x / 99).astype(np.uint8)
    np.testing.assert_array_equal(rgba[:, 2], expected_blue)


def test_a_colormap_of_the_wrong_length_is_an_error(tmp_path):
    def short(pts):
        return np.zeros((len(pts) - 1, 3), np.uint8)

    with pytest.raises(ValueError, match="99 colours for 100 points"):
        view(_Scalars(), save=str(tmp_path / "rec.rrd"),
             pipelines=[Pipeline("p", label_key=None, colormap_fn=short)])


def test_trajectory_is_part_of_every_3d_view(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(rerun_viewer.rr, "send_blueprint", lambda bp, *a, **k: sent.append(bp))

    class WithPose(_Scalars):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            pose = np.eye(4)
            pose[0, 3] = i
            sample.data["pose"] = pose
            return sample

    view(WithPose(), save=str(tmp_path / "rec.rrd"), pose_key="pose",
         pipelines=[Pipeline("a", label_key=None), Pipeline("b", label_key=None)])
    (blueprint,) = sent
    views = blueprint.root_container.contents
    assert len(views) == 2
    for v in views:
        # /world/trajectory and /world/robot live outside the view's origin.
        assert "/world/**" in v.contents


def test_range_filters_labels_with_their_points(logged, tmp_path):
    pytest.importorskip("apairo_transform")
    from apairo_visu.rerun.cli import main

    seq = tmp_path / "seq"
    for ch in ("lidar", "labels"):
        (seq / ch).mkdir(parents=True)
        (seq / ch / "timestamps.txt").write_text("".join(f"{0.1 * i}\n" for i in range(3)))
    for i in range(3):
        # point k: label k, at x = k -- except the first ten, moved 100 m out.
        pts = np.zeros((40, 4), np.float32)
        pts[:, 0] = np.arange(40)
        pts[:10, 0] += 100
        np.save(seq / "lidar" / f"{i:06d}.npy", pts)
        np.save(seq / "labels" / f"{i:06d}.npy", np.arange(40, dtype=np.int64))

    assert main([str(seq), "--lidar", "lidar", "--labels", "labels",
                 "--range", "50", "--save", str(tmp_path / "rec.rrd")]) == 0
    cloud = _clouds(logged)[0]
    x = _column(cloud.positions)[:, 0]
    ids = _column(cloud.class_ids).ravel()
    assert len(x) == len(ids) == 30
    np.testing.assert_array_equal(ids, x.astype(int))


def test_labels_of_the_wrong_length_are_an_error(tmp_path):
    class Mismatched(_Scalars):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample.data["labels"] = np.zeros(99, np.int64)
            return sample

    with pytest.raises(ValueError, match="99 labels"):
        view(Mismatched(), save=str(tmp_path / "rec.rrd"), pipelines=[Pipeline("p")])
