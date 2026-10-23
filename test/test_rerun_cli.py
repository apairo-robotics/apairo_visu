"""`apairo rerun` with no channel lists every channel the dataset has."""

import numpy as np
import pytest

pytest.importorskip("rerun")
from apairo_visu.rerun.cli import main  # noqa: E402


def test_no_channel_lists_every_channel_of_an_async_dataset(tmp_path, capsys):
    # The IMU fires first, so the timeline's first event holds the IMU only.
    seq = tmp_path / "seq"
    for ch, shape, t0 in (("imu", (6,), 0.0), ("lidar", (20, 4), 0.05)):
        d = seq / ch
        d.mkdir(parents=True)
        for i in range(3):
            np.save(d / f"{i:06d}.npy", np.zeros(shape, dtype=np.float32))
        (d / "timestamps.txt").write_text("".join(f"{t0 + 0.1 * i}\n" for i in range(3)))

    assert main([str(seq)]) == 2
    assert "imu, lidar" in capsys.readouterr().out
