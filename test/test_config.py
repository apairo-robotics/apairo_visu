"""Tests for ViewConfig and label config loading."""

import pytest

import apairo_visu
from apairo_visu import ViewConfig, load_label_config
from apairo_visu.config import BUILTIN_CONFIGS


def test_view_config_defaults():
    cfg = ViewConfig()
    assert cfg.point_key == "lidar"
    assert cfg.label_key == "labels"
    assert cfg.intensity_channel == 3


def test_importing_package_does_not_require_open3d():
    # The light layer must be importable headless; LidarViewer stays lazy.
    assert "open3d" not in dir(apairo_visu)
    assert callable(apairo_visu.load_label_config)


@pytest.mark.parametrize("name", sorted(BUILTIN_CONFIGS))
def test_builtin_label_configs_load(name):
    cfg = load_label_config(name)
    assert "color_map" in cfg
    assert "semantic_map" in cfg
    assert len(cfg["color_map"]) > 0


def test_lazy_lidar_viewer_attribute_error_for_unknown():
    with pytest.raises(AttributeError):
        _ = apairo_visu.does_not_exist


def test_open3d_is_an_optional_extra():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    assert not any(d.startswith("open3d") for d in project["dependencies"])
    assert "open3d" in project["optional-dependencies"]["open3d"]


def test_viewer_without_open3d_names_the_extra(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "open3d", None)  # import open3d -> ImportError
    monkeypatch.delitem(sys.modules, "apairo_visu.viewer", raising=False)
    with pytest.raises(ImportError, match=r"apairo-visu\[open3d\]"):
        importlib.import_module("apairo_visu.viewer")
