"""Tests for the studio: registry mapping and the FastAPI inspector API."""

import numpy as np
import pytest

from apairo_visu import graph

from test_graph import ArrayDataset

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from apairo_visu.studio.registry import StudioRegistry  # noqa: E402
from apairo_visu.studio.server import create_app  # noqa: E402


def build_chain():
    def double(arr):
        return arr * 2

    ds = ArrayDataset(n=4, keys=("lidar", "imu"))
    ds.transform("lidar", double, output="lidar_2x")
    return ds, ds.filter(np.array([0, 2]))


# --------------------------------------------------------------- registry


def test_describe_registry_maps_nodes_to_live_objects():
    base, view = build_chain()
    registry: dict = {}
    spec = graph.describe(view, registry=registry)

    assert set(registry) == {n.id for n in spec.nodes}
    by_label = {n.label: n.id for n in spec.nodes}
    assert registry[by_label["FilteredView"]] is view
    assert registry[by_label["ArrayDataset"]] is base
    owner, step_index = registry[by_label["double"]]
    assert owner is base and step_index == 0


def test_registry_detail_dataset_node():
    _, view = build_chain()
    reg = StudioRegistry([view])
    node_id = next(n.id for n in reg.spec.nodes if n.label == "FilteredView")

    detail = reg.detail(node_id)
    assert detail["kind"] == "dataset"
    assert detail["len"] == 2
    channels = {c["key"]: c for c in detail["channels"]}
    assert channels["lidar_2x"]["dtype"] == "float32"
    assert channels["lidar_2x"]["shape"] == [4, 3]
    assert "in-memory" in (detail["doc"] or "").lower() or detail["doc"]


def test_registry_detail_transform_node():
    _, view = build_chain()
    reg = StudioRegistry([view])
    node_id = next(n.id for n in reg.spec.nodes if n.label == "double")

    detail = reg.detail(node_id)
    assert detail["kind"] == "transform"
    assert detail["owner"] == "ArrayDataset"
    assert detail["step_index"] == 0
    assert detail["signature"] == "(arr)"


def test_registry_unknown_node_and_empty_datasets():
    _, view = build_chain()
    assert StudioRegistry([view]).detail("nope") is None
    with pytest.raises(ValueError):
        StudioRegistry([])


# --------------------------------------------------------------- REST API


@pytest.fixture()
def client():
    _, view = build_chain()
    return TestClient(create_app([view], title="studio test"))


def test_api_graph(client):
    payload = client.get("/api/graph").json()
    assert payload["title"] == "studio test"
    assert 'data-node="n0"' in payload["svg"]
    kinds = {n["kind"] for n in payload["nodes"]}
    assert kinds == {"dataset", "transform"}
    assert all({"src", "dst", "label"} <= set(e) for e in payload["edges"])


def test_api_node_detail_and_404(client):
    graph_payload = client.get("/api/graph").json()
    ds_node = next(n for n in graph_payload["nodes"] if n["label"] == "FilteredView")

    detail = client.get(f"/api/node/{ds_node['id']}").json()
    assert detail["kind"] == "dataset"
    assert any(c["key"] == "imu" for c in detail["channels"])

    assert client.get("/api/node/n999").status_code == 404


def test_protocol_roundtrip_and_decimation():
    from apairo_visu.studio import protocol

    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    decoded = protocol.decode_array(protocol.encode_array(arr))
    np.testing.assert_array_equal(decoded, arr)

    sub, original = protocol.decimate_rows(arr, 4)
    assert original == 8 and sub.shape == (4, 3)
    np.testing.assert_array_equal(sub, arr[::2])

    channels = protocol.encode_channels(
        {"pts": arr, "img": np.zeros((16, 16, 3), np.uint8), "weird": object()}, 4
    )
    assert channels["pts"]["full_rows"] == 8      # per-point: decimated
    assert "full_rows" not in channels["img"]     # image rows never dropped
    assert "repr" in channels["weird"]


def test_registry_sample_at_dataset_node():
    _, view = build_chain()
    reg = StudioRegistry([view])
    node_id = next(n.id for n in reg.spec.nodes if n.label == "FilteredView")

    result = reg.sample(node_id, 1)
    assert result["len"] == 2
    # FilteredView kept indices [0, 2]; frame 1 is parent frame 2, transformed.
    np.testing.assert_array_equal(
        result["data"]["lidar_2x"], np.full((4, 3), 4.0, np.float32)
    )
    with pytest.raises(IndexError):
        reg.sample(node_id, 2)
    assert reg.sample("nope", 0) is None


def test_registry_sample_at_step_prefix():
    def times_two(arr):
        return arr * 2

    def plus_one(arr):
        return arr + 1

    ds = ArrayDataset(n=3)
    ds.transform("lidar", times_two, output="lidar_2x")
    ds.transform("lidar_2x", plus_one)
    reg = StudioRegistry([ds])
    ids = {n.label: n.id for n in reg.spec.nodes}

    # Frame 1 raw is full of 1.0; after step 0 the new channel holds 2.0,
    # untouched by the later step; after step 1 it holds 3.0.
    after_first = reg.sample(ids["times_two"], 1)
    np.testing.assert_array_equal(
        after_first["data"]["lidar_2x"], np.full((4, 3), 2.0, np.float32)
    )
    after_second = reg.sample(ids["plus_one"], 1)
    np.testing.assert_array_equal(
        after_second["data"]["lidar_2x"], np.full((4, 3), 3.0, np.float32)
    )


def test_api_sample_endpoint(client):
    nodes = client.get("/api/graph").json()["nodes"]
    ds_node = next(n for n in nodes if n["label"] == "FilteredView")

    payload = client.get(f"/api/node/{ds_node['id']}/sample/0").json()
    assert payload["index"] == 0 and payload["len"] == 2
    lidar = payload["channels"]["lidar_2x"]
    assert lidar["dtype"] == "float32" and lidar["shape"] == [4, 3]

    filtered = client.get(
        f"/api/node/{ds_node['id']}/sample/0", params={"channels": "imu"}
    ).json()
    assert set(filtered["channels"]) == {"imu"}

    assert client.get(f"/api/node/{ds_node['id']}/sample/99").status_code == 400
    assert client.get("/api/node/n999/sample/0").status_code == 404


def test_api_sample_decimation():
    class BigDataset(ArrayDataset):
        def _load(self, idx):
            from apairo.core.sample import Sample
            return Sample(data={"lidar": np.zeros((1000, 3), np.float32)})

    app = create_app([BigDataset(n=2)], max_points=100)
    payload = TestClient(app).get("/api/node/n0/sample/0").json()
    lidar = payload["channels"]["lidar"]
    assert lidar["full_rows"] == 1000
    assert lidar["shape"][0] == 100


# ---------------------------------------------------------------- catalog


def _test_catalog():
    """A Catalog over a synthetic module (no dependency on the satellites)."""
    import types

    from apairo_visu.studio.catalog import Catalog

    mod = types.ModuleType("fake_transforms")

    class Scale:
        """Multiply the array by *factor*."""

        def __init__(self, factor: float = 2.0, clip: bool = False):
            self.factor, self.clip = factor, clip

        def __call__(self, arr):
            out = arr * self.factor
            return np.clip(out, 0, 1) if self.clip else out

    def shift(arr, offset=1.0):
        return arr + offset

    Scale.__module__ = shift.__module__ = "fake_transforms"
    mod.Scale, mod.shift = Scale, shift

    cat = Catalog(packages=())
    cat._scan_module(mod)
    return cat


def test_catalog_schema_apply_and_snippet():
    cat = _test_catalog()
    assert set(cat.entries) == {"fake_transforms.Scale", "fake_transforms.shift"}

    entry = cat.entries["fake_transforms.Scale"].to_dict()
    params = {p["name"]: p for p in entry["params"]}
    assert params["factor"]["type"] == "float" and params["factor"]["default"] == 2.0
    assert params["clip"]["type"] == "bool"

    arr = np.ones((4, 3), np.float32)
    out = cat.apply("fake_transforms.Scale", {"factor": "3"}, {"lidar": arr}, "lidar")
    np.testing.assert_array_equal(out, arr * 3)  # "3" coerced to float

    snippet = cat.snippet("fake_transforms.Scale", {"factor": 3.0}, "lidar")
    assert 'ds.transform("lidar", Scale(factor=3.0), output="lidar_scale")' in snippet
    assert "from fake_transforms import Scale" in snippet

    # offset's default is a float, so ints coerce to float in the snippet too.
    fn_snippet = cat.snippet("fake_transforms.shift", {"offset": 2}, "lidar")
    assert "partial(shift, offset=2.0)" in fn_snippet


def catalog_client():
    _, view = build_chain()
    app = create_app([view], title="studio test", catalog=_test_catalog())
    return TestClient(app)


def test_api_catalog_and_try():
    client = catalog_client()
    cat = client.get("/api/catalog").json()
    assert cat["modules"][0]["module"] == "fake_transforms"

    node = next(n for n in client.get("/api/graph").json()["nodes"]
                if n["label"] == "FilteredView")
    payload = {
        "entry_id": "fake_transforms.Scale",
        "node_id": node["id"],
        "channel": "lidar_2x",
        "kwargs": {"factor": 10},
        "index": 0,
    }
    result = client.post("/api/try", json=payload).json()
    assert "error" not in result
    from apairo_visu.studio.protocol import decode_array

    before = decode_array(result["before"])
    after = decode_array(result["after"])
    np.testing.assert_array_equal(after, before * 10)
    assert 'ds.transform("lidar_2x", Scale(factor=10.0)' in result["snippet"]

    # Application failures surface as data, not 500s.
    bad = client.post("/api/try", json={**payload, "kwargs": {"factor": "oops"}})
    assert bad.status_code == 200 and "error" in bad.json()

    assert client.post("/api/try", json={**payload, "entry_id": "nope"}).status_code == 400
    assert client.post("/api/try", json={**payload, "channel": "nope"}).status_code == 400


def test_real_satellite_catalog_available():
    pytest.importorskip("apairo_transform")
    from apairo_visu.studio.catalog import Catalog

    cat = Catalog(packages=("apairo_transform",))
    assert any(e.endswith(".RangeFilter") for e in cat.entries)
    arr = np.array([[1, 0, 0, 1], [100, 0, 0, 1]], np.float32)
    entry_id = next(e for e in cat.entries if e.endswith(".RangeFilter"))
    out = cat.apply(entry_id, {"max": 10}, {"pts": arr}, "pts")
    assert out.shape[0] == 1  # the 100 m point is dropped


# -------------------------------------------------------------- sequences


class SeqArrayDataset(ArrayDataset):
    """ArrayDataset that reports per-frame sequence ids like a root dataset."""

    class _Ref:
        def __init__(self, sequence, channel, row):
            self.sequence, self.channel, self.row = sequence, channel, row

    @property
    def frame_sequence_ids(self):
        return ["seq_a" if i < 2 else "seq_b" for i in range(self._n)]

    def frame_info(self, idx):
        return self._Ref("seq_a" if idx < 2 else "seq_b", "lidar", idx % 2)


def test_detail_exposes_sequence_ranges():
    reg = StudioRegistry([SeqArrayDataset(n=5)])
    node_id = reg.spec.nodes[0].id
    assert reg.detail(node_id)["sequences"] == [
        {"id": "seq_a", "start": 0, "stop": 2},
        {"id": "seq_b", "start": 2, "stop": 5},
    ]


def test_detail_without_sequence_info_has_no_sequences_key():
    reg = StudioRegistry([ArrayDataset(n=3)])
    node_id = reg.spec.nodes[0].id
    assert "sequences" not in reg.detail(node_id)


def test_sample_carries_frame_provenance():
    reg = StudioRegistry([SeqArrayDataset(n=5)])
    node_id = reg.spec.nodes[0].id
    assert reg.sample(node_id, 3)["frame"] == {
        "sequence": "seq_b", "channel": "lidar", "row": 1,
    }
    plain = StudioRegistry([ArrayDataset(n=3)])
    assert "frame" not in plain.sample(plain.spec.nodes[0].id, 0)


def test_synchronized_root_keeps_sequence_provenance(tmp_path):
    """The exact studio chain: multi-sequence root, synchronized per
    sequence then concatenated -- provenance must survive to the top."""
    import apairo

    for seq, n in (("seq_a", 3), ("seq_b", 2)):
        for ch in ("lidar", "imu"):
            d = tmp_path / seq / ch
            d.mkdir(parents=True)
            for i in range(n):
                shape = (20, 4) if ch == "lidar" else (6,)
                np.save(d / f"{i:06d}.npy", np.zeros(shape, dtype=np.float32))
            (d / "timestamps.txt").write_text(
                "".join(f"{0.1 * i}\n" for i in range(n)))

    ds = apairo.RawDataset(str(tmp_path)).synchronize(
        reference="lidar", method="nearest")
    if getattr(ds, "frame_sequence_ids", None) is None:
        pytest.skip("apairo without view provenance forwarding (< 0.6)")

    reg = StudioRegistry([ds])
    top = reg.spec.nodes[0].id
    assert reg.detail(top)["sequences"] == [
        {"id": "seq_a", "start": 0, "stop": 3},
        {"id": "seq_b", "start": 3, "stop": 5},
    ]
    assert reg.sample(top, 3)["frame"]["sequence"] == "seq_b"


def test_sample_endpoint_forwards_frame_provenance():
    app = create_app([SeqArrayDataset(n=5)])
    client = TestClient(app)
    node_id = app.state.registry.spec.nodes[0].id
    payload = client.get(f"/api/node/{node_id}/sample/0").json()
    assert payload["frame"]["sequence"] == "seq_a"
    detail = client.get(f"/api/node/{node_id}").json()
    assert [s["id"] for s in detail["sequences"]] == ["seq_a", "seq_b"]


# ----------------------------------------------------------------- series


class SignalDataset(ArrayDataset):
    """Frames carrying a 1-D signal channel and a per-point cloud."""

    def __init__(self, n: int = 5):
        super().__init__(n=n, keys=("imu", "lidar"))

    def _load(self, idx):
        from apairo.core.sample import Sample

        return Sample(data={
            "imu": np.array([idx, 10 * idx, -idx], dtype=np.float32),
            "lidar": np.full((30, 4), idx, dtype=np.float32),
        })


def test_series_indexes_1d_channels():
    reg = StudioRegistry([SignalDataset(n=4)])
    node_id = reg.spec.nodes[0].id
    result = reg.series(node_id, "imu", [0, 1])
    assert result["cols"]["0"] == [0.0, 1.0, 2.0, 3.0]
    assert result["cols"]["1"] == [0.0, 10.0, 20.0, 30.0]
    assert result["truncated"] is False


def test_series_reduces_cloud_columns_and_handles_bad_col():
    reg = StudioRegistry([SignalDataset(n=3)])
    node_id = reg.spec.nodes[0].id
    result = reg.series(node_id, "lidar", [2])
    assert result["cols"]["2"] == [0.0, 1.0, 2.0]  # per-frame column mean
    out_of_range = reg.series(node_id, "imu", [99])
    assert out_of_range["cols"]["99"] == [None, None, None]


def test_series_endpoint_range_and_errors():
    app = create_app([SignalDataset(n=5)])
    client = TestClient(app)
    node_id = app.state.registry.spec.nodes[0].id
    payload = client.get(
        f"/api/node/{node_id}/series/imu?cols=1&start=1&stop=3").json()
    assert payload["cols"]["1"] == [10.0, 20.0]
    assert client.get(f"/api/node/{node_id}/series/imu?cols=x").status_code == 400
    assert client.get("/api/node/nope/series/imu").status_code == 404


# ------------------------------------------------------- channel discovery


class AsyncishDataset(ArrayDataset):
    """One channel per frame, like an asynchronous raw timeline."""

    def __init__(self, n: int = 4, declared=("lidar", "camera")):
        super().__init__(n=n, keys=declared)

    def _load(self, idx):
        from apairo.core.sample import Sample

        data = ({"lidar": np.zeros((30, 4), np.float32)} if idx < 2
                else {"camera": np.zeros((8, 8, 3), np.uint8)})
        return Sample(data=data, timestamp=0.1 * idx)

    def frame_info(self, idx):
        class _Ref:
            sequence = None
            row = 0
        _Ref.channel = "lidar" if idx < 2 else "camera"
        return _Ref()


def test_channel_table_covers_channels_absent_at_frame_zero():
    reg = StudioRegistry([AsyncishDataset()])
    node_id = reg.spec.nodes[0].id
    channels = {c["key"]: c for c in reg.detail(node_id)["channels"]}
    assert set(channels) == {"lidar", "camera"}
    assert channels["camera"]["shape"] == [8, 8, 3]


def test_channel_table_lists_undiscoverable_keys_with_unknown_shape():
    reg = StudioRegistry([AsyncishDataset(declared=("lidar", "camera", "ghost"))])
    node_id = reg.spec.nodes[0].id
    channels = {c["key"]: c for c in reg.detail(node_id)["channels"]}
    assert channels["ghost"] == {"key": "ghost", "dtype": "?", "shape": None}


def test_frames_track_per_channel():
    reg = StudioRegistry([AsyncishDataset()])
    node_id = reg.spec.nodes[0].id
    assert reg.frames(node_id, "lidar")["indices"] == [0, 1]
    assert reg.frames(node_id, "camera")["indices"] == [2, 3]
    assert reg.frames(node_id, "nope")["indices"] == []


def test_frames_vector_track_matches_frame_info_on_real_root(tmp_path):
    """The vectorized timeline (frame_channel_ids) must agree with a
    frame_info scan, on a real multi-sequence async root -- the barakuda
    layout."""
    import apairo

    ds = apairo.RawDataset(str(_kitti_root(tmp_path)))
    reg = StudioRegistry([ds])
    node_id = reg.spec.nodes[0].id
    got = reg.frames(node_id, "camera")["indices"]
    expected = [i for i in range(len(ds))
                if getattr(ds.frame_info(i), "channel", None) == "camera"]
    assert got == expected and len(got) == 6  # both sequences covered


def _kitti_root(tmp_path, sequences=(("seq_a", 4), ("seq_b", 2))):
    """A barakuda-shaped root: sequences x channels of numbered .npy files."""
    for seq, n in sequences:
        for ch in ("lidar", "camera"):
            d = tmp_path / seq / ch
            d.mkdir(parents=True)
            for i in range(n):
                shape = (20, 4) if ch == "lidar" else (4, 6, 3)
                np.save(d / f"{i:06d}.npy", np.zeros(shape, dtype=np.float32))
            (d / "timestamps.txt").write_text(
                "".join(f"{0.1 * i}\n" for i in range(n)))
    return tmp_path


def test_frames_track_carries_file_stems(tmp_path):
    """The per-channel timeline names its files: a label lives in
    ``000002.npy``, and the flat index alone can never say so."""
    import apairo

    reg = StudioRegistry([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    node_id = reg.spec.nodes[0].id
    track = reg.frames(node_id, "lidar")
    # 4 frames in seq_a then 2 in seq_b -- each sequence restarts at 000000.
    assert track["stems"] == ["000000", "000001", "000002", "000003",
                              "000000", "000001"]
    assert len(track["indices"]) == len(track["stems"])


def test_sample_carries_the_file_stem(tmp_path):
    import apairo

    ds = apairo.RawDataset(str(_kitti_root(tmp_path)))
    reg = StudioRegistry([ds])
    node_id = reg.spec.nodes[0].id
    index = reg.frames(node_id, "lidar")["indices"][2]
    assert reg.sample(node_id, index)["frame"]["stem"] == "000002"


def test_entries_list_what_a_channel_holds(tmp_path):
    """The listing answers "which frames are in there", with the file that
    backs each one -- what no single sample can show."""
    import apairo

    reg = StudioRegistry([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    node_id = reg.spec.nodes[0].id

    listing = reg.entries(node_id, "lidar")
    assert listing["total"] == 6  # 4 frames in seq_a, 2 in seq_b
    assert listing["truncated"] is False
    first = listing["entries"][0]
    assert first["row"] == 0 and first["stem"] == "000000"
    assert first["sequence"] == "seq_a" and first["file"] == "000000.npy"
    assert first["bytes"] > 0 and first["timestamp"] == pytest.approx(0.0)
    # The index is what the frame slider seeks to.
    assert reg.sample(node_id, first["index"])["frame"]["channel"] == "lidar"
    # Rows restart at 0 in the next sequence, files and all.
    seq_b = [e for e in listing["entries"] if e["sequence"] == "seq_b"]
    assert [e["row"] for e in seq_b] == [0, 1]
    assert [e["file"] for e in seq_b] == ["000000.npy", "000001.npy"]


def test_entries_pages_and_filters_by_sequence(tmp_path):
    import apairo

    reg = StudioRegistry([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    node_id = reg.spec.nodes[0].id

    page = reg.entries(node_id, "lidar", start=1, stop=3)
    assert [e["pos"] for e in page["entries"]] == [1, 2]
    assert page["total"] == 6  # the total is the whole channel, not the page

    only_b = reg.entries(node_id, "lidar", sequence="seq_b")
    assert only_b["total"] == 2
    assert {e["sequence"] for e in only_b["entries"]} == {"seq_b"}

    assert reg.entries("nope", "lidar") is None


def test_entries_caps_the_page(tmp_path, monkeypatch):
    """Every row costs an os.stat, so a request is bounded and says so."""
    import apairo

    from apairo_visu.studio import registry as registry_module

    monkeypatch.setattr(registry_module, "_ENTRIES_CAP", 2)
    reg = StudioRegistry([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    listing = reg.entries(reg.spec.nodes[0].id, "lidar")
    assert listing["truncated"] is True
    assert len(listing["entries"]) == 2 and listing["total"] == 6


def test_entries_without_a_channel_timeline_lists_every_frame():
    """A synchronous dataset has no per-channel timeline: every frame carries
    the channel, so the listing is the frame axis itself."""
    reg = StudioRegistry([ArrayDataset(n=3, keys=("lidar",))])
    listing = reg.entries(reg.spec.nodes[0].id, "lidar")
    assert listing["total"] == 3
    assert [e["index"] for e in listing["entries"]] == [0, 1, 2]
    # No loader to name: file/bytes are simply absent, not an error.
    assert listing["entries"][0].get("file") is None


def test_entries_endpoint(tmp_path):
    import apairo

    app = create_app([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    client = TestClient(app)
    node_id = app.state.registry.spec.nodes[0].id

    payload = client.get(
        f"/api/node/{node_id}/entries/lidar",
        params={"start": 0, "stop": 2, "sequence": "seq_a"},
    ).json()
    assert payload["total"] == 4
    assert [e["file"] for e in payload["entries"]] == ["000000.npy", "000001.npy"]
    assert client.get("/api/node/nope/entries/lidar").status_code == 404


def test_locate_maps_a_stem_back_to_frame_indices(tmp_path):
    import apairo

    ds = apairo.RawDataset(str(_kitti_root(tmp_path)))
    reg = StudioRegistry([ds])
    node_id = reg.spec.nodes[0].id

    # Stems repeat across sequences and channels: unnarrowed, every match.
    every = reg.locate(node_id, "000001")["matches"]
    assert {(m["sequence"], m["channel"]) for m in every} == {
        ("seq_a", "lidar"), ("seq_a", "camera"),
        ("seq_b", "lidar"), ("seq_b", "camera"),
    }
    # Narrowed, exactly the one frame the user meant.
    one = reg.locate(node_id, "000001", channel="lidar", sequence="seq_b")["matches"]
    assert len(one) == 1
    ref = ds.frame_info(one[0]["index"])
    assert (ref.sequence, ref.channel, ref.row) == ("seq_b", "lidar", 1)

    assert reg.locate(node_id, "999999")["matches"] == []
    assert reg.locate("nope", "000001") is None


def test_locate_endpoint(tmp_path):
    import apairo

    app = create_app([apairo.RawDataset(str(_kitti_root(tmp_path)))])
    client = TestClient(app)
    node_id = app.state.registry.spec.nodes[0].id

    payload = client.get(
        f"/api/node/{node_id}/locate/000003",
        params={"channel": "lidar", "sequence": "seq_a"},
    ).json()
    assert [m["channel"] for m in payload["matches"]] == ["lidar"]
    assert client.get("/api/node/nope/locate/000003").status_code == 404


def test_locate_without_stems_reports_it():
    reg = StudioRegistry([ArrayDataset(n=3)])
    result = reg.locate(reg.spec.nodes[0].id, "000000")
    assert result["matches"] == [] and "error" in result


def test_frames_endpoint_and_sample_timestamp():
    app = create_app([AsyncishDataset()])
    client = TestClient(app)
    node_id = app.state.registry.spec.nodes[0].id
    assert client.get(f"/api/node/{node_id}/frames/camera").json()["indices"] == [2, 3]
    assert client.get("/api/node/nope/frames/camera").status_code == 404
    payload = client.get(f"/api/node/{node_id}/sample/2").json()
    assert payload["timestamp"] == pytest.approx(0.2)
    # Channel filter keeps the timestamp (the hold-last path relies on it).
    filtered = client.get(f"/api/node/{node_id}/sample/2?channels=camera").json()
    assert set(filtered["channels"]) == {"camera"}
    assert filtered["timestamp"] == pytest.approx(0.2)


# --------------------------------------------------------------- CLI plugin


def test_cli_registered_as_apairo_plugin():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="apairo.cli_plugins")}
    assert eps.get("studio") == "apairo_visu.studio.cli:main"


def test_cli_rejects_missing_directory(tmp_path):
    from apairo_visu.studio import cli

    assert cli.main([str(tmp_path / "nope")]) == 1


def test_cli_builds_dataset_and_launches(tmp_path, monkeypatch):
    import apairo

    from apairo_visu import studio
    from apairo_visu.studio import cli

    built, launched = {}, {}

    def fake_raw(directory, keys=None):
        built.update(directory=directory, keys=keys)
        return "DS"

    monkeypatch.setattr(apairo, "RawDataset", fake_raw)
    monkeypatch.setattr(
        studio, "launch", lambda *ds, **kw: launched.update(datasets=ds, **kw)
    )

    rc = cli.main(
        [str(tmp_path), "--keys", "lidar", "labels",
         "--port", "9000", "--no-browser", "--title", "custom"]
    )
    assert rc == 0
    assert built == {"directory": str(tmp_path), "keys": ["lidar", "labels"]}
    assert launched["datasets"] == ("DS",)
    assert launched["port"] == 9000
    assert launched["open_browser"] is False
    assert launched["title"] == "custom"


def test_cli_skips_channels_that_fail_to_load(tmp_path, monkeypatch, capsys):
    """When loading every channel at once fails, studio probes each channel and
    serves the ones that load, reporting the ones it dropped."""
    import apairo

    from apairo_visu import studio
    from apairo_visu.studio import cli

    def fake_raw(directory, keys=None):
        if keys is None:  # the initial all-channels load
            raise ValueError("boom: full load")
        if keys == ["bad"]:  # a malformed channel, probed alone
            raise ValueError("bad channel is malformed")
        if "bad" in keys:  # never reached: 'bad' is dropped before the final load
            raise AssertionError("dropped channel must not reach the final load")
        return "DS"

    monkeypatch.setattr(apairo, "RawDataset", fake_raw)
    monkeypatch.setattr(cli, "_declared_keys", lambda root: ["lidar", "imu", "bad"])
    launched = {}
    monkeypatch.setattr(
        studio, "launch", lambda *ds, **kw: launched.update(datasets=ds)
    )

    assert cli.main([str(tmp_path), "--no-browser"]) == 0
    assert launched["datasets"] == ("DS",)
    err = capsys.readouterr().err
    assert "skipped 1 channel" in err
    assert "bad: ValueError: bad channel is malformed" in err


def test_cli_errors_when_no_channel_loads(tmp_path, monkeypatch):
    import apairo

    from apairo_visu.studio import cli

    def fake_raw(directory, keys=None):
        raise ValueError("everything is broken")

    monkeypatch.setattr(apairo, "RawDataset", fake_raw)
    monkeypatch.setattr(cli, "_declared_keys", lambda root: ["a", "b"])
    assert cli.main([str(tmp_path), "--no-browser"]) == 1


def test_cli_sync_on_dropped_channel_errors(tmp_path, monkeypatch):
    import apairo

    from apairo_visu.studio import cli

    def fake_raw(directory, keys=None):
        if keys is None or keys == ["lidar"]:
            raise ValueError("lidar is malformed")
        return "DS"

    monkeypatch.setattr(apairo, "RawDataset", fake_raw)
    monkeypatch.setattr(cli, "_declared_keys", lambda root: ["lidar", "imu"])
    assert cli.main([str(tmp_path), "--sync", "lidar", "--no-browser"]) == 1


def test_declared_keys_reads_sequence_manifest(tmp_path):
    import yaml

    from apairo_visu.studio import cli

    apairo_dir = tmp_path / ".apairo"
    apairo_dir.mkdir()
    (apairo_dir / "channels.yaml").write_text(
        yaml.safe_dump(
            {"channels": {"lidar": {"loader": "npys"}, "imu": {"loader": "npy"}}}
        )
    )
    assert cli._declared_keys(tmp_path) == ["imu", "lidar"]


def test_declared_keys_unions_root_sequences(tmp_path):
    import yaml

    from apairo_visu.studio import cli

    for seq, chans in (("seq_a", ["lidar"]), ("seq_b", ["lidar", "camera"])):
        d = tmp_path / seq / ".apairo"
        d.mkdir(parents=True)
        (d / "channels.yaml").write_text(
            yaml.safe_dump({"channels": {c: {"loader": "npys"} for c in chans}})
        )
    assert cli._declared_keys(tmp_path) == ["camera", "lidar"]


def test_cli_sync_option(tmp_path, monkeypatch):
    import apairo

    from apairo_visu import studio
    from apairo_visu.studio import cli

    calls, launched = {}, {}

    class FakeDs:
        def synchronize(self, reference=None, method=None, tolerance=None):
            calls.update(reference=reference, method=method, tolerance=tolerance)
            return "SYNCED"

    monkeypatch.setattr(apairo, "RawDataset", lambda d, keys=None: FakeDs())
    monkeypatch.setattr(
        studio, "launch", lambda *ds, **kw: launched.update(datasets=ds)
    )

    rc = cli.main([str(tmp_path), "--sync", "lidar", "--tolerance", "0.05"])
    assert rc == 0
    assert calls == {"reference": "lidar", "method": "nearest", "tolerance": 0.05}
    assert launched["datasets"] == ("SYNCED",)


def test_cli_default_title_uses_directory_name(tmp_path, monkeypatch):
    import apairo

    from apairo_visu import studio
    from apairo_visu.studio import cli

    launched = {}
    monkeypatch.setattr(apairo, "RawDataset", lambda d, keys=None: "DS")
    monkeypatch.setattr(
        studio, "launch", lambda *ds, **kw: launched.update(**kw)
    )

    root = tmp_path / "barakuda_kitti"
    root.mkdir()
    assert cli.main([str(root), "--no-browser"]) == 0
    assert launched["title"] == "apairo studio -- barakuda_kitti"


def test_static_front_served_with_ssr_graph(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "apairo studio" in index.text
    assert 'data-node="n0"' in index.text  # SSR-lite: graph in first paint
    assert client.get("/src/app.js").status_code == 200
    assert client.get("/src/resize.js").status_code == 200
    assert client.get("/src/entriespanel.js").status_code == 200
    assert client.get("/style.css").status_code == 200
