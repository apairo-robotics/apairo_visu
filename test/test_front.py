"""The studio front's decoding and drawing helpers, run under Node.

The modules are plain ES modules with no DOM dependency at import time, so
Node can load them as shipped; a recording stand-in replaces the canvas.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "apairo_visu" / "studio" / "web" / "src"

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="needs Node.js")

HARNESS = """
const { decodeArray } = await import(%(arrays)s);
const views = await import(%(views)s);
function fakeCanvas(w = 480, h = 160) {
  const rects = [];
  const ctx = {
    clearRect() {}, fillText() {}, measureText: () => ({ width: 10 }),
    fillRect: (x, y, bw, bh) => rects.push([x, y, bw, bh]),
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData(img) { ctx.img = img; },
    set fillStyle(v) {}, set font(v) {},
  };
  return { width: w, height: h, getContext: () => ctx, rects, ctx };
}
const enc = (typed, dtype, shape) =>
  ({ dtype, shape, data: Buffer.from(typed.buffer).toString("base64") });
const out = {};

// int64 labels 0..19 (BigInt64Array once decoded)
const labels = new BigInt64Array(2000);
for (let i = 0; i < 2000; i++) labels[i] = BigInt(i %% 20);
const hist = fakeCanvas();
views.drawHist(hist, decodeArray(enc(labels, "int64", [2000])), "#000", "#000");
out.histBars = hist.rects.filter((r) => r[3] > 0).length;

// int64 raster of class ids 0..9
const ids = new BigInt64Array(1600);
for (let i = 0; i < 1600; i++) ids[i] = BigInt(i %% 10);
const raster = fakeCanvas();
views.drawRaster(raster, decodeArray(enc(ids, "int64", [40, 40])));
const colours = new Set();
for (let i = 0; i < 10; i++) colours.add(Array.from(raster.ctx.img.data.slice(i * 4, i * 4 + 3)).join());
out.rasterColours = colours.size;

// float16 [1, 2, 3] and bool [true, false, true]
out.float16 = Array.from(decodeArray(enc(new Uint16Array([0x3c00, 0x4000, 0x4200]), "float16", [3])).data);
out.bool = Array.from(decodeArray(enc(new Uint8Array([1, 0, 1]), "bool", [3])).data);
try { decodeArray(enc(new Uint8Array(4), "complex64", [1])); out.unknown = "decoded"; }
catch (err) { out.unknown = String(err.message); }

// float images: [0, 1] and already 0..255
for (const [name, value] of [["imageUnit", 0.8], ["image255", 204]]) {
  const c = fakeCanvas();
  views.drawImage(c, decodeArray(enc(new Float32Array(12).fill(value), "float32", [2, 2, 3])));
  out[name] = Array.from(c.ctx.img.data.slice(0, 4));
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def front(tmp_path_factory):
    script = tmp_path_factory.mktemp("front") / "harness.mjs"
    script.write_text(HARNESS % {
        "arrays": json.dumps((SRC / "arrays.js").as_uri()),
        "views": json.dumps((SRC / "views.js").as_uri()),
    })
    run = subprocess.run([node, str(script)], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def test_int64_channels_use_their_full_range(front):
    # BigInt values are not Number.isFinite: the range used to collapse to
    # [0, 1], so 20 label values drew 2 histogram bars and 10 class ids 2 colours.
    assert front["histBars"] == 20
    assert front["rasterColours"] == 10


def test_float16_and_bool_decode_to_their_values(front):
    assert front["float16"] == [1, 2, 3]
    assert front["bool"] == [1, 0, 1]


def test_an_unknown_dtype_is_refused_not_read_as_bytes(front):
    assert "unsupported dtype complex64" in front["unknown"]


def test_float_images_in_unit_range_are_scaled(front):
    assert front["imageUnit"] == [204, 204, 204, 255]
    assert front["image255"] == [204, 204, 204, 255]
