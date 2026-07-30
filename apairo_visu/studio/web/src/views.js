// Channel preview renderers -- all drawing happens client-side on raw arrays.
// One function per data kind; `previewKind` picks by shape/dtype.

import { categorical, finiteRange, rgbCss, viridis } from "./colors.js";

export function previewKind(shape, dtype) {
  if (shape.length === 3 && (shape[2] === 3 || shape[2] === 4)) return "image";
  // Per-point arrays with xyz columns open in the 3D viewer; 2-column
  // point lists fall back to the top-down BEV scatter.
  if (shape.length === 2 && shape[1] <= 32 && shape[0] > 16)
    return shape[1] >= 3 ? "cloud3d" : "cloud";
  if (shape.length === 2) return "raster";
  if (shape.length === 1 && shape[0] <= 32) return "values";
  if (shape.length === 1) return "hist";
  return "values";
}

export function stats(data) {
  let lo = Infinity, hi = -Infinity, sum = 0, n = 0;
  for (let i = 0; i < data.length; i++) {
    const v = Number(data[i]);
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    n += 1;
  }
  if (!n) return null;
  return { min: lo, max: hi, mean: sum / n };
}

const fmt = (v) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)
  ? v.toExponential(2) : +v.toFixed(3));

/* --------------------------------------------------------------- cloud */

// [lo, hi] percentile bounds of a sampled column -- robust to the outliers
// lidar clouds always carry (invalid returns at the origin, far echoes).
export function percentileBounds(arr, n, stride, col, loP = 0.02, hiP = 0.98) {
  const sampled = [];
  const step = Math.max(1, Math.floor(n / 5000));
  for (let i = 0; i < n; i += step) {
    const v = Number(arr[i * stride + col]);
    if (Number.isFinite(v)) sampled.push(v);
  }
  if (!sampled.length) return null;
  sampled.sort((a, b) => a - b);
  const lo = sampled[Math.floor(loP * (sampled.length - 1))];
  const hi = sampled[Math.floor(hiP * (sampled.length - 1))];
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

// Default BEV view of an (N, C) cloud on a w×h canvas: centred on the
// robust percentile bounds, fitting the larger span. {cx, cy} in world
// units, scale in px per world unit.
export function cloudFit(arr, w, h) {
  const [n, stride] = arr.shape;
  const bx = percentileBounds(arr.data, n, stride, 0);
  const by = percentileBounds(arr.data, n, stride, 1);
  if (!bx || !by) return null;
  const span = Math.max(bx[1] - bx[0], by[1] - by[0], 1e-6);
  return {
    cx: (bx[0] + bx[1]) / 2,
    cy: (by[0] + by[1]) / 2,
    scale: (Math.min(w, h) - 16) / span,
  };
}

// Robust [lo, hi] of a plain values array (2%-98% percentiles, sampled).
function percentileOfValues(values) {
  const sampled = [];
  const step = Math.max(1, Math.floor(values.length / 5000));
  for (let i = 0; i < values.length; i += step) {
    if (Number.isFinite(values[i])) sampled.push(values[i]);
  }
  if (!sampled.length) return [0, 1];
  sampled.sort((a, b) => a - b);
  const lo = sampled[Math.floor(0.02 * (sampled.length - 1))];
  const hi = sampled[Math.floor(0.98 * (sampled.length - 1))];
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

// Per-point color values for an (N, C) cloud: one of its own columns
// (key = number or "2"), or another per-point channel of the same sample
// (key = "ch:labels"). Label-like integer channels color categorically,
// everything else viridis over the robust range. Returns
// {values, categorical, label} or null when the key cannot be resolved.
export function cloudColorValues(arr, key, channels = null) {
  const [n, stride] = arr.shape;
  if (typeof key === "number" || /^\d+$/.test(String(key))) {
    const col = Math.min(Number(key), stride - 1);
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = Number(arr.data[i * stride + col]);
    return { values, categorical: false, label: `col ${col}` };
  }
  const name = String(key).replace(/^ch:/, "");
  const ch = channels?.[name];
  if (!ch || !ch.shape) return null;
  const rows = ch.shape[0];
  const chStride = ch.shape.length === 2 ? ch.shape[1] : 1;
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    values[i] = i < rows ? Number(ch.data[i * chStride]) : NaN;
  }
  return { values, categorical: /^u?int/.test(ch.dtype), label: name };
}

// value -> [r,g,b] mapper for a color spec (shared by BEV and 3D).
export function cloudColorMapper(spec) {
  if (spec.categorical) return (v) => categorical(v);
  const [lo, hi] = percentileOfValues(spec.values);
  return (v) => viridis((v - lo) / (hi - lo));
}

// Top-down scatter of an (N, C) per-point array. x = col 0, y = col 1;
// color by a spec from cloudColorValues (a plain column index is accepted
// and resolved for callers without external channels). `view` overrides
// the default fit ({cx, cy, scale} -- see cloudFit) so callers can
// zoom/pan.
export function drawCloud(canvas, arr, color, view = null, size = 1.7) {
  const [n, stride] = arr.shape;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const v = view ?? cloudFit(arr, w, h);
  if (!v) return;
  const { cx, cy, scale } = v;

  const spec = color && color.values
    ? color
    : cloudColorValues(arr, Number(color) || 0);
  const toColor = cloudColorMapper(spec);

  for (let i = 0; i < n; i++) {
    const x = arr.data[i * stride], y = arr.data[i * stride + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const sx = w / 2 + (x - cx) * scale;
    const sy = h / 2 - (y - cy) * scale; // world y up
    ctx.fillStyle = rgbCss(toColor(spec.values[i]));
    ctx.fillRect(sx, sy, size, size);
  }
}

/* --------------------------------------------------------------- image */

export function drawImage(canvas, arr, bgr = false) {
  const [h, w, c] = arr.shape;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const r = bgr ? 2 : 0, b = bgr ? 0 : 2;
  for (let i = 0; i < h * w; i++) {
    img.data[i * 4] = arr.data[i * c + r];
    img.data[i * 4 + 1] = arr.data[i * c + 1];
    img.data[i * 4 + 2] = arr.data[i * c + b];
    img.data[i * 4 + 3] = c === 4 ? arr.data[i * c + 3] : 255;
  }
  ctx.putImageData(img, 0, 0);
}

// 2-D scalar raster (H, W) -> viridis heatmap over its finite range.
export function drawRaster(canvas, arr) {
  const [h, w] = arr.shape;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const [lo, hi] = finiteRange(arr.data);
  for (let i = 0; i < h * w; i++) {
    const v = Number(arr.data[i]);
    const [r, g, b] = Number.isFinite(v) ? viridis((v - lo) / (hi - lo)) : [0, 0, 0];
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = Number.isFinite(v) ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
}

/* ------------------------------------------------------- zoomable raster */
// Pixel data goes into an offscreen canvas at its native resolution, and the
// visible canvas only ever blits a view of it. That separation is what makes
// zooming into a corner of a camera frame possible: the source keeps every
// pixel, the destination is sized to the panel, and `drawSource` decides
// which part lands where -- rescaling putImageData would resample the data.

// (H, W, C) uint8 -> an offscreen canvas at native resolution.
export function imageCanvas(arr, bgr = false) {
  const source = document.createElement("canvas");
  drawImage(source, arr, bgr);
  return source;
}

// (H, W) scalars -> an offscreen viridis heatmap at native resolution.
export function rasterCanvas(arr) {
  const source = document.createElement("canvas");
  drawRaster(source, arr);
  return source;
}

// The view that fits a sw×sh source into a dw×dh destination: centred, whole
// source visible. {cx, cy} in SOURCE pixels (y down), scale in destination
// px per source px -- the same {cx, cy, scale} shape cloudFit returns, so
// one pan/zoom binding drives both stages.
export function sourceFit(sw, sh, dw, dh) {
  return { cx: sw / 2, cy: sh / 2, scale: Math.min(dw / sw, dh / sh) };
}

// Blit `source` onto `canvas` under a {cx, cy, scale} view. Smoothing is off
// once magnified: at 4x a camera frame should show its pixels, not a blur.
export function drawSource(canvas, source, view) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!view || !(view.scale > 0)) return;
  ctx.imageSmoothingEnabled = view.scale < 1;
  ctx.drawImage(
    source,
    canvas.width / 2 - view.cx * view.scale,
    canvas.height / 2 - view.cy * view.scale,
    source.width * view.scale,
    source.height * view.scale,
  );
}

/* ----------------------------------------------------------- histogram */

export function drawHist(canvas, arr, accent, muted) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const [lo, hi] = finiteRange(arr.data);
  const BINS = 48;
  const counts = new Float64Array(BINS);
  for (let i = 0; i < arr.data.length; i++) {
    const v = Number(arr.data[i]);
    if (!Number.isFinite(v)) continue;
    const b = Math.min(BINS - 1, Math.floor(((v - lo) / (hi - lo)) * BINS));
    counts[b] += 1;
  }
  const peak = Math.max(...counts, 1);
  const bw = w / BINS;
  ctx.fillStyle = accent;
  for (let b = 0; b < BINS; b++) {
    const bh = (counts[b] / peak) * (h - 14);
    ctx.fillRect(b * bw + 1, h - bh, bw - 2, bh);
  }
  ctx.fillStyle = muted;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(fmt(lo), 2, h - 2);
  const hiText = String(fmt(hi));
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 2);
}

/* -------------------------------------------------------------- values */

export function valuesText(arr) {
  const vals = Array.from(arr.data, (v) => fmt(Number(v)));
  return `[${vals.join(", ")}]`;
}

export function statsLine(arr) {
  const s = stats(arr.data);
  if (!s) return "no finite values";
  return `min ${fmt(s.min)} · mean ${fmt(s.mean)} · max ${fmt(s.max)}`;
}

// Per-column {min, mean, max, nan} of an (N, C) array (col=null: whole buffer).
export function columnStats(arr, col = null) {
  const [n, stride] = arr.shape.length === 2 ? arr.shape : [arr.data.length, 1];
  const count = col === null ? arr.data.length : n;
  let lo = Infinity, hi = -Infinity, sum = 0, m = 0, nan = 0;
  for (let i = 0; i < count; i++) {
    const v = Number(col === null ? arr.data[i] : arr.data[i * stride + col]);
    if (!Number.isFinite(v)) { nan += 1; continue; }
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    m += 1;
  }
  if (!m) return null;
  return { min: lo, mean: sum / m, max: hi, nan };
}

// Value -> count for small-cardinality integer data (labels); null when the
// data has more than *cap* distinct values (then it is not label-like).
export function uniqueCounts(arr, cap = 16) {
  const counts = new Map();
  for (let i = 0; i < arr.data.length; i++) {
    const v = Number(arr.data[i]);
    counts.set(v, (counts.get(v) || 0) + 1);
    if (counts.size > cap) return null;
  }
  return [...counts.entries()].sort((a, b) => a[0] - b[0]);
}

export { fmt };

// Stats of ONE column of an (N, C) array -- a whole-buffer aggregate would
// mix x/y/z/intensity into a meaningless number.
export function columnStatsLine(arr, col, name) {
  const [n, stride] = arr.shape;
  let lo = Infinity, hi = -Infinity, sum = 0, m = 0;
  for (let i = 0; i < n; i++) {
    const v = Number(arr.data[i * stride + col]);
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    m += 1;
  }
  if (!m) return "no finite values";
  return `${name}: min ${fmt(lo)} · mean ${fmt(sum / m)} · max ${fmt(hi)}`;
}
