// Front-side colorization helpers (mirrored from splasher's colors.js).
// The server delivers raw arrays; every color decision happens here.

const VIRIDIS = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];

const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);

// t in [0,1] -> [r,g,b] 0..255 (linear interpolation between viridis anchors).
export function viridis(t) {
  if (!Number.isFinite(t)) t = 0;
  t = clamp01(t);
  const n = VIRIDIS.length - 1;
  const pos = t * n;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n);
  const f = pos - lo;
  const a = VIRIDIS[lo], b = VIRIDIS[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// Finite [min,max] bounds of a TypedArray (NaNs ignored).
export function finiteRange(arr) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

export const rgbCss = ([r, g, b]) => `rgb(${r | 0},${g | 0},${b | 0})`;

// Categorical palette for label-like integer channels (tab20-style, 20
// distinct hues). Indexed by label value modulo the palette size.
const CATEGORICAL = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
  [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
  [174, 199, 232], [255, 187, 120], [152, 223, 138], [255, 152, 150], [197, 176, 213],
  [196, 156, 148], [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
];

export function categorical(value) {
  if (!Number.isFinite(value)) return [127, 127, 127];
  const n = CATEGORICAL.length;
  const i = ((Math.round(value) % n) + n) % n;
  return CATEGORICAL[i];
}
