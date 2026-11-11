// numpy array decoding: {dtype, shape, data(base64)} -> TypedArray over the
// buffer (house codec, mirrored from splasher's api.js).

const TYPED = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, int16: Int16Array, int32: Int32Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
  bool: Uint8Array, // numpy stores one 0/1 byte per element
};

// IEEE half -> number. Every float16 value is exact in a Float32Array, which
// the drawing code reads like any other float channel.
function halfToNumber(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function decodeFloat16(buffer) {
  const halves = new Uint16Array(buffer);
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i++) out[i] = halfToNumber(halves[i]);
  return out;
}

export function decodeArray(o) {
  if (!o || o.repr !== undefined) return null;
  const Ctor = TYPED[o.dtype];
  // Reading an unknown dtype as raw bytes would draw a plausible but wrong
  // picture: refuse it instead (the server only sends the dtypes above).
  if (!Ctor && o.dtype !== "float16") throw new Error(`unsupported dtype ${o.dtype}`);
  const bin = atob(o.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return {
    data: Ctor ? new Ctor(bytes.buffer) : decodeFloat16(bytes.buffer),
    shape: o.shape,
    dtype: o.dtype,
    fullRows: o.full_rows ?? null, // set when the server decimated rows
  };
}
