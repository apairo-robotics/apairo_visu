// numpy array decoding: {dtype, shape, data(base64)} -> TypedArray over the
// buffer (house codec, mirrored from splasher's api.js).

const TYPED = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, int16: Int16Array, int32: Int32Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
};

export function decodeArray(o) {
  if (!o || o.repr !== undefined) return null;
  const bin = atob(o.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const Ctor = TYPED[o.dtype] || Uint8Array;
  return {
    data: new Ctor(bytes.buffer),
    shape: o.shape,
    dtype: o.dtype,
    fullRows: o.full_rows ?? null, // set when the server decimated rows
  };
}
