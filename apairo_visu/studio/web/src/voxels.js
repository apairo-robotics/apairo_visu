// The voxel view of a point cloud: which cell every point falls into, and
// which cells the scan occupies.
//
// Pure arithmetic over an (n * 3) position buffer -- no three.js, no DOM, no
// server call -- so a 3D panel can show what a given `voxel_size` would do to
// a scan *before* anyone writes VoxelisePointCloud into the recipe: the cells
// the scan really occupies, and the cloud that would come out the other side
// (one centroid per cell).
//
// Cells sit on the absolute world grid (floor(x / size)), the convention
// apairo's voxelisers use, so a cell drawn here is the cell the preprocessor
// would build -- not one aligned to wherever this particular scan happens to
// start.

// The linear cell index is a double, so the grid may hold at most 2^53 cells
// before two of them would address the same slot. A finer grid than that over
// a cloud this wide is a mistake, not a view: the caller says so instead of
// silently colliding cells.
const MAX_ADDRESSABLE = Number.MAX_SAFE_INTEGER;

/* --------------------------------------------------------------- build */

// Voxelize the first `n` points of `pos` ((n * 3) xyz, finite) at `size`
// metres. Returns null when the grid is unbuildable (empty cloud, silly size,
// or more cells than can be addressed); otherwise:
//
//   size, n, cells                     the grid, the points in it, occupied cells
//   cellOf   Int32Array(n)             point -> cell
//   count    Int32Array(cells)         points per cell
//   start    Int32Array(cells + 1)     CSR offsets into `members`...
//   members  Int32Array(n)             ...cell -> its point indices
//   centers  Float32Array(cells * 3)   cell centre (grid-aligned) -- the boxes
//   centroids Float32Array(cells * 3)  mean of the cell's points -- the reduction
export function voxelize(pos, n, size) {
  if (!(size > 0) || !(n > 0)) return null;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let p = 0; p < n; p++) {
    const x = pos[p * 3], y = pos[p * 3 + 1], z = pos[p * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxZ)) return null;

  const i0 = Math.floor(minX / size);
  const j0 = Math.floor(minY / size);
  const k0 = Math.floor(minZ / size);
  const nx = Math.floor(maxX / size) - i0 + 1;
  const ny = Math.floor(maxY / size) - j0 + 1;
  const nz = Math.floor(maxZ / size) - k0 + 1;
  // Written as a negation so a NaN extent falls out here too.
  if (!(nx * ny * nz <= MAX_ADDRESSABLE)) return null;

  // Pass 1: every point to a dense cell id, discovered in first-seen order.
  const cellOf = new Int32Array(n);
  const lookup = new Map(); // linear grid index -> dense cell id
  const cx = [], cy = [], cz = [];
  let cells = 0;
  for (let p = 0; p < n; p++) {
    const a = Math.floor(pos[p * 3] / size) - i0;
    const b = Math.floor(pos[p * 3 + 1] / size) - j0;
    const c = Math.floor(pos[p * 3 + 2] / size) - k0;
    const lin = (a * ny + b) * nz + c;
    let cell = lookup.get(lin);
    if (cell === undefined) {
      cell = cells++;
      lookup.set(lin, cell);
      cx.push(a + i0);
      cy.push(b + j0);
      cz.push(c + k0);
    }
    cellOf[p] = cell;
  }

  // Pass 2: counting sort into CSR membership -- one flat Int32Array instead
  // of `cells` little arrays, which at a hundred thousand cells is the
  // difference between a few hundred KB and a few MB of object headers.
  const count = new Int32Array(cells);
  for (let p = 0; p < n; p++) count[cellOf[p]]++;
  const start = new Int32Array(cells + 1);
  for (let c = 0; c < cells; c++) start[c + 1] = start[c] + count[c];
  const cursor = start.slice(0, cells);
  const members = new Int32Array(n);
  for (let p = 0; p < n; p++) members[cursor[cellOf[p]]++] = p;

  const centers = new Float32Array(cells * 3);
  for (let c = 0; c < cells; c++) {
    centers[c * 3] = (cx[c] + 0.5) * size;
    centers[c * 3 + 1] = (cy[c] + 0.5) * size;
    centers[c * 3 + 2] = (cz[c] + 0.5) * size;
  }

  // Sum in doubles: a cell holding a few hundred points tens of metres out
  // would lose millimetres to float32 rounding, and the centroid is the one
  // number here that claims to be the voxelised cloud.
  const sums = new Float64Array(cells * 3);
  for (let p = 0; p < n; p++) {
    const c = cellOf[p] * 3;
    sums[c] += pos[p * 3];
    sums[c + 1] += pos[p * 3 + 1];
    sums[c + 2] += pos[p * 3 + 2];
  }
  const centroids = new Float32Array(cells * 3);
  for (let c = 0; c < cells; c++) {
    const k = count[c];
    centroids[c * 3] = sums[c * 3] / k;
    centroids[c * 3 + 1] = sums[c * 3 + 1] / k;
    centroids[c * 3 + 2] = sums[c * 3 + 2] / k;
  }

  return { size, n, cells, cellOf, count, start, members, centers, centroids };
}

/* -------------------------------------------------------------- colour */

// The colour of each cell's representative when the cloud is reduced to one
// point per cell: the majority colour among the cell's members, which is what
// VoxeliseLabels does with the labels themselves -- so a voxel straddling
// grass and a tree trunk comes out the colour the preprocessor would give it,
// not a blend of the two that belongs to neither class. The cloud keeps its
// own colouring throughout; nothing here invents a colour for a voxel.
export function majorityColors(vox, rgb) {
  const out = new Float32Array(vox.cells * 3);
  const tally = new Map(); // packed rgb -> occurrences, reused per cell
  for (let c = 0; c < vox.cells; c++) {
    tally.clear();
    let best = 0, bestCount = -1;
    for (let m = vox.start[c]; m < vox.start[c + 1]; m++) {
      const p = vox.members[m] * 3;
      const packed = (Math.round(rgb[p] * 255) << 16)
        | (Math.round(rgb[p + 1] * 255) << 8)
        | Math.round(rgb[p + 2] * 255);
      const seen = (tally.get(packed) || 0) + 1;
      tally.set(packed, seen);
      if (seen > bestCount) { bestCount = seen; best = packed; }
    }
    out[c * 3] = ((best >> 16) & 255) / 255;
    out[c * 3 + 1] = ((best >> 8) & 255) / 255;
    out[c * 3 + 2] = (best & 255) / 255;
  }
  return out;
}
