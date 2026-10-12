//! Pluggable voxel-grid storage behind `Map` (W0 facade). Today: `DenseStore`
//! (a flat row-major `Vec`, identical to the pre-refactor `Map` grid).
//! W1 adds `BrickmapStore` — a sparse map of dense `BRICK³` bricks for big worlds.
use crate::CellKind;
use std::collections::HashMap;

/// The grid interface `Map` delegates to. Bounds-tolerant reads return `Empty`/`0`
/// (mirrors the legacy DDA contract — the cast loop never branches on bounds).
pub trait VoxelStore {
    /// The grid's `(sx, sy, sz)` dimensions.
    fn size(&self) -> (usize, usize, usize);
    /// True if `(x, y, z)` is inside the grid.
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool;
    /// The cell kind at `(x, y, z)`. Out-of-bounds reads return `Empty`.
    fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind;
    /// The cell's side/tag data at `(x, y, z)`. Out-of-bounds reads return `0`.
    fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16;
    /// Write kind + tag; out-of-bounds ignored. Returns `true` iff a cell was
    /// actually written (so `Map` can maintain its `has_holo` cache).
    fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) -> bool;
    /// Count of cells currently holding `kind`.
    fn count(&self, kind: CellKind) -> usize;
}

/// Dense flat row-major grid: `index = x + sx*(y + sy*z)`. Byte-for-byte the
/// storage `Map` used before the W0 facade.
#[derive(Clone, Debug)]
pub struct DenseStore {
    sx: usize,
    sy: usize,
    sz: usize,
    grid: Vec<CellKind>,
    side_data: Vec<u16>,
}

impl DenseStore {
    /// Allocate a `sx`×`sy`×`sz` dense grid, all cells `Empty` / side-data `0`.
    pub fn new(sx: usize, sy: usize, sz: usize) -> Self {
        DenseStore {
            sx,
            sy,
            sz,
            grid: vec![CellKind::Empty; sx * sy * sz],
            side_data: vec![0u16; sx * sy * sz],
        }
    }
    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.sx * (y + self.sy * z)
    }
}

impl VoxelStore for DenseStore {
    #[inline]
    fn size(&self) -> (usize, usize, usize) {
        (self.sx, self.sy, self.sz)
    }
    #[inline]
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.sx
            && (y as usize) < self.sy
            && (z as usize) < self.sz
    }
    #[inline]
    fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        if self.in_bounds(x, y, z) {
            self.grid[self.idx(x as usize, y as usize, z as usize)]
        } else {
            CellKind::Empty
        }
    }
    #[inline]
    fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16 {
        if self.in_bounds(x, y, z) {
            self.side_data[self.idx(x as usize, y as usize, z as usize)]
        } else {
            0
        }
    }
    #[inline]
    fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) -> bool {
        if x < self.sx && y < self.sy && z < self.sz {
            let i = self.idx(x, y, z);
            self.grid[i] = kind;
            self.side_data[i] = side_data;
            true
        } else {
            false
        }
    }
    fn count(&self, kind: CellKind) -> usize {
        self.grid.iter().filter(|&&c| c == kind).count()
    }
}

// ---------------------------------------------------------------------------
// W1: BrickmapStore — sparse big-world store.
// ---------------------------------------------------------------------------

/// Edge length of a dense brick (16³ = 4096 cells per brick).
pub const BRICK: i32 = 16;

/// A dense `BRICK³` block of cells. Materialized on first write.
#[derive(Clone, Debug)]
struct Brick {
    kinds: Vec<CellKind>, // BRICK^3, row-major (lx + BRICK*(ly + BRICK*lz))
    tags: Vec<u16>,
}

impl Brick {
    fn empty() -> Self {
        let n = (BRICK * BRICK * BRICK) as usize;
        Brick {
            kinds: vec![CellKind::Empty; n],
            tags: vec![0u16; n],
        }
    }
    #[inline]
    fn local_idx(lx: i32, ly: i32, lz: i32) -> usize {
        (lx + BRICK * (ly + BRICK * lz)) as usize
    }
}

/// Sparse brickmap: a logical `dims` volume backed by a HashMap of resident
/// bricks. Absent bricks read as `Empty`; writes materialize the brick. Memory
/// is O(written bricks), not O(volume) — the big-world store.
#[derive(Clone, Debug)]
pub struct BrickmapStore {
    sx: usize,
    sy: usize,
    sz: usize,
    bricks: HashMap<(i32, i32, i32), Brick>,
}

impl BrickmapStore {
    /// A `sx`×`sy`×`sz` brickmap with no resident bricks (reads as all `Empty`).
    pub fn new(sx: usize, sy: usize, sz: usize) -> Self {
        BrickmapStore {
            sx,
            sy,
            sz,
            bricks: HashMap::new(),
        }
    }

    /// Number of resident (materialized) bricks — for tests/diagnostics.
    pub fn resident_bricks(&self) -> usize {
        self.bricks.len()
    }

    #[inline]
    fn split(x: i32, y: i32, z: i32) -> ((i32, i32, i32), (i32, i32, i32)) {
        // Euclidean floor-div/mod so negative coords map correctly (in-bounds
        // coords are >=0, but keep it total).
        let bc = (
            x.div_euclid(BRICK),
            y.div_euclid(BRICK),
            z.div_euclid(BRICK),
        );
        let lc = (
            x.rem_euclid(BRICK),
            y.rem_euclid(BRICK),
            z.rem_euclid(BRICK),
        );
        (bc, lc)
    }

    /// Compact dump: dims, then per resident brick `(coord, BRICK³ kinds u8, BRICK³ tags u16-le)`.
    /// The override-brick payload W5 streaming builds on.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for d in [self.sx as u32, self.sy as u32, self.sz as u32] {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(self.bricks.len() as u32).to_le_bytes());
        let n = (BRICK * BRICK * BRICK) as usize;
        for (&(bx, by, bz), brick) in &self.bricks {
            for c in [bx, by, bz] {
                out.extend_from_slice(&c.to_le_bytes());
            }
            for i in 0..n {
                out.push(brick.kinds[i].to_u8());
            }
            for i in 0..n {
                out.extend_from_slice(&brick.tags[i].to_le_bytes());
            }
        }
        out
    }

    /// Decode bytes written by [`to_bytes`](Self::to_bytes). Returns `None` on any
    /// truncation or encoding error — never panics.
    pub fn from_bytes(bytes: &[u8]) -> Option<BrickmapStore> {
        let mut p = 0usize;

        let rd_u32 = |b: &[u8], p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let rd_i32 = |b: &[u8], p: &mut usize| -> Option<i32> {
            let v = i32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };

        let sx = rd_u32(bytes, &mut p)? as usize;
        let sy = rd_u32(bytes, &mut p)? as usize;
        let sz = rd_u32(bytes, &mut p)? as usize;
        let nbricks = rd_u32(bytes, &mut p)? as usize;
        let n = (BRICK * BRICK * BRICK) as usize;
        let mut bricks = HashMap::with_capacity(nbricks);
        for _ in 0..nbricks {
            let coord = (
                rd_i32(bytes, &mut p)?,
                rd_i32(bytes, &mut p)?,
                rd_i32(bytes, &mut p)?,
            );
            let mut brick = Brick::empty();
            for i in 0..n {
                brick.kinds[i] = CellKind::from_u8(*bytes.get(p)?);
                p += 1;
            }
            for i in 0..n {
                brick.tags[i] = u16::from_le_bytes(bytes.get(p..p + 2)?.try_into().ok()?);
                p += 2;
            }
            bricks.insert(coord, brick);
        }
        Some(BrickmapStore { sx, sy, sz, bricks })
    }
}

impl VoxelStore for BrickmapStore {
    #[inline]
    fn size(&self) -> (usize, usize, usize) {
        (self.sx, self.sy, self.sz)
    }
    #[inline]
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.sx
            && (y as usize) < self.sy
            && (z as usize) < self.sz
    }
    #[inline]
    fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        if !self.in_bounds(x, y, z) {
            return CellKind::Empty;
        }
        let (bc, lc) = Self::split(x, y, z);
        match self.bricks.get(&bc) {
            Some(b) => b.kinds[Brick::local_idx(lc.0, lc.1, lc.2)],
            None => CellKind::Empty,
        }
    }
    #[inline]
    fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16 {
        if !self.in_bounds(x, y, z) {
            return 0;
        }
        let (bc, lc) = Self::split(x, y, z);
        match self.bricks.get(&bc) {
            Some(b) => b.tags[Brick::local_idx(lc.0, lc.1, lc.2)],
            None => 0,
        }
    }
    #[inline]
    fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) -> bool {
        let (xi, yi, zi) = (x as i32, y as i32, z as i32);
        if !self.in_bounds(xi, yi, zi) {
            return false;
        }
        let (bc, lc) = Self::split(xi, yi, zi);
        let brick = self.bricks.entry(bc).or_insert_with(Brick::empty);
        let i = Brick::local_idx(lc.0, lc.1, lc.2);
        brick.kinds[i] = kind;
        brick.tags[i] = side_data;
        true
    }
    fn count(&self, kind: CellKind) -> usize {
        self.bricks
            .values()
            .flat_map(|b| b.kinds.iter())
            .filter(|&&c| c == kind)
            .count()
    }
}

// ---------------------------------------------------------------------------
// W5: ProcGenStore — procedural base ⊕ persistent overrides.
// ---------------------------------------------------------------------------

/// Procedural world with persistent overrides: unwritten cells come from
/// `procedural_base` (zero storage); a write materializes the brick FROM the
/// base, then overrides. Edited bricks are the only persisted state (W5).
///
/// **Determinism (DP7):** unwritten cells are a pure fn of `(seed,x,y,z)`.
/// Editing then reverting a cell does NOT auto-restore (override bricks are
/// explicit) — the override brick persists until explicitly replaced.
///
/// **`count` semantics:** only resident (edited) bricks are counted. The
/// procedural base spans the entire logical volume and cannot be O(1) counted.
#[derive(Clone, Debug)]
pub struct ProcGenStore {
    sx: usize,
    sy: usize,
    sz: usize,
    seed: u64,
    bricks: HashMap<(i32, i32, i32), Brick>,
}

impl ProcGenStore {
    /// A `sx`×`sy`×`sz` procedural store seeded by `seed`, with no edited
    /// (resident) bricks yet — every unwritten cell reads from the procedural
    /// base.
    pub fn new(sx: usize, sy: usize, sz: usize, seed: u64) -> Self {
        ProcGenStore {
            sx,
            sy,
            sz,
            seed,
            bricks: HashMap::new(),
        }
    }

    /// Number of resident (materialized, edited) bricks.
    pub fn resident_bricks(&self) -> usize {
        self.bricks.len()
    }

    #[inline]
    fn split(x: i32, y: i32, z: i32) -> ((i32, i32, i32), (i32, i32, i32)) {
        (
            (
                x.div_euclid(BRICK),
                y.div_euclid(BRICK),
                z.div_euclid(BRICK),
            ),
            (
                x.rem_euclid(BRICK),
                y.rem_euclid(BRICK),
                z.rem_euclid(BRICK),
            ),
        )
    }

    /// Fill a free-standing brick from the procedural base at brick coord `bc`.
    /// Used by [`to_bytes`](Self::to_bytes) and the `set_cell` entry closure.
    fn fill_from_base(seed: u64, bc: (i32, i32, i32)) -> Brick {
        let mut b = Brick::empty();
        for lz in 0..BRICK {
            for ly in 0..BRICK {
                for lx in 0..BRICK {
                    let (wx, wy, wz) = (bc.0 * BRICK + lx, bc.1 * BRICK + ly, bc.2 * BRICK + lz);
                    b.kinds[Brick::local_idx(lx, ly, lz)] =
                        crate::gen::procedural_base(seed, wx, wy, wz);
                }
            }
        }
        b
    }

    /// Compact dump: u64 seed + dims (3×u32) + edited-brick count (u32) + per
    /// brick `(coord: 3×i32, BRICK³ kinds u8, BRICK³ tags u16-le)`. Unwritten
    /// cells are regenerated from `procedural_base` on load — only overrides persist.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.seed.to_le_bytes());
        for d in [self.sx as u32, self.sy as u32, self.sz as u32] {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(self.bricks.len() as u32).to_le_bytes());
        let n = (BRICK * BRICK * BRICK) as usize;
        for (&(bx, by, bz), brick) in &self.bricks {
            for c in [bx, by, bz] {
                out.extend_from_slice(&c.to_le_bytes());
            }
            for i in 0..n {
                out.push(brick.kinds[i].to_u8());
            }
            for i in 0..n {
                out.extend_from_slice(&brick.tags[i].to_le_bytes());
            }
        }
        out
    }

    /// Decode bytes written by [`to_bytes`](Self::to_bytes). Returns `None` on any
    /// truncation or encoding error — never panics.
    pub fn from_bytes(bytes: &[u8]) -> Option<ProcGenStore> {
        let mut p = 0usize;

        let rd_u32 = |b: &[u8], p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let rd_u64 = |b: &[u8], p: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let rd_i32 = |b: &[u8], p: &mut usize| -> Option<i32> {
            let v = i32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };

        let seed = rd_u64(bytes, &mut p)?;
        let sx = rd_u32(bytes, &mut p)? as usize;
        let sy = rd_u32(bytes, &mut p)? as usize;
        let sz = rd_u32(bytes, &mut p)? as usize;
        let nbricks = rd_u32(bytes, &mut p)? as usize;
        let n = (BRICK * BRICK * BRICK) as usize;
        let mut bricks = HashMap::with_capacity(nbricks);
        for _ in 0..nbricks {
            let coord = (
                rd_i32(bytes, &mut p)?,
                rd_i32(bytes, &mut p)?,
                rd_i32(bytes, &mut p)?,
            );
            let mut brick = Brick::empty();
            for i in 0..n {
                brick.kinds[i] = crate::CellKind::from_u8(*bytes.get(p)?);
                p += 1;
            }
            for i in 0..n {
                brick.tags[i] = u16::from_le_bytes(bytes.get(p..p + 2)?.try_into().ok()?);
                p += 2;
            }
            bricks.insert(coord, brick);
        }
        Some(ProcGenStore {
            sx,
            sy,
            sz,
            seed,
            bricks,
        })
    }
}

impl VoxelStore for ProcGenStore {
    #[inline]
    fn size(&self) -> (usize, usize, usize) {
        (self.sx, self.sy, self.sz)
    }
    #[inline]
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.sx
            && (y as usize) < self.sy
            && (z as usize) < self.sz
    }
    #[inline]
    fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        if !self.in_bounds(x, y, z) {
            return CellKind::Empty;
        }
        let (bc, lc) = Self::split(x, y, z);
        match self.bricks.get(&bc) {
            Some(b) => b.kinds[Brick::local_idx(lc.0, lc.1, lc.2)],
            None => crate::gen::procedural_base(self.seed, x, y, z), // base, unstored
        }
    }
    #[inline]
    fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16 {
        if !self.in_bounds(x, y, z) {
            return 0;
        }
        let (bc, lc) = Self::split(x, y, z);
        match self.bricks.get(&bc) {
            Some(b) => b.tags[Brick::local_idx(lc.0, lc.1, lc.2)],
            None => 0, // base carries no tags
        }
    }
    fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) -> bool {
        let (xi, yi, zi) = (x as i32, y as i32, z as i32);
        if !self.in_bounds(xi, yi, zi) {
            return false;
        }
        let (bc, lc) = Self::split(xi, yi, zi);
        // Snapshot seed before the mutable borrow of `self.bricks` (DP5/F4):
        // the brick is initialized from the procedural base so that editing ONE
        // cell does NOT blank the rest of the brick — unedited neighbours keep
        // their procedural-base values.
        let seed = self.seed;
        let brick = self
            .bricks
            .entry(bc)
            .or_insert_with(|| Self::fill_from_base(seed, bc));
        let i = Brick::local_idx(lc.0, lc.1, lc.2);
        brick.kinds[i] = kind;
        brick.tags[i] = side_data;
        true
    }
    fn count(&self, kind: CellKind) -> usize {
        // Only resident (edited) bricks are counted. The base is infinite/unstored
        // and cannot be scanned without iterating the full logical volume.
        self.bricks
            .values()
            .flat_map(|b| b.kinds.iter())
            .filter(|&&c| c == kind)
            .count()
    }
}

// ---------------------------------------------------------------------------
// W1: MapStore enum — Dense (default, byte-identical) or Brick.
// ---------------------------------------------------------------------------

/// Which backing store a `Map` uses. `Dense` is the default (byte-identical to
/// pre-W0); `Brick` is the sparse big-world store (W1); `Proc` is the
/// procedural base ⊕ persistent-override store (W5).
/// Delegates `VoxelStore`.
#[derive(Clone, Debug)]
pub enum MapStore {
    /// Flat row-major grid — byte-identical to pre-W0 `Map` storage.
    Dense(DenseStore),
    /// Sparse map of resident bricks — the big-world store (W1).
    Brick(BrickmapStore),
    /// Procedural base ⊕ persistent overrides (W5).
    Proc(ProcGenStore),
}

impl VoxelStore for MapStore {
    #[inline]
    fn size(&self) -> (usize, usize, usize) {
        match self {
            MapStore::Dense(s) => s.size(),
            MapStore::Brick(s) => s.size(),
            MapStore::Proc(s) => s.size(),
        }
    }
    #[inline]
    fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        match self {
            MapStore::Dense(s) => s.in_bounds(x, y, z),
            MapStore::Brick(s) => s.in_bounds(x, y, z),
            MapStore::Proc(s) => s.in_bounds(x, y, z),
        }
    }
    #[inline]
    fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        match self {
            MapStore::Dense(s) => s.cell_at(x, y, z),
            MapStore::Brick(s) => s.cell_at(x, y, z),
            MapStore::Proc(s) => s.cell_at(x, y, z),
        }
    }
    #[inline]
    fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16 {
        match self {
            MapStore::Dense(s) => s.side_data_at(x, y, z),
            MapStore::Brick(s) => s.side_data_at(x, y, z),
            MapStore::Proc(s) => s.side_data_at(x, y, z),
        }
    }
    #[inline]
    fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) -> bool {
        match self {
            MapStore::Dense(s) => s.set_cell(x, y, z, kind, side_data),
            MapStore::Brick(s) => s.set_cell(x, y, z, kind, side_data),
            MapStore::Proc(s) => s.set_cell(x, y, z, kind, side_data),
        }
    }
    fn count(&self, kind: CellKind) -> usize {
        match self {
            MapStore::Dense(s) => s.count(kind),
            MapStore::Brick(s) => s.count(kind),
            MapStore::Proc(s) => s.count(kind),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_store_roundtrips_and_bounds() {
        let mut s = DenseStore::new(4, 4, 2);
        assert_eq!(s.size(), (4, 4, 2));
        assert_eq!(s.cell_at(1, 1, 0), CellKind::Empty);
        assert!(s.set_cell(1, 1, 0, CellKind::WallSolid, 7));
        assert_eq!(s.cell_at(1, 1, 0), CellKind::WallSolid);
        assert_eq!(s.side_data_at(1, 1, 0), 7);
        assert!(!s.set_cell(99, 0, 0, CellKind::Floor, 0)); // OOB ignored
        assert_eq!(s.cell_at(-1, 0, 0), CellKind::Empty); // OOB read → Empty
        assert_eq!(s.count(CellKind::WallSolid), 1);
    }

    #[test]
    fn brickmap_matches_dense_in_bounds() {
        let mut b = BrickmapStore::new(40, 40, 8);
        assert_eq!(b.size(), (40, 40, 8));
        assert_eq!(b.cell_at(10, 20, 3), CellKind::Empty); // absent brick → Empty
        assert!(b.set_cell(10, 20, 3, CellKind::WallSolid, 5));
        assert_eq!(b.cell_at(10, 20, 3), CellKind::WallSolid);
        assert_eq!(b.side_data_at(10, 20, 3), 5);
        assert!(!b.set_cell(40, 0, 0, CellKind::Floor, 0)); // OOB ignored
        assert_eq!(b.cell_at(-1, 0, 0), CellKind::Empty); // OOB read
        assert_eq!(b.count(CellKind::WallSolid), 1);
    }

    #[test]
    fn brickmap_is_sparse_for_big_worlds() {
        // A 2048^3 logical world with two far-apart writes must hold ~2 bricks,
        // never 2048^3 cells.
        let mut b = BrickmapStore::new(2048, 2048, 2048);
        b.set_cell(1, 1, 1, CellKind::WallSolid, 0);
        b.set_cell(2000, 2000, 2000, CellKind::Floor, 0);
        assert_eq!(b.resident_bricks(), 2);
        assert_eq!(b.cell_at(2000, 2000, 2000), CellKind::Floor);
        assert_eq!(b.cell_at(1000, 1000, 1000), CellKind::Empty);
    }

    #[test]
    fn brickmap_serialization_roundtrips() {
        let mut b = BrickmapStore::new(128, 128, 16);
        b.set_cell(20, 30, 4, CellKind::WallSolid, 7);
        b.set_cell(100, 10, 1, CellKind::Floor, 0);
        let bytes = b.to_bytes();
        let b2 = BrickmapStore::from_bytes(&bytes).expect("decode");
        assert_eq!(b2.size(), (128, 128, 16));
        assert_eq!(b2.cell_at(20, 30, 4), CellKind::WallSolid);
        assert_eq!(b2.side_data_at(20, 30, 4), 7);
        assert_eq!(b2.cell_at(100, 10, 1), CellKind::Floor);
        assert_eq!(b2.resident_bricks(), b.resident_bricks());
    }

    // -----------------------------------------------------------------------
    // W5: ProcGenStore tests.
    // -----------------------------------------------------------------------

    #[test]
    fn procgen_reads_base_until_overridden() {
        let seed = 7;
        let mut s = ProcGenStore::new(256, 256, 32, seed);
        // Unwritten cell == the procedural base (no storage used).
        let (x, y, z) = (100, 120, 4);
        assert_eq!(
            s.cell_at(x, y, z),
            crate::gen::procedural_base(seed, x, y, z)
        );
        assert_eq!(s.resident_bricks(), 0);
        // Override wins and persists.
        assert!(s.set_cell(x as usize, y as usize, z as usize, CellKind::Portal, 3));
        assert_eq!(s.cell_at(x, y, z), CellKind::Portal);
        assert_eq!(s.side_data_at(x, y, z), 3);
        assert_eq!(s.resident_bricks(), 1); // exactly the edited brick
    }

    #[test]
    fn procgen_materializes_brick_from_base() {
        // Editing ONE cell must NOT wipe the brick's other cells to Empty —
        // they must retain their procedural-base values (DP5/F4 coherence).
        let seed = 9;
        let mut s = ProcGenStore::new(64, 64, 16, seed);
        let (ex, ey, ez) = (10, 10, 5);
        // pick a neighbor in the same brick that the base makes solid/empty:
        let (nx, ny, nz) = (11, 10, 5);
        let base_neighbor = crate::gen::procedural_base(seed, nx, ny, nz);
        s.set_cell(ex, ey, ez, CellKind::WallMatrix, 0); // edit one cell
        assert_eq!(s.cell_at(nx, ny, nz), base_neighbor); // neighbor still = base
    }

    #[test]
    fn procgen_persists_only_edits() {
        let mut s = ProcGenStore::new(256, 256, 32, 7);
        s.set_cell(40, 40, 4, CellKind::WallSolid, 2);
        let bytes = s.to_bytes();
        let s2 = ProcGenStore::from_bytes(&bytes).expect("decode");
        assert_eq!(s2.cell_at(40, 40, 4), CellKind::WallSolid); // edit restored
        assert_eq!(s2.side_data_at(40, 40, 4), 2);
        let (ux, uy, uz) = (200, 10, 1);
        assert_eq!(
            s2.cell_at(ux, uy, uz),
            crate::gen::procedural_base(7, ux, uy, uz)
        ); // base regenerated
        assert_eq!(s2.resident_bricks(), 1);
    }
}
