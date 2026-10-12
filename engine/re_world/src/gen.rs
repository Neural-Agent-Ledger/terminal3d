//! Deterministic 3D procedural density: a pure fn of (seed, x, y, z) → CellKind.
//! Hash-based value noise (no float nondeterminism, no external dep), so a brick
//! generated at octave N is identical every run/platform — the determinism the
//! scale hierarchy + override layer depend on. (FBM/worley quality is a later
//! polish; W3a needs a big SPARSE structured world to render + measure.)
use crate::CellKind;

#[inline]
fn hash3(seed: u64, x: i32, y: i32, z: i32) -> u32 {
    // SplitMix-style integer hash; total, branch-free, platform-stable.
    let mut h = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as u32 as u64)
            .wrapping_mul(0xC2B2AE3D27D4EB4F)
            .rotate_left(21)
        ^ (z as u32 as u64)
            .wrapping_mul(0x165667B19E3779F9)
            .rotate_left(42);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    (h & 0xFFFF_FFFF) as u32
}

/// Density at a world cell as a fixed-point [0,4095].
#[inline]
fn density(seed: u64, x: i32, y: i32, z: i32) -> u32 {
    // Two octaves of value-noise (cell + half-resolution), summed — cheap fBm-ish.
    let coarse = hash3(seed, x >> 3, y >> 3, z >> 3) & 0xFFF;
    let fine = hash3(seed ^ 0xABCD, x >> 1, y >> 1, z >> 1) & 0xFFF;
    (coarse * 2 + fine) / 3
}

/// CellKind at a world cell. Below a height-biased threshold → solid (with a
/// little material variety); else Empty. Height bias makes a floor-heavy,
/// cavernous structure rather than uniform static.
pub fn procedural_base(seed: u64, x: i32, y: i32, z: i32) -> CellKind {
    let d = density(seed, x, y, z);
    // Lower z is denser (ground), higher z sparser (air) — a simple terrain bias.
    let threshold = 2200u32.saturating_add((z.max(0) as u32).saturating_mul(60));
    if d < threshold {
        CellKind::Empty
    } else {
        match (hash3(seed ^ 0x55, x, y, z) % 8) as u8 {
            0..=4 => CellKind::WallSolid,
            5 | 6 => CellKind::Floor,
            _ => CellKind::WallMatrix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn procedural_base_is_deterministic() {
        for &(x, y, z) in &[(0, 0, 0), (137, 42, 9), (-5, 1000, 3)] {
            assert_eq!(procedural_base(7, x, y, z), procedural_base(7, x, y, z));
        }
    }
    #[test]
    fn procedural_base_is_nontrivial() {
        // Over a 32³ sample there must be BOTH solid and empty cells (not all one).
        let mut solid = 0;
        let mut empty = 0;
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    match procedural_base(1, x, y, z) {
                        CellKind::Empty => empty += 1,
                        _ => solid += 1,
                    }
                }
            }
        }
        assert!(
            solid > 100 && empty > 100,
            "got solid={solid} empty={empty}"
        );
    }
    #[test]
    fn different_seed_differs() {
        let a: Vec<_> = (0..64).map(|i| procedural_base(1, i, 0, 0)).collect();
        let b: Vec<_> = (0..64).map(|i| procedural_base(2, i, 0, 0)).collect();
        assert_ne!(a, b);
    }
}
