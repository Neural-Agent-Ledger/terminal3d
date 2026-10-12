//! `re_render::truchet` — EDGE-MATCHED tiling: ornament whose features CONTINUE
//! across cell boundaries.
//!
//! ## Why this module exists
//!
//! Every procedural pattern in this crate samples per cell and independently:
//! `hash01(hash2i(cx, cy))`, an fBm tap at the cell's coordinate, a modulo lattice.
//! Independent sampling gives *texture* — and it is why our walls read as noise laid
//! over a surface rather than as a surface with structure. Nothing ever **continues**.
//! A crack stops at the cell edge. A brick bond does not interlock. A conduit does not
//! run anywhere; it just speckles.
//!
//! Wang tiling fixes exactly that, and the procedural (hash-driven) form of it keeps
//! every property this codebase needs: O(1) per cell, no storage, no scan order, and
//! fully deterministic — so it is golden-hash-safe and replayable.
//!
//! ## The trick: hash the EDGE, not the cell
//!
//! A Wang tile is a square with a colour on each side; neighbours must agree on their
//! shared side. The naive way to satisfy that is to place tiles in scan order and
//! backtrack, which needs state and an order. The procedural way needs neither:
//!
//! > **Derive each edge's value from a hash of THE EDGE's own coordinates.**
//!
//! The edge between `(cx, cy)` and `(cx, cy-1)` is identified by one `(cx, cy)` pair
//! whichever side you ask from. So cell A's north edge and cell B's south edge are
//! *the same function call on the same arguments*, and agreement is not enforced —
//! it is structural. `connections_agree_across_every_shared_edge` is the test, but
//! the property is true by construction, not by luck.
//!
//! ## What a "colour" means here
//!
//! One bit: does the conduit cross this edge or not. The cell's four edge bits then
//! name exactly which sides it connects, and the glyph is the box-drawing character
//! that connects those sides — `│ ─ └ ┘ ┌ ┐ ├ ┤ ┬ ┴ ┼` and the four stubs. So the
//! ornament reads as pipework/circuitry that genuinely runs, which is the right
//! vocabulary for this game rather than a decorative coincidence.
//!
//! [`DENSITY`] controls how often an edge is crossed, so the same machinery spans a
//! sparse scattering of stubs through to a dense lattice.
//!
//! Pure compute: no Bevy, no terminal, no allocation, no `unsafe`.

use crate::ascii_shader::hash2i;

/// Edge bits of a cell's connection mask, matching `crate::contour`'s N/E/S/W order
/// so the two modules can be read side by side.
pub const N: u8 = 1 << 0;
/// East edge bit — see [`N`].
pub const E: u8 = 1 << 1;
/// South edge bit — see [`N`].
pub const S: u8 = 1 << 2;
/// West edge bit — see [`N`].
pub const W: u8 = 1 << 3;

/// The sixteen connection glyphs, indexed by an `N|E|S|W` mask. The glyph connects
/// EXACTLY the sides whose bits are set.
///
/// Note this is the opposite convention to [`crate::contour::contour_glyph`], which
/// is given the sides that carry an *edge* and strokes toward the free directions.
/// Here a set bit means "the run leaves through this side", so the glyph is read
/// directly. Two different questions; deliberately not shared.
pub const CONNECT_GLYPHS: [char; 16] = [
    ' ', // 0000 — nothing crosses; blank cell
    '╵', // 0001 — N
    '╶', // 0010 — E
    '└', // 0011 — N+E
    '╷', // 0100 — S
    '│', // 0101 — N+S
    '┌', // 0110 — E+S
    '├', // 0111 — N+E+S
    '╴', // 1000 — W
    '┘', // 1001 — N+W
    '─', // 1010 — E+W
    '┴', // 1011 — N+E+W
    '┐', // 1100 — S+W
    '┤', // 1101 — N+S+W
    '┬', // 1110 — E+S+W
    '┼', // 1111 — all four
];

/// Fraction of edges that carry a connection at the default density. Half is the
/// classic Wang/Truchet look — dense enough to run, open enough to read as a pattern
/// rather than a grid.
pub const DENSITY: f32 = 0.5;

/// Default TILES PER WORLD CELL for a surface dressed in this lattice.
///
/// One tile per world cell (`1.0`) is the naive reading and it does not work: a wall
/// cell is drawn as a block of terminal cells, so a single tile becomes one glyph
/// smeared across all of them — a flat band, not ornament. Four subdivides a wall
/// cell into a 4x4 lattice, which at ordinary viewing distances puts roughly one
/// glyph per tile: dense enough to read as pipework, coarse enough that the junctions
/// stay distinguishable.
///
/// Continuity is unaffected by the choice — the index floors a CONTINUOUS world
/// coordinate, so any positive factor still lands on one global grid.
pub const WALL_SCALE: f32 = 4.0;

/// Salt distinguishing HORIZONTAL edge hashes from VERTICAL ones. Without it the two
/// families would collide at coordinates where they share `(a, b)`, correlating the
/// north edge of a cell with its west edge and visibly biasing the pattern toward
/// corners.
const AXIS_SALT: i32 = 0x5F37;

/// Is the HORIZONTAL edge above cell `(cx, cy)` crossed?
///
/// This edge separates `(cx, cy-1)` from `(cx, cy)`, and is named by the coordinates
/// of the cell BELOW it — so `h_edge(cx, cy)` is the north edge of `(cx, cy)` and the
/// south edge of `(cx, cy-1)`, one call, one answer.
#[inline]
pub fn h_edge(cx: i32, cy: i32, seed: i32, density: f32) -> bool {
    edge_bit(hash2i(cx.wrapping_add(seed), cy), density)
}

/// Is the VERTICAL edge left of cell `(cx, cy)` crossed?
///
/// Separates `(cx-1, cy)` from `(cx, cy)`; named by the cell to its RIGHT, so it is
/// the west edge of `(cx, cy)` and the east edge of `(cx-1, cy)`.
#[inline]
pub fn v_edge(cx: i32, cy: i32, seed: i32, density: f32) -> bool {
    edge_bit(
        hash2i(cx, cy.wrapping_add(seed).wrapping_add(AXIS_SALT)),
        density,
    )
}

/// Threshold a hash into a connection bit at `density` (clamped to `[0, 1]`).
#[inline]
fn edge_bit(h: u32, density: f32) -> bool {
    // `h as f32 / 2^32` in [0,1); compare against the density.
    let x = (h >> 8) as f32 / ((1u32 << 24) as f32);
    x < density.clamp(0.0, 1.0)
}

/// The `N|E|S|W` connection mask for cell `(cx, cy)`.
///
/// Each side is read from the SHARED edge hash, so this cell's east bit is bit-equal
/// to its right neighbour's west bit, and its south bit to the cell below's north
/// bit. No neighbour is consulted and no order is imposed — the agreement falls out
/// of naming each edge once.
#[inline]
pub fn connections(cx: i32, cy: i32, seed: i32, density: f32) -> u8 {
    let mut m = 0u8;
    if h_edge(cx, cy, seed, density) {
        m |= N;
    }
    if v_edge(cx + 1, cy, seed, density) {
        m |= E;
    }
    if h_edge(cx, cy + 1, seed, density) {
        m |= S;
    }
    if v_edge(cx, cy, seed, density) {
        m |= W;
    }
    m
}

/// The box-drawing glyph for cell `(cx, cy)` — the connection run through that cell.
///
/// `' '` when nothing crosses, which at low [`DENSITY`] is most cells.
#[inline]
pub fn glyph(cx: i32, cy: i32, seed: i32, density: f32) -> char {
    CONNECT_GLYPHS[(connections(cx, cy, seed, density) & 0x0F) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE property. For every pair of orthogonally adjacent cells, the bit each one
    /// reports for their shared side is the same bit — so a run leaving one cell
    /// always arrives in the next, and the ornament is connected rather than
    /// speckled. This is what independent per-cell hashing cannot give.
    #[test]
    fn connections_agree_across_every_shared_edge() {
        for seed in [0, 1, 7, -3] {
            for density in [0.25, 0.5, 0.75] {
                for cy in -4..4 {
                    for cx in -4..4 {
                        let me = connections(cx, cy, seed, density);
                        let right = connections(cx + 1, cy, seed, density);
                        let below = connections(cx, cy + 1, seed, density);
                        assert_eq!(
                            me & E != 0,
                            right & W != 0,
                            "({cx},{cy}) east must equal ({}, {cy}) west",
                            cx + 1
                        );
                        assert_eq!(
                            me & S != 0,
                            below & N != 0,
                            "({cx},{cy}) south must equal ({cx}, {}) north",
                            cy + 1
                        );
                    }
                }
            }
        }
    }

    /// Deterministic and stateless: the same cell always yields the same tile, in any
    /// order, with no dependence on what was asked before. Required for the golden
    /// hashes and for a re-cast frame to be identical.
    #[test]
    fn tiling_is_deterministic_and_order_independent() {
        let forward: Vec<u8> = (0..64).map(|i| connections(i % 8, i / 8, 5, 0.5)).collect();
        let backward: Vec<u8> = (0..64)
            .rev()
            .map(|i| connections(i % 8, i / 8, 5, 0.5))
            .collect();
        let mut b = backward;
        b.reverse();
        assert_eq!(forward, b, "tile choice must not depend on visit order");
        // And stable across repeated calls.
        assert_eq!(connections(3, 4, 5, 0.5), connections(3, 4, 5, 0.5));
    }

    /// Density actually controls occupancy, and the extremes are exact: 0 leaves every
    /// cell blank, 1 makes every cell a full cross. A pattern knob that saturated
    /// early or never reached blank would be much harder to author against.
    #[test]
    fn density_spans_blank_to_full() {
        for cy in -3..3 {
            for cx in -3..3 {
                assert_eq!(glyph(cx, cy, 0, 0.0), ' ', "density 0 → nothing connects");
                assert_eq!(
                    glyph(cx, cy, 0, 1.0),
                    '┼',
                    "density 1 → everything connects"
                );
            }
        }
        // In between, occupancy tracks density monotonically over a decent sample.
        let occupancy = |d: f32| -> f32 {
            let mut set = 0u32;
            for cy in -16..16 {
                for cx in -16..16 {
                    set += connections(cx, cy, 11, d).count_ones();
                }
            }
            f32::from(set as u16) / (32.0 * 32.0 * 4.0)
        };
        let (lo, mid, hi) = (occupancy(0.25), occupancy(0.5), occupancy(0.75));
        assert!(
            lo < mid && mid < hi,
            "occupancy must rise with density: {lo} {mid} {hi}"
        );
        assert!(
            (mid - 0.5).abs() < 0.06,
            "density 0.5 should sit near half: {mid}"
        );
    }

    /// The pattern must not be degenerate — a "tiling" that emits three glyphs is a
    /// grid, not ornament. Over a modest field, most of the vocabulary should appear.
    #[test]
    fn a_field_uses_most_of_the_glyph_vocabulary() {
        use std::collections::HashSet;
        let mut seen: HashSet<char> = HashSet::new();
        for cy in -12..12 {
            for cx in -12..12 {
                seen.insert(glyph(cx, cy, 2, 0.5));
            }
        }
        assert!(
            seen.len() >= 14,
            "expected most of the 16 connection glyphs, saw {}: {seen:?}",
            seen.len()
        );
    }

    /// The two edge families must be independent. Sharing one hash would correlate a
    /// cell's north edge with its west edge and bend the whole field toward corners —
    /// visible as a diagonal grain, and the reason [`AXIS_SALT`] exists.
    #[test]
    fn horizontal_and_vertical_edges_are_uncorrelated() {
        let mut both = 0u32;
        let mut n_only = 0u32;
        let mut w_only = 0u32;
        let mut neither = 0u32;
        for cy in -20..20 {
            for cx in -20..20 {
                match (h_edge(cx, cy, 0, 0.5), v_edge(cx, cy, 0, 0.5)) {
                    (true, true) => both += 1,
                    (true, false) => n_only += 1,
                    (false, true) => w_only += 1,
                    (false, false) => neither += 1,
                }
            }
        }
        let total = (both + n_only + w_only + neither) as f32;
        // Independent fair bits ⇒ each quadrant ≈ 25%. Allow generous slack; the test
        // is for CORRELATION (which would push one pair toward 50/0), not for a
        // perfect hash.
        for (name, n) in [
            ("both", both),
            ("n", n_only),
            ("w", w_only),
            ("neither", neither),
        ] {
            let f = n as f32 / total;
            assert!(
                (0.15..0.35).contains(&f),
                "{name} quadrant at {f:.3} — the two edge families look correlated"
            );
        }
    }
}
