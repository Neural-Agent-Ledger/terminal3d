//! Boundary vs interior classification + the per-pair contrast metrics.
//!
//! Walking the 4-connected neighbour pairs of a framebuffer, each pair is either
//! a **boundary** (a real object edge: a surface, material, or depth jump) or
//! **interior** (same surface, texture variation only). For each pair we measure
//! two contrast axes:
//!
//! * **colour contrast** — Oklab ΔE between the two cells' *effective colours*
//!   (`lerp(bg, fg, ink)`),
//! * **shape contrast** — `|ink(a) − ink(b)|` (grayscale silhouette legibility).
//!
//! The headline `separation` score (in [`crate::metrics`]) is
//! `boundary_color_contrast − interior_color_contrast`: large means objects pop.

use re_core::{Cell, Framebuffer};

use crate::color::{self, Oklab};
use crate::ink;

/// A neighbour pair's classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairClass {
    /// A real object edge (surface / material / depth discontinuity).
    Boundary,
    /// Same object — texture variation only.
    Interior,
}

/// Depth difference above which a same-surface pair still counts as a boundary
/// (a depth cliff — e.g. a doorway behind a near wall). In world-cell units.
pub const DEPTH_JUMP_EPS: f32 = 0.75;

/// The *effective colour* of a cell in Oklab: `lerp(bg, fg, ink(glyph))` decoded
/// to linear sRGB and converted to Oklab. This is the single signal every colour
/// metric reads.
#[inline]
pub fn effective_oklab(cell: &Cell) -> Oklab {
    let bg = color::unpack_linear(cell.bg);
    let fg = color::unpack_linear(cell.fg);
    let t = ink::coverage(cell.glyph);
    let eff = color::lerp_linear(bg, fg, t);
    color::linear_to_oklab(eff)
}

/// Classify a neighbour pair. Boundary iff the surface kind differs, the material
/// tag differs, or the depth jump exceeds [`DEPTH_JUMP_EPS`]. Two non-finite
/// depths (both open sky) are treated as equal (no jump).
#[inline]
pub fn classify(a: &Cell, b: &Cell) -> PairClass {
    if a.surface != b.surface || a.material != b.material {
        return PairClass::Boundary;
    }
    let da = a.depth;
    let db = b.depth;
    let depth_jump = match (da.is_finite(), db.is_finite()) {
        (true, true) => (da - db).abs() > DEPTH_JUMP_EPS,
        (false, false) => false,
        _ => true, // finite vs infinite = a real edge
    };
    if depth_jump {
        PairClass::Boundary
    } else {
        PairClass::Interior
    }
}

/// The edge orientation of a contrast pair, for `edge_orientation_ratio`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeDir {
    /// A vertical edge — the two cells are horizontally adjacent (left/right).
    Vertical,
    /// A horizontal edge — the two cells are vertically adjacent (up/down).
    Horizontal,
}

/// Running accumulators over all neighbour pairs of a frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContrastAccum {
    /// Running sum of colour ΔE over all boundary pairs.
    pub boundary_color_sum: f64,
    /// Running sum of shape (ink) contrast over all boundary pairs.
    pub boundary_shape_sum: f64,
    /// Number of boundary pairs visited.
    pub boundary_count: u64,
    /// Running sum of colour ΔE over all interior pairs.
    pub interior_color_sum: f64,
    /// Running sum of shape (ink) contrast over all interior pairs.
    pub interior_shape_sum: f64,
    /// Number of interior pairs visited.
    pub interior_count: u64,
    /// Count of boundary edges by orientation (drives `edge_orientation_ratio`).
    pub vertical_edges: u64,
    /// Count of horizontal boundary edges.
    pub horizontal_edges: u64,
}

impl ContrastAccum {
    /// Mean boundary colour contrast (Oklab ΔE); 0 if no boundary pairs.
    pub fn boundary_color(&self) -> f32 {
        mean(self.boundary_color_sum, self.boundary_count)
    }
    /// Mean boundary shape (ink) contrast; 0 if no boundary pairs.
    pub fn boundary_shape(&self) -> f32 {
        mean(self.boundary_shape_sum, self.boundary_count)
    }
    /// Mean interior colour contrast (Oklab ΔE); 0 if no interior pairs.
    pub fn interior_color(&self) -> f32 {
        mean(self.interior_color_sum, self.interior_count)
    }
    /// Mean interior shape (ink) contrast; 0 if no interior pairs.
    pub fn interior_shape(&self) -> f32 {
        mean(self.interior_shape_sum, self.interior_count)
    }
    /// vertical / horizontal boundary-edge ratio. `f32::INFINITY` if there are
    /// vertical edges but no horizontal ones; `0` if neither.
    pub fn edge_orientation_ratio(&self) -> f32 {
        if self.horizontal_edges == 0 {
            if self.vertical_edges == 0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            self.vertical_edges as f32 / self.horizontal_edges as f32
        }
    }
}

#[inline]
fn mean(sum: f64, count: u64) -> f32 {
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Accumulate boundary/interior colour + shape contrast over every 4-connected
/// neighbour pair in `fb`. Each pair is visited once (right + down edges only).
pub fn accumulate(fb: &Framebuffer) -> ContrastAccum {
    let cols = fb.cols();
    let rows = fb.rows();
    let mut acc = ContrastAccum::default();
    if cols == 0 || rows == 0 {
        return acc;
    }

    // Precompute effective Oklab + ink per cell once (each is read by up to two
    // edges) so we don't recompute the lerp/Oklab for every neighbour.
    let mut lab = Vec::with_capacity(cols * rows);
    let mut inks = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let c = fb.get(col, row).expect("in-bounds");
            lab.push(effective_oklab(c));
            inks.push(ink::coverage(c.glyph));
        }
    }
    let idx = |col: usize, row: usize| row * cols + col;

    for row in 0..rows {
        for col in 0..cols {
            let a = fb.get(col, row).expect("in-bounds");
            let ia = idx(col, row);
            // Right neighbour (vertical edge).
            if col + 1 < cols {
                let b = fb.get(col + 1, row).expect("in-bounds");
                let ib = idx(col + 1, row);
                add_pair(
                    &mut acc,
                    a,
                    b,
                    lab[ia],
                    lab[ib],
                    inks[ia],
                    inks[ib],
                    EdgeDir::Vertical,
                );
            }
            // Down neighbour (horizontal edge).
            if row + 1 < rows {
                let b = fb.get(col, row + 1).expect("in-bounds");
                let ib = idx(col, row + 1);
                add_pair(
                    &mut acc,
                    a,
                    b,
                    lab[ia],
                    lab[ib],
                    inks[ia],
                    inks[ib],
                    EdgeDir::Horizontal,
                );
            }
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn add_pair(
    acc: &mut ContrastAccum,
    a: &Cell,
    b: &Cell,
    lab_a: Oklab,
    lab_b: Oklab,
    ink_a: f32,
    ink_b: f32,
    dir: EdgeDir,
) {
    let color_c = color::delta_e(lab_a, lab_b) as f64;
    let shape_c = (ink_a - ink_b).abs() as f64;
    match classify(a, b) {
        PairClass::Boundary => {
            acc.boundary_color_sum += color_c;
            acc.boundary_shape_sum += shape_c;
            acc.boundary_count += 1;
            match dir {
                EdgeDir::Vertical => acc.vertical_edges += 1,
                EdgeDir::Horizontal => acc.horizontal_edges += 1,
            }
        }
        PairClass::Interior => {
            acc.interior_color_sum += color_c;
            acc.interior_shape_sum += shape_c;
            acc.interior_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::SurfaceKind;

    fn cell(
        glyph: char,
        fg: u32,
        bg: u32,
        depth: f32,
        surface: SurfaceKind,
        material: u16,
    ) -> Cell {
        Cell {
            glyph: u32::from(glyph),
            fg,
            bg,
            depth,
            surface,
            material,
            ..Cell::default()
        }
    }

    /// A 3×3 frame split down the middle: left column a near bright wall, the
    /// rest a far dark floor. The surface boundary between them must score high
    /// boundary contrast and low interior contrast.
    #[test]
    fn boundary_high_interior_low() {
        let mut fb = Framebuffer::new(3, 3);
        for row in 0..3 {
            // Left column: bright near wall.
            fb.set(
                0,
                row,
                cell('#', 0x00FFFFFF, 0x00202020, 1.0, SurfaceKind::WallSolid, 0),
            );
            // Cols 1,2: dark far floor.
            for col in 1..3 {
                fb.set(
                    col,
                    row,
                    cell('.', 0x00303030, 0x00000000, 8.0, SurfaceKind::Floor, 0),
                );
            }
        }
        let acc = accumulate(&fb);
        assert!(acc.boundary_count > 0, "expected boundary pairs");
        assert!(acc.interior_count > 0, "expected interior pairs");
        let sep = acc.boundary_color() - acc.interior_color();
        assert!(
            sep > 0.1,
            "boundary should dominate interior (b={}, i={}, sep={})",
            acc.boundary_color(),
            acc.interior_color(),
            sep
        );
    }

    /// A flat single-surface frame: every neighbour is interior, separation ≈ 0.
    #[test]
    fn flat_frame_scores_zero_separation() {
        let mut fb = Framebuffer::new(4, 4);
        for row in 0..4 {
            for col in 0..4 {
                fb.set(
                    col,
                    row,
                    cell('.', 0x00404040, 0x00101010, 5.0, SurfaceKind::Floor, 0),
                );
            }
        }
        let acc = accumulate(&fb);
        assert_eq!(acc.boundary_count, 0, "flat frame has no boundaries");
        let sep = acc.boundary_color() - acc.interior_color();
        assert!(
            sep.abs() < 1e-4,
            "flat frame separation should be ~0, got {sep}"
        );
        assert!(
            acc.interior_color() < 1e-4,
            "uniform interior contrast should be ~0"
        );
    }

    #[test]
    fn depth_jump_is_a_boundary() {
        // Same surface, big depth gap → boundary.
        let a = cell('#', 0x00FFFFFF, 0, 1.0, SurfaceKind::WallSolid, 0);
        let b = cell('#', 0x00FFFFFF, 0, 5.0, SurfaceKind::WallSolid, 0);
        assert_eq!(classify(&a, &b), PairClass::Boundary);
        // Same surface, tiny depth diff → interior.
        let c = cell('#', 0x00FFFFFF, 0, 1.1, SurfaceKind::WallSolid, 0);
        assert_eq!(classify(&a, &c), PairClass::Interior);
    }
}
