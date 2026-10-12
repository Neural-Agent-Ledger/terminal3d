//! `re_render::slope` — DIAGONAL silhouette de-angularization ("уйти от угловатости").
//!
//! The 4-neighbour contour pass ([`crate::contour`]) draws a clean axis-aligned
//! box-drawing outline. But on a SLOPED silhouette — a peaked roof, a 45° stair of
//! solid blocks, the diagonal flank of a pyramid — that pass can only place a
//! *corner* glyph (`┐ └ ┌ ┘`) in each cell, so a 45° run renders as a visible
//! STAIRCASE of little corners. The eye reads "blocky", not "sloped".
//!
//! This module is the de-angularizer. It runs INSIDE the contour pass: for each
//! cell the contour pass has already classified as an OUTER silhouette corner, we
//! re-examine the EIGHT neighbours. When the cell sits on a consistent DIAGONAL run
//! (the solid mass marches steadily down-and-left, up-and-right, …), we replace the
//! stair-step corner with a SLOPE glyph from a swappable [`SlopeTheme`]:
//!
//! * the **line** outline glyph `╱`/`╲` (U+2571/U+2572) — the 45° diagonal stroke
//!   that replaces the `┐`/`└` stair corner, OR
//! * the **chamfer triangle** `◤ ◥ ◣ ◢` (U+25E4/25E5/25E3/25E2) that fills the
//!   SOLID half of the cell so the FILLED silhouette of a solid roof reads as one
//!   continuous slope, not a flight of steps. This is the bigger win for a solid
//!   mass; the line glyph is for a thin one-cell-wide outline.
//!
//! ## What is exact vs. approximated
//!
//! * **45° (1:1) runs are EXACT.** `╱`/`╲` and the corner triangles are true 45°
//!   half-cell diagonals; a 1-up-1-over stair maps to them with no error.
//! * **Shallower-than-45° (2:1, 3:1 …) runs are APPROXIMATED.** A run that steps
//!   one cell over every two cells of travel has no dedicated half-step glyph at the
//!   1×1 cell level. The optional sub-cell pass (the 2×2 `▚`/`▞` quadrant combo,
//!   [`subcell_half_step`]) gives ONE intermediate angle (~26°/~63°) by splitting
//!   the diagonal across two cells; anything between is snapped to the nearer of
//!   {45°, the 2:1 half-step}. We never claim a continuous angle — ASCII gives a
//!   few exact rungs and we approximate to the nearest.
//!
//! ## Data-driven (smart-materials §5 `ContourTheme`)
//!
//! The glyph vocabulary is a [`SlopeTheme`] table, not hard-coded literals, so the
//! look is configuration: a Unicode theme ([`SlopeTheme::UNICODE`]) and an ASCII
//! degrade theme ([`SlopeTheme::ASCII`], `/ \`) ship built-in, and `/iterate-render`
//! can swap the set. The detection logic is fixed; only the glyphs vary.
//!
//! Pure compute, allocation-free, no terminal — like the rest of `re_render`.

// ───────────────────────── 8-neighbour bit layout ────────────────────────────
//
// The eight neighbours of a cell, as an 8-bit "solid presence" mask. Bits run
// clockwise from North. A bit is SET when that neighbour is SOLID (part of the
// silhouette's filled mass); CLEAR when it is open space / far background. The four
// diagonal bits are what the 4-neighbour contour pass cannot see — they are what
// let us tell a 45° slope from a true right-angle corner.

/// North neighbour solid (row-1, same col).
pub const NB_N: u8 = 1 << 0;
/// North-East neighbour solid (row-1, col+1).
pub const NB_NE: u8 = 1 << 1;
/// East neighbour solid (same row, col+1).
pub const NB_E: u8 = 1 << 2;
/// South-East neighbour solid (row+1, col+1).
pub const NB_SE: u8 = 1 << 3;
/// South neighbour solid (row+1, same col).
pub const NB_S: u8 = 1 << 4;
/// South-West neighbour solid (row+1, col-1).
pub const NB_SW: u8 = 1 << 5;
/// West neighbour solid (same row, col-1).
pub const NB_W: u8 = 1 << 6;
/// North-West neighbour solid (row-1, col-1).
pub const NB_NW: u8 = 1 << 7;

// ───────────────────────── slope orientation ─────────────────────────────────

/// Which way a 45° silhouette slope faces, named by the OUTER (open-space) corner —
/// the diagonal the outline runs along and the SOLID quadrant the chamfer fills.
///
/// The OUTLINE glyph follows the open↔solid DIVIDING DIAGONAL of the cell, NOT the
/// casual reading of the name: for `UpRight` the open wedge is NE and the solid mass
/// is SW, so the dividing line runs from the NW corner to the SE corner — a `╲`. (The
/// earlier code/docs paired each direction with the OPPOSITE diagonal, which made an
/// ascending stair render as if it descended — the corner-direction bug, #11.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopeDir {
    /// Open up-and-right (N+E edges); solid mass to the SW. The open↔solid divide runs
    /// NW-corner→SE-corner → outline `╲`, fill `◣` (the SW solid wedge). The RIGHT
    /// flank of a peak — solid steps DOWN as you go RIGHT.
    UpRight,
    /// Open up-and-left (N+W edges); solid mass to the SE. Divide NE→SW → outline `╱`,
    /// fill `◢` (SE wedge). The LEFT flank of a peak — solid steps DOWN as you go LEFT.
    UpLeft,
    /// Open down-and-right (S+E edges); solid mass to the NW. Divide NE→SW → outline
    /// `╱`, fill `◤` (NW wedge). An overhang / inverted slope facing down-right.
    DownRight,
    /// Open down-and-left (S+W edges); solid mass to the NE. Divide NW→SE → outline
    /// `╲`, fill `◥` (NE wedge). An overhang / inverted slope facing down-left.
    DownLeft,
}

// ───────────────────────── swappable glyph theme ─────────────────────────────

/// A swappable diagonal/triangle glyph set — the slope analogue of the
/// smart-materials `ContourTheme`: one detection logic, the glyph
/// vocabulary is DATA. Indexed by [`SlopeDir`] for the line and the triangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlopeTheme {
    /// Theme name (for `/iterate-render` selection / debug).
    pub name: &'static str,
    /// Diagonal OUTLINE glyph per [`SlopeDir`], in order
    /// `[UpRight, UpLeft, DownRight, DownLeft]`. The 45° stroke that replaces a
    /// stair corner on a thin (one-cell-wide) outline.
    pub line: [char; 4],
    /// Full-cell CHAMFER TRIANGLE per [`SlopeDir`], same order. Fills the SOLID
    /// half of the cell so a filled silhouette reads as a continuous slope.
    pub triangle: [char; 4],
}

impl SlopeTheme {
    /// The shipping Unicode theme: true half-cell diagonals + Geometric-Shapes
    /// triangles. All width-1 in a monospace font.
    ///
    /// Line:     `╱` U+2571, `╲` U+2572.
    /// Triangle: `◤` U+25E4 (upper-left), `◥` U+25E5 (upper-right),
    ///           `◣` U+25E3 (lower-left), `◢` U+25E2 (lower-right).
    ///
    /// Per [`SlopeDir`]: the triangle fills the SOLID quadrant and the line follows the
    /// open↔solid DIVIDING diagonal (the corner-to-corner cut), NOT its mirror —
    ///   UpRight   → solid SW → `◣`, divide NW→SE → line `╲`
    ///   UpLeft    → solid SE → `◢`, divide NE→SW → line `╱`
    ///   DownRight → solid NW → `◤`, divide NE→SW → line `╱`
    ///   DownLeft  → solid NE → `◥`, divide NW→SE → line `╲`
    /// (Bug #11 fix: the `line` row was previously the mirror of the divide, so an
    /// ascending stair read as descending. Triangles were already correct.)
    pub const UNICODE: SlopeTheme = SlopeTheme {
        name: "unicode",
        line: ['╲', '╱', '╱', '╲'],
        triangle: ['◣', '◢', '◤', '◥'],
    };

    /// ASCII degrade theme for glyph-poor terminals: `/` and `\` for the diagonal,
    /// and the same `/ \` reused as the "triangle" (ASCII has no half-cell triangle,
    /// so the chamfer degrades to the bare diagonal stroke — still reads as a slope,
    /// just unfilled). Selected by the present-layer capability probe, never here.
    pub const ASCII: SlopeTheme = SlopeTheme {
        name: "ascii",
        // Same divide as UNICODE: `\` ↔ `╲`, `/` ↔ `╱` (bug #11 fix — was mirrored).
        line: ['\\', '/', '/', '\\'],
        // The ASCII "triangle" is just the bare diagonal stroke (no half-cell fill), so
        // it follows the SAME divide as `line` — not its mirror.
        triangle: ['\\', '/', '/', '\\'],
    };

    /// The outline diagonal glyph for a slope direction.
    #[inline]
    pub fn line_for(&self, dir: SlopeDir) -> char {
        self.line[dir as usize]
    }

    /// The chamfer triangle glyph for a slope direction.
    #[inline]
    pub fn triangle_for(&self, dir: SlopeDir) -> char {
        self.triangle[dir as usize]
    }
}

impl Default for SlopeTheme {
    fn default() -> Self {
        SlopeTheme::UNICODE
    }
}

// ───────────────────────── detection ─────────────────────────────────────────

/// Build the 8-neighbour SOLID-presence mask for a cell from its eight neighbours.
/// Each entry is `Some(true)` if that neighbour is solid, `Some(false)` if it is
/// open (sky/far), `None` if off-grid. Off-grid is treated as OPEN (not solid): the
/// frame border is the open side, so a slope running off-screen still reads as a
/// slope rather than being suppressed.
///
/// Order: `[N, NE, E, SE, S, SW, W, NW]` — matches the `NB_*` bit order.
#[inline]
pub fn neighbour_mask(solid: [bool; 8]) -> u8 {
    let mut m = 0u8;
    for (i, &s) in solid.iter().enumerate() {
        if s {
            m |= 1 << i;
        }
    }
    m
}

/// Decide whether an OUTER silhouette CORNER cell is actually a 45° SLOPE, and which
/// way it faces. Returns `None` when the corner is a genuine right-angle (a true box
/// corner, e.g. the top of a wall meeting its side) that should KEEP its stair glyph.
///
/// `outer_mask` is the 4-bit N/E/S/W edge mask the contour pass built (bits:
/// `N=1, E=2, S=4, W=8`). Only the four CORNER masks (one H + one V edge) can be a
/// slope; a straight run (`N` only, `E` only, …) or a T/cross is never a slope.
///
/// `nb8` is the 8-neighbour SOLID mask from [`neighbour_mask`].
///
/// ## The diagonal-continuity test
///
/// A corner cell with edges open toward, say, N and E has its solid mass to the SW.
/// For it to be a *slope* (not a lone right-angle), the silhouette must CONTINUE
/// along the diagonal: the cell's solid mass should extend toward the SW corner AND
/// the run should keep stepping that way. Concretely, for `SlopeDir::UpRight`
/// (open NE, solid SW) we require:
///
/// * the **diagonal-into-solid** neighbour (SW) is SOLID — the mass continues into
///   the slope, AND
/// * the **anti-diagonal** neighbour (NE) is OPEN — the open wedge continues, AND
/// * the run is not a 1-cell nub: at least ONE of the two axis neighbours bridging
///   to the next step is solid (S or W), so a single isolated block does not get
///   chamfered into a meaningless triangle.
///
/// This is the marching condition for a 45° staircase: each step has solid on the
/// inner diagonal and open on the outer diagonal. A right-angle box corner FAILS it
/// (its inner diagonal neighbour is solid but BOTH axis-bridges are solid in a way
/// that makes it a filled corner, or the outer diagonal is also solid) — see the
/// per-direction guards below.
pub fn classify_slope(outer_mask: u8, nb8: u8) -> Option<SlopeDir> {
    // 4-neighbour edge bits (must match contour.rs N/E/S/W).
    const E_N: u8 = 1 << 0;
    const E_E: u8 = 1 << 1;
    const E_S: u8 = 1 << 2;
    const E_W: u8 = 1 << 3;

    // Only the four pure CORNER masks are slope candidates.
    let dir = match outer_mask {
        m if m == (E_N | E_E) => SlopeDir::UpRight, // open N+E, solid SW
        m if m == (E_N | E_W) => SlopeDir::UpLeft,  // open N+W, solid SE
        m if m == (E_S | E_E) => SlopeDir::DownRight, // open S+E, solid NW
        m if m == (E_S | E_W) => SlopeDir::DownLeft, // open S+W, solid NE
        _ => return None,
    };

    // Per-direction diagonal-continuity guard. (inner_diag, outer_diag, bridge_a,
    // bridge_b) = the SOLID diagonal the mass runs into, the OPEN diagonal opposite,
    // and the two axis neighbours one of which must be solid to prove a run.
    let (inner_diag, outer_diag, bridge_a, bridge_b) = match dir {
        SlopeDir::UpRight => (NB_SW, NB_NE, NB_S, NB_W),
        SlopeDir::UpLeft => (NB_SE, NB_NW, NB_S, NB_E),
        SlopeDir::DownRight => (NB_NW, NB_SE, NB_N, NB_W),
        SlopeDir::DownLeft => (NB_NE, NB_SW, NB_N, NB_E),
    };

    let inner_solid = nb8 & inner_diag != 0;
    let outer_open = nb8 & outer_diag == 0;
    let bridged = (nb8 & bridge_a != 0) || (nb8 & bridge_b != 0);

    if inner_solid && outer_open && bridged {
        Some(dir)
    } else {
        None
    }
}

/// Convenience predicate: does this cell sit on a 45° slope (any direction)?
#[inline]
pub fn is_slope(outer_mask: u8, nb8: u8) -> bool {
    classify_slope(outer_mask, nb8).is_some()
}

// ───────────────────────── sub-45° half-step (sub-cell, OPTIONAL) ─────────────

/// One half of a SHALLOW (sub-45°, ≈2:1) slope rendered across TWO cells via the
/// 2×2 quadrant glyphs (`▚`/`▞`), giving ONE intermediate angle between flat and
/// 45°. This is the APPROXIMATED rung: a run that advances one cell of
/// rise every two cells of travel.
///
/// `dir` is the 45° family the shallow run belongs to; `leading` is true for the
/// FIRST of the two cells in the step (the half that carries the diagonal quadrant),
/// false for the trailing half (which carries the filled/empty continuation).
///
/// Returns the quadrant glyph to place. Pairs with [`SlopeTheme`] for the colours at
/// the call site. The mapping uses the anti-diagonal `▞` (TR+BL) and main-diagonal
/// `▚` (TL+BR) quadrants from [`crate::quadrant::QUADRANT_GLYPHS`].
///
/// ## Wiring, and how two earlier correctness gaps were closed
///
/// This is WIRED behind [`crate::contour::SlopeStyle::Triangle`]: the 45° (1:1)
/// de-angularizer ([`classify_slope`]) and this sub-45° (2:1) half-step BOTH run
/// in the contour pass. The half-step fires only after the safe
/// [`classify_shallow_slope`] run-detector confirms a genuine 2:1 staircase over a
/// ±2-cell sampler; a flat top or a 45° run never triggers it (see that function +
/// its tests). Two earlier correctness gaps were closed as follows:
///
/// 1. **The 2:1 RUN detector** is [`classify_shallow_slope`]: it samples the silhouette
///    TOP-edge row at `c−2 … c+2`, requires a strict ONE-row step every TWO columns in a
///    consistent direction, and rejects both the flat run (step 0) and the 45° run (step
///    2 over ±2 columns). The `leading`/trailing phase falls out of which side the row
///    actually steps on.
/// 2. **Top- vs. bottom-edge glyph variants.** [`subcell_half_step`] takes a `top_edge`
///    flag: a TOP-edge (roofline, solid BELOW the slope line) keeps the cell's lower band
///    FILLED (`▄`/the diagonal+lower combo) so the cell never reads as an empty gap; a
///    BOTTOM-edge (rising floor, solid ABOVE) keeps the upper band filled. The old
///    bottom-quadrant-only mapping was the gap bug; the variants fix it.
#[inline]
pub fn subcell_half_step(dir: SlopeDir, leading: bool) -> char {
    // Back-compat shim: the original two-arg entry kept for the existing unit tests.
    // Defaults to the TOP-edge (roofline) variant — the common case the contour pass
    // wires. New call sites use [`subcell_half_step_edge`].
    subcell_half_step_edge(dir, leading, true)
}

/// As [`subcell_half_step`], but with an explicit `top_edge` flag selecting the glyph
/// family so the FILLED half of the cell is on the correct side (no empty-gap artifact).
///
/// * `top_edge = true` (a ROOFLINE: open ABOVE, solid BELOW the slope line) → the
///   cell's lower band is filled. The leading (about-to-step) half carries the diagonal
///   quadrant; the trailing half carries the FULL lower half-block `▄` so the descending
///   mass stays solid under the slope instead of leaving a hole.
/// * `top_edge = false` (a rising BOTTOM edge: solid ABOVE, open below) → the upper band
///   is filled: leading half the diagonal, trailing half the full upper half-block `▀`.
///
/// The diagonal quadrant family follows the 45° line of `dir` (post-#11 divide): the
/// `╲`-family (UpRight/DownLeft) uses the MAIN diagonal `▚` (TL+BR, a `╲` stroke); the
/// `╱`-family (UpLeft/DownRight) uses the ANTI-diagonal `▞` (TR+BL, a `╱` stroke). The
/// result is ONE intermediate angle (~26°/63°) between flat `─` and the 45° stroke —
/// the single approximated rung the spec promises — leaning the SAME way as the run.
#[inline]
pub fn subcell_half_step_edge(dir: SlopeDir, leading: bool, top_edge: bool) -> char {
    // The diagonal quadrant for the leading half (the ▚/▞ that carries the slope line).
    let diag = match dir {
        SlopeDir::UpRight | SlopeDir::DownLeft => '▚', // ╲-family → main diagonal
        SlopeDir::UpLeft | SlopeDir::DownRight => '▞', // ╱-family → anti-diagonal
    };
    if leading {
        // The half that carries the visible diagonal stroke.
        diag
    } else if top_edge {
        // Roofline: keep the lower band solid under the slope (no gap).
        '▄'
    } else {
        // Rising floor edge: keep the upper band solid above the slope.
        '▀'
    }
}

/// Decide whether a TOP-edge silhouette cell at column `c` (reference row `r`) sits on a
/// genuine SHALLOW (sub-45°, 2:1) DIAGONAL RUN, and if so which 45° family it belongs to
/// and whether `c` is the LEADING (about-to-step) half of its 2-column rung.
///
/// Returns `None` for a FLAT top (no run) and for a 45° run (the 1:1 [`classify_slope`]
/// path owns those) — the two negatives that, if mis-fired, would shatter flat wall tops
/// or fight the 45° pass. This is the safe run-detector the half-step gates on.
///
/// `top_row(col)` reports the silhouette TOP-edge row at `col`: the row index of the
/// topmost solid cell of the local silhouette segment, or `None` if that column is open
/// (no solid) in the sampled neighbourhood. The contour pass passes a closure that scans
/// a few rows around `r` for the first solid cell from the top.
///
/// ## The 2:1 test (why it is safe)
///
/// Let `b = top_row(c)`. We require, for a single direction `s ∈ {−1, +1}`:
///
/// * `top_row(c−2) == b − s` AND `top_row(c+2) == b + s` — the boundary descends/rises
///   by EXACTLY ONE row over TWO columns on each side: the defining 2:1 slope. A FLAT
///   top has `top_row(c±2) == b` (step 0) → rejected. A 45° run has `top_row(c±2) ==
///   b ± 2` (step 2 over 2 cols) → rejected.
/// * the intermediate samples `top_row(c−1)` and `top_row(c+1)` each lie BETWEEN the
///   endpoints (`b−s … b+s` inclusive) and never jump more than one row per column — so
///   a noisy/zig-zag edge (not a clean run) is rejected too.
///
/// The `dir` returned is the 45° family the shallow run approximates: a boundary that
/// steps DOWN as the column INCREASES (`s == +1`, going right) is the UpLeft `╲` family
/// (solid mass to the SE, descending right); stepping UP to the right (`s == −1`) is the
/// UpRight `╱` family (descending left). The `leading` half is the column at which the
/// row actually changes versus its same-rung neighbour: if `top_row(c+1) != b` the step
/// is immediately to the right, so `c` is the LEADING half of the rung; otherwise the
/// step is to the left and `c` is the trailing half.
#[inline]
pub fn classify_shallow_slope<F: Fn(i32) -> Option<i32>>(
    c: i32,
    top_row: F,
) -> Option<(SlopeDir, bool)> {
    let b = top_row(c)?;
    let lm2 = top_row(c - 2)?;
    let lp2 = top_row(c + 2)?;
    let lm1 = top_row(c - 1)?;
    let lp1 = top_row(c + 1)?;

    // Direction: the consistent one-row-per-two-column step. `s = +1` ⇒ the boundary
    // row INCREASES (descends on screen) as the column increases.
    let s = lp2 - b;
    if s != b - lm2 {
        return None; // the two sides disagree → not a single consistent run.
    }
    if s.abs() != 1 {
        // step 0 (flat top) or |step| ≥ 2 (45° or steeper) — both NOT a 2:1 run.
        return None;
    }

    // The intermediate samples must each sit between the endpoints and advance by at
    // most one row per column (a clean monotone staircase, not a zig-zag). For s = +1
    // the run is non-decreasing: b-1 ≤ lm1 ≤ b ≤ lp1 ≤ b+1; symmetric for s = -1.
    let (lo, hi) = if s > 0 {
        (b - 1, b + 1)
    } else {
        (b + 1, b - 1)
    };
    let between = |v: i32| (v - lo) * (v - hi) <= 0; // lo ≤ v ≤ hi (order-agnostic)
    if !between(lm1) || !between(lp1) {
        return None;
    }
    // No single-column jump may exceed one row (rules out a 2-step zig that happens to
    // average to 2:1).
    if (b - lm1).abs() > 1 || (lp1 - b).abs() > 1 || (lm1 - lm2).abs() > 1 || (lp2 - lp1).abs() > 1
    {
        return None;
    }

    // The 45° family (bug #11): a roofline DESCENDING-RIGHT (`s == +1`, top edge moves
    // DOWN as the column increases) has, at each cell, open up-and-right (NE) and solid
    // down-and-left (SW) → `SlopeDir::UpRight` (`╲` family). DESCENDING-LEFT (`s == -1`,
    // i.e. rising to the right) is open NW / solid SE → `UpLeft` (`╱` family). The
    // previous mapping was swapped, so a 2:1 stair leaned the wrong way.
    let dir = if s > 0 {
        SlopeDir::UpRight
    } else {
        SlopeDir::UpLeft
    };

    // Leading/trailing phase: the half where the row actually steps relative to the
    // same-rung neighbour. If the boundary changes immediately to the RIGHT of c
    // (`lp1 != b`), c is the LEADING half of this rung; otherwise the step is to the
    // left and c is the TRAILING half.
    let leading = lp1 != b;
    Some((dir, leading))
}

// ───────────────────────── neighbour helper for the pass ─────────────────────

/// Build the 8-neighbour solid mask for the cell at `(c, r)` from a grid accessor.
/// `is_solid` reports whether the cell at a (col, row) is part of the silhouette
/// mass; off-grid coordinates are reported as OPEN (`false`) by the closure the
/// caller passes (the contour pass clamps at the border).
///
/// Kept here so the detection and its neighbour gathering live together; the contour
/// pass calls this with a closure over its snapshot.
#[inline]
pub fn gather_nb8<F: Fn(isize, isize) -> bool>(c: usize, r: usize, is_solid: F) -> u8 {
    let (c, r) = (c as isize, r as isize);
    neighbour_mask([
        is_solid(c, r - 1),     // N
        is_solid(c + 1, r - 1), // NE
        is_solid(c + 1, r),     // E
        is_solid(c + 1, r + 1), // SE
        is_solid(c, r + 1),     // S
        is_solid(c - 1, r + 1), // SW
        is_solid(c - 1, r),     // W
        is_solid(c - 1, r - 1), // NW
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4-neighbour edge bits (mirror contour.rs).
    const E_N: u8 = 1 << 0;
    const E_E: u8 = 1 << 1;
    const E_S: u8 = 1 << 2;
    const E_W: u8 = 1 << 3;

    #[test]
    fn theme_triangle_fills_the_solid_quadrant() {
        let t = SlopeTheme::UNICODE;
        // The line follows the open↔solid DIVIDING diagonal (corner-to-corner cut), the
        // triangle fills the SOLID wedge. (Bug #11: line is the divide, not its mirror.)
        // UpRight: open NE, solid SW → divide NW→SE → line ╲ + lower-left triangle ◣.
        assert_eq!(t.triangle_for(SlopeDir::UpRight), '◣');
        assert_eq!(t.line_for(SlopeDir::UpRight), '╲');
        // UpLeft: open NW, solid SE → divide NE→SW → line ╱ + lower-right ◢.
        assert_eq!(t.triangle_for(SlopeDir::UpLeft), '◢');
        assert_eq!(t.line_for(SlopeDir::UpLeft), '╱');
        // DownRight: open SE, solid NW → divide NE→SW → line ╱ + upper-left ◤.
        assert_eq!(t.triangle_for(SlopeDir::DownRight), '◤');
        assert_eq!(t.line_for(SlopeDir::DownRight), '╱');
        // DownLeft: open SW, solid NE → divide NW→SE → line ╲ + upper-right ◥.
        assert_eq!(t.triangle_for(SlopeDir::DownLeft), '◥');
        assert_eq!(t.line_for(SlopeDir::DownLeft), '╲');
    }

    #[test]
    fn ascii_theme_degrades_to_slashes() {
        let t = SlopeTheme::ASCII;
        // `\` ↔ `╲`, `/` ↔ `╱`, following the same divide as UNICODE (bug #11 fix).
        assert_eq!(t.line_for(SlopeDir::UpRight), '\\');
        assert_eq!(t.line_for(SlopeDir::UpLeft), '/');
        assert_eq!(t.triangle_for(SlopeDir::DownRight), '/');
    }

    #[test]
    fn upright_slope_classifies_when_diagonal_continues() {
        // Open N+E (corner mask), solid SW continues the run, with a bridge to S.
        let outer = E_N | E_E;
        let nb8 = NB_SW | NB_S | NB_W; // solid SW + bridges; NE open.
        assert_eq!(classify_slope(outer, nb8), Some(SlopeDir::UpRight));
    }

    #[test]
    fn right_angle_box_corner_is_not_a_slope() {
        // A filled box corner: open N+E but the outer diagonal NE is ALSO solid
        // (the corner is part of a filled rectangle, not a diagonal run) → no slope.
        let outer = E_N | E_E;
        let nb8 = NB_SW | NB_S | NB_W | NB_NE; // NE solid → fails outer_open guard.
        assert_eq!(classify_slope(outer, nb8), None);
    }

    #[test]
    fn isolated_nub_is_not_a_slope() {
        // Open N+E, solid SW, but NEITHER bridge (S, W) is solid → a lone diagonal
        // touch, not a run → no chamfer.
        let outer = E_N | E_E;
        let nb8 = NB_SW; // only the inner diagonal; no axis bridge.
        assert_eq!(classify_slope(outer, nb8), None);
    }

    #[test]
    fn straight_run_is_never_a_slope() {
        // A single straight edge (only N) is a flat top, not a corner → no slope.
        assert_eq!(classify_slope(E_N, NB_S | NB_E | NB_W), None);
        // A T-junction / cross is never a slope.
        assert_eq!(classify_slope(E_N | E_E | E_W, 0xFF), None);
    }

    #[test]
    fn all_four_diagonal_orientations_classify() {
        // UpLeft: open N+W, solid SE.
        assert_eq!(
            classify_slope(E_N | E_W, NB_SE | NB_S | NB_E),
            Some(SlopeDir::UpLeft)
        );
        // DownRight: open S+E, solid NW.
        assert_eq!(
            classify_slope(E_S | E_E, NB_NW | NB_N | NB_W),
            Some(SlopeDir::DownRight)
        );
        // DownLeft: open S+W, solid NE.
        assert_eq!(
            classify_slope(E_S | E_W, NB_NE | NB_N | NB_E),
            Some(SlopeDir::DownLeft)
        );
    }

    #[test]
    fn gather_nb8_orders_neighbours_correctly() {
        // Mark only the N and SW neighbours solid; verify the right bits land.
        let nb = gather_nb8(5, 5, |c, r| (c, r) == (5, 4) || (c, r) == (4, 6));
        assert_eq!(nb & NB_N, NB_N, "N must be set");
        assert_eq!(nb & NB_SW, NB_SW, "SW must be set");
        assert_eq!(nb & NB_E, 0, "E must be clear");
    }

    #[test]
    fn subcell_half_step_uses_diagonal_quadrants() {
        // Post-#11: the quadrant leans the SAME way as the run. UpRight is the `╲`
        // family → main diagonal `▚`; UpLeft is the `╱` family → anti-diagonal `▞`.
        assert_eq!(subcell_half_step(SlopeDir::UpRight, true), '▚');
        assert_eq!(subcell_half_step(SlopeDir::UpLeft, true), '▞');
    }

    // ── sub-45° (2:1) run detector — the safe-wiring tests ───────────────────────

    /// Helper: a `top_row` closure over an explicit (col → top-edge-row) table. Any
    /// column outside the table is OPEN (`None`).
    fn rows_from(pairs: &[(i32, i32)]) -> impl Fn(i32) -> Option<i32> + '_ {
        move |c| pairs.iter().find(|(col, _)| *col == c).map(|(_, r)| *r)
    }

    #[test]
    fn shallow_slope_fires_on_a_true_2to1_descending_right_run() {
        // Top-edge row descends one per TWO columns going right: a 2:1 ╲-family run.
        //   col:  0 1 2 3 4 5 6
        //   row:  3 3 4 4 5 5 6
        let table = [(0, 3), (1, 3), (2, 4), (3, 4), (4, 5), (5, 5), (6, 6)];
        let tr = rows_from(&table);
        // Bug #11: a descending-RIGHT roofline is open NE / solid SW → UpRight (`╲`).
        let got = classify_shallow_slope(3, &tr);
        assert!(
            matches!(got, Some((SlopeDir::UpRight, _))),
            "a clean 2:1 descending-right run must classify as UpRight, got {got:?}"
        );
    }

    #[test]
    fn shallow_slope_fires_on_a_true_2to1_descending_left_run() {
        // Top-edge row descends one per two columns going LEFT (rises to the right):
        //   col:  0 1 2 3 4 5 6
        //   row:  6 5 5 4 4 3 3
        let table = [(0, 6), (1, 5), (2, 5), (3, 4), (4, 4), (5, 3), (6, 3)];
        let tr = rows_from(&table);
        let got = classify_shallow_slope(3, &tr);
        // Bug #11: descending-LEFT (rising to the right) is open NW / solid SE → UpLeft (`╱`).
        assert!(
            matches!(got, Some((SlopeDir::UpLeft, _))),
            "a clean 2:1 descending-left run must classify as UpLeft, got {got:?}"
        );
    }

    #[test]
    fn shallow_slope_is_silent_on_a_flat_top() {
        // A perfectly flat wall TOP edge: same row every column → step 0 → NO trigger.
        // This is the critical negative: a wrong detector would shatter flat tops.
        let table: Vec<(i32, i32)> = (0..8).map(|c| (c, 4)).collect();
        let tr = rows_from(&table);
        for c in 1..7 {
            assert_eq!(
                classify_shallow_slope(c, &tr),
                None,
                "a flat top at col {c} must NEVER classify as a shallow slope"
            );
        }
    }

    #[test]
    fn shallow_slope_is_silent_on_a_45deg_run() {
        // A 45° (1:1) run steps one row per ONE column → over ±2 the span is 2, not 1.
        // The 1:1 path (`classify_slope`) owns these; the shallow detector must NOT fire.
        //   col:  0 1 2 3 4 5 6
        //   row:  1 2 3 4 5 6 7
        let table: Vec<(i32, i32)> = (0..8).map(|c| (c, 1 + c)).collect();
        let tr = rows_from(&table);
        for c in 2..6 {
            assert_eq!(
                classify_shallow_slope(c, &tr),
                None,
                "a 45° run at col {c} must NOT classify as a 2:1 shallow slope"
            );
        }
    }

    #[test]
    fn shallow_slope_is_silent_on_a_noisy_zigzag() {
        // A zig-zag that is not a clean monotone run must be rejected even if some
        // ±2 endpoints happen to differ by one.
        //   col:  0 1 2 3 4 5 6
        //   row:  3 5 4 3 4 5 4   (jumps of 2 — not a clean staircase)
        let table = [(0, 3), (1, 5), (2, 4), (3, 3), (4, 4), (5, 5), (6, 4)];
        let tr = rows_from(&table);
        assert_eq!(
            classify_shallow_slope(3, &tr),
            None,
            "a noisy zig-zag top edge must not be read as a shallow slope"
        );
    }

    #[test]
    fn shallow_slope_is_silent_at_an_open_column() {
        // If any of the ±2 samples is OPEN (no solid → None), the run is unprovable.
        let table = [(0, 3), (1, 3), (2, 4), (3, 4) /* col 4,5,6 missing */];
        let tr = rows_from(&table);
        assert_eq!(
            classify_shallow_slope(3, &tr),
            None,
            "an incomplete neighbourhood must not classify (no guessing past open space)"
        );
    }

    #[test]
    fn shallow_slope_leading_phase_alternates_across_a_rung() {
        // Across a 2:1 run, consecutive same-rung cells alternate leading/trailing so
        // the diagonal quadrant lands once per rung (not on both halves).
        //   col:  0 1 2 3 4 5 6
        //   row:  3 3 4 4 5 5 6
        let table = [(0, 3), (1, 3), (2, 4), (3, 4), (4, 5), (5, 5), (6, 6)];
        let tr = rows_from(&table);
        // col 2 steps relative to col1 (row 3→4 happened at col2): the step is at col2,
        // so col2's right neighbour (col3) is SAME row → col2 is the trailing half;
        // col3's right neighbour (col4) steps → col3 is the leading half.
        let (_, lead2) = classify_shallow_slope(2, &tr).unwrap();
        let (_, lead3) = classify_shallow_slope(3, &tr).unwrap();
        assert_ne!(
            lead2, lead3,
            "adjacent same-rung cells must alternate phase"
        );
    }

    #[test]
    fn subcell_half_step_edge_fills_the_correct_band_no_gap() {
        // TOP edge (roofline, solid below): the trailing half must be a FILLED lower
        // band (`▄`), never an empty/quarter glyph that would read as a gap.
        assert_eq!(subcell_half_step_edge(SlopeDir::UpLeft, false, true), '▄');
        assert_eq!(subcell_half_step_edge(SlopeDir::UpRight, false, true), '▄');
        // BOTTOM edge (rising floor, solid above): trailing half fills the UPPER band.
        assert_eq!(subcell_half_step_edge(SlopeDir::UpLeft, false, false), '▀');
        assert_eq!(subcell_half_step_edge(SlopeDir::UpRight, false, false), '▀');
        // Leading half carries the diagonal quadrant on either edge (post-#11: UpLeft
        // is the `╱`-family → `▞`, UpRight the `╲`-family → `▚`).
        assert_eq!(subcell_half_step_edge(SlopeDir::UpLeft, true, true), '▞');
        assert_eq!(subcell_half_step_edge(SlopeDir::UpRight, true, true), '▚');
        // The back-compat two-arg entry equals the TOP-edge variant.
        assert_eq!(
            subcell_half_step(SlopeDir::UpLeft, true),
            subcell_half_step_edge(SlopeDir::UpLeft, true, true)
        );
    }

    /// The sub-45° half-step helper yields the correct quadrant mapping
    /// for all four orientations × both halves. The two-arg entry defaults to the
    /// TOP-edge (roofline) variant: leading = the diagonal quadrant, trailing = the
    /// FILLED lower band `▄` (no empty gap — the old `▖`/`▗` gap bug is fixed).
    #[test]
    fn subcell_half_step_full_mapping_is_correct() {
        // Post-#11 families: `╲` = {UpRight, DownLeft} → `▚`; `╱` = {UpLeft, DownRight} → `▞`.
        assert_eq!(subcell_half_step(SlopeDir::UpRight, true), '▚');
        assert_eq!(subcell_half_step(SlopeDir::UpRight, false), '▄');
        assert_eq!(subcell_half_step(SlopeDir::DownLeft, true), '▚');
        assert_eq!(subcell_half_step(SlopeDir::DownLeft, false), '▄');
        assert_eq!(subcell_half_step(SlopeDir::UpLeft, true), '▞');
        assert_eq!(subcell_half_step(SlopeDir::UpLeft, false), '▄');
        assert_eq!(subcell_half_step(SlopeDir::DownRight, true), '▞');
        assert_eq!(subcell_half_step(SlopeDir::DownRight, false), '▄');
        for dir in [
            SlopeDir::UpRight,
            SlopeDir::UpLeft,
            SlopeDir::DownRight,
            SlopeDir::DownLeft,
        ] {
            for leading in [true, false] {
                let g = subcell_half_step(dir, leading);
                assert!(
                    "▘▝▖▗▚▞▌▐▀▄█".contains(g),
                    "half-step glyph {g:?} must be a quadrant block"
                );
            }
        }
    }
}
