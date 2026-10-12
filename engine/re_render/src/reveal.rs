//! `reveal` — the **reveal compositor** for the FOV-zoom transition.
//!
//! When the camera's field of view changes (a zoom-in to a narrow FOV, or a
//! zoom-out to a wide one), we don't snap from the old frame to the new one — we
//! *redraw* the new frame **over** the old one in a screen-space order, so the eye
//! reads a sweep: a disc that grows from the centre, a ring that closes in from the
//! rim, or a spiral that unwinds. This module is the pure compositor that, given an
//! `old` frame, a `new` frame, and a `progress` in `[0,1]`, decides per cell whether
//! it shows NEW (already swept past) or OLD (not yet reached).
//!
//! # Where it sits
//!
//! This is Layer A, pure compute (`re_core` only — no terminal, no Bevy, no clock).
//! It is intentionally NOT wired into the live render: the FOV-lerp, the frame
//! snapshotting, and the key binding live downstream in `re_game` / `re_present`,
//! which call [`composite_reveal`] once per transition frame with the progress they
//! computed. Same `(old, new, progress, mode, feather)` → byte-identical `out`.
//!
//! # The spatial order (reused, not reinvented)
//!
//! The reveal order is driven entirely by the existing aspect-corrected
//! [`crate::posteffect::screen_radius`] (a visual circle on a 2:1 terminal) and
//! [`crate::posteffect::spiral_index`] (monotonic-from-centre swirl). We normalise
//! each by its value at the frame **corner** (the farthest cell), so `progress == 1`
//! reveals even the corners. The four modes differ only in which end of that order
//! the sweep starts from.
//!
//! # The feather (smoothness)
//!
//! A hard wipe is a crisp ring/spiral edge — correct, but it can read as a jagged
//! moving boundary. An optional **feather band** of configurable width softens the
//! frontier: cells within the band on the revealed side are shown but *degraded* —
//! their colour is blended back toward the OLD frame proportionally to how close they
//! are to the edge, so the wipe dissolves rather than snaps. Feather width `0` is a
//! pure hard wipe (and is byte-exact at the endpoints).

use re_core::{Cell, Framebuffer};

use crate::posteffect::{screen_radius, spiral_index};

/// In which screen-space order the NEW frame is swept in over the OLD one.
///
/// The two `*Out` modes grow the new frame from the centre outward (the natural read
/// for a **zoom-IN** to a narrow FOV — the world rushes at you from the middle); the
/// two `*In` modes close it in from the outer rim (the read for a **zoom-to-WIDE**
/// FOV — the edges of the world peel inward). `Radial*` sweeps a clean disc/ring;
/// `Spiral*` unwinds the same order as a swirl.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevealMode {
    /// Disc growing from the centre outward. NEW where `radius/max <= progress`.
    /// Use for zoom-IN.
    RadialOut,
    /// Ring closing from the outer rim inward. NEW where `radius/max >= 1-progress`.
    /// Use for zoom-to-WIDE.
    RadialIn,
    /// Spiral unwinding from the centre outward. NEW where
    /// `spiral/max <= progress`.
    Spiral,
    /// Spiral closing from the rim inward. NEW where `spiral/max >= 1-progress`.
    SpiralIn,
    /// A stable per-cell HASH order, biased toward the top rows first, growing
    /// outward with no single geometric centre — used by the host's level-
    /// **entry** "resolve" transition (a per-cell threshold field rather than
    /// a radial/spiral sweep, so the entry reveal reads as the scene
    /// "resolving into focus" cell by cell instead of growing from one visual
    /// centre). NEW where `hashed_order/1 <= progress` (see [`hashed_order`] —
    /// its own weighted mix is already normalized to `[0,1]`, so unlike the
    /// radial/spiral orders it needs no corner sample).
    Hashed,
}

impl RevealMode {
    /// True for the rim-in variants (`RadialIn`, `SpiralIn`) — the reveal grows from
    /// the outer edge inward rather than the centre outward.
    #[inline]
    fn is_inward(self) -> bool {
        matches!(self, RevealMode::RadialIn | RevealMode::SpiralIn)
    }

    /// True for the spiral variants — order by `spiral_index` rather than
    /// `screen_radius`.
    #[inline]
    fn is_spiral(self) -> bool {
        matches!(self, RevealMode::Spiral | RevealMode::SpiralIn)
    }

    /// True for [`RevealMode::Hashed`] — order by [`hashed_order`] rather than
    /// a radial/spiral screen-space order.
    #[inline]
    fn is_hashed(self) -> bool {
        matches!(self, RevealMode::Hashed)
    }
}

/// The per-cell reveal order for [`RevealMode::Hashed`]: a weighted mix of a
/// stable cell hash, a top-to-bottom row bias, and a faint diagonal stripe —
/// own formula, own weights (study-only inspiration, not a copy: see the
/// `RevealMode::Hashed` doc). Already normalized to `[0,1]` (each term is a
/// `[0,1)` value and the weights sum to `1.0`), so it needs no corner-sample
/// normalization the way [`screen_radius`]/[`spiral_index`] do.
///
/// Weights: `0.6` hash (the bulk of the texture — no two neighbouring cells
/// reveal in lockstep), `0.27` row bias (the top of the screen resolves
/// first, which reads as "the ceiling loads in, then the floor"), `0.13`
/// diagonal stripe (a faint directional grain so same-row cells don't all
/// hash independently with no visible structure at all).
#[inline]
fn hashed_order(col: usize, row: usize, cols: usize, rows: usize) -> f32 {
    let h = crate::ascii_shader::hash01(crate::ascii_shader::hash2i(col as i32, row as i32));
    let row_frac = if rows > 1 {
        row as f32 / (rows - 1) as f32
    } else {
        0.0
    };
    let denom = (cols + rows).saturating_sub(2).max(1) as f32;
    let diag = ((col + row) as f32 / denom).clamp(0.0, 1.0);
    (0.6 * h + 0.27 * row_frac + 0.13 * diag).clamp(0.0, 1.0)
}

/// The reveal **lead** for one cell at `progress`: positive once the frontier
/// has passed it, non-positive while it is still OLD. Shared by
/// [`composite_reveal_feathered`] and [`apply_scramble_frontier`] so both
/// agree, cell for cell, on exactly where the frontier sits.
#[inline]
fn reveal_lead(
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
    mode: RevealMode,
    progress: f32,
    max_order: f32,
) -> f32 {
    let order = if mode.is_hashed() {
        hashed_order(col, row, cols, rows)
    } else if mode.is_spiral() {
        spiral_index(col, row, cols, rows)
    } else {
        screen_radius(col, row, cols, rows)
    };
    let t01 = (order / max_order).clamp(0.0, 1.0);
    if mode.is_inward() {
        t01 - (1.0 - progress)
    } else {
        progress - t01
    }
}

/// The normalizing denominator [`reveal_lead`] divides the raw per-cell order
/// by. [`RevealMode::Hashed`]'s [`hashed_order`] is already `[0,1]`-bounded by
/// construction (no geometric corner to sample); every other mode uses the
/// frame-corner [`corner_order`].
#[inline]
fn max_order_for(cols: usize, rows: usize, mode: RevealMode) -> f32 {
    if mode.is_hashed() {
        1.0
    } else {
        corner_order(cols, rows, mode.is_spiral()).max(f32::EPSILON)
    }
}

/// Default feather band width, in normalized-order units (the same `[0,1]` space the
/// reveal threshold lives in). `0` is a hard wipe; this is a gentle softening of the
/// frontier. Roughly 1.5 cells of radius on a typical frame, expressed as a fraction
/// of the corner radius — small, so the band is a thin moving ribbon, not a wide fade.
pub const DEFAULT_FEATHER: f32 = 0.06;

/// Composite `new` over `old` into `out` by the reveal `mode` at `progress`, with a
/// **hard wipe** (no feather). Convenience wrapper over [`composite_reveal_feathered`]
/// with feather `0.0`.
///
/// Contract:
/// * `old`, `new`, `out` must share dimensions. If any differ from `out`, `out` is
///   left untouched (defensive — the caller owns matching-size buffers).
/// * `out` must be **distinct** from `old` and `new` (no aliasing). The function
///   writes `out` cell-by-cell while reading `old`/`new`, so aliasing would corrupt
///   the source mid-pass. Callers keep three buffers (or a dedicated scratch `out`).
/// * `progress <= 0.0` ⇒ `out` is a byte-identical copy of `old`.
/// * `progress >= 1.0` ⇒ `out` is a byte-identical copy of `new`.
/// * every cell is defined for every input — empty cells, equal frames, any size.
pub fn composite_reveal(
    old: &Framebuffer,
    new: &Framebuffer,
    out: &mut Framebuffer,
    progress: f32,
    mode: RevealMode,
) {
    composite_reveal_feathered(old, new, out, progress, mode, 0.0);
}

/// Composite `new` over `old` into `out` by the reveal `mode` at `progress`, with an
/// optional `feather` band (in normalized-order units, e.g. [`DEFAULT_FEATHER`]) that
/// softens the reveal frontier. `feather <= 0.0` is a hard wipe identical to
/// [`composite_reveal`].
///
/// See [`composite_reveal`] for the dimension / aliasing / endpoint contract.
///
/// # How the feather softens the edge
///
/// For each cell we compute its normalized order `t01 ∈ [0,1]` and a signed *lead*
/// `d` = how far the reveal frontier has already passed this cell (positive = already
/// revealed). A cell with `d >= feather` is fully NEW; `d <= 0` is fully OLD; in
/// between (`0 < d < feather`) the cell is on the revealed side but freshly uncovered,
/// so we show the NEW glyph but blend its colour toward the OLD colour by
/// `1 - d/feather` (full old-tint right at the edge, fading to pure new as the
/// frontier moves on). The result is a thin dissolving ribbon instead of a crisp ring.
pub fn composite_reveal_feathered(
    old: &Framebuffer,
    new: &Framebuffer,
    out: &mut Framebuffer,
    progress: f32,
    mode: RevealMode,
    feather: f32,
) {
    let cols = out.cols();
    let rows = out.rows();
    // Dimension guard: all three must match `out`, else leave `out` untouched.
    if old.cols() != cols || old.rows() != rows || new.cols() != cols || new.rows() != rows {
        return;
    }
    if cols == 0 || rows == 0 {
        return;
    }

    // Endpoint fast-paths: exact copies, no per-cell order math.
    if progress <= 0.0 {
        copy_into(old, out);
        return;
    }
    if progress >= 1.0 {
        copy_into(new, out);
        return;
    }

    let p = progress.clamp(0.0, 1.0);
    let feather = feather.max(0.0);
    let max_order = max_order_for(cols, rows, mode);

    for row in 0..rows {
        for col in 0..cols {
            // `lead` d = signed distance (in normalized-order units) by which the
            // frontier has passed this cell. d > 0 ⇒ revealed; d <= 0 ⇒ still old.
            let d = reveal_lead(col, row, cols, rows, mode, p, max_order);

            let cell = if d <= 0.0 {
                // Not yet reached — raw OLD.
                *old.get(col, row).unwrap()
            } else if feather <= 0.0 || d >= feather {
                // Fully revealed — raw NEW.
                *new.get(col, row).unwrap()
            } else {
                // Inside the feather band on the revealed side: NEW glyph, colour
                // blended back toward OLD by how fresh the reveal is.
                let new_c = *new.get(col, row).unwrap();
                let old_c = *old.get(col, row).unwrap();
                feather_cell(&new_c, &old_c, d / feather)
            };
            out.set(col, row, cell);
        }
    }
}

/// The order value at the frame corner `(0,0)` — the farthest cell from centre, hence
/// the maximum of `screen_radius` / `spiral_index` over the grid. Used to normalise
/// the per-cell order into `[0,1]` so `progress == 1` reveals the corners too.
#[inline]
fn corner_order(cols: usize, rows: usize, spiral: bool) -> f32 {
    if spiral {
        spiral_index(0, 0, cols, rows)
    } else {
        screen_radius(0, 0, cols, rows)
    }
}

/// Copy every cell of `src` into `dst` (same dims assumed; the caller guards size).
#[inline]
fn copy_into(src: &Framebuffer, dst: &mut Framebuffer) {
    let cols = dst.cols();
    let rows = dst.rows();
    for row in 0..rows {
        for col in 0..cols {
            if let Some(c) = src.get(col, row) {
                dst.set(col, row, *c);
            }
        }
    }
}

/// Build a feathered frontier cell: the NEW cell's glyph + geometry, but its `fg`/`bg`
/// blended toward the OLD cell's colours. `t` is the *reveal freshness* in `[0,1]`:
/// `t = 0` right at the frontier (full OLD tint), `t = 1` at the band's inner edge
/// (pure NEW). The blend weight toward OLD is therefore `1 - t`.
///
/// Only colour is degraded — glyph/depth/surface come from NEW — so the softening
/// reads as a brief colour dissolve at the moving edge, not a glyph swap.
#[inline]
fn feather_cell(new_c: &Cell, old_c: &Cell, t: f32) -> Cell {
    let w_old = (1.0 - t).clamp(0.0, 1.0);
    let mut c = *new_c;
    c.fg = blend(new_c.fg, old_c.fg, w_old);
    c.bg = blend(new_c.bg, old_c.bg, w_old);
    c
}

/// Linearly blend packed `0x00RRGGBB` colours: `from*(1-w) + to*w` per channel.
/// `w = 0` ⇒ `from`, `w = 1` ⇒ `to`. Used by the feather to tint NEW toward OLD.
#[inline]
fn blend(from: u32, to: u32, w: f32) -> u32 {
    let w = w.clamp(0.0, 1.0);
    let lerp = |a: u32, b: u32| {
        let a = a as f32;
        let b = b as f32;
        (a + (b - a) * w).round().clamp(0.0, 255.0) as u32
    };
    let (fr, fg, fb) = ((from >> 16) & 0xFF, (from >> 8) & 0xFF, from & 0xFF);
    let (tr, tg, tb) = ((to >> 16) & 0xFF, (to >> 8) & 0xFF, to & 0xFF);
    (lerp(fr, tr) << 16) | (lerp(fg, tg) << 8) | lerp(fb, tb)
}

/// A deterministic, seeded glyph pick from `glyphs` for one cell. `glyphs`
/// empty ⇒ a space (callers guard against this before calling in a hot loop).
#[inline]
fn scramble_glyph(seed: u64, col: usize, row: usize, glyphs: &[char]) -> char {
    if glyphs.is_empty() {
        return ' ';
    }
    let a = (col as i32) ^ (seed as i32);
    let b = (row as i32) ^ ((seed >> 32) as i32);
    let h = crate::ascii_shader::hash2i(a, b);
    glyphs[(h as usize) % glyphs.len()]
}

/// Optional decoration layered OVER an already-composited reveal frame (`out`,
/// produced by [`composite_reveal_feathered`] with the SAME `mode`/`progress`):
/// a thin **scramble-glyph frontier band** just ahead of the reveal, still on
/// the OLD side (`-band <= d <= 0`, see [`reveal_lead`]) — cells there are
/// overwritten with a glyph from `glyphs` and tinted toward `tint` by up to
/// `tint_strength`, fading to no tint at the far edge of the band (`d ==
/// -band`) and full tint right at the frontier (`d == 0`). Cells outside the
/// band, and cells already revealed (`d > 0`, left as raw NEW by the earlier
/// composite call), are untouched.
///
/// Pure decoration: it never changes WHICH cells are OLD vs NEW, only how the
/// still-OLD ones just ahead of the frontier are drawn. `band <= 0`, an empty
/// `glyphs`, or a `(cols, rows)` mismatch against `out` is a no-op — the
/// caller's plain [`composite_reveal_feathered`] result stands unmodified.
#[allow(clippy::too_many_arguments)]
pub fn apply_scramble_frontier(
    out: &mut Framebuffer,
    mode: RevealMode,
    progress: f32,
    band: f32,
    glyphs: &[char],
    tint: u32,
    tint_strength: f32,
    seed: u64,
) {
    let cols = out.cols();
    let rows = out.rows();
    if cols == 0 || rows == 0 || band <= 0.0 || glyphs.is_empty() {
        return;
    }
    let p = progress.clamp(0.0, 1.0);
    let max_order = max_order_for(cols, rows, mode);
    let strength = tint_strength.clamp(0.0, 1.0);

    for row in 0..rows {
        for col in 0..cols {
            let d = reveal_lead(col, row, cols, rows, mode, p, max_order);
            if d > 0.0 || d < -band {
                // Already revealed (raw NEW), or too far ahead of the frontier
                // to be in the band at all.
                continue;
            }
            let Some(mut c) = out.get(col, row).copied() else {
                continue;
            };
            c.glyph = u32::from(scramble_glyph(seed, col, row, glyphs));
            // `local`: 0 at the band's far edge (d = -band), 1 right at the
            // frontier (d = 0) — the tint fades OUT as a cell gets farther
            // from the reveal, not a flat wash over the whole band.
            let local = (1.0 + d / band).clamp(0.0, 1.0);
            c.fg = blend(c.fg, tint, strength * local);
            c.bg = blend(c.bg, tint, strength * local * 0.4);
            out.set(col, row, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, Framebuffer, SurfaceKind};

    /// A solid frame where every cell carries the given glyph + colours, so we can
    /// tell OLD from NEW by inspecting any cell.
    fn solid(cols: usize, rows: usize, glyph: char, fg: u32, bg: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                fb.set(
                    col,
                    row,
                    Cell {
                        glyph: u32::from(glyph),
                        fg,
                        bg,
                        surface: SurfaceKind::WallSolid,
                        ..Cell::default()
                    },
                );
            }
        }
        fb
    }

    fn old_frame(cols: usize, rows: usize) -> Framebuffer {
        solid(cols, rows, 'O', 0x0010_2030, 0x0001_0203)
    }
    fn new_frame(cols: usize, rows: usize) -> Framebuffer {
        solid(cols, rows, 'N', 0x00C0_D0E0, 0x00F0_E0D0)
    }

    fn center(cols: usize, rows: usize) -> (usize, usize) {
        (cols / 2, rows / 2)
    }

    // ── endpoint contract: progress 0 → old, progress 1 → new ──────────────

    #[test]
    fn progress_zero_is_old_byte_identical() {
        let old = old_frame(40, 20);
        let new = new_frame(40, 20);
        for mode in [
            RevealMode::RadialOut,
            RevealMode::RadialIn,
            RevealMode::Spiral,
            RevealMode::SpiralIn,
            RevealMode::Hashed,
        ] {
            let mut out = Framebuffer::new(40, 20);
            composite_reveal(&old, &new, &mut out, 0.0, mode);
            assert_eq!(
                out.cells(),
                old.cells(),
                "progress 0 must equal OLD ({mode:?})"
            );
            // Also for negative / sub-zero progress.
            composite_reveal(&old, &new, &mut out, -0.5, mode);
            assert_eq!(
                out.cells(),
                old.cells(),
                "progress<0 must equal OLD ({mode:?})"
            );
        }
    }

    #[test]
    fn progress_one_is_new_byte_identical() {
        let old = old_frame(40, 20);
        let new = new_frame(40, 20);
        for mode in [
            RevealMode::RadialOut,
            RevealMode::RadialIn,
            RevealMode::Spiral,
            RevealMode::SpiralIn,
            RevealMode::Hashed,
        ] {
            let mut out = Framebuffer::new(40, 20);
            composite_reveal(&old, &new, &mut out, 1.0, mode);
            assert_eq!(
                out.cells(),
                new.cells(),
                "progress 1 must equal NEW ({mode:?})"
            );
            composite_reveal(&old, &new, &mut out, 1.5, mode);
            assert_eq!(
                out.cells(),
                new.cells(),
                "progress>1 must equal NEW ({mode:?})"
            );
        }
    }

    // ── mid-progress RadialOut: centre NEW, corner OLD ─────────────────────

    #[test]
    fn radial_out_mid_reveals_center_not_corner() {
        let cols = 60;
        let rows = 30;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::RadialOut);

        let (cx, cy) = center(cols, rows);
        assert_eq!(
            out.get(cx, cy).unwrap().glyph,
            u32::from('N'),
            "RadialOut@0.5: centre is revealed (NEW)"
        );
        assert_eq!(
            out.get(0, 0).unwrap().glyph,
            u32::from('O'),
            "RadialOut@0.5: corner is not yet revealed (OLD)"
        );
    }

    // ── mid-progress RadialIn: corner NEW, centre OLD (the opposite) ───────

    #[test]
    fn radial_in_mid_reveals_corner_not_center() {
        let cols = 60;
        let rows = 30;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::RadialIn);

        let (cx, cy) = center(cols, rows);
        assert_eq!(
            out.get(0, 0).unwrap().glyph,
            u32::from('N'),
            "RadialIn@0.5: corner is revealed first (NEW)"
        );
        assert_eq!(
            out.get(cx, cy).unwrap().glyph,
            u32::from('O'),
            "RadialIn@0.5: centre is last (OLD)"
        );
    }

    // ── revealed cells are EXACTLY the new frame's cells (no corruption) ───

    #[test]
    fn revealed_cells_equal_new_exactly() {
        let cols = 50;
        let rows = 26;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        // Hard wipe (no feather) so revealed cells must be byte-exact NEW.
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::RadialOut);
        for row in 0..rows {
            for col in 0..cols {
                let c = out.get(col, row).unwrap();
                // Every cell is either exactly OLD or exactly NEW — never garbage.
                let is_old = c == old.get(col, row).unwrap();
                let is_new = c == new.get(col, row).unwrap();
                assert!(
                    is_old || is_new,
                    "hard wipe cell ({col},{row}) must be raw OLD or raw NEW"
                );
            }
        }
    }

    // ── Spiral monotonic reveal: a cell revealed at p stays revealed at p+ ──

    #[test]
    fn spiral_reveal_is_monotonic_in_progress() {
        let cols = 50;
        let rows = 26;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);

        let mut prev_revealed = 0usize;
        let mut last_set: Vec<bool> = vec![false; cols * rows];
        for step in 0..=10 {
            let p = step as f32 / 10.0;
            let mut out = Framebuffer::new(cols, rows);
            composite_reveal(&old, &new, &mut out, p, RevealMode::Spiral);

            let mut revealed = 0usize;
            for row in 0..rows {
                for col in 0..cols {
                    let is_new = out.get(col, row).unwrap().glyph == u32::from('N');
                    let idx = row * cols + col;
                    if is_new {
                        revealed += 1;
                    }
                    // Monotonic: a cell once revealed never reverts as progress grows.
                    if last_set[idx] {
                        assert!(
                            is_new,
                            "Spiral: cell ({col},{row}) un-revealed when progress rose to {p}"
                        );
                    }
                    last_set[idx] = is_new;
                }
            }
            assert!(
                revealed >= prev_revealed,
                "Spiral: revealed count must not shrink as progress rises ({revealed} < {prev_revealed} at p={p})"
            );
            prev_revealed = revealed;
        }
        // By the end the whole frame is revealed.
        assert_eq!(
            prev_revealed,
            cols * rows,
            "Spiral@1.0 must reveal every cell"
        );
    }

    // ── feather: a frontier cell is degraded (neither raw old nor raw new) ─

    #[test]
    fn feather_softens_frontier_cells() {
        let cols = 80;
        let rows = 40;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        // A wide feather so the band is easy to catch.
        composite_reveal_feathered(&old, &new, &mut out, 0.5, RevealMode::RadialOut, 0.15);

        // Scan for at least one cell that is neither raw OLD nor raw NEW: that is a
        // feathered frontier cell (NEW glyph, colour blended toward OLD).
        let mut found_degraded = false;
        for row in 0..rows {
            for col in 0..cols {
                let c = out.get(col, row).unwrap();
                let is_old = c == old.get(col, row).unwrap();
                let is_new = c == new.get(col, row).unwrap();
                if !is_old && !is_new {
                    // It must be a softened NEW cell: NEW glyph, but colours between.
                    assert_eq!(c.glyph, u32::from('N'), "feather cell keeps the NEW glyph");
                    found_degraded = true;
                }
            }
        }
        assert!(
            found_degraded,
            "feather>0 must produce at least one degraded frontier cell"
        );
    }

    #[test]
    fn feather_zero_is_a_hard_wipe() {
        let cols = 80;
        let rows = 40;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut hard = Framebuffer::new(cols, rows);
        let mut feathered0 = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut hard, 0.5, RevealMode::RadialOut);
        composite_reveal_feathered(&old, &new, &mut feathered0, 0.5, RevealMode::RadialOut, 0.0);
        // feather 0 must equal the hard wipe exactly.
        assert_eq!(hard.cells(), feathered0.cells());
        // And every cell of the hard wipe is raw OLD or raw NEW (no degradation).
        for row in 0..rows {
            for col in 0..cols {
                let c = hard.get(col, row).unwrap();
                let is_old = c == old.get(col, row).unwrap();
                let is_new = c == new.get(col, row).unwrap();
                assert!(
                    is_old || is_new,
                    "feather=0 cell must be raw OLD or raw NEW"
                );
            }
        }
    }

    // ── degenerate inputs are defined ──────────────────────────────────────

    #[test]
    fn mismatched_dims_leave_out_untouched() {
        let old = old_frame(40, 20);
        let new = new_frame(10, 10); // wrong size
        let mut out = old_frame(40, 20); // pre-seed with OLD
        let before = out.cells().to_vec();
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::RadialOut);
        assert_eq!(
            out.cells(),
            before.as_slice(),
            "size mismatch must not corrupt out"
        );
    }

    #[test]
    fn empty_grid_is_a_noop() {
        let old = Framebuffer::new(0, 0);
        let new = Framebuffer::new(0, 0);
        let mut out = Framebuffer::new(0, 0);
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::Spiral);
        assert_eq!(out.cells().len(), 0);
    }

    #[test]
    fn equal_frames_reveal_identically() {
        // If OLD == NEW, any progress/mode yields that same frame.
        let f = solid(30, 16, 'X', 0x0011_2233, 0x0044_5566);
        let mut out = Framebuffer::new(30, 16);
        composite_reveal_feathered(&f, &f, &mut out, 0.5, RevealMode::Spiral, 0.1);
        assert_eq!(
            out.cells(),
            f.cells(),
            "equal frames must composite to themselves"
        );
    }

    // ── feather endpoints stay exact (the contract holds with feather on) ──

    #[test]
    fn feathered_endpoints_are_exact() {
        let old = old_frame(48, 24);
        let new = new_frame(48, 24);
        let mut out = Framebuffer::new(48, 24);
        composite_reveal_feathered(&old, &new, &mut out, 0.0, RevealMode::SpiralIn, 0.2);
        assert_eq!(
            out.cells(),
            old.cells(),
            "feathered progress 0 still exact OLD"
        );
        composite_reveal_feathered(&old, &new, &mut out, 1.0, RevealMode::SpiralIn, 0.2);
        assert_eq!(
            out.cells(),
            new.cells(),
            "feathered progress 1 still exact NEW"
        );
    }

    // ── Hashed mode: monotonic, deterministic, top-biased ──────────────────

    #[test]
    fn hashed_reveal_is_monotonic_in_progress() {
        let cols = 50;
        let rows = 26;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);

        let mut prev_revealed = 0usize;
        let mut last_set: Vec<bool> = vec![false; cols * rows];
        for step in 0..=10 {
            let p = step as f32 / 10.0;
            let mut out = Framebuffer::new(cols, rows);
            composite_reveal(&old, &new, &mut out, p, RevealMode::Hashed);

            let mut revealed = 0usize;
            for row in 0..rows {
                for col in 0..cols {
                    let is_new = out.get(col, row).unwrap().glyph == u32::from('N');
                    let idx = row * cols + col;
                    if is_new {
                        revealed += 1;
                    }
                    if last_set[idx] {
                        assert!(
                            is_new,
                            "Hashed: cell ({col},{row}) un-revealed when progress rose to {p}"
                        );
                    }
                    last_set[idx] = is_new;
                }
            }
            assert!(
                revealed >= prev_revealed,
                "Hashed: revealed count must not shrink as progress rises ({revealed} < {prev_revealed} at p={p})"
            );
            prev_revealed = revealed;
        }
        assert_eq!(
            prev_revealed,
            cols * rows,
            "Hashed@1.0 must reveal every cell"
        );
    }

    #[test]
    fn hashed_reveal_is_deterministic() {
        // Same inputs, same order: two independent runs at the same progress
        // must agree cell-for-cell (no wall-clock, no thread-local RNG).
        let cols = 44;
        let rows = 22;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut a = Framebuffer::new(cols, rows);
        let mut b = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut a, 0.37, RevealMode::Hashed);
        composite_reveal(&old, &new, &mut b, 0.37, RevealMode::Hashed);
        assert_eq!(
            a.cells(),
            b.cells(),
            "Hashed reveal must be a pure function of its inputs"
        );
    }

    #[test]
    fn hashed_reveal_favors_top_rows_early() {
        // The row-bias term means, on average, the top of the frame should be
        // AT LEAST as revealed as the bottom at a low progress — not an
        // exact per-cell guarantee (the hash term dominates), but true in
        // aggregate, which is what "resolves top-down" means here.
        let cols = 60;
        let rows = 40;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut out, 0.25, RevealMode::Hashed);

        let half = rows / 2;
        let count_new = |r0: usize, r1: usize| -> usize {
            (r0..r1)
                .flat_map(|row| (0..cols).map(move |col| (col, row)))
                .filter(|&(col, row)| out.get(col, row).unwrap().glyph == u32::from('N'))
                .count()
        };
        let top = count_new(0, half);
        let bottom = count_new(half, rows);
        assert!(
            top >= bottom,
            "Hashed@0.25: top half ({top} revealed) should lead the bottom half ({bottom}) on average"
        );
    }

    // ── apply_scramble_frontier ─────────────────────────────────────────────

    #[test]
    fn scramble_frontier_only_touches_the_band_ahead_of_the_reveal() {
        let cols = 60;
        let rows = 30;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut out, 0.4, RevealMode::Hashed);
        let before = out.cells().to_vec();

        let glyphs = ['/', '\\', '|', '.'];
        apply_scramble_frontier(
            &mut out,
            RevealMode::Hashed,
            0.4,
            0.08,
            &glyphs,
            0x0040_FF80,
            0.8,
            7,
        );

        let mut touched = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                let b = before[row * cols + col];
                let a = *out.get(col, row).unwrap();
                if a != b {
                    touched += 1;
                    // A touched cell must have started as raw OLD (revealed NEW
                    // cells are never touched by the scramble pass).
                    assert_eq!(
                        b,
                        *old.get(col, row).unwrap(),
                        "scramble only touches still-OLD cells"
                    );
                    // And its glyph must be one of the scramble glyphs.
                    let g = char::from_u32(a.glyph).unwrap();
                    assert!(
                        glyphs.contains(&g),
                        "scrambled cell must use a band glyph, got {g:?}"
                    );
                }
            }
        }
        assert!(
            touched > 0,
            "a non-degenerate band must scramble at least one cell"
        );
    }

    #[test]
    fn scramble_frontier_zero_band_is_a_noop() {
        let cols = 40;
        let rows = 20;
        let old = old_frame(cols, rows);
        let new = new_frame(cols, rows);
        let mut out = Framebuffer::new(cols, rows);
        composite_reveal(&old, &new, &mut out, 0.5, RevealMode::Hashed);
        let before = out.cells().to_vec();

        apply_scramble_frontier(
            &mut out,
            RevealMode::Hashed,
            0.5,
            0.0,
            &['#'],
            0x00FF_0000,
            1.0,
            1,
        );
        assert_eq!(out.cells(), before.as_slice(), "band<=0 must be a no-op");

        apply_scramble_frontier(
            &mut out,
            RevealMode::Hashed,
            0.5,
            0.1,
            &[],
            0x00FF_0000,
            1.0,
            1,
        );
        assert_eq!(
            out.cells(),
            before.as_slice(),
            "empty glyph set must be a no-op"
        );
    }
}
