//! `re_render::color_drift` — the per-cell Oklch offset field. Effects do not
//! paint colour directly: they deposit OFFSETS into a per-frame field, and one
//! pass applies the field to F, B and U in Oklch, where a hue shift keeps
//! perceived lightness.
//!
//! # Split with `re_game`
//!
//! This module is PURE ENGINE MATH, like [`crate::palette_gate`]: it knows
//! nothing about pipes, puzzle state, emitters, glitch or the audio clock. It
//! defines the data shape a drift SOURCE has ([`DriftSource`], [`GazeSource`])
//! and the one function that turns a snapshot of sources into cell offsets
//! ([`apply_drift`]). Turning game concepts (a powered pipe, `GlitchPhosphorState`,
//! the camera pose, an emitter's glow accumulator) into a fresh
//! [`DriftSources`] EVERY FRAME is `re_game`'s job — the spec's §3.2 source
//! table lists seven of them, none of which this crate depends on.
//!
//! # The model (§3.1)
//!
//! ```text
//! D(cell) = Σ_i  w_i(cell) · env_i(t) · (ΔL_i, ΔC_i, Δh_i)
//! w_i     = falloff(|cell.hit − src_i| world)   |   screen-space for gaze
//! apply:  (L,C,h) ← (L + clampL(ΔL), min(C + ΔC, gamut(L,h)), h + clamp_h(Δh))
//! ```
//!
//! `env_i(t)` (the source's time envelope — dwell timers, decay half-lives,
//! toggle eases) is baked into [`DriftSource::delta_l`]/`delta_c`/`delta_h_deg`
//! BEFORE it reaches this module: the caller evaluates its own envelope once
//! per frame and hands over the already-scaled offset. That keeps this module
//! stateless and rebuilding the field fresh every frame is then "for free" —
//! R4 ("everything decays… the field is rebuilt every frame from sources, so
//! nothing accumulates") holds by construction, not by a decay timer living
//! here.
//!
//! Runs AFTER the composite resolve, BEFORE `re_postfx` (spec §3.1) — pipeline
//! ordering is `re_game`'s responsibility; this module only does the per-cell
//! accumulate + apply.
//!
//! # Readability rules enforced HERE vs. elsewhere (spec §3.3)
//!
//! | Rule | Where |
//! |---|---|
//! | R1 (semantic surfaces: `\|ΔL\|≤0.04`, `Δh=0`) | HERE — [`apply_drift`]'s `is_semantic` predicate |
//! | R2 (amplitude caps) | HERE — [`MAX_SOURCE_DELTA_H_AMBIENT_DEG`]/[`MAX_SOURCE_DELTA_H_NEAR_DEG`] clamp every source on read |
//! | R3 (tempo: drift <0.2 Hz, shimmer ≤3 Hz, flashes ≤3/s) | the CALLER's envelope shape — this module has no notion of time, so it cannot enforce a rate |
//! | R4 (everything decays, rebuilt every frame) | free, by this module's statelessness (see above) |
//! | R5 (contrast floor `ΔE_ok(F,B) ≥ 0.10` when κ≥0.1) | HERE — a best-effort safety net (§ below); the real κ table lives in `ascii_shader`/`glyph_family`, not `Cell`, so this uses `glyph != ' '` as a coarse non-blank proxy |
//! | R6 (deadband-aware amplitude) | the CALLER's source design — this module does not know the present layer's byte budget |
//! | R7 (Ansi256 hold-rule) | `re_present` — outside this crate entirely |

use re_core::oklab::{oklab_to_oklch, oklch_to_packed, wrap_hue_delta};
use re_core::{Cell, Framebuffer, Vec2};

/// A WORLD-anchored drift source: an authored Oklch offset that falls off
/// radially from `world_pos` (matched against `Cell::hit`, the exact
/// world-space point each cell looks at — floors AND walls both carry it).
///
/// The envelope (dwell/decay/ease) is already folded into `delta_l`/`delta_c`/
/// `delta_h_deg` by the caller — see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftSource {
    /// World-space `(x, y)` centre, matched against `Cell::hit`.
    pub world_pos: Vec2,
    /// Falloff radius in world units. `w = 0` at `dist >= radius`.
    pub radius: f32,
    /// Lightness offset, ALREADY envelope-scaled.
    pub delta_l: f32,
    /// Chroma offset, ALREADY envelope-scaled.
    pub delta_c: f32,
    /// Hue offset in DEGREES, ALREADY envelope-scaled. Clamped on read to
    /// [`MAX_SOURCE_DELTA_H_NEAR_DEG`] (R2: "Δh ≤ 60° only inside a source's
    /// radius").
    pub delta_h_deg: f32,
}

/// The SCREEN-anchored gaze source (spec §3.2: "cells within 12% of screen
/// centre, after 1.5s of dwell" — the dwell timer and the 12% radius are
/// authored by the caller; this struct just carries the already-computed
/// screen centre + radius + envelope-scaled offset).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GazeSource {
    /// Screen-space centre in NORMALISED `[0,1]×[0,1]` cell coordinates (so it
    /// is resolution-independent — the caller converts its `(col,row)` gaze
    /// point once per frame).
    pub screen_center: (f32, f32),
    /// Falloff radius, in the SAME normalised units (e.g. `0.12` for "12% of
    /// screen").
    pub radius: f32,
    /// Lightness offset, ALREADY envelope-scaled.
    pub delta_l: f32,
    /// Chroma offset, ALREADY envelope-scaled.
    pub delta_c: f32,
    /// Hue offset in DEGREES, ALREADY envelope-scaled. Clamped on read the
    /// same way as [`DriftSource::delta_h_deg`].
    pub delta_h_deg: f32,
}

/// One frame's worth of drift sources — rebuilt fresh every frame by the
/// caller (R4). `world` sources are matched against each cell's world hit;
/// `gaze` (at most one, since there is only one camera) is matched against
/// screen position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DriftSources {
    /// World-anchored sources, matched against each cell's `Cell::hit`.
    pub world: Vec<DriftSource>,
    /// The single screen-anchored gaze source, if the caller's dwell timer
    /// has armed one this frame.
    pub gaze: Option<GazeSource>,
}

/// R2 — "Ambient Δh ≤ 25°": the cap applied when a cell is only partially
/// inside a source's falloff (`w < 1`), scaled by `w`. See
/// [`clamped_source_delta_h`].
pub const MAX_SOURCE_DELTA_H_AMBIENT_DEG: f32 = 25.0;

/// R2 — "Δh ≤ 60° only inside a source's radius" (`w` at or near `1`): the
/// hard ceiling no single source's hue offset may exceed regardless of `w`.
pub const MAX_SOURCE_DELTA_H_NEAR_DEG: f32 = 60.0;

/// R1 — the lightness cap on a "semantic" surface (puzzle-legible material):
/// `|ΔL| ≤ 0.04`.
pub const SEMANTIC_MAX_DELTA_L: f32 = 0.04;

/// R1's `Δh=0` is enforced by zeroing `dh_deg` outright on semantic cells (see
/// `apply_drift_with`), but that alone is not sufficient: a large, UNcapped
/// chroma offset can walk the Oklab `(L,a,b)` point outside the sRGB gamut,
/// and the clamp-to-`[0,1]` that `pack_linear` applies on the round-trip
/// distorts the *apparent* hue by several degrees even though the Oklch hue
/// value itself never moved. R1's actual intent — a semantic surface's hue
/// never visibly shifts — requires capping chroma too, so this small cap
/// keeps every semantic-cell offset inside the gamut where hue is exact.
pub const SEMANTIC_MAX_DELTA_C: f32 = 0.02;

/// R5 — the Oklab ΔE(F,B) floor a non-blank cell must keep after drift.
pub const CONTRAST_FLOOR_DELTA_E: f32 = 0.10;

/// Smooth radial falloff: `1` at the centre, `0` at/after `radius`, a smooth
/// (not linear) ease between — so a source's edge does not read as a ring.
#[inline]
fn radial_falloff(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    let t = (dist / radius).clamp(0.0, 1.0);
    let s = 1.0 - t;
    s * s * (3.0 - 2.0 * s) // smoothstep(0,1,s) — eases both ends
}

/// R2, applied to one source's authored hue delta at weight `w`: the ambient
/// cap always applies; the wider near-source cap only applies once `w` is
/// (near) `1`. Interpolating the cap itself by `w` means a cell half inside a
/// source's radius cannot borrow the FULL near-source allowance.
#[inline]
fn clamped_source_delta_h(delta_h_deg: f32, w: f32) -> f32 {
    let cap = MAX_SOURCE_DELTA_H_AMBIENT_DEG
        + (MAX_SOURCE_DELTA_H_NEAR_DEG - MAX_SOURCE_DELTA_H_AMBIENT_DEG) * w.clamp(0.0, 1.0);
    delta_h_deg.clamp(-cap, cap)
}

/// The accumulated `(ΔL, ΔC, Δh_deg)` offset at world point `hit`, folding in
/// every world source. Public so a caller (e.g. a debug overlay, or a future
/// screen-space source) can inspect the field without re-deriving it.
#[must_use]
pub fn accumulate_world(sources: &[DriftSource], hit: Vec2) -> (f32, f32, f32) {
    let mut dl = 0.0_f32;
    let mut dc = 0.0_f32;
    let mut dh = 0.0_f32;
    for src in sources {
        let dist = (hit - src.world_pos).length();
        let w = radial_falloff(dist, src.radius);
        if w <= 0.0 {
            continue;
        }
        dl += src.delta_l * w;
        dc += src.delta_c * w;
        dh += clamped_source_delta_h(src.delta_h_deg, w) * w;
    }
    (dl, dc, dh)
}

/// The gaze contribution at a cell's SCREEN position (normalised `[0,1]²`).
#[must_use]
fn accumulate_gaze(gaze: Option<&GazeSource>, screen_pos: (f32, f32)) -> (f32, f32, f32) {
    let Some(g) = gaze else {
        return (0.0, 0.0, 0.0);
    };
    let dx = screen_pos.0 - g.screen_center.0;
    let dy = screen_pos.1 - g.screen_center.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let w = radial_falloff(dist, g.radius);
    if w <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    (
        g.delta_l * w,
        g.delta_c * w,
        clamped_source_delta_h(g.delta_h_deg, w) * w,
    )
}

/// `true` for cells this module treats as "semantic" (R1: puzzle-legible
/// surfaces — nodes, pipes, switches, doors, notes). `Cell` has no dedicated
/// semantic bit yet (that would be a `[material] semantic = true` authoring
/// key, not landed), so this is a best-effort proxy using data that already
/// exists: `Cell::material` is documented as "surface-specific extra tag
/// (node/door id); 0 when unused" — a non-zero value already IS a semantic
/// tag today. A future authoring key can widen this without changing
/// [`apply_drift`]'s signature (callers may also supply this predicate
/// pre-composed with their own check via [`apply_drift_with`]).
#[must_use]
pub fn is_semantic_default(cell: &Cell) -> bool {
    cell.material != 0
}

/// Apply `sources` to every cell of `fb`, using [`is_semantic_default`] for
/// R1. Convenience wrapper over [`apply_drift_with`] for the common case.
pub fn apply_drift(fb: &mut Framebuffer, sources: &DriftSources) {
    apply_drift_with(fb, sources, is_semantic_default);
}

/// Apply `sources` to every cell of `fb`, fg + bg + underline colour, with a
/// caller-supplied `is_semantic` predicate (R1). A cell with no colour change
/// this frame (`w == 0` everywhere reachable) is left byte-identical — an
/// empty [`DriftSources`] is therefore a true no-op, so a level authoring no
/// drift sources renders exactly as before.
pub fn apply_drift_with(
    fb: &mut Framebuffer,
    sources: &DriftSources,
    is_semantic: impl Fn(&Cell) -> bool,
) {
    if sources.world.is_empty() && sources.gaze.is_none() {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            let (wl, wc, wh) = accumulate_world(&sources.world, s.hit);
            let screen_pos = if cols > 1 && rows > 1 {
                (
                    col as f32 / (cols - 1) as f32,
                    row as f32 / (rows - 1) as f32,
                )
            } else {
                (0.5, 0.5)
            };
            let (gl, gc, gh) = accumulate_gaze(sources.gaze.as_ref(), screen_pos);
            let (mut dl, mut dc, mut dh_deg) = (wl + gl, wc + gc, wh + gh);
            if dl == 0.0 && dc == 0.0 && dh_deg == 0.0 {
                continue;
            }

            let semantic = is_semantic(s);
            if semantic {
                // R1: |ΔL| ≤ 0.04, Δh = 0. Chroma also gets a small cap here
                // (see `SEMANTIC_MAX_DELTA_C`) — not because R1's text names
                // it, but because an uncapped chroma offset can gamut-clip on
                // the round-trip and leak an apparent hue shift, defeating
                // Δh=0 in the colour a player actually sees. L2's colour
                // puzzle still has ample chroma range within this cap.
                dl = dl.clamp(-SEMANTIC_MAX_DELTA_L, SEMANTIC_MAX_DELTA_L);
                dc = dc.clamp(-SEMANTIC_MAX_DELTA_C, SEMANTIC_MAX_DELTA_C);
                dh_deg = 0.0;
            }

            let mut c: Cell = *s;
            c.fg = drift_packed(c.fg, dl, dc, dh_deg);
            c.bg = drift_packed(c.bg, dl, dc, dh_deg);
            if c.style.underline != re_core::NO_UNDERLINE {
                c.style.underline = drift_packed(c.style.underline, dl, dc, dh_deg);
            }

            if !semantic {
                enforce_contrast_floor(&mut c, s);
            }

            fb.set(col, row, c);
        }
    }
}

/// Apply one `(ΔL, ΔC, Δh_deg)` offset to a packed colour in Oklch space.
#[inline]
fn drift_packed(packed: u32, dl: f32, dc: f32, dh_deg: f32) -> u32 {
    let lch = oklab_to_oklch(re_core::oklab::linear_to_oklab(
        re_core::oklab::unpack_linear(packed),
    ));
    let dh = wrap_hue_delta(dh_deg.to_radians());
    let out = re_core::oklab::Oklch {
        l: (lch.l + dl).clamp(0.0, 1.0),
        c: (lch.c + dc).max(0.0),
        h: lch.h + dh,
    };
    oklch_to_packed(out)
}

/// R5 safety net: if `c`'s fg/bg ΔE has fallen under [`CONTRAST_FLOOR_DELTA_E`]
/// on a non-blank cell (`glyph != ' '`, the coarse κ>0 proxy — see the module
/// docs), nudge fg AWAY from bg in lightness by just enough to restore the
/// floor. Lightness (not hue/chroma) is the nudge axis because it is the
/// cheapest, least perceptually-disruptive way to restore separation, and
/// because R1 already protects semantic surfaces (this runs only when
/// `!semantic`, i.e. surfaces where nudging is safe to begin with).
fn enforce_contrast_floor(c: &mut Cell, original: &Cell) {
    if c.glyph == u32::from(' ') {
        return; // κ ≈ 0 — nothing to protect, per R5's own "κ≥0.1" gate.
    }
    let fg_lab = re_core::oklab::linear_to_oklab(re_core::oklab::unpack_linear(c.fg));
    let bg_lab = re_core::oklab::linear_to_oklab(re_core::oklab::unpack_linear(c.bg));
    let de = re_core::oklab::delta_e(fg_lab, bg_lab);
    if de >= CONTRAST_FLOOR_DELTA_E {
        return;
    }
    // Preserve the ORIGINAL (pre-drift) fg/bg lightness ORDER (which one was
    // brighter before drift), and push apart along L until the floor is met
    // or we hit the gamut edge — a few iterations of a stable, monotone step
    // converge quickly and never overshoot into a visible pop.
    let orig_fg_l = re_core::oklab::linear_to_oklab(re_core::oklab::unpack_linear(original.fg)).l;
    let orig_bg_l = re_core::oklab::linear_to_oklab(re_core::oklab::unpack_linear(original.bg)).l;
    let fg_is_lighter = orig_fg_l >= orig_bg_l;

    let mut fg = fg_lab;
    let mut bg = bg_lab;
    for _ in 0..8 {
        let cur = re_core::oklab::delta_e(fg, bg);
        if cur >= CONTRAST_FLOOR_DELTA_E {
            break;
        }
        let step = 0.02;
        if fg_is_lighter {
            fg.l = (fg.l + step).min(1.0);
            bg.l = (bg.l - step).max(0.0);
        } else {
            fg.l = (fg.l - step).max(0.0);
            bg.l = (bg.l + step).min(1.0);
        }
    }
    c.fg = re_core::oklab::pack_linear(re_core::oklab::oklab_to_linear(fg));
    c.bg = re_core::oklab::pack_linear(re_core::oklab::oklab_to_linear(bg));
}

#[cfg(test)]
mod tests;
