//! `re_render::mirror` — the reflective-surface (MIRROR) sub-system.
//!
//! A wall whose resolved [`crate::Material`] is `reflective` casts a SECOND DDA
//! bounce: the primary ray is reflected off the mirror's face and the reflected
//! scene is shaded into the mirror's screen slice, composited at the mirror's
//! CLARITY (authored, or the live recognition-beat override). A clear mirror shows
//! a crisp cool-tinted reflection; a "broken" one degrades toward sparse glitch
//! glyphs on near-black.
//!
//! This module is a pure mechanical extraction from `lib.rs`. The raycast core
//! still builds the reflected [`crate::RayHit`], wraps it in a [`MirrorView`], and
//! calls [`mirror_composite`] per mirror cell; the geometry primitives (`dda_hit`)
//! and the `MirrorView` construction stay in the cast where they belong.
//!
//! Pure compute, no terminal, no allocation, deterministic (a fixed hash on the
//! screen cell drives the glitch — no clock, no `rand`).

use crate::ascii_shader::MaterialRegistry;
use crate::{surface_material, MirrorClarity, RayHit};
use re_core::{Cell, SurfaceKind};

/// How far (world units, along the reflected ray) the reflected DDA origin is
/// nudged off the mirror's hit face, so the walk does not immediately re-hit the
/// mirror's own cell (the analogue of `SHADOW_BIAS` for the reflection ray).
pub(crate) const MIRROR_ORIGIN_NUDGE: f32 = 1e-3;

/// Mirror sheen tint (cool, slightly blue) the reflected scene is multiplied by,
/// so a reflection reads as seen-through-glass rather than the raw geometry.
const MIRROR_SHEEN: glam::Vec3 = glam::Vec3::new(0.72, 0.80, 0.92);

/// The reflected-bounce result + the clarity it composites at. Built once per
/// column when the primary ray hits a mirror; consumed by the wall-row loop.
#[derive(Clone, Copy)]
pub(crate) struct MirrorView {
    /// The reflected ray's grid hit (its own surface/material/relief is shaded).
    pub(crate) refl: RayHit,
    /// Reflected ray direction (un-normalised; used for its view/relief vectors).
    pub(crate) refl_dir_x: f32,
    pub(crate) refl_dir_y: f32,
    /// Effective clarity `[0,1]` for this mirror (authored, or the live override).
    pub(crate) clarity: f32,
}

/// True if a wall surface (its `side_data` material tag → registry material) is a
/// MIRROR. The reflective flag lives on the resolved [`crate::Material`]; a non-wall
/// surface (floor/ceiling/object) is never reflected (objects are puzzle markers).
#[inline]
pub(crate) fn surface_is_mirror(
    registry: &MaterialRegistry,
    surface: SurfaceKind,
    tag: u16,
) -> bool {
    surface.is_wall() && registry.get(surface_material(surface, tag)).reflective
}

/// The effective mirror clarity for the hit surface, given the live override.
#[inline]
pub(crate) fn surface_clarity(
    registry: &MaterialRegistry,
    surface: SurfaceKind,
    tag: u16,
    mirror: MirrorClarity,
) -> f32 {
    mirror.effective(registry.get(surface_material(surface, tag)).clarity)
}

/// Composite the reflected scene cell `cell` (already shaded from the reflected ray)
/// into a mirror cell at screen `(col, row)`, at the mirror's `clarity`.
///
/// * sheen-tint the reflected colours cool/dark (always — a clear mirror is glass);
/// * `refl_hit == false` (reflection is the void) → a dim cool wash, no detail;
/// * `clarity >= 1` → crisp (sheen only);
/// * `clarity < 1` → degrade toward sparse glitch glyphs on near-empty/black, a
///   deterministic per-cell hash deciding which cells survive and as what — so a
///   "collapsed" mirror reads as static rather than a clean image. The blend is
///   smooth, so the game animating `clarity 0→1` plays as the reflection resolving.
///
/// Pure + deterministic (a fixed hash on `(col,row)` — no clock, no `rand`, no
/// allocation), so it is golden-reproducible. `(col,row)` seed the glitch so the
/// noise is world-stable per screen cell.
pub(crate) fn mirror_composite(
    mut cell: Cell,
    refl_hit: bool,
    clarity: f32,
    col: usize,
    row: i32,
) -> Cell {
    use crate::ascii_shader::{dim_rgb, hash2i, mix_rgb, pack_rgb, unpack_rgb};
    let clarity = clarity.clamp(0.0, 1.0);

    // 1. Mirror sheen: tint the reflected colours cool + slightly darker, so even a
    //    perfect reflection reads as glass, not the raw geometry. Applied at full
    //    strength regardless of clarity (a clear mirror is still a mirror).
    let sheen = |packed: u32| -> u32 { pack_rgb(unpack_rgb(packed) * MIRROR_SHEEN) };
    cell.fg = sheen(cell.fg);
    cell.bg = sheen(cell.bg);
    if !refl_hit {
        // The reflection is the void: a dim cool wash, no reflected detail. Still
        // ride the clarity (a broken void mirror is even darker / blank).
        let v = 0.18 + 0.32 * clarity;
        cell.fg = dim_rgb(cell.fg, v);
        cell.bg = dim_rgb(cell.bg, v);
        cell.glyph = u32::from(' ');
        return cell;
    }

    if clarity >= 0.999 {
        return cell; // crisp reflection — sheen only.
    }

    // 2. Degrade toward glitch/empty as clarity → 0. The "broken" target is a dark,
    //    mostly-empty surface with sparse scrambled glyphs. A deterministic hash on
    //    the screen cell decides, per cell, whether a glitch glyph survives and what
    //    it is — so a low-clarity mirror reads as static/scramble, not a clean image.
    let h = hash2i(col as i32, row);
    // Fraction of cells that keep ANY glyph at this clarity: near 0 clarity almost
    // everything blanks (emptiness); the survivors scatter as glitch.
    let keep = (0.12 + 0.88 * clarity).clamp(0.0, 1.0);
    let cell_roll = (h & 0xFFFF) as f32 / 65535.0;

    // Broken colour: a dim, desaturated cold grey-blue the surviving glyphs sit on.
    const GLITCH_FG: u32 = 0x0046506E;
    const GLITCH_BG: u32 = 0x00060810;
    // A small scramble set (box-drawing / noise glyphs) for the surviving cells.
    const GLITCH_GLYPHS: [u32; 6] = [
        0x2591, // ░
        0x2592, // ▒
        0x2580, // ▀
        0x2584, // ▄
        0x23,   // '#'
        0x3F,   // '?'
    ];

    // Blend colour from the broken palette (clarity 0) toward the crisp reflection
    // (clarity 1), so the resolve is a smooth fade.
    cell.fg = mix_rgb(GLITCH_FG, cell.fg, clarity);
    cell.bg = mix_rgb(GLITCH_BG, cell.bg, clarity);

    if cell_roll > keep {
        // This cell collapsed to emptiness (the dominant read at low clarity).
        cell.glyph = u32::from(' ');
        // Keep a faint broken bg so the void still tints; near-black at clarity 0.
        cell.bg = dim_rgb(cell.bg, 0.4 + 0.6 * clarity);
    } else if clarity < GLITCH_SCRAMBLE_BELOW {
        // A surviving cell at low clarity shows a SCRAMBLED glyph (static), not the
        // true reflected glyph — the image is corrupted, not just dim.
        let gi = ((h >> 16) as usize) % GLITCH_GLYPHS.len();
        cell.glyph = GLITCH_GLYPHS[gi];
    }
    // (At higher clarity a surviving cell keeps its real reflected glyph, blended
    //  colour only — so the reflection sharpens smoothly toward crisp.)
    cell
}

/// Below this clarity a surviving mirror cell shows a SCRAMBLED glitch glyph (the
/// reflection is corrupt static); above it the cell keeps its true reflected glyph
/// and only the colour fades — so the image resolves cleanly as clarity rises.
const GLITCH_SCRAMBLE_BELOW: f32 = 0.55;
