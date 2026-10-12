//! `re_render::facade` — the structured Facade / window-bay texture recipe.
//!
//! Pure `(u_world, v_world, seed) -> FacadeTexel`: a bay × floor grid of window
//! units carved out of the wall's world-anchored plane coordinate, with per-unit
//! lit/dark variation (seeded, not per-frame noise — the pattern only moves if the
//! author changes the knobs), mullion/sill framing between units, an optional
//! ground-floor band and a thin sign row along the wall's base. A small per-style
//! glyph table ([`FacadeStyle`](crate::facade::FacadeStyle)) makes the SAME
//! lattice serve interior set-dressing too — a server rack's LED bank, a wall
//! of lockers — not only exterior facades.
//!
//! Reused unchanged by `sky::skyline`: a distant tower samples this exact
//! function for its own window rhythm, so the two features share one look.
//!
//! ## LOD
//!
//! At range, a screen cell covers many bay/floor units — sampling the fine grid
//! there would alias into noise as the camera moves (a different sub-unit lands in
//! each cell each frame). [`sample`](crate::facade::sample) takes the caller's
//! `world_per_cell` footprint
//! (see [`crate::ascii_shader::ShadeLod::world_per_cell`]) and, once it exceeds
//! about half a bay/floor period, COARSENS the effective grid to roughly one
//! sample per footprint and drops the mullion frame — the surface collapses to a
//! sparse, stable "lit-dot rhythm" instead of shimmering detail. Because the
//! coarsened lattice is still keyed off the WORLD coordinate (just a wider
//! quantization step), it stays anchored under camera motion rather than
//! flickering between neighbours frame to frame.

use crate::ascii_shader::{hash01, hash2i, hash3i};

/// Which window-unit GLYPH TABLE a facade samples from. Render-side mirror of
/// [`re_world::FacadeStyle`] — see [`crate::facade_style_from_world`] for the 1:1
/// map (the established pattern for every plain-data recipe enum, e.g.
/// `substrate::VoiceMode` / `re_world::VoiceMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FacadeStyle {
    /// Office windows: even bay/floor rhythm, moderate lit fraction.
    #[default]
    Office,
    /// Residential windows: same lattice, distinct glyph set for variety.
    Residential,
    /// Industrial bays: wider units, sparser lighting.
    Industrial,
    /// Interior set-dressing: a server rack's LED bank, sampled off the same lattice.
    ServerRack,
    /// Interior set-dressing: a wall of lockers, sampled off the same lattice.
    Lockers,
}

/// The authored knobs a [`sample`] call reads. Mirrors
/// [`re_world::MaterialRecipe::Facade`] 1:1 (minus `tex_id`-style plumbing — this
/// recipe needs none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FacadeParams {
    /// Which glyph table / window-unit rhythm to sample.
    pub style: FacadeStyle,
    /// Window-bay columns per world-cell width (along the wall).
    pub bay: f32,
    /// Window-floor rows per world-cell height (up the wall).
    pub floor_h: f32,
    /// Fraction of units that hash LIT `[0,1]`.
    pub lit: f32,
    /// Per-building hash salt — two facades with the same knobs but different
    /// seeds show different lit windows.
    pub seed: i32,
    /// Draw a distinct band across the wall's bottom world-cell height.
    pub ground_band: bool,
    /// Draw a thin accent band just above the ground band.
    pub sign_row: bool,
}

/// One shaded facade unit: a glyph plus an ink multiplier in `[0,1]` (the caller
/// folds this into its own palette taps, exactly like
/// [`crate::substrate::tone`]'s caller does). `frame` marks a mullion/sill texel so
/// a caller that wants to tint frames differently from glass can (today's sampler
/// in `ascii_shader.rs` does not; the flag is carried for that future use and for
/// tests).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FacadeTexel {
    /// The glyph to draw for this texel.
    pub glyph: char,
    /// Ink multiplier in `[0,1]` the caller folds into its own palette taps.
    pub ink: f32,
    /// Whether this texel is a mullion/sill frame texel rather than glass.
    pub frame: bool,
}

/// Frame (mullion/sill) half-width, as a fraction of one bay/floor unit. Fixed
/// rather than authored — the recipe's grammar tunes RHYTHM (bay/floor/lit), not
/// frame thickness, and one constant keeps every style's grid legible at the same
/// zoom. A narrower value (e.g. `0.12`) puts the frame's WORLD thickness
/// (`MULLION_WIDTH / bay`, e.g. `0.04` world-cells at `bay=3`) below one screen
/// cell across most of the playable viewing range, so the grid lines sample
/// inconsistently (present in some columns, skipped in others) and read as noise
/// rather than a frame. `0.2` keeps a solid majority of each unit open as glass
/// while making the frame band resolve reliably.
const MULLION_WIDTH: f32 = 0.2;

/// World-height (in world cells) of the ground/storefront band, when
/// [`FacadeParams::ground_band`] is set. One world-cell tall — the base storey.
const GROUND_BAND_HEIGHT: f32 = 1.0;

/// World-height of the sign row above the ground band, when
/// [`FacadeParams::sign_row`] is set. Thin — an accent strip, not another floor.
const SIGN_ROW_HEIGHT: f32 = 0.2;

/// Below this many world-cells per bay/floor PERIOD, a screen cell's footprint no
/// longer resolves individual units — see the module docs' LOD section.
const LOD_COLLAPSE_FACTOR: f32 = 0.5;

struct StyleGlyphs {
    mullion_v: char,
    mullion_h: char,
    lit: char,
    lit_alt: char,
    dark: char,
    ground: char,
    sign: char,
    lit_ink: f32,
    dark_ink: f32,
    frame_ink: f32,
}

fn glyphs_for(style: FacadeStyle) -> StyleGlyphs {
    match style {
        // Plain rectangular panes, calm framing — an office tower. `lit`/`dark`
        // are both SOLID-coverage block glyphs (not a stipple) so a pane reads as
        // a filled rectangle at any zoom; only the ink/colour tells lit from dark,
        // matching how the mullion (a thin line glyph) is meant to contrast
        // against a filled pane rather than against more speckle. Every glyph
        // below is checked present in Consolas (the reference terminal font);
        // `▮`/`▯` are NOT and render as tofu, which is why the Lockers style
        // below uses a different pair.
        FacadeStyle::Office => StyleGlyphs {
            mullion_v: '│',
            mullion_h: '─',
            lit: '▓',
            lit_alt: '█',
            dark: '▒',
            ground: '▄',
            sign: '═',
            lit_ink: 1.0,
            dark_ink: 0.32,
            frame_ink: 0.7,
        },
        // Smaller units, warmer variety — apartment windows. Same solid-pane
        // language as Office; the per-style read comes from hue (the material's
        // own fg/bg) and the thinner `frame_ink`, not from a sparser glyph.
        FacadeStyle::Residential => StyleGlyphs {
            mullion_v: '│',
            mullion_h: '─',
            lit: '▓',
            lit_alt: '▒',
            dark: '▒',
            ground: '▒',
            sign: '─',
            lit_ink: 0.95,
            dark_ink: 0.28,
            frame_ink: 0.55,
        },
        // Fewer, heavier openings, grimy ink — a plant/warehouse wall.
        FacadeStyle::Industrial => StyleGlyphs {
            mullion_v: '█',
            mullion_h: '▓',
            lit: '▓',
            lit_alt: '▒',
            dark: '▒',
            ground: '▓',
            sign: '▬',
            lit_ink: 0.85,
            dark_ink: 0.45,
            frame_ink: 0.85,
        },
        // Dense small dots on thin vertical rails — a server rack's LED bank. The
        // ONE style that stays deliberately sparse: a rack's individual LEDs are
        // genuinely point lights, not panes, so the dot read is correct here
        // (unlike a residential/office pane, which must NOT be a stipple).
        FacadeStyle::ServerRack => StyleGlyphs {
            mullion_v: '│',
            mullion_h: '·',
            lit: '•',
            lit_alt: '∙',
            dark: '·',
            ground: '▬',
            sign: '═',
            lit_ink: 1.0,
            dark_ink: 0.18,
            frame_ink: 0.5,
        },
        // Vertical door-like units — a locker-room wall (closed vs. in-use). Solid
        // block panes (a locker door is opaque sheet metal either way); `▮`/`▯`
        // replaced with the same `▓`/`▒` pane language the other solid styles use
        // — Consolas has no glyph for `▮`/`▯` (tofu).
        FacadeStyle::Lockers => StyleGlyphs {
            mullion_v: '│',
            mullion_h: '─',
            lit: '▓',
            lit_alt: '▓',
            dark: '▒',
            ground: '▬',
            sign: '─',
            lit_ink: 0.9,
            dark_ink: 0.4,
            frame_ink: 0.75,
        },
    }
}

/// Sample the facade at world-anchored `(u_world, v_world)` — the wall's ALONG and
/// UP coordinates (world cells), matching [`crate::ascii_shader::SampleCtx::wall_along`]
/// / [`crate::ascii_shader::SampleCtx::wall_up`]. `world_per_cell` is the caller's
/// current LOD footprint (`0.0` ⇒ near, never collapse — see
/// [`crate::ascii_shader::ShadeLod::world_per_cell`]).
///
/// Pure: no `time`, no interior mutability — same inputs, same texel, every call.
#[must_use]
pub fn sample(u_world: f32, v_world: f32, p: &FacadeParams, world_per_cell: f32) -> FacadeTexel {
    let bay = p.bay.max(0.25);
    let floor_h = p.floor_h.max(0.25);
    let g = glyphs_for(p.style);

    // The ground band / sign row read against the WALL'S ABSOLUTE base, not the
    // tiled bay/floor grid — a strip once along the whole facade, not once per
    // world cell.
    if p.ground_band && v_world < GROUND_BAND_HEIGHT {
        return ground_band_texel(u_world, p, &g);
    }
    if p.sign_row && (GROUND_BAND_HEIGHT..GROUND_BAND_HEIGHT + SIGN_ROW_HEIGHT).contains(&v_world) {
        return FacadeTexel {
            glyph: g.sign,
            ink: g.frame_ink * 1.3,
            frame: true,
        };
    }

    // LOD collapse: once the screen footprint covers more than half a bay/floor
    // period, stop resolving individual units — quantize the grid down toward
    // (roughly) one sample per footprint. `k >= 1` always, so this only ever
    // COARSENS (never sharpens) the near-tier grid.
    let bay_period = 1.0 / bay;
    let floor_period = 1.0 / floor_h;
    let finest = bay_period.min(floor_period);
    let collapse = world_per_cell > finest * LOD_COLLAPSE_FACTOR;
    let k = if collapse {
        (world_per_cell / finest.max(1e-4)).max(1.0)
    } else {
        1.0
    };
    let eff_bay = bay / k;
    let eff_floor = floor_h / k;

    let bx = u_world * eff_bay;
    let ix = bx.floor() as i32;
    let fu = bx - bx.floor();
    let fy = v_world * eff_floor;
    let iy = fy.floor() as i32;
    let fv = fy - fy.floor();

    // Mullion/sill frame band. At the near tier both the vertical (bay) and
    // horizontal (floor) seams draw. Once collapsed, the vertical seam is
    // dropped (a coarsened bay column aliases as it's crossed, shimmering under
    // camera motion) but the HORIZONTAL floor line is kept — floor banding is
    // the strongest distance cue a tower silhouette has (see the visual-review
    // reference shots' distinct storey stripes), and because only `v_world`
    // determines it, a floor seam stays anchored to one world height and does
    // not shimmer the way a coarsened bay column would. The result at range is
    // a lit-dot rhythm cut by stable horizontal floor lines, not a flat field.
    let du = fu.min(1.0 - fu);
    let dv = fv.min(1.0 - fv);
    if (!collapse && du < MULLION_WIDTH) || dv < MULLION_WIDTH {
        let glyph = if !collapse && du < dv {
            g.mullion_v
        } else {
            g.mullion_h
        };
        return FacadeTexel {
            glyph,
            ink: g.frame_ink,
            frame: true,
        };
    }

    let h = hash3i(p.seed, ix, iy);
    let is_lit = hash01(h) < p.lit.clamp(0.0, 1.0);
    let variety = hash01(h.rotate_left(11) ^ 0x9E37_79B9);
    let glyph = if is_lit {
        if variety < 0.5 {
            g.lit
        } else {
            g.lit_alt
        }
    } else {
        g.dark
    };
    let ink = if is_lit { g.lit_ink } else { g.dark_ink };
    FacadeTexel {
        glyph,
        ink,
        frame: false,
    }
}

fn ground_band_texel(u_world: f32, p: &FacadeParams, g: &StyleGlyphs) -> FacadeTexel {
    // A denser, brighter rhythm than the upper floors — the storefront read —
    // hashed per along-position so the base is not a flat fill.
    let ix = (u_world * (p.bay.max(0.25) * 2.0)).floor() as i32;
    let h = hash2i(p.seed, ix);
    let lit = hash01(h) < (p.lit.clamp(0.0, 1.0) * 1.4).min(0.9);
    FacadeTexel {
        glyph: g.ground,
        ink: if lit { g.lit_ink } else { g.dark_ink * 1.15 },
        frame: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(style: FacadeStyle) -> FacadeParams {
        FacadeParams {
            style,
            bay: 3.0,
            floor_h: 4.0,
            lit: 0.35,
            seed: 7,
            ground_band: false,
            sign_row: false,
        }
    }

    #[test]
    fn pure_function_same_inputs_same_texel() {
        let p = params(FacadeStyle::Office);
        let a = sample(1.37, 2.61, &p, 0.0);
        let b = sample(1.37, 2.61, &p, 0.0);
        assert_eq!(a, b);
    }

    #[test]
    fn every_style_reads_as_reliable_glyphs() {
        for style in [
            FacadeStyle::Office,
            FacadeStyle::Residential,
            FacadeStyle::Industrial,
            FacadeStyle::ServerRack,
            FacadeStyle::Lockers,
        ] {
            let g = glyphs_for(style);
            for c in [
                g.mullion_v,
                g.mullion_h,
                g.lit,
                g.lit_alt,
                g.dark,
                g.ground,
                g.sign,
            ] {
                assert!(
                    crate::glyph_family::is_reliable(c),
                    "{style:?}: {c:?} must be a reliable glyph"
                );
            }
        }
    }

    #[test]
    fn a_sweep_produces_both_lit_and_dark_units() {
        let p = params(FacadeStyle::Office);
        let mut any_lit = false;
        let mut any_dark = false;
        for iy in 0..40 {
            for ix in 0..40 {
                let u = (ix as f32 + 0.5) / p.bay;
                let v = (iy as f32 + 0.5) / p.floor_h;
                let t = sample(u, v, &p, 0.0);
                if t.frame {
                    continue;
                }
                if t.ink >= glyphs_for(p.style).lit_ink - 1e-6 {
                    any_lit = true;
                } else {
                    any_dark = true;
                }
            }
        }
        assert!(
            any_lit && any_dark,
            "a 40x40 sweep must hit both lit and dark units"
        );
    }

    #[test]
    fn a_frame_band_appears_near_the_top_tier() {
        let p = params(FacadeStyle::Office);
        // The unit edge closest to u=0 is exactly on a mullion seam.
        let t = sample(0.001, 0.5 / p.floor_h, &p, 0.0);
        assert!(t.frame, "a sample right at a bay seam must be framed");
    }

    #[test]
    fn collapsing_lod_drops_the_vertical_seam_but_keeps_the_floor_line() {
        let p = params(FacadeStyle::Office);
        // A huge footprint forces the collapse branch.
        let far_world_per_cell = 50.0;
        let a = sample(4.2, 6.7, &p, far_world_per_cell);
        let b = sample(4.2, 6.7, &p, far_world_per_cell);
        assert_eq!(a, b);

        // A collapsed tower must still show BOTH a plain lit/dark unit (between
        // floor lines) and at least one horizontal floor-line hit — visual review
        // 2026-09-27 pass 2 asked for "far LOD collapses to a lit-dot rhythm that
        // still shows floor lines", not a flat dotted field. The vertical (bay)
        // seam is the one that drops at collapse (see `sample`'s doc comment); a
        // frame hit at this tier must always be the horizontal glyph.
        let mut saw_unframed = false;
        let mut saw_floor_line = false;
        for i in 0..400 {
            let v = i as f32 * 0.05;
            let t = sample(4.2, v, &p, far_world_per_cell);
            if t.frame {
                saw_floor_line = true;
                assert_eq!(
                    t.glyph,
                    glyphs_for(p.style).mullion_h,
                    "a collapsed-tier frame hit must be the horizontal floor glyph, \
                     never the dropped vertical seam"
                );
            } else {
                saw_unframed = true;
            }
        }
        assert!(
            saw_unframed,
            "a collapsed sweep must still show plain units between floor lines"
        );
        assert!(
            saw_floor_line,
            "a collapsed sweep must still show at least one floor line"
        );
    }

    #[test]
    fn ground_band_overrides_the_grid_at_the_wall_base() {
        let mut p = params(FacadeStyle::Office);
        p.ground_band = true;
        let g = glyphs_for(p.style);
        let t = sample(1.5, 0.4, &p, 0.0);
        assert_eq!(t.glyph, g.ground);
    }

    #[test]
    fn sign_row_sits_just_above_the_ground_band() {
        let mut p = params(FacadeStyle::Office);
        p.ground_band = true;
        p.sign_row = true;
        let g = glyphs_for(p.style);
        let t = sample(1.5, GROUND_BAND_HEIGHT + 0.05, &p, 0.0);
        assert_eq!(t.glyph, g.sign);
    }

    #[test]
    fn different_seeds_move_the_lit_pattern() {
        let p_a = params(FacadeStyle::Office);
        let mut p_b = p_a;
        p_b.seed = 999;
        let mut differs = false;
        for iy in 0..20 {
            for ix in 0..20 {
                let u = (ix as f32 + 0.5) / p_a.bay;
                let v = (iy as f32 + 0.5) / p_a.floor_h;
                if sample(u, v, &p_a, 0.0) != sample(u, v, &p_b, 0.0) {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "two seeds over a 20x20 sweep must disagree somewhere"
        );
    }
}
