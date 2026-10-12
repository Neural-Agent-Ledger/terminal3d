// Test-module assert_eq! on f32 values are intentional exact-equal checks
// (effect parameter boundary values: exactly 0.0 / 1.0 clamp sentinels).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_postfx` — a deterministic, composable POST-PROCESS "shader" stack over the
//! ASCII raycaster's shaded [`Framebuffer`].
//!
//! # What it is
//!
//! A screen-space / cinematic effect layer that runs AFTER the shade pass has
//! produced a clean [`Framebuffer`] and BEFORE the present layer (and, ideally,
//! before `re_glitch`, so corruption sits on top). It reads depth / surface /
//! fg / bg and modifies fg / bg / glyph only — never geometry, never the cell
//! `style`. Seven composable passes turn the flat-shaded scene cinematic:
//! edge-outline, SSAO, bloom, depth-of-field, color-grade, vignette, dither.
//!
//! # The two halves (same architecture as `re_glitch`)
//!
//! 1. **The pure core** — [`apply_postfx`]`(fb, cfg)`. No Bevy, no clock, no
//!    `rand`: a deterministic function of exactly `(fb, cfg)`. Same inputs →
//!    byte-identical output. This is the testable, reusable heart, shared by the
//!    plugin and the binary's `--postfx` dump path.
//! 2. **The Bevy glue** — [`PostfxPlugin`] + the [`PostfxConfig`] resource. The
//!    plugin keeps a *clean snapshot* of the shaded frame and each present writes
//!    `postfx(clean) → Framebuffer`, so effects never COMPOUND across ticks even
//!    when the dirty-flagged renderer skips re-shading (see [`PostfxPlugin`]).
//!
//! # Determinism & purity
//!
//! Every pass is `f32`-only integer-indexed math — no global state, no RNG, no
//! time term (the dither pattern is a pure function of cell coordinates). Depends
//! only on `re_core` (the [`Framebuffer`]/[`Cell`] contract) + `glam`. Additive
//! by default: [`PostfxConfig::off`] (the plugin default) leaves the frame
//! byte-identical, so the shipped levels render exactly as before unless a level
//! or setting opts in.
//!
//! [`Cell`]: re_core::Cell

mod color;
mod config;
mod effects;
mod plugin;

pub use config::{
    BloomFx, ChromaticFx, ColorTempFx, DitherFx, DofFx, EdgeFx, FilmGrainFx, GradeFx, PosterizeFx,
    PostfxConfig, PostfxStage, ScanlineFx, SsaoFx, VignetteFx,
};
pub use plugin::{PostfxDitherPlugin, PostfxPlugin, PostfxSnapshot};

use re_core::Framebuffer;

/// Copy every cell of `src` into `dst` (same dimensions assumed) without
/// reallocating `dst`'s backing storage. The ping-pong primitive: a reused
/// scratch buffer is re-seeded from the working frame before each
/// neighbour-sampling pass instead of cloning a fresh one.
#[inline]
fn copy_into(dst: &mut Framebuffer, src: &Framebuffer) {
    debug_assert_eq!(dst.cols(), src.cols());
    debug_assert_eq!(dst.rows(), src.rows());
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            if let Some(c) = src.get(col, row) {
                dst.set(col, row, *c);
            }
        }
    }
}

/// Apply every enabled effect of `cfg` to `fb`, in a fixed deterministic pipeline
/// order.
///
/// PURE: the result depends on exactly `(fb, cfg)` — call it twice with the same
/// inputs and you get byte-identical frames. No Bevy, no clock, no RNG. This is
/// the function the plugin and the `--postfx` dump path both call.
///
/// The pipeline ORDER is deliberate and fixed (changing it changes the look, so
/// it is part of the contract) and is enumerated ONCE in
/// [`PostfxStage::ORDER`](config::PostfxStage::ORDER) — this driver walks exactly
/// that list. It follows the spec §15 strict stack (atmosphere → light → structure
/// → distortion → colour → stylization → quantization), preserving the FROZEN
/// subsequence `ssao → bloom → edge → dof → grade → vignette → DITHER LAST`; every
/// new stage slots strictly BETWEEN those frozen ones, never reordering them:
///  1. **SSAO** — darken recessed backgrounds from the clean depth buffer.
///  2. **bloom** — light first: bright cells bleed glow (soft-knee threshold).
///  3. **edge-outline** — structure AFTER light (crisp, unbloomed ink; «свет до
///     контуров»; audit #3). Now also fires on luma discontinuities.
///  4. **DoF** — far cells soften toward their neighbour average + desaturate.
///  5. **chromatic aberration** — the «distortion» slot: a lens fg/bg channel
///     split that grows toward the frame edge (after structure, before colour).
///  6. **grade / tonemap** — the global mood grade (ACES or Reinhard) on fg + bg.
///  7. **colour temperature / tint** — white-balance warm/cool + magenta/green.
///  8. **posterize** — optional float-domain colour banding for an ASCII-poster
///     look (distinct from the palette quantization of the dither pass).
///  9. **vignette** — radial edge darkening (quintic falloff) over the graded image.
/// 10. **scanlines / CRT mask** — retro row/column darkening stylization.
/// 11. **film grain** — cell+slow-time-anchored luminance wobble (stable per bucket,
///     no per-frame shimmer).
/// 12. **dither** — ordered Bayer dithering LAST, quantising the float of every
///     prior pass into the terminal palette («дизер последним»). Uniquely
///     deferrable past an external glitch layer via `skip_dither`.
///
/// The five neighbour-sampling passes (ssao/bloom/edge/dof/chromatic) sample a
/// CLEAN copy of the frame as it entered that stage (so neighbour reads never see a
/// half-processed cell). Per audit #5 those copies are **ping-ponged through one
/// reused scratch buffer** (`copy_into` re-seeds it) rather than a fresh clone per
/// pass. The master [`PostfxConfig::intensity`] scales every effect where scaling
/// is meaningful; at `intensity == 0` (or all effects disabled) this is the
/// identity — `fb` is left byte-identical (see the `off_is_identity` test).
pub fn apply_postfx(fb: &mut Framebuffer, cfg: &PostfxConfig) {
    apply_postfx_ex(fb, cfg, false);
}

/// As [`apply_postfx`], but when `skip_dither` is `true` the final ordered-dither
/// pass is NOT applied — so a caller can run the rest of the stack, then a glitch
/// layer, then call [`apply_dither`] as the genuine LAST quantisation (audit #4:
/// «DITHER LAST», after glitch's colour-mutating passes — otherwise bit-rot /
/// dropout / invert / aberration re-introduce un-dithered banding *after* dither).
pub fn apply_postfx_ex(fb: &mut Framebuffer, cfg: &PostfxConfig, skip_dither: bool) {
    // Fast identity path: a zero master means "no post-process at all", so
    // shipped levels with the default-off config pay nothing and the frame is
    // provably unchanged.
    if cfg.intensity <= 0.0 {
        return;
    }

    // One reused scratch buffer, ping-ponged across the four neighbour-sampling
    // passes: before each pass `copy_into` re-seeds it from the working frame,
    // the pass reads it as the CLEAN source and writes selectively into `fb`.
    // Identical result to `let src = fb.clone()` per pass, minus the allocations.
    // Allocated lazily so the global per-cell passes (grade/vignette/dither) and
    // the all-disabled neighbour case cost nothing.
    let needs_scratch = cfg.ssao.enabled
        || cfg.bloom.enabled
        || cfg.edge.enabled
        || cfg.dof.enabled
        || cfg.chromatic.enabled;
    let mut scratch = if needs_scratch {
        Some(fb.clone())
    } else {
        None
    };

    // 1. SSAO — sample the clean depth buffer for neighbour occlusion.
    if cfg.ssao.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::ssao::apply(fb, src, &cfg.ssao, cfg.intensity);
    }

    // 2. Bloom — LIGHT before structure: threshold + bleed from the pre-bloom image.
    if cfg.bloom.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::bloom::apply(fb, src, &cfg.bloom, cfg.intensity);
    }

    // 3. Edge-outline — STRUCTURE on top of the glow (crisp lines, unbloomed).
    if cfg.edge.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::edge::apply(fb, src, &cfg.edge, cfg.intensity);
    }

    // 4. DoF — far cells blend toward the (clean) neighbour average.
    if cfg.dof.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::dof::apply(fb, src, &cfg.dof, cfg.intensity);
    }

    // 5. Chromatic aberration — the «distortion» slot: a lens channel split that
    //    samples offset neighbours (clean src), strongest at the frame edge.
    if cfg.chromatic.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::chromatic::apply(fb, src, &cfg.chromatic, cfg.intensity);
    }

    // 6–9. Global per-cell color passes, in place (no neighbour sampling):
    //      grade → colour-temp → posterize → vignette.
    effects::grade::apply(fb, &cfg.grade, cfg.intensity);
    effects::color_temp::apply(fb, &cfg.color_temp, cfg.intensity);
    effects::posterize::apply(fb, &cfg.posterize, cfg.intensity);
    effects::vignette::apply(fb, &cfg.vignette, cfg.intensity);

    // 10–11. CRT stylization: scanlines/mask then cell+slow-time-anchored grain.
    effects::scanlines::apply(fb, &cfg.scanlines, cfg.intensity);
    effects::film_grain::apply(fb, &cfg.film_grain, cfg.intensity);

    // 12. Dither LAST — unless the caller defers it past a later glitch layer.
    if !skip_dither {
        effects::dither::apply(fb, &cfg.dither, cfg.intensity);
    }
}

/// Apply ONLY the ordered-dither pass — the final quantisation that must run as
/// the absolute last screen-space operation (after postfx AND after any glitch
/// layer). Pairs with [`apply_postfx_ex`]`(.., skip_dither = true)` so the
/// dither genuinely closes the pipeline (spec §15 «DITHER LAST»; audit #4).
///
/// No-op when the master `intensity` is zero or dither is disabled, so the
/// `off`/zero presets stay byte-identical.
pub fn apply_dither(fb: &mut Framebuffer, cfg: &PostfxConfig) {
    if cfg.intensity <= 0.0 {
        return;
    }
    effects::dither::apply(fb, &cfg.dither, cfg.intensity);
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, SurfaceKind};

    /// A small gradient scene: a near wall band, a recessed far band, one bright
    /// cell, varied surfaces — enough for every pass to find something to do.
    fn scene(cols: usize, rows: usize) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                let near = col < cols / 2;
                let c = Cell {
                    glyph: u32::from(if near { '#' } else { '.' }),
                    fg: if near { 0x00A0_A0A0 } else { 0x0040_4040 },
                    bg: 0x0020_2020,
                    depth: if near { 2.0 } else { 9.0 },
                    surface: if near {
                        SurfaceKind::WallSolid
                    } else {
                        SurfaceKind::Floor
                    },
                    ..Default::default()
                };
                fb.set(col, row, c);
            }
        }
        fb
    }

    /// Equality over the contract fields we touch (glyph/fg/bg) plus style, so a
    /// test can assert "unchanged" or "changed" precisely.
    fn frames_equal(a: &Framebuffer, b: &Framebuffer) -> bool {
        if a.cols() != b.cols() || a.rows() != b.rows() {
            return false;
        }
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                let (x, y) = (a.get(col, row).unwrap(), b.get(col, row).unwrap());
                if x.glyph != y.glyph || x.fg != y.fg || x.bg != y.bg || x.style != y.style {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn off_is_identity() {
        let fb0 = scene(16, 10);
        let mut fb = fb0.clone();
        apply_postfx(&mut fb, &PostfxConfig::off());
        assert!(
            frames_equal(&fb0, &fb),
            "off config must leave the frame byte-identical"
        );
    }

    #[test]
    fn enabled_but_zero_master_is_identity() {
        let fb0 = scene(16, 10);
        let mut fb = fb0.clone();
        // Every effect on, but master 0 → fast identity path.
        let cfg = PostfxConfig::cinematic().with_intensity(0.0);
        apply_postfx(&mut fb, &cfg);
        assert!(
            frames_equal(&fb0, &fb),
            "master 0 must short-circuit to identity"
        );
    }

    #[test]
    fn deterministic() {
        let fb0 = scene(20, 12);
        let cfg = PostfxConfig::cinematic();
        let mut a = fb0.clone();
        let mut b = fb0.clone();
        apply_postfx(&mut a, &cfg);
        apply_postfx(&mut b, &cfg);
        assert!(
            frames_equal(&a, &b),
            "same (fb,cfg) must produce identical output"
        );
    }

    #[test]
    fn style_field_is_preserved() {
        let mut fb = scene(12, 8);
        // Stamp a non-default style on one cell.
        let mut c = *fb.get(3, 3).unwrap();
        c.style = re_core::CellStyle::default()
            .with_bold()
            .with_underline_color(0x00FF_0000);
        fb.set(3, 3, c);
        let before = *fb.get(3, 3).unwrap();
        apply_postfx(&mut fb, &PostfxConfig::cinematic());
        let after = *fb.get(3, 3).unwrap();
        assert_eq!(
            before.style, after.style,
            "postfx must never touch the cell style field"
        );
    }

    #[test]
    fn each_effect_changes_the_frame() {
        let presets: &[(&str, PostfxConfig)] = &[
            ("edge", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.edge.enabled = true;
                c
            }),
            ("ssao", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.ssao.enabled = true;
                c
            }),
            ("bloom", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.bloom.enabled = true;
                c.bloom.threshold = 0.3; // our brightest cell is mid-grey
                c
            }),
            ("dof", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.dof.enabled = true;
                c.dof.focus_dist = 2.0;
                c.dof.range = 4.0;
                c
            }),
            ("grade", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.grade.enabled = true;
                c.grade.contrast = 1.5;
                c
            }),
            ("vignette", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.vignette.enabled = true;
                c
            }),
            ("dither", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.dither.enabled = true;
                c.dither.amount = 0.2;
                c
            }),
            ("chromatic", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.chromatic.enabled = true;
                c.chromatic.amount = 2.0;
                c.chromatic.edge_only = 0.0; // fringe everywhere so the small scene shifts
                c
            }),
            ("color_temp", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.color_temp.enabled = true;
                c.color_temp.temperature = 0.5;
                c
            }),
            ("posterize", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.posterize.enabled = true;
                c.posterize.levels = 3; // harsh banding so our greys snap
                c
            }),
            ("scanlines", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.scanlines.enabled = true;
                c.scanlines.strength = 0.5;
                c.scanlines.period = 1;
                c
            }),
            ("film_grain", {
                let mut c = PostfxConfig::off();
                c.intensity = 1.0;
                c.film_grain.enabled = true;
                c.film_grain.amount = 0.2;
                c
            }),
        ];
        for (name, cfg) in presets {
            let fb0 = scene(20, 12);
            let mut fb = fb0.clone();
            apply_postfx(&mut fb, cfg);
            assert!(
                !frames_equal(&fb0, &fb),
                "effect '{name}' must change the frame"
            );
        }
    }

    /// Every new effect must be a NO-OP when its `enabled` flag is false, even at
    /// full master intensity — so the identity preset stays byte-identical and a
    /// disabled effect never sneaks a change in.
    #[test]
    fn each_new_effect_is_noop_when_off() {
        // Configs with the master on but the named effect explicitly disabled.
        let mut full_on = PostfxConfig::off();
        full_on.intensity = 1.0;
        // Each tuple: a label + a closure that returns a cfg with ONLY non-enabled
        // params set for one effect (its `enabled` left false).
        let cases: &[(&str, PostfxConfig)] = &[
            ("chromatic", {
                let mut c = full_on;
                c.chromatic.amount = 3.0;
                c.chromatic.edge_only = 0.0;
                c
            }),
            ("color_temp", {
                let mut c = full_on;
                c.color_temp.temperature = 1.0;
                c.color_temp.tint = 1.0;
                c
            }),
            ("posterize", {
                let mut c = full_on;
                c.posterize.levels = 2;
                c
            }),
            ("scanlines", {
                let mut c = full_on;
                c.scanlines.strength = 1.0;
                c.scanlines.mask = 1.0;
                c
            }),
            ("film_grain", {
                let mut c = full_on;
                c.film_grain.amount = 0.5;
                c.film_grain.time_bucket = 99;
                c
            }),
        ];
        for (name, cfg) in cases {
            let fb0 = scene(20, 12);
            let mut fb = fb0.clone();
            apply_postfx(&mut fb, cfg);
            assert!(
                frames_equal(&fb0, &fb),
                "disabled effect '{name}' must leave the frame byte-identical"
            );
        }
    }

    /// Film grain must be TEMPORALLY STABLE: a static cell at a FIXED time bucket
    /// is byte-identical across repeated applies (no per-frame shimmer), but a
    /// DIFFERENT bucket re-rolls the grain (it does evolve over slow time).
    #[test]
    fn film_grain_is_stable_within_a_bucket_and_rerolls_across_buckets() {
        let fb0 = scene(24, 16);
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.film_grain.enabled = true;
        cfg.film_grain.amount = 0.15;

        // Same bucket, applied twice from the same clean source → identical.
        cfg.film_grain.time_bucket = 7;
        let mut a = fb0.clone();
        apply_postfx(&mut a, &cfg);
        let mut b = fb0.clone();
        apply_postfx(&mut b, &cfg);
        assert!(
            frames_equal(&a, &b),
            "grain at a FIXED bucket must be byte-stable (no per-frame shimmer)"
        );

        // A different bucket must re-roll the grain (it evolves over slow time).
        let mut c = fb0.clone();
        cfg.film_grain.time_bucket = 8;
        apply_postfx(&mut c, &cfg);
        assert!(
            !frames_equal(&a, &c),
            "a different bucket must re-roll the grain (it crawls over slow time)"
        );
    }

    /// Posterize must reduce the number of distinct fg colours: a smooth grey ramp
    /// quantizes to at most `levels` values.
    #[test]
    fn posterize_reduces_distinct_colors() {
        use std::collections::HashSet;
        let cols = 32;
        let mut fb = Framebuffer::new(cols, 1);
        for col in 0..cols {
            let g = ((col * 255 / (cols - 1)) & 0xFF) as u32;
            fb.set(
                col,
                0,
                Cell {
                    fg: (g << 16) | (g << 8) | g,
                    bg: 0,
                    depth: 3.0,
                    ..Default::default()
                },
            );
        }
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.posterize.enabled = true;
        cfg.posterize.levels = 4;
        apply_postfx(&mut fb, &cfg);
        let mut distinct = HashSet::new();
        for col in 0..cols {
            distinct.insert(fb.get(col, 0).unwrap().fg);
        }
        assert!(
            distinct.len() <= 4,
            "posterize to 4 levels must leave <= 4 distinct greys (got {})",
            distinct.len()
        );
    }

    /// Colour temperature must actually warm the image: a warm push boosts the red
    /// channel and cuts the blue on a neutral grey.
    #[test]
    fn color_temp_warm_boosts_red_cuts_blue() {
        let mut fb = Framebuffer::new(2, 2);
        for row in 0..2 {
            for col in 0..2 {
                fb.set(
                    col,
                    row,
                    Cell {
                        fg: 0x0080_8080,
                        bg: 0x0080_8080,
                        depth: 3.0,
                        ..Default::default()
                    },
                );
            }
        }
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.color_temp.enabled = true;
        cfg.color_temp.temperature = 0.6; // warm
        apply_postfx(&mut fb, &cfg);
        let c = fb.get(0, 0).unwrap();
        let r = (c.fg >> 16) & 0xFF;
        let b = c.fg & 0xFF;
        assert!(r > 0x80, "warm temp must boost red (got {r:#04x})");
        assert!(b < 0x80, "warm temp must cut blue (got {b:#04x})");
    }

    #[test]
    fn ssao_darkens_a_recessed_cell() {
        // A single deep cell surrounded by near cells: SSAO must darken its bg.
        let mut fb = Framebuffer::new(5, 5);
        for row in 0..5 {
            for col in 0..5 {
                fb.set(
                    col,
                    row,
                    Cell {
                        bg: 0x0080_8080,
                        depth: 2.0,
                        surface: SurfaceKind::WallSolid,
                        ..Default::default()
                    },
                );
            }
        }
        // Center cell is far behind its neighbours (a recess).
        fb.set(
            2,
            2,
            Cell {
                bg: 0x0080_8080,
                depth: 6.0,
                surface: SurfaceKind::WallSolid,
                ..Default::default()
            },
        );
        let before = fb.get(2, 2).unwrap().bg;
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.ssao.enabled = true;
        cfg.ssao.strength = 0.8;
        cfg.ssao.bias = 0.1;
        apply_postfx(&mut fb, &cfg);
        let after = fb.get(2, 2).unwrap().bg;
        assert!(
            after < before,
            "recessed cell bg must darken (before={before:#08x} after={after:#08x})"
        );
        // A flat neighbour (corner, all same depth) must be (near) untouched.
        let n_before = 0x0080_8080;
        let n_after = fb.get(0, 0).unwrap().bg;
        assert!(
            n_after >= color_scaled(n_before, 0.5),
            "flat-region cell should darken far less"
        );
    }

    /// Helper: a packed grey scaled by k, for loose comparisons.
    fn color_scaled(c: u32, k: f32) -> u32 {
        let g = ((c & 0xFF) as f32 * k) as u32;
        (g << 16) | (g << 8) | g
    }

    #[test]
    fn bloom_brightens_a_neighbour_of_a_bright_cell() {
        // A dark field with one bright cell; bloom must lift a neighbour's fg.
        let mut fb = Framebuffer::new(7, 7);
        for row in 0..7 {
            for col in 0..7 {
                fb.set(
                    col,
                    row,
                    Cell {
                        fg: 0x0010_1010,
                        bg: 0,
                        depth: 3.0,
                        ..Default::default()
                    },
                );
            }
        }
        fb.set(
            3,
            3,
            Cell {
                fg: 0x00FF_FFFF, // a hot cell
                bg: 0,
                depth: 3.0,
                ..Default::default()
            },
        );
        let nb_before = fb.get(2, 3).unwrap().fg;
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.bloom.enabled = true;
        cfg.bloom.threshold = 0.6;
        cfg.bloom.radius = 1;
        cfg.bloom.intensity = 1.0;
        apply_postfx(&mut fb, &cfg);
        let nb_after = fb.get(2, 3).unwrap().fg;
        assert!(
            nb_after > nb_before,
            "neighbour of a bright cell must brighten (before={nb_before:#08x} after={nb_after:#08x})"
        );
    }

    #[test]
    fn edge_outline_fires_on_a_depth_discontinuity() {
        // Left half near, right half far → a vertical depth cliff at the seam.
        // The edge pass must darken fg at the seam but leave a flat-interior cell
        // (deep in the near region, far from any discontinuity) alone.
        let mut fb = Framebuffer::new(10, 6);
        for row in 0..6 {
            for col in 0..10 {
                let near = col < 5;
                fb.set(
                    col,
                    row,
                    Cell {
                        glyph: u32::from('#'),
                        fg: 0x0090_9090,
                        depth: if near { 2.0 } else { 8.0 },
                        surface: SurfaceKind::WallSolid, // same surface → pure depth edge
                        ..Default::default()
                    },
                );
            }
        }
        let seam_before = fb.get(4, 3).unwrap().fg; // last near col, adjacent to far
        let interior_before = fb.get(1, 3).unwrap().fg; // deep in near region
        let mut cfg = PostfxConfig::off();
        cfg.intensity = 1.0;
        cfg.edge.enabled = true;
        cfg.edge.depth_threshold = 0.2;
        cfg.edge.strength = 0.8;
        apply_postfx(&mut fb, &cfg);
        let seam_after = fb.get(4, 3).unwrap().fg;
        let interior_after = fb.get(1, 3).unwrap().fg;
        assert!(
            seam_after < seam_before,
            "edge cell fg must be inked darker"
        );
        assert_eq!(
            interior_after, interior_before,
            "flat interior must be untouched"
        );
    }

    /// Audit #3: BLOOM must run BEFORE EDGE so the contour ink lands on top of the
    /// glow and is not itself smeared by bloom. We prove the order by reconstructing
    /// both orderings by hand and asserting `apply_postfx` matches bloom→edge (the
    /// new, correct order) and DIFFERS from edge→bloom (the old, buggy order).
    fn bloom_edge_scene() -> Framebuffer {
        // A bright vertical seam next to a depth/surface cliff: bloom has something
        // hot to bleed, and edge has a discontinuity to ink, in the same region.
        let mut fb = Framebuffer::new(9, 7);
        for row in 0..7 {
            for col in 0..9 {
                let near = col < 4;
                fb.set(
                    col,
                    row,
                    Cell {
                        glyph: u32::from('#'),
                        fg: if col == 4 { 0x00FF_FFFF } else { 0x0030_3030 },
                        bg: 0x0010_1010,
                        depth: if near { 2.0 } else { 9.0 },
                        surface: if near {
                            SurfaceKind::WallSolid
                        } else {
                            SurfaceKind::Floor
                        },
                        ..Default::default()
                    },
                );
            }
        }
        fb
    }

    fn bloom_edge_cfg() -> PostfxConfig {
        let mut c = PostfxConfig::off();
        c.intensity = 1.0;
        c.bloom.enabled = true;
        c.bloom.threshold = 0.5;
        c.bloom.radius = 2;
        c.bloom.intensity = 1.0;
        c.edge.enabled = true;
        c.edge.depth_threshold = 0.2;
        c.edge.strength = 0.9;
        c
    }

    #[test]
    fn bloom_runs_before_edge() {
        let scene0 = bloom_edge_scene();
        let cfg = bloom_edge_cfg();

        // What apply_postfx actually produces.
        let mut actual = scene0.clone();
        apply_postfx(&mut actual, &cfg);

        // Reference: the CORRECT order — bloom (light) THEN edge (structure).
        let mut correct = scene0.clone();
        {
            let src = correct.clone();
            effects::bloom::apply(&mut correct, &src, &cfg.bloom, cfg.intensity);
            let src = correct.clone();
            effects::edge::apply(&mut correct, &src, &cfg.edge, cfg.intensity);
        }

        // Reference: the OLD buggy order — edge THEN bloom (edge ink gets bloomed).
        let mut buggy = scene0.clone();
        {
            let src = buggy.clone();
            effects::edge::apply(&mut buggy, &src, &cfg.edge, cfg.intensity);
            let src = buggy.clone();
            effects::bloom::apply(&mut buggy, &src, &cfg.bloom, cfg.intensity);
        }

        assert!(
            frames_equal(&actual, &correct),
            "apply_postfx must compose bloom BEFORE edge (spec §15: light before contours)"
        );
        assert!(
            !frames_equal(&correct, &buggy),
            "the two orders must actually differ on this scene (test is meaningful)"
        );
    }

    /// Audit #5: the ping-pong scratch buffer must give BYTE-IDENTICAL output to
    /// the old clone-per-pass implementation. We reconstruct the old per-pass-clone
    /// pipeline (with the NEW order) by hand and compare to `apply_postfx`.
    #[test]
    fn ping_pong_matches_clone_per_pass() {
        let scene0 = scene(24, 15);
        let cfg = PostfxConfig::cinematic();

        let mut actual = scene0.clone();
        apply_postfx(&mut actual, &cfg);

        // Hand-rolled clone-per-pass reference in the SAME order as apply_postfx.
        let mut reference = scene0.clone();
        if cfg.ssao.enabled {
            let src = reference.clone();
            effects::ssao::apply(&mut reference, &src, &cfg.ssao, cfg.intensity);
        }
        if cfg.bloom.enabled {
            let src = reference.clone();
            effects::bloom::apply(&mut reference, &src, &cfg.bloom, cfg.intensity);
        }
        if cfg.edge.enabled {
            let src = reference.clone();
            effects::edge::apply(&mut reference, &src, &cfg.edge, cfg.intensity);
        }
        if cfg.dof.enabled {
            let src = reference.clone();
            effects::dof::apply(&mut reference, &src, &cfg.dof, cfg.intensity);
        }
        if cfg.chromatic.enabled {
            let src = reference.clone();
            effects::chromatic::apply(&mut reference, &src, &cfg.chromatic, cfg.intensity);
        }
        effects::grade::apply(&mut reference, &cfg.grade, cfg.intensity);
        effects::color_temp::apply(&mut reference, &cfg.color_temp, cfg.intensity);
        effects::posterize::apply(&mut reference, &cfg.posterize, cfg.intensity);
        effects::vignette::apply(&mut reference, &cfg.vignette, cfg.intensity);
        effects::scanlines::apply(&mut reference, &cfg.scanlines, cfg.intensity);
        effects::film_grain::apply(&mut reference, &cfg.film_grain, cfg.intensity);
        effects::dither::apply(&mut reference, &cfg.dither, cfg.intensity);

        assert!(
            frames_equal(&actual, &reference),
            "ping-pong scratch must be byte-identical to clone-per-pass (audit #5 is pure perf)"
        );
    }

    /// Audit #4: `skip_dither` must run the whole stack EXCEPT the final dither, and
    /// a following `apply_dither` must reproduce the full `apply_postfx` result —
    /// proving dither can be deferred to run last (after a glitch layer) without
    /// changing the composition when nothing runs in between.
    #[test]
    fn skip_dither_then_apply_dither_equals_full_postfx() {
        let scene0 = scene(20, 12);
        let mut cfg = PostfxConfig::cinematic();
        cfg.dither.enabled = true; // cinematic has dither off; force it on to test
        cfg.dither.amount = 0.08;

        let mut full = scene0.clone();
        apply_postfx(&mut full, &cfg);

        let mut deferred = scene0.clone();
        apply_postfx_ex(&mut deferred, &cfg, true);
        // (a glitch layer would run here in the real pipeline)
        apply_dither(&mut deferred, &cfg);

        assert!(
            frames_equal(&full, &deferred),
            "skip_dither + later apply_dither must equal a single full apply_postfx"
        );

        // And skip_dither WITHOUT the trailing dither must differ from the full run
        // (proving the dither pass is genuinely deferred, not silently dropped).
        let mut no_dither = scene0.clone();
        apply_postfx_ex(&mut no_dither, &cfg, true);
        assert!(
            !frames_equal(&full, &no_dither),
            "skip_dither must actually omit the dither (the deferred pass is load-bearing)"
        );
    }

    #[test]
    fn apply_dither_is_noop_when_off() {
        let scene0 = scene(12, 8);
        let mut fb = scene0.clone();
        apply_dither(&mut fb, &PostfxConfig::off());
        assert!(
            frames_equal(&scene0, &fb),
            "apply_dither must be identity on the off preset"
        );
    }

    #[test]
    fn no_compounding_via_repeated_apply_from_clean() {
        // The plugin's anti-compound contract: postfx(clean) is idempotent in the
        // sense that re-deriving from the SAME clean source always yields the same
        // frame, no matter how many times — never a postfx-of-a-postfx.
        let clean = scene(18, 11);
        let cfg = PostfxConfig::cinematic();
        let mut once = clean.clone();
        apply_postfx(&mut once, &cfg);
        // Simulate 5 "idle ticks": each writes clean → live, then postfx.
        let mut live = clean.clone();
        for _ in 0..5 {
            live = clean.clone();
            apply_postfx(&mut live, &cfg);
        }
        assert!(
            frames_equal(&once, &live),
            "re-deriving from the clean snapshot must never compound"
        );
        // And applying postfx a SECOND time in place (the bug we prevent) must
        // differ from the single pass — proving compounding is real and that the
        // snapshot pattern (not in-place) is what avoids it.
        let mut twice = once.clone();
        apply_postfx(&mut twice, &cfg);
        assert!(
            !frames_equal(&once, &twice),
            "applying in place twice DOES compound — the snapshot pattern is necessary"
        );
    }
}
