// Test-module assert_eq! on f32 values are intentional exact-equal checks
// (intensity clamp boundaries: exactly 0.0 / 1.0).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_glitch` — a deterministic, configurable VISUAL GLITCH post-process over
//! the ASCII raycaster's [`Framebuffer`].
//!
//! # What it is
//!
//! A terminal-native **corruption** layer that runs AFTER the shade pass has
//! produced a clean [`Framebuffer`] and BEFORE the present layer draws it. It
//! mutates glyph/fg/bg (never geometry) to make the rendered scene read as a
//! degrading signal — scanlines, bit-rot, tearing, RGB-split, dropout, warp,
//! inversion pulses. A host game can drive the corruption density as a
//! narrative timer rather than cosmetic noise — this layer is the visual
//! face of whatever that timer represents.
//!
//! # The two halves
//!
//! 1. **The pure core** — [`apply_glitch`]`(fb, cfg, time, seed)`. No Bevy, no
//!    clock, no `rand`: a deterministic function of exactly its four inputs.
//!    Same `(fb, cfg, time, seed)` → byte-identical output. This is the testable,
//!    reusable heart, shared by the plugin and the binary's `--dump --glitch`.
//! 2. **The Bevy glue** — [`GlitchPlugin`] + the [`GlitchConfig`] resource. The
//!    plugin keeps a *clean snapshot* of the shaded frame and each tick writes
//!    `glitch(snapshot) → Framebuffer`, so a glitch never COMPOUNDS across ticks
//!    even when the dirty-flagged renderer skips re-shading (see [`GlitchPlugin`]).
//!
//! # Determinism
//!
//! Every random-looking decision is a hash of integer coordinates salted by the
//! master `seed`, an effect-local `seed_off`, and a *quantized* time slice — so
//! visuals evolve with `time` but never reseed nondeterministically. `f32`-only
//! maths, no global state. This mirrors `re_render::ascii_shader`'s hash-based
//! determinism so golden tests stay reproducible.
//!
//! # Purity
//!
//! Depends only on `re_core` (the [`Framebuffer`]/`Cell` contract) + `glam`.
//! No terminal, no game crates. Additive by default: [`GlitchConfig::off`] (the
//! plugin default) leaves the frame byte-identical, so the shipped levels render
//! exactly as before unless a level / setting opts in.

mod effects;
mod hash;
mod plugin;

pub use effects::*;
pub use plugin::{GlitchPlugin, GlitchSnapshot};

use bevy::prelude::Resource;
use re_core::Framebuffer;

/// Copy every cell of `src` into `dst` (same dimensions assumed) without
/// reallocating `dst`'s backing storage. The ping-pong primitive: one reused
/// scratch buffer is re-seeded from the working frame before each
/// displacement/aberration pass instead of cloning a fresh one (audit #5).
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

/// Stateful history for the phosphor-persistence effect: the last PRESENTED
/// (glitched) frame, blended dimly under the next one to leave decay trails.
///
/// This is the ONE stateful piece — it cannot live in the pure [`apply_glitch`]
/// (which is a function of its inputs only), so the [`GlitchPlugin`] owns it as a
/// resource and feeds it in. Kept here at the crate root so it is a public,
/// inspectable part of the plugin's surface.
#[derive(Resource, Default)]
pub struct GlitchPhosphorState {
    /// The previous presented frame; `None` until the first glitched frame, or
    /// after phosphor is disabled (so re-enabling starts clean, no stale trail).
    pub prev: Option<Framebuffer>,
}

/// Apply every enabled effect of `cfg` to `fb`, in a fixed deterministic order,
/// for animation `time` (seconds) and master `seed`.
///
/// PURE: the result depends on exactly `(fb, cfg, time, seed)` — call it twice
/// with the same inputs and you get byte-identical frames. No Bevy, no clock, no
/// RNG. This is the function the plugin and the `--dump --glitch` path both call.
///
/// The effect ORDER is deliberate and fixed (changing it changes the look, so it
/// is part of the contract):
/// 1. **warp** — per-column vertical / per-row horizontal sine displacement.
/// 2. **block jitter** — rectangular regions jump a few cells.
/// 3. **row shift / tearing** — horizontal slices slide left/right.
/// 4. **chromatic aberration** — fg/bg channels split into red/blue ghosts.
/// 5. **bit-rot** — a fraction of cells become garbage glyphs (optionally spread).
/// 6. **scanlines / CRT** — periodic dim rows + a rolling bright/dark band.
/// 7. **dropout** — cells flicker to dead black (signal loss).
/// 8. **invert / posterize** — banded or full-frame colour pulses.
/// 9. **phosphor** — faint persistence trails (needs the optional prev-frame).
///
/// Geometry-displacing effects (1–3) run first on the clean buffer; colour /
/// per-cell corruption (4–8) run after, on the displaced layout; persistence (9)
/// blends last so trails ghost the final glitched image. The master
/// [`GlitchConfig::intensity`] scales every effect; at `intensity == 0` (or with
/// all effects disabled) this is the identity — `fb` is left untouched.
pub fn apply_glitch(fb: &mut Framebuffer, cfg: &GlitchConfig, time: f32, seed: u64) {
    // Fast identity path: a zero master scalar means "no corruption at all",
    // so shipped levels with the default-off config pay almost nothing and the
    // frame is provably unchanged (see the `off_is_identity` test).
    if cfg.intensity <= 0.0 {
        return;
    }

    // 1–4 need a CLEAN source copy so a shifted/jittered/split region samples the
    // un-glitched layout, not a half-glitched one. Per audit #5 we ping-pong ONE
    // reused scratch buffer through all four (re-seeding it with `copy_into`
    // before each pass) instead of cloning a fresh frame per pass — same output,
    // ~3 fewer full-frame allocations. Allocated lazily so the colour-only path
    // (no displacement/aberration enabled) costs nothing.
    let needs_scratch =
        cfg.warp.enabled || cfg.block.enabled || cfg.tear.enabled || cfg.aberration.enabled;
    let mut scratch = if needs_scratch {
        Some(fb.clone())
    } else {
        None
    };

    // 1–3: geometry displacement (re-arranges existing cells from the clean copy).
    if cfg.warp.enabled || cfg.block.enabled || cfg.tear.enabled {
        if cfg.warp.enabled {
            let src = scratch.as_mut().unwrap();
            copy_into(src, fb);
            effects::warp::apply(fb, src, &cfg.warp, cfg.intensity, time, seed);
        }
        if cfg.block.enabled {
            let src = scratch.as_mut().unwrap();
            copy_into(src, fb);
            effects::block::apply(fb, src, &cfg.block, cfg.intensity, time, seed);
        }
        if cfg.tear.enabled {
            let src = scratch.as_mut().unwrap();
            copy_into(src, fb);
            effects::tear::apply(fb, src, &cfg.tear, cfg.intensity, time, seed);
        }
    }

    // 4: chromatic aberration — splits colour channels reading the un-split image.
    if cfg.aberration.enabled {
        let src = scratch.as_mut().unwrap();
        copy_into(src, fb);
        effects::aberration::apply(fb, src, &cfg.aberration, cfg.intensity, time, seed);
    }

    // 5–8: per-cell, in place (each reads + writes the same cell, no neighbour
    // sampling beyond bit-rot's own spread which reads the working buffer on
    // purpose for the contagion feel).
    effects::bitrot::apply(fb, &cfg.bitrot, cfg.intensity, time, seed);
    effects::scanlines::apply(fb, &cfg.scanlines, cfg.intensity, time, seed);
    effects::dropout::apply(fb, &cfg.dropout, cfg.intensity, time, seed);
    effects::invert::apply(fb, &cfg.invert, cfg.intensity, time, seed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, SurfaceKind};

    /// A content-rich deterministic frame (mirrors the integration test's).
    fn make_frame(cols: usize, rows: usize) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                let glyph = b'!' as u32 + ((col * 7 + row * 13) % 90) as u32;
                let fg = (((col * 17) & 0xFF) << 16
                    | ((row * 23) & 0xFF) << 8
                    | ((col * row) & 0xFF)) as u32;
                let bg = ((((col + 5) * 11) & 0xFF) << 16 | ((row * 3) & 0xFF) << 8 | 0x20) as u32;
                fb.set(
                    col,
                    row,
                    Cell {
                        glyph,
                        fg,
                        bg,
                        depth: 1.0 + col as f32,
                        surface: SurfaceKind::WallSolid,
                        ..Cell::default()
                    },
                );
            }
        }
        fb
    }

    fn frames_equal(a: &Framebuffer, b: &Framebuffer) -> bool {
        if a.cols() != b.cols() || a.rows() != b.rows() {
            return false;
        }
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                let (x, y) = (a.get(col, row).unwrap(), b.get(col, row).unwrap());
                if x.glyph != y.glyph || x.fg != y.fg || x.bg != y.bg {
                    return false;
                }
            }
        }
        true
    }

    /// Audit #5: the ping-pong scratch buffer must produce BYTE-IDENTICAL output to
    /// the old clone-per-pass implementation (the refactor is PURE PERF). We rebuild
    /// the old fresh-clone-per-pass pipeline by hand and compare to `apply_glitch`.
    #[test]
    fn ping_pong_matches_clone_per_pass() {
        let base = make_frame(80, 32);
        let cfg = GlitchConfig::heavy();
        let (time, seed) = (2.5f32, 0xABCDu64);

        let mut actual = base.clone();
        apply_glitch(&mut actual, &cfg, time, seed);

        // Hand-rolled clone-per-pass reference, same order as apply_glitch.
        let mut reference = base.clone();
        if cfg.warp.enabled {
            let src = reference.clone();
            effects::warp::apply(&mut reference, &src, &cfg.warp, cfg.intensity, time, seed);
        }
        if cfg.block.enabled {
            let src = reference.clone();
            effects::block::apply(&mut reference, &src, &cfg.block, cfg.intensity, time, seed);
        }
        if cfg.tear.enabled {
            let src = reference.clone();
            effects::tear::apply(&mut reference, &src, &cfg.tear, cfg.intensity, time, seed);
        }
        if cfg.aberration.enabled {
            let src = reference.clone();
            effects::aberration::apply(
                &mut reference,
                &src,
                &cfg.aberration,
                cfg.intensity,
                time,
                seed,
            );
        }
        effects::bitrot::apply(&mut reference, &cfg.bitrot, cfg.intensity, time, seed);
        effects::scanlines::apply(&mut reference, &cfg.scanlines, cfg.intensity, time, seed);
        effects::dropout::apply(&mut reference, &cfg.dropout, cfg.intensity, time, seed);
        effects::invert::apply(&mut reference, &cfg.invert, cfg.intensity, time, seed);

        assert!(
            frames_equal(&actual, &reference),
            "ping-pong scratch must be byte-identical to clone-per-pass (audit #5 is pure perf)"
        );
    }
}
