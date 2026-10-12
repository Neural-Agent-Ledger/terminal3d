//! [`PostfxPlugin`] — runs the post-process stack each frame, AFTER the shade
//! pass produced the [`Framebuffer`] and BEFORE the present layer draws it
//! (ideally before `re_glitch`, so corruption sits on top of the cinematic look).
//!
//! # The compounding hazard (why a snapshot exists)
//!
//! `re_render`'s raycaster is dirty-flagged: on an idle, non-animated frame it
//! SKIPS re-shading and leaves the previous [`Framebuffer`] intact for the
//! present layer to reuse. If we post-processed the live `Framebuffer` IN PLACE
//! every tick, an idle frame would re-process the already-processed buffer —
//! bloom would double, the grade would stack, AO would crush to black. It would
//! COMPOUND without bound.
//!
//! The fix (mirrors `re_glitch`): keep a [`PostfxSnapshot`] of the CLEAN shaded
//! frame. Each tick we refresh the snapshot only when the renderer actually
//! produced a new frame (`Framebuffer::is_changed()`) or the size changed, then
//! write `postfx(clean_snapshot) → Framebuffer`. The live buffer is therefore
//! always exactly "one post-process pass over the latest clean frame" — never a
//! postfx-of-a-postfx — even across many idle ticks.
//!
//! Ordering: the system runs in `PostUpdate`. Correctness does NOT depend on the
//! exact intra-stage order — it snapshots whatever clean frame exists and never
//! compounds. The binary adds `PostfxPlugin` after `RenderPlugin` and BEFORE
//! `GlitchPlugin` so postfx runs first and glitch corrupts the graded image.
//!
//! # DITHER LAST (audit #4)
//!
//! The post-process stack runs here WITHOUT its final ordered-dither pass
//! ([`apply_postfx_ex`]`(.., skip_dither = true)`). The dither — the operation
//! that quantises the float of every pass into the terminal palette — must be the
//! ABSOLUTE last screen-space op, *after* `re_glitch`'s colour-mutating passes
//! (bit-rot / dropout / invert / aberration); otherwise glitch re-introduces
//! un-dithered banding on top of an already-dithered frame. The binary therefore
//! adds a separate [`PostfxDitherPlugin`] AFTER `GlitchPlugin`, whose system runs
//! the deferred dither as the genuine final pass (spec §15 «дизер последним»).

use bevy::prelude::*;
use re_core::Framebuffer;

#[cfg(test)]
use crate::apply_postfx;
use crate::config::PostfxConfig;
use crate::{apply_dither, apply_postfx_ex, copy_into};

/// A clean copy of the most recently SHADED framebuffer, kept so the post-process
/// is re-derived from an un-processed source every tick (see the module docs).
#[derive(Resource, Default)]
pub struct PostfxSnapshot {
    /// The clean shaded frame; `None` until the first frame is captured.
    clean: Option<Framebuffer>,
}

/// Adds the [`PostfxConfig`] resource (default [`PostfxConfig::off`] — additive,
/// shipped levels unaffected), the clean snapshot, and the post-process system to
/// `PostUpdate`.
pub struct PostfxPlugin;

impl Plugin for PostfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PostfxConfig>()
            .init_resource::<PostfxSnapshot>()
            // PostUpdate; the snapshot logic is correct regardless of intra-stage
            // order. Real ordering (after the render cast, before glitch) is
            // achieved by plugin add-order in the binary.
            .add_systems(PostUpdate, postfx_post_process);
    }
}

/// Film-grain buckets per second: how fast the cell-anchored grain re-rolls. A
/// handful per second reads as gentle "crawling" film grain — fast enough to feel
/// alive, slow enough that it never strobes per render-frame (the persistence
/// principle: the grain is anchored on `(cell, bucket)` and stable within a
/// bucket).
const GRAIN_BUCKETS_PER_SEC: f32 = 8.0;

/// The per-frame post-process.
///
/// 1. If the renderer produced a NEW clean frame this tick (`fb.is_changed()`) or
///    the size changed, refresh the [`PostfxSnapshot`] from it.
/// 2. If postfx is off (or no snapshot yet), leave the live `Framebuffer` alone.
/// 3. Otherwise overwrite the live `Framebuffer` with the clean snapshot and run
///    [`apply_postfx`] on it — so the result is always one pass over the clean
///    frame, never compounding.
fn postfx_post_process(
    cfg: Res<PostfxConfig>,
    time: Res<Time>,
    mut fb: ResMut<Framebuffer>,
    mut snap: ResMut<PostfxSnapshot>,
) {
    // 1. Refresh the clean snapshot when the renderer produced a fresh frame, or
    //    when the dimensions changed (resize) so we never sample a stale size.
    let size_changed = snap
        .clean
        .as_ref()
        .map(|c| c.cols() != fb.cols() || c.rows() != fb.rows())
        .unwrap_or(true);
    if fb.is_changed() || size_changed {
        snap.clean = Some(fb.clone());
    }

    // 2. Off → identity. Leave the live frame (the clean shaded image) untouched.
    if cfg.intensity <= 0.0 {
        return;
    }
    let Some(clean) = snap.clean.as_ref() else {
        return; // nothing shaded yet
    };

    // 3. Write clean → live, then post-process the live buffer ONCE — but WITHOUT
    //    the final dither pass: dither is deferred to PostfxDitherPlugin, which
    //    runs after GlitchPlugin so it is the genuine last quantisation (audit #4).
    if fb.cols() == clean.cols() && fb.rows() == clean.rows() {
        copy_into(&mut fb, clean);
    } else {
        *fb = clean.clone();
    }
    // Drive the film-grain bucket from a SLOW clock so the cell-anchored grain
    // crawls instead of shimmering per render-frame. The pure core stays a function
    // of (fb, cfg): we just stamp the current bucket into a local copy of the cfg.
    let bucket = (time.elapsed_secs() * GRAIN_BUCKETS_PER_SEC).floor() as i64 as u32;
    let cfg = cfg.with_grain_bucket(bucket);
    apply_postfx_ex(&mut fb, &cfg, true);
}

/// The deferred final-dither plugin: runs the post-process dither pass as the
/// ABSOLUTE last screen-space operation, AFTER both [`PostfxPlugin`] and (in the
/// binary's add-order) `GlitchPlugin`. This guarantees dither quantises the float
/// of EVERY prior pass — postfx grade AND glitch colour corruption — into the
/// terminal palette, instead of glitch re-banding an already-dithered frame
/// (spec §15 «DITHER LAST»; audit #4).
///
/// Reads the same [`PostfxConfig`] resource as [`PostfxPlugin`]; a no-op when
/// postfx is off, so shipped levels are unaffected.
pub struct PostfxDitherPlugin;

impl Plugin for PostfxDitherPlugin {
    fn build(&self, app: &mut App) {
        // PostfxConfig is already added by PostfxPlugin; init defensively in case
        // this plugin is added standalone.
        app.init_resource::<PostfxConfig>()
            .add_systems(PostUpdate, postfx_final_dither);
    }
}

/// Apply the deferred dither over the live (post-processed, glitched) frame.
fn postfx_final_dither(cfg: Res<PostfxConfig>, mut fb: ResMut<Framebuffer>) {
    apply_dither(&mut fb, &cfg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, SurfaceKind};

    fn scene(cols: usize, rows: usize) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                let near = col < cols / 2;
                fb.set(
                    col,
                    row,
                    Cell {
                        glyph: u32::from('#'),
                        fg: 0x00A0_A0A0,
                        bg: 0x0020_2020,
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

    fn frames_equal(a: &Framebuffer, b: &Framebuffer) -> bool {
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

    /// The plugin must NOT compound across idle ticks: with the snapshot held
    /// fixed (no new clean frame), repeated runs leave the live frame identical to
    /// a single postfx-from-clean, even though the system runs many times.
    #[test]
    fn snapshot_prevents_compounding_across_idle_ticks() {
        let clean = scene(16, 10);
        let cfg = PostfxConfig::cinematic();

        // Reference: a single postfx pass over the clean frame.
        let mut reference = clean.clone();
        apply_postfx(&mut reference, &cfg);

        // Simulate the plugin's body across 4 idle ticks where the snapshot is
        // already populated with the clean frame and is NOT refreshed.
        let mut snap = PostfxSnapshot {
            clean: Some(clean.clone()),
        };
        let mut live = clean.clone();
        for _ in 0..4 {
            // (idle: fb not changed → snapshot kept) write clean → live, postfx.
            let c = snap.clean.as_ref().unwrap().clone();
            copy_into(&mut live, &c);
            apply_postfx(&mut live, &cfg);
        }
        let _ = &mut snap;
        assert!(
            frames_equal(&reference, &live),
            "idle ticks must not compound: live must equal one postfx-from-clean"
        );
    }
}
