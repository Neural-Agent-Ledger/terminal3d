//! [`GlitchPlugin`] — runs the glitch post-process each frame, AFTER the shade
//! pass produced the [`Framebuffer`] and BEFORE the present layer draws it.
//!
//! # The compounding hazard (why a snapshot exists)
//!
//! `re_render`'s raycaster is dirty-flagged: on an idle, non-animated frame it
//! SKIPS re-shading and leaves the previous [`Framebuffer`] intact for the
//! present layer to reuse. If we glitched the live `Framebuffer` IN PLACE every
//! tick, an idle frame would re-glitch the already-glitched buffer — corruption
//! would COMPOUND without bound (bit-rot eats itself, tears stack into garbage).
//!
//! The fix: keep a [`GlitchSnapshot`] of the CLEAN shaded frame. Each tick we
//! refresh the snapshot only when the renderer actually produced a new frame
//! (`Framebuffer::is_changed()`), then write `glitch(clean_snapshot) →
//! Framebuffer`. The live buffer is therefore always exactly "one glitch pass
//! over the latest clean frame" — never a glitch-of-a-glitch — even across many
//! idle ticks, while still EVOLVING because `time` advances.
//!
//! Ordering: the system runs in `PostUpdate` AFTER `re_render`'s `raycast_sys`
//! (we depend on the shaded frame). The binary adds `GlitchPlugin` after
//! `RenderPlugin`; Bevy's default `PostUpdate` ordering plus `chain`-free
//! scheduling means we explicitly order after the cast via an `.after` on the
//! public symbol path isn't available here (cross-crate), so we instead rely on
//! reading `Framebuffer` state: even if we ran before the cast on frame 0, we
//! simply snapshot what's there and glitch it — correctness does not depend on
//! the exact intra-stage order, only on snapshotting the clean frame.

use bevy::prelude::*;
use re_core::Framebuffer;

use crate::effects::{color, GlitchConfig};
use crate::{apply_glitch, GlitchPhosphorState};

/// A clean copy of the most recently SHADED framebuffer, kept so the glitch is
/// re-derived from an un-corrupted source every tick (see the module docs). Also
/// caches the dimensions so a terminal resize re-snapshots cleanly.
#[derive(Resource, Default)]
pub struct GlitchSnapshot {
    /// The clean shaded frame; `None` until the first frame is captured.
    clean: Option<Framebuffer>,
}

/// The seed all glitch passes use. Fixed → deterministic; visuals vary by `time`
/// (the `Res<Time>` elapsed seconds), never by reseeding.
const GLITCH_SEED: u64 = 0x5245_5601_4744_4C54; // "REV\x01" + "GLT"

/// Adds the [`GlitchConfig`] resource (default [`GlitchConfig::off`] — additive,
/// shipped levels unaffected), the snapshot + phosphor state, and the
/// post-process system to `PostUpdate` (after the render cast).
pub struct GlitchPlugin;

impl Plugin for GlitchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlitchConfig>()
            .init_resource::<GlitchSnapshot>()
            .init_resource::<GlitchPhosphorState>()
            // Runs in PostUpdate. We do NOT hard-chain after re_render's private
            // system symbol (cross-crate); instead the snapshot logic is correct
            // regardless of intra-stage order — it captures whatever clean frame
            // exists and never compounds. Real ordering after the cast is
            // achieved by adding this plugin AFTER RenderPlugin in the binary,
            // and Bevy preserving add-order for same-set systems with no other
            // constraints.
            .add_systems(PostUpdate, glitch_post_process);
    }
}

/// The per-frame post-process.
///
/// 1. If the renderer produced a NEW clean frame this tick (`fb.is_changed()`)
///    or the size changed, refresh the [`GlitchSnapshot`] from it.
/// 2. If glitch is off (or no snapshot yet), leave the live `Framebuffer` alone.
/// 3. Otherwise overwrite the live `Framebuffer` with the clean snapshot and run
///    [`apply_glitch`] on it for the current `time` — so the result is always one
///    glitch pass over the clean frame, never compounding.
/// 4. Apply phosphor persistence (stateful: blends the previous PRESENTED frame
///    under the new glitched one) and store the result for the next tick.
fn glitch_post_process(
    cfg: Res<GlitchConfig>,
    time: Res<Time>,
    mut fb: ResMut<Framebuffer>,
    mut snap: ResMut<GlitchSnapshot>,
    mut ghost: ResMut<GlitchPhosphorState>,
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
    //    Also drop any phosphor history so re-enabling starts clean.
    if cfg.intensity <= 0.0 {
        ghost.prev = None;
        return;
    }
    let Some(clean) = snap.clean.as_ref() else {
        return; // nothing shaded yet
    };

    // 3. Write clean → live, then glitch the live buffer ONCE for this time.
    let elapsed = time.elapsed_secs();
    // Overwrite live from the clean snapshot without reallocating when sizes match.
    if fb.cols() == clean.cols() && fb.rows() == clean.rows() {
        copy_into(&mut fb, clean);
    } else {
        *fb = clean.clone();
    }
    apply_glitch(&mut fb, &cfg, elapsed, GLITCH_SEED);

    // 4. Phosphor persistence: blend the previous presented frame under this one.
    if cfg.phosphor.enabled {
        let p = (cfg.phosphor.persistence * cfg.phosphor.intensity * cfg.intensity).clamp(0.0, 1.0);
        if p > 0.0 {
            if let Some(prev) = ghost.prev.as_ref() {
                if prev.cols() == fb.cols() && prev.rows() == fb.rows() {
                    blend_ghost(&mut fb, prev, p);
                }
            }
        }
        ghost.prev = Some(fb.clone());
    } else {
        ghost.prev = None;
    }
}

/// Copy every cell of `src` into `dst` (same dimensions assumed) without
/// reallocating `dst`'s backing storage.
fn copy_into(dst: &mut Framebuffer, src: &Framebuffer) {
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            if let Some(c) = src.get(col, row) {
                dst.set(col, row, *c);
            }
        }
    }
}

/// Phosphor blend: where the current cell is darker than the ghost would leave
/// it, lift it toward a dimmed copy of the previous frame's colour — faint decay
/// trails. We MAX toward the dimmed ghost so trails only ever ADD a faint glow,
/// never darken the live image. Pure per-cell colour op.
fn blend_ghost(fb: &mut Framebuffer, prev: &Framebuffer, persistence: f32) {
    for row in 0..fb.rows() {
        for col in 0..fb.cols() {
            let (Some(cur), Some(old)) = (fb.get(col, row), prev.get(col, row)) else {
                continue;
            };
            let mut c = *cur;
            c.fg = ghost_max(c.fg, old.fg, persistence);
            c.bg = ghost_max(c.bg, old.bg, persistence);
            fb.set(col, row, c);
        }
    }
}

/// Per-channel `max(current, previous * persistence)` — a faded trail that can
/// only brighten, so motion leaves a dim wake without muddying the new frame.
#[inline]
fn ghost_max(cur: u32, prev: u32, k: f32) -> u32 {
    let (cr, cg, cb) = color::unpack(cur);
    let (pr, pg, pb) = color::unpack(prev);
    let dim = |p: u8| (p as f32 * k) as u8;
    color::pack(cr.max(dim(pr)), cg.max(dim(pg)), cb.max(dim(pb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_only_brightens() {
        // The trail can only raise a channel, never lower it.
        let cur = color::pack(10, 200, 30);
        let prev = color::pack(255, 0, 255);
        let out = ghost_max(cur, prev, 0.5);
        let (r, g, b) = color::unpack(out);
        assert!(r >= 10 && g >= 200 && b >= 30, "ghost must never darken");
        assert!(r >= 127, "bright prev red must lift the dim current red");
    }
}
