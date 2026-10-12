//! Full-screen tunnel set-piece — reuses the log-polar [`fractal`](crate::fractal)
//! field over the WHOLE framebuffer (no geometry, bypassing the raycaster) to
//! paint a hypnotic forward dive. Pure compute, **zero terminal, zero per-cell
//! heap allocation**.
//!
//! UV is measured from the screen centre. The phase runs a `SpinZoom`-style
//! coupled scroll (the zoom rushes rings out of the centre — reads as flying
//! forward — and the spin curls them into a Droste dive). The fg/bg palette and
//! the body-glyph ramp are sampled the SAME way `Material::sample_fractal`
//! samples its portal material, so the tunnel and the wall portal read as one
//! coherent look. `depth` is driven by the ring phase so near rings read close.
use std::sync::LazyLock;

use glam::Vec3;
use noise::Perlin;
use re_core::{Cell, Framebuffer, SurfaceKind};

use crate::ascii_shader::{body_ramp, pack_rgb, Palette, SHADER_SEED};
use crate::fractal::{log_polar_field, FractalParams, LN2};

/// The shared `Perlin` for the tunnel's fBm enrichment. Built ONCE for the whole
/// process (not per cell, not per frame) — mirrors the shader's fixed-seed
/// determinism so the same `(coord, t)` always yields the same frame.
static TUNNEL_PERLIN: LazyLock<Perlin> = LazyLock::new(|| Perlin::new(SHADER_SEED));

/// Cool teal → hot magenta ring sweep (matches `portal_spiral`'s fg palette) so
/// the dive reads as the same material the wall portal uses.
static TUNNEL_FG: LazyLock<Palette> = LazyLock::new(|| Palette {
    a: Vec3::new(0.70, 0.45, 0.30),
    b: Vec3::new(0.30, 0.25, 0.40),
    c: Vec3::new(1.0, 1.0, 1.0),
    d: Vec3::new(0.0, 0.20, 0.55),
});

/// Dark companion bg (matches `portal_spiral`'s bg palette).
static TUNNEL_BG: LazyLock<Palette> = LazyLock::new(|| Palette {
    a: Vec3::new(0.07, 0.04, 0.06),
    b: Vec3::new(0.04, 0.02, 0.04),
    c: Vec3::new(1.0, 1.0, 1.0),
    d: Vec3::new(0.0, 0.1, 0.2),
});

/// Minimum bg brightness fraction at a ring trough — keeps the tube wall visible
/// even where the height field bottoms out. Mirrors the material's bg floor.
const TUNNEL_BG_FLOOR: f32 = 0.25;
/// Near/far depth band the ring phase maps into. Near rings (`ring_phase ≈ 0`)
/// read close; far rings (`≈ 1`) read deep. Arbitrary monotone band — there is no
/// real geometry, this is only a depth CUE for the present layer's dirty-diff.
const TUNNEL_DEPTH_NEAR: f32 = 1.0;
const TUNNEL_DEPTH_SPAN: f32 = 8.0;

/// The fixed shape + motion of one tunnel dive. `speed` scales the dive rate,
/// `twist` curls the arms into a log-spiral, `octave_warp` adds fBm grain, `arms`
/// sets the angular repetition (`0` = pure concentric rings).
#[derive(Clone, Copy, Debug)]
pub struct TunnelParams {
    /// Scales the dive rate.
    pub speed: f32,
    /// Curls the arms into a log-spiral.
    pub twist: f32,
    /// Adds fBm grain.
    pub octave_warp: f32,
    /// Angular repetition (`0` = pure concentric rings).
    pub arms: u8,
}

impl Default for TunnelParams {
    /// A curling Droste dive (matches the `portal_spiral` look) — the default the
    /// game's `Diving` transition plays.
    fn default() -> Self {
        TunnelParams {
            speed: 1.0,
            twist: 0.9,
            octave_warp: 0.5,
            arms: 5,
        }
    }
}

/// Map a height/density scalar `[0,1]` to a body-glyph from the canonical ramp,
/// exactly as `Material::sample_fractal` does (densest glyph at `height = 1`).
#[inline]
fn ramp_glyph(brightness: f32) -> u32 {
    let ramp = body_ramp();
    let n = ramp.len();
    let idx = ((1.0 - brightness.clamp(0.0, 1.0)) * (n as f32 - 1.0) + 0.5) as usize;
    u32::from(ramp[idx.min(n - 1)])
}

/// Fill the WHOLE framebuffer with the log-polar warp, centred on the screen,
/// bypassing the raycaster (there is no geometry). `t` is the dive's elapsed time
/// (seconds); `params` fix the shape + speed. Writes `{glyph, fg, bg, depth}` (and
/// a flat surface tag) into every cell.
///
/// Determinism: a pure function of `(t, params, framebuffer size)`. At `t = 0`
/// with `arms = 0, twist = 0` the field is purely radial, so the frame is
/// radially symmetric about the centre (proven by the tests).
pub fn render_tunnel(fb: &mut Framebuffer, t: f32, params: &TunnelParams) {
    let perlin: &Perlin = &TUNNEL_PERLIN;
    let fg_pal: &Palette = &TUNNEL_FG;
    let bg_pal: &Palette = &TUNNEL_BG;

    let p = FractalParams {
        arms: params.arms,
        octave_warp: params.octave_warp,
        twist: params.twist,
    };

    // SpinZoom-style coupled phase: over one LN2 of zoom the spin advances a whole
    // TAU AND the field's `twist*log_radius` per-octave drift is cancelled, so both
    // seams coincide and the dive loops without a visible jump for ANY twist. The
    // coupling lives in one place — `spinzoom_phase`.
    let phase = crate::fractal::spinzoom_phase(t * params.speed * LN2, params.twist);

    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let inv_w = 1.0 / cols as f32;
    let inv_h = 1.0 / rows as f32;

    for y in 0..rows {
        let v = (y as f32 + 0.5) * inv_h;
        for x in 0..cols {
            let u = (x as f32 + 0.5) * inv_w;
            let s = log_polar_field(u, v, phase, &p, perlin);

            // Ring-phase hue, tinted by the arm index — same shaping as the portal
            // material so the tunnel and the wall portal share a colour family.
            let hue = fg_pal.eval(s.ring_phase + 0.15 * s.angle_idx);
            // Dark companion bg, deepened toward the ring troughs for contrast.
            let bg = bg_pal.eval(s.ring_phase * 0.5)
                * (TUNNEL_BG_FLOOR + (1.0 - TUNNEL_BG_FLOOR) * s.height);

            let cell = Cell {
                glyph: ramp_glyph(s.height),
                fg: pack_rgb(hue.clamp(Vec3::ZERO, Vec3::ONE)),
                bg: pack_rgb(bg.clamp(Vec3::ZERO, Vec3::ONE)),
                // Depth cue only (no real geometry): near rings read close.
                depth: TUNNEL_DEPTH_NEAR + s.ring_phase * TUNNEL_DEPTH_SPAN,
                surface: SurfaceKind::WallSolid,
                ..Cell::default()
            };
            fb.set(x, y, cell);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::Framebuffer;

    #[test]
    fn tunnel_fills_every_cell() {
        let mut fb = Framebuffer::new(60, 24);
        render_tunnel(
            &mut fb,
            0.0,
            &TunnelParams {
                speed: 1.0,
                twist: 0.6,
                octave_warp: 0.5,
                arms: 0,
            },
        );
        // Every cell is written ⇒ a finite (non-default) depth, no holes. (A
        // glyph check is wrong here: the ramp's trailing spaces are valid INKED
        // results for ring troughs, so "blank" must be measured by the depth
        // having been set, not by the glyph.)
        let holes = fb.cells().iter().filter(|c| !c.depth.is_finite()).count();
        assert_eq!(holes, 0, "tunnel must fill the whole framebuffer");
    }

    #[test]
    fn tunnel_is_radially_symmetric_at_t0() {
        let mut fb = Framebuffer::new(61, 61);
        render_tunnel(
            &mut fb,
            0.0,
            &TunnelParams {
                speed: 1.0,
                twist: 0.0,
                octave_warp: 0.0,
                arms: 0,
            },
        );
        // arms=0, twist=0, t=0 ⇒ the field depends only on radius, so a cell and
        // its mirror about the centre column (x=30) read the same glyph.
        let g = |x: usize, y: usize| fb.get(x, y).unwrap().glyph;
        assert_eq!(g(40, 30), g(20, 30), "mirror across vertical centre");
        // And mirror about the centre row (y=30).
        assert_eq!(g(30, 40), g(30, 20), "mirror across horizontal centre");
    }

    /// Largest per-channel fg-colour delta between two same-size frames, as a
    /// fraction of full range `[0,1]`. A CONTINUITY metric: the seamless dive must
    /// have a tiny delta across the loop wrap, a large one across a half-period.
    /// (Packed-RGB byte-equality is the wrong tool — a perfectly continuous field
    /// sampled `1e-6` s apart still differs in the colour LSBs; the seamlessness
    /// proof at exact `1e-3` tolerance lives in `fractal::tests`. Here we assert the
    /// player-visible MAGNITUDE of the wrap discontinuity.)
    fn max_fg_delta(a: &Framebuffer, b: &Framebuffer) -> f32 {
        assert_eq!(a.cells().len(), b.cells().len());
        let chan = |p: u32, sh: u32| ((p >> sh) & 0xFF) as f32 / 255.0;
        let mut m = 0.0f32;
        for (ca, cb) in a.cells().iter().zip(b.cells().iter()) {
            for sh in [16, 8, 0] {
                m = m.max((chan(ca.fg, sh) - chan(cb.fg, sh)).abs());
            }
        }
        m
    }

    #[test]
    fn tunnel_dive_loops_seamlessly_at_shipped_twist() {
        // Regression guard: the default dive ships twist=0.9. One loop period is the
        // time `t*speed*LN2` takes to advance a whole LN2 (t = 1/speed). The seam the
        // player sees is the wrap point: the frame just BEFORE the wrap (t→period⁻)
        // must be CONTINUOUS with the frame at t=0. `spinzoom_phase` cancels the
        // `twist*log_radius` drift, so the field is continuous there; the old coupling
        // jumped ~half an arm-period — a large, visible colour hitch.
        //
        // We measure the player-visible MAGNITUDE of the wrap discontinuity (max fg
        // channel delta) rather than packed-byte equality: at `period - 1e-6` even a
        // perfectly seamless field shifts the colour LSBs, so exact equality could
        // never hold. The broken coupling produced a LARGE delta here (verified ≫0.1);
        // the seamless coupling keeps it near machine epsilon.
        let params = TunnelParams::default(); // speed=1.0, twist=0.9, arms=5
        let period = 1.0 / params.speed;
        let near_wrap = period - 1e-6;
        let mut start = Framebuffer::new(48, 24);
        let mut end = Framebuffer::new(48, 24);
        render_tunnel(&mut start, 0.0, &params);
        render_tunnel(&mut end, near_wrap, &params);
        let wrap_delta = max_fg_delta(&start, &end);
        assert!(
            wrap_delta < 0.02,
            "tunnel dive must loop seamlessly at twist=0.9 across the wrap; \
             max fg delta = {wrap_delta} (broken coupling would jump ≫0.1)"
        );
        // Non-vacuous: the half-period frame must DIFFER substantially from the seam,
        // so the continuity check above can't pass on a constant field.
        let mut mid = Framebuffer::new(48, 24);
        render_tunnel(&mut mid, period / 2.0, &params);
        let mid_delta = max_fg_delta(&start, &mid);
        assert!(
            mid_delta > 0.1,
            "tunnel field barely moves across a half-period (delta={mid_delta}) — \
             the seamless-loop check would pass trivially"
        );
    }

    #[test]
    fn tunnel_is_deterministic() {
        let params = TunnelParams::default();
        let mut a = Framebuffer::new(48, 20);
        let mut b = Framebuffer::new(48, 20);
        render_tunnel(&mut a, 0.7, &params);
        render_tunnel(&mut b, 0.7, &params);
        assert_eq!(a.cells(), b.cells(), "same (t, params) ⇒ identical frame");
    }
}
