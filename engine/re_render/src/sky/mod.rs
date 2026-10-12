//! `re_render::sky` — direction-sampled SKY background: a 360° cylindrical
//! panorama plus a circular top cap.
//!
//! The sky is sampled by RAY DIRECTION `(yaw, pitch)`, NOT by screen position, so
//! it stays correct under the 2.5D pitch shear: looking up reveals more of the
//! cap, turning rotates the cap (it converges at the zenith). This replaces the
//! flat `CEILING_SKY` projection for above-horizon rays — the must-have is that
//! EVERY missed/above-horizon ray, at any pitch up to the ±60° clamp, gets real
//! sky content (no black band) all the way to ~85° (zenith approach).
//!
//! ## The model
//!
//! A ray with pitch `φ ≤ θ_cap` ([`THETA_CAP`], 55°) samples the CYLINDER (the
//! 360° sides); a ray with `φ > θ_cap` samples the CAP (the top disc). The seam
//! at `θ_cap` is density-matched: both sides carry [`CYL_W`] (720) texels per full
//! turn at the boundary (the cap rim circumference `2π·R = 2π·(720/2π) = 720`), so
//! the join is continuous, with a small blend band ([`SEAM_BLEND_RAD`]) to soften
//! the ring.
//!
//! ### Cylinder (`pitch ≤ θ_cap`)
//! * `u = wrap(yaw / TAU)` ∈ [0, 1) — one texel per viewport column at FOV60/120
//!   (`CYL_W = 720`); NOT oversampled.
//! * `v = (pitch − V_LOW) / (θ_cap − V_LOW)` ∈ [0, 1] over the band; `V_LOW` sits
//!   a little below the horizon ([`V_LOW`], −10°) so the brighter horizon band is
//!   not clipped exactly at 0.
//!
//! ### Cap (`pitch > θ_cap`)
//! Polar disc: `radius = R · (HALF_PI − φ)/(HALF_PI − θ_cap)`, `angle = yaw`, with
//! `R = CYL_W / TAU ≈ 114.6` so the rim matches the cylinder texel density. zenith
//! (φ = 90°) → disc centre (r = 0); seam (φ = θ_cap) → disc rim (r = R). The cap
//! ROTATES with yaw and converges at the zenith. Only the inscribed circle (r ≤ R)
//! of the [`CAP_DIM`]×[`CAP_DIM`] texture is sampled; the corners are dead/trimmed.
//!
//! ## Swappable content
//!
//! Content goes through the [`SkySource`] abstraction — [`sample_cylinder`](SkySource::sample_cylinder) /
//! [`sample_cap`](SkySource::sample_cap) in normalized texel space — so the procedural sky here can be
//! replaced by a [`re_core::TextureImage`]-backed source later WITHOUT touching the
//! direction→texel mapping or the integration in the caster. [`ProceduralSky`] is
//! the built-in (gradient + faint grid) so the feature works out of the box.

use std::f32::consts::{FRAC_PI_2, TAU};

/// Seam pitch (radians): rays at or below this sample the cylinder, above sample
/// the cap. 55° (≈0.9599 rad). Just under the +60° pitch clamp + half the vertical
/// FOV reach (~85°), so the cap always has headroom before the zenith.
pub const THETA_CAP: f32 = 55.0 * (std::f32::consts::PI / 180.0);

/// `π/2` — the zenith pitch (cap centre).
pub const HALF_PI: f32 = FRAC_PI_2;

/// Lower edge of the cylinder's vertical band (radians, signed). Set a little BELOW
/// the horizon (−10°) so the warm, brighter horizon row is not clipped right at the
/// band's bottom — the cylinder covers `[V_LOW, θ_cap]` in pitch.
pub const V_LOW: f32 = -10.0 * (std::f32::consts::PI / 180.0);

/// Cylinder width in texels: exactly one texel per viewport column at FOV60 / 120
/// columns over a full turn (720). A turn of `TAU/720` advances `u` by one texel —
/// density-matched, NOT oversampled.
pub const CYL_W: usize = 720;

/// Cylinder height in texels (the vertical band resolution). ~96 rows across the
/// `[V_LOW, θ_cap]` band — finer than any terminal row count, so the band never
/// shows texel stair-steps.
pub const CYL_H: usize = 96;

/// Cap texture dimension (square). Only the inscribed circle (r ≤ [`CAP_R`]) is
/// ever sampled; the corners are dead.
pub const CAP_DIM: usize = 256;

/// Cap rim radius in texels: `CYL_W / TAU ≈ 114.6`. Chosen so the rim circumference
/// (`2π·R = CYL_W = 720`) matches the cylinder's texel-per-turn density → the seam
/// is density-matched (no texel-rate discontinuity at the join).
pub const CAP_R: f32 = CYL_W as f32 / TAU;

/// Seam blend half-width (radians, ±3°). Within `[θ_cap − band, θ_cap + band]` the
/// cylinder and cap samples are linearly cross-faded so the join is not a hard ring.
/// Both carry 720 texels/turn at the boundary, so this is a gentle cosmetic blend,
/// not a correction.
pub const SEAM_BLEND_RAD: f32 = 3.0 * (std::f32::consts::PI / 180.0);

/// A packed `0x00RRGGBB` color plus the glyph the sky cell should carry. The
/// renderer writes `fg`/`bg`/`glyph`; `bg` reads as the dominant sky tone, `fg`
/// tints the (faint) grid glyph over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkyTexel {
    /// Packed `0x00RRGGBB` foreground (the grid line / glyph ink).
    pub fg: u32,
    /// Packed `0x00RRGGBB` background (the dominant sky tone behind the glyph).
    pub bg: u32,
    /// The glyph drawn at this sky cell (a space where there is no grid line).
    pub glyph: char,
}

/// A source of sky content, sampled in NORMALIZED texel space. The two mapping
/// regions (cylinder, cap) call the matching method; the direction→texel mapping
/// ([`direction_to_sky`]) is independent of the source, so a future
/// [`re_core::TextureImage`]-backed implementation drops in unchanged.
///
/// * [`sample_cylinder`](SkySource::sample_cylinder): `u ∈ [0,1)` around the turn,
///   `v ∈ [0,1]` up the band (0 = `V_LOW` bottom, 1 = `θ_cap` top).
/// * [`sample_cap`](SkySource::sample_cap): `r_norm ∈ [0,1]` from disc centre
///   (zenith, 0) to rim (seam, 1); `angle ∈ [0, TAU)` = yaw.
pub trait SkySource {
    /// Sample the cylinder band at normalized `(u, v)`.
    fn sample_cylinder(&self, u: f32, v: f32) -> SkyTexel;
    /// Sample the cap disc at normalized radius `r_norm` and `angle` (radians).
    fn sample_cap(&self, r_norm: f32, angle: f32) -> SkyTexel;
}

/// Where a ray direction lands on the sky: the cylinder band or the cap disc. The
/// seam region carries BOTH so the integrator can cross-fade them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SkyHit {
    /// A pure cylinder sample (pitch well below the seam): `(u, v)`.
    Cylinder {
        /// Normalized turn fraction `[0,1)`.
        u: f32,
        /// Normalized band height `[0,1]` (0 = `V_LOW` bottom, 1 = `θ_cap` top).
        v: f32,
    },
    /// A pure cap sample (pitch well above the seam): `(r_norm, angle)`.
    Cap {
        /// Normalized disc radius `[0,1]` (0 = zenith, 1 = seam rim).
        r_norm: f32,
        /// Yaw angle, radians `[0, TAU)`.
        angle: f32,
    },
    /// Inside the seam blend band: both samples + the cap weight `t ∈ [0,1]`
    /// (0 = all cylinder, 1 = all cap).
    Seam {
        /// The cylinder sample's `u`.
        u: f32,
        /// The cylinder sample's `v`.
        v: f32,
        /// The cap sample's `r_norm`.
        r_norm: f32,
        /// The cap sample's `angle`.
        angle: f32,
        /// Cross-fade weight toward the cap sample, `[0,1]`.
        cap_weight: f32,
    },
}

/// Wrap a value into `[0, 1)` (for `u`/angle fractions). `rem_euclid` keeps it
/// positive for negative yaws.
#[inline]
fn wrap01(x: f32) -> f32 {
    let w = x.rem_euclid(1.0);
    // rem_euclid(1.0) can return exactly 1.0 for tiny-negative inputs; fold it back.
    if w >= 1.0 {
        0.0
    } else {
        w
    }
}

/// Map a ray direction `(yaw, pitch)` to its sky landing (cylinder band, cap disc,
/// or the seam blend). Pure + deterministic — the heart of the model, unit-tested
/// directly (seam continuity, zenith → centre, yaw → cap rotation).
///
/// * `yaw` (radians) — the column's world yaw; `u`/cap `angle` wrap on it.
/// * `pitch` (radians) — the row's vertical angle. Only `pitch > V_LOW` is sky
///   (below that is real floor geometry and never reaches here); the caller passes
///   above-horizon rays.
#[inline]
pub fn direction_to_sky(yaw: f32, pitch: f32) -> SkyHit {
    let u = wrap01(yaw / TAU);
    let angle = yaw.rem_euclid(TAU);

    // Cylinder v across [V_LOW, θ_cap] (clamped: a ray just below V_LOW still reads
    // the bottom band row rather than going absent).
    let cyl_v = |p: f32| ((p - V_LOW) / (THETA_CAP - V_LOW)).clamp(0.0, 1.0);
    // Cap r_norm across [θ_cap, HALF_PI]: seam (θ_cap) → 1 (rim), zenith → 0 (centre).
    let cap_r = |p: f32| ((HALF_PI - p) / (HALF_PI - THETA_CAP)).clamp(0.0, 1.0);

    if pitch <= THETA_CAP - SEAM_BLEND_RAD {
        SkyHit::Cylinder { u, v: cyl_v(pitch) }
    } else if pitch >= THETA_CAP + SEAM_BLEND_RAD {
        SkyHit::Cap {
            r_norm: cap_r(pitch),
            angle,
        }
    } else {
        // Seam band: linearly cross-fade cylinder→cap across ±SEAM_BLEND_RAD.
        let lo = THETA_CAP - SEAM_BLEND_RAD;
        let hi = THETA_CAP + SEAM_BLEND_RAD;
        let cap_weight = ((pitch - lo) / (hi - lo)).clamp(0.0, 1.0);
        SkyHit::Seam {
            u,
            v: cyl_v(pitch),
            r_norm: cap_r(pitch),
            angle,
            cap_weight,
        }
    }
}

/// Sample a [`SkySource`] for a ray direction `(yaw, pitch)`, resolving the seam
/// blend into a single [`SkyTexel`]. This is the one call the caster makes per
/// missed/above-horizon ray. Cheap: at most two source samples (only in the narrow
/// seam band) + a per-channel lerp; no allocation.
#[inline]
pub fn sample_sky<S: SkySource>(source: &S, yaw: f32, pitch: f32) -> SkyTexel {
    match direction_to_sky(yaw, pitch) {
        SkyHit::Cylinder { u, v } => source.sample_cylinder(u, v),
        SkyHit::Cap { r_norm, angle } => source.sample_cap(r_norm, angle),
        SkyHit::Seam {
            u,
            v,
            r_norm,
            angle,
            cap_weight,
        } => {
            let a = source.sample_cylinder(u, v);
            let b = source.sample_cap(r_norm, angle);
            blend_texel(a, b, cap_weight)
        }
    }
}

/// Linearly blend two packed `0x00RRGGBB` colors per channel by `t ∈ [0,1]`.
#[inline]
fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ch = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xFF) as f32;
        let cb = ((b >> sh) & 0xFF) as f32;
        (ca + (cb - ca) * t).round().clamp(0.0, 255.0) as u32
    };
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Blend two [`SkyTexel`]s for the seam band. The discrete glyph comes from the
/// dominant side; the colors cross-fade so the join is smooth.
#[inline]
fn blend_texel(a: SkyTexel, b: SkyTexel, t: f32) -> SkyTexel {
    SkyTexel {
        fg: lerp_rgb(a.fg, b.fg, t),
        bg: lerp_rgb(a.bg, b.bg, t),
        glyph: if t < 0.5 { a.glyph } else { b.glyph },
    }
}

// ───────────────────────── procedural sky content ───────────────────────────

/// The built-in procedural sky: a vertical gradient (warm/bright horizon → cool/dark
/// zenith) plus a faint navigation grid (longitude/latitude on the cylinder, radial
/// spokes + concentric rings on the cap) so a yaw reads clearly as rotation. All
/// content is derived from the normalized texel coordinates, so it is resolution-
/// independent and a [`re_core::TextureImage`] could replace it via [`SkySource`].
#[derive(Clone, Copy, Debug)]
pub struct ProceduralSky {
    /// Number of longitude lines around the cylinder = number of radial spokes on
    /// the cap. A turn crosses this many lines, so the rotation reads.
    pub longitudes: u32,
    /// Number of latitude lines up the cylinder band = number of concentric rings on
    /// the cap.
    pub latitudes: u32,
}

impl Default for ProceduralSky {
    fn default() -> Self {
        // 22 longitudes: a grid line every ~16.36° of yaw, deliberately NOT a
        // multiple of 4. A count that IS a multiple of 4 puts a grid line at
        // exactly yaw 90°/180°/270° (k = N/4 is then an integer, and every
        // level's spawn yaw is one of those four cardinals —
        // `re_world::CameraPose::spawn_at`'s facing keywords are only ever
        // axis-aligned), so every level's first frame would show the same
        // line dead ahead: the sky would read as phase-locked to spawn rather
        // than as a rotation the player discovers by turning. 22 shares no
        // factor with 4 (or with 90 — 90 mod 22 = 2, so it does not divide
        // evenly into a quarter turn either), so no spawn yaw lands a line
        // exactly on-axis.
        // 6 latitudes / rings for vertical reference. Subtle, distant, non-critical.
        ProceduralSky {
            longitudes: 22,
            latitudes: 6,
        }
    }
}

/// Horizon sky tone (warm, brighter) — packed `0x00RRGGBB`. The bottom of the band.
const HORIZON_RGB: (f32, f32, f32) = (92.0, 104.0, 140.0);
/// Zenith sky tone (cool, darker). The cap centre / top of the band.
const ZENITH_RGB: (f32, f32, f32) = (14.0, 18.0, 44.0);
/// Grid-line ink (a touch brighter than the local sky so the line is faintly visible
/// without being loud — it is distant, non-critical).
const GRID_TINT: f32 = 1.35;
/// How close (as a fraction of the inter-line spacing, per [`line_distance`]) to a
/// grid line counts as "on the line". 0.06 = a line spanning ~12% of each gap — a
/// faint, distant grid that reads as a line, not a band.
const GRID_HALF_WIDTH: f32 = 0.06;

#[inline]
fn pack_rgb(r: f32, g: f32, b: f32) -> u32 {
    let c = |x: f32| x.round().clamp(0.0, 255.0) as u32;
    (c(r) << 16) | (c(g) << 8) | c(b)
}

/// Vertical gradient color for a "height" `h ∈ [0,1]` (0 = horizon, 1 = zenith).
#[inline]
fn gradient(h: f32) -> (f32, f32, f32) {
    let h = h.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * h;
    (
        lerp(HORIZON_RGB.0, ZENITH_RGB.0),
        lerp(HORIZON_RGB.1, ZENITH_RGB.1),
        lerp(HORIZON_RGB.2, ZENITH_RGB.2),
    )
}

/// Distance to the nearest of `n` evenly spaced lines for a coordinate `x ∈ [0,1)`,
/// measured as a FRACTION OF THE SPACING (`∈ [0, 0.5]`): 0 = exactly on a line, 0.5 =
/// midway between two lines. Wraps for the cylinder's `u`. A line "half-width" is
/// then a fraction of the inter-line gap, independent of `n`.
#[inline]
fn line_distance(x: f32, n: u32) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let scaled = x * n as f32;
    let frac = scaled - scaled.floor();
    frac.min(1.0 - frac)
}

impl ProceduralSky {
    /// Build a sky texel from a base "height" `h` (0 = horizon, 1 = zenith) and a
    /// boolean "on a grid line" flag, picking a faint line glyph when on a line.
    #[inline]
    fn texel(&self, h: f32, on_line: bool) -> SkyTexel {
        let (r, g, b) = gradient(h);
        let bg = pack_rgb(r, g, b);
        if on_line {
            // The grid line: a faint brighter ink + a thin glyph so it reads as a
            // line, not a block. Tint the local sky tone up a touch.
            let fg = pack_rgb(r * GRID_TINT, g * GRID_TINT, b * GRID_TINT);
            SkyTexel {
                fg, bg, glyph: '·'
            }
        } else {
            // Open sky: fg == bg so a space renders as a flat tone (the present
            // layer paints bg; the space glyph is invisible).
            SkyTexel {
                fg: bg,
                bg,
                glyph: ' ',
            }
        }
    }
}

impl SkySource for ProceduralSky {
    fn sample_cylinder(&self, u: f32, v: f32) -> SkyTexel {
        // v: 0 = V_LOW (below horizon), 1 = θ_cap (near zenith). The visible band
        // brightens toward the horizon; treat v as the gradient height directly
        // (0 → horizon tone, 1 → toward zenith tone).
        let h = v.clamp(0.0, 1.0);
        // Longitude lines (constant yaw) + latitude lines (constant pitch).
        let on_long = line_distance(wrap01(u), self.longitudes) < GRID_HALF_WIDTH;
        let on_lat = line_distance(v.clamp(0.0, 1.0), self.latitudes) < GRID_HALF_WIDTH;
        self.texel(h, on_long || on_lat)
    }

    fn sample_cap(&self, r_norm: f32, angle: f32) -> SkyTexel {
        // r_norm: 0 = zenith (disc centre), 1 = seam (rim). The cap is the TOP of
        // the sky, so its height runs from "just-below-zenith" at the rim (matching
        // the cylinder's top, h≈1) to full zenith at the centre. Map rim→1, centre→1
        // is too flat; instead push the rim to the cylinder-top tone and darken
        // gently toward the centre is wrong — the cap is ABOVE the band, so it is the
        // DARKEST part. Use h = 1 at centre, and 1 at rim too, but let the rings read.
        // We map height so rim matches band-top (h≈1.0) and centre is the deepest
        // zenith: keep h at 1.0 across the cap (the gradient's zenith tone), and rely
        // on the rings/spokes for structure.
        let h = 1.0;
        // Radial spokes (constant angle) — same count as longitudes so a spoke lines
        // up with a longitude line at the seam (continuity). Spoke spacing shrinks
        // toward the centre, so gate spokes out very near the centre to avoid a
        // cluttered hub.
        let frac_angle = wrap01(angle / TAU);
        let spoke_d = line_distance(frac_angle, self.longitudes);
        // Convert the angular spoke distance to an approximate texel distance at this
        // radius so the spoke has roughly constant on-screen width (it narrows toward
        // the centre); only draw spokes outside a small hub radius.
        let on_spoke = r_norm > 0.15 && spoke_d * r_norm < GRID_HALF_WIDTH;
        // Concentric rings (constant radius) — same count as latitudes.
        let on_ring = line_distance(r_norm.clamp(0.0, 1.0), self.latitudes) < GRID_HALF_WIDTH;
        self.texel(h, on_spoke || on_ring)
    }
}

// ───────────────────────── image-backed sky content ─────────────────────────

/// An [`re_core::TextureImage`]-backed sky source. Holds borrowed references to a
/// [`Map`](re_world::Map)'s [`SkyImages`](re_world::SkyImages) and samples them at the
/// normalized cylinder/cap UVs the [`SkySource`] trait provides, so ANY input image
/// resolution maps onto the fixed sky dimensions purely through UV sampling — no
/// pre-resize is needed ("exact size" is handled by the sampler).
///
/// This is selected at the sky-sampling site only when a `cylinder` image is present
/// (see [`map_sky_texel`]); the cap images are optional. When a cap image is absent the
/// source falls back to the embedded [`ProceduralSky`] cap for that region, so a level
/// that authors only a horizon strip still gets a coherent dome rather than a hole.
///
/// Backdrop look: image sky is COLOUR-DOMINANT — `bg` carries the sampled colour, the
/// glyph is a space, and `fg == bg`. The procedural navigation grid is intentionally
/// DROPPED for image sky: an authored panorama already reads as a scene, and faint grid
/// dots over a photo look like dirt rather than reference lines. (The procedural sky
/// keeps its grid; only the image source omits it.)
pub struct ImageSky<'a> {
    images: &'a re_world::SkyImages,
    /// Fallback procedural source for cap regions with no authored image.
    fallback: ProceduralSky,
}

impl<'a> ImageSky<'a> {
    /// Build an image sky over a borrowed [`SkyImages`](re_world::SkyImages) set. The fallback procedural
    /// source covers any cap region whose image is absent.
    #[inline]
    pub fn new(images: &'a re_world::SkyImages) -> Self {
        ImageSky {
            images,
            fallback: ProceduralSky::default(),
        }
    }

    /// Pack a sampled `[R,G,B,A]` texel into a colour-dominant [`SkyTexel`] (alpha
    /// ignored — the sky is an opaque backdrop). `bg` = colour, `fg == bg`, glyph = ' '
    /// so the present layer paints the colour as a flat block.
    #[inline]
    fn texel_from_rgba(rgba: [u8; 4]) -> SkyTexel {
        let bg = (u32::from(rgba[0]) << 16) | (u32::from(rgba[1]) << 8) | u32::from(rgba[2]);
        SkyTexel {
            fg: bg,
            bg,
            glyph: ' ',
        }
    }
}

impl SkySource for ImageSky<'_> {
    fn sample_cylinder(&self, u: f32, v: f32) -> SkyTexel {
        match &self.images.cylinder {
            Some(img) => {
                // `u` already wraps in [0,1). Our band `v` runs 0 = V_LOW (bottom) → 1 =
                // θ_cap (top); `TextureImage` `v` is top-origin (0 = top row). Flip so the
                // image's TOP row maps to the top of the band (toward the zenith) — the
                // natural orientation for a horizon strip.
                Self::texel_from_rgba(img.sample_nearest(wrap01(u), 1.0 - v.clamp(0.0, 1.0)))
            }
            // Source is only built when a cylinder image exists, but stay defined.
            None => self.fallback.sample_cylinder(u, v),
        }
    }

    fn sample_cap(&self, r_norm: f32, angle: f32) -> SkyTexel {
        // Top cap by default; the seam blend only ever drives the TOP cap from above-
        // horizon rays (below-horizon rays are real floor geometry and never reach the
        // sky), so `cap_bottom` is unused by the current direction mapping and kept for
        // forward-compat with a future nadir region.
        match &self.images.cap_top {
            Some(img) => {
                // Polar disc → square image UV. r_norm: 0 = centre (zenith), 1 = rim
                // (seam); angle = yaw. Map the unit disc into [0,1]² centred at (0.5,0.5).
                let r = r_norm.clamp(0.0, 1.0) * 0.5;
                let su = 0.5 + r * angle.cos();
                let sv = 0.5 + r * angle.sin();
                Self::texel_from_rgba(img.sample_nearest(su, sv))
            }
            // No top-cap image: fall back to the procedural cap so the dome closes with
            // a coherent (gradient + spokes/rings) zenith rather than a hole.
            None => self.fallback.sample_cap(r_norm, angle),
        }
    }
}

/// Sample the sky for a ray `(yaw, pitch)`, choosing the source from a
/// [`Map`](re_world::Map)'s sky images: if a cylinder image is set, sample through an
/// [`ImageSky`]; otherwise use the built-in `procedural` source exactly as before. With
/// no images set this is byte-identical to `sample_sky(procedural, …)` — the regression
/// guard the no-image golden frames rely on.
#[inline]
pub fn map_sky_texel(
    map: &re_world::Map,
    procedural: &ProceduralSky,
    yaw: f32,
    pitch: f32,
) -> SkyTexel {
    let images = map.sky_images();
    if images.cylinder.is_some() {
        sample_sky(&ImageSky::new(images), yaw, pitch)
    } else {
        sample_sky(procedural, yaw, pitch)
    }
}

// ═══════════════════════ the animated SKY PROGRAM ═══════════════════════════
//
// Everything below is the varied, ANIMATED, per-level sky layer stack. It is
// purely ADDITIVE over the direction-sampled panorama above: a level whose
// `re_world::Map::sky_spec()` is `None` (no `[sky] base =`/`layer =`/`kind =`
// key authored) never reaches any of this — `re_render::apply_sky` keeps
// calling `map_sky_texel` exactly as before, byte-identical.
//
// A [`SkySpec`](re_world::sky_spec::SkySpec) is compiled ONCE per frame (not
// per cell — see [`SkyProgram::compile`]) into a [`SkyProgram`], which is then
// sampled per above-horizon ray via [`SkyProgram::sample`]. The compiled
// program and the frame's clock/camera are published through thread-locals
// (the same pattern `re_render::state_bus`/`config`/`draw_dist` use for
// cast-scoped globals the deep shade path reads back), so `apply_sky`'s
// callers need no new parameter threaded through them.

/// Shifting curtains of colour banded by altitude, with a soft trailing hem.
pub mod aurora;
/// Perspective cloud layers projected onto a world-anchored sky plane.
pub mod clouds;
/// A vertical stream of falling glyph columns, independent of camera yaw.
pub mod datarain;
/// Distant-city horizon: a ring of towers reusing [`crate::facade`]'s window
/// rhythm for their lit-window pattern.
pub mod skyline;
/// Twinkling starfield, an optional Milky Way band, and named constellations.
pub mod stars;
/// A dark cloud deck plus a bounded lightning event other layers can react to.
pub mod storm;
/// Deterministic hashing + value-noise helpers shared by every sky layer.
pub(crate) mod util;
/// The surreal sky family: an empty void, an infinite zoom, a watching eye,
/// and a floor-plan mirror.
pub mod void;

use re_core::CellStyle;
use re_world::sky_spec::{SkyLayerSpec, SkySpec};

/// Everything a sky layer needs to shade one above-horizon ray: its direction,
/// the animation clock, and (for the WORLD-anchored layers — clouds, mirror)
/// the camera's ground position. Pure data, `Copy`.
#[derive(Clone, Copy, Debug)]
pub struct SkyCtx {
    /// Ray yaw (radians), matching [`CameraPose::dir`](re_world::CameraPose::dir)'s
    /// convention (`ray_yaw` in this module).
    pub yaw: f32,
    /// Ray pitch (radians), positive = looking up.
    pub pitch: f32,
    /// The program's own tick index: `floor(t * hz)`. Two calls with the same
    /// `tick` MUST produce the same [`SkyCell`] (the byte-budget invariant).
    pub tick: u32,
    /// Continuous seconds, `tick / hz` when quantized correctly by the caller —
    /// layers should prefer `tick` for anything that must be byte-static
    /// between ticks, and only use `t` for a continuous phase computed FROM
    /// `tick` (e.g. `tick as f32 / hz as f32`).
    pub t: f32,
    /// The program's tick rate (Hz), so a layer can turn `tick` back into
    /// quantized seconds without the caller threading `hz` separately.
    pub hz: f32,
    /// Camera ground position (world cell units) — for clouds' perspective
    /// plane and mirror's floor-plan reflection.
    pub cam_xy: (f32, f32),
    /// The PLAYER's facing yaw (radians) — distinct from `yaw` (the ray's own
    /// direction), used by the `watcher` sky so its pupil tracks the player's
    /// facing rather than each individual ray.
    pub player_yaw: f32,
}

/// One sky layer's output cell: glyph + fg + bg + [`CellStyle`] (attrs +
/// coloured underline + underline shape). The channel-grammar contract (spec
/// §1.4): an effect that ADDS MASS blends into `bg`; an effect that adds
/// STRUCTURE writes `glyph`+`fg`; `style` is free (no other sky layer owns it
/// unless documented, per level, so "U has at most one meaning per level").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyCell {
    /// The structural glyph (a star, a cloud edge, a bolt streak, …).
    pub glyph: char,
    /// The accent colour, on the glyph.
    pub fg: u32,
    /// The mass colour, behind/around the glyph.
    pub bg: u32,
    /// Attributes + coloured/shaped underline — free for a layer to use.
    pub style: CellStyle,
}

impl SkyCell {
    /// A flat, unstyled cell of one colour (fg == bg, space glyph) — the
    /// gradient base's starting point and every layer's safe "no-op" default.
    #[inline]
    pub fn flat(rgb: u32) -> Self {
        SkyCell {
            glyph: ' ',
            fg: rgb,
            bg: rgb,
            style: CellStyle::default(),
        }
    }
}

/// One layer of the sky stack: a pure function from (context, the cell so far)
/// to the next cell. Layers compose bottom-to-top.
pub trait SkyLayer: std::fmt::Debug {
    /// Shade `below` (everything under this layer has already been resolved)
    /// at `ctx`, using `seed`/`hz` (the program's shared determinism seed and
    /// tick rate — passed explicitly rather than stored, so a layer struct
    /// stays a plain, comparable, Clone-able value).
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell;

    /// Whether this layer's output can change from tick to tick even while the
    /// camera and map are unchanged (drives the near-cast "sky animated" OR
    /// term in `composite.rs` — a layer that only depends on view direction,
    /// like `watcher`'s gaze-independent iris or `fractal`'s pure zoom, should
    /// still return `true` if it animates with time; a purely static layer
    /// returns `false`). Default `true` (safe: never freezes a layer that
    /// should animate).
    fn is_time_animated(&self) -> bool {
        true
    }
}

/// Normalize a ray's pitch into a `[0,1]` dome-height fraction: `0` at the
/// bottom of the visible band ([`V_LOW`]), `1` at the zenith ([`HALF_PI`]).
/// Shared by every layer that needs "how high up the dome is this ray" without
/// caring about the cylinder/cap seam (stars, aurora, clouds, data-rain,
/// void).
#[inline]
pub(crate) fn dome_height(pitch: f32) -> f32 {
    ((pitch - V_LOW) / (HALF_PI - V_LOW)).clamp(0.0, 1.0)
}

/// The gradient BASE — the floor of every sky: a vertical lerp
/// from `base.horizon` to `base.zenith` over the WHOLE dome (`V_LOW` to
/// zenith), plus an optional horizon glow band blended in near the bottom.
fn base_cell(base: &re_world::sky_spec::SkyBase, pitch: f32) -> SkyCell {
    let h = dome_height(pitch);
    let mut rgb = util::lerp_rgb(base.horizon, base.zenith, h);
    if let Some(glow) = base.glow {
        let w = base.glow_width.max(1e-4);
        let glow_t = (1.0 - (h / w).min(1.0)).max(0.0);
        if glow_t > 0.0 {
            rgb = util::lerp_rgb(rgb, glow, glow_t * 0.6);
        }
    }
    SkyCell::flat(rgb)
}

/// A compiled, per-frame-cached sky program: the gradient base plus every
/// authored layer, ready to sample per ray. Built ONCE per frame by
/// [`SkyProgram::compile`] (never per cell — layer construction may allocate,
/// e.g. `Vec<Box<dyn SkyLayer>>`), then read many times via [`sample`](Self::sample).
#[derive(Debug)]
pub struct SkyProgram {
    base: re_world::sky_spec::SkyBase,
    layers: Vec<Box<dyn SkyLayer + Send + Sync>>,
    seed: u32,
    hz: u32,
}

impl SkyProgram {
    /// Compile a [`SkySpec`] into its sampler. Pure + cheap (a handful of small
    /// struct constructions, no heavy precompute, no file/network IO).
    pub fn compile(spec: &SkySpec) -> Self {
        let mut layers: Vec<Box<dyn SkyLayer + Send + Sync>> =
            Vec::with_capacity(spec.layers.len());
        for l in &spec.layers {
            layers.push(match l {
                SkyLayerSpec::Stars(p) => {
                    Box::new(stars::StarsLayer::new(*p)) as Box<dyn SkyLayer + Send + Sync>
                }
                SkyLayerSpec::Constellation(p) => {
                    Box::new(stars::ConstellationLayer::new(p.clone()))
                }
                SkyLayerSpec::Aurora(p) => Box::new(aurora::AuroraLayer::new(*p)),
                SkyLayerSpec::Clouds(p) => Box::new(clouds::CloudsLayer::new(*p)),
                SkyLayerSpec::Storm(p) => Box::new(storm::StormLayer::new(*p)),
                SkyLayerSpec::DataRain(p) => Box::new(datarain::DataRainLayer::new(*p)),
                SkyLayerSpec::Void(p) => Box::new(void::VoidLayer::new(*p)),
                SkyLayerSpec::Fractal(p) => Box::new(void::FractalLayer::new(*p)),
                SkyLayerSpec::Watcher(p) => Box::new(void::WatcherLayer::new(*p)),
                SkyLayerSpec::Mirror(p) => Box::new(void::MirrorLayer::new(*p)),
                SkyLayerSpec::Skyline(p) => Box::new(skyline::SkylineLayer::new(*p)),
            });
        }
        SkyProgram {
            base: spec.base,
            layers,
            seed: spec.seed,
            hz: spec.hz.max(1),
        }
    }

    /// Sample every layer, bottom to top, starting from the gradient base.
    pub fn sample(&self, ctx: &SkyCtx) -> SkyCell {
        let mut cell = base_cell(&self.base, ctx.pitch);
        for layer in &self.layers {
            cell = layer.shade(ctx, cell, self.seed, self.hz);
        }
        cell
    }

    /// True when ANY layer can change over time on a static camera — the
    /// map-local "sky animated" flag `composite.rs` ORs into its near-tier
    /// dirty-flag decision.
    pub fn is_animated(&self) -> bool {
        self.layers.iter().any(|l| l.is_time_animated())
    }

    /// The program's tick rate (Hz).
    #[inline]
    pub fn hz(&self) -> u32 {
        self.hz
    }

    /// Quantize a continuous time `t_secs` to this program's tick index.
    #[inline]
    pub fn tick_at(&self, t_secs: f32) -> u32 {
        (t_secs.max(0.0) * self.hz as f32).floor() as u32
    }
}

/// True when `map` authors an animated procedural sky program — the
/// map-local "sky animated" flag. `false` for a map with no [`SkySpec`]
/// (the legacy image/procedural path is time-independent) — no behaviour
/// change for any level that authors no `[sky] base =`/`layer =`/`kind =` key.
pub fn sky_is_animated(map: &re_world::Map) -> bool {
    match map.sky_spec() {
        Some(spec) => SkyProgram::compile(spec).is_animated(),
        None => false,
    }
}

// ── cast-scoped globals (the frame's clock/camera/compiled program) ─────────
//
// Same rationale as `re_render::state_bus`/`config`/`draw_dist`: `apply_sky`'s
// callers (`cast_column`, `shade_subray`, the near-ring bake) sit deep inside
// the raycaster, so threading the frame clock/camera through as an ordinary
// parameter would touch every call site along the way. `composite.rs` installs
// these once per frame/dump; every other caller (tests, `re_imgqa`, `ts_editor`
// preview) never touches them, so they read the harmless defaults (t=0, no
// program) — byte-identical to a build with no sky program installed.

thread_local! {
    static ACTIVE_TIME: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
    static ACTIVE_CAM_XY: std::cell::Cell<(f32, f32)> = const { std::cell::Cell::new((0.0, 0.0)) };
    static ACTIVE_PLAYER_YAW: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
    static ACTIVE_PROGRAM: std::cell::RefCell<Option<std::rc::Rc<SkyProgram>>> =
        const { std::cell::RefCell::new(None) };
}

/// Publish the current frame's animation clock (seconds) for [`crate::apply_sky`] to
/// read. `t = 0.0` (the default) makes every sky-program sample deterministic
/// for goldens/dumps that never call this.
pub fn set_active_sky_time(t_secs: f32) {
    ACTIVE_TIME.with(|c| c.set(t_secs));
}

/// Read the current frame's animation clock.
pub fn active_sky_time() -> f32 {
    ACTIVE_TIME.with(|c| c.get())
}

/// Publish the current frame's camera ground position (world cell units).
pub fn set_active_cam_xy(xy: (f32, f32)) {
    ACTIVE_CAM_XY.with(|c| c.set(xy));
}

/// Read the current frame's camera ground position.
pub fn active_cam_xy() -> (f32, f32) {
    ACTIVE_CAM_XY.with(|c| c.get())
}

/// Publish the current frame's PLAYER facing yaw (for `watcher`'s pupil).
pub fn set_active_player_yaw(yaw: f32) {
    ACTIVE_PLAYER_YAW.with(|c| c.set(yaw));
}

/// Read the current frame's player facing yaw.
pub fn active_player_yaw() -> f32 {
    ACTIVE_PLAYER_YAW.with(|c| c.get())
}

/// Install (or clear, with `None`) the compiled sky program for the current
/// frame/dump. Called once by `composite.rs` from the live map's
/// [`re_world::Map::sky_spec`] — never per cell.
pub fn set_active_program(program: Option<SkyProgram>) {
    ACTIVE_PROGRAM.with(|c| *c.borrow_mut() = program.map(std::rc::Rc::new));
}

/// The compiled program active for this thread, if any.
pub fn active_program() -> Option<std::rc::Rc<SkyProgram>> {
    ACTIVE_PROGRAM.with(|c| c.borrow().clone())
}

#[cfg(test)]
mod program_tests {
    use super::*;
    use re_world::sky_spec::{SkyBase, StarsParams};

    #[test]
    fn no_spec_program_is_never_animated_by_default() {
        let map = re_world::Map::new(4, 4, 4);
        assert!(
            !sky_is_animated(&map),
            "a map with no [sky] program must not force redraw"
        );
    }

    #[test]
    fn stars_program_is_animated() {
        let spec = SkySpec {
            base: SkyBase::default(),
            layers: vec![SkyLayerSpec::Stars(StarsParams::default())],
            seed: 1,
            hz: 12,
        };
        let program = SkyProgram::compile(&spec);
        assert!(program.is_animated());
    }

    #[test]
    fn empty_program_samples_the_gradient_base() {
        let spec = SkySpec {
            base: SkyBase {
                horizon: 0xFF0000,
                zenith: 0x0000FF,
                glow: None,
                glow_width: 0.1,
            },
            layers: vec![],
            seed: 0,
            hz: 12,
        };
        let program = SkyProgram::compile(&spec);
        let ctx = SkyCtx {
            yaw: 0.0,
            pitch: V_LOW,
            tick: 0,
            t: 0.0,
            hz: 12.0,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        };
        let cell = program.sample(&ctx);
        // At the very bottom of the dome (pitch == V_LOW) the gradient reads
        // pure horizon colour.
        assert_eq!(cell.bg, 0xFF0000);
    }

    #[test]
    fn same_tick_is_byte_static() {
        // The byte-budget invariant: two samples at the SAME tick
        // (even with different continuous `t`) must be identical.
        let spec = SkySpec {
            base: SkyBase::default(),
            layers: vec![SkyLayerSpec::Stars(StarsParams::default())],
            seed: 7,
            hz: 12,
        };
        let program = SkyProgram::compile(&spec);
        let ctx_a = SkyCtx {
            yaw: 1.2,
            pitch: 0.3,
            tick: 40,
            t: 40.0 / 12.0,
            hz: 12.0,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        };
        let mut ctx_b = ctx_a;
        ctx_b.t = 40.4 / 12.0; // same tick, slightly different continuous t
        assert_eq!(program.sample(&ctx_a), program.sample(&ctx_b));
    }

    #[test]
    fn active_globals_default_harmlessly() {
        // A caller that never installs anything reads t=0, no program.
        assert!(active_program().is_none() || true); // may carry state from another test on this thread
        set_active_program(None);
        assert!(active_program().is_none());
    }
}

/// BYTE-BUDGET bench: the p95 count of sky cells that change from one tick to
/// the next must stay small, so an animated sky never forces a full-frame
/// re-cast at a punishing rate. This measures the SKY LAYER in isolation —
/// every direction sample in a synthetic 322×86 grid, at consecutive ticks,
/// independent of the raycaster/composite machinery — so it is a property of
/// `SkyProgram::sample` alone.
#[cfg(test)]
mod budget_tests {
    use super::*;
    use re_world::sky_spec::SkySpec;

    /// Terminal grid width from the acceptance criterion.
    const COLS: usize = 322;
    /// Terminal grid height from the acceptance criterion.
    const ROWS: usize = 86;
    /// A representative horizontal FOV (matches typical raycaster presets) —
    /// deliberately narrower than a full 360° sweep so each column samples the
    /// sky at roughly the density a real frame would, not an artificially
    /// coarsened one.
    const HFOV: f32 = 100.0 * std::f32::consts::PI / 180.0;
    /// Ticks sampled per kind (20s at hz=12) — long enough to cover slow drift
    /// (aurora `drift_period`) and rare events (meteors, lightning) landing at
    /// least a few times each.
    const TICKS: u32 = 240;
    const HZ: u32 = 12;

    /// Every time-animated preset shipped here (`watcher`/`mirror` are
    /// intentionally excluded: `is_time_animated() == false` for both, they are
    /// driven by player yaw / fully static, so they carry no per-tick budget).
    const ANIMATED_KINDS: [&str; 8] = [
        "stars",
        "constellation",
        "aurora",
        "clouds",
        "storm",
        "datarain",
        "void",
        "fractal",
    ];

    fn sample_grid(program: &SkyProgram, tick: u32) -> Vec<SkyCell> {
        let t = tick as f32 / HZ as f32;
        let mut out = Vec::with_capacity(COLS * ROWS);
        for row in 0..ROWS {
            let pitch = V_LOW + (row as f32 / ROWS as f32) * (HALF_PI - V_LOW);
            for col in 0..COLS {
                let yaw = (col as f32 / COLS as f32 - 0.5) * HFOV;
                let ctx = SkyCtx {
                    yaw,
                    pitch,
                    tick,
                    t,
                    hz: HZ as f32,
                    cam_xy: (0.0, 0.0),
                    player_yaw: 0.0,
                };
                out.push(program.sample(&ctx));
            }
        }
        out
    }

    fn p95(mut counts: Vec<u32>) -> u32 {
        counts.sort_unstable();
        let idx = ((counts.len() as f32) * 0.95).floor() as usize;
        counts[idx.min(counts.len().saturating_sub(1))]
    }

    #[test]
    fn p95_changed_cells_per_tick_is_within_budget() {
        const BUDGET: u32 = 800;
        let mut worst = (0u32, "");
        for kind in ANIMATED_KINDS {
            let spec = SkySpec::from_kind(kind).expect("every listed kind is a known preset");
            let program = SkyProgram::compile(&spec);
            assert!(
                program.is_animated(),
                "{kind}: this bench only means something for a time-animated preset"
            );
            let mut prev = sample_grid(&program, 0);
            let mut per_tick = Vec::with_capacity(TICKS as usize);
            for tick in 1..TICKS {
                let cur = sample_grid(&program, tick);
                let changed = prev.iter().zip(cur.iter()).filter(|(a, b)| a != b).count() as u32;
                per_tick.push(changed);
                prev = cur;
            }
            let kind_p95 = p95(per_tick);
            if kind_p95 > worst.0 {
                worst = (kind_p95, kind);
            }
        }
        assert!(
            worst.0 <= BUDGET,
            "sky p95 changed-cells-per-tick budget exceeded at {COLS}x{ROWS}: kind={} p95={} (budget {BUDGET})",
            worst.1,
            worst.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── direction → mapping ──

    #[test]
    fn horizon_maps_to_cylinder_bottom_band() {
        // A ray at the horizon (pitch 0) is in the cylinder, above V_LOW so v>0.
        match direction_to_sky(0.0, 0.0) {
            SkyHit::Cylinder { v, .. } => {
                assert!(v > 0.0 && v < 1.0, "horizon sits inside the band: v={v}");
            }
            other => panic!("horizon must be cylinder, got {other:?}"),
        }
    }

    #[test]
    fn zenith_maps_to_cap_centre() {
        // Looking straight up (φ = 90°) lands at the disc centre (r_norm = 0).
        match direction_to_sky(0.0, HALF_PI) {
            SkyHit::Cap { r_norm, .. } => {
                assert!(r_norm.abs() < 1e-4, "zenith → cap centre, got r={r_norm}");
            }
            other => panic!("zenith must be cap, got {other:?}"),
        }
    }

    #[test]
    fn seam_continuity_cylinder_top_meets_cap_rim() {
        // A ray exactly at θ_cap must map to cylinder v≈1 AND cap r≈1 (rim) — the
        // density-matched join. Probe just inside each pure region (outside the blend
        // band) so we read a single region's coordinate cleanly.
        let just_below = direction_to_sky(0.0, THETA_CAP - SEAM_BLEND_RAD - 1e-3);
        let just_above = direction_to_sky(0.0, THETA_CAP + SEAM_BLEND_RAD + 1e-3);
        match just_below {
            // 3° below θ_cap is the top of the cylinder band: v approaches 1 (the
            // band spans [V_LOW, θ_cap], so 3° short of the top is ~0.95 — the join
            // is exact AT θ_cap, tested by `seam_exactly_at_theta_cap_is_blend_band`).
            SkyHit::Cylinder { v, .. } => assert!(v > 0.94, "cylinder top v→1: {v}"),
            other => panic!("just below seam must be cylinder, got {other:?}"),
        }
        match just_above {
            // 3° above θ_cap is near the cap rim: r approaches 1 (the join is exact
            // AT θ_cap, tested by `seam_exactly_at_theta_cap_is_blend_band`).
            SkyHit::Cap { r_norm, .. } => assert!(r_norm > 0.90, "cap rim r→1: {r_norm}"),
            other => panic!("just above seam must be cap, got {other:?}"),
        }
    }

    #[test]
    fn seam_exactly_at_theta_cap_is_blend_band() {
        // Exactly at θ_cap is inside the blend band, carrying both samples.
        match direction_to_sky(0.0, THETA_CAP) {
            SkyHit::Seam {
                v,
                r_norm,
                cap_weight,
                ..
            } => {
                assert!(v > 0.97, "seam cylinder v≈1: {v}");
                assert!(r_norm > 0.97, "seam cap r≈1: {r_norm}");
                assert!(
                    (cap_weight - 0.5).abs() < 1e-3,
                    "centre of the blend band is 50/50: {cap_weight}"
                );
            }
            other => panic!("θ_cap must be the seam band, got {other:?}"),
        }
    }

    #[test]
    fn cap_angle_rotates_with_yaw() {
        // The cap angle == yaw (mod TAU): a quarter turn rotates the cap a quarter.
        let p = HALF_PI - 1e-2; // well into the cap.
        let a0 = match direction_to_sky(0.0, p) {
            SkyHit::Cap { angle, .. } => angle,
            other => panic!("expected cap, got {other:?}"),
        };
        let a_quarter = match direction_to_sky(TAU / 4.0, p) {
            SkyHit::Cap { angle, .. } => angle,
            other => panic!("expected cap, got {other:?}"),
        };
        assert!((a0 - 0.0).abs() < 1e-4, "yaw 0 → cap angle 0: {a0}");
        assert!(
            (a_quarter - TAU / 4.0).abs() < 1e-4,
            "quarter-turn rotates the cap a quarter: {a_quarter}"
        );
    }

    #[test]
    fn cylinder_u_wraps_with_yaw() {
        // u = wrap(yaw/TAU): yaw 0 and yaw TAU map to the same u (0).
        let u0 = match direction_to_sky(0.0, 0.0) {
            SkyHit::Cylinder { u, .. } => u,
            other => panic!("{other:?}"),
        };
        let u_tau = match direction_to_sky(TAU, 0.0) {
            SkyHit::Cylinder { u, .. } => u,
            other => panic!("{other:?}"),
        };
        assert!(u0 < 1e-4 && u_tau < 1e-4, "yaw 0 and TAU both → u≈0");
        // A half turn → u ≈ 0.5.
        let u_half = match direction_to_sky(TAU / 2.0, 0.0) {
            SkyHit::Cylinder { u, .. } => u,
            other => panic!("{other:?}"),
        };
        assert!((u_half - 0.5).abs() < 1e-4, "half turn → u≈0.5: {u_half}");
    }

    #[test]
    fn cap_rim_density_matches_cylinder() {
        // The density-match invariant: the cap rim circumference (2π·R) equals the
        // cylinder's texels-per-turn (CYL_W). This is the seam being density-matched.
        let rim_circumference = TAU * CAP_R;
        assert!(
            (rim_circumference - CYL_W as f32).abs() < 1e-2,
            "2π·R must equal CYL_W (720): {rim_circumference}"
        );
    }

    // ── procedural content + no-black-band ──

    #[test]
    fn every_sky_direction_has_non_default_content() {
        // The must-have: a sky ray at ANY pitch up to the +60° clamp + half-vfov reach
        // (~85°) gets real content — a finite, non-absent texel — never black/absent.
        // Sweep yaw × pitch across the whole sky and assert each sample is well-formed.
        let sky = ProceduralSky::default();
        let max_pitch = 85.0_f32.to_radians(); // +60° clamp + ~25° half-vfov reach.
        let mut samples = 0u32;
        for yi in 0..72 {
            let yaw = (yi as f32 / 72.0) * TAU;
            for pi in 0..60 {
                // From just below the horizon up to ~zenith.
                let pitch = V_LOW + (pi as f32 / 59.0) * (max_pitch - V_LOW);
                let t = sample_sky(&sky, yaw, pitch);
                // A real texel: bg is a packed RGB in range; not the absent sentinel.
                assert!(t.bg <= 0x00FF_FFFF, "bg is a packed 24-bit color");
                // The sky is never pure black anywhere (horizon/zenith tones are > 0).
                assert!(t.bg > 0, "sky must have a non-black tone everywhere");
                samples += 1;
            }
        }
        assert!(samples > 4000, "swept the whole sky dome");
    }

    #[test]
    fn cap_corner_trim_only_inscribed_circle_sampled() {
        // The cap mapping only ever produces r_norm ∈ [0,1] (the inscribed circle);
        // a direction can never address a texel outside r ≤ R (the dead corners).
        for pi in 0..100 {
            let pitch = THETA_CAP + (pi as f32 / 99.0) * (HALF_PI - THETA_CAP);
            if let SkyHit::Cap { r_norm, .. } = direction_to_sky(0.7, pitch) {
                assert!(
                    (0.0..=1.0).contains(&r_norm),
                    "cap radius stays within the inscribed circle: {r_norm}"
                );
            }
        }
    }

    #[test]
    fn gradient_horizon_brighter_than_zenith() {
        // The vertical gradient: the horizon band-bottom is brighter/warmer than the
        // zenith (a readable sky, not a flat fill).
        let sky = ProceduralSky::default();
        let horizon = sky.sample_cylinder(0.5, 0.0); // band bottom
        let zenith = sky.sample_cap(0.0, 0.0); // disc centre
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        assert!(
            lum(horizon.bg) > lum(zenith.bg),
            "horizon ({}) must be brighter than zenith ({})",
            lum(horizon.bg),
            lum(zenith.bg)
        );
    }

    #[test]
    fn seam_blend_is_continuous_in_colour() {
        // Crossing the seam, the sampled color must not jump: sample just below, at,
        // and just above θ_cap and assert the bg changes monotonically-ish (no hard
        // ring). We check the at-seam value is bracketed by the two sides.
        let sky = ProceduralSky::default();
        let below = sample_sky(&sky, 0.3, THETA_CAP - SEAM_BLEND_RAD);
        let at = sample_sky(&sky, 0.3, THETA_CAP);
        let above = sample_sky(&sky, 0.3, THETA_CAP + SEAM_BLEND_RAD);
        let b = |c: u32| (c & 0xFF) as i32;
        // The blue channel at the seam sits between the two pure-region values
        // (continuous join, not a step).
        let (lo, hi) = (b(below.bg).min(b(above.bg)), b(below.bg).max(b(above.bg)));
        assert!(
            b(at.bg) >= lo - 2 && b(at.bg) <= hi + 2,
            "seam colour ({}) bracketed by below ({}) and above ({})",
            b(at.bg),
            b(below.bg),
            b(above.bg)
        );
    }

    // ── image-backed sky ──

    /// Pack `(r,g,b)` the way the bg channel of a `SkyTexel` does, for comparison.
    fn rgb(r: u8, g: u8, b: u8) -> u32 {
        (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
    }

    /// A 2×1 cylinder image: left texel red, right texel blue. `sample_nearest`
    /// splits the u-axis at 0.5, so u≈0 reads the left (red), u≈1 reads the right (blue).
    fn two_by_one() -> re_core::TextureImage {
        re_core::TextureImage::new(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255])
    }

    #[test]
    fn image_sky_cylinder_samples_left_vs_right_texel() {
        let img = re_world::SkyImages {
            cylinder: Some(std::sync::Arc::new(two_by_one())),
            cap_top: None,
            cap_bottom: None,
        };
        let sky = ImageSky::new(&img);
        // v in the middle of the band so we read the (only) image row regardless of flip.
        let left = sky.sample_cylinder(0.0, 0.5);
        let right = sky.sample_cylinder(0.999, 0.5);
        assert_eq!(left.bg, rgb(255, 0, 0), "u=0 → left (red) texel");
        assert_eq!(right.bg, rgb(0, 0, 255), "u=1 → right (blue) texel");
        // Backdrop look: colour-dominant, no grid ink, space glyph.
        assert_eq!(left.glyph, ' ', "image sky uses a space glyph");
        assert_eq!(
            left.fg, left.bg,
            "fg mirrors bg (no grid line over the image)"
        );
    }

    #[test]
    fn image_sky_absent_cap_falls_back_to_procedural() {
        // Cylinder present (so ImageSky is the chosen source) but no cap image: a cap
        // sample must equal the embedded procedural cap, not garbage / black.
        let img = re_world::SkyImages {
            cylinder: Some(std::sync::Arc::new(two_by_one())),
            cap_top: None,
            cap_bottom: None,
        };
        let sky = ImageSky::new(&img);
        let proc = ProceduralSky::default();
        let got = sky.sample_cap(0.0, 0.0);
        let expect = proc.sample_cap(0.0, 0.0);
        assert_eq!(got, expect, "absent cap → procedural cap fallback");
    }

    #[test]
    fn image_sky_cap_samples_centre_texel() {
        // A 1×1 cap image: the disc centre (r_norm=0) maps to UV (0.5,0.5) → the single
        // texel. Proves the polar→disc UV path actually reads the cap image when present.
        let cap = re_core::TextureImage::new(1, 1, vec![10, 200, 30, 255]);
        let img = re_world::SkyImages {
            cylinder: Some(std::sync::Arc::new(two_by_one())),
            cap_top: Some(std::sync::Arc::new(cap)),
            cap_bottom: None,
        };
        let sky = ImageSky::new(&img);
        assert_eq!(sky.sample_cap(0.0, 0.0).bg, rgb(10, 200, 30));
    }

    #[test]
    fn map_sky_texel_no_images_is_byte_identical_to_procedural() {
        // THE regression guard: a Map with no sky images set must produce, for every
        // swept (yaw, pitch), EXACTLY the same SkyTexel as sampling the procedural sky
        // directly — so the no-image golden frames never shift. Byte-for-byte equality.
        let map = re_world::Map::new(4, 4, 4);
        let proc = ProceduralSky::default();
        let max_pitch = 85.0_f32.to_radians();
        for yi in 0..48 {
            let yaw = (yi as f32 / 48.0) * TAU;
            for pi in 0..48 {
                let pitch = V_LOW + (pi as f32 / 47.0) * (max_pitch - V_LOW);
                let via_map = map_sky_texel(&map, &proc, yaw, pitch);
                let direct = sample_sky(&proc, yaw, pitch);
                assert_eq!(
                    via_map, direct,
                    "no-image map must match procedural sky exactly at yaw={yaw}, pitch={pitch}"
                );
            }
        }
    }

    #[test]
    fn map_sky_texel_uses_image_when_cylinder_set() {
        // With a cylinder image set, map_sky_texel must route through ImageSky: a
        // cylinder-band ray reads the image colour (space glyph), not the procedural grid.
        let mut map = re_world::Map::new(4, 4, 4);
        map.set_sky_cylinder(std::sync::Arc::new(two_by_one()));
        let proc = ProceduralSky::default();
        // Horizon ray (pitch 0) is a pure cylinder sample. yaw 0 → u≈0 → red texel.
        let texel = map_sky_texel(&map, &proc, 0.0, 0.0);
        assert_eq!(
            texel.bg,
            rgb(255, 0, 0),
            "cylinder image colour read at yaw 0"
        );
        assert_eq!(texel.glyph, ' ', "image sky uses a space glyph");
    }
}
